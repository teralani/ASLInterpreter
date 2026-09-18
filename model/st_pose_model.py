import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from model.spatial_attention import SpatialAttentionWithBias
from utils.velocity import add_velocity
from utils.skeleton import (
    LEFT_SHOULDER_IDX, NUM_JOINTS, RIGHT_SHOULDER_IDX, WINDOW_FRAMES, MODEL_CHANNELS,
    LEFT_HAND_START, RIGHT_HAND_START, NUM_HAND_JOINTS,
    LEFT_WRIST_POSE_IDX, RIGHT_WRIST_POSE_IDX,
)

def stitch_hands_to_body(x:torch.Tensor, mask:torch.Tensor):
    """
    Repositions hands to fit in the same world landmarks as the pose landmarks.
    Each hand's joints are translated so that its wrists land on the matching pose wrist

    Args:
        x: (B, T, J, 3) raw coordinates
        mask: (B, T, J) joint validity, 1.0 = detected, 0.0 = not detected
    Returns:
        (x, mask) with the same shape but with the hand translation. 
        If a pose wrist is not found, the associated hand is zeroed out.
    """
    x = x.clone()
    mask = mask.clone()

    def _stitch(hand_start: int, wrist_pose_idx: int):
        pose_ok = mask[..., wrist_pose_idx] > 0.5   # (B, T)
        hand_ok = mask[..., hand_start] > 0.5       # (B, T)

        pose_wrist = x[..., wrist_pose_idx, :]
        hand_wrist = x[..., hand_start, :]
        delta = pose_wrist - hand_wrist
        
        sl = slice(hand_start, hand_start + NUM_HAND_JOINTS)
        can_stitch = (pose_ok & hand_ok).unsqueeze(-1).unsqueeze(-1) # (B, T, 1, 1)

        translated = x[..., sl, :] + delta.unsqueeze(-2)
        x[..., sl, :] = torch.where(can_stitch, translated, x[..., sl, :])

        # hand but no wrist
        unanchored = (hand_ok & ~pose_ok).unsqueeze(-1) # (B, T, 1)
        x[..., sl, :] = torch.where(
            unanchored.unsqueeze(-1), torch.zeros_like(x[..., sl, :]), x[..., sl, :]
        )
        mask[..., sl] = torch.where(
            unanchored, torch.zeros_like(mask[..., sl]), mask[..., sl]
        )

    _stitch(LEFT_HAND_START, LEFT_WRIST_POSE_IDX)
    _stitch(RIGHT_HAND_START, RIGHT_WRIST_POSE_IDX)
    return x, mask

def normalize_body_relative(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """
    Body-relative normalization: center each frame on the mid-shoulder point and scale the whole clip by its shoulder width.
 
    x: (B, T, J, 3) raw coordinates
    mask: (B, T, J) joint validity, 1.0 = detected
    Returns: x normalized the same shape as x.
    """
    B, T, J, _ = x.shape
    valid = mask > 0.5  # (B, T, J)
 
    left = x[..., LEFT_SHOULDER_IDX, :]   # (B, T, 3)
    right = x[..., RIGHT_SHOULDER_IDX, :]  # (B, T, 3)
    both_ok = valid[..., LEFT_SHOULDER_IDX] & valid[..., RIGHT_SHOULDER_IDX]  # (B, T)
    both_ok_f = both_ok.unsqueeze(-1).to(x.dtype)  # (B, T, 1)
 
    shoulder_mid = (left + right) / 2.0  # (B, T, 3)
 

    fallback_center = (x * vj).sum(dim=2) / vj.sum(dim=2).clamp(min=1)  # (B, T, 3)
 
    center = both_ok_f * shoulder_mid + (1 - both_ok_f) * fallback_center  # (B, T, 3)
    x_centered = x - center.unsqueeze(2)  # (B, T, J, 3)
 
    shoulder_width = (left - right).norm(dim=-1)  # (B, T)
    shoulder_width = torch.where(
        both_ok, shoulder_width, torch.full_like(shoulder_width, float("nan"))
    )
    scale = torch.nanmedian(shoulder_width, dim=1).values  # (B,)
 
    fallback_scale = x_centered.abs().amax(dim=(1, 2, 3)).clamp(min=eps)  # (B,)
    scale = torch.where(torch.isnan(scale), fallback_scale, scale).clamp(min=eps)
 
    return x_centered / scale.view(B, 1, 1, 1)

class STBlock(nn.Module):

    def __init__(self, dim: int, heads: int, adj, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.spatial = SpatialAttentionWithBias(dim, heads, adj)
        self.norm1 = nn.LayerNorm(dim)

        self.temporal = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim)
        )

        self.norm3 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, valid_mask):
        """
        x:            (B, T, J, D)
        causal_mask:  (T, T) bool, True = blocked
        valid_mask:   (B, T, J, 1) float, 1.0 = valid joint/frame, 0.0 = invalid
        """

        B, T, J, D = x.shape

        residual = x
        x_flat = self.norm1(x).view(B * T, J, D)
        x_flat = self.spatial(x_flat)
        x = residual + self.dropout(x_flat.view(B, T, J, D))
        x = x * valid_mask

        residual = x
        x_t = self.norm2(x).permute(0, 2, 1, 3).reshape(B * J, T, D)
        x_t, _ = self.temporal(x_t, x_t, x_t, attn_mask = causal_mask)
        x_t = x_t.view(B, J, T, D).permute(0, 2, 1, 3) # (B, T, J, D)
        x = residual + self.dropout(x_t)
        x = x * valid_mask

        residual = x
        x = residual + self.dropout(self.ff(self.norm3(x)))
        x = x * valid_mask

        return x

class STPoseModel(nn.Module):
    """

    forward() uses raw inputs (B, T, J, 3) and mask (B, T, J)

    Hand stitching, normalization, then velocity computation is calculated.
    """
    def __init__(
            self, adj, num_classes:int = 2000, hidden_dim:int=256, num_heads:int = 8, 
            num_layers: int = 2, dropout:float = 0.1, 
            T:int = WINDOW_FRAMES, num_joints:int = NUM_JOINTS):
        super().__init__()

        assert adj.shape == (num_joints, num_joints), (
            f"adj must be ({num_joints}, {num_joints}) to match the joint layout, got {tuple(adj.shape)}"
        )

        self.T = T
        self.num_joints = num_joints

        self.embed = nn.Linear(MODEL_CHANNELS, hidden_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)
        self.embed_norm = nn.LayerNorm(hidden_dim)

        self.pos = nn.Parameter(torch.zeros(1, T, 1, hidden_dim))

        self.joint_embed = nn.Parameter(torch.zeros(1, 1, num_joints, hidden_dim))

        from utils.skeleton import BONE_PARENTS
        self.register_buffer("bone_parents", BONE_PARENTS)

        self.blocks = nn.ModuleList([
            STBlock(hidden_dim, num_heads, adj, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

        self.cls = nn.Linear(hidden_dim, num_classes)
        nn.init.xavier_uniform_(self.cls.weight)
        nn.init.zeros_(self.cls.bias)

    def _assert_finite(self, t: torch.Tensor, name: str):
        if not torch.isfinite(t).all():
            finite_mask = torch.isfinite(t)
            if finite_mask.any():
                vals = t[finite_mask]
                stats = {
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "mean": float(vals.mean()),
                    "std": float(vals.std())
                }
            else:
                stats = {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}

            bad = (~torch.isfinite(t)).nonzero()
            print("first bad:", bad[:10])
            raise RuntimeError(f"Non-finite values in {name}: {stats}")

    def forward(self, x, mask=None):
        if mask is None:
            mask = (x.abs().sum(dim=-1) > 0).float()
        
        # Connect hand to pose wrists
        x, mask = stitch_hands_to_body(x, mask)
        valid_joints = mask > 0.5
        
        # Per sample scale normalization
        x = normalize_body_relative(x, mask)
        self._assert_finite(x, "after_normalize")


        from utils.skeleton import BONE_PARENTS
        from utils.velocity import add_bone

        bone = add_bone(x, mask, self.bone_parents)
        x = torch.cat([x, bone], dim=-1) # 6 channels: pos + bone
        x = add_velocity(x, mask)

        
        self._assert_finite(x, "after_add_velocity")
        assert x.shape[-1] == MODEL_CHANNELS, f"Expected {MODEL_CHANNELS} channels. got {x.shape[-1]}"

        B, T, J, C = x.shape

        valid_frames = valid_joints.any(dim=2).clone()
        no_valid_frame = (~valid_frames).all(dim=1)
        if no_valid_frame.any():
            valid_frames[no_valid_frame, 0] = True

        no_valid_joint = ~valid_joints.any(dim=2, keepdim=True) & valid_frames.unsqueeze(-1)
        valid_joints_for_mask = valid_joints.clone()
        if no_valid_joint.any():
            valid_joints_for_mask[..., 0] = valid_joints_for_mask[..., 0] | no_valid_joint.squeeze(-1)
        
        x = self.embed(x) # (B, T, J, D)
        x = self.embed_norm(x)
        self._assert_finite(x, "after_embed")

        x = x + self.pos[:, :T] + self.joint_embed

        valid_mask = valid_joints_for_mask.unsqueeze(-1).to(x.dtype) # (B, T, J, 1)

        device = x.device
        causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

        for i, block in enumerate(self.blocks):
            x = block(x, causal_mask, valid_mask)
            self._assert_finite(x, f"after_block_{i}")

        x = self.final_norm(x)
        self._assert_finite(x, "after_final_norm")

        # Masked mean pooling across joints (B, T, D)
        vj = valid_joints.unsqueeze(-1).to(x.dtype)
        x = (x * vj).sum(dim=2) / vj.sum(dim=2).clamp(min=2)
        self._assert_finite(x, "after_joint_pool")

        # Masked mean pooling across time (B, D)
        valid_frames_f = valid_frames.unsqueeze(-1).to(x.dtype)
        x = (x * valid_frames_f).sum(dim=1) / valid_frames_f.sum(dim=1).clamp(min=1)
        self._assert_finite(x, "before classifier")

        return self.cls(x)
    
    @torch.no_grad
    def predict_window(self, landmarks, mask, device=None):
        """
        Args:
            landmarks: (T, J, 3) for a single window, or (B, T, J, 3) for batch of windows.
            mask: (T, J) or (B, T, J) matching `landmarks`.
            device: optional 
 
        Returns:
            (num_classes,) or (B, num_classes) softmax probabilities.
        """

        was_training = self.training
        self.eval()
        try:
            landmarks = torch.as_tensor(landmarks, dtype = torch.float32)
            mask_t = torch.as_tensor(mask, dtype = torch.float32)

            single = landmarks.dim() == 3
            if single:
                landmarks = landmarks.unsqueeze(0)
                mask_t = mask_t.unsqueeze(0)

            if device is not None:
                landmarks, mask_t = landmarks.to(device), mask_t.to(device)

            probs = torch.softmax(self(landmarks, mask_t), dim = -1)
            return probs.squeeze(0) if single else probs
        finally:
            self.train(was_training)