import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from model.spatial_attention import SpatialAttentionWithBias
from utils.velocity import add_velocity
from utils.skeleton import (
    NUM_JOINTS, WINDOW_FRAMES, MODEL_CHANNELS,
    LEFT_HAND_START, RIGHT_HAND_START, NUM_HAND_JOINTS,
    LEFT_WRIST_POSE_IDX, RIGHT_WRIST_POSE_IDX,
)

def stitch_hands_to_body(x:torch.Tensor, mask:torch.Tensor):
    """
    Repositions hands to fit in the same world landmarks as the pose landmarks.
    Each hand's joints are rigidly translated so that its wrists land on the 
    matching pose wrist, preserving the hands' shape/orientation while giving it 
    a real body-relative position.

    Args:
        x:      (B, T, J, 3) raw coordinates
        mask:   (B, T, J) joint validity, 1.0 = detected, 0.0 = not detected
    Returns:
        (x, mask) with the same shape but with the hand translation. If a pose 
        wrist is not found, the associated hand is zeroed out.
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

        # There is a hand but no pose wrist
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

class STPoseModel(nn.Module):
    """
    Spatiotemporal pose model for isolated sign recognition

    forward() accepts raw inputs (B, T, J, 3) and mask (B, T, J)

    Hand stitching, normalization, then velocity computation is calculated.
    """
    def __init__(self, adj, num_classes:int = 2000, hidden_dim:int=128, num_heads:int = 4, T:int = WINDOW_FRAMES, num_joints:int = NUM_JOINTS):
        super().__init__()

        assert adj.shape == (num_joints, num_joints), (
            f"adj must be ({num_joints}, {num_joints}) to match the joint layout, got {tuple(adj.shape)}"
        )

        self.T = T
        self.num_joints = num_joints

        self.embed = nn.Linear(MODEL_CHANNELS, hidden_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)

        self.spatial = SpatialAttentionWithBias(hidden_dim, num_heads, adj)

        self.temporal = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # initializes small positional embeddings to avoid large activations
        self.pos = nn.Parameter(torch.zeros(1, T, hidden_dim))

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
            raise RuntimeError(f"Non-finite values detected in {name}: {stats}")

    def forward(self, x, mask=None):
        if mask is None:
            mask = (x.abs().sum(dim=-1) > 0).float()
        
        # Connect hand to pose wrists
        x, mask = stitch_hands_to_body(x, mask)
        valid_joints = mask > 0.5
        
        # Per sample scale normalization
        max_abs = x.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-12)
        x = x / max_abs

        # velocity added        
        x = add_velocity(x, mask)
        self._assert_finite(x, "after_add_velocity")
        assert x.shape[-1] == 6, f"Expected {MODEL_CHANNELS} channels, got {x.shape[-1]}"

        B, T, J, C = x.shape

        valid_frames = valid_joints.any(dim=2).clone()
        no_valid_frame = (~valid_frames).all(dim=1)
        if no_valid_frame.any():
            valid_frames[no_valid_frame, 0] = True
        key_padding_mask = ~valid_frames
        
        x = self.embed(x) # (B, T, J, D)
        self._assert_finite(x, "after_embed")
        x = x.view(B * T, J, -1) # (B*T, J, D)

        # Joint-level key_padding_mask for spatial attention
        valid_joints_flat = valid_joints.reshape(B * T, J).clone()
        no_valid_joint = (~valid_joints_flat).all(dim=1)
        if no_valid_joint.any():
            valid_joints_flat[no_valid_joint, 0] = True
        joint_kpm = ~valid_joints_flat

        x = self.spatial(x, key_padding_mask=joint_kpm) # (B*T, J, D)
        self._assert_finite(x, "after_spatial")
        x = x.view(B, T, J, -1)

        # Masked mean pooling across joints
        vj = valid_joints_flat.view(B, T, J).unsqueeze(-1).to(x.dtype)
        x = (x * vj).sum(dim=2) / vj.sum(dim=2).clamp(min=1)
        self._assert_finite(x, "after_joint_pool")

        # Positional information added up to timestep
        x = x + self.pos[:, :T]

        self._assert_finite(x, "after_pos_add")

        # Causal attention mask
        device = x.device
        attn_mask = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)


        x, _ = self.temporal(
            x, x, x,
            attn_mask = attn_mask,
            key_padding_mask = key_padding_mask
        )
        self._assert_finite(x, "after_temporal")

        valid_frames_f = valid_frames.unsqueeze(-1).to(x.dtype) # (B, T, 1)

        # aggregates over time for a clip-level representation (B, D)
        x = (x * valid_frames_f).sum(dim=1)
        x = x / valid_frames_f.sum(dim=1).clamp(min=1)

        self._assert_finite(x, "before_classifier")
        return self.cls(x)
    
    @torch.no_grad
    def predict_window(self, landmarks, mask, device=None):
        """
        Convenience entry point for a live sliding-window caller.
 
        A live loop should keep a rolling deque of the last `self.T` raw
        (J, 3) coordinate frames plus (J,) validity masks, in exactly
        extract.py's joint layout -- the same per-frame output
        `_extract_frame_keypoints` already produces, just buffered instead
        of written to disk. No stitching or normalization is needed on the
        caller's side; `forward()` does that internally, identically to
        how ASLDataset's samples are processed at training time.
 
        Args:
            landmarks: (T, J, 3) for a single window, or (B, T, J, 3) for a
                batch of windows.
            mask: (T, J) or (B, T, J) matching `landmarks`.
            device: optional device to move tensors to before inference.
 
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