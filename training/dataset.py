import numpy as np
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from utils.skeleton import NUM_JOINTS, WINDOW_FRAMES, RAW_CHANNELS, build_left_right_swap

logger = logging.getLogger(__name__)

class ASLDataset(Dataset):
    def __init__(
        self, processed_dir, wsasl_json, split="train",
        augment = None,
        jitter_std: float = 0.01,
        joint_dropout_prob:float = 0.05,
        frame_dropout_prob: float = 0.05,
        mirror_prob: float = 0.5,
        left_right_swap = None
        ):

        self.processed_dir = Path(processed_dir)
        self.T = WINDOW_FRAMES
        self.num_joints = NUM_JOINTS
        self.C = RAW_CHANNELS

        # dataset augmentation
        self.augment = (split == "train") if augment is None else augment
        self.jitter_std = jitter_std
        self.joint_dropout_prob = joint_dropout_prob
        self.frame_dropout_prob = frame_dropout_prob
        self.mirror_prob = mirror_prob

        self.left_right_swap = (
            build_left_right_swap() if left_right_swap is None else left_right_swap
        )

        self.files = sorted(
            f for f in self.processed_dir.glob("*.npy")
            if not f.stem.endswith("_mask")
        )

        with open(wsasl_json, "r") as f:
            wsasl = json.load(f)

        processed_stems = {
            f.stem for f in Path(processed_dir).glob("*.npy") if not f.stem.endswith("_mask")
        }

        split_counts = {}
        for entry in wsasl:
            g = entry["gloss"]
            counts = split_counts.setdefault(g, {"train": 0, "val": 0, "test": 0})
            for inst in entry["instances"]:
                if inst["video_id"] in processed_stems and inst["split"] in counts:
                    counts[inst["split"]] += 1

        # Only glosses with at least one processed instance in EVERY split are eligible
        eligible = [
            g for g, c in split_counts.items()
            if c["train"] > 0 and c["val"] > 0 and c["test"] > 0
        ]

        # num_classes = 100
        num_classes = 1e6 # all classes


        eligible.sort(key=lambda g: sum(split_counts[g].values()), reverse=True)

        if num_classes > 3000:
            top_glosses = set(eligible[:])
        else:
            top_glosses = set(eligible[:num_classes])  # WLASL-100 but only from splits-complete glosses


        self.gloss2id = {g: idx for idx, g in enumerate(sorted(top_glosses))}
        bad = [(g, i) for g, i in self.gloss2id.items() if not (0 <= i < num_classes)]
        assert not bad, f"gloss2id has out-of-range indices: {bad}"
        
        total_instances = sum(
            len(entry["instances"]) for entry in wsasl if entry["gloss"] in top_glosses
        )
        processed_instances = sum(
            1 for entry in wsasl if entry["gloss"] in top_glosses
            for inst in entry["instances"] if inst["video_id"] in processed_stems
        )
        print(f"{processed_instances}/{total_instances} instances processed ({processed_instances/total_instances:.1%})")

        # number of classes available
        self.num_classes = len(self.gloss2id)

        # Build video_id -> class_id map
        self.video_to_label = {}

        for entry in wsasl:
            if entry["gloss"] not in self.gloss2id:
                continue
            label = self.gloss2id[entry["gloss"]]

            for inst in entry["instances"]:
                if inst["split"] == split:
                    self.video_to_label[inst["video_id"]] = label

        # Filter only valid processed files
        self.files = [f for f in self.files if f.stem in self.video_to_label]

    def __len__(self):
        return len(self.files)
    
    def _load_mask(self, stem):
        mask_path = self.processed_dir / f"{stem}_mask.npy"
        if not mask_path.exists():
            logger.warning("No mask file for %s -- check the extraction run that produced this sample.", stem)
            return None
        try:
            return np.load(mask_path)
        except:
            logger.exception("Failed to load mask for %s", stem)
            return None

    def __getitem__(self, idx):
        stem = self.files[idx].stem
        x = np.load(self.files[idx])  # (T, J, 3) -> (WINDOW_FRAMES, NUM_JOINTS, 3)
        T, J, C = self.T, self.num_joints, self.C
        mask = self._load_mask(stem)

        if x.ndim != 3 or x.shape[0] == 0:
            x_out = np.zeros((T, J, C), dtype=np.float32)
            mask_out = np.zeros((T, J), dtype=np.float32)
        
        elif x.shape[0] == T and x.shape[1] == J:
            x_out = x.astype(np.float32)
            if mask is not None and mask.shape == (T, J):
                mask_out = mask.astype(np.float32)
            elif mask is not None:
                mask_out = np.zeros((T, J), dtype=np.float32)
                min_t, min_j = min(mask.shape[0], T), min(mask.shape[1], J)
                mask_out[:min_t, :min_j] = mask[:min_t, :min_j]
            else:
                mask_out = (np.abs(x_out).sum(axis=-1) > 0).astype(np.float32)

        else:
            logger.warning(
                "%s has shape %s, expected (%d, %d, 3)",
                stem, x.shape, T, J,
            )
            if x.shape[0] < T:
                x_out = np.zeros((T, J, C), dtype=np.float32)
                min_j = min(x.shape[1], J)
                x_out[:x.shape[0], :min_j] = x[:, :min_j]
                mask_src = mask
            else:
                idxs = np.linspace(0, x.shape[0] - 1, T).astype(int)
                x_sub = x[idxs]
                x_out = np.zeros((T, J, C), dtype=np.float32)
                min_j = min(x_sub.shape[1], J)
                x_out[:, :min_j] = x_sub[:, :min_j]
                mask_src = mask[idxs] if mask is not None else None
 
            if mask_src is not None:
                mask_out = np.zeros((T, J), dtype=np.float32)
                min_t, min_j = min(mask_src.shape[0], T), min(mask_src.shape[1], J)
                mask_out[:min_t, :min_j] = mask_src[:min_t, :min_j]
            else:
                mask_out = (np.abs(x_out).sum(axis=-1) > 0).astype(np.float32)
        
        # x_out = np.nan_to_num(x_out, nan = 0.0, posinf = 1e5, neginf = -1e5).astype(np.float32)
        # mask_out = np.nan_to_num(mask_out, nan = 0.0, posinf = 1e5, neginf = -1e5).astype(np.float32)

        bad = ~np.isfinite(x_out)           # (T, J, 3)
        bad_joint = bad.any(axis=-1) | ~np.isfinite(mask_out)  # (T, J)

        x_out = np.where(bad, 0.0, x_out).astype(np.float32)
        mask_out = np.where(bad_joint, 0.0, mask_out).astype(np.float32)

        if self.augment:
            x_out, mask_out = self._augment(x_out, mask_out)    

        if not (np.abs(x_out).sum(axis=(1, 2)) > 0).any():
            x_out[0, :, :] = 1e-6
            mask_out[0, :] = 1.0
 
        y = self.video_to_label[stem]

        return (
             (torch.from_numpy(x_out).float(), torch.from_numpy(mask_out).float()),
            torch.tensor(y, dtype=torch.long), stem
        )


    def _augment(self, x, mask):
        T, J, C = x.shape
        valid = mask > 0.5

        # spatial jitter
        if self.jitter_std > 0:
            noise = np.random.normal(0.0, self.jitter_std, size=x.shape).astype(np.float32)
            x = x + noise * valid[..., None]

        # joint dropout
        if self.joint_dropout_prob > 0:
            drop = (np.random.rand(T, J) < self.joint_dropout_prob) & valid
            x = np.where(drop[..., None], 0.0, x)
            mask = np.where(drop, 0.0, mask)
            valid = mask > 0.5

        # frame dropout 
        if self.frame_dropout_prob > 0:
            frame_drop = np.random.rand(T) < self.frame_dropout_prob
            x[frame_drop] = 0.0
            mask[frame_drop] = 0.0

        # mirroring
        if (
            self.mirror_prob > 0
            and self.left_right_swap is not None
            and np.random.rand() < self.mirror_prob
        ):
            x = x[:, self.left_right_swap, :].copy()
            mask = mask[:, self.left_right_swap].copy()
            x[..., 0] = -x[..., 0]  # flip along x-axis

        return x.astype(np.float32), mask.astype(np.float32)