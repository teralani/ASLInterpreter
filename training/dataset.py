import numpy as np
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path

class ASLDataset(Dataset):
    def __init__(self, processed_dir, wsasl_json, split="train"):
        self.processed_dir = Path(processed_dir)
        self.T = 30
        self.files = list(self.processed_dir.glob("*.npy"))

        with open(wsasl_json, "r") as f:
            wsasl = json.load(f)

        # Build gloss vocabulary
        self.gloss2id = {
            entry["gloss"]: idx
            for idx, entry in enumerate(wsasl)
        }

        # Build video_id → class_id map
        self.video_to_label = {}

        for entry in wsasl:
            gloss = entry["gloss"]
            label = self.gloss2id[gloss]

            for inst in entry["instances"]:
                if inst["split"] == split:
                    self.video_to_label[inst["video_id"]] = label

        # Filter only valid processed files
        self.files = [
            f for f in self.files
            if f.stem in self.video_to_label
        ]

        # number of classes available
        self.num_classes = len(self.gloss2id)

        # Infer number of joints from the first processed file (robust fallback to 59)
        if len(self.files) > 0:
            try:
                sample = np.load(self.files[0])
                if sample.ndim >= 2:
                    self.num_joints = int(sample.shape[1])
                else:
                    self.num_joints = 59
            except Exception:
                self.num_joints = 59
        else:
            self.num_joints = 59

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])  # (T, J, 3) or empty

        # try to load optional mask saved alongside coordinates (stem_mask.npy)
        mask_path = self.processed_dir / f"{self.files[idx].stem}_mask.npy"
        mask = None
        if mask_path.exists():
            try:
                mask = np.load(mask_path)  # (F, J) or similar
            except Exception:
                mask = None

        T, J, C = self.T, self.num_joints, 3

        # ----------------------------------
        # Handle empty or malformed samples
        # ----------------------------------
        if x.ndim != 3 or x.shape[0] == 0:
            x_out = np.zeros((T, J, C), dtype=np.float32)
            mask_out = np.zeros((T, J), dtype=np.float32)

        # ----------------------------------
        # Pad short sequences
        # ----------------------------------
        elif x.shape[0] < T:
            x_out = np.zeros((T, J, C), dtype=np.float32)
            if x.shape[1] == J:
                x_out[:x.shape[0]] = x
            else:
                min_j = min(x.shape[1], J)
                x_out[:x.shape[0], :min_j] = x[:, :min_j]
            # pad mask similarly if available
            if mask is not None:
                mask_out = np.zeros((T, J), dtype=np.float32)
                if mask.shape[1] == J:
                    mask_out[:mask.shape[0]] = mask
                else:
                    min_j = min(mask.shape[1], J)
                    mask_out[:mask.shape[0], :min_j] = mask[:, :min_j]
            else:
                mask_out = np.zeros((T, J), dtype=np.float32)

        # ----------------------------------
        # Uniformly subsample long sequences
        # ----------------------------------
        elif x.shape[0] > T:
            idxs = np.linspace(0, x.shape[0] - 1, T).astype(int)
            x_sub = x[idxs]
            if x_sub.shape[1] == J:
                x_out = x_sub
            else:
                x_out = np.zeros((T, J, C), dtype=np.float32)
                min_j = min(x_sub.shape[1], J)
                x_out[:, :min_j] = x_sub[:, :min_j]
            # subsample mask if available
            if mask is not None:
                mask_sub = mask[idxs]
                if mask_sub.shape[1] == J:
                    mask_out = mask_sub
                else:
                    mask_out = np.zeros((T, J), dtype=np.float32)
                    min_j = min(mask_sub.shape[1], J)
                    mask_out[:, :min_j] = mask_sub[:, :min_j]
            else:
                mask_out = np.zeros((T, J), dtype=np.float32)

        else:
            if x.shape[1] == J:
                x_out = x
            else:
                x_out = np.zeros((T, J, C), dtype=np.float32)
                min_j = min(x.shape[1], J)
                x_out[:, :min_j] = x[:, :min_j]
            if mask is not None:
                if mask.shape[1] == J:
                    mask_out = mask
                else:
                    mask_out = np.zeros((T, J), dtype=np.float32)
                    min_j = min(mask.shape[1], J)
                    mask_out[:, :min_j] = mask[:, :min_j]
            else:
                mask_out = np.zeros((T, J), dtype=np.float32)

        # sanitize numerical issues
        x_out = np.nan_to_num(x_out, nan=0.0, posinf=1e5, neginf=-1e5).astype(np.float32)
        mask_out = np.nan_to_num(mask_out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

        # normalize per-sample to avoid extremely large coordinates
        max_abs = float(np.max(np.abs(x_out))) if x_out.size else 0.0
        if max_abs > 1e-12:
            x_out = x_out / max_abs

        # ----------------------------------
        # Label lookup
        # ----------------------------------
        video_id = self.files[idx].stem
        y = self.video_to_label[video_id]


        # ensure at least one valid frame (avoid fully-masked sequences)
        if not (np.abs(x_out).sum(axis=(1,2)) > 0).any():
            x_out[0, :, :] = 1e-6
            mask_out[0, :] = 1.0

        return (torch.from_numpy(x_out).float(), torch.from_numpy(mask_out).float()), torch.tensor(y, dtype=torch.long)