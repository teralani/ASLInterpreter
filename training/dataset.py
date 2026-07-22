import numpy as np
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from utils.skeleton import NUM_JOINTS, WINDOW_FRAMES, RAW_CHANNELS

logger = logging.getLogger(__name__)

class ASLDataset(Dataset):
    def __init__(self, processed_dir, wsasl_json, split="train"):
        self.processed_dir = Path(processed_dir)
        self.T = WINDOW_FRAMES
        self.num_joints = NUM_JOINTS
        self.C = RAW_CHANNELS

        self.files = sorted(
            f for f in self.processed_dir.glob("*.npy")
            if not f.stem.endswith("_mask")
        )

        with open(wsasl_json, "r") as f:
            wsasl = json.load(f)

        # Build gloss vocabulary
        self.gloss2id = {
            entry["gloss"]: idx
            for idx, entry in enumerate(wsasl)
        }

        # number of classes available
        self.num_classes = len(self.gloss2id)

        # Build video_id → class_id map
        self.video_to_label = {}

        for entry in wsasl:
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
            logger.warning(
                "No mask file for %s -- check the extraction run that "
                "produced this sample.", stem,
            )
            return None
        try:
            return np.load(mask_path)
        except:
            logger.exception("Failed to load mask for %s", stem)
            return None

    def __getitem__(self, idx):
        stem = self.files[idx].stem
        x = np.load(self.files[idx])  # (T, J, 3) ==> (WINDOW_FRAMES, NUM_JOINTS, 3)
        T, J, C = self.T, self.num_joints, self.C
        mask = self._load_mask(stem)

        if x.ndim != 3 or x.shape[0] == 0:
            x_out = np.zeros((T, J, C), dtype=np.float32)
            mask_out = np.zeros((T, J), dtype=np.float32)
        
        elif x.shape[0] == T and x.shape[1] == J:
            # When extraction works
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
                "%s has shape %s, expected (%d, %d, 3) -- "
                "padding/subsampling as a fallback. Check the extraction run.",
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
        
        x_out = np.nan_to_num(x_out, nan = 0.0, posinf = 1e5, neginf = -1e5).astype(np.float32)
        mask_out = np.nan_to_num(mask_out, nan = 0.0, posinf = 1e5, neginf = -1e5).astype(np.float32)

        if not (np.abs(x_out).sum(axis=(1, 2)) > 0).any():
            x_out[0, :, :] = 1e-6
            mask_out[0, :] = 1.0
 
        y = self.video_to_label[stem]

        return (
             (torch.from_numpy(x_out).float(), torch.from_numpy(mask_out).float()),
            torch.tensor(y, dtype=torch.long),
        )