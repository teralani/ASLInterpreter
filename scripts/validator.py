import numpy as np
from pathlib import Path

OUT = Path("data/processed")
bad = []
shapes = {}
for p in OUT.glob("*.npy"):
    a = np.load(p)
    shapes.setdefault(a.ndim and a.shape[1] if a.ndim >= 2 else None, 0)
    if np.isnan(a).any() or np.isinf(a).any():
        bad.append((p, "NaN/Inf"))
    else:
        max_abs = float(np.max(np.abs(a))) if a.size else 0.0
        if max_abs > 1e4:
            bad.append((p, f"Large values max_abs={max_abs:.1f}"))
print("Shape counts (joint dim):", shapes)
if bad:
    print("Problem files:")
    for p, reason in bad:
        print(p, reason)
else:
    print("No NaN/Inf or extreme-value files found.")