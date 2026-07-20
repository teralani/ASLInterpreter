import numpy as np
from pathlib import Path
folder_name = "data/old_processed"
for p in Path(folder_name).glob("*.npy"):
    a = np.load(p)
    if not np.isfinite(a).all() or np.allclose(a, 0):
        print(p, "empty_or_bad", "all_zero:", np.allclose(a,0), "nan:", np.isnan(a).any(), "inf:", np.isinf(a).any())

# x = np.load(file)
# print(x)