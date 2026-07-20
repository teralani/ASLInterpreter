from pathlib import Path
import sys
sys.path.append(str(Path('.').resolve()))
import importlib
import training.dataset as ds
import model.st_pose_model as sm
importlib.reload(ds)
importlib.reload(sm)
print('loaded OK')
