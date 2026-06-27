from pathlib import Path
from PIL import Image
import numpy as np

root = Path("/root/autodl-tmp/datasetss/loveda")

paths = list(root.glob("Train/Train/*/masks_png/*.png"))

print("mask数量：", len(paths))

values = set()

for p in paths[:100]:
    mask = np.array(Image.open(p))
    values.update(np.unique(mask).tolist())

print("前100个mask标签：")
print(sorted(values))