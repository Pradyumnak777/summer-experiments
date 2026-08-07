import torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

mask_dir = "data/singlecell_mask"
t = transforms.Compose([transforms.Resize((128,128), interpolation=transforms.InterpolationMode.NEAREST),
                         transforms.ToTensor()])
bad = []
for f in sorted(Path(mask_dir).glob("*.png")):
    m = t(Image.open(f).convert("L"))
    if m.sum() == 0:
        bad.append(f.name)
print(f"{len(bad)} empty masks out of {len(list(Path(mask_dir).glob('*.png')))}")
print(bad[:20])