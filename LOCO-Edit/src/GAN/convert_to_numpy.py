# GAN/convert_to_npy.py
import numpy as np
from pathlib import Path
from PIL import Image

CHA_DIR = "data/singlecell_chA_split/train"   # same dirs diffusion/VAE used
CHB_DIR = "data/singlecell_chB_split/train"
OUT_DIR = "GAN/data_npy"                       # where the stacked arrays go
SIZE    = 128

def main():
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    chA = sorted(Path(CHA_DIR).glob('*.png'))
    chB = sorted(Path(CHB_DIR).glob('*.png'))
    assert len(chA) == len(chB) and len(chA) > 0, "channel folder mismatch"

    for i, (a_path, b_path) in enumerate(zip(chA, chB)):
        a = np.array(Image.open(a_path).convert('L').resize((SIZE, SIZE)), dtype=np.uint8)
        b = np.array(Image.open(b_path).convert('L').resize((SIZE, SIZE)), dtype=np.uint8)
        np.save(out / f'{i:06d}.npy', np.stack([a, b], axis=0))   # (2, H, W) uint8

    print(f"wrote {len(chA)} .npy files to {out}")

if __name__ == "__main__":
    main()