# GAN/convert_to_npy.py
import numpy as np
from pathlib import Path
from PIL import Image

CHA_DIR = "data/singlecell_chA_split/train"   # same dirs diffusion/VAE used
CHB_DIR = "data/singlecell_chB_split/train"
OUT_DIR = "GAN/data_npy_masked_new"                       # where the stacked arrays go
MASK_DIR = "data/singlecell_mask"
SIZE    = 128

def main():
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    chA = sorted(Path(CHA_DIR).glob('*.png'))
    chB = sorted(Path(CHB_DIR).glob('*.png'))
    assert len(chA) == len(chB) and len(chA) > 0, "channel folder mismatch"

    n_written, n_skipped = 0, 0
    for a_path, b_path in zip(chA, chB):
        base = a_path.name[:-len("_chA.png")]
        mask_path = Path(MASK_DIR) / f"{base}_mask.png"
        mask = Image.open(mask_path).convert('L').resize((SIZE, SIZE), resample=Image.BILINEAR)
        mask = np.array(mask).astype(np.float32) / 255.0   # soft alpha in [0,1], no threshold

        if not mask.any():
            n_skipped += 1
            continue

        a = np.array(Image.open(a_path).convert('L').resize((SIZE, SIZE)), dtype=np.uint8)
        b = np.array(Image.open(b_path).convert('L').resize((SIZE, SIZE)), dtype=np.uint8)

        # arithmetic blend, not np.where -- mask is a float alpha now, not a boolean,
        # so np.where(mask, a, 0) would just treat any nonzero alpha as "fully keep"
        a_masked = (mask * a).astype(np.uint8)
        b_masked = (mask * b).astype(np.uint8)

        np.save(out / f'{n_written:06d}.npy', np.stack([a_masked, b_masked], axis=0))
        n_written += 1

    print(f"wrote {n_written} masked .npy files to {out} ({n_skipped} skipped: empty mask)")


if __name__ == "__main__":
    main()