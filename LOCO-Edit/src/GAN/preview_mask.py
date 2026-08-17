import numpy as np
from pathlib import Path
from PIL import Image

DIRS = {
    "old_hard":  "GAN/data_npy_masked",
    "new_soft":  "GAN/data_npy_masked_new",
}
OUT_DIR = "GAN/preview_out"
INDEX   = 0   # which sample (000000.npy, 000001.npy, ...) to preview

def main():
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)

    for tag, npy_dir in DIRS.items():
        npy_path = Path(npy_dir) / f"{INDEX:06d}.npy"
        if not npy_path.exists():
            print(f"[{tag}] skipping, not found: {npy_path}")
            continue

        arr = np.load(npy_path)   # [2, H, W], uint8
        a, b = arr[0], arr[1]

        Image.fromarray(a).save(out / f"{INDEX:06d}_{tag}_chA.png")
        Image.fromarray(b).save(out / f"{INDEX:06d}_{tag}_chB.png")

        nonzero_frac = (a > 0).mean()
        print(f"[{tag}] {npy_path}  shape={arr.shape}  chA nonzero_frac={nonzero_frac:.3f}  "
              f"chA min/max={a.min()}/{a.max()}")

    print(f"\nwrote previews to {OUT_DIR}")


if __name__ == "__main__":
    main()
