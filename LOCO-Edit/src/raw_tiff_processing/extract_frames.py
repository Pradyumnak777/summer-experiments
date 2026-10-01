import json

import numpy as np
from PIL import Image

from .config import (
    CAPTION_A,
    CAPTION_B,
    CAPTION_COMPOSITE,
    CH_A_INDEX,
    CH_B_INDEX,
    FIXED_DIR,
    LOWER_PCT,
    OUT_DIR_A,
    OUT_DIR_B,
    OUT_DIR_MASK,
    OUT_DIR_PREVIEW,
    RAW_DIR,
    SAVE_PREVIEWS,
    UPPER_PCT,
)
from .fix_tiffs import load_and_fix


def channel_to_uint8_percentile(channel_stack, lo_pct=LOWER_PCT, hi_pct=UPPER_PCT):
    #clip the tails before rescaling, otherwise a single hot pixel pins the top of the range and
    #squashes the real signal into the bottom few uint8 levels
    lo = np.percentile(channel_stack, lo_pct)
    hi = np.percentile(channel_stack, hi_pct)
    if hi <= lo:
        lo, hi = float(channel_stack.min()), float(channel_stack.max())
    scaled = (channel_stack.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def mask_to_uint8(mask_stack):
    #masks are label data with no outliers to reject, so plain min-max keeps the edges hard
    lo, hi = float(mask_stack.min()), float(mask_stack.max())
    if hi <= lo:
        return np.zeros_like(mask_stack, dtype=np.uint8)
    scaled = (mask_stack.astype(np.float32) - lo) / (hi - lo)
    return (scaled * 255).astype(np.uint8)


def merge_preview(chA, chB):
    return np.dstack([chB, chA, chB]).astype(np.uint8)  #magenta=R&B, green=G


def find_cell_files(subject_dir):
    """yield (cell_id, tif_path, mask_path_or_None) for every non-mask .tif directly in subject_dir"""
    for tif_path in sorted(subject_dir.glob("*.tif")):
        if tif_path.stem.endswith("mask"):
            continue
        cell_id = tif_path.stem
        mask_path = subject_dir / f"{cell_id}mask.tif"
        yield cell_id, tif_path, (mask_path if mask_path.exists() else None)


def process_all():
    for out_dir in (OUT_DIR_A, OUT_DIR_B, OUT_DIR_MASK):
        out_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_PREVIEWS:
        OUT_DIR_PREVIEW.mkdir(parents=True, exist_ok=True)

    rows_A, rows_B, rows_preview = [], [], []
    subject_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())

    for subject_dir in subject_dirs:
        for cell_id, tif_path, mask_path in find_cell_files(subject_dir):
            rel_dir = subject_dir.relative_to(RAW_DIR)
            data, _ = load_and_fix(tif_path, FIXED_DIR / rel_dir / tif_path.name)
            if data is None:
                continue
            if data.ndim != 4:
                print(f"SKIP  {tif_path}  (expected 4D TCYX, got shape {data.shape})")
                continue
            frames, channels, h, w = data.shape
            if channels < 2:
                print(f"SKIP  {tif_path}  (only {channels} channel(s))")
                continue

            base_tag = f"{subject_dir.name}_cell{cell_id}"
            print(f"{base_tag}: {frames} frames, {channels} channels, {h}x{w}, dtype={data.dtype}")

            chA_u8 = channel_to_uint8_percentile(data[:, CH_A_INDEX])
            chB_u8 = channel_to_uint8_percentile(data[:, CH_B_INDEX])

            mask_u8 = None
            if mask_path is not None:
                mask_data, _ = load_and_fix(mask_path, FIXED_DIR / rel_dir / mask_path.name)
                if mask_data is not None and mask_data.shape[0] == frames:
                    mask_u8 = mask_to_uint8(mask_data)
                else:
                    print(f"  mask frame count mismatch; skipping mask for {base_tag}")

            for fi in range(frames):
                base = f"{base_tag}_f{fi:03d}"
                fA = f"{base}_chA.png"
                fB = f"{base}_chB.png"

                Image.fromarray(chA_u8[fi], mode="L").save(OUT_DIR_A / fA)
                Image.fromarray(chB_u8[fi], mode="L").save(OUT_DIR_B / fB)

                if mask_u8 is not None:
                    Image.fromarray(mask_u8[fi], mode="L").save(OUT_DIR_MASK / f"{base}_mask.png")

                if SAVE_PREVIEWS:
                    Image.fromarray(merge_preview(chA_u8[fi], chB_u8[fi])).save(
                        OUT_DIR_PREVIEW / f"{base}_preview.png"
                    )

                rows_A.append({"file_name": fA, "text": CAPTION_A})
                rows_B.append({"file_name": fB, "text": CAPTION_B})
                rows_preview.append({"file_name": f"{base}_preview.png", "text": CAPTION_COMPOSITE})

    #the split stage reads the chA manifest back in this write order, so the shuffle stays reproducible
    manifests = [(OUT_DIR_A, rows_A), (OUT_DIR_B, rows_B)]
    if SAVE_PREVIEWS:
        manifests.append((OUT_DIR_PREVIEW, rows_preview))
    for out_dir, rows in manifests:
        with (out_dir / "metadata.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    print(f"\ntotal frames saved: {len(rows_A)}")
    print(f"fixed tiffs -> {FIXED_DIR}")
    print(f"chA -> {OUT_DIR_A}")
    print(f"chB -> {OUT_DIR_B}")
    print(f"masks -> {OUT_DIR_MASK}")
    if SAVE_PREVIEWS:
        print(f"previews -> {OUT_DIR_PREVIEW}")


def main():
    process_all()


if __name__ == "__main__":
    main()
