# prepare_data_singlecells.py
import json
from pathlib import Path
import numpy as np
import tifffile as tiff
from PIL import Image

SRC_ROOT = Path("single_cell_files_fixed")

OUT_DIR_A   = Path("data/singlecell_chA")
OUT_DIR_B   = Path("data/singlecell_chB")
OUT_DIR_MASK = Path("data/singlecell_mask")
PREVIEW_DIR = Path("data/singlecell_previews")

CAPTION_A = "an image of gene burst in a microscopic cell"
CAPTION_B = "an image of condensate in a microscopic cell"
CAPTION_COMPOSITE = "an image of a cell with magenta condensate and green gene burst"
SAVE_PREVIEWS = True
LOWER_PCT = 0.35
UPPER_PCT = 99.65

CH_A_INDEX = 0
CH_B_INDEX = 1


def channel_to_uint8_percentile(channel_stack, lo_pct=LOWER_PCT, hi_pct=UPPER_PCT):
    lo = np.percentile(channel_stack, lo_pct)
    hi = np.percentile(channel_stack, hi_pct)
    if hi <= lo:
        lo, hi = float(channel_stack.min()), float(channel_stack.max())
    scaled = (channel_stack.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def mask_to_uint8(mask_stack):
    lo, hi = float(mask_stack.min()), float(mask_stack.max())
    if hi <= lo:
        return np.zeros_like(mask_stack, dtype=np.uint8)
    scaled = (mask_stack.astype(np.float32) - lo) / (hi - lo)
    return (scaled * 255).astype(np.uint8)


def merge_preview(chA, chB):
    return np.dstack([chB, chA, chB]).astype(np.uint8)  # magenta=R&B, green=G


def find_cell_files(subject_dir: Path):
    """Yield (cell_id, tif_path, mask_path_or_None) for every non-mask .tif directly in subject_dir."""
    for tif_path in sorted(subject_dir.glob("*.tif")):
        if tif_path.stem.endswith("mask"):
            continue
        cell_id = tif_path.stem
        mask_path = subject_dir / f"{cell_id}mask.tif"
        yield cell_id, tif_path, (mask_path if mask_path.exists() else None)


def process_all():
    OUT_DIR_A.mkdir(parents=True, exist_ok=True)
    OUT_DIR_B.mkdir(parents=True, exist_ok=True)
    OUT_DIR_MASK.mkdir(parents=True, exist_ok=True)
    if SAVE_PREVIEWS:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    rows_A, rows_B, rows_preview = [], [], []
    subject_dirs = sorted(d for d in SRC_ROOT.iterdir() if d.is_dir())

    for subject_dir in subject_dirs:
        for cell_id, tif_path, mask_path in find_cell_files(subject_dir):
            data = tiff.imread(tif_path)
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
                mask_data = tiff.imread(mask_path)
                if mask_data.shape[0] == frames:
                    mask_u8 = mask_to_uint8(mask_data)
                else:
                    print(f"  mask frame count mismatch ({mask_data.shape[0]} vs {frames}); skipping mask")

            for fi in range(frames):
                base = f"{base_tag}_f{fi:03d}"
                fA = f"{base}_chA.png"
                fB = f"{base}_chB.png"
                

                Image.fromarray(chA_u8[fi], mode='L').save(OUT_DIR_A / fA)
                Image.fromarray(chB_u8[fi], mode='L').save(OUT_DIR_B / fB)

                if mask_u8 is not None:
                    Image.fromarray(mask_u8[fi], mode='L').save(OUT_DIR_MASK / f"{base}_mask.png")

                if SAVE_PREVIEWS:
                    Image.fromarray(merge_preview(chA_u8[fi], chB_u8[fi])).save(
                        PREVIEW_DIR / f"{base}_preview.png"
                    )

                rows_A.append({"file_name": fA, "text": CAPTION_A})
                rows_B.append({"file_name": fB, "text": CAPTION_B})
                rows_preview.append({"file_name": f"{base}_preview.png", "text": CAPTION_COMPOSITE})

    for out_dir, rows in [(OUT_DIR_A, rows_A), (OUT_DIR_B, rows_B), (PREVIEW_DIR, rows_preview)]:
        with (out_dir / "metadata.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    print(f"\ntotal frames saved: {len(rows_A)}")
    print(f"chA -> {OUT_DIR_A}")
    print(f"chB -> {OUT_DIR_B}")
    print(f"masks -> {OUT_DIR_MASK}")
    if SAVE_PREVIEWS:
        print(f"previews -> {PREVIEW_DIR}")


if __name__ == "__main__":
    process_all()