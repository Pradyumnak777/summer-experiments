'''
mirrors the annotation/derived-tiff subfolders under each raw_tiffs/<subject_dir>/ (hand_drawn_mask,
masked_img, output, etc, at any nesting depth) into data/annotations/, converting every tiff stack to
per-frame pngs. top-level <id>.tif / <id>mask.tif are skipped since those already go through
extract_frames.py into data/singlecell_*_new.

png files (eg output/*graphs.png) are copied through unchanged. gif and csv files are left alone,
they are rendered summaries / tables, not frame stacks.
'''

import shutil

import numpy as np
import tifffile
from PIL import Image

from ..config import ANNOTATIONS_OUT_DIR, RAW_DIR
from ..extract_frames import channel_to_uint8_percentile, mask_to_uint8


def to_uint8_stack(arr):
    """classify by content, not source dtype: binary/label stacks (values are just {0,1}, eg masks)
    always go through the min-max mask scaling so they are visible rather than a near-invisible 0/1
    png, continuous intensity stacks go through the percentile scaling, matching how extract_frames.py
    always runs mask data through mask_to_uint8 regardless of its original dtype"""
    if np.unique(arr).size <= 2:
        return mask_to_uint8(arr)
    if arr.dtype == np.uint8:
        return arr
    return channel_to_uint8_percentile(arr)


def write_frames(stack_u8, out_dir, stem):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fi in range(stack_u8.shape[0]):
        Image.fromarray(stack_u8[fi], mode="L").save(out_dir / f"{stem}_f{fi:03d}.png")
    return stack_u8.shape[0]


def convert_tif(src_path, dst_dir):
    """returns (n_frames_written, n_channels) or None if the file was skipped"""
    try:
        arr = tifffile.imread(src_path)
    except Exception as e:
        print(f"SKIP  {src_path}  (unreadable: {e})")
        return None

    stem = src_path.stem

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]  #treat a bare 2d image as a single frame

    if arr.ndim == 3:
        stack_u8 = to_uint8_stack(arr)
        n_frames = write_frames(stack_u8, dst_dir / stem, stem)
        return n_frames, None

    if arr.ndim == 4:
        #TCYX: split channels into their own subfolders, per the raw <id>.tif convention
        n_channels = arr.shape[1]
        n_frames = None
        for c in range(n_channels):
            stack_u8 = to_uint8_stack(arr[:, c])
            n_frames = write_frames(stack_u8, dst_dir / stem / f"ch{c}", f"{stem}_ch{c}")
        return n_frames, n_channels

    print(f"SKIP  {src_path}  (unexpected shape {arr.shape})")
    return None


def process_all():
    n_tif, n_png, n_skipped_other, n_skipped_bad = 0, 0, 0, 0

    subject_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    for subject_dir in subject_dirs:
        #only files inside a subfolder qualify, <id>.tif / <id>mask.tif directly in subject_dir are
        #skipped since those are handled by extract_frames.py already
        files = [p for p in subject_dir.rglob("*") if p.is_file() and p.parent != subject_dir]
        if not files:
            continue

        print(f"{subject_dir.name}: {len(files)} file(s) under subfolders")
        for src_path in sorted(files):
            rel = src_path.relative_to(RAW_DIR)
            dst_dir = ANNOTATIONS_OUT_DIR / rel.parent
            suffix = src_path.suffix.lower()

            if suffix in (".tif", ".tiff"):
                result = convert_tif(src_path, dst_dir)
                if result is None:
                    n_skipped_bad += 1
                else:
                    n_tif += 1
            elif suffix == ".png":
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_dir / src_path.name)
                n_png += 1
            else:
                n_skipped_other += 1

    print(f"\ntif stacks converted : {n_tif}")
    print(f"png files copied     : {n_png}")
    print(f"skipped (gif/csv/etc): {n_skipped_other}")
    print(f"skipped (unreadable) : {n_skipped_bad}")
    print(f"annotations -> {ANNOTATIONS_OUT_DIR}")


def main():
    process_all()


if __name__ == "__main__":
    main()
