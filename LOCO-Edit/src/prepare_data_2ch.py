# prepare_data_2ch.py
import os, json
from pathlib import Path
import numpy as np
import tifffile as tiff
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
TIFFS = [
    "OneDrive_1_6-17-2026/bleach_corrected_DPHM_Sox2_MCP_halo549_snap646-01_MIP_merged.tif",
    "OneDrive_1_6-17-2026/bleach_corrected_DPHM_Sox2_MCP_halo549_snap646-02_MIP_merged.tif",
    "OneDrive_1_6-17-2026/bleach_corrected_DPHM_Sox2_MCP_halo549_snap646-03_MIP_merged.tif",
]

OUT_DIR_A    = Path("data/microscopy_lora_green")           # chA training images + metadata
OUT_DIR_B    = Path("data/microscopy_lora_magenta")           # chB training images + metadata
PREVIEW_DIR  = Path("data/microscopy_lora_previews_test")      # green/magenta composites, inspection only

CAPTION_A = "an image of green cells in fluorescent microscopy"
CAPTION_B = "an image of magenta cells in fluorescent microscopy"
TILE = 512

SAVE_PREVIEWS = True
# LOWER_PCT = 0.35    
# UPPER_PCT = 99.65

# Replace LOWER_PCT / UPPER_PCT with these:
CH_A_LO = 0      # Min from ImageJ B&C for Channel 1
CH_A_HI = 821    # Max from ImageJ B&C for Channel 1
CH_B_LO = 0      # Min from ImageJ B&C for Channel 2
CH_B_HI = 2501    # Max from ImageJ B&C for Channel 2

CH_A_INDEX = 0   # Sox2 / halo549 (green LUT) — flip to 1 if previews look swapped vs ImageJ
CH_B_INDEX = 1   # MCP  / snap646 (magenta LUT)


def first_existing_axis(axes, candidates):
    for candidate in candidates:
        if candidate in axes:
            return candidate
    return None


def move_to_tcyx(data, axes):
    if "C" not in axes:
        raise ValueError(f"No channel axis found in TIFF axes {axes!r}")
    y_axis = first_existing_axis(axes, "Y")
    x_axis = first_existing_axis(axes, "X")
    if y_axis is None or x_axis is None:
        raise ValueError(f"No Y/X spatial axes found in TIFF axes {axes!r}")

    keep = ["T"] if "T" in axes else []
    keep += ["C", y_axis, x_axis]

    slicer, reduced = [], []
    for axis, size in zip(axes, data.shape):
        if axis in keep:
            slicer.append(slice(None)); reduced.append(axis)
        elif size == 1:
            slicer.append(0)
        else:
            raise ValueError(f"Unsupported non-singleton axis {axis!r} size {size}")
    data = data[tuple(slicer)]
    axes = "".join(reduced)

    if "T" not in axes:
        data = np.expand_dims(data, 0); axes = "T" + axes
    order = [axes.index(a) for a in "TCYX"]
    return np.moveaxis(data, order, range(4)), "TCYX"


# def channel_to_uint8(channel_stack):
#     # stretch once across the whole stack so frames stay comparable (data is bleach-corrected)
#     lo = np.percentile(channel_stack, LOWER_PCT)
#     hi = np.percentile(channel_stack, UPPER_PCT)
#     if hi <= lo:
#         lo, hi = float(channel_stack.min()), float(channel_stack.max())
#     scaled = (channel_stack.astype(np.float32) - lo) / max(hi - lo, 1e-6)
#     return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)

def channel_to_uint8(channel_stack, lo, hi):
    scaled = (channel_stack.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)

def tile_frame(frame, size=TILE):
    h, w = frame.shape[:2]
    ny, nx = h // size, w // size
    off_y = (h - ny * size) // 2
    off_x = (w - nx * size) // 2
    for iy in range(ny):
        for ix in range(nx):
            y, x = off_y + iy * size, off_x + ix * size
            yield frame[y:y+size, x:x+size]


def merge_preview(chA, chB):
    return np.dstack([chB, chA, chB]).astype(np.uint8)  # magenta=R&B, green=G


def process_tiffs():
    OUT_DIR_A.mkdir(parents=True, exist_ok=True)
    OUT_DIR_B.mkdir(parents=True, exist_ok=True)
    if SAVE_PREVIEWS:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    rows_A, rows_B = [], []

    for vi, name in enumerate(TIFFS):
        path = os.path.join(SRC_DIR, name)
        with tiff.TiffFile(path) as tf:
            series = tf.series[0]
            data, axes = move_to_tcyx(series.asarray(), series.axes)
        frames, channels, h, w = data.shape
        if channels < 2:
            raise ValueError(f"{name}: expected >=2 channels, found {channels}")

        print(f"{name}: {frames} frames, {channels} channels, {h}x{w}, dtype={data.dtype}")

        chA_u8 = channel_to_uint8(data[:, CH_A_INDEX, :, :], CH_A_LO, CH_A_HI)
        chB_u8 = channel_to_uint8(data[:, CH_B_INDEX, :, :], CH_B_LO, CH_B_HI)

        saved = 0
        for fi in range(frames):
            for ti, (tA, tB) in enumerate(zip(tile_frame(chA_u8[fi]), tile_frame(chB_u8[fi]))):
                base = f"v{vi:02d}_f{saved:03d}_t{ti}"
                fA = f"{base}_chA.png"
                fB = f"{base}_chB.png"

                Image.fromarray(tA, mode='L').save(OUT_DIR_A / fA)
                Image.fromarray(tB, mode='L').save(OUT_DIR_B / fB)

                if SAVE_PREVIEWS:
                    Image.fromarray(merge_preview(tA, tB)).save(PREVIEW_DIR / f"{base}_preview.png")

                rows_A.append({"file_name": fA, "text": CAPTION_A})
                rows_B.append({"file_name": fB, "text": CAPTION_B})
            saved += 1
        print(f"  -> {saved} frames saved, {len(rows_A)} tile-pairs so far")

    for out_dir, rows in [(OUT_DIR_A, rows_A), (OUT_DIR_B, rows_B)]:
        with (out_dir / "metadata.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    print(f"\ntotal tile-pairs: {len(rows_A)}")
    print(f"chA -> {OUT_DIR_A}")
    print(f"chB -> {OUT_DIR_B}")
    if SAVE_PREVIEWS:
        print(f"previews -> {PREVIEW_DIR}")


if __name__ == "__main__":
    process_tiffs()