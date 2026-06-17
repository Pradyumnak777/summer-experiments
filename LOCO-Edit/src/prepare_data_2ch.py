# prepare_data_2ch.py
import os, json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

VIDEO_DIR = Path(__file__).resolve().parent  # videos are in the src folder
VIDEOS = [
    "bleach_corrected_DPHM_Sox2_MCP_halo549_snap646-01_MIP_merged (1)-1.avi",
    "bleach_corrected_DPHM_Sox2_MCP_halo549_snap646-02_MIP_merged_h264-1.mp4",
    "bleach_corrected_DPHM_Sox2_MCP_halo549_snap646-03_MIP_merged_h264-1.mp4",
]
OUT_DIR = Path("data/microscopy_lora_2ch")
PREVIEW_DIR = Path("data/microscopy_lora_2ch_previews")  # green/magenta merged PNGs for visual inspection only
CAPTION = "an image of cells in fluroscent microscopy"
TILE = 512  # square crop size
SAVE_PREVIEWS = True  # set False to skip preview generation (saves time/disk)


def tile_frame(frame, size=TILE):
    # cut a frame into non-overlapping size x size tiles. centered grid, leftover edges dropped.
    h, w = frame.shape[:2]
    ny, nx = h // size, w // size       # how many full tiles fit on each axis (708x2048 -> 1 x 4)

    '''
    below is for margins..
    off_y is nothing, as 2048 is divisible by 512. no margin
    for off_x, 708/512, so drop 98px on each side(left and right)
    '''
    off_y = (h - ny * size) // 2        # center the grid so margins are dropped evenly
    off_x = (w - nx * size) // 2
    for iy in range(ny):
        for ix in range(nx):
            y, x = off_y + iy * size, off_x + ix * size
            yield frame[y:y+size, x:x+size]


def split_channels(tile):
    # tile is RGB uint8 [H,W,3]. green LUT lives in G, magenta LUT lives in R(=B).
    # peel back into two raw grayscale intensity maps. NO color anymore.
    R, G, B = tile[..., 0], tile[..., 1], tile[..., 2]
    chA = G      # green marker intensity (e.g. Sox2 / halo549)
    chB = R      # magenta marker intensity (e.g. MCP / snap646). R approx B for green/magenta scheme.
    return chA, chB


def merge_preview(chA, chB):
    # chA, chB: [H,W] uint8 grayscale arrays. reconstructs the familiar green/magenta composite.
    # this is ONLY for human inspection — never feed this to the model.
    rgb = np.dstack([chB, chA, chB]).astype(np.uint8)  # magenta=R&B, green=G
    return rgb


def process_vid(fps=25):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PREVIEWS:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for vi, vid in enumerate(VIDEOS):
        vid_path = os.path.join(VIDEO_DIR, vid)

        # load this video as frames based on frame rate, and save
        cap = cv2.VideoCapture(vid_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        step = max(1, round(native_fps / fps))  # e.g. native=50, target=25 -> keep every 2nd frame

        read_idx, saved = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if read_idx % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2 is BGR by default
                # slice each kept frame into 512x512 tiles -> ~4 images per frame here
                for ti, tile in enumerate(tile_frame(frame)):
                    chA, chB = split_channels(tile)

                    # two grayscale PNGs per tile. each is a valid viewable B/W image (mode 'L').
                    base = f"v{vi:02d}_f{saved:03d}_t{ti}"
                    fA   = f"{base}_chA.png"   # green marker
                    fB   = f"{base}_chB.png"   # magenta marker
                    Image.fromarray(chA, mode='L').save(OUT_DIR / fA)
                    Image.fromarray(chB, mode='L').save(OUT_DIR / fB)

                    # optional: save green/magenta merged preview for visual sanity-checking
                    if SAVE_PREVIEWS:
                        preview = merge_preview(chA, chB)
                        Image.fromarray(preview).save(PREVIEW_DIR / f"{base}_preview.png")

                    rows.append({"chA": fA, "chB": fB, "text": CAPTION})
                saved += 1
            read_idx += 1
        cap.release()
        print(f"{vid}: kept {saved} frames -> {len(rows)} tile-pairs so far")

    with (OUT_DIR / "metadata.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"total: {len(rows)} tile-pairs -> {OUT_DIR}")
    if SAVE_PREVIEWS:
        print(f"previews -> {PREVIEW_DIR}")


if __name__ == "__main__":
    process_vid()