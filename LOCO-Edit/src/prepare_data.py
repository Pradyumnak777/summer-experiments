# prepare_data.py
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
OUT_DIR = Path("data/microscopy_lora_new")
CAPTION = "an image of cells in fluroscent microscopy"
TILE = 512  #square crop size


def tile_frame(frame, size=TILE):
    #cut a frame into non-overlapping size x size tiles. centered grid, leftover edges dropped.
    h, w = frame.shape[:2]
    ny, nx = h // size, w // size       # how many full tiles fit on each axis (708x2048 -> 1 x 4)
    
    '''
    below is for margins..
    off_y is nothing, as 2048 is d ivisible by 512. no margin
    for off_x, 708/512, so drop 98px on each side(left and righ)
    '''
    off_y = (h - ny * size) // 2        # center the grid so margins are dropped evenly
    off_x = (w - nx * size) // 2
    for iy in range(ny):
        for ix in range(nx):
            y, x = off_y + iy * size, off_x + ix * size
            yield frame[y:y+size, x:x+size] 


def process_vid(fps=25):
    OUT_DIR.mkdir(parents=True, exist_ok=True) #create it
    rows = []

    for vi, vid in enumerate(VIDEOS):
        vid_path = os.path.join(VIDEO_DIR, vid)

        #now load this video as frames based on frame rate, and save
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
                #slice each kept frame into 512x512 tiles -> ~4 images per frame here
                for ti, tile in enumerate(tile_frame(frame)):
                    fname = f"v{vi:02d}_f{saved:03d}_t{ti}.png"
                    Image.fromarray(tile).save(OUT_DIR / fname)
                    rows.append({"file_name": fname, "text": CAPTION})
                saved += 1
            read_idx += 1
        cap.release()
        print(f"{vid}: kept {saved} frames -> {saved * (len(rows)//max(saved,1))} tiles so far")

    with (OUT_DIR / "metadata.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"total: {len(rows)} tiles -> {OUT_DIR}")


if __name__ == "__main__":
    process_vid()