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
OUT_DIR = Path("data/microscopy_lora")
FRAMES_PER_VIDEO = 70
CAPTION = "a sks microscopy image of fluorescent transcriptional condensates"

def extract_one(video_path: Path, video_idx: int, out_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total <= 0:
        raise RuntimeError(f"Could not read {video_path}")
    idxs = np.linspace(0, n_total - 1, FRAMES_PER_VIDEO, dtype=int)
    rows = []
    for i, fidx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Pad short side to a minimum of 512 so random crops are always valid.
        h, w = frame.shape[:2]
        pad_h = max(0, 512 - h)
        pad_w = max(0, 512 - w)
        if pad_h or pad_w:
            frame = np.pad(
                frame,
                ((pad_h // 2, pad_h - pad_h // 2),
                 (pad_w // 2, pad_w - pad_w // 2),
                 (0, 0)),
                mode="constant", constant_values=0,
            )
        fname = f"v{video_idx:02d}_f{i:03d}.png"
        Image.fromarray(frame).save(out_dir / fname)
        rows.append({"file_name": fname, "text": CAPTION})
    cap.release()
    return rows

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for vi, vname in enumerate(VIDEOS):
        path = VIDEO_DIR / vname
        assert path.exists(), f"Missing {path}"
        rows = extract_one(path, vi, OUT_DIR)
        all_rows.extend(rows)
        print(f"video {vi}: {len(rows)} frames")
    # diffusers' image-folder loader wants metadata.jsonl
    with (OUT_DIR / "metadata.jsonl").open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Total frames: {len(all_rows)} -> {OUT_DIR}")

if __name__ == "__main__":
    main()