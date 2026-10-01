'''
this script reads the raw per-cell tiffs and writes playable mp4s per cell, instead of exploding
them into per frame pngs the way extract_frames does. five videos come out per cell:

    chA_videos_{fps}/         chB_videos_{fps}/         raw channel
    chA_videos_{fps}_masked/  chB_videos_{fps}_masked/  background zeroed by the per frame mask
    mask_videos_{fps}/                                  the mask itself, encoded losslessly

these are a *viewing* artifact, not a training one. anything that trains reads the lossless pngs
from extract_frames and applies the mask at load time (VAE_disent/data_utils.py), which keeps the
fill value and any mask dilation changeable without re-encoding anything. h264 is lossy, so never
feed the frames decoded out of these back into a model

normalisation matches extract_frames exactly: percentile clip over the whole T stack, not per frame,
so the video keeps the real intensity changes over time. a per frame rescale would flatten exactly
the bursting behaviour these movies exist to show

frame rate: nothing in the tiffs records one, see the VIDEO_FPS note in config.py
'''

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tifffile
from PIL import Image

from .config import (
    CH_A_INDEX,
    CH_B_INDEX,
    MASK_CRF,
    MASK_FILL,
    MASK_RESAMPLE,
    RAW_DIR,
    VIDEO_CRF,
    VIDEO_FPS,
    VIDEO_RESAMPLE,
    VIDEO_SIZE,
    VIDEO_WORKERS,
    video_out_dir,
)
from .extract_frames import channel_to_uint8_percentile, find_cell_files

RESAMPLE_FILTERS = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}


def find_ffmpeg():
    #the system ffmpeg is the one with libx264 built in, imageio ships its own as a fallback
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("no ffmpeg binary found, install ffmpeg or imageio-ffmpeg")


def resize_stack(stack_u8, size, resample):
    """resize a (T,H,W) uint8 stack frame by frame, through pil so it matches torchvision"""
    if stack_u8.shape[1] == size and stack_u8.shape[2] == size:
        return stack_u8
    flt = RESAMPLE_FILTERS[resample]
    return np.stack([np.array(Image.fromarray(f).resize((size, size), flt)) for f in stack_u8])


def encode(frames_u8, out_path, ffmpeg, fps, crf):
    """pipe a (T,H,W) uint8 stack straight into ffmpeg as rawvideo and write an h264 mp4"""
    t, h, w = frames_u8.shape

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-framerate", str(fps),
        "-i", "-",
        "-an",
        #a no-op at an even VIDEO_SIZE, it is here so an odd size does not break yuv420p
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(frames_u8).tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {out_path}\n{proc.stderr.decode(errors='replace')}")


def encode_cell(job, ffmpeg, fps, size, crf, mask_crf):
    """read one cell tiff plus its mask and write all five videos, returns a line for the log"""
    base_tag, tif_path, mask_path, outs = job

    #read straight through rather than going via fix_tiffs.load_and_fix, that one writes a corrected
    #tiff copy out to disk as a side effect and we do not need one to encode
    data = tifffile.imread(tif_path)
    if data.ndim != 4:
        return False, f"SKIP  {tif_path}  (expected 4D TCYX, got shape {data.shape})"
    frames, channels, h, w = data.shape
    if channels < 2:
        return False, f"SKIP  {tif_path}  (only {channels} channel(s))"

    #resize the mask separately with nearest, exactly like data_utils.py, so it stays strictly binary
    #rather than picking up interpolated edge values that would bleed the crop outline
    mask_u8 = None
    if mask_path is not None:
        raw_mask = tifffile.imread(mask_path)
        if raw_mask.shape[0] != frames or raw_mask.shape[-2:] != (h, w):
            return False, f"SKIP  {base_tag}  (mask shape {raw_mask.shape} vs image {data.shape})"
        mask_u8 = resize_stack((raw_mask > 0).astype(np.uint8) * 255, size, MASK_RESAMPLE)

    for ch_index, channel in ((CH_A_INDEX, "A"), (CH_B_INDEX, "B")):
        stack = channel_to_uint8_percentile(data[:, ch_index])
        stack = resize_stack(stack, size, VIDEO_RESAMPLE)
        encode(stack, outs[channel], ffmpeg, fps, crf)

        if mask_u8 is not None:
            masked = np.where(mask_u8 > 0, stack, np.uint8(MASK_FILL))
            encode(masked, outs[f"{channel}_masked"], ffmpeg, fps, crf)

    if mask_u8 is not None:
        encode(mask_u8, outs["mask"], ffmpeg, fps, mask_crf)

    kept = float((mask_u8 > 0).mean()) if mask_u8 is not None else float("nan")
    return True, (
        f"{base_tag}: {frames} frames, {h}x{w} -> {size}x{size} @ {fps} fps "
        f"({frames / fps:.1f}s), mask keeps {kept:.1%}"
    )


def build_out_dirs(fps):
    return {
        "A": video_out_dir("A", fps),
        "B": video_out_dir("B", fps),
        "A_masked": video_out_dir("A", fps, masked=True),
        "B_masked": video_out_dir("B", fps, masked=True),
        "mask": video_out_dir("mask", fps),
    }


def collect_jobs(fps, overwrite=False, limit=None):
    dirs = build_out_dirs(fps)
    jobs, skipped, no_mask = [], 0, 0

    for subject_dir in sorted(d for d in RAW_DIR.iterdir() if d.is_dir()):
        for cell_id, tif_path, mask_path in find_cell_files(subject_dir):
            base_tag = f"{subject_dir.name}_cell{cell_id}"
            outs = {
                "A": dirs["A"] / f"{base_tag}_chA.mp4",
                "B": dirs["B"] / f"{base_tag}_chB.mp4",
                "A_masked": dirs["A_masked"] / f"{base_tag}_chA_masked.mp4",
                "B_masked": dirs["B_masked"] / f"{base_tag}_chB_masked.mp4",
                "mask": dirs["mask"] / f"{base_tag}_mask.mp4",
            }
            if mask_path is None:
                no_mask += 1
            expected = [v for k, v in outs.items()
                        if mask_path is not None or k in ("A", "B")]
            if not overwrite and all(p.exists() for p in expected):
                skipped += 1
                continue
            jobs.append((base_tag, tif_path, mask_path, outs))
            if limit is not None and len(jobs) >= limit:
                return jobs, skipped, no_mask, dirs
    return jobs, skipped, no_mask, dirs


def process_all(fps=VIDEO_FPS, size=VIDEO_SIZE, crf=VIDEO_CRF, mask_crf=MASK_CRF,
                overwrite=False, limit=None, workers=VIDEO_WORKERS):
    ffmpeg = find_ffmpeg()
    jobs, skipped, no_mask, dirs = collect_jobs(fps, overwrite=overwrite, limit=limit)
    for out_dir in dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(jobs)} cells to encode ({skipped} already done), {workers} workers")
    if no_mask:
        print(f"note: {no_mask} cells have no mask, they get raw videos only")
    print()

    written, failed = 0, 0
    #ffmpeg does the actual work in a subprocess so threads are enough here, the gil is released
    #for the whole duration of each subprocess.run
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(encode_cell, j, ffmpeg, fps, size, crf, mask_crf): j for j in jobs}
        for done in as_completed(futures):
            base_tag = futures[done][0]
            try:
                ok, message = done.result()
            except Exception as exc:
                failed += 1
                print(f"FAIL  {base_tag}: {exc}")
                continue
            written += ok
            failed += not ok
            print(f"[{written + failed}/{len(jobs)}] {message}")

    return written, skipped, failed, dirs


def main():
    parser = argparse.ArgumentParser(
        description="encode the raw per-cell tiffs into raw, masked and mask mp4s"
    )
    parser.add_argument("--fps", type=float, default=VIDEO_FPS,
                        help=f"playback frame rate, also names the output dirs "
                             f"(default {VIDEO_FPS}, see config.py)")
    parser.add_argument("--size", type=int, default=VIDEO_SIZE,
                        help=f"square output size (default {VIDEO_SIZE})")
    parser.add_argument("--crf", type=int, default=VIDEO_CRF,
                        help=f"x264 quality for the image channels, lower is better "
                             f"(default {VIDEO_CRF})")
    parser.add_argument("--mask-crf", type=int, default=MASK_CRF,
                        help=f"x264 quality for the mask videos (default {MASK_CRF}, lossless)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-encode cells that already have all their videos")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many cells, for a quick look")
    parser.add_argument("--workers", type=int, default=VIDEO_WORKERS,
                        help=f"parallel encodes (default {VIDEO_WORKERS}, nproc is {os.cpu_count()})")
    args = parser.parse_args()

    written, skipped, failed, dirs = process_all(
        fps=args.fps, size=args.size, crf=args.crf, mask_crf=args.mask_crf,
        overwrite=args.overwrite, limit=args.limit, workers=args.workers,
    )

    print(f"\ncells encoded: {written}  (already done: {skipped}, failed/skipped: {failed})")
    for key, out_dir in dirs.items():
        print(f"{key:<9} -> {out_dir}")


if __name__ == "__main__":
    main()
