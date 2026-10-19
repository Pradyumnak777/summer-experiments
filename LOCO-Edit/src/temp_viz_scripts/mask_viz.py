'''
look at one hand-drawn gene burst mask against both channels, to see which channel it actually sits on

pick the split (SET) and an index (CELL_IDX). CELL_IDX indexes the burst subset (get_gene_burst_intensity,
like vae_guided.py), so every index is a frame with a non-empty hand-drawn mask.
saves one figure, on the raw crops (what the annotators saw, not the 128px cell-masked model input),
upscaled so the small masks show:
    top row:    burst mask | mask boundary on chA | mask boundary on chB
    bottom row:            | raw chA               | raw chB

saved as temp_viz_scripts/mask_viz/<set>_idx<CELL_IDX>.png

run from src/:
    python temp_viz_scripts/mask_viz.py
'''
import os
import sys
import numpy as np
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

#run from src/, so the VAE_disent and disentanglement_semantic_eval packages resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VAE_disent.data_utils import twoChannelDataset
from disentanglement_semantic_eval.gan_dci import get_gene_burst_intensity, load_burst_mask, burst_mask_path

SET      = "train"      #"train" or "validation"
CELL_IDX = 700                 #index into the burst subset of SET (frames with a hand-drawn burst mask)
CHA_DIR  = f"data/singlecell_chA_split/{SET}"
CHB_DIR  = f"data/singlecell_chB_split/{SET}"
MASK_DIR = "data/singlecell_mask"

SCALE    = 4                 #nearest upscale of the ~115-133px crops
LINE     = 2                 #boundary thickness in px (after upscaling)
COLOR    = (255, 0, 0)       #boundary color

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask_viz")


if __name__ == "__main__":
    #burst subset, like vae_guided.py. the train scan takes a while, so the subset indices are cached per set
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = f"{OUT_DIR}/burst_idx_{SET}.npz"
    if os.path.exists(cache):
        burst_idx = np.load(cache)["burst_idx"]
    else:
        _, _, burst_idx = get_gene_burst_intensity(dataset)
        np.savez(cache, burst_idx=burst_idx)

    i = burst_idx[CELL_IDX]                                  #CELL_IDX in the subset -> index in the full dataset
    chA_path, chB_path = dataset.chA_files[i], dataset.chB_files[i]
    #a video can have several annotation folders (hand_drawn_mask, hand_drawn_mask2, ..), this is the one picked
    mask_path = burst_mask_path(chB_path)
    mask_rel = Path(*mask_path.parts[-3:])                  #<annotation folder>/<cell folder>/<frame>.png

    #raw crop resolution: no resize, the hand-drawn mask is drawn on the same crop
    burst = load_burst_mask(chB_path, lambda img: torch.from_numpy(np.array(img))).numpy() > 0   #[H, W]
    chA = np.array(Image.open(chA_path).convert('L'))
    chB = np.array(Image.open(chB_path).convert('L'))

    def up(a):
        return np.kron(a, np.ones((SCALE, SCALE), dtype=a.dtype))

    burst_up = up(burst.astype(np.uint8)).astype(bool)
    #drawn just outside the mask, so the (small) masked spot itself stays visible
    boundary = ndimage.binary_dilation(burst_up, iterations=LINE) & ~burst_up

    def with_boundary(img):
        rgb = np.repeat(up(img)[:, :, None], 3, axis=2)
        rgb[boundary] = COLOR
        return rgb

    #2x3 grid, the bottom-left slot under the mask is left empty
    panels = {(0, 0): (burst_up, "hand-drawn burst mask"),
              (0, 1): (with_boundary(chA), "chA + mask boundary"),
              (0, 2): (with_boundary(chB), "chB + mask boundary"),
              (1, 1): (up(chA), "chA (raw)"),
              (1, 2): (up(chB), "chB (raw)")}
    fig, axes = plt.subplots(2, 3, figsize=(12, 9), layout="constrained")   #constrained leaves room for the titles
    for (r, c), ax in np.ndenumerate(axes):
        ax.axis("off")
        if (r, c) in panels:
            img, title = panels[(r, c)]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255 if img.dtype == np.uint8 else 1, interpolation="nearest")
            ax.set_title(title, fontsize=12)
    video, cell_frame = chB_path.name[:-len("_chB.png")].rsplit("_cell", 1)
    fig.suptitle(f"{SET} idx {CELL_IDX}  |  cell{cell_frame}\n{video}\nmask: {mask_rel}", fontsize=10)

    out = f"{OUT_DIR}/{SET}_idx{CELL_IDX}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"{SET}: burst subset index {CELL_IDX} of {len(burst_idx)} (full dataset index {i})")
    print(f"chA: {chA_path.name}")
    print(f"chB: {chB_path.name}")
    print(f"mask: {mask_path}")
    print(f"saved {out}")
