'''
column view of the conditioned 2D model (new method), focused on the hand-drawn gene burst:
    chA = denoised target (queries), chB = condition (the 64 key tokens, 8x8 grid over chB)
    chA is generated from pure noise with chB as the condition (how the model was trained and is used,
    as in infer.py), attention is recorded over the whole trajectory (generate_and_collect from
    cross_attn_viz.py, unchanged). --start_t N instead starts from the real chA noised to step N

one figure per chB KEY token that lies ON or right next to the hand-drawn burst mask:
    chB (key) + token box | hand-drawn mask + token box | mask overlaid on chB + token box |
    that token's attention received from chA (over the generated chA) | generated chA (plain)

run from src/:
    python conditioned_diffusion_2d/cross_attn_burst.py --split val --index 1020
    python conditioned_diffusion_2d/cross_attn_burst.py --split train --index 31607   #same cell as the old-method run
'''
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))
from conditioned_diffusion_2d.data_utils import PairedCropDataset, recording_id
from conditioned_diffusion_2d.infer import CKPT_DIR, load_model
from conditioned_diffusion_2d.cross_attn_viz import (LAYER_GROUPS, IMG_SIZE, SRC_LABEL, TGT_LABEL,
                                                     generate_and_collect, to_img, overlay_on)
from disentanglement_semantic_eval.gan_dci import pick_burst_cell_dirs

ANNOT_DIR = SRC_ROOT / "data/annotations"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--index", type=int, default=1020, help="crop index within --split")
    p.add_argument("--ckpt", default=str(CKPT_DIR / "model_ema_epoch90.pt"))
    p.add_argument("--layer", default="up16", help="layer group (" + ", ".join(LAYER_GROUPS) + ") or names")
    p.add_argument("--start_t", type=int, default=None,
                   help="start from the real chA noised to this step instead of pure noise (default: pure noise)")
    p.add_argument("--num_inference_steps", type=int, default=1000)
    p.add_argument("--near_ring", type=int, default=1, help="also take key tokens this many tokens from the mask")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=str(SRC_ROOT / "conditioned_diffusion_2d/cross_attn_maps"))
    p.add_argument("--device", default="cuda:9")
    return p.parse_args()


def load_burst_mask(base):
    #hand-drawn burst mask for this crop, resized to 128x128 the same way the cell mask is
    rec = recording_id(base)
    cell, frame = base.rsplit("_cell", 1)[1].split("_f")
    d = pick_burst_cell_dirs(ANNOT_DIR / rec).get(int(cell))
    if d is None:
        return None, None
    path = d / f"{d.name}_f{frame}.png"
    if not path.exists():
        return None, None
    m = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
    return np.array(m) > 0, d.parent.name


def pick_key_tokens(burst, side, near_ring):
    #key tokens whose cell overlaps the mask ("on"), plus neighbours up to near_ring away ("near")
    cell = IMG_SIZE // side
    on = burst.reshape(side, cell, side, cell).any(axis=(1, 3))
    near = on.copy()
    for _ in range(near_ring):
        padded = np.pad(near, 1)
        near = np.zeros_like(near)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                near |= padded[1 + dy:1 + dy + side, 1 + dx:1 + dx + side]
    return [(y, x, "on" if on[y, x] else "near") for y in range(side) for x in range(side) if near[y, x]]


def mask_overlay(base_gray, burst):
    #burst pixels tinted red on top of the grayscale image
    rgb = np.stack([base_gray] * 3, axis=-1)
    rgb[burst] = 0.45 * rgb[burst] + 0.55 * np.array([1.0, 0.15, 0.15])
    return rgb


def token_box(ax, y, x, cell, color="yellow"):
    ax.add_patch(patches.Rectangle((x * cell, y * cell), cell, cell, linewidth=2, edgecolor=color, facecolor="none"))


def main():
    args = parse_args()
    layer_names = LAYER_GROUPS.get(args.layer, args.layer.split(","))

    dataset = PairedCropDataset(split=args.split)
    sample = dataset[args.index]
    burst, mask_src = load_burst_mask(sample["name"])
    if burst is None or not burst.any():
        raise SystemExit(f"{args.split} #{args.index} ({sample['name']}) has no hand-drawn burst mask in this frame")

    device = torch.device(args.device)
    model = load_model(Path(args.ckpt), device)
    cond = sample["cond"][None].to(device)
    real = sample["target"][None].to(device)
    gen, per_query, src_side, tgt_side, n_steps = generate_and_collect(
        model, layer_names, cond, real, args.start_t, args.num_inference_steps, args.seed)
    start_label = "pure noise" if args.start_t is None else f"T={args.start_t}"
    start_tag = "noise" if args.start_t is None else f"startT{args.start_t}"

    src_np, tgt_real_np, tgt_gen_np = to_img(cond), to_img(real), to_img(gen)
    key_cell = IMG_SIZE / src_side

    sample_dir = (Path(args.out_dir) / f"{args.split}_{args.index}" /
                  f"burst_keys_{Path(args.ckpt).stem}_{args.layer.replace(',', '+')}_{start_tag}")
    tok_dir = sample_dir / "tokens"
    tok_dir.mkdir(parents=True, exist_ok=True)

    #context: chB (keys) | real chA | generated chA
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, (src_np, tgt_real_np, tgt_gen_np),
                              (f"{SRC_LABEL} (condition / keys)", f"{TGT_LABEL} real", f"{TGT_LABEL} generated from {start_label}")):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(sample_dir / "triplet.png", dpi=120); plt.close(fig)

    #which key tokens were picked, on the mask overlay
    tokens = pick_key_tokens(burst, src_side, args.near_ring)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask_overlay(src_np, burst))
    for (y, x, kind) in tokens:
        token_box(ax, y, x, key_cell, color="yellow" if kind == "on" else "cyan")
    ax.set_title(f"picked {SRC_LABEL} key tokens\n(yellow = on the burst mask, cyan = next to it)"); ax.axis("off")
    fig.tight_layout(); fig.savefig(sample_dir / "picked_tokens.png", dpi=120); plt.close(fig)

    #column view: fix a chB key, heatmap over chA positions. one colour scale over all key tokens (as in cross_attn_viz.py)
    per_token = per_query.T                                                   # [chB tokens, chA positions]
    vmin, vmax = per_token.min().item(), per_token.max().item()
    def shared_normalize(m):
        return ((m - vmin) / (vmax - vmin + 1e-8)).numpy()

    for (y, x, kind) in tokens:
        heat = shared_normalize(per_token[y * src_side + x]).reshape(tgt_side, tgt_side)
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))
        axes[0].imshow(src_np, cmap="gray", vmin=0, vmax=1); token_box(axes[0], y, x, key_cell)
        axes[0].set_title(f"{SRC_LABEL} (key), token ({y},{x})")
        axes[1].imshow(burst, cmap="gray"); token_box(axes[1], y, x, key_cell)
        axes[1].set_title("hand-drawn burst mask")
        axes[2].imshow(mask_overlay(src_np, burst)); token_box(axes[2], y, x, key_cell)
        axes[2].set_title(f"burst mask on {SRC_LABEL}")
        axes[3].imshow(overlay_on(tgt_gen_np, heat))
        axes[3].set_title(f"attention this {SRC_LABEL} token receives from {TGT_LABEL}\n(from {start_label}, summed over {n_steps} steps)")
        axes[4].imshow(tgt_gen_np, cmap="gray", vmin=0, vmax=1)
        axes[4].set_title(f"{TGT_LABEL} generated (plain)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{args.split} sample {args.index} | key token ({y},{x}) {kind} the burst mask", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.9]) #leave room for the suptitle above the panel titles
        fig.savefig(tok_dir / f"token_{y:02d}_{x:02d}_{kind}.png", dpi=110)
        plt.close(fig)

    n_on = sum(k == "on" for _, _, k in tokens)
    print(f"crop   : {args.split} #{args.index} {sample['name']} (mask from {mask_src})")
    print(f"layers : {', '.join(layer_names)} ({tgt_side}x{tgt_side} {TGT_LABEL} queries x {src_side}x{src_side} "
          f"{SRC_LABEL} keys), {n_steps} steps from {start_label}")
    print(f"saved  : {sample_dir}  ({len(tokens)} token figures: {n_on} on, {len(tokens) - n_on} near)")


if __name__ == "__main__":
    main()
