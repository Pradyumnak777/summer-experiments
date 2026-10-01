'''
vsualizing the cross attention layers here..

generate chA for one chB crop with the conditioned 2D model while recording cross-attention, laid
out like infer_cross_attn_v2.py. in every cross-attention layer the query comes from the chA
features being denoised (one per location of that layer's grid) and the keys are the 64 chB tokens
(8x8 grid over chB). so the recorded map A is [chA positions, chB tokens], averaged over heads, the
chosen layers, and the whole denoising trajectory.

--view row:    fix a chA position, heatmap over chB   -> grid.png, points/point_yy_xx.png
--view column: fix a chB token, heatmap over the generated chA (the SD / DAAM "map for token X"
               convention) -> grid_flipped.png, points_flipped/token_yy_xx.png
both also save triplet.png (chB | real chA | generated chA). heatmaps share one colour scale.

generation starts from pure noise by default (real generation, as in infer.py). --start_t N instead
starts from the real chA noised to step N, like infer_cross_attn_v2.py's START_T.

--layer takes one layer name (see --list_layers), a comma-separated list, or a group: up16 (default),
up8, down16, down8, mid. layers in one run must share a resolution.

run from src/ (on a GPU training isn't using):
    python conditioned_diffusion_2d/cross_attn_viz.py --list_layers
    CUDA_VISIBLE_DEVICES=4 python conditioned_diffusion_2d/cross_attn_viz.py --split val --index 100 --view row
    CUDA_VISIBLE_DEVICES=4 python conditioned_diffusion_2d/cross_attn_viz.py --split val --index 100 --view column \
        --layer down_blocks.3.attentions.0 --start_t 400
'''
import argparse, math, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDPMScheduler, Transformer2DModel
from diffusers.models.attention_processor import AttnProcessor

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))
from conditioned_diffusion_2d.data_utils import PairedCropDataset
from conditioned_diffusion_2d.infer import CKPT_DIR, NUM_TRAIN_TIMESTEPS, load_model
from conditioned_diffusion_2d.model import build_unet

IMG_SIZE = 128
SRC_LABEL, TGT_LABEL = "chB", "chA"
LAYER_GROUPS = {
    "up16":   ["up_blocks.2.attentions.0", "up_blocks.2.attentions.1", "up_blocks.2.attentions.2"],
    "up8":    ["up_blocks.1.attentions.0", "up_blocks.1.attentions.1", "up_blocks.1.attentions.2"],
    "down16": ["down_blocks.3.attentions.0", "down_blocks.3.attentions.1"],
    "down8":  ["down_blocks.4.attentions.0", "down_blocks.4.attentions.1"],
    "mid":    ["mid_block.attentions.0"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--index", type=int, default=0, help="crop index within --split (see infer.py --list)")
    p.add_argument("--name", default=None, help="crop name instead of --index (quote it: names contain a space)")
    p.add_argument("--ckpt", default=str(CKPT_DIR / "model_ema_epoch90.pt"))
    p.add_argument("--view", choices=["row", "column"], default="row")
    p.add_argument("--layer", default="up16", help="layer name, comma-separated names, or a group: "
                                                   + ", ".join(LAYER_GROUPS))
    p.add_argument("--start_t", type=int, default=None,
                   help="start from the real chA noised to this step instead of pure noise")
    p.add_argument("--num_inference_steps", type=int, default=1000, help="DDPM steps over the full schedule")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_point_figs", action="store_true", help="skip the per-point / per-token figures")
    p.add_argument("--out_dir", default=str(SRC_ROOT / "conditioned_diffusion_2d/cross_attn_maps"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--list_layers", action="store_true", help="print the cross-attention layers and exit")
    return p.parse_args()


def cross_attn_layers(unet):
    return [name for name, m in unet.named_modules() if isinstance(m, Transformer2DModel)]


class AttentionRecorder:
    '''
    switch one Attention module to diffusers' eager processor (same math as the default fused one,
    but it computes the softmax weights explicitly) and keep a running sum of those weights.
    '''
    def __init__(self, attn):
        self.attn, self.total, self.calls = attn, None, 0
        self._orig_processor = attn.processor
        attn.set_processor(AttnProcessor())
        compute = attn.get_attention_scores

        def record(query, key, attention_mask=None):
            probs = compute(query, key, attention_mask)                      # [heads, n_query, n_key]
            m = probs.detach().float().reshape(-1, attn.heads, *probs.shape[1:]).mean(1)[0]
            self.total = m if self.total is None else self.total + m
            self.calls += 1
            return probs

        attn.get_attention_scores = record

    def mean(self):
        return self.total / self.calls

    def remove(self):
        del self.attn.get_attention_scores                                    #back to the class method
        self.attn.set_processor(self._orig_processor)


@torch.no_grad()
def generate_and_collect(model, layer_names, cond, real, start_t, num_inference_steps, seed):
    recorders = [AttentionRecorder(model.unet.get_submodule(n).transformer_blocks[0].attn2) for n in layer_names]
    scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
    scheduler.set_timesteps(num_inference_steps)
    torch.manual_seed(seed)
    noise = torch.randn_like(real)
    if start_t is None:
        timesteps, x = scheduler.timesteps, noise
    else:
        timesteps = scheduler.timesteps[scheduler.timesteps <= start_t]
        x = scheduler.add_noise(real, noise, timesteps[:1])                  #real chA, noised to start_t

    tokens = model.encode(cond)
    for t in timesteps:
        noise_pred = model(x, t.reshape(1).to(x.device), tokens=tokens)
        x = scheduler.step(noise_pred, t, x).prev_sample

    maps = [r.mean() for r in recorders]
    for r in recorders:
        r.remove()
    if len({m.shape for m in maps}) > 1:
        raise SystemExit(f"layers {layer_names} have different resolutions {[tuple(m.shape) for m in maps]}; "
                         f"pick layers from one group")
    per_query = torch.stack(maps).mean(0).cpu()                               # [chA positions, chB tokens]
    return x, per_query, int(math.sqrt(per_query.shape[1])), int(math.sqrt(per_query.shape[0])), len(timesteps)


def to_img(t):
    return (t[0, 0] * 0.5 + 0.5).clamp(0, 1).cpu().numpy()

def upsample(m, size=IMG_SIZE):
    t = torch.tensor(m)[None, None]
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()

def overlay_on(base_gray, heat):
    heat_up = upsample(heat)
    gray_rgb = np.stack([base_gray] * 3, axis=-1)
    return (0.5 * cm.jet(heat_up)[..., :3] + 0.5 * gray_rgb).clip(0, 1)


def save_triplet(sample_dir, src_np, tgt_real_np, tgt_gen_np, start_label):
    plt.imsave(f"{sample_dir}/{SRC_LABEL}.png", src_np, cmap="gray")
    plt.imsave(f"{sample_dir}/{TGT_LABEL}_real.png", tgt_real_np, cmap="gray")
    plt.imsave(f"{sample_dir}/{TGT_LABEL}_generated.png", tgt_gen_np, cmap="gray")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(src_np, cmap="gray"); axes[0].set_title(f"{SRC_LABEL} (source)"); axes[0].axis("off")
    axes[1].imshow(tgt_real_np, cmap="gray"); axes[1].set_title(f"{TGT_LABEL} (real)"); axes[1].axis("off")
    axes[2].imshow(tgt_gen_np, cmap="gray"); axes[2].set_title(f"{TGT_LABEL} (generated, {start_label})"); axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/triplet.png", dpi=120)
    plt.close(fig)


def save_row_view(sample_dir, src_np, tgt_gen_np, per_query, src_side, tgt_side, point_figs):
    '''fix a chA position (row of A), heatmap over chB tokens, drawn on chB.'''
    vmin, vmax = per_query.min().item(), per_query.max().item()
    def shared_normalize(m):
        return ((m - vmin) / (vmax - vmin + 1e-8)).numpy()

    fig, axes = plt.subplots(tgt_side, tgt_side, figsize=(tgt_side * 1.3, tgt_side * 1.3))
    for y in range(tgt_side):
        for x in range(tgt_side):
            m = shared_normalize(per_query[y * tgt_side + x]).reshape(src_side, src_side)
            axes[y, x].imshow(overlay_on(src_np, m))
            axes[y, x].set_xticks([]); axes[y, x].set_yticks([])
    fig.suptitle(f"per-{TGT_LABEL}-position attn over {SRC_LABEL}, summed over full trajectory, "
                 f"one cell per {TGT_LABEL} position ({tgt_side}x{tgt_side}={tgt_side * tgt_side} total)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))   #leave room for the title
    fig.savefig(f"{sample_dir}/grid.png", dpi=150)
    plt.close(fig)

    if not point_figs:
        return
    points_dir = Path(sample_dir) / "points"
    points_dir.mkdir(exist_ok=True)
    cell_size = IMG_SIZE / tgt_side
    for y in range(tgt_side):
        for x in range(tgt_side):
            m = shared_normalize(per_query[y * tgt_side + x]).reshape(src_side, src_side)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(tgt_gen_np, cmap="gray")
            axes[0].add_patch(patches.Rectangle((x * cell_size, y * cell_size), cell_size, cell_size,
                                                linewidth=2, edgecolor="red", facecolor="none"))
            axes[0].set_title(f"{TGT_LABEL} position ({y},{x})"); axes[0].axis("off")
            axes[1].imshow(overlay_on(src_np, m))
            axes[1].set_title(f"this {TGT_LABEL} position attends to {SRC_LABEL}\n(summed over full trajectory)")
            axes[1].axis("off")
            axes[2].imshow(src_np, cmap="gray"); axes[2].set_title(f"{SRC_LABEL} (plain)"); axes[2].axis("off")
            fig.tight_layout()
            fig.savefig(points_dir / f"point_{y:02d}_{x:02d}.png", dpi=100)
            plt.close(fig)


def save_column_view(sample_dir, src_np, tgt_gen_np, per_query, src_side, tgt_side, point_figs):
    '''fix a chB token (column of A), heatmap over chA positions, drawn on the generated chA.'''
    per_token = per_query.T                                                   # [chB tokens, chA positions]
    vmin, vmax = per_token.min().item(), per_token.max().item()
    def shared_normalize(m):
        return ((m - vmin) / (vmax - vmin + 1e-8)).numpy()

    fig, axes = plt.subplots(src_side, src_side, figsize=(src_side * 1.3, src_side * 1.3))
    for y in range(src_side):
        for x in range(src_side):
            m = shared_normalize(per_token[y * src_side + x]).reshape(tgt_side, tgt_side)
            axes[y, x].imshow(overlay_on(tgt_gen_np, m))
            axes[y, x].set_xticks([]); axes[y, x].set_yticks([])
    fig.suptitle(f"per-{SRC_LABEL}-token attn received from {TGT_LABEL}, summed over full trajectory, "
                 f"one cell per {SRC_LABEL} token ({src_side}x{src_side}={src_side * src_side} total)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))   #leave room for the title
    fig.savefig(f"{sample_dir}/grid_flipped.png", dpi=150)
    plt.close(fig)

    if not point_figs:
        return
    points_dir = Path(sample_dir) / "points_flipped"
    points_dir.mkdir(exist_ok=True)
    cell_size = IMG_SIZE / src_side
    for y in range(src_side):
        for x in range(src_side):
            m = shared_normalize(per_token[y * src_side + x]).reshape(tgt_side, tgt_side)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(src_np, cmap="gray")
            axes[0].add_patch(patches.Rectangle((x * cell_size, y * cell_size), cell_size, cell_size,
                                                linewidth=2, edgecolor="red", facecolor="none"))
            axes[0].set_title(f"{SRC_LABEL} token ({y},{x})"); axes[0].axis("off")
            axes[1].imshow(overlay_on(tgt_gen_np, m))
            axes[1].set_title(f"this {SRC_LABEL} token's attention over {TGT_LABEL}\n(summed over full trajectory)")
            axes[1].axis("off")
            axes[2].imshow(tgt_gen_np, cmap="gray"); axes[2].set_title(f"{TGT_LABEL} (plain)"); axes[2].axis("off")
            fig.tight_layout()
            fig.savefig(points_dir / f"token_{y:02d}_{x:02d}.png", dpi=100)
            plt.close(fig)


def main():
    args = parse_args()
    all_layers = cross_attn_layers(build_unet())
    if args.list_layers:
        for name in all_layers:
            print(name)
        print("groups: " + ", ".join(f"{g} = {', '.join(v)}" for g, v in LAYER_GROUPS.items()))
        return
    layer_names = LAYER_GROUPS.get(args.layer, args.layer.split(","))
    unknown = [n for n in layer_names if n not in all_layers]
    if unknown:
        raise SystemExit(f"not cross-attention layers: {unknown}; see --list_layers")

    dataset = PairedCropDataset(split=args.split)
    if args.name is not None and args.name not in dataset.bases:
        raise SystemExit(f"no crop named {args.name!r} in the {args.split} split (quote names; see infer.py --list)")
    index = dataset.bases.index(args.name) if args.name else args.index
    sample = dataset[index]

    device = torch.device(args.device)
    model = load_model(Path(args.ckpt), device)
    cond = sample["cond"][None].to(device)
    real = sample["target"][None].to(device)
    gen, per_query, src_side, tgt_side, n_steps = generate_and_collect(
        model, layer_names, cond, real, args.start_t, args.num_inference_steps, args.seed)

    start_label = "pure noise" if args.start_t is None else f"START_T={args.start_t}"
    ckpt = Path(args.ckpt)
    sample_dir = (Path(args.out_dir) / f"{args.split}_{index}" /
                  f"{ckpt.stem}_{args.layer.replace(',', '+')}_{'noise' if args.start_t is None else f'startT{args.start_t}'}")
    sample_dir.mkdir(parents=True, exist_ok=True)

    src_np, tgt_real_np, tgt_gen_np = to_img(cond), to_img(real), to_img(gen)
    save_triplet(sample_dir, src_np, tgt_real_np, tgt_gen_np, start_label)
    view = save_row_view if args.view == "row" else save_column_view
    view(sample_dir, src_np, tgt_gen_np, per_query, src_side, tgt_side, not args.no_point_figs)

    print(f"crop   : {args.split} #{index} {sample['name']}")
    print(f"layers : {', '.join(layer_names)} ({tgt_side}x{tgt_side} {TGT_LABEL} positions x "
          f"{src_side}x{src_side} {SRC_LABEL} tokens), {n_steps} steps from {start_label}")
    print(f"saved  : {sample_dir}")


if __name__ == "__main__":
    main()
