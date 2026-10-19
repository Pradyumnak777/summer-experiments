'''
row view of the old cross-attention model (decoupled cross-attn adapter on the ddpm unet, see
cross_attn_modules.py), focused on the hand-drawn gene burst:
    AtoB: chA = source/keys (encoder), chB = denoiser/queries
    the real chB is noised to START_T and denoised with chA as the condition, attention is
    recorded over the whole trajectory (same setup as infer_cross_attn_v2.py, VIEW_MODE = "row")

one figure per chB query token that lies ON or right next to the hand-drawn burst mask:
    chB (query) + token box | hand-drawn mask + token box | mask overlaid on chB + token box |
    that token's attention over chA | chA (plain)

run from src/:
    python infer_cross_attn_burst.py
'''
import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from diffusers import DDPMScheduler

from VAE_disent.diffusion_model import build_unet
from VAE_disent.data_utils import twoChannelDataset
from cross_attn_modules import ChannelEncoder, install_cross_attn, set_tokens, set_store_attn
from disentanglement_semantic_eval.gan_dci import pick_burst_cell_dirs

import debugpy
# debugpy.listen(("127.0.0.1", 5678)) #127.0.0.1, cz only local host can talk to the port
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")

DEVICE = torch.device("cuda:9")
SET    = "validation"
CHA_DIR  = f"data/singlecell_chA_new_split/{SET}"
CHB_DIR  = f"data/singlecell_chB_new_split/{SET}"
MASK_DIR = f"data/singlecell_mask_new_split/{SET}"
ANNOT_DIR = "data/annotations"

#AtoB: encoder = chA (keys), denoiser = chB (queries). same checkpoints as infer_cross_attn_v2.py
TGT_CKPT = "diffusion_checkpoints/ddpm_chB_128_masked/unet_ema_epoch40.pt"   #denoiser (chB)
SRC_CKPT = "diffusion_checkpoints/ddpm_chA_128_masked/unet_ema_epoch40.pt"   #encoder  (chA)
ADAPTER  = "cross_attn_checkpoints/AtoB_v2/adapter_ema_epoch49.pt"
TGT_IDX, SRC_IDX = 1, 0
SRC_LABEL, TGT_LABEL = "chA", "chB"

IMG_SIZE            = 128
TOKEN_DIM           = 256
STOP_BLOCK          = 3
START_T             = 400
NUM_INFERENCE_STEPS = 1000
SAMPLE_INDICES = [1646]   #val crop with a burst clearly visible in chB (video 86, cell 6, f042)
NEAR_RING      = 1        #also take query tokens up to this many tokens away from the mask
SEED           = 0
OUT_DIR        = f"cross_attn/AtoB_v2/{SET}/burst_tokens_startT_{START_T}"
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)

#----- models (identical to infer_cross_attn_v2.py) -----
denoiser = build_unet(IMG_SIZE, channels=1).to(DEVICE)
denoiser.load_state_dict(torch.load(TGT_CKPT, map_location=DEVICE))
denoiser.requires_grad_(False)

src_unet = build_unet(IMG_SIZE, channels=1)
src_unet.load_state_dict(torch.load(SRC_CKPT, map_location="cpu"))
encoder = ChannelEncoder(src_unet, stop_at_block=STOP_BLOCK, token_dim=TOKEN_DIM).to(DEVICE)

procs = install_cross_attn(denoiser, token_dim=TOKEN_DIM, scale=1.0)
for p in procs.values():
    p.to(DEVICE)

state = torch.load(ADAPTER, map_location=DEVICE)
encoder.proj.load_state_dict(state["encoder_proj"])
for name, p in procs.items():
    p.load_state_dict(state["procs"][name])

scheduler = DDPMScheduler(num_train_timesteps=1000)
dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)
denoiser.eval(); encoder.eval()


@torch.no_grad()
def generate_and_collect(src_img, tgt_img, mask):
    #same as infer_cross_attn_v2.py: real target noised to START_T, attention summed over the
    #trajectory and the up_blocks layers -> per_query [Q (chB positions), N_tok (chA tokens)]
    tokens = encoder(src_img)
    set_tokens(procs, tokens)
    B, N = tokens.shape[0], tokens.shape[1]
    src_side = int(math.sqrt(N))

    #keys restricted to chA tokens inside the cell mask
    token_mask = F.adaptive_max_pool2d(mask, output_size=(src_side, src_side))
    token_mask = (token_mask.flatten(1) > 0.5)
    for p in procs.values():
        p.token_mask = token_mask

    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    timesteps = scheduler.timesteps[scheduler.timesteps <= START_T]

    def merge_heads(attn_map):
        return attn_map.view(B, -1, *attn_map.shape[1:]).mean(dim=1)[0]   # [Q, N_tok]

    noise = torch.randn_like(tgt_img)
    t0 = torch.full((B,), START_T, device=DEVICE, dtype=torch.long)
    latents = scheduler.add_noise(tgt_img, noise, t0)

    up_names, layer_sum, tgt_side = None, None, None
    for t in timesteps:
        set_store_attn(procs, True)
        noise_pred = denoiser(latents, t.reshape(1).to(DEVICE)).sample
        set_store_attn(procs, False)

        if up_names is None:
            up_names = [name for name, p in procs.items() if p.attn_map is not None and "up_blocks" in name]
            q_sizes = {procs[name].attn_map.shape[1] for name in up_names}
            assert len(q_sizes) == 1, f"up_block layers have mismatched Q sizes: {q_sizes}"
            max_q = q_sizes.pop()
            tgt_side = int(math.sqrt(max_q))
            layer_sum = torch.zeros(max_q, N, device=DEVICE)

        for name in up_names:
            layer_sum += merge_heads(procs[name].attn_map)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    per_query = (layer_sum / (len(timesteps) * len(up_names))).cpu()   # [Q, N_tok]
    return latents, per_query, src_side, tgt_side, len(timesteps)


def to_img(t):
    return (t[0, 0] * 0.5 + 0.5).clamp(0, 1).cpu().numpy()

def upsample(m, size=IMG_SIZE):
    t = torch.tensor(m)[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()

def overlay_on(base_gray, heat):
    heat_up = upsample(heat)
    gray_rgb = np.stack([base_gray] * 3, axis=-1)
    return (0.5 * cm.jet(heat_up)[..., :3] + 0.5 * gray_rgb).clip(0, 1)

def mask_overlay(base_gray, burst):
    #burst pixels tinted red on top of the grayscale image
    rgb = np.stack([base_gray] * 3, axis=-1)
    tint = np.array([1.0, 0.15, 0.15])
    rgb[burst] = 0.45 * rgb[burst] + 0.55 * tint
    return rgb


def load_burst_mask(name):
    #hand-drawn burst mask for this crop, resized to the model's 128x128 like the cell mask
    base = name[:-len("_chA.png")]
    video, cell_frame = base.rsplit("_cell", 1)
    cell, frame = cell_frame.split("_f")
    cell_dirs = pick_burst_cell_dirs(Path(ANNOT_DIR) / video)
    d = cell_dirs.get(int(cell))
    if d is None:
        return None, None
    path = d / f"{d.name}_f{frame}.png"
    if not path.exists():
        return None, None
    m = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
    return np.array(m) > 0, d.parent.name


def pick_query_tokens(burst, tgt_side):
    #query tokens whose cell overlaps the mask ("on"), plus their neighbours up to NEAR_RING away ("near")
    cell = IMG_SIZE // tgt_side
    on = burst.reshape(tgt_side, cell, tgt_side, cell).any(axis=(1, 3))
    near = on.copy()
    for _ in range(NEAR_RING):
        grown = near.copy()
        grown[1:, :] |= near[:-1, :]; grown[:-1, :] |= near[1:, :]
        grown[:, 1:] |= near[:, :-1]; grown[:, :-1] |= near[:, 1:]
        grown[1:, 1:] |= near[:-1, :-1]; grown[:-1, :-1] |= near[1:, 1:]
        grown[1:, :-1] |= near[:-1, 1:]; grown[:-1, 1:] |= near[1:, :-1]
        near = grown
    tokens = [(y, x, "on" if on[y, x] else "near") for y in range(tgt_side) for x in range(tgt_side) if near[y, x]]
    return tokens


def token_box(ax, y, x, cell, color="yellow"):
    ax.add_patch(patches.Rectangle((x * cell, y * cell), cell, cell, linewidth=2, edgecolor=color, facecolor="none"))


for i in SAMPLE_INDICES:
    x, mask = dataset[i]
    name = dataset.chA_files[i].name
    burst, mask_src = load_burst_mask(name)
    if burst is None or not burst.any():
        print(f"sample {i}: no hand-drawn burst mask for this frame, skipped")
        continue

    x = x.unsqueeze(0).to(DEVICE)
    mask = mask.unsqueeze(0).to(DEVICE)
    src_img = x[:, SRC_IDX:SRC_IDX+1]
    tgt_img = x[:, TGT_IDX:TGT_IDX+1]

    recon, per_query, src_side, tgt_side, n_steps = generate_and_collect(src_img, tgt_img, mask)
    src_np, tgt_real_np, tgt_gen_np = to_img(src_img), to_img(tgt_img), to_img(recon)
    cell = IMG_SIZE / tgt_side

    sample_dir = f"{OUT_DIR}/sample_{i}"
    tok_dir = f"{sample_dir}/tokens"
    os.makedirs(tok_dir, exist_ok=True)

    #context: chA (source) | real chB | generated chB
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, (src_np, tgt_real_np, tgt_gen_np),
                              (f"{SRC_LABEL} (source / keys)", f"{TGT_LABEL} real (query)", f"{TGT_LABEL} denoised from T={START_T}")):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(f"{sample_dir}/triplet.png", dpi=120); plt.close(fig)

    #which query tokens were picked, on the mask overlay
    tokens = pick_query_tokens(burst, tgt_side)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask_overlay(tgt_real_np, burst))
    for (y, xx, kind) in tokens:
        token_box(ax, y, xx, cell, color="yellow" if kind == "on" else "cyan")
    ax.set_title(f"picked {TGT_LABEL} query tokens\n(yellow = on the burst mask, cyan = next to it)"); ax.axis("off")
    fig.tight_layout(); fig.savefig(f"{sample_dir}/picked_tokens.png", dpi=120); plt.close(fig)

    #heatmaps share one colour scale across all query tokens of this sample (as in infer_cross_attn_v2.py)
    vmin, vmax = per_query.min().item(), per_query.max().item()
    def shared_normalize(m):
        return ((m - vmin) / (vmax - vmin + 1e-8)).numpy()

    for (y, xx, kind) in tokens:
        q_idx = y * tgt_side + xx
        heat = shared_normalize(per_query[q_idx]).reshape(src_side, src_side)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))
        axes[0].imshow(tgt_real_np, cmap="gray", vmin=0, vmax=1); token_box(axes[0], y, xx, cell)
        axes[0].set_title(f"{TGT_LABEL} (query), token ({y},{xx})")
        axes[1].imshow(burst, cmap="gray"); token_box(axes[1], y, xx, cell)
        axes[1].set_title("hand-drawn burst mask")
        axes[2].imshow(mask_overlay(tgt_real_np, burst)); token_box(axes[2], y, xx, cell)
        axes[2].set_title(f"burst mask on {TGT_LABEL}")
        axes[3].imshow(overlay_on(src_np, heat))
        axes[3].set_title(f"this {TGT_LABEL} token's attention over {SRC_LABEL}\n(T={START_T}, summed over {n_steps} steps)")
        axes[4].imshow(src_np, cmap="gray", vmin=0, vmax=1)
        axes[4].set_title(f"{SRC_LABEL} (keys, plain)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"val sample {i} | query token ({y},{xx}) {kind} the burst mask", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.9]) #leave room for the suptitle above the panel titles
        fig.savefig(f"{tok_dir}/token_{y:02d}_{xx:02d}_{kind}.png", dpi=110)
        plt.close(fig)

    n_on = sum(k == "on" for _, _, k in tokens)
    print(f"sample {i} ({name}) -> {sample_dir}/  query grid {tgt_side}x{tgt_side}, key grid {src_side}x{src_side}, "
          f"{len(tokens)} token figures ({n_on} on, {len(tokens) - n_on} near), mask from {mask_src}")
