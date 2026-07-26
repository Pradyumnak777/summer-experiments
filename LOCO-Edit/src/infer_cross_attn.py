'''
run the trained cross-attn adapter on HELD-OUT test pairs and visualize what the chB
denoiser attends to over chA's tokens. two views per sample:
  1. aggregate.png     - chA importance summed over the WHOLE chB image (coarse, global)
  2. grid.png          - ALL chB grid positions at once, one small chA-attention heatmap
                          per cell, arranged to match chB's own spatial layout
  3. points/*.png      - the same per-location maps, saved individually, full size
each sample gets its own folder under OUT_DIR.
run from src/:  python infer_cross_attn.py
'''
import os, math
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler

from VAE_disent.diffusion_model import build_unet
from VAE_disent.data_utils import twoChannelDataset
from cross_attn_modules import ChannelEncoder, install_cross_attn, set_tokens, set_store_attn

DEVICE   = torch.device("cuda:9")
#HELD-OUT test dirs (the 2k the models never saw) - adjust names to your split
CHA_DIR  = "data/singlecell_chA_split/validation"
CHB_DIR  = "data/singlecell_chB_split/validation"

TGT_CKPT   = "diffusion_checkpoints/ddpm_chB_128/unet_ema_epoch10.pt"
SRC_CKPT   = "diffusion_checkpoints/ddpm_chA_128/unet_ema_epoch10.pt"
ADAPTER    = "cross_attn_checkpoints/AtoB/adapter_epoch5.pt"
TGT_IDX, SRC_IDX = 1, 0

IMG_SIZE     = 128
TOKEN_DIM    = 256
STOP_BLOCK   = 3
PROBE_T      = 200        #timestep to probe attention at (mid-low, structure is forming)
N_SAMPLES    = 6          #how many held-out samples to run
OUT_DIR      = "cross_attn_checkpoints/AtoB/attn_maps"
os.makedirs(OUT_DIR, exist_ok=True)

#---- rebuild the exact same setup, then load adapter weights ----
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
dataset = twoChannelDataset(CHA_DIR, CHB_DIR)

denoiser.eval(); encoder.eval()


@torch.no_grad()
def attn_maps(src_img, tgt_img):
    '''
    one forward pass, attn capture on. returns:
      agg       - chA importance summed over ALL of chB (the coarse global view)
      per_query - [Q, N_tok], EVERY chB position's own view of the chA tokens,
                  nothing subsampled or summed away
      src_side, tgt_side - grid side lengths for chA tokens and chB positions
    '''
    tokens = encoder(src_img)
    set_tokens(procs, tokens)
    set_store_attn(procs, True)

    noise = torch.randn_like(tgt_img)
    t = torch.full((1,), PROBE_T, device=DEVICE, dtype=torch.long)
    noisy = scheduler.add_noise(tgt_img, noise, t)
    _ = denoiser(noisy, t).sample                     #runs attn, fills proc.attn_map

    set_store_attn(procs, False)

    N = tokens.shape[1]
    src_side = int(math.sqrt(N))

    #---- aggregate: chA importance, summed over ALL chB positions + ALL attn layers ----
    agg = torch.zeros(N, device=DEVICE)
    for p in procs.values():
        if p.attn_map is not None:
            agg += p.attn_map.sum(dim=(0, 1))
    agg = (agg / agg.max().clamp_min(1e-8)).view(src_side, src_side).cpu().numpy()

    #---- per-location: use the attn layer with the most query positions ----
    best_map = max(
        (p.attn_map for p in procs.values() if p.attn_map is not None),
        key=lambda m: m.shape[1]
    )
    Q = best_map.shape[1]
    tgt_side = int(math.sqrt(Q))
    assert tgt_side * tgt_side == Q, f"Q_len={Q} isn't a perfect square, pick a different layer"

    per_query = best_map.mean(dim=0)                    # [Q, N_tok] - EVERY position kept
    return agg, per_query.cpu(), src_side, tgt_side


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

def normalize(m):
    m = m - m.min()
    return m / m.max().clamp_min(1e-8) if torch.is_tensor(m) else m / (m.max() or 1.0)


def save_sample(sample_dir, chA, chB, agg, per_query, src_side, tgt_side):
    os.makedirs(sample_dir, exist_ok=True)
    points_dir = f"{sample_dir}/points"
    os.makedirs(points_dir, exist_ok=True)

    plt.imsave(f"{sample_dir}/chA.png", chA, cmap="gray")
    plt.imsave(f"{sample_dir}/chB.png", chB, cmap="gray")

    #---- aggregate figure ----
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(chA, cmap="gray"); axes[0].set_title("chA (source)"); axes[0].axis("off")
    axes[1].imshow(overlay_on(chA, agg))
    axes[1].set_title("aggregate attn over chA\n(summed over ALL of chB)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/aggregate.png", dpi=120)
    plt.close(fig)

    #---- full grid mosaic: every chB position, arranged to match chB's own layout ----
    fig, axes = plt.subplots(tgt_side, tgt_side, figsize=(tgt_side * 1.3, tgt_side * 1.3))
    for y in range(tgt_side):
        for x in range(tgt_side):
            q_idx = y * tgt_side + x
            m = normalize(per_query[q_idx]).view(src_side, src_side).numpy()
            ax = axes[y, x]
            ax.imshow(overlay_on(chA, m))
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"per-location attn over chA, one cell per chB position ({tgt_side}x{tgt_side}={tgt_side*tgt_side} total)")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/grid.png", dpi=150)
    plt.close(fig)

    #---- every point ALSO saved individually, full size ----
    for y in range(tgt_side):
        for x in range(tgt_side):
            q_idx = y * tgt_side + x
            m = normalize(per_query[q_idx]).view(src_side, src_side).numpy()

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(chB, cmap="gray")
            py, px = y / tgt_side * IMG_SIZE, x / tgt_side * IMG_SIZE
            axes[0].scatter([px], [py], c="red", s=40, marker="x")
            axes[0].set_title(f"chB point ({y},{x})"); axes[0].axis("off")
            axes[1].imshow(overlay_on(chA, m))
            axes[1].set_title("attends to (chA)"); axes[1].axis("off")
            fig.tight_layout()
            fig.savefig(f"{points_dir}/point_{y:02d}_{x:02d}.png", dpi=100)
            plt.close(fig)


idx = torch.linspace(0, len(dataset) - 1, N_SAMPLES).long()
for n, i in enumerate(idx):
    x = dataset[int(i)].unsqueeze(0).to(DEVICE)       # [1, 2, H, W]
    src_img = x[:, SRC_IDX:SRC_IDX+1]
    tgt_img = x[:, TGT_IDX:TGT_IDX+1]

    agg, per_query, src_side, tgt_side = attn_maps(src_img, tgt_img)
    chA = to_img(src_img)
    chB = to_img(tgt_img)

    sample_dir = f"{OUT_DIR}/sample_{n:02d}"
    save_sample(sample_dir, chA, chB, agg, per_query, src_side, tgt_side)
    print(f"sample {n:02d} (dataset idx {int(i)}) -> {sample_dir}/  "
          f"({tgt_side*tgt_side} points, grid.png + {tgt_side*tgt_side} individual files)")

print(f"\ndone. {N_SAMPLES} samples saved under {OUT_DIR}/sample_XX/")