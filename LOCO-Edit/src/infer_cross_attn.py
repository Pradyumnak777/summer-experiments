'''
run the trained cross-attn adapter on HELD-OUT test pairs and visualize, for each
chA region/token, WHERE in chB the denoiser draws on it. two views per sample:
  1. aggregate.png     - chA importance summed over the WHOLE chB image (coarse, global)
  2. grid.png          - ALL chA tokens at once, one small chB-attention heatmap
                          per chA cell, arranged to match chA's own spatial layout
  3. points/*.png      - the same per-token maps, saved individually, full size,
                          with a bounding box on chA showing that token's cell
each sample gets its own folder under OUT_DIR.
run from src/:  python infer_cross_attn.py

NOTE: TGT_CKPT/SRC_CKPT must be the SAME checkpoints ADAPTER was trained against -
the trained to_k_img/to_v_img weights are fit to those specific backbones' hidden-
state statistics. Bumping these to a later epoch needs retraining train_cross_attn.py
against the new checkpoints first, not just swapping the path here.
'''
import os, math
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from diffusers import DDPMScheduler

from VAE_disent.diffusion_model import build_unet
from VAE_disent.data_utils import twoChannelDataset
from cross_attn_modules import ChannelEncoder, install_cross_attn, set_tokens, set_store_attn

DEVICE   = torch.device("cuda:9")
#HELD-OUT test dirs (the 2k the models never saw) - adjust names to your split
CHA_DIR  = "data/singlecell_chA_split/train"
CHB_DIR  = "data/singlecell_chB_split/train"

TGT_CKPT   = "diffusion_checkpoints/ddpm_chB_128/unet_ema_epoch45.pt"
SRC_CKPT   = "diffusion_checkpoints/ddpm_chA_128/unet_ema_epoch45.pt"
ADAPTER    = "cross_attn_checkpoints/AtoB/adapter_ema_final.pt"
TGT_IDX, SRC_IDX = 1, 0

IMG_SIZE      = 128
TOKEN_DIM     = 256
STOP_BLOCK    = 3
PROBE_T       = 800        #timestep to probe attention at (mid-low, structure is forming)
N_SAMPLES     = 7          #how many held-out samples to run
N_NOISE_DRAWS = 5          #average the attn map over this many independent noise draws
SEED          = 0
OUT_DIR       = f"cross_attn/AtoB/attn_maps_probe_{PROBE_T}"
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)

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
    averages N_NOISE_DRAWS independent noise draws at PROBE_T, and averages across
    EVERY attn layer tied at chB's finest queryable resolution (5 layers tie at
    Q=64 in this unet - picking just one arbitrarily throws the other 4 away).
    returns:
      agg        - chA importance summed over ALL of chB (the coarse global view)
      per_token  - [N_tok, Q], for EVERY chA token, its full spread over chB positions
      src_side, tgt_side - grid side lengths for chA tokens and chB positions
    '''
    tokens = encoder(src_img)
    set_tokens(procs, tokens)
    B, N = tokens.shape[0], tokens.shape[1]
    src_side = int(math.sqrt(N))

    def run_once():
        set_store_attn(procs, True)
        noise = torch.randn_like(tgt_img)
        t = torch.full((B,), PROBE_T, device=DEVICE, dtype=torch.long)
        noisy = scheduler.add_noise(tgt_img, noise, t)
        _ = denoiser(noisy, t).sample
        set_store_attn(procs, False)

    def merge_heads(attn_map):
        return attn_map.view(B, -1, *attn_map.shape[1:]).mean(dim=1)[0]   # [Q, N_tok]

    run_once()
    max_q = max(p.attn_map.shape[1] for p in procs.values() if p.attn_map is not None)
    tgt_side = int(math.sqrt(max_q))
    assert tgt_side * tgt_side == max_q, f"Q={max_q} isn't a perfect square, pick a different layer"
    finest = [name for name, p in procs.items()
              if p.attn_map is not None and p.attn_map.shape[1] == max_q]

    agg_sum   = torch.zeros(N, device=DEVICE)
    layer_sum = torch.zeros(max_q, N, device=DEVICE)

    for _ in range(N_NOISE_DRAWS):
        run_once()
        for name, p in procs.items():
            if p.attn_map is None:
                continue
            m = merge_heads(p.attn_map)
            agg_sum += m.sum(dim=0)
            if name in finest:
                layer_sum += m

    agg = (agg_sum / agg_sum.max().clamp_min(1e-8)).view(src_side, src_side).cpu().numpy()
    avg_map = layer_sum / (N_NOISE_DRAWS * len(finest))   # [Q, N_tok]
    per_token = avg_map.T.cpu()                            # [N_tok, Q] - chA-token-centric view

    return agg, per_token, src_side, tgt_side


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


def save_sample(sample_dir, chA, chB, agg, per_token, src_side, tgt_side):
    os.makedirs(sample_dir, exist_ok=True)
    points_dir = f"{sample_dir}/points"
    os.makedirs(points_dir, exist_ok=True)

    plt.imsave(f"{sample_dir}/chA.png", chA, cmap="gray")
    plt.imsave(f"{sample_dir}/chB.png", chB, cmap="gray")

    #---- aggregate figure: which chA regions matter most, summed over ALL of chB ----
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(chA, cmap="gray"); axes[0].set_title("chA (source)"); axes[0].axis("off")
    axes[1].imshow(overlay_on(chA, agg))
    axes[1].set_title("aggregate importance per chA region\n(summed over ALL of chB)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/aggregate.png", dpi=120)
    plt.close(fig)

    vmin, vmax = per_token.min().item(), per_token.max().item()
    def shared_normalize(m):
        return ((m - vmin) / (vmax - vmin + 1e-8)).numpy()

    #---- full grid mosaic: every chA token, arranged to match chA's own layout ----
    fig, axes = plt.subplots(src_side, src_side, figsize=(src_side * 1.3, src_side * 1.3))
    for y in range(src_side):
        for x in range(src_side):
            n_idx = y * src_side + x
            m = shared_normalize(per_token[n_idx]).reshape(tgt_side, tgt_side)
            ax = axes[y, x]
            ax.imshow(overlay_on(chB, m))
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"per-chA-token attn over chB, one cell per chA position "
                 f"({src_side}x{src_side}={src_side*src_side} total)")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/grid.png", dpi=150)
    plt.close(fig)

    #---- every chA token ALSO saved individually, full size, with a bounding box ----
    cell_size = IMG_SIZE / src_side
    for y in range(src_side):
        for x in range(src_side):
            n_idx = y * src_side + x
            m = shared_normalize(per_token[n_idx]).reshape(tgt_side, tgt_side)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(chA, cmap="gray")
            rect = patches.Rectangle((x * cell_size, y * cell_size), cell_size, cell_size,
                                      linewidth=2, edgecolor="red", facecolor="none")
            axes[0].add_patch(rect)
            axes[0].set_title(f"chA token ({y},{x})"); axes[0].axis("off")
            axes[1].imshow(overlay_on(chB, m))
            axes[1].set_title("chB attends to this token"); axes[1].axis("off")
            fig.tight_layout()
            fig.savefig(f"{points_dir}/point_{y:02d}_{x:02d}.png", dpi=100)
            plt.close(fig)


idx = torch.linspace(0, len(dataset) - 1, N_SAMPLES).long()
for n, i in enumerate(idx):
    x = dataset[int(i)].unsqueeze(0).to(DEVICE)       # [1, 2, H, W]
    src_img = x[:, SRC_IDX:SRC_IDX+1]
    tgt_img = x[:, TGT_IDX:TGT_IDX+1]

    agg, per_token, src_side, tgt_side = attn_maps(src_img, tgt_img)
    chA = to_img(src_img)
    chB = to_img(tgt_img)

    sample_dir = f"{OUT_DIR}/sample_{n:02d}"
    save_sample(sample_dir, chA, chB, agg, per_token, src_side, tgt_side)
    print(f"sample {n:02d} (dataset idx {int(i)}) -> {sample_dir}/  "
          f"({src_side*src_side} chA tokens, grid.png + {src_side*src_side} individual files)")

print(f"\ndone. {N_SAMPLES} samples saved under {OUT_DIR}/sample_XX/")