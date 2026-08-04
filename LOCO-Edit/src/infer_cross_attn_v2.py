
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

import debugpy
# debugpy.listen(("127.0.0.1", 5678)) #127.0.0.1, cz only local host can talk to the port
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")

DEVICE   = torch.device("cuda:9")
SET = "train"
CHA_DIR  = f"data/singlecell_chA_split/{SET}"
CHB_DIR  = f"data/singlecell_chB_split/{SET}"

TGT_CKPT   = "diffusion_checkpoints/ddpm_chB_128/unet_ema_epoch45.pt"
SRC_CKPT   = "diffusion_checkpoints/ddpm_chA_128/unet_ema_epoch45.pt"
ADAPTER    = "cross_attn_checkpoints/AtoB_new_maskedchA/adapter_ema_epoch30.pt"
# ADAPTER    = "cross_attn_checkpoints/AtoB/adapter_ema_epoch20.pt"

TGT_IDX, SRC_IDX = 1, 0

IMG_SIZE            = 128
TOKEN_DIM           = 256
STOP_BLOCK          = 3
START_T             = 600   
NUM_INFERENCE_STEPS = 1000  
SAMPLE_INDICES = [7]   #dataset indices to run
SEED           = 0
OUT_DIR        = f"cross_attn/AtoB_queryview/{SET}/attn_maps_real_noised_startT{START_T}"
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)

denoiser = build_unet(IMG_SIZE, channels=1).to(DEVICE)
denoiser.load_state_dict(torch.load(TGT_CKPT, map_location=DEVICE))
denoiser.requires_grad_(False)

src_unet = build_unet(IMG_SIZE, channels=1)
src_unet.load_state_dict(torch.load(SRC_CKPT, map_location="cpu"))
encoder = ChannelEncoder(src_unet, stop_at_block=STOP_BLOCK, token_dim=TOKEN_DIM).to(DEVICE)

#attaches cross attn weights..(ts is also called during training)
procs = install_cross_attn(denoiser, token_dim=TOKEN_DIM, scale=1.0)
for p in procs.values():
    p.to(DEVICE)

#loads model weights onto the newly attached cross attn matrices in the above code line.
state = torch.load(ADAPTER, map_location=DEVICE)
encoder.proj.load_state_dict(state["encoder_proj"])
for name, p in procs.items():
    p.load_state_dict(state["procs"][name])

scheduler = DDPMScheduler(num_train_timesteps=1000)
dataset = twoChannelDataset(CHA_DIR, CHB_DIR)

denoiser.eval(); encoder.eval()


@torch.no_grad()
def generate_and_collect(src_img, tgt_img):
    tokens = encoder(src_img)
    set_tokens(procs, tokens)
    B, N = tokens.shape[0], tokens.shape[1]
    src_side = int(math.sqrt(N))

    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    timesteps = scheduler.timesteps[scheduler.timesteps <= START_T]

    def merge_heads(attn_map):
        return attn_map.view(B, -1, *attn_map.shape[1:]).mean(dim=1)[0]   # [Q, N_tok]

    noise = torch.randn_like(tgt_img)
    t0 = torch.full((B,), START_T, device=DEVICE, dtype=torch.long)
    latents = scheduler.add_noise(tgt_img, noise, t0)   # REAL chB, properly noised to START_T

    up_names, max_q, tgt_side = None, None, None
    agg_sum, layer_sum = None, None

    for t in timesteps:
        set_store_attn(procs, True)
        t_batch = t.reshape(1).to(DEVICE)
        noise_pred = denoiser(latents, t_batch).sample
        set_store_attn(procs, False)

        if up_names is None:
            #resolve which layers/shapes we're accumulating, once - same every step,
            #since it's determined by architecture (which layer), not by t
            up_names = [name for name, p in procs.items()
                        if p.attn_map is not None and "up_blocks" in name]
            assert up_names, "no up_blocks attention layers found - check install_cross_attn naming"
            q_sizes = {procs[name].attn_map.shape[1] for name in up_names}
            assert len(q_sizes) == 1, f"up_block layers have mismatched Q sizes: {q_sizes}"
            max_q = q_sizes.pop()
            tgt_side = int(math.sqrt(max_q))
            assert tgt_side * tgt_side == max_q, f"Q={max_q} isn't a perfect square"
            agg_sum   = torch.zeros(N, device=DEVICE)
            layer_sum = torch.zeros(max_q, N, device=DEVICE)

        for name in up_names:
            m = merge_heads(procs[name].attn_map)
            agg_sum += m.sum(dim=0)
            layer_sum += m

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    agg = (agg_sum / agg_sum.max().clamp_min(1e-8)).view(src_side, src_side).cpu().numpy()
    avg_map = layer_sum / (len(timesteps) * len(up_names))   # [Q, N_tok]
    # per_token = avg_map.T.cpu()                               # [N_tok, Q]
    
    #to get the reverse view (fixing query, overlaying on keys)
    per_query = avg_map.cpu()        # [Q, N_tok]     

    return latents, agg, per_query, src_side, tgt_side


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


def save_sample(sample_dir, chA, chB_real, chB_gen, agg, per_query, src_side, tgt_side):
    os.makedirs(sample_dir, exist_ok=True)
    points_dir = f"{sample_dir}/points"
    os.makedirs(points_dir, exist_ok=True)

    plt.imsave(f"{sample_dir}/chA.png", chA, cmap="gray")
    plt.imsave(f"{sample_dir}/chB_real.png", chB_real, cmap="gray")
    plt.imsave(f"{sample_dir}/chB_generated.png", chB_gen, cmap="gray")

    #source / real target / reconstructed target, side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(chA, cmap="gray"); axes[0].set_title("chA (source)"); axes[0].axis("off")
    axes[1].imshow(chB_real, cmap="gray"); axes[1].set_title("chB (real)"); axes[1].axis("off")
    axes[2].imshow(chB_gen, cmap="gray"); axes[2].set_title(f"chB (reconstructed, START_T={START_T})"); axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/triplet.png", dpi=120)
    plt.close(fig)
    

    #aggregate figure: which chA regions matter most, summed over the WHOLE
    #trajectory AND the whole chB image 
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(chA, cmap="gray"); axes[0].set_title("chA (source)"); axes[0].axis("off")
    axes[1].imshow(overlay_on(chA, agg))
    axes[1].set_title("aggregate importance per chA region\n(summed over full trajectory + all of chB)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/aggregate.png", dpi=120)
    plt.close(fig)

    vmin, vmax = per_query.min().item(), per_query.max().item()
    def shared_normalize(m):
        return ((m - vmin) / (vmax - vmin + 1e-8)).numpy()

    fig, axes = plt.subplots(tgt_side, tgt_side, figsize=(tgt_side * 1.3, tgt_side * 1.3))
    for y in range(tgt_side):
        for x in range(tgt_side):
            q_idx = y * tgt_side + x
            m = shared_normalize(per_query[q_idx]).reshape(src_side, src_side)
            ax = axes[y, x]
            ax.imshow(overlay_on(chA, m))          # chA, not chB_gen
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"per-chB-position attn over chA, summed over full trajectory, "
                f"one cell per chB position ({tgt_side}x{tgt_side}={tgt_side*tgt_side} total)")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/grid.png", dpi=150)
    plt.close(fig)

    cell_size = IMG_SIZE / tgt_side                 # was src_side
    for y in range(tgt_side):
        for x in range(tgt_side):
            q_idx = y * tgt_side + x
            m = shared_normalize(per_query[q_idx]).reshape(src_side, src_side)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(chB_gen, cmap="gray")     # box goes on chB now
            rect = patches.Rectangle((x * cell_size, y * cell_size), cell_size, cell_size,
                                    linewidth=2, edgecolor="red", facecolor="none")
            axes[0].add_patch(rect)
            axes[0].set_title(f"chB position ({y},{x})"); axes[0].axis("off")
            axes[1].imshow(overlay_on(chA, m))       # heatmap on chA
            axes[1].set_title("this chB position attends to chA\n(summed over full trajectory)")
            axes[1].axis("off")
            fig.tight_layout()
            fig.savefig(f"{points_dir}/point_{y:02d}_{x:02d}.png", dpi=100)
            plt.close(fig)


for i in SAMPLE_INDICES:
    x = dataset[i].unsqueeze(0).to(DEVICE)            
    src_img = x[:, SRC_IDX:SRC_IDX+1]
    tgt_img = x[:, TGT_IDX:TGT_IDX+1]

    recon, agg, per_query, src_side, tgt_side = generate_and_collect(src_img, tgt_img)
    chA = to_img(src_img)
    chB_real = to_img(tgt_img)
    chB_gen = to_img(recon)

    sample_dir = f"{OUT_DIR}/sample_{i}"
    save_sample(sample_dir, chA, chB_real, chB_gen, agg, per_query, src_side, tgt_side)
    n_steps = len(scheduler.timesteps[scheduler.timesteps <= START_T])
    print(f"sample idx {i} -> {sample_dir}/  "
          f"({src_side*src_side} chA tokens, {n_steps} steps aggregated, "
          f"grid.png + {tgt_side*tgt_side} individual files)")

print(f"\ndone. {len(SAMPLE_INDICES)} samples saved under {OUT_DIR}/sample_<idx>/")