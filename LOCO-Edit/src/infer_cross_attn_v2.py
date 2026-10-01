
import os, math
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from diffusers import DDPMScheduler
import torch.nn.functional as F
from VAE_disent.diffusion_model import build_unet
from VAE_disent.data_utils import twoChannelDataset
from cross_attn_modules import ChannelEncoder, install_cross_attn, set_tokens, set_store_attn
from pathlib import Path
from PIL import Image

import debugpy
# debugpy.listen(("127.0.0.1", 5678)) #127.0.0.1, cz only local host can talk to the port
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")

DEVICE   = torch.device("cuda:9")
SET = "train"
CHA_DIR  = f"data/singlecell_chA_new_split/{SET}"
CHB_DIR  = f"data/singlecell_chB_new_split/{SET}"

#NOTE: toggle direction here, everything below (ckpts, indices, labels, adapter, out dir)
#adapts to this. "AtoB": encoder=chA, denoiser=chB | "BtoA": encoder=chB, denoiser=chA
DIRECTION     = "AtoB"
ADAPTER_EPOCH = "epoch49"

if DIRECTION == "AtoB":
    TGT_CKPT = "diffusion_checkpoints/ddpm_chB_128_masked/unet_ema_epoch40.pt"   #denoiser (chB)
    SRC_CKPT = "diffusion_checkpoints/ddpm_chA_128_masked/unet_ema_epoch40.pt"   #encoder  (chA)
    TGT_IDX, SRC_IDX = 1, 0
    SRC_LABEL, TGT_LABEL = "chA", "chB"
elif DIRECTION == "BtoA":
    TGT_CKPT = "diffusion_checkpoints/ddpm_chA_128_masked/unet_ema_epoch40.pt"   #denoiser (chA)
    SRC_CKPT = "diffusion_checkpoints/ddpm_chB_128_masked/unet_ema_epoch40.pt"   #encoder  (chB)
    TGT_IDX, SRC_IDX = 0, 1
    SRC_LABEL, TGT_LABEL = "chB", "chA"
else:
    raise ValueError(f"unknown DIRECTION: {DIRECTION}")

ADAPTER = f"cross_attn_checkpoints/{DIRECTION}_v2/adapter_ema_{ADAPTER_EPOCH}.pt"

#NOTE: toggle view here. "row": fix a target-channel query, heatmap over source-channel
#keys -> points/, grid.png | "column": fix a source-channel key, heatmap over target-channel
#queries (the SD/DAAM "attention map for token X" convention) -> points_flipped/, grid_flipped.png
VIEW_MODE = "row"

IMG_SIZE            = 128
TOKEN_DIM           = 256
STOP_BLOCK          = 3
START_T             = 400
NUM_INFERENCE_STEPS = 1000
SAMPLE_INDICES = [431]   #dataset indices to run
SEED           = 0
OUT_DIR        = f"cross_attn/{DIRECTION}_v2/{SET}/attn_maps_real_noised_startT_{START_T}"
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
dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir = f"data/singlecell_mask_new_split/{SET}")

denoiser.eval(); encoder.eval()


@torch.no_grad()
def generate_and_collect(src_img, tgt_img, mask):
    tokens = encoder(src_img)
    set_tokens(procs, tokens)
    B, N = tokens.shape[0], tokens.shape[1]
    src_side = int(math.sqrt(N))

    #added for masking
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
    latents = scheduler.add_noise(tgt_img, noise, t0)   # REAL target channel, properly noised to START_T

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


def save_sample(sample_dir, src_np, tgt_real_np, tgt_gen_np, agg, per_query, src_side, tgt_side):
    os.makedirs(sample_dir, exist_ok=True)
    points_dir = f"{sample_dir}/points"
    os.makedirs(points_dir, exist_ok=True)

    plt.imsave(f"{sample_dir}/{SRC_LABEL}.png", src_np, cmap="gray")
    plt.imsave(f"{sample_dir}/{TGT_LABEL}_real.png", tgt_real_np, cmap="gray")
    plt.imsave(f"{sample_dir}/{TGT_LABEL}_generated.png", tgt_gen_np, cmap="gray")

    #source / real target / reconstructed target, side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(src_np, cmap="gray"); axes[0].set_title(f"{SRC_LABEL} (source)"); axes[0].axis("off")
    axes[1].imshow(tgt_real_np, cmap="gray"); axes[1].set_title(f"{TGT_LABEL} (real)"); axes[1].axis("off")
    axes[2].imshow(tgt_gen_np, cmap="gray"); axes[2].set_title(f"{TGT_LABEL} (reconstructed, START_T={START_T})"); axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/triplet.png", dpi=120)
    plt.close(fig)


    #aggregate figure: which source-channel regions matter most, summed over the WHOLE
    #trajectory AND the whole target-channel image
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(src_np, cmap="gray"); axes[0].set_title(f"{SRC_LABEL} (source)"); axes[0].axis("off")
    axes[1].imshow(overlay_on(src_np, agg))
    axes[1].set_title(f"aggregate importance per {SRC_LABEL} region\n(summed over full trajectory + all of {TGT_LABEL})")
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
            ax.imshow(overlay_on(src_np, m))          # source channel, not generated target
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"per-{TGT_LABEL}-position attn over {SRC_LABEL}, summed over full trajectory, "
                f"one cell per {TGT_LABEL} position ({tgt_side}x{tgt_side}={tgt_side*tgt_side} total)")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/grid.png", dpi=150)
    plt.close(fig)

    cell_size = IMG_SIZE / tgt_side                 # was src_side
    for y in range(tgt_side):
        for x in range(tgt_side):
            q_idx = y * tgt_side + x
            m = shared_normalize(per_query[q_idx]).reshape(src_side, src_side)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(tgt_gen_np, cmap="gray")     # box goes on generated target now
            rect = patches.Rectangle((x * cell_size, y * cell_size), cell_size, cell_size,
                                    linewidth=2, edgecolor="red", facecolor="none")
            axes[0].add_patch(rect)
            axes[0].set_title(f"{TGT_LABEL} position ({y},{x})"); axes[0].axis("off")
            axes[1].imshow(overlay_on(src_np, m))       # heatmap on source channel
            axes[1].set_title(f"this {TGT_LABEL} position attends to {SRC_LABEL}\n(summed over full trajectory)")
            axes[1].axis("off")
            axes[2].imshow(src_np, cmap="gray")         # plain source channel, the image the heatmap is overlaid on
            axes[2].set_title(f"{SRC_LABEL} (plain)")
            axes[2].axis("off")
            fig.tight_layout()
            fig.savefig(f"{points_dir}/point_{y:02d}_{x:02d}.png", dpi=100)
            plt.close(fig)


def save_handdrawn_mask_context(sample_dir, mask_movie_dir, frame_idx):
    """plots the hand-drawn mask frames at frame_idx-1, frame_idx, frame_idx+1 side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, offset in zip(axes, (-1, 0, 1)):
        f_idx = frame_idx + offset
        frame_path = mask_movie_dir / f"{mask_movie_dir.name}_f{f_idx:03d}.png"
        label = {-1: "frame-1", 0: "frame", 1: "frame+1"}[offset]
        if f_idx < 0 or not frame_path.exists():
            ax.axis("off")
            ax.set_title(f"{label} (f{f_idx:03d}, n/a)")
            continue
        m = np.array(Image.open(frame_path).convert("L"))
        ax.imshow(m, cmap="gray")
        ax.set_title(f"{label} (f{f_idx:03d})")
        ax.axis("off")
    fig.suptitle("hand-drawn mask, target frame ± 1")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/handdrawn_mask_context.png", dpi=120)
    plt.close(fig)


def save_sample_flipped(sample_dir, src_np, tgt_real_np, tgt_gen_np, agg, per_query, src_side, tgt_side):
    os.makedirs(sample_dir, exist_ok=True)
    points_dir = f"{sample_dir}/points_flipped"
    os.makedirs(points_dir, exist_ok=True)

    plt.imsave(f"{sample_dir}/{SRC_LABEL}.png", src_np, cmap="gray")
    plt.imsave(f"{sample_dir}/{TGT_LABEL}_real.png", tgt_real_np, cmap="gray")
    plt.imsave(f"{sample_dir}/{TGT_LABEL}_generated.png", tgt_gen_np, cmap="gray")

    #source / real target / reconstructed target, side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(src_np, cmap="gray"); axes[0].set_title(f"{SRC_LABEL} (source)"); axes[0].axis("off")
    axes[1].imshow(tgt_real_np, cmap="gray"); axes[1].set_title(f"{TGT_LABEL} (real)"); axes[1].axis("off")
    axes[2].imshow(tgt_gen_np, cmap="gray"); axes[2].set_title(f"{TGT_LABEL} (reconstructed, START_T={START_T})"); axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/triplet.png", dpi=120)
    plt.close(fig)

    #aggregate figure: which source-channel regions matter most, summed over the WHOLE
    #trajectory AND the whole target-channel image (this is already the "flipped" total,
    #so it's unchanged from save_sample - see note below)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(src_np, cmap="gray"); axes[0].set_title(f"{SRC_LABEL} (source)"); axes[0].axis("off")
    axes[1].imshow(overlay_on(src_np, agg))
    axes[1].set_title(f"aggregate importance per {SRC_LABEL} region\n(summed over full trajectory + all of {TGT_LABEL})")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/aggregate.png", dpi=120)
    plt.close(fig)

    per_token = per_query.T   # [N_tok, Q] - fix a source-channel KEY, heat over target-channel QUERIES

    vmin, vmax = per_token.min().item(), per_token.max().item()
    def shared_normalize(m):
        return ((m - vmin) / (vmax - vmin + 1e-8)).numpy()

    fig, axes = plt.subplots(src_side, src_side, figsize=(src_side * 1.3, src_side * 1.3))
    for y in range(src_side):
        for x in range(src_side):
            k_idx = y * src_side + x
            m = shared_normalize(per_token[k_idx]).reshape(tgt_side, tgt_side)
            ax = axes[y, x]
            ax.imshow(overlay_on(tgt_gen_np, m))          # generated target now, not source
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"per-{SRC_LABEL}-token attn received from {TGT_LABEL}, summed over full trajectory, "
                f"one cell per {SRC_LABEL} token ({src_side}x{src_side}={src_side*src_side} total)")
    fig.tight_layout()
    fig.savefig(f"{sample_dir}/grid_flipped.png", dpi=150)
    plt.close(fig)

    cell_size = IMG_SIZE / src_side                 # box now sized to a source-channel token
    for y in range(src_side):
        for x in range(src_side):
            k_idx = y * src_side + x
            m = shared_normalize(per_token[k_idx]).reshape(tgt_side, tgt_side)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(src_np, cmap="gray")         # box goes on source channel now
            rect = patches.Rectangle((x * cell_size, y * cell_size), cell_size, cell_size,
                                    linewidth=2, edgecolor="red", facecolor="none")
            axes[0].add_patch(rect)
            axes[0].set_title(f"{SRC_LABEL} token ({y},{x})"); axes[0].axis("off")
            axes[1].imshow(overlay_on(tgt_gen_np, m))    # heatmap on generated target
            axes[1].set_title(f"this {SRC_LABEL} token's attention over {TGT_LABEL}\n(summed over full trajectory)")
            axes[1].axis("off")
            axes[2].imshow(tgt_gen_np, cmap="gray")      # plain generated target, the image the heatmap is overlaid on
            axes[2].set_title(f"{TGT_LABEL} (plain)")
            axes[2].axis("off")
            fig.tight_layout()
            fig.savefig(f"{points_dir}/token_{y:02d}_{x:02d}.png", dpi=100)
            plt.close(fig)




for i in SAMPLE_INDICES:
    x, mask = dataset[i]
    x = x.unsqueeze(0).to(DEVICE)
    mask = mask.unsqueeze(0).to(DEVICE)

    '''
    check for corresponding human annotations
    '''
    name = dataset.chA_files[i].name
    subject, rest = name.split("_cell", 1)
    cell_id, rest = rest.split("_f", 1)
    frame = rest.split("_")[0]

    hand_mask_dir = Path("data/annotations") / subject / "hand_drawn_mask"
    has_mask = (hand_mask_dir / f"{cell_id}_handdrawnmask_movie").exists()


    src_img = x[:, SRC_IDX:SRC_IDX+1]
    tgt_img = x[:, TGT_IDX:TGT_IDX+1]

    recon, agg, per_query, src_side, tgt_side = generate_and_collect(src_img, tgt_img, mask)
    src_np = to_img(src_img)
    tgt_real_np = to_img(tgt_img)
    tgt_gen_np = to_img(recon)

    sample_dir = f"{OUT_DIR}/sample_{i}"

    if VIEW_MODE == "row":
        save_sample(sample_dir, src_np, tgt_real_np, tgt_gen_np, agg, per_query, src_side, tgt_side)
    elif VIEW_MODE == "column":
        save_sample_flipped(sample_dir, src_np, tgt_real_np, tgt_gen_np, agg, per_query, src_side, tgt_side)
    else:
        raise ValueError(f"unknown VIEW_MODE: {VIEW_MODE}")

    if has_mask:
        mask_movie_dir = hand_mask_dir / f"{cell_id}_handdrawnmask_movie"
        save_handdrawn_mask_context(sample_dir, mask_movie_dir, int(frame))

    n_steps = len(scheduler.timesteps[scheduler.timesteps <= START_T])
    if VIEW_MODE == "row":
        print(f"sample idx {i} -> {sample_dir}/  "
              f"({src_side*src_side} {SRC_LABEL} tokens, {n_steps} steps aggregated, "
              f"grid.png + {tgt_side*tgt_side} individual files)")
    else:
        print(f"sample idx {i} -> {sample_dir}/  "
              f"({tgt_side*tgt_side} {TGT_LABEL} queries, {n_steps} steps aggregated, "
              f"grid_flipped.png + {src_side*src_side} individual files)")

'''
below for visualziing all query heatmap- overlaid on a query img
'''
# for i in SAMPLE_INDICES:
#     x, mask = dataset[i]
#     x = x.unsqueeze(0).to(DEVICE)
#     mask = mask.unsqueeze(0).to(DEVICE)

#     src_img = x[:, SRC_IDX:SRC_IDX+1]
#     tgt_img = x[:, TGT_IDX:TGT_IDX+1]

#     recon, agg, per_query, src_side, tgt_side = generate_and_collect(src_img, tgt_img, mask)
#     src_np = to_img(src_img)
#     tgt_real_np = to_img(tgt_img)
#     tgt_gen_np = to_img(recon)

#     sample_dir = f"{OUT_DIR}/sample_{i}"
#     # save_sample(sample_dir, src_np, tgt_real_np, tgt_gen_np, agg, per_query, src_side, tgt_side)
#     save_sample_flipped(sample_dir, src_np, tgt_real_np, tgt_gen_np, agg, per_query, src_side, tgt_side)

#     n_steps = len(scheduler.timesteps[scheduler.timesteps <= START_T])
#     print(f"sample idx {i} -> {sample_dir}/  "
#           f"({src_side*src_side} {SRC_LABEL} tokens, {n_steps} steps aggregated, "
#           f"grid.png + grid_flipped.png + "
#           f"{tgt_side*tgt_side + src_side*src_side} individual files)")

print(f"\ndone. {len(SAMPLE_INDICES)} samples saved under {OUT_DIR}/sample_<idx>/")


