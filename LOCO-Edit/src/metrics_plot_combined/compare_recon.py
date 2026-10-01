'''
takes one validation image, reconstructs it with all 3 models and plots them side by side:
    - VAE        : encode -> decode(mu), same as metrics_vae/recon.py
    - GAN        : optimization-based inversion (project_batch) -> G.synthesis, same as metrics_GAN/recon.py
    - diffusion  : ddim inversion up to T -> ddim denoise back to 0, same as metrics_diffusion/recon.py

saves one combined figure (rows = chA/chB, cols = original + 3 recons) plus every panel as its
own raw 128x128 png. per-image mse/ssim go to metrics.txt.

run from src/:
    python -m metrics_plot_combined.compare_recon --idx 42 --t 500
    python -m metrics_plot_combined.compare_recon --name sub3_cell7_f012 --t 500 --mask_recon
'''

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
from skimage.metrics import structural_similarity
from torchvision.utils import save_image

from VAE_disent.data_utils import twoChannelDataset
from VAE_disent.model import twoChannelVAE
from VAE_disent.diffusion_model import build_unet

REPO_DIR = "GAN/stylegan2-ada-pytorch"
sys.path.insert(0, REPO_DIR)
import legacy
from inversion_opti import project_batch, compute_w_stats

#same data + checkpoints as the three metrics_*/recon.py scripts, so the plot matches the reported numbers
SET        = "validation"
CHA_DIR    = f"data/singlecell_chA_split/{SET}"
CHB_DIR    = f"data/singlecell_chB_split/{SET}"
MASK_DIR   = "data/singlecell_mask"

VAE_CKPT   = "vae_checkpoints_masked/beta_vae_beta=1_16/vae_epoch99.pt"
LATENT_DIM = 16
GAN_PKL    = "GAN/stylegan2-ada-pytorch/training-runs/00010-data_npy-auto1-kimg1000-ada-bg/network-snapshot-001000.pkl"
DIFF_CKPT  = "diffusion_checkpoints/ddpm_2ch_128_masked/unet_ema_epoch80.pt"
IMG_SIZE   = 128

OUT_ROOT   = "metrics_plot_combined/outputs"


def unnormalize(t):
    #[-1,1] -> [0,1], same as the metric scripts so data_range=1.0 holds
    return (t * 0.5 + 0.5).clamp(0, 1)


#VAE
@torch.no_grad()
def recon_vae(x, device):
    model = twoChannelVAE(latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(VAE_CKPT, map_location=device))
    model.to(device).eval()
    mu, _ = model.encode(x)
    return model.decode(mu)   #decode the mean, no sampling


#GAN
def recon_gan(x, device, num_steps):
    with open(GAN_PKL, "rb") as f:
        nets = legacy.load_network_pkl(f)
    G = nets["G_ema"].to(device).eval().requires_grad_(False)
    D = nets["D"].to(device).eval().requires_grad_(False)
    D.b4.mbstd.group_size = 1   #batch of 1, minibatch-std needs group size 1

    w_avg, w_std = compute_w_stats(G, device)
    target = ((x * 0.5 + 0.5).clamp(0, 1) * 255.0).round()   #project_batch wants [0,255]
    ws = project_batch(G, D, targets=target, w_avg=w_avg, w_std=w_std,
                       num_steps=num_steps, device=device, verbose=True)
    with torch.no_grad():
        return G.synthesis(ws, noise_mode="const"), ws


#diffusion
scheduler = DDPMScheduler(num_train_timesteps=1000)


def eps_pred(model, x, t_int):
    t = torch.full((x.shape[0],), t_int, device=x.device, dtype=torch.long)
    with torch.autocast(device_type="cuda", dtype=torch.float16):   #matches metrics_diffusion/recon.py
        eps = model(x, t).sample
    return eps.float()


def ddim_step(model, x_t, t_int, t_prev_int, ac):
    #one deterministic ddim reverse step, eta=0
    eps = eps_pred(model, x_t, t_int)
    ab_t = ac[t_int]
    ab_prev = ac[t_prev_int] if t_prev_int > 0 else torch.tensor(1.0, device=x_t.device)
    x0_pred = (x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    return ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps


def ddim_inversion_step(model, x_prev, t_prev_int, t_int, ac):
    #mirror of ddim_step, walks from t_prev up to t using eps at the point we already have
    eps = eps_pred(model, x_prev, t_prev_int)
    ab_prev = ac[t_prev_int] if t_prev_int > 0 else torch.tensor(1.0, device=x_prev.device)
    ab_t = ac[t_int]
    x0_pred = (x_prev - (1 - ab_prev).sqrt() * eps) / ab_prev.sqrt()
    return ab_t.sqrt() * x0_pred + (1 - ab_t).sqrt() * eps


@torch.no_grad()
def recon_diffusion(x, device, t_end, steps):
    model = build_unet(IMG_SIZE, channels=2).to(device)
    model.load_state_dict(torch.load(DIFF_CKPT, map_location=device))
    model.eval().requires_grad_(False)
    ac = scheduler.alphas_cumprod.to(device)

    #0 -> T (inversion)
    ts = torch.linspace(0, t_end, steps + 1).round().long().tolist()
    x_t = x
    for a, b in zip(ts[:-1], ts[1:]):
        if a != b:
            x_t = ddim_inversion_step(model, x_t, a, b, ac)

    #T -> 0 (denoise)
    ts = ts[::-1]
    x0 = x_t
    for a, b in zip(ts[:-1], ts[1:]):
        if a != b:
            x0 = ddim_step(model, x0, a, b, ac)
    return x0


#---------------- metrics + plotting ----------------
def per_channel_metrics(x_u, xr_u):
    #x_u, xr_u: [2,H,W] in [0,1] -> list of (mse, ssim) per channel, full frame
    out = []
    for c in range(2):
        a, b = x_u[c].cpu().numpy(), xr_u[c].cpu().numpy()
        out.append((float(((a - b) ** 2).mean()), float(structural_similarity(a, b, data_range=1.0))))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=int, default=0, help="index into the validation set (same order as the metric scripts)")
    p.add_argument("--name", type=str, default=None, help="sample base name e.g. sub3_cell7_f012, overrides --idx")
    p.add_argument("--t", type=int, default=500, help="diffusion inversion timestep T")
    p.add_argument("--ddim_steps", type=int, default=100, help="ddim steps for both inversion and denoising")
    p.add_argument("--gan_steps", type=int, default=1000, help="gan inversion optimization steps")
    p.add_argument("--mask_recon", action="store_true",
                   help="mask all 3 recons with the cell mask before plotting/scoring (like metrics_GAN does)")
    p.add_argument("--device", type=str, default="cuda:9")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)
    if args.name is not None:
        names = [f.name[:-len("_chA.png")] for f in dataset.chA_files]
        assert args.name in names, f"{args.name} not in the {SET} set (after empty-mask filtering)"
        idx = names.index(args.name)
    else:
        idx = args.idx
    base = dataset.chA_files[idx].name[:-len("_chA.png")]
    print(f"sample {idx}: {base}")

    x, mask = dataset[idx]
    x, mask = x.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)   #[1,2,H,W], [1,1,H,W]

    print("VAE ...")
    r_vae = recon_vae(x, device)
    print(f"GAN inversion ({args.gan_steps} steps) ...")
    r_gan, gan_ws = recon_gan(x, device, args.gan_steps)
    print(f"diffusion (T={args.t}, {args.ddim_steps} ddim steps) ...")
    r_diff = recon_diffusion(x, device, args.t, args.ddim_steps)

    recons = {"VAE": r_vae, "GAN": r_gan, f"Diffusion (T={args.t})": r_diff}
    if args.mask_recon:
        #same masking as metrics_GAN/recon.py, applied to every method so they're comparable
        recons = {k: torch.where(mask > 0.5, v, torch.full_like(v, -1.0)) for k, v in recons.items()}

    x_u = unnormalize(x[0])
    recons_u = {k: unnormalize(v[0]) for k, v in recons.items()}
    scores = {k: per_channel_metrics(x_u, v) for k, v in recons_u.items()}

    tag = f"{idx:04d}_{base}_T{args.t}" + ("_masked" if args.mask_recon else "")
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    #inverted gan latent, same format as inversion_opti.py's projected_w.npz ([1, num_ws, w_dim])
    #so it loads straight into gan_latent_traversal_inversion.py / gan_pca_inverted_w.py
    np.savez(f"{out_dir}/gan_projected_w.npz", w=gan_ws.cpu().numpy())

    #separate images, raw pixels, one png per method per channel
    fnames = {"VAE": "vae", "GAN": "gan", f"Diffusion (T={args.t})": f"diffusion_T{args.t}"}
    for c, ch in enumerate(["chA", "chB"]):
        save_image(x_u[c:c+1], f"{out_dir}/original_{ch}.png")
        for k, v in recons_u.items():
            save_image(v[c:c+1], f"{out_dir}/{fnames[k]}_{ch}.png")

    #combined figure: rows = channels, cols = original + recons
    cols = [("Original", x_u)] + list(recons_u.items())
    fig, axes = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 6.8), layout="constrained")
    for c, ch in enumerate(["chA", "chB"]):
        for j, (title, img) in enumerate(cols):
            ax = axes[c, j]
            ax.imshow(img[c].cpu().numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_title(title, fontsize=12)
            if j == 0:
                ax.set_ylabel(ch, fontsize=12)
    fig.suptitle(f"{base} (val idx {idx})" + ("  [recons masked]" if args.mask_recon else ""), fontsize=12)
    fig.savefig(f"{out_dir}/combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    #per-image numbers, full frame, [0,1]
    with open(f"{out_dir}/metrics.txt", "w") as f:
        f.write(f"sample: {base}  (val idx {idx})\nmask_recon: {args.mask_recon}\n\n")
        f.write(f"{'method':22} {'chA MSE':>9} {'chA SSIM':>9} {'chB MSE':>9} {'chB SSIM':>9}\n")
        for k, s in scores.items():
            f.write(f"{k:22} {s[0][0]:9.5f} {s[0][1]:9.3f} {s[1][0]:9.5f} {s[1][1]:9.3f}\n")
    print(open(f"{out_dir}/metrics.txt").read())
    print(f"saved to {out_dir}/")


if __name__ == "__main__":
    main()
