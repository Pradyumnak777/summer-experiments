'''
difference between inverted image and the actual image.
GAN optimization-based inversion metrics: MSE, PSNR, SSIM, LPIPS.
'''

import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
import numpy as np
from VAE_disent.data_utils import twoChannelDataset
import sys
import warnings
warnings.filterwarnings("ignore")

REPO_DIR = "GAN/stylegan2-ada-pytorch"
sys.path.insert(0, REPO_DIR)
import legacy
from inversion_opti import project_batch, compute_w_stats

DEVICE = torch.device("cuda:9")
torch.backends.cudnn.benchmark = True   # fixed input shape every step -> let cudnn autotune convs

import lpips
loss_fn = lpips.LPIPS(net='alex').to(DEVICE)

SET      = "validation"
PKL      = "GAN/stylegan2-ada-pytorch/training-runs/00013-data_npy_masked-auto1-kimg1000-ada-bg/network-snapshot-001000.pkl"
CHA_DIR  = f"data/singlecell_chA_split/{SET}"
CHB_DIR  = f"data/singlecell_chB_split/{SET}"
SEED = 0
BATCH_SIZE = 64   #using batched projection
MAX_SAMPLES = None
NUM_STEPS = 1000
W_CACHE = f"metrics_GAN/projected_ws_{SET}.pt"   # so re-running metrics doesn't redo the 1000-step optimization


def unnormalize(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


def lpips_per_channel(x, x_recon):
    scores = torch.empty(x.size(0), 2)
    for c in range(2):
        a = x[:, c:c+1].repeat(1, 3, 1, 1)
        b = x_recon[:, c:c+1].repeat(1, 3, 1, 1)
        scores[:, c] = loss_fn(a, b).flatten().cpu()
    return scores


def to_target_range(x):
    # twoChannelDataset gives [B,2,H,W] in [-1,1]; project_batch wants [B,2,H,W] in [0,255]
    return ((x * 0.5 + 0.5).clamp(0, 1) * 255.0).round()


def evaluate(G, D, loader, device, w_avg, w_std, num_steps=NUM_STEPS,
             max_samples=MAX_SAMPLES, log_file="metrics_GAN/progress.txt"):
    mses, ssims, lpipss, all_ws = [], [], [], []
    n_batches = max_samples if max_samples is not None else len(loader)

    with open(log_file, "w") as f:
        for i, (x, _) in enumerate(loader):
            if max_samples is not None and i >= max_samples:
                break
            x = x.to(device)
            target = to_target_range(x)   # [B, 2, H, W], [0,255]

            f.write(f"inverting batch {i+1}/{n_batches} (size {x.shape[0]})...\n")
            f.flush()

            ws_final = project_batch(G, D, targets=target, w_avg=w_avg, w_std=w_std,
                                      num_steps=num_steps, device=device, verbose=False)
            with torch.no_grad():
                x_recon = G.synthesis(ws_final, noise_mode='const')

            all_ws.append(ws_final.detach().cpu())

            x_u, xr_u = unnormalize(x), unnormalize(x_recon)
            mses.append(((xr_u - x_u) ** 2).mean(dim=(2, 3)).cpu())

            x_np, xr_np = x_u.detach().cpu().numpy(), xr_u.detach().cpu().numpy()
            batch_ssim = torch.empty(x.shape[0], 2)
            for b in range(x.shape[0]):
                for c in range(2):
                    batch_ssim[b, c] = structural_similarity(x_np[b, c], xr_np[b, c], data_range=1.0)
            ssims.append(batch_ssim)

            lpipss.append(lpips_per_channel(x, x_recon))

            f.write(f"inverted batch {i+1}/{n_batches}\n")
            f.flush()

    torch.save(torch.cat(all_ws), W_CACHE)
    return torch.cat(mses), torch.cat(ssims), torch.cat(lpipss)


if __name__ == "__main__":
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir="data/singlecell_mask")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with open(PKL, 'rb') as f:
        nets = legacy.load_network_pkl(f)
        G = nets['G_ema'].to(DEVICE).eval()
        D = nets['D'].to(DEVICE).eval()
    G.requires_grad_(False)
    D.requires_grad_(False)
    
    D.b4.mbstd.group_size = 1 #due to error (last batch is size=9..)

    w_avg, w_std = compute_w_stats(G, DEVICE)   #computed once for the whole run

    mse, ssim, lp = evaluate(G, D, loader, DEVICE, w_avg, w_std, NUM_STEPS, MAX_SAMPLES)
    psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))

    # print(f"\ncheckpoint : {PKL}")
    # print(f"split      : {SET}   ({mse.shape[0]} samples, full frame)\n")
    # print(f"{'':5} {'MSE':>20} {'PSNR (dB)':>20} {'SSIM':>18} {'LPIPS':>18}")
    # for c, name in [(0, "chA"), (1, "chB")]:
    #     print(f"{name:5} {mse[:, c].mean().item():9.5f} +- {mse[:, c].std().item():<8.5f}"
    #           f"{psnr[:, c].mean().item():9.2f} +- {psnr[:, c].std().item():<8.2f}"
    #           f"{ssim[:, c].mean().item():8.3f} +- {ssim[:, c].std().item():<7.3f}"
    #           f"{lp[:, c].mean().item():8.3f} +- {lp[:, c].std().item():<7.3f}")
    
    METRICS_OUT = f"metrics_GAN/metrics_{SET}.txt"

    lines = []
    lines.append(f"checkpoint : {PKL}")
    lines.append(f"split      : {SET}   ({mse.shape[0]} samples, full frame)\n")
    lines.append(f"{'':5} {'MSE':>20} {'PSNR (dB)':>20} {'SSIM':>18} {'LPIPS':>18}")
    for c, name in [(0, "chA"), (1, "chB")]:
        lines.append(
            f"{name:5} {mse[:, c].mean().item():9.5f} +- {mse[:, c].std().item():<8.5f}"
            f"{psnr[:, c].mean().item():9.2f} +- {psnr[:, c].std().item():<8.2f}"
            f"{ssim[:, c].mean().item():8.3f} +- {ssim[:, c].std().item():<7.3f}"
            f"{lp[:, c].mean().item():8.3f} +- {lp[:, c].std().item():<7.3f}"
        )

    report = "\n".join(lines)
    print("\n" + report)

    with open(METRICS_OUT, "w") as f:
        f.write(report + "\n")
