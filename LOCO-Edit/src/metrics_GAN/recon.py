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
import torch.nn.functional as F
import os
from torchvision import transforms
from torchvision.utils import save_image
import warnings
warnings.filterwarnings("ignore")

REPO_DIR = "GAN/stylegan2-ada-pytorch"
sys.path.insert(0, REPO_DIR)
import legacy
from inversion_opti import project_batch, compute_w_stats

DEVICE = torch.device("cuda:9")
torch.backends.cudnn.benchmark = True   # fixed input shape every step -> let cudnn autotune convs

#loading deep feature metrics
import lpips
loss_fn = lpips.LPIPS(net='alex').to(DEVICE)
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE).eval()
dino_model.requires_grad_(False)


SET      = "validation"
PKL      = "GAN/stylegan2-ada-pytorch/training-runs/00010-data_npy-auto1-kimg1000-ada-bg/network-snapshot-001000.pkl"
CHA_DIR  = f"data/singlecell_chA_split/{SET}"
CHB_DIR  = f"data/singlecell_chB_split/{SET}"
SEED = 0
BATCH_SIZE = 64   #using batched projection
MAX_SAMPLES = None
NUM_STEPS = 1000
W_CACHE = f"metrics_GAN/projected_ws_{SET}.pt"   # so re-running metrics doesn't redo the 1000-step optimization

#reqd. transforms for dino
dino_norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def unnormalize(t):
    return (t * 0.5 + 0.5).clamp(0, 1)

def dino_patch_similarity_per_channel(x_u, xr_u):
    
    scores = torch.empty(x_u.size(0), 2)
    
    with torch.no_grad():
        for c in range(2):
            # 1. Repeat single channel to 3 channels and resize to 224x224 (divisible by 14)
            a = x_u[:, c:c+1].repeat(1, 3, 1, 1)
            b = xr_u[:, c:c+1].repeat(1, 3, 1, 1)
            
            a_res = F.interpolate(a, size=(224, 224), mode='bicubic', antialias=True)
            b_res = F.interpolate(b, size=(224, 224), mode='bicubic', antialias=True)
            
            # 2. Apply ImageNet normalization
            a_norm = dino_norm(a_res)
            b_norm = dino_norm(b_res)
            
            # 3. Extract patch tokens [B, num_patches, embed_dim] (excluding CLS)
            feat_a = dino_model.get_intermediate_layers(a_norm, n=1)[0]
            feat_b = dino_model.get_intermediate_layers(b_norm, n=1)[0]
            
            # 4. Normalize along feature dimension and compute cosine similarity per patch
            feat_a = F.normalize(feat_a, dim=-1)
            feat_b = F.normalize(feat_b, dim=-1)
            
            # Average across spatial patches -> shape [B]
            patch_sim = (feat_a * feat_b).sum(dim=-1).mean(dim=-1) # this is cosine similarty per patch
            scores[:, c] = patch_sim.cpu()
            
    return scores


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
             max_samples=MAX_SAMPLES, log_file="metrics_GAN/progress.txt",
             save_dir=f"metrics_GAN/images_{SET}"):
    
    os.makedirs(save_dir, exist_ok=True)
    sample_idx = 0

    mses, ssims, lpipss, dinos, all_ws = [], [], [], [], []
    n_batches = max_samples if max_samples is not None else len(loader)

    with open(log_file, "w") as f:
        for i, (x, mask) in enumerate(loader):
            if max_samples is not None and i >= max_samples:
                break
            x = x.to(device)
            mask = mask.to(device).float()
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
                
            target = to_target_range(x)

            start_msg = f"Inverting batch {i+1}/{n_batches} (size {x.shape[0]})...\n"
            print(start_msg.strip())
            f.write(start_msg)
            f.flush()

            ws_final = project_batch(G, D, targets=target, w_avg=w_avg, w_std=w_std,
                                     num_steps=num_steps, device=device, verbose=False)
            with torch.no_grad():
                x_recon = G.synthesis(ws_final, noise_mode='const')

            all_ws.append(ws_final.detach().cpu())

            # 1. Unnormalized and clamped tensors
            # x_u, xr_u = unnormalize(x), unnormalize(x_recon)
            # x_recon_clamped = x_recon.clamp(-1.0, 1.0)
            
            x_recon_masked = torch.where(mask > 0.5, x_recon, torch.tensor(-1.0, device=device))
            
            x_u = unnormalize(x)
            xr_u = unnormalize(x_recon_masked) * mask
            
            # 2. Pixel & Traditional Metrics
            batch_mse = ((xr_u - x_u) ** 2).mean(dim=(2, 3)).cpu()  # [B, 2]
            mses.append(batch_mse)

            x_np, xr_np = x_u.detach().cpu().numpy(), xr_u.detach().cpu().numpy()
            batch_ssim = torch.empty(x.shape[0], 2)
            for b in range(x.shape[0]):
                for c in range(2):
                    batch_ssim[b, c] = structural_similarity(x_np[b, c], xr_np[b, c], data_range=1.0)
            ssims.append(batch_ssim)

            # 3. Deep Feature Metrics (LPIPS + DINOv2 Patch Sim)
            batch_lpips = lpips_per_channel(x, x_recon_masked)  # [B, 2]
            lpipss.append(batch_lpips)

            batch_dino = dino_patch_similarity_per_channel(x_u, xr_u)  # [B, 2]
            dinos.append(batch_dino)

            # 4. Batch Logging
            batch_psnr = 10.0 * torch.log10(1.0 / batch_mse.clamp(min=1e-12))
            
            log_lines = [f"\n--- Batch {i+1}/{n_batches} Summary (size {x.shape[0]}) ---"]
            log_lines.append(f"{'':5} {'MSE':>20} {'PSNR (dB)':>20} {'SSIM':>18} {'LPIPS':>18} {'DINO Patch':>18}")
            for c, name in [(0, "chA"), (1, "chB")]:
                mse_std  = batch_mse[:, c].std().item() if x.shape[0] > 1 else 0.0
                psnr_std = batch_psnr[:, c].std().item() if x.shape[0] > 1 else 0.0
                ssim_std = batch_ssim[:, c].std().item() if x.shape[0] > 1 else 0.0
                lp_std   = batch_lpips[:, c].std().item() if x.shape[0] > 1 else 0.0
                dino_std = batch_dino[:, c].std().item() if x.shape[0] > 1 else 0.0

                log_lines.append(
                    f"{name:5} {batch_mse[:, c].mean().item():9.5f} +- {mse_std:<8.5f}"
                    f"{batch_psnr[:, c].mean().item():9.2f} +- {psnr_std:<8.2f}"
                    f"{batch_ssim[:, c].mean().item():8.3f} +- {ssim_std:<7.3f}"
                    f"{batch_lpips[:, c].mean().item():8.3f} +- {lp_std:<7.3f}"
                    f"{batch_dino[:, c].mean().item():8.3f} +- {dino_std:<7.3f}"
                )
            log_lines.append("-" * 103)
            batch_report = "\n".join(log_lines) + "\n"

            print(batch_report)
            f.write(batch_report)
            f.flush()

            # 5. Save Images
            channel_names = ["chA", "chB"]
            for b in range(x.shape[0]):
                sample_folder = os.path.join(save_dir, f"{sample_idx:04d}")
                os.makedirs(sample_folder, exist_ok=True)

                for c_idx, ch_name in enumerate(channel_names):
                    save_image(x_u[b, c_idx:c_idx+1], os.path.join(sample_folder, f"orig_{ch_name}.png"))
                    save_image(xr_u[b, c_idx:c_idx+1], os.path.join(sample_folder, f"recon_{ch_name}.png"))

                sample_idx += 1

    torch.save(torch.cat(all_ws), W_CACHE)
    return torch.cat(mses), torch.cat(ssims), torch.cat(lpipss), torch.cat(dinos)


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
    
    D.b4.mbstd.group_size = 1

    w_avg, w_std = compute_w_stats(G, DEVICE)

    mse, ssim, lp, dino_sim = evaluate(G, D, loader, DEVICE, w_avg, w_std, NUM_STEPS, MAX_SAMPLES)
    psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))

    METRICS_OUT = f"metrics_GAN/metrics_{SET}.txt"

    lines = []
    lines.append(f"checkpoint : {PKL}")
    lines.append(f"split      : {SET}   ({mse.shape[0]} samples, full frame)\n")
    lines.append(f"{'':5} {'MSE':>20} {'PSNR (dB)':>20} {'SSIM':>18} {'LPIPS':>18} {'DINO Patch':>18}")
    for c, name in [(0, "chA"), (1, "chB")]:
        lines.append(
            f"{name:5} {mse[:, c].mean().item():9.5f} +- {mse[:, c].std().item():<8.5f}"
            f"{psnr[:, c].mean().item():9.2f} +- {psnr[:, c].std().item():<8.2f}"
            f"{ssim[:, c].mean().item():8.3f} +- {ssim[:, c].std().item():<7.3f}"
            f"{lp[:, c].mean().item():8.3f} +- {lp[:, c].std().item():<7.3f}"
            f"{dino_sim[:, c].mean().item():8.3f} +- {dino_sim[:, c].std().item():<7.3f}"
        )

    report = "\n".join(lines)
    print("\n" + report)

    with open(METRICS_OUT, "w") as f:
        f.write(report + "\n")