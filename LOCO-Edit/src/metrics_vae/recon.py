"""
metrics_vae/recon.py

reconstruction metrics for the 2 channel beta vae.
mse, psnr and ssim over the full 128x128 frame, reported per channel.
"""
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
import lpips

from VAE_disent.model import twoChannelVAE
from VAE_disent.data_utils import twoChannelDataset

DEVICE     = torch.device("cuda:9")
SET        = "validation"
CHA_DIR    = f"data/singlecell_chA_split/{SET}"
CHB_DIR    = f"data/singlecell_chB_split/{SET}"
MASK_DIR   = "data/singlecell_mask"
CKPT       = "vae_checkpoints_masked/beta_vae_beta=1_16/vae_epoch99.pt"
LATENT_DIM = 16
BATCH_SIZE = 32

loss_fn = lpips.LPIPS(net='alex').to(DEVICE)


def unnormalize(t):
    #model lives in [-1,1] because of normalize(0.5,0.5) and the tanh at the end.
    #push it to [0,1] so psnr and ssim can just use data_range=1.0
    return (t * 0.5 + 0.5).clamp(0, 1)


def lpips_per_channel(x, x_recon):
    #x, x_recon: [B, 2, H, W] in [-1,1]. lpips wants 3-channel input, so replicate each channel
    scores = torch.empty(x.size(0), 2)
    for c in range(2):
        a = x[:, c:c+1].repeat(1, 3, 1, 1)
        b = x_recon[:, c:c+1].repeat(1, 3, 1, 1)
        scores[:, c] = loss_fn(a, b).flatten().cpu()
    return scores


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mses, ssims, kls, lpipss = [], [], [], []

    for x, _ in loader:
        #dataset gives back (masked_image, mask). the image is already masked,
        #we just don't use the mask tensor for anything here
        x = x.to(device)
        # x_recon, mu, logvar = model(x)
        
        mu, logvar = model.encode(x)
        x_recon = model.decode(mu)

        x_u  = unnormalize(x)
        xr_u = unnormalize(x_recon)

        #plain mse, averaged over every pixel in the frame. shape is [batch, 2]
        mses.append(((xr_u - x_u) ** 2).mean(dim=(2, 3)).cpu())

        #kl per image, summed across the 16 latent dims.
        kls.append((-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).cpu())

        #ssim wants numpy and only does one 2d image at a time, so loop it
        x_np, xr_np = x_u.cpu().numpy(), xr_u.cpu().numpy()
        batch_ssim = torch.empty(x.size(0), 2)
        for i in range(x.size(0)):
            for c in range(2):
                batch_ssim[i, c] = structural_similarity(x_np[i, c], xr_np[i, c], data_range=1.0) #this is 8bit image, so data_range is 1.0 by default
        ssims.append(batch_ssim)

        #lpips wants the raw [-1,1] tensors, not the unnormalized [0,1] ones
        lpipss.append(lpips_per_channel(x, x_recon))

    return torch.cat(mses), torch.cat(ssims), torch.cat(kls), torch.cat(lpipss)


if __name__ == "__main__":
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)
    #shuffle off so the numbers are the same every run
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = twoChannelVAE(latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.to(DEVICE)

    mse, ssim, kl, lp = evaluate(model, loader, DEVICE)

    #psnr is just mse in log units
    psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))

    print(f"\ncheckpoint : {CKPT}")
    print(f"split      : {SET}   ({mse.shape[0]} samples, full frame)\n")
    print(f"{'':5} {'MSE':>20} {'PSNR (dB)':>20} {'SSIM':>18} {'LPIPS':>18}")
    for c, name in [(0, "chA"), (1, "chB")]:
        print(f"{name:5} {mse[:, c].mean().item():9.5f} +- {mse[:, c].std().item():<8.5f}"
              f"{psnr[:, c].mean().item():9.2f} +- {psnr[:, c].std().item():<8.2f}"
              f"{ssim[:, c].mean().item():8.3f} +- {ssim[:, c].std().item():<7.3f}"
              f"{lp[:, c].mean().item():8.3f} +- {lp[:, c].std().item():<7.3f}")
    print(f"\nboth channels pooled MSE : {mse.mean().item():.5f}")
    # print(f"KL (nats/sample, {LATENT_DIM} dims) : {kl.mean().item():.3f} +- {kl.std().item():.3f}")