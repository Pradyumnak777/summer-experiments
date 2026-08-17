'''
chose a timestep, apply tweedie formula on each img in the validation dataset, then perform same 
recon loss b/w them both as done for VAE (mse, psnr and ssim)
'''
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import DDPMScheduler
from skimage.metrics import structural_similarity

from VAE_disent.data_utils import twoChannelDataset
from VAE_disent.diffusion_model import build_unet



DEVICE   = torch.device("cuda:4")

import lpips
loss_fn = lpips.LPIPS(net='alex').to(DEVICE)
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE).eval()
dino_model.requires_grad_(False)

CKPT     = "diffusion_checkpoints/ddpm_2ch_128_masked/unet_ema_epoch80.pt"
CHA_DIR  = "data/singlecell_chA_split/validation"
CHB_DIR  = "data/singlecell_chB_split/validation"

IMG_SIZE   = 128
BATCH_SIZE = 128
EDIT_T     = 400

DENOISE_STEPS = 25
SEED       = 0

scheduler = DDPMScheduler(num_train_timesteps=1000)
alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)

#reqd. transforms for dino
dino_norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def unnormalize(t):
    #model lives in [-1,1] so push to [0,1], same as the vae script so data_range=1.0 stays valid
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


def get_x0(model, x_t, t_int):
    #tweedie, predicts the clean x0 from a noised x_t at timestep t
    t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        eps = model(x_t, t).sample
    eps = eps.float()
    ab = alphas_cumprod[t_int]
    return (x_t - (1 - ab).sqrt() * eps) / ab.sqrt()


def ddim_step(model, x_t, t_int, t_prev_int):
    #one deterministic ddim reverse step, eta=0, from t_int down to t_prev_int
    t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        eps = model(x_t, t).sample
    eps = eps.float()
    ab_t = alphas_cumprod[t_int]
    #t_prev<=0 counts as fully clean
    ab_prev = alphas_cumprod[t_prev_int] if t_prev_int > 0 else torch.tensor(1.0, device=x_t.device)
    x0_pred = (x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    return ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps


def ddim_denoise(model, x_t, t_start, denoise_steps=DENOISE_STEPS):
    #runs the full reverse trajectory from t_start down to 0, this is what actually gets presented
    timesteps = torch.linspace(t_start, 0, denoise_steps + 1).round().long().tolist()
    x = x_t
    for i in range(len(timesteps) - 1):
        t_cur, t_next = timesteps[i], timesteps[i + 1]
        if t_cur == t_next:
            continue
        x = ddim_step(model, x, t_cur, t_next)
    return x

def lpips_per_channel(x, x_recon):
    #x, x_recon: [B, 2, H, W] in [-1,1]. lpips wants 3-channel input, so replicate each channel
    scores = torch.empty(x.size(0), 2)
    for c in range(2):
        a = x[:, c:c+1].repeat(1, 3, 1, 1)
        b = x_recon[:, c:c+1].repeat(1, 3, 1, 1)
        scores[:, c] = loss_fn(a, b).flatten().cpu()
    return scores



# @torch.no_grad()
# def evaluate(model, loader, device, t_int):
#     mses, ssims, lpipss = [], [], []

#     for x, _ in loader:
#         #dataset gives back (masked_image, mask), only the image is needed here
#         x = x.to(device)

#         #noise the real image up to t then tweedie back. this x0_hat is the edit anchor
#         noise = torch.randn_like(x)
#         t_batch = torch.full((x.shape[0],), t_int, device=device, dtype=torch.long)
#         x_t = scheduler.add_noise(x, noise, t_batch)
#         x0_hat = get_x0(model, x_t, t_int)

#         x_u  = unnormalize(x)
#         xr_u = unnormalize(x0_hat)

#         #mse per image per channel over the whole frame. shape is [batch, 2]
#         mses.append(((xr_u - x_u) ** 2).mean(dim=(2, 3)).cpu())

#         #ssim wants numpy and does one 2d image at a time so loop it
#         x_np, xr_np = x_u.cpu().numpy(), xr_u.cpu().numpy()
#         batch_ssim = torch.empty(x.size(0), 2)
#         for i in range(x.size(0)):
#             for c in range(2):
#                 batch_ssim[i, c] = structural_similarity(x_np[i, c], xr_np[i, c], data_range=1.0)
#         ssims.append(batch_ssim)
        
#         lpipss.append(lpips_per_channel(x, x0_hat))
        
        

#     return torch.cat(mses), torch.cat(ssims), torch.cat(lpipss)


@torch.no_grad()
def evaluate(model, loader, device, t_int, denoise_steps=DENOISE_STEPS):
    mses, ssims, lpipss, dinos = [], [], [], []
    count = 0
    for x, _ in loader:
        print(f"batch{BATCH_SIZE} count: {count}")
        #dataset gives back (masked_image, mask), only the image is needed here
        x = x.to(device)

        #noise the real image up to t, then run the full reverse trajectory back down. this is the true presented anchor
        noise = torch.randn_like(x)
        t_batch = torch.full((x.shape[0],), t_int, device=device, dtype=torch.long)
        x_t = scheduler.add_noise(x, noise, t_batch)
        x0_hat = ddim_denoise(model, x_t, t_int, denoise_steps)

        x_u  = unnormalize(x)
        xr_u = unnormalize(x0_hat)

        #mse per image per channel over the whole frame. shape is [batch, 2]
        mses.append(((xr_u - x_u) ** 2).mean(dim=(2, 3)).cpu())

        #ssim wants numpy and does one 2d image at a time so loop it
        x_np, xr_np = x_u.cpu().numpy(), xr_u.cpu().numpy()
        batch_ssim = torch.empty(x.size(0), 2)
        for i in range(x.size(0)):
            for c in range(2):
                batch_ssim[i, c] = structural_similarity(x_np[i, c], xr_np[i, c], data_range=1.0)
        ssims.append(batch_ssim)
        
        lpipss.append(lpips_per_channel(x, x0_hat))
        dinos.append(dino_patch_similarity_per_channel(x_u, xr_u))
        count+=1
    return torch.cat(mses), torch.cat(ssims), torch.cat(lpipss), torch.cat(dinos)




if __name__ == "__main__":
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir="data/singlecell_mask")
    #shuffle off so the numbers repeat run to run
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    #same seed every run, otherwise the random noise makes the numbers wobble
    torch.manual_seed(SEED)

    model = build_unet(IMG_SIZE, channels=2).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    model.requires_grad_(False)

    mse, ssim, lp, dino_sim = evaluate(model, loader, DEVICE, EDIT_T)

    #psnr is just mse in log units so it comes free
    psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))

    print(f"\ncheckpoint : {CKPT}")
    print(f"timestep   : {EDIT_T}")
    print(f"denoise steps : {DENOISE_STEPS}")
    print(f"samples    : {mse.shape[0]} (full frame)\n")
    print(f"{'':5} {'MSE':>20} {'PSNR (dB)':>20} {'SSIM':>18} {'LPIPS':>18} {'DINO Patch':>18}")
    for c, name in [(0, "chA"), (1, "chB")]:
        print(f"{name:5} {mse[:, c].mean().item():9.5f} +- {mse[:, c].std().item():<8.5f}"
            f"{psnr[:, c].mean().item():9.2f} +- {psnr[:, c].std().item():<8.2f}"
            f"{ssim[:, c].mean().item():8.3f} +- {ssim[:, c].std().item():<7.3f}"
            f"{lp[:, c].mean().item():8.3f} +- {lp[:, c].std().item():<7.3f}"
            f"{dino_sim[:, c].mean().item():8.3f} +- {dino_sim[:, c].std().item():<7.3f}")