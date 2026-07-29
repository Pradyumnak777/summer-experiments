'''
sample form both models at a certain epoch, using the same seeded noise, and save 
the plot under a new directory /diffusion_1ch_comparisons/
'''
import os
import torch
from torchvision.utils import make_grid, save_image
from diffusers import DDPMScheduler
from diffusion_model import build_unet

MODEL_weight_loaded = "diffusion_checkpoints/ddpm_chB_128/unet_ema_epoch40.pt"
MODEL_standard = "diffusion_checkpoints/ddpm_chB_128_no_preweights/unet_ema_epoch40.pt"

DEVICE   = torch.device("cuda:9")
IMG_SIZE = 128
N        = 2
SEED     = 0
EPOCH    = 40
OUT_DIR  = "diffusion_1ch_comparisons"

scheduler = DDPMScheduler(num_train_timesteps=1000)


@torch.no_grad()
def sample_from_ckpt(ckpt_path, noise):
    model = build_unet(IMG_SIZE, channels=1).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    scheduler.set_timesteps(1000)
    x = noise.clone()
    for t in scheduler.timesteps:
        noise_pred = model(x, t).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

    x = (x * 0.5 + 0.5).clamp(0, 1)
    return x


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    noise = torch.randn(N, 1, IMG_SIZE, IMG_SIZE, device=DEVICE, generator=generator)

    x_weight_loaded = sample_from_ckpt(MODEL_weight_loaded, noise)
    x_standard      = sample_from_ckpt(MODEL_standard, noise)

    save_image(make_grid(x_weight_loaded, nrow=N), f"{OUT_DIR}/epoch{EPOCH}_warmstart.png")
    save_image(make_grid(x_standard, nrow=N), f"{OUT_DIR}/epoch{EPOCH}_scratch.png")

    comparison = torch.cat([x_weight_loaded, x_standard], dim=0)
    save_image(make_grid(comparison, nrow=N), f"{OUT_DIR}/epoch{EPOCH}_comparison.png")

    print(f"saved comparison grids to {OUT_DIR}/")