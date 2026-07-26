'''
generate fresh 1-channel cells from the trained DDPM checkpoint.
use the EMA checkpoint, it gives cleaner samples.
'''
import os
import torch
from torchvision.utils import make_grid, save_image
from diffusers import DDPMScheduler
from diffusion_model import build_unet

DEVICE   = torch.device("cuda:9")
CKPT     = "diffusion_checkpoints/ddpm_chA_128/unet_ema_epoch10.pt"
IMG_SIZE = 128
N        = 16
OUT      = "diffusion_checkpoints/ddpm_chA_128/generated"
os.makedirs(OUT, exist_ok=True)

model = build_unet(IMG_SIZE, channels=1).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

scheduler = DDPMScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(1000)

x = torch.randn(N, 1, IMG_SIZE, IMG_SIZE, device=DEVICE)
with torch.no_grad():
    for t in scheduler.timesteps:
        noise_pred = model(x, t).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

x = (x * 0.5 + 0.5).clamp(0, 1)
save_image(make_grid(x, nrow=4), f"{OUT}/chA.png")
print(f"saved generated samples to {OUT}/")