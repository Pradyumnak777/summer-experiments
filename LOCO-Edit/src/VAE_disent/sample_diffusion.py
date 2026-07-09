'''
generate fresh 2-channel cells from a trained DDPM checkpoint.
use the EMA checkpoint - it gives cleaner samples.
'''
import os
import torch
from torchvision.utils import make_grid, save_image
from diffusers import DDPMScheduler
from diffusion_model import build_unet

DEVICE   = torch.device("cuda:9")
CKPT     = "diffusion_checkpoints/ddpm_2ch_128/unet_ema_epoch199.pt"
IMG_SIZE = 128
N        = 16
OUT      = "diffusion_checkpoints/ddpm_2ch_128/generated"
os.makedirs(OUT, exist_ok=True)

model = build_unet(IMG_SIZE, channels=2).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

scheduler = DDPMScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(1000)

x = torch.randn(N, 2, IMG_SIZE, IMG_SIZE, device=DEVICE)
with torch.no_grad():
    for t in scheduler.timesteps:
        noise_pred = model(x, t).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

x = (x * 0.5 + 0.5).clamp(0, 1)
for ch, name in [(0, "chA"), (1, "chB")]:
    save_image(make_grid(x[:, ch:ch+1], nrow=4), f"{OUT}/{name}.png")

#magenta/green composite, matching your preview convention
comp = torch.cat([x[:, 1:2], x[:, 0:1], x[:, 1:2]], dim=1)
save_image(make_grid(comp, nrow=4), f"{OUT}/composite.png")
print(f"saved generated samples to {OUT}/")