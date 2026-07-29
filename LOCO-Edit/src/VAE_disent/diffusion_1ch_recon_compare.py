'''
1. take a real chB image (say from data/singlecell_chB_split/train)
2. Load the 2 models- with pretrained wts and without
3. Perform reconstruction using tweedie's formula (like in the locoedit/jacobian analysis). that is, after regressing back to a
specific noise timestep, apply tweedies formula to get the reconstruction
4. Compare the reconstructions of the 2 models (plot using mmatplotlib or smthng), and save under new dir "diffusion_1ch_recon_coparisons/"
'''

import torch
import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
from diffusion_model import build_unet

MODEL_weight_loaded = "diffusion_checkpoints/ddpm_chB_128/unet_ema_epoch20.pt"
MODEL_standard = "diffusion_checkpoints/ddpm_chB_128_no_preweights/unet_ema_epoch20.pt"


DEVICE   = torch.device("cuda:9")
OUT_DIR  = "diffusion_1ch_recon_coparisons"

CHB_DIR  = "data/singlecell_chB_split/train"
IMG_SIZE = 128
ANCHOR   = 0           # which image in CHB_DIR to use
EDIT_T   = 500         # timestep to noise to, then tweedie's formula back
SEED     = 0
EPOCH = 20

scheduler = DDPMScheduler(num_train_timesteps=1000)
alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)


def load_model(ckpt_path):
    model = build_unet(IMG_SIZE, channels=1).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    model.requires_grad_(False)
    return model


#the PMP/tweedie's estimator - predicts clean x0 from a noised x_t at timestep t
def get_x0(model, x_t, t_int):
    t = torch.tensor(t_int, device=x_t.device)
    eps = model(x_t, t).sample
    ab  = alphas_cumprod[t_int]
    return (x_t - (1 - ab).sqrt() * eps) / ab.sqrt()


def load_real_chB_image(chb_dir, anchor):
    files = sorted([f for f in Path(chb_dir).glob('*.png')])
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    img = Image.open(files[anchor]).convert('L')
    return transform(img).unsqueeze(0)   # [1, 1, H, W]


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    x0_real = load_real_chB_image(CHB_DIR, ANCHOR).to(DEVICE)
    noise   = torch.randn_like(x0_real)
    x_t = scheduler.add_noise(x0_real, noise, torch.tensor([EDIT_T], device=DEVICE))

    model_weight_loaded = load_model(MODEL_weight_loaded)
    model_standard       = load_model(MODEL_standard)

    with torch.no_grad():
        x0_hat_weight_loaded = get_x0(model_weight_loaded, x_t, EDIT_T)
        x0_hat_standard      = get_x0(model_standard, x_t, EDIT_T)

    mse_weight_loaded = F.mse_loss(x0_hat_weight_loaded, x0_real).item()
    mse_standard      = F.mse_loss(x0_hat_standard, x0_real).item()
    print(f"reconstruction MSE (t={EDIT_T}): warm-start={mse_weight_loaded:.5f}  scratch={mse_standard:.5f}")

    orig       = (x0_real[0, 0].cpu()               * 0.5 + 0.5).clamp(0, 1)
    recon_wl   = (x0_hat_weight_loaded[0, 0].cpu()  * 0.5 + 0.5).clamp(0, 1)
    recon_std  = (x0_hat_standard[0, 0].cpu()       * 0.5 + 0.5).clamp(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, title in zip(
        axes,
        [orig, recon_wl, recon_std],
        ["original", f"warm-start recon (t={EDIT_T})", f"scratch recon (t={EDIT_T})"],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/anchor{ANCHOR}_t{EDIT_T}_recon_comparison.png", dpi=200)
    plt.close()

    print(f"saved reconstruction comparison to {OUT_DIR}/anchor{ANCHOR}_t{EDIT_T}__recon_comparison.png")