import os
import debugpy
import torch
from VAE_disent.diffusion_model import build_unet
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers import DDPMScheduler
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

if os.getenv("DEBUGPY", "0") == "1":
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach on 5678...")
    debugpy.wait_for_client()

device = torch.device("cuda:9")

feat_cache = {}
grad_cache = {}

#config stuff
CKPT     = "diffusion_checkpoints/ddpm_2ch_128_masked/unet_ema_epoch80.pt"
IMG_SIZE = 128
K_FULL   = 5          # top-k directions for the full SVD
K_BLOCK  = 5          # top-k for each channel block
N_ITER   = 5          # subspace iterations


#model specifications
model = build_unet(IMG_SIZE, channels=2).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))

for module in model.modules():
    if isinstance(module, Attention):
        module.set_processor(AttnProcessor())

model.eval()
model.requires_grad_(False)

scheduler = DDPMScheduler(num_train_timesteps=1000)
alphas_cumprod = scheduler.alphas_cumprod.to(device)

H = W = IMG_SIZE
SHAPE = (1, 2, H, W)


'''
functions associated w/ model-
'''

#the PMP estimator
def get_x0(x_t, t_int):
    t = torch.tensor(t_int, device=x_t.device)
    eps = model(x_t, t).sample
    ab  = alphas_cumprod[t_int]
    return (x_t - (1 - ab).sqrt() * eps) / ab.sqrt()


def act_fwd_hook(module, inputs, output):
    feat_cache['x'] = output[0] if isinstance(output, tuple) else output

def act_bwd_hook(module, grad_input, grad_output):
    grad_cache['x'] = grad_output[0]


def _make_cam(feat, grad):
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feat).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam.float(), size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    cam = cam[0, 0].cpu().numpy()
    if cam.max() > 1e-8:
        cam = cam / cam.max()
    return cam


def gradcam_1(x_t, v_k, lam, activation_layer, edit_t):
    h1 = activation_layer.register_forward_hook(act_fwd_hook)
    h2 = activation_layer.register_full_backward_hook(act_bwd_hook)
    try:
        with torch.no_grad():
            x0_base = get_x0(x_t, edit_t)

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x_t_edit = (x_t + lam * v_k).detach().requires_grad_(True)
            x0_edit = get_x0(x_t_edit, edit_t) #tweedie's formula at this timestep
            feat = feat_cache['x']

            
            '''
            both below scalars are ESTIMATES from a noisy timestep, and NOT the final edit..
            '''
            scalar_A = (x0_edit[:, 0] - x0_base[:, 0]).pow(2).sum()
            scalar_B = (x0_edit[:, 1] - x0_base[:, 1]).pow(2).sum()

            scalar_A.backward(retain_graph=True)
            grad_A = grad_cache['x'].detach().clone()

            scalar_B.backward()
            grad_B = grad_cache['x'].detach().clone()

        feat = feat.detach()
        cam_A = _make_cam(feat, grad_A)
        cam_B = _make_cam(feat, grad_B)

        # the actual edited reconstruction (x0 of the perturbed z_t), same display convention as get_recon
        edit_recon = (x0_edit[0].detach() * 0.5 + 0.5).clamp(0, 1).cpu().numpy()   # [2, H, W] in [0, 1]

        return cam_A, cam_B, edit_recon, float(scalar_A.detach().cpu()), float(scalar_B.detach().cpu())

    finally:
        h1.remove()
        h2.remove()


# recon = the base Tweedie x0 (no edit). same for every direction, so compute it once
# and reuse it as the grayscale backdrop for every overlay.
def get_recon(x_t, edit_t):
    with torch.no_grad():
        x0 = get_x0(x_t, edit_t)                 # [1, 2, H, W], data lives in [-1, 1]
    recon = (x0[0] * 0.5 + 0.5).clamp(0, 1)      # match diffusion_jacobian.py display convention
    return recon.cpu().numpy()                   # [2, H, W] in [0, 1]


# cols now: base reconstruction | edited reconstruction (the actual moved image) | GradCAM overlay
def save_gradcam_figure(recon, edit_recon, cam_A, cam_B, score_A, score_B, k, out_dir, alpha=0.5):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row, (ch, name, cam) in enumerate([(0, "chA", cam_A), (1, "chB", cam_B)]):
        cam_colored = cm.jet(cam)[..., :3]
        gray = recon[ch]
        gray_edit = edit_recon[ch]
        gray_rgb = np.stack([gray] * 3, axis=-1)  # overlay sits on the BASE img now
        overlay = (alpha * cam_colored + (1 - alpha) * gray_rgb).clip(0, 1)

        axes[row, 0].imshow(gray, cmap="gray")
        axes[row, 0].set_title(f"{name} reconstruction")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gray_edit, cmap="gray")
        axes[row, 1].set_title(f"{name} edited (x0 of z_t + lam*v_k)")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"{name}")
        axes[row, 2].axis("off")

    fig.suptitle(f"direction {k}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"dir{k:02d}_gradcam.png"), dpi=120)
    plt.close(fig)

    plt.imsave(os.path.join(out_dir, f"dir{k:02d}_camA_raw.png"), (cam_A * 255).astype(np.uint8), cmap="gray")
    plt.imsave(os.path.join(out_dir, f"dir{k:02d}_camB_raw.png"), (cam_B * 255).astype(np.uint8), cmap="gray")

if __name__ == "__main__":
    '''
    load custom diffusion model for running
    '''
    RUN_DIR = f"diffusion_checkpoints/ddpm_2ch_128_masked/500/jacobian_epoch80"
    x_t, EDIT_T = torch.load(f"{RUN_DIR}/data_files/x_t.pt", map_location=torch.device(device)) #loading from tuple (dited in diffusion_jacobian.py)
    Vd = torch.load(f"{RUN_DIR}/data_files/Vd.pt", map_location=torch.device(device))
    EDIT_T = int(EDIT_T)   # get_x0 indexes alphas_cumprod[EDIT_T], so keep it a plain int

    lam = 7.0
    activation_layer = model.up_blocks[1]

    OUT_DIR = f"{RUN_DIR}/gradcam"
    os.makedirs(OUT_DIR, exist_ok=True)

    # grayscale backdrop shared by every direction's overlay
    recon = get_recon(x_t, EDIT_T)

    for k in range(K_FULL):
        v_k = Vd[:, k].view(1, 2, H, W) #a single direction
        cam_A, cam_B, edit_recon, score_A, score_B = gradcam_1(x_t, v_k, lam, activation_layer, EDIT_T)
        print(f"dir{k:02d}: score_A={score_A:.4f}, score_B={score_B:.4f}")
        save_gradcam_figure(recon, edit_recon, cam_A, cam_B, score_A, score_B, k, OUT_DIR)


    print(f"done. figures in {OUT_DIR}/")
