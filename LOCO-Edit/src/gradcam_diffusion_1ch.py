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
K_FULL   = 5          # top-k directions from Vd to sweep through
K_BLOCK  = 5          # top-k for each channel block
N_ITER   = 5          # subspace iterations
EPS      = 1e-8


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


'''
one DIRECTED experiment: perturb the src INPUT channel only, watch the tgt OUTPUT channel.
because the src-channel slice is the ONLY thing we add (other input channel stays 0),
the tgt-output response is purely the off-diagonal Jacobian term J_{tgt<-src}. that's the
clean cross-channel dependence, no within-channel confound.

  src_ch=1, tgt_ch=0  ->  "A <- B"  (does chA output depend on chB input?)
  src_ch=0, tgt_ch=1  ->  "B <- A"

returns BOTH channels' edited images now - src_edit is the manipulation check (should
visibly move, that's the direct effect), tgt_edit is the actual cross effect we care about.
'''
def directed_gradcam(x_t, direction, lam, activation_layer, edit_t, src_ch, tgt_ch, x0_base):
    src_slice = direction[:, src_ch:src_ch+1]                 # [1, 1, H, W]
    norm = src_slice.flatten().norm()
    if norm < EPS:
        #this direction has basically no energy in the src channel -> nothing to push
        return None

    delta = torch.zeros_like(x_t)
    delta[:, src_ch:src_ch+1] = (src_slice / norm) * lam

    h1 = activation_layer.register_forward_hook(act_fwd_hook)
    h2 = activation_layer.register_full_backward_hook(act_bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x_t_edit  = (x_t + delta).detach().requires_grad_(True)
            x0_edit   = get_x0(x_t_edit, edit_t)
            feat      = feat_cache['x']

            tgt_delta = x0_edit[:, tgt_ch] - x0_base[:, tgt_ch]   #the CROSS response
            scalar    = tgt_delta.pow(2).sum()
            scalar.backward()
            grad = grad_cache['x'].detach().clone()

        cam = _make_cam(feat.detach(), grad)

        #both channels' actual edited images - src is the manipulation check, tgt is the cross effect
        src_edit = (x0_edit[0, src_ch].detach() * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
        tgt_edit = (x0_edit[0, tgt_ch].detach() * 0.5 + 0.5).clamp(0, 1).cpu().numpy()

        cross_resp  = float(scalar.detach().cpu())
        direct_resp = float((x0_edit[0, src_ch] - x0_base[0, src_ch]).pow(2).sum().detach().cpu())

        return dict(cam=cam, src_edit=src_edit, tgt_edit=tgt_edit,
                    cross_resp=cross_resp, direct_resp=direct_resp)
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


'''
rows are the two DIRECTED dependencies, not the two channels:
  row 0 = A <- B : chB input perturbed, chA output response
  row 1 = B <- A : chA input perturbed, chB output response

cols read left to right as a story:
  0-1: SOURCE channel, base vs edited  - what you pushed, and confirmation it moved
       (direct_resp - should be large, this is the sanity check)
  2-3: TARGET channel, base vs edited  - untouched input, this is the cross effect
       (cross_resp - the number you actually care about)
  4  : GradCAM of the target's response, overlaid on the target's edited image
'''
def save_directed_figure(recon, res_AB, res_BA, k, out_dir):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    rows = [("A<-B", 0, res_AB), ("B<-A", 1, res_BA)]
    for row, (label, tgt_ch, res) in enumerate(rows):
        for col in range(5):
            axes[row, col].axis("off")
        if res is None:
            axes[row, 0].set_title(f"{label}: no src energy in dir {k}")
            continue
        src_ch = 1 - tgt_ch

        axes[row, 0].imshow(recon[src_ch], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{label}: ch{src_ch} (src) base")
        axes[row, 1].imshow(res['src_edit'], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"ch{src_ch} (src) edited\ndirect={res['direct_resp']:.1f}")

        axes[row, 2].imshow(recon[tgt_ch], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"ch{tgt_ch} (tgt) base")
        axes[row, 3].imshow(res['tgt_edit'], cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title(f"ch{tgt_ch} (tgt) edited\ncross={res['cross_resp']:.1f}")

        cam_colored = cm.jet(res['cam'])[..., :3]
        gray_edit   = np.stack([res['tgt_edit']] * 3, axis=-1)
        overlay     = (0.5 * cam_colored + 0.5 * gray_edit).clip(0, 1)
        axes[row, 4].imshow(overlay)
        axes[row, 4].set_title(f"ch{tgt_ch} GradCAM")

    fig.suptitle(f"direction {k}  (cols 0-1: what was pushed | cols 2-3: what responded | col 4: where)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"dir{k:02d}_directed.png"), dpi=120)
    plt.close(fig)

    for label, res in [("AB", res_AB), ("BA", res_BA)]:
        if res is None:
            continue
        plt.imsave(os.path.join(out_dir, f"dir{k:02d}_cam_{label}_raw.png"),
                   (res['cam'] * 255).astype(np.uint8), cmap="gray")


if __name__ == "__main__":
    '''
    load custom diffusion model for running
    '''
    RUN_DIR = f"diffusion_checkpoints/ddpm_2ch_128/jacobian_epoch80"
    x_t, EDIT_T = torch.load(f"{RUN_DIR}/data_files/x_t.pt", map_location=torch.device(device)) #loading from tuple (edited in diffusion_jacobian.py)
    Vd = torch.load(f"{RUN_DIR}/data_files/Vd.pt", map_location=torch.device(device))
    EDIT_T = int(EDIT_T)   # get_x0 indexes alphas_cumprod[EDIT_T], so keep it a plain int

    lam = 20
    activation_layer = model.up_blocks[1]

    OUT_DIR = f"{RUN_DIR}/gradcam_directed"
    os.makedirs(OUT_DIR, exist_ok=True)

    #base recon (no edit), shared backdrop
    recon = get_recon(x_t, EDIT_T)
    #base x0 tensor too - directed_gradcam needs it to form the response delta
    with torch.no_grad():
        x0_base = get_x0(x_t, EDIT_T)

    for k in range(K_FULL):
        v_k = Vd[:, k].view(1, 2, H, W)   #a single full-space singular direction

        #A <- B : perturb chB input (src=1), watch chA output (tgt=0)
        res_AB = directed_gradcam(x_t, v_k, lam, activation_layer, EDIT_T,
                                  src_ch=1, tgt_ch=0, x0_base=x0_base)
        #B <- A : perturb chA input (src=0), watch chB output (tgt=1)
        res_BA = directed_gradcam(x_t, v_k, lam, activation_layer, EDIT_T,
                                  src_ch=0, tgt_ch=1, x0_base=x0_base)

        s_AB = res_AB['cross_resp'] if res_AB else float("nan")
        s_BA = res_BA['cross_resp'] if res_BA else float("nan")
        r_AB = (res_AB['cross_resp'] / res_AB['direct_resp']) if res_AB else float("nan")
        r_BA = (res_BA['cross_resp'] / res_BA['direct_resp']) if res_BA else float("nan")
        print(f"dir{k:02d}: A<-B cross={s_AB:7.2f} (cross/direct={r_AB:.3f}) | "
              f"B<-A cross={s_BA:7.2f} (cross/direct={r_BA:.3f})")

        save_directed_figure(recon, res_AB, res_BA, k, OUT_DIR)

    print(f"done. figures in {OUT_DIR}/")