'''
similar to diffusion_jacobian.py, but modified also to edit only regions that are masked
'''

import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from diffusers import DDPMScheduler
import debugpy
from data_utils import twoChannelDataset
from diffusion_model import build_unet
from diffusers.models.attention_processor import Attention, AttnProcessor
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


DEVICE   = torch.device("cuda:9")
CKPT     = "diffusion_checkpoints/ddpm_2ch_128_masked/unet_ema_epoch80.pt"
CHA_DIR  = "data/singlecell_chA_new_split/validation"
CHB_DIR  = "data/singlecell_chB_new_split/validation"
IMG_SIZE = 128
EDIT_T   = 500        #at what timestep to perform PMP/tweedies formula
ANCHOR   = 15          #sum frame to anchor on
K_FULL   = 10          # top-k directions for the full SVD
K_BG     = 30          # top-k background/complement directions for nullspace projection
# K_BLOCK  = 5          # top-k for each channel block
# N_ITER   = 5        #similar to locoedt(?)
EDIT_SCALE = 7.0     # sweep range for alpha (unit directions need a large multiplier - c.f.
N_STEPS  = 7           # images per traversal strip (odd number -> exact center = alpha=0)
SEED     = 0           # reproducibility: fixes both x_t's noise and the random SVD init
MIN_ITER = 10
MAX_ITER = 20

OUT_DIR  = f"diffusion_checkpoints/ddpm_2ch_128_masked/{EDIT_T}/masked_jacobian_epoch80_ddim_inversion"


DENOISE_STEPS = 100
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)

if os.getenv("DEBUGPY", "0") == "1":
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach on 5678...")
    debugpy.wait_for_client()

# ---- model + scheduler ----
model = build_unet(IMG_SIZE, channels=2).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))

# UNet2DModel has no set_attn_processor(); reach into each attention module instead
for module in model.modules():
    if isinstance(module, Attention):
        module.set_processor(AttnProcessor())   # non-fused path, forward-AD compatible

model.eval()
model.requires_grad_(False)

scheduler = DDPMScheduler(num_train_timesteps=1000)
alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)

H = W = IMG_SIZE
SHAPE = (1, 2, H, W)


#the PMPM estimator
def get_x0(x_t, t_int):
    t = torch.tensor(t_int, device=x_t.device)
    eps = model(x_t, t).sample
    ab  = alphas_cumprod[t_int]
    return (x_t - (1 - ab).sqrt() * eps) / ab.sqrt()


#sm efficient way to calculate jacobian as full one will ahve a billion entires and will probs crash
def build_linear_ops(x_t, t_int):
    f = lambda x: get_x0(x, t_int) #this is the function being differentiated (PMP predictor)
    _, vjp_raw = torch.func.vjp(f, x_t)
    def jvp(v):
        _, out = torch.func.jvp(f, (x_t,), (v,))
        return out.detach()
    def vjp(u):
        return vjp_raw(u)[0].detach()
    return jvp, vjp


#randomized top-k SVD of a linear operator given as flat jvp/vjp ----
def jacobian_svd(jvp_flat, vjp_flat, d_in, d_out, k, min_iter=10, max_iter=100, tol=1e-3):
    V = torch.linalg.qr(torch.randn(d_in, k, device=DEVICE))[0]
    for i in range(max_iter):

        V_prev = V.detach().clone()

        Y = torch.stack([jvp_flat(V[:, j]) for j in range(k)], dim=1)
        Y = torch.linalg.qr(Y)[0]
        Z = torch.stack([vjp_flat(Y[:, j]) for j in range(k)], dim=1)
        V = torch.linalg.qr(Z)[0]

        conv = torch.dist(V_prev, V).item()
        print(f"jacobian_svd: iter {i} convergence dist = {conv:.4e}")
        if i >= min_iter and conv < tol:
            print(f"jacobian_svd: converged after {i+1} iterations (dist={conv:.2e})")
            break

    Y = torch.stack([jvp_flat(V[:, j]) for j in range(k)], dim=1)
    Q = torch.linalg.qr(Y)[0]
    Bt = torch.stack([vjp_flat(Q[:, j]) for j in range(k)], dim=1)
    Ub, S, Vh = torch.linalg.svd(Bt.T, full_matrices=False)
    U = Q @ Ub
    return U, S, Vh.T


def make_full_ops(jvp, vjp):
    D = 2 * H * W
    def jvp_flat(v):  return jvp(v.view(SHAPE)).reshape(D)
    def vjp_flat(u):  return vjp(u.view(SHAPE)).reshape(D)
    return jvp_flat, vjp_flat, D, D


def make_block_ops(jvp, vjp, in_ch, out_ch): #flattens only that specific channel's half!!
    D = H * W
    def jvp_flat(v_in):
        v = torch.zeros(SHAPE, device=DEVICE)
        v[0, in_ch] = v_in.view(H, W)
        return jvp(v)[0, out_ch].reshape(D)
    def vjp_flat(u_out):
        u = torch.zeros(SHAPE, device=DEVICE)
        u[0, out_ch] = u_out.view(H, W)
        return vjp(u)[0, in_ch].reshape(D)
    return jvp_flat, vjp_flat, D, D


def expand_block_v(v_block, in_ch):
    '''place a block-SVD direction (lives in one channel only) back into a real,
    usable 2-channel x_t perturbation - zero in the other channel, by construction.'''
    v_full = torch.zeros(SHAPE, device=DEVICE)
    v_full[0, in_ch] = v_block.view(H, W)
    return v_full


@torch.no_grad()
def save_direction_image(v_block, tag):
    '''visualize the raw input-space singular vector: the actual perturbation pattern
    injected into the input channel - i.e. *what changed in the input* before decoding.
    signed direction -> min-max normalized to [0,1] just for viewing.'''
    v = v_block.view(H, W)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    save_image(v.unsqueeze(0), f"{OUT_DIR}/{tag}_input_dir.png")


@torch.no_grad()
def save_edit_traversal(x_t, v_full, tag, edit_scale=EDIT_SCALE, n_steps=N_STEPS):
    values = torch.linspace(-edit_scale, edit_scale, n_steps, device=DEVICE)
    x_batch = x_t + values.view(n_steps, 1, 1, 1) * v_full        # [n_steps, 2, H, W]
    # x0_batch = get_x0(x_batch, EDIT_T)                             # [n_steps, 2, H, W]
    x0_batch = ddim_denoise(x_batch, EDIT_T)                       
    
    
    imgs = (x0_batch * 0.5 + 0.5).clamp(0, 1)
    for ch, name in [(0, "chA"), (1, "chB")]:
        grid = make_grid(imgs[:, ch:ch+1], nrow=n_steps)
        save_image(grid, f"{OUT_DIR}/{tag}_{name}.png")


@torch.no_grad()
def save_original_vs_recon(x0_real, x0_hat_center):
    orig  = (x0_real[0]        * 0.5 + 0.5).clamp(0, 1)   # [2, H, W]
    recon = (x0_hat_center[0]  * 0.5 + 0.5).clamp(0, 1)   # [2, H, W]

    for ch, name in [(0, "chA"), (1, "chB")]:
        pair = torch.stack([orig[ch], recon[ch]], dim=0).unsqueeze(1)  # [2, 1, H, W]
        grid = make_grid(pair, nrow=2)   # left = original, right = reconstruction
        save_image(grid, f"{OUT_DIR}/anchor_recon_{name}.png")

    mse_A = F.mse_loss(x0_hat_center[:, 0], x0_real[:, 0]).item()
    mse_B = F.mse_loss(x0_hat_center[:, 1], x0_real[:, 1]).item()
    print(f"anchor reconstruction check (t={EDIT_T}): MSE chA={mse_A:.5f}  chB={mse_B:.5f}")
    print(f"  saved -> {OUT_DIR}/anchor_recon_chA.png, anchor_recon_chB.png (left=original, right=recon)")


#one deterministic ddim reverse step, eta=0, from t_int down to t_prev_int
def ddim_step(x_t, t_int, t_prev_int):
    t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
    eps = model(x_t, t).sample
    ab_t = alphas_cumprod[t_int]
    #treat t_prev<=0 as fully clean, matches how tweedie's x0 estimate is defined
    ab_prev = alphas_cumprod[t_prev_int] if t_prev_int > 0 else torch.tensor(1.0, device=x_t.device)
    x0_pred = (x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    return ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps


#runs the full reverse trajectory from t_start down to 0, this is what actually gets "presented"
@torch.no_grad()
def ddim_denoise(x_t, t_start, denoise_steps=DENOISE_STEPS):
    timesteps = torch.linspace(t_start, 0, denoise_steps + 1).round().long().tolist()
    x = x_t
    for i in range(len(timesteps) - 1):
        t_cur, t_next = timesteps[i], timesteps[i + 1]
        if t_cur == t_next:
            continue
        x = ddim_step(x, t_cur, t_next)
    return x

#instead of noising like SDEedit
def ddim_inversion_step(x_prev, t_prev_int, t_int):
    t_prev = torch.full((x_prev.shape[0],), t_prev_int, device=x_prev.device, dtype=torch.long)
    eps = model(x_prev, t_prev).sample
    #treat t_prev<=0 as fully clean, matches how tweedie's x0 estimate is defined
    ab_prev = alphas_cumprod[t_prev_int] if t_prev_int > 0 else torch.tensor(1.0, device=x_prev.device)
    ab_t = alphas_cumprod[t_int]
    x0_pred = (x_prev - (1 - ab_prev).sqrt() * eps) / ab_prev.sqrt()
    return ab_t.sqrt() * x0_pred + (1 - ab_t).sqrt() * eps


#instead of noising like SDEedit
#runs the full deterministic forward trajectory from 0 up to t_end, this is what
#actually produces x_t for a real image (mirror of ddim_denoise, walked the other way)
@torch.no_grad()
def ddim_inversion(x0_real, t_end, inversion_steps=DENOISE_STEPS):
    timesteps = torch.linspace(0, t_end, inversion_steps + 1).round().long().tolist()
    x = x0_real
    for i in range(len(timesteps) - 1):
        t_cur, t_next = timesteps[i], timesteps[i + 1]
        if t_cur == t_next:
            continue
        x = ddim_inversion_step(x, t_cur, t_next)
    return x


if __name__ == "__main__":
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir="data/singlecell_mask_new_split/validation") #loading data
    x0_real, _ = dataset[ANCHOR] #actual img, dataloader mask discarded - using the hand-drawn mask below instead
    x0_real = x0_real.unsqueeze(0).to(DEVICE)

    #same convention as infer_cross_attn_v2.py: find the hand-drawn mask frame for this sample
    name = dataset.chA_files[ANCHOR].name
    subject, rest = name.split("_cell", 1)
    cell_id, rest = rest.split("_f", 1)
    frame = int(rest.split("_")[0])

    hand_mask_dir  = Path("data/annotations") / subject / "hand_drawn_mask"
    mask_movie_dir = hand_mask_dir / f"{cell_id}_handdrawnmask_movie"
    mask_path      = mask_movie_dir / f"{mask_movie_dir.name}_f{frame:03d}.png"
    assert mask_path.exists(), f"no hand-drawn mask found at {mask_path}"

    mask_np = np.array(Image.open(mask_path).convert("L"))
    mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0) / 255.0   # [1,1,h,w]
    mask = F.interpolate(mask, size=(H, W), mode="nearest").to(DEVICE)           # [1,1,H,W]
    save_image(mask[0], f"{OUT_DIR}/hand_drawn_mask.png") #so we can sanity check what got loaded
    
    '''
    below is like SDEedit
    '''
    # noise   = torch.randn_like(x0_real)
    # x_t = scheduler.add_noise(x0_real, noise, torch.tensor([EDIT_T], device=DEVICE)) #image noised to timestep t

    '''
    below is using ddim_inversion
    '''
    x_t = ddim_inversion(x0_real, EDIT_T)
    
    with torch.no_grad():
        '''
        below basically applies the PMP/tweedie's formula which predicts clean image from a noised image
        '''
        x0_hat_center = get_x0(x_t, EDIT_T)
        
    
    
    '''
        APPLY mask here!!

        1. get x0_hat_center_masked
        2. get x0_hat_center_masked-composite (opposite mask)
    '''
    mask_full      = mask.expand(1, 2, H, W)
    mask_flat      = (mask_full.reshape(-1) > 0.5).float()   # Omega - ROI (masked-in)
    comp_mask_flat = 1.0 - mask_flat                          # Omega^C - background/composite

    x0_hat_center_masked           = x0_hat_center * mask_full
    x0_hat_center_masked_composite = x0_hat_center * (1 - mask_full)

    '''
        now, perform 2 jacobians - d(x0_hat_center_masked)/d(x_t) and
        d(x0_hat_center_masked_composite)/d(x_t)

    '''
    def make_masked_ops(jvp, vjp, out_mask_flat):
        D = 2 * H * W
        def jvp_flat(v):  return jvp(v.view(SHAPE)).reshape(D) * out_mask_flat
        def vjp_flat(u):  return vjp((u * out_mask_flat).view(SHAPE)).reshape(D)
        return jvp_flat, vjp_flat, D, D

    save_original_vs_recon(x0_real, x0_hat_center) #saving original vs PMP predicted on image

    jvp, vjp = build_linear_ops(x_t, EDIT_T)

    
    #below are the 2 jacobians
    jf_roi, vf_roi, din, dout = make_masked_ops(jvp, vjp, mask_flat)
    jf_bg,  vf_bg,  _, _      = make_masked_ops(jvp, vjp, comp_mask_flat)

    #ROI jacobian: candidate edit directions. background jacobian: what NOT to move
    U_roi, S_roi, Vd_roi = jacobian_svd(jf_roi, vf_roi, din, dout, K_FULL, MIN_ITER, MAX_ITER)
    U_bg,  S_bg,  Vbar   = jacobian_svd(jf_bg,  vf_bg,  din, dout, K_BG,   MIN_ITER, MAX_ITER)

    os.makedirs(f"{OUT_DIR}/data_files", exist_ok=True)
    torch.save((x_t.detach().cpu(), EDIT_T), f"{OUT_DIR}/data_files/x_t.pt") #this is the noise, to be used in gradCAM
    torch.save(Vd_roi.detach().cpu(), f"{OUT_DIR}/data_files/Vd_roi.pt") #

    print("\nbelow are the LOCAL (nullspace-projected) edit directions")
    print("dir |  sigma   | A-energy | B-energy | verdict")
    U_img = U_roi.T.view(K_FULL, 2, H, W)
    for i in range(K_FULL):
        eA = U_img[i, 0].pow(2).sum().item() #u is the edit direction o the output..?
        eB = U_img[i, 1].pow(2).sum().item()
        fracA = eA / (eA + eB + 1e-8)
        verdict = "A-specific" if fracA > 0.8 else "B-specific" if fracA < 0.2 else "SHARED/coupled"
        print(f" {i:2d} | {S_roi[i].item():8.3f} |  {fracA:5.2f}   |  {1-fracA:5.2f}   | {verdict}")

        v   = Vd_roi[:, i]
        v_p = v - Vbar @ (Vbar.T @ v)   #nullspace projection - strip the background-moving component
        v_p = v_p / v_p.norm().clamp_min(1e-8)
        v_i = v_p.view(1, 2, H, W)
        save_edit_traversal(x_t, v_i, f"local_dir{i}")

    # below is the OLD, non-masked jacobian code (full image, no locality guarantee)
    # jf, vf, din, dout = make_full_ops(jvp, vjp) #flattened jvp and vjp basically
    # U, S, Vd = jacobian_svd(jf, vf, din, dout, K_FULL, MIN_ITER, MAX_ITER) #U- output space(X_0) dirs. Vd- input space(X_t) dirs. S- singular values

    # os.makedirs(f"{OUT_DIR}/data_files", exist_ok=True)
    # torch.save((x_t.detach().cpu(), EDIT_T), f"{OUT_DIR}/data_files/x_t.pt") #this is the noise, to be used in gradCAM
    # torch.save(Vd.detach().cpu(), f"{OUT_DIR}/data_files/Vd.pt") #

    # print("\nbelow are the actual edit directions for the full 32,768 jacobian (2 channel)")
    # print("dir |  sigma   | A-energy | B-energy | verdict")
    # U_img = U.T.view(K_FULL, 2, H, W)
    # for i in range(K_FULL):
    #     eA = U_img[i, 0].pow(2).sum().item() #u is the edit direction o the output..?
    #     eB = U_img[i, 1].pow(2).sum().item()
    #     fracA = eA / (eA + eB + 1e-8)
    #     verdict = "A-specific" if fracA > 0.8 else "B-specific" if fracA < 0.2 else "SHARED/coupled"
    #     print(f" {i:2d} | {S[i].item():8.3f} |  {fracA:5.2f}   |  {1-fracA:5.2f}   | {verdict}")
    #     v_i = Vd[:, i].view(1, 2, H, W) #the input directions, this is multiplied with the anchors?
    #     save_edit_traversal(x_t, v_i, f"full_dir{i}")


        '''
        #NOTE: below for channelwise!
        '''

    # print("\nbelow are J_AB, J_BA, etc")
    # print("block | " + " | ".join(f"s{j}" for j in range(K_BLOCK)))
    # blocks = {"J_AA": (0, 0), "J_BB": (1, 1), "J_BA (A->B)": (0, 1), "J_AB (B->A)": (1, 0)}
    # cross_blocks = {"J_BA (A->B)", "J_AB (B->A)"}   # off-diagonal only -> where the real coupling lives

    # spectra = {}   # collect all blocks' singular values for plotting
    # for name, (in_ch, out_ch) in blocks.items():
    #     jb, vb, di, do = make_block_ops(jvp, vjp, in_ch, out_ch)
    #     Ub, Sb, Vb = jacobian_svd(jb, vb, di, do, K_BLOCK, MIN_ITER, MAX_ITER)
    #     print(f"{name:11s} | " + " | ".join(f"{s:6.3f}" for s in Sb.tolist()))
    #     spectra[name] = Sb.tolist()

    #     if name in cross_blocks:
    #         safe_name = name.split(" ")[0]   # "J_BA", "J_AB" - strip the "(A->B)" for filenames
    #         for i in range(K_BLOCK):
    #             save_direction_image(Vb[:, i], f"{safe_name}_dir{i}")   # the input perturbation pattern (what changed)
    #             v_full = expand_block_v(Vb[:, i], in_ch) #NOTE: pads the vector with zeroes for the other channel..!
    #             save_edit_traversal(x_t, v_full, f"{safe_name}_dir{i}")

    # plt.figure(figsize=(6, 4))
    # for name, s_vals in spectra.items():
    #     plt.plot(range(len(s_vals)), s_vals, marker="o", label=name)
    # plt.yscale("log")                      # cross-block values likely much smaller -> log scale shows both clearly
    # plt.xlabel("rank (s0, s1, ...)")
    # plt.ylabel("singular value")
    # plt.title(f"Block Jacobian singular value spectra (t={EDIT_T})")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"{OUT_DIR}/singular_value_spectra.png")
    # plt.close()

    print(f"\nsaved traversal grids to {OUT_DIR}/")