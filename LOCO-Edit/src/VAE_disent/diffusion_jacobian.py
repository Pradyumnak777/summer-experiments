'''
unsupervised disentanglement directions for the 2-channel DDPM, LOCO-Edit style.
the PMP (posterior mean predictor) is get_x0: noisy x_t -> clean x0_hat.
we SVD its Jacobian J = d x0_hat / d x_t.

with 2 channels, J has a block structure:
    J = [[J_AA, J_AB],
         [J_BA, J_BB]],   J_XY = d(x0 chan X)/d(x_t chan Y)
- full SVD          -> semantic edit directions (traversal: sweep alpha, decode, center = real recon)
- off-diagonal SVD  -> "how does moving one channel affect the other" (coupling), same traversal format
'''
import os
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from diffusers import DDPMScheduler
from data_utils import twoChannelDataset
from diffusion_model import build_unet
from diffusers.models.attention_processor import Attention, AttnProcessor


DEVICE   = torch.device("cuda:9")
CKPT     = "diffusion_checkpoints/ddpm_2ch_128/unet_ema_epoch50.pt"
CHA_DIR  = "data/singlecell_chA_split/train"
CHB_DIR  = "data/singlecell_chB_split/train"
OUT_DIR  = "diffusion_checkpoints/ddpm_2ch_128/jacobian_epoch50"
IMG_SIZE = 128
EDIT_T   = 300        # timestep to linearize at. larger = coarser/semantic, smaller = fine detail
ANCHOR   = 1          # which dataset image to anchor on
K_FULL   = 5          # top-k directions for the full SVD
K_BLOCK  = 5          # top-k for each channel block
N_ITER   = 5          # subspace iterations (your power-method loop count)
EDIT_SCALE = 15.0     # sweep range for alpha (unit directions need a large multiplier - c.f. your
                       # own repo's get_delta_zt_via_grad, which uses 20.0. TUNE THIS: too small ->
                       # no visible change; too large -> image degrades into garbage.
N_STEPS  = 7           # images per traversal strip (odd number -> exact center = alpha=0)
os.makedirs(OUT_DIR, exist_ok=True)

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


# ---- the PMP: noisy x_t -> clean x0 estimate (DDPM closed form) ----
def get_x0(x_t, t_int):
    t = torch.tensor(t_int, device=x_t.device)
    eps = model(x_t, t).sample
    ab  = alphas_cumprod[t_int]
    return (x_t - (1 - ab).sqrt() * eps) / ab.sqrt()


# ---- linearize the PMP at a fixed x_t: build reusable JVP / VJP operators ----
def build_linear_ops(x_t, t_int):
    f = lambda x: get_x0(x, t_int)
    _, vjp_raw = torch.func.vjp(f, x_t)
    def jvp(v):
        _, out = torch.func.jvp(f, (x_t,), (v,))
        return out.detach()
    def vjp(u):
        return vjp_raw(u)[0].detach()
    return jvp, vjp


# ---- randomized top-k SVD of a linear operator given as flat jvp/vjp ----
def jacobian_svd(jvp_flat, vjp_flat, d_in, d_out, k, n_iter):
    V = torch.linalg.qr(torch.randn(d_in, k, device=DEVICE))[0]
    for _ in range(n_iter):
        Y = torch.stack([jvp_flat(V[:, i]) for i in range(k)], dim=1)
        Y = torch.linalg.qr(Y)[0]
        Z = torch.stack([vjp_flat(Y[:, i]) for i in range(k)], dim=1)
        V = torch.linalg.qr(Z)[0]
    Y = torch.stack([jvp_flat(V[:, i]) for i in range(k)], dim=1)
    Q = torch.linalg.qr(Y)[0]
    Bt = torch.stack([vjp_flat(Q[:, i]) for i in range(k)], dim=1)
    Ub, S, Vh = torch.linalg.svd(Bt.T, full_matrices=False)
    U = Q @ Ub
    return U, S, Vh.T


def make_full_ops(jvp, vjp):
    D = 2 * H * W
    def jvp_flat(v):  return jvp(v.view(SHAPE)).reshape(D)
    def vjp_flat(u):  return vjp(u.view(SHAPE)).reshape(D)
    return jvp_flat, vjp_flat, D, D


def make_block_ops(jvp, vjp, in_ch, out_ch):
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
def save_edit_traversal(x_t, v_full, tag, edit_scale=EDIT_SCALE, n_steps=N_STEPS):
    '''
    sweep alpha in [-edit_scale, edit_scale], apply x_t + alpha*v_full, decode each
    with the one-step PMP (get_x0) - same "preview" approach your own LOCO-Edit repo
    uses in get_delta_zt_via_grad, so no expensive full re-sampling needed.
    CENTER column (alpha=0) is the actual current reconstruction, unedited.
    '''
    values = torch.linspace(-edit_scale, edit_scale, n_steps, device=DEVICE)
    x_batch = x_t + values.view(n_steps, 1, 1, 1) * v_full        # [n_steps, 2, H, W]
    x0_batch = get_x0(x_batch, EDIT_T)                             # [n_steps, 2, H, W]

    imgs = (x0_batch * 0.5 + 0.5).clamp(0, 1)
    for ch, name in [(0, "chA"), (1, "chB")]:
        grid = make_grid(imgs[:, ch:ch+1], nrow=n_steps)
        save_image(grid, f"{OUT_DIR}/{tag}_{name}.png")


@torch.no_grad()
def save_original_vs_recon(x0_real, x0_hat_center):
    '''
    sanity check BEFORE trusting any traversal: does the unedited (alpha=0)
    reconstruction at t=EDIT_T actually resemble the real input image?
    top row = original, bottom row = reconstruction, per channel.
    x0_hat_center is exactly the same value as the center column (alpha=0)
    of every traversal strip above - this just pairs it directly against
    the ground truth for a clean side-by-side look.
    '''
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


# ============================ run ============================
dataset = twoChannelDataset(CHA_DIR, CHB_DIR)
x0_real = dataset[ANCHOR].unsqueeze(0).to(DEVICE)
noise   = torch.randn_like(x0_real)
x_t = scheduler.add_noise(x0_real, noise, torch.tensor([EDIT_T], device=DEVICE))

# ---- 0) sanity check: does the unedited reconstruction match the real image? ----
with torch.no_grad():
    x0_hat_center = get_x0(x_t, EDIT_T)
save_original_vs_recon(x0_real, x0_hat_center)

jvp, vjp = build_linear_ops(x_t, EDIT_T)

# ---- 1) FULL jacobian SVD: semantic directions, saved as real edit traversals ----
jf, vf, din, dout = make_full_ops(jvp, vjp)
U, S, Vd = jacobian_svd(jf, vf, din, dout, K_FULL, N_ITER)

print("\n=== FULL Jacobian SVD (semantic edit directions) ===")
print("dir |  sigma   | A-energy | B-energy | verdict")
U_img = U.T.view(K_FULL, 2, H, W)
for i in range(K_FULL):
    eA = U_img[i, 0].pow(2).sum().item()
    eB = U_img[i, 1].pow(2).sum().item()
    fracA = eA / (eA + eB + 1e-8)
    verdict = "A-specific" if fracA > 0.8 else "B-specific" if fracA < 0.2 else "SHARED/coupled"
    print(f" {i:2d} | {S[i].item():8.3f} |  {fracA:5.2f}   |  {1-fracA:5.2f}   | {verdict}")
    v_i = Vd[:, i].view(1, 2, H, W)
    save_edit_traversal(x_t, v_i, f"full_dir{i}")

# ---- 2) CHANNEL-BLOCK SVD: coupling strength + real traversal for the CROSS blocks ----
print("\n=== per-block top singular values (coupling structure) ===")
print("block | " + " | ".join(f"s{j}" for j in range(K_BLOCK)))
blocks = {"J_AA": (0, 0), "J_BB": (1, 1), "J_BA (A->B)": (0, 1), "J_AB (B->A)": (1, 0)}
cross_blocks = {"J_BA (A->B)", "J_AB (B->A)"}   # off-diagonal only -> where the real coupling lives

for name, (in_ch, out_ch) in blocks.items():
    jb, vb, di, do = make_block_ops(jvp, vjp, in_ch, out_ch)
    Ub, Sb, Vb = jacobian_svd(jb, vb, di, do, K_BLOCK, N_ITER)
    print(f"{name:11s} | " + " | ".join(f"{s:6.3f}" for s in Sb.tolist()))

    if name in cross_blocks:
        safe_name = name.split(" ")[0]   # "J_BA", "J_AB" - strip the "(A->B)" for filenames
        for i in range(K_BLOCK):
            v_full = expand_block_v(Vb[:, i], in_ch)
            save_edit_traversal(x_t, v_full, f"{safe_name}_dir{i}")

print(f"\nsaved traversal grids to {OUT_DIR}/")