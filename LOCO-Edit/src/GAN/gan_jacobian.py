# gan_jacobian.py
import os
import sys
import torch
from torchvision.utils import make_grid, save_image

REPO_DIR = "/scratch/pbk5339/summer/LOCO-Edit/src/GAN/stylegan2-ada-pytorch"
sys.path.insert(0, REPO_DIR)
import legacy

# ---- config ----
DEVICE     = torch.device("cuda:0")
PKL        = "stylegan2-ada-pytorch/training-runs/00010-data_npy-auto1-kimg1000-ada-bg/network-snapshot-001000.pkl"
OUT_DIR    = "gan_jacobian_results"
IMG_SIZE   = 128
TRUNC_PSI  = 1.0      # no truncation for analysis -> study the full W-space geometry, not a narrowed neighborhood
K_FULL     = 5        # top-k directions, full 2-channel output
K_CHANNEL  = 5         # top-k directions restricted to a single output channel
N_ITER     = 5         # subspace/power iterations
EDIT_SCALE = 12.0       # sweep range for alpha in W-space. TUNE: watch the first traversal grids.
N_STEPS    = 5
SEED       = 0
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)

# ---- load generator ----
with open(PKL, 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE).eval()
G.requires_grad_(False)

H = W = IMG_SIZE
NUM_WS = G.num_ws
W_DIM  = G.w_dim
D_IN   = NUM_WS * W_DIM
D_OUT  = 2 * H * W

print(f"loaded G_ema: w_dim={W_DIM}, num_ws={NUM_WS}, img_channels={G.img_channels}, res={G.img_resolution}")

# ---- anchor sample: one fixed w to build the Jacobian around ----
z = torch.randn([1, G.z_dim], device=DEVICE)
c = torch.zeros([1, G.c_dim], device=DEVICE)
with torch.no_grad():
    ws_anchor  = G.mapping(z, c, truncation_psi=TRUNC_PSI)      # [1, NUM_WS, W_DIM]
    img_anchor = G.synthesis(ws_anchor, noise_mode='const')      # [1, 2, H, W]

save_image((img_anchor[0, :1] * 0.5 + 0.5).clamp(0, 1), f"{OUT_DIR}/anchor_chA.png")
save_image((img_anchor[0, 1:2] * 0.5 + 0.5).clamp(0, 1), f"{OUT_DIR}/anchor_chB.png")

ws_flat_anchor = ws_anchor.reshape(-1).detach()


def decode(ws_flat):
    '''the function being differentiated: flat w-vector -> flat image.
    Unlike diffusion's get_x0 (PMP/Tweedie's formula applied to a noise-prediction network),
    this is just the generator's own forward pass - no reformulation needed.'''
    ws = ws_flat.view(1, NUM_WS, W_DIM)
    img = G.synthesis(ws, noise_mode='const')
    return img.reshape(-1)


def build_linear_ops(w_flat):
    '''
    No torch.func here (stylegan2-ada-pytorch needs PyTorch 1.7.1, which predates functorch/
    torch.func). VJP is one ordinary backward pass. JVP uses the standard double-backward
    trick: g(u0) = J^T u0 is linear and differentiable in u0 (create_graph=True), and
    d/du0 <g(u0), v> = J v.
    '''
    w = w_flat.clone().requires_grad_(True)
    img = decode(w)   # graph rooted at w, reused by every jvp/vjp call below

    def vjp(u):
        return torch.autograd.grad(img, w, grad_outputs=u, retain_graph=True)[0].detach()

    def jvp(v):
        u0 = torch.zeros_like(img, requires_grad=True)
        g = torch.autograd.grad(img, w, grad_outputs=u0, create_graph=True, retain_graph=True)[0]
        scalar = (g * v).sum()
        return torch.autograd.grad(scalar, u0, retain_graph=True)[0].detach()

    return jvp, vjp


# ---- randomized top-k SVD of a linear operator given as flat jvp/vjp (identical to diffusion_jacobian.py) ----
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


def make_channel_ops(jvp, vjp, out_ch):
    '''
    Restrict the OUTPUT side to one channel. Unlike diffusion's block Jacobian
    (input AND output were both 2-channel images there), here the input is w - it has no
    channel structure to split, so only the output side gets restricted.
    '''
    mask = torch.zeros(2, H, W, device=DEVICE)
    mask[out_ch] = 1.0
    mask_flat = mask.reshape(-1).bool()

    def jvp_ch(v):
        return jvp(v)[mask_flat]

    def vjp_ch(u_ch):
        u_full = torch.zeros(D_OUT, device=DEVICE)
        u_full[mask_flat] = u_ch
        return vjp(u_full)

    return jvp_ch, vjp_ch, D_IN, H * W


@torch.no_grad()
def save_edit_traversal(ws_flat, v_flat, tag, edit_scale=EDIT_SCALE, n_steps=N_STEPS):
    values = torch.linspace(-edit_scale, edit_scale, n_steps, device=DEVICE)
    ws_batch = (ws_flat.unsqueeze(0) + values.unsqueeze(1) * v_flat.unsqueeze(0)).view(n_steps, NUM_WS, W_DIM)
    img_batch = G.synthesis(ws_batch, noise_mode='const')
    imgs = (img_batch * 0.5 + 0.5).clamp(0, 1)
    for ch, name in [(0, "chA"), (1, "chB")]:
        grid = make_grid(imgs[:, ch:ch + 1], nrow=n_steps)
        save_image(grid, f"{OUT_DIR}/{tag}_{name}.png")


# ---- main ----
jvp, vjp = build_linear_ops(ws_flat_anchor)

# full Jacobian: w (D_IN) -> both image channels (D_OUT)
U, S, Vd = jacobian_svd(jvp, vjp, D_IN, D_OUT, K_FULL, N_ITER)

print("\ntop directions from the full Jacobian (w -> 2-channel image)")
print("dir |  sigma   | A-energy | B-energy | verdict")
U_img = U.T.view(K_FULL, 2, H, W)
for i in range(K_FULL):
    eA = U_img[i, 0].pow(2).sum().item()
    eB = U_img[i, 1].pow(2).sum().item()
    fracA = eA / (eA + eB + 1e-8)
    verdict = "A-specific" if fracA > 0.8 else "B-specific" if fracA < 0.2 else "SHARED/coupled"
    print(f" {i:2d} | {S[i].item():8.3f} |  {fracA:5.2f}   |  {1 - fracA:5.2f}   | {verdict}")
    save_edit_traversal(ws_flat_anchor, Vd[:, i], f"full_dir{i}")

print("\ntop directions restricted to a single output channel")
for out_ch, name in [(0, "outA"), (1, "outB")]:
    jvp_ch, vjp_ch, di, do = make_channel_ops(jvp, vjp, out_ch)
    Uc, Sc, Vc = jacobian_svd(jvp_ch, vjp_ch, di, do, K_CHANNEL, N_ITER)
    print(f"{name}: " + " | ".join(f"s{j}={Sc[j].item():.3f}" for j in range(K_CHANNEL)))
    for i in range(K_CHANNEL):
        save_edit_traversal(ws_flat_anchor, Vc[:, i], f"{name}_dir{i}")

print(f"\nsaved traversal grids to {OUT_DIR}/")