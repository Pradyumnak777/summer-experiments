import os
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from diffusers import DDPMScheduler
from data_utils import twoChannelDataset
from torch.utils.data import DataLoader
from diffusion_model import build_unet
from diffusers.models.attention_processor import Attention, AttnProcessor
from tqdm import tqdm

DEVICE   = torch.device("cuda:8")
CKPT     = "diffusion_checkpoints/ddpm_2ch_128_masked/unet_ema_epoch80.pt"
CHA_DIR  = "data/singlecell_chA_split/train"
CHB_DIR  = "data/singlecell_chB_split/train"
IMG_SIZE = 128
EDIT_T   = 500
K_FULL   = 10
N_ITERS  = 3           # 3 fixed iterations for randomized SVD
BATCH_SIZE = 32         # Set to 4, 8, or 16 depending on available VRAM
OUT_DIR  = f"diffusion_train-set_dirs/ddpm_2ch_128_masked/{EDIT_T}_epoch80_ddim_inversion"
DENOISE_STEPS = 100

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(0)

# ---- model + scheduler ----
model = build_unet(IMG_SIZE, channels=2).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))

for module in model.modules():
    if isinstance(module, Attention):
        module.set_processor(AttnProcessor())

model.eval()
model.requires_grad_(False)

scheduler = DDPMScheduler(num_train_timesteps=1000)
alphas_cumprod = scheduler.alphas_cumprod.to(DEVICE)

H = W = IMG_SIZE
D = 2 * H * W

# 1. Single-element PMP estimator for torch.func transformations
def get_x0_single(x_single, t_val):
    # x_single shape: [2, H, W]
    x_in = x_single.unsqueeze(0)
    t = torch.tensor([t_val], device=x_single.device, dtype=torch.long)
    eps = model(x_in, t).sample.squeeze(0)
    ab = alphas_cumprod[t_val]
    return (x_single - (1.0 - ab).sqrt() * eps) / ab.sqrt()

# 2. Build vectorized linear operators across batch dimension
def build_batched_linear_ops(x_t_batch, t_val):
    # x_t_batch shape: [B, 2, H, W]
    f = lambda x: get_x0_single(x, t_val)

    # 1. Batched JVP: Evaluate directional derivative along v_in
    def jvp_flat(V_batch):
        # V_batch shape: [B, D] -> view as [B, 2, H, W]
        v_in = V_batch.view(-1, 2, H, W)
        _, out = torch.func.vmap(
            torch.func.jvp, in_dims=(None, (0,), (0,))
        )(f, (x_t_batch,), (v_in,))
        return out.detach().reshape(-1, D)

    # Helper function that performs both VJP computation and pull-back evaluation
    # Returning a tensor directly keeps vmap happy
    def single_vjp_eval(x, u):
        _, vjp_fn = torch.func.vjp(f, x)
        return vjp_fn(u)[0]

    # 2. Batched VJP: Map the self-contained helper across batch dimension 0
    def vjp_flat(U_batch):
        # U_batch shape: [B, D] -> view as [B, 2, H, W]
        u_in = U_batch.view(-1, 2, H, W)
        out = torch.func.vmap(single_vjp_eval, in_dims=(0, 0))(x_t_batch, u_in)
        return out.detach().reshape(-1, D)

    return jvp_flat, vjp_flat

# 3. Batched randomized SVD (operates on [B, D, K] in parallel)
def batched_jacobian_svd(jvp_flat, vjp_flat, B, d_dim, k, n_iter=N_ITERS):
    # Initial orthonormal subspace: [B, D, K]
    V = torch.linalg.qr(torch.randn(B, d_dim, k, device=DEVICE))[0]

    for _ in range(n_iter):
        # Forward pass: apply JVP column-by-column across the batch
        Y = torch.stack([jvp_flat(V[:, :, j]) for j in range(k)], dim=-1)   # [B, D, K]
        Y = torch.linalg.qr(Y)[0]

        # Adjoint pass: apply VJP column-by-column across the batch
        Z = torch.stack([vjp_flat(Y[:, :, j]) for j in range(k)], dim=-1)   # [B, D, K]
        V = torch.linalg.qr(Z)[0]

    # Final projection to recover singular values and vectors
    Y = torch.stack([jvp_flat(V[:, :, j]) for j in range(k)], dim=-1)
    Q = torch.linalg.qr(Y)[0]                                              # [B, D, K]
    Bt = torch.stack([vjp_flat(Q[:, :, j]) for j in range(k)], dim=-1)    # [B, D, K]

    # Transpose last two dimensions for batched thin SVD: [B, K, D]
    Ub, S, Vh = torch.linalg.svd(Bt.transpose(-2, -1), full_matrices=False)
    U = torch.bmm(Q, Ub)                                                   # [B, D, K]
    Vd = Vh.transpose(-2, -1)                                              # [B, D, K]
    return U, S, Vd

# Deterministic DDIM functions (unchanged)
def ddim_step(x_t, t_int, t_prev_int):
    t = torch.full((x_t.shape[0],), t_int, device=x_t.device, dtype=torch.long)
    eps = model(x_t, t).sample
    ab_t = alphas_cumprod[t_int]
    ab_prev = alphas_cumprod[t_prev_int] if t_prev_int > 0 else torch.tensor(1.0, device=x_t.device)
    x0_pred = (x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    return ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps

@torch.no_grad()
def ddim_inversion_step(x_prev, t_prev_int, t_int):
    t_prev = torch.full((x_prev.shape[0],), t_prev_int, device=x_prev.device, dtype=torch.long)
    eps = model(x_prev, t_prev).sample
    ab_prev = alphas_cumprod[t_prev_int] if t_prev_int > 0 else torch.tensor(1.0, device=x_prev.device)
    ab_t = alphas_cumprod[t_int]
    x0_pred = (x_prev - (1 - ab_prev).sqrt() * eps) / ab_prev.sqrt()
    return ab_t.sqrt() * x0_pred + (1 - ab_t).sqrt() * eps

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
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir="data/singlecell_mask")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)

    save_dir = f"{OUT_DIR}/data_files"
    os.makedirs(save_dir, exist_ok=True)

    element_idx = 0
    for x0_real, _ in tqdm(loader, desc="Batched Inversion & SVD"):
        x0_real = x0_real.to(DEVICE)
        B_curr = x0_real.shape[0]

        # 1. Batched DDIM inversion
        x_t = ddim_inversion(x0_real, EDIT_T)

        # 2. Build vectorized operators for current batch
        jvp_flat, vjp_flat = build_batched_linear_ops(x_t, EDIT_T)

        # 3. Parallel randomized SVD across the batch
        U, S, Vd = batched_jacobian_svd(jvp_flat, vjp_flat, B_curr, D, K_FULL, n_iter=N_ITERS)

        # 4. Save results per individual element
        for b in range(B_curr):
            torch.save(
                (x_t[b].detach().cpu(), EDIT_T),
                f"{save_dir}/x_t_{element_idx:06d}.pt"
            )
            torch.save(
                Vd[b].detach().cpu(),
                f"{save_dir}/Vd_{element_idx:06d}.pt"
            )
            element_idx += 1