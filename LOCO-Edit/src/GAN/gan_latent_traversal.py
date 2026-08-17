import os
import sys
import torch
from torchvision.utils import make_grid, save_image

REPO_DIR = "/scratch/pbk5339/summer/LOCO-Edit/src/GAN/stylegan2-ada-pytorch"
sys.path.insert(0, REPO_DIR)
import legacy

DEVICE     = torch.device("cuda:9")
PKL        = "stylegan2-ada-pytorch/training-runs/00013-data_npy_masked-auto1-kimg1000-ada-bg/network-snapshot-001000.pkl"
OUT_DIR    = "gan_latent_traversal_masked"
TRUNC_PSI  = 1.0       # no truncation -> study the raw learned W-space
N_DIMS     = 5         # traverse the first 5 raw W dimensions
EDIT_SCALE = 7.0       
N_STEPS    = 5         # odd -> exact center column is alpha=0 (the untouched image)
SEED       = 0
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)

# ---- load generator ----
with open(PKL, 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE).eval()
G.requires_grad_(False)

NUM_WS = G.num_ws
W_DIM  = G.w_dim
print(f"loaded G_ema: w_dim={W_DIM}, num_ws={NUM_WS}, img_channels={G.img_channels}, res={G.img_resolution}")

'''
#NOTE: see below that a random generation is being traversed. in jacobian-diffusion previously, worked on
an already 
'''
z = torch.randn([1, G.z_dim], device=DEVICE)
c = torch.zeros([1, G.c_dim], device=DEVICE)
with torch.no_grad():
    ws_anchor = G.mapping(z, c, truncation_psi=TRUNC_PSI)   # [1, NUM_WS, W_DIM]

# the multipliers for the columns: symmetric around 0, so center = untouched image
alphas = torch.linspace(-EDIT_SCALE, EDIT_SCALE, N_STEPS, device=DEVICE)   # e.g. [-8,...,0,...,+8]


@torch.no_grad()
def traverse_dim(dim):
    '''walk W-dimension `dim` from -EDIT_SCALE to +EDIT_SCALE, same w added to every layer'''
    ws_batch = ws_anchor.repeat(N_STEPS, 1, 1).clone()   # [N_STEPS, NUM_WS, W_DIM]
    # add alpha to this one coordinate, across ALL num_ws style layers
    ws_batch[:, :, dim] += alphas.view(N_STEPS, 1)
    img = G.synthesis(ws_batch, noise_mode='const')      # [N_STEPS, 2, H, W]
    return (img * 0.5 + 0.5).clamp(0, 1)


# per-dim strips (one row = one dim's traversal), plus a combined grid
rows_A, rows_B = [], []
for dim in range(N_DIMS):
    imgs = traverse_dim(dim)                              # [N_STEPS, 2, H, W]
    # each channel saved as its own left->right strip
    save_image(make_grid(imgs[:, 0:1], nrow=N_STEPS), f"{OUT_DIR}/axis{dim}_chA.png")
    save_image(make_grid(imgs[:, 1:2], nrow=N_STEPS), f"{OUT_DIR}/axis{dim}_chB.png")
    rows_A.append(imgs[:, 0:1])
    rows_B.append(imgs[:, 1:2])
    print(f"dim {dim}: center col (idx {N_STEPS//2}) = untouched image, alpha range [{-EDIT_SCALE}, {EDIT_SCALE}]")

# stacked grid: N_DIMS rows x N_STEPS cols, center column = original
grid_A = make_grid(torch.cat(rows_A, dim=0), nrow=N_STEPS)
grid_B = make_grid(torch.cat(rows_B, dim=0), nrow=N_STEPS)
save_image(grid_A, f"{OUT_DIR}/all_axes_chA.png")
save_image(grid_B, f"{OUT_DIR}/all_axes_chB.png")

print(f"\nsaved to {OUT_DIR}/  (rows = W-dims 0..{N_DIMS-1}, columns = alpha sweep, center column = original)")