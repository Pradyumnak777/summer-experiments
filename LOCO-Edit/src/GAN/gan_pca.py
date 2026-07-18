# gan_ganspace.py — PCA over sampled W vectors (the GANSpace method)
import os
import sys
import torch
from torchvision.utils import make_grid, save_image

REPO_DIR = "/scratch/pbk5339/summer/LOCO-Edit/src/GAN/stylegan2-ada-pytorch"
sys.path.insert(0, REPO_DIR)
import legacy

#config stuff
DEVICE      = torch.device("cuda:9")
PKL         = "stylegan2-ada-pytorch/training-runs/00010-data_npy-auto1-kimg1000-ada-bg/network-snapshot-001000.pkl"
OUT_DIR     = "gan_pca_results"
TRUNC_PSI   = 1.0       
N_SAMPLES   = 100000      #how many w's we build the PCA from
BATCH       = 100       
K           = 5         #top-k principal components to look at
EDIT_SCALE  = 5.0 
N_STEPS     = 5        
SEED        = 0
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)

#load the generator
with open(PKL, 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE).eval()
G.requires_grad_(False)

NUM_WS = G.num_ws
W_DIM  = G.w_dim
print(f"loaded G_ema: w_dim={W_DIM}, num_ws={NUM_WS}, img_channels={G.img_channels}, res={G.img_resolution}")


#step 1: sample a ton of w's from the model. each w is 512-dim, this is what we PCA on
@torch.no_grad()
def sample_ws(n):
    chunks = []
    for i in range(0, n, BATCH):
        b = min(BATCH, n - i)
        z = torch.randn([b, G.z_dim], device=DEVICE)
        c = torch.zeros([b, G.c_dim], device=DEVICE)
        ws = G.mapping(z, c, truncation_psi=TRUNC_PSI)   #[b, NUM_WS, W_DIM]
        chunks.append(ws[:, 0, :].cpu())                  #just the single shared w, before it gets copied to every layer
    return torch.cat(chunks, dim=0)                        #[n, W_DIM]


W_samples = sample_ws(N_SAMPLES).to(DEVICE)               #[N_SAMPLES, 512]
print(f"sampled {N_SAMPLES} w vectors, shape={tuple(W_samples.shape)}")

#step 2: PCA. center the samples, then SVD. the rows of Vt are the principal directions in the 512-dim space
w_mean = W_samples.mean(dim=0)                             #the "average" w -> clean point to traverse from
centered = W_samples - w_mean
U, S, Vt = torch.linalg.svd(centered, full_matrices=False)   #Vt[i] = i-th principal direction (512-dim)
explained_var = (S ** 2) / (N_SAMPLES - 1)

#print how much of the total variation each component explains
print("\ntop principal components (directions of max variance across the sampled w's)")
print("comp | singular value | % var explained")
total_var = explained_var.sum().item()
for i in range(K):
    pct = 100 * explained_var[i].item() / total_var
    print(f"  {i:2d} |    {S[i].item():8.3f}   |   {pct:5.2f}%")

#step 3: walk along each PCA direction, anchored at the mean w. same traversal mechanic as the axis one
alphas = torch.linspace(-EDIT_SCALE, EDIT_SCALE, N_STEPS, device=DEVICE)
ws_anchor_full = w_mean.unsqueeze(0).repeat(NUM_WS, 1)    #broadcast mean-w to all layers -> [NUM_WS, 512]

for i in range(K):
    v_i = Vt[i]                                           #the i-th direction, already unit norm
    ws_batch = ws_anchor_full.unsqueeze(0).repeat(N_STEPS, 1, 1).clone()   #[N_STEPS, NUM_WS, 512]
    ws_batch += alphas.view(N_STEPS, 1, 1) * v_i.view(1, 1, -1)             #same nudge at every layer
    with torch.no_grad():
        img = G.synthesis(ws_batch, noise_mode='const')
    imgs = (img * 0.5 + 0.5).clamp(0, 1)
    save_image(make_grid(imgs[:, 0:1], nrow=N_STEPS), f"{OUT_DIR}/pca{i}_chA.png")
    save_image(make_grid(imgs[:, 1:2], nrow=N_STEPS), f"{OUT_DIR}/pca{i}_chB.png")

print(f"\nsaved to {OUT_DIR}/  (component 0..{K-1}, center column = mean-w image)")