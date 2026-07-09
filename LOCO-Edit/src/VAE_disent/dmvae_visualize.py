'''
visualize the private/shared split of a trained DMVAE.
anchor on one real paired image, sweep one latent dim at a time (holding
everything else fixed), decode BOTH channels every time.

what to look for:
  - sweep a priv_A dim -> chA should move, chB MUST be pixel-identical 
  - sweep a priv_B dim -> chB should move, chA MUST be pixel-identical 
  - sweep a shared dim -> both chA and chB may move (this is the real empirical test)
'''
import os
import torch
from torchvision.utils import make_grid, save_image
from model_dmvae import DMVAE
from data_utils import twoChannelDataset

MODEL_PATH = "vae_checkpoints/dmvae_p8_s8/dmvae_epoch50.pt"
PRIV_DIM   = 8
SHARED_DIM = 8
DEVICE     = torch.device("cuda:0")
CHA_DIR    = "data/singlecell_chA_split/train"
CHB_DIR    = "data/singlecell_chB_split/train"
OUT_DIR    = "vae_checkpoints/dmvae_p8_s8/traversals"
os.makedirs(OUT_DIR, exist_ok=True)

model = DMVAE(PRIV_DIM, SHARED_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

dataset = twoChannelDataset(CHA_DIR, CHB_DIR)


def decode_pair(zP_A, zP_B, zS):
    with torch.no_grad():
        recon_A = model.dec_A(torch.cat([zP_A, zS], dim=1))
        recon_B = model.dec_B(torch.cat([zP_B, zS], dim=1))
    return recon_A.squeeze(0), recon_B.squeeze(0)


def traverse_group(anchors, group, dim, span=3, steps=7):
    zP_A_a, zP_B_a, zS_a = anchors
    center = {"priv_A": zP_A_a, "priv_B": zP_B_a, "shared": zS_a}[group][dim].item()
    values = torch.linspace(center - span, center + span, steps)

    A_frames, B_frames = [], []
    for v in values:
        zP_A, zP_B, zS = zP_A_a.clone(), zP_B_a.clone(), zS_a.clone()
        if group == "priv_A":
            zP_A[dim] = v
        elif group == "priv_B":
            zP_B[dim] = v
        else:
            zS[dim] = v

        recon_A, recon_B = decode_pair(zP_A.unsqueeze(0), zP_B.unsqueeze(0), zS.unsqueeze(0))
        A_frames.append(recon_A)
        B_frames.append(recon_B)

    return torch.stack(A_frames), torch.stack(B_frames)


def save_group_traversal(A_frames, B_frames, group, dim, out_dir=OUT_DIR):
    A_frames = (A_frames * 0.5 + 0.5).clamp(0, 1)
    B_frames = (B_frames * 0.5 + 0.5).clamp(0, 1)
    save_image(make_grid(A_frames, nrow=A_frames.shape[0]), f"{out_dir}/{group}_dim{dim}_chA.png")
    save_image(make_grid(B_frames, nrow=B_frames.shape[0]), f"{out_dir}/{group}_dim{dim}_chB.png")


#anchor on one real paired image
x = dataset[0].unsqueeze(0).to(DEVICE)
xA, xB = x[:, 0:1], x[:, 1:2]
with torch.no_grad():
    muP_A, muP_B, muS = model.encode(xA, xB)
anchors = (muP_A.squeeze(0), muP_B.squeeze(0), muS.squeeze(0))

for group, dim_count in [("priv_A", PRIV_DIM), ("priv_B", PRIV_DIM), ("shared", SHARED_DIM)]:
    for dim in range(dim_count):
        A_frames, B_frames = traverse_group(anchors, group, dim)
        save_group_traversal(A_frames, B_frames, group, dim)
        print(f"saved {group} dim {dim}")