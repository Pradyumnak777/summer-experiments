'''
visualize DMVAE reconstruction quality: real images vs their reconstructions,
for both channels, plus a numeric MSE check.
'''
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from model_dmvae import DMVAE
from data_utils import twoChannelDataset

MODEL_PATH = "vae_checkpoints/dmvae_p8_s8/dmvae_epoch99.pt"
PRIV_DIM   = 8
SHARED_DIM = 8
DEVICE     = torch.device("cuda:9")
CHA_DIR    = "data/singlecell_chA_split/validation"
CHB_DIR    = "data/singlecell_chB_split/validation"
OUT_DIR    = "vae_checkpoints/dmvae_p8_s8/recon_checks"
N_SHOW     = 8   # how many examples to put in the preview grid
os.makedirs(OUT_DIR, exist_ok=True)

model = DMVAE(PRIV_DIM, SHARED_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

dataset = twoChannelDataset(CHA_DIR, CHB_DIR)
loader  = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)


def save_recon_preview(xA, xB, recon_A, recon_B, out_dir=OUT_DIR):
    #un-normalize from [-1,1] -> [0,1] for viewing
    xA      = (xA[:N_SHOW]      * 0.5 + 0.5).clamp(0, 1)
    xB      = (xB[:N_SHOW]      * 0.5 + 0.5).clamp(0, 1)
    recon_A = (recon_A[:N_SHOW] * 0.5 + 0.5).clamp(0, 1)
    recon_B = (recon_B[:N_SHOW] * 0.5 + 0.5).clamp(0, 1)

    for x, recon, name in [(xA, recon_A, "chA"), (xB, recon_B, "chB")]:
        pairs = torch.cat([x, recon], dim=0)   # top row = real, bottom row = recon
        grid  = make_grid(pairs, nrow=N_SHOW)
        save_image(grid, f"{out_dir}/{name}_recon.png")


@torch.no_grad()
def evaluate():
    x = next(iter(loader)).to(DEVICE)   # [B, 2, 128, 128]
    xA, xB = x[:, 0:1], x[:, 1:2]

    out = model(xA, xB)
    recon_A, recon_B = out["recon_A"], out["recon_B"]

    #quick numeric check alongside the images
    mse_A = F.mse_loss(recon_A, xA).item()
    mse_B = F.mse_loss(recon_B, xB).item()
    print(f"per-pixel MSE  chA: {mse_A:.5f}   chB: {mse_B:.5f}")

    save_recon_preview(xA, xB, recon_A, recon_B)
    print(f"saved reconstruction grids to {OUT_DIR}/chA_recon.png and {OUT_DIR}/chB_recon.png")
    print("top row = real input, bottom row = reconstruction")


if __name__ == "__main__":
    evaluate()