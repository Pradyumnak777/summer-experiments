'''
load one model and check for disentangled features
'''
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from model import twoChannelVAE
from data_utils import twoChannelDataset

MODEL_PATH = "vae_checkpoints/beta_vae_beta=1/vae_epoch99.pt"
LATENT_DIM = 16          # must match what MODEL_PATH was trained with
DEVICE     = torch.device("cuda:9")

CHA_DIR = "data/singlecell_chA_split/validation"
CHB_DIR = "data/singlecell_chB_split/validation"
OUT_DIR = "vae_checkpoints/beta_vae_beta=1_16/disentangle_checks_val"
os.makedirs(OUT_DIR, exist_ok=True)

model = twoChannelVAE(latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

dataset = twoChannelDataset(CHA_DIR, CHB_DIR)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)


def active_units(model, loader):
    #per-dimension KL, averaged over a batch -> which latent dims are actually "on"
    x = next(iter(loader)).to(DEVICE)
    with torch.no_grad():
        mu, logvar = model.encode(x)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, latent_dim]
        kl_per_dim = kl_per_dim.mean(dim=0)                          # [latent_dim]
    return kl_per_dim


def traverse(model, mu_anchor, dim, span=3, steps=7):
    center = mu_anchor[dim].item()
    values = torch.linspace(center - span, center + span, steps)
    outputs = []
    with torch.no_grad():
        for v in values:
            z = mu_anchor.clone()
            z[dim] = v
            recon = model.decode(z.unsqueeze(0))
            outputs.append(recon.squeeze(0))
    return torch.stack(outputs)

def save_traversal(images, dim, out_dir=OUT_DIR):
    images = (images * 0.5 + 0.5).clamp(0, 1)
    for ch, name in [(0, "chA"), (1, "chB")]:
        grid = make_grid(images[:, ch:ch+1], nrow=images.shape[0])
        save_image(grid, f"{out_dir}/dim{dim}_{name}.png")


if __name__ == "__main__":
    kl_per_dim = active_units(model, loader)
    print("per-dimension KL (higher = more active):")
    for i, kl in enumerate(kl_per_dim):
        print(f"  dim {i}: {kl.item():.4f}")

    x_anchor = dataset[0].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mu_anchor, _ = model.encode(x_anchor)
    mu_anchor = mu_anchor.squeeze(0)  # [latent_dim]

    values = torch.linspace(-3, 3, 7)  #prior is N(0,1), so +-3 covers ~all its mass
    for dim in range(LATENT_DIM):
        images = traverse(model, mu_anchor, dim)  # values computed inside now
        save_traversal(images, dim)
