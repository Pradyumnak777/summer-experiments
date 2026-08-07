'''
load one model and check for disentangled features
'''
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from model import twoChannelVAE
from data_utils import twoChannelDataset
import debugpy
import random
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# debugpy.listen(("127.0.0.1", 5678)) #127.0.0.1, cz only local host can talk to the port
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")

SAVE_DIR = "vae_checkpoints_masked/beta_vae_beta=1_16"
MODEL_PATH = f"{SAVE_DIR}/vae_epoch99.pt"
LATENT_DIM = 16          # must match what MODEL_PATH was trained with
DEVICE     = torch.device("cuda:9")
SET = "validation"
CHA_DIR = f"data/singlecell_chA_split/{SET}"
CHB_DIR = f"data/singlecell_chB_split/{SET}"
idx = 10
OUT_DIR = f"{SAVE_DIR}/{SET}/latent_traversal_sample_{idx}"
model = twoChannelVAE(latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir="data/singlecell_mask")
loader  = DataLoader(dataset, batch_size=64, shuffle=True)


def active_units(model, loader):
    #per-dimension KL, averaged over a batch -> which latent dims are actually "on"
    # idx = 1000
    fixed_sample, _ = dataset[idx] #returns the mask too now
    os.makedirs(OUT_DIR, exist_ok=True)
    x = fixed_sample.unsqueeze(0).to(DEVICE)
    '''
    saving above img for comparison
    '''
    tensor_img = x.detach().cpu()
    if tensor_img.ndim == 4:
        tensor_img = tensor_img[0]
        
    for i in range(tensor_img.shape[0]):
        channel_data = tensor_img[i].numpy()
        plt.imsave(f"{OUT_DIR}/channel_{i}.png", channel_data, cmap="gray")
    
    
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

def save_center_reconstruction(model, mu_anchor, out_dir=OUT_DIR):
    """Decodes the unperturbed anchor latent vector and saves its channels."""
    model.eval()
    with torch.no_grad():
        # Decode the anchor latent vector (add back batch dimension)
        recon = model.decode(mu_anchor.unsqueeze(0))  # shape [1, C, H, W]
        recon = (recon * 0.5 + 0.5).clamp(0, 1)       # normalize if needed
        
        # Save each channel of the center reconstruction
        for ch, name in [(0, "chA"), (1, "chB")]:
            save_image(recon[:, ch:ch+1], f"{out_dir}/center_recon_{name}.png")

if __name__ == "__main__":
    kl_per_dim = active_units(model, loader)
    print("per-dimension KL (higher = more active):")
    for i, kl in enumerate(kl_per_dim):
        print(f"  dim {i}: {kl.item():.4f}")

    x_anchor, _ = dataset[idx]
    x_anchor = x_anchor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mu_anchor, _ = model.encode(x_anchor)
    mu_anchor = mu_anchor.squeeze(0)  # [latent_dim]

    values = torch.linspace(-3, 3, 7)  #prior is N(0,1), so +-3 covers ~all its mass
    for dim in range(LATENT_DIM):
        images = traverse(model, mu_anchor, dim)  # values computed inside now
        save_traversal(images, dim)

    save_center_reconstruction(model, mu_anchor)