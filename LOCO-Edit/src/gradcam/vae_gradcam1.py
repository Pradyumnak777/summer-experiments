import torch
from VAE_disent.model import twoChannelVAE
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from VAE_disent.data_utils import twoChannelDataset
from disentanglement_semantic_eval.gan_dci import get_gene_burst_intensity, load_burst_mask
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
import os

DEVICE = torch.device("cuda:9")
# CHA_PATH = "data/singlecell_chA_split/train/20260611_HP_bh1-PP7_15_TIGRE-PCP-mSG_17-01_processed-Orthogonal Projection-146_bc_cell0_f000_chA.png"
# CHB_PATH = "data/singlecell_chB_split/train/20260611_HP_bh1-PP7_15_TIGRE-PCP-mSG_17-01_processed-Orthogonal Projection-146_bc_cell0_f000_chB.png"
SET = "validation"
CHA_DIR = f"data/singlecell_chA_split/{SET}"
CHB_DIR = f"data/singlecell_chB_split/{SET}"
idx = 150   #index into the burst subset of SET (frames with a hand-drawn burst mask), not the full dataset

#below is VAE config..
BETA = 1
LATENT_DIM = 16
# TARGET_DIM = 0

#"burst_sample" so these don't mix with the older sample_{idx} folders, which indexed the full dataset
SAVE_DIR = f"gradcam/vae_guided/guided_vae_beta=1_16_intensity-z0_area-z1_aspect-z2/{SET}/burst_sample_{idx}"

LATENT_CODES = 16
os.makedirs(SAVE_DIR, exist_ok=True)

dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir="data/singlecell_mask")
#keep only the frames with an annotated gene burst, burst_idx maps a subset index back to the full dataset
subset, burst_int, burst_idx = get_gene_burst_intensity(dataset)
loader  = DataLoader(subset, batch_size=64, shuffle=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class GradCAMVAE:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        
        # Register hooks
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        self.activations = output.detach()  # (B, C, H, W)
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # (B, C, H, W)
    
    def __call__(self, input_tensor, latent_idx):
        self.model.zero_grad()
        input_tensor.requires_grad_(True)
        
        #foward pass thru the model
        mu, logvar = self.model.encode(input_tensor)
        # z = self.model.reparametrize(mu, logvar)
        
        #
        self.model.zero_grad()
        # z[:, latent_idx].sum().backward()  # this is the chosen SCALAR to backprop!
        mu[:, latent_idx].sum().backward()     #this is the chosen SCALAR to backprop!
        
        # Compute GradCAM
        grads = self.gradients[0]  # (C, H, W)
        acts = self.activations[0]  # (C, H, W)
        
        # Weights: average gradient across spatial dims
        weights = grads.mean(dim=(1, 2))  # (C,)
        
        # Weighted activation sum
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=0)  # (H, W)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        cam = cam.unsqueeze(0).unsqueeze(0)                       # (1, 1, 8, 8)
        cam = F.interpolate(cam, size=(128, 128), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()                # (128, 128)

        return cam

    
def mask_ring(mask):
    #mask: [1, H, W] binary -> [H, W] bool outline, the 1 px ring just outside the mask
    #(dilate by a 3x3 max pool, minus the mask), so drawing it never covers a pixel inside the mask
    dilated = F.max_pool2d(mask.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    return (dilated - mask)[0].bool().cpu().numpy()


def save_grid(raw_img_tensor, ring, cam, save_path, title, mask_name):
    #one row of 3: channel | channel with the mask outline (red) | gradcam heatmap over the channel

    #un-normalize back to [0, 1] for display
    img = (raw_img_tensor.squeeze().cpu().numpy() * 0.5) + 0.5
    img = np.clip(img, 0, 1)

    outlined = np.repeat(img[..., None], 3, axis=-1)   #[H, W, 3], rgb so the outline can be red
    outlined[ring] = [1.0, 0.0, 0.0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("input")
    axes[1].imshow(outlined)
    axes[1].set_title(f"{mask_name} mask outline")
    axes[2].imshow(img, cmap='gray', vmin=0, vmax=1)
    heat = axes[2].imshow(cam, cmap='jet', alpha=0.45, vmin=0, vmax=1)  #overlay heatmap with transparency
    axes[2].set_title("gradcam")
    fig.colorbar(heat, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis('off')
    fig.suptitle(title)
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"saved to {save_path}")


if __name__ == "__main__":
    #
    model_path = f"vae_checkpoints_masked/guided_vae_beta=1_16_intensity-z0_area-z1_aspect-z2/vae_final.pt"
    model = twoChannelVAE(latent_dim=16)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    #loaidn ginput
    # imgA = Image.open(CHA_PATH).convert('L')
    # imgB = Image.open(CHB_PATH).convert('L')
    # chA_tensor = transform(imgA)  # (1, 128, 128)
    # chB_tensor = transform(imgB)  # (1, 128, 128)
    # input_tensor = torch.cat([chA_tensor, chB_tensor], dim=0).unsqueeze(0).to(DEVICE)  # (1, 2, 128, 128)

    gradcam = GradCAMVAE(model, model.conv4)
    fixed_sample, _ = subset[idx]
    #separate this into 2 channels
    chA_tensor = fixed_sample[0]  # Shape: [H, W]
    chB_tensor = fixed_sample[1]

    #every subset frame has a non-empty hand-drawn burst mask, so there's no fallback to the cell mask
    #(frames without a burst are never picked, so nothing gets highlighted that isn't a burst)
    full_idx = burst_idx[idx]                                                   #subset index -> full dataset index
    mask_name = "burst"
    burst = load_burst_mask(dataset.chB_files[full_idx], dataset.mask_transform)  #[1, 128, 128]
    ring = mask_ring(burst)
    print(f"burst subset index {idx} of {len(subset)} (full dataset index {full_idx}): {dataset.chA_files[full_idx].name}")

    for i in range(LATENT_CODES):
        latent_idx = i
        input_tensor = fixed_sample.unsqueeze(0).to(DEVICE) #[1, 2, H, W]
        cam = gradcam(input_tensor, latent_idx=latent_idx)

        save_grid(chA_tensor, ring, cam, f"{SAVE_DIR}/latent_{latent_idx}_chA.png", f"chA, latent {latent_idx}", mask_name)
        save_grid(chB_tensor, ring, cam, f"{SAVE_DIR}/latent_{latent_idx}_chB.png", f"chB, latent {latent_idx}", mask_name)
    
    
