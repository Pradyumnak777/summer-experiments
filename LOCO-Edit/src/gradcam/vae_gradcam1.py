import torch
from VAE_disent.model import twoChannelVAE
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from VAE_disent.data_utils import twoChannelDataset
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
idx = 10
SAVE_DIR = f"gradcam/v1/beta_vae_beta=1_16/{SET}/sample_{idx}"
LATENT_CODES = 16
os.makedirs(SAVE_DIR, exist_ok=True)

dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir="data/singlecell_mask")
loader  = DataLoader(dataset, batch_size=64, shuffle=True)

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

    
def save_overlay(raw_img_tensor, cam, save_path, title):
    
    # un-normalize back to [0, 1] for display
    img = (raw_img_tensor.squeeze().cpu().numpy() * 0.5) + 0.5
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.45)  # overlay heatmap with transparency
    plt.title(title)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"saved to {save_path}")


if __name__ == "__main__":
    #
    model_path = "vae_checkpoints_masked/beta_vae_beta=1_16/vae_epoch99.pt"
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
    fixed_sample, _ = dataset[idx]
    #separate this into 2 channels
    chA_tensor = fixed_sample[0]  # Shape: [H, W]
    chB_tensor = fixed_sample[1]
    
    for i in range(LATENT_CODES):
        latent_idx = i
        input_tensor = fixed_sample.unsqueeze(0).to(DEVICE) #[1, 2, H, W]
        cam = gradcam(input_tensor, latent_idx=latent_idx)

        save_overlay(chA_tensor, cam, f"{SAVE_DIR}/latent_{latent_idx}_chA.png", f"chA, latent {latent_idx}")
        save_overlay(chB_tensor, cam, f"{SAVE_DIR}/latent_{latent_idx}_chB.png", f"chB, latent {latent_idx}")
    
    
