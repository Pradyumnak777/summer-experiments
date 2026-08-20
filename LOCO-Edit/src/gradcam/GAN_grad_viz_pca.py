'''
gradient/jacobian sensitivity heatmap for the styleGAN architecture. ccurrent pipeline involves-

1. inverting image using optimization based method into W-space (alternatives: w+ space and S-space)
2. construct image using theat W space vector
3. output image
4. "edits" are tweaks to the W-space vector (eg- plain traversal or GANspace)
'''

import os
import sys
import torch
import pickle
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
REPO_DIR = "/scratch/pbk5339/summer/LOCO-Edit/src/GAN/stylegan2-ada-pytorch"
sys.path.insert(0, REPO_DIR)
import legacy
import numpy as np
from VAE_disent.data_utils import twoChannelDataset
from torch.utils.data import DataLoader

import debugpy
debugpy.listen(("127.0.0.1", 5678))
print("Waiting for debugger attach on port 5678...")
debugpy.wait_for_client()
print("Debugger attached! Running code...")


#config
DEVICE = torch.device("cuda:9")
SET      = "validation"
PKL      = "GAN/stylegan2-ada-pytorch/training-runs/00010-data_npy-auto1-kimg1000-ada-bg/network-snapshot-001000.pkl"
CHA_DIR  = f"data/singlecell_chA_split/{SET}"
CHB_DIR  = f"data/singlecell_chB_split/{SET}"
SEED = 0
IDX = 0
NUM_STEPS = 1000
DIM_TO_PROBE = 0
OUT_DIR  = "gradcam/gan_viz_results_pca"
NPY_DATA = "GAN/data_npy_masked_val/000000.npy"
W_DIR = "GAN/gan_inversion_results_masked/val/000000/projected_w.npz"
PCA_DIR = "GAN/gan_pca_results_inverted_w_masked/dirs.npz"

def jacobian_gen(G, w_invert, v_pca_dir):
    '''
    d(x_out)/d(w_i)
    
    -> computed using the forward mode AD to ease computation(?) 
    '''
    def decode(w):
        if w.ndim == 2: # If shape is [1, 512]
            ws = w.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)
        else: # If shape is [1, num_ws, 512]
            ws = w
        img = G.synthesis(ws, noise_mode='const')
        return img.view(-1)
    
    img_flat = decode(w_invert)
    
    v = torch.zeros_like(w_invert)
    
    v = torch.zeros_like(w_invert)
    
    
    
    if w_invert.ndim == 2:
        # Standard W space: [1, 512]
        v[0] = v_pca_dir
        print("w-space")
    else:
        # Extended W+ space: [1, num_ws, 512]
        # Broadcast the 512-dim PCA vector across all layers
        v[0, :] = v_pca_dir.view(1, -1)
        print("w+ space")
    
    u0 = torch.zeros_like(img_flat, requires_grad=True)
    g = torch.autograd.grad(img_flat, w_invert, grad_outputs=u0, create_graph=True, retain_graph=True)[0]

    # J * v evaluates partial derivative wrt ONLY dim_to_probe
    jvp = torch.autograd.grad((g * v).sum(), u0, retain_graph=True)[0]
    
    return jvp, img_flat #jvp is the jacobian, d(x_out)/d(w_i) result..

def apply_colormap_and_overlay(heatmap_2d, recon_img_rgb, alpha=0.5):
    '''
    Applies a JET colormap to a 2D heatmap [H, W] and overlays it on an RGB image [3, H, W].
    '''
    # Convert heatmap to numpy array
    heatmap_np = heatmap_2d.squeeze().cpu().detach().numpy()
    
    # Apply JET colormap using Matplotlib (returns RGBA [H, W, 4])
    cmap = plt.get_cmap('jet')
    colored_heatmap = cmap(heatmap_np)[:, :, :3] # Take RGB channels [H, W, 3]
    
    # Convert back to PyTorch Tensor [3, H, W]
    heatmap_rgb = torch.from_numpy(colored_heatmap).permute(2, 0, 1).to(recon_img_rgb.device).float()
    
    # Blend image and heatmap: (1 - alpha) * Base + alpha * Heatmap
    overlay = (1.0 - alpha) * recon_img_rgb + alpha * heatmap_rgb
    overlay = overlay.clamp(0.0, 1.0)
    
    return heatmap_rgb, overlay


def to_rgb_display(img_tensor):
    # Pad a zero 3rd channel for 2-channel single-cell data
    zero_ch = torch.zeros((1, H, W), device=img_tensor.device)
    return torch.cat([img_tensor, zero_ch], dim=0)



if __name__ == "__main__":
    # dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir="data/singlecell_mask")
    # x, _ = dataset[IDX] # chA and chB img
    raw = np.load(NPY_DATA)
    x = torch.from_numpy(raw).float() / 255.0 * 2.0 - 1.0
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load network
    with open(PKL, 'rb') as f:
        nets = legacy.load_network_pkl(f)
        G = nets['G_ema'].to(DEVICE).eval()
    
    # Prepare target image
    x = x.to(DEVICE)
    x_target = ((x * 0.5 + 0.5).clamp(0, 1) * 255.0).round()
    
    #checking if its W space or not-
    w = np.load(W_DIR)["w"]
    check = np.abs(w - w[:, :1, :]).max()
    print(check) #maxium difference between the 12 nums of 512 dim Ws
    
    # Load pre-computed W-space vector instead of inverting
    w_invert = torch.tensor(w, device=DEVICE, dtype=torch.float32)
    w_invert = w_invert.detach().requires_grad_(True)
    
    pca_data = np.load(PCA_DIR)
    
    Vt = torch.tensor(pca_data["Vt"], device=DEVICE, dtype=torch.float32)
    v_dir = Vt[DIM_TO_PROBE]
    
    jvp_flat, img_flat = jacobian_gen(G, w_invert, v_dir)
    C, H, W = x.shape

    recon_img = img_flat.view(C, H, W).detach()
    recon_norm = (recon_img * 0.5 + 0.5).clamp(0, 1)
    orig_norm  = (x * 0.5 + 0.5).clamp(0, 1)

    grad_image = jvp_flat.view(C, H, W).detach()
    heat_ch    = grad_image.abs()                          # [2,H,W] per-channel
    heat_comb  = torch.norm(grad_image, dim=0, keepdim=True)  # [1,H,W] combined

    # raw energy BEFORE normalizing -- lets you rank dims against each other later
    print(f"dim {DIM_TO_PROBE} energy: chA={heat_ch[0].sum():.4f}  chB={heat_ch[1].sum():.4f}")

    def norm01(t):
        return (t - t.min()) / (t.max() - t.min() + 1e-8)

    def overlay(heat2d, base_rgb, alpha=0.6):
        '''alpha scales WITH the heat, so low-sensitivity areas keep the base image
           instead of getting a flat blue jet(0) wash'''
        h = heat2d.squeeze().cpu().numpy()
        hm = torch.from_numpy(plt.get_cmap('jet')(h)[:, :, :3]).permute(2, 0, 1).float().to(base_rgb.device)
        a  = alpha * torch.from_numpy(h).float().to(base_rgb.device).unsqueeze(0)
        return ((1 - a) * base_rgb + a * hm).clamp(0, 1)

    dim_dir = os.path.join(OUT_DIR, f"dim_{DIM_TO_PROBE}")
    os.makedirs(dim_dir, exist_ok=True)

    # grayscale per-channel views
    for ch, name in [(0, "chA"), (1, "chB")]:
        save_image(orig_norm[ch:ch+1],  os.path.join(dim_dir, f"original_{name}.png"))
        save_image(recon_norm[ch:ch+1], os.path.join(dim_dir, f"recon_{name}.png"))
        h = norm01(heat_ch[ch:ch+1])
        save_image(h, os.path.join(dim_dir, f"heat_{name}.png"))
        base = recon_norm[ch:ch+1].repeat(3, 1, 1)          # grayscale -> rgb for the overlay
        save_image(overlay(h, base), os.path.join(dim_dir, f"overlay_{name}.png"))

    # keep the false-color composite too, it's genuinely useful for chA/chB colocalization
    save_image(to_rgb_display(orig_norm),  os.path.join(dim_dir, "original_rgb.png"))
    save_image(to_rgb_display(recon_norm), os.path.join(dim_dir, "recon_rgb.png"))
    # save_image(overlay(norm01(heat_comb), to_rgb_display(recon_norm)),
    #            os.path.join(dim_dir, "overlay_rgb.png"))

    print(f"saved to {dim_dir}")