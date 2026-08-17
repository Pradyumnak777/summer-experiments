'''
gradCAM for the styleGAN architecture. ccurrent pipeline involves-

1. Inverting image using optimization based method into W-space (alternatives: w+ space and S-space)
2. construct image using theat W space vector
3. Output image
4. "Edits" are tweaks to the W-space vector (eg- plain traversal or GANspace)
'''
import os
import sys
import torch
import pickle
from torchvision.utils import make_grid, save_image

REPO_DIR = "/scratch/pbk5339/summer/LOCO-Edit/src/GAN/stylegan2-ada-pytorch"
sys.path.insert(0, REPO_DIR)
import legacy
from inversion_opti import project, compute_w_stats
import numpy as np
from VAE_disent.data_utils import twoChannelDataset
from torch.utils.data import DataLoader

#config
DEVICE = torch.device("cuda:9")
SET      = "validation"
PKL      = "GAN/stylegan2-ada-pytorch/training-runs/00010-data_npy-auto1-kimg0600-ada-bg/network-snapshot-001000.pkl"
CHA_DIR  = f"data/singlecell_chA_split/{SET}"
CHB_DIR  = f"data/singlecell_chB_split/{SET}"
SEED = 0
IDX = 0
NUM_STEPS = 1000


class vizHook:
    def __init__(self, target_module):
        self.activations = None
        self.gradients = None

        '''
        #NOTE: register_forward_hook is an inbuilt method into every pytorch module (like conv2d, etc)
        '''
        self.forward_handle = target_module.register_forward_hook(self._forward_hook) #for activations
        self.backward_handle = target_module.register_full_backward_hook(self._backward_hook) #for grads
    
    def _forward_hook(self, module, input, output):
        # output in StyleGAN conv layers is a 4D Tensor: [B, C, H, W]
        self.activations = output #output of the target module
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def remove(self):
        self.forward_handle.remove()
        self.backward_handle.remove()
        

if __name__ == "__main__":
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir="data/singlecell_mask")
    x, _ = dataset[IDX] #chA and chB img
    # loader = DataLoader(dataset, batch_size=1, shuffle=False) 
    # cell_images,_ = next(iter(loader))
    
    # img = cell_images[IDX] #chA and chB imgs 
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    '''
    pass this to the GAN now
    '''
    with open(PKL, 'rb') as f:
        nets = legacy.load_network_pkl(f)
        G = nets['G_ema'].to(DEVICE).eval() #generator
        D = nets['D'].to(DEVICE).eval() #discriminator
    
    # G.requires_grad_(False) #not sure yet..
    # D.requires_grad_(False)
    
    '''
    invert this image to get the W vector
    '''
    # w_avg, w_std = compute_w_stats(G, DEVICE) #needed only if many images ig
    
    x = x.to(DEVICE)
    x = ((x * 0.5 + 0.5).clamp(0, 1) * 255.0).round() #X needs to be in target range for the functtion
    w_steps = project(G, D, target=x, num_steps=NUM_STEPS, device=DEVICE, verbose=False)
    w_invert = w_steps[-1].unsqueeze(0)
    w_invert = w_invert.detach().requires_grad_(True) #required for hooks!
    # with torch.no_grad():
    '''
    HOOK HERE!! for gradcam/visualization!
    '''
    target_layer = 
    
    with torch.enable_grad():
        x_recon = G.synthesis(w_invert, noise_mode='const')
    
    
    