'''
differ4ence between inverted image and the actual image?
'''

import torch
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from skimage.metrics import structural_similarity

from VAE_disent.data_utils import twoChannelDataset
from VAE_disent.diffusion_model import build_unet

DEVICE   = torch.device("cuda:9")

import lpips
loss_fn = lpips.LPIPS(net='alex').to(DEVICE)

CKPT     = "diffusion_checkpoints/ddpm_2ch_128_masked/unet_ema_epoch80.pt"
CHA_DIR  = "data/singlecell_chA_split/validation"
CHB_DIR  = "data/singlecell_chB_split/validation"
SEED = 0
BATCH_SIZE = 128


if __name__ == "__main__":
    '''
    1. load data
    2. invert the image to get W(save on d isk? or save space?)
    3. get reconstr. image, compare them both using metrics
    '''
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir="data/singlecell_mask")
    #shuffle off so the numbers repeat run to run
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
