'''
training a single channel diffusion model by reusing weights from 2-channel diffusion model
'''
import torch

DEVICE      = torch.device("cuda:9")
CH_DIR = "data/singlecell_chA_split/train"
SAVE_DIR    = "diffusion_checkpoints/ddpm_chA_128"
IMG_SIZE    = 128



