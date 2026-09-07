'''
VAE was trained on 16 latent codes..

1. size- major axis, minor axis, area of cell
2. shape- convex,(?) aspect ratio
3. intesity- average intensity, integrated intensity
4. distance- center-to-center, edge-to-edge
'''
import torch
from torch.utils.data import DataLoader
from VAE_disent.model import twoChannelVAE
from VAE_disent.data_utils import twoChannelDataset

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

DEVICE     = torch.device("cuda:9")
SET        = "validation"
CHA_DIR    = f"data/singlecell_chA_split/{SET}"
CHB_DIR    = f"data/singlecell_chB_split/{SET}"
MASK_DIR   = "data/singlecell_mask"
CKPT       = "vae_checkpoints_masked/beta_vae_beta=1_16/vae_epoch99.pt"
LATENT_DIM = 16
BATCH_SIZE = 32


def size(data_2ch, mask):
    #1. measure area of the mask (compared to full image)

    #2. width of mask (first pixel on the left to the last pixel on the right)
    
    #3. height of the mask (first pixel at the top to the lowest pixel at teh bottom)
    
    #NOTE: return all 3, (each one could be a specific "generative factor")

def shape(data_2ch, mask):
    #aspect ratio- (major axis)/(minor axis)
    #measure both height and width and select the major/minor axis based on their values. then divide


def intensity(data_2ch, mask):
    
    #1. average intensity (average of all pixel intensities..)
    
    #2. integrated intensity- (sum of all pixel intensities..NOT average..)


if __name__ == "__main__":
    #load training data
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    
    
