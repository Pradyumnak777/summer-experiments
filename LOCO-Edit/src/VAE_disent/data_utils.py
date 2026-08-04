import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import os
from pathlib import Path
from PIL import Image

class twoChannelDataset(Dataset):
    def __init__(self, chA_dir, chB_dir, mask_dir = None):
        super().__init__()        
        self.chA_files = sorted([f for f in Path(chA_dir).glob('*.png')]) #only stores strings, not actual files..
        self.chB_files = sorted([f for f in Path(chB_dir).glob('*.png')])
        self.mask_dir = mask_dir
        #defining a transform
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mask_transform = transforms.Compose([
        transforms.Resize((128, 128),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        ])

                
    def __len__(self):
        #calculate no. of img files in either dir
        return len(self.chA_files)
    
    def __getitem__(self, index):
        #get a file from each directory
        imgA_path = self.chA_files[index]
        imgB_path = self.chB_files[index]
        
        #load both channels as grayscale
        imgA = Image.open(imgA_path).convert('L') #'L' is grayscale(?)
        imgB = Image.open(imgB_path).convert('L')
        
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        
        x = torch.cat([imgA, imgB], dim=0) #[2, 128, 128]
        
        if self.mask_dir is None:
            return x 
        #eg: "sub3_cell7_f012_chA.png" -> "sub3_cell7_f012" -> "sub3_cell7_f012_mask.png"
        base = imgA_path.name[:-len("_chA.png")]
        img_mask = Image.open(f"{self.mask_dir}/{base}_mask.png").convert('L')
        mask = self.mask_transform(img_mask)         # [1, 128, 128], values 0.0 or 1.0
        # mask = (mask > 0.5).float()
        x_masked = torch.where(mask.bool(), x, torch.full_like(x, -1.0)) #applies mask to the image   
        return x_masked, mask #is now the masked img
    
    