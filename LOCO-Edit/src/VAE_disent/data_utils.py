import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import os
from pathlib import Path
from PIL import Image

class twoChannelDataset(Dataset):
    def __init__(self, chA_dir, chB_dir):
        super().__init__()        
        self.chA_files = sorted([f for f in Path(chA_dir).glob('*.png')]) #only stores strings, not actual files..
        self.chB_files = sorted([f for f in Path(chB_dir).glob('*.png')])
        
        #defining a transform
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
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
            
        return torch.cat([imgA, imgB], dim=0) #2 channels -> (2, h, w) NOT (2, 1, h, w)
    
    