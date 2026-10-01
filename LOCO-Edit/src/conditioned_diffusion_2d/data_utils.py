'''
paired chA / chB single-cell crops for the conditioned 2D diffusion model.

same crops and preprocessing as the 2D DDPMs (VAE_disent twoChannelDataset / singleChannelDataset):
133x133 png -> 128x128, [-1, 1], pixels outside the cell mask set to -1.
the split is different: the old train/validation folders split by frame, so 602 of 624 cells
appear in both. here both folders are pooled and whole recordings are held out instead, the same
recording(s) as the video model (video_diffusion.data_utils.VAL_RECORDINGS).
'''
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from video_diffusion.data_utils import VAL_RECORDINGS

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
CHA_DIR  = DATA_ROOT / "singlecell_chA_split"    #train/ and validation/ are pooled
CHB_DIR  = DATA_ROOT / "singlecell_chB_split"
MASK_DIR = DATA_ROOT / "singlecell_mask"


def recording_id(base):
    '''"<recording>_cell3_f012" -> "<recording>", written with hyphens like the video filenames.'''
    return base.rsplit("_cell", 1)[0].replace(" ", "-")


class PairedCropDataset(Dataset):
    '''
    returns a dict:
        target: [1, 128, 128] chA in [-1, 1]
        cond:   [1, 128, 128] chB in [-1, 1], same crop
        mask:   [1, 128, 128] in {0, 1}
        name:   "<recording>_cellN_fMMM"
    split="train" / "val" keeps crops outside / inside VAL_RECORDINGS; None keeps everything.
    crops with an empty mask are dropped, as in twoChannelDataset.
    '''
    def __init__(self, split=None):
        super().__init__()
        if split not in (None, "train", "val"):
            raise ValueError(f"split must be None, 'train' or 'val', got {split!r}")
        chA = {p.name[:-len("_chA.png")]: p for p in CHA_DIR.glob("*/*_chA.png")}
        chB = {p.name[:-len("_chB.png")]: p for p in CHB_DIR.glob("*/*_chB.png")}
        unpaired = set(chA) ^ set(chB)
        if unpaired:
            raise ValueError(f"{len(unpaired)} crops exist in only one channel, e.g. {sorted(unpaired)[:3]}")

        bases = sorted(b for b in chA
                       if split is None or (recording_id(b) in VAL_RECORDINGS) == (split == "val"))
        self.bases = [b for b in bases if np.asarray(Image.open(MASK_DIR / f"{b}_mask.png")).any()]
        self.n_empty = len(bases) - len(self.bases)
        self.chA_files = [chA[b] for b in self.bases]
        self.chB_files = [chB[b] for b in self.bases]

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, index):
        base = self.bases[index]
        mask = self.mask_transform(Image.open(MASK_DIR / f"{base}_mask.png").convert("L"))
        keep = mask.bool()
        chA = self.transform(Image.open(self.chA_files[index]).convert("L"))
        chB = self.transform(Image.open(self.chB_files[index]).convert("L"))
        return {"target": torch.where(keep, chA, torch.full_like(chA, -1.0)),
                "cond":   torch.where(keep, chB, torch.full_like(chB, -1.0)),
                "mask":   mask,
                "name":   base}
