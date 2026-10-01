'''
paired clip loader for the video diffusion model.

each .mp4 is one cell: 60 frames, 128x128, 7 fps (~8.6 s). grayscale is stored as 3
identical colour channels. target (the channel being generated, chA by default) and
cond (the conditioning channel, chB by default) are loaded TOGETHER with the same
frame window, so frame t of cond always corresponds to frame t of target. two separately
shuffled loaders would break that.
'''
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

#opencv's internal threads fight with DataLoader worker processes
cv2.setNumThreads(0)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
CHA_DIR  = DATA_ROOT / "chA_videos_7"
CHB_DIR  = DATA_ROOT / "chB_videos_7"
MASK_DIR = DATA_ROOT / "mask_videos_7"

#"<recording>_cell12_chA.mp4" / "_chB_masked.mp4" / "_mask.mp4" -> "<recording>_cell12"
_SUFFIX = re.compile(r"_(chA|chB|mask)(_masked)?\.mp4$")

#held out as whole recordings so no recording is in both train and val.
#-174_bc has 24 cells -> 600 train / 24 val
VAL_RECORDINGS = (
    "20260619_HP_bh1-PP7_15_TIGRE_PCP-mSG_17-04_processed-Orthogonal-Projection-174_bc",
)


def cell_id(path):
    return _SUFFIX.sub("", Path(path).name)


def recording_id(cell):
    return cell.rsplit("_cell", 1)[0]


def load_video(path, start=0, num_frames=None):
    '''decode frames [start, start + num_frames) -> uint8 [T, H, W]. num_frames=None reads to the end.'''
    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while num_frames is None or len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= start:
            frames.append(frame[..., 0])
        i += 1
    cap.release()
    if not frames or (num_frames is not None and len(frames) < num_frames):
        raise RuntimeError(f"{path}: wanted frames [{start}, {start + (num_frames or 0)}), got {len(frames)}")
    return np.stack(frames)


def to_tensor(u8):
    '''uint8 [T, H, W] -> float [1, T, H, W] in [-1, 1], (C, T, H, W) layout expected by UNet3DConditionModel.'''
    return torch.from_numpy(u8).float().div(255).mul(2).sub(1).unsqueeze(0)


class PairedVideoDataset(Dataset):
    '''
    returns a dict:
        target: [1, T, H, W]  in [-1, 1]
        cond:   [1, T, H, W]  in [-1, 1], same frames as target
        mask:   [1, T, H, W]  in {0, 1}   (only if mask_dir is given)
        cell, start: which cell and which frame the clip starts at

    num_frames=None returns the full 60-frame clip. otherwise a num_frames-long window,
    starting at a random frame if random_start else at frame 0.
    clips_per_video=k instead cuts every video into k equal, non-overlapping clips
    (k=2 -> frames 0-29 and 30-59), each its own sample; num_frames/random_start are then ignored.
    with mask_dir, pixels outside the mask are set to -1 in both target and cond,
    same as the 2D twoChannelDataset.
    split="train" / "val" keeps cells outside / inside VAL_RECORDINGS; None keeps everything.
    '''
    def __init__(self, target_dir=CHA_DIR, cond_dir=CHB_DIR, mask_dir=None,
                 num_frames=None, random_start=True, clips_per_video=1, split=None):
        super().__init__()
        self.num_frames = num_frames
        self.random_start = random_start
        self.clips_per_video = clips_per_video

        target = {cell_id(p): p for p in Path(target_dir).glob("*.mp4")}
        cond   = {cell_id(p): p for p in Path(cond_dir).glob("*.mp4")}
        mask   = {cell_id(p): p for p in Path(mask_dir).glob("*.mp4")} if mask_dir else None

        missing = set(target) - set(cond) | (set(target) - set(mask) if mask is not None else set())
        if missing:
            raise ValueError(f"{len(missing)} target clips have no cond/mask partner, e.g. {sorted(missing)[:3]}")

        if split not in (None, "train", "val"):
            raise ValueError(f"split must be None, 'train' or 'val', got {split!r}")
        self.cells = sorted(c for c in target
                            if split is None or (recording_id(c) in VAL_RECORDINGS) == (split == "val"))
        self.target_files = [target[c] for c in self.cells]
        self.cond_files   = [cond[c] for c in self.cells]
        self.mask_files   = [mask[c] for c in self.cells] if mask is not None else None

        self.lengths = []
        for p in self.target_files:
            cap = cv2.VideoCapture(str(p))
            self.lengths.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
        if clips_per_video > 1:
            self.num_frames = min(self.lengths) // clips_per_video
        elif num_frames is not None and min(self.lengths) < num_frames:
            raise ValueError(f"num_frames={num_frames} but the shortest clip has {min(self.lengths)} frames")

    def __len__(self):
        return len(self.cells) * self.clips_per_video

    def __getitem__(self, index):
        index, k = divmod(index, self.clips_per_video)
        T = self.num_frames
        if self.clips_per_video > 1:
            start = k * T
        elif T is None:
            start = 0
        elif self.random_start:
            start = np.random.randint(0, self.lengths[index] - T + 1)
        else:
            start = 0

        target = to_tensor(load_video(self.target_files[index], start, T))
        cond   = to_tensor(load_video(self.cond_files[index], start, T))
        out = {"cell": self.cells[index], "start": start}

        if self.mask_files is not None:
            mask = torch.from_numpy(load_video(self.mask_files[index], start, T) > 127).float().unsqueeze(0)
            target = torch.where(mask.bool(), target, torch.full_like(target, -1.0))
            cond   = torch.where(mask.bool(), cond, torch.full_like(cond, -1.0))
            out["mask"] = mask

        out["target"], out["cond"] = target, cond
        return out


def build_dataloader(batch_size=4, shuffle=True, num_workers=4, **dataset_kwargs):
    '''batched tensors come out as [B, 1, T, H, W].'''
    ds = PairedVideoDataset(**dataset_kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=shuffle, pin_memory=True, persistent_workers=num_workers > 0)
