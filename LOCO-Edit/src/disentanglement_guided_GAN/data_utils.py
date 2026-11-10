'''
data utils for trainnig attribute classifiers (using the interfaceGAN method here)

1. take the train frames which have a hand-drawn gene burst mask
2. invert each one to a shared w (inversion_opti.project_batch, same as GAN/gan_inversion_train-set.py)
3. save the w together with its burst factors + binary (median split) labels, one .npz per frame,
   named by the cell's base name ("<video>_cell12_f010.npz") so every w links back to its images/masks

run from src/ (one process per gpu, the frames are split between them):
    CUDA_VISIBLE_DEVICES=8,9 torchrun --nproc_per_node=2 -m disentanglement_guided_GAN.data_utils
or on a single gpu (also the easiest way to debug):
    CUDA_VISIBLE_DEVICES=9 python -m disentanglement_guided_GAN.data_utils
'''
import os
import sys
import importlib.util
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
import debugpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VAE_disent.data_utils import twoChannelDataset
from disentanglement_semantic_eval.gan_dci import get_burst_factors, BURST_FACTORS

REPO_DIR = "/scratch/pbk5339/summer/LOCO-Edit/src/GAN/stylegan2-ada-pytorch"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
import legacy

#load the inversion functions dynamically (stylegan2-ada-pytorch isn't a package)
spec = importlib.util.spec_from_file_location("inversion_opti", Path(REPO_DIR) / "inversion_opti.py")
inversion_opti = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inversion_opti)
project_batch = inversion_opti.project_batch
compute_w_stats = inversion_opti.compute_w_stats

#config stuff
#no DEVICE here, every rank gets its own gpu in setup_DDP (pick the physical gpus with CUDA_VISIBLE_DEVICES)
SET      = "train"
CHA_DIR  = f"data/singlecell_chA_split/{SET}"
CHB_DIR  = f"data/singlecell_chB_split/{SET}"
MASK_DIR = "data/singlecell_mask"
PKL      = "GAN/stylegan2-ada-pytorch/training-runs/00010-data_npy-auto1-kimg1000-ada-bg/network-snapshot-001000.pkl"
SAVE_DIR = "GAN/gan_inverted_ws_gene_burst_masked_train"
SIZE     = 128
BATCH    = 64
SEED     = 0


def setup_DDP():
    #every inversion is independent, so nothing is synced between gpus, each rank just takes its share of the frames.
    #the process group is only for the barrier at the end. gloo (cpu) + a long timeout, so a rank that finishes
    #hours before the other doesn't time out while waiting there
    #without torchrun (plain python -m ...) there's no LOCAL_RANK -> run as a single rank 0
    if "LOCAL_RANK" not in os.environ:
        return 0, 1
    dist.init_process_group("gloo", timeout=timedelta(hours=48))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def cleanup_DDP():
    if dist.is_initialized():
        dist.barrier()   #wait for every rank to finish saving
        dist.destroy_process_group()


def get_filtered_data(dataset_og):
    #takes the original dataset and then creates a subset based on the availalbility of gene burst masks(like in some other files)
    #same frames as gan_dci.get_burst_factors, factors is [N, 3] in BURST_FACTORS order (intensity, log area, log aspect)
    #burst_idx maps a row of factors back to its index in dataset_og
    subset, factors, burst_idx = get_burst_factors(dataset_og)
    return subset, factors, burst_idx


def load_target(chA_path, chB_path):
    #same preprocessing as GAN/convert_to_numpy_new.py (what the old inversions were run on), so the w's are comparable:
    #resize to 128, multiply in the soft (bilinear) cell mask, uint8 [2, H, W] in [0, 255] like project_batch expects
    base = chA_path.name[:-len("_chA.png")]
    mask = Image.open(Path(MASK_DIR) / f"{base}_mask.png").convert('L').resize((SIZE, SIZE), resample=Image.BILINEAR)
    mask = np.array(mask).astype(np.float32) / 255.0
    a = np.array(Image.open(chA_path).convert('L').resize((SIZE, SIZE)), dtype=np.uint8)
    b = np.array(Image.open(chB_path).convert('L').resize((SIZE, SIZE)), dtype=np.uint8)
    return np.stack([(mask * a).astype(np.uint8), (mask * b).astype(np.uint8)], axis=0)


if __name__ == "__main__":
    local_rank, world_size = setup_DDP()
    DEVICE = torch.device(f"cuda:{local_rank}")   #with CUDA_VISIBLE_DEVICES=8,9: rank 0 -> gpu 8, rank 1 -> gpu 9
    is_main = local_rank == 0
    torch.manual_seed(SEED + local_rank)
    os.makedirs(SAVE_DIR, exist_ok=True)   #exist_ok, so both ranks calling it is fine

    #debugging: under torchrun every rank runs this file, so only ONE rank may listen. a second rank listening on
    #the same port crashes with "address already in use". rank 1 keeps running on its own meanwhile (nothing waits
    #on it until the final barrier). to step through everything instead, run on a single gpu without torchrun
    #NOTE: don't also set DEBUGPY=1 -> inversion_opti.py then calls debugpy.listen at import time, on every rank
    # if local_rank == 0:
    #     debugpy.listen(("127.0.0.1", 5678))
    #     print("[rank 0] waiting for debugger to attach...")
    #     debugpy.wait_for_client()

    #get the normal dataset
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)

    #get the filtered dataset, for train set which has gene burst masks
    #(the factors come from the images + burst masks, not the w's, so they're computed before the inversion.
    #that way every saved file already carries its labels, and a crashed run can just resume)
    subset, factors, burst_idx = get_filtered_data(dataset)

    #look at the range of every factor, and binarize at the median: above -> 1 (high), at/below -> 0 (low)
    #area/aspect are logged in get_burst_factors, the median split is the same either way as log is monotonic
    #(every rank computes these itself, it's deterministic so they all get the same thresholds/labels)
    thresholds = np.median(factors, axis=0)
    labels = (factors > thresholds).astype(np.int64)    #[N, 3]
    if is_main:
        print("\nfactor    |   min   | median  |   max   | labeled high")
        for k, name in enumerate(BURST_FACTORS):
            f = factors[:, k]
            print(f"{name:9s} | {f.min():7.3f} | {thresholds[k]:7.3f} | {f.max():7.3f} | {labels[:, k].sum()}/{len(f)}")

    #frames already saved by a previous run are skipped, the rest is split between the ranks (every world_size-th frame)
    #all ranks see the same disk state here, so the split has no overlaps/gaps. stop any older run before launching
    bases = [dataset.chA_files[i].name[:-len("_chA.png")] for i in burst_idx]
    pending = [j for j, base in enumerate(bases) if not (Path(SAVE_DIR) / f"{base}.npz").exists()]
    rank_pending = pending[local_rank::world_size]
    if is_main:
        print(f"\n{len(bases) - len(pending)}/{len(bases)} burst frames already saved, {len(pending)} left to invert "
              f"across {world_size} gpu(s), ~{len(rank_pending)} each")

    #load the frozen generator & discriminator (D is needed for the feature matching loss)
    with open(PKL, 'rb') as f:
        network_dict = legacy.load_network_pkl(f)
    G = network_dict['G_ema'].to(DEVICE).eval().requires_grad_(False)
    D = network_dict['D'].to(DEVICE).eval().requires_grad_(False)
    w_avg, w_std = compute_w_stats(G, DEVICE)
    
    '''
    below the inversion is run: 
    '''
    #run through inversion_opti, batched to get the inverted W space vector.
    #progress bar only on rank 0 (it shows rank 0's share, the other rank runs at about the same pace)
    for start in tqdm(range(0, len(rank_pending), BATCH), desc="inverting burst frames (rank 0)", disable=not is_main):
        batch = rank_pending[start:start + BATCH]
        targets = torch.from_numpy(np.stack([
            load_target(dataset.chA_files[burst_idx[j]], dataset.chB_files[burst_idx[j]]) for j in batch
        ]))
        #D's minibatch-std layer (D.b4.mbstd, group_size=4) reshapes the batch into groups of 4, so a tail batch
        #like 37 would crash. pad up to a multiple of 4 by repeating the last frame, the extra w's are dropped below
        #(the loss only uses the hooked blocks before b4, so the padded frames don't affect the real ones)
        n_pad = -len(batch) % 4
        if n_pad:
            targets = torch.cat([targets, targets[-1:].repeat(n_pad, 1, 1, 1)], dim=0)
        '''
        project_batch() is the inversion run in batched way. "targets" are the images(>1) sent to be inverted. 
        '''
        ws = project_batch(G, D, targets, w_avg, w_std, device=DEVICE)[:len(batch)]   #[B, num_ws, 512], all rows identical (shared W)

        #save as a file that contains both the w space vector and the meta data (labels for each attribute)
        #w is stored as the single shared [512] vector, repeat it to [num_ws, 512] before G.synthesis
        for j, w in zip(batch, ws[:, 0].cpu().numpy()):
            np.savez(
                Path(SAVE_DIR) / f"{bases[j]}.npz",
                w=w,                                     #[512]
                base=bases[j],
                chA_path=str(dataset.chA_files[burst_idx[j]]),
                chB_path=str(dataset.chB_files[burst_idx[j]]),
                factor_names=np.array(BURST_FACTORS),
                factors=factors[j],                      #[3] raw values (intensity, log area, log aspect)
                labels=labels[j],                        #[3] 0 = at/below median, 1 = above
                thresholds=thresholds,                   #[3] the medians used for the split
            )

    cleanup_DDP()
    if is_main:
        n_saved = sum((Path(SAVE_DIR) / f"{base}.npz").exists() for base in bases)
        print(f"\n{n_saved}/{len(bases)} w's + labels saved in {SAVE_DIR}/")


#NOTE: (after these are saved, another script should now train an SVM< to find the directions between hyperplane regions- do later..)
