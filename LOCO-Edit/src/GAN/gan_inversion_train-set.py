import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

# 1. Setup repository path before importing legacy
REPO_DIR = "/scratch/pbk5339/summer/LOCO-Edit/src/GAN/stylegan2-ada-pytorch"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import legacy
import dnnlib
import importlib.util

# Load inversion functions dynamically
file_path = Path(REPO_DIR) / "inversion_opti.py"
spec = importlib.util.spec_from_file_location("inversion_opti", file_path)
inversion_opti = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inversion_opti)

project_batch = inversion_opti.project_batch
compute_w_stats = inversion_opti.compute_w_stats

PKL = "GAN/stylegan2-ada-pytorch/training-runs/00010-data_npy-auto1-kimg1000-ada-bg/network-snapshot-001000.pkl"
SAVE_DIR = "GAN/gan_inverted_ws_train-set"
DATA_DIR = "GAN/data_npy_masked_new"
BATCH = 64
SEED = 0

def setup_DDP():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_DDP():
    dist.destroy_process_group()

if __name__ == "__main__":
    local_rank = setup_DDP()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(SEED + local_rank)

    # Only rank 0 creates the directory
    if local_rank == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
    dist.barrier()  # Synchronize so all workers see the directory

    # Load frozen generator & discriminator per rank
    with open(PKL, 'rb') as f:
        network_dict = legacy.load_network_pkl(f)
        G = network_dict['G_ema'].to(device).eval()
        D = network_dict['D'].to(device).eval()
    G.requires_grad_(False)
    D.requires_grad_(False)

    w_avg, w_std = compute_w_stats(G, device)

    # 1. Deterministic sort across all ranks
    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".npy")])

    # 2. Check for previously saved files
    pending_files = [
        f for f in all_files 
        if not os.path.exists(os.path.join(SAVE_DIR, f))
    ]
    skipped_count = len(all_files) - len(pending_files)

    # 3. Print resume status strictly on rank 0
    if local_rank == 0:
        print(f"[Rank 0] Total files: {len(all_files)}")
        print(f"[Rank 0] Skipped already processed: {skipped_count}")
        print(f"[Rank 0] Remaining to process across {world_size} GPUs: {len(pending_files)}")

    # 4. Partition remaining files evenly across GPU ranks
    rank_files = pending_files[local_rank::world_size]

    twoch_data_batch = []
    batch_filenames = []

    # Display outer progress bar only on rank 0
    iterator = tqdm(rank_files, desc="Processing files (Rank 0)", leave=True) if local_rank == 0 else rank_files

    for f in iterator:
        file_path = os.path.join(DATA_DIR, f)
        array_data = np.load(file_path)
        twoch_data_batch.append(torch.from_numpy(array_data))
        batch_filenames.append(f)

        if len(twoch_data_batch) == BATCH:
            batch_tensor = torch.stack(twoch_data_batch).to(device)
            # Pass verbose=(local_rank == 0) to silence inner tqdm on other ranks
            w_batch = project_batch(
                G, D, batch_tensor, w_avg, w_std, device=device, verbose=(local_rank == 0)
            )

            for fname, w_vec in zip(batch_filenames, w_batch.cpu().numpy()):
                np.save(os.path.join(SAVE_DIR, fname), w_vec)

            twoch_data_batch.clear()
            batch_filenames.clear()

    # Process remaining tail batch on this rank
    if len(twoch_data_batch) > 0:
        batch_tensor = torch.stack(twoch_data_batch).to(device)
        w_batch = project_batch(
            G, D, batch_tensor, w_avg, w_std, device=device, verbose=(local_rank == 0)
        )

        for fname, w_vec in zip(batch_filenames, w_batch.cpu().numpy()):
            np.save(os.path.join(SAVE_DIR, fname), w_vec)

    dist.barrier()  # Synchronize all GPUs before shutting down
    cleanup_DDP()