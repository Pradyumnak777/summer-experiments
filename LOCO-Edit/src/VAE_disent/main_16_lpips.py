'''
training a beta-VAE on 2-channel (chA, chB) microscopy crops
same as the beta_vae_beta=1_16 baseline, plus an LPIPS perceptual term on the recon

multi-GPU (DDP): the global batch is still 64 (16 per GPU x 4), so steps/epoch and the
optimization match the single-GPU baseline. run from src/:
    CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 VAE_disent/main_16_lpips.py
'''
import os
import csv
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image
from torch.utils.data import Subset
import random
import lpips

from data_utils import twoChannelDataset
from model import twoChannelVAE

CHA_DIR = "data/singlecell_chA_split/train"
CHB_DIR = "data/singlecell_chB_split/train"

BATCH_SIZE  = 64          #global batch, split evenly across GPUs
NUM_EPOCHS  = 100
LR          = 3e-4
BETA        = 1
LATENT_DIM  = 16

#MSE/KL from model.loss are summed over pixels per image, LPIPS is a per-image mean.
#so LPIPS is multiplied by the pixel count (2*128*128) -> LPIPS_WEIGHT acts like a per-pixel weight next to MSE
LPIPS_WEIGHT = 0.1
NUM_PIXELS   = 2 * 128 * 128

SAVE_DIR = f"vae_checkpoints_masked/beta_vae_beta={BETA}_{LATENT_DIM}_lpips"

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
world      = dist.get_world_size()
is_main    = dist.get_rank() == 0
torch.cuda.set_device(local_rank)
DEVICE = torch.device("cuda", local_rank)

if is_main:
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/recon_previews", exist_ok=True)



dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir="data/singlecell_mask")

'''
below for subset test
'''
# SUBSET_SIZE = 150
# random.seed(42)
# subset_indices = random.sample(range(len(dataset)), SUBSET_SIZE)
# dataset = Subset(dataset, subset_indices)


sampler = DistributedSampler(dataset, shuffle=True)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE // world, sampler=sampler, num_workers=4, drop_last=True)

model = twoChannelVAE(latent_dim=LATENT_DIM).to(DEVICE)
ddp_model = DDP(model, device_ids=[local_rank])
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

#VGG for training (repo evaluates with net='alex', so this keeps eval less circular). frozen
lpips_fn = lpips.LPIPS(net='vgg', verbose=is_main).to(DEVICE).eval()
for p in lpips_fn.parameters():
    p.requires_grad_(False)

def lpips_per_image(x, x_recon):
    #lpips wants 3ch in [-1,1]; inputs already in [-1,1], so repeat each channel to 3 and average the 2 channels
    per_ch = []
    for c in (0, 1):
        a = x[:, c:c+1].repeat(1, 3, 1, 1)
        b = x_recon[:, c:c+1].repeat(1, 3, 1, 1)
        per_ch.append(lpips_fn(a, b).flatten())  # [B]
    return torch.stack(per_ch, dim=0).mean(dim=0)  # [B]

if is_main:
    log_file = open(f"{SAVE_DIR}/loss_log.csv", "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["global_step", "epoch", "total_loss", "recon_loss", "kl_loss", "lpips"])

def save_recon_preview(x, x_recon, epoch):
    #un-normalize from [-1,1] back to [0,1] for viewing
    x       = (x[:8]       * 0.5 + 0.5).clamp(0, 1)
    x_recon = (x_recon[:8] * 0.5 + 0.5).clamp(0, 1)
    for ch, name in [(0, "chA"), (1, "chB")]:
        pairs = torch.cat([x[:, ch:ch+1], x_recon[:, ch:ch+1]], dim=0)  #inputs then recons
        grid = make_grid(pairs, nrow=8)
        save_image(grid, f"{SAVE_DIR}/recon_previews/epoch{epoch}_{name}.png")

global_step = 0
for epoch in range(NUM_EPOCHS):
    sampler.set_epoch(epoch)
    model.train()
    for step, (x,mask) in enumerate(loader):
        x = x.to(DEVICE)

        x_recon, mu, logvar = ddp_model(x)
        mask = mask.to(DEVICE)
        total_loss, recon_loss, kl_loss = model.loss(x, x_recon, mu, logvar, beta=BETA, l1_weight=0.0)

        #perceptual term, scaled to the summed-per-image footing of MSE/KL
        lpips_val  = lpips_per_image(x, x_recon).mean()
        total_loss = total_loss + LPIPS_WEIGHT * NUM_PIXELS * lpips_val

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        #average the per-GPU losses so the log matches a single 64-image batch
        logged = torch.stack([total_loss, recon_loss, kl_loss, lpips_val]).detach()
        dist.all_reduce(logged)
        logged = (logged / world).tolist()
        if is_main:
            log_writer.writerow([global_step, epoch] + logged)
        global_step += 1

        if is_main and step % 20 == 0:
            print(f"epoch {epoch} | step {step} | total {logged[0]:.2f} | "
                  f"recon {logged[1]:.2f} | kl {logged[2]:.2f} | lpips {logged[3]:.4f}", flush=True)

    if is_main:
        log_file.flush()

    if is_main and (epoch == 0 or (epoch + 1) % 10 == 0):
        model.eval()
        with torch.no_grad():
            x_check, _ = next(iter(loader))
            x_check = x_check.to(DEVICE)

            mu_check, _ = model.encode(x_check)
            x_recon_check = model.decode(mu_check)

        save_recon_preview(x_check, x_recon_check, epoch)
        torch.save(model.state_dict(), f"{SAVE_DIR}/vae_epoch{epoch}.pt")
    dist.barrier()

if is_main:
    log_file.close()
    torch.save(model.state_dict(), f"{SAVE_DIR}/vae_final.pt")
dist.destroy_process_group()
