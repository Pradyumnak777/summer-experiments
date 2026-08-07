'''
training a beta-VAE on 2-channel (chA, chB) microscopy crops
'''
import os
import csv
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torch.utils.data import Subset
import random

from data_utils import twoChannelDataset
from model import twoChannelVAE

CHA_DIR = "data/singlecell_chA_split/train"
CHB_DIR = "data/singlecell_chB_split/train"

DEVICE      = torch.device("cuda:9")
BATCH_SIZE  = 128
NUM_EPOCHS  = 100
LR          = 3e-4
BETA        = 0
LATENT_DIM  = 16

SAVE_DIR = f"vae_autoenc_checkpoints_masked_2/beta_vae_beta={BETA}_{LATENT_DIM}"
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


loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

model = twoChannelVAE(latent_dim=LATENT_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

log_file = open(f"{SAVE_DIR}/loss_log.csv", "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["global_step", "epoch", "total_loss", "recon_loss", "kl_loss"])

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
    model.train()
    for step, (x,mask) in enumerate(loader):
        x = x.to(DEVICE)

        x_recon, mu, logvar = model(x)
        mask = mask.to(DEVICE)
        total_loss, recon_loss, kl_loss = model.loss(x, x_recon, mu, logvar, beta=BETA, l1_weight=0.6)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        log_writer.writerow([global_step, epoch, total_loss.item(), recon_loss.item(), kl_loss.item()])
        global_step += 1

        if step % 20 == 0:
            print(f"epoch {epoch} | step {step} | total {total_loss.item():.2f} | "
                  f"recon {recon_loss.item():.2f} | kl {kl_loss.item():.2f}")

    log_file.flush()

    if epoch == 0 or (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            x_check, _ = next(iter(loader))
            x_check = x_check.to(DEVICE)

            mu_check, _ = model.encode(x_check)
            x_recon_check = model.decode(mu_check)
            
        save_recon_preview(x_check, x_recon_check, epoch)
        torch.save(model.state_dict(), f"{SAVE_DIR}/vae_epoch{epoch}.pt")

log_file.close()
torch.save(model.state_dict(), f"{SAVE_DIR}/vae_final.pt")
