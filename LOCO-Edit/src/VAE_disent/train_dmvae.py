import os, csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_utils import twoChannelDataset
from model_dmvae import DMVAE, kl_standard_normal

DEVICE     = torch.device("cuda:9")   # free one
CHA_DIR    = "data/singlecell_chA_split/train"
CHB_DIR    = "data/singlecell_chB_split/train"
SAVE_DIR   = "vae_checkpoints/dmvae_p8_s8"
PRIV_DIM   = 8
SHARED_DIM = 8
BATCH_SIZE = 64
NUM_EPOCHS = 100
LR         = 1e-4
BETA       = 1.0    # KL weight; >1 to push disentanglement (was 1 = plain VAE before!)
os.makedirs(SAVE_DIR, exist_ok=True)

dataset = twoChannelDataset(CHA_DIR, CHB_DIR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=4, drop_last=True)

model = DMVAE(PRIV_DIM, SHARED_DIM).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR)

log = csv.writer(open(f"{SAVE_DIR}/loss_log.csv", "w", newline=""))
log.writerow(["step", "epoch", "total", "rec_A", "rec_B", "kl"])
step = 0

for epoch in range(NUM_EPOCHS):
    for x in loader:
        x  = x.to(DEVICE)                 # [B, 2, 128, 128]
        xA = x[:, 0:1]                    # keep channel dim -> [B, 1, 128, 128]
        xB = x[:, 1:2]

        out = model(xA, xB)

        B = xA.shape[0]
        rec_A = F.mse_loss(out["recon_A"], xA, reduction='sum') / B
        rec_B = F.mse_loss(out["recon_B"], xB, reduction='sum') / B

        kl  = kl_standard_normal(*out["priv_A"])
        kl += kl_standard_normal(*out["priv_B"])
        kl += kl_standard_normal(*out["shared"])

        loss = rec_A + rec_B + BETA * kl
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"ep {epoch} step {step} | tot {loss.item():.4f} "
                  f"| recA {rec_A.item():.4f} recB {rec_B.item():.4f} "
                  f"| kl {kl.item():.4f}")
            log.writerow([step, epoch, loss.item(),
                          rec_A.item(), rec_B.item(), kl.item()])
        step += 1

    if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
        torch.save(model.state_dict(), f"{SAVE_DIR}/dmvae_epoch{epoch}.pt")

print("done")