'''
supervised guided-VAE (Ding et al., CVPR 2020, "guided variational autoencoder for disentanglement learning")
on the 2-channel cell crops, tying gene burst intensity to the 1st latent code

same config as the beta_vae_beta=1_16 baseline (beta=1, 16 latents, batch 64, lr 3e-4, adamw, plain mse recon),
plus the paper's excitation + inhibition terms:
    z = (z_t, z_rst), z_t = z[:, 0] (burst intensity), z_rst = z[:, 1:] (everything else)
    excitation: a head on z_t has to predict the burst intensity -> pulls intensity INTO z_t
    inhibition: an adversary on z_rst tries to predict the burst intensity, the encoder is trained to
                defeat it -> pushes intensity OUT of the other 15 codes

the paper uses binary attributes + classifiers (log-likelihood), burst intensity is continuous so both
heads are regressors and the log-likelihood becomes a (negative) mse on the standardized intensity

only the train frames with an annotated gene burst are used (see get_gene_burst_intensity in gan_dci.py),
the burst intensity is the mean chA (the gene burst channel) under the hand-drawn burst mask

run from src/:
    python disentanglement_guided/vae_guided.py
'''
import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

#run from src/, so the VAE_disent and disentanglement_semantic_eval packages resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VAE_disent.data_utils import twoChannelDataset
from VAE_disent.model import twoChannelVAE
from disentanglement_semantic_eval.gan_dci import get_gene_burst_intensity, load_burst_mask

import debugpy
# debugpy.listen(("127.0.0.1", 5678)) #127.0.0.1, cz only local host can talk to the port
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")

CHA_DIR  = "data/singlecell_chA_split/train"
CHB_DIR  = "data/singlecell_chB_split/train"
MASK_DIR = "data/singlecell_mask"

DEVICE      = torch.device("cuda:9")
BATCH_SIZE  = 64
NUM_EPOCHS  = 100
LR          = 3e-4
BETA        = 1
LATENT_DIM  = 16
TARGET_DIM  = 0           #latent code tied to the gene burst intensity

#model.loss returns recon/kl summed over pixels per image (~240 recon, ~49 kl at the end of the baseline),
#while the excitation/inhibition mse is on the standardized intensity (~1 when uninformative),
#so the weights put them on the same footing as the kl term

'''
actual loss is: recon + β·KL + EXC_WEIGHT·exc_mse - INH_WEIGHT·inh_mse

#NOTE: we are "minimizing" the exc_mse but are "maximizing" the error of the inhibitor
'''
EXC_WEIGHT = 50.0
INH_WEIGHT = 50.0
ADV_LR     = 3e-4

#None -> train from scratch like the baseline, or a path to start from the baseline weights
#eg "vae_checkpoints_masked/beta_vae_beta=1_16/vae_epoch99.pt"
INIT_CKPT = None

SAVE_DIR = f"vae_checkpoints_masked/guided_vae_beta={BETA}_{LATENT_DIM}_burst_z{TARGET_DIM}"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/recon_previews", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/traversal_previews", exist_ok=True)


class burstLabelDataset(torch.utils.data.Dataset):
    #wraps the burst subset so each item also carries its standardized burst intensity
    def __init__(self, subset, labels):
        self.subset = subset
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, mask = self.subset[i]
        return x, mask, self.labels[i]


#labels: the scan over all train frames takes a while, so cache it next to the checkpoints.
#the cache records which channel the intensity came from, older caches (no "channel" key) may be chB, so they get recomputed
BURST_CH = "chA"  #gene burst channel
dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)
label_cache = f"{SAVE_DIR}/burst_labels.npz"
cached = np.load(label_cache) if os.path.exists(label_cache) else None
if cached is not None and "channel" in cached and str(cached["channel"]) == BURST_CH:
    burst_int, burst_idx = cached["burst_int"], cached["burst_idx"]
    subset = torch.utils.data.Subset(dataset, burst_idx)
else:
    subset, burst_int, burst_idx = get_gene_burst_intensity(dataset) #mean chA under the burst mask

#standardize the intensity (as in the dci scripts), keep the stats to map z_t back to real intensity later
y_mean, y_std = float(burst_int.mean()), float(burst_int.std() + 1e-8)
np.savez(label_cache, burst_int=burst_int, burst_idx=burst_idx, y_mean=y_mean, y_std=y_std, channel=BURST_CH)
labels = (burst_int - y_mean) / y_std

train_set = burstLabelDataset(subset, labels)
loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
print(f"training on {len(train_set)} burst frames, {len(loader)} steps/epoch")

#fixed preview cells, the same ones every epoch so the recon/traversal previews compare cleanly across epochs.
#picked evenly across the sorted burst intensities (dimmest burst first), so the rows also span the intensity range
N_PREVIEW = 5
preview_idx = np.argsort(labels)[np.linspace(0, len(labels) - 1, N_PREVIEW).astype(int)]
x_preview = torch.stack([train_set[i][0] for i in preview_idx]).to(DEVICE)


#burst masks of the preview cells, burst_idx maps a train_set index back to the full dataset
#resized like the model input (nearest, so it stays binary), the raw crops are ~131-133 px
burst_preview = torch.stack([load_burst_mask(dataset.chB_files[burst_idx[i]], dataset.mask_transform)
                             for i in preview_idx]).to(DEVICE)

model = twoChannelVAE(latent_dim=LATENT_DIM).to(DEVICE)
if INIT_CKPT is not None:
    model.load_state_dict(torch.load(INIT_CKPT, map_location=DEVICE))

#excitation head w_t: z_t -> intensity. linear on purpose, so moving along z_t moves the intensity monotonically
exc_head = nn.Linear(1, 1).to(DEVICE)

#inhibition adversary C_t: z_rst -> intensity. an mlp, so it can also catch nonlinear leakage into the other codes
adversary = nn.Sequential(
    nn.Linear(LATENT_DIM - 1, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 1),
).to(DEVICE)

#the encoder/decoder and the excitation head share an optimizer, the adversary gets its own (it plays the other side)
optimizer = torch.optim.AdamW(list(model.parameters()) + list(exc_head.parameters()), lr=LR)
adv_optimizer = torch.optim.AdamW(adversary.parameters(), lr=ADV_LR)


def split_z(z):
    #z_t is the target code, z_rst is the other LATENT_DIM - 1 codes
    z_t = z[:, TARGET_DIM:TARGET_DIM + 1]
    z_rst = torch.cat([z[:, :TARGET_DIM], z[:, TARGET_DIM + 1:]], dim=1)
    return z_t, z_rst


log_file = open(f"{SAVE_DIR}/loss_log.csv", "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["global_step", "epoch", "total_loss", "recon_loss", "kl_loss", "exc_mse", "inh_mse", "adv_mse"])

def save_recon_preview(x, x_recon, epoch):
    #un-normalize from [-1,1] back to [0,1] for viewing
    x       = (x[:N_PREVIEW]       * 0.5 + 0.5).clamp(0, 1)
    x_recon = (x_recon[:N_PREVIEW] * 0.5 + 0.5).clamp(0, 1)
    for ch, name in [(0, "chA"), (1, "chB")]:
        pairs = torch.cat([x[:, ch:ch+1], x_recon[:, ch:ch+1]], dim=0)  #inputs then recons
        grid = make_grid(pairs, nrow=N_PREVIEW)
        save_image(grid, f"{SAVE_DIR}/recon_previews/epoch{epoch}_{name}.png")

def save_traversal_preview(mu, x, burst, epoch, n_cells=N_PREVIEW, n_steps=7, span=3.0):
    #walk one code at a time from -span to +span (posterior std units), other codes fixed.
    #z_t should brighten the burst along each row, the other codes should leave the burst alone
    #saved as traversal_previews/epoch{epoch}/z{dim}/{chA,chB}.png, the target code's folder is tagged "_target"
    #rows (chA and chB): input | burst mask | burst mask overlaid on the input (red) | traversal steps
    alphas = torch.linspace(-span, span, n_steps, device=mu.device)

    #the 3 reference columns don't depend on the code being walked, so build them once per channel (rgb, for the red overlay)
    burst = burst[:n_cells]
    ref_cols = {}
    for ch in (0, 1):
        x_in = (x[:n_cells, ch:ch+1] * 0.5 + 0.5).clamp(0, 1)      #[n_cells, 1, H, W]
        dimmed = x_in * (1 - 0.5 * burst)                          #50% red blend inside the burst
        overlay = torch.cat([dimmed + 0.5 * burst, dimmed, dimmed], dim=1)
        ref_cols[ch] = torch.stack([x_in.repeat(1, 3, 1, 1), burst.repeat(1, 3, 1, 1), overlay], dim=1)  #[n_cells, 3, 3, H, W]

    for dim in range(LATENT_DIM):
        z = mu[:n_cells].unsqueeze(1).repeat(1, n_steps, 1)           #[n_cells, n_steps, LATENT_DIM]
        z[:, :, dim] = alphas.view(1, n_steps)
        imgs = model.decode(z.view(-1, LATENT_DIM))
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)

        dim_dir = f"{SAVE_DIR}/traversal_previews/epoch{epoch}/z{dim}" + ("_target" if dim == TARGET_DIM else "")
        os.makedirs(dim_dir, exist_ok=True)
        for ch, name in [(0, "chA"), (1, "chB")]:
            trav = imgs[:, ch:ch+1].reshape(n_cells, n_steps, 1, *imgs.shape[-2:]).repeat(1, 1, 3, 1, 1)
            rows = torch.cat([ref_cols[ch], trav], dim=1)             #[n_cells, 3 + n_steps, 3, H, W]
            save_image(make_grid(rows.flatten(0, 1), nrow=3 + n_steps), f"{dim_dir}/{name}.png")


global_step = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    for step, (x, mask, y) in enumerate(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE).unsqueeze(1)    #[B, 1]

        #encode/sample/decode by hand (instead of model forward) so the sampled z is available for the heads
        mu, logvar = model.encode(x)
        z = model.reparametrize(mu, logvar)
        x_recon = model.decode(z)
        z_t, z_rst = split_z(z)

        #1. adversary step: best possible regressor of intensity from z_rst (the max over C_t in eq. 7).
        #z_rst is detached, so this step only updates the adversary
        adv_mse = F.mse_loss(adversary(z_rst.detach()), y)
        adv_optimizer.zero_grad()
        adv_mse.backward()
        adv_optimizer.step()

        #2. vae step: elbo + excitation - inhibition (eq. 8)
        total_loss, recon_loss, kl_loss = model.loss(x, x_recon, mu, logvar, beta=BETA, l1_weight=0.0)
        exc_mse = F.mse_loss(exc_head(z_t), y)
        inh_mse = F.mse_loss(adversary(z_rst), y)

        #the encoder wants the adversary to fail, ie maximize its mse. on standardized labels, mse = 1 is exactly
        #"no better than predicting the mean" = z_rst carries no intensity info, so it is capped there,
        #pushing past it would only make the encoder chase the adversary's blind spots
        loss = total_loss + EXC_WEIGHT * exc_mse - INH_WEIGHT * torch.clamp(inh_mse, max=1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logged = [loss.item(), recon_loss.item(), kl_loss.item(), exc_mse.item(), inh_mse.item(), adv_mse.item()]
        log_writer.writerow([global_step, epoch] + logged)
        global_step += 1

        if step % 20 == 0:
            print(f"epoch {epoch} | step {step} | total {logged[0]:.2f} | recon {logged[1]:.2f} | kl {logged[2]:.2f} | "
                  f"exc {logged[3]:.3f} | inh {logged[4]:.3f} | adv {logged[5]:.3f}", flush=True)

    log_file.flush()

    if epoch == 0 or (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            mu_check, _ = model.encode(x_preview)
            x_recon_check = model.decode(mu_check)

            save_recon_preview(x_preview, x_recon_check, epoch)
            save_traversal_preview(mu_check, x_preview, burst_preview, epoch)
        #vae saved as a plain state_dict like the baseline, so disent_quant_VAE.py etc. load it unchanged
        torch.save(model.state_dict(), f"{SAVE_DIR}/vae_epoch{epoch}.pt")
        torch.save({"exc_head": exc_head.state_dict(), "adversary": adversary.state_dict(),
                    "y_mean": y_mean, "y_std": y_std}, f"{SAVE_DIR}/heads_epoch{epoch}.pt")

log_file.close()
torch.save(model.state_dict(), f"{SAVE_DIR}/vae_final.pt")
torch.save({"exc_head": exc_head.state_dict(), "adversary": adversary.state_dict(),
            "y_mean": y_mean, "y_std": y_std}, f"{SAVE_DIR}/heads_final.pt")
