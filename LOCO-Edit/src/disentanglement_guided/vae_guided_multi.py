'''
supervised guided-VAE (Ding et al., CVPR 2020, "guided variational autoencoder for disentanglement learning")
on the 2-channel cell crops, with several gene burst factors, each tied to its own latent code:
    z0: burst intensity (mean chA under the hand-drawn burst mask)
    z1: burst area      (log px, at the 128 px model resolution)
    z2: burst aspect    (log major/minor axis)
see get_burst_factors in gan_dci.py for how each is measured

the multi-attribute version of vae_guided.py (which is the single-factor, intensity only case). per factor k with code d_k:
    excitation: a linear head on z[:, d_k] has to predict factor k -> pulls factor k INTO z_dk
    inhibition: an mlp adversary on every other code (z without d_k) tries to predict factor k, the encoder is
                trained to defeat it -> pushes factor k OUT of every other code, including the other target codes
    loss = recon + β·KL + Σ_k (EXC_WEIGHT·exc_mse_k - INH_WEIGHT·inh_mse_k)

because the adversary for k also sees the other target codes, factors that are correlated in the data fight each other.
these three were picked for being close to independent (spearman: intensity-area 0.27, intensity-aspect 0.15,
area-aspect -0.10). area, major and minor axis would all be ~0.9 correlated, ie one "size" factor

only the train frames with an annotated gene burst are used

run from src/:
    python disentanglement_guided/vae_guided_multi.py
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
from disentanglement_semantic_eval.gan_dci import get_burst_factors, load_burst_mask, BURST_FACTORS, ASPECT_MAX

import debugpy
debugpy.listen(("127.0.0.1", 5678)) #127.0.0.1, cz only local host can talk to the port
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
print("Debugger attached! Running code...")

CHA_DIR  = "data/singlecell_chA_split/train"
CHB_DIR  = "data/singlecell_chB_split/train"
MASK_DIR = "data/singlecell_mask"

DEVICE      = torch.device("cuda:9")
BATCH_SIZE  = 64
NUM_EPOCHS  = 100
LR          = 3e-4
BETA        = 1
LATENT_DIM  = 16

#factor -> the latent code tied to it. the names have to be columns of get_burst_factors (BURST_FACTORS)
TARGETS = {"intensity": 0, "area": 1, "aspect": 2}
FACTORS = list(TARGETS)
TARGET_DIMS = [TARGETS[f] for f in FACTORS]
assert all(f in BURST_FACTORS for f in FACTORS), f"factors have to be in {BURST_FACTORS}"
assert len(set(TARGET_DIMS)) == len(TARGET_DIMS), "each factor needs its own code"

'''
actual loss is: recon + β·KL + Σ_k (EXC_WEIGHT·exc_mse_k - INH_WEIGHT·inh_mse_k)

#NOTE: the weights are PER factor, same as the single-factor run (so the intensity term is weighted the same).
with 3 factors the guidance terms together can reach ~3x that, against recon ~240. if recon gets visibly worse,
lower them (eg 50 / len(FACTORS))
'''
EXC_WEIGHT = 50.0
INH_WEIGHT = 50.0
ADV_LR     = 3e-4

#None -> train from scratch like the baseline, or a path to start from the baseline weights
#eg "vae_checkpoints_masked/beta_vae_beta=1_16/vae_epoch99.pt"
INIT_CKPT = None

SAVE_DIR = f"vae_checkpoints_masked/guided_vae_beta={BETA}_{LATENT_DIM}_" + "_".join(f"{f}-z{d}" for f, d in TARGETS.items())
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/recon_previews", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/traversal_previews", exist_ok=True)


class burstLabelDataset(torch.utils.data.Dataset):
    #wraps the burst subset so each item also carries its standardized factors [len(FACTORS)]
    def __init__(self, subset, labels):
        self.subset = subset
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, mask = self.subset[i]
        return x, mask, self.labels[i]


#labels: the scan over all train frames takes a while, so cache it next to the checkpoints.
#the cache records the channel + how the factors were measured, a cache that doesn't match gets recomputed
BURST_CH = "chA"  #gene burst channel
dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)
label_cache = f"{SAVE_DIR}/burst_labels.npz"
cached = np.load(label_cache) if os.path.exists(label_cache) else None
if (cached is not None and "channel" in cached and str(cached["channel"]) == BURST_CH
        and "factor_names" in cached and list(cached["factor_names"]) == BURST_FACTORS
        and float(cached["aspect_max"]) == ASPECT_MAX):
    all_factors, burst_idx = cached["factors"], cached["burst_idx"]
    subset = torch.utils.data.Subset(dataset, burst_idx)
else:
    subset, all_factors, burst_idx = get_burst_factors(dataset)
    np.savez(label_cache, factors=all_factors, burst_idx=burst_idx, factor_names=np.array(BURST_FACTORS),
             channel=BURST_CH, aspect_max=ASPECT_MAX)

#keep only the factors this run uses, in FACTORS order
factors = all_factors[:, [BURST_FACTORS.index(f) for f in FACTORS]]   #[num of files, num of factors = 3 in this case]


'''
NOTE:
1. (mean)intensity is avg of pixel intensiyties i nthe gene burst mask. no log.
2. area = num of pixels in the hand drawn mask - area = raw_px * (128/H) * (128/W). THis is then taken log of.
3. aspect ratio: [skimage.measure.regionprops] gives major and minor axis. divide those both to get the aspect ratio, then take natural log
'''

#standardize each factor on its own (as in the dci scripts), keep the stats to map the heads back to real values later
#(area/aspect are in log units, so real = exp(pred * y_std + y_mean) for those two)
y_mean, y_std = factors.mean(0), factors.std(0) + 1e-8
labels = (factors - y_mean) / y_std #this is to standardize wts and not make one term less important thatn the other due to different range of values

# print("factor spearman correlations (the inhibition terms fight over anything correlated):")
# ranks = labels.argsort(0).argsort(0)
# print(np.array2string(np.corrcoef(ranks, rowvar=False), precision=2), FACTORS)

train_set = burstLabelDataset(subset, labels)
loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
print(f"training on {len(train_set)} burst frames, {len(loader)} steps/epoch")

#fixed preview cells, the same ones every epoch so the recon/traversal previews compare cleanly across epochs.
#picked evenly across the sorted burst intensities (dimmest burst first), so the rows also span the intensity range
N_PREVIEW = 3
preview_idx = np.argsort(labels[:, 0])[np.linspace(0, len(labels) - 1, N_PREVIEW).astype(int)]
x_preview = torch.stack([train_set[i][0] for i in preview_idx]).to(DEVICE)

#burst masks of the preview cells, burst_idx maps a train_set index back to the full dataset
#resized like the model input (nearest, so it stays binary), the raw crops are 93-173 px
burst_preview = torch.stack([load_burst_mask(dataset.chB_files[burst_idx[i]], dataset.mask_transform)
                             for i in preview_idx]).to(DEVICE)

model = twoChannelVAE(latent_dim=LATENT_DIM).to(DEVICE)
if INIT_CKPT is not None:
    model.load_state_dict(torch.load(INIT_CKPT, map_location=DEVICE))

#excitation heads w_k: z_dk -> factor k. linear on purpose, so moving along z_dk moves the factor monotonically
#NOTE: num of heads equals the num of factors we're trying to control..
exc_heads = nn.ModuleList([nn.Linear(1, 1) for _ in FACTORS]).to(DEVICE)

#inhibition adversaries C_k: z without d_k -> factor k. mlps, so they can also catch nonlinear leakage into the other codes.
#one per factor, they can't share weights since each sees a different set of codes and predicts a different factor
def make_adversary():
    return nn.Sequential(
        nn.Linear(LATENT_DIM - 1, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 1),
    )
    
#NOTE: 3 adversaries here too..
adversaries = nn.ModuleList([make_adversary() for _ in FACTORS]).to(DEVICE)

#the encoder/decoder and the excitation heads share an optimizer, the adversaries get their own (they play the other side).
#one optimizer for all adversaries is fine, their parameters don't overlap so summing their losses keeps them independent
optimizer = torch.optim.AdamW(list(model.parameters()) + list(exc_heads.parameters()), lr=LR)
adv_optimizer = torch.optim.AdamW(adversaries.parameters(), lr=ADV_LR)


def split_z(z, dim):
    #z_t is the code at dim, z_rst is the other LATENT_DIM - 1 codes
    z_t = z[:, dim:dim + 1]
    z_rst = torch.cat([z[:, :dim], z[:, dim + 1:]], dim=1)
    return z_t, z_rst


log_file = open(f"{SAVE_DIR}/loss_log.csv", "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["global_step", "epoch", "total_loss", "recon_loss", "kl_loss"]
                    + [f"{kind}_{f}" for f in FACTORS for kind in ("exc_mse", "inh_mse", "adv_mse")])

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
    #each target code should change only its own factor along each row, the free codes should leave the burst alone
    #saved as traversal_previews/epoch{epoch}/z{dim}/{chA,chB}.png, target code folders are tagged with their factor ("z1_area")
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

    dim_tag = {d: f for f, d in TARGETS.items()}
    for dim in range(LATENT_DIM):
        z = mu[:n_cells].unsqueeze(1).repeat(1, n_steps, 1)           #[n_cells, n_steps, LATENT_DIM]
        z[:, :, dim] = alphas.view(1, n_steps)
        imgs = model.decode(z.view(-1, LATENT_DIM))
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)

        dim_dir = f"{SAVE_DIR}/traversal_previews/epoch{epoch}/z{dim}" + (f"_{dim_tag[dim]}" if dim in dim_tag else "")
        os.makedirs(dim_dir, exist_ok=True)
        for ch, name in [(0, "chA"), (1, "chB")]:
            trav = imgs[:, ch:ch+1].reshape(n_cells, n_steps, 1, *imgs.shape[-2:]).repeat(1, 1, 3, 1, 1)
            rows = torch.cat([ref_cols[ch], trav], dim=1)             #[n_cells, 3 + n_steps, 3, H, W]
            save_image(make_grid(rows.flatten(0, 1), nrow=3 + n_steps), f"{dim_dir}/{name}.png")

def heads_state():
    #everything needed to map the heads back to real factor values
    return {"factors": FACTORS, "target_dims": TARGET_DIMS,
            "exc_heads": exc_heads.state_dict(), "adversaries": adversaries.state_dict(),
            "y_mean": y_mean.tolist(), "y_std": y_std.tolist()}


global_step = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    for step, (x, mask, y) in enumerate(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)                 #[B, K]

        #encode/sample/decode by hand (instead of model forward) so the sampled z is available for the heads
        mu, logvar = model.encode(x)
        z = model.reparametrize(mu, logvar)
        x_recon = model.decode(z)

        #1. adversary step: best possible regressor of each factor from its z_rst (the max over C_t in eq. 7).
        #z is detached, so this step only updates the adversaries
        z_d = z.detach()
        adv_mses = [F.mse_loss(adversaries[k](split_z(z_d, d)[1]), y[:, k:k+1]) for k, d in enumerate(TARGET_DIMS)]
        adv_optimizer.zero_grad()
        sum(adv_mses).backward()
        adv_optimizer.step()

        #2. vae step: elbo + Σ_k (excitation - inhibition) (eq. 8)
        total_loss, recon_loss, kl_loss = model.loss(x, x_recon, mu, logvar, beta=BETA, l1_weight=0.0)
        exc_mses, inh_mses = [], []
        for k, d in enumerate(TARGET_DIMS):
            z_t, z_rst = split_z(z, d)
            exc_mses.append(F.mse_loss(exc_heads[k](z_t), y[:, k:k+1]))
            inh_mses.append(F.mse_loss(adversaries[k](z_rst), y[:, k:k+1]))

        #the encoder wants each adversary to fail, ie maximize its mse. on standardized labels, mse = 1 is exactly
        #"no better than predicting the mean" = z_rst carries no info on that factor, so it is capped there,
        #pushing past it would only make the encoder chase the adversary's blind spots
        guide = sum(EXC_WEIGHT * e - INH_WEIGHT * torch.clamp(i, max=1.0) for e, i in zip(exc_mses, inh_mses))
        loss = total_loss + guide

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        per_factor = [m.item() for k in range(len(FACTORS)) for m in (exc_mses[k], inh_mses[k], adv_mses[k])]
        log_writer.writerow([global_step, epoch, loss.item(), recon_loss.item(), kl_loss.item()] + per_factor)
        global_step += 1

        if step % 20 == 0:
            fmt = lambda ms: "/".join(f"{m.item():.3f}" for m in ms)
            print(f"epoch {epoch} | step {step} | total {loss.item():.2f} | recon {recon_loss.item():.2f} | "
                  f"kl {kl_loss.item():.2f} | exc {fmt(exc_mses)} | inh {fmt(inh_mses)} | adv {fmt(adv_mses)} "
                  f"({'/'.join(FACTORS)})", flush=True)

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
        torch.save(heads_state(), f"{SAVE_DIR}/heads_epoch{epoch}.pt")

log_file.close()
torch.save(model.state_dict(), f"{SAVE_DIR}/vae_final.pt")
torch.save(heads_state(), f"{SAVE_DIR}/heads_final.pt")
