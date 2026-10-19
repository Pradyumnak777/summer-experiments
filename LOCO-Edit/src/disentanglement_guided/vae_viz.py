'''
just taking a trained model and then visaulizing the the traversal per dimension

same traversal as the training previews in vae_guided.py / vae_guided_multi.py, but for ONE chA/chB pair picked by CELL_IDX
(change it to look at different cells). CELL_IDX indexes the burst subset (get_burst_factors, same frames as
get_gene_burst_intensity), so every index is a frame with a hand-drawn burst mask.
each image is one row: input | burst mask | mask overlaid on the input (red) | traversal steps -SPAN..+SPAN,
for both chA and chB. the walk is centered on the cell's own code value (mu[dim] - SPAN .. mu[dim] + SPAN),
so the middle step is exactly the reconstruction, it gets a green border
(unlike the training previews, which set the code to -SPAN..+SPAN absolute)
every traversal step (recon included) has the burst mask outline drawn on it (red, 1 px just outside the mask,
so the pixels inside the burst stay untouched), to show where to look. for the shape codes (RECON_ONLY_RING, area/aspect)
only the recon gets it, as the burst is meant to change shape along the walk and the original outline would hide that

the target codes' images are drawn with matplotlib instead, with a title per panel. for intensity, each panel is titled
with the mean intensity of that channel inside the burst mask (128 px decoded image, [0, 1]), so you can read off how
the burst brightens along the walk. for area/aspect, the steps get their offset and the input gets the cell's measured value

saved as disentanglement_guided/vae_traversal_viz/<run>/<ckpt>_<set>_idx<CELL_IDX>/z{dim}/{chA,chB}.png
(target code folders are tagged with their factor, eg "z1_area"), plus info.txt with which frame it is and its factor values

run from src/:
    python disentanglement_guided/vae_viz.py
'''
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import make_grid, save_image

#run from src/, so the VAE_disent and disentanglement_semantic_eval packages resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VAE_disent.data_utils import twoChannelDataset
from VAE_disent.model import twoChannelVAE
from disentanglement_semantic_eval.gan_dci import get_burst_factors, load_burst_mask, BURST_FACTORS, ASPECT_MAX

import debugpy
# debugpy.listen(("127.0.0.1", 5678)) #127.0.0.1, cz only local host can talk to the port
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")

SET     = "validation"      #"train" or "validation"
CELL_IDX = 18                 #index into the burst subset of SET (frames with a hand-drawn burst mask)
CHA_DIR  = f"data/singlecell_chA_split/{SET}"
CHB_DIR  = f"data/singlecell_chB_split/{SET}"
MASK_DIR = "data/singlecell_mask"

#must match the training config of the run
#factor -> latent code, like TARGETS in vae_guided_multi.py. for the single-factor vae_guided.py runs: {"intensity": 0}
RUN_DIR    = "vae_checkpoints_masked/guided_vae_beta=1_16_intensity-z0_area-z1_aspect-z2"
LATENT_DIM = 16
TARGETS    = {"intensity": 0, "area": 1, "aspect": 2}

#None -> vae_final.pt if the run finished, else the latest vae_epoch*.pt (changes while the run is still training).
#there is no validation-based "best" checkpoint saved. set a path to pin one, eg when comparing cells
CKPT = None

DEVICE  = torch.device("cuda:9")
N_STEPS = 7                  #odd, so there is a middle step (= the reconstruction)
SPAN    = 3.0                #walk each code from mu - SPAN to mu + SPAN, other codes fixed at the cell's posterior mean
BORDER  = 3                  #px of green border around the reconstruction
RING_RGB = (1.0, 0.0, 0.0)   #color of the burst mask outline on the traversal steps
RECON_ONLY_RING = {"area", "aspect"}   #factor codes whose outline goes on the recon only, the rest get it on every step
assert N_STEPS % 2 == 1, "N_STEPS has to be odd for the middle step to be the reconstruction"

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vae_traversal_viz")


def pick_ckpt(run_dir):
    final = Path(run_dir) / "vae_final.pt"
    if final.exists():
        return final
    epochs = sorted(Path(run_dir).glob("vae_epoch*.pt"), key=lambda p: int(p.stem[len("vae_epoch"):]))
    assert epochs, f"no checkpoints in {run_dir}"
    return epochs[-1]


def ref_columns(x_ch, burst):
    #x_ch: one channel [1, H, W] in [0, 1], burst: [1, H, W] binary
    #-> input | burst mask | mask overlaid on the input (50% red blend), as rgb [3, 3, H, W]
    dimmed = x_ch * (1 - 0.5 * burst)
    overlay = torch.cat([dimmed + 0.5 * burst, dimmed, dimmed], dim=0)
    return torch.stack([x_ch.repeat(3, 1, 1), burst.repeat(3, 1, 1), overlay], dim=0)


def mask_ring(burst):
    #burst: [1, H, W] binary -> [1, H, W] binary outline, the 1 px ring just outside the mask
    #(dilate by a 3x3 max pool, minus the mask), so drawing it never covers a burst pixel
    dilated = F.max_pool2d(burst.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    return dilated - burst


def save_annotated_row(row, titles, recon_col, suptitle, path):
    #row: [n_cols, 3, H, W] rgb in [0, 1], one panel per column with its title (matplotlib, so the values are readable)
    n_cols = row.shape[0]
    fig, axes = plt.subplots(1, n_cols, figsize=(1.5 * n_cols, 2.1))
    for k, ax in enumerate(axes):
        ax.imshow(row[k].permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
        ax.set_title(titles[k], fontsize=8, color="green" if k == recon_col else "black")
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ckpt = Path(CKPT) if CKPT is not None else pick_ckpt(RUN_DIR)
    model = twoChannelVAE(latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    #burst subset, like the training script. the train scan takes a while, so reuse the run's label cache there
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)
    #only trust a multi-factor cache (vae_guided_multi.py) computed on chA (the gene burst channel) with the same aspect clip,
    #single-factor caches have no area/aspect and older ones may be chB, so those get recomputed
    label_cache = Path(RUN_DIR) / "burst_labels.npz"
    cached = np.load(label_cache) if SET == "train" and label_cache.exists() else None
    if (cached is not None and "channel" in cached and str(cached["channel"]) == "chA"
            and "factor_names" in cached and list(cached["factor_names"]) == BURST_FACTORS
            and float(cached["aspect_max"]) == ASPECT_MAX):
        factors, burst_idx = cached["factors"], cached["burst_idx"]
    else:
        _, factors, burst_idx = get_burst_factors(dataset)

    i = burst_idx[CELL_IDX]                                  #CELL_IDX in the subset -> index in the full dataset
    x = dataset[i][0].to(DEVICE)                             #[2, 128, 128], cell-masked like the training input
    burst = load_burst_mask(dataset.chB_files[i], dataset.mask_transform).to(DEVICE)

    out_dir = f"{OUT_ROOT}/{Path(RUN_DIR).name}/{ckpt.stem}_{SET}_idx{CELL_IDX}"
    x_vis = (x * 0.5 + 0.5).clamp(0, 1)                      #un-normalize from [-1,1] back to [0,1] for viewing
    refs = {ch: ref_columns(x_vis[ch:ch+1], burst) for ch in (0, 1)}
    ring = mask_ring(burst)                                  #[1, H, W]
    ring_rgb = torch.tensor(RING_RGB, device=DEVICE).view(3, 1, 1)
    in_burst = burst[0].bool()                               #[H, W], for the mean intensity inside the mask
    intensity_dim = TARGETS.get("intensity")                 #None if this run has no intensity code

    #the cell's measured factors (from its hand-drawn mask), shown on the input panel of the area/aspect plots
    intensity, log_area, log_aspect = factors[CELL_IDX]    #BURST_FACTORS order
    measured = {"area": f"area {np.exp(log_area):.0f} px", "aspect": f"aspect {np.exp(log_aspect):.2f}"}

    with torch.no_grad():
        mu, _ = model.encode(x.unsqueeze(0))                 #[1, LATENT_DIM]
        alphas = torch.linspace(-SPAN, SPAN, N_STEPS, device=DEVICE)

        dim_tag = {d: f for f, d in TARGETS.items()}
        for dim in range(LATENT_DIM):
            z = mu.repeat(N_STEPS, 1)                         #[N_STEPS, LATENT_DIM]
            z[:, dim] = mu[0, dim] + alphas                   #alpha = 0 in the middle -> z = mu, the reconstruction
            imgs = (model.decode(z) * 0.5 + 0.5).clamp(0, 1)   #[N_STEPS, 2, H, W]

            dim_dir = f"{out_dir}/z{dim}" + (f"_{dim_tag[dim]}" if dim in dim_tag else "")
            os.makedirs(dim_dir, exist_ok=True)
            #which traversal steps get the burst mask outline: only the recon for the shape codes, every step otherwise
            ring_steps = torch.ones(N_STEPS, device=DEVICE)
            if dim_tag.get(dim) in RECON_ONLY_RING:
                ring_steps = torch.zeros(N_STEPS, device=DEVICE)
                ring_steps[N_STEPS // 2] = 1
            step_ring = ring * ring_steps.view(N_STEPS, 1, 1, 1)                 #[N_STEPS, 1, H, W]

            for ch, name in [(0, "chA"), (1, "chB")]:
                trav = imgs[:, ch:ch+1].repeat(1, 3, 1, 1)                       #[N_STEPS, 3, H, W]
                trav = trav * (1 - step_ring) + ring_rgb * step_ring
                row = torch.cat([refs[ch], trav], dim=0)                         #[3 + N_STEPS, 3, H, W]

                #green border around the reconstruction (middle traversal step)
                recon = row[3 + N_STEPS // 2]
                green = torch.tensor([0.0, 1.0, 0.0], device=DEVICE).view(3, 1, 1)
                recon[:, :BORDER, :] = green
                recon[:, -BORDER:, :] = green
                recon[:, :, :BORDER] = green
                recon[:, :, -BORDER:] = green

                #free codes: plain grid, no titles
                if dim not in dim_tag:
                    save_image(make_grid(row, nrow=3 + N_STEPS), f"{dim_dir}/{name}.png")
                    continue

                if dim == intensity_dim:
                    #intensity code: title every panel with the mean of this channel inside the burst mask.
                    #taken from imgs (before the outline/border are drawn), the input one from the 128 px input, so all are comparable
                    #(the info.txt label is on the raw crop, so it differs slightly)
                    in_mean = x_vis[ch][in_burst].mean().item()
                    step_means = imgs[:, ch][:, in_burst].mean(1).tolist()       #[N_STEPS]
                    titles = [f"input\n{in_mean:.3f}", "burst mask", "overlay"]
                    titles += [f"recon\n{m:.3f}" if s == N_STEPS // 2 else f"{a:+.1f}\n{m:.3f}"
                               for s, (a, m) in enumerate(zip(alphas.tolist(), step_means))]
                    suptitle = (f"z{dim} ({dim_tag[dim]}), {name}: mean {name} inside the burst mask, [0, 1]"
                                f"   (steps = offset from the cell's own z{dim})")
                else:
                    #shape codes (area/aspect): the decoded bursts have no mask to measure, so the steps only get their offset,
                    #the input panel gets the cell's measured value to compare the walk against
                    titles = [f"input\n{measured.get(dim_tag[dim], '')}", "burst mask", "overlay"]
                    titles += ["recon" if s == N_STEPS // 2 else f"{a:+.1f}" for s, a in enumerate(alphas.tolist())]
                    suptitle = (f"z{dim} ({dim_tag[dim]}), {name}"
                                f"   (steps = offset from the cell's own z{dim}, outline = hand-drawn burst on the recon)")
                save_annotated_row(row, titles, 3 + N_STEPS // 2, suptitle, f"{dim_dir}/{name}.png")

    info = [f"checkpoint: {ckpt}",
            f"set: {SET}, burst subset index {CELL_IDX} of {len(burst_idx)} (full dataset index {i})",
            f"chA: {dataset.chA_files[i].name}",
            f"chB: {dataset.chB_files[i].name}",
            f"burst intensity (mean chA under the mask, raw crop): {intensity:.4f}",
            f"burst area (px at 128 res): {np.exp(log_area):.1f}",
            f"burst aspect ratio (major/minor, clipped to [1, {ASPECT_MAX}]): {np.exp(log_aspect):.2f}"]
    with open(f"{out_dir}/info.txt", "w") as f:
        f.write("\n".join(info) + "\n")
    print("\n".join(info))
    print(f"saved to {out_dir}")
