'''
grayscale latent traversals + grad-cam, one figure:
rows = (z0, chA), (z0, chB), (z1, chA), (z1, chB)
cols = traversal over alpha (decoder output), then the grad-cam map for that latent over that input channel

grad-cam is the GradCAMVAE class from gradcam/vae_gradcam1.py (last conv layer, ie conv4, as in
liu et al. cvpr 2020), run on the same cell as the traversals. conv4 has already mixed the two channels,
so it gives ONE map per latent: the chA and chB grad-cam panels of a latent show the same map

same beta=1, 16-latent baseline vae as make_vae_disentanglement_diagram.py. run from src/:
    python temp_images/make_vae_traversal_gradcam.py           #grayscale
    python temp_images/make_vae_traversal_gradcam.py --color   #chA green, chB magenta, saved as *_color
'''
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VAE_disent.data_utils import twoChannelDataset
from VAE_disent.model import twoChannelVAE
from gradcam.vae_gradcam1 import GradCAMVAE #importing also runs that script's module-level setup (no outputs written)

DEVICE   = torch.device("cuda:9")
CKPT     = "vae_checkpoints_masked/beta_vae_beta=1_16/vae_epoch99.pt"
CELL     = "Projection-146_bc_cell12_f010" #same cell as the disentanglement diagram
DIMS     = [0, 1]
ALPHAS   = [-3.0, -1.5, 0.0, 1.5, 3.0]     #offsets added to the cell's own code, 0 = plain reconstruction
COLOR    = "--color" in sys.argv
OUT_STEM = "temp_images/vae_traversal_gradcam" + ("_color" if COLOR else "")
#frame on the alpha = 0 (reconstruction) column: green in grayscale, deep orange in colour (green would blend with chA)
RECON_BORDER = "#E8590C" if COLOR else "#2CA02C"

#channel colormaps: grayscale, or chA green / chB magenta (same LUTs as the disentanglement diagram)
GREEN   = LinearSegmentedColormap.from_list("g", ["black", "#39FF5A"])
MAGENTA = LinearSegmentedColormap.from_list("m", ["black", "#FF3DF0"])
CH_CMAP = {0: GREEN, 1: MAGENTA} if COLOR else {0: "gray", 1: "gray"}

#----- data + model -----
ds = twoChannelDataset("data/singlecell_chA_split/train", "data/singlecell_chB_split/train", mask_dir="data/singlecell_mask")
names = [p.name for p in ds.chA_files]
idx = next(i for i, n in enumerate(names) if CELL in n)
x = ds[idx][0].unsqueeze(0).to(DEVICE)

model = twoChannelVAE(latent_dim=16)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.to(DEVICE)
model.eval()

#traversals (decoder side)
with torch.no_grad():
    mu, _ = model.encode(x)
    trav = {}
    for d in DIMS:
        z = mu.repeat(len(ALPHAS), 1)
        z[:, d] += torch.tensor(ALPHAS, device=DEVICE)
        trav[d] = (model.decode(z) * 0.5 + 0.5).clamp(0, 1).cpu().numpy() #[n_alpha, 2, H, W]

#grad-cam (encoder side), exactly the vae_gradcam1 computation: backprop mu[:, d] to conv4
gradcam = GradCAMVAE(model, model.conv4)
cams = {d: gradcam(x.clone(), latent_idx=d) for d in DIMS} #[128, 128] each, in [0, 1]

x_np = (x[0] * 0.5 + 0.5).clamp(0, 1).detach().cpu().numpy() #original input, the grad-cam background

#----- figure -----
p, gap, cam_gap = 1.5, 0.06, 0.35   #panel size, gap between panels, extra gap before the grad-cam column (inches)
orig_gap = 0.35                     #extra gap between the original image column and the traversal
cap_h, head_h = 0.42, 0.6           #caption band under each row, header band on top
left, right = 0.25, 1.0             #right margin holds the colorbar
n_rows, n_trav = 2 * len(DIMS), len(ALPHAS)
W = left + p + orig_gap + n_trav * p + (n_trav - 1) * gap + cam_gap + p + right
H = head_h + n_rows * (p + cap_h) + 0.1
fig = plt.figure(figsize=(W, H), facecolor="white")

def add_panel(col_x, row_y, img, cam=None, border=None, cmap="gray"):
    ax = fig.add_axes([col_x / W, row_y / H, p / W, p / H])
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    im = None
    if cam is not None:
        im = ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1) #same overlay as save_overlay in vae_gradcam1
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(3.0 if border else 0.6)
        s.set_edgecolor(border or "#888888")
    return im

orig_x = left                       #1st column: the original input channel
trav_left = orig_x + p + orig_gap
col_xs = [trav_left + j * (p + gap) for j in range(n_trav)]
cam_x = col_xs[-1] + p + cam_gap

#column headers
head_y = H - head_h / 2
fig.text((orig_x + p / 2) / W, head_y / H, "original", ha="center", va="center", fontsize=11, fontweight="bold")
for xx, a in zip(col_xs, ALPHAS):
    fig.text((xx + p / 2) / W, head_y / H, "recon" if a == 0 else rf"$\alpha$ = {a:+g}", ha="center", va="center", color=RECON_BORDER if a == 0 else "black",
             fontsize=11, fontweight="bold" if a == 0 else "normal")
fig.text((cam_x + p / 2) / W, head_y / H, "gradient-based\nvisualization", ha="center", va="center", fontsize=10.5, fontweight="bold")

rows = [(d, c) for d in DIMS for c in (0, 1)]
for r, (d, c) in enumerate(rows):
    row_y = H - head_h - (r + 1) * (p + cap_h) + cap_h #bottom of this row's panels
    ch_name = "chA" if c == 0 else "chB"
    add_panel(orig_x, row_y, x_np[c], cmap=CH_CMAP[c])
    fig.text((orig_x + p / 2) / W, (row_y - cap_h * 0.45) / H, f"input {ch_name}", ha="center", va="center", fontsize=10)
    for j, xx in enumerate(col_xs):
        add_panel(xx, row_y, trav[d][j, c], border=RECON_BORDER if ALPHAS[j] == 0 else None, cmap=CH_CMAP[c])
    im = add_panel(cam_x, row_y, x_np[c], cams[d])
    fig.text((trav_left + (n_trav * p + (n_trav - 1) * gap) / 2) / W, (row_y - cap_h * 0.45) / H,
             f"perturb z{d}, {ch_name}", ha="center", va="center", fontsize=11.5)
    fig.text((cam_x + p / 2) / W, (row_y - cap_h * 0.45) / H, f"z{d}, on {ch_name}", ha="center", va="center", fontsize=10)

#one shared colorbar for the grad-cam column
cax = fig.add_axes([(cam_x + p + 0.2) / W, (H - head_h - n_rows * (p + cap_h) + cap_h) / H, 0.15 / W,
                    (n_rows * (p + cap_h) - cap_h) / H])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1)), cax=cax)

fig.savefig(f"{OUT_STEM}.png", dpi=200)
fig.savefig(f"{OUT_STEM}.pdf")
fig.savefig(f"{OUT_STEM}.svg")
print(f"saved {OUT_STEM}.png / .pdf / .svg")

#----- every panel as its own image (native 128x128, no borders / labels) -----
#<OUT_STEM>_panels/: original_cell_chX.png, gradient_viz_zK_chX.png (this one with its colour scale)
#<OUT_STEM>_panels/perturbations/: zK_chX_alpha<a>.png, with the alpha = 0 column saved as zK_chX_recon.png
panel_dir = f"{OUT_STEM}_panels"
pert_dir = f"{panel_dir}/perturbations"
os.makedirs(pert_dir, exist_ok=True)

def colorize(img, c):
    #same look as the figure panels: the channel's colormap, or plain grayscale
    return plt.get_cmap(CH_CMAP[c])(np.clip(img, 0, 1))[..., :3]

def save_cam_with_scale(path, img, cam):
    #same overlay as the figure (grayscale input under jet at alpha 0.45), plus its 0-1 colour scale
    f, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    f.colorbar(plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1)), ax=ax, fraction=0.046, pad=0.04)
    f.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(f)

for c in (0, 1):
    ch_name = "chA" if c == 0 else "chB"
    plt.imsave(f"{panel_dir}/original_cell_{ch_name}.png", colorize(x_np[c], c))
    for d in DIMS:
        save_cam_with_scale(f"{panel_dir}/gradient_viz_z{d}_{ch_name}.png", x_np[c], cams[d])
        for j, a in enumerate(ALPHAS):
            tag = "recon" if a == 0 else f"alpha{a:+g}"
            plt.imsave(f"{pert_dir}/z{d}_{ch_name}_{tag}.png", colorize(trav[d][j, c], c))
print(f"saved individual panels to {panel_dir}/ (perturbations in {pert_dir}/)")
