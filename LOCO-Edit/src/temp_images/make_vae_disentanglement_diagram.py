'''
diagram of the vae disentanglement idea on real data:
input chA + chB -> encoder -> 16 latent codes -> perturb ONE code (others fixed) -> decoder -> change in both channels

uses the beta=1, 16-latent baseline vae and real traversals (not illustrations). run from src/:
    python temp_images/make_vae_disentanglement_diagram.py
'''
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VAE_disent.data_utils import twoChannelDataset
from VAE_disent.model import twoChannelVAE

DEVICE   = torch.device("cuda:9")
CKPT     = "vae_checkpoints_masked/beta_vae_beta=1_16/vae_epoch99.pt"
CELL     = "Projection-146_bc_cell12_f010" #a 2026-06 video (burst annotations for these videos line up with chA, see per-video check)
N_KNOWN  = 5                                #z0..z4 drawn as "known" factors, the rest "unknown"
KNOWN_COLOR, UNKNOWN_COLOR = "#1F6FD1", "#E07B00"
DIMS     = [(0, KNOWN_COLOR), (1, KNOWN_COLOR)] #(latent code to perturb, frame colour), both from the known group
ALPHAS   = [-3.0, -1.5, 0.0, 1.5, 3.0]     #offsets added to the cell's own code, 0 = plain reconstruction
OUT_STEM = "temp_images/vae_disentanglement_diagram"
FIG_W, FIG_H = 16, 7.2
CROP_X   = 0.815                            #keep this fraction of the width (traversal blocks end ~0.79)
CROP_X0  = 0.04                             #left crop edge: the compact inputs start at ~0.062

#chA shown in green, chB in magenta (same LUTs as the composite previews)
GREEN   = LinearSegmentedColormap.from_list("g", ["black", "#39FF5A"])
MAGENTA = LinearSegmentedColormap.from_list("m", ["black", "#FF3DF0"])

#----- run the vae -----
ds = twoChannelDataset("data/singlecell_chA_split/train", "data/singlecell_chB_split/train", mask_dir="data/singlecell_mask")
names = [p.name for p in ds.chA_files]
idx = next(i for i, n in enumerate(names) if CELL in n)
x = ds[idx][0].unsqueeze(0).to(DEVICE)

model = twoChannelVAE(16).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

with torch.no_grad():
    mu, _ = model.encode(x)
    trav = {}
    for d, _ in DIMS:
        z = mu.repeat(len(ALPHAS), 1)
        z[:, d] += torch.tensor(ALPHAS, device=DEVICE)
        trav[d] = (model.decode(z) * 0.5 + 0.5).clamp(0, 1).cpu().numpy() #[n_alpha, 2, H, W]

x_np = (x[0] * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
mu_np = mu[0].cpu().numpy()

#numbers behind the caption: cell area (fraction of pixels above background) and mean brightness per channel
for d, _ in DIMS:
    imgs = trav[d]
    area = [(im.max(0) > 0.15).mean() for im in imgs]
    fg = [im.max(0) > 0.15 for im in imgs]
    bA = [im[0][m].mean() for im, m in zip(imgs, fg)]
    bB = [im[1][m].mean() for im, m in zip(imgs, fg)]
    print(f"z{d}: area {area[0]:.3f} -> {area[-1]:.3f} | chA brightness {bA[0]:.3f} -> {bA[-1]:.3f} | chB brightness {bB[0]:.3f} -> {bB[-1]:.3f}")

#----- figure -----
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
canvas = fig.add_axes([0, 0, 1, 1])
canvas.set_xlim(0, 1)
canvas.set_ylim(0, 1)
canvas.axis("off")

def show(img, cmap, rect, border=None):
    ax = fig.add_axes(rect)
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(border or "#888888")
        s.set_linewidth(2 if border else 0.6)
    return ax

def arrow(x0, y0, x1, y1, color="#444444"):
    canvas.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=2, color=color))

def label(x, y, s, size=12, weight="normal", color="#222222", ha="center"):
    canvas.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va="center")

#1. input (compact: smaller images, stacked close, right next to the backbone)
img_w, img_x = 0.07, 0.062
img_h = img_w * 16 / 7.2
show(x_np[0], GREEN,   [img_x, 0.545, img_w, img_h])
show(x_np[1], MAGENTA, [img_x, 0.31, img_w, img_h])
label(img_x + img_w / 2, 0.73, "input cell", size=12, weight="bold")
label(img_x + img_w / 2, 0.523, "chA (condensate)", size=10)
label(img_x + img_w / 2, 0.288, "chB (gene burst)", size=10)

#2. feature backbone (any model that maps the image to a set of codes, the vae here)
arrow(img_x + img_w + 0.004, 0.493, 0.165, 0.493) #passes between the chA label and the chB image
canvas.add_patch(FancyBboxPatch((0.175, 0.423), 0.052, 0.14, boxstyle="round,pad=0.008",
                                facecolor="#E6E9EE", edgecolor="#5A6E85", lw=1.5))
label(0.201, 0.493, "feature\nbackbone", size=11, weight="bold")
arrow(0.237, 0.493, 0.262, 0.493)

#3. disentangled features: 16 grey codes, grouped into known (z0..z4) and unknown (z5..z15)
box_h, box_w, top, bx = 0.032, 0.042, 0.82, 0.31
GROUP_GAP = 0.014 #vertical gap between the known and unknown groups
def code_y(k):
    #bottom edge of code k, codes after the known group are shifted down by the gap
    return top - (k + 1) * box_h - (GROUP_GAP if k >= N_KNOWN else 0.0)
for k in range(16):
    y = code_y(k)
    canvas.add_patch(Rectangle((bx, y), box_w, box_h * 0.9, facecolor="#D9D9D9", edgecolor="#8C8C8C", lw=0.6))
    label(bx - 0.005, y + box_h * 0.45, f"z{k}", size=8.5, color="#444444", ha="right")
label(bx + box_w / 2 - 0.01, 0.885, "disentangled features", size=12, weight="bold")
label(bx + box_w / 2 - 0.01, 0.855, "take 16 latent codes in a VAE, for example", size=9, color="#555555")

def group_box(k_first, k_last, color, name):
    #frame around codes k_first..k_last, with the group name written sideways on the left
    y_lo = code_y(k_last) - 0.005
    y_hi = code_y(k_first) + box_h * 0.9 + 0.005
    canvas.add_patch(Rectangle((bx - 0.026, y_lo), box_w + 0.031, y_hi - y_lo, facecolor="none", edgecolor=color, lw=2.2))
    canvas.text(bx - 0.034, (y_lo + y_hi) / 2, name, fontsize=11, fontweight="bold", color=color,
                rotation=90, ha="center", va="center")

group_box(0, N_KNOWN - 1, KNOWN_COLOR, "known")
group_box(N_KNOWN, 15, UNKNOWN_COLOR, "unknown")

#perturbation note, under the codes (that is where the edit happens)
canvas.add_patch(FancyBboxPatch((0.267, 0.075), 0.13, 0.15, boxstyle="round,pad=0.008", facecolor="#FFF8EC", edgecolor="#C9A66B", lw=1))
label(0.332, 0.20, "perturb ONE code,", size=10, weight="bold")
label(0.332, 0.172, "keep the other 15 fixed", size=9.5)
label(0.332, 0.132, r"$z_k \rightarrow z_k + \alpha$", size=12)
label(0.332, 0.095, r"($\alpha$ = how far the code is pushed)", size=8.5, color="#555555")

#4. traversals: one block per perturbed code, rows = chA / chB, columns = alpha
x0, cw, gap = 0.47, 0.057, 0.006
ch = cw * 16 / 7.2
blocks_y = [0.54, 0.155]
for (d, color), by in zip(DIMS, blocks_y):
    canvas.add_patch(FancyBboxPatch((x0 - 0.035, by - 0.015), len(ALPHAS) * (cw + gap) + 0.04, 2 * ch + 0.065,
                                    boxstyle="round,pad=0.005", facecolor="none", edgecolor=color, lw=2))
    label(x0 + len(ALPHAS) * (cw + gap) / 2, by + 2 * ch + 0.03, f"perturb z{d}", size=12, weight="bold", color=color)
    for j, a in enumerate(ALPHAS):
        xx = x0 + j * (cw + gap)
        show(trav[d][j, 0], GREEN,   [xx, by + ch + 0.006, cw, ch], border="white" if a == 0 else None)
        show(trav[d][j, 1], MAGENTA, [xx, by, cw, ch], border="white" if a == 0 else None)
    label(x0 - 0.018, by + ch * 1.5, "chA", size=9.5)
    label(x0 - 0.018, by + ch * 0.5, "chB", size=9.5)
    #arrow straight from that code's box to its block
    arrow(bx + box_w + 0.004, code_y(d) + box_h * 0.45, x0 - 0.037, by + ch, color=color)

#alpha axis under the bottom block: values, then arrows pushing left (negative) and right (positive) from the recon
col_c = [x0 + j * (cw + gap) + cw / 2 for j in range(len(ALPHAS))]
mid = col_c[len(ALPHAS) // 2]
for xx, a in zip(col_c, ALPHAS):
    label(xx, 0.115, "recon" if a == 0 else f"{a:+g}", size=10, weight="bold" if a == 0 else "normal")
canvas.add_patch(FancyArrowPatch((mid - 0.012, 0.082), (col_c[0] - 0.02, 0.082), arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#444444"))
canvas.add_patch(FancyArrowPatch((mid + 0.012, 0.082), (col_c[-1] + 0.02, 0.082), arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#444444"))
label((mid + col_c[0]) / 2, 0.058, r"push negative ($\alpha < 0$)", size=9.5, color="#444444")
label((mid + col_c[-1]) / 2, 0.058, r"push positive ($\alpha > 0$)", size=9.5, color="#444444")
label(mid, 0.025, r"perturbation $\alpha$ added to the latent code", size=10.5, color="#444444")

#crop off the empty strip right of the traversal blocks and the empty band where the title was
crop = Bbox([[FIG_W * CROP_X0, 0], [FIG_W * CROP_X, FIG_H * 0.925]])
fig.savefig(f"{OUT_STEM}.png", dpi=200, bbox_inches=crop)
fig.savefig(f"{OUT_STEM}.pdf", bbox_inches=crop)
fig.savefig(f"{OUT_STEM}.svg", bbox_inches=crop)
print(f"saved {OUT_STEM}.png / .pdf / .svg")
