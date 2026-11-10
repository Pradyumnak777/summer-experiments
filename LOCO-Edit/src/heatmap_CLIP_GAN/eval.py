'''
gan version of heatmap_CLIP/eval.py: takes one gan heatmap and returns its similarity with every sentence in the label
space (label_set in data_utils.py), using the clip finetuned on the gan heatmaps (finetune.py), or the plain pretrained one

then, on the gan side, for the cell the heatmap was made from (same file name, see make_heatmaps in data_utils.py):
    - its burst attributes (all of LABEL_FACTORS), on the input with the hand-drawn burst mask, and on the reconstruction
      G(w) with the bounded burst detection (mask_track_bounded in vae_viz.py, as the generated burst has no hand-drawn mask)
    - every direction (the 3 svm ext30 normals + the top 10 ganspace pcs) walked around the cell's own inverted w, each
      step generated and its burst detected the same bounded way, titled with the value of the attribute clip chose.
      the gan stand-in for the vae's latent traversal: w + alpha * d instead of mu + alpha * e_d, alpha within +-SPAN sd
      of where the inverted train w's sit along d (the vae walked +-3 in its N(0, 1) latents, ie +-3 sd too)
saved as viz_new/<heatmap name>/: heatmap.png, cell.png, stats.txt (the printed numbers), traversal/<dir>.png (one per
direction, the heatmap's own one tagged "heatmap")

run from src/ (gpu 9):
    python -m heatmap_CLIP_GAN.eval
'''
import shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.measure import regionprops
import open_clip

from heatmap_CLIP_GAN.data_utils import (DATA_VAL_DIR, LABEL_FACTORS, SEEN, label_set, known_dirs, unknown_dirs,
                                         TRAIN_W_DIR, VAL_W_DIR, gan_data)
from disentanglement_guided_GAN.train_svm import load_data
from disentanglement_guided_GAN.testing.svm_edit import get_burst, get_cell_mask
from disentanglement_semantic_eval.gan_dci import ASPECT_MAX
from disentanglement_guided.vae_viz import mask_track_bounded, detect_each
from disentanglement_guided.burst_detect import load_spot_model

import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

DEVICE     = torch.device("cuda:9")
MODEL_NAME = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
CKPT       = "heatmap_CLIP_GAN/checkpoints/best.pt"   #None -> the pretrained biomed clip, ie zero-shot
#a heatmap png, None -> the first intensity heatmap in data/. data/<factor>, data_val/<factor> or data_unknown/pca<k>
# IMAGE_PATH = "heatmap_CLIP_GAN/data_val/area/heatmaps/20260611_HP_bh1-PP7_15_TIGRE-PCP-mSG_17-01_processed-Orthogonal Projection-146_bc_cell3_f047.png"

IMAGE_PATH = "heatmap_CLIP_GAN/data_unknown/pca3/heatmaps/20260611_HP_bh1-PP7_15_TIGRE-PCP-mSG_17-01_processed-Orthogonal Projection-146_bc_cell0_f007.png"


N_STEPS    = 7       #odd, so the middle step is the reconstruction
SPAN       = 5.0     #each direction walks w +- SPAN sd (sd of the inverted train w's along it), the vae walked +-3 too
VIZ_DIR    = "heatmap_CLIP_GAN/viz_new"
RED        = (1.0, 0.0, 0.0)   #hand-drawn burst outline, on the input
YELLOW     = (1.0, 1.0, 0.0)   #bounded detected burst outline, on the generated steps (same color as vae_viz.py)

#how each attribute is printed. area/axes are in px at the 128 px model resolution
FMT = {"intensity": "{:.3f}", "area": "{:.0f} px", "aspect ratio": "{:.2f}", "major axis": "{:.1f} px", "minor axis": "{:.1f} px"}
#which attributes go in the traversal panel titles: a known (svm) direction shows its own factor, a pca direction all 3.
#direction name -> attribute name in burst_attributes ("aspect" is the svm/folder name, "aspect ratio" the attribute)
KNOWN_ATTR  = {"intensity": "intensity", "area": "area", "aspect": "aspect ratio"}
# SHORT       = {"intensity": "int", "area": "area", "aspect ratio": "asp"}   #labels when 3 are stacked in one title
#the 2 unseen factors (no svm direction of their own), added under every direction's titles so a clip pick of them can be checked
UNSEEN_ATTR = ["major axis", "minor axis"]
SHORT       = {"intensity": "int", "area": "area", "aspect ratio": "asp", "major axis": "maj", "minor axis": "min"}   #labels when stacked
#traversal plot layout, in inches
PANEL_IN    = 1.75   #each square panel
GAP_IN      = 0.15   #between panels, rows, and at the edges
LINE_IN     = 0.19   #one panel title line at fontsize 10
SUPTITLE_IN = 0.6    #the 2 line suptitle


def load_model(ckpt=CKPT):
    #-> model, its preprocess, and the features of all the label sentences [len(LABEL_FACTORS), 512]
    #same as heatmap_CLIP.eval.load_model (copied, importing that file pulls in heatmap_CLIP.data_utils' vae setup)
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(DEVICE).eval()

    #the text features depend on the checkpoint (finetuning trains the text tower too), so they're computed per model
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    label_tokens = tokenizer([label_set[k] for k in range(len(LABEL_FACTORS))]).to(DEVICE)
    with torch.no_grad():
        txt_feat = model.encode_text(label_tokens, normalize=True).float()
    return model, preprocess_val, txt_feat


@torch.no_grad()
def heatmap_similarities(image, model, preprocess, txt_feat):
    #image: path or PIL image of one heatmap -> list of cosine similarities, one per label in the label space
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    x = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)     #[1, 3, 224, 224]
    img_feat = model.encode_image(x, normalize=True).float()          #[1, 512]
    return (img_feat @ txt_feat.T)[0].tolist()


def all_dirs():
    #every direction that gets walked: the 3 svm ext30 normals, then pca0..pca9 -> {name: [512] unit vector}
    return {**known_dirs(), **unknown_dirs()}


def heatmap_direction(heatmap_path):
    #the direction the heatmap explains, from its folder: data/intensity -> "intensity", data_unknown/pca4 -> "pca4"
    return Path(heatmap_path).parent.parent.name


def find_cell(heatmap_path):
    #the cell the heatmap was made from -> its inverted w [512], the input chA/chB [2, 128, 128] in [0, 1] (cell-masked,
    #like the inversion saw it), cell mask [H, W], hand-drawn burst mask [H, W] (128 px), and its split.
    #data_val/ was made from the validation w's, data/ and data_unknown/ from the train ones
    set_name = "validation" if Path(DATA_VAL_DIR).name in Path(heatmap_path).parts else "train"
    w_dir = VAL_W_DIR if set_name == "validation" else TRAIN_W_DIR
    d = np.load(Path(w_dir) / f"{Path(heatmap_path).stem}.npz")
    chA_path, chB_path = Path(str(d["chA_path"])), Path(str(d["chB_path"]))
    x = gan_data.load_target(chA_path, chB_path).astype(np.float32) / 255.0
    return d["w"], x, get_cell_mask(chA_path), get_burst(chB_path), set_name


def direction_sds(dirs):
    #how far the inverted train w's spread along every direction (std of w . d), the unit of the walk
    W = load_data(TRAIN_W_DIR)[0]                                      #[N, 512]
    return {name: float((W @ d).std()) for name, d in dirs.items()}


def load_G():
    with open(gan_data.PKL, 'rb') as f:
        return gan_data.legacy.load_network_pkl(f)['G_ema'].to(DEVICE).eval().requires_grad_(False)


def burst_attributes(chA, mask):
    #chA: [H, W] in [0, 1], mask: [H, W] bool -> {factor: value} for every factor in LABEL_FACTORS, None if no burst.
    #same as heatmap_CLIP.eval.burst_attributes (measured like get_burst_factors in gan_dci.py, but not logged)
    if mask is None or not mask.any():
        return None
    p = regionprops(mask.astype(np.uint8))[0]
    major, minor = p.major_axis_length, p.minor_axis_length
    return {"intensity": float(chA[mask].mean()), "area": float(mask.sum()),
            "aspect ratio": float(np.clip(major / max(minor, 1e-6), 1.0, ASPECT_MAX)),
            "major axis": float(major), "minor axis": float(minor)}


def fmt_attr(attrs, factor):
    return "-" if attrs is None else FMT[factor].format(attrs[factor])


def traverse(G, w, dirs, sds, cell_mask, burst, spot_model):
    #every direction walked around the cell's w (w + alpha * d, alpha within +-SPAN sd along d), generated, and the burst
    #detected on each step's chA -> chA per step [n_dirs, N_STEPS, H, W], bounded burst mask per step (same shape, bool),
    #the steps in sd units [N_STEPS]
    steps = np.linspace(-SPAN, SPAN, N_STEPS)                          #in sd units, 0 in the middle -> the reconstruction
    D = np.stack(list(dirs.values()))                                  #[n_dirs, 512]
    alphas = steps[None] * np.array([sds[n] for n in dirs])[:, None]   #[n_dirs, N_STEPS], raw step along each unit d
    ws = torch.from_numpy((w[None, None] + alphas[..., None] * D[:, None]).reshape(-1, w.shape[0]).astype(np.float32)).to(DEVICE)
    with torch.no_grad():   #same shared W at every layer as the inversion
        imgs = ((G.synthesis(ws.unsqueeze(1).repeat(1, G.num_ws, 1), noise_mode='const') + 1) / 2).clamp(0, 1)[:, 0]
    imgs = imgs.cpu().numpy()                                          #[n_dirs * N_STEPS, H, W], chA (the gene burst channel)

    #bounded: the burst can only be found within a margin of the hand-drawn one, so it can't jump to a bright speck elsewhere
    track_bounded = lambda f: mask_track_bounded(f, cell_mask, burst, spot_model)
    masks = detect_each(imgs, track_bounded).astype(bool)
    shape = (len(dirs), N_STEPS, *imgs.shape[-2:])
    return imgs.reshape(shape), masks.reshape(shape), steps


def outlined(gray, mask, rgb):
    #gray: [H, W] in [0, 1] -> [H, W, 3], with the 1 px ring just outside mask drawn in rgb (the burst pixels stay untouched)
    out = np.repeat(gray[..., None], 3, axis=2)
    if mask is not None:
        out[binary_dilation(mask) & ~mask] = rgb
    return out


def save_cell(path, x):
    #the cell as the gan inversion (and the heatmap) saw it: 128 px, cell-masked. x: [2, H, W] in [0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.8))
    for ax, ch, name in zip(axes, (0, 1), ("chA (gene burst)", "chB")):
        ax.imshow(x[ch], cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(Path(path).parent.name, fontsize=7)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def attr_lines(attrs, shown):
    #the panel title lines for the attributes in shown: one -> just its value (like before), several -> "int 0.853" per line
    if len(shown) == 1:
        return fmt_attr(attrs, shown[0])
    return "\n".join(f"{SHORT[f]} {fmt_attr(attrs, f)}" for f in shown)


def save_traversals(trav_dir, x_chA, burst, imgs, masks, steps, dirs, sds, heat_dir, chosen, orig_attrs):
    #one plot per direction, trav_dir/<dir>.png: input (hand-drawn burst, red) | the walk (bounded detected burst, yellow).
    #panel titles: a known (svm) direction shows its own attribute (intensity / area / aspect ratio), a pca direction all 3,
    #plus major / minor axis (the unseen factors) for both, regardless of what clip chose (clip's choice is only mentioned in the suptitle). the heatmap's own direction is tagged "heatmap"
    trav_dir.mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(dirs):
        tags = ["heatmap"] if name == heat_dir else []
        kind = "svm ext30" if name in known_dirs() else "ganspace"
        # shown = [KNOWN_ATTR[name]] if name in KNOWN_ATTR else list(KNOWN_ATTR.values())
        #own factor (svm) or all 3 known (pca), then the 2 unseen axes for every direction
        shown = ([KNOWN_ATTR[name]] if name in KNOWN_ATTR else list(KNOWN_ATTR.values())) + UNSEEN_ATTR
        n_lines = len(shown)
        # fig, axes = plt.subplots(1, N_STEPS + 1, figsize=(1.9 * (N_STEPS + 1), 3.0))
        # fig, axes = plt.subplots(1, N_STEPS + 1, figsize=(1.9 * (N_STEPS + 1), 3.0 + 0.35 * (n_lines - 1)))   #taller for 3 lines
        #2 rows: the outlined panels on top, the same images without any outline below (the outline can hide a small burst)
        # fig, axes2 = plt.subplots(2, N_STEPS + 1, figsize=(1.9 * (N_STEPS + 1), 3.0 + 0.35 * (n_lines - 1) + 1.9))
        #panels placed in inches (tight_layout spaced the rows differently for 3 vs 5 title lines):
        #suptitle, then the panel titles (n_lines + the step line), then the 2 rows of square panels
        n_cols = N_STEPS + 1
        fig_w = n_cols * PANEL_IN + (n_cols + 1) * GAP_IN
        fig_h = SUPTITLE_IN + LINE_IN * (n_lines + 1) + 2 * PANEL_IN + 3 * GAP_IN
        fig = plt.figure(figsize=(fig_w, fig_h))
        axes2 = np.array([[fig.add_axes([(GAP_IN + c * (PANEL_IN + GAP_IN)) / fig_w,
                                          (GAP_IN + (1 - r) * (PANEL_IN + GAP_IN)) / fig_h,   #row 0 on top
                                          PANEL_IN / fig_w, PANEL_IN / fig_h]) for c in range(n_cols)] for r in range(2)])
        axes, axes_plain = axes2
        axes_plain[0].imshow(x_chA, cmap="gray", vmin=0, vmax=1)
        for s in range(N_STEPS):
            axes_plain[s + 1].imshow(imgs[k, s], cmap="gray", vmin=0, vmax=1)
        axes[0].imshow(outlined(x_chA, burst, RED))
        # axes[0].set_title(f"input\n{fmt_attr(orig_attrs, chosen)}", fontsize=10)
        axes[0].set_title(f"input\n{attr_lines(orig_attrs, shown)}", fontsize=10)
        for s in range(N_STEPS):
            step = "recon" if s == N_STEPS // 2 else f"{steps[s]:+.1f} sd"
            axes[s + 1].imshow(outlined(imgs[k, s], masks[k, s], YELLOW))
            # axes[s + 1].set_title(f"{step}\n{fmt_attr(burst_attributes(imgs[k, s], masks[k, s]), chosen)}", fontsize=10,
            #                       color="green" if s == N_STEPS // 2 else "black")
            axes[s + 1].set_title(f"{step}\n{attr_lines(burst_attributes(imgs[k, s], masks[k, s]), shown)}", fontsize=10,
                                  color="green" if s == N_STEPS // 2 else "black")
        # for ax in axes:
        for ax in axes2.ravel():
            ax.axis("off")
        # fig.suptitle(f"{name} ({kind}" + (", " + ", ".join(tags) if tags else "") + f") walked around the cell's own w, chA, "
        #              f"1 sd = {sds[name]:.2f}   |   titles: {chosen} (clip's choice)\nred = hand-drawn burst on the input, "
        #              f"yellow = bounded detected burst on each generated step, - = no burst detected", fontsize=10)
        fig.suptitle(f"{name} ({kind}" + (", " + ", ".join(tags) if tags else "") + f") walked around the cell's own w, chA, "
                     f"1 sd = {sds[name]:.2f}   |   titles: {', '.join(shown)}   |   clip's choice: {chosen}\n"
                     f"red = hand-drawn burst on the input, yellow = bounded detected burst on each generated step, "
                     f"- = no burst detected   |   bottom row: same images, no outline", fontsize=10)
        # fig.tight_layout(rect=(0, 0, 1, 0.84))      #room for the 2 line suptitle above the panel titles
        #room for the 2 line suptitle above the panel titles: the same ~0.48 in as before, so a smaller fraction of a taller fig
        # fig.tight_layout(rect=(0, 0, 1, 1 - 0.16 * 3.0 / fig.get_figheight()))   #axes are placed by hand now
        # fig.savefig(trav_dir / ("_".join([name] + tags) + ".png"), dpi=150)
        fig.savefig(trav_dir / ("_".join([name] + tags) + ".png"), dpi=150, bbox_inches="tight")   #trims the empty bottom of the taller figs
        plt.close(fig)


if __name__ == "__main__":
    torch.cuda.set_device(DEVICE)
    image_path = IMAGE_PATH if IMAGE_PATH is not None else sorted(Path("heatmap_CLIP_GAN/data/intensity/heatmaps").glob("*.png"))[0]
    model, preprocess, txt_feat = load_model(CKPT)
    sims = heatmap_similarities(image_path, model, preprocess, txt_feat)

    #softmax over the label space with the model's logit scale, the cosines alone are often close together
    probs = (model.logit_scale.exp().item() * torch.tensor(sims)).softmax(0).tolist()

    lines = [f"heatmap: {image_path}", f"model: {CKPT if CKPT is not None else 'pretrained (zero-shot)'}"]
    best = max(range(len(sims)), key=lambda k: sims[k])
    for k, (s, p) in enumerate(zip(sims, probs)):
        tag = "seen  " if LABEL_FACTORS[k] in SEEN else "unseen"
        lines.append(f"  {k} {LABEL_FACTORS[k]:>12s} ({tag}) | sim {s:.4f} | prob {p:.3f}" + ("   <- best" if k == best else ""))
    chosen = LABEL_FACTORS[best]

    #the gan side: the cell this heatmap was made from, its burst attributes, and every direction walked
    heat_dir = heatmap_direction(image_path)
    w, x, cell_mask, burst, set_name = find_cell(image_path)
    dirs = all_dirs()
    sds = direction_sds(dirs)
    imgs, masks, steps = traverse(load_G(), w, dirs, sds, cell_mask, burst, load_spot_model())
    orig_attrs = burst_attributes(x[0], burst)                                       #hand-drawn mask
    recon_attrs = burst_attributes(imgs[0, N_STEPS // 2], masks[0, N_STEPS // 2])    #any row's middle step is the recon

    lines.append(f"\ncell: {set_name} split, heatmap explains direction {heat_dir}")
    lines.append(f"  {'':>12s}   {'input (hand-drawn)':>18s}   {'recon (bounded detection)':>25s}")
    for f in LABEL_FACTORS:
        lines.append(f"  {f:>12s}   {fmt_attr(orig_attrs, f):>18s}   {fmt_attr(recon_attrs, f):>25s}" + ("   <- clip's choice" if f == chosen else ""))
    k_heat = list(dirs).index(heat_dir)
    walk = [fmt_attr(burst_attributes(imgs[k_heat, s], masks[k_heat, s]), chosen) for s in range(N_STEPS)]
    lines.append(f"\n{chosen} along the {heat_dir} walk ({', '.join(f'{a:+.1f}' for a in steps)} sd, 1 sd = {sds[heat_dir]:.2f}):\n  "
                 + "  ".join(walk))
    print("\n".join(lines))

    out_dir = Path(VIZ_DIR, Path(image_path).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(image_path, out_dir / "heatmap.png")
    (out_dir / "stats.txt").write_text("\n".join(lines) + "\n")   #the vae eval.py only printed these
    save_cell(out_dir / "cell.png", x)
    save_traversals(out_dir / "traversal", x[0], burst, imgs, masks, steps, dirs, sds, heat_dir, chosen, orig_attrs)
    print(f"\nsaved to {out_dir}/")
