'''
takes one heatmap image and returns its similarity with every sentence in the label space (label_set in data_utils.py),
using the finetuned biomed clip from finetune.py (or the plain pretrained one, to compare against zero-shot)

the similarity is the cosine between the image and sentence features, list index k = label k

then, on the vae side, for the cell the heatmap was made from (same file name, see gradcam_getData in data_utils.py):
    - its burst attributes (all of LABEL_FACTORS), on the input with the hand-drawn burst mask, and on the reconstruction
      with the bounded burst detection (mask_track_bounded in vae_viz.py, as the decoded burst has no hand-drawn mask)
    - every latent (z0..z15) perturbed around the cell's own value, each step decoded and its burst detected the same
      bounded way, titled with the value of the attribute clip chose
saved as viz_new/<heatmap name>/: heatmap.png, cell.png, traversal/z{d}.png (one per latent, the heatmap's one tagged "heatmap")

run from src/:
    python -m heatmap_CLIP.eval
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

from heatmap_CLIP.data_utils import DATA_DIR, DATA_VAL_DIR, LABEL_FACTORS, SEEN, TARGETS, label_set
from VAE_disent.model import twoChannelVAE
from VAE_disent.data_utils import twoChannelDataset
from disentanglement_semantic_eval.gan_dci import load_burst_mask, ASPECT_MAX
from disentanglement_guided.vae_viz import mask_track_bounded, detect_each
from disentanglement_guided.burst_detect import load_spot_model

DEVICE     = torch.device("cuda:9")
MODEL_NAME = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
CKPT       = "heatmap_CLIP/checkpoints/best.pt"   #None -> the pretrained biomed clip, ie zero-shot
# IMAGE_PATH = "heatmap_CLIP/data_unknown/z2/heatmaps/20260611_HP_bh1-PP7_15_TIGRE-PCP-mSG_17-01_processed-Orthogonal Projection-146_bc_cell0_f000.png"                                 #a heatmap png, None -> the first intensity heatmap in DATA_DIR
IMAGE_PATH = "heatmap_CLIP/data_unknown/z4/heatmaps/20260611_HP_bh1-PP7_15_TIGRE-PCP-mSG_17-01_processed-Orthogonal Projection-146_bc_cell0_f000.png"                                 #a heatmap png, None -> the first intensity heatmap in DATA_DIR

#the guided vae the heatmaps were made with (see data/done.txt)
VAE_RUN    = "vae_checkpoints_masked/guided_vae_beta=1_16_intensity-z0_area-z1_aspect-z2"
LATENT_DIM = 16
N_STEPS    = 7       #odd, so the middle step is the reconstruction
SPAN       = 3.0     #each latent walks from mu - SPAN to mu + SPAN, the others fixed at the cell's mu
VIZ_DIR    = "heatmap_CLIP/viz_new"
RED        = (1.0, 0.0, 0.0)   #hand-drawn burst outline, on the input
YELLOW     = (1.0, 1.0, 0.0)   #bounded detected burst outline, on the decoded steps (same color as vae_viz.py)

#how each attribute is printed. area/axes are in px at the 128 px model resolution
FMT = {"intensity": "{:.3f}", "area": "{:.0f} px", "aspect ratio": "{:.2f}", "major axis": "{:.1f} px", "minor axis": "{:.1f} px"}


def load_model(ckpt=CKPT):
    #-> model, its preprocess, and the features of all the label sentences [len(LABEL_FACTORS), 512]
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


def heatmap_latent(heatmap_path):
    #the latent the heatmap explains, from its folder: data/intensity -> z0, data/area -> z1, data_unknown/z4 -> z4
    folder = Path(heatmap_path).parent.parent.name
    return TARGETS[folder] if folder in TARGETS else int(folder[1:])


def find_cell(heatmap_path):
    #the cell the heatmap was made from -> model input [2, 128, 128] (cell-masked, in [-1, 1]), cell mask [H, W],
    #hand-drawn burst mask [H, W] (both resized to 128 like the input), and its split
    #data_val/ was made from the validation split, data/ and data_unknown/ from the train split
    set_name = "validation" if Path(DATA_VAL_DIR).name in Path(heatmap_path).parts else "train"
    dataset = twoChannelDataset(f"data/singlecell_chA_split/{set_name}", f"data/singlecell_chB_split/{set_name}",
                                mask_dir="data/singlecell_mask")
    name = Path(heatmap_path).stem + "_chA.png"
    i = next(k for k, p in enumerate(dataset.chA_files) if p.name == name)
    x, cell_mask = dataset[i]
    burst = load_burst_mask(dataset.chB_files[i], dataset.mask_transform)[0].numpy().astype(bool)
    return x, cell_mask[0].numpy(), burst, set_name


def load_vae():
    vae = twoChannelVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(torch.load(f"{VAE_RUN}/vae_final.pt", map_location=DEVICE))
    return vae.eval()


def burst_attributes(chA, mask):
    #chA: [H, W] in [0, 1], mask: [H, W] bool -> {factor: value} for every factor in LABEL_FACTORS, None if no burst.
    #measured like get_burst_factors in gan_dci.py (but not logged), all burst pixels taken as one region
    if not mask.any():
        return None
    p = regionprops(mask.astype(np.uint8))[0]
    major, minor = p.major_axis_length, p.minor_axis_length
    return {"intensity": float(chA[mask].mean()), "area": float(mask.sum()),
            "aspect ratio": float(np.clip(major / max(minor, 1e-6), 1.0, ASPECT_MAX)),
            "major axis": float(major), "minor axis": float(minor)}


def fmt_attr(attrs, factor):
    return "-" if attrs is None else FMT[factor].format(attrs[factor])


def traverse(vae, x, cell_mask, burst, spot_model):
    #every latent walked around the cell's mu (the others fixed), decoded, and the burst detected on each step's chA
    #-> chA per step [LATENT_DIM, N_STEPS, H, W], bounded burst mask per step (same shape, bool), the offsets
    with torch.no_grad():
        mu, _ = vae.encode(x.unsqueeze(0).to(DEVICE))                     #[1, LATENT_DIM]
        alphas = torch.linspace(-SPAN, SPAN, N_STEPS, device=DEVICE)
        z = mu.repeat(LATENT_DIM, N_STEPS, 1)                              #[LATENT_DIM, N_STEPS, LATENT_DIM]
        for d in range(LATENT_DIM):
            z[d, :, d] += alphas                                           #alpha = 0 in the middle -> the reconstruction
        imgs = (vae.decode(z.view(-1, LATENT_DIM)) * 0.5 + 0.5).clamp(0, 1)[:, 0]   #chA, the gene burst channel
    imgs = imgs.cpu().numpy()                                               #[LATENT_DIM * N_STEPS, H, W]

    #bounded: the burst can only be found within a margin of the hand-drawn one, so it can't jump to a bright speck elsewhere
    track_bounded = lambda f: mask_track_bounded(f, cell_mask, burst, spot_model)
    masks = detect_each(imgs, track_bounded).astype(bool)
    shape = (LATENT_DIM, N_STEPS, *imgs.shape[-2:])
    return imgs.reshape(shape), masks.reshape(shape), alphas.tolist()


def outlined(gray, mask, rgb):
    #gray: [H, W] in [0, 1] -> [H, W, 3], with the 1 px ring just outside mask drawn in rgb (the burst pixels stay untouched)
    out = np.repeat(gray[..., None], 3, axis=2)
    out[binary_dilation(mask) & ~mask] = rgb
    return out


def save_cell(path, x):
    #the cell as the vae (and gradcam) sees it: 128 px, cell-masked
    x = (x * 0.5 + 0.5).clamp(0, 1).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.8))
    for ax, ch, name in zip(axes, (0, 1), ("chA (gene burst)", "chB")):
        ax.imshow(x[ch], cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(Path(path).parent.name, fontsize=7)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_traversals(trav_dir, x_chA, burst, imgs, masks, alphas, heat_dim, chosen, orig_attrs):
    #one plot per latent, trav_dir/z{d}.png: input (hand-drawn burst, red) | the walk (bounded detected burst, yellow),
    #every panel titled with the clip-chosen attribute. target codes are tagged with their factor ("z1_area"),
    #the heatmap's own latent with "heatmap"
    dim_tag = {d: f for f, d in TARGETS.items()}
    trav_dir.mkdir(parents=True, exist_ok=True)
    for d in range(LATENT_DIM):
        tags = ([dim_tag[d]] if d in dim_tag else []) + (["heatmap"] if d == heat_dim else [])
        fig, axes = plt.subplots(1, N_STEPS + 1, figsize=(1.9 * (N_STEPS + 1), 3.0))
        axes[0].imshow(outlined(x_chA, burst, RED))
        axes[0].set_title(f"input\n{fmt_attr(orig_attrs, chosen)}", fontsize=10)
        for s in range(N_STEPS):
            step = "recon" if s == N_STEPS // 2 else f"{alphas[s]:+.1f}"
            axes[s + 1].imshow(outlined(imgs[d, s], masks[d, s], YELLOW))
            axes[s + 1].set_title(f"{step}\n{fmt_attr(burst_attributes(imgs[d, s], masks[d, s]), chosen)}", fontsize=10,
                                  color="green" if s == N_STEPS // 2 else "black")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"z{d}" + (f" ({', '.join(tags)})" if tags else "") + f" walked around the cell's own value, chA   |   "
                     f"titles: {chosen} (clip's choice)\nred = hand-drawn burst on the input, yellow = bounded detected "
                     f"burst on each decoded step, - = no burst detected", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.84))      #room for the 2 line suptitle above the panel titles
        fig.savefig(trav_dir / ("_".join([f"z{d}"] + tags) + ".png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    image_path = IMAGE_PATH if IMAGE_PATH is not None else sorted(Path(DATA_DIR, "intensity", "heatmaps").glob("*.png"))[0]
    model, preprocess, txt_feat = load_model(CKPT)
    sims = heatmap_similarities(image_path, model, preprocess, txt_feat)

    #softmax over the label space with the model's logit scale, the cosines alone are often close together
    probs = (model.logit_scale.exp().item() * torch.tensor(sims)).softmax(0).tolist()   #.item(), logit_scale is on the gpu, sims on the cpu

    print(f"heatmap: {image_path}")
    print(f"model: {CKPT if CKPT is not None else 'pretrained (zero-shot)'}")
    best = max(range(len(sims)), key=lambda k: sims[k])
    for k, (s, p) in enumerate(zip(sims, probs)):
        tag = "seen  " if LABEL_FACTORS[k] in SEEN else "unseen"
        print(f"  {k} {LABEL_FACTORS[k]:>12s} ({tag}) | sim {s:.4f} | prob {p:.3f}" + ("   <- best" if k == best else ""))
    chosen = LABEL_FACTORS[best]

    #the vae side: the cell this heatmap was made from, its burst attributes, and every latent perturbed
    heat_dim = heatmap_latent(image_path)
    x, cell_mask, burst, set_name = find_cell(image_path)
    x_chA = (x[0] * 0.5 + 0.5).clamp(0, 1).numpy()                         #input chA in [0, 1], what gradcam saw
    imgs, masks, alphas = traverse(load_vae(), x, cell_mask, burst, load_spot_model())
    orig_attrs = burst_attributes(x_chA, burst)                             #hand-drawn mask
    recon_attrs = burst_attributes(imgs[0, N_STEPS // 2], masks[0, N_STEPS // 2])   #any row's middle step is the recon

    print(f"\ncell: {set_name} split, heatmap explains z{heat_dim}")
    print(f"  {'':>12s}   {'input (hand-drawn)':>18s}   {'recon (bounded detection)':>25s}")
    for f in LABEL_FACTORS:
        print(f"  {f:>12s}   {fmt_attr(orig_attrs, f):>18s}   {fmt_attr(recon_attrs, f):>25s}" + ("   <- clip's choice" if f == chosen else ""))
    walk = [fmt_attr(burst_attributes(imgs[heat_dim, s], masks[heat_dim, s]), chosen) for s in range(N_STEPS)]
    print(f"\n{chosen} along the z{heat_dim} walk ({', '.join(f'{a:+.1f}' for a in alphas)}):\n  " + "  ".join(walk))

    out_dir = Path(VIZ_DIR, Path(image_path).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(image_path, out_dir / "heatmap.png")
    save_cell(out_dir / "cell.png", x)
    save_traversals(out_dir / "traversal", x_chA, burst, imgs, masks, alphas, heat_dim, chosen, orig_attrs)
    print(f"\nsaved to {out_dir}/")
