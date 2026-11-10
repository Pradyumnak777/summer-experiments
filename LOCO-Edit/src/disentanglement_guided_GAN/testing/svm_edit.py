'''
apply one of the interfaceGAN svm directions (disentanglement_guided_GAN/svm_models/) to an image, plot the edit

the image is picked by its base name ("<video>_cell12_f010"). if data_utils.py already inverted it (burst frames),
its saved w is used, otherwise it gets inverted here first (~1 min on one gpu at 1000 steps)

the burst intensity in the titles is measured like row 3 of disentanglement_guided/vae_viz.py: the rule based detector
(christina's mask_track) re-run on every image on its own, but bounded to within BOUND_MARGIN px of the hand-drawn burst
(mask_track_bounded), so each edit step is measured on ITS OWN burst instead of the original hand-drawn outline

the last column is the gan heatmap of the direction: |J n| at the original w (jacobian_gen in gradcam/GAN_grad_viz_pca.py),
ie how much every output pixel moves for a small step along n, overlaid on the recon like that script does

run from src/:
    python -m disentanglement_guided_GAN.testing.svm_edit                                      #all 3 attrs, ext30, random inverted burst frame
    python -m disentanglement_guided_GAN.testing.svm_edit --base "<video>_cell12_f010"
    python -m disentanglement_guided_GAN.testing.svm_edit --attrs area --model median        #just one of them
saved as OUT_DIR/<base>/<attr><model suffix>.png, eg outputs_svm_new/<base>/intensity_ext30.png
'''
import argparse
import os
import random
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from disentanglement_guided_GAN import data_utils   #reuses load_target, PKL and the inversion functions
from disentanglement_semantic_eval.gan_dci import burst_mask_path, ASPECT_MAX
from skimage.measure import regionprops
#same bounded rule based detection as vae_viz.py, imported so the gan and vae numbers are measured the same way
from disentanglement_guided.burst_detect import load_spot_model
from disentanglement_guided.vae_viz import mask_track_bounded, detect_each, bound_region, BOUND_MARGIN

import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

#config stuff
SVM_DIR = "disentanglement_guided_GAN/svm_models"
# OUT_DIR = "disentanglement_guided_GAN/testing/outputs_svm"   #each model's plots go in OUT_DIR/<model>/
OUT_DIR = "disentanglement_guided_GAN/testing/outputs_svm_new"   #one subfolder per image, OUT_DIR/<base>/, every attr's plot in it
ATTRS   = ["intensity", "area", "aspect"]
SIZE    = 128

#svm models to choose from (--model) -> their file suffix in SVM_DIR, eg ext30 -> svm_intensity_ext30.joblib
#add new ones here when train_svm.py saves another version
MODELS  = {
    "median": "",          #train_boundary, median split
    "ext30":  "_ext30",    #train_boundary_2, top/bottom 30%
}


def get_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--attr", default="intensity", choices=["intensity", "area", "aspect"])
    p.add_argument("--attrs", nargs="+", default=ATTRS, choices=ATTRS, help="one plot per attr, default: all 3")
    p.add_argument("--model", default="ext30", choices=list(MODELS), help="which svm version, see MODELS")
    p.add_argument("--base", default=None, help='"<video>_cell12_f010", default: a random already-inverted burst frame')
    p.add_argument("--set", default="train", help="split to look for the pngs in, if the frame isn't inverted yet")
    #the directions are unit norm, w's spread ~2 along them (std of w @ n) and sit within ~[-6, 6] of the boundary,
    #so +-6 walks from one end of the data to the other
    p.add_argument("--alpha_max", type=float, default=10.0)
    p.add_argument("--n_steps", type=int, default=7, help="odd, so alpha=0 (the plain reconstruction) is included")
    p.add_argument("--inv_steps", type=int, default=1000, help="inversion steps, only used if the frame isn't inverted yet")
    p.add_argument("--gpu", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def get_w(base, set_name, inv_steps, G, D, device):
    #saved w if data_utils.py already inverted this frame, otherwise invert it now
    npz = Path(data_utils.SAVE_DIR) / f"{base}.npz"
    if npz.exists():
        d = np.load(npz)
        return torch.from_numpy(d["w"]).to(device), Path(str(d["chA_path"])), Path(str(d["chB_path"]))

    chA_path = Path(f"data/singlecell_chA_split/{set_name}/{base}_chA.png")
    chB_path = Path(f"data/singlecell_chB_split/{set_name}/{base}_chB.png")
    assert chA_path.exists(), f"no saved w and no png for {base} in {set_name}"
    print(f"{base} isn't inverted yet, inverting ({inv_steps} steps)..")
    w_avg, w_std = data_utils.compute_w_stats(G, device)
    target = torch.from_numpy(data_utils.load_target(chA_path, chB_path)).unsqueeze(0)   #[1, 2, H, W]
    ws = data_utils.project_batch(G, D, target, w_avg, w_std, num_steps=inv_steps, device=device, verbose=True)
    return ws[0, 0], chA_path, chB_path   #[512], the shared w


def get_burst(chB_path):
    #hand-drawn burst mask at 128 (nearest, like the dataset's mask_transform), None if this frame has none
    try:
        path = burst_mask_path(chB_path)
    except KeyError:   #cell was never annotated
        return None
    if not path.exists():
        return None
    burst = np.array(Image.open(path).convert('L').resize((SIZE, SIZE), resample=Image.NEAREST)) > 0
    return burst if burst.any() else None


def get_cell_mask(chA_path):
    #binary cell mask at 128 (nearest, like the dataset's mask_transform), mask_track only looks inside it
    base = chA_path.name[:-len("_chA.png")]
    mask = Image.open(Path(data_utils.MASK_DIR) / f"{base}_mask.png").convert('L').resize((SIZE, SIZE), resample=Image.NEAREST)
    return (np.array(mask) > 0).astype(np.float32)


def burst_value(chA, m, attr):
    #the quantity the direction is about, measured inside a burst mask m like gan_dci.get_burst_factors, but not logged
    #(area/aspect are logged there for the svm labels, the raw values are easier to read, the ordering is the same).
    #chA: [H, W] in [0, 1] at 128 px, m: [H, W] bool. "-" = no burst detected in the region
    if not m.any():
        return "-"
    if attr == "intensity":
        return f"{chA[m].mean():.3f}"
    if attr == "area":
        return f"{int(m.sum())} px"   #already at the 128 px model resolution, no rescale needed
    p = regionprops(m.astype(np.uint8))[0]   #all burst pixels as one region
    return f"{np.clip(p.major_axis_length / max(p.minor_axis_length, 1e-6), 1.0, ASPECT_MAX):.2f}"


def heat_overlay(gray, heat, alpha=0.6):
    #same overlay as GAN_grad_viz_pca.py: heat min-maxed to [0, 1], jet blended in with an opacity that scales with the
    #heat, so low-sensitivity areas keep the image instead of a flat blue wash. gray, heat: [H, W] -> [H, W, 3]
    h = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    a = alpha * h[..., None]
    return ((1 - a) * gray[..., None] + a * plt.get_cmap("jet")(h)[..., :3]).clip(0, 1)


if __name__ == "__main__":
    args = get_args()
    DEVICE = torch.device(f"cuda:{args.gpu}")
    #imported here, not at the top: GAN_grad_viz_pca.py imports get_w from this file, so a top level import is circular
    from gradcam.GAN_grad_viz_pca import jacobian_gen
    tag = MODELS[args.model]
    # out_dir = f"{OUT_DIR}/{args.model}"
    # os.makedirs(out_dir, exist_ok=True)

    base = args.base
    if base is None:
        random.seed(args.seed)
        base = random.choice(sorted(Path(data_utils.SAVE_DIR).glob("*.npz"))).stem
    #one subfolder per image, every attr's plot goes in it (the model suffix is in the file name)
    out_dir = f"{OUT_DIR}/{base}"
    os.makedirs(out_dir, exist_ok=True)

    with open(data_utils.PKL, 'rb') as f:
        network_dict = data_utils.legacy.load_network_pkl(f)
    G = network_dict['G_ema'].to(DEVICE).eval().requires_grad_(False)
    D = network_dict['D'].to(DEVICE).eval().requires_grad_(False)

    w, chA_path, chB_path = get_w(base, args.set, args.inv_steps, G, D, DEVICE)

    #everything that only depends on the image (not on the direction) is done once, before the attr loop
    target = data_utils.load_target(chA_path, chB_path).astype(np.float32) / 255.0   #[2, H, W]
    burst = get_burst(chB_path)

    #bounded rule based burst masks (row 3 of vae_viz.py): detected on chA (the gene burst channel) of every image on its
    #own, only within BOUND_MARGIN px of the hand-drawn burst. the edit steps can move/resize the burst, so each one is
    #measured inside its own detected mask instead of the target's hand-drawn outline. needs a hand-drawn burst to bound by
    if burst is not None:
        cell_mask = get_cell_mask(chA_path)
        spot_model = load_spot_model()                                    #spotiflow, loaded once
        track_bounded = lambda f: mask_track_bounded(f, cell_mask, burst, spot_model)
        target_det = detect_each(target[0:1], track_bounded)[0].astype(bool)   #[H, W]
        search = bound_region(burst, cell_mask)                               #where the detector was allowed to look
        target_iou = (target_det & burst).sum() / max((target_det | burst).sum(), 1)
    # fmt = lambda img, m: f"{img[0][m].mean():.3f}" if m.any() else "-"     #"-" = no burst detected in the region

    #walk along the direction: w + alpha * n, same nudge at every layer (shared W)
    alphas = torch.linspace(-args.alpha_max, args.alpha_max, args.n_steps, device=DEVICE)

    for attr in args.attrs:
        #the svm direction (unit norm, points low -> high) + the full model for the signed distance to the boundary
        # clf = joblib.load(f"{SVM_DIR}/svm_{args.attr}{tag}.joblib")
        # n = torch.from_numpy(np.load(f"{SVM_DIR}/boundary_{args.attr}{tag}.npy")[0]).to(DEVICE)   #[512]
        clf = joblib.load(f"{SVM_DIR}/svm_{attr}{tag}.joblib")
        n = torch.from_numpy(np.load(f"{SVM_DIR}/boundary_{attr}{tag}.npy")[0]).to(DEVICE)   #[512]
        dist0 = float(clf.decision_function(w.cpu().numpy()[None])[0] / np.linalg.norm(clf.coef_))
        #signed distance: > 0 -> high side of the hyperplane, < 0 -> low side. crosses over at alpha = -dist0
        print(f"{attr}: original w: dist {dist0:+.2f} -> {'high' if dist0 > 0 else 'low'} {attr} side "
              f"(crosses the boundary at alpha {-dist0:+.2f})")

        ws = (w.unsqueeze(0) + alphas[:, None] * n.unsqueeze(0)).unsqueeze(1).repeat(1, G.num_ws, 1)   #[n_steps, num_ws, 512]
        with torch.no_grad():
            imgs = ((G.synthesis(ws, noise_mode='const') + 1) / 2).clamp(0, 1).cpu().numpy()   #[n_steps, 2, H, W] in [0, 1]

        if burst is not None:
            step_dets = detect_each(imgs[:, 0], track_bounded).astype(bool)        #[n_steps, H, W]

        #gan heatmap: |J n| at the original w, per channel. [1, 512] -> the shared W branch of jacobian_gen
        w_in = w.unsqueeze(0).float().detach().requires_grad_(True)
        jvp_flat, _ = jacobian_gen(G, w_in, n)
        heat = jvp_flat.view(2, SIZE, SIZE).abs().detach().cpu().numpy()       #[2, H, W]

        #plot: rows chA / chB, columns target | edits. on chA: the bounded detected burst in yellow (every column), plus the
        #hand-drawn burst in red and the search region dashed grey on the target. titles carry the signed distance to the
        #boundary and the mean chA inside that column's own bounded mask
        cols = [("target", target, None)] + [(f"alpha {a:+.1f}" + (" (recon)" if abs(a) < 1e-6 else ""), imgs[i], dist0 + a)
                                             for i, a in enumerate(alphas.tolist())]
        # fig, axes = plt.subplots(2, len(cols), figsize=(1.9 * len(cols), 4.6))
        fig, axes = plt.subplots(2, len(cols) + 1, figsize=(1.9 * (len(cols) + 1), 4.6))   #+1 for the gan heatmap
        for k, (name, img, dist) in enumerate(cols):
            title = name if dist is None else f"{name}\ndist {dist:+.2f}"
            if burst is not None:
                det = target_det if k == 0 else step_dets[k - 1]
                # title += f"\nburst chA {fmt(img, det)}" + (f"\nIoU w/ hand-drawn {target_iou:.2f}" if k == 0 else "")
                #the direction's own quantity (intensity / area / aspect), measured inside that column's bounded mask
                title += f"\nburst {attr} {burst_value(img[0], det, attr)}" + (f"\nIoU w/ hand-drawn {target_iou:.2f}" if k == 0 else "")
            for ch in range(2):
                ax = axes[ch, k]
                ax.imshow(img[ch], cmap="gray", vmin=0, vmax=1)
                if ch == 0 and burst is not None:
                    if k == 0:
                        ax.contour(burst, levels=[0.5], colors="r", linewidths=0.6)
                        ax.contour(search, levels=[0.5], colors="0.6", linewidths=0.5, linestyles="dashed")
                    if det.any():
                        ax.contour(det, levels=[0.5], colors="yellow", linewidths=0.6)
                ax.set_xticks([]); ax.set_yticks([])
            axes[0, k].set_title(title, fontsize=7)

        #last column: the gan heatmap over the recon (the image at w, where the jacobian is taken), no mask outlines.
        #the energy (sum of |J n|) is before the min-max, so it's the number to compare across directions, and the share of
        #chA's energy inside the burst vs the burst's share of the image says if the direction is local to it
        recon = imgs[args.n_steps // 2]
        heat_title = f"GAN heatmap |J n| at recon\nenergy chA {heat[0].sum():.1f} | chB {heat[1].sum():.1f}"
        if burst is not None:
            heat_title += f"\nchA in burst {100 * heat[0][burst].sum() / heat[0].sum():.1f}% (area {100 * burst.mean():.1f}%)"
        for ch in range(2):
            ax = axes[ch, -1]
            ax.imshow(heat_overlay(recon[ch], heat[ch]))
            # if ch == 0 and burst is not None:
            #     ax.contour(burst, levels=[0.5], colors="r", linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, -1].set_title(heat_title, fontsize=7)

        axes[0, 0].set_ylabel("chA", fontsize=8); axes[1, 0].set_ylabel("chB", fontsize=8)
        # fig.suptitle(f"svm direction: {args.attr} (low -> high), model: {args.model}   |   {base}\n"
        fig.suptitle(f"svm direction: {attr} (low -> high), model: {args.model}   |   {base}\n"
                     f"yellow: rule based burst mask, detected per image within {BOUND_MARGIN} px of the hand-drawn burst "
                     f"(red, dashed = search region), - = none detected\n"
                     f"last column: |J n| at the original w (gradcam/GAN_grad_viz_pca.py), min-max per channel", fontsize=8)
        fig.tight_layout()

        # out_path = f"{out_dir}/{args.attr}_{base}.png"
        out_path = f"{out_dir}/{attr}{tag}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)   #3 figures per run now, so each one is closed instead of plt.show()
        print(f"saved {out_path}")
    # if matplotlib.get_backend().lower() != "agg":   #no display on the server -> just the saved png
    #     plt.show()
