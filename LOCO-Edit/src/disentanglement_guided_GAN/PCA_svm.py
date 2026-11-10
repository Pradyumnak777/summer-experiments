'''
this file will be an attempt to combine both GANspace PCA method and
the SVM direction finding method(intensity for now)

run from src/:
    python -m disentanglement_guided_GAN.PCA_svm                                     #intensity, ext30, random inverted burst frame
    python -m disentanglement_guided_GAN.PCA_svm --k 10 --base "<video>_cell12_f010"
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
from scipy.stats import spearmanr

from disentanglement_guided_GAN import data_utils
from disentanglement_guided_GAN.train_svm import load_data
#same image loading/inversion + burst mask helpers as svm_edit.py, so the plots are directly comparable
from disentanglement_guided_GAN.testing.svm_edit import MODELS, SVM_DIR, get_w, get_burst, get_cell_mask
from disentanglement_semantic_eval.gan_dci import PCA_DIRS_PATH, K_COMPONENTS
from disentanglement_guided.burst_detect import load_spot_model
from disentanglement_guided.vae_viz import mask_track_bounded, detect_each, bound_region, BOUND_MARGIN

import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

#config stuff
OUT_DIR = "disentanglement_guided_GAN/combined_viz"   #-> GANspace/ and SVM_dirs/ under it


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attr", default="intensity", choices=["intensity", "area", "aspect"])
    p.add_argument("--model", default="ext30", choices=list(MODELS), help="which svm version, see svm_edit.MODELS")
    p.add_argument("--k", type=int, default=K_COMPONENTS, help="top-k ganspace directions (gan_dci uses 10)")
    p.add_argument("--base", default=None, help='"<video>_cell12_f010", default: a random already-inverted burst frame')
    p.add_argument("--set", default="train", help="split to look for the pngs in, if the frame isn't inverted yet")
    #raw alpha range (all directions are unit norm), like svm_edit.py, set separately for the svm and the pca directions.
    #the inverted w's spread ~1.7-2.3 (sd) along each of them, the titles also show every step in those sd units
    p.add_argument("--alpha_max_svm", type=float, default=15.0)
    p.add_argument("--alpha_max_pca", type=float, default=7.0)
    p.add_argument("--n_steps", type=int, default=7, help="odd, so alpha=0 (the plain reconstruction) is included")
    p.add_argument("--inv_steps", type=int, default=1000, help="inversion steps, only used if the frame isn't inverted yet")
    p.add_argument("--gpu", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_pca_dirs(k):
    #the ganspace directions saved by GAN/gan_pca_inverted_w.py: pca over 100k w's sampled from the mapping network of the
    #same generator (00010) the burst frames were inverted with. the gan_pca scripts run everything at import time (no
    #functions), so their saved dirs.npz is loaded here instead (the same file gan_dci.py uses)
    d = np.load(PCA_DIRS_PATH)
    var_pct = 100 * d["S"] ** 2 / (d["S"] ** 2).sum()
    return d["Vt"][:k].astype(np.float32), var_pct[:k]   #[k, 512] unit norm rows, % of the sampled w variance each


def walk(G, w, d, alphas):
    #w + alpha * d for every alpha, same nudge at every layer (shared W) -> [n_steps, 2, H, W] in [0, 1]
    ws = (w.unsqueeze(0) + alphas[:, None] * d.unsqueeze(0)).unsqueeze(1).repeat(1, G.num_ws, 1)
    with torch.no_grad():
        return ((G.synthesis(ws, noise_mode='const') + 1) / 2).clamp(0, 1).cpu().numpy()


def plot_walk(target, imgs, col_names, dists, det, suptitle, out_path):
    #same layout as svm_edit.py: rows chA / chB, columns target | steps. det is None (no hand-drawn burst) or a dict with
    #the bounded rule based masks (yellow), the hand-drawn burst (red) and the search region (dashed grey, target only)
    fmt = lambda img, m: f"{img[0][m].mean():.3f}" if m.any() else "-"   #"-" = no burst detected in the region
    cols = [("target", target, None)] + list(zip(col_names, imgs, dists))
    fig, axes = plt.subplots(2, len(cols), figsize=(1.9 * len(cols), 4.8))
    for k, (name, img, dist) in enumerate(cols):
        title = name if dist is None else f"{name}\nsvm dist {dist:+.2f}"
        if det is not None:
            m = det["target"] if k == 0 else det["steps"][k - 1]
            title += f"\nburst chA {fmt(img, m)}"
        for ch in range(2):
            ax = axes[ch, k]
            ax.imshow(img[ch], cmap="gray", vmin=0, vmax=1)
            if ch == 0 and det is not None:
                if k == 0:
                    ax.contour(det["burst"], levels=[0.5], colors="r", linewidths=0.6)
                    ax.contour(det["search"], levels=[0.5], colors="0.6", linewidths=0.5, linestyles="dashed")
                if m.any():
                    ax.contour(m, levels=[0.5], colors="yellow", linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, k].set_title(title, fontsize=7)
    axes[0, 0].set_ylabel("chA", fontsize=8); axes[1, 0].set_ylabel("chB", fontsize=8)
    fig.suptitle(suptitle, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    args = get_args()
    DEVICE = torch.device(f"cuda:{args.gpu}")
    tag = MODELS[args.model]
    os.makedirs(OUT_DIR, exist_ok=True)

    #import the ganspace functionalities from a previous script- check under
    #GAN/gan_pca or gan_pca_inverted..
    Vt, var_pct = load_pca_dirs(args.k)   #[k, 512]

    #after finding the the top-k directions, get teh direction
    #to move W space in for intensity, according to the SVM
    clf = joblib.load(f"{SVM_DIR}/svm_{args.attr}{tag}.joblib")
    n = np.load(f"{SVM_DIR}/boundary_{args.attr}{tag}.npy")[0]   #[512], unit norm, points low -> high
    coef_norm = np.linalg.norm(clf.coef_)

    #as these both are in the same space, 2 things will be done-
    #1. check similarity of intensity/svm direction with all of the found k directions and report it.
    #all directions are unit norm, so the dot product is the cosine similarity. a pca direction's sign is arbitrary
    #(svd can return v or -v), so only |cos| matters. for scale: a random unit direction in 512-d has |cos| ~ 1/sqrt(512)
    #~ 0.044 with any fixed one. sum of cos^2 = how much of the svm direction lies inside the span of the top-k
    cos = Vt @ n   #[k]

    #the inverted burst frames tell how the real data sits along each direction: its spread (std) and the spearman
    #between the position along it and the raw factor
    W, _, F, _, factor_names = load_data(data_utils.SAVE_DIR)
    f_attr = F[:, factor_names.index(args.attr)]
    dirs = [(f"pca{i}", Vt[i]) for i in range(args.k)] + [(f"svm_{args.attr}{tag}", n)]
    sds = {name: float((W @ d).std()) for name, d in dirs}
    rhos = {name: spearmanr(W @ d, f_attr)[0] for name, d in dirs}

    report = [f"svm direction: {args.attr}, model {args.model}  vs  top-{args.k} ganspace directions ({PCA_DIRS_PATH})",
              f"random-direction baseline |cos| ~ {1 / np.sqrt(W.shape[1]):.3f}",
              f"spearman: position along the direction (inverted burst w's, n={len(W)}) vs raw {args.attr}\n",
              " dir  | % var (sampled w) |  cos w/ svm | sd (inverted w) | spearman"]
    for i in range(args.k):
        report.append(f" pca{i:<2d}|      {var_pct[i]:5.2f}%       |   {cos[i]:+.3f}   |      {sds[f'pca{i}']:.3f}      | {rhos[f'pca{i}']:+.3f}")
    svm_name = dirs[-1][0]
    report.append(f" svm  |         -         |   +1.000   |      {sds[svm_name]:.3f}      | {rhos[svm_name]:+.3f}"
                  f"   (incl. the svm's own train frames, so higher than its val spearman)")
    report.append(f"\nfraction of the svm direction inside the top-{args.k} span (sum cos^2): {(cos ** 2).sum():.3f}"
                  f"   (random baseline ~ {args.k / W.shape[1]:.3f})")
    print("\n".join(report))
    report_path = f"{OUT_DIR}/similarity_{args.attr}{tag}_k{args.k}.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"saved {report_path}\n")

    #also, under combined_viz/, and then under GANspace/ and under SVM_dirs/, store the respective perturbations as matplotlib plots for every direction (similar to svm_edit.py)
    base = args.base
    if base is None:
        random.seed(args.seed)
        base = random.choice(sorted(Path(data_utils.SAVE_DIR).glob("*.npz"))).stem
    #one subfolder per image, so runs on different frames don't pile up in the same folder
    gs_dir, svm_dir = f"{OUT_DIR}/{base}/GANspace", f"{OUT_DIR}/{base}/SVM_dirs"
    os.makedirs(gs_dir, exist_ok=True); os.makedirs(svm_dir, exist_ok=True)

    with open(data_utils.PKL, 'rb') as f:
        network_dict = data_utils.legacy.load_network_pkl(f)
    G = network_dict['G_ema'].to(DEVICE).eval().requires_grad_(False)
    D = network_dict['D'].to(DEVICE).eval().requires_grad_(False)   #only used if the frame has to be inverted first

    w, chA_path, chB_path = get_w(base, args.set, args.inv_steps, G, D, DEVICE)
    target = data_utils.load_target(chA_path, chB_path).astype(np.float32) / 255.0   #[2, H, W]
    burst = get_burst(chB_path)
    if burst is not None:   #bounded rule based detection, same as svm_edit.py (needs a hand-drawn burst to bound by)
        cell_mask = get_cell_mask(chA_path)
        spot_model = load_spot_model()
        track_bounded = lambda fr: mask_track_bounded(fr, cell_mask, burst, spot_model)
        target_det = detect_each(target[0:1], track_bounded)[0].astype(bool)
        search = bound_region(burst, cell_mask)

    for k, (name, d) in enumerate(dirs):
        is_svm = k == args.k
        alpha_max = args.alpha_max_svm if is_svm else args.alpha_max_pca
        alphas = np.linspace(-alpha_max, alpha_max, args.n_steps)
        #flip a pca direction so +alpha moves the svm score up too (its sign is arbitrary anyway), so every plot reads
        #"low -> high" in the same direction as the svm one
        sign = 1.0 if is_svm or cos[k] >= 0 else -1.0
        d_t = torch.from_numpy(sign * d).to(DEVICE)
        imgs = walk(G, w, d_t, torch.from_numpy(alphas.astype(np.float32)).to(DEVICE))

        #svm score of every edited w: moving alpha along a pca direction moves it by alpha * |cos|
        w_edit = w.cpu().numpy()[None] + alphas[:, None] * sign * d[None]
        dists = clf.decision_function(w_edit) / coef_norm
        col_names = [f"alpha {a:+.1f} ({a / sds[name]:+.1f} sd)" + (" (recon)" if abs(a) < 1e-6 else "") for a in alphas]

        det = None
        if burst is not None:
            det = {"target": target_det, "steps": detect_each(imgs[:, 0], track_bounded).astype(bool),
                   "burst": burst, "search": search}

        if is_svm:
            head = f"svm direction: {args.attr} (low -> high), model: {args.model}"
            out_path = f"{svm_dir}/{args.attr}{tag}.png"
        else:
            head = (f"ganspace {name} ({var_pct[k]:.2f}% var), cos w/ svm {args.attr} {cos[k]:+.3f}"
                    + (", sign flipped to match the svm" if sign < 0 else ""))
            out_path = f"{gs_dir}/{name}.png"
        plot_walk(target, imgs, col_names, dists, det,
                  f"{head}   |   {base}\nyellow: rule based burst mask within {BOUND_MARGIN} px of the hand-drawn burst "
                  f"(red, dashed = search region), - = none detected", out_path)

    #2. (TODO later)



