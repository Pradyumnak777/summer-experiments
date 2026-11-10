'''
asks the question:

given some point int he W space, let it rest on either side of the SVM plane.
The prediction is currently unimportant to us. Instead, if the normal vector gotten form the plane is applied to that
point and the point is moved in the direciton of the normal vector, does it reach the region of 'higher' intensity?

for every inverted w of a split (the svm's train frames, and its held out cells):
1. where does the recon sit: signed distance of w to the hyperplane, > 0 = bright side, < 0 = dim side
2. push it along the normal, 3 steps down and 3 steps up (--alphas), and generate each one
3. measure the burst intensity on every generated image on its own (bounded detector, same as svm_edit.py)
4. score it against the recon, whichever side the recon was on:
   a push is correct if a down push is dimmer than the recon, or an up push is brighter than the recon
   score = fraction of the 6 pushes that are correct. a direction that does nothing useful sits ~0.5
   spearman (alpha vs intensity, per frame) is kept as a second number: are the stronger pushes also further along?

no burst found on a push = that push failed in "score" (a burst that vanishes might be a real dimming, so
"score_detected" is the same thing only over the pushes that had a burst). a frame whose recon has no burst has nothing
to compare against, so it gets no score

run from src/:
    python -m disentanglement_guided_GAN.custom_metric                                #ext20 svm, train + val, every frame
    python -m disentanglement_guided_GAN.custom_metric --n_frames 16 --workers 8      #quick test
    python -m disentanglement_guided_GAN.custom_metric --random_dir                   #same loop along a random direction (control)
saved as OUT_DIR/intensity_<model>[_random]_<split>.csv (one row per frame) + summary_intensity_<model>[_random].txt
'''
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")   #before numpy, every detector worker would otherwise grab all 96 cores
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import csv
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from disentanglement_guided_GAN import data_utils
from disentanglement_guided_GAN.train_svm import load_data, cell_groups
from disentanglement_guided_GAN.testing.svm_edit import SVM_DIR, get_burst, get_cell_mask
from disentanglement_guided.burst_detect import load_spot_model
from disentanglement_guided.vae_viz import mask_track_bounded, detect_each

import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

#config stuff
OUT_DIR = "disentanglement_guided_GAN/custom_metric_outputs"
ATTR    = "intensity"
CHUNK   = 256   #images per G.synthesis call


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ext20", help="svm file suffix, eg ext20 -> svm_intensity_ext20.joblib")
    #the w's sit within ~[-6, 6] of the boundary, so the biggest push goes about one end of the data to the other
    p.add_argument("--alphas", nargs=3, type=float, default=[2.0, 4.0, 6.0], help="the push sizes, used as both -a and +a")
    p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    #these two have to be what the svm was trained with, or the "held out" cells aren't held out. ext20 was 0.3, seed 0
    p.add_argument("--val_ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_frames", type=int, default=None, help="random subset per split, default: every frame")
    p.add_argument("--random_dir", action="store_true", help="push along a random unit direction instead of the normal")
    p.add_argument("--workers", type=int, default=24, help="cpu processes for the detector")
    p.add_argument("--batch", type=int, default=128, help="frames generated + measured at a time")
    p.add_argument("--gpu", type=int, default=9)
    return p.parse_args()


def init_worker():
    #the detector only runs on the cpu, so the workers never touch the shared gpus
    global SPOT
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    SPOT = load_spot_model()


def measure(item):
    #item: (base name, chA [n_alphas, H, W] in [0, 1]) -> burst intensity per image (nan = no burst found), error or None
    base, chA = item
    nans = np.full(len(chA), np.nan, dtype=np.float32)
    try:
        d = np.load(Path(data_utils.SAVE_DIR) / f"{base}.npz")
        cell_mask = get_cell_mask(Path(str(d["chA_path"])))
        burst = get_burst(Path(str(d["chB_path"])))
        assert burst is not None, "no hand-drawn burst"
        masks = detect_each(chA, lambda f: mask_track_bounded(f, cell_mask, burst, SPOT)).astype(bool)
    except Exception as e:   #one bad frame shouldn't kill a long run, these get counted in the summary
        return nans, f"{base}: {e!r}"
    for i, m in enumerate(masks):
        if m.any():
            nans[i] = chA[i][m].mean()
    return nans, None


@torch.no_grad()
def walk_batch(G, w, n, alphas):
    #w: [B, 512], n: [512], alphas: [n_alphas] -> chA of every w + alpha * n, [B, n_alphas, H, W] in [0, 1]
    ws = (w[:, None] + alphas[None, :, None] * n[None, None]).view(-1, w.shape[1])
    out = []
    for i in range(0, len(ws), CHUNK):
        batch = ws[i:i + CHUNK].unsqueeze(1).repeat(1, G.num_ws, 1)   #same nudge at every layer (shared W)
        out.append(((G.synthesis(batch, noise_mode='const')[:, 0] + 1) / 2).clamp(0, 1).cpu())
    return torch.cat(out).numpy().reshape(len(w), len(alphas), G.img_resolution, G.img_resolution)


def push_scores(I, alphas):
    #I: [N, n_alphas] measured intensity, nan = no burst. every push is compared to that frame's own recon
    zero = np.flatnonzero(alphas == 0)[0]
    I0 = I[:, zero, None]
    down, up = I[:, alphas < 0], I[:, alphas > 0]
    ok_down, ok_up = down < I0, up > I0   #nan compares False, so a missing burst is a failed push
    has_recon = ~np.isnan(I0[:, 0])

    def frac(ok):
        return np.where(has_recon, ok.mean(1), np.nan)

    ok_all = np.concatenate([ok_down, ok_up], axis=1).astype(np.float32)
    detected = ~np.isnan(np.concatenate([down, up], axis=1))
    #only over the pushes that had a burst, frames with none of those stay nan
    n_detected = detected.sum(1)
    score_detected = np.divide((ok_all * detected).sum(1), n_detected, out=np.full(len(I), np.nan), where=n_detected > 0)
    score_detected[~has_recon] = np.nan
    return {"score": frac(ok_all), "down_score": frac(ok_down), "up_score": frac(ok_up), "score_detected": score_detected}


def frame_spearman(I, alphas):
    #alpha vs intensity for each frame on its own. nan if a push had no burst, or nothing changed (a constant has no ranks)
    out = np.full(len(I), np.nan)
    for i, row in enumerate(I):
        if not np.isnan(row).any() and np.ptp(row) > 0:
            out[i] = spearmanr(alphas, row)[0]
    return out


def summarize(split, I, dist0, scores, rho, alphas, n_errors):
    def line(name, m):
        if not m.any():
            return f"  {name:6s} n=0"
        return (f"  {name:6s} n={m.sum():5d} | score {np.nanmean(scores['score'][m]):.3f} | down {np.nanmean(scores['down_score'][m]):.3f} "
                f"| up {np.nanmean(scores['up_score'][m]):.3f} | detected only {np.nanmean(scores['score_detected'][m]):.3f} "
                f"| spearman {np.nanmean(rho[m]):+.3f} (n={(~np.isnan(rho[m])).sum()})")

    zero = np.flatnonzero(alphas == 0)[0]
    has_recon = ~np.isnan(I[:, zero])
    lines = [f"\n{split}: {len(I)} frames, recon has a burst in {has_recon.sum()}, detector errors {n_errors}",
             line("all", has_recon), line("bright", has_recon & (dist0 > 0)), line("dim", has_recon & (dist0 <= 0)),
             f"  {'alpha':>18s} " + " ".join(f"{a:+7.1f}" for a in alphas),
             f"  {'mean intensity':>18s} " + " ".join(f"{v:7.3f}" for v in np.nanmean(I, axis=0)),
             f"  {'no burst found %':>18s} " + " ".join(f"{v:7.1f}" for v in 100 * np.isnan(I).mean(0))]
    return lines


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}")
    tag = f"{ATTR}_{args.model}" + ("_random" if args.random_dir else "")
    os.makedirs(OUT_DIR, exist_ok=True)

    W, _, F, bases, factor_names = load_data(data_utils.SAVE_DIR)
    f_attr = F[:, factor_names.index(ATTR)]
    tr_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed).split(W, groups=cell_groups(bases)))
    split_idx = {"train": tr_idx, "val": val_idx}

    clf = joblib.load(f"{SVM_DIR}/svm_{ATTR}_{args.model}.joblib")
    coef_norm = np.linalg.norm(clf.coef_)
    rng = np.random.default_rng(args.seed)
    if args.random_dir:
        n = rng.standard_normal(W.shape[1]).astype(np.float32)
        n /= np.linalg.norm(n)
    else:
        n = np.load(f"{SVM_DIR}/boundary_{ATTR}_{args.model}.npy")[0]   #[512], unit norm, points low -> high

    with open(data_utils.PKL, 'rb') as f:
        G = data_utils.legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False)

    alphas = np.array(sorted([-a for a in args.alphas] + [0.0] + list(args.alphas)), dtype=np.float32)   #[7], recon in the middle
    alphas_t, n_t = torch.from_numpy(alphas).to(device), torch.from_numpy(n).to(device)
    summary = [f"direction: {'random unit vector' if args.random_dir else 'svm normal'}, svm {ATTR} {args.model}, alphas {alphas.tolist()}",
               "score = fraction of the 6 pushes that move the intensity the right way compared to the recon (~0.5 = no useful effect)"]

    with get_context("spawn").Pool(args.workers, initializer=init_worker) as pool:
        for split in args.splits:
            idx = split_idx[split]
            if args.n_frames is not None:
                idx = np.sort(rng.choice(idx, min(args.n_frames, len(idx)), replace=False))

            I, n_errors = [], 0
            for start in tqdm(range(0, len(idx), args.batch), desc=f"{split}: walking + measuring"):
                batch = idx[start:start + args.batch]
                chA = walk_batch(G, torch.from_numpy(W[batch]).to(device), n_t, alphas_t)   #[B, 7, H, W]
                for vals, err in pool.map(measure, list(zip(bases[batch], chA)), chunksize=1):
                    I.append(vals)
                    if err is not None:
                        n_errors += 1
                        print(f"\ndetector error, {err}")
            I = np.stack(I)   #[N, 7]

            #where the svm puts the recon and every push (>0 = bright side). for the normal this is just dist0 + alpha
            dist = np.stack([clf.decision_function(W[idx] + a * n) / coef_norm for a in alphas], axis=1)   #[N, 7]
            dist0 = dist[:, np.flatnonzero(alphas == 0)[0]]
            scores = push_scores(I, alphas)
            rho = frame_spearman(I, alphas)

            header = (["base", "split", "real_intensity", "recon_side"] + [f"intensity_a{a:+g}" for a in alphas]
                      + [f"svm_dist_a{a:+g}" for a in alphas] + ["score", "down_score", "up_score", "score_detected", "spearman"])
            csv_path = f"{OUT_DIR}/{tag}_{split}.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i, j in enumerate(idx):
                    writer.writerow([bases[j], split, f"{f_attr[j]:.4f}", "bright" if dist0[i] > 0 else "dim"]
                                    + [f"{v:.4f}" for v in I[i]] + [f"{v:.3f}" for v in dist[i]]
                                    + [f"{scores[k][i]:.4f}" for k in ("score", "down_score", "up_score", "score_detected")]
                                    + [f"{rho[i]:.4f}"])
            print(f"saved {csv_path}")
            summary += summarize(split, I, dist0, scores, rho, alphas, n_errors)

    print("\n".join(summary))
    with open(f"{OUT_DIR}/summary_{tag}.txt", "w") as f:
        f.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
