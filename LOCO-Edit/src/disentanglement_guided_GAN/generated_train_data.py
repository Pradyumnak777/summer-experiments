'''
1. test whether the GAN generation is adequate. Also perform factor calculation on it. This will be the first step
2. Run a loop over that setup to generate around 100k samples. Then, slect a threshold, split into extremes, and train SVM on this.

this file generates the dataset for 2: random w -> generated chA + chB -> cellpose cell mask -> rule based burst mask ->
the same factors as the inverted w's (intensity, log area, log aspect), measured on the generated sample itself

saved as OUT_DIR/<seed, 6 digits>/ with
    <seed>_chA.png, <seed>_chB.png   the generated images as they came out of G, 8 bit, background not removed
    <seed>.npz                       w [512] + factors, same fields as data_utils.py, plus the masks (see save_sample)
the seed fixes the z, so a folder's sample can always be regenerated. seeds with no cell or no burst get no folder, so the
numbers have gaps. the labels are not saved: an extreme split needs the thresholds of the whole set, so train makes them

safe to stop and rerun, it carries on after the highest seed already saved

run from src/:
    python -m disentanglement_guided_GAN.generated_train_data                        #100k samples on cuda:9
    python -m disentanglement_guided_GAN.generated_train_data --n_samples 64         #quick test
    python -m disentanglement_guided_GAN.generated_train_data --inspect_seed 3       #just the figure of one sample
'''
import os
os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", "/scratch/pbk5339/caches/cellpose")   #before importing cellpose, default is ~/.cellpose
os.environ.setdefault("OMP_NUM_THREADS", "2")   #before numpy, every detector worker would otherwise grab all 96 cores
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import shutil
from multiprocessing import get_context
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   #no display on the server
import matplotlib.pyplot as plt
import numpy as np
import torch
from cellpose.models import CellposeModel
from PIL import Image
from skimage.measure import regionprops
from tqdm import tqdm

from disentanglement_guided_GAN import data_utils   #reuses PKL and legacy (the stylegan loader)
from disentanglement_guided.burst_detect import load_spot_model
#mask_track is the unbounded detector (no hand-drawn burst needed), same one mask_track_bounded wraps
from disentanglement_guided.vae_viz import mask_track, detect_each
from disentanglement_semantic_eval.gan_dci import BURST_FACTORS, ASPECT_MAX

#config stuff
OUT_DIR     = "GAN/gan_generated_ws_gene_burst_masked"
INSPECT_DIR = "disentanglement_guided_GAN/generated_cells"
PSI         = 1.0   #no truncation, so the samples cover the full distribution the gan learned


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=100000, help="samples to end up with, the ones already saved count")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=24, help="cpu processes for the burst detector")
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--inspect_seed", type=int, default=None, help="only save the figure of this one seed, nothing else")
    p.add_argument("--gpu", type=int, default=9)
    return p.parse_args()


def segment_cell(img, model):
    #img: [2, H, W] in [0, 1] -> [H, W] bool mask of the main cell, None if it finds no cell
    #cellpose runs on each channel on its own, the largest region over both runs is kept (so the partial blobs at the edges
    #get dropped, and a cell missed on one channel can still come from the other). the mask then gets applied to both channels
    best = None
    for ch in img:
        #.copy() as cellpose normalizes the array in place, which would change the chA the burst is measured on
        labels = model.eval(ch.copy())[0]   #[H, W], 0 = background, 1..n = cells
        if labels.max() == 0:
            continue
        mask = labels == 1 + np.argmax(np.bincount(labels.ravel())[1:])   #the largest region of this channel
        if best is None or mask.sum() > best.sum():
            best = mask
    return best


def detect_burst(chA, cell_mask, spot_model):
    #rule based detection without a hand-drawn burst to bound it, so it searches the whole cell
    #chA: [H, W] in [0, 1], cell_mask: [H, W] bool -> [H, W] bool burst mask, empty = no burst found
    cell_mask = cell_mask.astype(np.float32)
    return detect_each(chA[None], lambda f: mask_track(f, cell_mask, spot_model))[0].astype(bool)


def burst_factors(chA, burst):
    #same factors as gan_dci.get_burst_factors, in BURST_FACTORS order. the images are already at the model's 128 px, so the
    #area needs no rescaling. chA: [H, W] in [0, 1], burst: [H, W] bool, not empty -> [3] (intensity, log area, log aspect)
    p = regionprops(burst.astype(np.uint8))[0]   #all burst pixels as one region
    aspect = np.clip(p.major_axis_length / max(p.minor_axis_length, 1e-6), 1.0, ASPECT_MAX)
    return np.array([chA[burst].mean(), np.log(p.area), np.log(aspect)], dtype=np.float32)


def generate(G, seeds, device):
    #seeds: list of ints -> w [B, 512], images [B, 2, H, W] in uint8. each z comes from its own seed, so a sample doesn't
    #depend on the batch it was generated in
    z = np.stack([np.random.RandomState(int(s)).randn(G.z_dim) for s in seeds])
    z = torch.from_numpy(z).float().to(device)
    with torch.no_grad():
        ws = G.mapping(z, None, truncation_psi=PSI)   #[B, num_ws, 512], all rows identical
        imgs = ((G.synthesis(ws, noise_mode='const') + 1) / 2).clamp(0, 1)
    return ws[:, 0].cpu().numpy(), (imgs * 255).round().to(torch.uint8).cpu().numpy()


def save_sample(out_dir, seed, w, img8, cell_mask, burst, factors):
    #written to a hidden folder and renamed at the end, so a run that gets killed can't leave a half written sample
    base = f"{seed:06d}"
    tmp, final = Path(out_dir) / f".tmp_{base}", Path(out_dir) / base
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    Image.fromarray(img8[0]).save(tmp / f"{base}_chA.png")
    Image.fromarray(img8[1]).save(tmp / f"{base}_chB.png")
    np.savez_compressed(
        tmp / f"{base}.npz",
        w=w,                                     #[512]
        base=base,
        seed=seed,
        chA_path=str(final / f"{base}_chA.png"),
        chB_path=str(final / f"{base}_chB.png"),
        factor_names=np.array(BURST_FACTORS),
        factors=factors,                         #[3] raw values (intensity, log area, log aspect)
        cell_mask=cell_mask,                     #[H, W] bool, what the burst was searched in
        burst_mask=burst,                        #[H, W] bool, what the factors were measured on
        #cells cut off by the frame (a partial neighbour can win the largest region), so these can be filtered at train time
        touches_border=bool(cell_mask[0].any() or cell_mask[-1].any() or cell_mask[:, 0].any() or cell_mask[:, -1].any()),
    )
    os.rename(tmp, final)


def init_worker():
    #the detector only runs on the cpu, so the workers never touch the shared gpus
    global SPOT
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    SPOT = load_spot_model()


def finish_sample(item):
    #runs in a worker: burst detection + factors + saving. -> "saved", "no_burst", or the error text
    out_dir, seed, w, img8, cell_mask = item
    try:
        chA = img8[0].astype(np.float32) / 255.0   #measured on the same 8 bit values the png stores
        burst = detect_burst(chA * cell_mask, cell_mask, SPOT)
        if not burst.any():
            return "no_burst"
        save_sample(out_dir, seed, w, img8, cell_mask, burst, burst_factors(chA, burst))
        return "saved"
    except Exception as e:   #one bad sample shouldn't kill a long run
        return f"seed {seed}: {e!r}"


def inspect_seed(seed, G, model, device):
    #one sample's figure, to look at it by eye. columns: generated | cell mask applied. red = cell outline, yellow = detected burst
    os.makedirs(INSPECT_DIR, exist_ok=True)
    _, img8 = generate(G, [seed], device)
    img = img8[0].astype(np.float32) / 255.0
    cell_mask = segment_cell(img, model)
    assert cell_mask is not None, f"cellpose found no cell for seed {seed}"
    segmented = img * cell_mask          #[2, H, W], everything outside the main cell is 0

    burst = detect_burst(segmented[0], cell_mask, load_spot_model())
    print(f"cell covers {100 * cell_mask.mean():.1f}% of the frame")
    print(f"burst: {int(burst.sum())} px, mean chA {segmented[0][burst].mean():.3f}" if burst.any() else "burst: none found")

    fig, axes = plt.subplots(2, 2, figsize=(7, 7.3))
    for row, name in enumerate(["chA (gene burst)", "chB"]):
        axes[row, 0].imshow(img[row], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 0].contour(cell_mask, levels=[0.5], colors="r", linewidths=0.7)
        axes[row, 1].imshow(segmented[row], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 0].set_ylabel(name, fontsize=9)
    if burst.any():
        axes[0, 1].contour(burst, levels=[0.5], colors="yellow", linewidths=0.7)
    axes[0, 0].set_title("generated, cell outline", fontsize=9)
    axes[0, 1].set_title("cell mask applied, detected burst", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"generated cell, seed {seed}, psi {PSI}", fontsize=9)
    fig.tight_layout()
    out_path = f"{INSPECT_DIR}/gen_seed{seed}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}")

    with open(data_utils.PKL, 'rb') as f:
        G = data_utils.legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False)
    model = CellposeModel(gpu=True, device=device)

    if args.inspect_seed is not None:
        inspect_seed(args.inspect_seed, G, model, device)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for tmp in Path(args.out_dir).glob(".tmp_*"):   #left by a run that got killed mid save
        shutil.rmtree(tmp)
    saved_seeds = [int(p.name) for p in Path(args.out_dir).iterdir() if p.name.isdigit()]
    n_saved = len(saved_seeds)
    seed = max(saved_seeds) + 1 if saved_seeds else 0
    print(f"{n_saved} samples already saved in {args.out_dir}, next seed {seed}, target {args.n_samples}")

    counts = {"no_cell": 0, "no_burst": 0, "errors": 0}
    with get_context("spawn").Pool(args.workers, initializer=init_worker) as pool:
        pbar = tqdm(total=args.n_samples, initial=n_saved, desc="samples saved", smoothing=0.05)
        while n_saved < args.n_samples:
            seeds = list(range(seed, seed + min(args.batch, args.n_samples - n_saved)))
            seed = seeds[-1] + 1
            ws, imgs = generate(G, seeds, device)

            items = []
            for s, w, img8 in zip(seeds, ws, imgs):
                cell_mask = segment_cell(img8.astype(np.float32) / 255.0, model)
                if cell_mask is None:
                    counts["no_cell"] += 1
                else:
                    items.append((args.out_dir, s, w, img8, cell_mask))

            n_new = 0
            for result in pool.map(finish_sample, items, chunksize=1):
                if result == "saved":
                    n_new += 1
                elif result == "no_burst":
                    counts["no_burst"] += 1
                else:
                    counts["errors"] += 1
                    tqdm.write(f"error, {result}")
            n_saved += n_new
            pbar.update(n_new)
            pbar.set_postfix(seed=seed, **counts)
        pbar.close()

    print(f"\ndone: {n_saved} samples in {args.out_dir}, this run: tried up to seed {seed - 1}, {counts}")


if __name__ == "__main__":
    main()
