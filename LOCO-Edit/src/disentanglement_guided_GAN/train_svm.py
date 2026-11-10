'''
paper reference: interfaceGAN
using the stored files along with emtadata(labels of chosen attributes),
find the SVM directions from "no" to "yes" for that specific attribute..

save these svm models under 'disentanglement_guided_GAN/svm_models/'

To run:
python -m disentanglement_guided_GAN.train_svm
'''
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC
from tqdm import tqdm

import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

#run from src/: python -m disentanglement_guided_GAN.train_svm

#config stuff
DATA_DIR  = "GAN/gan_inverted_ws_gene_burst_masked_train"   #written by data_utils.py, one .npz per burst frame
SAVE_DIR  = "disentanglement_guided_GAN/svm_models"
VAL_RATIO = 0.3     #interfaceGAN uses a 70/30 train/val split
C         = 1.0     #svm regularization
SEED      = 0
KEEP_RATIO = 0.2    #extreme split: train on the top/bottom KEEP_RATIO*100% of each factor, the middle is dropped (in this case, its 80%)
TAG       = f"_ext{int(KEEP_RATIO * 100)}"   #suffix for the extreme split files, so the median ones aren't overwritten


def load_data(data_dir):
    #stack every saved frame -> W [N, 512], median labels [N, n_attr], raw factors [N, n_attr], base names [N]
    files = sorted(Path(data_dir).glob("*.npz"))
    W, Y, F, bases = [], [], [], []
    for f in tqdm(files, desc="loading inverted w's"):
        d = np.load(f)
        W.append(d["w"])
        Y.append(d["labels"])
        F.append(d["factors"])   #raw values, the extreme split makes its own labels from these
        bases.append(str(d["base"]))
    factor_names = [str(n) for n in d["factor_names"]]   #same in every file
    return np.stack(W), np.stack(Y), np.stack(F), np.array(bases), factor_names


def cell_groups(bases):
    #"<video>_cell12_f010" -> "<video>_cell12". frames of the same cell are near duplicates, so a random frame split
    #would put almost the same image in train and val and inflate the val accuracy -> split by cell instead
    return np.array([b.rsplit("_f", 1)[0] for b in bases])


def train_boundary(W_tr, y_tr, W_val, y_val):
    #linear svm on the raw w's (no standardizing, the normal has to live in the same space we edit in)
    #dual=False is the faster solver when n_samples > n_features (~9k vs 512)
    clf = LinearSVC(C=C, dual=False, max_iter=10000, random_state=SEED)
    clf.fit(W_tr, y_tr)
    train_acc = clf.score(W_tr, y_tr)
    val_acc = clf.score(W_val, y_val)

    #the hyperplane normal points from label 0 ("no"/low) to label 1 ("yes"/high), unit norm so an edit
    #w + alpha * n moves w by exactly alpha. same [1, 512] shape as interfaceGAN's boundary files
    n = clf.coef_ / np.linalg.norm(clf.coef_)
    return clf, n.astype(np.float32), train_acc, val_acc


def train_boundary_2(W_tr, f_tr, W_val, f_val, y_val_median, groups_tr, keep_ratio=KEEP_RATIO): #this is using extreme split instead of median
    #frames near the median get opposite labels while looking the same, so only the clear cases are used (like interfaceGAN,
    #which keeps the top/bottom 2%): label 1 = top keep_ratio of the factor, 0 = bottom keep_ratio, the middle is dropped
    #thresholds come from the train frames only, the val extremes are cut with the same thresholds
    lo, hi = np.quantile(f_tr, keep_ratio), np.quantile(f_tr, 1 - keep_ratio)
    tr_keep, val_keep = (f_tr <= lo) | (f_tr >= hi), (f_val <= lo) | (f_val >= hi)
    y_tr, y_val = (f_tr[tr_keep] >= hi).astype(np.int64), (f_val[val_keep] >= hi).astype(np.int64)

    clf = LinearSVC(C=C, dual=False, max_iter=10000, random_state=SEED)
    clf.fit(W_tr[tr_keep], y_tr)

    #accuracy on the extremes is inflated (it's an easier task), so it can't be compared to the median split's val acc.
    #the two other numbers use ALL val frames: the median split acc (same task as train_boundary), and the spearman
    #(rank) correlation between the svm score and the raw factor, the most honest single number
    stats = {
        "train_acc_ext": clf.score(W_tr[tr_keep], y_tr),
        "val_acc_ext": clf.score(W_val[val_keep], y_val),
        "val_acc_median": clf.score(W_val, y_val_median),
        "val_spearman": spearmanr(clf.decision_function(W_val), f_val)[0],
        "n_train": int(tr_keep.sum()),
        #the factor values each class actually covers, in the train frames
        "range_low": (float(f_tr.min()), float(lo)),
        "range_high": (float(hi), float(f_tr.max())),
        #if a class is made of only a few cells/videos, the svm can just learn which cell/video it is
        "cells_low": len(np.unique(groups_tr[tr_keep][y_tr == 0])),
        "cells_high": len(np.unique(groups_tr[tr_keep][y_tr == 1])),
        "videos_low": len(np.unique([g.rsplit("_cell", 1)[0] for g in groups_tr[tr_keep][y_tr == 0]])),
        "videos_high": len(np.unique([g.rsplit("_cell", 1)[0] for g in groups_tr[tr_keep][y_tr == 1]])),
    }

    #same as train_boundary: unit normal, points from low (0) to high (1)
    n = clf.coef_ / np.linalg.norm(clf.coef_)
    return clf, n.astype(np.float32), stats


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    W, Y, F, bases, factor_names = load_data(DATA_DIR)
    groups = cell_groups(bases)
    print(f"{len(W)} frames from {len(np.unique(groups))} cells, w dim {W.shape[1]}, attributes: {factor_names}")

    #one split by cell, shared by every attribute
    tr_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED).split(W, groups=groups))
    print(f"train: {len(tr_idx)} frames, val: {len(val_idx)} frames (no cell in both)\n")

    boundaries, summary = {}, []

    #median split (train_boundary), files without a suffix
    # for k, name in enumerate(factor_names):
    #     y = Y[:, k]
    #     clf, n, train_acc, val_acc = train_boundary(W[tr_idx], y[tr_idx], W[val_idx], y[val_idx])
    #     boundaries[name] = n
    #
    #     joblib.dump(clf, f"{SAVE_DIR}/svm_{name}.joblib")    #full model (coef_ + intercept_)
    #     np.save(f"{SAVE_DIR}/boundary_{name}.npy", n)       #just the unit direction, [1, 512]
    #
    #     line = (f"{name:9s} | train acc {train_acc:.3f} | val acc {val_acc:.3f} "
    #             f"| val high/total {y[val_idx].sum()}/{len(val_idx)}")
    #     print(line)
    #     summary.append(line)

    #extreme split (train_boundary_2), files get the TAG suffix, eg svm_intensity_ext30.joblib
    summary.append(f"extreme split: top/bottom {int(KEEP_RATIO * 100)}% of each factor (train thresholds)")
    for k, name in enumerate(factor_names):
        clf, n, s = train_boundary_2(W[tr_idx], F[tr_idx, k], W[val_idx], F[val_idx, k], Y[val_idx, k], groups[tr_idx])
        boundaries[name] = n

        joblib.dump(clf, f"{SAVE_DIR}/svm_{name}{TAG}.joblib")
        np.save(f"{SAVE_DIR}/boundary_{name}{TAG}.npy", n)

        summary.append(f"\n{name}  (trained on {s['n_train']} frames)")
        unit = "" if name == "intensity" else "  (log)"   #area and aspect are logged in get_burst_factors
        summary.append(f"  train factor range  low [{s['range_low'][0]:.3f}, {s['range_low'][1]:.3f}] | high [{s['range_high'][0]:.3f}, {s['range_high'][1]:.3f}]{unit}")
        summary.append(f"  acc on extremes     train {s['train_acc_ext']:.3f} | val {s['val_acc_ext']:.3f}")
        summary.append(f"  all val frames      median split acc {s['val_acc_median']:.3f} | spearman(svm score, raw factor) {s['val_spearman']:+.3f}")
        summary.append(f"  train cells/videos  low {s['cells_low']}/{s['videos_low']} | high {s['cells_high']}/{s['videos_high']}")
    print("\n".join(summary))

    #how much the directions overlap, editing along one will also move the others by this much (cos sim)
    #(interfaceGAN's conditional manipulation projects these out if they turn out to be big)
    N = np.concatenate([boundaries[name] for name in factor_names], axis=0)   #[n_attr, 512]
    cos = N @ N.T
    summary.append("\ncosine similarity between directions:")
    summary.append("          " + " ".join(f"{name:>9s}" for name in factor_names))
    for k, name in enumerate(factor_names):
        summary.append(f"{name:9s} " + " ".join(f"{c:9.3f}" for c in cos[k]))
    print("\n".join(summary[-(len(factor_names) + 2):]))

    with open(f"{SAVE_DIR}/summary{TAG}.txt", "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"\nsaved svm models + boundaries to {SAVE_DIR}/")
