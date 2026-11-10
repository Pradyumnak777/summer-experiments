'''
rbf (kernel) svm version of train_svm.py, on the same extreme split of the inverted train burst frames

same data, same split by cell, same top/bottom KEEP_RATIO labels and the same stats as train_boundary_2, so the numbers
line up with svm_models/summary_ext30.txt. differences:
    - w is standardized first (rbf is a distance, so it's scale sensitive), the scaler is part of the saved pipeline
    - C and gamma are picked by a grouped cv on the train cells only (the val cells stay untouched)
    - an rbf boundary is curved, so there's no single normal to save. the direction at a point w is the gradient of the
      decision function there (rbf_grad), different at every w. saved: the fitted pipeline per attr
    - the linear ext30 svm is scored on the same val frames next to it, plus how the rbf gradients compare to its normal

--split median: all train frames instead, with the median split labels from data_utils.py (like train_boundary), scored
next to the linear median svm (svm_models/svm_<attr>.joblib). files without the ext suffix, eg svm_rbf_intensity.joblib

run from src/ (cpu only, no gpu needed):
    python -m disentanglement_guided_GAN.train_svm_rbf                    #extreme split (default)
    python -m disentanglement_guided_GAN.train_svm_rbf --split median     #entire train set
'''
import argparse
import os

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#same data + split settings as the linear one, so both see exactly the same train/val frames
from disentanglement_guided_GAN.train_svm import load_data, cell_groups, DATA_DIR, VAL_RATIO, SEED, KEEP_RATIO, TAG
from disentanglement_guided_GAN.train_svm import SAVE_DIR as LINEAR_DIR

import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

#config stuff
SAVE_DIR  = "disentanglement_guided_GAN/svm_models_rbf"
CV_FOLDS  = 3       #grouped by cell, on the train extremes only
N_JOBS    = 9       #cpu processes for the grid search (shared box, keep it modest)
#gamma in units of 1/n_features (sklearn's "scale" on standardized data is ~1/512): 1x = that default, smaller = smoother,
#closer to linear, bigger = more local bumps around the support vectors
GRID = {"svc__C": [0.1, 1.0, 10.0], "svc__gamma": [g / 512 for g in (0.1, 1.0, 10.0)]}


def rbf_grad(pipe, W):
    #the rbf direction at every w: gradient of the decision function f(w) = sum_i a_i exp(-gamma |x - s_i|^2) + b,
    #x = (w - mu) / sigma the standardized w, s_i the support vectors, a_i = dual_coef_ (sign included, + -> label 1/high)
    #df/dx = sum_i a_i * -2 gamma (x - s_i) k(x, s_i), and df/dw = df/dx / sigma (chain rule through the scaler)
    #W: [N, 512] -> [N, 512], not normalized (its norm goes to 0 far from the support vectors)
    scaler, svc = pipe.named_steps["standardscaler"], pipe.named_steps["svc"]
    X = scaler.transform(W)                                            #[N, 512]
    S, a = svc.support_vectors_, svc.dual_coef_[0]                     #[n_sv, 512], [n_sv]
    sq = (X ** 2).sum(1)[:, None] + (S ** 2).sum(1)[None] - 2 * X @ S.T   #[N, n_sv] squared distances
    K = np.exp(-svc._gamma * np.maximum(sq, 0))                         #[N, n_sv]
    AK = K * a[None]                                                   #[N, n_sv]
    grad_x = -2 * svc._gamma * (AK.sum(1, keepdims=True) * X - AK @ S)  #sum_i a_i k_i (x - s_i), without materializing x - s_i
    return grad_x / scaler.scale_[None]


def extreme_labels(f_tr, f_val):
    #same as train_boundary_2: thresholds from the train frames only, middle (1 - 2 * KEEP_RATIO) dropped
    lo, hi = np.quantile(f_tr, KEEP_RATIO), np.quantile(f_tr, 1 - KEEP_RATIO)
    tr_keep, val_keep = (f_tr <= lo) | (f_tr >= hi), (f_val <= lo) | (f_val >= hi)
    return tr_keep, val_keep, (f_tr[tr_keep] >= hi).astype(np.int64), (f_val[val_keep] >= hi).astype(np.int64)


def score(clf, W_tr_ext, y_tr, W_val, val_keep, y_val, y_val_median, f_val):
    #the same 4 numbers as train_boundary_2, for any fitted classifier (linear or rbf)
    return {
        "train_acc_ext": clf.score(W_tr_ext, y_tr),
        "val_acc_ext": clf.score(W_val[val_keep], y_val),
        "val_acc_median": clf.score(W_val, y_val_median),
        "val_spearman": spearmanr(clf.decision_function(W_val), f_val)[0],
    }


def train_boundary_rbf(W_tr, f_tr, W_val, f_val, y_val_median, groups_tr):
    tr_keep, val_keep, y_tr, y_val = extreme_labels(f_tr, f_val)

    #grid search on the train extremes, folds split by cell like the train/val split (frames of one cell are near copies)
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    search = GridSearchCV(pipe, GRID, cv=GroupKFold(n_splits=CV_FOLDS), scoring="accuracy", n_jobs=N_JOBS)
    search.fit(W_tr[tr_keep], y_tr, groups=groups_tr[tr_keep])
    clf = search.best_estimator_   #refit on all the train extremes with the best C/gamma

    stats = score(clf, W_tr[tr_keep], y_tr, W_val, val_keep, y_val, y_val_median, f_val)
    stats.update({"n_train": int(tr_keep.sum()), "n_sv": int(clf.named_steps["svc"].n_support_.sum()),
                  "C": search.best_params_["svc__C"], "gamma_x512": search.best_params_["svc__gamma"] * 512,
                  "cv_acc": search.best_score_})
    return clf, stats, (tr_keep, val_keep, y_tr, y_val)


def train_boundary_rbf_median(W_tr, y_tr, W_val, y_val, f_val, groups_tr):
    #the rbf version of train_boundary: every train frame, labels = above/below the median (data_utils.py's labels).
    #same grid search, folds split by cell
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    search = GridSearchCV(pipe, GRID, cv=GroupKFold(n_splits=CV_FOLDS), scoring="accuracy", n_jobs=N_JOBS)
    search.fit(W_tr, y_tr, groups=groups_tr)
    clf = search.best_estimator_

    stats = score_median(clf, W_tr, y_tr, W_val, y_val, f_val)
    stats.update({"n_train": len(W_tr), "n_sv": int(clf.named_steps["svc"].n_support_.sum()),
                  "C": search.best_params_["svc__C"], "gamma_x512": search.best_params_["svc__gamma"] * 512,
                  "cv_acc": search.best_score_})
    return clf, stats


def score_median(clf, W_tr, y_tr, W_val, y_val, f_val):
    #train_boundary's numbers + the spearman, for any fitted classifier (linear or rbf)
    return {
        "train_acc": clf.score(W_tr, y_tr),
        "val_acc": clf.score(W_val, y_val),
        "val_spearman": spearmanr(clf.decision_function(W_val), f_val)[0],
    }


def direction_stats(clf, W_val, n_linear):
    #how "linear" the rbf directions are, on the val frames:
    #  consistency = |mean of the unit gradients|, 1 -> every w gets the same direction (like the linear svm), ~0 -> all over
    #  cos w/ linear = mean cos between each w's gradient and the linear ext30 normal (both point low -> high)
    #  cos of mean = the averaged rbf direction vs the linear normal
    g = rbf_grad(clf, W_val)
    norms = np.linalg.norm(g, axis=1)
    u = g / (norms[:, None] + 1e-12)
    mean_u = u.mean(0)
    return {
        "consistency": float(np.linalg.norm(mean_u)),
        "cos_linear_mean": float((u @ n_linear).mean()),
        "cos_linear_std": float((u @ n_linear).std()),
        "cos_of_mean": float(mean_u @ n_linear / (np.linalg.norm(mean_u) + 1e-12)),
        "grad_norm_median": float(np.median(norms)),
    }


def direction_line(d):
    return (f"  directions on val: consistency {d['consistency']:.3f} (1 = same everywhere) | "
            f"cos w/ linear normal {d['cos_linear_mean']:+.3f} +- {d['cos_linear_std']:.3f} | "
            f"cos of mean dir {d['cos_of_mean']:+.3f} | median |grad| {d['grad_norm_median']:.3g}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="ext", choices=["ext", "median"],
                   help="ext: top/bottom KEEP_RATIO of the train frames, median: all train frames, median split labels")
    args = p.parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)

    W, Y, F, bases, factor_names = load_data(DATA_DIR)
    groups = cell_groups(bases)
    print(f"{len(W)} frames from {len(np.unique(groups))} cells, w dim {W.shape[1]}, attributes: {factor_names}")

    #the same split by cell as train_svm.py (same seed + data -> the same cells in val)
    tr_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED).split(W, groups=groups))
    print(f"train: {len(tr_idx)} frames, val: {len(val_idx)} frames (no cell in both)\n")

    if args.split == "median":
        summary = [f"rbf svm, median split: every train frame, labels from data_utils.py (above/below the median), "
                   f"C/gamma by {CV_FOLDS}-fold grouped cv on the train frames",
                   f"grid: C {GRID['svc__C']}, gamma x512 {[round(g * 512, 3) for g in GRID['svc__gamma']]}"]
        for k, name in enumerate(factor_names):
            print(f"{name}: grid search..", flush=True)
            y = Y[:, k]
            clf, s = train_boundary_rbf_median(W[tr_idx], y[tr_idx], W[val_idx], y[val_idx], F[val_idx, k], groups[tr_idx])
            joblib.dump(clf, f"{SAVE_DIR}/svm_rbf_{name}.joblib")

            #the linear median svm (train_boundary) on the exact same frames
            lin = joblib.load(f"{LINEAR_DIR}/svm_{name}.joblib")
            n_lin = np.load(f"{LINEAR_DIR}/boundary_{name}.npy")[0]
            sl = score_median(lin, W[tr_idx], y[tr_idx], W[val_idx], y[val_idx], F[val_idx, k])
            d = direction_stats(clf, W[val_idx], n_lin)

            summary.append(f"\n{name}  (trained on {s['n_train']} frames, {s['n_sv']} support vectors, "
                           f"best C {s['C']}, gamma {s['gamma_x512']:.1f}/512, cv acc {s['cv_acc']:.3f})")
            summary.append(f"                      {'rbf':>7s}   {'linear':>7s}")
            for key, label in [("train_acc", "train acc"), ("val_acc", "val acc"), ("val_spearman", "val spearman")]:
                summary.append(f"  {label:20s} {s[key]:7.3f}   {sl[key]:7.3f}")
            summary.append(direction_line(d))
            print("\n".join(summary[-6:]), flush=True)

        with open(f"{SAVE_DIR}/summary.txt", "w") as f:
            f.write("\n".join(summary) + "\n")
        print(f"\nsaved rbf svm pipelines + summary to {SAVE_DIR}/")
        raise SystemExit

    summary = [f"rbf svm, extreme split: top/bottom {int(KEEP_RATIO * 100)}% of each factor (train thresholds), "
               f"C/gamma by {CV_FOLDS}-fold grouped cv on the train extremes",
               f"grid: C {GRID['svc__C']}, gamma x512 {[round(g * 512, 3) for g in GRID['svc__gamma']]}"]
    for k, name in enumerate(factor_names):
        print(f"{name}: grid search..", flush=True)
        clf, s, (tr_keep, val_keep, y_tr, y_val) = train_boundary_rbf(W[tr_idx], F[tr_idx, k], W[val_idx], F[val_idx, k],
                                                                     Y[val_idx, k], groups[tr_idx])
        joblib.dump(clf, f"{SAVE_DIR}/svm_rbf_{name}{TAG}.joblib")   #scaler + svc, decision_function takes raw w's

        #the linear ext30 svm on the exact same frames, so the two are side by side
        lin = joblib.load(f"{LINEAR_DIR}/svm_{name}{TAG}.joblib")
        n_lin = np.load(f"{LINEAR_DIR}/boundary_{name}{TAG}.npy")[0]
        sl = score(lin, W[tr_idx][tr_keep], y_tr, W[val_idx], val_keep, y_val, Y[val_idx, k], F[val_idx, k])
        d = direction_stats(clf, W[val_idx], n_lin)

        summary.append(f"\n{name}  (trained on {s['n_train']} frames, {s['n_sv']} support vectors, "
                       f"best C {s['C']}, gamma {s['gamma_x512']:.1f}/512, cv acc {s['cv_acc']:.3f})")
        summary.append(f"                      {'rbf':>7s}   {'linear':>7s}")
        for key, label in [("train_acc_ext", "train acc extremes"), ("val_acc_ext", "val acc extremes"),
                           ("val_acc_median", "val acc median split"), ("val_spearman", "val spearman")]:
            summary.append(f"  {label:20s} {s[key]:7.3f}   {sl[key]:7.3f}")
        summary.append(direction_line(d))
        print("\n".join(summary[-7:]), flush=True)

    with open(f"{SAVE_DIR}/summary{TAG}.txt", "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"\nsaved rbf svm pipelines + summary to {SAVE_DIR}/")
