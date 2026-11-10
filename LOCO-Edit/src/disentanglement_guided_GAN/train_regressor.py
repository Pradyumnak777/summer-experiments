'''
regression version of train_svm.py: no binarized labels, every train frame is fit on its raw (continuous) factor value
(intensity, log area, log aspect, the same "factors" data_utils.py saved), same inverted w's and the same split by cell

two directions per factor, both [1, 512] unit norm, pointing low -> high like the svm boundaries (so svm_edit.py can use them):
    - ridge: the regression coefficients beta / |beta| (the "decoder", what predicts the factor best). alpha (the ridge
      strength) is picked by a grouped cv on the train cells. bonus: the prediction is beta . w + b, so a step of alpha
      along it moves the predicted factor by alpha * |beta| (saved as slope)
    - cov: cov(w, factor) / |...|, the "pattern" (haufe et al. 2014), how the factor shows up in w. the continuous version
      of the difference of class means, and the same as the first pls component
the ridge weights can lean on dims that only cancel noise, the cov one can't, so the two can edit differently

scored on the val cells, next to both linear svms (median + ext30) on the exact same frames

run from src/ (cpu only, no gpu needed):
    python -m disentanglement_guided_GAN.train_regressor
'''
import os

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV

#same data + split settings as the svms, so all of them see exactly the same train/val frames
from disentanglement_guided_GAN.train_svm import load_data, cell_groups, DATA_DIR, VAL_RATIO, SEED, TAG
from disentanglement_guided_GAN.train_svm import SAVE_DIR as SVM_DIR

import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

#config stuff
SAVE_DIR = "disentanglement_guided_GAN/regressor_models"
CV_FOLDS = 3
#ridge strengths to try, on the raw w's (no standardizing, the direction has to live in w space). very strong ridge
#(alpha -> inf) turns the direction into the cov one, so the top of the range is where the two meet
ALPHAS   = np.logspace(-2, 8, 21)


def train_ridge(W_tr, f_tr, groups_tr):
    #ridge on the raw w's, alpha by grouped cv (folds split by cell, frames of one cell are near copies)
    search = GridSearchCV(Ridge(), {"alpha": ALPHAS}, cv=GroupKFold(n_splits=CV_FOLDS), scoring="r2")
    search.fit(W_tr, f_tr, groups=groups_tr)
    reg = search.best_estimator_
    beta = reg.coef_
    return reg, (beta / np.linalg.norm(beta))[None].astype(np.float32), float(np.linalg.norm(beta)), search


def cov_direction(W_tr, f_tr):
    #cov(w, f) over the train frames, unit norm. points low -> high by construction (positive covariance with f)
    c = (W_tr - W_tr.mean(0)).T @ (f_tr - f_tr.mean()) / len(f_tr)
    return (c / np.linalg.norm(c))[None].astype(np.float32)


def val_spearman(W_val, d, f_val):
    #rank correlation between the position along a unit direction and the raw factor, on the val frames
    return spearmanr(W_val @ d[0], f_val)[0]


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    W, Y, F, bases, factor_names = load_data(DATA_DIR)
    groups = cell_groups(bases)
    print(f"{len(W)} frames from {len(np.unique(groups))} cells, w dim {W.shape[1]}, attributes: {factor_names}")

    #the same split by cell as train_svm.py (same seed + data -> the same cells in val)
    tr_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED).split(W, groups=groups))
    print(f"train: {len(tr_idx)} frames, val: {len(val_idx)} frames (no cell in both)\n")
    W_tr, W_val = W[tr_idx], W[val_idx]

    summary = [f"ridge regression on the raw factors (intensity, log area, log aspect), every train frame, "
               f"alpha by {CV_FOLDS}-fold grouped cv",
               "val spearman = rank corr between the position along the direction (w . d) and the raw factor, val cells"]
    dirs = {}
    for k, name in enumerate(factor_names):
        f_tr, f_val = F[tr_idx, k], F[val_idx, k]
        reg, d_ridge, slope, search = train_ridge(W_tr, f_tr, groups[tr_idx])
        d_cov = cov_direction(W_tr, f_tr)
        dirs[name] = d_ridge

        joblib.dump(reg, f"{SAVE_DIR}/ridge_{name}.joblib")
        np.save(f"{SAVE_DIR}/direction_ridge_{name}.npy", d_ridge)   #[1, 512], unit norm, low -> high
        np.save(f"{SAVE_DIR}/direction_cov_{name}.npy", d_cov)

        #the two linear svm normals, scored the same way on the same val frames
        n_med = np.load(f"{SVM_DIR}/boundary_{name}.npy")
        n_ext = np.load(f"{SVM_DIR}/boundary_{name}{TAG}.npy")

        unit = {"intensity": "intensity", "area": "log area", "aspect": "log aspect"}[name]
        summary.append(f"\n{name}  (best alpha {search.best_params_['alpha']:.3g}, cv r2 {search.best_score_:.3f})")
        summary.append(f"  ridge fit          train r2 {reg.score(W_tr, f_tr):.3f} | val r2 {r2_score(f_val, reg.predict(W_val)):.3f}")
        summary.append(f"  slope              a step of 1 along the ridge direction moves the predicted {unit} by {slope:.4f}"
                       f"   (factor sd on train {f_tr.std():.4f})")
        summary.append(f"  val spearman       ridge {val_spearman(W_val, d_ridge, f_val):+.3f} | cov {val_spearman(W_val, d_cov, f_val):+.3f}"
                       f" | svm median {val_spearman(W_val, n_med, f_val):+.3f} | svm ext30 {val_spearman(W_val, n_ext, f_val):+.3f}")
        summary.append(f"  cos w/ ridge       cov {(d_cov @ d_ridge.T).item():+.3f} | svm median {(n_med @ d_ridge.T).item():+.3f}"
                       f" | svm ext30 {(n_ext @ d_ridge.T).item():+.3f}")
        print("\n".join(summary[-5:]), flush=True)

    #how much the ridge directions overlap, same table as train_svm.py
    N = np.concatenate([dirs[name] for name in factor_names], axis=0)
    cos = N @ N.T
    summary.append("\ncosine similarity between the ridge directions:")
    summary.append("          " + " ".join(f"{name:>9s}" for name in factor_names))
    for k, name in enumerate(factor_names):
        summary.append(f"{name:9s} " + " ".join(f"{c:9.3f}" for c in cos[k]))
    print("\n".join(summary[-(len(factor_names) + 2):]))

    with open(f"{SAVE_DIR}/summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"\nsaved ridge models + directions to {SAVE_DIR}/")
