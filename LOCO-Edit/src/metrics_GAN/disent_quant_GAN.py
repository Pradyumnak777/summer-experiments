import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from VAE_disent.data_utils import twoChannelDataset

import debugpy
debugpy.listen(("127.0.0.1", 5678)) #127.0.0.1, cz only local host can talk to the port
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
print("Debugger attached! Running code...")


DEVICE           = torch.device("cuda:9")
SET              = "train"
CHA_DIR          = f"data/singlecell_chA_split/{SET}"
CHB_DIR          = f"data/singlecell_chB_split/{SET}"
MASK_DIR         = "data/singlecell_mask"
PCA_DIRS_PATH    = "GAN/gan_pca_results_inverted_w_masked/dirs.npz"
W_SPACE_DATA_DIR = "GAN/gan_inverted_ws_train-set"
OUT_METRICS_DIR  = "metrics_GAN"
K_COMPONENTS     = 10   # Number of top GANSpace directions to use as codes
BATCH_SIZE       = 64

# ----------------- Ground Truth Factors -----------------
def size(mask):
    m = mask.squeeze().cpu().numpy() > 0.5
    total_px = m.size
    area = m.sum() / total_px

    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0.0, 0.0, 0.0

    height = rows.max() - rows.min() + 1
    width  = cols.max() - cols.min() + 1
    return area, float(width), float(height)

def shape(mask):
    m = mask.squeeze().cpu().numpy() > 0.5
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0.0

    height = rows.max() - rows.min() + 1
    width  = cols.max() - cols.min() + 1
    major = max(height, width)
    minor = max(min(height, width), 1)
    return float(major / minor)

def intensity(data_2ch, mask):
    m = mask.squeeze().cpu().numpy() > 0.5
    img = data_2ch.cpu().numpy()
    img = (img * 0.5) + 0.5  # Rescale from [-1, 1] to [0, 1]
    pixel_vals = img[:, m]

    if pixel_vals.size == 0:
        return 0.0, 0.0
    return float(pixel_vals.mean()), float(pixel_vals.sum())

def extract_dataset_factors(dataset):
    """Computes the ground-truth factors Z for all dataset images."""
    factors = []
    print("Extracting ground-truth factors from dataset masks...")
    for idx in tqdm(range(len(dataset)), desc="Computing Factors"):
        x_masked, mask = dataset[idx]
        area, width, height = size(mask)
        aspect_ratio = shape(mask)
        avg_int, int_int = intensity(x_masked, mask)
        factors.append([area, width, height, aspect_ratio, avg_int, int_int])
    return np.array(factors, dtype=np.float32)

def compute_ganspace_codes(w_dir, pca_path, k_components):
    """Loads inverted w vectors and projects them onto top-K GANSpace directions."""
    pca_data = np.load(pca_path)
    Vt = pca_data["Vt"][:k_components]      # Shape: [K, 512]
    w_mean = pca_data["w_mean"]              # Shape: [512]
    V = Vt.T                                 # Projection basis: [512, K]

    # Load all inverted latents matching file order
    w_files = sorted([f for f in os.listdir(w_dir) if f.endswith(".npy") or f.endswith(".npz")])
    codes = []
    print(f"Projecting {len(w_files)} inverted W vectors onto top {k_components} PCA directions...")
    
    for fname in tqdm(w_files, desc="Projecting W"):
        path = os.path.join(w_dir, fname)
        raw = np.load(path)
        w_vec = raw["w"] if isinstance(raw, np.lib.npyio.NpzFile) else raw
        
        # Flatten or select base W vector: [num_ws, 512] -> [512]
        if w_vec.ndim >= 2:
            w_base = w_vec[0] if w_vec.shape[0] == 12 else w_vec.squeeze()
        else:
            w_base = w_vec

        # Center and project: [512] @ [512, K] -> [K]
        c = (w_base - w_mean) @ V
        codes.append(c)

    return np.array(codes, dtype=np.float32)

# ----------------- DCI Metrics -----------------
def fit_lasso_importance_matrix(C_train, Z_train):
    c_mean, c_std = C_train.mean(0), C_train.std(0) + 1e-8
    z_mean, z_std = Z_train.mean(0), Z_train.std(0) + 1e-8
    C_norm = (C_train - c_mean) / c_std
    Z_norm = (Z_train - z_mean) / z_std

    D_dim, K_factors = C_norm.shape[1], Z_norm.shape[1]
    R = np.zeros((D_dim, K_factors))
    models = []
    for j in range(K_factors):
        reg = LassoCV(cv=5, max_iter=2000).fit(C_norm, Z_norm[:, j])
        R[:, j] = np.abs(reg.coef_)
        models.append(reg)

    return R, models, (c_mean, c_std, z_mean, z_std)

def disentanglement_scores(R):
    eps = 1e-12
    D_dim, K_factors = R.shape
    P = R / (R.sum(axis=1, keepdims=True) + eps)
    H = -np.sum(P * (np.log(P + eps) / np.log(K_factors)), axis=1)
    D_i = 1.0 - H
    rho = R.sum(axis=1) / (R.sum() + eps)
    overall_disentanglement = np.sum(rho * D_i)
    return D_i, rho, overall_disentanglement

def completeness_scores(R):
    eps = 1e-12
    D_dim, K_factors = R.shape
    P_tilde = R / (R.sum(axis=0, keepdims=True) + eps)
    H = -np.sum(P_tilde * (np.log(P_tilde + eps) / np.log(D_dim)), axis=0)
    C_j = 1.0 - H
    overall_completeness = C_j.mean()
    return C_j, overall_completeness

def informativeness_scores(models, C_test, Z_test, stats):
    c_mean, c_std, z_mean, z_std = stats
    C_norm = (C_test - c_mean) / c_std
    Z_norm = (Z_test - z_mean) / z_std

    errors = []
    for j, reg in enumerate(models):
        z_pred = reg.predict(C_norm)
        errors.append(mean_squared_error(Z_norm[:, j], z_pred))

    errors = np.array(errors)
    return errors, errors.mean()

if __name__ == "__main__":
    os.makedirs(OUT_METRICS_DIR, exist_ok=True)

    # 1. Extract ground truth factors Z from masked dataset
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)
    Z = extract_dataset_factors(dataset)

    # 2. Build code matrix C by projecting inverted W latents onto GANSpace directions
    C = compute_ganspace_codes(W_SPACE_DATA_DIR, PCA_DIRS_PATH, K_COMPONENTS)

    # Validate row alignment
    min_samples = min(len(C), len(Z))
    C, Z = C[:min_samples], Z[:min_samples]
    print(f"Evaluating DCI on {min_samples} aligned samples (Codes: {C.shape}, Factors: {Z.shape})")

    # 3. Fit Lasso regressions and evaluate DCI
    C_train, C_test, Z_train, Z_test = train_test_split(C, Z, test_size=0.2, random_state=0)
    R, lasso_models, stats = fit_lasso_importance_matrix(C_train, Z_train)

    D_i, rho, overall_disentanglement = disentanglement_scores(R)
    C_j, overall_completeness = completeness_scores(R)
    err_j, overall_informativeness = informativeness_scores(lasso_models, C_test, Z_test, stats)

    factor_names = ["area", "width", "height", "aspect_ratio", "avg_intensity", "integrated_intensity"]

    print("\n=== Disentanglement (per GANSpace direction) ===")
    for i, (d, r) in enumerate(zip(D_i, rho)):
        print(f"  PC{i:02d}: D={d:.3f}  rho={r:.3f}")
    print(f"Overall disentanglement: {overall_disentanglement:.3f}\n")

    print("=== Completeness (per generative factor) ===")
    for name, c in zip(factor_names, C_j):
        print(f"  {name:>20s}: C={c:.3f}")
    print(f"Overall completeness: {overall_completeness:.3f}\n")

    print("=== Informativeness (per generative factor, normalized MSE) ===")
    for name, e in zip(factor_names, err_j):
        print(f"  {name:>20s}: err={e:.3f}")
    print(f"Overall informativeness (mean err): {overall_informativeness:.3f}\n")

    # 4. Save visualization matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(R, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(factor_names)))
    ax.set_xticklabels(factor_names, rotation=45, ha="right")
    ax.set_yticks(range(K_COMPONENTS))
    ax.set_yticklabels([f"PC_{i}" for i in range(K_COMPONENTS)])
    ax.set_ylabel("GANSpace Component (Code)")
    ax.set_xlabel("Generative Factor")
    ax.set_title("StyleGAN (GANSpace) Lasso Importance Matrix R")
    plt.tight_layout()
    plt.savefig(f"{OUT_METRICS_DIR}/ganspace_dci_R_matrix.png")
    print(f"Saved importance matrix plot to {OUT_METRICS_DIR}/ganspace_dci_R_matrix.png")