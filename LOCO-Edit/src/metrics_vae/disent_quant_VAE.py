'''
VAE was trained on 16 latent codes..

1. size- major axis, minor axis, area of cell
2. shape- convex,(?) aspect ratio
3. intesity- average intensity, integrated intensity
4. distance- center-to-center, edge-to-edge
'''
import torch
from torch.utils.data import DataLoader
from VAE_disent.model import twoChannelVAE
from VAE_disent.data_utils import twoChannelDataset

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import debugpy
# debugpy.listen(("127.0.0.1", 5678)) #127.0.0.1, cz only local host can talk to the port
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")


DEVICE     = torch.device("cuda:9")
SET        = "train"
CHA_DIR    = f"data/singlecell_chA_split/{SET}"
CHB_DIR    = f"data/singlecell_chB_split/{SET}"
MASK_DIR   = "data/singlecell_mask"
CKPT       = "vae_checkpoints_masked/beta_vae_beta=1_16/vae_epoch99.pt"
LATENT_DIM = 16
BATCH_SIZE = 32


def size(data_2ch, mask):
    #1. measure area of the mask (compared to full image)

    #2. width of mask (first pixel on the left to the last pixel on the right)
    
    #3. height of the mask (first pixel at the top to the lowest pixel at teh bottom)
    
    #NOTE: return all 3, (each one could be a specific "generative factor")
    m = mask.squeeze().cpu().numpy() > 0.5  # [H, W] bool
    total_px = m.size
    area = m.sum() / total_px

    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0.0, 0.0, 0.0

    height = rows.max() - rows.min() + 1
    width  = cols.max() - cols.min() + 1
    return area, width, height

def shape(data_2ch, mask):
    #aspect ratio- (major axis)/(minor axis)
    #measure both height and width and select the major/minor axis based on their values. then divide
    m = mask.squeeze().cpu().numpy() > 0.5
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0.0

    height = rows.max() - rows.min() + 1
    width  = cols.max() - cols.min() + 1
    major = max(height, width)
    minor = max(min(height, width), 1)  # avoid div by 0
    return major / minor

def intensity(data_2ch, mask):
    
    #1. average intensity (average of all pixel intensities..)
    
    #2. integrated intensity- (sum of all pixel intensities..NOT average..)
    m = mask.squeeze().cpu().numpy() > 0.5
    img = data_2ch.cpu().numpy()
    img = (img * 0.5) + 0.5  # undo Normalize(mean=0.5, std=0.5) back to [0, 1]
    pixel_vals = img[:, m]   # [2, num_mask_pixels], masked-in pixels only

    if pixel_vals.size == 0:
        return 0.0, 0.0
    return pixel_vals.mean(), pixel_vals.sum()


def encode_dataset(loader, model):
    #run every sample through the encoder, building the [N, D] code matrix C
    #and the matching [N, K] generative-factor matrix Z
    codes = []
    factors = []
    model.eval()
    with torch.no_grad():
        for x_masked, mask in loader:
            x_masked = x_masked.to(DEVICE)
            mu, logvar = model.encode(x_masked)
            codes.append(mu.cpu().numpy())

            for i in range(x_masked.size(0)):
                area, width, height = size(x_masked[i], mask[i])
                aspect_ratio = shape(x_masked[i], mask[i])
                avg_int, int_int = intensity(x_masked[i], mask[i])
                factors.append([area, width, height, aspect_ratio, avg_int, int_int])

    C = np.concatenate(codes, axis=0)
    Z = np.array(factors)
    return C, Z


def fit_lasso_importance_matrix(C_train, Z_train):
    #standardize codes/factors to zero mean, unit variance (as in the paper)
    c_mean, c_std = C_train.mean(0), C_train.std(0) + 1e-8
    z_mean, z_std = Z_train.mean(0), Z_train.std(0) + 1e-8
    C_norm = (C_train - c_mean) / c_std
    Z_norm = (Z_train - z_mean) / z_std

    D, K = C_norm.shape[1], Z_norm.shape[1] #D= 16 here, K = 6 (generative factors)
    R = np.zeros((D, K)) #16x16 -> importance matrix
    models = []
    for j in range(K):
        #cross-validated alpha in place of a hand-tuned val-set sweep
        reg = LassoCV(cv=5).fit(C_norm, Z_norm[:, j])
        R[:, j] = np.abs(reg.coef_) #the coefficients (or weights, as noted in the paper!)
        models.append(reg)

    return R, models, (c_mean, c_std, z_mean, z_std)


def fit_randomforest_importance_matrix(C_train, Z_train):
    #standardize codes/factors to zero mean, unit variance (as in the paper)
    c_mean, c_std = C_train.mean(0), C_train.std(0) + 1e-8
    z_mean, z_std = Z_train.mean(0), Z_train.std(0) + 1e-8
    C_norm = (C_train - c_mean) / c_std
    Z_norm = (Z_train - z_mean) / z_std
    
    D, K = C_norm.shape[1], Z_norm.shape[1] #D= 16 here, K = 6 (generative factors)
    R = np.zeros((D, K)) #16x16 -> importance matrix
    models = []
    
    for j in range(K):
        #train the regressors
        rf = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=20)
        '''
        #NOTE:
        below, 1. C_norm is the features (random subspaces) AND
        2. Z_norm[:, j] are the data elements for that specific factor (could be area, width,etc)
        
        '''
        rf.fit(C_norm, Z_norm[:, j])
        R[:, j] = rf.feature_importances_
        models.append(rf)
    
    return R, models, (c_mean, c_std, z_mean, z_std)
        
def disentanglement_scores(R):
    eps = 1e-12
    D, K = R.shape
    P = R / (R.sum(axis=1, keepdims=True) + eps)          # P[i,j]: importance of ci for zj
    H = -np.sum(P * (np.log(P + eps) / np.log(K)), axis=1)  # entropy, base K
    D_i = 1 - H

    rho = R.sum(axis=1) / R.sum()
    overall_disentanglement = np.sum(rho * D_i)
    return D_i, rho, overall_disentanglement


def completeness_scores(R):
    eps = 1e-12
    D, K = R.shape
    P_tilde = R / (R.sum(axis=0, keepdims=True) + eps)              # P_tilde[i,j]: importance of ci for zj, normalized over i
    H = -np.sum(P_tilde * (np.log(P_tilde + eps) / np.log(D)), axis=0)  # entropy, base D
    C_j = 1 - H

    overall_completeness = C_j.mean()
    return C_j, overall_completeness


def informativeness_scores(models, C_test, Z_test, stats):
    c_mean, c_std, z_mean, z_std = stats
    C_norm = (C_test - c_mean) / c_std
    Z_norm = (Z_test - z_mean) / z_std

    errors = []
    for j, reg in enumerate(models):
        z_pred = reg.predict(C_norm)
        errors.append(mean_squared_error(Z_norm[:, j], z_pred))  # z's already standardized -> normalized error

    errors = np.array(errors)
    return errors, errors.mean()


if __name__ == "__main__":
    #load training data
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = twoChannelVAE(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    C, Z = encode_dataset(loader, model)

    #held-out split so informativeness isn't measured on the same data the lasso was fit on
    C_train, C_test, Z_train, Z_test = train_test_split(C, Z, test_size=0.2, random_state=0)

    '''
    lasso below
    '''    

    # R, lasso_models, stats = fit_lasso_importance_matrix(C_train, Z_train)
    
    '''
    random forest method below
    '''
    print("fitting random forest..")
    R, lasso_models, stats = fit_randomforest_importance_matrix(C_train, Z_train)
    print("random forest successfully fit..")

    D_i, rho, overall_disentanglement = disentanglement_scores(R)
    C_j, overall_completeness = completeness_scores(R)
    err_j, overall_informativeness = informativeness_scores(lasso_models, C_test, Z_test, stats)

    factor_names = ["area", "width", "height", "aspect_ratio", "avg_intensity", "integrated_intensity"]

    print("=== Disentanglement (per code variable) ===")
    for i, (d, r) in enumerate(zip(D_i, rho)):
        print(f"  c{i:02d}: D={d:.3f}  rho={r:.3f}")
    print(f"Overall disentanglement: {overall_disentanglement:.3f}\n")

    print("=== Completeness (per generative factor) ===")
    for name, c in zip(factor_names, C_j):
        print(f"  {name:>20s}: C={c:.3f}")
    print(f"Overall completeness: {overall_completeness:.3f}\n")

    print("=== Informativeness (per generative factor, normalized MSE) ===")
    for name, e in zip(factor_names, err_j):
        print(f"  {name:>20s}: err={e:.3f}")
    print(f"Overall informativeness (mean err): {overall_informativeness:.3f}\n")

    #Hinton-style visualisation of R (rows=code vars, cols=generative factors)
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(R, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(factor_names)))
    ax.set_xticklabels(factor_names, rotation=45, ha='right')
    ax.set_ylabel("code variable")
    ax.set_title("relative importance matrix R")
    plt.tight_layout()
    plt.savefig("metrics_vae/disent_quant_R_matrix.png")