'''
this is to evaluate if the trained models can perform semantic disnentaglement-

1. assume intensity of the gene burst is a factor we want to manipulate..
'''
import torch
from torch.utils.data import DataLoader, Subset
from VAE_disent.data_utils import twoChannelDataset
from pathlib import Path
from PIL import Image

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
ANNOT_DIR  = "data/annotations" #gene burst annotations, data/annotations/<video>/hand_drawn_mask*/<cell>_handdrawnmask_movie/
CKPT       = "GAN/stylegan2-ada-pytorch/training-runs/00010-data_npy-auto1-kimg1000-ada-bg/network-snapshot-001000.pkl" #GAN checkpoint
W_SPACE_DATA_DIR = "GAN/gan_inverted_ws_train-set" #w space vectors for ht entire training set
LATENT_DIM = 16
BATCH_SIZE = 32
K_COMPONENTS = 10   # Number of top GANSpace directions to use as codes
PCA_DIRS_PATH    = "GAN/gan_pca_results_inverted_w_masked/dirs.npz" #PCA gotten by sampling 100k w space vectors..

'''
1. from the gene burst gt masks and the original train set images, get the intensity values of that specific spot/mask (for all available ones in the train set)
2. train a random forest classifier with features/random subspaces = latent codes and the data elements as the specific ground truth factors..
3. get the matrix, but itll be amatrix with a single column, as therte only one known genenrative factor.

#NOTE 1: We need both known + unknown genrative factors, this method is primarily for known factors only...

#NOTE 2:  dimension that only encodes unknown factors gets little weight, so it barely affects D. 
A dimension that encodes one known factor plus several unknown ones still looks perfectly disentangled, because the probes can't see the extra content. 
D therefore overestimates disentanglement
'''


def pick_burst_cell_dirs(video_dir):
    #a video can have duplicate annotations (hand_drawn_mask, hand_drawn_mask2, hand_drawn_mask_2..)
    #the folder with the most annotated cells wins, cells it is missing are filled in from the others
    hd_dirs = [d for d in Path(video_dir).glob("hand_drawn_mask*") if d.is_dir()]
    hd_dirs = sorted(hd_dirs, key=lambda d: len(list(d.iterdir())), reverse=True)

    cell_dirs = {} #cell number -> its folder of per-frame burst masks
    for hd in hd_dirs:
        for c in hd.iterdir():
            cell = int(c.name.split("_")[0]) #"12_handdrawnmask_movie" -> 12
            if cell not in cell_dirs:
                cell_dirs[cell] = c
    return cell_dirs


def burst_frames(dataset):
    #yields (idx, chA_path, burst) for every frame with a non-empty hand-drawn burst mask,
    #burst is the raw bool mask, same size as the raw chA crop (93-173 px, the crops vary)
    cell_dirs_per_video = {}

    for idx, chA_path in enumerate(dataset.chA_files):
        #eg: "<video>_cell12_f010_chA.png" -> video, cell 12, frame "010"
        base = chA_path.name[:-len("_chA.png")]
        video, cell_frame = base.rsplit("_cell", 1)
        cell, frame = cell_frame.split("_f")
        cell = int(cell)

        #annotation folders use "Orthogonal-Projection", the single cell crops use "Orthogonal Projection"
        if video not in cell_dirs_per_video:
            video_dir = Path(ANNOT_DIR) / video.replace(" ", "-")
            cell_dirs_per_video[video] = pick_burst_cell_dirs(video_dir) if video_dir.is_dir() else {}
        cell_dirs = cell_dirs_per_video[video]
        if cell not in cell_dirs:
            continue #this cell was never annotated

        burst_path = cell_dirs[cell] / f"{cell_dirs[cell].name}_f{frame}.png"
        if not burst_path.exists():
            continue
        burst = np.array(Image.open(burst_path).convert('L')) > 0
        if not burst.any():
            continue #empty mask -> no burst in this frame
        yield idx, chA_path, burst


def get_gene_burst_intensity(dataset): #load the masks, and get the gene burst intensity(my getting the portion of the actual cell iamge which is covered by that binary mask) for all available in the train set
    #return a modified dataset with only the values which have the gene bursts
    burst_idx, burst_int = [], []
    for idx, chA_path, burst in burst_frames(dataset):
        #mask only says where the burst is, the intensity comes from the actual chA pixels under it
        #(same raw crop as the burst mask, scaled to [0, 1] like intensity() in the metrics scripts)
        #NOTE: chA is the gene burst channel (confirmed), not chB
        img = np.array(Image.open(chA_path).convert('L')).astype(np.float32) / 255.0
        burst_idx.append(idx)
        burst_int.append(img[burst].mean())

    burst_idx = np.array(burst_idx)
    burst_int = np.array(burst_int, dtype=np.float32)
    print(f"{len(burst_idx)}/{len(dataset)} train frames have an annotated gene burst")

    #burst_idx indexes the full dataset, so it also picks the matching rows of the inverted w's
    return Subset(dataset, burst_idx), burst_int, burst_idx


BURST_FACTORS = ["intensity", "area", "aspect"]  #columns of get_burst_factors
MODEL_RES  = 128   #the model input size, every crop gets resized to this
ASPECT_MAX = 4.0   #near-line masks (minor axis ~0) blow the aspect ratio up to ~1e6, ~2% of bursts are past 2.35

def get_burst_factors(dataset):
    #same frames as get_gene_burst_intensity, but three factors per burst (columns in BURST_FACTORS order):
    #   intensity: mean chA under the burst mask, in [0, 1]
    #   area:      log(burst area in px), measured at MODEL_RES so a burst counts as big as the model sees it
    #              (the raw crops are 93-173 px, so raw px areas aren't comparable across cells)
    #   aspect:    log(major / minor axis), from the mask's second moments (rotation invariant, unlike a bounding box),
    #              clipped to [1, ASPECT_MAX] first. scale invariant, so the raw mask is fine
    #area and aspect are logged as both are heavily right skewed (area 2-299 px), so a few huge bursts don't dominate the mse
    from skimage.measure import regionprops
    burst_idx, factors = [], []
    for idx, chA_path, burst in burst_frames(dataset):
        img = np.array(Image.open(chA_path).convert('L')).astype(np.float32) / 255.0
        H, W = burst.shape
        p = regionprops(burst.astype(np.uint8))[0]   #all burst pixels as one region (all but ~0.05% are 1 blob anyway)
        area = p.area * (MODEL_RES / H) * (MODEL_RES / W)
        aspect = np.clip(p.major_axis_length / max(p.minor_axis_length, 1e-6), 1.0, ASPECT_MAX)
        burst_idx.append(idx)
        factors.append([img[burst].mean(), np.log(area), np.log(aspect)])

    burst_idx = np.array(burst_idx)
    factors = np.array(factors, dtype=np.float32)    #[N, len(BURST_FACTORS)]
    print(f"{len(burst_idx)}/{len(dataset)} frames have an annotated gene burst")
    return Subset(dataset, burst_idx), factors, burst_idx


def burst_mask_path(chB_path):
    #path of the hand-drawn burst mask of one frame, same filename -> annotation lookup as get_gene_burst_intensity
    #the chB filename is only used for the lookup (chA/chB share the "<video>_cell12_f010" base), the burst itself is in chA
    #eg: "<video>_cell12_f010_chB.png" -> video, cell 12, frame "010"
    base = chB_path.name[:-len("_chB.png")]
    video, cell_frame = base.rsplit("_cell", 1)
    cell, frame = cell_frame.split("_f")
    cell_dir = pick_burst_cell_dirs(Path(ANNOT_DIR) / video.replace(" ", "-"))[int(cell)]
    return cell_dir / f"{cell_dir.name}_f{frame}.png"


def load_burst_mask(chB_path, mask_transform):
    #mask_transform: pass the dataset's mask_transform (nearest resize to 128), so the mask lines up with the model input
    burst = Image.open(burst_mask_path(chB_path)).convert('L')
    return (mask_transform(burst) > 0).float()   #[1, 128, 128]




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
        
        '''
        the codes(alternative to the latent codes in VAEs) here represent how far alaong the edit direction that 
        specific W vector(unique to a single cell) is. As there are 10 dimensions, it will be a list for each W, of length 10.

        positive means the image sits further along v_i than the population average, negative means it sits in the opposite direction from the average
        '''
        c = (w_base - w_mean) @ V #component of (w_base- w_mean) along the V directions(10 in this case, which is also the number of codes)
        codes.append(c)

    return np.array(codes, dtype=np.float32)


if __name__ == "__main__":
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)
    # loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    subset, Z, Z_indices = get_gene_burst_intensity(dataset)
    
    #inverted w latents from the train space
    C = compute_ganspace_codes(W_SPACE_DATA_DIR, PCA_DIRS_PATH, K_COMPONENTS)
    #both C, and Z should match
    #C has a row for every train cell, Z only for the burst frames -> pick C's rows by the burst indices
    assert len(C) == len(dataset), f"{len(C)} inverted w's vs {len(dataset)} dataset cells, row order would not line up"
    C = C[Z_indices]
    print(f"Evaluating DCI on {len(C)} aligned samples (Codes: {C.shape}, Factors: {Z.shape})")


    '''
    generating only the matrix, not the scores
    '''
    #standardize codes/factor to zero mean, unit variance (as in the paper)
    C_norm = (C - C.mean(0)) / (C.std(0) + 1e-8)
    Z_norm = (Z - Z.mean()) / (Z.std() + 1e-8)

    #one regressor, as there is only one known factor (burst intensity) -> R is [K_COMPONENTS, 1]
    print("fitting random forest..")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=20, n_jobs=-1)
    rf.fit(C_norm, Z_norm)
    R = rf.feature_importances_.reshape(-1, 1) #importance of each PC code for burst intensity
    print("random forest successfully fit..")

    print("\n=== importance of each GANSpace direction for gene burst intensity ===")
    for i, r in enumerate(R[:, 0]):
        print(f"  PC{i:02d}: {r:.3f}")

    out_dir = "semantic_disentanglement_eval"
    np.save(f"{out_dir}/gan_burst_R_matrix.npy", R)

    fig, ax = plt.subplots(figsize=(3, 6))
    im = ax.imshow(R, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0])
    ax.set_xticklabels(["burst_intensity"])
    ax.set_yticks(range(K_COMPONENTS))
    ax.set_yticklabels([f"PC_{i}" for i in range(K_COMPONENTS)])
    ax.set_ylabel("GANSpace Component (Code)")
    ax.set_title("random forest importance matrix R")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/gan_burst_R_matrix.png")
    print(f"saved importance matrix to {out_dir}/gan_burst_R_matrix.png")