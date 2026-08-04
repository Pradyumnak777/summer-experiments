import matplotlib.pyplot as plt
from VAE_disent.data_utils import twoChannelDataset

CHA_DIR  = "data/singlecell_chA_split/train"
CHB_DIR  = "data/singlecell_chB_split/train"
MASK_DIR = "data/singlecell_mask"
N_SAMPLES = 6
OUT_PATH  = "debug_masked_chA.png"

dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)

fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(9, 3 * N_SAMPLES))

for i in range(N_SAMPLES):
    x, mask = dataset[i]                 # x: [2,128,128] in [-1,1], mask: [1,128,128] in {0,1}
    src_img = x[0:1]                     # chA channel

    src_img_masked = src_img * mask - (1 - mask)

    to_disp = lambda t: (t[0] * 0.5 + 0.5).clamp(0, 1).numpy()   # un-normalize -> [0,1]

    axes[i, 0].imshow(to_disp(src_img), cmap="gray")
    axes[i, 0].set_title(f"chA real (idx {i})"); axes[i, 0].axis("off")

    axes[i, 1].imshow(mask[0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes[i, 1].set_title("mask"); axes[i, 1].axis("off")

    axes[i, 2].imshow(to_disp(src_img_masked), cmap="gray")
    axes[i, 2].set_title("chA masked (fed to encoder)"); axes[i, 2].axis("off")

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=120)
print(f"saved -> {OUT_PATH}")