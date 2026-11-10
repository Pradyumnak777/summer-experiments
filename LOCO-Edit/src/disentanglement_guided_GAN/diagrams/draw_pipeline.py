'''
overview diagram of the gan side: a two channel cell is inverted into stylegan2's 512-d w-space, and from there two ways
of finding directions to edit along:
    - GANspace (pca on w, unsupervised) -> top-k directions, unknown meaning
    - interfaceGAN svm (train_svm.py, supervised) -> one direction per measured burst factor, known meaning

run from src/:
    python -m disentanglement_guided_GAN.diagrams.draw_pipeline
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from PIL import Image

OUT_DIR = Path(__file__).parent

#example cell (same one as combined_viz/), shown masked like data_utils.load_target
BASE     = "20260611_HP_bh1-PP7_15_TIGRE-PCP-mSG_17-01_processed-Orthogonal Projection-146_bc_cell3_f022"
CHA_PATH = f"data/singlecell_chA_split/train/{BASE}_chA.png"
CHB_PATH = f"data/singlecell_chB_split/train/{BASE}_chB.png"
MASK     = f"data/singlecell_mask/{BASE}_mask.png"

K_SHOWN  = ["PC 1", "PC 2", "PC 3", "PC 4", "PC 5"]   #first few of the top-k, then a "..." and PC k
KNOWN    = ["intensity", "area", "aspect ratio"]      #the factors train_svm.py fits a boundary for

#colors: blue = known (anchored to a measured factor), orange = unknown (data-driven), like the vae diagram
C_KNOWN     = "#2f7ed8"
C_KNOWN_BG  = "#eaf2fc"
C_UNK       = "#f26b3a"
C_UNK_FILL  = "#f8b89c"
C_UNK_BG    = "#fdf0ea"
C_BACKBONE  = "#d5dde8"
C_INK       = "#222222"
C_MUTED     = "#555555"


def load_channel(path):
    #grayscale channel times the soft cell mask, contrast stretched to the cell's own 1-99.5 percentile for display
    mask = np.array(Image.open(MASK).convert("L").resize(Image.open(path).size, resample=Image.BILINEAR)) / 255.0
    img = np.array(Image.open(path).convert("L")).astype(np.float32) * mask
    lo, hi = np.percentile(img[mask > 0.5], [1, 99.5])
    return ((img - lo) / (hi - lo)).clip(0, 1) * (mask > 0.05)


def arrow(ax, start, end, color=C_INK, lw=1.6):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0,
                                                          mutation_scale=16), zorder=1)


def method_block(ax, x, y, w, h, title, lines, ec, fc):
    #rounded block with a bold title and a few smaller lines under it -> (left, right, y) for the arrows
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=1.8, zorder=2))
    ax.text(x + w / 2, y + h / 2 - 0.3, title, ha="center", va="top", fontsize=13, weight="bold", color=C_INK, zorder=3)
    ax.text(x + w / 2, y - 0.25, lines, ha="center", va="center", fontsize=10, color=C_MUTED, zorder=3, linespacing=1.5)
    return x, x + w, y


def direction_stack(ax, x, y_top, names, color, fill, dots_before_last=False):
    #a column of bars with their names on the left, like the latent stack in the vae diagram -> the stack's bounding box
    bar_w, bar_h, gap = 1.1, 0.42, 0.14
    y = y_top
    rows = names[:-1] + ["dots"] + names[-1:] if dots_before_last else names
    for name in rows:
        if name == "dots":
            ax.text(x + bar_w / 2, y - bar_h / 2 + 0.05, "⋮", ha="center", va="center", fontsize=16, color=C_INK)
        else:
            ax.add_patch(Rectangle((x, y - bar_h), bar_w, bar_h, fc=fill, ec=color, lw=1.0, zorder=2))
            ax.text(x - 0.15, y - bar_h / 2, name, ha="right", va="center", fontsize=11, color=C_INK)
        y -= bar_h + gap
    return y + gap   #bottom edge of the last row


def frame(ax, x0, y0, x1, y1, color):
    #dashed box around a group of directions
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="none", ec=color, lw=2.0, ls="--", zorder=1))


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(17, 8.2))
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0, 8.4)
    ax.set_aspect("equal")
    ax.axis("off")

    #input cell, two channels in grayscale
    ax.text(1.35, 7.4, "Input cell\n(two channels)", ha="center", va="top", fontsize=13, weight="bold", color=C_INK)
    img_w = 2.0
    for path, y0, name in [(CHA_PATH, 4.55, "chA"), (CHB_PATH, 1.25, "chB")]:
        ax.imshow(load_channel(path), cmap="gray", vmin=0, vmax=1, extent=(0.35, 0.35 + img_w, y0, y0 + img_w), zorder=2)
        ax.text(1.35, y0 - 0.3, name, ha="center", va="center", fontsize=12, color=C_INK)

    #brackets from the two channels into the backbone
    bx, mid = 2.75, 4.15
    for yc in (4.55 + img_w / 2, 1.25 + img_w / 2):
        ax.plot([0.35 + img_w + 0.08, bx, bx], [yc, yc, mid], color=C_INK, lw=1.4)
    arrow(ax, (bx, mid), (3.3, mid))

    #backbone: stylegan2, the cell is inverted into one shared w
    bb = Polygon([(3.3, 2.35), (3.3, 5.95), (6.2, 5.0), (6.2, 3.3)], closed=True, fc=C_BACKBONE, ec=C_INK, lw=1.6, zorder=2)
    ax.add_patch(bb)
    ax.text(4.65, 4.4, "StyleGAN2", ha="center", va="center", fontsize=14, weight="bold", color=C_INK, zorder=3)
    ax.text(4.65, 3.8, "inverted to\nw ∈ ℝ⁵¹²", ha="center", va="center", fontsize=10, color=C_MUTED, zorder=3)

    #direction stacks first: top-k from pca (unknown), one per factor from the svm (known), so the method blocks can be
    #centered on them and every arrow stays straight
    stack_x, frame_x = 13.6, 12.05
    gs_top = 7.85
    gs_bot = direction_stack(ax, stack_x, gs_top, K_SHOWN + ["PC k"], C_UNK, C_UNK_FILL, dots_before_last=True)
    frame(ax, frame_x, gs_bot - 0.15, stack_x + 1.3, gs_top + 0.15, C_UNK)
    svm_top = 2.95
    svm_bot = direction_stack(ax, stack_x, svm_top, KNOWN, C_KNOWN, C_KNOWN)
    frame(ax, frame_x, svm_bot - 0.15, stack_x + 1.3, svm_top + 0.15, C_KNOWN)

    #two branches out of the same w-space
    blk_x, blk_w, blk_h = 7.4, 3.4, 2.0
    gs = method_block(ax, blk_x, (gs_top + gs_bot) / 2, blk_w, blk_h, "GANspace (PCA)",
                      "PCA on the 512-D w-space\ntop-k directions\n(unsupervised)", C_UNK, C_UNK_BG)
    svm = method_block(ax, blk_x, (svm_top + svm_bot) / 2, blk_w, blk_h, "SVM-based",
                       "linear SVM per factor on w\nnormal of the boundary\n(supervised, InterFaceGAN)", C_KNOWN, C_KNOWN_BG)

    #elbow: out of the backbone, split, then straight into each block
    split_x = 6.8
    ax.plot([6.2, split_x], [4.15, 4.15], color=C_INK, lw=1.6, zorder=1)
    ax.plot([split_x, split_x], [svm[2], gs[2]], color=C_INK, lw=1.6, zorder=1)
    for left, right, yb in (gs, svm):
        arrow(ax, (split_x, yb), (left, yb))
        arrow(ax, (right, yb), (frame_x, yb))

    #what each group is, to the right of its stack
    tx = stack_x + 1.55
    ax.text(tx, (gs_top + gs_bot) / 2 + 0.35, "Top-k directions", ha="left", va="center", fontsize=13, weight="bold", color=C_UNK)
    ax.text(tx, (gs_top + gs_bot) / 2 - 0.35, "unknown /\ndata-driven\n(meaning not given)", ha="left", va="center",
            fontsize=10.5, style="italic", color=C_UNK)
    ax.text(tx, (svm_top + svm_bot) / 2 + 0.35, "Known directions", ha="left", va="center", fontsize=13, weight="bold", color=C_KNOWN)
    ax.text(tx, (svm_top + svm_bot) / 2 - 0.35, "anchored to measured\ngene burst factors", ha="left", va="center",
            fontsize=10.5, style="italic", color=C_KNOWN)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"gan_directions.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved to {OUT_DIR}/gan_directions.png/.pdf")
