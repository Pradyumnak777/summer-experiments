'''
block diagram of the biomed clip finetuning (finetune.py) and its evaluation (finetune.evaluate / eval.py)

run from src/:
    python -m heatmap_CLIP.diagrams.draw_pipeline
'''
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path(__file__).parent

#block colors, by role
C_INPUT   = "#e8e8e8"
C_TRAIN   = "#f6c28b"    #encoder being finetuned
C_FROZEN  = "#9ec5e8"    #finetuned encoder, weights fixed at eval
C_FEAT    = "#b7dfb0"
C_OP      = "#d9c8ef"
C_OUT     = "#f4a6a6"

#column centers and widths, shared by both panels so the blocks line up
COL_X = [1.5, 4.4, 6.65, 9.0, 11.8]
COL_W = [2.6, 2.0, 1.6, 2.3, 2.3]


def box(ax, col, y, h, text, color, fontsize=9, weight="normal"):
    #draws a rounded block centered at (COL_X[col], y) -> (x, y, w, h), used for the arrows
    x, w = COL_X[col], COL_W[col]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="#333333", lw=1.2, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight, zorder=3)
    return (x, y, w, h)


def right(b): return (b[0] + b[2] / 2, b[1])
def left(b):  return (b[0] - b[2] / 2, b[1])
def top(b):   return (b[0], b[1] + b[3] / 2)
def bottom(b): return (b[0], b[1] - b[3] / 2)


def arrow(ax, start, end, style="-|>", ls="-", rad=0.0, color="#333333", text=None, text_xy=None):
    ax.annotate("", xy=end, xytext=start, zorder=1,
                arrowprops=dict(arrowstyle=style, color=color, lw=1.4, ls=ls, shrinkA=2, shrinkB=2,
                                connectionstyle=f"arc3,rad={rad}"))
    if text is not None:
        ax.text(*text_xy, text, ha="center", va="center", fontsize=8.5, color=color, style="italic")


def panel(ax, y0, y1, title):
    #light background behind one stage, title in its top left corner
    ax.add_patch(FancyBboxPatch((0.05, y0), 13.3, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="#fafafa", ec="#999999", lw=1.0, ls="--", zorder=0))
    ax.text(0.25, y1 - 0.25, title, ha="left", va="top", fontsize=12, weight="bold")


def draw_training(ax):
    panel(ax, 5.0, 9.6, "training  (finetune.py, FLYP-style)")
    ax.text(0.25, 9.0, "data/ :  12,890 GradCAM heatmaps per factor (z0 = intensity, z1 = area), "
            "split 90/10 by cell  |  batch 64, 5 epochs", fontsize=9, color="#444444")

    img_y, txt_y = 7.85, 6.05
    #image tower
    b_img  = box(ax, 0, img_y, 1.15, "heatmap batch\nGradCAM jet over chA\n[B, 3, 224, 224]", C_INPUT)
    b_ienc = box(ax, 1, img_y, 1.0, "image encoder\nViT-B/16", C_TRAIN, weight="bold")
    b_if   = box(ax, 2, img_y, 0.8, "img_feat\n[B, 512]", C_FEAT)
    #text tower
    b_txt  = box(ax, 0, txt_y, 1.5, "label sentence per image\nlabel_tokens[labels]\n\"a heatmap over a cell regions\n"
                 "affected when the {intensity |\narea} of the gene burst ...\"", C_INPUT, fontsize=8.5)
    b_tenc = box(ax, 1, txt_y, 1.0, "text encoder\nPubMedBERT", C_TRAIN, weight="bold")
    b_tf   = box(ax, 2, txt_y, 0.8, "txt_feat\n[B, 512]", C_FEAT)
    #contrastive head
    mid_y = (img_y + txt_y) / 2
    b_sim  = box(ax, 3, mid_y, 1.2, "B x B similarity\nlogit_scale ·\nimg_feat @ txt_featᵀ", C_OP)
    b_loss = box(ax, 4, mid_y, 1.2, "symmetric CLIP loss\nimage i <-> text i\n(CE over rows + cols)", C_OUT, weight="bold")

    for a, b in [(b_img, b_ienc), (b_ienc, b_if), (b_txt, b_tenc), (b_tenc, b_tf), (b_sim, b_loss)]:
        arrow(ax, right(a), left(b))
    arrow(ax, right(b_if), left(b_sim), rad=-0.15)
    arrow(ax, right(b_tf), left(b_sim), rad=0.15)

    #backprop goes into both towers
    arrow(ax, top(b_loss), top(b_ienc), ls="--", rad=0.25, color="#c0392b",
          text="backprop into both towers\n(AdamW, lr 1e-5, warmup + cosine)", text_xy=(9.0, 8.3))
    arrow(ax, bottom(b_loss), bottom(b_tenc), ls="--", rad=-0.25, color="#c0392b")


def draw_eval(ax):
    panel(ax, 0.1, 4.7, "evaluation  (finetune.evaluate per epoch, eval.py per heatmap)")
    ax.text(0.25, 4.1, "val split of data/ (seen heatmaps), or data_unknown/z2..z15 to probe the unseen labels  |  "
            "best.pt = best val seen acc", fontsize=9, color="#444444")

    img_y, txt_y = 3.0, 1.2
    b_img  = box(ax, 0, img_y, 1.0, "heatmap(s)\n[N, 3, 224, 224]", C_INPUT)
    b_ienc = box(ax, 1, img_y, 1.0, "image encoder\n(finetuned, no grad)", C_FROZEN, weight="bold")
    b_if   = box(ax, 2, img_y, 0.8, "img_feat\n[N, 512]", C_FEAT)
    b_txt  = box(ax, 0, txt_y, 1.4, "all 5 label sentences\nseen: intensity, area\nunseen: aspect ratio,\n"
                 "major axis, minor axis", C_INPUT, fontsize=8.5)
    b_tenc = box(ax, 1, txt_y, 1.0, "text encoder\n(finetuned, no grad)", C_FROZEN, weight="bold")
    b_tf   = box(ax, 2, txt_y, 0.8, "txt_feat\n[5, 512]", C_FEAT)

    mid_y = (img_y + txt_y) / 2
    b_sim  = box(ax, 3, mid_y, 1.2, "cosine similarity\nimg_feat @ txt_featᵀ\n[N, 5]", C_OP)
    b_out  = box(ax, 4, mid_y, 1.9, "argmax -> predicted factor\n\nseen acc: argmax over 2\nall-5 acc: argmax over 5\n\n"
                 "eval.py: softmax(logit_scale\n· sims) -> probs", C_OUT, fontsize=8.5)

    for a, b in [(b_img, b_ienc), (b_ienc, b_if), (b_txt, b_tenc), (b_tenc, b_tf), (b_sim, b_out)]:
        arrow(ax, right(a), left(b))
    arrow(ax, right(b_if), left(b_sim), rad=-0.15)
    arrow(ax, right(b_tf), left(b_sim), rad=0.15)


def legend(ax):
    items = [(C_INPUT, "input"), (C_TRAIN, "trained"), (C_FROZEN, "finetuned, fixed"), (C_FEAT, "features"),
             (C_OP, "similarity"), (C_OUT, "loss / output")]
    for k, (c, name) in enumerate(items):
        x = 2.2 + k * 1.85
        ax.add_patch(FancyBboxPatch((x, -0.42), 0.3, 0.22, boxstyle="round,pad=0.01", fc=c, ec="#333333", lw=0.8))
        ax.text(x + 0.4, -0.31, name, va="center", fontsize=9)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(-0.6, 9.7)
    ax.axis("off")

    draw_training(ax)
    draw_eval(ax)
    legend(ax)

    fig.suptitle("BiomedCLIP finetuning on GradCAM heatmaps", fontsize=14, weight="bold", y=0.985)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"finetune_pipeline.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved to {OUT_DIR}/finetune_pipeline.png/.pdf")
