'''
training-only diagram of the biomed clip finetuning (finetune.py), with the B x B similarity drawn as a partial table
of an example batch (4 of the 64), so the targets and the duplicate sentences are visible

run from src/:
    python -m heatmap_CLIP.diagrams.draw_training
'''
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).parent

C_INPUT = "#e8e8e8"
C_TRAIN = "#f6c28b"
C_FEAT  = "#b7dfb0"
C_OUT   = "#f4a6a6"
#table cells, by what the loss does with them
C_POS   = "#9fd89a"    #diagonal, the target
C_DUP   = "#f7d58b"    #same sentence as the target, still a negative
C_NEG   = "#eeeeee"    #different sentence, a real negative
C_INK   = "#222222"
C_MUTED = "#666666"

#example batch, as in the table: intensity, area, intensity, area
BATCH = ["int", "area", "int", "area"]
CELL_W, CELL_H = 0.68, 0.62
TABLE_X, TABLE_TOP = 9.8, 4.3    #top left corner of the cells (headers sit outside it)


def box(ax, x, y, w, h, text, color, fontsize=9.5, weight="normal"):
    #rounded block centered at (x, y) -> (x, y, w, h), used for the arrows
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="#333333", lw=1.2, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight, zorder=3)
    return (x, y, w, h)


def right(b): return (b[0] + b[2] / 2, b[1])
def left(b):  return (b[0] - b[2] / 2, b[1])
def top(b):   return (b[0], b[1] + b[3] / 2)
def bottom(b): return (b[0], b[1] - b[3] / 2)


def arrow(ax, start, end, ls="-", rad=0.0, color="#333333"):
    ax.annotate("", xy=end, xytext=start, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, ls=ls, shrinkA=2, shrinkB=2,
                                connectionstyle=f"arc3,rad={rad}"))


def path_arrow(ax, pts, color="#c0392b"):
    #dashed polyline through pts, arrowhead on the last segment, so it can go around the table instead of through it
    xs, ys = zip(*pts[:-1])
    ax.plot(xs, ys, color=color, lw=1.4, ls="--", zorder=1)
    arrow(ax, pts[-2], pts[-1], ls="--", color=color)


def cell(ax, i, j, text, color, fontsize=11, weight="normal"):
    #cell in row i, column j of the table (row 0 at the top)
    x, y = TABLE_X + j * CELL_W, TABLE_TOP - (i + 1) * CELL_H
    ax.add_patch(Rectangle((x, y), CELL_W, CELL_H, fc=color, ec="white", lw=2, zorder=2))
    ax.text(x + CELL_W / 2, y + CELL_H / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight,
            color=C_INK, zorder=3)


def draw_table(ax):
    n = len(BATCH)
    #column headers: the batch's texts, one copy of the label sentence per image
    for j, f in enumerate(BATCH):
        ax.text(TABLE_X + (j + 0.5) * CELL_W, TABLE_TOP + 0.1, f"txt{j}\nS_{f}", ha="center", va="bottom",
                fontsize=9, color=C_INK)
    ax.text(TABLE_X + (n + 0.5) * CELL_W, TABLE_TOP + 0.1, "...", ha="center", va="bottom", fontsize=11)
    #row headers: the batch's heatmaps
    for i, f in enumerate(BATCH):
        ax.text(TABLE_X - 0.1, TABLE_TOP - (i + 0.5) * CELL_H, f"img{i}  ({f} map)", ha="right", va="center",
                fontsize=9, color=C_INK)
    ax.text(TABLE_X - 0.1, TABLE_TOP - (n + 0.5) * CELL_H, "...", ha="right", va="center", fontsize=11)

    for i, fi in enumerate(BATCH):
        for j, fj in enumerate(BATCH):
            if i == j:
                cell(ax, i, j, "✓", C_POS, fontsize=14, weight="bold")
            elif fi == fj:
                cell(ax, i, j, "same", C_DUP, fontsize=8.5)
            else:
                cell(ax, i, j, "−", C_NEG)
        cell(ax, i, n, "...", "none")
    for j in range(n + 1):
        cell(ax, n, j, "...", "none")

    ax.text(TABLE_X + (n + 1) * CELL_W / 2, TABLE_TOP + 0.75, "B x B similarity  (logit_scale · cos), example 4 of 64",
            ha="center", va="bottom", fontsize=9.5, weight="bold", color=C_INK)

    #legend in the bottom left, out of the way of the backprop paths
    items = [(C_POS, "target (positive)"), (C_DUP, "same sentence, but counted as a negative"),
             (C_NEG, "different sentence, negative")]
    for k, (c, name) in enumerate(items):
        y = 1.2 - k * 0.32
        ax.add_patch(Rectangle((0.3, y - 0.1), 0.28, 0.2, fc=c, ec="#999999", lw=0.6))
        ax.text(0.68, y, name, va="center", fontsize=8.5, color=C_MUTED)

    #(x, y, w, h) of the whole table, cells only, for the arrows
    w, h = (n + 1) * CELL_W, (n + 1) * CELL_H
    return (TABLE_X + w / 2, TABLE_TOP - h / 2, w, h)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(15.5, 7.4))
    ax.set_xlim(0, 15.7)
    ax.set_ylim(0.3, 7.7)
    ax.axis("off")

    txt_y, img_y = 6.0, 2.75
    b_txt  = box(ax, 1.7, txt_y, 3.0, 1.2, "label sentence per image\nlabel_tokens[labels]\n"
                 "\"... when the {intensity | area}\nof the gene burst is changing\"", C_INPUT, fontsize=8.5)
    b_tenc = box(ax, 4.6, txt_y, 2.0, 1.0, "text encoder\nPubMedBERT", C_TRAIN, weight="bold")
    b_tf   = box(ax, 6.7, txt_y, 1.5, 0.8, "txt_feat\n[B, 512]", C_FEAT)
    b_img  = box(ax, 1.7, img_y, 3.0, 1.1, "heatmap batch\nGradCAM jet over chA\n[B, 3, 224, 224]", C_INPUT)
    b_ienc = box(ax, 4.6, img_y, 2.0, 1.0, "image encoder\nViT-B/16", C_TRAIN, weight="bold")
    b_if   = box(ax, 6.7, img_y, 1.5, 0.8, "img_feat\n[B, 512]", C_FEAT)
    b_tab  = draw_table(ax)
    b_loss = box(ax, 14.6, b_tab[1], 1.9, 1.5, "symmetric\nCLIP loss\n\nCE per row\n+ CE per column", C_OUT,
                 weight="bold")

    for a, b in [(b_txt, b_tenc), (b_tenc, b_tf), (b_img, b_ienc), (b_ienc, b_if)]:
        arrow(ax, right(a), left(b))
    arrow(ax, right(b_tf), (b_tab[0], TABLE_TOP + 0.75 + 0.35), rad=-0.25)     #texts -> the columns
    arrow(ax, right(b_if), (TABLE_X - 1.55, img_y))                            #images -> the rows
    arrow(ax, right(b_tab), left(b_loss))

    #backprop into both towers, routed above and below the table
    lx = b_loss[0]
    path_arrow(ax, [top(b_loss), (lx, 7.2), (b_tenc[0], 7.2), top(b_tenc)])
    path_arrow(ax, [bottom(b_loss), (lx, 0.75), (b_ienc[0], 0.75), bottom(b_ienc)])
    ax.text((lx + b_tenc[0]) / 2, 7.3, "backprop into both towers  (AdamW, lr 1e-5)", ha="center", va="bottom",
            fontsize=8.5, color="#c0392b", style="italic")

    ax.set_title("BiomedCLIP finetuning on GradCAM heatmaps (training)", fontsize=13, weight="bold")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"training_pipeline.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved to {OUT_DIR}/training_pipeline.png/.pdf")
