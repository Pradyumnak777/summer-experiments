'''
saves one figure per heatmap: the heatmap, the label set, and the finetuned clip's score for every label in it
(cosine similarity + softmax prob, same as eval.py), with the chosen label highlighted

train = heatmap_CLIP/data (includes the cells finetune.py held out for its val acc, tagged in the title)
val   = heatmap_CLIP/data_val (frames the guided vae never trained on)
saved as viz/<train|val>/<factor folder>/<heatmap name>.png, plus viz/<train|val>/scores.csv with every score

run from src/:
    python -m heatmap_CLIP.viz_predictions
'''
import csv
from multiprocessing import Pool
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from heatmap_CLIP.data_utils import CLIP_Data, DATA_DIR, DATA_VAL_DIR, LABEL_FACTORS, SEEN, label_set
from heatmap_CLIP.eval import CKPT, DEVICE, load_model
from heatmap_CLIP.finetune import SEED, VAL_FRAC, split_by_cell

VIZ_DIR     = "heatmap_CLIP/viz"
SETS        = {"train": DATA_DIR, "val": DATA_VAL_DIR}
LIMIT       = None    #max heatmaps per set, None -> all of them
BATCH_SIZE  = 128
NUM_WORKERS = 8       #processes drawing the figures, the gpu part is done before

C_BAR    = "#c9c9c9"   #every label's bar
C_CHOSEN = "#2a6fdb"   #the chosen (argmax) label's bar
C_INK    = "#222222"
C_MUTED  = "#666666"


@torch.no_grad()
def score_set(dataset, model, txt_feat):
    #-> [N, 5] cosine similarities between every heatmap and every label sentence, in dataset.samples order
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    sims = []
    for images, _ in tqdm(loader, desc="scoring"):
        img_feat = model.encode_image(images.to(DEVICE), normalize=True).float()
        sims.append((img_feat @ txt_feat.T).cpu())
    return torch.cat(sims)


def plot_one(args):
    #one heatmap -> one figure: the heatmap on the left, a bar per label on the right
    path, label, sims, probs, split_tag, out_path = args
    pred = max(range(len(sims)), key=lambda k: sims[k])

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 2.3]})
    ax_img.imshow(Image.open(path).convert("RGB"))
    ax_img.set_axis_off()
    ax_img.set_title(f"true: {LABEL_FACTORS[label]}", fontsize=11, color=C_INK)

    y = range(len(LABEL_FACTORS))
    ax_bar.barh(y, probs, height=0.6, color=[C_CHOSEN if k == pred else C_BAR for k in y])
    for k in y:
        ax_bar.text(probs[k] + 0.015, k, f"p {probs[k]:.3f}  |  cos {sims[k]:.4f}", va="center", fontsize=9,
                    color=C_INK if k == pred else C_MUTED)
    ax_bar.set_yticks(list(y))
    ax_bar.set_yticklabels([f"{k}  {f} ({'seen' if f in SEEN else 'unseen'})" + ("  <- true" if k == label else "")
                            for k, f in enumerate(LABEL_FACTORS)], fontsize=10)
    for k, tick in enumerate(ax_bar.get_yticklabels()):
        tick.set_fontweight("bold" if k == label else "normal")
    ax_bar.invert_yaxis()                                     #label 0 at the top
    ax_bar.set_xlim(0, 1.32)
    ax_bar.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_bar.set_xlabel("softmax prob over the label set (logit_scale * cos)", fontsize=9, color=C_MUTED)
    ax_bar.grid(axis="x", color="#e6e6e6", lw=0.8)
    ax_bar.set_axisbelow(True)
    for side in ["top", "right"]:
        ax_bar.spines[side].set_visible(False)
    verdict = "correct" if pred == label else "WRONG"
    ax_bar.set_title(f"chosen: {LABEL_FACTORS[pred]}  ({verdict})", fontsize=11, color=C_INK, loc="left")

    fig.suptitle(f"{Path(path).name}   [{split_tag}]", fontsize=8.5, color=C_MUTED)
    fig.text(0.5, 0.015, f'label set: "{label_set[0].replace(LABEL_FACTORS[0], "{x}")}"', ha="center",
             fontsize=8.5, color=C_MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def split_tags(set_name, dataset):
    #which split each heatmap was in during finetuning, so train-cell heatmaps aren't mistaken for held-out ones
    if set_name != "train":
        return ["vae val frame, unseen in finetuning"] * len(dataset)
    _, val_set = split_by_cell(dataset, VAL_FRAC, SEED)      #same seed as finetune.py -> the same cells
    held_out = set(val_set.indices)
    return ["finetune val cell" if i in held_out else "finetune train cell" for i in range(len(dataset))]


if __name__ == "__main__":
    model, preprocess, txt_feat = load_model(CKPT)
    scale = model.logit_scale.exp().item()

    for set_name, data_dir in SETS.items():
        dataset = CLIP_Data(data_dir, LABEL_FACTORS, SEEN, transform=preprocess)
        tags = split_tags(set_name, dataset)
        if LIMIT is not None:
            dataset.samples = dataset.samples[:LIMIT // 2] + dataset.samples[-(LIMIT // 2):]   #some of each factor
            tags = tags[:LIMIT // 2] + tags[-(LIMIT // 2):]
        sims = score_set(dataset, model, txt_feat)
        probs = (scale * sims).softmax(1)

        out_root = Path(VIZ_DIR, set_name)
        jobs = []
        for (path, label), s, p, tag in zip(dataset.samples, sims.tolist(), probs.tolist(), tags):
            out_path = out_root / path.parent.parent.name / path.name      #<factor folder>/<heatmap name>.png
            out_path.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((path, label, s, p, tag, out_path))

        #every score in one csv too, easier to aggregate than the figures
        with open(out_root / "scores.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "true", "pred", "split"] + [f"cos_{x}" for x in LABEL_FACTORS] + [f"p_{x}" for x in LABEL_FACTORS])
            for path, label, s, p, tag, _ in jobs:
                pred = max(range(len(s)), key=lambda k: s[k])
                writer.writerow([path.name, LABEL_FACTORS[label], LABEL_FACTORS[pred], tag] + s + p)

        with Pool(NUM_WORKERS) as pool:
            list(tqdm(pool.imap_unordered(plot_one, jobs, chunksize=32), total=len(jobs), desc=f"plotting {set_name}"))
        print(f"{set_name}: {len(jobs)} figures, accuracy over all 5 {(sims.argmax(1) == torch.tensor([l for _, l in dataset.samples])).float().mean():.3f}")
