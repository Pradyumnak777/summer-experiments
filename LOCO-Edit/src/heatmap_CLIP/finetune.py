'''
here, BioMed-CLIP will be loaded and finetuned.

finetuning follows FLYP (Goyal et al., CVPR 2023, "finetune like you pretrain"): each image is paired with the sentence of
its label, and both towers are trained with the same contrastive loss clip was pretrained with. classifying is then zero-shot,
ie the image goes to the label sentence its features are closest to

run from src/:
    python -m heatmap_CLIP.finetune
'''
import csv
import math
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import open_clip


from heatmap_CLIP.data_utils import CLIP_Data, DATA_DIR, LABEL_FACTORS, SEEN, label_set

DEVICE       = torch.device("cuda:9")
BATCH_SIZE   = 64
NUM_EPOCHS   = 5
LR           = 1e-5
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
VAL_FRAC     = 0.1    #fraction of cells held out for validation
SEED         = 42
SAVE_DIR     = "heatmap_CLIP/checkpoints"


def split_by_cell(dataset, val_frac, seed):
    #split by cell, not by frame: neighbouring frames of one cell are near copies, so a frame split would leak.
    #file stem is "<video>_cell12_f010" -> cell key "<video>_cell12"
    cell_of = [p.stem.rsplit("_f", 1)[0] for p, _ in dataset.samples]
    cells = sorted(set(cell_of))
    random.Random(seed).shuffle(cells)
    val_cells = set(cells[:int(len(cells) * val_frac)])
    train_idx = [i for i, c in enumerate(cell_of) if c not in val_cells]
    val_idx   = [i for i, c in enumerate(cell_of) if c in val_cells]
    print(f"{len(cells) - len(val_cells)} train cells ({len(train_idx)} samples), {len(val_cells)} val cells ({len(val_idx)} samples)")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def warmup_cosine(step, total_steps):
    #lr multiplier: linear warmup, then cosine decay to 0 (the schedule FLYP uses)
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    return 0.5 * (1 + math.cos(math.pi * (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)))


@torch.no_grad()
def evaluate(model, loader, label_tokens, seen_idx):
    #zero-shot classification against the label sentences. the text tower is trained too, so their features are recomputed
    #seen acc: picking among the seen sentences only (chance = 1 / len(SEEN))
    #all acc: picking among all 5, so a seen heatmap pulled towards an unseen sentence counts as wrong
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        txt_feat = model.encode_text(label_tokens, normalize=True).float()        #[5, 512]
    correct_seen = correct_all = n = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            img_feat = model.encode_image(images, normalize=True)
        sims = img_feat.float() @ txt_feat.T                                       #[B, 5]
        correct_all  += (sims.argmax(1) == labels).sum().item()
        correct_seen += (seen_idx[sims[:, seen_idx].argmax(1)] == labels).sum().item()
        n += len(labels)
    model.train()
    return correct_seen / n, correct_all / n


if __name__ == "__main__":
    torch.manual_seed(SEED)

    #load the data using the CLIP_Data dataset
    #the clip preprocess is set as its transform once the model is loaded below
    dataset = CLIP_Data(DATA_DIR, LABEL_FACTORS, SEEN)
    train_set, val_set = split_by_cell(dataset, VAL_FRAC, SEED) #val set is used for eval later...

    #load bioMedCLIP
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model = model.to(DEVICE)

    #preprocess_val (resize + center crop to 224) for training too, preprocess_train's random crop can cut off the cell's edge
    dataset.transform = preprocess_val
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    #biomed clip needs (image, "text"), not image label
    #during finetuning then, as my dataloader outptus will be of the form- (image, label),
    #it must be made into (image, "text"), using the mapping
    #the label is the index into label_set, so all 5 sentences are tokenized once and label_tokens[labels] is the batch's text
    label_tokens = tokenizer([label_set[k] for k in range(len(LABEL_FACTORS))]).to(DEVICE)   #[5, 256]
    seen_idx = torch.tensor([LABEL_FACTORS.index(f) for f in SEEN], device=DEVICE)

    #NOTE: with only 2 seen labels, ~half the batch shares an image's sentence, and the contrastive loss counts those
    #identical sentences as negatives too. so the loss can't reach 0 (its floor is ~log(images per label in the batch)),
    #the val accuracy is the number to watch, not the loss
    clip_loss = open_clip.ClipLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: warmup_cosine(step, total_steps))

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    log_file = open(f"{SAVE_DIR}/loss_log.csv", "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["global_step", "epoch", "loss", "lr", "logit_scale"])
    val_file = open(f"{SAVE_DIR}/val_log.csv", "w", newline="")
    val_writer = csv.writer(val_file)
    val_writer.writerow(["epoch", "val_seen_acc", "val_all_acc"])

    #epoch -1 = zero-shot, before any finetuning
    seen_acc, all_acc = evaluate(model, val_loader, label_tokens, seen_idx)
    val_writer.writerow([-1, seen_acc, all_acc])
    print(f"zero-shot | val seen acc {seen_acc:.3f} | val all-5 acc {all_acc:.3f}", flush=True)

    best_acc = -1.0
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            texts = label_tokens[labels]                                           #[B, 256], each image's own label sentence
            with torch.autocast("cuda", dtype=torch.bfloat16):
                img_feat = model.encode_image(images, normalize=True)              #[B, 512]
                txt_feat = model.encode_text(texts, normalize=True)                #[B, 512]
            #symmetric clip loss: image i should match text i (and text i image i) against the rest of the batch
            loss = clip_loss(img_feat.float(), txt_feat.float(), model.logit_scale.exp())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            log_writer.writerow([global_step, epoch, loss.item(), scheduler.get_last_lr()[0], model.logit_scale.exp().item()])
            global_step += 1
        log_file.flush()

        seen_acc, all_acc = evaluate(model, val_loader, label_tokens, seen_idx)
        val_writer.writerow([epoch, seen_acc, all_acc])
        val_file.flush()
        print(f"epoch {epoch} | last loss {loss.item():.3f} | val seen acc {seen_acc:.3f} | val all-5 acc {all_acc:.3f}", flush=True)

        #only the best one is kept, each save is ~800 MB
        if seen_acc > best_acc:
            best_acc = seen_acc
            torch.save(model.state_dict(), f"{SAVE_DIR}/best.pt")

    log_file.close()
    val_file.close()
    print(f"best val seen acc {best_acc:.3f}, saved to {SAVE_DIR}/best.pt")
