'''
gan version of heatmap_CLIP/data_utils.py: the same heatmap folders, but from the stylegan2 directions instead of the
guided vae's latents, so clip can be finetuned/evaluated on them the same way

known directions (the 3 svm ext30 boundaries, disentanglement_guided_GAN/svm_models/boundary_<factor>_ext30.npy):
    data/<factor>/heatmaps/<base>.png        train burst frames (the 12,890 inverted ones)
    data_val/<factor>/heatmaps/<base>.png    validation burst frames (inverted here first, see below)
unknown directions (the top 10 ganspace pca directions, the same dirs.npz gan_dci.py / PCA_svm.py use):
    data_unknown/pca<k>/heatmaps/<base>.png  train burst frames, pca0 = the most variance
same base names ("<video>_cell12_f010.png") in every folder, so the same name = the same frame

heatmap: the same gan heatmap as the last column of disentanglement_guided_GAN/testing/svm_edit.py, the chA one:
|J d| on chA at the frame's inverted w, ie how much every chA pixel moves for a small step along the direction d (the
jacobian-vector product of gradcam/GAN_grad_viz_pca.jacobian_gen, batched here), drawn with svm_edit's heat_overlay
(min-max per image, jet with an opacity that grows with the heat) over the reconstruction's chA, where the jacobian is taken.
all in fp32 with tf32 off (see jvp_batch), batching then gives exactly the same maps as jacobian_gen one at a time

the validation burst frames were never inverted, so they're inverted first, the same way as
disentanglement_guided_GAN/data_utils.py does the train ones, and saved next to them (VAL_W_DIR) so they can be reused

run from src/ (gpu 9 only):
    python -m heatmap_CLIP_GAN.data_utils
'''
from pathlib import Path

import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from disentanglement_guided_GAN import data_utils as gan_data   #PKL, load_target, the inversion functions, train SAVE_DIR
from disentanglement_semantic_eval.gan_dci import PCA_DIRS_PATH, BURST_FACTORS
from VAE_disent.data_utils import twoChannelDataset
from disentanglement_guided_GAN.testing.svm_edit import heat_overlay   #the overlay of svm_edit.py's heatmap column

import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

#config stuff
DEVICE           = torch.device("cuda:9")
DATA_DIR         = "heatmap_CLIP_GAN/data"
DATA_VAL_DIR     = "heatmap_CLIP_GAN/data_val"       #same layout, from the validation split
DATA_UNKNOWN_DIR = "heatmap_CLIP_GAN/data_unknown"   #one folder per pca direction instead of per factor
SVM_DIR          = "disentanglement_guided_GAN/svm_models"
SVM_TAG          = "_ext30"
K_UNKNOWN        = 10                                 #top-k ganspace directions
TRAIN_W_DIR      = gan_data.SAVE_DIR                  #GAN/gan_inverted_ws_gene_burst_masked_train
VAL_W_DIR        = "GAN/gan_inverted_ws_gene_burst_masked_val"
INV_BATCH        = 64                                 #inversion batch, same as disentanglement_guided_GAN/data_utils.py
FRAMES_PER_BATCH = 8                                  #frames per jacobian batch (x the number of directions), ~8 GB on gpu 9, 32 runs out
# ALPHA            = 0.45                               #heatmap opacity over the cell, same as heatmap_CLIP


#the text label set, the same sentences/seen split as heatmap_CLIP/data_utils.py so the gan and vae clips are comparable.
#(copied, not imported: importing heatmap_CLIP.data_utils runs gradcam/vae_gradcam1.py's dataset setup)
#aspect heatmaps exist here too (data/aspect/), but the aspect svm is near chance (val spearman 0.07), so it stays unseen
LABEL_FACTORS = ["intensity", "area", "aspect ratio", "major axis", "minor axis"]
SEEN = ["intensity", "area"]
label_set = {k: f"a heatmap over a cell regions affected when the {x} of the gene burst is changing"
             for k, x in enumerate(LABEL_FACTORS)}


class CLIP_Data(Dataset):
    #same as heatmap_CLIP.data_utils.CLIP_Data: (heatmap, label) pairs from data_dir/<seen factor>/heatmaps/*.png
    def __init__(self, data_dir, label_factors, seen, transform=None):
        super().__init__()
        #label = index of the factor in label_factors (the full label set), so seen and unseen labels share one index space
        #transform: pass the clip preprocess here when training, defaults to a plain [3, H, W] tensor in [0, 1]
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.samples = []
        for f in seen:
            label = label_factors.index(f)
            self.samples += [(p, label) for p in sorted(Path(data_dir, f, "heatmaps").glob("*.png"))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert("RGB")), label


def known_dirs():
    #factor -> [512] unit svm normal (low -> high)
    return {f: np.load(f"{SVM_DIR}/boundary_{f}{SVM_TAG}.npy")[0].astype(np.float32) for f in BURST_FACTORS}


def unknown_dirs(k=K_UNKNOWN):
    #"pca<i>" -> [512] unit ganspace direction (its sign is arbitrary, doesn't matter for |J d|)
    Vt = np.load(PCA_DIRS_PATH)["Vt"][:k].astype(np.float32)
    return {f"pca{i}": Vt[i] for i in range(k)}


# def heatmap_overlay(gray, cam):
#     #same as heatmap_CLIP.data_utils.heatmap_overlay (copied, importing that file runs gradcam/vae_gradcam1.py's setup):
#     #gray: [H, W] cell channel in [0, 1], cam: [H, W] in [0, 1] -> [H, W, 3] uint8, jet heatmap blended onto the cell
#     heat = cm.jet(cam)[..., :3]
#     rgb = (1 - ALPHA) * gray[..., None] + ALPHA * heat
#     return (rgb * 255).round().astype(np.uint8)


def jvp_batch(G, ws, dirs):
    #ws, dirs: [B, 512] -> J_i d_i for every row [B, 2, H, W] + the recon of every row [B, 2, H, W] in [0, 1]
    #same double backward trick as jacobian_gen (g = J^T u with a dummy u, then d(g . d)/du = J d), but with a different
    #direction per row. the rows don't interact in the synthesis network (no batch statistics, constant noise)
    #fp32 + tf32 off like jacobian_gen: with the fp16 layers, batch 1 vs > 1 take different conv paths and don't agree,
    #with both off they're identical (and match a finite difference to its own ~2-3% error). tf32 is put back after
    prev = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # ws = ws.detach().requires_grad_(True)
        # # img = G.synthesis(ws.unsqueeze(1).repeat(1, G.num_ws, 1), noise_mode='const')   #shared W, like the inversions
        # img = G.synthesis(ws.unsqueeze(1).repeat(1, G.num_ws, 1), noise_mode='const', force_fp32=True)
        # u = torch.zeros_like(img, requires_grad=True)
        # g = torch.autograd.grad(img, ws, grad_outputs=u, create_graph=True)[0]          #[B, 512]
        # jvp = torch.autograd.grad((g * dirs).sum(), u)[0]                               #[B, 2, H, W]
        # return jvp, ((img.detach() + 1) / 2).clamp(0, 1)
        #forward mode ad instead of the double backward: the same J d (matches it to ~4e-4), in about one forward pass,
        #~20x faster and less memory (the double backward was ~22 h for every frame x direction, this is ~0.5 h)
        synth = lambda x: G.synthesis(x.unsqueeze(1).repeat(1, G.num_ws, 1), noise_mode='const', force_fp32=True)
        with torch.no_grad():
            img, jvp = torch.func.jvp(synth, (ws.detach(),), (dirs,))                   #[B, 2, H, W] each
        return jvp, ((img + 1) / 2).clamp(0, 1)
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev


def invert_val(G, D):
    #the validation burst frames -> VAL_W_DIR/<base>.npz, the same file format as disentanglement_guided_GAN/data_utils.py.
    #labels use the TRAIN medians (the svms were trained on those), already saved frames are skipped (resumable)
    dataset = twoChannelDataset("data/singlecell_chA_split/validation", "data/singlecell_chB_split/validation",
                                mask_dir=gan_data.MASK_DIR)
    _, factors, burst_idx = gan_data.get_filtered_data(dataset)
    thresholds = np.load(next(Path(TRAIN_W_DIR).glob("*.npz")))["thresholds"]
    labels = (factors > thresholds).astype(np.int64)

    Path(VAL_W_DIR).mkdir(parents=True, exist_ok=True)
    bases = [dataset.chA_files[i].name[:-len("_chA.png")] for i in burst_idx]
    pending = [j for j, b in enumerate(bases) if not (Path(VAL_W_DIR) / f"{b}.npz").exists()]
    print(f"validation: {len(bases) - len(pending)}/{len(bases)} burst frames already inverted, {len(pending)} left")
    if not pending:
        return
    w_avg, w_std = gan_data.compute_w_stats(G, DEVICE)
    for start in tqdm(range(0, len(pending), INV_BATCH), desc="inverting val burst frames"):
        batch = pending[start:start + INV_BATCH]
        targets = torch.from_numpy(np.stack([
            gan_data.load_target(dataset.chA_files[burst_idx[j]], dataset.chB_files[burst_idx[j]]) for j in batch]))
        n_pad = -len(batch) % 4   #D's minibatch-std layer needs a multiple of 4, see disentanglement_guided_GAN/data_utils.py
        if n_pad:
            targets = torch.cat([targets, targets[-1:].repeat(n_pad, 1, 1, 1)], dim=0)
        ws = gan_data.project_batch(G, D, targets, w_avg, w_std, device=DEVICE)[:len(batch)]
        for j, w in zip(batch, ws[:, 0].cpu().numpy()):
            np.savez(Path(VAL_W_DIR) / f"{bases[j]}.npz", w=w, base=bases[j],
                     chA_path=str(dataset.chA_files[burst_idx[j]]), chB_path=str(dataset.chB_files[burst_idx[j]]),
                     factor_names=np.array(BURST_FACTORS), factors=factors[j], labels=labels[j], thresholds=thresholds)


def make_heatmaps(G, w_dir, out_dir, dirs):
    #every inverted frame in w_dir x every direction in dirs -> out_dir/<dir name>/heatmaps/<base>.png
    #frames whose heatmaps are all there already are skipped (resumable), done.txt marks the folder as complete
    out = {name: Path(out_dir, name, "heatmaps") for name in dirs}
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)
    names = list(dirs)
    D_mat = torch.from_numpy(np.stack([dirs[n] for n in names])).to(DEVICE)   #[n_dirs, 512]

    files = sorted(Path(w_dir).glob("*.npz"))
    todo = [f for f in files if not all((out[n] / f"{f.stem}.png").exists() for n in names)]
    print(f"{out_dir}: {len(files) - len(todo)}/{len(files)} frames done already, {len(todo)} left, {len(names)} directions")
    for start in tqdm(range(0, len(todo), FRAMES_PER_BATCH), desc=f"heatmaps -> {out_dir}"):
        batch = [np.load(f) for f in todo[start:start + FRAMES_PER_BATCH]]
        w = torch.from_numpy(np.stack([b["w"] for b in batch])).to(DEVICE)       #[B, 512]
        #every frame with every direction: rows are frame-major, [B * n_dirs, 512]
        ws = w.repeat_interleave(len(names), dim=0)
        ds = D_mat.repeat(len(batch), 1)
        # heat = jvp_batch(G, ws, ds).norm(dim=1)                                  #[B * n_dirs, H, W], l2 over chA/chB
        jvp, recon = jvp_batch(G, ws, ds)
        heat = jvp[:, 0].abs()                                                   #[B * n_dirs, H, W], |J d| on chA, like svm_edit
        heat = heat.view(len(batch), len(names), *heat.shape[-2:]).cpu().numpy()
        recon = recon[::len(names), 0].cpu().numpy()                             #[B, H, W], recon chA (same for every direction)

        for b, h, rec in zip(batch, heat, recon):
            # #the input chA the overlay goes on, preprocessed like the inversion saw it (128 px, cell-masked)
            # chA = gan_data.load_target(Path(str(b["chA_path"])), Path(str(b["chB_path"])))[0].astype(np.float32) / 255.0
            for n, hm in zip(names, h):
                # cam = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)               #per image min-max, like the gradcam cam
                # Image.fromarray(heatmap_overlay(chA, cam)).save(out[n] / f"{b['base']}.png")
                #svm_edit's heat_overlay: min-max inside, [H, W, 3] in [0, 1] -> uint8 png
                Image.fromarray((heat_overlay(rec, hm) * 255).round().astype(np.uint8)).save(out[n] / f"{b['base']}.png")

    Path(out_dir, "done.txt").write_text(f"{w_dir}, {len(files)} frames, directions: {', '.join(names)} "
                                         f"(svm {SVM_TAG} / ganspace {PCA_DIRS_PATH})\n")


if __name__ == "__main__":
    torch.cuda.set_device(DEVICE)   #so anything that picks "the current gpu" lands on 9 too

    with open(gan_data.PKL, 'rb') as f:
        network_dict = gan_data.legacy.load_network_pkl(f)
    G = network_dict['G_ema'].to(DEVICE).eval().requires_grad_(False)
    D = network_dict['D'].to(DEVICE).eval().requires_grad_(False)   #only for inverting the val frames

    #only once per folder, delete a folder (or its done.txt) to rebuild it
    if not Path(DATA_DIR, "done.txt").exists():
        make_heatmaps(G, TRAIN_W_DIR, DATA_DIR, known_dirs())
    #skipped for now: data_val is only an extra check (like heatmap_CLIP's viz_predictions.py), finetuning uses data/ only.
    #256/689 val frames are already inverted in VAL_W_DIR, uncomment to finish them (resumes) and make data_val
    # if not Path(DATA_VAL_DIR, "done.txt").exists():
    #     invert_val(G, D)
    #     make_heatmaps(G, VAL_W_DIR, DATA_VAL_DIR, known_dirs())
    if not Path(DATA_UNKNOWN_DIR, "done.txt").exists():
        make_heatmaps(G, TRAIN_W_DIR, DATA_UNKNOWN_DIR, unknown_dirs())

    #the same frames under every folder, ie the correspondence held
    names = [{p.name for p in Path(DATA_DIR, f, "heatmaps").glob("*.png")} for f in BURST_FACTORS]
    names += [{p.name for p in Path(DATA_UNKNOWN_DIR, f"pca{i}", "heatmaps").glob("*.png")} for i in range(K_UNKNOWN)]
    assert all(n == names[0] for n in names), "known/unknown heatmaps don't line up"
    print(f"all {len(names)} train folders line up ({len(names[0])} frames each)")
