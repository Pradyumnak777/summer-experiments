'''
this file will be used to create a preliminary dataset to finetune a CLIP model.

1. let the label space be K- {0: f"a heatmap over a cell regions affected when the {x} of the gene burst is changing", 1: "...{x}..", ...}
Here, "x" would be the concerend generative factor- like intensity, area, etc.

2. I believe the data would be of form (heatmap image, label)

run from src/:
    python -m heatmap_CLIP.data_utils
'''
from pathlib import Path
import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from gradcam.vae_gradcam1 import GradCAMVAE
from VAE_disent.model import twoChannelVAE
from VAE_disent.data_utils import twoChannelDataset
from disentanglement_semantic_eval.gan_dci import get_gene_burst_intensity

DEVICE   = torch.device("cuda:9")
DATA_DIR = "heatmap_CLIP/data"
DATA_VAL_DIR = "heatmap_CLIP/data_val"   #same layout, from the validation split (frames the guided vae never trained on)
DATA_UNKNOWN_DIR = "heatmap_CLIP/data_unknown"   #same layout, but one folder per remaining latent (z2..z15) instead of per factor
TARGETS  = {"intensity": 0, "area": 1}   #factor -> its latent code in the guided multi vae (TARGETS in vae_guided_multi.py)
ALPHA    = 0.45                           #heatmap opacity over the cell, same as the gradcam grids

#the text label set (see the global comment), at the top so finetune.py uses the exact same sentences
LABEL_FACTORS = ["intensity", "area", "aspect ratio", "major axis", "minor axis"]
SEEN = ["intensity", "area"]
label_set = {k: f"a heatmap over a cell regions affected when the {x} of the gene burst is changing"
             for k, x in enumerate(LABEL_FACTORS)}

class CLIP_Data(Dataset):
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


def heatmap_overlay(gray, cam):
    #gray: [H, W] cell channel in [0, 1], cam: [H, W] in [0, 1] -> [H, W, 3] uint8, jet heatmap blended onto the cell
    heat = cm.jet(cam)[..., :3]
    rgb = (1 - ALPHA) * gray[..., None] + ALPHA * heat
    return (rgb * 255).round().astype(np.uint8)


def gradcam_getData(vae_model_dir, set_name="train", out_dir=DATA_DIR, targets=TARGETS): #this will get the model, then use the gradCAM VAE class
    #targets: folder name -> latent code, one heatmap folder per entry
    model = twoChannelVAE(latent_dim=16).to(DEVICE)
    model.load_state_dict(torch.load(f"{vae_model_dir}/vae_final.pt", map_location=DEVICE))
    model.eval()
    gradcam = GradCAMVAE(model, model.conv4)

    #run gradcam using the GradCAMVAE class over the entire train dataset WITH gene burst masks (so the subset)
    dataset = twoChannelDataset(f"data/singlecell_chA_split/{set_name}", f"data/singlecell_chB_split/{set_name}",
                                mask_dir="data/singlecell_mask")
    subset, _, burst_idx = get_gene_burst_intensity(dataset)

    #save under folder heatmap_CLIP/data/ :- intensity/heatmaps/ - this is for the z0 latent

    #next, save area/heatmaps - this is for the z1 latent
    out = {f: Path(out_dir, f, "heatmaps") for f in targets}
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)

    #handle the file names such that the correspondence is maintained across intensity-heatmap
    #every factor saves the frame under its own base name ("<video>_cell12_f010.png"), so the same name = the same frame
    for k in tqdm(range(len(subset)), desc="gradcam"):
        x, _ = subset[k]                                             #[2, 128, 128], cell-masked
        base = dataset.chA_files[burst_idx[k]].name[:-len("_chA.png")]
        chA = (x[0].numpy() * 0.5 + 0.5).clip(0, 1)                  #un-normalize the gene burst channel to [0, 1]
        for f, dim in targets.items():
            cam = gradcam(x.unsqueeze(0).to(DEVICE), latent_idx=dim)  #[128, 128] in [0, 1]
            Image.fromarray(heatmap_overlay(chA, cam)).save(out[f] / f"{base}.png")

    #marks the cache as complete, an interrupted run doesn't leave it looking finished
    Path(out_dir, "done.txt").write_text(f"{vae_model_dir}/vae_final.pt, {set_name}, {len(subset)} frames\n")

'''
below is gradCAM data, but for all the other unknown latents (excluding z0 and z1) 
'''
def gradcam_getData_unknown(vae_model_dir, set_name="train", out_dir=DATA_UNKNOWN_DIR):
    #same frames and file names as data/, saved as data_unknown/z{d}/heatmaps/ for every code that isn't in TARGETS
    unknown = {f"z{d}": d for d in range(16) if d not in TARGETS.values()}   #z2..z15
    gradcam_getData(vae_model_dir, set_name=set_name, out_dir=out_dir, targets=unknown)


if __name__ == "__main__":
    VAE_MODEL_DIR = "vae_checkpoints_masked/guided_vae_beta=1_16_intensity-z0_area-z1_aspect-z2"

    #run gradcam_getData()
    #only once, delete heatmap_CLIP/data/ to rebuild
    if not Path(DATA_DIR, "done.txt").exists():
        gradcam_getData(VAE_MODEL_DIR)
    if not Path(DATA_VAL_DIR, "done.txt").exists():
        gradcam_getData(VAE_MODEL_DIR, set_name="validation", out_dir=DATA_VAL_DIR)
    if not Path(DATA_UNKNOWN_DIR, "done.txt").exists():
        gradcam_getData_unknown(VAE_MODEL_DIR)

    #prepare the text label set like i mentioned in the above/global comment
    #(LABEL_FACTORS, SEEN and label_set are defined at the top of the file)

    '''
    the blow is primarily a test to check if this is working, as the actual training is not performed in this script
    '''
    #pass this data/folder to CLIP_Data class, and accordingly create tuples like (image, label), and let the label
    #be matched to some "label set", like i mentioned in global comment. The label set will have [intensity, area, aspect ratio, major axis, minor axis].
    #There will be only 2 seen labels(intensity and area), and the other 3 will be unseen during training.
    dataset = CLIP_Data(DATA_DIR, LABEL_FACTORS, SEEN)
    print(f"{len(dataset)} samples")
    for k, text in label_set.items():
        n = sum(label == k for _, label in dataset.samples)
        print(f"  {k}: {n:6d} {'seen  ' if LABEL_FACTORS[k] in SEEN else 'unseen'} | {text}")

    img, label = dataset[0]
    print(f"sample 0: image {tuple(img.shape)}, label {label} -> {label_set[label]}")

    #the same frame names under every seen factor, ie the correspondence held
    names = [{p.name for p in Path(DATA_DIR, f, "heatmaps").glob("*.png")} for f in SEEN]
    assert all(n == names[0] for n in names), "intensity/area heatmaps don't line up"
    print("intensity/area heatmaps line up")
