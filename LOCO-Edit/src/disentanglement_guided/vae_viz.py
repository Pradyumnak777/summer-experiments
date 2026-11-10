'''
just taking a trained model and then visaulizing the the traversal per dimension

same traversal as the training previews in vae_guided.py / vae_guided_multi.py, but for ONE chA/chB pair picked by CELL_IDX
(change it to look at different cells). CELL_IDX indexes the burst subset (get_burst_factors, same frames as
get_gene_burst_intensity), so every index is a frame with a hand-drawn burst mask.
each image is one row: input | burst mask | mask overlaid on the input (red) | traversal steps -SPAN..+SPAN,
for both chA and chB. the walk is centered on the cell's own code value (mu[dim] - SPAN .. mu[dim] + SPAN),
so the middle step is exactly the reconstruction, it gets a green border
(unlike the training previews, which set the code to -SPAN..+SPAN absolute)
every traversal step (recon included) has the burst mask outline drawn on it (red, 1 px just outside the mask,
so the pixels inside the burst stay untouched), to show where to look. for the shape codes (RECON_ONLY_RING, area/aspect)
only the recon gets it, as the burst is meant to change shape along the walk and the original outline would hide that
every image has 2 rows:
    top (static):   the above, with the hand-drawn burst mask in red
    bottom (tracked): input | burst mask DETECTED on the input (title = its IoU with the hand-drawn one) | overlay (cyan),
                      then every step gets the outline of its own tracked burst mask, from mask_track (christina's detection,
                      see burst_detect.py) on the decoded chA. the target codes' titles get the mean intensity inside it,
                      "-" where no burst was detected

the target codes' images are drawn with matplotlib instead, with a title per panel. for intensity, each panel is titled
with the mean intensity of that channel inside the burst mask (128 px decoded image, [0, 1]), so you can read off how
the burst brightens along the walk. for area/aspect, the steps get their offset and the input gets the cell's measured value

saved as disentanglement_guided/vae_traversal_viz/<run>/<ckpt>_<set>_idx<CELL_IDX>/z{dim}/{chA,chB}.png
(target code folders are tagged with their factor, eg "z1_area"), plus info.txt with which frame it is and its factor values

run from src/:
    python disentanglement_guided/vae_viz.py
'''
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import make_grid, save_image

#run from src/, so the VAE_disent and disentanglement_semantic_eval packages resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VAE_disent.data_utils import twoChannelDataset
from VAE_disent.model import twoChannelVAE
from disentanglement_semantic_eval.gan_dci import get_burst_factors, load_burst_mask, BURST_FACTORS, ASPECT_MAX, ANNOT_DIR
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt
from PIL import Image
from disentanglement_guided.burst_detect import (PARAMETERS, PEAK_THRESHOLD_REL, PEAK_MIN_DISTANCE, load_spot_model, fit_image,
                                                 classify_bursts, getSpotLoc, classify_spots, dot_mask_from_mean_threshold, pick_radius)

import debugpy
# debugpy.listen(("127.0.0.1", 5678)) #127.0.0.1, cz only local host can talk to the port
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")

SET     = "train"      #"train" or "validation"
CELL_IDX = 70                 #index into the burst subset of SET (frames with a hand-drawn burst mask)
CHA_DIR  = f"data/singlecell_chA_split/{SET}"
CHB_DIR  = f"data/singlecell_chB_split/{SET}"
MASK_DIR = "data/singlecell_mask"

#must match the training config of the run
#factor -> latent code, like TARGETS in vae_guided_multi.py. for the single-factor vae_guided.py runs: {"intensity": 0}
RUN_DIR    = "vae_checkpoints_masked/guided_vae_beta=1_16_intensity-z0_area-z1_aspect-z2"
LATENT_DIM = 16
TARGETS    = {"intensity": 0, "area": 1, "aspect": 2}

#None -> vae_final.pt if the run finished, else the latest vae_epoch*.pt (changes while the run is still training).
#there is no validation-based "best" checkpoint saved. set a path to pin one, eg when comparing cells
CKPT = None

DEVICE  = torch.device("cuda:9")
N_STEPS = 7                  #odd, so there is a middle step (= the reconstruction)
SPAN    = 3.0                #walk each code from mu - SPAN to mu + SPAN, other codes fixed at the cell's posterior mean
BORDER  = 3                  #px of green border around the reconstruction
RING_RGB = (1.0, 0.0, 0.0)   #color of the burst mask outline on the traversal steps
TRACK_RGB = (0.0, 1.0, 1.0)  #color of the tracked burst outline (mask_track), cyan so it isn't mistaken for the hand-drawn one
BOUND_RGB = (1.0, 1.0, 0.0)  #color of the bounded tracked burst outline (mask_track_bounded), yellow
SAVED_RGB = (1.0, 0.0, 1.0)  #color of christina's saved dot mask (data/annotations/<video>/masked_img), magenta
SAVED_RADIUS = 5             #which of her saved radius variants (3, 4, 5) to show
BOUND_MARGIN = 15            #px around the hand-drawn burst that mask_track_bounded searches in (loose on purpose)
N_REF    = 3                 #reference columns before the traversal: input | burst mask | overlay
RECON_ONLY_RING = {"area", "aspect"}   #factor codes whose outline goes on the recon only, the rest get it on every step
assert N_STEPS % 2 == 1, "N_STEPS has to be odd for the middle step to be the reconstruction"

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vae_traversal_viz")


def pick_ckpt(run_dir):
    final = Path(run_dir) / "vae_final.pt"
    if final.exists():
        return final
    epochs = sorted(Path(run_dir).glob("vae_epoch*.pt"), key=lambda p: int(p.stem[len("vae_epoch"):]))
    assert epochs, f"no checkpoints in {run_dir}"
    return epochs[-1]


def ref_columns(x_ch, burst, rgb=RING_RGB):
    #x_ch: one channel [1, H, W] in [0, 1], burst: [1, H, W] binary, rgb: overlay color (red = hand-drawn, cyan = detected)
    #-> input | burst mask | mask overlaid on the input (50% color blend), as rgb [3, 3, H, W]
    dimmed = x_ch * (1 - 0.5 * burst)
    overlay = dimmed.repeat(3, 1, 1) + 0.5 * burst * torch.tensor(rgb, device=x_ch.device).view(3, 1, 1)
    return torch.stack([x_ch.repeat(3, 1, 1), burst.repeat(3, 1, 1), overlay], dim=0)


def mask_ring(burst):
    #burst: [1, H, W] binary -> [1, H, W] binary outline, the 1 px ring just outside the mask
    #(dilate by a 3x3 max pool, minus the mask), so drawing it never covers a burst pixel
    dilated = F.max_pool2d(burst.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]
    return dilated - burst


def load_saved_mask(chB_path, mask_transform, radius=SAVED_RADIUS):
    #christina's saved output for this frame: data/annotations/<video>/masked_img/<cell>_masked_radius<r>/..._f<frame>.png,
    #the raw image inside her dot mask (0 outside), same size as the crop. -> [1, 128, 128] binary mask, None if there's none.
    #same filename -> annotation lookup as burst_mask_path in gan_dci.py
    base = chB_path.name[:-len("_chB.png")]
    video, cell_frame = base.rsplit("_cell", 1)
    cell, frame = cell_frame.split("_f")
    path = Path(ANNOT_DIR) / video.replace(" ", "-") / "masked_img" / f"{cell}_masked_radius{radius}" / f"{cell}_masked_radius{radius}_f{frame}.png"
    if not path.exists():
        return None
    return (mask_transform(Image.open(path).convert('L')) > 0).float()


def save_annotated_rows(rows, titles, recon_col, suptitle, path):
    #rows: list of [n_cols, 3, H, W] rgb in [0, 1], titles: one list per row. one panel per column with its title
    #(matplotlib, so the values are readable)
    #a row can be shorter than the first one (eg only the reference columns), its missing panels are left empty
    n_rows, n_cols = len(rows), rows[0].shape[0]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 2.3 * n_rows), squeeze=False)
    for r in range(n_rows):
        for k, ax in enumerate(axes[r]):
            ax.axis("off")
            if k >= rows[r].shape[0]:
                continue
            ax.imshow(rows[r][k].permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
            ax.set_title(titles[r][k], fontsize=8, color="green" if k == recon_col else "black")
    fig.suptitle(suptitle, fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
'''
below will be the function to "track" the gene burst, adjusted for my pixel usecase
'''

def mask_track(frames, cell_mask, spot_model, radii=(3, 4, 5), search_mask=None):
    #frames: [N, H, W] one channel in [0, 1], the traversal steps play the role of her movie frames. cell_mask: [H, W] binary
    #spot_model: load_spot_model() (spotiflow), loaded once outside since it's the same for every call
    #search_mask: None = look in the whole cell (her method), or [H, W] binary region the burst has to be in (mask_track_bounded)
    #-> dict: the burst mask per frame [N, H, W] at the picked radius, the spot per frame [N, 2] (row, col), found per frame [N]
    #(plus the gaussian-only spot/found, before spotiflow)
    final_img = frames * cell_mask[None]                              #like raw_img * mask_img in her script
    frame_num = len(final_img)
    mask_stack = np.repeat(cell_mask[None], frame_num, axis=0)

    #run peak_local_max() to find bright spots, set a threshold(!tune)
    #here, minimum distance bw bright spots also needs to be tuned.. (in the tiff files, it was 20 (pixels?))

    #check that its a blob- fit_image() function from christina's files
    #15x15 box was the deafault, tune according to my data
    parameter_fit = [[] for _ in range(frame_num)]
    r_square = [[] for _ in range(frame_num)]
    for t in range(frame_num):
        #bounded: peaks only inside the search region (threshold_rel is then relative to the brightest pixel there),
        #the gaussian is still fit on the full image, so its box isn't cut off at the region's edge
        peak_img = final_img[t] if search_mask is None else final_img[t] * search_mask
        peaks = peak_local_max(peak_img, exclude_border=False, threshold_rel=PEAK_THRESHOLD_REL, min_distance=PEAK_MIN_DISTANCE)
        for peak in peaks:
            try:
                popt, r2, _, _ = fit_image(final_img[t], peak, PARAMETERS['fit_size'])
                if r2 >= PARAMETERS['min_r_square']:
                    parameter_fit[t].append(popt)
                    r_square[t].append(r2)
            except Exception:
                continue   #fits that can't run (eg a flat box near the border) are skipped, as in her script

    #classify_bursts(), should pass a lot of threhsolds (tune for my case?)
    bkg_int = np.zeros(frame_num)                                      #her script passes zeros here too
    _, gauss_found, gauss_spot, _ = classify_bursts(parameter_fit, r_square, bkg_int, PARAMETERS)

    #spotiflow: its spot per frame is the final position, the gaussian spot above only picks between several spotiflow spots
    spot_list = getSpotLoc(spot_model, frame_num, frames, mask_stack, PARAMETERS['circle_radius'], gauss_spot, gauss_found,
                           search_mask=search_mask)
    found, spot_position = classify_spots(spot_list)

    #build the mask
    #flood fill around the spotiflow spot for each radius, keep the one picked by her r² rule (circle_radius if none gets a fit)
    masks, threshes, integrated = {}, {}, {}
    for r in radii:
        dotmasks, integrated[r], _, threshes[r], _ = dot_mask_from_mean_threshold(final_img, r, frame_num, found, spot_position)
        masks[r] = np.stack(dotmasks).astype(bool)
    best_radius = pick_radius(threshes, integrated, gauss_found, radii)
    radius = best_radius if best_radius != 0 else PARAMETERS['circle_radius']

    return {"masks": masks[radius], "radius": radius, "spot_position": spot_position, "found": found,
            "gauss_spot_position": gauss_spot, "gauss_found": gauss_found}


def bound_region(burst_mask, cell_mask, margin=BOUND_MARGIN):
    #burst_mask, cell_mask: [H, W] binary -> [H, W] bool, every cell pixel within margin px of the hand-drawn burst
    return (distance_transform_edt(~burst_mask.astype(bool)) <= margin) & cell_mask.astype(bool)


def mask_track_bounded(frames, cell_mask, burst_mask, spot_model, margin=BOUND_MARGIN, radii=(3, 4, 5)):
    #mask_track, but the burst can only be found near the hand-drawn one (within margin px, see bound_region),
    #so it can't jump to a bright speck elsewhere in the nucleus. same output, plus the search region
    search_mask = bound_region(burst_mask, cell_mask, margin)
    out = mask_track(frames, cell_mask, spot_model, radii=radii, search_mask=search_mask)
    out["search_mask"] = search_mask
    return out


def detect_each(images, detect):
    #images: [N, H, W]. runs detect on every image on its own, repeated min_signal_duration times as a still "movie"
    #(her rules want a burst to last that many frames) -> [N, H, W] masks
    masks = []
    for img in images:
        still = np.repeat(img[None], PARAMETERS['min_signal_duration'], axis=0)
        masks.append(detect(still)["masks"][0])
    return np.stack(masks)


if __name__ == "__main__":
    ckpt = Path(CKPT) if CKPT is not None else pick_ckpt(RUN_DIR)
    model = twoChannelVAE(latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    #burst subset, like the training script. the train scan takes a while, so reuse the run's label cache there
    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR)
    #only trust a multi-factor cache (vae_guided_multi.py) computed on chA (the gene burst channel) with the same aspect clip,
    #single-factor caches have no area/aspect and older ones may be chB, so those get recomputed
    label_cache = Path(RUN_DIR) / "burst_labels.npz"
    cached = np.load(label_cache) if SET == "train" and label_cache.exists() else None
    if (cached is not None and "channel" in cached and str(cached["channel"]) == "chA"
            and "factor_names" in cached and list(cached["factor_names"]) == BURST_FACTORS
            and float(cached["aspect_max"]) == ASPECT_MAX):
        factors, burst_idx = cached["factors"], cached["burst_idx"]
    else:
        _, factors, burst_idx = get_burst_factors(dataset)

    i = burst_idx[CELL_IDX]                                  #CELL_IDX in the subset -> index in the full dataset
    x = dataset[i][0].to(DEVICE)                             #[2, 128, 128], cell-masked like the training input
    burst = load_burst_mask(dataset.chB_files[i], dataset.mask_transform).to(DEVICE)
    cell_mask = dataset[i][1][0].numpy()                     #[H, W] cell mask, for mask_track
    spot_model = load_spot_model()                           #spotiflow, loaded once for every code's traversal

    out_dir = f"{OUT_ROOT}/{Path(RUN_DIR).name}/{ckpt.stem}_{SET}_idx{CELL_IDX}"
    x_vis = (x * 0.5 + 0.5).clamp(0, 1)                      #un-normalize from [-1,1] back to [0,1] for viewing
    ring = mask_ring(burst)                                  #[1, H, W]
    ring_rgb = torch.tensor(RING_RGB, device=DEVICE).view(3, 1, 1)
    track_rgb = torch.tensor(TRACK_RGB, device=DEVICE).view(3, 1, 1)
    bound_rgb = torch.tensor(BOUND_RGB, device=DEVICE).view(3, 1, 1)
    in_burst = burst[0].bool()                               #[H, W], for the mean intensity inside the mask

    #the detection on the original input too, to check it against the hand-drawn mask. rules want a burst to last
    #min_signal_duration frames, so the input is repeated that many times as a still "movie" (every copy gets the same mask)
    
    '''
    #NOTE: this is for the rule based detection on an original image,
    to mimic a movie, the same frame is repeated many times.
    '''
    still = np.repeat(x_vis[0].cpu().numpy()[None], PARAMETERS['min_signal_duration'], axis=0)   #chA, the gene burst channel
    det = mask_track(still, cell_mask, spot_model)
    det_mask = torch.from_numpy(det["masks"][0]).float().unsqueeze(0).to(DEVICE)                 #[1, H, W]
    in_det = det_mask[0].bool()
    det_iou = (in_det & in_burst).sum().item() / (in_det | in_burst).sum().item()
    det_title = f"detected mask\nIoU {det_iou:.2f}" if in_det.any() else "detected mask\nno burst"

    #row 3 (bounded): the same detection on the input, but only near the hand-drawn burst (mask_track_bounded)
    burst_np = burst[0].cpu().numpy().astype(bool)
    track_bounded = lambda f: mask_track_bounded(f, cell_mask, burst_np, spot_model)
    bdet_mask = torch.from_numpy(detect_each(x_vis[0:1].cpu().numpy(), track_bounded)[0]).float().unsqueeze(0).to(DEVICE)
    in_bdet = bdet_mask[0].bool()
    bdet_iou = (in_bdet & in_burst).sum().item() / (in_bdet | in_burst).sum().item()
    bdet_title = f"bounded mask\nIoU {bdet_iou:.2f}" if in_bdet.any() else "bounded mask\nno burst"
    search = torch.from_numpy(bound_region(burst_np, cell_mask)).float().to(DEVICE)                  #[H, W]

    #reference columns per row: hand-drawn mask in red (row 1, static), detected mask in cyan (row 2, tracked),
    #bounded detected mask in yellow (row 3, its mask panel also shows the search region in grey)
    refs_static = {ch: ref_columns(x_vis[ch:ch+1], burst, RING_RGB) for ch in (0, 1)}
    refs_track = {ch: ref_columns(x_vis[ch:ch+1], det_mask, TRACK_RGB) for ch in (0, 1)}
    refs_bound = {ch: ref_columns(x_vis[ch:ch+1], bdet_mask, BOUND_RGB) for ch in (0, 1)}
    for ch in (0, 1):
        refs_bound[ch][1] = (0.3 * search + 0.7 * bdet_mask[0]).repeat(3, 1, 1)

    #row 4 (saved): christina's own saved dot mask for this frame (her pipeline on the raw movie), reference columns only,
    #as there's nothing to run on the perturbed images. an empty mask panel if this frame has none saved
    saved = load_saved_mask(dataset.chB_files[i], dataset.mask_transform)
    saved_mask = saved.to(DEVICE) if saved is not None else torch.zeros_like(burst)                 #[1, H, W]
    in_saved = saved_mask[0].bool()
    saved_iou = (in_saved & in_burst).sum().item() / (in_saved | in_burst).sum().item()
    saved_title = (f"saved r={SAVED_RADIUS} mask\nIoU {saved_iou:.2f}" if in_saved.any()
                   else f"saved r={SAVED_RADIUS} mask\n" + ("empty" if saved is not None else "not found"))
    refs_saved = {ch: ref_columns(x_vis[ch:ch+1], saved_mask, SAVED_RGB) for ch in (0, 1)}
    intensity_dim = TARGETS.get("intensity")                 #None if this run has no intensity code
    fmt = lambda v: "-" if v is None else f"{v:.3f}"         #"-" = no tracked burst, so no intensity

    #the cell's measured factors (from its hand-drawn mask), shown on the input panel of the area/aspect plots
    intensity, log_area, log_aspect = factors[CELL_IDX]    #BURST_FACTORS order
    measured = {"area": f"area {np.exp(log_area):.0f} px", "aspect": f"aspect {np.exp(log_aspect):.2f}"}

    with torch.no_grad():
        mu, _ = model.encode(x.unsqueeze(0))                 #[1, LATENT_DIM]
        alphas = torch.linspace(-SPAN, SPAN, N_STEPS, device=DEVICE)

        dim_tag = {d: f for f, d in TARGETS.items()}
        for dim in range(LATENT_DIM):
            z = mu.repeat(N_STEPS, 1)                         #[N_STEPS, LATENT_DIM]
            z[:, dim] = mu[0, dim] + alphas                   #alpha = 0 in the middle -> z = mu, the reconstruction
            imgs = (model.decode(z) * 0.5 + 0.5).clamp(0, 1)   #[N_STEPS, 2, H, W]

            dim_dir = f"{out_dir}/z{dim}" + (f"_{dim_tag[dim]}" if dim in dim_tag else "")
            os.makedirs(dim_dir, exist_ok=True)
            #row 1 (static): which traversal steps get the hand-drawn burst mask outline: only the recon for the shape codes,
            #every step otherwise
            ring_steps = torch.ones(N_STEPS, device=DEVICE)
            if dim_tag.get(dim) in RECON_ONLY_RING:
                ring_steps = torch.zeros(N_STEPS, device=DEVICE)
                ring_steps[N_STEPS // 2] = 1
            static_ring = ring * ring_steps.view(N_STEPS, 1, 1, 1)                #[N_STEPS, 1, H, W]

            #row 2 (tracked): every step gets the outline of its own burst mask, from mask_track on the decoded chA
            #(the gene burst channel). a step where no burst is found gets no outline
            # track = mask_track(imgs[:, 0].cpu().numpy(), cell_mask, spot_model)   #(the 7 steps as one 7-frame movie)
            #each step is detected on its own instead: repeated min_signal_duration times as a still "movie", like the original
            #input, so a step's detection doesn't depend on its neighbouring steps (and the recon gets the same mask in every walk)
            step_masks = []
            for s in range(N_STEPS):
                still_s = np.repeat(imgs[s, 0].cpu().numpy()[None], PARAMETERS['min_signal_duration'], axis=0)   #[3, H, W]
                step_masks.append(mask_track(still_s, cell_mask, spot_model)["masks"][0])
            track = {"masks": np.stack(step_masks)}                                         #[N_STEPS, H, W]
            tracked = torch.from_numpy(track["masks"]).float().unsqueeze(1).to(DEVICE)       #[N_STEPS, 1, H, W]
            track_ring = F.max_pool2d(tracked, kernel_size=3, stride=1, padding=1) - tracked  #1 px just outside, like mask_ring
            in_track = tracked[:, 0].bool()                                                   #[N_STEPS, H, W]

            #row 3 (bounded): like row 2, every step on its own, but the burst can only be found near the hand-drawn one
            bounded = torch.from_numpy(detect_each(imgs[:, 0].cpu().numpy(), track_bounded)).float().unsqueeze(1).to(DEVICE)
            bound_ring = F.max_pool2d(bounded, kernel_size=3, stride=1, padding=1) - bounded   #[N_STEPS, 1, H, W]
            in_bound = bounded[:, 0].bool()                                                    #[N_STEPS, H, W]

            for ch, name in [(0, "chA"), (1, "chB")]:
                trav = imgs[:, ch:ch+1].repeat(1, 3, 1, 1)                       #[N_STEPS, 3, H, W]
                trav_static = trav * (1 - static_ring) + ring_rgb * static_ring
                trav_track = trav * (1 - track_ring) + track_rgb * track_ring
                trav_bound = trav * (1 - bound_ring) + bound_rgb * bound_ring
                row_static = torch.cat([refs_static[ch], trav_static], dim=0)   #[N_REF + N_STEPS, 3, H, W]
                row_track = torch.cat([refs_track[ch], trav_track], dim=0)
                row_bound = torch.cat([refs_bound[ch], trav_bound], dim=0)

                #green border around the reconstruction (middle traversal step)
                green = torch.tensor([0.0, 1.0, 0.0], device=DEVICE).view(3, 1, 1)
                for row in (row_static, row_track, row_bound):
                    recon = row[N_REF + N_STEPS // 2]
                    recon[:, :BORDER, :] = green
                    recon[:, -BORDER:, :] = green
                    recon[:, :, :BORDER] = green
                    recon[:, :, -BORDER:] = green

                #free codes: plain grid, no titles (row 1 static, row 2 tracked, row 3 bounded)
                if dim not in dim_tag:
                    row_saved = torch.cat([refs_saved[ch], torch.zeros_like(trav)])   #padded with black, make_grid needs full rows
                    save_image(make_grid(torch.cat([row_static, row_track, row_bound, row_saved]), nrow=N_REF + N_STEPS),
                               f"{dim_dir}/{name}.png")
                    continue

                #row 1 titles, as before
                if dim == intensity_dim:
                    #intensity code: title every panel with the mean of this channel inside the burst mask.
                    #taken from imgs (before the outline/border are drawn), the input one from the 128 px input, so all are comparable
                    #(the info.txt label is on the raw crop, so it differs slightly)
                    in_mean = x_vis[ch][in_burst].mean().item()
                    step_means = imgs[:, ch][:, in_burst].mean(1).tolist()       #[N_STEPS]
                    titles_static = [f"input\n{in_mean:.3f}", "burst mask", "overlay"]
                    titles_static += [f"recon\n{m:.3f}" if s == N_STEPS // 2 else f"{a:+.1f}\n{m:.3f}"
                                      for s, (a, m) in enumerate(zip(alphas.tolist(), step_means))]
                else:
                    #shape codes (area/aspect): the decoded bursts have no mask to measure, so the steps only get their offset,
                    #the input panel gets the cell's measured value to compare the walk against
                    titles_static = [f"input\n{measured.get(dim_tag[dim], '')}", "burst mask", "overlay"]
                    titles_static += ["recon" if s == N_STEPS // 2 else f"{a:+.1f}" for s, a in enumerate(alphas.tolist())]

                #row 2 titles: mean of this channel inside the tracked mask of each step ("-" where nothing was detected),
                #the input one inside the mask detected on the input
                in_mean_track = x_vis[ch][in_det].mean().item() if in_det.any() else None
                step_means_track = [imgs[s, ch][in_track[s]].mean().item() if in_track[s].any() else None for s in range(N_STEPS)]
                titles_track = [f"input\n{fmt(in_mean_track)}", det_title, "overlay"]
                titles_track += [f"recon\n{fmt(m)}" if s == N_STEPS // 2 else f"{a:+.1f}\n{fmt(m)}"
                                 for s, (a, m) in enumerate(zip(alphas.tolist(), step_means_track))]

                #row 3 titles: same as row 2, inside the bounded mask
                in_mean_bound = x_vis[ch][in_bdet].mean().item() if in_bdet.any() else None
                step_means_bound = [imgs[s, ch][in_bound[s]].mean().item() if in_bound[s].any() else None for s in range(N_STEPS)]
                titles_bound = [f"input\n{fmt(in_mean_bound)}", bdet_title, "overlay"]
                titles_bound += [f"recon\n{fmt(m)}" if s == N_STEPS // 2 else f"{a:+.1f}\n{fmt(m)}"
                                 for s, (a, m) in enumerate(zip(alphas.tolist(), step_means_bound))]

                suptitle = (f"z{dim} ({dim_tag[dim]}), {name}   (steps = offset from the cell's own z{dim}, - = no burst detected)\n"
                            f"top: hand-drawn burst mask (red)" + (f", mean {name} inside it" if dim == intensity_dim else "")
                            + f"   |   middle: tracked burst mask (cyan), mean {name} inside it"
                            + f"   |   3rd: tracked within {BOUND_MARGIN} px of the hand-drawn burst (yellow, grey = search region)\n"
                            + f"4th: from data/annotations saved radius-{SAVED_RADIUS} dot mask for this frame (magenta, data/annotations/.../masked_img)")
                in_mean_saved = x_vis[ch][in_saved].mean().item() if in_saved.any() else None
                titles_saved = [f"input\n{fmt(in_mean_saved)}", saved_title, "overlay"]
                save_annotated_rows([row_static, row_track, row_bound, refs_saved[ch]],
                                    [titles_static, titles_track, titles_bound, titles_saved],
                                    N_REF + N_STEPS // 2, suptitle, f"{dim_dir}/{name}.png")

    info = [f"checkpoint: {ckpt}",
            f"set: {SET}, burst subset index {CELL_IDX} of {len(burst_idx)} (full dataset index {i})",
            f"chA: {dataset.chA_files[i].name}",
            f"chB: {dataset.chB_files[i].name}",
            f"burst intensity (mean chA under the mask, raw crop): {intensity:.4f}",
            f"burst area (px at 128 res): {np.exp(log_area):.1f}",
            f"burst aspect ratio (major/minor, clipped to [1, {ASPECT_MAX}]): {np.exp(log_aspect):.2f}",
            f"burst detected on the input (mask_track): " + (f"{int(in_det.sum())} px, IoU with the hand-drawn mask {det_iou:.2f}"
                                                             if in_det.any() else "none"),
            f"burst detected on the input, within {BOUND_MARGIN} px of the hand-drawn one (mask_track_bounded): "
            + (f"{int(in_bdet.sum())} px, IoU with the hand-drawn mask {bdet_iou:.2f}" if in_bdet.any() else "none"),
            f"christina's saved radius-{SAVED_RADIUS} dot mask (masked_img): "
            + (f"{int(in_saved.sum())} px, IoU with the hand-drawn mask {saved_iou:.2f}" if in_saved.any()
               else ("empty" if saved is not None else "not found"))]
    with open(f"{out_dir}/info.txt", "w") as f:
        f.write("\n".join(info) + "\n")
    print("\n".join(info))
    print(f"saved to {out_dir}")
