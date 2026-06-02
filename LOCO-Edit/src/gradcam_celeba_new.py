import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image
import debugpy

sys.path.append('.')
from utils.define_argparser import parse_args, preset
from modules.edit import EditUncondDiffusion

if os.getenv("DEBUGPY", "0") == "1":
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach on 5678...")
    debugpy.wait_for_client()


# how many denoising steps to skip between saved maps.
# for_steps is asserted to 100 for this model, so the tail is <=100 steps.
# set to 1 to keep every timestep, 5 to keep every 5th, etc.
TIMESTEP_STRIDE = 1


def gradcam_over_trajectory(edit, xt, vk, lam, act_layer, hspace_layer, stride=1):
    feat_cache, grad_cache, h_cache = {}, {}, {}

    def act_fwd_hook(m, i, o): #runs during forward pass
        feat_cache['x'] = o[0] if isinstance(o, tuple) else o #stores activation (last encoder layer, before h-space)

    def act_bwd_hook(m, gi, go): #runs during backprop..
        grad_cache['x'] = go[0] #stores the gradient coming back..

    def h_fwd_hook(m, i, o): #grabs the bottleneck = h-space output
        h_cache['x'] = o[0] if isinstance(o, tuple) else o

    h1 = act_layer.register_forward_hook(act_fwd_hook)
    h2 = act_layer.register_full_backward_hook(act_bwd_hook)
    h3 = hspace_layer.register_forward_hook(h_fwd_hook)

    # tiny helper: unet may return a tensor OR a ModelOutput with .sample
    def _unet_et(x, t):
        out = edit.unet(x, t)
        return out if isinstance(out, torch.Tensor) else out.sample

    results = []
    try:
        timesteps = edit.scheduler.timesteps
        start = int(edit.edit_t_idx) #WHERE the edit happens (set via edit_t in the bash file)

        xt_base = xt.detach().clone()
        # inject vk ONCE at the edit timestep. from here on we just denoise both trajectories.
        xt_edit = xt_base + lam * vk.detach().view_as(xt_base)

        # walk the tail: edit_t -> x_0. at each step we collect h-space and do one gradCAM.
        for i in range(start, len(timesteps)):
            t = timesteps[i].to(edit.device)

            # ---- base h-space: no grad, it's just a frozen target ----
            with torch.no_grad():
                et_base = _unet_et(xt_base, t)
            h_base = h_cache['x'].detach()

            # ---- edited h-space: needs grad so gradCAM can flow back ----
            edit.unet.zero_grad(set_to_none=True)
            with torch.enable_grad():
                et_edit = _unet_et(xt_edit, t) #this fires both hooks (h-space + activation layer)
                h_edit = h_cache['x']
                feat = feat_cache['x'] #activation of the layer just before h-space

                # delta h = h(edited) - h(base). NOTE: this is a FEATURE MAP, not a scalar!
                delta_h = h_edit - h_base
                '''
                #NOTE!: pow(2).sum() collapses delta_h into ONE number.
                # THAT is the value we backpropogate (similar to logit in original gradCAM)
                '''
                scalar = delta_h.pow(2).sum()
                scalar.backward() #this calls the bwd hook!!!

            score_val = float(scalar.detach().cpu())

            feat = feat.detach() #has outputs of forward activations
            grad = grad_cache['x'].detach() #has gradients of d(scalar)/d(activation map)
            weights = grad.mean(dim=(2, 3), keepdim=True) #these are the new weights (dims 2/3 are i,j!, as seen in formula)
            cam = F.relu((weights * feat).sum(dim=1, keepdim=True)) #CAM formula
            cam = F.interpolate(cam.float(), size=(256, 256), mode='bilinear', align_corners=False) #upsample
            cam = cam[0, 0].cpu().numpy() #2d plot now
            if cam.max() > 1e-8:
                cam = cam / cam.max()

            if (i - start) % stride == 0:
                results.append((i, int(t.item()), cam, score_val))

            # ---- advance BOTH trajectories one DDIM step (plain denoising, edit NOT re-applied) ----
            with torch.no_grad():
                xt_base = edit.scheduler.step(
                    et_base, t, xt_base, eta=0, use_clipped_model_output=None, generator=None
                ).prev_sample.detach()
                xt_edit = edit.scheduler.step(
                    et_edit.detach(), t, xt_edit, eta=0, use_clipped_model_output=None, generator=None
                ).prev_sample.detach()

        return results

    finally:
        h1.remove()
        h2.remove()
        h3.remove()


def save_overlay(orig_rgb, cam, path, alpha=0.5):
    colored = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    overlay = (alpha * colored + (1 - alpha) * orig_rgb).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(path)


if __name__ == '__main__':
    args = parse_args()
    args = preset(args)

    edit = EditUncondDiffusion(args) #loads diff model (initialization)
    edit.scheduler.set_timesteps(edit.for_steps, device=edit.device)

    #load the exact xt saved by main.py (no renoising or repeating)
    xt_path = os.path.join(edit.result_folder, f'xt-edit_{edit.edit_t}T.pt')
    assert os.path.exists(xt_path), f'missing {xt_path} — run main.py first'
    xt = torch.load(xt_path, map_location=edit.device).type(edit.dtype)
    xt = xt.to(edit.device)

    # load basis saved by main.py
    basis_dir = os.path.join(
        edit.result_folder, 'basis',
        f'local_basis-{edit.edit_t}T-select-mask-{args.mask_index}',
    )
    vT_path = os.path.join(basis_dir, f'vT-modify-pca-rank-{args.pca_rank}.pt')

    # cuz we need to have the SVD vectors output first!!!!
    assert os.path.exists(vT_path), f'missing {vT_path} — run main.py first'
    vT_modify = torch.load(vT_path, map_location=edit.device).type(edit.dtype)
    print(f'loaded vT_modify: {vT_modify.shape}')

    # load the saved original face for overlay
    orig_path = os.path.join(edit.result_folder, 'original.png')
    assert os.path.exists(orig_path), 'original.png not found — run main.py first'
    orig_rgb = np.array(Image.open(orig_path).convert('RGB').resize((256, 256)))

    out_dir = os.path.join(edit.result_folder, 'gradcam_hspace')
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(orig_rgb).save(os.path.join(out_dir, 'original.png'))

    # h-space = the bottleneck (mid_block). the layer JUST BEFORE it = last encoder block.
    # (same idea as the VAE paper using the last layer of the encoder)
    hspace_layer = edit.unet.mid_block
    act_layer = edit.unet.down_blocks[-1]

    lam = edit.x_space_guidance_scale * edit.x_space_guidance_edit_step

    for k in range(min(args.pca_rank, vT_modify.shape[0])):
        vk = vT_modify[k] / (vT_modify[k].norm() + 1e-8) #kth direction vector..

        # one folder per direction, then one map per timestep inside it
        dir_out = os.path.join(out_dir, f'dir{k:02d}')
        os.makedirs(dir_out, exist_ok=True)

        results = gradcam_over_trajectory(edit, xt, vk, lam, act_layer, hspace_layer, stride=TIMESTEP_STRIDE)
        print(f'direction {k}: collected {len(results)} timestep maps')

        for (i, t_val, cam, score) in results:
            print(f'  dir{k:02d} step{i:03d} (t={t_val}): score={score:.4f}, cam>0.5 area={(cam>0.5).sum()}')
            save_overlay(orig_rgb, cam, os.path.join(dir_out, f'step{i:03d}_t{t_val:04d}.png'))
            Image.fromarray((cam*255).astype(np.uint8)).save(
                os.path.join(dir_out, f'step{i:03d}_t{t_val:04d}_raw.png'))

    print(f'Done. Heatmaps in {out_dir}')