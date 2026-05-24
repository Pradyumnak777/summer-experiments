import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image

sys.path.append('.')
from utils.define_argparser import parse_args, preset
from modules.edit import EditStableDiffusion


def gradcam_for_direction(edit, zt, u_k, mask, target_module):
    """Run one GradCAM pass for SVD direction with vector u_k."""
    feat_cache, grad_cache = {}, {}

    def fwd_hook(m, i, o):
        feat_cache['x'] = o[0] if isinstance(o, tuple) else o
    def bwd_hook(m, gi, go):
        grad_cache['x'] = go[0]

    h1 = target_module.register_forward_hook(fwd_hook)
    h2 = target_module.register_full_backward_hook(bwd_hook)

    try:
        zt_grad = zt.detach().clone().requires_grad_(True)
        t_idx = edit.edit_t_idx
        t = edit.scheduler.timesteps[t_idx]

        with torch.enable_grad():
            x0_masked = edit.get_x0(
                zt_grad, t, t_idx,
                edit.for_prompt_emb, edit.edit_prompt_emb, edit.null_prompt_emb,
                mask=mask, mode="null+(for-null)"
            )   # [1, num_mask_pixels]
            u_k_dev = u_k.to(x0_masked.device, x0_masked.dtype)
            score = (x0_masked[0] * u_k_dev).sum()

        edit.unet.zero_grad(set_to_none=True)
        edit.vae.zero_grad(set_to_none=True)
        score.backward()

        feat = feat_cache['x']        # [1, C, Hf, Wf]
        grad = grad_cache['x']        # [1, C, Hf, Wf]

        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feat).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam.float(), size=(512, 512), mode='bilinear', align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        return cam, float(score.detach().cpu())
    finally:
        h1.remove(); h2.remove()


def save_overlay(orig_rgb, cam, path, alpha=0.5):
    colored = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    overlay = (alpha * colored + (1 - alpha) * orig_rgb).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(path)


if __name__ == '__main__':
    args = parse_args()
    args = preset(args)

    edit = EditStableDiffusion(args)

    # re-derive zt at edit_t (same path as main.py)
    edit.scheduler.set_timesteps(edit.for_steps)
    if edit.dataset_name == 'Random':
        zT = torch.randn(1, 4, 64, 64, dtype=edit.dtype, device=edit.device)
    else:
        zT = edit.run_DDIMinversion(idx=edit.sample_idx)

    zt, t, t_idx = edit.DDIMforwardsteps(
        zT, t_start_idx=0, t_end_idx=edit.edit_t_idx,
        for_prompt_emb=edit.for_prompt_emb,
        edit_prompt_emb=edit.edit_prompt_emb,
        null_prompt_emb=edit.null_prompt_emb,
        mode="null+(for-null)",
    )

    # load saved SVD basis (same folder structure as the main run)
    basis_dir = os.path.join(
        edit.result_folder, 'basis',
        f'local_basis-{edit.edit_t}T-pca-rank-{args.pca_rank}-select-mask{args.mask_index}',
    )
    u_modify_path = os.path.join(basis_dir, 'u-modify.pt')
    assert os.path.exists(u_modify_path), f'missing {u_modify_path} — run main.py first'
    u_modify = torch.load(u_modify_path, map_location=edit.device).type(edit.dtype)
    print(f'loaded u_modify: {u_modify.shape}')

    # mask used during SVD
    assert edit.mask_path and os.path.exists(edit.mask_path), 'mask_path not set'
    mask = torch.load(edit.mask_path, map_location=edit.device).bool()

    # original image (preprocessed, same as model sees)
    x0_orig = edit.dataset[edit.sample_idx]
    orig_rgb = ((x0_orig[0].permute(1,2,0).cpu().float().numpy() + 1) / 2 * 255).clip(0,255).astype(np.uint8)

    out_dir = os.path.join(edit.result_folder, 'gradcam')
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(orig_rgb).save(os.path.join(out_dir, 'original.png'))

    # choose which U-Net submodule to hook
    # mid_block is the bottleneck (8x8 spatial in SD), cheapest. Try up_blocks[1] (16x16) or up_blocks[2] (32x32) for finer maps.
    # target_module = edit.unet.mid_block
    
    '''
    trying up block
    '''
    target_module = edit.unet.up_blocks[2].resnets[-1]   # last ResBlock in up_blocks[2]

    num_directions = min(5, u_modify.size(0))   # edit this to visualize more
    for k in range(num_directions):
        u_k = u_modify[k]
        cam, score = gradcam_for_direction(edit, zt, u_k, mask, target_module)
        print(f'direction {k}: score={score:.4f}, cam max-region size={(cam>0.5).sum()}')

        save_overlay(orig_rgb, cam, os.path.join(out_dir, f'gradcam_dir{k:02d}.png'))
        Image.fromarray((cam*255).astype(np.uint8)).save(
            os.path.join(out_dir, f'gradcam_dir{k:02d}_raw.png'))

    print(f'Done. Heatmaps in {out_dir}')