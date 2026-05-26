import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image

sys.path.append('.')
from utils.define_argparser import parse_args, preset
from modules.edit import EditStableDiffusion


def gradcam_for_direction(edit, zt, vk, lam, mask, target_module):
    feat_cache, grad_cache = {}, {}

    def fwd_hook(m, i, o):
        feat_cache['x'] = o[0] if isinstance(o, tuple) else o

    def bwd_hook(m, gi, go):
        grad_cache['x'] = go[0]

    h1 = target_module.register_forward_hook(fwd_hook)
    h2 = target_module.register_full_backward_hook(bwd_hook)

    try:
        # zero grad BEFORE forward, not after
        edit.unet.zero_grad(set_to_none=True)
        edit.vae.zero_grad(set_to_none=True)

        t_idx = edit.edit_t_idx
        t = edit.scheduler.timesteps[t_idx]

        zt_base = zt.detach().clone()
        zt_edit = zt_base + lam * vk.detach().view_as(zt_base)  # same operation as main.py

        # baseline x0 — no gradient needed, computed outside graph
        with torch.no_grad():
            x0_base = edit.get_x0(
                zt_base, t, t_idx,
                edit.for_prompt_emb, edit.edit_prompt_emb, edit.null_prompt_emb,
                mask=None, mode="null+(for-null)"
            )
        x0_base = x0_base.detach()

        with torch.enable_grad():
            x0_edited = edit.get_x0(
                zt_edit, t, t_idx,
                edit.for_prompt_emb, edit.edit_prompt_emb, edit.null_prompt_emb,
                mask=None, mode="null+(for-null)"
            )
            # NOTE:!! this is the "logit" equivalent- total squared change in the masked region
            # caused by applying vk, which is exactly what main.py produces as lambda varies
            delta = x0_edited - x0_base
            # scalar = delta[0][mask].pow(2).sum()
            scalar = delta[0].pow(2).sum() #on whole image...

            scalar.backward(retain_graph=True)    # inside enable_grad

        score_val = float(scalar.detach().cpu())

        feat = feat_cache['x'].detach()
        grad = grad_cache['x'].detach()

        weights = grad.mean(dim=(2, 3), keepdim=True) #weighted sum to get a final single matrix feature map
        cam = F.relu((weights * feat).sum(dim=1, keepdim=True)) #relu to remove negatives..
        cam = F.interpolate(cam.float(), size=(512, 512), mode='bilinear', align_corners=False) #interpolate to input img size..
        cam = cam[0, 0].cpu().numpy()
        if cam.max() > 1e-8:
            cam = cam / cam.max()

        return cam, score_val

    finally:
        h1.remove()
        h2.remove()

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
    vT_modify_path = os.path.join(basis_dir, 'vT-modify.pt')
    assert os.path.exists(vT_modify_path), f'missing {vT_modify_path} — run main.py first'
    vT_modify = torch.load(vT_modify_path, map_location=edit.device).type(edit.dtype)
    print(f'loaded vT_modify: {vT_modify.shape}')

    # mask used during SVD
    if edit.mask_path and os.path.exists(edit.mask_path):
        mask = torch.load(edit.mask_path, map_location=edit.device).bool()
    else:
        mask = torch.ones(3, 512, 512, dtype=torch.bool, device=edit.device)

    # mask = torch.load(edit.mask_path, map_location=edit.device).bool()

    # original image (preprocessed, same as model sees)
    orig_path = os.path.join(edit.result_folder, 'original.png')
    assert os.path.exists(orig_path), 'original.png not found — run main.py first'
    orig_rgb = np.array(Image.open(orig_path).convert('RGB').resize((512, 512)))

    out_dir = os.path.join(edit.result_folder, 'gradcam')
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(orig_rgb).save(os.path.join(out_dir, 'original.png'))

    # choose which U-Net submodule to hook
    # mid_block is the bottleneck (8x8 spatial in SD), cheapest. Try up_blocks[1] (16x16) or up_blocks[2] (32x32) for finer maps.
    # target_module = edit.unet.mid_block
    
    '''
    trying up block
    '''
    target_module = edit.unet.up_blocks[3].resnets[-1]   # last ResBlock in up_blocks[3]

    lam = edit.x_space_guidance_scale * edit.x_space_guidance_edit_step  # same step size as main.py

    num_directions = args.pca_rank   # edit this to visualize more
    for k in range(num_directions): #this is top k..
        vk = vT_modify[k] / (vT_modify[k].norm() + 1e-8) #normalizing
        cam, score = gradcam_for_direction(edit, zt, vk, lam, mask, target_module)
        print(f'direction {k}: score={score:.4f}, cam max-region size={(cam>0.5).sum()}')

        save_overlay(orig_rgb, cam, os.path.join(out_dir, f'gradcam_dir{k:02d}.png'))
        Image.fromarray((cam*255).astype(np.uint8)).save(
            os.path.join(out_dir, f'gradcam_dir{k:02d}_raw.png'))

    print(f'Done. Heatmaps in {out_dir}')