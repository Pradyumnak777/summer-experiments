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



def gradcam_for_direction(edit, xt, vk, lam, target_module):
    feat_cache, grad_cache = {}, {}

    def fwd_hook(m, i, o): #runs during forward pass
        feat_cache['x'] = o[0] if isinstance(o, tuple) else o #stores activation(after ReLU?)

    def bwd_hook(m, gi, go): #runs during backprop..
        grad_cache['x'] = go[0] #stores the gradient coming back..

    h1 = target_module.register_forward_hook(fwd_hook)
    h2 = target_module.register_full_backward_hook(bwd_hook)

    try:
        edit.unet.zero_grad(set_to_none=True)
        t_idx = edit.edit_t_idx
        t = edit.scheduler.timesteps[t_idx]
        t = t.to(edit.device)

        vk = vk.to(xt.device)

        xt_base = xt.detach().clone()
        xt_edit = xt_base + lam * vk.detach().view_as(xt_base)

        with torch.no_grad():
            x0_base = edit.get_x0(t, xt_base, mask=None)
        x0_base = x0_base.detach()

        with torch.enable_grad():
            x0_edited = edit.get_x0(t, xt_edit, mask=None)
            delta = x0_edited - x0_base
            scalar = delta[0].pow(2).sum()
            '''
            #NOTE!: below is the final values use to backpropogate 
            # (similar to logit in original gradCAM)
            '''
            scalar.backward(retain_graph=True) #this calls the bwd hook!!!

        score_val = float(scalar.detach().cpu())
        feat = feat_cache['x'].detach() #has outputs of forward activations
        grad = grad_cache['x'].detach() #has gradients of d(scalar)/d(activation map)
        weights = grad.mean(dim=(2, 3), keepdim=True) #these are the new weights (dims 2/3 are i,j!, as seen in formula)
        cam = F.relu((weights * feat).sum(dim=1, keepdim=True)) #CAM formula
        cam = F.interpolate(cam.float(), size=(256, 256), mode='bilinear', align_corners=False) #upsample
        cam = cam[0, 0].cpu().numpy() #2d plot now
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

    edit = EditUncondDiffusion(args) #loads diff model (initialization)
    edit.scheduler.set_timesteps(edit.for_steps, device=edit.device)

    # Random dataset: no inversion, just draw xT (xT is gaussian noise(?))
    xT = torch.randn(1, 3, edit.image_size, edit.image_size,
                     dtype=edit.dtype, device=edit.device)
    torch.manual_seed(args.seed)  # same seed as main run
    xT = torch.randn(1, 3, edit.image_size, edit.image_size,
                     dtype=edit.dtype, device=edit.device)

    '''
    #NOTE: here, edit_t_idx is what I set in the bash file argument for 
    WHERE the edit needs to happen.
    
    below line runs from noise(x_T) to the edit timestep defined in above line
    '''
    xt, t, t_idx = edit.DDIMforwardsteps(xT, t_start_idx=0, t_end_idx=edit.edit_t_idx)
    xt = xt.to(edit.device) #noisy image/latent at the edit timestep

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

    out_dir = os.path.join(edit.result_folder, 'gradcam')
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(orig_rgb).save(os.path.join(out_dir, 'original.png'))

    target_module = edit.unet.up_blocks[1]
    lam = edit.x_space_guidance_scale * edit.x_space_guidance_edit_step

    for k in range(min(args.pca_rank, vT_modify.shape[0])):
        vk = vT_modify[k] / (vT_modify[k].norm() + 1e-8) #kth direction vector..
        cam, score = gradcam_for_direction(edit, xt, vk, lam, target_module)
        print(f'direction {k}: score={score:.4f}, cam>0.5 area={(cam>0.5).sum()}')
        save_overlay(orig_rgb, cam, os.path.join(out_dir, f'gradcam_dir{k:02d}.png'))
        Image.fromarray((cam*255).astype(np.uint8)).save(
            os.path.join(out_dir, f'gradcam_dir{k:02d}_raw.png'))

    print(f'Done. Heatmaps in {out_dir}')