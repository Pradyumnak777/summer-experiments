import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image
import debugpy

sys.path.append('.')
from utils.define_argparser import parse_args, preset
from modules.edit import EditStableDiffusion

if os.getenv("DEBUGPY", "0") == "1":
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach on 5678...")
    debugpy.wait_for_client()


# how many denoising steps to skip between saved maps.
# the tail (edit_t -> z0) is (for_steps - edit_t_idx) steps. 1 keeps every step, 5 every 5th.
TIMESTEP_STRIDE = 1

#CFG mode
#the h-space single forward ignores this and uses the conditional (for) prompt only!!(why?- because 2 grads flow back to the bottleneck: 
# h from conditional, and h from null/unconditional, which complicates stuff)
#this is short hand for-
'''
noise_final = noise_uncond + s * (noise_cond - noise_uncond)
'''
MODE = "null+(for-null)"


def gradcam_over_trajectory_pixel(edit, zt, vk, lam, act_layer, stride=1):
    feat_cache, grad_cache = {}, {}

    def act_fwd_hook(m, i, o): #runs during forward pass
        #m: modeule being hooked, i: input, o: output...
        #here module is the block like up_blocks, etc.
        feat_cache['x'] = o[0] if isinstance(o, tuple) else o #stores activation (a decoder/up layer now)

    def act_bwd_hook(m, gi, go): #runs during backprop..
        grad_cache['x'] = go[0] #stores the gradient coming back..

    h1 = act_layer.register_forward_hook(act_fwd_hook)
    h2 = act_layer.register_full_backward_hook(act_bwd_hook)

    '''
    noise_final = noise_uncond + s * (noise_cond - noise_uncond)
    '''
    do_cfg = edit.guidance_scale > 1.0

    def _cfg_et(z, t): #CFG noise prediction, used ONLY for stepping (no grad)
        return edit._classifer_free_guidance(
            z, t, edit.for_prompt_emb, edit.edit_prompt_emb, edit.null_prompt_emb,
            mode=MODE, do_classifier_free_guidance=do_cfg,
        )

    results = []
    try:
        timesteps = edit.scheduler.timesteps
        start = int(edit.edit_t_idx) #WHERE the edit happens (set via edit_t in the bash file)

        zt_base = zt.detach().clone()
        # inject vk ONCE at the edit timestep. from here on we just denoise both trajectories.
        zt_edit = zt_base + lam * vk.detach().view_as(zt_base)

        #edit_t -> z_0. at each step collect x0 and do one gradCAM.
        for i in range(start, len(timesteps)):
            t = timesteps[i].to(edit.device)

            #predicted clean image x0 of the NON-edited path. constant target, no grad.
            with torch.no_grad():
                x0_base = edit.get_x0(
                    zt_base, t, i,
                    edit.for_prompt_emb, edit.edit_prompt_emb, edit.null_prompt_emb,
                    mask=None, mode=MODE,
                )
            x0_base = x0_base.detach()

            #predicted clean image x0 of the EDITED path. needs grad so it flows back to up_blocks.
            edit.unet.zero_grad(set_to_none=True)
            edit.vae.zero_grad(set_to_none=True)
            with torch.enable_grad():
                zt_edit_g = zt_edit.detach().requires_grad_(True)   
                x0_edit = edit.get_x0( #full forward thru U-Net + VAE -> Tweedie x0 (pixels)
                    zt_edit_g, t, i,
                    edit.for_prompt_emb, edit.edit_prompt_emb, edit.null_prompt_emb,
                    mask=None, mode=MODE,
                )
                feat = feat_cache['x'] #activation of a decoder (up) layer

                # delta x0 = x0(edited) - x0(base). a pixel-space IMAGE diff, NOT a scalar!
                delta = x0_edit - x0_base
                '''
                #NOTE!: pow(2).sum() collapses delta into ONE number.
                # THAT is the value we backpropogate (the "logit"). same as the old pixel method.
                '''
                scalar = delta.pow(2).sum()
                scalar.backward() #this calls the bwd hook!!!

            score_val = float(scalar.detach().cpu())

            feat = feat.detach() #has outputs of forward activations
            grad = grad_cache['x'].detach() #has gradients of d(scalar)/d(activation map)
            weights = grad.mean(dim=(2, 3), keepdim=True) #GAP over spatial -> per-channel weight
            cam = F.relu((weights * feat).sum(dim=1, keepdim=True)) #CAM formula
            cam = F.interpolate(cam.float(), size=(512, 512), mode='bilinear', align_corners=False) #upsample
            cam = cam[0, 0].cpu().numpy() #2d plot now
            if cam.max() > 1e-8:
                cam = cam / cam.max()

            if (i - start) % stride == 0:
                results.append((i, int(t.item()), cam, score_val))

            #go 1 DDIM step forward, for both base and edit.. (CFG dynamics, no grad)
            with torch.no_grad():
                et_base = _cfg_et(zt_base, t)
                zt_base = edit.scheduler.step(et_base, t, zt_base, eta=0).prev_sample.detach()
                et_edit = _cfg_et(zt_edit, t)
                zt_edit = edit.scheduler.step(et_edit, t, zt_edit, eta=0).prev_sample.detach()

        return results

    finally:
        h1.remove()
        h2.remove()



def gradcam_over_trajectory(edit, zt, vk, lam, act_layer, hspace_layer, stride=1):
    feat_cache, grad_cache, h_cache = {}, {}, {}

    def act_fwd_hook(m, i, o): #runs during forward pass
        feat_cache['x'] = o[0] if isinstance(o, tuple) else o #stores activation (last encoder layer, before h-space)

    def act_bwd_hook(m, gi, go): #runs during backprop..
        grad_cache['x'] = go[0] #stores the gradient coming back..

    def h_fwd_hook(m, i, o): #grabs the bottleneck = h-space output
        h_cache['x'] = o[0] if isinstance(o, tuple) else o

    h1 = act_layer.register_forward_hook(act_fwd_hook)
    h2 = act_layer.register_full_backward_hook(act_bwd_hook)
    '''
    #NOTE- the bwlow hook is needed cuz the scalar is an intermediate value. In the pixel variant, it was a direct output of the NN
    but here, it is not and it needs to be grabbed..
    '''
    h3 = hspace_layer.register_forward_hook(h_fwd_hook)

    do_cfg = edit.guidance_scale > 1.0

    #h-space forward only for ocndiitonal run, not for uncond (as cfg is involved)
    #a CFG batched bottleneck that would mix conditional/unconditional h-spaces.)
    def _hspace_forward(z, t):
        emb = edit.for_prompt_emb.repeat(z.size(0), 1, 1) #only 'for_prompt'(which is cond.) is involved..
        return edit.unet(z, t, encoder_hidden_states=emb).sample

    def _cfg_et(z, t): #CFG noise prediction, used ONLY for stepping (no grad)
        return edit._classifer_free_guidance(
            z, t, edit.for_prompt_emb, edit.edit_prompt_emb, edit.null_prompt_emb,
            mode=MODE, do_classifier_free_guidance=do_cfg,
        )

    results = []
    try:
        timesteps = edit.scheduler.timesteps
        start = int(edit.edit_t_idx) #WHERE the edit happens (set via edit_t in the bash file)

        zt_base = zt.detach().clone()
        # inject vk ONCE at the edit timestep. from here on we just denoise both trajectories.
        '''
        #NOTE!!!- gradCAM visualized only over one increment of lambda!
        '''
        zt_edit = zt_base + lam * vk.detach().view_as(zt_base)

        #edit_t -> z_0. at each step collect h-space and do one gradCAM.
        for i in range(start, len(timesteps)):
            t = timesteps[i].to(edit.device)

            #below is h-space of non edited at timestep t. It is treated as constant.
            with torch.no_grad():
                _ = _hspace_forward(zt_base, t)
            h_base = h_cache['x'].detach() #THIS gets the h-space value for conditional run..(of original)

            #below is hspace of z_t + v_k, part of the NN output. this will flow back.
            edit.unet.zero_grad(set_to_none=True)
            with torch.enable_grad():
                zt_edit_g = zt_edit.detach().requires_grad_(True)
                _ = _hspace_forward(zt_edit_g, t) #this fires both hooks (h-space + activation layer)
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
            weights = grad.mean(dim=(2, 3), keepdim=True) #these are the new weights (dims 2/3 are i,j!)
            cam = F.relu((weights * feat).sum(dim=1, keepdim=True)) #CAM formula
            cam = F.interpolate(cam.float(), size=(512, 512), mode='bilinear', align_corners=False) #upsample
            cam = cam[0, 0].cpu().numpy() #2d plot now
            if cam.max() > 1e-8:
                cam = cam / cam.max()

            if (i - start) % stride == 0:
                results.append((i, int(t.item()), cam, score_val))

            #go 1 DDIM step forward, for both base and edit.. (CFG dynamics, no grad)
            with torch.no_grad():
                et_base = _cfg_et(zt_base, t)
                zt_base = edit.scheduler.step(et_base, t, zt_base, eta=0).prev_sample.detach()
                et_edit = _cfg_et(zt_edit, t)
                zt_edit = edit.scheduler.step(et_edit, t, zt_edit, eta=0).prev_sample.detach()

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

    edit = EditStableDiffusion(args) #loads diff model (initialization)
    edit.scheduler.set_timesteps(edit.for_steps, device=edit.device)
    
    #dont want gradients for below stuff..
    edit.for_prompt_emb  = edit.for_prompt_emb.detach()
    edit.edit_prompt_emb = edit.edit_prompt_emb.detach()
    edit.null_prompt_emb = edit.null_prompt_emb.detach()


    #load the exact zt saved by main.py (no renoising or repeating). its zt because its a latent DM
    zt_path = os.path.join(edit.result_folder, "xt.pt")
    assert os.path.exists(zt_path), f'missing {zt_path} — run main.py first'
    zt = torch.load(zt_path, map_location=edit.device).type(edit.dtype)
    zt = zt.to(edit.device)

    # load basis saved by main.py (SD folder naming differs from CelebA)
    basis_dir = os.path.join(
        edit.result_folder, 'basis',
        f'local_basis-{edit.edit_t}T-"{edit.edit_prompt}"-pca-rank-{args.pca_rank}-select-mask{args.mask_index}',
    )
    vT_path = os.path.join(basis_dir, 'vT-modify.pt')

    # cuz we need to have the SVD vectors output first!!!!
    assert os.path.exists(vT_path), f'missing {vT_path} — run main.py first'
    vT_modify = torch.load(vT_path, map_location=edit.device).type(edit.dtype)
    print(f'loaded vT_modify: {vT_modify.shape}')

    # load the saved original image for overlay (SD = 512)
    orig_path = os.path.join(edit.result_folder, 'original.png')
    assert os.path.exists(orig_path), 'original.png not found — run main.py first'
    orig_rgb = np.array(Image.open(orig_path).convert('RGB').resize((512, 512)))

    # h-space = the bottleneck (mid_block). the layer JUST BEFORE it = last encoder block.
    # (same idea as the VAE paper using the last layer of the encoder)
    
    '''
    uncomment below if using h-space vector derived scalar
    '''
    hspace_layer = edit.unet.mid_block
    act_layer = edit.unet.down_blocks[-1]
    # act_layer = edit.unet.down_blocks[0]
    out_dir = os.path.join(edit.result_folder, 'gradcam_hspace')

    '''
    uncomment below if performing pixel space level derived scalar
    (this is the MATCHED scalar for SD: the basis comes from the x0 pullback)
    '''
    # act_layer = edit.unet.up_blocks[1]
    # out_dir = os.path.join(edit.result_folder, 'gradcam_pixel')

    # ===================== NEW: layer-sweep setup =====================
    # pick which scalar variant the sweep runs with: 'hspace' or 'pixel'.
    # (your uncomment-toggle above is kept as reference; this flag drives execution
    #  and sets out_dir accordingly.)
    VARIANT = 'hspace'
    if VARIANT == 'pixel':
        out_dir = os.path.join(edit.result_folder, 'gradcam_pixel')

    # build the U-Net layers to sweep over (DF-CAM style: across the whole net).
    # the h-space scalar lives at mid_block, so only layers UPSTREAM of it
    # (encoder down_blocks + mid_block) get gradient; up_blocks are downstream of
    # the bottleneck -> zero gradient, so we add them ONLY for the pixel variant
    # (whose scalar is the decoded output, downstream of everything).
    sweep_layers = []
    for bi, blk in enumerate(edit.unet.down_blocks):
        sweep_layers.append((f'down_blocks_{bi}', blk))
    sweep_layers.append(('mid_block', edit.unet.mid_block))
    if VARIANT == 'pixel':
        for bi, blk in enumerate(edit.unet.up_blocks):
            sweep_layers.append((f'up_blocks_{bi}', blk))
    # ==================================================================

    '''
    resume normal execution..
    '''
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(orig_rgb).save(os.path.join(out_dir, 'original.png'))

    '''
    below is for single step edit, to the right!!
    '''
    lam = edit.x_space_guidance_scale * edit.x_space_guidance_edit_step
    
    '''
    below is for edit to the right, but matching the magnitude of 1st RHS of output
    '''
    # vis_stride = (edit.x_space_guidance_num_step + 1) // args.vis_num
    # lam = edit.x_space_guidance_scale * edit.x_space_guidance_edit_step * vis_stride

    # NEW: outer loop sweeps the hooked layer; each layer gets its own subfolder,
    # and the per-direction (dir{k}) folders live inside it, as before.
    for layer_name, act_layer in sweep_layers:
        layer_out = os.path.join(out_dir, layer_name)
        os.makedirs(layer_out, exist_ok=True)
        Image.fromarray(orig_rgb).save(os.path.join(layer_out, 'original.png'))
        print(f'===== sweeping act_layer = {layer_name} =====')

        for k in range(min(args.pca_rank, vT_modify.shape[0])):
            vk = vT_modify[k] / (vT_modify[k].norm() + 1e-8) #kth direction vector..(normalizing)

            # one folder per direction, then one map per timestep inside it
            dir_out = os.path.join(layer_out, f'dir{k:02d}')
            os.makedirs(dir_out, exist_ok=True)

            '''
            uncomment the one below accordingly (pixel or h-space)
            '''
            if VARIANT == 'hspace':
                results = gradcam_over_trajectory(edit, zt, vk, lam, act_layer, hspace_layer, stride=TIMESTEP_STRIDE)
            else:
                results = gradcam_over_trajectory_pixel(edit, zt, vk, lam, act_layer, stride=TIMESTEP_STRIDE)

            print(f'[{layer_name}] direction {k}: collected {len(results)} timestep maps')

            for (i, t_val, cam, score) in results:
                print(f'  dir{k:02d} step{i:03d} (t={t_val}): score={score:.4f}, cam>0.5 area={(cam>0.5).sum()}')
                save_overlay(orig_rgb, cam, os.path.join(dir_out, f'step{i:03d}_t{t_val:04d}.png'))
                Image.fromarray((cam*255).astype(np.uint8)).save(
                    os.path.join(dir_out, f'step{i:03d}_t{t_val:04d}_raw.png'))

    print(f'Done. Heatmaps in {out_dir}')