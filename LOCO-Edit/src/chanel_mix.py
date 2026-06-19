# channel_coupling.py
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import debugpy

sys.path.append('.')
from utils.define_argparser import parse_args, preset
from modules.edit import EditStableDiffusion


if os.getenv("DEBUGPY", "0") == "1":
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach on 5678...")
    debugpy.wait_for_client()


# same CFG mode used everywhere else (stepping + x0 estimate)
MODE = "null+(for-null)"


@torch.no_grad()
def denoise_to_x0(edit, zt_start, do_cfg):
    #step from edit_t all the way to z0, return the FINAL decoded x0 (pixels [1,3,512,512]).
    #the last get_x0 (smallest t) is the fully-denoised image, which is what we compare.
    timesteps = edit.scheduler.timesteps
    start = int(edit.edit_t_idx)
    zt = zt_start.detach().clone()
    x0 = None
    for i in range(start, len(timesteps)):
        t = timesteps[i].to(edit.device)
        x0 = edit.get_x0(
            zt, t, i,
            edit.for_prompt_emb, edit.edit_prompt_emb, edit.null_prompt_emb,
            mask=None, mode=MODE,
        )
        et = edit._classifer_free_guidance(
            zt, t, edit.for_prompt_emb, edit.edit_prompt_emb, edit.null_prompt_emb,
            mode=MODE, do_classifier_free_guidance=do_cfg,
        )
        zt = edit.scheduler.step(et, t, zt, eta=0).prev_sample
    return x0


def split_channels(delta):
    #delta: [1,3,H,W] = x0_edit - x0_base (the RESPONSE to a direction).
    #green lives in G plane, magenta in R(=B) -> (R+B)/2.
    dG = delta[:, 1:2]  #this results in [1, 1, 512, 512]
    dM = 0.5 * (delta[:, 0:1] + delta[:, 2:3])  # this is [1, 1, 512, 512]. (as 2 channels are summed)
    return dG, dM


def delta_to_rgb(dG, dM):
    #visualize the change in the SAME green/magenta scheme: green pixels = green moved,
    #magenta pixels = magenta moved, WHITE = both moved (= coupled region).
    g = dG[0, 0].abs().float().cpu().numpy()
    m = dM[0, 0].abs().float().cpu().numpy()
    vmax = max(g.max(), m.max()) + 1e-8 #shared max so relative magnitude is preserved
    g = g / vmax
    m = m / vmax
    rgb = np.stack([m, g, m], axis=-1)          # magenta=R&B, green=G
    return (rgb * 255).astype(np.uint8)



def make_trio(dG, dM):
    # returns side-by-side: [green gray | magenta gray | delta_rgb]
    g = dG[0, 0].abs().float().cpu().numpy()
    m = dM[0, 0].abs().float().cpu().numpy()
    vmax = max(g.max(), m.max()) + 1e-8
    g_norm = (g / vmax * 255).astype(np.uint8)
    m_norm = (m / vmax * 255).astype(np.uint8)
    # plain grayscale panels (R=G=B), stacked as RGB
    green_panel   = np.stack([g_norm, g_norm, g_norm], axis=-1)
    magenta_panel = np.stack([m_norm, m_norm, m_norm], axis=-1)
    delta_panel   = delta_to_rgb(dG, dM)
    return np.concatenate([green_panel, magenta_panel, delta_panel], axis=1)

if __name__ == '__main__':
    args = parse_args()
    args = preset(args)

    edit = EditStableDiffusion(args)
    edit.scheduler.set_timesteps(edit.for_steps, device=edit.device)
    edit.for_prompt_emb  = edit.for_prompt_emb.detach()
    edit.edit_prompt_emb = edit.edit_prompt_emb.detach()
    edit.null_prompt_emb = edit.null_prompt_emb.detach()

    do_cfg = edit.guidance_scale > 1.0

    # load zt + directions saved by main.py (whole-image, since disabled the mask)
    zt = torch.load(os.path.join(edit.result_folder, "xt.pt"),
                    map_location=edit.device).type(edit.dtype).to(edit.device)

    basis_dir = os.path.join(
        edit.result_folder, 'basis',
        f'local_basis-{edit.edit_t}T-pca-rank-{args.pca_rank}-select-mask{args.mask_index}',
    )
    vT_modify = torch.load(os.path.join(basis_dir, 'vT-modify.pt'),
                           map_location=edit.device).type(edit.dtype)
    print(f'loaded vT_modify: {vT_modify.shape}')

    orig_rgb = np.array(Image.open(
        os.path.join(edit.result_folder, 'original.png')).convert('RGB').resize((512, 512)))

    out_dir = os.path.join(edit.result_folder, 'channel_coupling')
    os.makedirs(out_dir, exist_ok=True)

    # same single-step magnitude as the gradcam script. controls edit poower(?)
    MULT = edit.x_space_guidance_num_step
    lam = edit.x_space_guidance_scale * edit.x_space_guidance_edit_step * MULT

    # base trajectory is shared across directions -> compute once
    x0_base = denoise_to_x0(edit, zt, do_cfg).detach()

    rows = []
    for k in range(min(args.pca_rank, vT_modify.shape[0])):
        vk = vT_modify[k] / (vT_modify[k].norm() + 1e-8)
        zt_edit = zt + lam * vk.view_as(zt)

        x0_edit = denoise_to_x0(edit, zt_edit, do_cfg).detach()
        delta = x0_edit - x0_base                       #this is the SCALAR in gradcam, [1,3,512,512]

        avg_rb = (delta[:, 0:1] + delta[:, 2:3]) / 2
        delta[:, 0:1] = avg_rb
        delta[:, 2:3] = avg_rb
        
        dG, dM = split_channels(delta)
        
        
        # R=B sanity check: if this ratio is large, magenta split is unreliable for this direction
        # rb_diff = (delta[:, 0:1] - delta[:, 2:3]).abs().mean().item()
        # dm_mean  = dM.abs().mean().item()
        # ratio    = rb_diff / (dm_mean + 1e-8)
        # print(f'dir{k:02d}: |R-B|_mean={rb_diff:.4f}  dM_mean={dm_mean:.4f}  ratio={ratio:.3f}')
        # eG = dG.pow(2).sum().item()
        # eM = dM.pow(2).sum().item()
        # total = eG + eM + 1e-12
        # coupling = min(eG, eM) / (max(eG, eM) + 1e-12)  # 1 = fully coupled, 0 = single-channel

        # rows.append((k, eG, eM, eG / total, eM / total, coupling))
        # print(f'dir{k:02d}: energy_G={eG:.3f}  energy_M={eM:.3f}  '
        #       f'(G%={eG/total:.2f} M%={eM/total:.2f})  coupling={coupling:.2f}')

        # spatial map of the response in green/magenta scheme (white = coupled)
        # Image.fromarray(delta_to_rgb(dG, dM)).save(
        #     os.path.join(out_dir, f'dir{k:02d}_response.png'))
        
        # spatial map of the response in green/magenta scheme (white = coupled)
        Image.fromarray(make_trio(dG, dM)).save(
            os.path.join(out_dir, f'dir{k:02d}_trio.png'))

    #
    # with open(os.path.join(out_dir, 'coupling.csv'), 'w') as f:
    #     f.write('dir,energy_G,energy_M,G_frac,M_frac,coupling\n')
    #     for r in rows:
    #         f.write(','.join(str(x) for x in r) + '\n')

    print(f'Done. Results in {out_dir}')