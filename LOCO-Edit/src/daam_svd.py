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

#how many denoising steps to skip between saved maps (same convention as gradcam_svd_semantic.py)
TIMESTEP_STRIDE = 1

#words whose heatmaps is wanted. must appear in CAPTURE_PROMPT below.
WORDS = ["glasses", "man"]

#formula for perfomring denoising..
MODE = "null+(for-null)"

#run the edited trajectory (zt + lam*vk) and save base/edit/difference maps,
COMPARE_EDIT = True

DAAM_RES = 64  #output overlay size


class AttnStore:
    #just a bucket the attn processor dumps maps into. one entry per cross-attn layer
    def __init__(self):
        self.enabled = False
        self.maps = []  # list of [heads, hw, 77] tensors (one per layer, per capture pass)

    def add(self, probs):
        if self.enabled:
            self.maps.append(probs.detach().float().cpu())

    def reset(self):
        self.maps = []


class DAAMCrossAttnProcessor:
    # plain attention so the softmax matrix actually exists in memory.
    # xformers fuses it away so there's nothing to grab , so custom written
    def __init__(self, store):
        self.store = store

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        attention_mask = attn.prepare_attention_mask(
            attention_mask, encoder_hidden_states.shape[1], batch_size)

        # Q from the image, K/V from the text tokens
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key   = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.store.add(attention_probs)  # [B*heads, hw, 77]

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def install_daam(unet, store):
    #swap my processor onto every cross-attn (attn2) layer, remember the old ones to put back

    originals = {}
    for name, module in unet.named_modules():
        if name.endswith('attn2'):
            originals[name] = module.processor
            module.processor = DAAMCrossAttnProcessor(store)
    return originals


def restore(unet, originals):
    #putting xformers back
    for name, module in unet.named_modules():
        if name in originals:
            module.processor = originals[name]


def token_indices(tokenizer, prompt, word):
    #find where `word` sits in the 77-token sequence. returns a list bc a word can split into pieces
    prompt_ids = tokenizer(prompt).input_ids                      # [BOS, ..., EOS]
    word_ids = tokenizer(word, add_special_tokens=False).input_ids
    for s in range(len(prompt_ids) - len(word_ids) + 1):
        if prompt_ids[s:s + len(word_ids)] == word_ids:
            return list(range(s, s + len(word_ids)))
    raise ValueError(f'"{word}" not found in "{prompt}"')


def layers_to_heatmap(maps, idxs):
    #turn this timestep's 16 attn maps into ONE 64x64 heatmap for a word
    total = torch.zeros(DAAM_RES, DAAM_RES)
    for probs in maps:  #[heads, hw, 77]. there's 16 attn2 layers (cross attn layers)
        hw = probs.shape[1]
        r = int(hw ** 0.5)
        m = probs[:, :, idxs].sum(dim=-1)  #[heads, hw] of only CONCERNED words (like "glasses")
        m = m.view(-1, 1, r, r) #some spatial grid [heads, 1, r, r]
        m = F.interpolate(m, size=(DAAM_RES, DAAM_RES), mode='bicubic', align_corners=False)
        #[heads, 1, 64, 64]
        total += m.sum(dim=0)[0].clamp(min=0) #sum over heads.
    return total                              


def daam_over_trajectory(edit, zt0, store, word_idxs, capture_emb, stride=1):
    '''
    denoise edit_t -> z0 
    1. a throwaway forward w/ the edit prompt JUST to grab attention (no stepping/actual denoising)
    2. the real CFG step that actually moves zt (actual denoising)
    '''
    do_cfg = edit.guidance_scale > 1.0

    def _cfg_et(z, t):
        return edit._classifer_free_guidance(
            z, t, edit.for_prompt_emb, edit.edit_prompt_emb, edit.null_prompt_emb,
            mode=MODE, do_classifier_free_guidance=do_cfg)

    timesteps = edit.scheduler.timesteps
    start = int(edit.edit_t_idx)
    zt = zt0.detach().clone()

    per_step = [] # (i, t, normalized 2d heatmap)
    global_maps = {w: torch.zeros(DAAM_RES, DAAM_RES) for w in word_idxs} #sum over all timesteps

    with torch.no_grad():
        for i in range(start, len(timesteps)):
            t = timesteps[i].to(edit.device)

            # capture pass- this runs the Unet (ONCE!) over the edit prompt(denoisis from 0.7/0.6 to get cross attn maps)
            store.reset(); store.enabled = True #this is for storage of the cross attention maps
            _ = edit.unet(zt, t, encoder_hidden_states=capture_emb).sample
            store.enabled = False

            step_maps = {}
            for w, idxs in word_idxs.items():
                hm = layers_to_heatmap(store.maps, idxs)
                global_maps[w] += hm
                step_maps[w] = hm
            if (i - start) % stride == 0:
                per_step.append((i, int(t.item()), step_maps))

            # one DDIM step with normal CFG dynamics
            et = _cfg_et(zt, t)
            zt = edit.scheduler.step(et, t, zt, eta=0).prev_sample.detach()

    return per_step, global_maps


def to_img(hm):
    hm = hm / (hm.max() + 1e-8)
    hm = F.interpolate(hm[None, None], size=(512, 512), mode='bilinear', align_corners=False)
    return hm[0, 0].numpy()


def save_overlay(orig_rgb, cam, path, alpha=0.5):
    colored = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    overlay = (alpha * colored + (1 - alpha) * orig_rgb).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def save_signed(diff, path):
    """edit-vs-base difference: red = attention gained, blue = lost."""
    d = diff / (diff.abs().max() + 1e-8)
    d = F.interpolate(d[None, None], size=(512, 512), mode='bilinear', align_corners=False)[0, 0].numpy()
    rgb = (cm.bwr(0.5 + d / 2)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


if __name__ == '__main__':
    args = parse_args()
    args = preset(args)

    edit = EditStableDiffusion(args)
    edit.scheduler.set_timesteps(edit.for_steps, device=edit.device)

    #this is the prompt that will change the base. eg here "glasses" is added..
    CAPTURE_PROMPT = edit.edit_prompt
    capture_emb = edit.edit_prompt_emb.detach() #all prompts padded to 77 token length as per CLIP..
    #1024 dim text embeddings- [1, 77, 1024] is total size..

    word_idxs = {w: token_indices(edit.pipe.tokenizer, CAPTURE_PROMPT, w) for w in WORDS}
    print('token indices:', word_idxs)
    '''
    migh tlook like..
    index:  0     1    2      3   4    5     6     7        8    9…76
    token: [BOS]  a   photo  of   a   man  with  glasses  [EOS]  [PAD…]
    '''

    #loading noised latent
    zt_path = os.path.join(edit.result_folder, "xt.pt")
    assert os.path.exists(zt_path), f'missing {zt_path} — run main.py first'
    zt = torch.load(zt_path, map_location=edit.device).type(edit.dtype)

    orig_path = os.path.join(edit.result_folder, 'original.png')
    assert os.path.exists(orig_path), 'original.png not found — run main.py first'
    orig_rgb = np.array(Image.open(orig_path).convert('RGB').resize((512, 512)))

    out_dir = os.path.join(edit.result_folder, 'daam')
    os.makedirs(out_dir, exist_ok=True)

    store = AttnStore()
    originals = install_daam(edit.unet, store)
    try:
        #base trajectory- [WITHOUT adding the v_k directin vector..]
        per_step, global_base = daam_over_trajectory(
            edit, zt, store, word_idxs, capture_emb, stride=TIMESTEP_STRIDE)
        for w in WORDS:
            wdir = os.path.join(out_dir, f'base_{w}')
            os.makedirs(wdir, exist_ok=True)
            for (i, t_val, maps) in per_step:
                save_overlay(orig_rgb, to_img(maps[w]), os.path.join(wdir, f'step{i:03d}_t{t_val:04d}.png'))
            save_overlay(orig_rgb, to_img(global_base[w]), os.path.join(out_dir, f'GLOBAL_base_{w}.png'))

        #edited trajectory- [WITHOUT adding the v_k direction vector..]
        if COMPARE_EDIT:
            basis_dir = os.path.join(
                edit.result_folder, 'basis',
                f'local_basis-{edit.edit_t}T-"{edit.edit_prompt}"-pca-rank-{args.pca_rank}-select-mask{args.mask_index}')
            vT = torch.load(os.path.join(basis_dir, 'vT-modify.pt'),
                            map_location=edit.device).type(edit.dtype)
            vk = vT[0] / (vT[0].norm() + 1e-8)
            lam = edit.x_space_guidance_scale * edit.x_space_guidance_edit_step
            zt_edit = zt + lam * vk.view_as(zt) #new trajectory..

            per_step_e, global_edit = daam_over_trajectory(
                edit, zt_edit, store, word_idxs, capture_emb, stride=TIMESTEP_STRIDE)
            for w in WORDS:
                wdir = os.path.join(out_dir, f'edit_{w}')
                os.makedirs(wdir, exist_ok=True)
                for (i, t_val, maps) in per_step_e:
                    save_overlay(orig_rgb, to_img(maps[w]), os.path.join(wdir, f'step{i:03d}_t{t_val:04d}.png'))
                save_overlay(orig_rgb, to_img(global_edit[w]), os.path.join(out_dir, f'GLOBAL_edit_{w}.png'))
                save_signed(global_edit[w] - global_base[w], os.path.join(out_dir, f'DIFF_{w}.png'))
    finally:
        restore(edit.unet, originals)

    print(f'Done. Heatmaps in {out_dir}')