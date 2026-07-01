import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from main import channelEncode, hook_unet, channelAttnProcessor
import debugpy
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

# if os.getenv("DEBUGPY", "0") == "1":
#     debugpy.listen(("0.0.0.0", 5678))
#     print("Waiting for debugger attach on 5678...")
#     debugpy.wait_for_client()


#---config---
DEVICE       = torch.device("cuda:0")
BASE_MODEL   = "Manojb/stable-diffusion-2-1-base"
LORA_PATH    = "checkpoints_new/sd21_magenta"
CKPT_PATH    = "checkpoints_new/channel_cond_epoch19_new.pt"
GREEN_DIR    = "data/microscopy_lora_green"
FIXED_PROMPT = "an image of magenta cells in fluorescent microscopy"
NUM_STEPS    = 50
OUT_DIR      = "channel_cond/eval_outputs"

import os
os.makedirs(OUT_DIR, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
pipe.load_lora_weights(LORA_PATH)
unet         = pipe.unet.to(DEVICE)
vae          = pipe.vae.to(DEVICE)
text_encoder = pipe.text_encoder.to(DEVICE)
tokenizer    = pipe.tokenizer

scheduler = DDIMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
scheduler.set_timesteps(NUM_STEPS)

for model in [unet, vae, text_encoder]:
    for param in model.parameters():
        param.requires_grad = False

hook_unet(unet)
unet.to(DEVICE)
channel_encoder = channelEncode(vae).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
channel_encoder.load_state_dict(ckpt["channel_encoder"])
for name, module in unet.named_modules():
    if isinstance(module, channelAttnProcessor) and name in ckpt["attn_processors"]:
        module.load_state_dict(ckpt["attn_processors"][name])

channel_encoder.eval()
unet.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

with torch.no_grad():
    text_inputs = tokenizer(
        [FIXED_PROMPT], return_tensors="pt",
        padding="max_length", max_length=tokenizer.model_max_length, truncation=True
    ).input_ids.to(DEVICE)
    text_emb = text_encoder(text_inputs)[0]


def generate(green_input):
    with torch.no_grad():
        if isinstance(green_input, torch.Tensor):
            green_tensor = green_input
        else:
            green_img = Image.open(green_input).convert("RGB")
            green_tensor = transform(green_img).unsqueeze(0).to(DEVICE)

        image_tokens = channel_encoder(green_tensor)

        for module in unet.modules():
            if isinstance(module, channelAttnProcessor):
                module.image_tokens = image_tokens

        latent = torch.randn(1, 4, 64, 64, device=DEVICE)

        for t in scheduler.timesteps:
            noise_pred = unet(latent, t, encoder_hidden_states=text_emb).sample
            latent = scheduler.step(noise_pred, t, latent).prev_sample

        latent = latent / vae.config.scaling_factor
        image = vae.decode(latent).sample
        image = (image.clamp(-1, 1) + 1) / 2

    return green_tensor, image


def get_unseen_crop(img_path, tile=512):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    off_x = (w - tile) // 2
    return img.crop((off_x, 256, off_x + tile, 256 + tile))


def get_green_to_magenta_maps(green_input, token_ids=None, suffix="seen", img_name=None):

    if img_name is None and isinstance(green_input, str):
        img_name = os.path.splitext(os.path.basename(green_input))[0]
    elif img_name is None:
        img_name = "tensor"

    if isinstance(green_input, torch.Tensor):
        green_tensor = green_input
    else:
        green_tensor = transform(Image.open(green_input).convert("RGB")).unsqueeze(0).to(DEVICE)

    image_tokens = channel_encoder(green_tensor)

    procs = []
    for module in unet.modules():
        if isinstance(module, channelAttnProcessor):
            module.image_tokens = image_tokens
            module.store_attn = True
            procs.append(module)

    if token_ids is None:
        token_ids = list(range(64))

    accum = {k: torch.zeros(512, 512) for k in token_ids}

    latent = torch.randn(1, 4, 64, 64, device=DEVICE)
    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = unet(latent, t, encoder_hidden_states=text_emb).sample
            latent = scheduler.step(noise_pred, t, latent).prev_sample

            for m in procs:
                a = m.attn_map.mean(dim=0).cpu()
                Q_len = a.shape[0]
                mag_grid = int(round(Q_len ** 0.5))
                for k in token_ids:
                    col_k = a[:, k].reshape(mag_grid, mag_grid)
                    up = F.interpolate(col_k[None, None], size=(512, 512),
                                       mode="bilinear", align_corners=False).squeeze()
                    accum[k] += up

        magenta = vae.decode(latent / vae.config.scaling_factor).sample
        magenta = ((magenta.clamp(-1, 1) + 1) / 2).squeeze().cpu().permute(1, 2, 0).numpy()

    green_vis = (green_tensor.squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)

    # print per-token max so you can see the distribution
    token_maxes = {k: accum[k].max().item() for k in token_ids}
    for k in token_ids:
        print(f"token {k:02d} (r={k//8}, c={k%8}): max={token_maxes[k]:.4f}")

    # percentile-based normalization: honest across tokens but doesn't let
    # 1-2 outliers crush everything else to black
    all_maxes = sorted(token_maxes.values())
    p95 = all_maxes[int(0.95 * len(all_maxes))]
    global_ref = p95 + 1e-6
    print(f"p95 token max = {p95:.4f}  (used as colormap reference)")

    token_dir = f"channel_cond/eval_outputs/tokens_{suffix}_{img_name}_new"
    os.makedirs(token_dir, exist_ok=True)

    for k in token_ids:
        heat = (accum[k] / global_ref).clamp(0, 1)
        row_g, col_g = k // 8, k % 8

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(green_vis)
        axes[0].add_patch(plt.Rectangle(
            (col_g * 64, row_g * 64), 64, 64,
            edgecolor="lime", facecolor="none", lw=2
        ))
        axes[0].set_title(f"green token {k} (r={row_g}, c={col_g})  max={token_maxes[k]:.3f}")
        axes[0].axis("off")

        axes[1].imshow(magenta)
        axes[1].imshow(heat.numpy(), cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[1].set_title("magenta influence")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(token_dir, f"token_{k:02d}.png"), dpi=120, bbox_inches="tight")
        plt.close()

    print(f"saved {len(token_ids)} token maps → {token_dir}/")

    for module in unet.modules():
        if isinstance(module, channelAttnProcessor):
            module.store_attn = False


def make_composite(green_tensor, magenta_tensor,
                   out_path="channel_cond/eval_outputs/composite.png"):
    green_gray   = green_tensor.squeeze().mean(dim=0).cpu()
    magenta_gray = magenta_tensor.squeeze().mean(dim=0).cpu()
    green_gray   = (green_gray * 0.5 + 0.5).clamp(0, 1)
    magenta_gray = magenta_gray.clamp(0, 1)
    composite = torch.stack([magenta_gray, green_gray, magenta_gray], dim=0)
    save_image(composite, out_path)
    print(f"saved composite → {out_path}")


if __name__ == "__main__":
    use_unseen = False
    raw_path = "data/microscopy_lora_green/v00_f000_t2_chA.png"

    suffix = "unseen" if use_unseen else "seen"

    if use_unseen:
        img_name = os.path.splitext(os.path.basename(raw_path))[0] + "_unseen"
        crop = get_unseen_crop(raw_path)
        crop.save(f"channel_cond/eval_outputs/input_crop_{suffix}_{img_name}.png")
        green_input = transform(crop).unsqueeze(0).to(DEVICE)
    else:
        green_input = raw_path
        img_name = os.path.splitext(os.path.basename(raw_path))[0]

    torch.manual_seed(42)
    green_tensor, generated = generate(green_input)

    green_vis = (green_tensor.clamp(-1, 1) + 1) / 2
    comparison = torch.cat([green_vis, generated], dim=3)
    save_image(comparison, f"channel_cond/eval_outputs/eval_{suffix}_{img_name}.png")
    print(f"saved eval_{suffix}_{img_name}.png")

    torch.manual_seed(42)
    get_green_to_magenta_maps(green_tensor, suffix=suffix, img_name=img_name)
    make_composite(green_tensor, generated, out_path=f"channel_cond/eval_outputs/composite_{suffix}_{img_name}.png")