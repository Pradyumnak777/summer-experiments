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

if os.getenv("DEBUGPY", "0") == "1":
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach on 5678...")
    debugpy.wait_for_client()


#---config---
DEVICE       = torch.device("cuda:0")
BASE_MODEL   = "Manojb/stable-diffusion-2-1-base"
LORA_PATH    = "checkpoints_new/sd21_magenta/pytorch_lora_weights.safetensors"
CKPT_PATH    = "checkpoints_new/channel_cond_epoch19.pt"
GREEN_DIR    = "data/microscopy_lora_green"
FIXED_PROMPT = "an image of magenta cells in fluorescent microscopy"
NUM_STEPS    = 50   #ddim is way faster than ddpm, 50 steps is enough for eval
OUT_DIR      = "channel_cond/eval_outputs"

import os
os.makedirs(OUT_DIR, exist_ok=True)

#load base + lora same as training
pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
pipe.load_lora_weights(LORA_PATH)
unet         = pipe.unet.to(DEVICE)
vae          = pipe.vae.to(DEVICE)
text_encoder = pipe.text_encoder.to(DEVICE)
tokenizer    = pipe.tokenizer

#swap to ddim for fast inference
scheduler = DDIMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
scheduler.set_timesteps(NUM_STEPS)

#freeze everything
for model in [unet, vae, text_encoder]:
    for param in model.parameters():
        param.requires_grad = False

#hook unet + build channel encoder
hook_unet(unet)
unet.to(DEVICE) 
channel_encoder = channelEncode(vae).to(DEVICE)

#load trained weights
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
channel_encoder.load_state_dict(ckpt["channel_encoder"])
for name, module in unet.named_modules():
    if isinstance(module, channelAttnProcessor) and name in ckpt["attn_processors"]:
        module.load_state_dict(ckpt["attn_processors"][name])

channel_encoder.eval()
unet.eval()

#transform for loading green images (same as training)
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

'''
center crop for custom_img to genreate unseen data..
'''

transform = transforms.Compose([
    transforms.Resize(512),              #scale shorter edge to 512, keeps aspect ratio
    transforms.CenterCrop((512, 512)),   #cut 512x512 from center
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#pre-compute text embeddings once
with torch.no_grad():
    text_inputs = tokenizer(
        [FIXED_PROMPT], return_tensors="pt",
        padding="max_length", max_length=tokenizer.model_max_length, truncation=True
    ).input_ids.to(DEVICE)
    text_emb = text_encoder(text_inputs)[0]  #[1, 77, 768]

def generate(green_input):
    with torch.no_grad():
        #check if input is already a tensor or a path
        if isinstance(green_input, torch.Tensor):
            green_tensor = green_input
        else:
            green_img = Image.open(green_input).convert("RGB")
            green_tensor = transform(green_img).unsqueeze(0).to(DEVICE)

        image_tokens = channel_encoder(green_tensor)  #[1, 64, 768]

        #set tokens on all processors
        for module in unet.modules():
            if isinstance(module, channelAttnProcessor):
                module.image_tokens = image_tokens #handing green img tokens for cross attn..

        #start from pure noise
        latent = torch.randn(1, 4, 64, 64, device=DEVICE) #generation of magenta..

        #ddim denoising loop
        for t in scheduler.timesteps:
            noise_pred = unet(latent, t, encoder_hidden_states=text_emb).sample #passed img tokens above, now passing text embs..
            latent = scheduler.step(noise_pred, t, latent).prev_sample

        #decode latent -> image
        latent = latent / vae.config.scaling_factor
        image = vae.decode(latent).sample
        image = (image.clamp(-1, 1) + 1) / 2

    return green_tensor, image


def get_unseen_crop(img_path, tile=512):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size  #PIL: (width, height)

    #match prepare_data_2ch tiling: horizontal center is fixed, tiles stack vertically
    off_x = (w - tile) // 2          #same horizontal center the training tiles used
    #training y positions are 0, 512, 1024, 1536 -> pick 256 (between t0 and t1), unseen
    y = 256
    x = off_x
    crop = img.crop((x, y, x + tile, y + tile))
    return crop
# if __name__ == "__main__":
#     #run on first 5 green images for a quick visual check
#     green_files = sorted([f for f in os.listdir(GREEN_DIR) if f.endswith(".png")])[:5]

#     for fname in green_files:
#         green_path = os.path.join(GREEN_DIR, fname)
#         green_tensor, generated = generate(green_path)

#         #unnormalize green for saving
#         green_vis = (green_tensor.clamp(-1, 1) + 1) / 2

#         #save side by side: green | generated magenta
#         comparison = torch.cat([green_vis, generated], dim=3)  #concat along width
#         out_path = os.path.join(OUT_DIR, fname.replace("chA", "eval"))
#         save_image(comparison, out_path)
#         print(f"saved {out_path}")

def get_green_to_magenta_maps(green_input, token_ids=None,
                              out_path="channel_cond/eval_outputs/green_to_magenta.png"):
    if isinstance(green_input, torch.Tensor):
        green_tensor = green_input
    else:
        green_tensor = transform(Image.open(green_input).convert("RGB")).unsqueeze(0).to(DEVICE)

    image_tokens = channel_encoder(green_tensor)

    #enable capture on ALL processors, collect references
    procs = []
    for module in unet.modules():
        if isinstance(module, channelAttnProcessor):
            module.image_tokens = image_tokens
            module.store_attn = True
            procs.append(module)

    #run full denoising, sum attention maps across all steps and all layers
    #each layer has different Q_len so we upsample to 512x512 immediately
    n_tokens = 64
    if token_ids is None:
        token_ids = [0, 7, 27, 36, 56, 63]

    #accum[k] = running sum of upsampled magenta heatmap for green token k, shape [512, 512]
    accum = {k: torch.zeros(512, 512) for k in token_ids}

    latent = torch.randn(1, 4, 64, 64, device=DEVICE)
    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = unet(latent, t, encoder_hidden_states=text_emb).sample
            latent = scheduler.step(noise_pred, t, latent).prev_sample

            for m in procs:
                a = m.attn_map.mean(dim=0).cpu()        #[Q_len, 64] avg over heads
                Q_len = a.shape[0]
                mag_grid = int(round(Q_len ** 0.5))     #e.g. 8, 16, 32, 64

                for k in token_ids:
                    col_k = a[:, k].reshape(mag_grid, mag_grid)  #[mag_grid, mag_grid]
                    #upsample to 512x512 so all layers can be summed
                    up = F.interpolate(col_k[None, None], size=(512, 512),
                                       mode="bilinear", align_corners=False).squeeze()
                    accum[k] += up

        magenta = vae.decode(latent / vae.config.scaling_factor).sample
        magenta = ((magenta.clamp(-1, 1) + 1) / 2).squeeze().cpu().permute(1, 2, 0).numpy()

    green_vis = (green_tensor.squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)

    n = len(token_ids)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    for i, k in enumerate(token_ids):
        row_g, col_g = k // 8, k % 8   #which 64x64 patch in the green image

        #top row: green image with the chosen patch boxed
        axes[0, i].imshow(green_vis)
        axes[0, i].add_patch(plt.Rectangle(
            (col_g * 64, row_g * 64), 64, 64,
            edgecolor="lime", facecolor="none", lw=2
        ))
        axes[0, i].set_title(f"token {k} (row={row_g}, col={col_g})")
        axes[0, i].axis("off")

        #bottom row: that token's summed attention heatmap over the magenta image
        heat = accum[k]
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
        axes[1, i].imshow(magenta)
        axes[1, i].imshow(heat.numpy(), cmap="jet", alpha=0.5)
        axes[1, i].set_title(f"→ magenta influence")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved → {out_path}")

    #cleanup
    for module in unet.modules():
        if isinstance(module, channelAttnProcessor):
            module.store_attn = False    

def make_composite(green_tensor, magenta_tensor,
                   out_path="channel_cond/eval_outputs/composite.png"):
    #convert both to single-channel grayscale (they're RGB but grayscale content)
    green_gray   = green_tensor.squeeze().mean(dim=0).cpu()    #[512, 512]
    magenta_gray = magenta_tensor.squeeze().mean(dim=0).cpu()  #[512, 512]

    #unnormalize from [-1,1] → [0,1]
    green_gray   = (green_gray * 0.5 + 0.5).clamp(0, 1)
    magenta_gray = magenta_gray.clamp(0, 1)  #generate() already outputs [0,1]

    #composite: magenta=R&B, green=G (same as prepare_data_2ch merge_preview)
    composite = torch.stack([magenta_gray, green_gray, magenta_gray], dim=0)  #[3, 512, 512]
    save_image(composite, out_path)
    print(f"saved composite → {out_path}")


if __name__ == "__main__":
    raw_path = "test_data_2/0.jpg"
    
    #get unseen crop instead of resizing/stretching the whole image
    crop = get_unseen_crop(raw_path)
    crop.save("channel_cond/eval_outputs/unseen_input_crop.png")  #save so you can inspect the input

    green_tensor = transform(crop).unsqueeze(0).to(DEVICE)
    torch.manual_seed(7)
    green_tensor, generated = generate(green_tensor)  #pass tensor directly

    #unnormalize green for saving
    green_vis = (green_tensor.clamp(-1, 1) + 1) / 2
    #save side by side: green input | generated magenta
    comparison = torch.cat([green_vis, generated], dim=3)
    save_image(comparison, "channel_cond/eval_outputs/unseen_eval.png")
    print("saved unseen_eval.png")
    
    torch.manual_seed(7)
    get_green_to_magenta_maps(green_tensor)
    
    make_composite(green_tensor, generated)  