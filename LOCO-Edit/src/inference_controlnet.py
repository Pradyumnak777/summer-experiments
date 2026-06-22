# infer_controlnet.py
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

DEVICE          = "cuda:0"          # physical GPU 9 via CUDA_VISIBLE_DEVICES=9
BASE_MODEL      = "checkpoints_new/sd21_chB_fused"
CONTROLNET_DIR  = "checkpoints_new/controlnet_chA2chB"
PROMPT          = "an image of cells in fluorescent microscopy"
NUM_STEPS       = 30
GUIDANCE_SCALE  = 3.0
SEED            = 0
OUT_DIR         = Path("controlnet_results")

# chA images to run inference on — edit this list or pass as args
INPUT_IMAGES = [
    "data/microscopy_lora_chA/v00_f000_t0_chA.png",
    "data/microscopy_lora_chA/v00_f000_t1_chA.png",
]

OUT_DIR.mkdir(exist_ok=True)

controlnet = ControlNetModel.from_pretrained(CONTROLNET_DIR, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe = pipe.to(DEVICE)

generator = torch.Generator(device=DEVICE).manual_seed(SEED)

import numpy as np
from PIL import Image

def tile_image(img_path, size=512):
    """Yield (tile_image, row, col) using the same grid logic as prepare_data_2ch.py"""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    ny, nx = h // size, w // size
    off_y = (h - ny * size) // 2
    off_x = (w - nx * size) // 2
    for iy in range(ny):
        for ix in range(nx):
            y = off_y + iy * size
            x = off_x + ix * size
            tile = Image.fromarray(arr[y:y+size, x:x+size])
            yield tile, iy, ix

def to_magenta(gray_img):
    """Convert grayscale PIL image to magenta-colorized RGB."""
    g = np.array(gray_img.convert("L"))
    return Image.fromarray(
        np.dstack([g, np.zeros_like(g), g]).astype(np.uint8)
    )

for input_path in INPUT_IMAGES:
    input_path = Path(input_path)
    base = input_path.stem.replace("_chA", "")

    # load condition (chA / green)
    cond = Image.open(input_path).convert("RGB").resize((512, 512))

    # generate predicted chB
    pred = pipe(
        PROMPT,
        image=cond,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]                        # PIL RGB, grayscale-looking

    # save: grayscale pred, magenta pred, green condition
    pred_gray    = pred.convert("L")
    pred_magenta = to_magenta(pred)
    cond_green   = np.array(pred.convert("L"))   # reuse luminance for green LUT
    cond_green   = Image.fromarray(
        np.dstack([np.zeros_like(cond_green), cond_green, np.zeros_like(cond_green)]).astype(np.uint8)
    )

    pred_gray.save(   OUT_DIR / f"{base}_pred_chB_gray.png")
    pred_magenta.save(OUT_DIR / f"{base}_pred_chB_magenta.png")

    # side-by-side: [chA green | predicted chB magenta]
    side = Image.new("RGB", (1024, 512))
    side.paste(cond_green, (0,   0))
    side.paste(pred_magenta,  (512, 0))
    side.save(OUT_DIR / f"{base}_compare.png")

    # if ground-truth chB exists, add it to the side-by-side
    gt_path = Path(str(input_path).replace("chA", "chB").replace("microscopy_lora_chA", "microscopy_lora_chB"))
    if gt_path.exists():
        gt = Image.open(gt_path).convert("L").resize((512, 512))
        gt_magenta = to_magenta(gt)
        trio = Image.new("RGB", (1536, 512))
        trio.paste(cond_green,   (0,    0))
        trio.paste(pred_magenta, (512,  0))
        trio.paste(gt_magenta,   (1024, 0))
        trio.save(OUT_DIR / f"{base}_trio.png")   # green | predicted | ground truth
        print(f"{base}: saved trio (green | pred | GT)")
    else:
        print(f"{base}: saved compare (green | pred)")

print(f"\nall results -> {OUT_DIR}/")