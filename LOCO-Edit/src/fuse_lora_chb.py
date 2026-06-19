# fuse_lora_chB.py
import torch
from diffusers import StableDiffusionPipeline

BASE = "Manojb/stable-diffusion-2-1-base"
LORA = "checkpoints_new/sd21_chB"         
OUT  = "checkpoints_new/sd21_chB_fused"

pipe = StableDiffusionPipeline.from_pretrained(BASE, torch_dtype=torch.float32, safety_checker=None)
pipe.load_lora_weights(LORA)               
pipe.fuse_lora()
pipe.unload_lora_weights()
pipe.save_pretrained(OUT)
print("fused model ->", OUT)