# # test_lora.py
# import torch
# from diffusers import StableDiffusionPipeline

# BASE = "runwayml/stable-diffusion-v1-5"
# LORA = "checkpoints/sd15_microscopy_lora"

# pipe = StableDiffusionPipeline.from_pretrained(BASE, torch_dtype=torch.float16).to("cuda")
# pipe.load_lora_weights(LORA)
# pipe.safety_checker = None  # microscopy may trip false positives

# out = pipe(
#     "a sks microscopy image of fluorescent transcriptional condensates",
#     num_inference_steps=30,
#     guidance_scale=4.5,
#     num_images_per_prompt=4,
# ).images
# for i, im in enumerate(out):
#     im.save(f"gen_{i}.png")