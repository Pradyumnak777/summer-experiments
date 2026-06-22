import torch
from diffusers import UNet2DConditionModel

'''
1. use SD VAE to encode the gren channels
2. Hook this as cross attention into SD2.1 finetuned on magenta
3. then trian
'''
FINETUNED_MODEL_PATH = "checkpoints_new/sd21_magenta/pytorch_lora_weights.safetensors"
unet = UNet2DConditionModel.from_pretrained("Manojb/stable-diffusion-2-1-base", subfolder="unet")



if __name__ == "__main__":
    
    #some testing scripts
     