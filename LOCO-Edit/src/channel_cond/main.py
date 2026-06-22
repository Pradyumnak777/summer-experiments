import torch.nn as nn
import torch
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
'''
1. use SD VAE to encode the gren channels
2. Hook this as cross attention into SD2.1 finetuned on magenta
3. then trian
'''
FINETUNED_MODEL_PATH = "checkpoints_new/sd21_magenta/pytorch_lora_weights.safetensors"
unet = UNet2DConditionModel.from_pretrained("Manojb/stable-diffusion-2-1-base", subfolder="unet")
vae = AutoencoderKL.from_pretrained("Manojb/stable-diffusion-2-1-base", subfolder="vae") #for encoding channel

class channelEncode(nn.Module): 
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False #freezing all
            
        #trainable projection: [B, 4, 64, 64] → [B, 768, 8, 8] → [B, 64, 768]
        self.projection = nn.Sequential(
            nn.Conv2d(4, 256, kernel_size=3, padding=1),   # [B, 256, 64, 64]
            nn.SiLU(),
            nn.Conv2d(256, 768, kernel_size=8, stride=8),  # [B, 768, 8, 8]
        )

        
    def encode(self, channel_img):
        with torch.no_grad():
            latent_dist = self.vae.encode(channel_img).latent_dist
            latent = latent_dist.sample()
        return latent #ts is of dimension [B, 4, 64, 64]
    def forward(self, channel_img):
        latent = self.encode(channel_img)
        #now, tokens need to be output (similar to text tokens for conditioning)
        x = self.projection(latent) #[B, 768, 8, 8]
        x = x.flatten(2)         #[B, 768, 64]
        x = x.permute(0, 2, 1)    #[B, 64, 768]
        return x





if __name__ == "__main__":
    #some testing scripts
     