import torch.nn as nn
import torch
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
'''
1. use SD VAE to encode the gren channels
2. Hook this as cross attention into SD2.1 finetuned on magenta
3. then trian
'''
#train on cuda device 9 only! its free

FINETUNED_MODEL_PATH = "checkpoints_new/sd21_cell_chB/pytorch_lora_weights.safetensors"
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
            nn.Conv2d(4, 256, kernel_size=3, padding=1),   # [B, 256, 32, 32]
            nn.SiLU(),
            nn.Conv2d(256, 768, kernel_size=4, stride=4),  # [B, 768, 8, 8]
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




'''
be defauly, unet atn2 layers take K,V from the text conditioning. need to complement/add that with
custom K,V from the channel encoder..

add new attn ALONGSIDE it
'''
class channelAttnProcessor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.to_k_img = nn.Linear(768, hidden_dim, bias=False)
        self.to_v_img = nn.Linear(768, hidden_dim, bias=False)
        self.image_tokens = None  #set this before each forward pass
        
        self.store_attn = False  #flip on to capture attention during forward pass
        self.attn_map   = None
        
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        '''
        FROZEN original text attention below
        '''
        query = attn.to_q(hidden_states)
        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        text_weights = attn.get_attention_scores(query, key, attention_mask)
        text_out     = torch.bmm(text_weights, value)
        text_out     = attn.batch_to_head_dim(text_out)

        '''
        ADDING image/channel attention (trainable!)
        '''
        key_img   = self.to_k_img(self.image_tokens)
        value_img = self.to_v_img(self.image_tokens)

        key_img   = attn.head_to_batch_dim(key_img)
        value_img = attn.head_to_batch_dim(value_img)

        img_weights = attn.get_attention_scores(query, key_img)
        img_out     = torch.bmm(img_weights, value_img)
        img_out     = attn.batch_to_head_dim(img_out)
        
        img_weights = attn.get_attention_scores(query, key_img)
        if self.store_attn:
            self.attn_map = img_weights.detach()  #[B*heads, Q_len, 64]

        #add
        out = text_out + img_out
        out = attn.to_out[0](out)   # linear projection
        out = attn.to_out[1](out)   # dropout
        return out
    
def hook_unet(unet):
    for name, module in unet.named_modules():
        if name.endswith("attn2"):
            hidden_dim = module.to_k.out_features  #get the right dim for this layer
            module.processor = channelAttnProcessor(hidden_dim)




if __name__ == "__main__":
    #some testing scripts
    #  for name, module in unet.named_modules():
    #     if "attn2" in name:
    #         print(name, "→", type(module).__name__)
    hook_unet(unet)
    for name, module in unet.named_modules():
        if name.endswith("attn2"):
            print(name, "→", type(module.processor).__name__)
