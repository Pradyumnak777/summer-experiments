'''
chA denoiser conditioned on chB through the UNet's built-in cross-attention.

UNet2DConditionModel is the 2D counterpart of the video model's UNet3DConditionModel: same
channel widths, and cross-attention at the same resolutions (16x16, 8x8 and the 4x4 mid block).
chB is tokenized by the frozen chB DDPM trunk (ChannelEncoder, as in the video model and the
earlier 2D cross-attention work): 8x8 feature map -> 64 tokens of width 256 per image.
'''
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

from VAE_disent.diffusion_model import build_unet as build_unet_2d
from cross_attn_modules import ChannelEncoder


def build_unet(size=128, channels=1, cross_attention_dim=256):
    return UNet2DConditionModel(
        sample_size=size,
        in_channels=channels,
        out_channels=channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        #down resolutions: 128, 64, 32, 16, 8, 4 -> cross-attention at 16x16 and 8x8
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D",
        ),
        #up blocks are listed lowest resolution first: 4, 8, 16, 32, 64, 128
        up_block_types=(
            "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
            "UpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
        cross_attention_dim=cross_attention_dim, #NOTE: cross_attndim needs ot equal
        #token_dim, which is the channel width of the chB encoded tokens.. [B, 64, 256]
        
        #diffusers naming bug: for this class attention_head_dim is the NUMBER of heads
        attention_head_dim=8,
    )


class ConditionedUNet(nn.Module):
    def __init__(self, unet, encoder):
        super().__init__()
        self.unet = unet
        self.encoder = encoder   #chB [B, 1, H, W] -> tokens [B, 64, D]

    def encode(self, cond):
        return self.encoder(cond)

    def forward(self, sample, timestep, cond=None, tokens=None):
        '''pass cond, or tokens from encode() to skip re-encoding (e.g. every sampling step).'''
        if tokens is None:
            tokens = self.encode(cond)
        return self.unet(sample, timestep, encoder_hidden_states=tokens).sample


def build_model(chB_ckpt, stop_block=3, token_dim=256, size=128):
    src_unet = build_unet_2d(size, channels=1)
    src_unet.load_state_dict(torch.load(chB_ckpt, map_location="cpu"))
    encoder = ChannelEncoder(src_unet, stop_at_block=stop_block, token_dim=token_dim)
    return ConditionedUNet(build_unet(size, 1, cross_attention_dim=token_dim), encoder)
