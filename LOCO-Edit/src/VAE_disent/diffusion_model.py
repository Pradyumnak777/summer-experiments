'''
2-channel UNet for pixel-space DDPM on 128x128 cell crops.
Defined once so train + sample scripts always agree on architecture.
'''
from diffusers import UNet2DModel


def build_unet(img_size=128, channels=2):
    return UNet2DModel(
        sample_size=img_size,
        in_channels=channels,
        out_channels=channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
    )