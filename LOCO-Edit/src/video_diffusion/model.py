import torch
from torch.utils.checkpoint import checkpoint
from diffusers import UNet3DConditionModel, Transformer2DModel
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_3d_blocks import TemporalConvLayer
from diffusers.models.transformers.transformer_temporal import TransformerTemporalModel


#idea: build a single channel (say chA) diffuser model, but it needs to be conditoned
#on its corresponding chB?

#previosuly trained chB encoder can be used here to encoder the corresopnding chB frames?

def build_unet(frame_size=128, channels=1, cross_attention_dim=256):
    return UNet3DConditionModel(
        sample_size=frame_size,
        in_channels=channels,
        out_channels=channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        #down resolutions: 128, 64, 32, 16, 8, 4 -> attention at 16x16 and 8x8
        down_block_types=(
            "DownBlock3D", "DownBlock3D","DownBlock3D",
            "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D",
        ),
        #up blocks are listed lowest resolution first: 4, 8, 16, 32, 64, 128
        up_block_types=(
            "UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D",
            "UpBlock3D", "UpBlock3D", "UpBlock3D",
        ),
        cross_attention_dim=cross_attention_dim,
        #diffusers naming bug: for this class attention_head_dim is the NUMBER of heads
        attention_head_dim=8,
    )


def enable_gradient_checkpointing(unet):
    '''
    recompute every resnet / temporal conv / attention layer's activations in backward instead
    of storing them. diffusers 0.27's 3D blocks ignore their gradient_checkpointing flag, so
    wrap the layers here. per-layer (not per-block) keeps the recompute peak small.
    '''
    layer_types = (ResnetBlock2D, TemporalConvLayer, Transformer2DModel, TransformerTemporalModel)
    for m in unet.modules():
        if isinstance(m, layer_types):
            def ckpt_forward(*args, _fwd=m.forward, **kwargs):
                if not torch.is_grad_enabled():
                    return _fwd(*args, **kwargs)
                return checkpoint(_fwd, *args, use_reentrant=False, **kwargs)
            m.forward = ckpt_forward
    
    
'''
#NOTE: test out with diffusion transformer later (for both 2D and 3D cases..)
'''