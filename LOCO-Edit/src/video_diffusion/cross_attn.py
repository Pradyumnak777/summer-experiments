'''
this is for ideating an IP adapter like method for videos, where a video can be conditioned on a video.

frame-matched conditioning: frame t of the generated clip cross-attends only to frame t of the
cond clip. the stock UNet3DConditionModel.forward repeat_interleaves encoder_hidden_states over
frames (one token set shared by the whole clip), so instead each cond frame is tokenized on its
own and a pre-hook on every Transformer2DModel (the only modules that read encoder_hidden_states)
swaps in the per-frame tokens [B*T, N, D]. the UNet folds frames into the batch as b*T + t, and
cond is folded the same way, so the rows line up.
'''
import torch.nn as nn
from diffusers import Transformer2DModel


class FrameConditionedUNet(nn.Module):
    def __init__(self, unet, encoder):
        super().__init__()
        self.unet = unet
        self.encoder = encoder   #frames [N, 1, H, W] -> tokens [N, n_tok, D]
        self._tokens = None
        for m in unet.modules():
            if isinstance(m, Transformer2DModel):
                m.register_forward_pre_hook(self._inject_tokens, with_kwargs=True)

    def _inject_tokens(self, module, args, kwargs):
        kwargs["encoder_hidden_states"] = self._tokens
        return args, kwargs

    def encode(self, cond):
        '''cond clip [B, 1, T, H, W] -> per-frame tokens [B*T, n_tok, D], folded like the UNet folds frames.'''
        B, C, T, H, W = cond.shape
        return self.encoder(cond.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W))

    def forward(self, sample, timestep, cond=None, tokens=None):
        '''pass cond, or tokens from encode() to skip re-encoding (e.g. every sampling step).'''
        if tokens is None:
            tokens = self.encode(cond)
        #kept (not reset) after forward: gradient checkpointing would re-run the hooks during backward
        self._tokens = tokens
        #only there to satisfy forward()'s signature; the hooks replace it before any attention uses it
        placeholder = tokens.new_zeros(sample.shape[0], 1, tokens.shape[-1])
        return self.unet(sample, timestep, encoder_hidden_states=placeholder).sample
