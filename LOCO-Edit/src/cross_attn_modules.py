
import math
import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention


class ChannelEncoder(nn.Module):
    '''
    run a CLEAN image through conv_in + the first few down blocks of a trained unet,
    read out the feature map, flatten to spatial tokens. token (i,j) <-> a patch of
    the input image, so the tokens ARE spatial.
    '''
    def __init__(self, src_unet, stop_at_block=3, token_dim=256):
        super().__init__()
        #reuse the trained pieces (frozen)
        self.conv_in        = src_unet.conv_in
        self.time_proj      = src_unet.time_proj
        self.time_embedding = src_unet.time_embedding
        self.down_blocks    = nn.ModuleList(src_unet.down_blocks[:stop_at_block + 1])

        tapped_ch = src_unet.config.block_out_channels[stop_at_block]
        #1x1 conv: feature channels -> a fixed token width. this is the ONLY trainable
        #bit of the encoder
        self.proj = nn.Conv2d(tapped_ch, token_dim, kernel_size=1)

        #freeze the reused trunk
        for p in (list(self.conv_in.parameters())
                  + list(self.time_embedding.parameters())
                  + list(self.down_blocks.parameters())):
            p.requires_grad = False

    @torch.no_grad()
    def _trunk(self, x, t):
        #down blocks are FiLM-conditioned on temb so we still feed a timestep.
        #t=0 means "treat this as near noise-free" (DIFT-style clean feature extraction)
        temb = self.time_proj(t)
        temb = self.time_embedding(temb.to(x.dtype))
        h = self.conv_in(x)
        for block in self.down_blocks:
            h, _ = block(hidden_states=h, temb=temb)   #returns (h, res_samples)
        return h

    def forward(self, x, t=None):
        B = x.shape[0]
        if t is None:
            t = torch.zeros(B, device=x.device, dtype=torch.long)
        h = self._trunk(x, t)                    # [B, C, h', w']  (frozen, no grad)
        h = self.proj(h)                         # [B, token_dim, h', w']  (trainable)
        tokens = h.flatten(2).transpose(1, 2)    # [B, h'*w', token_dim]  spatial tokens
        return tokens


class CrossAttnProcessor(nn.Module):
    '''
    decoupled cross-attention. the denoiser's ORIGINAL self-attention runs unchanged
    (frozen weights), and we ADD a second attention over the source-channel tokens
    with its own trainable K,V. the two outputs are summed - IP-adapter style.
    to_v_img is zero-init so at the start the added branch contributes nothing, i.e.
    training begins as the untouched unconditional denoiser.
    '''
    def __init__(self, hidden_dim, token_dim, scale=1.0):
        super().__init__()
        self.to_k_img = nn.Linear(token_dim, hidden_dim, bias=False) #this is the learnable weight matrix for K (keys)
        self.to_v_img = nn.Linear(token_dim, hidden_dim, bias=False) #similarly for values
        nn.init.zeros_(self.to_v_img.weight)   #start as a no-op
        
        '''
        below is the scale of the cross attention weight. making this learnable
        '''
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        
        self.tokens = None        #set before each forward (source-channel tokens)
        self.store_attn = False   #flip on at inference to capture the cross-attn map
        self.attn_map = None
        self.token_mask = None ##NOTE: masking out edge tokens and considering only ones within the mask!!

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)  # [B, HW, C]

        #UNet2DModel self-attention has a group_norm before to_q - must replicate it
        #or the frozen branch won't match the original numerics
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        #---- original self-attention (frozen) ----
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        q = attn.head_to_batch_dim(query)
        k = attn.head_to_batch_dim(key)
        v = attn.head_to_batch_dim(value)
        self_probs = attn.get_attention_scores(q, k, attention_mask)
        out = torch.bmm(self_probs, v)
        out = attn.batch_to_head_dim(out)

        #---- added cross-attention to the source tokens (trainable) ----
        if self.tokens is not None:
            k_img = attn.head_to_batch_dim(self.to_k_img(self.tokens))
            v_img = attn.head_to_batch_dim(self.to_v_img(self.tokens))

            img_attn_mask = None
            if self.token_mask is not None:
                b, n_tok = self.token_mask.shape
                bias = torch.zeros(b, 1, n_tok, device=q.device, dtype=q.dtype)
                bias.masked_fill_(~self.token_mask.unsqueeze(1), float('-inf'))
                img_attn_mask = attn.prepare_attention_mask(bias, n_tok, b)

            img_probs = attn.get_attention_scores(q, k_img, img_attn_mask)
            img_out   = attn.batch_to_head_dim(torch.bmm(img_probs, v_img))
            if self.store_attn:
                self.attn_map = img_probs.detach()
            out = out + self.scale * img_out


        #original output projection (frozen)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if input_ndim == 4:
            out = out.transpose(1, 2).view(b, c, h, w)
        if attn.residual_connection:
            out = out + residual
        out = out / attn.rescale_output_factor
        return out


def install_cross_attn(unet, token_dim, scale=1.0):
    #swap the self-attn processor on every Attention module for our decoupled one.
    #returns {name: proc} so we can grab params + push tokens in later.
    procs = {}
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            hidden_dim = module.to_q.out_features
            proc = CrossAttnProcessor(hidden_dim, token_dim, scale=scale)
            module.set_processor(proc)
            procs[name] = proc
    return procs


def set_tokens(procs, tokens):
    for p in procs.values():
        p.tokens = tokens

def set_store_attn(procs, flag):
    for p in procs.values():
        p.store_attn = flag