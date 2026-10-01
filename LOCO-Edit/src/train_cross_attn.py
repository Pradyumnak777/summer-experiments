
import os, csv, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDPMScheduler

from VAE_disent.diffusion_model import build_unet
from VAE_disent.data_utils import twoChannelDataset
from cross_attn_modules import ChannelEncoder, install_cross_attn, set_tokens
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()


def setup_DDP():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_DDP():
    dist.destroy_process_group()

def set_token_mask(procs, mask):
    for p in procs.values():
        p.token_mask = mask

def clear_token_mask(procs):
    for p in procs.values():
        p.token_mask = None


if __name__ == "__main__":
    local_rank = setup_DDP()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if local_rank == 0:
        print(f"Using device: {device}")

    CHA_DIR  = "data/singlecell_chA_split/train"
    CHB_DIR  = "data/singlecell_chB_split/train"

    #NOTE: old below
    # TGT_CKPT = "diffusion_checkpoints/ddpm_chB_128_masked/unet_ema_epoch40.pt"   #denoiser (chB)
    # SRC_CKPT = "diffusion_checkpoints/ddpm_chA_128_masked/unet_ema_epoch40.pt"   #encoder  (chA)

    #NOTE: new below-
    TGT_CKPT = "diffusion_checkpoints/ddpm_chA_128_masked/unet_ema_epoch40.pt"   #denoiser (chA)
    SRC_CKPT = "diffusion_checkpoints/ddpm_chB_128_masked/unet_ema_epoch40.pt"   #encoder  (chB)

    TGT_IDX  = 0     #chA is channel index 0 in the stacked [2,H,W] tensor
    SRC_IDX  = 1     #chB is channel index 1

    SAVE_DIR   = "cross_attn_checkpoints/BtoA_v2"
    IMG_SIZE   = 128
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LR         = 1e-4
    TOKEN_DIM  = 256
    STOP_BLOCK = 3       #tap chA encoder after down_blocks[3] -> 8x8 = 64 tokens
    # TOKEN_DROP = 0.1
    if local_rank == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)

    denoiser = build_unet(IMG_SIZE, channels=1).to(local_rank)
    denoiser.load_state_dict(torch.load(TGT_CKPT, map_location=f"cuda:{local_rank}"))
    denoiser.requires_grad_(False)

    src_unet = build_unet(IMG_SIZE, channels=1)
    src_unet.load_state_dict(torch.load(SRC_CKPT, map_location="cpu"))
    encoder = ChannelEncoder(src_unet, stop_at_block=STOP_BLOCK, token_dim=TOKEN_DIM).to(local_rank)

    procs = install_cross_attn(denoiser, token_dim=TOKEN_DIM, scale=1.0)
    for p in procs.values():
        p.to(local_rank)

    #only these train: encoder proj + the added K,V in every attn layer
    trainable = list(encoder.proj.parameters())
    for p in procs.values():
        trainable += list(p.to_k_img.parameters()) + list(p.to_v_img.parameters()) + [p.scale]

    if local_rank == 0:
        print(f"trainable params: {sum(t.numel() for t in trainable)/1e6:.2f}M")

    #NOTE: sending trainable-param-holding modules to DDP
    encoder  = DDP(encoder, device_ids=[local_rank], output_device=local_rank)
    denoiser = DDP(denoiser, device_ids=[local_rank], output_device=local_rank)

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir="data/singlecell_mask")

    train_sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )

    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         sampler=train_sampler, num_workers=4, drop_last=True)

    optimizer = torch.optim.AdamW(trainable, lr=LR)
    lr_sched  = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=200,
        num_training_steps=len(loader) * NUM_EPOCHS,
    )
    ema = EMAModel(trainable, decay=0.9999, use_ema_warmup=True, power=0.75)

    if local_rank == 0:
        log_file   = open(f"{SAVE_DIR}/loss_log.csv", "w", newline="")
        log_writer = csv.writer(log_file); log_writer.writerow(["step", "epoch", "loss"])

    total_steps = len(loader) * NUM_EPOCHS
    progress = tqdm(total=total_steps, desc="training", dynamic_ncols=True, disable=(local_rank != 0))
    global_step = 0

    def save_ckpt(tag):
        torch.save({
            "encoder_proj": encoder.module.proj.state_dict(),
            "procs": {name: p.state_dict() for name, p in procs.items()},
        }, f"{SAVE_DIR}/adapter_{tag}.pt")

        ema.store(trainable)
        ema.copy_to(trainable)
        torch.save({
            "encoder_proj": encoder.module.proj.state_dict(),
            "procs": {name: p.state_dict() for name, p in procs.items()},
        }, f"{SAVE_DIR}/adapter_ema_{tag}.pt")
        ema.restore(trainable)

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        for x, mask in loader:
            x = x.to(local_rank)                          # [B, 2, H, W]
            mask = mask.to(local_rank) #[b, 1, h, w]

            src_img = x[:, SRC_IDX:SRC_IDX+1]         # chA, the conditioning channel
            tgt_img = x[:, TGT_IDX:TGT_IDX+1]         # chB, the channel we denoise
            B = x.shape[0]

            # src_img_masked = src_img * mask - (1 - mask) #cell kept, background -> black


            tokens = encoder(src_img)                 # [B, N, token_dim]
            #token dropout -> lets us do classifier-free guidance at inference
            # keep = (torch.rand(B, 1, 1, device=DEVICE) >= TOKEN_DROP)
            # tokens = tokens * keep
            # set_tokens(procs, tokens)
            '''
            new token masking below
            '''
            # downsample cell mask to the 8x8 token grid; "keep" if ANY cell pixel falls in that patch
            token_mask = F.adaptive_max_pool2d(mask, output_size=(8, 8))   # [B,1,8,8]
            token_mask = (token_mask.flatten(1) > 0.5)                     # [B, 64] bool

            # keep = (torch.rand(B, 1, 1, device=DEVICE))
            # tokens = tokens * keep
            set_tokens(procs, tokens)
            set_token_mask(procs, token_mask)

            noise = torch.randn_like(tgt_img)

            '''
            below was randomly sampling timestep to denoise from. chose more "challening" timesteps,
            with more noise, so the model cant easily denoise using its chB weights
            '''
            # t = torch.randint(0, 1000, (B,), device=DEVICE).long()

            T_SKEW = 1.5   # >1 biases sampled t toward HIGH noise, where chA conditioning
                   # actually has to matter for generation; 1.0 = back to plain uniform
            u = torch.rand(B, device=local_rank)
            t = (u ** (1.0 / T_SKEW) * 1000).long().clamp(max=999)

            noisy = scheduler.add_noise(tgt_img, noise, t)
            noise_pred = denoiser(noisy, t).sample

            '''
            trying a heuristic!!
            '''
            # per_pix = F.mse_loss(noise_pred, noise, reduction="none")
            # loss    = (per_pix * mask).sum() / mask.sum().clamp_min(1.0)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            lr_sched.step()
            ema.step(trainable)


            global_step += 1
            if local_rank == 0:
                log_writer.writerow([global_step, epoch, loss.item()])
                progress.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}")
            progress.update(1)

        if local_rank == 0 and (epoch % 10 == 0 or epoch == NUM_EPOCHS - 1):
            save_ckpt(f"epoch{epoch}")
            log_file.flush()
            tqdm.write(f"  [ckpt] saved adapter at epoch {epoch}")

    progress.close()
    if local_rank == 0:
        log_file.close()
        save_ckpt("final")
        print(f"done. adapters in {SAVE_DIR}/")

    cleanup_DDP()
