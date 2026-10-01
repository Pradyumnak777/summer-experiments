'''
training a video diffusion model for a single chA, conditioned frame-by-frame on the
corresponding chB clip through cross-attention.

- every 60-frame video is cut in half -> two 30-frame clips (frames 0-29 and 30-59).
  chA and chB halves always come from the same frames.
- the UNet3DConditionModel (video_diffusion/model.py) is trained from scratch.
- chB frames are tokenized by the frozen 2D chB DDPM trunk (ChannelEncoder, same as the 2D
  cross-attention work) plus a trainable 1x1 proj. frame t of chA only cross-attends to
  frame t of chB (video_diffusion/cross_attn.py).

run from src/:
    torchrun --nproc_per_node=4 video_diffusion/train.py
'''
import os, sys, csv, time, argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))
from video_diffusion.data_utils import PairedVideoDataset, CHA_DIR, CHB_DIR, MASK_DIR, VAL_RECORDINGS
from video_diffusion.model import build_unet, enable_gradient_checkpointing
from video_diffusion.cross_attn import FrameConditionedUNet
from VAE_disent.diffusion_model import build_unet as build_unet_2d
from cross_attn_modules import ChannelEncoder

CHB_ENCODER_CKPT = SRC_ROOT / "diffusion_checkpoints/ddpm_chB_128_masked/unet_ema_epoch40.pt"
FRAME_SIZE  = 128
TOKEN_DIM   = 256
STOP_BLOCK  = 3        #tap the chB trunk after down_blocks[3] -> 8x8 = 64 tokens per frame
NUM_TRAIN_TIMESTEPS = 1000
PREVIEW_STEPS = 50     #DDIM steps for the preview clips
WARMUP_STEPS  = 500
PREVIEW_TRAIN_CLIP = 0     #fixed clip indices so previews are comparable across epochs
PREVIEW_VAL_CLIP   = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", default=str(SRC_ROOT / "video_checkpoints/chA_given_chB_halves"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2, help="clips per GPU (2 x 30 frames uses ~15 GiB)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_every", type=int, default=50, help="epochs between checkpoints")
    p.add_argument("--preview_every", type=int, default=1, help="epochs between preview images")
    p.add_argument("--print_every", type=int, default=50, help="steps between loss printouts (~2.5 min)")
    p.add_argument("--resume", default=None,
                   help="path to a resume_last.pt to continue from, or 'auto' for <save_dir>/resume_last.pt")
    p.add_argument("--restart_lr", type=float, default=None,
                   help="with --resume: start a fresh warmup + cosine schedule over the remaining epochs, "
                        "peaking at this LR, instead of continuing the old schedule (use when raising --epochs)")
    p.add_argument("--init_from", default=None,
                   help="model_*.pt / model_ema_*.pt to start from (weights only; fresh optimizer and schedule)")
    p.add_argument("--offset_noise", type=float, default=0.0,
                   help="std of a per-clip constant added to the training noise (0.1 is typical). stops the model "
                        "reading overall brightness from the signal left at t=999, which makes samples grey")
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def build_model():
    src_unet = build_unet_2d(FRAME_SIZE, channels=1)
    src_unet.load_state_dict(torch.load(CHB_ENCODER_CKPT, map_location="cpu"))
    encoder = ChannelEncoder(src_unet, stop_at_block=STOP_BLOCK, token_dim=TOKEN_DIM)
    unet = build_unet(FRAME_SIZE, channels=1, cross_attention_dim=TOKEN_DIM)
    enable_gradient_checkpointing(unet)   #30 frames don't fit in 24 GB otherwise
    return FrameConditionedUNet(unet, encoder)


@torch.no_grad()
def save_preview(model, ema, trainable, samples, noise_scheduler, device, path):
    '''
    generate chA for the preview clips with the EMA weights. rows per clip, top to bottom:
    chB, real chA, generated chA; clips in order (train clip first, then val clip).
    '''
    ema.store(trainable)
    ema.copy_to(trainable)
    model.eval()

    cond = torch.stack([s["cond"] for s in samples]).to(device)     # [n, 1, T, H, W]
    real = torch.stack([s["target"] for s in samples]).to(device)

    sampler = DDIMScheduler.from_config(noise_scheduler.config)
    sampler.set_timesteps(PREVIEW_STEPS)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = model.encode(cond)                                  #encode chB once, reuse every step
        #fixed seed so previews differ between epochs only because the model changed
        x = torch.randn(real.shape, generator=torch.Generator(device).manual_seed(0), device=device)
        for t in sampler.timesteps:
            noise_pred = model(x, t.to(device), tokens=tokens)
            x = sampler.step(noise_pred.float(), t, x).prev_sample

    model.train()
    ema.restore(trainable)

    frames = slice(None, None, 3)                                    #every 3rd frame -> 10 columns
    rows = []
    for i in range(len(samples)):
        for clip in (cond[i], real[i], x[i]):
            rows.append(clip[:, frames].permute(1, 0, 2, 3))         # [10, 1, H, W]
    grid = torch.cat(rows)
    save_image(make_grid((grid * 0.5 + 0.5).clamp(0, 1), nrow=rows[0].shape[0]), path)


def main():
    args = parse_args()
    local_rank = setup_ddp()
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    is_main = rank == 0
    if is_main:
        os.makedirs(f"{args.save_dir}/samples", exist_ok=True)

    dataset = PairedVideoDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR, clips_per_video=2, split="train")
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4,
                        drop_last=True, pin_memory=True, persistent_workers=True)

    model = build_model().to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]   #whole video UNet + encoder proj
    if args.init_from:
        init = torch.load(args.init_from, map_location=device)
        model.unet.load_state_dict(init["unet"])
        model.encoder.proj.load_state_dict(init["encoder_proj"])
        del init

    resume_path = f"{args.save_dir}/resume_last.pt" if args.resume == "auto" else args.resume
    if args.restart_lr is not None and not resume_path:
        raise ValueError("--restart_lr only makes sense together with --resume")
    resume = torch.load(resume_path, map_location=device) if resume_path else None
    if resume is not None:
        if resume["steps_per_epoch"] != len(loader):
            raise ValueError(f"checkpoint was trained with {resume['steps_per_epoch']} steps/epoch, this run has "
                             f"{len(loader)}; use the same --batch_size and number of GPUs to resume")
        model.unet.load_state_dict(resume["unet"])
        model.encoder.proj.load_state_dict(resume["encoder_proj"])

    ddp_model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)

    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    total_steps = len(loader) * args.epochs
    ema = EMAModel(trainable, decay=0.9999, use_ema_warmup=True, power=0.75)
    ema.to(device)

    #the LR schedule runs from global step sched_start to total_steps. it's 0 unless a run was resumed
    #with --restart_lr, and that start point is saved so later resumes rebuild the same schedule
    start_epoch, global_step = 0, 0
    sched_start, sched_peak = 0, args.lr
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        ema.load_state_dict(resume["ema"])
        ema.to(device)
        start_epoch, global_step = resume["epoch"] + 1, resume["global_step"]
        sched_start, sched_peak = resume.get("sched_start", 0), resume.get("sched_peak", args.lr)
    if args.restart_lr is not None:
        sched_start, sched_peak = global_step, args.restart_lr
        if total_steps - sched_start <= WARMUP_STEPS:
            raise ValueError(f"only {total_steps - sched_start} steps left; raise --epochs to restart the LR schedule")
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] = sched_peak   #the new schedule scales from this peak
    lr_sched = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS,
                                               num_training_steps=total_steps - sched_start)
    if resume is not None and args.restart_lr is None:
        lr_sched.load_state_dict(resume["lr_sched"])
    del resume

    if is_main:
        val_dataset = PairedVideoDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR, clips_per_video=2, split="val")
        preview_samples = [dataset[PREVIEW_TRAIN_CLIP], val_dataset[PREVIEW_VAL_CLIP]]
        log_path = f"{args.save_dir}/loss_log.csv"
        if resume_path and os.path.exists(log_path):
            #drop rows logged after the checkpoint (the partial epoch that crashed) so steps don't repeat
            with open(log_path, newline="") as f:
                rows = [r for r in csv.reader(f)][1:]
            rows = [r for r in rows if int(r[0]) <= global_step]
            log_file = open(log_path, "w", newline="")
            log_writer = csv.writer(log_file)
            log_writer.writerow(["step", "epoch", "loss"])
            log_writer.writerows(rows)
        else:
            log_file = open(log_path, "w", newline="")
            log_writer = csv.writer(log_file)
            log_writer.writerow(["step", "epoch", "loss"])
        print("=" * 60)
        if resume_path:
            print(f"resumed from       : {resume_path} (continuing at epoch {start_epoch}, step {global_step})")
        if args.init_from:
            print(f"initialized from   : {args.init_from}")
        print(f"offset noise       : {args.offset_noise}")
        print(f"LR schedule        : {WARMUP_STEPS}-step warmup to {sched_peak:.1e} at step {sched_start}, "
              f"cosine to 0 at step {total_steps}")
        print(f"GPUs               : {world}")
        print(f"train clips        : {len(dataset)} ({len(dataset) // 2} cells x 2 halves, {dataset.num_frames} frames)")
        print(f"held out           : {len(VAL_RECORDINGS)} recording(s), {len(val_dataset)} clips, previews only")
        print(f"previews           : {preview_samples[0]['cell']} (train), {preview_samples[1]['cell']} (val)")
        print(f"batch              : {args.batch_size} per GPU, {args.batch_size * world} total")
        print(f"steps / epoch      : {len(loader)}")
        print(f"total steps        : {total_steps}")
        print(f"trainable params   : {sum(p.numel() for p in trainable) / 1e6:.1f}M")
        print(f"checkpoints every  : {args.save_every} epochs -> {args.save_dir}")
        print("=" * 60)

    def save_checkpoint(tag):
        state = lambda: {"unet": model.unet.state_dict(),
                         "encoder_proj": model.encoder.proj.state_dict(),
                         "chB_encoder_ckpt": str(CHB_ENCODER_CKPT),
                         "stop_block": STOP_BLOCK, "token_dim": TOKEN_DIM}
        torch.save(state(), f"{args.save_dir}/model_{tag}.pt")
        ema.store(trainable)
        ema.copy_to(trainable)
        torch.save(state(), f"{args.save_dir}/model_ema_{tag}.pt")
        ema.restore(trainable)

    def save_resume_state(epoch):
        #everything needed to continue exactly where this epoch ended; overwritten every epoch
        path = f"{args.save_dir}/resume_last.pt"
        torch.save({"unet": model.unet.state_dict(),
                    "encoder_proj": model.encoder.proj.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_sched": lr_sched.state_dict(),
                    "ema": ema.state_dict(),
                    "epoch": epoch, "global_step": global_step,
                    "sched_start": sched_start, "sched_peak": sched_peak,
                    "steps_per_epoch": len(loader)}, path + ".tmp")
        os.replace(path + ".tmp", path)   #a crash mid-write leaves the previous file intact

    start_step = global_step
    window_loss = torch.zeros((), device=device)   #loss summed since the last printout
    train_start = time.time()
    progress = tqdm(total=total_steps, initial=global_step, desc="training", dynamic_ncols=True,
                    disable=not is_main)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            x = batch["target"].to(device, non_blocking=True)       # [B, 1, 30, 128, 128] chA
            cond = batch["cond"].to(device, non_blocking=True)      # same frames of chB
            noise = torch.randn_like(x)
            if args.offset_noise > 0:
                #same shift for every frame and pixel of a clip, drawn fresh per clip
                noise = noise + args.offset_noise * torch.randn(x.shape[0], x.shape[1], 1, 1, 1, device=device)
            t = torch.randint(0, NUM_TRAIN_TIMESTEPS, (x.shape[0],), device=device).long()
            noisy = noise_scheduler.add_noise(x, noise, t)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                noise_pred = ddp_model(noisy, t, cond=cond)
            loss = F.mse_loss(noise_pred.float(), noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            lr_sched.step()
            ema.step(trainable)

            global_step += 1
            window_loss += loss.detach()
            if is_main:
                log_writer.writerow([global_step, epoch, loss.item()])
                progress.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}",
                                     lr=f"{lr_sched.get_last_lr()[0]:.2e}")
            progress.update(1)

            if global_step % args.print_every == 0:
                #mean over the window and over all GPUs, so it's much less noisy than one step's loss
                dist.all_reduce(window_loss)
                if is_main:
                    elapsed = time.time() - train_start
                    eta = (total_steps - global_step) * elapsed / (global_step - start_step)
                    tqdm.write(f"  step {global_step}/{total_steps} | epoch {epoch} | "
                               f"loss {window_loss.item() / (args.print_every * world):.4f} | "
                               f"elapsed {elapsed / 3600:.2f}h | ETA {eta / 3600:.2f}h")
                    log_file.flush()
                window_loss.zero_()

        last = epoch == args.epochs - 1
        if is_main:
            log_file.flush()
            save_resume_state(epoch)
            if epoch % args.save_every == 0 or last:
                save_checkpoint(f"epoch{epoch}")
                tqdm.write(f"  [checkpoint] epoch {epoch}, {(time.time() - train_start) / 3600:.2f}h elapsed")
            if epoch % args.preview_every == 0 or last:
                save_preview(model, ema, trainable, preview_samples, noise_scheduler, device,
                             f"{args.save_dir}/samples/epoch{epoch}.png")
        dist.barrier()

    progress.close()
    if is_main:
        log_file.close()
        print(f"done. {global_step - start_step} steps in {(time.time() - train_start) / 3600:.2f} hours.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
