'''
train a 2D diffusion model that generates chA single-cell crops conditioned on the matching chB
crop, through the UNet's built-in cross-attention (conditioned_diffusion_2d/model.py).

vanilla setup, matching the 2D DDPMs: default DDPM schedule, noise-prediction MSE, trained from
scratch, AdamW with warmup + cosine, EMA weights for previews. chB is tokenized by the frozen chB
DDPM trunk plus a trainable 1x1 proj. the held-out val recording is the same one as the video model.

each epoch saves a preview (rows: train chB, real chA, generated chA, then the same for val) and
appends brightness stats of generated vs real cells to preview_stats.csv. resume_last.pt is
overwritten every epoch; continue an interrupted run with --resume auto.

run from src/:
    torchrun --nproc_per_node=4 conditioned_diffusion_2d/train.py
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

import debugpy
debugpy.listen(("127.0.0.1", 5678))
print("waiting for debugger to attach...")
debugpy.wait_for_client()
print("debugger attached!")

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))
from conditioned_diffusion_2d.data_utils import PairedCropDataset
from conditioned_diffusion_2d.model import build_model

CHB_ENCODER_CKPT = SRC_ROOT / "diffusion_checkpoints/ddpm_chB_128_masked/unet_ema_epoch40.pt"
TOKEN_DIM   = 256
STOP_BLOCK  = 3        #tap the chB trunk after down_blocks[3] -> 8x8 = 64 tokens
NUM_TRAIN_TIMESTEPS = 1000
WARMUP_STEPS  = 500
PREVIEW_STEPS = 50     #DDIM steps for previews
PREVIEW_N     = 8      #images per split in each preview


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", default=str(SRC_ROOT / "conditioned_diffusion_2d_checkpoints_test/chA_given_chB"))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32, help="images per GPU (16 uses ~10 GiB)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_every", type=int, default=10, help="epochs between checkpoints")
    p.add_argument("--preview_every", type=int, default=1, help="epochs between preview images")
    p.add_argument("--print_every", type=int, default=200, help="steps between loss printouts")
    p.add_argument("--resume", default=None,
                   help="path to a resume_last.pt to continue from, or 'auto' for <save_dir>/resume_last.pt")
    return p.parse_args()


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def pick_preview(dataset, n):
    '''n crops spread evenly over the (name-sorted) dataset, so they come from different cells.'''
    step = len(dataset) // n
    return [dataset[i * step + step // 2] for i in range(n)]


@torch.no_grad()
def save_preview(model, ema, trainable, groups, noise_scheduler, device, path):
    '''
    generate chA for each preview group with the EMA weights, from fixed-seed noise.
    returns per-group brightness stats: mean of generated / real inside the cell, and of the background.
    '''
    ema.store(trainable)
    ema.copy_to(trainable)
    model.eval()

    rows, stats = [], {}
    for name, samples in groups.items():
        cond = torch.stack([s["cond"] for s in samples]).to(device)
        real = torch.stack([s["target"] for s in samples]).to(device)
        cell = torch.stack([s["mask"] for s in samples]).to(device).bool()
        sampler = DDIMScheduler.from_config(noise_scheduler.config)
        sampler.set_timesteps(PREVIEW_STEPS)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = model.encode(cond)
            x = torch.randn(real.shape, generator=torch.Generator(device).manual_seed(0), device=device)
            for t in sampler.timesteps:
                noise_pred = model(x, t.to(device), tokens=tokens)
                x = sampler.step(noise_pred.float(), t, x).prev_sample
        rows += [cond, real, x]
        stats[name] = (x[cell].mean().item(), real[cell].mean().item(), x[~cell].mean().item())

    model.train()
    ema.restore(trainable)
    grid = torch.cat(rows)
    save_image(make_grid((grid * 0.5 + 0.5).clamp(0, 1), nrow=PREVIEW_N), path)
    return stats


def main():
    args = parse_args()
    local_rank = setup_ddp()
    
    #setting up debugpy
    if local_rank == 0 and os.environ.get("DEBUG"):
        import debugpy
        debugpy.listen(("127.0.0.1", 5678))
        print("waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("debugger attached!")

    
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    is_main = rank == 0
    if is_main:
        os.makedirs(f"{args.save_dir}/samples", exist_ok=True)

    dataset = PairedCropDataset(split="train")
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=6,
                        drop_last=True, pin_memory=True, persistent_workers=True)

    model = build_model(CHB_ENCODER_CKPT, stop_block=STOP_BLOCK, token_dim=TOKEN_DIM).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]   #whole UNet + encoder proj

    resume_path = f"{args.save_dir}/resume_last.pt" if args.resume == "auto" else args.resume
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
    lr_sched = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS,
                                               num_training_steps=total_steps)
    ema = EMAModel(trainable, decay=0.9999, use_ema_warmup=True, power=0.75)
    ema.to(device)

    start_epoch, global_step = 0, 0
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        lr_sched.load_state_dict(resume["lr_sched"])
        ema.load_state_dict(resume["ema"])
        ema.to(device)
        start_epoch, global_step = resume["epoch"] + 1, resume["global_step"]
    del resume

    if is_main:
        val_dataset = PairedCropDataset(split="val")
        preview_groups = {"train": pick_preview(dataset, PREVIEW_N), "val": pick_preview(val_dataset, PREVIEW_N)}
        log_path, stats_path = f"{args.save_dir}/loss_log.csv", f"{args.save_dir}/preview_stats.csv"
        if resume_path:
            #drop rows logged after the checkpoint (a partial epoch) so nothing repeats
            for path, key in ((log_path, 0), (stats_path, 0)):
                if os.path.exists(path):
                    with open(path, newline="") as f:
                        rows = list(csv.reader(f))
                    limit = global_step if path == log_path else start_epoch - 1
                    with open(path, "w", newline="") as f:
                        csv.writer(f).writerows([rows[0]] + [r for r in rows[1:] if int(r[key]) <= limit])
        else:
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "epoch", "loss"])
            with open(stats_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "split", "gen_cell_mean", "real_cell_mean", "gen_background_mean"])
        log_file = open(log_path, "a", newline="")
        log_writer = csv.writer(log_file)
        print("=" * 60)
        if resume_path:
            print(f"resumed from       : {resume_path} (continuing at epoch {start_epoch}, step {global_step})")
        print(f"GPUs               : {world}")
        print(f"train crops        : {len(dataset)} ({dataset.n_empty} empty-mask crops dropped)")
        print(f"val crops          : {len(val_dataset)} (held-out recording, previews only)")
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
        path = f"{args.save_dir}/resume_last.pt"
        torch.save({"unet": model.unet.state_dict(),
                    "encoder_proj": model.encoder.proj.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_sched": lr_sched.state_dict(),
                    "ema": ema.state_dict(),
                    "epoch": epoch, "global_step": global_step,
                    "steps_per_epoch": len(loader)}, path + ".tmp")
        os.replace(path + ".tmp", path)   #a crash mid-write leaves the previous file intact

    start_step = global_step
    window_loss = torch.zeros((), device=device)
    train_start = time.time()
    progress = tqdm(total=total_steps, initial=global_step, desc="training", dynamic_ncols=True,
                    disable=not is_main)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            x = batch["target"].to(device, non_blocking=True)       # [B, 1, 128, 128] chA
            cond = batch["cond"].to(device, non_blocking=True)      # matching chB crop
            noise = torch.randn_like(x)
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
                tqdm.write(f"  [checkpoint] epoch {epoch}")
            if epoch % args.preview_every == 0 or last:
                stats = save_preview(model, ema, trainable, preview_groups, noise_scheduler, device,
                                     f"{args.save_dir}/samples/epoch{epoch}.png")
                with open(f"{args.save_dir}/preview_stats.csv", "a", newline="") as f:
                    w = csv.writer(f)
                    for split, (gen, real, bg) in stats.items():
                        w.writerow([epoch, split, f"{gen:.4f}", f"{real:.4f}", f"{bg:.4f}"])
                tqdm.write("  [preview] epoch {} | ".format(epoch) + " | ".join(
                    f"{s}: cell gen {g:+.2f} real {r:+.2f}, background {b:+.2f}" for s, (g, r, b) in stats.items()))
        dist.barrier()

    progress.close()
    if is_main:
        log_file.close()
        print(f"done. {global_step - start_step} steps in {(time.time() - train_start) / 3600:.2f} hours.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
