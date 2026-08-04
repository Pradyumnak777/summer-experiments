'''
train an unconditional 2-channel DDPM from scratch on the cell crops.
objective is just: predict the noise added to a clean image. no KL, no
posterior collapse, no PoE - simpler than the DMVAE.
'''
import os, csv, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel

from data_utils import twoChannelDataset
from diffusion_model import build_unet

DEVICE      = torch.device("cuda:9")
CHA_DIR     = "data/singlecell_chA_split/train"
CHB_DIR     = "data/singlecell_chB_split/train"
SAVE_DIR    = "diffusion_checkpoints/ddpm_2ch_128_masked"
IMG_SIZE    = 128
BATCH_SIZE  = 32
NUM_EPOCHS  = 100
LR          = 1e-4
NUM_TRAIN_TIMESTEPS = 1000
SAVE_EVERY  = 10          # epochs between checkpoints + sample previews
PRINT_EVERY = 200         # steps between logged print lines
os.makedirs(f"{SAVE_DIR}/samples", exist_ok=True)

dataset = twoChannelDataset(CHA_DIR, CHB_DIR, mask_dir="data/singlecell_mask")
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=4, drop_last=True)

model     = build_unet(IMG_SIZE, channels=2).to(DEVICE)
scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_sched  = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=500,
    num_training_steps=len(loader) * NUM_EPOCHS,
)

#EMA weights -> materially better samples for diffusion, standard practice
ema = EMAModel(model.parameters(), decay=0.9999, use_ema_warmup=True, power=0.75)
ema.to(DEVICE)

log_file   = open(f"{SAVE_DIR}/loss_log.csv", "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["step", "epoch", "loss"])

steps_per_epoch = len(loader)
total_steps     = steps_per_epoch * NUM_EPOCHS

print("="*60)
print(f"device            : {DEVICE}")
print(f"dataset size       : {len(dataset)} images")
print(f"batch size         : {BATCH_SIZE}")
print(f"steps / epoch      : {steps_per_epoch}")
print(f"total epochs       : {NUM_EPOCHS}")
print(f"total steps        : {total_steps}")
print(f"checkpoints every  : {SAVE_EVERY} epochs -> {SAVE_DIR}")
print("="*60)


@torch.no_grad()
def sample_preview(n, epoch):
    #generate n images with the EMA weights, then restore training weights
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    model.eval()

    scheduler.set_timesteps(NUM_TRAIN_TIMESTEPS)
    x = torch.randn(n, 2, IMG_SIZE, IMG_SIZE, device=DEVICE)
    for t in tqdm(scheduler.timesteps, desc=f"sampling epoch {epoch}", leave=False):
        noise_pred = model(x, t).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

    model.train()
    ema.restore(model.parameters())

    x = (x * 0.5 + 0.5).clamp(0, 1)
    for ch, name in [(0, "chA"), (1, "chB")]:
        save_image(make_grid(x[:, ch:ch+1], nrow=n),
                   f"{SAVE_DIR}/samples/epoch{epoch}_{name}.png")


global_step = 0
train_start = time.time()
#single bar spanning the WHOLE run (all epochs), not recreated per epoch
progress = tqdm(total=total_steps, desc="training", dynamic_ncols=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    tqdm.write(f"\n--- starting epoch {epoch}/{NUM_EPOCHS} ---")

    for x,_ in loader:
        x = x.to(DEVICE)                              # [B, 2, 128, 128] in [-1,1]
        noise = torch.randn_like(x)
        bsz   = x.shape[0]
        timesteps = torch.randint(0, NUM_TRAIN_TIMESTEPS, (bsz,),
                                  device=DEVICE).long()

        noisy_x    = scheduler.add_noise(x, noise, timesteps)
        noise_pred = model(noisy_x, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)          # predict the noise

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_sched.step()
        ema.step(model.parameters())

        global_step += 1
        progress.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}",
                             lr=f"{lr_sched.get_last_lr()[0]:.2e}")
        progress.update(1)

        log_writer.writerow([global_step, epoch, loss.item()])

        if global_step % PRINT_EVERY == 0:
            elapsed   = time.time() - train_start
            rate      = global_step / elapsed
            eta_hours = (total_steps - global_step) / rate / 3600
            tqdm.write(f"  step {global_step}/{total_steps} | epoch {epoch} | "
                      f"loss {loss.item():.4f} | {rate:.2f} it/s | "
                      f"elapsed {elapsed/3600:.2f}h | ETA {eta_hours:.2f}h")
            log_file.flush()

    if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS - 1:
        sample_preview(8, epoch)
        torch.save(model.state_dict(), f"{SAVE_DIR}/unet_epoch{epoch}.pt")
        #also save the EMA weights
        ema.store(model.parameters()); ema.copy_to(model.parameters())
        torch.save(model.state_dict(), f"{SAVE_DIR}/unet_ema_epoch{epoch}.pt")
        ema.restore(model.parameters())
        tqdm.write(f"  [checkpoint] saved model + samples at epoch {epoch}")

progress.close()
log_file.close()
total_hours = (time.time() - train_start) / 3600
print(f"\ndone. trained {total_steps} steps in {total_hours:.2f} hours.")