'''
training a single channel diffusion model by reusing weights from 2-channel diffusion model
warm-start: copy the whole 2ch unet, slice the in/out convs down to 1 channel, then fine-tune.
'''
import os, csv, time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel

from diffusion_model import build_unet

DEVICE      = torch.device("cuda:9")
CH_DIR      = "data/singlecell_chB_split/train"   #training single channel
SAVE_DIR    = "diffusion_checkpoints/ddpm_chB_128_no_preweights"
IMG_SIZE    = 128


TARGET_CHANNEL = 1        #0 = chA, 1 = chB
BATCH_SIZE     = 32
NUM_EPOCHS     = 80        #warm start converges faster than the 200 the 2ch needed from scratch
LR             = 5e-5      #lower than scratch LR since we're fine-tuning a good init
NUM_TRAIN_TIMESTEPS = 1000
SAVE_EVERY     = 5
PRINT_EVERY    = 200

#L2-SP(?)- weights penalized furthey they stray from the pretrained ones?
USE_L2SP    = False
# L2SP_WEIGHT = 1e-4

os.makedirs(f"{SAVE_DIR}/samples", exist_ok=True)


# def port_2ch_to_1ch(pretrained_path, target_channel, img_size):
#     sd_2ch = torch.load(pretrained_path, map_location="cpu")
#     model  = build_unet(img_size, channels=1)
#     sd_1ch = model.state_dict()

#     #only these 3 tensors depend on channel count, everything else copies straight over
#     for k, v in sd_2ch.items():
#         if k == "conv_in.weight":
#             #slice, not average, for the specific channel wts
#             sd_1ch[k] = v[:, target_channel:target_channel+1].clone()
#         elif k == "conv_out.weight":
#             sd_1ch[k] = v[target_channel:target_channel+1].clone()
#         elif k == "conv_out.bias":
#             sd_1ch[k] = v[target_channel:target_channel+1].clone()
#         else:
#             sd_1ch[k] = v.clone()

#     model.load_state_dict(sd_1ch)
#     return model


class singleChannelDataset(Dataset):
    def __init__(self, ch_dir):
        super().__init__()
        self.files = sorted([f for f in Path(ch_dir).glob('*.png')])
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert('L')
        return self.transform(img)


dataset = singleChannelDataset(CH_DIR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=4, drop_last=True)

# model     = port_2ch_to_1ch(PRETRAINED_MODEL, TARGET_CHANNEL, IMG_SIZE).to(DEVICE)
model = build_unet(IMG_SIZE, channels=1).to(DEVICE) 
scheduler = DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)

if USE_L2SP:
    ref_params = {k: v.detach().clone() for k, v in model.named_parameters()}

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_sched  = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=200,
    num_training_steps=len(loader) * NUM_EPOCHS,
)

ema = EMAModel(model.parameters(), decay=0.9999, use_ema_warmup=True, power=0.75)
ema.to(DEVICE)

log_file   = open(f"{SAVE_DIR}/loss_log.csv", "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["step", "epoch", "loss"])

steps_per_epoch = len(loader)
total_steps     = steps_per_epoch * NUM_EPOCHS

print(f"device            : {DEVICE}")
# print(f"warm-started from  : {PRETRAINED_MODEL}")
print(f"target channel     : {TARGET_CHANNEL}  ({'chA' if TARGET_CHANNEL==0 else 'chB'})")
# print(f"L2-SP              : {USE_L2SP}  (weight {L2SP_WEIGHT})")
print(f"dataset size       : {len(dataset)} images")
print(f"total epochs       : {NUM_EPOCHS}  |  total steps : {total_steps}")


@torch.no_grad()
def sample_preview(n, epoch):
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    model.eval()

    scheduler.set_timesteps(NUM_TRAIN_TIMESTEPS)
    x = torch.randn(n, 1, IMG_SIZE, IMG_SIZE, device=DEVICE)
    for t in tqdm(scheduler.timesteps, desc=f"sampling epoch {epoch}", leave=False):
        noise_pred = model(x, t).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

    model.train()
    ema.restore(model.parameters())

    x = (x * 0.5 + 0.5).clamp(0, 1)
    save_image(make_grid(x, nrow=n), f"{SAVE_DIR}/samples/epoch{epoch}.png")


global_step = 0
train_start = time.time()
#tqdm tracks it/s and elapsed/remaining on its own, just feed it total_steps up front
progress = tqdm(total=total_steps, desc="training", dynamic_ncols=True)

'''
code for resuing if it fails/reboot
'''

RESUME_FROM = None

start_epoch = 0
global_step = 0
if RESUME_FROM and os.path.exists(RESUME_FROM):
    ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_sched.load_state_dict(ckpt["lr_sched"])
    ema.load_state_dict(ckpt["ema"])
    start_epoch  = ckpt["epoch"] + 1
    global_step  = ckpt["global_step"]
    print(f"resumed from epoch {ckpt['epoch']}, step {global_step}")

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    tqdm.write(f"\n--- starting epoch {epoch}/{NUM_EPOCHS} ---")

    for x in loader:
        x = x.to(DEVICE)
        noise = torch.randn_like(x)
        bsz   = x.shape[0]
        timesteps = torch.randint(0, NUM_TRAIN_TIMESTEPS, (bsz,),
                                  device=DEVICE).long()

        noisy_x    = scheduler.add_noise(x, noise, timesteps)
        noise_pred = model(noisy_x, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)

        if USE_L2SP:
            l2sp = sum(((p - ref_params[k])**2).sum()
                       for k, p in model.named_parameters() if k in ref_params)
            loss = loss + L2SP_WEIGHT * l2sp

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_sched.step()
        ema.step(model.parameters())

        global_step += 1

        #tqdm's own format_dict has rate/remaining already computed, just pull from it
        rate      = progress.format_dict["rate"] or 0
        eta_hours = ((total_steps - global_step) / rate / 3600) if rate else float("inf")
        progress.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}",
                             lr=f"{lr_sched.get_last_lr()[0]:.2e}",
                             eta_h=f"{eta_hours:.2f}")
        progress.update(1)
        log_writer.writerow([global_step, epoch, loss.item()])

        if global_step % PRINT_EVERY == 0:
            log_file.flush()

    if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS - 1:
        sample_preview(8, epoch)
        torch.save(model.state_dict(), f"{SAVE_DIR}/unet_epoch{epoch}.pt")
        ema.store(model.parameters()); ema.copy_to(model.parameters())
        torch.save(model.state_dict(), f"{SAVE_DIR}/unet_ema_epoch{epoch}.pt")
        ema.restore(model.parameters())

        #resume checkpoint - separate from the inspection snapshots above,
        #this one gets overwritten every time, only ever need the latest
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_sched": lr_sched.state_dict(),
            "ema": ema.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }, f"{SAVE_DIR}/resume_latest.pt")

        tqdm.write(f"  [checkpoint] saved model + samples at epoch {epoch}")

progress.close()
log_file.close()
total_hours = (time.time() - train_start) / 3600
print(f"\ndone. trained {total_steps} steps in {total_hours:.2f} hours.")