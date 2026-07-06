import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data_util import PairedMicroscopyDataset
from main import channelEncode, hook_unet, channelAttnProcessor
import os
import csv


SAVE_DIR = "checkpoints_new/channel_cond_v2_redo"   
os.makedirs(SAVE_DIR, exist_ok=True)
#conf
DEVICE       = torch.device("cuda:9")  #cuda:9 only! its free
LORA_PATH    = "checkpoints_new/sd21_cell_chB"  #actual saved path
BASE_MODEL   = "Manojb/stable-diffusion-2-1-base"
GREEN_DIR    = "data/singlecell_chA_split/train"
MAGENTA_DIR  = "data/singlecell_chB_split/train"
FIXED_PROMPT = "an image of condensate in a microscopic cell"
BATCH_SIZE   = 16
NUM_EPOCHS   = 15
LR           = 1e-4

#load everything via pipeline, cleanest way to get all components + lora together
pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(LORA_PATH)
unet         = pipe.unet.to(DEVICE)
vae          = pipe.vae.to(DEVICE)
text_encoder = pipe.text_encoder.to(DEVICE)
tokenizer    = pipe.tokenizer
scheduler    = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

#freeze everything first, trainable parts get added on top after
for model in [unet, vae, text_encoder]:
    for param in model.parameters():
        param.requires_grad = False

#hook unet AFTER freezing so new processor K,V projections stay trainable
channel_encoder = channelEncode(vae).to(DEVICE, dtype=torch.bfloat16)

hook_unet(unet)
unet.to(DEVICE)
for module in unet.modules():
    if isinstance(module, channelAttnProcessor):
        module.to(DEVICE, dtype=torch.bfloat16)   # to_k_img/to_v_img default to fp32 otherwise

#only optimize the new stuf- channel encoder projection + image attn K,V
trainable_params = list(channel_encoder.projection.parameters())
for module in unet.modules():
    if isinstance(module, channelAttnProcessor):
        trainable_params += list(module.parameters())

optimizer = AdamW(trainable_params, lr=LR)

dataset = PairedMicroscopyDataset(GREEN_DIR, MAGENTA_DIR, size=256)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#loss logger
log_file = open(f"{SAVE_DIR}/loss_log.csv", "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["global_step", "epoch", "loss"])
ema = None
global_step = 0

#compute text embeddings once outside loop, prompt never changes so no point redoing every step
with torch.no_grad():
    text_inputs = tokenizer(
        [FIXED_PROMPT],
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True
    ).input_ids.to(DEVICE)
    text_embeddings = text_encoder(text_inputs)[0]  #[1, 77, 768]
    
    null_inputs = tokenizer(
        [""],
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True
    ).input_ids.to(DEVICE)
    null_embeddings = text_encoder(null_inputs)[0]  #[1, 77, 768]

for epoch in range(NUM_EPOCHS):
    for step, (green_imgs, magenta_imgs) in enumerate(loader):
        green_imgs   = green_imgs.to(DEVICE, dtype=torch.bfloat16)
        magenta_imgs = magenta_imgs.to(DEVICE, dtype=torch.bfloat16)
        B = green_imgs.shape[0]

        #encode magenta-> clean latent, scale it (SD always expects scaled latents)
        with torch.no_grad():
            magenta_latents = vae.encode(magenta_imgs).latent_dist.sample()
            magenta_latents = magenta_latents * vae.config.scaling_factor

        #random noise + random timestep for forward diffusion
        noise     = torch.randn_like(magenta_latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=DEVICE).long()

        #corrupt the clean latent with noise
        noisy_latents = scheduler.add_noise(magenta_latents, noise, timesteps)

        #green channel -> image tokens [B, 64, 768]
        image_tokens = channel_encoder(green_imgs)
        
        '''
        drop image tokens sometimes, so that CFG can be used..
        '''
        # if torch.rand(1).item() < 0.1:
        #     image_tokens = torch.zeros_like(image_tokens)
        keep = (torch.rand(B, 1, 1, device=DEVICE) >= 0.1)
        image_tokens = image_tokens * keep

        #inject image tokens into every attn processor before unet sees the batch
        for module in unet.modules():
            if isinstance(module, channelAttnProcessor):
                module.image_tokens = image_tokens

        #expand text embeddings to match batch size
        text_emb = text_embeddings.expand(B, -1, -1)  #[B, 77, 768]
        
        # #null ocnditioning 10% of the times..
        # if torch.rand(1).item() < 0.1:
        #     text_emb = null_embeddings.expand(B, -1, -1)

        #unet predicts the noise we added
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #logging!
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        log_writer.writerow([global_step, epoch, loss.item()])
        global_step += 1
        if step % 100 == 0:
            log_file.flush()



        if step % 10 == 0:
            print(f"epoch {epoch} | step {step} | loss {loss.item():.4f} | ema {ema:.4f}")

    if epoch % 5 == 0:
        torch.save({
            "channel_encoder": channel_encoder.state_dict(),
            "attn_processors": {
                name: module.state_dict()
                for name, module in unet.named_modules()
                if isinstance(module, channelAttnProcessor)
            }
        }, f"{SAVE_DIR}/channel_cond_cell_epoch{epoch}_new.pt")

#save channel encoder + new attn processors
torch.save({
    "channel_encoder": channel_encoder.state_dict(),
    "attn_processors": {
        name: module.state_dict()
        for name, module in unet.named_modules()
        if isinstance(module, channelAttnProcessor)
    }
}, f"{SAVE_DIR}/channel_cond_cell_epoch{epoch}_new.pt")

log_file.close()