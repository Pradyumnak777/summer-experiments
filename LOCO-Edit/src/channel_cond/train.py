import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data_util import PairedMicroscopyDataset
from main import channelEncode, hook_unet, channelAttnProcessor

#conf
DEVICE       = torch.device("cuda")  #cuda:9 only! its free
LORA_PATH    = "checkpoints_new/sd21_magenta"  #actual saved path
BASE_MODEL   = "Manojb/stable-diffusion-2-1-base"
GREEN_DIR    = "data/microscopy_lora_green"
MAGENTA_DIR  = "data/microscopy_lora_magenta"
FIXED_PROMPT = "an image of magenta cells in fluorescent microscopy"
BATCH_SIZE   = 2
NUM_EPOCHS   = 20
LR           = 1e-4

#load everything via pipeline, cleanest way to get all components + lora together
pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
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
hook_unet(unet)
unet.to(DEVICE)
channel_encoder = channelEncode(vae).to(DEVICE)

#only optimize the new stuf- channel encoder projection + image attn K,V
trainable_params = list(channel_encoder.projection.parameters())
for module in unet.modules():
    if isinstance(module, channelAttnProcessor):
        trainable_params += list(module.parameters())

optimizer = AdamW(trainable_params, lr=LR)

dataset = PairedMicroscopyDataset(GREEN_DIR, MAGENTA_DIR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

for epoch in range(NUM_EPOCHS):
    for step, (green_imgs, magenta_imgs) in enumerate(loader):
        green_imgs   = green_imgs.to(DEVICE)
        magenta_imgs = magenta_imgs.to(DEVICE)
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

        #inject image tokens into every attn processor before unet sees the batch
        for module in unet.modules():
            if isinstance(module, channelAttnProcessor):
                module.image_tokens = image_tokens

        #expand text embeddings to match batch size
        text_emb = text_embeddings.expand(B, -1, -1)  #[B, 77, 768]
        
        #randomly drop text 50% of the time so model cant just cheat with text
        if torch.rand(1).item() < 0.5:
            text_emb = torch.zeros_like(text_emb)  #null conditioning

        #unet predicts the noise we added
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"epoch {epoch} | step {step} | loss {loss.item():.4f}")

#save channel encoder + new attn processors
torch.save({
    "channel_encoder": channel_encoder.state_dict(),
    "attn_processors": {
        name: module.state_dict()
        for name, module in unet.named_modules()
        if isinstance(module, channelAttnProcessor)
    }
}, f"checkpoints_new/channel_cond_epoch{epoch}.pt")