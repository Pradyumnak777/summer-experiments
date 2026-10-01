'''
fine-tune the chA-given-chB video model from its epoch-250 EMA weights with offset noise.

why: with the default DDPM schedule, t=999 still holds 0.64% of the real clip. averaged over a
whole 30-frame clip that reveals the clip's overall brightness, so the model learned to read it
instead of generating it, and samples from pure noise come out grey with low contrast. offset
noise (a random per-clip constant added to the training noise) hides that clue, so the model
has to learn to set the brightness itself.

this runs train.py with fine-tuning defaults: same train/val split and data (both 30-frame
halves), weights from --init_from, a fresh optimizer and a fresh warmup + cosine schedule over
the 50 epochs, fixed-seed previews every epoch. any train.py argument can be overridden, e.g.
--epochs 80 or --offset_noise 0.05, and --resume auto continues an interrupted fine-tune.

run from src/:
    torchrun --nproc_per_node=4 video_diffusion/finetune.py
while another torchrun job is running, pick free GPUs and a different port:
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port 29501 video_diffusion/finetune.py
'''
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))
from video_diffusion import train

CKPT_DIR = SRC_ROOT / "video_checkpoints/chA_given_chB_halves"

FINETUNE_DEFAULTS = [
    "--init_from", str(CKPT_DIR / "model_ema_epoch250.pt"),
    "--save_dir", str(SRC_ROOT / "video_finetune_checkpoints/offset_noise_from_ep250"),
    "--epochs", "50",
    "--lr", "5e-5",
    "--offset_noise", "0.1",
    "--save_every", "25",
]

if __name__ == "__main__":
    #defaults go first so anything given on the command line overrides them
    sys.argv[1:1] = FINETUNE_DEFAULTS
    train.main()
