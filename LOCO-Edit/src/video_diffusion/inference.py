'''
generate chA for one 30-frame chB clip with a trained video model and save it as a 7 fps mp4.

pick a cell from the train or val split (the same split training used) and a half
(0 = frames 0-29, 1 = frames 30-59). only chB goes into the model; generation starts from
noise, exactly like the training previews. the real chA clip is loaded only to build a
side-by-side comparison video (chB | real chA | generated chA).

run from src/:
    python video_diffusion/inference.py --split val --list              #show cells and their indices
    python video_diffusion/inference.py --split val --cell 0 --half 0
    python video_diffusion/inference.py --split train --cell 12 --half 1 --ckpt <path to model_ema_epochN.pt>
'''
import argparse, re, subprocess, sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, DDIMScheduler

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))
from video_diffusion.data_utils import PairedVideoDataset, CHA_DIR, CHB_DIR, MASK_DIR
from video_diffusion.model import build_unet
from video_diffusion.cross_attn import FrameConditionedUNet
from video_diffusion.train import FRAME_SIZE, NUM_TRAIN_TIMESTEPS
from VAE_disent.diffusion_model import build_unet as build_unet_2d
from cross_attn_modules import ChannelEncoder

CKPT_DIR = SRC_ROOT / "video_checkpoints/chA_given_chB_halves"
FPS = 7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--cell", default="0", help="index within the split (see --list) or the full cell id")
    p.add_argument("--half", type=int, choices=[0, 1], default=0, help="0 = frames 0-29, 1 = frames 30-59")
    p.add_argument("--ckpt", default=None, help="model_ema_epochN.pt to use (default: highest epoch in CKPT_DIR)")
    p.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=str(SRC_ROOT / "video_diffusion/inference_outputs"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--list", action="store_true", help="print the cells in --split and exit")
    return p.parse_args()


def latest_ema_ckpt():
    ckpts = list(CKPT_DIR.glob("model_ema_epoch*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no model_ema_epoch*.pt in {CKPT_DIR}")
    return max(ckpts, key=lambda p: int(re.search(r"epoch(\d+)", p.name).group(1)))


def load_model(path, device):
    '''rebuild the model from the settings stored in the checkpoint.'''
    state = torch.load(path, map_location="cpu")
    src_unet = build_unet_2d(FRAME_SIZE, channels=1)
    src_unet.load_state_dict(torch.load(state["chB_encoder_ckpt"], map_location="cpu"))
    encoder = ChannelEncoder(src_unet, stop_at_block=state["stop_block"], token_dim=state["token_dim"])
    encoder.proj.load_state_dict(state["encoder_proj"])
    unet = build_unet(FRAME_SIZE, channels=1, cross_attention_dim=state["token_dim"])
    unet.load_state_dict(state["unet"])
    return FrameConditionedUNet(unet, encoder).to(device).eval()


@torch.no_grad()
def generate(model, cond, steps, seed):
    '''cond: chB clip [1, 1, T, H, W] in [-1, 1] -> generated chA clip, same shape.'''
    sampler = DDIMScheduler.from_config(DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS).config)
    sampler.set_timesteps(steps)
    generator = torch.Generator(cond.device).manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = model.encode(cond)                     #chB encoded once, reused every step
        x = torch.randn(cond.shape, generator=generator, device=cond.device)
        for t in sampler.timesteps:
            noise_pred = model(x, t.to(cond.device), tokens=tokens)
            x = sampler.step(noise_pred.float(), t, x).prev_sample
    return x


def to_uint8(clip):
    '''[1, T, H, W] in [-1, 1] -> uint8 [T, H, W].'''
    return ((clip[0].clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()


def write_mp4(frames, path):
    '''uint8 [T, H, W] -> H.264 mp4 at FPS, same codec and frame rate as the source videos.'''
    T, H, W = frames.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def main():
    args = parse_args()
    dataset = PairedVideoDataset(CHA_DIR, CHB_DIR, mask_dir=MASK_DIR, clips_per_video=2, split=args.split)

    if args.list:
        for i, cell in enumerate(dataset.cells):
            print(f"{i:4d}  {cell}")
        return

    cell_idx = int(args.cell) if args.cell.isdigit() else dataset.cells.index(args.cell)
    sample = dataset[cell_idx * dataset.clips_per_video + args.half]
    ckpt = Path(args.ckpt) if args.ckpt else latest_ema_ckpt()
    epoch = re.search(r"epoch(\d+)", ckpt.name).group(1)
    print(f"checkpoint : {ckpt}")
    print(f"clip       : {args.split} cell {cell_idx} ({sample['cell']}), "
          f"frames {sample['start']}-{sample['start'] + dataset.num_frames - 1}")

    device = torch.device(args.device)
    model = load_model(ckpt, device)
    cond = sample["cond"].unsqueeze(0).to(device)       # [1, 1, 30, 128, 128]
    pred = generate(model, cond, args.steps, args.seed)[0]

    pred_u8 = to_uint8(pred)
    real_u8 = to_uint8(sample["target"])
    cond_u8 = to_uint8(sample["cond"])
    gap = np.full((pred_u8.shape[0], pred_u8.shape[1], 2), 255, np.uint8)   #thin white divider
    compare = np.concatenate([cond_u8, gap, real_u8, gap, pred_u8], axis=2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_cell{cell_idx}_half{args.half}_epoch{epoch}"
    write_mp4(pred_u8, out_dir / f"{stem}_pred_chA.mp4")
    write_mp4(compare, out_dir / f"{stem}_compare.mp4")
    print(f"saved      : {out_dir / (stem + '_pred_chA.mp4')}")
    print(f"             {out_dir / (stem + '_compare.mp4')}  (chB | real chA | generated chA)")


if __name__ == "__main__":
    main()
