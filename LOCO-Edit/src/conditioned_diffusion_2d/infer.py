'''
generate chA for one chB crop with a trained conditioned 2D model, and save a one-row figure:
input chB | real chA | generated chA.

pick the crop from the train or val split (the same recording-level split training used) by
index or by name. only chB goes into the model; generation starts from noise, as in the previews.

run from src/:
    python conditioned_diffusion_2d/infer.py --split val --list            #crop names and indices
    python conditioned_diffusion_2d/infer.py --split val --index 100
    python conditioned_diffusion_2d/infer.py --split train --name "<crop name>" --ckpt <model_ema_epochN.pt>
while training is using GPUs 0-3, run it on a free one:
    CUDA_VISIBLE_DEVICES=4 python conditioned_diffusion_2d/infer.py --split val --index 100
'''
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from diffusers import DDPMScheduler, DDIMScheduler

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))
from conditioned_diffusion_2d.data_utils import PairedCropDataset
from conditioned_diffusion_2d.model import build_model

CKPT_DIR = SRC_ROOT / "conditioned_diffusion_2d_checkpoints/chA_given_chB"
NUM_TRAIN_TIMESTEPS = 1000   #must match training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--index", type=int, default=0, help="crop index within --split (see --list)")
    p.add_argument("--name", default=None, help="crop name instead of --index, e.g. '<recording>_cell3_f012'")
    p.add_argument("--ckpt", default=str(CKPT_DIR / "model_ema_epoch90.pt"))
    p.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=str(SRC_ROOT / "conditioned_diffusion_2d/inference_outputs"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--list", action="store_true", help="print the crops in --split and exit")
    return p.parse_args()


def load_model(path, device):
    '''rebuild the model from the settings stored in the checkpoint.'''
    state = torch.load(path, map_location="cpu")
    model = build_model(state["chB_encoder_ckpt"], stop_block=state["stop_block"], token_dim=state["token_dim"])
    model.unet.load_state_dict(state["unet"])
    model.encoder.proj.load_state_dict(state["encoder_proj"])
    return model.to(device).eval()


@torch.no_grad()
def generate(model, cond, steps, seed):
    '''cond: chB [B, 1, H, W] in [-1, 1] -> generated chA, same shape.'''
    sampler = DDIMScheduler.from_config(DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS).config)
    sampler.set_timesteps(steps)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = model.encode(cond)
        x = torch.randn(cond.shape, generator=torch.Generator(cond.device).manual_seed(seed), device=cond.device)
        for t in sampler.timesteps:
            noise_pred = model(x, t.to(cond.device), tokens=tokens)
            x = sampler.step(noise_pred.float(), t, x).prev_sample
    return x


def to_display(img):
    '''[1, H, W] in [-1, 1] -> [H, W] in [0, 1].'''
    return (img[0].float().clamp(-1, 1) * 0.5 + 0.5).cpu().numpy()


def main():
    args = parse_args()
    dataset = PairedCropDataset(split=args.split)

    if args.list:
        for i, name in enumerate(dataset.bases):
            print(f"{i:6d}  {name}")
        return

    if args.name is not None and args.name not in dataset.bases:
        raise SystemExit(f"no crop named {args.name!r} in the {args.split} split (names contain a space, so quote them; "
                         f"see --list)")
    index = dataset.bases.index(args.name) if args.name else args.index
    sample = dataset[index]
    ckpt = Path(args.ckpt)
    print(f"checkpoint : {ckpt}")
    print(f"crop       : {args.split} #{index}  {sample['name']}")

    device = torch.device(args.device)
    model = load_model(ckpt, device)
    pred = generate(model, sample["cond"].unsqueeze(0).to(device), args.steps, args.seed)[0]

    panels = [("input chB", sample["cond"]), ("real chA", sample["target"]), ("generated chA", pred)]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(to_display(img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{args.split} #{index}  |  {ckpt.stem}  |  seed {args.seed}", fontsize=9)
    fig.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.split}_{index}_{ckpt.stem}_seed{args.seed}.png"
    fig.savefig(out, dpi=150)
    print(f"saved      : {out}")


if __name__ == "__main__":
    main()
