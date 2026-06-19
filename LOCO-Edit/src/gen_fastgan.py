# gen_fastgan.py — generate & save images from a trained lightweight-gan checkpoint
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
from lightweight_gan.lightweight_gan import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name',        default='microscopy')
    p.add_argument('--results_dir', default='checkpoints_new/fastgan_microscopy/results')
    p.add_argument('--models_dir',  default='checkpoints_new/fastgan_microscopy/models')
    p.add_argument('--load_num',    type=int, default=5,  help='checkpoint number -> model_<n>.pt')
    p.add_argument('--num',         type=int, default=64, help='how many images to generate')
    p.add_argument('--ema',         action='store_true',  help='use EMA generator instead of raw')
    p.add_argument('--out',         default='checkpoints_new/fastgan_microscopy/generated')
    p.add_argument('--seed',        type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # Trainer reconstructs the exact architecture from the saved .config.json
    trainer = Trainer(name=args.name, results_dir=args.results_dir, models_dir=args.models_dir, use_aim=False)
    trainer.load(args.load_num)
    trainer.GAN.eval()

    G = trainer.GAN.GE if args.ema else trainer.GAN.G   # GE = EMA, G = raw
    device = next(G.parameters()).device
    latent_dim = trainer.GAN.latent_dim

    out_dir = Path(args.out) / ('ema' if args.ema else 'raw')
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = []
    with torch.no_grad():
        for i in range(args.num):
            z = torch.randn(1, latent_dim, device=device)
            img = G(z).clamp_(0., 1.)            # lightweight-gan outputs are clamped to [0,1]
            save_image(img, out_dir / f'gen_{i:03d}.png')
            imgs.append(img.cpu())

    # also save a contact-sheet grid for quick viewing
    grid = torch.cat(imgs, dim=0)
    nrow = int(args.num ** 0.5)
    save_image(grid, Path(args.out) / f"grid_{'ema' if args.ema else 'raw'}.png", nrow=nrow)

    print(f'wrote {args.num} images to {out_dir}')
    print(f'grid -> {Path(args.out) / ("grid_ema.png" if args.ema else "grid_raw.png")}')


if __name__ == '__main__':
    main()