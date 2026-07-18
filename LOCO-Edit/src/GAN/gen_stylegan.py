import os
import sys
import argparse
import torch
import numpy as np
from torchvision.utils import make_grid, save_image

REPO_DIR = "/scratch/pbk5339/summer/LOCO-Edit/src/GAN/stylegan2-ada-pytorch"
sys.path.insert(0, REPO_DIR)
import legacy   


if os.getenv("DEBUGPY", "0") == "1":
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach on port 5678...")
    debugpy.wait_for_client()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pkl',    required=True, help='path to network-snapshot-XXXXXX.pkl')
    p.add_argument('--out',    default='gen_out')
    p.add_argument('--num',    type=int, default=64, help='how many images to generate')
    p.add_argument('--nrow',   type=int, default=8,  help='images per row in the grid')
    p.add_argument('--seed',   type=int, default=0)
    p.add_argument('--trunc',  type=float, default=1.0,
                   help='truncation psi. 1.0 = raw/diverse; <1 pulls toward the mean (cleaner, less varied)')
    args = p.parse_args()

    device = torch.device('cuda:0')         
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    
    with open(args.pkl, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() #loads the generator..
    print(f'loaded G_ema: z_dim={G.z_dim}, c_dim={G.c_dim}, img_channels={G.img_channels}, res={G.img_resolution}')

    # unconditional model -> c_dim is 0, so the label tensor is shape [num, 0]
    z = torch.randn([args.num, G.z_dim], device=device)
    c = torch.zeros([args.num, G.c_dim], device=device)

    with torch.no_grad():
        img = G(z, c, truncation_psi=args.trunc, noise_mode='const')   # [num, 2, H, W], roughly [-1, 1]

    # StyleGAN output convention -> [0, 1] for saving. (network range ~[-1,1] -> *0.5+0.5)
    img = (img * 0.5 + 0.5).clamp(0, 1).cpu()

    # # 1) false-color composite grid (chA=red, chB=green, blue=0) — matches how reals.png looked
    # zeros = torch.zeros_like(img[:, :1])
    # rgb = torch.cat([img[:, :1], img[:, 1:2], zeros], dim=1)          # [num, 3, H, W]
    # save_image(make_grid(rgb, nrow=args.nrow), os.path.join(args.out, 'gen_falsecolor.png'))

    #each channel on its own
    save_image(make_grid(img[:, :1],  nrow=args.nrow), os.path.join(args.out, 'gen_chA.png'))
    save_image(make_grid(img[:, 1:2], nrow=args.nrow), os.path.join(args.out, 'gen_chB.png'))

    print(f'wrote 3 grids ({args.num} samples each) to {args.out}/')


if __name__ == '__main__':
    main()