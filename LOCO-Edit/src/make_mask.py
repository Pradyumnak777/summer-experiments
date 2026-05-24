import argparse, os
import numpy as np
import torch
from PIL import Image

def make_color_mask(img_path, kind, out_path, size=512):
    img = Image.open(img_path).convert('RGB')

    # match ImgDataset.__getitem__ exactly
    w, h = img.size
    crop_size = min(w, h)
    left   = (w - crop_size) / 2
    top    = (h - crop_size) / 2
    right  = (w + crop_size) / 2
    bottom = (h + crop_size) / 2
    img = img.crop((left, top, right, bottom))
    img = img.resize((size, size))

    a = np.array(img).astype(np.int16)

    R, G, B = a[..., 0], a[..., 1], a[..., 2]

    if kind == 'green':
        # bright green, not very red, not very blue
        m = (G > 80) & (G - R > 25) & (G - B > 25)
    elif kind == 'magenta':
        # high R AND B, lower G  (magenta = R + B)
        m = (R > 80) & (B > 80) & (R - G > 15) & (B - G > 15)
    elif kind == 'both':
        green   = (G > 80) & (G - R > 25) & (G - B > 25)
        magenta = (R > 80) & (B > 80) & (R - G > 15) & (B - G > 15)
        m = green | magenta
    else:
        raise ValueError(f"unknown kind: {kind}")

    preview = (np.dstack([a, np.where(m, 255, 0)]) ).astype(np.uint8)
    Image.fromarray(np.where(m[..., None], a, a // 4).astype(np.uint8)).save(
        out_path.replace('.pt', '_preview.png'))

    # mask shape required by edit.py: [3, H, W] bool
    mask_t = torch.from_numpy(m).bool().unsqueeze(0).repeat(3, 1, 1)
    torch.save(mask_t, out_path)
    print(f'saved mask: {out_path}, true-pixels={mask_t[0].sum().item()} / {size*size}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--img', required=True)
    p.add_argument('--kind', choices=['green', 'magenta', 'both'], required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    make_color_mask(args.img, args.kind, args.out)