'''
this script is to inspect raw tiff files and find out thier shape, size, etc-
raw tiffs are generally of size (?)
'''

import argparse
from pathlib import Path

import numpy as np
import tifffile

from .fix_tiffs import infer_axes


def inspect(path):
    path = Path(path)
    size_mb = path.stat().st_size / (1024 * 1024)

    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        arr = series.asarray()
        axes = series.axes or infer_axes(arr.ndim) or "?"
        n_pages = len(tf.pages)
        page0 = tf.pages[0]
        tags = {t.name: t.value for t in page0.tags}

        print(f"file        : {path}")
        print(f"size        : {size_mb:.2f} MB")
        print(f"shape       : {arr.shape}")
        print(f"axes        : {axes}")
        print(f"dtype       : {arr.dtype}")
        print(f"n_pages     : {n_pages}")
        print(f"min / max   : {arr.min()} / {arr.max()}")
        print(f"mean        : {np.mean(arr):.4f}")

        print("\nkey tags (page 0):")
        for key in ("ImageWidth", "ImageLength", "BitsPerSample", "SampleFormat",
                    "PhotometricInterpretation", "Compression", "SamplesPerPixel"):
            if key not in tags:
                continue
            value = tags[key]
            #per-sample tags repeat one value SamplesPerPixel times, collapse that down for readability
            if isinstance(value, tuple) and len(set(value)) == 1:
                value = f"{value[0]} (x{len(value)})"
            print(f"  {key:<26}: {value}")

        if tf.imagej_metadata:
            print("\nimagej_metadata:")
            for k, v in tf.imagej_metadata.items():
                print(f"  {k}: {v}")

        if tf.ome_metadata:
            print("\nome_metadata (raw xml, truncated):")
            print(f"  {tf.ome_metadata[:500]}")


def main():
    parser = argparse.ArgumentParser(description="inspect a tiff file's shape, dtype, and metadata")
    parser.add_argument("path", type=Path, help="path to a .tif/.tiff file")
    args = parser.parse_args()
    inspect(args.path)


if __name__ == "__main__":
    main()
