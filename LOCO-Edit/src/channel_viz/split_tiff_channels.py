#!/usr/bin/env python3
"""Analyze and split a two-channel time-lapse TIFF.

The script reads ImageJ/OME-style TIFF axes with tifffile, separates the
channel axis, writes one TIFF stack per channel, and saves per-frame intensity
statistics for quality control.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile as tiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a multi-channel biological signal TIFF into channel stacks."
    )
    parser.add_argument("input_tif", type=Path, help="Input TIFF file")
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to '<input stem>_split_channels'.",
    )
    return parser.parse_args()


def first_existing_axis(axes: str, candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in axes:
            return candidate
    return None


def move_to_tcyx(data: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    """Return data as T,C,Y,X and a normalized axes string."""
    if "C" not in axes:
        raise ValueError(f"No channel axis found in TIFF axes {axes!r}")

    y_axis = first_existing_axis(axes, "Y")
    x_axis = first_existing_axis(axes, "X")
    if y_axis is None or x_axis is None:
        raise ValueError(f"No Y/X spatial axes found in TIFF axes {axes!r}")

    keep_axes = []
    if "T" in axes:
        keep_axes.append("T")
    keep_axes.extend(["C", y_axis, x_axis])

    # Drop singleton axes such as Z/S if present; fail loudly for non-singletons.
    slicer = []
    reduced_axes = []
    for axis, size in zip(axes, data.shape):
        if axis in keep_axes:
            slicer.append(slice(None))
            reduced_axes.append(axis)
        elif size == 1:
            slicer.append(0)
        else:
            raise ValueError(
                f"Unsupported non-singleton axis {axis!r} with size {size}; "
                "please split Z/extra axes before channel separation."
            )
    data = data[tuple(slicer)]
    axes = "".join(reduced_axes)

    if "T" not in axes:
        data = np.expand_dims(data, axis=0)
        axes = "T" + axes

    order = [axes.index(axis) for axis in "TCYX"]
    return np.moveaxis(data, order, range(4)), "TCYX"


def channel_stats(channel_data: np.ndarray, channel_index: int) -> list[dict[str, float | int]]:
    rows = []
    for frame_index, frame in enumerate(channel_data, start=1):
        rows.append(
            {
                "channel": channel_index,
                "frame": frame_index,
                "min": int(frame.min()),
                "max": int(frame.max()),
                "mean": float(frame.mean()),
                "median": float(np.median(frame)),
                "std": float(frame.std()),
                "sum": int(frame.sum()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    input_tif = args.input_tif.expanduser().resolve()
    outdir = args.outdir or input_tif.with_name(f"{input_tif.stem}_split_channels")
    outdir.mkdir(parents=True, exist_ok=True)

    with tiff.TiffFile(input_tif) as tif:
        series = tif.series[0]
        raw_axes = series.axes
        data = series.asarray()
        imagej_metadata = tif.imagej_metadata or {}

    data_tcyx, axes = move_to_tcyx(data, raw_axes)
    frames, channels, height, width = data_tcyx.shape
    if channels < 2:
        raise ValueError(f"Expected at least 2 channels, found {channels}")

    print(f"Input: {input_tif}")
    print(f"Original axes/shape: {raw_axes} {data.shape}")
    print(f"Normalized axes/shape: {axes} {data_tcyx.shape}")
    print(f"Frames: {frames}; channels: {channels}; size: {height} x {width}")

    all_rows = []
    for channel_zero_based in range(channels):
        channel_index = channel_zero_based + 1
        channel_data = data_tcyx[:, channel_zero_based, :, :]
        output_tif = outdir / f"{input_tif.stem}_channel{channel_index}.tif"

        metadata = {
            "axes": "TYX",
            "unit": imagej_metadata.get("unit"),
            "spacing": imagej_metadata.get("spacing"),
        }
        tiff.imwrite(
            output_tif,
            channel_data,
            imagej=True,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )
        all_rows.extend(channel_stats(channel_data, channel_index))

        print(
            f"Wrote channel {channel_index}: {output_tif} "
            f"shape={channel_data.shape} dtype={channel_data.dtype}"
        )

    stats_csv = outdir / f"{input_tif.stem}_channel_stats.csv"
    with stats_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["channel", "frame", "min", "max", "mean", "median", "std", "sum"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote stats: {stats_csv}")


if __name__ == "__main__":
    main()
