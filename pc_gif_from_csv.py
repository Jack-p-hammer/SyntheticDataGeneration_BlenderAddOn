#!/usr/bin/env python3
# pc_gif_from_csv.py — build a GIF from a single-object CSV exported by your Blender add-on.

import argparse
import csv
import math
import random
from typing import Dict
from pathlib import Path as _Path

import numpy as np
import imageio.v3 as iio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description='Make a GIF of a 3D point cloud from a single-object CSV.')
    p.add_argument('input_csv', type=_Path)
    p.add_argument('output_gif', type=_Path)
    p.add_argument('--fps', type=int, default=12)
    p.add_argument('--elev', type=float, default=20.0)
    p.add_argument('--azim', type=float, default=-60.0)
    p.add_argument('--rotate', type=int, default=0)
    p.add_argument('--rotate-speed', type=float, default=2.0)
    p.add_argument('--dpi', type=int, default=120)
    p.add_argument('--size', type=int, nargs=2, default=[800,800], metavar=('W','H'))
    p.add_argument('--point-size', type=float, default=2.0)
    p.add_argument('--downsample', type=float, default=0.0)
    p.add_argument('--tight', type=int, default=1)
    p.add_argument('--bg', type=str, default='white', choices=['white','black'])
    return p.parse_args()


def read_single_object_csv(path: _Path):
    data: Dict[int, np.ndarray] = {}
    with path.open('r', newline='') as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        if len(header) < 4 or header[0].strip().lower() != 'frame':
            raise ValueError("CSV must start with 'frame' followed by x1,y1,z1,... columns")
        ntriples = (len(header) - 1) // 3
        expected_len = 1 + 3 * ntriples
        for row in rdr:
            if not row:
                continue
            f = int(float(row[0]))
            vals = row[1:expected_len]
            arr = np.array(list(map(float, vals)), dtype=np.float32).reshape(-1,3)
            data[f] = arr
    frames = sorted(data.keys())
    return frames, data


def compute_global_bounds(data: Dict[int, np.ndarray]):
    if not data:
        return (-1,1), (-1,1), (-1,1)
    all_pts = np.concatenate([v for v in data.values() if v.size > 0], axis=0)
    if all_pts.size == 0:
        return (-1,1), (-1,1), (-1,1)
    min_xyz = all_pts.min(axis=0)
    max_xyz = all_pts.max(axis=0)
    span = np.maximum(max_xyz - min_xyz, 1e-6)
    pad = 0.02 * span
    lo = min_xyz - pad
    hi = max_xyz + pad
    max_span = float(np.max(hi - lo))
    center = (hi + lo) / 2.0
    lo_eq = center - 0.5 * max_span
    hi_eq = center + 0.5 * max_span
    return (lo_eq[0], hi_eq[0]), (lo_eq[1], hi_eq[1]), (lo_eq[2], hi_eq[2])


def frame_to_image(points, xlim, ylim, zlim, size_px=(800, 800), dpi=120,
                   elev=20.0, azim=-60.0, point_size=2.0, bg="white"):
    import numpy as np
    import matplotlib.pyplot as plt

    w, h = size_px
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    if bg == "black":
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.w_xaxis.set_pane_color((0, 0, 0, 0))
        ax.w_yaxis.set_pane_color((0, 0, 0, 0))
        ax.w_zaxis.set_pane_color((0, 0, 0, 0))
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")

    ax.view_init(elev=elev, azim=azim)

    if points.size > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=point_size)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.grid(False)

    # Keep layout tight
    plt.tight_layout()

    # Draw, then grab RGBA buffer and drop alpha
    fig.canvas.draw()
    img_rgba = np.asarray(fig.canvas.buffer_rgba())  # (H, W, 4)
    img_rgb = img_rgba[..., :3].copy()               # (H, W, 3)
    plt.close(fig)
    return img_rgb


def main():
    args = parse_args()
    frames, data = read_single_object_csv(args.input_csv)
    if not frames:
        raise SystemExit('No frames found in CSV.')
    if args.downsample and args.downsample > 0.0:
        rng = random.Random(12345)
        for f in frames:
            pts = data[f]
            if pts.size == 0:
                continue
            keep = max(1, int(math.ceil(len(pts) * min(1.0, max(0.0, args.downsample)))))
            idx = np.array(rng.sample(range(len(pts)), keep))
            data[f] = pts[idx]
    xlim, ylim, zlim = compute_global_bounds(data)
    images = []
    az = args.azim
    for f in frames:
        if args.rotate:
            az += args.rotate_speed
        img = frame_to_image(data[f], xlim, ylim, zlim, size_px=tuple(args.size), dpi=args.dpi,
                             elev=args.elev, azim=az, point_size=args.point_size, bg=args.bg)
        images.append(img)
    args.output_gif.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output_gif, images, plugin='pillow', duration=1000.0/args.fps, loop=0)
    print(f"[ok] Wrote GIF: {args.output_gif}  ({len(images)} frames @ {args.fps} fps)")


if __name__ == '__main__':
    main()