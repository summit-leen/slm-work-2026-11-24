# -*- coding: utf-8 -*-
"""用实测帧跑拼接式重建（版本 B）。

    python code/recon_real.py --obj captures/rec_obj_mean.npy \
                             --flat captures/rec_flat_mean.npy --bin 8

输入是相机原始帧（3860×2178 float），会先按 bin 下采样到模型网格，
用平场做照明归一化，再交给 mla_recon.mosaic_recon。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mla_recon as M  # noqa: E402


def bin_img(a: np.ndarray, k: int) -> np.ndarray:
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="captures/rec_obj_mean.npy")
    ap.add_argument("--flat", default="captures/rec_flat_mean.npy")
    ap.add_argument("--csv", default="patterns/B_rla420_lenses.csv")
    ap.add_argument("--bin", type=int, default=8)
    ap.add_argument("--fov", type=float, default=4.2)
    ap.add_argument("--d", type=float, default=200.0)
    ap.add_argument("--s", type=float, default=50.0)
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--rl", type=int, default=25)
    ap.add_argument("--out", default="captures/real_B")
    args = ap.parse_args()

    obj = np.load(args.obj).astype(np.float64)
    flat = np.load(args.flat).astype(np.float64)
    obj_b = bin_img(obj, args.bin)
    flat_b = bin_img(flat, args.bin)
    H, W = obj_b.shape
    print(f"下采样 {args.bin}× → {H}×{W}")

    # 照明归一化：平场是 SLM 全 0（同一物体），用它抹平照明不均
    y = obj_b / np.maximum(flat_b, 1e-6)
    y = y - np.median(y)                       # 去掉直穿光偏置
    y = np.clip(y, 0, None)
    y = y / max(np.percentile(y, 99.5), 1e-9)
    print(f"归一化后 y：mean {y.mean():.3f}  max {y.max():.2f}")

    lens = M.load_lenses(args.csv, args.n, args.fov, args.s, args.d)
    A = M.MLAOperator(lens, args.n, args.fov, (H, W), M.SENSOR_PX_MM * args.bin)
    x = M.mosaic_recon(A, y, rl_iters=args.rl)
    x = np.rot90(x, 2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(f"{args.out}_rec.npy", x)
    M._save_views(np.zeros_like(x), y, x, args.out)
    print(f"已保存 {args.out}_rec.npy / _rec.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
