# -*- coding: utf-8 -*-
"""在 SLM 上显示随机相位图案（散斑用）。

每个像素（或 block x block 的小块）取 0..max_grey 的均匀随机灰度，
经面板 alpha=0.645 周期/255 折算即 0..1.29pi 的随机相位。

    python code/random_phase.py --seed 1 --show --hold 0
    python code/random_phase.py --seed 1 --block 2 --save patterns/speckle
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def make_pattern(h=768, w=1024, seed=1, block=1, max_grey=255) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bh, bw = (h + block - 1) // block, (w + block - 1) // block
    small = rng.integers(0, max_grey + 1, size=(bh, bw), dtype=np.uint8)
    if block > 1:
        small = np.kron(small, np.ones((block, block), np.uint8))
    return np.ascontiguousarray(small[:h, :w])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--block", type=int, default=1)
    ap.add_argument("--max-grey", type=int, default=255)
    ap.add_argument("--w", type=int, default=1024)
    ap.add_argument("--h", type=int, default=768)
    ap.add_argument("--save", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--hold", type=float, default=0.0, help="0 = 常驻")
    ap.add_argument("--monitor", type=int, default=1)
    args = ap.parse_args()

    pat = make_pattern(args.h, args.w, args.seed, args.block, args.max_grey)
    phase = pat.astype(np.float64) / 255.0 * 0.645 * 2 * np.pi
    print(f"随机相位块 {args.block}x{args.block}，灰度 0..{args.max_grey} "
          f"→ 相位 0..{phase.max():.2f} rad，均值 {phase.mean():.2f} rad")

    if args.save:
        base = args.save
        np.save(base + "_phase.npy", pat)
        try:
            from PIL import Image
            Image.fromarray(pat).save(base + "_phase.png")
            print("已保存", base + "_phase.npy / .png")
        except Exception as e:
            print("PNG 未保存:", type(e).__name__)

    if args.show:
        from slm_driver import SLM
        slm = SLM(monitor=args.monitor)
        w, h = slm.getSize()
        img = pat if (h, w) == pat.shape else make_pattern(h, w, args.seed, args.block, args.max_grey)
        slm.update(np.ascontiguousarray(img))
        print(f"显示 {w}x{h}, hold={args.hold or 'inf'}")
        globals()["_keep"] = slm
        slm.run(hold=args.hold or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
