# -*- coding: utf-8 -*-
"""在随机透镜阵列上叠一层随机相位背景（混合散射体）。

阵列口径内保留透镜相位，其余面积填随机相位；背景调制深度可调，
用来调整"散斑底"的强弱，从而让焦点更突出（或更不突出）。

    python code/hybrid_pattern.py --bg-max 255 --save patterns/hyb_full   # 全幅随机背景
    python code/hybrid_pattern.py --bg-max 64  --save patterns/hyb_A      # 弱背景（≈0.32π）
    python code/hybrid_pattern.py --bg-max 64  --show --hold 0            # 直接显示
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build(array_npy: str, bg_max: int, seed: int) -> np.ndarray:
    """读入阵列图案，口径内保留原值，口径外填 [0, bg_max] 均匀随机灰度。"""
    lens = np.load(array_npy)
    if lens.ndim != 2:
        raise SystemExit(f"{array_npy} 不是二维相位图")
    rng = np.random.default_rng(seed)
    mask = lens > 0
    bg = rng.integers(0, bg_max + 1, size=lens.shape, dtype=np.uint8)
    out = np.where(mask, lens, bg).astype(np.uint8)
    print(f"阵列 {mask.sum()} px ({100 * mask.mean():.1f}%), "
          f"背景 0..{bg_max}, 均灰 {out.mean():.1f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="阵列 + 随机相位背景的混合图案")
    ap.add_argument("--array", default="patterns/rla_phase.npy", help="阵列图案 npy")
    ap.add_argument("--bg-max", type=int, default=255, help="背景随机相位上限（0..255）")
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--save", default=None, help="输出前缀（存 _phase.npy/_png）")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--hold", type=float, default=0.0, help="0 = 常驻")
    ap.add_argument("--monitor", type=int, default=1)
    args = ap.parse_args(argv)

    gray = build(args.array, args.bg_max, args.seed)

    if args.save:
        import json

        from PIL import Image
        prefix = args.save
        d = os.path.dirname(prefix)
        if d:
            os.makedirs(d, exist_ok=True)
        np.save(prefix + "_phase.npy", gray)
        Image.fromarray(gray, mode="L").save(prefix + "_phase.png")
        meta = dict(array=args.array, bg_max=args.bg_max, seed=args.seed,
                    mean_gray=float(gray.mean()))
        with open(prefix + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        print(f"已保存 {prefix}_phase.npy / _phase.png")

    if args.show:
        from slm_driver import SLM
        slm = SLM(monitor=args.monitor)
        w, h = slm.getSize()
        if gray.shape != (h, w):
            raise SystemExit(f"图案 {gray.shape} 与面板 {(h, w)} 不符")
        slm.update(gray)
        print(f"显示 {args.hold or 'inf'} s ... (关窗口结束)")
        slm.run(hold=args.hold or None)
        globals()["_keepalive"] = slm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
