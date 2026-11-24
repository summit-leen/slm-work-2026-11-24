# -*- coding: utf-8 -*-
"""密铺微透镜阵列（hex 栅格，塞进被照亮的圆形区域）。

和 random_lens_array.py 的区别：位置不是随机撒，而是六角密排，覆盖率高，
用于"每个子孔径各自成一张子图"的成像/光场实验。

    # 成像模式：d=20 mm, s=50 mm -> f = d*s/(d+s) = 14.29 mm
    python code/dense_mla.py --d 20 --s 50 --area-mm 3.1 --save patterns/dma_d20s50 --show --hold 0

    # 只看设计（算覆盖率、环数、边缘采样）
    python code/dense_mla.py --d 20 --s 50 --diag
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random_lens_array as rla  # noqa: E402

W, H = 1024, 768


def hex_points(cx, cy, pitch, radius):
    """六角栅格点，落在半径 radius 的圆内。"""
    rows = int(math.floor(radius / (pitch * math.sqrt(3) / 2))) + 1
    pts = []
    for i in range(-rows, rows + 1):
        y = i * pitch * math.sqrt(3) / 2
        if abs(y) > radius:
            continue
        dx = pitch * (0.5 if i % 2 else 0.0)
        n = int(math.floor((math.sqrt(max(radius ** 2 - y * y, 0)) - dx) / pitch))
        for j in range(-n, n + 1):
            x = j * pitch + dx
            if x * x + y * y <= radius ** 2:
                pts.append((cx + x, cy + y))
    return pts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=float, default=20.0, help="物面到面板 mm")
    ap.add_argument("--s", type=float, default=50.0, help="面板到相机 mm")
    ap.add_argument("--f", type=float, default=None, help="焦距 mm（缺省按 1/f=1/d+1/s 算）")
    ap.add_argument("--area-mm", type=float, default=3.1, help="被照亮区域直径 mm")
    ap.add_argument("--r", type=float, default=0.0, help="透镜半径 px（0=取混叠上限）")
    ap.add_argument("--gap", type=float, default=4.0, help="透镜间隙 px")
    ap.add_argument("--pitch", type=float, default=18.0)
    ap.add_argument("--lam", type=float, default=620.0)
    ap.add_argument("--gamma", type=float, default=400.0)
    ap.add_argument("--cx", type=float, default=512.0)
    ap.add_argument("--cy", type=float, default=384.0)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="位置抖动（占 pitch 的比例，0=规则栅格，0.35 左右仍不重叠）")
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--save", default=None)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--hold", type=float, default=0.0)
    args = ap.parse_args()

    p = args.pitch * 1e-6
    lam = args.lam * 1e-9
    f_mm = args.f or (args.d * args.s / (args.d + args.s))
    f = f_mm * 1e-3
    r_lim = lam * f / (2 * p) / p      # px
    r = args.r or min(r_lim * 0.95, 60.0)
    if r > r_lim:
        print(f"警告：r={r:.1f} px 超过混叠上限 {r_lim:.1f} px")

    R = args.area_mm * 1e-3 / 2 / p    # px
    pitch = 2 * r + args.gap
    pts = hex_points(args.cx, args.cy, pitch, R)
    if args.jitter > 0:
        rng = np.random.default_rng(args.seed)
        amp = args.jitter * pitch
        jit = []
        for x, y in pts:
            dx, dy = rng.uniform(-amp, amp, 2)
            # 抖动后必须仍在有效区内，且与已有的保持不重叠
            if math.hypot(x + dx - args.cx, y + dy - args.cy) + r > R:
                dx = dy = 0.0
            jit.append((x + dx, y + dy))
        pts = jit
    ns = argparse.Namespace(pitch=args.pitch, lam=args.lam, gamma=args.gamma,
                            sign=-1, tilt=(0.0, 0.0))
    lenses = [rla.Lens(x, y, r, f, p) for x, y in pts]
    pat = rla.build_pattern(W, H, lenses, ns)

    m = args.s / args.d
    print(f"f = {f_mm:.2f} mm (d={args.d:g}, s={args.s:g} → 放大率 {m:.2f})")
    print(f"半径 r = {r:.1f} px ({r*p*1e6:.0f} um)，栅格 pitch = {pitch:.1f} px，"
          f"有效区直径 {args.area_mm:g} mm ({2*R:.0f} px)")
    print(f"透镜 {len(pts)} 枚，覆盖 {len(pts)*math.pi*r*r/(math.pi*R*R)*100:.1f}%")
    zones = (r * p) ** 2 / (2 * lam * f)
    print(f"每枚环数 {zones:.2f}，边缘采样 {r*p*p/(lam*f):.3f} 周期/px")
    if zones < 1:
        print("注意：环数 <1，透镜太弱（等效缩孔），焦斑会变大")
    airy = 1.22 * lam * f / (r * p)
    print(f"艾里半径 {airy*1e6:.0f} um → 物方 {airy/m*1e6:.0f} um；"
          f"子图视场 {2*r*p*1e3/m:.3f} mm；总视场 {2*R*p*1e3/m:.2f} mm")
    print(f"相位图 {pat.shape} uint8 均值 {pat.mean():.1f}，灰度最大 {pat.max()}")

    if args.save:
        base = args.save
        np.save(base + "_phase.npy", pat)
        from PIL import Image
        Image.fromarray(pat).save(base + "_phase.png")
        if args.diag or True:
            prev = rla.render_aa(W, H, lenses, ns, ss=3)
            Image.fromarray(prev).save(base + "_preview.png")
        print("已保存", base + "_phase.npy / .png / _preview.png")

    if args.show:
        from slm_driver import SLM
        slm = SLM(monitor=1)
        w, h = slm.getSize()
        if (h, w) != pat.shape:
            raise SystemExit(f"图案 {pat.shape} 与面板 {(h, w)} 不符")
        slm.update(np.ascontiguousarray(pat))
        print(f"显示中 {(h, w)} ... hold={args.hold or 'inf'}")
        globals()["_keep"] = slm
        slm.run(hold=args.hold or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
