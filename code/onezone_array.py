# -*- coding: utf-8 -*-
"""单带（one-zone）微透镜阵列：每枚透镜只写一个菲涅尔带。

单带透镜的焦距与半径被唯一绑定：  f = r^2 / (2*lambda)
所以给定 f 就得到 r = sqrt(2*lambda*f)，反之亦然。
本脚本按随机 f（默认 30-50 mm）算出各自半径，用六角栅格铺满整个面板。

    python code/onezone_array.py --diag
    python code/onezone_array.py --save patterns/onezone --show --hold 0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PITCH_UM = 18.0
DEFAULT_LAM_NM = 620.0


def hex_points(cx: float, cy: float, pitch: float, radius: float):
    """返回圆内六角栅格点。"""
    pts = []
    dy = pitch * np.sqrt(3) / 2.0
    ny = int(np.ceil(radius / dy)) + 1
    nx = int(np.ceil(radius / pitch)) + 1
    for j in range(-ny, ny + 1):
        for i in range(-nx, nx + 1):
            x = cx + (i + 0.5 * (j & 1)) * pitch
            y = cy + j * dy
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                pts.append((x, y))
    return pts


def build(args):
    p = args.pitch * 1e-6
    lam = args.lam * 1e-9
    W, H = args.w, args.h
    cx, cy = W / 2.0, H / 2.0

    # 单带关系：r = sqrt(2 lam f) / p  (px)
    r_of_f = lambda f_mm: np.sqrt(2 * lam * f_mm * 1e-3) / p
    r_min, r_max = r_of_f(args.f_min), r_of_f(args.f_max)

    rng = np.random.default_rng(args.seed)
    pitch = 2.0 * r_max * (1.0 + args.gap_frac)
    area_r = args.area_mm * 1e-3 / 2.0 / p if args.area_mm > 0 else max(W, H) / 2.0

    lens = []
    if args.layout == "hex":
        cand = hex_points(cx, cy, pitch, area_r)
        for (x, y) in cand:
            if not (0 <= x < W and 0 <= y < H):
                continue
            f = rng.uniform(args.f_min, args.f_max)
            r = r_of_f(f)
            if x - r < 0 or y - r < 0 or x + r >= W or y + r >= H:
                continue
            lens.append((x, y, r, f))
            if args.n and len(lens) >= args.n:
                break
    else:  # 随机摆放（RSA），互不重叠
        n_target = args.n or 150
        gap = args.gap_frac
        for _ in range(200000):
            if len(lens) >= n_target:
                break
            x = rng.uniform(0, W)
            y = rng.uniform(0, H)
            f = rng.uniform(args.f_min, args.f_max)
            r = r_of_f(f)
            if x - r < 0 or y - r < 0 or x + r >= W or y + r >= H:
                continue
            if (x - cx) ** 2 + (y - cy) ** 2 > area_r ** 2:
                continue
            ok = True
            for (x2, y2, r2, _f2) in lens:
                if (x - x2) ** 2 + (y - y2) ** 2 < (r + r2 + gap * 4) ** 2:
                    ok = False
                    break
            if ok:
                lens.append((x, y, r, f))

    # 相位
    Y, X = np.mgrid[0:H, 0:W]
    pat = np.zeros((H, W), np.float64)
    for (x, y, r, f) in lens:
        x0, x1 = int(max(0, x - r - 1)), int(min(W, x + r + 2))
        y0, y1 = int(max(0, y - r - 1)), int(min(H, y + r + 2))
        yy = Y[y0:y1, x0:x1] - y
        xx = X[y0:y1, x0:x1] - x
        rr2 = xx * xx + yy * yy
        m = rr2 <= r * r
        phi = -np.pi * rr2 * p * p / (lam * f * 1e-3)
        pat[y0:y1, x0:x1][m] = np.mod(phi[m], 2 * np.pi)
    img = np.clip(np.round(pat / (2 * np.pi) * 255), 0, 255).astype(np.uint8)
    return img, lens, r_min, r_max, pitch


def describe(img, lens, r_min, r_max, pitch, args):
    cov = sum(np.pi * r * r for _, _, r, _ in lens) / (img.shape[0] * img.shape[1])
    print("面板        %dx%d, 像素间距 %.0f um" % (img.shape[1], img.shape[0], args.pitch))
    print("波长         %0.0f nm -> 单带关系 f = r^2/(2 lambda)" % args.lam)
    print("焦距         随机 %.1f .. %.1f mm" % (args.f_min, args.f_max))
    print("对应半径     %.2f .. %.2f px (%.0f .. %.0f um)" % (r_min, r_max, r_min * args.pitch, r_max * args.pitch))
    print("透镜数量     %d" % len(lens))
    print("栅格节距     %.2f px (%.0f um)" % (pitch, pitch * args.pitch))
    print("覆盖率       %.1f%%" % (cov * 100))
    print("带数         1.00（每枚）")
    print("相位图       灰度 0..255，均值 %.1f" % img.mean())
    fs = np.array([f for _, _, _, f in lens])
    if len(fs):
        print("焦距分布     %.2f .. %.2f mm, 均值 %.2f" % (fs.min(), fs.max(), fs.mean()))
        a = 2 * np.array([r for _, _, r, _ in lens]) * args.pitch * 1e-6   # 直径 (m)
        airy = 1.22 * (args.lam * 1e-9) * (fs * 1e-3) / a                   # 半径 (m)
        print("艾里半径     %.0f .. %.0f um (中位 %.0f um)" % (airy.min() * 1e6, airy.max() * 1e6, np.median(airy) * 1e6))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f-min", type=float, default=30.0)
    ap.add_argument("--f-max", type=float, default=50.0)
    ap.add_argument("--gap-frac", type=float, default=0.04, help="相邻圆之间的相对间隙")
    ap.add_argument("--area-mm", type=float, default=0.0, help="0 = 铺满整幅")
    ap.add_argument("--n", type=int, default=0, help="0 = 铺满（hex）/ 150（random 默认）")
    ap.add_argument("--layout", choices=["hex", "random"], default="hex")
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--pitch", type=float, default=DEFAULT_PITCH_UM)
    ap.add_argument("--lam", type=float, default=DEFAULT_LAM_NM)
    ap.add_argument("--w", type=int, default=1024)
    ap.add_argument("--h", type=int, default=768)
    ap.add_argument("--save", default="")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--hold", type=float, default=30.0)
    ap.add_argument("--monitor", type=int, default=1)
    args = ap.parse_args(argv)

    img, lens, r_min, r_max, pitch = build(args)
    describe(img, lens, r_min, r_max, pitch, args)

    if args.save:
        base = args.save
        np.save(base + "_phase.npy", img)
        from PIL import Image
        Image.fromarray(img).save(base + "_phase.png")
        with open(base + "_lenses.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "cx_px", "cy_px", "r_px", "f_mm", "zones", "a_mm"])
            for i, (x, y, r, f) in enumerate(lens):
                w.writerow([i, "%.2f" % x, "%.2f" % y, "%.2f" % r, "%.3f" % f, "1.00", "%.4f" % (2 * r * args.pitch / 1000)])
        print("已保存 %s_phase.npy/.png/_lenses.csv" % base)

    if args.show:
        from slm_driver import SLM
        slm = SLM(monitor=args.monitor, verbose=False)
        w, h = slm.getSize()
        if (h, w) != img.shape:
            print("尺寸不符：%s vs %s" % (img.shape, (h, w)))
            return 2
        slm.update(img)
        print("显示 %g s ..." % (args.hold or 0))
        globals()["_keep"] = slm
        slm.run(hold=args.hold or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
