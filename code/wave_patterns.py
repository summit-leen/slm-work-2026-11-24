# -*- coding: utf-8 -*-
"""三种"定制散斑"相位图案：条纹、圆环、水波。

    python code/wave_patterns.py --which all --save-prefix patterns/W

W1  一维随机相位   phi = 2*pi*u(x)      → 远场被压成一条水平条纹
W2  轴棱锥(锥面)   phi = -2*pi*r/Lam     → 远场汇聚成一圈，半径 R = s*lam/Lam
W3  Kolmogorov屏   PSD ~ f^(-11/3)      → 大尺度起伏，散斑呈水波状

注意：面板行程只有 1.29π（gray 255），本脚本按 2π→255 映射，实际只用到 64.5% 的
设计行程；W2 的锥面斜率受采样极限约束 Lam >= 2p（否则外圈环带 < 1 像素）。
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

P = 18e-6          # 像素间距 (m)
LAM = 620e-9       # 波长 (m)
S = 0.050          # 面板到相机 (m)
H, W = 768, 1024


def grey(phi: np.ndarray) -> np.ndarray:
    """把相位 [rad] 映射到 0..255（2π → 255）。"""
    return np.round(((phi % (2 * np.pi)) / (2 * np.pi)) * 255).astype(np.uint8)


def w1_stripes(seed: int = 1) -> np.ndarray:
    """一维随机相位：每列一个随机值，沿 y 不变 → 远场水平条纹。"""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, size=W)
    return grey(np.broadcast_to(u[None, :], (H, W)))


def w2_axicon(lam_nm: float = 620.0, ring_mm: float = 0.78,
              jitter: float = 0.0, seed: int = 1) -> np.ndarray:
    """轴棱锥：远场成半径 ring_mm 的环。"""
    lam = lam_nm * 1e-9
    lam_ax = S * lam / (ring_mm * 1e-3)          # 锥面周期
    if lam_ax < 2 * P:
        raise ValueError(f"锥面周期 {lam_ax*1e6:.1f} um < 2p={2*P*1e6:.0f} um，会混叠；"
                         f"请把 ring_mm 调小（上限约 {S*lam/(2*P)*1e3:.2f} mm）")
    y, x = np.mgrid[0:H, 0:W]
    r = np.hypot((x - (W - 1) / 2) * P, (y - (H - 1) / 2) * P)
    phi = -2 * np.pi * r / lam_ax
    if jitter > 0:
        phi = phi + np.random.default_rng(seed).uniform(0, jitter * np.pi, (H, W))
    return grey(phi)


def w3_kolmogorov(f0: float = 0.01, seed: int = 1) -> np.ndarray:
    """Kolmogorov 相位屏：PSD ∝ (f/f0)^(-11/3)，远场呈水波状。"""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    FX, FY = np.meshgrid(fx, fy)
    f = np.hypot(FX, FY)
    f[0, 0] = f[0, 1]
    psd = (f / f0) ** (-11.0 / 3.0)
    cn = rng.normal(size=(H, W)) + 1j * rng.normal(size=(H, W))
    phi = np.fft.ifft2(cn * np.sqrt(psd)).real
    phi -= phi.min()
    phi *= 2 * np.pi / phi.max()
    return grey(phi)


def save(pat: np.ndarray, prefix: str, tag: str) -> None:
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    np.save(f"{prefix}{tag}_phase.npy", pat)
    Image.fromarray(pat).save(f"{prefix}{tag}_phase.png")
    print(f"  {tag}: 均值 {pat.mean():.1f}  std {pat.std():.1f}  → {prefix}{tag}_phase.npy",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", "W1", "W2", "W3"])
    ap.add_argument("--save-prefix", default="patterns/W")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ring-mm", type=float, default=0.78)
    ap.add_argument("--f0", type=float, default=0.01)
    ap.add_argument("--jitter", type=float, default=1.0, help="W2 叠加随机相位的幅度（×π）")
    args = ap.parse_args()

    todo = ["W1", "W2", "W3"] if args.which == "all" else [args.which]
    print(f"面板 {W}x{H}，p={P*1e6:.0f} um，s={S*1e3:.0f} mm", flush=True)
    for tag in todo:
        if tag == "W1":
            pat = w1_stripes(args.seed)
        elif tag == "W2":
            pat = w2_axicon(ring_mm=args.ring_mm, jitter=args.jitter, seed=args.seed)
            print(f"  W2 环半径 {args.ring_mm} mm（远场）", flush=True)
        else:
            pat = w3_kolmogorov(args.f0, args.seed)
        save(pat, args.save_prefix, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
