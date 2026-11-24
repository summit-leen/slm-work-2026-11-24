# -*- coding: utf-8 -*-
"""用版本 B（420 枚随机透镜阵列）做图像重建的仿真与反演。

前向模型（非相干、每枚透镜各自成像后叠加）：
    对第 l 枚透镜：物面正前方尺寸 ≈ 口径 a_l 的一小块，
    以放大率 m_l = s/s_l 投到靶面，再与它的离焦+衍射核 h_l 卷积，
    最后把 420 个结果叠到传感器上。

    y = Σ_l  C_l^T [ ( R_{m_l} C_l x ) ∗ h_l ]

其中 C_l 是"取第 l 枚透镜对应的物面小块"(含边界零填充)，R 是缩放，
C_l^T 是把该小块按位置写回（重叠区相加）。反演用 FISTA + 各向同性 TV。

用法：
    python code/mla_recon.py --d 200 --obj bars --iters 120
    python code/mla_recon.py --d 200 --obj arrow --noise 2 --tv 0.02
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PITCH_MM = 18e-3          # SLM 像素 18 µm
LAM_MM = 620e-6           # 620 nm
SENSOR_PX_MM = 2e-3       # 相机像元 2 µm


# ---------------------------------------------------------------- 物面
def make_object(kind: str, n: int, fov_mm: float) -> np.ndarray:
    """在 n×n 网格上画测试物（fov_mm 见方），返回 [0,1] 透射率。"""
    px = fov_mm / n
    y, x = (np.mgrid[0:n, 0:n] + 0.5) * px - fov_mm / 2
    img = np.zeros((n, n), np.float64)

    if kind == "bars":                      # 不同宽度的条纹（分辨率靶）
        widths = [0.60, 0.40, 0.28, 0.20, 0.14]   # mm
        y0 = -fov_mm / 2 + 0.6
        for i, wdt in enumerate(widths):
            for k in range(3):
                yc = y0 + i * 0.9 + k * 2 * wdt
                img[(np.abs(y - yc) < wdt / 2) & (np.abs(x) < fov_mm / 3)] = 1
    elif kind == "arrow":                   # 箭头（他们的标定卡）
        img[(np.abs(y) < 0.25) & (x > -1.2) & (x < 0.9)] = 1
        img[(x > 0.55) & (x < 1.15) & (np.abs(y) < 0.75 - (x - 0.55) * 0.9)] = 1
    elif kind == "dots":                    # 点阵
        for yy in (-1.2, 0.0, 1.2):
            for xx in (-1.2, 0.0, 1.2):
                img[(x - xx) ** 2 + (y - yy) ** 2 < 0.18 ** 2] = 1
    elif kind == "slit":                    # 单缝 + 一组窄缝
        img[(np.abs(x - 1.6) < 0.15) & (np.abs(y) < 1.6)] = 1
        for k, wdt in enumerate((0.30, 0.20, 0.12)):
            img[(np.abs(y - (-1.0 + 1.0 * k)) < wdt / 2) & (np.abs(x + 1.4) < 0.8)] = 1
    else:
        raise SystemExit(f"未知测试物 {kind}")
    return img


# ---------------------------------------------------------------- 透镜参数
def load_lenses(csv_path: str, n_obj: int, fov_mm: float,
                s_mm: float, d_mm: float):
    """读 B 的 csv，换算成物面像素坐标、口径、放大率、PSF。"""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    cx = np.array([float(r["cx_px"]) for r in rows])
    cy = np.array([float(r["cy_px"]) for r in rows])
    rr = np.array([float(r["r_px"]) for r in rows])
    ff = np.array([float(r["f_mm"]) for r in rows])

    a_mm = 2 * rr * PITCH_MM                    # 口径（mm）
    sl = ff * d_mm / (d_mm - ff)                # 像距（mm）
    m = s_mm / sl                                # 放大率
    geo = a_mm * np.abs(1 - m)                  # 几何弥散（mm）
    dif = 2.44 * LAM_MM * sl / a_mm             # 衍射斑（mm）
    blur = np.hypot(geo, dif)                   # 总弥散（mm）

    # SLM 面板坐标 → 物面坐标（阵列与物面 1:1，面板中心对物面中心）
    panel_mm = 1024 * PITCH_MM
    obj_px = fov_mm / n_obj
    lx = (cx * PITCH_MM - panel_mm / 2) / obj_px + n_obj / 2
    ly = (cy * PITCH_MM - 768 * PITCH_MM / 2) / obj_px + n_obj / 2

    return dict(cx=lx, cy=ly, a=a_mm, m=m, blur=blur, f=ff, sl=sl, r=rr)


# ---------------------------------------------------------------- 前向/伴随
def _bilinear(p, q):
    """(q,p) 双线性采样矩阵：把 p 长的向量重采样成 q 长。"""
    R = np.zeros((q, p))
    pos = (np.arange(q) + 0.5) * p / q - 0.5
    i0 = np.floor(pos).astype(int)
    f = pos - i0
    for j in range(q):
        a = int(np.clip(i0[j], 0, p - 1))
        b = int(np.clip(i0[j] + 1, 0, p - 1))
        R[j, a] += 1 - f[j]
        R[j, b] += f[j]
    return R


class MLAOperator:
    """y = A x：每枚透镜取出物面正前方口径大小的一块，按放大率重采样到靶面像素，
    与它的离焦+衍射核卷积后放回靶面位置；A^T 为严格转置。"""

    def __init__(self, lens, n_obj, fov_mm, sensor_shape, sensor_px_mm,
                 psf_cap_px=71):
        self.n_obj = n_obj
        self.fov = fov_mm
        self.sensor = sensor_shape
        self.spx = sensor_px_mm
        self.obj_px = fov_mm / n_obj
        self.patch = []
        H, W = sensor_shape
        for k in range(len(lens["cx"])):
            a, m = lens["a"][k], lens["m"][k]
            p = int(np.clip(round(a / self.obj_px), 3, n_obj))
            q = int(np.clip(round(a * m / self.spx), 2, min(H, W)))
            R = _bilinear(p, q)
            sx = (lens["cx"][k] - n_obj / 2) * self.obj_px / self.spx + W / 2
            sy = (lens["cy"][k] - n_obj / 2) * self.obj_px / self.spx + H / 2
            sigma = lens["blur"][k] / self.spx / 2.355
            ks = int(np.clip(6 * sigma + 1, 1, psf_cap_px))
            if ks % 2 == 0:
                ks += 1
            h = _gauss_kernel(ks, max(sigma, 0.6))
            self.patch.append(dict(cx=lens["cx"][k], cy=lens["cy"][k],
                                   p=p, q=q, R=R, sx=sx, sy=sy, h=h,
                                   a2=a * a))

    def forward(self, x):
        H, W = self.sensor
        y = np.zeros((H, W))
        for it in self.patch:
            blk, _ = self._crop(x, it["cy"], it["cx"], it["p"])
            blk = it["R"] @ blk @ it["R"].T            # p×p -> q×q
            blk = fftconvolve(blk, it["h"], mode="same")
            y += _place(blk, it["sy"], it["sx"], H, W) * it["a2"]
        return y

    def adjoint(self, y):
        x = np.zeros((self.n_obj, self.n_obj))
        for it in self.patch:
            blk = _take(y, it["sy"], it["sx"], it["q"])
            blk = fftconvolve(blk, it["h"], mode="same")   # 对称核
            blk = it["R"].T @ blk @ it["R"]                # q×q -> p×p
            _add_crop(x, blk, it["cy"], it["cx"], it["a2"])
        return x

    @staticmethod
    def _crop(x, cy, cx, p):
        n = x.shape[0]
        out = np.zeros((p, p))
        y0, x0 = int(round(cy - p / 2)), int(round(cx - p / 2))
        ys, ye = max(0, y0), min(n, y0 + p)
        xs, xe = max(0, x0), min(n, x0 + p)
        if ys < ye and xs < xe:
            out[ys - y0:ye - y0, xs - x0:xe - x0] = x[ys:ye, xs:xe]
        return out, (y0, x0)


def _gauss_kernel(k, sigma):
    ax = (np.arange(k) - (k - 1) / 2)
    g = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _place(blk, sy, sx, H, W):
    out = np.zeros((H, W), blk.dtype)
    p = blk.shape[0]
    y0, x0 = int(round(sy - p / 2)), int(round(sx - p / 2))
    ys, ye = max(0, y0), min(H, y0 + p)
    xs, xe = max(0, x0), min(W, x0 + p)
    if ys < ye and xs < xe:
        out[ys:ye, xs:xe] = blk[ys - y0:ye - y0, xs - x0:xe - x0]
    return out


def _take(y, sy, sx, p):
    H, W = y.shape
    y0, x0 = int(round(sy - p / 2)), int(round(sx - p / 2))
    out = np.zeros((p, p), y.dtype)
    ys, ye = max(0, y0), min(H, y0 + p)
    xs, xe = max(0, x0), min(W, x0 + p)
    if ys < ye and xs < xe:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = y[ys:ye, xs:xe]
    return out


def _fit(blk, p):
    if blk.shape[0] == p:
        return blk
    return ndimage.zoom(blk, p / blk.shape[0], order=1,
                        output_shape=(p, p))


def _add_crop(x, blk, cy, cx, w):
    n = x.shape[0]
    p = blk.shape[0]
    y0, x0 = int(round(cy - p / 2)), int(round(cx - p / 2))
    ys, ye = max(0, y0), min(n, y0 + p)
    xs, xe = max(0, x0), min(n, x0 + p)
    if ys < ye and xs < xe:
        x[ys:ye, xs:xe] += w * blk[ys - y0:ye - y0, xs - x0:xe - x0]


# ---------------------------------------------------------------- 反演
def _grad(u):
    gy = np.zeros_like(u); gx = np.zeros_like(u)
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    return gy, gx


def _div(gy, gx):
    out = np.zeros_like(gy)
    out[:-1, :] -= gy[:-1, :]
    out[1:, :] += gy[:-1, :]
    out[:, :-1] -= gx[:, :-1]
    out[:, 1:] += gx[:, :-1]
    return out


def _tv_val_grad(u, eps=1e-3):
    gy, gx = _grad(u)
    mag = np.sqrt(gy * gy + gx * gx + eps * eps)
    val = mag.sum()
    gy, gx = gy / mag, gx / mag
    return val, _div(gy, gx)


def operator_norm(A, n, iters=15):
    v = np.random.default_rng(0).standard_normal((n, n))
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = A.adjoint(A.forward(v))
        lam = np.linalg.norm(w)
        if lam == 0:
            return 1.0
        v = w / lam
    return lam


def _dtv(qy, qx):
    """梯度算子的转置 Dᵀ（与 _grad 的 forward difference 严格配对）。"""
    out = np.zeros_like(qy)
    out[:-1, :] -= qy[:-1, :]
    out[1:, :] += qy[:-1, :]
    out[:, :-1] -= qx[:, :-1]
    out[:, 1:] += qx[:, :-1]
    return out


def pdhg(A, y, iters, lam, n, verbose=False):
    """min 0.5||Ax-y||^2 + lam*TV(x) + i_[0,1](x)，Chambolle-Pock 原-对偶。

    不再对 TV 做 eps 平滑，因此步长只由 ||K||^2 = ||A||^2 + 8 决定
    （K = [A; D]），不会被 eps 的 Lipschitz 常数拖死。
    """
    na2 = operator_norm(A, n)
    kn = np.sqrt(na2 + 8.0)
    tau = 0.99 / kn
    sig = 0.99 / kn
    x = np.zeros((n, n))
    xb = x.copy()
    p = np.zeros_like(y)
    qy = np.zeros((n, n))
    qx = np.zeros((n, n))
    for k in range(iters):
        # 对偶上升：数据项 + 梯度场（球投影）
        p = (p + sig * (A.forward(xb) - y)) / (1.0 + sig)
        gy, gx = _grad(xb)
        qy = qy + sig * gy
        qx = qx + sig * gx
        mag = np.sqrt(qy * qy + qx * qx)
        fac = np.minimum(1.0, lam / np.maximum(mag, 1e-12))
        qy = qy * fac
        qx = qx * fac
        # 原始下降
        g = A.adjoint(p) + _dtv(qy, qx)
        xn = np.clip(x - tau * g, 0, 1)
        xb = 2.0 * xn - x
        x = xn
        if verbose and (k % 20 == 0 or k == iters - 1):
            print(f"   iter {k:4d}  |r|={np.linalg.norm(A.forward(x) - y):.4g}",
                  flush=True)
    return x


def _rl_deconv(obs, psf, iters=25):
    """Richardson-Lucy 去卷积（非负）。psf 已归一化。"""
    obs = np.clip(obs, 1e-9, None)
    est = np.full_like(obs, max(obs.mean(), 1e-9))
    tpsf = psf[::-1, ::-1]
    for _ in range(iters):
        conv = fftconvolve(est, psf, mode="same")
        est = est * fftconvolve(obs / np.maximum(conv, 1e-9), tpsf, mode="same")
        est = np.clip(est, 0.0, None)
    return est


def mosaic_recon(A, y, rl_iters=25, gain=False):
    """拼接式重建：每枚透镜的子图单独去卷积，再按 1:1 拼回物面（重叠区取平均）。

    这是透镜阵列成像的标准做法，比全局反演稳健得多：
    每个子图是物面一小块与本透镜核的卷积，去卷积后直接放回去即可。
    注意子图相对物面是倒像（放大率符号为负），整体结果需旋转 180°。
    """
    x = np.zeros((A.n_obj, A.n_obj))
    w = np.zeros_like(x)
    ones = np.ones((A.n_obj, A.n_obj))
    for it in A.patch:
        a2 = it["a2"]
        if a2 <= 0:
            continue
        blk = _take(y, it["sy"], it["sx"], it["q"])
        peak = blk.max()
        if peak <= 0:
            continue
        blk = blk / peak                      # 每枚透镜只有相对强度有意义
        est = _rl_deconv(blk, it["h"], rl_iters)
        est = est / max(est.max(), 1e-9)
        blk_p = it["R"].T @ est @ it["R"]     # q×q -> p×p
        _add_crop(x, blk_p, it["cy"], it["cx"], 1.0)
        _add_crop(w, np.ones_like(blk_p), it["cy"], it["cx"], 1.0)
    out = x / np.maximum(w, 1e-9)
    out[w < 0.5] = 0.0
    return out


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="patterns/B_rla420_lenses.csv")
    ap.add_argument("--d", type=float, default=200.0, help="物面到面板 mm")
    ap.add_argument("--s", type=float, default=50.0, help="面板到传感器 mm")
    ap.add_argument("--fov", type=float, default=4.2, help="物方视场 mm")
    ap.add_argument("--n", type=int, default=192, help="物面网格")
    ap.add_argument("--sensor-bin", type=float, default=8.0, help="传感器 bin")
    ap.add_argument("--obj", default="bars")
    ap.add_argument("--noise", type=float, default=0.0, help="噪声相对强度")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tv", type=float, default=0.02)
    ap.add_argument("--method", default="mosaic", choices=["mosaic", "pdhg"])
    ap.add_argument("--rl", type=int, default=25, help="mosaic 每枚透镜的 RL 迭代")
    ap.add_argument("--out", default="captures/sim_B")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    obj = make_object(args.obj, args.n, args.fov)
    lens = load_lenses(args.csv, args.n, args.fov, args.s, args.d)
    spx = SENSOR_PX_MM * args.sensor_bin
    H, W = int(2178 / args.sensor_bin), int(3860 / args.sensor_bin)
    A = MLAOperator(lens, args.n, args.fov, (H, W), spx)

    y = A.forward(obj)
    peak = np.percentile(y, 99.9)
    y = y / max(peak, 1e-12)
    if args.noise > 0:
        y = y + args.noise * np.random.default_rng(0).standard_normal(y.shape) * 0.02
    print(f"前向完成：{len(lens['cx'])} 枚透镜，传感器 {H}×{W}，"
          f"信号 99.9% {peak:.3g}，用时 {time.time()-t0:.1f}s", flush=True)

    if args.method == "mosaic":
        xr = mosaic_recon(A, y, rl_iters=args.rl)
        xr = np.rot90(xr, 2)                  # 透镜成倒像
    else:
        xr = pdhg(A, y, args.iters, args.tv, args.n, verbose=args.verbose)
        xr = np.rot90(xr, 2)
    print(f"反演（{args.method}）用时 {time.time()-t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(f"{args.out}_obj.npy", obj)
    np.save(f"{args.out}_meas.npy", y)
    np.save(f"{args.out}_rec.npy", xr)

    psnr = 10 * np.log10(obj.var() / max(np.mean((xr - obj) ** 2), 1e-9))
    print(f"PSNR {psnr:.2f} dB")
    _save_views(obj, y, xr, args.out)
    return 0


def _save_views(obj, y, xr, prefix):
    from PIL import Image
    def nz(a):
        lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
        return np.clip((a - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(nz(obj)).save(f"{prefix}_obj.png")
    Image.fromarray(nz(y)).save(f"{prefix}_meas.png")
    Image.fromarray(nz(xr)).save(f"{prefix}_rec.png")


if __name__ == "__main__":
    raise SystemExit(main())
