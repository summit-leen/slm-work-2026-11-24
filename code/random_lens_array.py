# -*- coding: utf-8 -*-
"""随机透镜阵列相位图：每枚微透镜位置随机、焦距随机（默认 30-50 mm）。

和参考图一样是"多枚菲涅尔透镜铺成阵列"的相位图，可以直接显示到 SLM 上。
每枚透镜的相位
    phi(r) = -pi r^2 / (lambda f)     (mod 2pi)
再按 8 bit 映射成灰度（gray = phi_mod/(2pi)*255），面板上 gray=255 只有
alpha = Gamma/lambda = 0.645 个周期（620 nm，Gamma=400 nm），这是面板的物理
限制，不是本脚本的选择：焦点效率 sinc^2(1-alpha) ≈ 65%，另有约 20% 零级直穿。

两处物理约束（脚本会自动满足，不是可选装饰）
--------------------------------------------
1. 采样极限：单像素相位差 <= pi，否则环带欠采样、图案混叠。
       r <= r_alias = lambda*f/(2p)
   18 um / 620 nm / f=40 mm 时 r_alias ≈ 38 px（直径 76 px）。所以短焦透镜
   的口径必然小，脚本会把每枚透镜的半径夹到 alias_frac*r_alias 以内。
2. 透镜之间不重叠（可选 --overlap 允许交叠，交叠处相位相加再取模）。

用法
----
    python random_lens_array.py --diag                  # 只算，打印参数表
    python random_lens_array.py --show --hold 30        # 显示到 SLM 30 s
    python random_lens_array.py --save rla --show       # 存 npy/png/csv 并显示
    python random_lens_array.py --n 80 --f-min 30 --f-max 50 --seed 7
    python random_lens_array.py --tilt 0.1 --show       # 叠棱镜，把焦斑挪开零级
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

PITCH_UM = 18.0
LAM_NM = 620.0
GAMMA_NM = 400.0

# 不显示时用的面板尺寸（本面板 DISPLAY2 原生 1024x768；有 --w/--h 或 --show 时会覆盖）
PANEL_W_DEFAULT = 1024
PANEL_H_DEFAULT = 768

# wx 收尾顺序的坑：SLM 对象（持有 wx.App 与 frame）如果在 main() 返回时就被
# 析构，解释器退出阶段会以 127 退出（图案正常显示、诊断也 PASS，但退出码是错的）。
# 用模块级引用把它留到解释器收尾，就能正常 exit 0。相同问题也存在于 slm_lens.py。
_KEEPALIVE = None


# --------------------------------------------------------------------------- #
# 布局
# --------------------------------------------------------------------------- #
class Lens:
    __slots__ = ("cx", "cy", "r", "f", "zones", "p")

    def __init__(self, cx, cy, r, f, p):
        self.cx, self.cy, self.r, self.f = cx, cy, r, f
        self.p = p          # 像素间距 m
        self.zones = 0.0

    @property
    def as_dict(self):
        return dict(cx=self.cx, cy=self.cy, r=self.r, f=self.f, zones=self.zones)


def build_layout(width, height, args, rng):
    """随机序列放置（RSA）：先随机焦距 -> 由采样极限定最大半径 -> 再随机半径/位置。"""
    p = args.pitch * 1e-6
    lam = args.lam * 1e-9
    margin = args.margin
    lenses: list[Lens] = []
    tries = 0
    max_tries = max(20000, args.n * 3000)

    r_cap = 0.5 * min(width, height) - margin
    # --area-mm > 0 时把所有透镜约束在半径 area_r 的圆内（成像/拼接用）
    area_r = 0.5 * args.area_mm * 1e-3 / p if args.area_mm > 0 else 0.0
    area_cx = 0.5 * (width - 1.0) if args.cx is None else args.cx
    area_cy = 0.5 * (height - 1.0) if args.cy is None else args.cy
    while len(lenses) < args.n and tries < max_tries:
        tries += 1
        f = rng.uniform(args.f_min, args.f_max) * 1e-3
        r_alias = lam * f / (2.0 * p) / p               # px
        r_lim = min(args.alias_frac * r_alias, args.r_max, r_cap)
        if r_lim <= args.r_min + 1.0:
            continue
        # 半径分布：size_skew<1 偏大，>1 偏小；默认略偏大，画面更像参考图
        u = rng.random()
        r = args.r_min + (r_lim - args.r_min) * (u ** args.size_skew)
        if area_r > 0:
            if r + margin >= area_r:
                continue
            rr = math.sqrt(rng.random()) * (area_r - r - margin)
            th = rng.random() * 2.0 * math.pi
            cx = area_cx + rr * math.cos(th)
            cy = area_cy + rr * math.sin(th)
        else:
            if 2.0 * r + 2.0 * margin >= min(width, height):
                continue
            cx = rng.uniform(r + margin, width - 1.0 - r - margin)
            cy = rng.uniform(r + margin, height - 1.0 - r - margin)

        ok = True
        for L in lenses:
            dx, dy = cx - L.cx, cy - L.cy
            if math.hypot(dx, dy) < r + L.r + args.gap:
                ok = False
                break
        if not ok:
            continue
        L = Lens(cx, cy, r, f, p)
        L.zones = (r * p) ** 2 / (2.0 * lam * f)
        lenses.append(L)
    if len(lenses) < args.n:
        print(f"[warn] 只放下 {len(lenses)}/{args.n} 枚（版面/间距不够），"
              f"可减小 --n、--r-min 或 --gap", file=sys.stderr)
    return lenses


# --------------------------------------------------------------------------- #
# 相位图
# --------------------------------------------------------------------------- #
def build_pattern(width, height, lenses, args):
    p = args.pitch * 1e-6
    lam = args.lam * 1e-9
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    phase = np.zeros((height, width), dtype=np.float64)

    for L in lenses:
        dx = (xx - L.cx) * p
        dy = (yy - L.cy) * p
        r2 = dx * dx + dy * dy
        inside = r2 <= (L.r * p) ** 2
        phase[inside] += args.sign * math.pi * r2[inside] / (lam * L.f)

    qx, qy = args.tilt
    if qx or qy:
        phase += 2.0 * math.pi * (qx * (xx - (width - 1) / 2.0) +
                                  qy * (yy - (height - 1) / 2.0))

    gray = np.rint(np.mod(phase, 2.0 * math.pi) / (2.0 * math.pi) * 255.0)
    return np.clip(gray, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 诊断
# --------------------------------------------------------------------------- #
def describe(width, height, lenses, args) -> str:
    p = args.pitch * 1e-6
    lam = args.lam * 1e-9
    alpha = min(1.0, args.gamma / args.lam)
    etas = math.sin(math.pi * (1 - alpha)) / (math.pi * (1 - alpha))
    eta0 = math.sin(math.pi * alpha) / (math.pi * alpha)
    if not lenses:
        return "没有放下的透镜"
    fs = np.array([L.f for L in lenses]) * 1e3
    rs = np.array([L.r for L in lenses])
    zs = np.array([L.zones for L in lenses])
    edge = np.array([L.r * p * p / (lam * L.f) for L in lenses])
    lines = [
        f"面板        {width}x{height}, 像素间距 {args.pitch:g} um, "
        f"全口径 {min(width, height) * p * 1e3:.1f} mm",
        f"波长/延迟   {args.lam:g} nm, 行程 {args.gamma:g} nm -> alpha = {alpha:.3f} 周期",
        f"透镜数量    {len(lenses)}  (目标 {args.n})",
        f"焦距        {fs.min():.1f} .. {fs.max():.1f} mm (设计=实际)",
        f"口径半径    {rs.min():.0f} .. {rs.max():.0f} px  "
        f"(直径 {2 * rs.min():.0f} .. {2 * rs.max():.0f} px, "
        f"{2 * rs.min() * p * 1e3:.2f} .. {2 * rs.max() * p * 1e3:.2f} mm)",
        f"菲涅尔环数  {zs.min():.1f} .. {zs.max():.1f} 个周期",
        f"边缘采样    {edge.max():.3f} 周期/像素 (上限 0.5)",
        f"覆盖面积    {np.sum(np.pi * rs ** 2) / (width * height) * 100:.1f}%",
        "",
        f"焦点效率    {etas ** 2 * 100:.0f}%   (sinc^2(1-alpha))",
        f"零级直穿    {eta0 ** 2 * 100:.0f}%   (不随 f 移动的亮斑)",
        f"焦斑艾里半径  ~{1.22 * lam * np.mean(fs) * 1e-3 / (2 * rs.mean() * p) * 1e6:.1f} um",
    ]
    if args.tilt[0] or args.tilt[1]:
        q = math.hypot(*args.tilt)
        lines.append(f"离轴倾斜    q = ({args.tilt[0]:g}, {args.tilt[1]:g}) -> "
                     f"偏折 {q * lam / p * 1e3:.2f} mrad")
    return "\n".join(lines)


def write_csv(path, lenses, args):
    p = args.pitch * 1e-6
    lam = args.lam * 1e-9
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "cx_px", "cy_px", "r_px", "d_px", "f_mm",
                    "f_number", "zones", "edge_cycles_per_px"])
        for i, L in enumerate(lenses):
            w.writerow([i, f"{L.cx:.2f}", f"{L.cy:.2f}", f"{L.r:.2f}",
                        f"{2 * L.r:.2f}", f"{L.f * 1e3:.2f}",
                        f"{L.f / (2 * L.r * p):.1f}", f"{L.zones:.2f}",
                        f"{L.r * p * p / (lam * L.f):.4f}"])


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def verify(width, height, lenses, gray, args):
    """不依赖光路的数值自检：布局合法 + 相位图与解析式一致 + 不混叠。"""
    p = args.pitch * 1e-6
    lam = args.lam * 1e-9
    checks = []

    inside = all(L.cx - L.r >= 0 and L.cy - L.r >= 0 and
                 L.cx + L.r <= width - 1 and L.cy + L.r <= height - 1
                 for L in lenses)
    checks.append(("透镜全部落在面板内", inside, f"{len(lenses)} 枚"))

    dmin = float("inf")
    for i in range(len(lenses)):
        for j in range(i + 1, len(lenses)):
            a, b = lenses[i], lenses[j]
            d = math.hypot(a.cx - b.cx, a.cy - b.cy) - (a.r + b.r)
            dmin = min(dmin, d)
    checks.append(("透镜不重叠", dmin >= args.gap - 1e-6,
                   f"最小边缘间距 {dmin:.2f} px (要求 >= {args.gap:g})"))

    edge = max(L.r * p * p / (lam * L.f) for L in lenses)
    checks.append(("单像素相位差 <= pi", edge <= 0.5,
                   f"最大 {edge:.3f} 周期/像素 (上限 0.500)"))

    fs = [L.f * 1e3 for L in lenses]
    checks.append(("焦距在设计区间内",
                   min(fs) >= args.f_min - 1e-6 and max(fs) <= args.f_max + 1e-6,
                   f"{min(fs):.2f} .. {max(fs):.2f} mm"))

    # 相位图与解析式逐点比对（随机抽 400 个透镜内像素）
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(400):
        L = lenses[int(rng.integers(len(lenses)))]
        while True:
            dx = rng.uniform(-1, 1) * L.r
            dy = rng.uniform(-1, 1) * L.r
            if dx * dx + dy * dy <= L.r ** 2:
                break
        ix = int(round(L.cx + dx))
        iy = int(round(L.cy + dy))
        if not (0 <= ix < width and 0 <= iy < height):
            continue
        r2 = ((ix - L.cx) * p) ** 2 + ((iy - L.cy) * p) ** 2
        if r2 > (L.r * p) ** 2:
            continue          # 取整后落到口径外，那里 gray=0，不参与比对
        ref = np.mod(args.sign * math.pi * r2 / (lam * L.f), 2 * math.pi)
        got = gray[iy, ix] / 255.0 * 2 * math.pi
        d = abs(np.angle(np.exp(1j * (ref - got))))
        d = min(d, 2 * math.pi - d)
        err = max(err, math.degrees(d))
    checks.append(("相位图 = 解析式", err < 1.0, f"最大偏差 {err:.2f} deg (1 LSB)"))

    return checks


# --------------------------------------------------------------------------- #
# 预览图
# --------------------------------------------------------------------------- #
def render_aa(width, height, lenses, args, ss=4):
    """预览用抗锯齿渲染：在 ss 倍网格上取复振幅 exp(i*phi)，按像素做盒式积分
    （等价于一个像素内的场平均），再取相位映射到 8 bit。

    只用于看图。SLM 实际显示的是 build_pattern() 的 1:1 8 bit 图案：
    相位锯齿在 1-2 px 宽的环带上必然是像素化的，这是面板采样极限，不是渲染问题。
    """
    p = args.pitch * 1e-6
    lam = args.lam * 1e-9
    Wp, Hp = width * ss, height * ss
    acc = np.zeros((Hp, Wp), dtype=np.complex128)
    cov = np.zeros((Hp, Wp), dtype=np.float32)
    qx, qy = args.tilt
    for L in lenses:
        x0 = int(max(0, np.floor((L.cx - L.r) * ss)))
        x1 = int(min(Wp, np.ceil((L.cx + L.r) * ss) + 1))
        y0 = int(max(0, np.floor((L.cy - L.r) * ss)))
        y1 = int(min(Hp, np.ceil((L.cy + L.r) * ss) + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        xf, yf = xx / ss, yy / ss
        dx = (xf - L.cx) * p
        dy = (yf - L.cy) * p
        r2 = dx * dx + dy * dy
        inside = r2 <= (L.r * p) ** 2
        phi = args.sign * math.pi * r2 / (lam * L.f)
        if qx or qy:
            phi = phi + 2.0 * math.pi * (qx * (xf - (width - 1) / 2.0) +
                                         qy * (yf - (height - 1) / 2.0))
        e = np.exp(1j * phi)
        sub = acc[y0:y1, x0:x1]
        sub[inside] = e[inside]
        cv = cov[y0:y1, x0:x1]
        cv[inside] = 1.0
    A = acc.reshape(height, ss, width, ss).sum(axis=(1, 3))
    C = cov.reshape(height, ss, width, ss).sum(axis=(1, 3))
    out = np.zeros((height, width), dtype=np.float64)
    m = C > 0
    out[m] = np.mod(np.angle(A[m]), 2.0 * math.pi) / (2.0 * math.pi) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def save_previews(gray, lenses, path_prefix, args):
    from PIL import Image

    Image.fromarray(gray, mode="L").save(path_prefix + "_phase.png")

    # 抗锯齿渲染 -> 放大到参考图尺寸 (1448x1086 = 1024x768 的 1.414 倍)
    aa = render_aa(gray.shape[1], gray.shape[0], lenses, args)
    up = Image.fromarray(aa, mode="L").resize((1448, 1086), Image.BILINEAR)
    up.save(path_prefix + "_preview.png")



# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="随机透镜阵列相位图 (SLM)")
    ap.add_argument("--n", type=int, default=80, help="透镜数量 (默认 80)")
    ap.add_argument("--f-min", type=float, default=30.0, help="最短焦距 mm")
    ap.add_argument("--f-max", type=float, default=50.0, help="最长焦距 mm")
    ap.add_argument("--r-min", type=float, default=9.0, help="最小口径半径 px")
    ap.add_argument("--r-max", type=float, default=60.0, help="最大口径半径 px")
    ap.add_argument("--alias-frac", type=float, default=0.95,
                    help="半径取采样极限 lambda*f/(2p) 的多少倍 (<=1)")
    ap.add_argument("--size-skew", type=float, default=0.75,
                    help="半径分布指数：<1 偏大, >1 偏小")
    ap.add_argument("--gap", type=float, default=4.0, help="透镜边缘最小间隔 px")
    ap.add_argument("--margin", type=float, default=6.0, help="距面板边缘 px")
    ap.add_argument("--area-mm", type=float, default=0.0,
                    help="把透镜约束在直径 area-mm 的圆内（0 = 铺满整块面板）")
    ap.add_argument("--cx", type=float, default=None, help="圆的中心 x px（默认面板中心）")
    ap.add_argument("--cy", type=float, default=None, help="圆的中心 y px（默认面板中心）")
    ap.add_argument("--overlap", action="store_true", help="允许透镜交叠")
    ap.add_argument("--pitch", type=float, default=PITCH_UM, help="像素间距 um")
    ap.add_argument("--lam", type=float, default=LAM_NM, help="波长 nm")
    ap.add_argument("--gamma", type=float, default=GAMMA_NM, help="液晶延迟 nm")
    ap.add_argument("--sign", type=int, default=-1, choices=(-1, 1),
                    help="-1 会聚 (默认), +1 发散")
    ap.add_argument("--tilt", type=float, nargs=2, default=(0.0, 0.0),
                    metavar=("QX", "QY"), help="整幅叠线性相位，单位 周期/像素")
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--w", type=int, default=None, help="面板宽 (默认取 SLM)")
    ap.add_argument("--h", type=int, default=None, help="面板高 (默认取 SLM)")
    ap.add_argument("--monitor", type=int, default=1)
    ap.add_argument("--save", default=None, help="输出前缀，存 _phase.npy/_phase.png/_lenses.csv")
    ap.add_argument("--diag", action="store_true", help="只打印诊断，不显示")
    ap.add_argument("--show", action="store_true", help="显示到 SLM")
    ap.add_argument("--hold", type=float, default=30.0, help="显示秒数 (默认 30)")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    W, H = args.w, args.h
    slm = None
    if args.show:
        from slm_driver import SLM
        slm = SLM(monitor=args.monitor, verbose=False)
        sw, sh = slm.getSize()
        W = W or sw
        H = H or sh
    else:
        # 只算不显示时不开 SLM 窗口，避免黑屏闪一下
        W = W or PANEL_W_DEFAULT
        H = H or PANEL_H_DEFAULT

    lenses = build_layout(W, H, args, rng)
    gray = build_pattern(W, H, lenses, args)

    print(describe(W, H, lenses, args))
    checks = verify(W, H, lenses, gray, args)
    print("\n自检")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<22s} {detail}")
    print(f"\n图案: {W}x{H} uint8, 灰度 {gray.min()}..{gray.max()}, 均值 {gray.mean():.1f}")

    if args.save:
        base = args.save
        d = os.path.dirname(os.path.abspath(base))
        if d:
            os.makedirs(d, exist_ok=True)
        np.save(base + "_phase.npy", gray)
        write_csv(base + "_lenses.csv", lenses, args)
        save_previews(gray, lenses, base, args)
        print(f"已保存 {base}_phase.npy / _phase.png / _preview.png / _lenses.csv")

    if args.show and slm is not None:
        global _KEEPALIVE
        _KEEPALIVE = slm  # 留到解释器收尾，否则退出码 127（见文件头）
        print(f"\n显示 {args.hold:g} s ... (Ctrl+C 或关窗口结束)")
        slm.update(gray)
        slm.run(hold=args.hold)
        print("结束")
    elif slm is not None:
        slm.close()
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
