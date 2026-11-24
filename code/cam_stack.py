# -*- coding: utf-8 -*-
"""不碰 SLM，只抓一组相机帧并保存均值 / 首帧 / 统计。

    python code/cam_stack.py --exp 10000 --n 16 --out captures/x200_on
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from pypylon import pylon


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=float, default=10000, help="曝光 us")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--gain", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=0.0, help=">0 时限帧率")
    ap.add_argument("--out", default="captures/frame")
    args = ap.parse_args()

    tl = pylon.TlFactory.GetInstance()
    devs = tl.EnumerateDevices()
    if not devs:
        print("没有发现相机")
        return 1
    c = pylon.InstantCamera(tl.CreateDevice(devs[0]))
    c.Open()
    c.PixelFormat.Value = "Mono8"
    c.Width.Value = c.WidthMax.Value
    c.Height.Value = c.HeightMax.Value
    c.ExposureAuto.Value = "Off"
    c.GainAuto.Value = "Off"
    c.Gain.Value = args.gain
    c.ExposureTime.Value = float(args.exp)
    if args.fps > 0:
        try:
            c.AcquisitionFrameRateEnable.Value = True
            c.AcquisitionFrameRate.Value = float(args.fps)
        except Exception:
            pass

    fr = []
    c.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    t0 = time.time()
    while len(fr) < args.n and time.time() - t0 < 30:
        r = c.RetrieveResult(3000, pylon.TimeoutHandling_ThrowException)
        if r.GrabSucceeded():
            fr.append(r.GetArray().copy())
        r.Release()
    c.StopGrabbing()
    c.Close()

    if not fr:
        print("没抓到帧")
        return 1
    st = np.stack(fr).astype(np.float32)
    mean = st.mean(axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out + "_mean.npy", mean.astype(np.float32))
    np.save(args.out + "_first.npy", st[0].astype(np.uint8))
    noise = float(st.std(axis=0).mean())
    print(f"帧 {len(fr)} 张，曝光 {args.exp:.0f} us，shape {st.shape[1:]}")
    print(f"均值图 mean {mean.mean():.2f}  p99.9 {np.percentile(mean,99.9):.1f}  max {mean.max():.1f}")
    print(f"单帧饱和 {(st >= 255).mean()*100:.4f}%   帧间噪声 std {noise:.2f}")
    print(f"已存 {args.out}_mean.npy / _first.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
