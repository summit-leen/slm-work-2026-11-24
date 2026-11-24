# -*- coding: utf-8 -*-
"""Show test patterns on the SLM (extended desktop, second screen).

Examples
--------
    python slm_show.py --info                     # list displays, mark the SLM
    python slm_show.py frame --hold 20            # alignment frame for 20 s
    python slm_show.py grating --period 16 --hold 30
    python slm_show.py checker --hold 10          # 1-pixel checkerboard = 1:1 test
    python slm_show.py --file phase.npy --hold 30 # your own uint8 array
    python slm_show.py grating --bench 120        # measure achievable frame rate

Patterns: white, black, grating, checker, comb, blaze, frame.
Add --legacy to use slmpy instead of slm_driver (slmpy is noisy on Windows).
"""
import argparse
import os
import sys
import threading
import time

import numpy as np

import slm_driver


def display_info():
    import wx
    app = wx.App(False)  # noqa: F841
    rows = []
    for i in range(wx.Display.GetCount()):
        d = wx.Display(i)
        g = d.GetGeometry()
        rows.append((i, g.GetWidth(), g.GetHeight(), g.GetX(), g.GetY(),
                     d.GetScaleFactor(), d.IsPrimary(), d.GetName()))
    return rows


def make_pattern(name, W, H, period=16):
    y, x = np.mgrid[0:H, 0:W]
    if name == 'white':
        return np.full((H, W), 255, np.uint8)
    if name == 'black':
        return np.zeros((H, W), np.uint8)
    if name == 'grating':
        return (((x // max(1, period // 2)) % 2) * 255).astype(np.uint8)
    if name == 'checker':
        return (((x + y) % 2) * 255).astype(np.uint8)
    if name == 'comb':
        img = np.zeros((H, W), np.uint8)
        img[::32, :] = 255
        return img
    if name == 'blaze':  # 8-bit phase ramp: 0..255 across the width
        return np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
    if name == 'frame':
        img = np.zeros((H, W), np.uint8)
        img[0:2, :] = 255
        img[H - 2:H, :] = 255
        img[:, 0:2] = 255
        img[:, W - 2:W] = 255
        img[H // 2 - 1:H // 2 + 1, :] = 255
        img[:, W // 2 - 1:W // 2 + 1] = 255
        for cy in (20, H - 80):
            for cx in (20, W - 80):
                img[cy:cy + 60, cx:cx + 60] = 255
        img[::128, ::128] = 160
        return img
    raise SystemExit(f'unknown pattern: {name}')


def build_image(args, W, H):
    if args.file:
        img = np.load(args.file)
        print('loaded', args.file, img.shape, img.dtype)
    else:
        img = make_pattern(args.pattern, W, H, args.period)
        print('pattern', args.pattern, img.shape, img.dtype)
    if img.shape[:2] != (H, W):
        raise SystemExit(f'array {img.shape} does not match the SLM window {(H, W)}')
    return img


def run_legacy(args, img):
    import slmpy
    slm = slmpy.SLMdisplay(monitor=args.monitor, isImageLock=False)
    W, H = slm.getSize()
    slm.updateArray(img)
    time.sleep(args.hold)
    slm.close()
    print('closed (slmpy)')
    os._exit(0)


def main():
    ap = argparse.ArgumentParser(description='Show test patterns on the SLM.')
    ap.add_argument('pattern', nargs='?', default='frame',
                    help='white|black|grating|checker|comb|blaze|frame')
    ap.add_argument('--file', help='.npy array (H,W) uint8 to display instead')
    ap.add_argument('--hold', type=float, default=10.0, help='seconds on screen (default 10)')
    ap.add_argument('--monitor', type=int, default=1, help='display index of the SLM (default 1)')
    ap.add_argument('--period', type=int, default=16, help='grating period in pixels')
    ap.add_argument('--bench', type=int, default=0, help='send N alternating frames and report the rate')
    ap.add_argument('--legacy', action='store_true', help='use slmpy instead of slm_driver')
    ap.add_argument('--info', action='store_true', help='only list displays')
    args = ap.parse_args()

    for i, w, h, x, y, scale, primary, name in display_info():
        flag = '  <- SLM' if i == args.monitor else ''
        print(f'[{i}] {w}x{h} at ({x},{y}) scale={scale} primary={primary} {name}{flag}')
    if args.info:
        return 0

    slm = slm_driver.SLM(monitor=args.monitor)
    W, H = slm.getSize()
    img = build_image(args, W, H)

    if args.legacy:
        return run_legacy(args, img)

    done = threading.Event()

    def sequence():
        try:
            t0 = time.perf_counter()
            slm.update(img)
            print(f'first paint in {(time.perf_counter() - t0) * 1000:.1f} ms')
            if args.bench:
                alt = 255 - img
                n = args.bench
                base = slm.paint_count
                t0 = time.perf_counter()
                for i in range(n):
                    slm.update(img if i % 2 == 0 else alt)
                t_sub = time.perf_counter() - t0
                while slm.paint_count - base < n and time.perf_counter() - t0 < 20:
                    time.sleep(0.002)
                total = time.perf_counter() - t0
                print(f'bench: submit {n / t_sub:.0f} fps, painted {slm.paint_count - base}/{n} '
                      f'in {total:.2f} s -> {n / total:.1f} fps end-to-end')
                slm.update(img)
            print(f'holding {args.hold:.0f} s ...')
            time.sleep(args.hold)
        finally:
            slm.close()
            done.set()

    threading.Thread(target=sequence, daemon=True).start()
    try:
        slm.run()
    except KeyboardInterrupt:
        slm.close()
    done.wait(2.0)
    print('closed')
    sys.stdout.flush()
    os._exit(0)  # skip wx teardown: destroying the wx.App at exit can crash/hang


if __name__ == '__main__':
    main()
