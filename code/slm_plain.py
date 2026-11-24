# -*- coding: utf-8 -*-
"""在 SLM 上显示一帧纯灰（默认 0 = 不调制），常驻直到关窗。

    python code/slm_plain.py --grey 0            # 常驻
    python code/slm_plain.py --grey 128 --hold 20
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slm_driver import SLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grey", type=int, default=0)
    ap.add_argument("--hold", type=float, default=0.0, help="0 = 常驻")
    ap.add_argument("--monitor", type=int, default=1)
    args = ap.parse_args()

    slm = SLM(monitor=args.monitor)
    w, h = slm.getSize()          # getSize() 返回 (宽, 高)
    slm.update(np.full((h, w), args.grey, np.uint8))
    print(f"grey={args.grey} on {w}x{h}, hold={args.hold or 'inf'}", flush=True)
    slm.run(hold=args.hold or None)
    return 0


if __name__ == "__main__":
    globals()["_keep"] = None
    raise SystemExit(main())
