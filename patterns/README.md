# 图案说明

本目录只存 PNG（1024×768，8 bit 灰度，lossless）。上屏时用 `code/slm_show.py --file <png>`。

`.npy` 版本（uint8，供 numpy 直接读取）可用下列命令重新生成，结果逐字节一致：

| 文件 | 生成命令 |
| --- | --- |
| `A_rla420_hyb_phase` | `python code/random_lens_array.py --n 420 --f-min 30 --f-max 50 --gap 3 --margin 4 --seed 20261123 --save patterns/A_rla420` 再用 `code/hybrid_pattern.py --array ... --bg-max 255` |
| `B_rla420_phase` | `python code/random_lens_array.py --n 420 --f-min 30 --f-max 50 --gap 3 --margin 4 --seed 20261123 --save patterns/B_rla420` |
| `C_onezone_rand_phase` | `python code/onezone_array.py --n 150 --layout random --seed 20261124 --save patterns/C_onezone_rand` |
| `W1/W2/W3_phase` | `python code/wave_patterns.py --which all --save-prefix patterns/` |
