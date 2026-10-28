### 1. 数据集概览

| 数据集 | N | B | AllErr事件数 | FN事件数 | M_global | 冗余 | 问题 |
|---|---|---|---|---|---|---|---|
| uci | 6000 | 600 | 1506 | 521 | 16 | M_global=16, S in set: other 15 candidates not recoverable from (S,C) | 无 |
| fico | 2092 | 210 | 563 | 273 | 14 | M_global=14, S not in set | 无 |
| hmda | 32855 | 3286 | 7591 | 5451 | 2 | M_global=2 and S in set: SYM≡G2 algebraically; ID recoverable from G1 (S,C) -> ID-G1, ID-SYM cannot reflect new information | 无 |
| lendingclub | 5148 | 515 | 1868 | 386 | 4 | M_global=4, S in set: other 3 candidates not recoverable from (S,C) | 无 |
| gmc | 30000 | 3000 | 6249 | 452 | 17 | M_global=17, S in set: other 16 candidates not recoverable from (S,C) | 无 |
| hmeq | 1192 | 120 | 98 | 32 | 6 | M_global=6, S not in set | 无 |
| home_credit | 61503 | 6151 | 18660 | 1636 | 2 | M_global=2 and S in set: SYM≡G2 algebraically; ID recoverable from G1 (S,C) -> ID-G1, ID-SYM cannot reflect new information | 无 |

### 2a. 各方法结果（AllErr，10% 预算；格内依次为 命中 / capture / 队列 precision）

| dataset | G1 | G2 | SYM | ID | GX | G1add |
|---|---|---|---|---|---|---|
| uci | 448 / 29.7% / 74.7% | 450 / 29.9% / 75.0% | 451 / 29.9% / 75.2% | 460 / 30.5% / 76.7% | 447 / 29.7% / 74.5% | 452 / 30.0% / 75.3% |
| fico | 99 / 17.6% / 47.1% | 96 / 17.1% / 45.7% | 104 / 18.5% / 49.5% | 96 / 17.1% / 45.7% | 100 / 17.8% / 47.6% | 103 / 18.3% / 49.0% |
| hmda | 2426 / 32.0% / 73.8% | 2416 / 31.8% / 73.5% | 2420 / 31.9% / 73.6% | 2412 / 31.8% / 73.4% | 2420 / 31.9% / 73.6% | 2423 / 31.9% / 73.7% |
| lendingclub | 389 / 20.8% / 75.5% | 393 / 21.0% / 76.3% | 391 / 20.9% / 75.9% | 386 / 20.7% / 75.0% | 399 / 21.4% / 77.5% | 393 / 21.0% / 76.3% |
| gmc | 2729 / 43.7% / 91.0% | 2724 / 43.6% / 90.8% | 2732 / 43.7% / 91.1% | 2729 / 43.7% / 91.0% | 2743 / 43.9% / 91.4% | 2737 / 43.8% / 91.2% |
| hmeq | 62 / 63.3% / 51.7% | 63 / 64.3% / 52.5% | 63 / 64.3% / 52.5% | 62 / 63.3% / 51.7% | 59 / 60.2% / 49.2% | 60 / 61.2% / 50.0% |
| home_credit | 5572 / 29.9% / 90.6% | 5583 / 29.9% / 90.8% | 5563 / 29.8% / 90.4% | 5573 / 29.9% / 90.6% | 5564 / 29.8% / 90.5% | 5568 / 29.8% / 90.5% |

### 2b. 各方法结果（FN，10% 预算；格内依次为 命中 / capture / 队列 precision）

| dataset | G1 | G2 | SYM | ID | GX | G1add |
|---|---|---|---|---|---|---|
| uci | 124 / 23.8% / 20.7% | 112 / 21.5% / 18.7% | 113 / 21.7% / 18.8% | 107 / 20.5% / 17.8% | 128 / 24.6% / 21.3% | 122 / 23.4% / 20.3% |
| fico | 101 / 37.0% / 48.1% | 101 / 37.0% / 48.1% | 108 / 39.6% / 51.4% | 104 / 38.1% / 49.5% | 97 / 35.5% / 46.2% | 103 / 37.7% / 49.0% |
| hmda | 2426 / 44.5% / 73.8% | 2416 / 44.3% / 73.5% | 2420 / 44.4% / 73.6% | 2412 / 44.2% / 73.4% | 2420 / 44.4% / 73.6% | 2423 / 44.5% / 73.7% |
| lendingclub | 128 / 33.2% / 24.9% | 118 / 30.6% / 22.9% | 113 / 29.3% / 21.9% | 115 / 29.8% / 22.3% | 120 / 31.1% / 23.3% | 128 / 33.2% / 24.9% |
| gmc | 151 / 33.4% / 5.0% | 155 / 34.3% / 5.2% | 161 / 35.6% / 5.4% | 161 / 35.6% / 5.4% | 154 / 34.1% / 5.1% | 160 / 35.4% / 5.3% |
| hmeq | 23 / 71.9% / 19.2% | 22 / 68.8% / 18.3% | 22 / 68.8% / 18.3% | 22 / 68.8% / 18.3% | 22 / 68.8% / 18.3% | 24 / 75.0% / 20.0% |
| home_credit | 441 / 27.0% / 7.2% | 430 / 26.3% / 7.0% | 430 / 26.3% / 7.0% | 423 / 25.9% / 6.9% | 418 / 25.6% / 6.8% | 441 / 27.0% / 7.2% |

### 3. 主比较：AllErr，10%，ID − SYM 与 ID − G1（14 项，Bonferroni-14）

| 数据集 | 对比 | 参照命中 | ΔH | Δ每万人 | ΔH/B | 剩余空间 | 逐点95% | Bonferroni-14 | 实质增益 | 实质损失 | 可排除 | Δlog loss [95%] | 冗余标注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uci | ID-SYM | 451 | 9 | 15 | 0.015 | 149 | [-4, +17] | [-10, +22] | False | False | False | -0.00038 [-0.00211, +0.00135] |  |
| uci | ID-G1 | 448 | 12 | 20 | 0.02 | 152 | [-3, +24] | [-9, +30] | False | False | False | +0.00022 [-0.00224, +0.00269] |  |
| fico | ID-SYM | 104 | -8 | -38.24 | -0.0381 | 106 | [-19, +5] | [-24, +11] | False | False | False | +0.00066 [-0.00502, +0.00630] |  |
| fico | ID-G1 | 99 | -3 | -14.34 | -0.0143 | 111 | [-16, +12] | [-22, +19] | False | False | False | +0.00572 [-0.00124, +0.01252] |  |
| hmda | ID-SYM | 2420 | -8 | -2.43 | -0.0024 | 866 | [-12, +9] | [-18, +14] | False | False | True | -0.00000 [-0.00010, +0.00009] | G1 已可恢复，不可能反映新信息 |
| hmda | ID-G1 | 2426 | -14 | -4.26 | -0.0043 | 860 | [-22, +14] | [-30, +23] | False | False | True | +0.00013 [-0.00019, +0.00044] | G1 已可恢复，不可能反映新信息 |
| lendingclub | ID-SYM | 391 | -5 | -9.71 | -0.0097 | 124 | [-15, +8] | [-21, +14] | False | False | False | +0.00157 [+0.00005, +0.00309] |  |
| lendingclub | ID-G1 | 389 | -3 | -5.83 | -0.0058 | 126 | [-19, +12] | [-26, +19] | False | False | False | +0.00409 [+0.00144, +0.00678] |  |
| gmc | ID-SYM | 2732 | -3 | -1 | -0.001 | 268 | [-17, +13] | [-24, +20] | False | False | True | -0.00063 [-0.00103, -0.00023] |  |
| gmc | ID-G1 | 2729 | 0 | 0 | 0 | 271 | [-14, +17] | [-22, +25] | False | False | True | -0.00047 [-0.00101, +0.00006] |  |
| hmeq | ID-SYM | 63 | -1 | -8.39 | -0.0083 | 35 | [-4, +3] | [-6, +5] | False | False | False | -0.00019 [-0.00289, +0.00248] |  |
| hmeq | ID-G1 | 62 | 0 | 0 | 0 | 36 | [-4, +6] | [-7, +9] | False | False | False | +0.00148 [-0.00335, +0.00639] |  |
| home_credit | ID-SYM | 5563 | 10 | 1.63 | 0.0016 | 588 | [-9, +30] | [-19, +40] | False | False | True | +0.00006 [-0.00005, +0.00017] | G1 已可恢复，不可能反映新信息 |
| home_credit | ID-G1 | 5572 | 1 | 0.16 | 0.0002 | 579 | [-21, +18] | [-31, +28] | False | False | True | +0.00015 [-0.00005, +0.00034] | G1 已可恢复，不可能反映新信息 |

### 4. 次要比较（探索性，逐点区间）

| 数据集 | 目标 | 对比 | 参照命中 | ΔH | Δ每万人 | 剩余空间 | 逐点95% | 实质增益(逐点) | 实质损失 | 可排除 | Δlog loss | log loss 区间<0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uci | AllErr | G2-G1 | 448 | 2 | 3.33 | 152 | [-8, +11] | False | False | True | 0.00191 | False |
| uci | AllErr | SYM-G2 | 450 | 1 | 1.67 | 150 | [-8, +14] | False | False | True | -0.00131 | False |
| uci | AllErr | SYM-G1 | 448 | 3 | 5 | 152 | [-9, +17] | False | False | False | 0.00059 | False |
| uci | AllErr | GX-G1 | 448 | -1 | -1.67 | 152 | [-14, +12] | False | False | True | 0.0003 | False |
| uci | AllErr | G1-G1add | 452 | -4 | -6.67 | 148 | [-14, +8] | False | False | True | 0.00264 | False |
| uci | AllErr | G2-G1add | 452 | -2 | -3.33 | 148 | [-14, +10] | False | False | True | 0.00455 | False |
| uci | AllErr | SYM-G1add | 452 | -1 | -1.67 | 148 | [-12, +14] | False | False | True | 0.00323 | False |
| uci | AllErr | ID-G1add | 452 | 8 | 13.33 | 148 | [-6, +21] | False | False | False | 0.00285 | False |
| uci | AllErr | GX-G1add | 452 | -5 | -8.33 | 148 | [-17, +8] | False | False | True | 0.00294 | False |
| uci | FN | ID-SYM | 113 | -6 | -10 | 408 | [-19, +10] | False | False | True | -0.00038 | False |
| uci | FN | ID-G1 | 124 | -17 | -28.33 | 397 | [-33, +3] | False | False | True | 0.00022 | False |
| uci | FN | G2-G1 | 124 | -12 | -20 | 397 | [-27, +3] | False | False | True | 0.00191 | False |
| uci | FN | SYM-G2 | 112 | 1 | 1.67 | 409 | [-13, +16] | False | False | True | -0.00131 | False |
| uci | FN | SYM-G1 | 124 | -11 | -18.33 | 397 | [-28, +7] | False | False | True | 0.00059 | False |
| uci | FN | GX-G1 | 124 | 4 | 6.67 | 397 | [-17, +20] | False | False | True | 0.0003 | False |
| uci | FN | G1-G1add | 122 | 2 | 3.33 | 399 | [-15, +17] | False | False | True | 0.00264 | False |
| uci | FN | G2-G1add | 122 | -10 | -16.67 | 399 | [-28, +5] | False | False | True | 0.00455 | False |
| uci | FN | SYM-G1add | 122 | -9 | -15 | 399 | [-27, +7] | False | False | True | 0.00323 | False |
| uci | FN | ID-G1add | 122 | -15 | -25 | 399 | [-32, +3] | False | False | True | 0.00285 | False |
| uci | FN | GX-G1add | 122 | 6 | 10 | 399 | [-16, +20] | False | False | True | 0.00294 | False |
| fico | AllErr | G2-G1 | 99 | -3 | -14.34 | 111 | [-13, +6] | False | False | True | 0.00336 | False |
| fico | AllErr | SYM-G2 | 96 | 8 | 38.24 | 114 | [-3, +20] | False | False | False | 0.00171 | False |
| fico | AllErr | SYM-G1 | 99 | 5 | 23.9 | 111 | [-7, +17] | False | False | False | 0.00506 | False |
| fico | AllErr | GX-G1 | 99 | 1 | 4.78 | 111 | [-13, +13] | False | False | False | 0.00717 | False |
| fico | AllErr | G1-G1add | 103 | -4 | -19.12 | 107 | [-14, +7] | False | False | True | 0.00543 | False |
| fico | AllErr | G2-G1add | 103 | -7 | -33.46 | 107 | [-19, +5] | False | False | True | 0.00879 | False |
| fico | AllErr | SYM-G1add | 103 | 1 | 4.78 | 107 | [-11, +13] | False | False | False | 0.01049 | False |
| fico | AllErr | ID-G1add | 103 | -7 | -33.46 | 107 | [-19, +8] | False | False | True | 0.01115 | False |
| fico | AllErr | GX-G1add | 103 | -3 | -14.34 | 107 | [-16, +9] | False | False | True | 0.0126 | False |
| fico | FN | ID-SYM | 108 | -4 | -19.12 | 102 | [-14, +4] | False | False | True | 0.00066 | False |
| fico | FN | ID-G1 | 101 | 3 | 14.34 | 109 | [-11, +11] | False | False | False | 0.00572 | False |
| fico | FN | G2-G1 | 101 | 0 | 0 | 109 | [-10, +7] | False | False | True | 0.00336 | False |
| fico | FN | SYM-G2 | 101 | 7 | 33.46 | 109 | [-2, +16] | False | False | False | 0.00171 | False |
| fico | FN | SYM-G1 | 101 | 7 | 33.46 | 109 | [-6, +15] | False | False | False | 0.00506 | False |
| fico | FN | GX-G1 | 101 | -4 | -19.12 | 109 | [-15, +7] | False | False | True | 0.00717 | False |
| fico | FN | G1-G1add | 103 | -2 | -9.56 | 107 | [-10, +10] | False | False | True | 0.00543 | False |
| fico | FN | G2-G1add | 103 | -2 | -9.56 | 107 | [-12, +8] | False | False | True | 0.00879 | False |
| fico | FN | SYM-G1add | 103 | 5 | 23.9 | 107 | [-5, +15] | False | False | False | 0.01049 | False |
| fico | FN | ID-G1add | 103 | 1 | 4.78 | 107 | [-10, +10] | False | False | True | 0.01115 | False |
| fico | FN | GX-G1add | 103 | -6 | -28.68 | 107 | [-15, +6] | False | False | True | 0.0126 | False |
| hmda | AllErr | G2-G1 | 2426 | -10 | -3.04 | 860 | [-22, +12] | False | False | True | 5e-05 | False |
| hmda | AllErr | SYM-G2 | 2416 | 4 | 1.22 | 870 | [-14, +20] | False | False | True | 8e-05 | False |
| hmda | AllErr | SYM-G1 | 2426 | -6 | -1.83 | 860 | [-21, +16] | False | False | True | 0.00013 | False |
| hmda | AllErr | GX-G1 | 2426 | -6 | -1.83 | 860 | [-19, +14] | False | False | True | 0 | False |
| hmda | AllErr | G1-G1add | 2423 | 3 | 0.91 | 863 | [-22, +18] | False | False | True | 0.00028 | False |
| hmda | AllErr | G2-G1add | 2423 | -7 | -2.13 | 863 | [-27, +14] | False | False | True | 0.00033 | False |
| hmda | AllErr | SYM-G1add | 2423 | -3 | -0.91 | 863 | [-26, +17] | False | False | True | 0.00041 | False |
| hmda | AllErr | ID-G1add | 2423 | -11 | -3.35 | 863 | [-26, +15] | False | False | True | 0.00041 | False |
| hmda | AllErr | GX-G1add | 2423 | -3 | -0.91 | 863 | [-24, +15] | False | False | True | 0.00029 | False |
| hmda | FN | ID-SYM | 2420 | -8 | -2.43 | 866 | [-12, +9] | False | False | True | -0 | False |
| hmda | FN | ID-G1 | 2426 | -14 | -4.26 | 860 | [-22, +14] | False | False | True | 0.00013 | False |
| hmda | FN | G2-G1 | 2426 | -10 | -3.04 | 860 | [-22, +12] | False | False | True | 5e-05 | False |
| hmda | FN | SYM-G2 | 2416 | 4 | 1.22 | 870 | [-14, +20] | False | False | True | 8e-05 | False |
| hmda | FN | SYM-G1 | 2426 | -6 | -1.83 | 860 | [-21, +16] | False | False | True | 0.00013 | False |
| hmda | FN | GX-G1 | 2426 | -6 | -1.83 | 860 | [-19, +14] | False | False | True | 0 | False |
| hmda | FN | G1-G1add | 2423 | 3 | 0.91 | 863 | [-22, +18] | False | False | True | 0.00028 | False |
| hmda | FN | G2-G1add | 2423 | -7 | -2.13 | 863 | [-27, +14] | False | False | True | 0.00033 | False |
| hmda | FN | SYM-G1add | 2423 | -3 | -0.91 | 863 | [-26, +17] | False | False | True | 0.00041 | False |
| hmda | FN | ID-G1add | 2423 | -11 | -3.35 | 863 | [-26, +15] | False | False | True | 0.00041 | False |
| hmda | FN | GX-G1add | 2423 | -3 | -0.91 | 863 | [-24, +15] | False | False | True | 0.00029 | False |
| lendingclub | AllErr | G2-G1 | 389 | 4 | 7.77 | 126 | [-10, +13] | False | False | False | 0.00165 | False |
| lendingclub | AllErr | SYM-G2 | 393 | -2 | -3.89 | 122 | [-14, +11] | False | False | True | 0.00087 | False |
| lendingclub | AllErr | SYM-G1 | 389 | 2 | 3.89 | 126 | [-13, +13] | False | False | False | 0.00252 | False |
| lendingclub | AllErr | GX-G1 | 389 | 10 | 19.43 | 126 | [-12, +21] | False | False | False | 0.00212 | False |
| lendingclub | AllErr | G1-G1add | 393 | -4 | -7.77 | 122 | [-18, +12] | False | False | True | 0.00159 | False |
| lendingclub | AllErr | G2-G1add | 393 | 0 | 0 | 122 | [-16, +14] | False | False | False | 0.00324 | False |
| lendingclub | AllErr | SYM-G1add | 393 | -2 | -3.89 | 122 | [-16, +11] | False | False | True | 0.00411 | False |
| lendingclub | AllErr | ID-G1add | 393 | -7 | -13.6 | 122 | [-21, +8] | False | False | True | 0.00568 | False |
| lendingclub | AllErr | GX-G1add | 393 | 6 | 11.66 | 122 | [-14, +19] | False | False | False | 0.0037 | False |
| lendingclub | FN | ID-SYM | 113 | 2 | 3.89 | 273 | [-8, +15] | False | False | True | 0.00157 | False |
| lendingclub | FN | ID-G1 | 128 | -13 | -25.25 | 258 | [-23, +6] | False | False | True | 0.00409 | False |
| lendingclub | FN | G2-G1 | 128 | -10 | -19.43 | 258 | [-19, +4] | False | False | True | 0.00165 | False |
| lendingclub | FN | SYM-G2 | 118 | -5 | -9.71 | 268 | [-16, +8] | False | False | True | 0.00087 | False |
| lendingclub | FN | SYM-G1 | 128 | -15 | -29.14 | 258 | [-25, +2] | False | False | True | 0.00252 | False |
| lendingclub | FN | GX-G1 | 128 | -8 | -15.54 | 258 | [-21, +9] | False | False | True | 0.00212 | False |
| lendingclub | FN | G1-G1add | 128 | 0 | 0 | 258 | [-16, +11] | False | False | True | 0.00159 | False |
| lendingclub | FN | G2-G1add | 128 | -10 | -19.43 | 258 | [-24, +3] | False | False | True | 0.00324 | False |
| lendingclub | FN | SYM-G1add | 128 | -15 | -29.14 | 258 | [-28, -1] | False | False | True | 0.00411 | False |
| lendingclub | FN | ID-G1add | 128 | -13 | -25.25 | 258 | [-25, +3] | False | False | True | 0.00568 | False |
| lendingclub | FN | GX-G1add | 128 | -8 | -15.54 | 258 | [-23, +6] | False | False | True | 0.0037 | False |
| gmc | AllErr | G2-G1 | 2729 | -5 | -1.67 | 271 | [-16, +11] | False | False | True | -1e-05 | False |
| gmc | AllErr | SYM-G2 | 2724 | 8 | 2.67 | 276 | [-7, +20] | False | False | True | 0.00017 | False |
| gmc | AllErr | SYM-G1 | 2729 | 3 | 1 | 271 | [-11, +18] | False | False | True | 0.00016 | False |
| gmc | AllErr | GX-G1 | 2729 | 14 | 4.67 | 271 | [+0, +28] | False | False | False | -0.00081 | True |
| gmc | AllErr | G1-G1add | 2737 | -8 | -2.67 | 263 | [-20, +5] | False | False | True | 0.0011 | False |
| gmc | AllErr | G2-G1add | 2737 | -13 | -4.33 | 263 | [-24, +4] | False | False | True | 0.00109 | False |
| gmc | AllErr | SYM-G1add | 2737 | -5 | -1.67 | 263 | [-17, +9] | False | False | True | 0.00125 | False |
| gmc | AllErr | ID-G1add | 2737 | -8 | -2.67 | 263 | [-21, +9] | False | False | True | 0.00063 | False |
| gmc | AllErr | GX-G1add | 2737 | 6 | 2 | 263 | [-8, +22] | False | False | True | 0.00028 | False |
| gmc | FN | ID-SYM | 161 | 0 | 0 | 291 | [-11, +8] | False | False | True | -0.00063 | True |
| gmc | FN | ID-G1 | 151 | 10 | 3.33 | 301 | [-3, +22] | False | False | True | -0.00047 | False |
| gmc | FN | G2-G1 | 151 | 4 | 1.33 | 301 | [-4, +14] | False | False | True | -1e-05 | False |
| gmc | FN | SYM-G2 | 155 | 6 | 2 | 297 | [-6, +19] | False | False | True | 0.00017 | False |
| gmc | FN | SYM-G1 | 151 | 10 | 3.33 | 301 | [-1, +23] | False | False | True | 0.00016 | False |
| gmc | FN | GX-G1 | 151 | 3 | 1 | 301 | [-10, +15] | False | False | True | -0.00081 | True |
| gmc | FN | G1-G1add | 160 | -9 | -3 | 292 | [-20, +2] | False | False | True | 0.0011 | False |
| gmc | FN | G2-G1add | 160 | -5 | -1.67 | 292 | [-15, +6] | False | False | True | 0.00109 | False |
| gmc | FN | SYM-G1add | 160 | 1 | 0.33 | 292 | [-9, +13] | False | False | True | 0.00125 | False |
| gmc | FN | ID-G1add | 160 | 1 | 0.33 | 292 | [-11, +12] | False | False | True | 0.00063 | False |
| gmc | FN | GX-G1add | 160 | -6 | -2 | 292 | [-19, +6] | False | False | True | 0.00028 | False |
| hmeq | AllErr | G2-G1 | 62 | 1 | 8.39 | 36 | [-3, +6] | False | False | False | -0.00145 | False |
| hmeq | AllErr | SYM-G2 | 63 | 0 | 0 | 35 | [-4, +3] | False | False | False | 0.00312 | False |
| hmeq | AllErr | SYM-G1 | 62 | 1 | 8.39 | 36 | [-4, +6] | False | False | False | 0.00168 | False |
| hmeq | AllErr | GX-G1 | 62 | -3 | -25.17 | 36 | [-9, +4] | False | False | False | 0.00539 | False |
| hmeq | AllErr | G1-G1add | 60 | 2 | 16.78 | 38 | [-4, +7] | False | False | False | 0.00927 | False |
| hmeq | AllErr | G2-G1add | 60 | 3 | 25.17 | 38 | [-2, +8] | False | False | False | 0.00782 | False |
| hmeq | AllErr | SYM-G1add | 60 | 3 | 25.17 | 38 | [-3, +8] | False | False | False | 0.01094 | False |
| hmeq | AllErr | ID-G1add | 60 | 2 | 16.78 | 38 | [-3, +8] | False | False | False | 0.01075 | False |
| hmeq | AllErr | GX-G1add | 60 | -1 | -8.39 | 38 | [-8, +6] | False | False | False | 0.01465 | False |
| hmeq | FN | ID-SYM | 22 | 0 | 0 | 10 | [-2, +3] | False | False | False | -0.00019 | False |
| hmeq | FN | ID-G1 | 23 | -1 | -8.39 | 9 | [-2, +3] | False | False | False | 0.00148 | False |
| hmeq | FN | G2-G1 | 23 | -1 | -8.39 | 9 | [-2, +3] | False | False | False | -0.00145 | False |
| hmeq | FN | SYM-G2 | 22 | 0 | 0 | 10 | [-3, +2] | False | False | False | 0.00312 | False |
| hmeq | FN | SYM-G1 | 23 | -1 | -8.39 | 9 | [-2, +2] | False | False | False | 0.00168 | False |
| hmeq | FN | GX-G1 | 23 | -1 | -8.39 | 9 | [-2, +2] | False | False | False | 0.00539 | False |
| hmeq | FN | G1-G1add | 24 | -1 | -8.39 | 8 | [-4, +0] | False | False | False | 0.00927 | False |
| hmeq | FN | G2-G1add | 24 | -2 | -16.78 | 8 | [-4, +2] | False | False | False | 0.00782 | False |
| hmeq | FN | SYM-G1add | 24 | -2 | -16.78 | 8 | [-5, +1] | False | False | False | 0.01094 | False |
| hmeq | FN | ID-G1add | 24 | -2 | -16.78 | 8 | [-4, +1] | False | False | False | 0.01075 | False |
| hmeq | FN | GX-G1add | 24 | -2 | -16.78 | 8 | [-5, +1] | False | False | False | 0.01465 | False |
| home_credit | AllErr | G2-G1 | 5572 | 11 | 1.79 | 579 | [-10, +21] | False | False | True | 4e-05 | False |
| home_credit | AllErr | SYM-G2 | 5583 | -20 | -3.25 | 568 | [-35, +0] | False | False | True | 5e-05 | False |
| home_credit | AllErr | SYM-G1 | 5572 | -9 | -1.46 | 579 | [-29, +5] | False | False | True | 9e-05 | False |
| home_credit | AllErr | GX-G1 | 5572 | -8 | -1.3 | 579 | [-43, +20] | False | False | True | 0.00046 | False |
| home_credit | AllErr | G1-G1add | 5568 | 4 | 0.65 | 583 | [-16, +33] | False | False | True | 0.00047 | False |
| home_credit | AllErr | G2-G1add | 5568 | 15 | 2.44 | 583 | [-12, +40] | False | False | True | 0.00051 | False |
| home_credit | AllErr | SYM-G1add | 5568 | -5 | -0.81 | 583 | [-28, +21] | False | False | True | 0.00056 | False |
| home_credit | AllErr | ID-G1add | 5568 | 5 | 0.81 | 583 | [-18, +31] | False | False | True | 0.00061 | False |
| home_credit | AllErr | GX-G1add | 5568 | -4 | -0.65 | 583 | [-36, +29] | False | False | True | 0.00092 | False |
| home_credit | FN | ID-SYM | 430 | -7 | -1.14 | 1206 | [-24, +9] | False | False | True | 6e-05 | False |
| home_credit | FN | ID-G1 | 441 | -18 | -2.93 | 1195 | [-43, +4] | False | False | True | 0.00015 | False |
| home_credit | FN | G2-G1 | 441 | -11 | -1.79 | 1195 | [-36, +10] | False | False | True | 4e-05 | False |
| home_credit | FN | SYM-G2 | 430 | 0 | 0 | 1206 | [-20, +22] | False | False | True | 5e-05 | False |
| home_credit | FN | SYM-G1 | 441 | -11 | -1.79 | 1195 | [-35, +12] | False | False | True | 9e-05 | False |
| home_credit | FN | GX-G1 | 441 | -23 | -3.74 | 1195 | [-53, +6] | False | False | True | 0.00046 | False |
| home_credit | FN | G1-G1add | 441 | 0 | 0 | 1195 | [-24, +22] | False | False | True | 0.00047 | False |
| home_credit | FN | G2-G1add | 441 | -11 | -1.79 | 1195 | [-38, +9] | False | False | True | 0.00051 | False |
| home_credit | FN | SYM-G1add | 441 | -11 | -1.79 | 1195 | [-35, +9] | False | False | True | 0.00056 | False |
| home_credit | FN | ID-G1add | 441 | -18 | -2.93 | 1195 | [-43, +2] | False | False | True | 0.00061 | False |
| home_credit | FN | GX-G1add | 441 | -23 | -3.74 | 1195 | [-53, +4] | False | False | True | 0.00092 | False |

### 5. 可复现性检查（按事先写定的规则）

| contrast | target | role | 实质增益数据集数 | 实质损失数据集数 | 可排除数据集数 | 已定义数据集数 | 可复现增益(≥3且无损失) |
|---|---|---|---|---|---|---|---|
| G1-G1add | AllErr | secondary | 0 | 0 | 6 | 7 | False |
| G1-G1add | FN | secondary | 0 | 0 | 6 | 7 | False |
| G2-G1 | AllErr | secondary | 0 | 0 | 5 | 7 | False |
| G2-G1 | FN | secondary | 0 | 0 | 6 | 7 | False |
| G2-G1add | AllErr | secondary | 0 | 0 | 5 | 7 | False |
| G2-G1add | FN | secondary | 0 | 0 | 6 | 7 | False |
| GX-G1 | AllErr | secondary | 0 | 0 | 3 | 7 | False |
| GX-G1 | FN | secondary | 0 | 0 | 6 | 7 | False |
| GX-G1add | AllErr | secondary | 0 | 0 | 5 | 7 | False |
| GX-G1add | FN | secondary | 0 | 0 | 6 | 7 | False |
| ID-G1 | AllErr | primary | 0 | 0 | 3 | 7 | False |
| ID-G1 | FN | secondary | 0 | 0 | 5 | 7 | False |
| ID-G1add | AllErr | secondary | 0 | 0 | 5 | 7 | False |
| ID-G1add | FN | secondary | 0 | 0 | 6 | 7 | False |
| ID-SYM | AllErr | primary | 0 | 0 | 3 | 7 | False |
| ID-SYM | FN | secondary | 0 | 0 | 6 | 7 | False |
| SYM-G1 | AllErr | secondary | 0 | 0 | 3 | 7 | False |
| SYM-G1 | FN | secondary | 0 | 0 | 5 | 7 | False |
| SYM-G1add | AllErr | secondary | 0 | 0 | 5 | 7 | False |
| SYM-G1add | FN | secondary | 0 | 0 | 5 | 7 | False |
| SYM-G2 | AllErr | secondary | 0 | 0 | 5 | 7 | False |
| SYM-G2 | FN | secondary | 0 | 0 | 5 | 7 | False |

### 6. 判定规则的机械输入

```
{
 "身份线索_规则1": [],
 "对称表示线索_规则2": [],
 "仅GX改善_规则3": [],
 "只改善log_loss_规则4": [
  "gmc:GX-G1:AllErr",
  "gmc:GX-G1:FN",
  "gmc:ID-SYM:AllErr",
  "gmc:ID-SYM:FN"
 ],
 "主比较中未排除也未增益": [
  "fico:ID-G1",
  "fico:ID-SYM",
  "hmeq:ID-G1",
  "hmeq:ID-SYM",
  "lendingclub:ID-G1",
  "lendingclub:ID-SYM",
  "uci:ID-G1",
  "uci:ID-SYM"
 ]
}
```