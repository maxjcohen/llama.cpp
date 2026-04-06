# Optimization

## Step 0
Baseline

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7858 MiB):
  Device 0: NVIDIA Tegra X2, compute capability 6.2, VMM: no, VRAM: 7858 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |           pp512 |         17.67 ± 0.11 |
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |           tg128 |          6.64 ± 0.01 |

build: b2cef5024 (8667)
```

## Step 1
Cap Q6_K `mmq_x` at 40 on few-SM (≤4) pre-Volta Pascal GPUs.

**Change:** `ggml/src/ggml-cuda/mmq.cuh` — in `mul_mat_q_case<type>`, read `nsm` from device info and
override `mmq_x_max` to `min(mmq_x_max, 40)` when `type==Q6_K && NVIDIA && cc<700 && nsm<=4`.

**Why:** Q6_K shmem at mmq_x=64 is 26.8 KB → `floor(49152/26.8K)=1 block/SM` (only 2 active
blocks on TX2's 2 SMs). Capping at mmq_x=40 (23.7 KB) gives `floor(49152/23.7K)=2 blocks/SM`
(4 active blocks) — doubling occupancy. The extra tiles (7 vs 4 per row) are outweighed by 2×
better latency hiding.

```
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |      256 |           pp512 |         18.05 ± 0.04 |
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |      256 |           tg128 |          6.83 ± 0.00 |

build: f2212a98d (8668)
```

pp512: 17.67 → 18.05 t/s (+2.2%) ✓  tg128: 6.64 → 6.83 t/s (+2.9%) ✓

## Step 2
Enable flash attention at runtime (`-fa 1`).

**Change:** Runtime flag only — no code change. Pass `-fa 1` to `llama-bench` / `llama-server`.
Flash attention replaces the standard softmax attention with a tiled CUDA kernel that keeps the
KV tile in registers/shared memory, reducing global memory traffic during the attention pass.

```
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |      256 |  1 |           pp512 |         18.23 ± 0.03 |
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |      256 |  1 |           tg128 |          7.49 ± 0.00 |

build: f2212a98d (8668)
```

pp512: 18.05 → 18.23 t/s (+1.0%) ✓  tg128: 6.83 → 7.49 t/s (+9.7%) ✓

All subsequent benchmarks use `-fa 1`.

## Step 3
Optimal ubatch size for Q4_K_M is 512 (not 256 as for Q4_0).

**Change:** Runtime flag only — use `-ub 512` instead of `-ub 256`.
With ub=512 the entire pp512 prompt fits in a single batch pass.
Q4_0 optimal was 256; Q4_K_M benefits from larger tiles because mmq_x was
reduced to 40 for Q6_K layers (more kernel launches at smaller mmq_x amortize
better with one big batch).

Sweep results (all with `-fa 1`):
- ub=64:   pp512=17.75,  tg128=7.27
- ub=128:  pp512=18.42,  tg128=7.49
- ub=256:  pp512=18.23,  tg128=7.49
- **ub=512:  pp512=18.52±,  tg128=7.48** ← best
- ub=1024: pp512=18.13,  tg128=7.44

```
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |  1 |           pp512 |         18.65 ± 0.05 |
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |  1 |           tg128 |          7.49 ± 0.00 |

build: 159b3a4f5 (8670)
```

pp512: 18.23 → 18.52 t/s (+1.6%) ✓  tg128: 7.49 → 7.49 t/s (flat) ✓

All subsequent benchmarks use `-fa 1 -ub 512`.

## Step 4
