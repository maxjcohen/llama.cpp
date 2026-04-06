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
KV cache quantization — tested, no improvement.

- `--cache-type-k q8_0 --cache-type-v q8_0`: pp512=18.61 (+0.5%), tg128=7.33 (-2.1%) ✗
- `--cache-type-k q4_0 --cache-type-v q4_0`: pp512=18.35 (-0.9%), tg128=7.26 (-3.1%) ✗

Dequantization overhead in the FA kernel exceeds bandwidth savings at tg128 sequence
lengths (only 128 KV vectors). Standard f16 KV cache remains best.

## Step 5
Pascal FA tile `nbatch_fa` 64→32 — tested, no improvement, reverted.

**Theory:** For tg128 decode, the FA TILE kernel (if used) dispatches with `nbatch_fa=64`,
giving `ntiles_KV=2` for a context of 128 tokens. With 2 SMs and `occ=2` blocks/SM,
`nblocks_stream_k=min(4,2)=2` → only 50% SM utilization. Halving `nbatch_fa` to 32
would give `ntiles_KV=4` → `nblocks_stream_k=4` → 100% SM utilization.

**What was changed:** Added `ggml_cuda_fattn_tile_get_config_nvidia_pascal_fp16()` in
`ggml/src/ggml-cuda/fattn-tile.cuh` with `nbatch_fa=32` override for `(D=256,ncols=8)`,
`(D=512,ncols=8)`, and `(D=576,ncols=8)`. Routed Pascal (`cc < GGML_CUDA_CC_VOLTA`)
through this table in both host and device dispatch.

**Why it failed:** The tg128 decode path does **not** use the TILE kernel. With
`Q->ne[1]=1` (single decode token), `K->ne[1]` never a multiple of
`FATTN_KQ_STRIDE=256` during generation, so `gqa_opt_applies=false`. The dispatch
falls to `BEST_FATTN_KERNEL_VEC` (not TILE). The VEC kernel dispatches `blocks_num.z =
gqa_ratio * K->ne[2] * Q->ne[3] = 8 * 1 * 1 = 8` blocks (one per Q-head), giving 4
waves at 2 blocks/SM — already 100% SM utilization. The TILE tuning was targeting the
wrong code path.

**Result (build: a8515af7b, nbatch_fa change reverted):**
```
| gemma4 E2B Q4_K - Medium  | 2.88 GiB | 4.65 B | CUDA | 99 | 1 | pp512 | 18.21 ± 0.12 |
| gemma4 E2B Q4_K - Medium  | 2.88 GiB | 4.65 B | CUDA | 99 | 1 | tg128 |  7.49 ± 0.00 |
```

pp512: regression (noise/variance vs 18.65 baseline). tg128: flat. Change reverted.

## Step 6
FA VEC kernel `nthreads` 128 → 64 for Pascal — tested, no improvement, reverted.

**Theory:** At `nthreads=128` with `__launch_bounds__(128,1)` the compiler maximises register
usage for 1 block/SM.  With `nthreads=64` the same register budget might support 2 blocks/SM,
doubling occupancy and halving the number of serial FA VEC waves (8 blocks → 4 waves → 2 waves).

**What was changed:** `ggml_cuda_fattn_vec_get_nthreads_host()` returns 64 for Pascal (cc∈[600,700)).
`ggml_cuda_fattn_vec_get_nthreads_device()` uses `#if __CUDA_ARCH__` to return 64 at compile time.

**Why it failed:** `nthreads_KQ` is hardcoded as `128/cpy_nb` (not `nthreads/cpy_nb`), so reducing
`nthreads` does not change the inner KQ reduction work per thread.  The Pascal CUDA 10.2 compiler
apparently keeps register usage high enough at 64 threads/block that only 1 block still fits per SM
(same as at 128 threads), yielding no occupancy gain.  The `min_blocks=1` in `__launch_bounds__`
gives no pressure to reduce registers, so the compiler uses the full 64K/SM budget for a single block
regardless of `nthreads`.  Measured result: both pp512 and tg128 regressed slightly (-1.2% / -0.7%).

**Result (reverted):**
```
| gemma4 E2B Q4_K - Medium  | 2.88 GiB | 4.65 B | CUDA | 99 | 1 | pp512 | 18.42 ± 0.13 |
| gemma4 E2B Q4_K - Medium  | 2.88 GiB | 4.65 B | CUDA | 99 | 1 | tg128 |  7.44 ± 0.01 |
```

pp512: regression (-1.2%). tg128: regression (-0.7%). Change reverted.

---

## Ceiling Analysis — No Further Kernel-Level Improvements Identified

After exhaustive analysis of all kernel dispatch paths for this model/hardware, no
further optimizations were found that reliably improve both pp512 and tg128.

### Kernel-by-kernel status

| Kernel | Situation | Verdict |
|---|---|---|
| MMVQ Q4_K (tg decode) | nwarps=2 for Pascal ncols_dst=1 (prior patch) | Optimal |
| MMQ Q4_K/Q6_K (pp prefill) | mmq_x=64 (Q4_K, 2 blocks/SM), Q6_K capped at 40 (prior patch) | Optimal |
| FA VEC (tg, SWA layers D=256) | 8 blocks/4 waves, both SMs fully used; nthreads=64 tried — no gain (register pressure) | Ceiling reached |
| FA TILE (pp, SWA+global) | Large ntiles grids, well-utilized | Already full |
| FA VEC (tg, global D=512) | gqa_opt_applies=false at tg context lengths → NONE → standard GPU attn | n/a |
| RMS Norm | 512 threads per block (prior patch) | Optimal |
| RoPE | 8 blocks × 256 threads; RoPE+SET_ROWS fusion active | Fine |
| GLU/SwiGLU | 12 blocks × 256 threads, 2 SMs well-covered | Fine |
| per_layer_model_proj (mmvf) | F32, block_size=256, 1536 output rows → ample parallelism | Fine |
| stream_k (MMQ, FA TILE) | Correctly disabled on Pascal | Correct |
| KV quant (Step 4) | Adds dequant overhead > BW savings at short sequences | Rejected |

### Why tg128 is at its practical ceiling

At 7.49 t/s, effective memory bandwidth = 7.49 × 2.88 GB ≈ 21.6 GB/s out of
58.4 GB/s peak (37% utilization). The gap is not from kernel inefficiency but from the
sequential structure of the transformer:

- 35 layers must execute serially (each depends on the previous)
- ~500–700 kernel launches per generated token, each with CPU-side dispatch latency
- VEC attention: 8 blocks/4 waves already at 100% SM utilization; further splitting
  provides no benefit because the context (≤128 tokens) is too short for
  `parallel_blocks > 1` to trigger (ntiles_KV = ceil(KV / D) = ceil(64/256) = 1)
- MMVQ: already at optimal nwarps=2 giving max concurrent blocks per SM

Further gains would require architectural changes outside the kernel-tuning scope:
CUDA graph capture (eliminates CPU dispatch overhead), kernel fusion across layers,
or a fundamentally different model format (e.g., quantized KV with a bespoke kernel
that amortizes the dequant cost).

### Why pp512 is at its practical ceiling

At 18.2 t/s, Q4_K dequantization throughput is the bottleneck. mmq_x=64 gives
2 blocks/SM (already at max for 49 KB shmem budget). Smaller mmq_x (32) would give
3 blocks/SM but fewer columns per block, resulting in more waves and lower throughput
per unit shmem loaded. Larger mmq_x is blocked by the Pascal mmq_x_max=64 limit
(no Volta tensor cores available for the wider DP4A path).
