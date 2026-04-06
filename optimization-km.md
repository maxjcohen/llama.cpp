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

## Step 7
Route Q4_K (and all K-quant) matmuls to cuBLAS FP16 GEMM instead of MMQ on
pre-Volta GPUs with fast FP16 hardware. **pp512: 18.24 → 137.76 t/s (+655%).**

### Background

The previous ceiling analysis (Steps 1–6) concluded that pp512 was limited to ~18 t/s
by MMQ DP4A kernel throughput. That analysis was correct *within the MMQ path* but
missed a fundamentally better alternative: bypassing MMQ entirely.

### The insight

Jetson TX2 (cc 6.2) has FP16 arithmetic at 2× the FP32 rate — the same as Tesla P100
(cc 6.0). This is unlike consumer Pascal cc 6.1 (GTX 1080 etc.) which has FP16 at
1/64× rate. llama.cpp's `ggml_cuda_should_use_mmq()` returns `true` for **all**
quantized types on pre-Volta GPUs (the check is `!fp16_mma_hardware_available(cc)`),
unconditionally routing to the MMQ DP4A kernel.

For simple quant types like Q4_0, this is fine — the DP4A dequantization is cheap and
the kernel is well-tuned. But for K-quants (Q4_K, Q5_K, Q6_K), the super-block
dequantization inside the MMQ kernel is complex (scales, mins, nested bit unpacking)
and severely under-utilizes the DP4A units.

The alternative path — cuBLAS — first dequantizes Q4_K→FP16 via a dedicated convert
kernel, then calls `maxwell_hgemm` (NVIDIA's tuned FP16 GEMM). The one-time dequant
cost is amortized over the full GEMM, and `hgemm` runs at near-peak FP16 FLOPS.

### nsys profiling confirmed the bottleneck

**Before (MMQ path):** `mul_mat_q<…Q4_K…>` consumed 49,179 ms out of 73,733 ms total
GPU time (66.7%). Average 407 µs per call.

**After (cuBLAS path):** `maxwell_hgemm_128x32_nn` + `convert_unary` replaced MMQ.
Total GPU time dropped from 73,733 ms to 24,689 ms (-66%). The matmul portion went
from 49,179 ms to 4,210 ms (-91%).

### What was changed

`ggml/src/ggml-cuda/mmq.cu`, in `ggml_cuda_should_use_mmq()`, added a new early-return
**before** the `GGML_CUDA_FORCE_MMQ` short-circuit:

```c
// On pre-Volta NVIDIA GPUs with fast FP16 (cc 6.0, 6.2 — NOT 6.1),
// cuBLAS FP16 GEMM outperforms MMQ DP4A for K-quant batched matmuls.
if (GGML_CUDA_CC_IS_NVIDIA(cc) && !fp16_mma_hardware_available(cc)
        && fast_fp16_hardware_available(cc) && ne11 >= 2) {
    return false;
}
```

This returns `false` (= use cuBLAS) when:
1. NVIDIA GPU (not AMD)
2. Pre-Volta (no tensor cores / FP16 MMA)
3. Has fast FP16 hardware (cc 6.0 or 6.2, excludes cc 6.1)
4. Batch size ≥ 2 (single-token decode still uses MMVQ, unaffected)

The `ne11 >= 2` threshold was tested at 2, 8, and 64 — all produced the same result
since pp512 always dispatches with ne11=512. The threshold of 2 is conservative and
correct: at ne11=1 the MMVQ path is used instead (not MMQ), so this guard only
affects actual batched matmuls.

### Other approaches tested and rejected

- **CUDA graphs on Pascal:** Guard changed from `cc < AMPERE` to `cc < PASCAL`,
  graphs work (258 launches, 128 reused) but zero performance benefit — no
  hardware-accelerated graph dispatch on Pascal, `cudaGraphLaunch` costs ~8.7 ms
  each, and CPU overhead already overlaps GPU execution. Reverted.

- **`GGML_CUDA_FORCE_MMQ=ON` removal:** Tested building without it — identical
  results because the new early-return fires before the `FORCE_MMQ` check.

### Result

```
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |  1 |           pp512 |        137.76 ± 0.53 |
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |  1 |           tg128 |          7.46 ± 0.00 |
```

pp512: 18.24 → 137.76 t/s (+655%) ✓  tg128: 7.49 → 7.46 t/s (flat, within noise) ✓

### Cumulative progress from Step 0

| Metric | Step 0 baseline | Step 7 | Improvement |
|--------|---------------:|-------:|------------:|
| pp512  | 17.67 t/s      | 137.76 t/s | +680% |
| tg128  | 6.64 t/s       | 7.46 t/s   | +12.3% |

### Why this was missed earlier

The ceiling analysis in Steps 1–6 focused on tuning *within* the MMQ kernel (tile
sizes, occupancy, shmem budgets). It correctly concluded that MMQ was at its ceiling.
The breakthrough came from questioning the premise: why use MMQ at all when cuBLAS
can leverage the TX2's fast FP16 hardware? The llama.cpp dispatch logic treats all
pre-Volta GPUs identically, but cc 6.0/6.2 (Tesla P100, Jetson TX2) have fundamentally
different FP16 performance from cc 6.1 (consumer Pascal).

## Step 8
MMVQ `rows_per_cuda_block` 1→2 for Pascal ncols_dst=1 decode.
**tg128: 7.46 → 8.02 t/s (+7.5%).**

### Background

After Step 7, MMVQ (mul_mat_vec_q) is now the dominant kernel for tg128 decode,
consuming ~62% of total GPU time. With `nwarps=2` and `rows_per_cuda_block=1`,
each block computes a single output row. For hidden_dim=4096, this launches 4096
blocks. With only 2 SMs on the TX2, those blocks must serialize through ~200+ waves
(even with ~14-20 concurrent blocks per SM from the low register footprint).

### What was changed

`ggml/src/ggml-cuda/mmvq.cu`, in `calc_rows_per_block()`: for `MMVQ_PARAMETERS_PASCAL`
with `ncols_dst=1`, return 2 instead of 1. This makes each CUDA block compute 2
adjacent output rows, halving the grid from 4096 to 2048 blocks.

Each thread now does two `vec_dot_q_cuda()` calls per inner loop iteration (one per
row) using the same Q8 activation data, providing data reuse. The cost is slightly
higher register pressure: +21 regs for Q4_K non-fused (51→72), +8 for fused (64→72),
+22 for Q6_K (40→62).

### Alternatives tested

- `rows_per_cuda_block=4`: tg128=7.95 — worse than 2. The extra register pressure
  from 4 accumulator rows causes occupancy loss (14→~10 blocks/SM) that outweighs
  the grid reduction benefit.

### Result

```
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |  1 |           pp512 |        137.85 ± 0.23 |
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |  1 |           tg128 |          8.01 ± 0.01 |
```

pp512: 137.76 → 137.85 t/s (flat) ✓  tg128: 7.46 → 8.01 t/s (+7.4%) ✓

### Cumulative progress from Step 0

| Metric | Step 0 baseline | Step 8 | Improvement |
|--------|---------------:|-------:|------------:|
| pp512  | 17.67 t/s      | 137.85 t/s | +680% |
| tg128  | 6.64 t/s       | 8.01 t/s   | +20.6% |
