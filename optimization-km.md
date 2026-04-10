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

## Step 9
MMVQ `nwarps` 2→1 for Pascal ncols_dst=1 decode — eliminates inter-warp reduction.
**E2B tg128: 8.01 → 10.08 t/s (+25.8%). E4B tg128: 4.60 → 5.26 t/s (+14.3%).**

### Background

With nwarps=2 and rpb=2 from Step 8, each MMVQ block has 64 threads (2 warps).
The two warps split the K-dimension loop iterations, then synchronize via shared
memory to reduce partial sums. This synchronization has real cost:
1. Warp 0 writes partial sum to shared memory
2. `__syncthreads()` barrier
3. Warp 0 reads warp 1's partial sum and adds it
4. Warp 1's threads do nothing after the barrier — 50% thread waste in the epilogue

With only 2 SMs on TX2, occupancy is limited by register pressure (72 regs/thread
for Q4_K), not by thread count. Doubling warps does not help occupancy.

### What was changed

`ggml/src/ggml-cuda/mmvq.cu`, in `calc_nwarps()`: for `MMVQ_PARAMETERS_PASCAL`
with `ncols_dst=1`, return 1 instead of 2. Each block is now exactly 1 warp (32
threads) computing 2 rows. The inner loop runs 1 extra iteration per thread (3 vs
2 iterations for 2560-column rows in E2B) but avoids all synchronization overhead.

### Alternatives tested

- `nwarps=1, rpb=1`: 4.64 t/s E4B (worse than rpb=2's 5.26) — loses Q8 activation
  reuse across row pairs. Confirms rpb=2 is still important.
- `nwarps=2, rpb=4`: 4.56 t/s E4B (neutral) — the 4-accumulator register pressure
  offsets the grid reduction. Rejected.
- `nwarps=4`: 3.85 t/s E4B (-15.9%) — severe register pressure, reduced occupancy.
  Rejected.

### Result

```
E2B (from SD card, ub=512):
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |  1 |           pp512 |        137.56 ± 3.78 |
| gemma4 E2B Q4_K - Medium       |   2.88 GiB |     4.65 B | CUDA       |  99 |  1 |           tg128 |         10.08 ± 0.00 |

E4B (local storage, ub=256):
| gemma4 E4B Q4_K - Medium       |   4.62 GiB |     7.52 B | CUDA       |  99 |  1 |           pp512 |         53.21 ± 0.56 |
| gemma4 E4B Q4_K - Medium       |   4.62 GiB |     7.52 B | CUDA       |  99 |  1 |           tg128 |          5.26 ± 0.01 |
```

pp512: flat (uses cuBLAS, not MMVQ) ✓  tg128: +25.8% E2B, +14.3% E4B ✓

### E2B cumulative progress from Step 0

| Metric | Step 0 baseline | Step 9 | Improvement |
|--------|---------------:|-------:|------------:|
| pp512  | 17.67 t/s      | ~137.6 t/s | +679% |
| tg128  | 6.64 t/s       | 10.08 t/s  | +51.8% |

---

# E4B Optimization (Gemma 4 E4B Q4_K_M, 4.62 GiB, 7.52B params)

## E4B Step 0
Baseline — with all E2B optimizations applied (cuBLAS routing, rpb=2, nwarps=1, -fa 1).

E4B differs from E2B in several important ways:
- 42 layers (vs 35), embedding=2560, FFN=10240
- head_k=512, head_v=512 (E2B has 256) — requires D=512 FA kernels
- 8 attention heads, 2 KV heads (GQA ratio 4)
- shared_kv_layers=18, sliding_window=512
- embedding_length_per_layer_input=256 (per-layer f32 projection matrices)
- vocab_size=262144 → lm_head is [262144, 2560] Q4_K
- 253 Q4_K tensors, 42 Q6_K tensors, 423 f32 tensors

### SD card vs local storage

E4B was initially benchmarked from SD card at `/mnt/sdcard/`. After copying to
local storage (`models/`):

| Storage  | pp512    | tg128    |
|----------|----------|----------|
| SD card  | 22.89    | 4.32     |
| Local    | 42.39    | 4.56     |

pp512 improved +85% on local storage. SD card I/O is a massive bottleneck for
prompt processing. All subsequent E4B benchmarks use local storage.

### ubatch size sweep

E4B benefits from ub=256, unlike E2B which is optimal at ub=512. The reason is
likely that E4B's larger per-layer temp buffers create memory pressure at ub=512.

| ub   | pp512  | tg128 |
|------|--------|-------|
| 128  | 48.43  | 4.56  |
| 224  | 43.40  | 4.56  |
| 256  | 50.12  | 4.56  |
| 288  | 42.00  | 4.56  |
| 512  | 42.01  | 4.56  |

Non-power-of-2 values are worse than their neighbors. ub=256 is optimal (+19% over
default ub=512). tg128 is unaffected by ubatch.

### Baseline with nwarps=2 (before Step 9)

```
| gemma4 E4B Q4_K - Medium       |   4.62 GiB |     7.52 B | CUDA       |  99 |  1 |           pp512 |         50.12 ± 0.78 |
| gemma4 E4B Q4_K - Medium       |   4.62 GiB |     7.52 B | CUDA       |  99 |  1 |           tg128 |          4.60 ± 0.01 |

build: e2f295dbd (8728), ub=256
```

### Baseline with nwarps=1 (after Step 9)

```
| gemma4 E4B Q4_K - Medium       |   4.62 GiB |     7.52 B | CUDA       |  99 |  1 |           pp512 |         53.21 ± 0.56 |
| gemma4 E4B Q4_K - Medium       |   4.62 GiB |     7.52 B | CUDA       |  99 |  1 |           tg128 |          5.26 ± 0.01 |

build: bb54c7229 (8728), ub=256
```

pp512: 50.12 → 53.21 t/s (+6.2%, thermal variance) ✓  tg128: 4.60 → 5.26 t/s (+14.3%) ✓

## E4B Profiling Analysis

Profiled tg128 decode with nwarps=1 (`/tmp/llama-profile-e4b-nw1.sqlite`).
Wall time per decode step: ~213.2 ms (~4.69 t/s).

### Kernel time breakdown (tg128 per-step)

| Component | Time/step | % of wall | Detail |
|-----------|-----------|-----------|--------|
| Fused gate_up Q4_K MMVQ | ~70.6 ms | 33.1% | 5120 grid, 72 regs, 1669 µs avg, 12-24% BW util |
| Q6_K MMVQ (2560 rows) | ~31.1 ms | 14.6% | 1280 grid, 64 regs, 1471 µs avg |
| Q4_K plain MMVQ (2560 rows) | ~24.9 ms | 11.7% | 1280 grid, 72 regs, 393 µs avg |
| lm_head Q4_K MMVQ (262144 rows) | ~20.9 ms | 9.8% | 131072 grid, 20700 µs avg |
| f32 matmul (2560 rows, 128 threads) | ~11.0 ms | 5.2% | Per-layer f32 projections |
| f32 matmul (256 rows, 256 threads) | ~4.4 ms | 2.1% | |
| fp16 matmul (10752 rows) | ~2.9 ms | 1.4% | 129 calls only |
| Flash attention | ~3.5 ms | 1.7% | D=256 and D=512 tile kernels |
| Other | ~4.6 ms | | norms, rope, quantize, etc. |
| **GPU total** | ~185.3 ms | 86.9% | |
| **Non-kernel overhead** | ~28.0 ms | 13.1% | Launch overhead, CPU work |
| **Wall total** | ~213.2 ms | 100% | |

### Key insight: MMVQ is compute-bound, not bandwidth-bound

The fused gate_up kernel (33% of total time) reads only 14.7-29.5 MB per call but
takes 1669 µs — achieving only 12-24% of peak memory bandwidth (58.4 GB/s). The
bottleneck is computational: the `vec_dot_q4_K_q8_1` dequantization involves scales,
mins, and nested bit unpacking per super-block, and the fused path does 2× vec_dot
calls for gate+up.

cuBLAS GEMV for tg128 was analyzed and rejected: dequant Q4_K→fp16 produces a
52.4 MB intermediate matrix that must be written then read, roughly equaling the
MMVQ compute cost.

### Flash attention on E4B

D=512 head size requires the TILE FA kernel (no VEC or WMMA kernel supports D=512).
Pascal has no tensor cores, so TILE is the only option. FA is only 1.7% of tg128
time — not a meaningful optimization target. FA nbatch_fa=32 for D=512 was tested
and regressed -3.3% pp512.

### KV cache quantization on E4B

Without `-fa 1`, q8_0/q4_0 KV cache causes OOM. With `-fa 1`, even f16 KV can
OOM with standard attention path. FA is required for E4B on TX2.

## E4B Experiments Tried and Rejected

1. **Batched dequant kernel**: No improvement — bandwidth-limited, not launch-overhead-limited
2. **CUBLAS_COMPUTE_32F path**: -18.6% pp512 — cuBLAS 10.2 falls back to sgemm
3. **MMVQ nwarps=4**: -15.9% tg128 — register pressure, reduced occupancy
4. **FA tile nbatch_fa=32 for D=512**: -3.3% pp512
5. **MMVQ rpb=4 with nwarps=2**: 4.56 t/s (neutral), rejected
6. **MMVQ rpb=1 with nwarps=1**: 4.64 t/s (worse — loses Q8 activation reuse)
7. **Thread count -t 2 or -t 6**: worse than -t 4
8. **MMVQ rpb=4 with nwarps=1**: 5.21 t/s (-1.0%) — extra register pressure from
   4 accumulators outweighs grid reduction even with 1 warp. Rejected.
9. **MMVQ `__launch_bounds__` minBlocks=32**: 5.15 t/s (-2.1%) — forcing register
   spilling to increase occupancy causes local memory accesses that slow the inner
   loop more than the marginal concurrency gain. Rejected.

## E4B Profiling Details (nsys)

Total kernel launches per decode step: ~1,156. At ~24 µs average launch overhead
(CUDA 10.2, ARM CPU), this accounts for ~28 ms of non-kernel overhead per step
(13.1% of wall time). This is an architectural limit of the many-small-kernel
dispatch pattern on CUDA 10.2.

GPU kernel time breakdown (per tg128 decode step):
- Q4_K MMVQ (fused + plain + lm_head): 116.4 ms (54.6%)
- Q6_K MMVQ: 31.1 ms (14.6%)
- f32 matmul: 15.4 ms (7.2%)
- Flash attention: 3.5 ms (1.6%)
- fp16 matmul: 2.9 ms (1.4%)
- Norms + rope + quantize + other: 16.0 ms (7.5%)
- Non-kernel overhead: 28.0 ms (13.1%)

The MMVQ kernels dominate at 69.2% and are **compute-bound** — the Q4_K/Q6_K
dequantization involves scales, mins, and nested bit unpacking that saturates
ALU before memory bandwidth. cuBLAS GEMV is not viable for decode (dequant
intermediate exceeds original weight bandwidth).

lm_head alone is 20.7 ms (9.8%) — a [262144, 2560] Q4_K GEMV reading 377.5 MB
at 18.2 GB/s effective (31% of 58.4 GB/s peak), compute-limited by Q4_K dequant.

## E4B Current Performance

| Metric | Original (SD, ub=512, nw=2) | Current (local, ub=256, nw=1) | Improvement |
|--------|----------------------------:|------------------------------:|------------:|
| pp512  | 22.89 t/s                   | ~53.2 t/s                     | +132%       |
| tg128  | 4.32 t/s                    | ~5.26 t/s                     | +21.8%      |

Note: pp512 gain is primarily from local storage (+85%) and ub=256 (+19%).
tg128 gain is from nwarps=1 (+14.3%).

### Performance ceiling analysis

- **tg128**: 5.26 t/s × 4.62 GiB ≈ 24.3 GB/s out of 58.4 GB/s peak (42% utilization).
  MMVQ kernels are compute-bound at 12-24% BW utilization — the gap is from Q4_K
  dequant compute overhead. Reaching BW-limited performance would require ~11.5 t/s,
  implying ~2.2× potential headroom if dequant cost could be eliminated.
- **pp512**: ~53 t/s (ub=256). cuBLAS hgemm dominates. Thermal variance ±5 t/s.

### Remaining opportunities

The MMVQ decode path is fundamentally compute-bound by Q4_K dequantization on
sm_62. Without tensor cores (Volta+), dp4a is the only integer acceleration and
is already being used. Remaining actionable avenues:

1. **Q6_K VDR increase** (1→2): restructure `vec_dot_q6_K_q8_1` to process 2
   quant positions per call, halving loop iterations. Complex code change, ~17%
   target → potential ~3% total tg128 gain if 20% faster.
2. **f32 matmul rows_per_block**: add multi-row support to `mul_mat_vec_f` for
   small GPUs. ~7% target → potential ~1-2% total gain.
3. **Persistent MMVQ kernel**: launch exactly `2×max_blocks_per_SM` blocks and
   loop over rows internally, eliminating all block-scheduling overhead. Major
   architectural change, would help lm_head (9.8%) most.
4. **Fused lm_head + sampling**: combine matmul with top-k/argmax to avoid
   materializing all 262144 logits. Major change, highly model-specific.

None of these are expected to yield more than 5% total tg128 improvement.
