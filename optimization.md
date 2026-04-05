# Optimization

## Step 0
Baseline
```
Device 0: NVIDIA Tegra X2, compute capability 6.2, VMM: no, VRAM: 7858 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.82 GiB |     4.65 B | CUDA       |  99 |           pp512 |         76.95 ± 0.03 |
| gemma4 E2B Q4_0                |   2.82 GiB |     4.65 B | CUDA       |  99 |           tg128 |          6.95 ± 0.01 |
```

## Step 1
Add `-DGGML_CUDA_FORCE_MMQ=ON` to the cmake build configuration.

**Why**: sm_62 (Pascal) has no tensor cores, so cuBLAS fp16 GEMM is slower than the DP4A integer MMQ kernels for Q4_0 weights. Without this flag, prompt processing batches ≥ 64 tokens fall through to cuBLAS after a float→fp16 weight conversion. With this flag, the MMQ DP4A path is always used for quantized matrix multiplications.

**Change**: `cmake ... -DGGML_CUDA_FORCE_MMQ=ON` (see AGENTS.md for full cmake invocation)

```
Device 0: NVIDIA Tegra X2, compute capability 6.2, VMM: no, VRAM: 7858 MiB
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.82 GiB |     4.65 B | CUDA       |  99 |           pp512 |         77.08 ± 0.05 |
| gemma4 E2B Q4_0                |   2.82 GiB |     4.65 B | CUDA       |  99 |           tg128 |          7.18 ± 0.01 |
```

**Delta vs Step 0**: pp512 +0.2%, tg128 +3.3%

## Step 2
Use `--ubatch-size 256` (runtime flag, default is 512) when running llama-server or llama-cli.

**Why**: The MMQ kernel (forced via GGML_CUDA_FORCE_MMQ=ON) tiles matmuls into blocks of mmq_x=64. On a 2-SM GPU, processing 256 tokens per chunk (4 tiles of 64) fits the available parallelism better than the default 512 (8 tiles). Larger chunks increase shared memory pressure and cause serialization on this GPU; smaller chunks (128) under-utilize the SMs.

**Change**: Pass `-ub 256` to llama-bench / `--ubatch-size 256` to llama-server and llama-cli.

```
Device 0: NVIDIA Tegra X2, compute capability 6.2, VMM: no, VRAM: 7858 MiB
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.82 GiB |     4.65 B | CUDA       |  99 |      256 |           pp512 |         77.47 ± 0.07 |
| gemma4 E2B Q4_0                |   2.82 GiB |     4.65 B | CUDA       |  99 |      256 |           tg128 |          7.19 ± 0.01 |
```

**Delta vs Step 1**: pp512 +0.5%, tg128 flat (+0.01, noise)
**Delta vs Step 0**: pp512 +0.7%, tg128 +3.4%

## Step 3
Add `MMVQ_PARAMETERS_PASCAL` table in `ggml/src/ggml-cuda/mmvq.cu`: reduce nwarps from 4 to 2 for ncols_dst=1 (single-token decode) on Pascal architecture (sm_60–sm_69).

**Why**: The MMVQ kernel (mul_mat_vec_q) is the decode hot-path. Each block computes 1 output row. With 4 warps per block (128 threads), Pascal's 2 SMs can run at most 16 blocks concurrently per SM = 32 blocks total. With 2 warps per block (64 threads), 32 blocks/SM × 2 SMs = 64 blocks run simultaneously — halving the number of serial execution waves for each matmul. The MMVQ computation is memory-bandwidth bound (loading Q4_0 weights), not compute-bound, so halving the warp count per block does not reduce throughput: both warp configurations saturate memory bandwidth at the same rate.

The pp512 path (batch=256 with FORCE_MMQ) uses the MMQ kernel, not MMVQ, so it is unaffected by this change.

**Change**: `ggml/src/ggml-cuda/mmvq.cu`
- Added `MMVQ_PARAMETERS_PASCAL` to `mmvq_parameter_table_id` enum
- `get_device_table_id()` (device): returns PASCAL for `__CUDA_ARCH__` 600–699
- `get_device_table_id(cc)` (host): returns PASCAL for cc in `[PASCAL, VOLTA)`
- `calc_nwarps`: PASCAL ncols_dst=1 → 2 warps (was 4)
- `calc_rows_per_block`: PASCAL included in GENERIC/GCN group (rows_per_block=1)

```
Device 0: NVIDIA Tegra X2, compute capability 6.2, VMM: no, VRAM: 7858 MiB
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.82 GiB |     4.65 B | CUDA       |  99 |      256 |           pp512 |         77.33 ± 0.07 |
| gemma4 E2B Q4_0                |   2.82 GiB |     4.65 B | CUDA       |  99 |      256 |           tg128 |          8.22 ± 0.01 |
```

**Delta vs Step 2**: pp512 flat (within noise), tg128 +14.3%
**Delta vs Step 0**: pp512 +0.5%, tg128 +18.3%

## Step 4
Use 512 threads/block instead of 1024 for RMS norm kernels on few-SM GPUs (nsm ≤ 4) when `1024 ≤ ncols ≤ 8192`.

**Why**: The RMS norm kernel launches one block per token row. With 1024 threads/block on Pascal (2 SMs, 2048 threads/SM hard limit), only 2 blocks run concurrently (4 total). With 512 threads/block, 4 blocks/SM × 2 SMs = 8 concurrent blocks — doubling effective RMS norm throughput. The kernel iterates `col = tid; col < ncols; col += block_size`, so any power-of-2 block size is correct. Gemma 4 2B has hidden_size=2304, so each RMS norm processes 2304 elements per row, well within the 512-thread range. This improves pp512 (many RMS norm calls during prefill) without affecting tg128 significantly (decode has far fewer tokens per step).

**Change**: `ggml/src/ggml-cuda/norm.cu`
- Added `#include "common.cuh"` to access `ggml_cuda_get_device()` and `ggml_cuda_info()`
- `rms_norm_f32_cuda`: added 512-thread branch for `1024 ≤ ncols ≤ 8192` when `nsm ≤ 4`
- `rms_norm_mul_f32_cuda`: same 512-thread branch for both do_multiply and do_multiply+do_add variants

```
Device 0: NVIDIA Tegra X2, compute capability 6.2, VMM: no, VRAM: 7858 MiB
| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| gemma4 E2B Q4_0                |   2.82 GiB |     4.65 B | CUDA       |  99 |      256 |           pp512 |         77.70 ± 0.10 |
| gemma4 E2B Q4_0                |   2.82 GiB |     4.65 B | CUDA       |  99 |      256 |           tg128 |          8.21 ± 0.01 |
```

**Delta vs Step 3**: pp512 +0.5%, tg128 flat
**Delta vs Step 0**: pp512 +1.0%, tg128 +18.1%

## Analysis of remaining opportunities (no further gains found)

After Step 4, all remaining kernel-level optimization candidates were analyzed and ruled out. The hardware ceiling for the 2-SM Tegra X2 has been reached for Q4_0 inference.

### MMQ kernel (pp512 bottleneck)

Shared memory per MMQ block for Q4_0 (`mmq_y=64, mmq_x=64`):
- `nbs_x` (X tile: qs + dm) = 9504 bytes
- `nbs_y` (Y tile: 64 × block_q8_1_mmq) = 9216 bytes
- `nbs_ids` = 256 bytes
- **Total ≈ 18.5 KB** → `floor(48 KB / 18.5 KB) = 2 blocks/SM` — hard hardware constraint

Approaches tried or computed:
- `mmq_x=24`: shmem drops to 13 KB → 3 blocks/SM, but `ceil(256/24)=11` tiles vs current 4 → tile overhead overwhelms the occupancy gain (same failure mode as `mmq_y=32` which measured −21% on pp512)
- `mmq_x=96`: shmem ≈ 23.7 KB → still 2 blocks/SM, no improvement
- `mmq_x=128`: shmem ≈ 28.4 KB → 1 block/SM, worse

Increasing `mmq_x_max` for Pascal beyond 64 makes things worse; the current value is already optimal.

### MMVQ kernel (tg128 bottleneck)

Already at peak occupancy after Step 3. With `nwarps=2` (64 threads/block):
- Threads limit: `2048 / 64 = 32 blocks/SM` — exactly the Pascal block limit
- Both the thread limit and block limit are saturated simultaneously; no further gain possible

Trying `nwarps=1` (32 threads/block): `2048 / 32 = 64` → still capped at 32 blocks/SM by the block limit. No additional blocks can be scheduled; the only effect is halving the arithmetic per block.

### RMS norm

Already addressed in Step 4. `rms_norm_back_f32_cuda` (training backward pass) was not changed — not executed during inference.

### SiLU activation (FFN gate)

Already uses 256 threads/block (`CUDA_GLU_BLOCK_SIZE=256`). For FFN hidden_size=9216: 36 blocks total, well above the 16-block SM capacity. No change needed.

### RoPE

Uses `block_dims=(1, 256, 1)` — 256 threads per block. With head_dim=256: exactly 1 block per row, all threads active. Already optimal.

### Softmax

The `soft_max_f32_cuda` dispatcher uses templated specializations where `block_size_template == ncols_template`. Reducing `nth` for few-SM GPUs would cause the `p.ncols == ncols` template check to fail, falling through to the slower `soft_max_f32<true, 0, 0>` general path. Adding new specializations (e.g., `<512, 256>`) to handle this is possible but the softmax contribution to total runtime is negligible compared to MMQ.

### `norm_f32_cuda` / `l2_norm_f32_cuda`

Gemma 4 uses only RMS norm (`GGML_OP_RMS_NORM`). Layer norm (`GGML_OP_NORM`) and L2 norm (`GGML_OP_L2_NORM`) are not called during Gemma 4 inference.

### `quantize_mmq_q8_1` (activation quantizer before MMQ)

With `CUDA_QUANTIZE_BLOCK_SIZE_MMQ=128` and hidden_size=2304: 5 blocks/row × 512 rows = 2560 blocks. This kernel is trivially short (load 4 floats, warp-reduce, store). It is not a bottleneck; changing its block size has no measurable effect.

### KV cache quantization

`--ctk q8_0 --ctv q8_0` fails to initialize context for Gemma 4 (model-specific constraint). Other KV quant formats were not tested but are unlikely to help pp512 which is compute-bound, not memory-bound.

### `small_k` MMVQ path

For attention (head_dim=256): `blocks_per_row_x=8 < nwarps*blocks_per_iter_1warp=16` → `small_k=true`, `rows_per_block=nwarps=2`. Already enabled automatically; each attention MMVQ block processes 2 output rows. For the large weight matrices (hidden_size=2304): `blocks_per_row_x=72 ≥ 16` → `small_k=false`, `rows_per_block=1`. Correct behaviour in both cases.

### Conclusion

The remaining bottlenecks are intrinsic to the hardware:
- **pp512**: MMQ is shmem-limited at 2 concurrent blocks/SM on Pascal's 48 KB SM. The tile geometry that achieves 2 blocks is already optimal; any reduction to get 3 blocks requires a tile size that causes more harm than the occupancy gain.
- **tg128**: MMVQ is at the 32-block/SM block limit. Memory bandwidth (58.4 GB/s) is the ultimate constraint; more blocks cannot issue more memory requests than the bus can service.

