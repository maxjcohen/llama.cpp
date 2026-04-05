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

