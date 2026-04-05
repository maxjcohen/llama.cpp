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

