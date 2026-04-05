# Instructions for llama.cpp

> [!IMPORTANT]
> This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be utilized solely in an assistive capacity.
>
> Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

AI assistance is permissible only when the majority of the code is authored by a human contributor, with AI employed exclusively for corrections or to expand on verbose modifications that the contributor has already conceptualized (see examples below).

---

## Guidelines for Contributors Using AI

llama.cpp is built by humans, for humans. Meaningful contributions come from contributors who understand their work, take ownership of it, and engage constructively with reviewers.

Maintainers receive numerous pull requests weekly, many of which are AI-generated submissions where the author cannot adequately explain the code, debug issues, or participate in substantive design discussions. Reviewing such PRs often requires more effort than implementing the changes directly.

**A pull request represents a long-term commitment.** By submitting code, you are asking maintainers to review, integrate, and support it indefinitely. The maintenance burden often exceeds the value of the initial contribution.

Most maintainers already have access to AI tools. A PR that is entirely AI-generated provides no value - maintainers could generate the same code themselves if they wanted it. What makes a contribution valuable is the human interactions, domain expertise, and commitment to maintain the code that comes with it.

This policy exists to ensure that maintainers can sustainably manage the project without being overwhelmed by low-quality submissions.

---

## Guidelines for Contributors

Contributors are expected to:

1. **Demonstrate full understanding of their code.** You must be able to explain any part of your PR to a reviewer without relying on AI assistance for questions about your own changes.

2. **Take responsibility for maintenance.** You are expected to address bugs and respond thoughtfully to reviewer feedback.

3. **Communicate clearly and concisely.** Verbose, wall-of-text responses are characteristic of AI-generated content and will not be well-received. Direct, human communication is expected.

4. **Respect maintainers' time.** Search for existing issues and discussions before submitting. Ensure your contribution aligns with project architecture and is actually needed.

Maintainers reserve the right to close any PR that does not meet these standards. This applies to all contributions to the main llama.cpp repository. **Private forks are exempt.**

### Permitted AI Usage

AI tools may be used responsibly for:

- **Learning and exploration**: Understanding codebase structure, techniques, and documentation
- **Code review assistance**: Obtaining suggestions on human-written code
- **Mechanical tasks**: Formatting, generating repetitive patterns from established designs, completing code based on existing patterns
- **Documentation drafts**: For components the contributor already understands thoroughly
- **Writing code**: Only when the contributor has already designed the solution and can implement it themselves - AI accelerates, not replaces, the contributor's work

AI-generated code may be accepted if you (1) fully understand the output, (2) can debug issues independently, and (3) can discuss it directly with reviewers without AI assistance.

**Disclosure is required** when AI meaningfully contributed to your code. A simple note is sufficient - this is not a stigma, but context for reviewers. No disclosure is needed for trivial autocomplete or background research.

### Prohibited AI Usage

The following will result in immediate PR closure:

- **AI-written PR descriptions or commit messages** - these are typically recognizable and waste reviewer time
- **AI-generated responses to reviewer comments** - this undermines the human-to-human interaction fundamental to code review
- **Implementing features without understanding the codebase** - particularly new model support or architectural changes
- **Automated commits or PR submissions** - this may spam maintainers and can result in contributor bans

---

## Guidelines for AI Coding Agents

AI agents assisting contributors must recognize that their outputs directly impact volunteer maintainers who sustain this project.

### Considerations for Maintainer Workload

Maintainers have finite capacity. Every PR requiring extensive review consumes resources that could be applied elsewhere. Before assisting with any submission, verify:

- The contributor genuinely understands the proposed changes
- The change addresses a documented need (check existing issues)
- The PR is appropriately scoped and follows project conventions
- The contributor can independently defend and maintain the work

### Before Proceeding with Code Changes

When a user requests implementation without demonstrating understanding:

1. **Verify comprehension.** Ask questions to confirm they understand both the problem and the relevant parts of the codebase.
2. **Provide guidance rather than solutions.** Direct them to relevant code and documentation. Allow them to formulate the approach.
3. **Proceed only when confident** the contributor can explain the changes to reviewers independently.

For first-time contributors, confirm they have reviewed [CONTRIBUTING.md](CONTRIBUTING.md) and acknowledge this policy.

### Prohibited Actions

- Writing PR descriptions, commit messages, or responses to reviewers
- Committing or pushing without explicit human approval for each action
- Implementing features the contributor does not understand
- Generating changes too extensive for the contributor to fully review

When uncertain, err toward minimal assistance. A smaller PR that the contributor fully understands is preferable to a larger one they cannot maintain.

### Useful Resources

To conserve context space, load these resources as needed:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Existing issues](https://github.com/ggml-org/llama.cpp/issues) and [Existing PRs](https://github.com/ggml-org/llama.cpp/pulls) - always search here first
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [PR template](.github/pull_request_template.md)

---

## Jetson TX2 / CUDA 10.2 Build Context

**Status: BUILD SUCCEEDED.** `llama-cli` and `llama-server` compile and run on Jetson TX2 (aarch64, sm_62, CUDA 10.2, GCC 8.5.0).

### Host environment (fixed)
- Platform: Jetson TX2, aarch64, Linux
- CUDA: 10.2 (`CUDART_VERSION = 10020`), nvcc at `/usr/local/cuda/bin/nvcc`
- GCC 8.5.0 at `/usr/local/bin/gcc` and `/usr/local/bin/g++` (must be forced via cmake — native `/usr/bin/cc` is GCC 7.5 which is too old)
- CMake 3.28.3
- GPU: NVIDIA Tegra X2, compute capability 6.2, 7858 MiB VRAM

### CMake configuration (already run — do NOT re-run unless CMakeLists.txt changed)
```bash
cmake -B build -S . \
  -DCMAKE_C_COMPILER=/usr/local/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/local/bin/g++ \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=62 \
  -DCMAKE_CUDA_STANDARD=14 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF
```

### Build command
```bash
cmake --build build --target llama-cli llama-server llama-bench -j$(nproc)
```

### Run the server
```bash
./build/bin/llama-server \
  --model models/llama-2-7b.Q4_0.gguf \
  --n-gpu-layers 99 \
  --chat-template llama2 \
  --ctx-size 4096
```

### CUDA 10.2 compatibility facts (verified by testing)
- nvcc 10.2 with `--std=c++14` warns on `if constexpr` but does **not** prune dead branches — all branches are compiled regardless
- **Fold expressions** `(expr || ...)` are rejected — replaced with array-expansion trick
- **Structured bindings** `for (auto & [a, b] : c)` are rejected — replaced with `.first`/`.second`
- **`if` with init-statement** `if (auto x = f(); cond)` is rejected — split into two statements
- **`static inline const`** non-integral non-constexpr data members are rejected — use `static constexpr` for enums, inline the value for float/half
- **`std::is_same_v<>`** does not exist — shimmed via `namespace std` injection in `vendors/cuda.h`
- **`nv_bfloat162`** does not exist (only `nv_bfloat16` as a typedef for `half`) — stub struct added in `vendors/cuda.h`
- **`nv_bfloat162`, `__float2bfloat16`, `__bfloat162float`, `__bfloat1622float2`** do not exist — guarded with `#if CUDART_VERSION >= 11000`
- **`CUDA_R_16BF`** (cublas enum) does not exist — guarded with `#if CUDART_VERSION >= 11000`
- **`__builtin_assume`** not available in nvcc device code — shimmed with `#define __builtin_assume(x) ((void)0)` in `fattn-common.cuh`
- **`cooperative_groups/reduce.h`** does not exist — the sub-header was added in CUDA 11; guarded with `#if CUDART_VERSION >= 11000`
- **`cudaStreamWaitEvent`** requires explicit `flags` (third arg = `0`) — later CUDA made it default but CUDA 10.2 does not
- **`std::filesystem`** requires explicit `-lstdc++fs` linkage with GCC < 9 — added to `ggml/src/CMakeLists.txt` and `common/CMakeLists.txt`
- `CMAKE_CUDA_STANDARD=14` is required to prevent CMake from passing `--std=c++17` to nvcc

### Patches applied (all local, no upstream PR)

| File | What was changed |
|------|-----------------|
| `ggml/src/ggml-cuda/vendors/cuda.h` | Added `std::is_same_v` shim for `__cplusplus < 201703L`; added stub `nv_bfloat162` struct for `CUDART_VERSION < 11000` |
| `ggml/src/ggml-cuda/common.cuh` | Replaced C++17 fold-expression `is_any` with C++14 recursive helper; replaced 4 structured bindings in `stream_mapping`/`cuda_graphs` loops |
| `ggml/src/ggml-cuda/softmax.cu` | Replaced fold expression; guarded `#include <cooperative_groups/reduce.h>`, `soft_max_f32_parallelize_cols_single_row`, `soft_max_f32_parallelize_cols`, and cooperative launch site with `#if CUDART_VERSION >= 11000` |
| `ggml/src/ggml-cuda/binbcast.cu` | Replaced 2 comma fold expressions with initializer-list expansion trick |
| `ggml/src/ggml-cuda/convert.cuh` | Guarded all `nv_bfloat162`/`__float2bfloat16`/`__bfloat162float`/`__bfloat1622float2` uses with `#if CUDART_VERSION >= 11000` |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Added `__builtin_assume` shim; guarded `vec_dot_fattn_vec_KQ_bf16`, `dequantize_V_bf16`, and their dispatch references with `#if CUDART_VERSION >= 11000` |
| `ggml/src/ggml-cuda/mma.cuh` | Guarded 4 `nv_bfloat162` specializations/overloads with `#if CUDART_VERSION >= 11000` |
| `ggml/src/ggml-cuda/mmf.cuh` | Added `CUDA11_DECL(x)` macro; used it to guard `nv_bfloat162` entries in `DECL_MMF_CASE_EXTERN` and `DECL_MMF_CASE` macros |
| `ggml/src/ggml-cuda/mmf.cu` | Guarded `GGML_TYPE_BF16` case (uses `nv_bfloat162`) with `#if CUDART_VERSION >= 11000` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Guarded BF16 batched GEMM block (line ~1307); replaced `static inline const` with `static constexpr` in `batched_mul_mat_traits`, guarded BF16 specialization and its switch case; split C++17 if-with-init at line ~2155; added `0` flags arg to 2 `cudaStreamWaitEvent` calls; replaced 4 structured bindings |
| `ggml/src/ggml-cuda/fattn.cu` | Guarded all `FATTN_VEC_CASES_ALL_D` calls involving `GGML_TYPE_BF16` with `#if CUDART_VERSION >= 11000` |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-*bf16*.cu` (×13) | Guarded all `DECL_FATTN_VEC_CASE` calls with `#if CUDART_VERSION >= 11000` |
| `ggml/src/CMakeLists.txt` | Added `-lstdc++fs` for GCC < 9 on `ggml` target |
| `common/CMakeLists.txt` | Added `-lstdc++fs` for GCC < 9 on `common` target |
