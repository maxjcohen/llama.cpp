#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

// CUDA 10.2 / nvcc --std=c++14 compatibility: std::is_same_v is C++17.
// Inject it into namespace std when compiling below C++17 standard.
#if __cplusplus < 201703L
namespace std {
    template<class A, class B>
    constexpr bool is_same_v = is_same<A, B>::value;
} // namespace std
#endif // __cplusplus < 201703L

#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif // GGML_USE_NCCL

#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080

#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#if CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020

// CUDA < 11000 does not define nv_bfloat162; provide a stub so that templates
// that reference it as a type (e.g. in std::is_same_v branches) still compile.
// Actual instantiations with nv_bfloat162 must be guarded with
// #if CUDART_VERSION >= 11000.
#if CUDART_VERSION < 11000
struct nv_bfloat162 {
    nv_bfloat16 x;
    nv_bfloat16 y;
};
#endif // CUDART_VERSION < 11000
