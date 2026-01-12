#pragma once
#include <kernels/scheduling.hpp>
#include <cuda_runtime_api.h>

namespace kernel::matrix {

template <typename T>
T* gpu_allocate();

template <typename T>
void gpu_free(T* ptr);

using GPUFloatPool = kernel::ObjectPool<float*, 8, nullptr, 
                                      gpu_allocate<float>, gpu_free<float>>;
using GPUHalfPool = kernel::ObjectPool<uint16_t*, 8, nullptr, 
                                      gpu_allocate<uint16_t>, gpu_free<uint16_t>>;
using GPUBf16Pool = kernel::ObjectPool<uint16_t*, 8, nullptr, 
                                       gpu_allocate<uint16_t>, gpu_free<uint16_t>>;
using GPUBoolPool = kernel::ObjectPool<bool*, 8, nullptr, 
                                      gpu_allocate<bool>, gpu_free<bool>>;

extern GPUFloatPool global_gpu_float_pool;
extern GPUHalfPool global_gpu_half_pool;
extern GPUBf16Pool global_gpu_bf16_pool;
extern GPUBoolPool global_gpu_bool_pool;

} // namespace kernel::matrix
