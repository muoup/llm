#include "pools.cuh"
#include <cuda_runtime_api.h>

namespace kernel::matrix {

template <typename T>
T* gpu_allocate() {
    T* ptr;
    cudaMalloc(&ptr, sizeof(T));
    return ptr;
}

template <typename T>
void gpu_free(T* ptr) {
    cudaFree(ptr);
}

GPUFloatPool global_gpu_float_pool;
GPUHalfPool global_gpu_half_pool;
GPUBf16Pool global_gpu_bf16_pool;
GPUBoolPool global_gpu_bool_pool;

} // namespace kernel::matrix
