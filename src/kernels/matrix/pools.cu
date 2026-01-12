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

void gpu_free(void* ptr) {
    cudaFree(ptr);
}

GPUFloatPool global_gpu_float_pool;
GPUHalfPool global_gpu_half_pool;
GPUBf16Pool global_gpu_bf16_pool;
GPUBoolPool global_gpu_bool_pool;

// Explicit template instantiations
template float* gpu_allocate<float>();
template void gpu_free<float>(float*);
template uint16_t* gpu_allocate<uint16_t>();
template void gpu_free<uint16_t>(uint16_t*);
template bool* gpu_allocate<bool>();
template void gpu_free<bool>(bool*);

} // namespace kernel::matrix
