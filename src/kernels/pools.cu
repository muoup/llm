#include "pools.hpp"

#include <cublas.h>
#include <cuda_runtime_api.h>

template <typename T>
T* gpu_allocate() {
    void* ptr;
    cudaMallocManaged(&ptr, sizeof(T));
    cudaDeviceSynchronize();
    return (T*)ptr;
}

template <typename T>
void gpu_free(T* ptr) {
    cudaFree(ptr);
}

kernel::ObjectPool<float*> kernel::global_gpu_float_pool
    = kernel::ObjectPool<float*>(gpu_allocate<float>, gpu_free, nullptr, 8);
kernel::ObjectPool<std::uint16_t*> kernel::global_gpu_half_pool
    = kernel::ObjectPool<std::uint16_t*>(gpu_allocate<std::uint16_t>,
                                         gpu_free,
                                         nullptr,
                                         8);
kernel::ObjectPool<std::uint16_t*> kernel::global_gpu_bf16_pool
    = kernel::ObjectPool<std::uint16_t*>(gpu_allocate<std::uint16_t>,
                                         gpu_free,
                                         nullptr,
                                         8);
kernel::ObjectPool<bool*> kernel::global_gpu_bool_pool
    = kernel::ObjectPool<bool*>(gpu_allocate<bool>, gpu_free, nullptr, 8);

kernel::KernelStreamPool kernel::new_kernel_stream_pool(size_t pool_size) {
    return kernel::KernelStreamPool(create_kernel_stream, destroy_kernel_stream,
                                    nullptr, pool_size);
}

kernel::MatmulHandlePool kernel::new_matmul_handle_pool(size_t pool_size) {
    return kernel::MatmulHandlePool(create_matmul_handle, destroy_matmul_handle,
                                    nullptr, pool_size);
}
