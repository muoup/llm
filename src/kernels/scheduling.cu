#include "scheduling.hpp"

#include <cublas_api.h>
#include <cuda_runtime.h>

kernel::kernel_stream_t kernel::create_kernel_stream() {
    kernel_stream_t stream;
    cudaStreamCreate((cudaStream_t*) &stream);
    return stream;
}

void kernel::destroy_kernel_stream(kernel_stream_t stream) {
    cudaStreamDestroy((cudaStream_t) stream);
}

void kernel::wait_for_stream(kernel_stream_t stream) {
    cudaStreamSynchronize((cudaStream_t) stream);
}

kernel::matmul_handle_t kernel::create_matmul_handle() {
    matmul_handle_t handle;
    
    cublasCreate_v2((cublasHandle_t*) &handle);
    
    return handle;
}

void kernel::destroy_matmul_handle(kernel::matmul_handle_t stream) {
    cublasDestroy_v2((cublasHandle_t) stream);
}