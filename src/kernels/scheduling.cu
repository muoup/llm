#include "scheduling.hpp"

#include <cublas_api.h>
#include <cuda_runtime.h>

float kernel::get_device_ptr(kernel::float_device_ptr_t ptr) {
    // float f;
    // cudaMemcpy(&f, ptr, sizeof(float), cudaMemcpyDeviceToHost);
    return *(float*) ptr;
}

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

void kernel::wait_for_all_streams() {
    cudaDeviceSynchronize();
}

kernel::matmul_handle_t kernel::create_matmul_handle() {
    matmul_handle_t handle;
    
    cublasCreate_v2((cublasHandle_t*) &handle);
    
    return handle;
}

void kernel::destroy_matmul_handle(kernel::matmul_handle_t stream) {
    cublasDestroy_v2((cublasHandle_t) stream);
}