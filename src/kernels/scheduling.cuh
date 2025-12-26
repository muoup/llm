#pragma once

#include <kernels/scheduling.hpp>

#include <cuda_runtime_api.h>
#include <cublas_api.h>

inline cudaStream_t get_kernel_stream(kernel::kernel_stream_t stream) {
    return (cudaStream_t) stream;
}

inline cublasHandle_t get_matmul_handle(kernel::matmul_handle_t handle) {
    return (cublasHandle_t) handle;
}

inline float get_device_ptr(kernel::float_device_ptr_t ptr) {
    float f;
    cudaMemcpy(&f, ptr, sizeof(float), cudaMemcpyDeviceToHost);
    return f;
}