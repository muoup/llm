#include "scheduling.hpp"

#include <cublas_api.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <util/matrix.hpp>

float kernel::dereference_device_ptr(DataType type,
                                     kernel::float_device_ptr_t ptr) {
    switch (type) {
        case DataType::BFloat16:
            return __bfloat162float(*std::bit_cast<__nv_bfloat16*>(ptr));
        case DataType::Half:
            return __half2float(*std::bit_cast<__half*>(ptr));
        case DataType::Float:
            return *std::bit_cast<float*>(ptr);
    }

    std::abort();
}

void kernel::wait_for_stream(kernel_stream_t stream) {
    cudaStreamSynchronize((cudaStream_t)stream);
}

void kernel::wait_for_all_streams() {
    cudaDeviceSynchronize();
}

kernel::kernel_stream_t kernel::create_kernel_stream() {
    kernel::kernel_stream_t stream;
    cudaStreamCreate((cudaStream_t*)&stream);
    return stream;
}

void kernel::destroy_kernel_stream(kernel::kernel_stream_t stream) {
    cudaStreamDestroy((cudaStream_t)stream);
}

kernel::matmul_handle_t kernel::create_matmul_handle() {
    kernel::matmul_handle_t handle;

    cublasCreate_v2((cublasHandle_t*)&handle);

    return handle;
}

void kernel::destroy_matmul_handle(kernel::matmul_handle_t stream) {
    cublasDestroy_v2((cublasHandle_t)stream);
}
