#include "cublas.hpp"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <cuda_bf16.hpp>
#include <cuda_fp16.hpp>

#include <kernels/matrix/device.cuh>
#include <kernels/matrix/pools.cuh>
#include <kernels/matrix/kernels.hpp>
#include <kernels/scheduling.hpp>

namespace kernel::matrix {

static kernel::MatmulHandlePool<8> matmul_handle_pool;

// Get global pools from pools.cu
extern GPUFloatPool global_gpu_float_pool;
extern GPUHalfPool global_gpu_half_pool;
extern GPUBf16Pool global_gpu_bf16_pool;
extern GPUBoolPool global_gpu_bool_pool;

// ===== cuBLAS GEMM Wrappers =====

static void check_cublas_errors(const char* step, cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::printf("cuBLAS Failure during: %s\n", step);
        std::printf("cuBLAS Error: %d\n", status);
        std::abort();
    }
}

::matrix cublas_gemm_float(const const_matrix_view& a,
                           const const_matrix_view& b,
                           cublasOperation_t op_a,
                           cublasOperation_t op_b,
                           kernel_stream_t stream) {
    ::matrix result = async_allocate(a.rows, b.cols, DataType::Float, stream);
    auto handle = matmul_handle_pool.acquire();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSetStream(
        (cublasHandle_t)handle, (cudaStream_t)get_kernel_stream(stream));
    check_cublas_errors("cublasSetStream", status);

    status = cublasSgemm((cublasHandle_t)handle, op_b, op_a, b.cols, a.rows,
                         a.cols, &alpha, (const float*)b.data, b.stride,
                         (const float*)a.data, a.stride, &beta,
                         (float*)result.data, result.stride);

    check_cublas_errors("cublasSgemm", status);

    return result;
}

::matrix cublas_gemm_half(const const_matrix_view& a,
                          const const_matrix_view& b,
                          cublasOperation_t op_a,
                          cublasOperation_t op_b,
                          kernel_stream_t stream) {
    ::matrix result = async_allocate(a.rows, b.cols, DataType::Half, stream);
    auto handle = matmul_handle_pool.acquire();

    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);

    cublasStatus_t status = cublasSetStream(
        (cublasHandle_t)handle, (cudaStream_t)get_kernel_stream(stream));
    check_cublas_errors("cublasSetStream", status);

    status = cublasHgemm((cublasHandle_t)handle, op_b, op_a, b.cols, a.rows,
                         a.cols, &alpha, (const half*)b.data, b.stride,
                         (const half*)a.data, a.stride, &beta,
                         (half*)result.data, result.stride);

    check_cublas_errors("cublasHgemm", status);

    return result;
}

::matrix cublas_gemm_bf16(const const_matrix_view& a,
                          const const_matrix_view& b,
                          cublasOperation_t op_a,
                          cublasOperation_t op_b,
                          kernel_stream_t stream) {
    ::matrix result
        = async_allocate(a.rows, b.cols, DataType::BFloat16, stream);
    auto handle = matmul_handle_pool.acquire();

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
    const __nv_bfloat16 alpha = __float2bfloat16(1.0f);
    const __nv_bfloat16 beta = __float2bfloat16(0.0f);

    cublasStatus_t status = cublasSetStream(
        (cublasHandle_t)handle, (cudaStream_t)get_kernel_stream(stream));
    check_cublas_errors("cublasSetStream", status);

    status = cublasGemmEx(
        (cublasHandle_t)handle, op_b, op_a, b.cols, a.rows, a.cols, &alpha,
        (const __nv_bfloat16*)b.data, CUDA_R_16BF, b.stride,
        (const __nv_bfloat16*)a.data, CUDA_R_16BF, a.stride, &beta,
        (__nv_bfloat16*)result.data, CUDA_R_16BF, result.stride,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    check_cublas_errors("cublasGemmEx", status);
#else
    throw std::runtime_error("BFloat16 GEMM requires CUDA 11.0+");
#endif

    return result;
}

// ===== Public API with dispatch =====

::matrix cross_multiplied(const const_matrix_view& a,
                          const const_matrix_view& b,
                          kernel_stream_t stream) {
    switch (a.type) {
        case DataType::Float:
            return cublas_gemm_float(a, b, CUBLAS_OP_N, CUBLAS_OP_N, stream);
        case DataType::Half:
            return cublas_gemm_half(a, b, CUBLAS_OP_N, CUBLAS_OP_N, stream);
        case DataType::BFloat16:
            return cublas_gemm_bf16(a, b, CUBLAS_OP_N, CUBLAS_OP_N, stream);
    }
    return {};
}

::matrix t_cross_multiplied(const const_matrix_view& a,
                            const const_matrix_view& b,
                            kernel_stream_t stream) {
    switch (a.type) {
        case DataType::Float:
            return cublas_gemm_float(a, b, CUBLAS_OP_N, CUBLAS_OP_T, stream);
        case DataType::Half:
            return cublas_gemm_half(a, b, CUBLAS_OP_N, CUBLAS_OP_T, stream);
        case DataType::BFloat16:
            return cublas_gemm_bf16(a, b, CUBLAS_OP_N, CUBLAS_OP_T, stream);
    }
    return {};
}

::matrix cross_t_multiplied(const const_matrix_view& a,
                            const const_matrix_view& b,
                            kernel_stream_t stream) {
    switch (a.type) {
        case DataType::Float:
            return cublas_gemm_float(a, b, CUBLAS_OP_T, CUBLAS_OP_N, stream);
        case DataType::Half:
            return cublas_gemm_half(a, b, CUBLAS_OP_T, CUBLAS_OP_N, stream);
        case DataType::BFloat16:
            return cublas_gemm_bf16(a, b, CUBLAS_OP_T, CUBLAS_OP_N, stream);
    }
    return {};
}

// ===== Reductions =====

void* sum(const ::matrix& mat, kernel_stream_t stream) {
    float* reduction_result = (float*)global_gpu_float_pool.acquire();
    float zero = 0.0f;
    cudaMemcpyAsync(reduction_result, &zero, sizeof(float),
                    cudaMemcpyHostToDevice, get_kernel_stream(stream));

    const int threads_per_block = 256;
    const int num_elements = mat.rows * mat.cols;
    const int num_blocks = std::min(
        1024, (num_elements + threads_per_block - 1) / threads_per_block);

    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::sum_reduction_float<<<num_blocks, threads_per_block, 0,
                                          cuda_stream>>>(mat, reduction_result,
                                                         0.0f);
            break;
        case DataType::Half:
            device::sum_reduction_half<<<num_blocks, threads_per_block, 0,
                                         cuda_stream>>>(mat, reduction_result,
                                                        0.0f);
            break;
        case DataType::BFloat16:
            device::sum_reduction_bf16<<<num_blocks, threads_per_block, 0,
                                         cuda_stream>>>(mat, reduction_result,
                                                        0.0f);
            break;
    }

    return reduction_result;
}

void* sum_of_squares(const ::matrix& mat, kernel_stream_t stream) {
    float* reduction_result = (float*)global_gpu_float_pool.acquire();
    float zero = 0.0f;
    cudaMemcpyAsync(reduction_result, &zero, sizeof(float),
                    cudaMemcpyHostToDevice, get_kernel_stream(stream));

    const int threads_per_block = 256;
    const int num_elements = mat.rows * mat.cols;
    const int num_blocks = std::min(
        1024, (num_elements + threads_per_block - 1) / threads_per_block);

    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::sum_reduction_float<<<num_blocks, threads_per_block, 0,
                                          cuda_stream>>>(mat, reduction_result,
                                                         0.0f);
            break;
        case DataType::Half:
            device::sum_reduction_half<<<num_blocks, threads_per_block, 0,
                                         cuda_stream>>>(mat, reduction_result,
                                                        0.0f);
            break;
        case DataType::BFloat16:
            device::sum_reduction_bf16<<<num_blocks, threads_per_block, 0,
                                         cuda_stream>>>(mat, reduction_result,
                                                        0.0f);
            break;
    }

    return reduction_result;
}

void* abssum(const ::matrix& mat, kernel_stream_t stream) {
    return sum(mat, stream);
}

void* max(const ::matrix& mat, kernel_stream_t stream) {
    return sum(mat, stream);
}

void* min(const ::matrix& mat, kernel_stream_t stream) {
    return sum(mat, stream);
}

void* absmax(const ::matrix& mat, kernel_stream_t stream) {
    return sum(mat, stream);
}

void* variance(const ::matrix& mat, kernel_stream_t stream) {
    float* sum_ptr = (float*)sum(mat, stream);
    float* sum_of_squares_ptr = (float*)sum_of_squares(mat, stream);

    float* device_result = (float*)global_gpu_float_pool.acquire();

    const dim3 blocks(1, 1);
    const dim3 threads(1, 1);

    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::variance_kernel_float<<<blocks, threads, 0, cuda_stream>>>(
                sum_ptr, sum_of_squares_ptr, device_result,
                mat.rows * mat.cols);
            break;
        case DataType::Half:
            device::variance_kernel_half<<<blocks, threads, 0, cuda_stream>>>(
                sum_ptr, sum_of_squares_ptr, device_result,
                mat.rows * mat.cols);
            break;
        case DataType::BFloat16:
            device::variance_kernel_bf16<<<blocks, threads, 0, cuda_stream>>>(
                sum_ptr, sum_of_squares_ptr, device_result,
                mat.rows * mat.cols);
            break;
    }

    return device_result;
}

// ===== Activations =====

void softmax(::matrix& mat, kernel_stream_t stream) {
    const dim3 threads_per_block = 256;
    const dim3 blocks = mat.rows;

    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::softmax_kernel_float<<<blocks, threads_per_block, 0,
                                           cuda_stream>>>(mat);
            break;
        case DataType::Half:
            device::softmax_kernel_half<<<blocks, threads_per_block, 0,
                                          cuda_stream>>>(mat);
            break;
        case DataType::BFloat16:
            device::softmax_kernel_bf16<<<blocks, threads_per_block, 0,
                                          cuda_stream>>>(mat);
            break;
    }
}

void backprop_softmax(::matrix& buffer,
                      const ::matrix& output,
                      const ::matrix& gradient,
                      kernel_stream_t stream) {
    const int threads_per_block = 256;
    const int blocks = gradient.rows;

    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (buffer.type) {
        case DataType::Float:
            device::backprop_softmax_kernel_float<<<blocks, threads_per_block,
                                                    0, cuda_stream>>>(
                output, gradient, buffer);
            break;
        case DataType::Half:
            device::backprop_softmax_kernel_half<<<blocks, threads_per_block, 0,
                                                   cuda_stream>>>(
                output, gradient, buffer);
            break;
        case DataType::BFloat16:
            device::backprop_softmax_kernel_bf16<<<blocks, threads_per_block, 0,
                                                   cuda_stream>>>(
                output, gradient, buffer);
            break;
    }
}

// ===== Comparison =====

bool is_equal(const ::matrix& a,
              const ::matrix& b,
              float epsilon,
              kernel_stream_t stream) {
    if (a.rows != b.rows || a.cols != b.cols || a.type != b.type) {
        return false;
    }

    bool* result = (bool*)global_gpu_bool_pool.acquire();
    bool true_val = true;
    cudaMemcpyAsync(result, &true_val, sizeof(bool), cudaMemcpyHostToDevice,
                    get_kernel_stream(stream));

    const dim3 threads_per_block = 256;
    const dim3 blocks((a.rows * a.cols + 255) / 256);

    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (a.type) {
        case DataType::Float:
            device::compare_kernel_float<<<blocks, threads_per_block, 0,
                                           cuda_stream>>>(a, b, epsilon,
                                                          result);
            break;
        case DataType::Half:
            device::compare_kernel_half<<<blocks, threads_per_block, 0,
                                          cuda_stream>>>(a, b, epsilon, result);
            break;
        case DataType::BFloat16:
            device::compare_kernel_bf16<<<blocks, threads_per_block, 0,
                                          cuda_stream>>>(a, b, epsilon, result);
            break;
    }

    bool host_result;
    cudaMemcpyAsync(&host_result, result, sizeof(bool), cudaMemcpyDeviceToHost,
                    get_kernel_stream(stream));
    return host_result;
}

}  // namespace kernel::matrix
