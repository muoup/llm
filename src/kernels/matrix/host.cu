#include "host.hpp"

#include <curand.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <kernels/scheduling.cuh>
#include <kernels/matrix/device.cuh>
#include <kernels/pools.hpp>
#include <util/matrix.hpp>

namespace kernel::matrix {

// ===== Memory Management =====

::matrix async_allocate(size_t rows,
                        size_t cols,
                        DataType type,
                        kernel_stream_t stream) {
    ::matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.type = type;
    mat.stride = calculate_stride(rows, cols, type);

    size_t total_size = mat.stride * mat.rows;

    switch (type) {
        case DataType::Float:
            cudaMallocAsync(&mat.data, total_size * sizeof(float),
                            (cudaStream_t)get_kernel_stream(stream));
            break;
        case DataType::Half:
            cudaMallocAsync(&mat.data, total_size * sizeof(half),
                            (cudaStream_t)get_kernel_stream(stream));
            break;
        case DataType::BFloat16:
            cudaMallocAsync(&mat.data, total_size * sizeof(__nv_bfloat16),
                            (cudaStream_t)get_kernel_stream(stream));
            break;
    }

    return mat;
}

void* allocate_buffer(size_t size, DataType type, kernel_stream_t stream) {
    void* ptr = nullptr;
    
    switch (type) {
        case DataType::Float:
            cudaMalloc(&ptr, size * sizeof(float));
            break;
        case DataType::Half:
            cudaMalloc(&ptr, size * sizeof(half));
            break;
        case DataType::BFloat16:
            cudaMalloc(&ptr, size * sizeof(__nv_bfloat16));
            break;
    }
    
    return ptr;
}

void free_buffer(void* buffer, kernel_stream_t stream) {
    cudaFreeAsync(buffer, get_kernel_stream(stream));
}

// ===== Data Transfer =====

void load_into(::matrix& mat, const void* host_data, kernel_stream_t stream) {
    size_t total_size = mat.rows * mat.cols;
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            cudaMemcpyAsync(mat.data, host_data, total_size * sizeof(float),
                            cudaMemcpyHostToDevice, cuda_stream);
            break;
        case DataType::Half:
            cudaMemcpyAsync(mat.data, host_data, total_size * sizeof(half),
                            cudaMemcpyHostToDevice, cuda_stream);
            break;
        case DataType::BFloat16:
            cudaMemcpyAsync(mat.data, host_data,
                            total_size * sizeof(__nv_bfloat16),
                            cudaMemcpyHostToDevice, cuda_stream);
            break;
    }
}

void store_from(const ::matrix& mat, void* host_data, kernel_stream_t stream) {
    size_t total_size = mat.rows * mat.cols;
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            cudaMemcpyAsync(host_data, mat.data, total_size * sizeof(float),
                            cudaMemcpyDeviceToHost, cuda_stream);
            break;
        case DataType::Half:
            cudaMemcpyAsync(host_data, mat.data, total_size * sizeof(half),
                            cudaMemcpyDeviceToHost, cuda_stream);
            break;
        case DataType::BFloat16:
            cudaMemcpyAsync(host_data, mat.data,
                            total_size * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToHost, cuda_stream);
            break;
    }
}

// ===== Element Access =====

float get(const ::matrix& mat, size_t row, size_t col, kernel_stream_t stream) {
    switch (mat.type) {
        case DataType::Float:
            return ((const float*)mat.data)[row * mat.stride + col];
        case DataType::Half:
            return __half2float(
                ((const half*)mat.data)[row * mat.stride + col]);
        case DataType::BFloat16:
            return __bfloat162float(
                ((const __nv_bfloat16*)mat.data)[row * mat.stride + col]);
    }
    return 0.0f;
}

void set(::matrix& mat,
         size_t row,
         size_t col,
         float value,
         kernel_stream_t stream) {
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            cudaMemcpyAsync(&((float*)mat.data)[row * mat.stride + col], &value,
                            sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
            break;
        case DataType::Half: {
            half h = __float2half(value);
            cudaMemcpyAsync(&((half*)mat.data)[row * mat.stride + col], &h,
                            sizeof(half), cudaMemcpyHostToDevice, cuda_stream);
        } break;
        case DataType::BFloat16: {
            __nv_bfloat16 bf = __float2bfloat16(value);
            cudaMemcpyAsync(&((__nv_bfloat16*)mat.data)[row * mat.stride + col],
                            &bf, sizeof(__nv_bfloat16), cudaMemcpyHostToDevice,
                            cuda_stream);
        } break;
    }
}

void* get_addr(::matrix& mat, size_t row, size_t col) {
    switch (mat.type) {
        case DataType::Float:
            return &((float*)mat.data)[row * mat.stride + col];
        case DataType::Half:
            return &((half*)mat.data)[row * mat.stride + col];
        case DataType::BFloat16:
            return &((__nv_bfloat16*)mat.data)[row * mat.stride + col];
    }
    return nullptr;
}

const void* get_addr(const ::matrix& mat, size_t row, size_t col) {
    switch (mat.type) {
        case DataType::Float:
            return &((const float*)mat.data)[row * mat.stride + col];
        case DataType::Half:
            return &((const half*)mat.data)[row * mat.stride + col];
        case DataType::BFloat16:
            return &((const __nv_bfloat16*)mat.data)[row * mat.stride + col];
    }
    return nullptr;
}

// ===== Randomization =====

void randomize(::matrix& mat, float min, float max, kernel_stream_t stream) {
    curandGenerator_t generator = nullptr;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetStream(generator, (cudaStream_t)get_kernel_stream(stream));

    size_t total_size = mat.rows * mat.cols;

    switch (mat.type) {
        case DataType::Float:
            curandGenerateUniform(generator, (float*)mat.data, total_size);
            break;
        case DataType::Half:
        case DataType::BFloat16: {
            ::matrix temp
                = async_allocate(mat.rows, mat.cols, DataType::Float, stream);
            curandGenerateUniform(generator, (float*)temp.data, total_size);

            dim3 threads(256);
            dim3 blocks((total_size + 255) / 256);

            if (mat.type == DataType::Half) {
                device::convert_float_to_half<<<blocks, threads, 0,
                                        (cudaStream_t)get_kernel_stream(
                                            stream)>>>(
                    (float*)temp.data, (uint16_t*)mat.data, total_size);
            } else {
                device::convert_float_to_bf16<<<blocks, threads, 0,
                                        (cudaStream_t)get_kernel_stream(
                                            stream)>>>(
                    (float*)temp.data, (uint16_t*)mat.data, total_size);
            }
            cudaFreeAsync(temp.data,
                            (cudaStream_t)get_kernel_stream(stream));
        } break;
    }

    curandDestroyGenerator(generator);
}

void check_errors(const char* step) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("CUDA Error during: %s\n", step);
        std::printf("CUDA Error: %s\n", cudaGetErrorString(err));
        std::abort();
    }
}

// ===== Matrix Operations =====

::matrix clone(const ::const_matrix_view mat, kernel_stream_t stream) {
    ::matrix result = async_allocate(mat.rows, mat.cols, mat.type, stream);

    dim3 threads(16, 16);
    dim3 blocks((mat.cols + 15) / 16, (mat.rows + 15) / 16);

    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::
                copy_matrix_kernel_float<<<blocks, threads, 0, cuda_stream>>>(
                    result, mat);
            break;
        case DataType::Half:
            device::
                copy_matrix_kernel_half<<<blocks, threads, 0, cuda_stream>>>(
                    result, mat);
            break;
        case DataType::BFloat16:
            device::
                copy_matrix_kernel_bf16<<<blocks, threads, 0, cuda_stream>>>(
                    result, mat);
            break;
    }

    return result;
}

void set_all(::matrix& mat, float value, kernel_stream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((mat.cols + 15) / 16, (mat.rows + 15) / 16);

    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::set_all_kernel_float<<<blocks, threads, 0, cuda_stream>>>(
                mat, value);
            break;
        case DataType::Half:
            device::set_all_kernel_half<<<blocks, threads, 0, cuda_stream>>>(
                mat, value);
            break;
        case DataType::BFloat16:
            device::set_all_kernel_bf16<<<blocks, threads, 0, cuda_stream>>>(
                mat, value);
            break;
    }
}

void scale(::matrix& mat, float factor, kernel_stream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((mat.cols + 15) / 16, (mat.rows + 15) / 16);

    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::scale_kernel_float<<<blocks, threads, 0, cuda_stream>>>(
                mat, factor);
            break;
        case DataType::Half:
            device::scale_kernel_half<<<blocks, threads, 0, cuda_stream>>>(
                mat, factor);
            break;
        case DataType::BFloat16:
            device::scale_kernel_bf16<<<blocks, threads, 0, cuda_stream>>>(
                mat, factor);
            break;
    }
}

// ===== Row Operations =====

void transfer_row(::matrix& dest,
                  size_t dest_row,
                  const ::matrix& src,
                  size_t src_row,
                  kernel_stream_t stream) {
    dim3 threads(256);
    dim3 blocks((dest.cols + 255) / 256);
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (dest.type) {
        case DataType::Float:
            device::
                transfer_row_kernel_float<<<blocks, threads, 0, cuda_stream>>>(
                    dest, dest_row, src, src_row);
            break;
        case DataType::Half:
            device::
                transfer_row_kernel_half<<<blocks, threads, 0, cuda_stream>>>(
                    dest, dest_row, src, src_row);
            break;
        case DataType::BFloat16:
            device::
                transfer_row_kernel_bf16<<<blocks, threads, 0, cuda_stream>>>(
                    dest, dest_row, src, src_row);
            break;
    }
}

void set_row_vector(::matrix& mat,
                    size_t mat_row,
                    const ::matrix& vec,
                    size_t vec_row,
                    kernel_stream_t stream) {
    dim3 threads(256);
    dim3 blocks((mat.cols + 255) / 256);
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::
                transfer_row_kernel_float<<<blocks, threads, 0, cuda_stream>>>(
                    mat, mat_row, vec, vec_row);
            break;
        case DataType::Half:
            device::
                transfer_row_kernel_half<<<blocks, threads, 0, cuda_stream>>>(
                    mat, mat_row, vec, vec_row);
            break;
        case DataType::BFloat16:
            device::
                transfer_row_kernel_bf16<<<blocks, threads, 0, cuda_stream>>>(
                    mat, mat_row, vec, vec_row);
            break;
    }
}

::matrix get_row_vector(const ::matrix& mat,
                        size_t row,
                        kernel_stream_t stream) {
    ::matrix result = async_allocate(1, mat.cols, mat.type, stream);
    dim3 threads(256);
    dim3 blocks((mat.cols + 255) / 256);
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::
                transfer_row_kernel_float<<<blocks, threads, 0, cuda_stream>>>(
                    result, 0, mat, row);
            break;
        case DataType::Half:
            device::
                transfer_row_kernel_half<<<blocks, threads, 0, cuda_stream>>>(
                    result, 0, mat, row);
            break;
        case DataType::BFloat16:
            device::
                transfer_row_kernel_bf16<<<blocks, threads, 0, cuda_stream>>>(
                    result, 0, mat, row);
            break;
    }

    return result;
}

void add_row_vector(::matrix& mat,
                    size_t row,
                    const ::matrix& vec,
                    size_t vec_row,
                    float scale,
                    kernel_stream_t stream) {
    dim3 threads(256);
    dim3 blocks((mat.cols + 255) / 256);
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::add_row_vector_kernel_float<<<blocks, threads, 0,
                                                  cuda_stream>>>(
                mat, row, vec, vec_row, scale);
            break;
        case DataType::Half:
            device::
                add_row_vector_kernel_half<<<blocks, threads, 0, cuda_stream>>>(
                    mat, row, vec, vec_row, scale);
            break;
        case DataType::BFloat16:
            device::
                add_row_vector_kernel_bf16<<<blocks, threads, 0, cuda_stream>>>(
                    mat, row, vec, vec_row, scale);
            break;
    }
}

void atomic_add_row_vector(::matrix& mat,
                           size_t row,
                           const ::matrix& vec,
                           size_t vec_row,
                           float scale,
                           kernel_stream_t stream) {
    dim3 threads(256);
    dim3 blocks((mat.cols + 255) / 256);
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::atomic_add_row_vector_kernel_float<<<blocks, threads, 0,
                                                         cuda_stream>>>(
                mat, row, vec, vec_row, scale);
            break;
        case DataType::Half:
            device::atomic_add_row_vector_kernel_half<<<blocks, threads, 0,
                                                        cuda_stream>>>(
                mat, row, vec, vec_row, scale);
            break;
        case DataType::BFloat16:
            device::atomic_add_row_vector_kernel_bf16<<<blocks, threads, 0,
                                                        cuda_stream>>>(
                mat, row, vec, vec_row, scale);
            break;
    }
}

// ===== Matrix Addition =====

void add(::matrix& mat, const ::matrix& other, kernel_stream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((mat.cols + 15) / 16, (mat.rows + 15) / 16);
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::
                add_matrix_kernel_float<<<blocks, threads, 0, cuda_stream>>>(
                    mat, other);
            break;
        case DataType::Half:
            device::add_matrix_kernel_half<<<blocks, threads, 0, cuda_stream>>>(
                mat, other);
            break;
        case DataType::BFloat16:
            device::add_matrix_kernel_bf16<<<blocks, threads, 0, cuda_stream>>>(
                mat, other);
            break;
    }
}

void atomic_add(::matrix& mat, const ::matrix& other, kernel_stream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((mat.cols + 15) / 16, (mat.rows + 15) / 16);
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::add_matrix_atomic_kernel_float<<<blocks, threads, 0,
                                                     cuda_stream>>>(mat, other);
            break;
        case DataType::Half:
            device::add_matrix_atomic_kernel_half<<<blocks, threads, 0,
                                                    cuda_stream>>>(mat, other);
            break;
        case DataType::BFloat16:
            device::add_matrix_atomic_kernel_bf16<<<blocks, threads, 0,
                                                    cuda_stream>>>(mat, other);
            break;
    }
}

void add_scaled(::matrix& mat,
                const ::matrix& other,
                float factor,
                kernel_stream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((mat.cols + 15) / 16, (mat.rows + 15) / 16);
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::
                add_scaled_kernel_float<<<blocks, threads, 0, cuda_stream>>>(
                    mat, other, factor);
            break;
        case DataType::Half:
            device::add_scaled_kernel_half<<<blocks, threads, 0, cuda_stream>>>(
                mat, other, factor);
            break;
        case DataType::BFloat16:
            device::add_scaled_kernel_bf16<<<blocks, threads, 0, cuda_stream>>>(
                mat, other, factor);
            break;
    }
}

void add(::matrix& mat, float value, kernel_stream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((mat.cols + 15) / 16, (mat.rows + 15) / 16);
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::add_value_kernel_float<<<blocks, threads, 0, cuda_stream>>>(
                mat, value);
            break;
        case DataType::Half:
            device::add_value_kernel_half<<<blocks, threads, 0, cuda_stream>>>(
                mat, value);
            break;
        case DataType::BFloat16:
            device::add_value_kernel_bf16<<<blocks, threads, 0, cuda_stream>>>(
                mat, value);
            break;
    }
}

// ===== Slice Operations =====

void set_horizontal_slice(::matrix& mat,
                          size_t start_col,
                          const ::matrix& slice,
                          kernel_stream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((slice.cols + 15) / 16, (slice.rows + 15) / 16);
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::set_horizontal_slice_kernel_float<<<blocks, threads, 0,
                                                        cuda_stream>>>(
                mat, start_col, slice);
            break;
        case DataType::Half:
            device::set_horizontal_slice_kernel_half<<<blocks, threads, 0,
                                                       cuda_stream>>>(
                mat, start_col, slice);
            break;
        case DataType::BFloat16:
            device::set_horizontal_slice_kernel_bf16<<<blocks, threads, 0,
                                                       cuda_stream>>>(
                mat, start_col, slice);
            break;
    }
}

// ===== Masking =====

void mask_upper_triangle(::matrix& mat,
                         float mask_value,
                         kernel_stream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((mat.cols + 15) / 16, (mat.rows + 15) / 16);
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (mat.type) {
        case DataType::Float:
            device::mask_upper_triangle_kernel_float<<<blocks, threads, 0,
                                                       cuda_stream>>>(
                mat, mask_value);
            break;
        case DataType::Half:
            device::mask_upper_triangle_kernel_half<<<blocks, threads, 0,
                                                      cuda_stream>>>(
                mat, mask_value);
            break;
        case DataType::BFloat16:
            device::mask_upper_triangle_kernel_bf16<<<blocks, threads, 0,
                                                      cuda_stream>>>(
                mat, mask_value);
            break;
    }
}

// ===== Element-wise Operations =====

void element_wise_multiply(::matrix& a,
                           const ::matrix& b,
                           kernel_stream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((a.cols + 15) / 16, (a.rows + 15) / 16);
    cudaStream_t cuda_stream = (cudaStream_t)get_kernel_stream(stream);

    switch (a.type) {
        case DataType::Float:
            device::element_wise_multiply_kernel_float<<<blocks, threads, 0,
                                                         cuda_stream>>>(a, b);
            break;
        case DataType::Half:
            device::element_wise_multiply_kernel_half<<<blocks, threads, 0,
                                                        cuda_stream>>>(a, b);
            break;
        case DataType::BFloat16:
            device::element_wise_multiply_kernel_bf16<<<blocks, threads, 0,
                                                        cuda_stream>>>(a, b);
            break;
    }
}

}  // namespace kernel::matrix
