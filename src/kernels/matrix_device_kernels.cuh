#pragma once

#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <kernels/scheduling.cuh>
#include <kernels/matrix/host.hpp>
#include <util/matrix.hpp>

namespace kernel::matrix {

#define MATRIX_PROJECT(row, col, stride) ((row) * stride + (col))

static __device__ void* device_get_addr(void* data,
                                     const size_t stride,
                                     DataType type,
                                     const size_t row,
                                     const size_t col) {
    size_t offset = MATRIX_PROJECT(row, col, stride);
    switch (type) {
        case DataType::Float:
            return &((float*)data)[offset];
        case DataType::Half:
            return &((half*)data)[offset];
        case DataType::BFloat16:
            return &((__nv_bfloat16*)data)[offset];
    }
    return nullptr;
}

static __device__ const void* device_get_addr(const void* data,
                                        const size_t stride,
                                        DataType type,
                                        const size_t row,
                                        const size_t col) {
    size_t offset = MATRIX_PROJECT(row, col, stride);
    switch (type) {
        case DataType::Float:
            return &((const float*)data)[offset];
        case DataType::Half:
            return &((const half*)data)[offset];
        case DataType::BFloat16:
            return &((const __nv_bfloat16*)data)[offset];
    }
    return nullptr;
}

static __device__ float* device_get_addr(matrix_view data,
                                     const size_t row,
                                     const size_t col) {
    return (float*)device_get_addr(data.data, data.stride, data.type, row, col);
}

static __device__ const float* device_get_addr(const_matrix_view data,
                                         const size_t row,
                                         const size_t col) {
    return (const float*)device_get_addr(data.data, data.stride, data.type, row, col);
}

static __device__ void device_set(void* data,
                                const size_t stride,
                                DataType type,
                                const size_t row,
                                const size_t col,
                                const float value) {
    switch (type) {
        case DataType::Float:
            *(float*)device_get_addr(data, stride, type, row, col) = value;
            break;
        case DataType::Half:
            *(half*)device_get_addr(data, stride, type, row, col) = __float2half(value);
            break;
        case DataType::BFloat16:
            *(__nv_bfloat16*)device_get_addr(data, stride, type, row, col) = __float2bfloat16(value);
            break;
    }
}

static __device__ void device_set(matrix_view data,
                                const size_t row,
                                const size_t col,
                                const float value) {
    device_set(data.data, data.stride, data.type, row, col, value);
}

static __device__ float device_get(const void* data,
                                const size_t stride,
                                DataType type,
                                const size_t row,
                                const size_t col) {
    switch (type) {
        case DataType::Float:
            return *(const float*)device_get_addr(data, stride, type, row, col);
        case DataType::Half:
            return __half2float(*(const half*)device_get_addr(data, stride, type, row, col));
        case DataType::BFloat16:
            return __bfloat162float(*(const __nv_bfloat16*)device_get_addr(data, stride, type, row, col));
    }
    return 0.0f;
}

static __device__ float device_get(matrix_view data,
                                const size_t row,
                                const size_t col) {
    return device_get(data.data, data.stride, data.type, row, col);
}

static __device__ float device_get(const const_matrix_view data,
                                const size_t row,
                                const size_t col) {
    return device_get(data.data, data.stride, data.type, row, col);
}

static __device__ void device_offset_elem(void* data,
                                      const size_t stride,
                                      DataType type,
                                      const size_t row,
                                      const size_t col,
                                      float value) {
    switch (type) {
        case DataType::Float:
            *(float*)device_get_addr(data, stride, type, row, col) += value;
            break;
        case DataType::Half: {
            half h = *(half*)device_get_addr(data, stride, type, row, col);
            *(half*)device_get_addr(data, stride, type, row, col) = __float2half(__half2float(h) + value);
            break;
        }
        case DataType::BFloat16: {
            __nv_bfloat16 bf = *(__nv_bfloat16*)device_get_addr(data, stride, type, row, col);
            *(__nv_bfloat16*)device_get_addr(data, stride, type, row, col) = __float2bfloat16(__bfloat162float(bf) + value);
            break;
        }
    }
}

static __device__ void device_offset_elem(matrix_view data,
                                      const size_t row,
                                      const size_t col,
                                      float value) {
    device_offset_elem(data.data, data.stride, data.type, row, col, value);
}

static __device__ void device_offset_elem_atomic(float* data,
                                             const size_t stride,
                                             const size_t row,
                                             const size_t col,
                                             float value) {
    float* addr = (float*)device_get_addr(data, stride, DataType::Float, row, col);
    atomicAdd(addr, value);
}

static __device__ void device_offset_elem_atomic(void* data,
                                             const size_t stride,
                                             DataType type,
                                             const size_t row,
                                             const size_t col,
                                             float value) {
    if (type == DataType::Float) {
        float* addr = (float*)device_get_addr(data, stride, type, row, col);
        atomicAdd(addr, value);
    } else if (type == DataType::Half) {
        unsigned int* addr = (unsigned int*)device_get_addr(data, stride, type, row, col);
        atomicAdd(addr, __half_as_ushort(__float2half(value)));
    } else if (type == DataType::BFloat16) {
        unsigned int* addr = (unsigned int*)device_get_addr(data, stride, type, row, col);
        atomicAdd(addr, __bfloat16_as_ushort(__float2bfloat16(value)));
    }
}

static __device__ void device_offset_elem_atomic(matrix_view data,
                                             const size_t row,
                                             const size_t col,
                                             float value) {
    device_offset_elem_atomic(data.data, data.stride, data.type, row, col, value);
}

namespace device {

static __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset, 32);

    return val;
}

static __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset, 32));

    return val;
}

static __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32.0f) ? shared[lane] : 0;

    if (wid == 0)
        val = warp_reduce_sum(val);

    __syncthreads();

    return val;
}

static __device__ float block_reduce_max(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32.0f) ? shared[lane] : -CUDART_INF_F;

    if (wid == 0)
        val = warp_reduce_max(val);

    __syncthreads();

    return val;
}

}  // namespace device

template <auto Mapping>
__global__ void map_matrix_kernel(const const_matrix_view input,
                                  const matrix_view output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < input.rows && col < input.cols) {
        float val = kernel::matrix::device_get(input, row, col);
        kernel::matrix::device_set(output, row, col, Mapping(val));
    }
}

template <auto Mapping>
::matrix map_matrix(const ::matrix& input, kernel_stream_t stream = nullptr) {
    ::matrix output = kernel::matrix::async_allocate(input.rows, input.cols, input.type, stream);

    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);

    map_matrix_kernel<Mapping>
        <<<gridSize, blockSize, 0, get_kernel_stream(stream)>>>(input, output);
    return output;
}

template <auto Mapping>
void map_matrix_inplace(::matrix& input, kernel_stream_t stream = nullptr) {
    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);

    map_matrix_kernel<Mapping>
        <<<gridSize, blockSize, 0, get_kernel_stream(stream)>>>(input, input);
}

}  // namespace kernel::matrix
