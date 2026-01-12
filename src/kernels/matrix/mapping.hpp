#pragma once

#include <cuda_runtime_api.h>
#include <kernels/scheduling.cuh>
#include <util/matrix.hpp>

#ifdef __CUDACC__
#include <cuda_bf16.hpp>
#include <cuda_fp16.hpp>
#endif

namespace kernel::matrix {

struct DataMapper {
    __device__ float (*map_float)(float);
    __device__ uint16_t (*map_half)(uint16_t);
    __device__ uint16_t (*map_bf16)(uint16_t);

    __device__ static float identity_float(float x) { return x; }
    __device__ static uint16_t identity_half(uint16_t x) { return x; }
    __device__ static uint16_t identity_bf16(uint16_t x) { return x; }

    static constexpr DataMapper identity() {
        return { identity_float, identity_half, identity_bf16 };
    }

    __device__ void dispatch_inplace(void* data,
                                     size_t stride,
                                     DataType type,
                                     size_t row,
                                     size_t col) const {
        switch (type) {
            case DataType::Float:
                ((float*)data)[row * stride + col]
                    = map_float(((float*)data)[row * stride + col]);
                break;
            case DataType::Half:
                ((half*)data)[row * stride + col] = __ushort_as_half(map_half(
                    __half_as_ushort(((half*)data)[row * stride + col])));
                break;
            case DataType::BFloat16:
                ((__nv_bfloat16*)data)[row * stride + col]
                    = __ushort_as_bfloat16(map_bf16(__bfloat16_as_ushort(
                        ((__nv_bfloat16*)data)[row * stride + col])));
                break;
        }
    }

    __device__ void dispatch_copy(const void* src_data,
                                  void* dst_data,
                                  size_t src_stride,
                                  size_t dst_stride,
                                  DataType type,
                                  size_t row,
                                  size_t col) const {
        switch (type) {
            case DataType::Float:
                ((float*)dst_data)[row * dst_stride + col] = map_float(
                    ((const float*)src_data)[row * src_stride + col]);
                break;
            case DataType::Half:
                ((half*)dst_data)[row * dst_stride + col]
                    = __ushort_as_half(map_half(__half_as_ushort(
                        ((const half*)src_data)[row * src_stride + col])));
                break;
            case DataType::BFloat16:
                ((__nv_bfloat16*)dst_data)[row * dst_stride + col]
                    = __ushort_as_bfloat16(map_bf16(__bfloat16_as_ushort(
                        ((const __nv_bfloat16*)
                             src_data)[row * src_stride + col])));
                break;
        }
    }
};

#ifdef __CUDACC__

template <DataMapper Mapper>
__global__ void map_inplace_kernel(matrix_view data) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < data.rows && col < data.cols) {
        Mapper.dispatch_inplace(data.data, data.stride, data.type, row, col);
    }
}

template <DataMapper Mapper>
__global__ void map_copy_kernel(const const_matrix_view input,
                                matrix_view output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < input.rows && col < input.cols) {
        Mapper.dispatch_copy(input.data, output.data, input.stride,
                             output.stride, input.type, row, col);
    }
}

template <DataMapper Mapper>
::matrix map(const ::matrix& input, kernel_stream_t stream);

template <DataMapper Mapper>
void map_inplace(::matrix& input, kernel_stream_t stream);

#endif  // __CUDACC__

}  // namespace kernel::matrix
