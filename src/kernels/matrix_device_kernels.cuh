#pragma once

#include <cuda_runtime_api.h>
#include <util/matrix.hpp>

namespace kernel::matrix {

inline __device__ float* device_get_addr(float* data,
                                         const size_t stride,
                                         const size_t row,
                                         const size_t col) {
    return &(data[row + col * stride]);
}

inline __device__ float* device_get_addr(matrix_view data,
                                         const size_t row,
                                         const size_t col) {
    return device_get_addr(data.data, data.stride, row, col);
}

inline __device__ const float* device_get_addr(const float* data,
                                               const size_t stride,
                                               const size_t row,
                                               const size_t col) {
    return &(data[row + col * stride]);
}

inline __device__ const float* device_get_addr(const const_matrix_view data,
                                               const size_t row,
                                               const size_t col) {
    return device_get_addr(data.data, data.stride, row, col);
}

inline __device__ void device_set(float* data,
                                  const size_t stride,
                                  const size_t row,
                                  const size_t col,
                                  const float value) {
    *(device_get_addr(data, stride, row, col)) = value;
}

inline __device__ void device_set(matrix_view data,
                                  const size_t row,
                                  const size_t col,
                                  const float value) {
    device_set(data.data, data.stride, row, col, value);
}

inline __device__ float device_get(const float* data,
                                   const size_t stride,
                                   const size_t row,
                                   const size_t col) {
    return *(device_get_addr(data, stride, row, col));
}

inline __device__ float device_get(const matrix_view data,
                                   const size_t row,
                                   const size_t col) {
    return device_get(data.data, data.stride, row, col);
}

inline __device__ float device_get(const const_matrix_view data,
                                   const size_t row,
                                   const size_t col) {
    return device_get(data.data, data.stride, row, col);
}

inline __device__ void device_offset_elem(float* data,
                                          const size_t stride,
                                          const size_t row,
                                          const size_t col,
                                          float value) {
    *(device_get_addr(data, stride, row, col)) += value;
}

inline __device__ void device_offset_elem(matrix_view data,
                                          const size_t row,
                                          const size_t col,
                                          float value) {
    device_offset_elem(data.data, data.stride, row, col, value);
}

inline __device__ void device_offset_elem_atomic(float* data,
                                                 const size_t stride,
                                                 const size_t row,
                                                 const size_t col,
                                                 float value) {
    float* addr = device_get_addr(data, stride, row, col);
    atomicAdd(addr, value);
}

inline __device__ void device_offset_elem_atomic(matrix_view data,
                                                 const size_t row,
                                                 const size_t col,
                                                 float value) {
    device_offset_elem_atomic(data.data, data.stride, row, col, value);
}

template <auto Mapping>
__global__ void map_matrix_kernel(const const_matrix_view input, const matrix_view output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < input.rows && col < input.cols) {
        float val = kernel::matrix::device_get(input, row, col);
        kernel::matrix::device_set(output, row, col, Mapping(val));
    }
}

template <auto Mapping>
::matrix map_matrix(const ::matrix& input) {
    ::matrix output(input.rows, input.cols);

    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);

    map_matrix_kernel<Mapping>
        <<<gridSize, blockSize>>>(input, output);
    return output;
}

template <auto Mapping>
void map_matrix_inplace(::matrix& input) {
    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);

    map_matrix_kernel<Mapping>
        <<<gridSize, blockSize>>>(input, input);
}

}  // namespace kernel::matrix
