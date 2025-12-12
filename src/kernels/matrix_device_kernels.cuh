#pragma once

#include <cuda_runtime.h>
#include <util/matrix.hpp>

namespace kernel::matrix {

inline __device__ float* device_get_addr(float* data,
                                         const size_t stride,
                                         const size_t row,
                                         const size_t col) {
    return &(data[row + col * stride]);
}

inline __device__ const float* device_get_addr(const float* data,
                                         const size_t stride,
                                         const size_t row,
                                         const size_t col) {
    return &(data[row + col * stride]);
}

inline __device__ void device_set(float* data,
                                  const size_t stride,
                                  const size_t row,
                                  const size_t col,
                                  const float value) {
    *(device_get_addr(data, stride, row, col)) = value;
}

inline __device__ float device_get(const float* data,
                                   const size_t stride,
                                   const size_t row,
                                   const size_t col) {
    return *(device_get_addr(data, stride, row, col));
}

inline __device__ void device_offset_elem(float* data,
                                          const size_t stride,
                                          const size_t row,
                                          const size_t col,
                                          float value) {
    *(device_get_addr(data, stride, row, col)) += value;
}

inline __device__ void device_offset_elem_atomic(float* data,
                                                 const size_t stride,
                                                 const size_t row,
                                                 const size_t col,
                                                 float value) {
    float* addr = device_get_addr(data, stride, row, col);
    atomicAdd(addr, value);
}

template <auto Mapping>
__global__ void map_matrix_kernel(const float* input,
                                  float* output,
                                  std::uint64_t stride,
                                  std::uint64_t rows,
                                  std::uint64_t cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        float val = kernel::matrix::device_get(input, stride, row, col);
        kernel::matrix::device_set(output, stride, row, col, Mapping(val));
    }
}

template <auto Mapping>
::matrix map_matrix(const ::matrix& input) {
    ::matrix output(input.rows, input.cols);

    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);

    map_matrix_kernel<Mapping>
        <<<gridSize, blockSize>>>(input.data_ptr(), output.data_ptr(),
                                  input.stride, input.rows, input.cols);

    cudaDeviceSynchronize();
    return output;
}

template <auto Mapping>
void map_matrix_inplace(::matrix& input) {
    dim3 blockSize(16, 16);
    dim3 gridSize((input.cols + blockSize.x - 1) / blockSize.x,
                  (input.rows + blockSize.y - 1) / blockSize.y);

    map_matrix_kernel<Mapping>
        <<<gridSize, blockSize>>>(input.data_ptr(), input.data_ptr(),
                                  input.stride, input.rows, input.cols);

    cudaDeviceSynchronize();
}

}  // namespace kernel::matrix
