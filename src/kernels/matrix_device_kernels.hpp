#pragma once

#include <util/matrix.hpp>

#include <cuda_runtime.h>

namespace kernel::matrix {

inline __device__ void device_set(float* data,
                                  const size_t stride,
                                  const size_t rows,
                                  const size_t cols,
                                  const size_t row,
                                  const size_t col,
                                  const float value) {
    data[row + col * stride] = value;
}

inline __device__ float device_get(const float* data,
                                   const size_t stride,
                                   const size_t rows,
                                   const size_t cols,
                                   const size_t row,
                                   const size_t col) {
    return data[row + col * stride];
}

inline __device__ void device_offset_elem(const float* data,
                                          const size_t stride,
                                          const size_t rows,
                                          const size_t cols,
                                          const size_t row,
                                          const size_t col,
                                          float value) {
    auto current = device_get(data, stride, rows, cols, row, col);
    device_set(const_cast<float*>(data), stride, rows, cols, row, col,
               current + value);
}

}  // namespace kernel::matrix
