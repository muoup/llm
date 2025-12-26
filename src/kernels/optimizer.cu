#include "kernels/scheduling.hpp"
#include "optimizer.hpp"

#include <kernels/matrix_device_kernels.cuh>
#include <kernels/matrix_kernels.hpp>
#include <kernels/scheduling.cuh>
#include <util/matrix.hpp>

constexpr auto NORM_CLIP_MAX_MAG = 5.0f;

void kernel::optimizer::norm_clip(::matrix &gradient, kernel_stream_t stream) {
  const auto mag =
      kernel::matrix::sum_of_squares(gradient, stream) / gradient.size();
  CHECK_ERRORS("After absmax in norm_clip");

  if (mag > NORM_CLIP_MAX_MAG * NORM_CLIP_MAX_MAG) {
    const float scale = sqrtf((NORM_CLIP_MAX_MAG * NORM_CLIP_MAX_MAG) / mag);
    kernel::matrix::scale(gradient, scale, stream);
    CHECK_ERRORS("After scaling in norm_clip");
  }
}

static __global__ void regularize_gradient(const matrix_view gradient,
                                           const const_matrix_view parameters,
                                           float mag) {
  constexpr float regularization_strength = 0.0001f;

  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= gradient.rows || col >= gradient.cols) {
    return;
  }
 
  float device_value = kernel::matrix::device_get(gradient, row, col);
  
  if (mag > NORM_CLIP_MAX_MAG * NORM_CLIP_MAX_MAG) {
    device_value *= (NORM_CLIP_MAX_MAG * NORM_CLIP_MAX_MAG) / mag;
    kernel::matrix::device_set(gradient, row, col, device_value);
  }
  
  float param_value = kernel::matrix::device_get(parameters, row, col);
  float regularization = 2 * regularization_strength * param_value;
  kernel::matrix::device_set(gradient, row, col, device_value + regularization);
}

void kernel::optimizer::regularize_weight_gradient(::matrix &gradient,
                                                   const ::matrix &parameters,
                                                   kernel_stream_t stream) {
  MATRIX_ASSERT(gradient.rows == parameters.rows &&
                    gradient.cols == parameters.cols,
                "Dimension mismatch in regularize_gradient");

  dim3 threads_per_block(16, 16);
  dim3 blocks((gradient.rows + threads_per_block.x - 1) / threads_per_block.x,
              (gradient.cols + threads_per_block.y - 1) / threads_per_block.y);

  float sum_of_squares = kernel::matrix::sum_of_squares(gradient, stream) / gradient.size();
  regularize_gradient<<<blocks, threads_per_block, 0,
                        from_kernel_stream(stream)>>>(gradient, parameters,
                                                      sum_of_squares);
  CHECK_ERRORS("After regularize_gradient");
}

kernel::KernelStreamPool<4> parameter_optimization_pool;

void kernel::optimizer::adjust_parameter_matrix(::matrix &adjust,
                                                ::matrix &gradient,
                                                float learning_rate,
                                                kernel_stream_t stream) {
  MATRIX_ASSERT(adjust.rows == gradient.rows && adjust.cols == gradient.cols,
                "Dimension mismatch in adjust_parameter_matrix");

  if (stream == nullptr) {
    stream = parameter_optimization_pool.acquire();
  }

  dim3 threads_per_block(16, 16);
  dim3 blocks((adjust.rows + threads_per_block.x - 1) / threads_per_block.x,
              (adjust.cols + threads_per_block.y - 1) / threads_per_block.y);

  // kernel::optimizer::norm_clip(gradient);
  // kernel::optimizer::wait_for_operations();
  kernel::matrix::add_scaled(adjust, gradient, -learning_rate, stream);
  CHECK_ERRORS("After adjust_parameter_matrix");
}

void kernel::optimizer::wait_for_operations(kernel_stream_t stream) {
  cudaStreamSynchronize(from_kernel_stream(stream));
}
