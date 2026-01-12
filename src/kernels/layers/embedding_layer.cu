#include "embedding_layer.hpp"

#include <kernels/matrix/device.cuh>
#include <kernels/scheduling.cuh>

static __global__ void positional_encoding_kernel(matrix_view input) {
    const size_t token_i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t encoding_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (token_i < input.rows && encoding_i < input.cols / 2) {
        const auto offset = encoding_i * 2;
        const auto inner
            = token_i
              / std::pow(10000.0f, offset / static_cast<float>(input.cols));
              
        kernel::matrix::device::offset_elem(input, token_i, offset,
                                       std::sin(inner));
        kernel::matrix::device::offset_elem(input, token_i, offset + 1,
                                       std::cos(inner));
    }
}

void kernel::embedding::positional_encoding(::matrix& input, kernel_stream_t stream) {
    const dim3 blockSize(16, 16);
    const dim3 gridSize((input.rows + blockSize.x - 1) / blockSize.x,
                        (input.cols / 2 + blockSize.y - 1) / blockSize.y);

    positional_encoding_kernel<<<gridSize, blockSize, 0, get_kernel_stream(stream)>>>(input);
}
