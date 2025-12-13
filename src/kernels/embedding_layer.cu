#include "embedding_layer.hpp"

#include <kernels/matrix_device_kernels.cuh>

// static void positional_encoding(matrix& input) {
//     for (size_t token_i = 0; token_i < input.rows; ++token_i) {
//         for (size_t encoding_i = 0; encoding_i < input.cols / 2;
//         ++encoding_i) {
//             const auto offset = encoding_i * 2;
//             const auto inner
//                 = token_i
//                   / std::pow(10000, offset / static_cast<float>(input.cols));
//             input.offset(token_i, offset, std::sin(inner));
//             input.offset(token_i, offset + 1, std::cos(inner));
//         }
//     }
// }

static __global__ void positional_encoding_kernel(matrix_view input) {
    const size_t token_i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t encoding_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (token_i < input.rows && encoding_i < input.cols / 2) {
        const auto offset = encoding_i * 2;
        const auto inner
            = token_i / powf(10000.0f, offset / static_cast<float>(input.cols));
        kernel::matrix::device_offset_elem(input, token_i, offset, sinf(inner));
        kernel::matrix::device_offset_elem(input, token_i, offset + 1,
                                           cosf(inner));
    }
}

void kernel::embedding::positional_encoding(::matrix& input) {
    const dim3 blockSize(16, 16);
    const dim3 gridSize((input.rows + blockSize.x - 1) / blockSize.x,
                        (input.cols / 2 + blockSize.y - 1) / blockSize.y);

    positional_encoding_kernel<<<gridSize, blockSize>>>(input);
}
