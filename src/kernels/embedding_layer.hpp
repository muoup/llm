#pragma once

#include <kernels/scheduling.hpp>
#include <util/matrix.hpp>

namespace kernel::embedding {

void positional_encoding(::matrix& input, kernel_stream_t stream = nullptr);

}