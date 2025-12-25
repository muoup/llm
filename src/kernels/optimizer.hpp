#pragma once

#include <mutex>
#include <util/matrix.hpp>

namespace kernel::optimizer {

typedef void* kernel_stream_t;
    
void norm_clip(::matrix& gradient, kernel_stream_t stream = nullptr);

void adjust_regularize_parameter_matrix(::matrix& gradient, ::matrix& parameters, float learning_rate, kernel_stream_t stream = nullptr);

void adjust_parameter_matrix(::matrix& adjust,
                             ::matrix& gradient,
                             float learning_rate);

void wait_for_operations();
void wait_for_stream(kernel_stream_t stream);

template <typename ... streams>
void wait_for_streams(streams ... s) {
    (wait_for_stream(s), ...);
}

struct kernel_stream_pool {
    size_t stream_count;
    size_t next_stream;
    
    std::mutex next_stream_lock;
    std::vector<kernel_stream_t> streams;

    kernel_stream_pool(size_t stream_count);
    ~kernel_stream_pool();
    kernel_stream_t get_next_stream();
};

}  // namespace kernel::optimizer
