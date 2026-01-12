#pragma once

#include <cstddef>
#include <mutex>
#include <vector>

#include <util/matrix.hpp>

namespace kernel {
    // Abstracted stream type in case different backends are used later
    using kernel_stream_t = void*;
    using matmul_handle_t = void*;
    
    using float_device_ptr_t = void*;
    
    float dereference_device_ptr(DataType type, kernel::float_device_ptr_t ptr);
    void wait_for_stream(kernel_stream_t);
    
    template <typename ... Args>
    void wait_for_streams(Args ... args) {
        (wait_for_stream(args), ...);
    }
    
    void wait_for_all_streams();
    
    kernel_stream_t create_kernel_stream();
    void destroy_kernel_stream(kernel_stream_t stream);
    
    matmul_handle_t create_matmul_handle();
    void destroy_matmul_handle(matmul_handle_t handle);
}