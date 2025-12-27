#pragma once

#include <cstddef>
#include <mutex>
#include <vector>

namespace kernel {
    // Abstracted stream type in case different backends are used later
    using kernel_stream_t = void*;
    using matmul_handle_t = void*;
    
    using float_device_ptr_t = void*;
    
    kernel_stream_t create_kernel_stream();
    void destroy_kernel_stream(kernel_stream_t stream);
    
    void wait_for_stream(kernel_stream_t);
    
    template <typename ... Args>
    void wait_for_streams(Args ... args) {
        (wait_for_stream(args), ...);
    }
    
    void wait_for_all_streams();
    
    matmul_handle_t create_matmul_handle();
    void destroy_matmul_handle(matmul_handle_t handle);
    
    template <typename T, const size_t PoolSize, const T null_value, T(*init)(), void(*deinit)(T)> 
    struct ObjectPool {
        T streams[PoolSize] = { null_value };
        std::mutex pool_mutex;
        
        ObjectPool() {
            for (size_t i = 0; i < PoolSize; ++i) {
                streams[i] = null_value;
            }
        }
        
        ~ObjectPool() {
            for (size_t i = 0; i < PoolSize; ++i) {
                if (streams[i] != null_value) {
                    deinit(streams[i]);
                }
            }
        }
        
        T acquire() {
            pool_mutex.lock();
            
            static size_t current_index = 0;
            auto out_stream = streams[current_index];
            current_index = (current_index + 1) % PoolSize;
        
            if (out_stream == null_value) {
                out_stream = init();
                streams[current_index == 0 ? PoolSize - 1 : current_index - 1] = out_stream;
            }
            
            pool_mutex.unlock();
            return out_stream;
        }
    };
 
    template <const size_t PoolSize>
    using KernelStreamPool = ObjectPool<kernel_stream_t, PoolSize, nullptr, create_kernel_stream, destroy_kernel_stream>;
    
    template <const size_t PoolSize>
    using MatmulHandlePool = ObjectPool<matmul_handle_t, PoolSize, nullptr, create_matmul_handle, destroy_matmul_handle>;
    
    struct FixedStreamList {
        std::vector<kernel_stream_t> streams;
        
        FixedStreamList(size_t count) {
            streams.reserve(count);
        
            for (size_t i = 0; i < count; ++i) {
                streams.push_back(create_kernel_stream());
            }
        }
        
        FixedStreamList(FixedStreamList&&) = default;
        FixedStreamList& operator =(FixedStreamList&&) = default;
        
        ~FixedStreamList() {
            for (auto stream : streams) {
                destroy_kernel_stream(stream);
            }
        }
        
        constexpr kernel_stream_t operator [](size_t index) const {
            return streams[index];
        }
    };
}