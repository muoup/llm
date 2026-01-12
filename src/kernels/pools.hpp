#pragma once

#include <kernels/scheduling.hpp>

namespace kernel {

template <typename T>
struct ObjectPool {
    T* objs = nullptr;
    T null_value;
    size_t pool_size;
    std::mutex pool_mutex;

    T (*init)();
    void (*deinit)(T);

    ObjectPool(T (*init)(), void (*deinit)(T), T null_value, size_t pool_size)
        : null_value(null_value), init(init), deinit(deinit), pool_size(pool_size) {
        objs = new T[pool_size];
    }
    ~ObjectPool() {
        for (size_t i = 0; i < pool_size; ++i) {
            if (objs[i] != null_value) {
                deinit(objs[i]);
            }
        }

        delete[] objs;
    }

    T acquire() {
        pool_mutex.lock();

        static size_t current_index = 0;
        auto out_stream = objs[current_index];
        current_index = (current_index + 1) % pool_size;

        if (out_stream == null_value) {
            out_stream = init();
            objs[current_index == 0 ? pool_size - 1 : current_index - 1]
                = out_stream;
        }

        pool_mutex.unlock();
        return out_stream;
    }
};

using KernelStreamPool = ObjectPool<kernel_stream_t>;
using MatmulHandlePool = ObjectPool<matmul_handle_t>;

KernelStreamPool new_kernel_stream_pool(size_t pool_size);
MatmulHandlePool new_matmul_handle_pool(size_t pool_size);

extern ObjectPool<float*> global_gpu_float_pool; 
extern ObjectPool<uint16_t*> global_gpu_half_pool;
extern ObjectPool<uint16_t*> global_gpu_bf16_pool;
extern ObjectPool<bool*> global_gpu_bool_pool;

struct FixedStreamList {
    std::vector<kernel_stream_t> streams;

    FixedStreamList(size_t count) {
        streams.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            streams.push_back(create_kernel_stream());
        }
    }

    FixedStreamList(FixedStreamList&&) = default;
    FixedStreamList& operator=(FixedStreamList&&) = default;

    ~FixedStreamList() {
        for (auto stream : streams) {
            destroy_kernel_stream(stream);
        }
    }

    constexpr kernel_stream_t operator[](size_t index) const {
        return streams[index];
    }
};

}  // namespace kernel
