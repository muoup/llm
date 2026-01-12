#include "matrix_kernels.hpp"

#include <kernels/matrix_device_kernels.cuh>
#include <kernels/scheduling.cuh>
#include <kernels/scheduling.hpp>
#include <util/matrix.hpp>

#include <cublas_api.h>
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <device_atomic_functions.h>
#include <math_constants.h>

#include <float.h>

template <typename T>
T* gpu_allocate() {
    T* ptr;
    cudaMallocManaged(&ptr, sizeof(T));
    return ptr;
}

template <typename T>
void gpu_free(T* ptr) {
    cudaFree(ptr);
}

using GPUFloatPool = kernel::
    ObjectPool<float*, 8, nullptr, gpu_allocate<float>, gpu_free<float>>;
using GPUBoolPool
    = kernel::ObjectPool<bool*, 8, nullptr, gpu_allocate<bool>, gpu_free<bool>>;

GPUFloatPool global_gpu_float_pool;
GPUBoolPool global_gpu_bool_pool;

struct CurandGenerator {
    curandGenerator_t gen;

    CurandGenerator() {}

    operator curandGenerator_t() {
        auto status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

        if (status != CURAND_STATUS_SUCCESS) {
            std::puts("Failed to create cuRAND generator");
            std::printf("Error Code: %d\n", status);
            std::fflush(stdout);
            std::exit(1);
        }

        curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
        return gen;
    }
};

static CurandGenerator global_curand_generator;

void kernel::matrix::check_errors(const char* step) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("Failure during: %s\n", step);
        std::printf("CUDA error: %s\n", cudaGetErrorString(err));
        std::abort();
    }
}

::matrix kernel::matrix::async_allocate(const size_t rows,
                                        const size_t cols,
                                        kernel_stream_t stream) {
    ::matrix result;
    result.rows = rows;
    result.cols = cols;
    result.stride = calculate_stride(rows, cols);

    cudaMallocAsync(&result.data, result.buffer_size(),
                    get_kernel_stream(stream));
    cudaMemsetAsync(result.data, 0, result.buffer_size(),
                    get_kernel_stream(stream));
    return result;
}

float* kernel::matrix::allocate_buffer(const size_t size,
                                       kernel_stream_t stream) {
    float* data;
    cudaMallocAsync(&data, size, get_kernel_stream(stream));
    cudaMemsetAsync(data, 0, size, get_kernel_stream(stream));
    CHECK_ERRORS("Allocating matrix buffer");
    kernel::wait_for_stream(stream);
    return data;
}

void kernel::matrix::free_buffer(float* data) {
    cudaFreeAsync(data, nullptr);
}

float* kernel::matrix::gpu_unified_float_ptr() {
    return global_gpu_float_pool.acquire();
}

static __global__ void global_set(const matrix_view data,
                                  const size_t row,
                                  const size_t col,
                                  const float value) {
    kernel::matrix::device_set(data, row, col, value);
}

float* kernel::matrix::get_addr(::matrix& matrix,
                                const size_t row,
                                const size_t col) {
    return &matrix.data[MATRIX_PROJECT(row, col, matrix.stride)];
}

const float* kernel::matrix::get_addr(const ::matrix& matrix,
                                      const size_t row,
                                      const size_t col) {
    return &matrix.data[MATRIX_PROJECT(row, col, matrix.stride)];
}

void kernel::matrix::set(::matrix& matrix,
                         const size_t row,
                         const size_t col,
                         const float value,
                         kernel_stream_t stream) {
    global_set<<<1, 1, 0, get_kernel_stream(stream)>>>(matrix, row, col, value);
}

static __global__ void global_get(const const_matrix_view data,
                                  size_t row,
                                  size_t col,
                                  float* result) {
    *result = kernel::matrix::device_get(data, row, col);
}

float kernel::matrix::get(const ::matrix& matrix,
                          const size_t row,
                          const size_t col,
                          kernel_stream_t stream) {
    float* storage = global_gpu_float_pool.acquire();
    global_get<<<1, 1, 0, get_kernel_stream(stream)>>>(matrix, row, col,
                                                       storage);
    cudaStreamSynchronize(get_kernel_stream(stream));

    return *storage;
}

void kernel::matrix::load_into(::matrix& matrix,
                               const float* host_data,
                               kernel_stream_t stream) {
    cudaMemcpyAsync(matrix.data, host_data, matrix.buffer_size(),
                    cudaMemcpyHostToDevice, get_kernel_stream(stream));
}

void kernel::matrix::store_from(const ::matrix& matrix,
                                float* host_data,
                                kernel_stream_t stream) {
    cudaMemcpyAsync(host_data, matrix.data, matrix.buffer_size(),
                    cudaMemcpyDeviceToHost, get_kernel_stream(stream));
}

void kernel::matrix::randomize(::matrix& matrix,
                               const float min,
                               const float max,
                               kernel_stream_t stream) {
    curandSetStream(global_curand_generator, get_kernel_stream(stream));
    curandGenerateUniform(global_curand_generator, matrix.data,
                          matrix.buffer_size() / sizeof(float));
    const auto range = max - min;

    kernel::matrix::scale(matrix, range, stream);
    kernel::matrix::add(matrix, min, stream);
}

__global__ void copy_matrix_kernel(const matrix_view dest,
                                   const const_matrix_view src) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < src.rows && col < src.cols) {
        auto val = kernel::matrix::device_get(src, row, col);
        kernel::matrix::device_set(dest, row, col, val);
    }
}

matrix kernel::matrix::clone(const ::const_matrix_view other,
                             kernel_stream_t stream) {
    ::matrix result(other.rows, other.cols);

    const dim3 threads_per_block(16, 16);
    const dim3 blocks(
        (other.rows + threads_per_block.x - 1) / threads_per_block.x,
        (other.cols + threads_per_block.y - 1) / threads_per_block.y);
    copy_matrix_kernel<<<blocks, threads_per_block, 0,
                         get_kernel_stream(stream)>>>(result, other);

    return result;
}

__global__ void kernel_set_all(const matrix_view data, float value) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_size = data.rows * data.cols;

    if (idx < total_size) {
        const size_t row = idx / data.cols;
        const size_t col = idx % data.cols;

        kernel::matrix::device_set(data, row, col, value);
    }
}

void kernel::matrix::set_all(::matrix& mat,
                             float value,
                             kernel_stream_t stream) {
    const size_t total_size = mat.rows * mat.cols;
    const size_t threads_per_block = 256;
    const size_t blocks
        = (total_size + threads_per_block - 1) / threads_per_block;

    kernel_set_all<<<blocks, threads_per_block, 0, get_kernel_stream(stream)>>>(
        mat, value);
}

__device__ float kernel_fadd(float a, float b) {
    return a + b;
}

__device__ float kernel_fmaxf(float a, float b) {
    return fmaxf(a, b);
}

__device__ float individual_absmax(float a, float b) {
    return fmaxf(fabsf(a), fabsf(b));
}

__device__ void kernel_atomic_fadd(float* a, float b) {
    atomicAdd(a, b);
}

__device__ void kernel_atomic_fmax(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;

    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
        if (old == assumed)
            break;
    }
}

template <__device__ float (*ElementReducer)(float, float),
          __device__ void (*AtomicReducer)(float*, float)>
static __global__ void matrix_reduce_kernel(const const_matrix_view data,
                                            float* global_result,
                                            float identity) {
    size_t tid = threadIdx.x;
    size_t total_threads = blockDim.x;
    size_t size = data.rows * data.cols;

    float local_acc = identity;
    for (size_t i = blockIdx.x * total_threads + tid; i < size;
         i += gridDim.x * total_threads) {
        size_t r = i / data.cols;
        size_t c = i % data.cols;
        local_acc
            = ElementReducer(local_acc, kernel::matrix::device_get(data, r, c));
    }

    // Block reduction
    if constexpr (AtomicReducer == kernel_atomic_fadd) {
        local_acc = kernel::matrix::device::block_reduce_sum(local_acc);
    } else if constexpr (AtomicReducer == kernel_atomic_fmax) {
        local_acc = kernel::matrix::device::block_reduce_max(local_acc);
    } else {
        // Generic slow reduction for other types if needed, but we mostly use
        // sum/max
        static __shared__ float shared_data[1024];
        shared_data[tid] = local_acc;
        __syncthreads();
        for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
                shared_data[tid]
                    = ElementReducer(shared_data[tid], shared_data[tid + s]);
            __syncthreads();
        }
        local_acc = shared_data[0];
    }

    if (tid == 0) {
        AtomicReducer(global_result, local_acc);
    }
}

template <__device__ float (*ElementReducer)(float, float),
          __device__ void (*AtomicReducer)(float*, float)>
kernel::float_device_ptr_t run_reduction(const ::matrix& mat,
                                         float identity,
                                         kernel::kernel_stream_t stream
                                         = nullptr) {
    float* reduction_result = global_gpu_float_pool.acquire();
    cudaMemcpyAsync(reduction_result, &identity, sizeof(float),
                    cudaMemcpyHostToDevice, get_kernel_stream(stream));

    const size_t threads_per_block = 256;
    const size_t num_elements = mat.rows * mat.cols;
    const size_t num_blocks
        = std::min((size_t)1024,
                   (num_elements + threads_per_block - 1) / threads_per_block);

    matrix_reduce_kernel<ElementReducer, AtomicReducer>
        <<<num_blocks, threads_per_block, 0, get_kernel_stream(stream)>>>(
            mat, reduction_result, identity);

    return (kernel::float_device_ptr_t)reduction_result;
}

kernel::float_device_ptr_t kernel::matrix::sum(const ::matrix& mat,
                                               kernel_stream_t stream) {
    return run_reduction<kernel_fadd, kernel_atomic_fadd>(mat, 0.0f, stream);
}

__device__ float abs_val(float a, float b) {
    return a + fabsf(b);
}

kernel::float_device_ptr_t kernel::matrix::abssum(const ::matrix& mat,
                                                  kernel_stream_t stream) {
    return run_reduction<abs_val, kernel_atomic_fadd>(mat, 0.0f, stream);
}

__device__ float square_val(float a, float b) {
    return a + b * b;
}

kernel::float_device_ptr_t kernel::matrix::sum_of_squares(
    const ::matrix& mat,
    kernel_stream_t stream) {
    return run_reduction<square_val, kernel_atomic_fadd>(mat, 0.0f, stream);
}

kernel::float_device_ptr_t kernel::matrix::max(const ::matrix& mat,
                                               kernel_stream_t stream) {
    return run_reduction<kernel_fmaxf, kernel_atomic_fmax>(
        mat, std::numeric_limits<float>::min(), stream);
}

__device__ float kernel_fminf(float a, float b) {
    return fminf(a, b);
}

__device__ void kernel_atomic_fmin(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
        if (old == assumed)
            break;
    }
}

kernel::float_device_ptr_t kernel::matrix::min(const ::matrix& mat,
                                               kernel_stream_t stream) {
    return run_reduction<kernel_fminf, kernel_atomic_fmin>(
        mat, std::numeric_limits<float>::max(), stream);
}

kernel::float_device_ptr_t kernel::matrix::absmax(const ::matrix& mat,
                                                  kernel_stream_t stream) {
    return run_reduction<individual_absmax, kernel_atomic_fmax>(mat, 0.0f,
                                                                stream);
}

__global__ void variance_kernel(float* sum_ptr,
                                float* sum_sq_ptr,
                                float* result,
                                size_t total_elements) {
    float sum = *sum_ptr;
    float sum_of_squares = *sum_sq_ptr;
    float n = (float)total_elements;
    *result = (sum_of_squares / n) - (sum * sum) / (n * n);
}

kernel::float_device_ptr_t kernel::matrix::variance(const ::matrix& mat,
                                                    kernel_stream_t stream) {
    float_device_ptr_t sum_ptr = kernel::matrix::sum(mat, stream);
    float_device_ptr_t sum_of_squares_ptr
        = kernel::matrix::sum_of_squares(mat, stream);

    float* device_result = global_gpu_float_pool.acquire();

    variance_kernel<<<1, 1, 0, get_kernel_stream(stream)>>>(
        (float*)sum_ptr, (float*)sum_of_squares_ptr, device_result,
        mat.rows * mat.cols);

    return (float_device_ptr_t)device_result;
}

__global__ void kernel_scale(const matrix_view data, const float factor) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < data.rows && col < data.cols) {
        *(kernel::matrix::device_get_addr(data, row, col)) *= factor;
    }
}

void kernel::matrix::scale(::matrix& mat,
                           const float factor,
                           kernel_stream_t stream) {
    dim3 threads_per_block(16, 16);
    dim3 blocks((mat.rows + threads_per_block.x - 1) / threads_per_block.x,
                (mat.cols + threads_per_block.y - 1) / threads_per_block.y);

    kernel_scale<<<blocks, threads_per_block, 0, get_kernel_stream(stream)>>>(
        mat, factor);
}

static __global__ void kernel_transfer_row(const matrix_view dest,
                                           size_t dest_row,
                                           const const_matrix_view src,
                                           size_t src_row) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < src.cols && col < src.cols) {
        auto val = kernel::matrix::device_get(src, src_row, col);
        kernel::matrix::device_set(dest, dest_row, col, val);
    }
}

void kernel::matrix::transfer_row(::matrix& dest,
                                  const size_t dest_row,
                                  const ::matrix& src,
                                  const size_t src_row,
                                  kernel_stream_t stream) {
    const size_t threads_per_block = 256;
    const size_t blocks
        = (src.cols + threads_per_block - 1) / threads_per_block;

    kernel_transfer_row<<<blocks, threads_per_block, 0,
                          get_kernel_stream(stream)>>>(dest, dest_row, src,
                                                       src_row);
}

void kernel::matrix::set_row_vector(::matrix& mat,
                                    const size_t mat_row,
                                    const ::matrix& vec,
                                    const size_t vec_row,
                                    kernel_stream_t stream) {
    cudaMemcpyAsync(kernel::matrix::get_addr(mat, mat_row, 0),
                    kernel::matrix::get_addr(vec, vec_row, 0),
                    vec.cols * sizeof(float), cudaMemcpyDeviceToDevice,
                    get_kernel_stream(stream));
}

::matrix kernel::matrix::get_row_vector(const ::matrix& mat,
                                        const size_t row,
                                        kernel_stream_t stream) {
    ::matrix result(1, mat.cols);

    cudaMemcpyAsync(kernel::matrix::get_addr(result, 0, 0),
                    kernel::matrix::get_addr(mat, row, 0),
                    mat.cols * sizeof(float), cudaMemcpyDeviceToDevice,
                    get_kernel_stream(stream));

    return result;
}

static __global__ void add_row_vector_kernel(const matrix_view data,
                                             size_t data_row,
                                             const const_matrix_view row_vec,
                                             size_t offset_row,
                                             float scale) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < data.cols) {
        auto val = kernel::matrix::device_get(row_vec, offset_row, col);
        kernel::matrix::device_offset_elem(data, data_row, col, val * scale);
    }
}

void kernel::matrix::add_row_vector(::matrix& mat,
                                    const size_t row,
                                    const ::matrix& vec,
                                    size_t vec_row,
                                    float scale,
                                    kernel_stream_t stream) {
    const size_t threads_per_block = 256;
    const size_t blocks
        = (mat.cols + threads_per_block - 1) / threads_per_block;

    add_row_vector_kernel<<<blocks, threads_per_block, 0,
                            get_kernel_stream(stream)>>>(mat, row, vec, vec_row,
                                                         scale);
}

static __global__ void atomic_add_row_vector_kernel(
    const matrix_view data,
    size_t data_row,
    const const_matrix_view row_vec,
    size_t offset_row,
    float scale) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < data.cols) {
        auto val = kernel::matrix::device_get(row_vec, offset_row, col);
        kernel::matrix::device_offset_elem_atomic(data, data_row, col,
                                                  val * scale);
    }
}

void kernel::matrix::atomic_add_row_vector(::matrix& mat,
                                           const size_t row,
                                           const ::matrix& vec,
                                           size_t vec_row,
                                           float scale,
                                           kernel_stream_t stream) {
    const size_t threads_per_block = 256;
    const size_t blocks
        = (mat.cols + threads_per_block - 1) / threads_per_block;

    atomic_add_row_vector_kernel<<<blocks, threads_per_block, 0,
                                   get_kernel_stream(stream)>>>(mat, row, vec,
                                                                vec_row, scale);
}

static __global__ void set_horizontal_slice_kernel(
    const matrix_view data,
    const size_t start_col,
    const const_matrix_view slice) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < slice.rows && col < slice.cols) {
        auto val = kernel::matrix::device_get(slice, row, col);
        kernel::matrix::device_set(data, row, start_col + col, val);
    }
}

void kernel::matrix::set_horizontal_slice(::matrix& mat,
                                          const size_t start_col,
                                          const ::matrix& slice,
                                          kernel_stream_t stream) {
    const dim3 threads_per_block(16, 16);
    const dim3 blocks(
        (slice.rows + threads_per_block.x - 1) / threads_per_block.x,
        (slice.cols + threads_per_block.y - 1) / threads_per_block.y);

    set_horizontal_slice_kernel<<<blocks, threads_per_block, 0,
                                  get_kernel_stream(stream)>>>(mat, start_col,
                                                               slice);
}

static __global__ void matrix_add_matrix(const matrix_view data,
                                         const const_matrix_view offset) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < data.rows && col < data.cols) {
        float value = kernel::matrix::device_get(offset, row, col);
        kernel::matrix::device_offset_elem(data, row, col, value);
    }
}

void kernel::matrix::add(::matrix& mat,
                         const ::matrix& offset,
                         kernel_stream_t stream) {
    const dim3 threads_per_block(16, 16);
    const dim3 blocks(
        (mat.rows + threads_per_block.x - 1) / threads_per_block.x,
        (mat.cols + threads_per_block.y - 1) / threads_per_block.y);

    matrix_add_matrix<<<blocks, threads_per_block, 0,
                        get_kernel_stream(stream)>>>(mat, offset);
}

static __global__ void matrix_atomic_add_matrix(
    const matrix_view data,
    const const_matrix_view offset) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < data.rows && col < data.cols) {
        float value = kernel::matrix::device_get(offset, row, col);
        kernel::matrix::device_offset_elem_atomic(data, row, col, value);
    }
}

void kernel::matrix::atomic_add(::matrix& mat,
                                const ::matrix& offset,
                                kernel_stream_t stream) {
    const dim3 threads_per_block(16, 16);
    const dim3 blocks(
        (mat.rows + threads_per_block.x - 1) / threads_per_block.x,
        (mat.cols + threads_per_block.y - 1) / threads_per_block.y);

    matrix_atomic_add_matrix<<<blocks, threads_per_block, 0,
                               get_kernel_stream(stream)>>>(mat, offset);
}

static __global__ void kernel_add_scaled(const matrix_view data,
                                         const const_matrix_view other,
                                         const float factor) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= data.rows || col >= data.cols)
        return;

    float value = kernel::matrix::device_get(other, row, col);
    kernel::matrix::device_offset_elem(data, row, col, value * factor);
}

void kernel::matrix::add_scaled(::matrix& mat,
                                const ::matrix& other,
                                const float factor,
                                kernel_stream_t stream) {
    dim3 threads_per_block(16, 16);
    dim3 blocks((mat.rows + threads_per_block.x - 1) / threads_per_block.x,
                (mat.cols + threads_per_block.y - 1) / threads_per_block.y);

    kernel_add_scaled<<<blocks, threads_per_block, 0,
                        get_kernel_stream(stream)>>>(mat, other, factor);
}

static __global__ void kernel_add_value(const matrix_view data,
                                        const float value) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < data.rows && col < data.cols) {
        kernel::matrix::device_offset_elem(data, row, col, value);
    }
}

void kernel::matrix::add(::matrix& mat, float value, kernel_stream_t stream) {
    const dim3 threads_per_block(16, 16);
    const dim3 blocks(
        (mat.rows + threads_per_block.x - 1) / threads_per_block.x,
        (mat.cols + threads_per_block.y - 1) / threads_per_block.y);

    kernel_add_value<<<blocks, threads_per_block, 0,
                       get_kernel_stream(stream)>>>(mat, value);
}

__global__ void kernel_softmax(matrix_view data) {
    const size_t row = blockIdx.x;
    const size_t tid = threadIdx.x;

    float local_max = -CUDART_INF_F;
    for (size_t col = tid; col < data.cols; col += blockDim.x) {
        float val = kernel::matrix::device_get(data, row, col);
        local_max = fmaxf(local_max, val);
    }

    float row_max = kernel::matrix::device::block_reduce_max(local_max);

    __shared__ float shared_max;
    if (tid == 0) {
        shared_max = row_max;
    }

    __syncthreads();

    float local_sum = 0.0f;
    if (shared_max > -CUDART_INF_F) {
        for (size_t col = tid; col < data.cols; col += blockDim.x) {
            float val = kernel::matrix::device_get(data, row, col);
            float exp = __expf(val - shared_max);

            kernel::matrix::device_set(data, row, col, exp);
            local_sum += exp;
        }
    } else {
        for (size_t col = tid; col < data.cols; col += blockDim.x) {
            kernel::matrix::device_set(data, row, col, 0.0f);
        }
    }
    __syncthreads();

    float row_dot = kernel::matrix::device::block_reduce_sum(local_sum);

    __shared__ float shared_denom;
    if (tid == 0) {
        shared_denom = row_dot;
    }
    __syncthreads();

    if (shared_denom > 0.0f) {
        for (size_t col = tid; col < data.cols; col += blockDim.x) {
            float val = kernel::matrix::device_get(data, row, col);
            kernel::matrix::device_set(data, row, col, val / shared_denom);
        }
    }
}

void kernel::matrix::softmax(::matrix& mat, kernel_stream_t stream) {
    const dim3 threads_per_block = 256;
    const dim3 blocks = mat.rows;

    kernel_softmax<<<blocks, threads_per_block, 0, get_kernel_stream(stream)>>>(
        mat);
}

static __global__ void kernel_backprop_softmax(
    const const_matrix_view softmax_output,
    const const_matrix_view output_gradient,
    matrix_view softmax_gradient) {
    const size_t row = blockIdx.x;

    float local_dot = 0.0f;
    for (size_t col = threadIdx.x; col < softmax_output.cols;
         col += blockDim.x) {
        float s_j = kernel::matrix::device_get(softmax_output, row, col);
        float g_j = kernel::matrix::device_get(output_gradient, row, col);
        local_dot += s_j * g_j;
    }
    float row_dot = kernel::matrix::device::block_reduce_sum(local_dot);

    __shared__ float shared_dot;
    if (threadIdx.x == 0) {
        shared_dot = row_dot;
    }
    __syncthreads();

    row_dot = shared_dot;

    for (size_t col = threadIdx.x; col < softmax_gradient.cols;
         col += blockDim.x) {
        float s_j = kernel::matrix::device_get(softmax_output, row, col);
        float g_j = kernel::matrix::device_get(output_gradient, row, col);
        kernel::matrix::device_set(softmax_gradient, row, col,
                                   s_j * (g_j - row_dot));
    }
}

void kernel::matrix::backprop_softmax(::matrix& buffer,
                                      const ::matrix& output,
                                      const ::matrix& gradient,
                                      kernel_stream_t stream) {
    const size_t threads_per_block = 256;
    const size_t blocks = gradient.rows;

    kernel_backprop_softmax<<<blocks, threads_per_block, 0,
                              get_kernel_stream(stream)>>>(output, gradient,
                                                           buffer);
}

void __global__ kernel_mask_upper_triangle(const matrix_view data,
                                           const float mask_value) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < data.rows && col < data.cols && col > row) {
        kernel::matrix::device_set(data, row, col, mask_value);
    }
}

void kernel::matrix::mask_upper_triangle(::matrix& mat,
                                         const float mask_value,
                                         kernel_stream_t stream) {
    const dim3 threads_per_block(16, 16);
    const dim3 blocks(
        (mat.rows + threads_per_block.x - 1) / threads_per_block.x,
        (mat.cols + threads_per_block.y - 1) / threads_per_block.y);

    kernel_mask_upper_triangle<<<blocks, threads_per_block, 0,
                                 get_kernel_stream(stream)>>>(mat, mask_value);
}

static kernel::MatmulHandlePool<8> matmul_handle_pool;

matrix kernel::matrix::dot_product(const ::matrix& a,
                                   const ::matrix& b,
                                   kernel_stream_t stream) {
    throw std::runtime_error("Deprecated: Use element-wise operations");

    // ::matrix result = async_allocate(1, 1, stream);
    // auto handle = matmul_handle_pool.acquire();
    // cublasSetStream(get_matmul_handle(handle), get_kernel_stream(stream));

    // cublasSdot(get_matmul_handle(handle), a.rows * a.cols, a.data, 1, b.data,
    // 1,
    //            result.data);

    // return result;
}

matrix kernel::matrix::cross_multiplied(const ::const_matrix_view a,
                                        const ::const_matrix_view b,
                                        kernel_stream_t stream) {
    ::matrix result = async_allocate(a.rows, b.cols, stream);
    auto handle = matmul_handle_pool.acquire();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSetStream(get_matmul_handle(handle), get_kernel_stream(stream));
    cublasSgemm(get_matmul_handle(handle), CUBLAS_OP_N, CUBLAS_OP_N, b.cols,
                a.rows, a.cols, &alpha, b.data, b.stride, a.data, a.stride,
                &beta, result.data_ptr(), result.stride);
    CHECK_ERRORS("cross_multiplied");

    return result;
}

matrix kernel::matrix::cross_t_multiplied(const ::const_matrix_view a,
                                          const ::const_matrix_view b,
                                          kernel_stream_t stream) {
    ::matrix result = async_allocate(a.rows, b.rows, stream);
    auto handle = matmul_handle_pool.acquire();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSetStream(get_matmul_handle(handle), get_kernel_stream(stream));
    cublasSgemm(get_matmul_handle(handle), CUBLAS_OP_T, CUBLAS_OP_N, b.rows,
                a.rows, a.cols, &alpha, b.data, b.stride, a.data, a.stride,
                &beta, result.data_ptr(), result.stride);
    CHECK_ERRORS("cross_t_multiplied");

    return result;
}

matrix kernel::matrix::t_cross_multiplied(const ::const_matrix_view a,
                                          const ::const_matrix_view b,
                                          kernel_stream_t stream) {
    ::matrix result = async_allocate(a.cols, b.cols, stream);
    auto handle = matmul_handle_pool.acquire();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSetStream(get_matmul_handle(handle), get_kernel_stream(stream));
    cublasSgemm(get_matmul_handle(handle), CUBLAS_OP_N, CUBLAS_OP_T, b.cols,
                a.cols, a.rows, &alpha, b.data, b.stride, a.data, a.stride,
                &beta, result.data, result.stride);
    CHECK_ERRORS("t_cross_multiplied");

    return result;
}

__global__ void element_wise_multiply_kernel(matrix_view a_data,
                                             const const_matrix_view b_data) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < a_data.rows && col < a_data.cols) {
        float val_a = kernel::matrix::device_get(a_data, row, col);
        float val_b = kernel::matrix::device_get(b_data, row, col);
        kernel::matrix::device_set(a_data, row, col, val_a * val_b);
    }
}

void kernel::matrix::element_wise_multiply(::matrix& a,
                                           const ::matrix& b,
                                           kernel_stream_t stream) {
    const dim3 threads_per_block(16, 16);
    const dim3 blocks((a.rows + threads_per_block.x - 1) / threads_per_block.x,
                      (a.cols + threads_per_block.y - 1) / threads_per_block.y);

    element_wise_multiply_kernel<<<blocks, threads_per_block, 0,
                                   get_kernel_stream(stream)>>>(a, b);
}

__global__ void compare(const_matrix_view a,
                        const_matrix_view b,
                        float epsilon,
                        bool* result) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_size = a.rows * a.cols;

    if (idx < total_size) {
        const size_t row = idx / a.cols;
        const size_t col = idx % a.cols;

        float val_a = kernel::matrix::device_get(a, row, col);
        float val_b = kernel::matrix::device_get(b, row, col);

        if (fabsf(val_a - val_b) > epsilon) {
            *result = false;
        }
    }
};

bool kernel::matrix::is_equal(const ::matrix& a,
                              const ::matrix& b,
                              const float epsilon,
                              kernel_stream_t stream) {
    if (a.rows != b.rows || a.cols != b.cols) {
        return false;
    }

    bool* d_result = global_gpu_bool_pool.acquire();
    *d_result = true;

    compare<<<(a.rows * a.cols + 255) / 256, 256, 0,
              get_kernel_stream(stream)>>>(a, b, epsilon, d_result);

    return *d_result;
}
