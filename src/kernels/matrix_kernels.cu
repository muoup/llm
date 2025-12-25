#include "kernels/optimizer.hpp"

#include <kernels/matrix_device_kernels.cuh>
#include <kernels/matrix_kernels.hpp>
#include <util/matrix.hpp>

#include <cublas_api.h>
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <device_atomic_functions.h>
#include <math_constants.h>

#include <float.h>

static cudaStream_t s_from_stream(kernel::matrix::kernel_stream_t stream) {
    return static_cast<cudaStream_t>(stream);
}

struct CurandGenerator {
    curandGenerator_t gen = nullptr;

    CurandGenerator() {}

    operator curandGenerator_t() {
        if (gen != nullptr) {
            return gen;
        }

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

void cleanup_cublas();

struct cuBlasHandlePool {
    cublasHandle_t get_handle() {
        static thread_local cublasHandle_t handle = nullptr;
        
        if (handle == nullptr) {
            auto status = cublasCreate(&handle);

            if (status != CUBLAS_STATUS_SUCCESS) {
                std::puts("Failed to create cuBLAS handle");
                std::printf("Error Code: %d\n", status);
                std::fflush(stdout);
                std::exit(1);
            }
        }

        return handle;
    }

    // operator overload to allow easy usage
    operator cublasHandle_t() { return get_handle(); }
};

static cuBlasHandlePool cublas_handle_pool;

// Persistent buffer for small results (e.g. reductions)
static float* d_reduction_result = nullptr;
static bool* d_reduction_bool_result = nullptr;

void ensure_reduction_buffer() {
    if (d_reduction_result == nullptr) {
        cudaMalloc(&d_reduction_result, sizeof(float));
        cudaMalloc(&d_reduction_bool_result, sizeof(bool));
    }
}

constexpr size_t MAX_CLEANUP_STREAMS = 8;
static kernel::matrix::kernel_stream_t cleanup_streams[MAX_CLEANUP_STREAMS];

kernel::matrix::kernel_stream_t get_cleanup_stream() {
    static bool initialized = false;
    static size_t next_stream = 0;

    if (!initialized) [[unlikely]] {
        for (size_t i = 0; i < MAX_CLEANUP_STREAMS; ++i) {
            cudaStream_t stream;
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            cleanup_streams[i]
                = static_cast<kernel::matrix::kernel_stream_t>(stream);
        }
        initialized = true;
    }

    size_t stream = next_stream;
    next_stream = (next_stream + 1) % MAX_CLEANUP_STREAMS;

    return cleanup_streams[stream];
}

__global__ void test_print() {
    printf("Hello from CUDA kernel!\n");
}

void kernel::matrix::test_print() {
    ::test_print<<<1, 1>>>();
}

void kernel::matrix::check_errors(const char* step) {
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::printf("Failure during: %s\n", step);
        std::printf("CUDA error: %s\n", cudaGetErrorString(err));
        std::abort();
    }
}

static kernel::optimizer::kernel_stream_pool allocation_pool(4);

float* kernel::matrix::allocate_buffer(const size_t size) {
    kernel_stream_t stream = allocation_pool.get_next_stream();

    CHECK_ERRORS("Pre matrix buffer allocation");
    float* data;
    cudaMallocAsync(&data, size, s_from_stream(stream));
    kernel::optimizer::wait_for_stream(stream);
    cudaMemsetAsync(data, 0, size, s_from_stream(stream));
    kernel::optimizer::wait_for_stream(stream);
    CHECK_ERRORS("Allocating matrix buffer");
    return data;
}

::matrix kernel::matrix::async_create(const size_t rows,
                                      const size_t cols,
                                      kernel::matrix::kernel_stream_t stream) {
    size_t stride = ::matrix::calculate_stride(rows);
    size_t size = stride * rows * sizeof(float);

    ::matrix result;
    result.rows = rows;
    result.cols = cols;
    result.stride = stride;

    cudaMallocAsync(&result.data, size, s_from_stream(stream));
    kernel::optimizer::wait_for_stream(stream);
    cudaMemsetAsync(result.data, 0, size, s_from_stream(stream));
    CHECK_ERRORS("Async matrix buffer allocation");

    return result;
}

void kernel::matrix::free_buffer(float* data) {
    auto stream = allocation_pool.get_next_stream();

    cudaFreeAsync(data, s_from_stream(stream));
    kernel::optimizer::wait_for_stream(stream);
}

static __global__ void global_set(const matrix_view data,
                                  const size_t row,
                                  const size_t col,
                                  const float value) {
    kernel::matrix::device_set(data, row, col, value);
}

void kernel::matrix::set(::matrix& matrix,
                         const size_t row,
                         const size_t col,
                         const float value) {
    global_set<<<1, 1>>>(matrix, row, col, value);
}

static __global__ void global_get(const const_matrix_view data,
                                  size_t row,
                                  size_t col,
                                  float* result) {
    *result = kernel::matrix::device_get(data, row, col);
}

float kernel::matrix::get(const ::matrix& matrix,
                          const size_t row,
                          const size_t col) {
    ensure_reduction_buffer();
    global_get<<<1, 1>>>(matrix, row, col, d_reduction_result);
    float value;
    cudaMemcpy(&value, d_reduction_result, sizeof(float),
               cudaMemcpyDeviceToHost);
    return value;
}

void kernel::matrix::load_into(::matrix& matrix, const float* host_data) {
    cudaMemcpy(matrix.data, host_data, matrix.buffer_size(),
               cudaMemcpyHostToDevice);
}

void kernel::matrix::store_from(const ::matrix& matrix, float* host_data) {
    cudaMemcpy(host_data, matrix.data, matrix.buffer_size(),
               cudaMemcpyDeviceToHost);
}

void kernel::matrix::randomize(::matrix& matrix,
                               const float min,
                               const float max) {
    curandSetStream(global_curand_generator, nullptr);
    curandGenerateUniform(global_curand_generator, matrix.data,
                          matrix.buffer_size() / sizeof(float));

    const auto range = max - min;

    matrix.scale(range);
    kernel::optimizer::wait_for_operations();
    matrix.add(min);
}

__global__ void copy_matrix_kernel(const matrix_view dest,
                                   const const_matrix_view src) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = src.rows * src.cols;

    if (idx < total) {
        size_t row = idx % src.rows;
        size_t col = idx / src.rows;
        auto val = kernel::matrix::device_get(src, row, col);
        kernel::matrix::device_set(dest, row, col, val);
    }
}

matrix kernel::matrix::clone(const ::const_matrix_view other) {
    ::matrix result(other.rows, other.cols);

    const size_t threads_per_block = 256;
    const size_t blocks
        = (other.rows * other.cols + threads_per_block - 1) / threads_per_block;
    copy_matrix_kernel<<<blocks, threads_per_block>>>(result, other);

    return result;
}

__global__ void kernel_set_all(float* data,
                               const size_t stride,
                               const size_t rows,
                               const size_t cols,
                               float value) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_size = rows * cols;

    if (idx < total_size) {
        const size_t row = idx % rows;
        const size_t col = idx / rows;
        kernel::matrix::device_set(data, stride, row, col, value);
    }
}

void kernel::matrix::set_all(::matrix& mat, float value) {
    const size_t total_size = mat.rows * mat.cols;
    const size_t threads_per_block = 256;
    const size_t blocks
        = (total_size + threads_per_block - 1) / threads_per_block;

    kernel_set_all<<<blocks, threads_per_block>>>(mat.data, mat.stride,
                                                  mat.rows, mat.cols, value);
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
        size_t r = i % data.rows;
        size_t c = i / data.rows;
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
float run_reduction(const ::matrix& mat,
                    float identity,
                    kernel::matrix::kernel_stream_t stream = nullptr) {
    float* reduction_result;
    cudaMalloc(&reduction_result, sizeof(float));
    cudaMemcpy(reduction_result, &identity, sizeof(float),
               cudaMemcpyHostToDevice);

    const size_t threads_per_block = 256;
    const size_t num_elements = mat.rows * mat.cols;
    const size_t num_blocks
        = std::min((size_t)1024,
                   (num_elements + threads_per_block - 1) / threads_per_block);

    matrix_reduce_kernel<ElementReducer, AtomicReducer>
        <<<num_blocks, threads_per_block, 0, s_from_stream(stream)>>>(
            mat, reduction_result, identity);

    float h_result;
    cudaMemcpy(&h_result, reduction_result, sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(reduction_result);
    return h_result;
}

float kernel::matrix::sum(const ::matrix& mat, kernel_stream_t stream) {
    return run_reduction<kernel_fadd, kernel_atomic_fadd>(mat, 0.0f, stream);
}

__device__ float abs_val(float a, float b) {
    return a + fabsf(b);
}

float kernel::matrix::abssum(const ::matrix& mat) {
    return run_reduction<abs_val, kernel_atomic_fadd>(mat, 0.0f);
}

__device__ float square_val(float a, float b) {
    return a + b * b;
}

float kernel::matrix::sum_of_squares(const ::matrix& mat,
                                     kernel_stream_t stream) {
    return run_reduction<square_val, kernel_atomic_fadd>(mat, 0.0f, stream);
}

float kernel::matrix::max(const ::matrix& mat) {
    return run_reduction<kernel_fmaxf, kernel_atomic_fmax>(
        mat, std::numeric_limits<float>::min());
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

float kernel::matrix::min(const ::matrix& mat) {
    return run_reduction<kernel_fminf, kernel_atomic_fmin>(
        mat, std::numeric_limits<float>::max());
}

float kernel::matrix::absmax(const ::matrix& mat) {
    return run_reduction<individual_absmax, kernel_atomic_fmax>(mat, 0.0f);
}

float kernel::matrix::variance(const ::matrix& mat) {
    float sum = kernel::matrix::sum(mat);
    float sum_of_squares = kernel::matrix::sum_of_squares(mat);
    size_t size = mat.rows * mat.cols;

    return (sum_of_squares / size) - (sum * sum) / (size * size);
}

__global__ void kernel_scale(const matrix_view matrix, const float factor) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < matrix.rows && col < matrix.cols) {
        *(kernel::matrix::device_get_addr(matrix, row, col)) *= factor;
    }
}

void kernel::matrix::scale(::matrix& mat,
                           const float factor,
                           kernel_stream_t stream) {
    const dim3 threads_per_block(16, 16);
    const dim3 blocks(
        (mat.rows + threads_per_block.x - 1) / threads_per_block.x,
        (mat.cols + threads_per_block.y - 1) / threads_per_block.y);

    kernel_scale<<<blocks, threads_per_block, 0, s_from_stream(stream)>>>(
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
                                  const size_t src_row) {
    const size_t threads_per_block = 256;
    const size_t blocks
        = (src.cols + threads_per_block - 1) / threads_per_block;

    kernel_transfer_row<<<blocks, threads_per_block>>>(dest, dest_row, src,
                                                       src_row);
}

static __global__ void kernel_set_row_vector(const matrix_view data,
                                             size_t data_row,
                                             const const_matrix_view row_vector,
                                             size_t vector_row) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < data.cols) {
        auto val = kernel::matrix::device_get(row_vector, vector_row, col);
        kernel::matrix::device_set(data, data_row, col, val);
    }
}

void kernel::matrix::set_row_vector(::matrix& mat,
                                    const size_t mat_row,
                                    const ::matrix& vec,
                                    const size_t vec_row) {
    const size_t threads_per_block = 256;
    const size_t blocks
        = (mat.cols + threads_per_block - 1) / threads_per_block;

    kernel_set_row_vector<<<blocks, threads_per_block>>>(mat, mat_row, vec,
                                                         vec_row);
}

static __global__ void kernel_get_row_vector(const float* data,
                                             const size_t stride,
                                             const size_t rows,
                                             const size_t cols,
                                             const size_t row,
                                             float* vec,
                                             size_t vec_stride) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols) {
        const float val = kernel::matrix::device_get(data, stride, row, col);
        kernel::matrix::device_set(vec, vec_stride, 0, col, val);
    }
}

::matrix kernel::matrix::get_row_vector(const ::matrix& mat, const size_t row) {
    const size_t threads_per_block = 256;
    const size_t blocks
        = (mat.cols + threads_per_block - 1) / threads_per_block;

    ::matrix result(1, mat.cols);

    kernel_get_row_vector<<<blocks, threads_per_block>>>(
        mat.data, mat.stride, mat.rows, mat.cols, row, result.data,
        result.stride);

    return result;
}

static __global__ void add_row_vector_kernel(const matrix_view data,
                                             size_t data_row,
                                             const const_matrix_view offset,
                                             size_t offset_row) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < data.cols) {
        auto val = kernel::matrix::device_get(offset, offset_row, col);
        kernel::matrix::device_offset_elem(data, data_row, col, val);
    }
}

void kernel::matrix::add_row_vector(::matrix& mat,
                                    const size_t row,
                                    const ::matrix& vec,
                                    size_t vec_row) {
    const size_t threads_per_block = 256;
    const size_t blocks
        = (mat.cols + threads_per_block - 1) / threads_per_block;

    add_row_vector_kernel<<<blocks, threads_per_block>>>(mat, row, vec,
                                                         vec_row);
}

static __global__ void set_horizontal_slice_kernel(
    const size_t start_col,
    const matrix_view data,
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
                                  s_from_stream(stream)>>>(start_col, mat,
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

    matrix_add_matrix<<<blocks, threads_per_block, 0, s_from_stream(stream)>>>(
        mat, offset);
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

    kernel_add_scaled<<<blocks, threads_per_block, 0, s_from_stream(stream)>>>(
        mat, other, factor);
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

    kernel_add_value<<<blocks, threads_per_block, 0, s_from_stream(stream)>>>(
        mat, value);
}

__global__ void kernel_softmax(matrix_view data) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    float max_value = -CUDART_INF_F;

    for (size_t col = 0, cols = data.cols; col < cols; ++col) {
        float val = kernel::matrix::device_get(data, row, col);
        max_value = fmaxf(max_value, val);
    }

    float denominator = 0.0f;
    for (size_t col = 0, cols = data.cols; col < cols; ++col) {
        float val = kernel::matrix::device_get(data, row, col);
        denominator += expf(val - max_value);
    }

    for (size_t col = 0, cols = data.cols; col < cols; ++col) {
        float val = kernel::matrix::device_get(data, row, col);
        float softmaxed = expf(val - max_value) / denominator;
        kernel::matrix::device_set(data, row, col, softmaxed);
    }
}

void kernel::matrix::softmax(::matrix& mat, kernel_stream_t stream) {
    const size_t threads_per_block = 256;
    const size_t blocks
        = (mat.rows + threads_per_block - 1) / threads_per_block;

    kernel_softmax<<<blocks, threads_per_block, 0, s_from_stream(stream)>>>(
        mat);
}

static __global__ void kernel_backprop_softmax(
    const const_matrix_view softmax_output,
    const const_matrix_view output_gradient,
    matrix_view softmax_gradient) {
    const size_t row = blockIdx.x;
    const size_t tid = threadIdx.x;
    const size_t cols = softmax_gradient.cols;

    if (row >= softmax_output.rows)
        return;

    float local_s_dot = 0.0f;
    for (size_t col = tid; col < cols; col += blockDim.x) {
        float s_j = kernel::matrix::device_get(softmax_output, row, col);
        float g_j = kernel::matrix::device_get(output_gradient, row, col);
        local_s_dot += s_j * g_j;
    }

    float s_dot = kernel::matrix::device::block_reduce_sum(local_s_dot);
    __shared__ float shared_s_dot;
    if (tid == 0)
        shared_s_dot = s_dot;
    __syncthreads();

    for (size_t col = tid; col < cols; col += blockDim.x) {
        float s_j = kernel::matrix::device_get(softmax_output, row, col);
        float g_j = kernel::matrix::device_get(output_gradient, row, col);
        kernel::matrix::device_set(softmax_gradient, row, col,
                                   s_j * (g_j - shared_s_dot));
    }
}

::matrix kernel::matrix::backprop_softmax(const ::matrix& output,
                                          const ::matrix& gradient,
                                          kernel_stream_t stream) {
    ::matrix softmax_gradient(gradient.rows, gradient.cols);

    const size_t threads_per_block = 256;
    kernel_backprop_softmax<<<gradient.rows, threads_per_block, 0,
                              s_from_stream(stream)>>>(output, gradient,
                                                       softmax_gradient);

    return softmax_gradient;
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
                                 s_from_stream(stream)>>>(mat, mask_value);
}

matrix kernel::matrix::dot_product(const ::matrix& a, const ::matrix& b) {
    ::matrix result(a.rows, b.cols);
    cublasSdot(cublas_handle_pool, a.rows * a.cols, a.data, 1, b.data, 1,
               result.data);
    return result;
}

kernel::matrix::kernel_stream_t kernel::matrix::create_kernel_stream() {
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    return static_cast<kernel_stream_t>(stream);
}

void kernel::matrix::destroy_kernel_stream(kernel_stream_t stream) {
    cudaStreamDestroy(s_from_stream(stream));
}

matrix kernel::matrix::cross_multiplied(const ::const_matrix_view a,
                                        const ::const_matrix_view b,
                                        kernel_stream_t kernel_stream) {
    ::matrix result = ::matrix(a.rows, b.cols);
    auto handle = cublas_handle_pool.get_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSetStream(handle, s_from_stream(kernel_stream));
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, a.rows, b.cols, a.cols,
                &alpha, a.data, a.stride, b.data, b.stride, &beta,
                result.data_ptr(), result.stride);
    CHECK_ERRORS("cross_multiplied");

    return result;
}

matrix kernel::matrix::cross_t_multiplied(const ::const_matrix_view a,
                                          const ::const_matrix_view b,
                                          kernel_stream_t matmul_handle) {
    ::matrix result
        = ::matrix(a.rows, b.rows);  // kernel::matrix::async_create(a.rows,
                                     // b.rows, matmul_handle.stream);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto handle = cublas_handle_pool.get_handle();

    cublasSetStream(handle, s_from_stream(matmul_handle));
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, a.rows, b.rows, a.cols,
                &alpha, a.data, a.stride, b.data, b.stride, &beta,
                result.data_ptr(), result.stride);
    CHECK_ERRORS("cross_t_multiplied");

    return result;
}

matrix kernel::matrix::t_cross_multiplied(const ::const_matrix_view a,
                                          const ::const_matrix_view b,
                                          kernel_stream_t kernel_stream) {
    ::matrix result = ::matrix(a.cols, b.cols);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto handle = cublas_handle_pool.get_handle();

    cublasSetStream(handle, s_from_stream(kernel_stream));
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, a.cols, b.cols, a.rows,
                &alpha, a.data, a.stride, b.data, b.stride, &beta, result.data,
                result.stride);
    CHECK_ERRORS("t_cross_multiplied");

    return result;
}

__global__ void element_wise_multiply_kernel(matrix_view a_data,
                                             const const_matrix_view b_data) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < a_data.rows && col < a_data.cols) {
        float val_b = kernel::matrix::device_get(b_data, row, col);
        *kernel::matrix::device_get_addr(a_data, row, col) *= val_b;
    }
}

void kernel::matrix::element_wise_multiply(::matrix& a, const ::matrix& b) {
    const dim3 threads_per_block(16, 16);
    const dim3 blocks((a.rows + threads_per_block.x - 1) / threads_per_block.x,
                      (a.cols + threads_per_block.y - 1) / threads_per_block.y);

    element_wise_multiply_kernel<<<blocks, threads_per_block>>>(a, b);
}

__global__ void compare(const const_matrix_view a,
                        const const_matrix_view b,
                        const float epsilon,
                        bool* result) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < a.rows && col < a.cols) {
        float val_a = kernel::matrix::device_get(a, row, col);
        float val_b = kernel::matrix::device_get(b, row, col);

        if (fabsf(val_a - val_b) > epsilon) {
            *result = false;
        }
    }
};

bool kernel::matrix::is_equal(const ::matrix& a,
                              const ::matrix& b,
                              const float epsilon) {
    if (a.rows != b.rows || a.cols != b.cols) {
        return false;
    }

    ensure_reduction_buffer();

    const dim3 threads_per_block(16, 16);
    const dim3 blocks((a.rows + threads_per_block.x - 1) / threads_per_block.x,
                      (a.cols + threads_per_block.y - 1) / threads_per_block.y);

    const bool initial_value = true;

    cudaMemcpy(d_reduction_bool_result, &initial_value, sizeof(bool),
               cudaMemcpyHostToDevice);
    compare<<<threads_per_block, blocks>>>(a, b, epsilon,
                                           d_reduction_bool_result);

    bool h_result;
    cudaMemcpy(&h_result, d_reduction_bool_result, sizeof(bool),
               cudaMemcpyDeviceToHost);

    return h_result;
}
