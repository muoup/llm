#include "kernels/optimizer.hpp"
#include "matrix_device_kernels.cuh"
#include "matrix_kernels.hpp"

#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <device_atomic_functions.h>

#include <float.h>

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

void cleanup_cublas();

struct cublas_handle {
    cublasHandle_t handle;
    bool initialized = false;

    // The linker does not seem to like this function having undefined behaviour
    // checks.
    __attribute__((no_sanitize("address"), no_sanitize("undefined")))
    cublas_handle() {}

    void initialize() {
        auto status = cublasCreate(&handle);

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::puts("Failed to create cuBLAS handle");
            std::printf("Error Code: %d\n", status);
            std::fflush(stdout);
            std::exit(1);
        }

        initialized = true;
    }

    ~cublas_handle() { cublasDestroy(handle); }
    operator cublasHandle_t() {
        if (!initialized) {
            initialize();
        }

        return handle;
    }
};

static cublas_handle handle;

__global__ void test_print() {
    printf("Hello from CUDA kernel!\n");
}

void kernel::matrix::test_print() {
    ::test_print<<<1, 1>>>();
}

void kernel::matrix::check_errors(const char* step) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("Failure during: %s\n", step);
        std::printf("CUDA error: %s\n", cudaGetErrorString(err));
        std::abort();
    }
}

float* kernel::matrix::allocate_buffer(const size_t size) {
    float* data;
    cudaMalloc(&data, size);
    cudaMemset(data, 0, size);
    kernel::matrix::check_errors("Allocating matrix buffer");
    kernel::optimizer::wait_for_operations();
    return data;
}

void kernel::matrix::free_buffer(float* data) {
    cudaFree(data);
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
    float* storage;
    cudaMalloc(&storage, sizeof(float));
    global_get<<<1, 1>>>(matrix, row, col, storage);
    float value;
    cudaMemcpy(&value, storage, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(storage);

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
    curandGenerateUniform(global_curand_generator, matrix.data,
                          matrix.buffer_size() / sizeof(float));
    const auto range = max - min;

    matrix.add(-0.5f);
    kernel::optimizer::wait_for_operations();

    matrix.scale(range);
    kernel::optimizer::wait_for_operations();

    matrix.add(min);
    kernel::optimizer::wait_for_operations();
}

matrix kernel::matrix::clone(const ::matrix& other) {
    ::matrix result(other.rows, other.cols);
    cudaMemcpy(result.data, other.data, other.buffer_size(),
               cudaMemcpyDeviceToDevice);
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
        data[row + col * stride] = value;
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

struct reduction_mutex {
    int lock;
    float result;
};

template <__device__ float (*reducer)(float, float)>
static __global__ void matrix_reduce(const const_matrix_view data,
                                     float acc,
                                     reduction_mutex* result_mutex) {
    float partial_reduction = acc;
    std::uint64_t col = blockIdx.x;

    if (col >= data.cols) {
        return;
    }

    for (std::uint64_t row = 0; row < data.rows; ++row) {
        float val = kernel::matrix::device_get(data, row, col);
        partial_reduction = reducer(partial_reduction, val);
    }

    volatile int* lock = &result_mutex->lock;
    while (atomicExch((int*)lock, 1) == 1)
        ;

    result_mutex->result = reducer(result_mutex->result, partial_reduction);

    // Release lock
    atomicExch((int*)lock, 0);
};

template <__device__ float (*reducer)(float, float)>
float general_reduce(const ::matrix& mat, float acc) {
    reduction_mutex host_reference = { .lock = 0, .result = acc };
    reduction_mutex* d_result;
    cudaMalloc(&d_result, sizeof(reduction_mutex));
    cudaMemcpy(d_result, &host_reference, sizeof(reduction_mutex),
               cudaMemcpyHostToDevice);

    const size_t blocks = mat.cols;
    const size_t threads_per_block = 1;

    (matrix_reduce<reducer>)<<<blocks, threads_per_block>>>(mat, acc, d_result);
    cudaDeviceSynchronize();

    cudaMemcpy(&host_reference, d_result, sizeof(reduction_mutex),
               cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return host_reference.result;
}

__device__ float kernel_fadd(float a, float b) {
    return a + b;
}

float kernel::matrix::sum(const ::matrix& mat) {
    return general_reduce<kernel_fadd>(mat, 0.0f);
}

__device__ float square_sum(float a, float b) {
    return a + b * b;
}

float kernel::matrix::sum_of_squares(const ::matrix& mat) {
    return general_reduce<square_sum>(mat, 0.0f);
}

__device__ float kernel_fmaxf(float a, float b) {
    return a > b ? a : b;
}

float kernel::matrix::max(const ::matrix& mat) {
    return general_reduce<kernel_fmaxf>(mat, FLT_MIN);
}

__device__ float kernel_fminf(float a, float b) {
    return a < b ? a : b;
}

float kernel::matrix::min(const ::matrix& mat) {
    return general_reduce<kernel_fminf>(mat, FLT_MAX);
}

__device__ float kernel_fabsf(float a) {
    return a < 0 ? -a : a;
}

__device__ float individual_absmax(float a, float b) {
    return kernel_fmaxf(kernel_fabsf(a), kernel_fabsf(b));
}

float kernel::matrix::absmax(const ::matrix& mat) {
    return general_reduce<individual_absmax>(mat, 0.0f);
}

float kernel::matrix::variance(const ::matrix& mat) {
    float sum = kernel::matrix::sum(mat);
    float sum_of_squares = kernel::matrix::sum_of_squares(mat);

    return (sum_of_squares / (mat.rows * mat.cols))
           - (sum * sum) / (mat.rows * mat.cols * mat.rows * mat.cols);
}

__global__ void kernel_scale(float* data,
                             const size_t stride,
                             const size_t rows,
                             const size_t cols,
                             const float factor) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        *(kernel::matrix::device_get_addr(data, stride, row, col)) *= factor;
    }
}

void kernel::matrix::scale(::matrix& mat, const float factor) {
    dim3 threads_per_block(16, 16);
    dim3 blocks((mat.rows + threads_per_block.x - 1) / threads_per_block.x,
                (mat.cols + threads_per_block.y - 1) / threads_per_block.y);

    kernel_scale<<<blocks, threads_per_block>>>(mat.data, mat.stride, mat.rows,
                                                mat.cols, factor);
}

static __global__ void kernel_transfer_row(const matrix_view dest,
                                           size_t src_row,
                                           const const_matrix_view src,
                                           size_t dest_row) {
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

static __global__ void set_horizontal_slice_kernel(float* data,
                                                   const size_t stride,
                                                   const size_t rows,
                                                   const size_t start_col,
                                                   const float* slice,
                                                   const size_t slice_cols) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // We can assume here that slice_stride == data_stride since they both have
    // the same number of rows

    if (row < rows) {
        for (size_t col = 0; col < slice_cols; ++col) {
            auto val = kernel::matrix::device_get(slice, stride, row, col);
            kernel::matrix::device_set(data, stride, row, start_col + col, val);
        }
    }
}

void kernel::matrix::set_horizontal_slice(::matrix& mat,
                                          const size_t start_col,
                                          const ::matrix& slice) {
    const size_t threads_per_block = 256;
    const size_t blocks
        = (mat.rows + threads_per_block - 1) / threads_per_block;

    set_horizontal_slice_kernel<<<blocks, threads_per_block>>>(
        mat.data, mat.stride, mat.rows, start_col, slice.data, slice.cols);
}

static __global__ void get_horizontal_slice_kernel(const const_matrix_view data,
                                                   matrix_view slice,
                                                   const size_t start_col) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < slice.cols && row < slice.rows) {
        auto val = kernel::matrix::device_get(data, row, start_col + col);
        kernel::matrix::device_set(slice, row, col, val);
    }
}

::matrix kernel::matrix::get_horizontal_slice(const ::matrix& mat,
                                              const size_t start_col,
                                              const size_t slice_cols) {
    ::matrix slice(mat.rows, slice_cols);

    dim3 threads_per_block(16, 16);
    dim3 blocks(((mat.rows + threads_per_block.x - 1) / threads_per_block.x),
                ((slice_cols + threads_per_block.y - 1) / threads_per_block.y));

    get_horizontal_slice_kernel<<<blocks, threads_per_block>>>(mat, slice,
                                                               start_col);
    return slice;
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

void kernel::matrix::add(::matrix& mat, const ::matrix& offset) {
    const dim3 threads_per_block(16, 16);
    const dim3 blocks(
        (mat.rows + threads_per_block.x - 1) / threads_per_block.x,
        (mat.cols + threads_per_block.y - 1) / threads_per_block.y);

    matrix_add_matrix<<<blocks, threads_per_block>>>(mat, offset);
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
                                const float factor) {
    dim3 threads_per_block(16, 16);
    dim3 blocks((mat.rows + threads_per_block.x - 1) / threads_per_block.x,
                (mat.cols + threads_per_block.y - 1) / threads_per_block.y);

    kernel_add_scaled<<<blocks, threads_per_block>>>(mat, other, factor);
}

static __global__ void kernel_add_value(const matrix_view data,
                                        const float value) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < data.rows && col < data.cols) {
        kernel::matrix::device_offset_elem(data, row, col, value);
    }
}

void kernel::matrix::add(::matrix& mat, float value) {
    const dim3 threads_per_block(16, 16);
    const dim3 blocks(
        (mat.rows + threads_per_block.x - 1) / threads_per_block.x,
        (mat.cols + threads_per_block.y - 1) / threads_per_block.y);

    kernel_add_value<<<blocks, threads_per_block>>>(mat, value);
}

__global__ void kernel_softmax(const matrix_view data) {
    const size_t row = blockIdx.x;

    if (row < data.rows) {
        // Find max value for numerical stability
        float max_val = kernel::matrix::device_get(
            data, row, 0);  // data[row + 0 * stride];
        for (size_t j = 1; j < data.cols; ++j) {
            const float val = kernel::matrix::device_get(
                data, row, j);  // data[row + j * stride];
            if (val > max_val) {
                max_val = val;
            }
        }

        // Compute exponentials and sum
        float sum_exp = 0.0f;
        for (size_t j = 0; j < data.cols; ++j) {
            const float exp_val
                = expf(kernel::matrix::device_get(data, row, j) - max_val);
            kernel::matrix::device_set(data, row, j, exp_val);
            sum_exp += exp_val;
        }

        // Normalize to get probabilities
        for (size_t j = 0; j < data.cols; ++j) {
            const float val = kernel::matrix::device_get(data, row, j);  //
            kernel::matrix::device_set(data, row, j, val / sum_exp);
        }
    }
}

void kernel::matrix::softmax(::matrix& mat) {
    const size_t threads_per_block = 1;
    const size_t blocks = mat.rows;

    kernel_softmax<<<blocks, threads_per_block>>>(mat);
}

static __global__ void kernel_backprop_softmax(
    const const_matrix_view softmax_output,
    const const_matrix_view gradient,
    matrix_view softmax_gradient) {
    const size_t row = blockIdx.x;

    if (row < softmax_output.rows) {
        float s_dot = 0.0f;

        for (size_t col = 0; col < softmax_gradient.cols; ++col) {
            float s_j = kernel::matrix::device_get(softmax_output, row, col);
            float g_j = kernel::matrix::device_get(gradient, row, col);
            s_dot += s_j * g_j;
        }

        for (size_t col = 0; col < softmax_gradient.cols; ++col) {
            float s_j = kernel::matrix::device_get(softmax_output, row, col);
            float g_j = kernel::matrix::device_get(gradient, row, col);
            kernel::matrix::device_set(softmax_gradient, row, col,
                                       s_j * (g_j - s_dot));
        }
    }
}

::matrix kernel::matrix::backprop_softmax(const ::matrix& output,
                                          const ::matrix& gradient) {
    ::matrix softmax_gradient({ gradient.rows, gradient.cols });

    const size_t threads_per_block = 1;
    const size_t blocks = gradient.rows;

    kernel_backprop_softmax<<<blocks, threads_per_block>>>(output, gradient,
                                                           softmax_gradient);

    return softmax_gradient;
}

void __global__ kernel_mask_upper_triangle(const matrix_view data,
                                           const float mask_value) {
    const size_t row = data.rows;
    const size_t col = data.cols;

    if (col > row && (row < data.rows || col < data.cols)) {
        if (col > row) {
            kernel::matrix::device_set(data, row, col, mask_value);
        }
    }
}

void kernel::matrix::mask_upper_triangle(::matrix& mat,
                                         const float mask_value) {
    const size_t threads_per_block = 256;
    const dim3 blocks((mat.rows + threads_per_block - 1) / threads_per_block,
                      (mat.cols + threads_per_block - 1) / threads_per_block);

    kernel_mask_upper_triangle<<<blocks, threads_per_block>>>(mat, mask_value);
}

__global__ void matrixDotProductKernel(float* A,
                                       float* B,
                                       float* C,
                                       int stride_a,
                                       int stride_b,
                                       int stride_c,
                                       int M,
                                       int N,
                                       int K) {
    // Calculate global row and column indices for the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row + i * stride_a] * B[i + col * stride_b];
        }
        C[row + col * stride_c] = sum;
    }
}

matrix kernel::matrix::dot_product(const ::matrix& a, const ::matrix& b) {
    ::matrix result(a.rows, b.cols);

    dim3 blockSize(16, 16);
    dim3 gridSize((b.cols + blockSize.x - 1) / blockSize.x,
                  (a.rows + blockSize.y - 1) / blockSize.y);

    matrixDotProductKernel<<<gridSize, blockSize>>>(
        a.data, b.data, result.data, a.stride, b.stride, result.stride, a.rows,
        b.cols, a.cols);

    return result;
}

matrix kernel::matrix::cross_multiplied(const ::matrix& a, const ::matrix& b) {
    ::matrix result = ::matrix(a.rows, b.cols);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, a.rows, b.cols, a.cols,
                &alpha, a.data_ptr(), a.stride, b.data_ptr(), b.stride, &beta,
                result.data_ptr(), result.stride);
    kernel::matrix::check_errors("cross_multiplied");

    return result;
}

matrix kernel::matrix::cross_t_multiplied(const ::matrix& a,
                                          const ::matrix& b) {
    ::matrix result = ::matrix(a.rows, b.rows);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, a.rows, b.rows, a.cols,
                &alpha, a.data_ptr(), a.stride, b.data_ptr(), b.stride, &beta,
                result.data_ptr(), result.stride);
    kernel::matrix::check_errors("cross_t_multiplied");

    return result;
}

matrix kernel::matrix::t_cross_multiplied(const ::matrix& a,
                                          const ::matrix& b) {
    ::matrix result = ::matrix(a.cols, b.cols);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, a.cols, b.cols, a.rows,
                &alpha, a.data_ptr(), a.stride, b.data_ptr(), b.stride, &beta,
                result.data_ptr(), result.stride);
    kernel::matrix::check_errors("t_cross_multiplied");

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

void kernel::matrix::element_wise_multiply(::matrix& a, const ::matrix& b) {
    const dim3 threads_per_block(16, 16);
    const dim3 blocks((a.rows + threads_per_block.x - 1) / threads_per_block.x,
                      (a.cols + threads_per_block.y - 1) / threads_per_block.y);

    element_wise_multiply_kernel<<<blocks, threads_per_block>>>(a, b);
}

__global__ void compare(const float* a,
                        const float* b,
                        const size_t stride_a,
                        const size_t stride_b,
                        const size_t rows,
                        const size_t cols,
                        const float epsilon,
                        bool* result) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_size = rows * cols;

    if (idx < total_size) {
        const size_t row = idx % rows;
        const size_t col = idx / rows;
        float val_a = a[row + col * stride_a];
        float val_b = b[row + col * stride_b];
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

    bool* d_result;
    cudaMalloc(&d_result, sizeof(bool));
    cudaMemset(d_result, 1, sizeof(bool));

    compare<<<(a.rows * a.cols + 255) / 256, 256>>>(
        a.data, b.data, a.stride, b.stride, a.rows, a.cols, epsilon, d_result);

    bool h_result;
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return h_result;
}
