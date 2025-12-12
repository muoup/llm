#include "matrix_device_kernels.cuh"
#include "matrix_kernels.hpp"

#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <device_atomic_functions.h>

#include <float.h>

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

float* kernel::matrix::allocate_buffer(const size_t size) {
    float* data;
    cudaMalloc(&data, size * sizeof(float));
    cudaMemset(data, 0, size * sizeof(float));
    return data;
}

void kernel::matrix::free_buffer(float* data) {
    cudaFree(data);
}

static __global__ void global_set(float* data,
                                  const size_t stride,
                                  const size_t row,
                                  const size_t col,
                                  const float value) {
    kernel::matrix::device_set(data, stride, row, col, value);
}

void kernel::matrix::set(::matrix& matrix,
                         const size_t row,
                         const size_t col,
                         const float value) {
    global_set<<<1, 1>>>(matrix.data, matrix.stride, row, col, value);
}

static __global__ void global_get(const float* data,
                                  const size_t stride,
                                  const size_t row,
                                  const size_t col,
                                  float* result) {
    *result = kernel::matrix::device_get(data, stride, row, col);
}

float kernel::matrix::get(const ::matrix& matrix,
                          const size_t row,
                          const size_t col) {
    float* storage;
    cudaMalloc(&storage, sizeof(float));
    global_get<<<1, 1>>>(matrix.data, matrix.stride, row, col, storage);
    float value;
    cudaMemcpy(&value, storage, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(storage);

    return value;
}

void kernel::matrix::load_into(::matrix& matrix, const float* host_data) {
    cudaMemcpy(matrix.data, host_data, matrix.buffer_size() * sizeof(float),
               cudaMemcpyHostToDevice);
}

void kernel::matrix::store_from(const ::matrix& matrix, float* host_data) {
    cudaMemcpy(host_data, matrix.data, matrix.buffer_size() * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void kernel::matrix::randomize(::matrix& matrix,
                               const float min,
                               const float max) {
    static curandGenerator_t gen;
    static bool generator_initialized = false;

    if (!generator_initialized) {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
        generator_initialized = true;
    }

    curandGenerateUniform(gen, matrix.data,
                          matrix.buffer_size() / sizeof(float));
    const auto range = max - min;

    matrix.add(-0.5f);
    matrix.scale(range);
    matrix.add(min);
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

// 'matrix_reduce' is not really used in any hot paths as of right now, so the
// implementation will be simplified down to 1 thread per row, and some locking
// mechanism to map the partial reudction to the result. A more optimized
// version could use shared memory and parallel reduction within blocks, but
// that is more complex to implement.
template <__device__ float (*reducer)(float, float)>
static __global__ void matrix_reduce(const float* data,
                                     const std::uint64_t stride,
                                     const std::uint64_t rows,
                                     const std::uint64_t cols,
                                     float acc,
                                     reduction_mutex* result_mutex) {
    float partial_reduction = acc;
    std::uint64_t col = blockIdx.x;

    if (col >= cols) {
        return;
    }

    for (std::uint64_t row = 0; row < rows; ++row) {
        float val = kernel::matrix::device_get(data, stride, row, col);
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
    cudaMemcpy(&d_result, &host_reference, sizeof(reduction_mutex),
               cudaMemcpyHostToDevice);

    const size_t blocks = mat.cols;
    const size_t threads_per_block = 1;

    (matrix_reduce<reducer>)<<<blocks, threads_per_block>>>(
        mat.data, mat.stride, mat.rows, mat.cols, acc, d_result);

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

__device__ float kernel_fmaxf(float a, float b) {
    return a > b ? a : b;
}

float kernel::matrix::max(const ::matrix& mat) {
    return general_reduce<kernel_fadd>(mat, FLT_MIN);
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

__device__ float square_sum(float a, float b) {
    return a + b * b;
}

float kernel::matrix::variance(const ::matrix& mat) {
    float sum = kernel::matrix::sum(mat);
    float sum_of_squares = general_reduce<square_sum>(mat, 0.0f);

    return (sum_of_squares / (mat.rows * mat.cols))
           - (sum * sum) / (mat.rows * mat.cols * mat.rows * mat.cols);
}

__global__ void kernel_scale(float* data,
                             const size_t stride,
                             const size_t rows,
                             const size_t cols,
                             const float factor) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_size = rows * cols;

    if (idx < total_size) {
        const size_t row = idx % rows;
        const size_t col = idx / rows;
        data[row + col * stride] = data[row + col * stride] * factor;
    }
}

void kernel::matrix::scale(::matrix& mat, const float factor) {
    const size_t total_size = mat.rows * mat.cols;
    const size_t threads_per_block = 256;
    const size_t blocks
        = (total_size + threads_per_block - 1) / threads_per_block;

    kernel_scale<<<blocks, threads_per_block>>>(mat.data, mat.stride, mat.rows,
                                                mat.cols, factor);
}

static __global__ void kernel_set_row_vector(float* data,
                                             const size_t stride,
                                             const size_t rows,
                                             const size_t cols,
                                             const size_t row,
                                             const float* vec) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols) {
        data[row + col * stride] = vec[col];
    }
}

void kernel::matrix::set_row_vector(::matrix& mat,
                                    const size_t row,
                                    const ::matrix& vec) {
    const size_t threads_per_block = 256;
    const size_t blocks
        = (mat.cols + threads_per_block - 1) / threads_per_block;

    kernel_set_row_vector<<<blocks, threads_per_block>>>(
        mat.data, mat.stride, mat.rows, mat.cols, row, vec.data);
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
        vec[col * vec_stride] = data[row + col * stride];
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

static __global__ void add_row_vector_kernel(float* data,
                                             const size_t stride,
                                             const size_t rows,
                                             const size_t cols,
                                             const size_t row,
                                             const float* vec,
                                             size_t vec_stride) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols) {
        auto val = kernel::matrix::device_get(data, stride, row, col);
        kernel::matrix::device_set(data, stride, row, col,
                                   val + vec[col * vec_stride]);
    }
}

void kernel::matrix::add_row_vector(::matrix& mat,
                                    const size_t row,
                                    const ::matrix& vec) {
    const size_t threads_per_block = 256;
    const size_t blocks
        = (mat.cols + threads_per_block - 1) / threads_per_block;

    add_row_vector_kernel<<<blocks, threads_per_block>>>(
        mat.data, mat.stride, mat.rows, mat.cols, row, vec.data, vec.stride);
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

static __global__ void get_horizontal_slice_kernel(const float* data,
                                                   const size_t stride,
                                                   const size_t rows,
                                                   const size_t start_col,
                                                   float* slice,
                                                   const size_t slice_cols) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        for (size_t col = 0; col < slice_cols; ++col) {
            slice[row + col * stride] = data[row + (start_col + col) * stride];
        }
    }
}

::matrix kernel::matrix::get_horizontal_slice(const ::matrix& mat,
                                              const size_t start_col,
                                              const size_t slice_cols) {
    ::matrix slice(mat.rows, slice_cols);

    const size_t threads_per_block = 256;
    const size_t blocks
        = (mat.rows + threads_per_block - 1) / threads_per_block;

    get_horizontal_slice_kernel<<<blocks, threads_per_block>>>(
        mat.data, mat.stride, mat.rows, start_col, slice.data, slice.cols);

    return slice;
}

static __global__ void matrix_add_matrix(float* data,
                                         const size_t stride,
                                         const size_t rows,
                                         const size_t cols,
                                         const float* offset_data,
                                         const size_t offset_stride) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_size = rows * cols;

    if (idx < total_size) {
        const size_t row = idx % rows;
        const size_t col = idx / rows;
        data[row + col * stride] += offset_data[row + col * offset_stride];
    }
}

void kernel::matrix::add(::matrix& mat, const ::matrix& offset) {
    const size_t total_size = mat.rows * mat.cols;
    const size_t threads_per_block = 256;
    const size_t blocks
        = (total_size + threads_per_block - 1) / threads_per_block;

    matrix_add_matrix<<<blocks, threads_per_block>>>(
        mat.data, mat.stride, mat.rows, mat.cols, offset.data, offset.stride);
}

static __global__ void matrix_add_value(float* data,
                                        const size_t stride,
                                        const size_t rows,
                                        const size_t cols,
                                        const float* other_data,
                                        const size_t other_stride,
                                        const float factor) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_size = rows * cols;

    if (idx < total_size) {
        const size_t row = idx % rows;
        const size_t col = idx / rows;
        data[row + col * stride]
            += other_data[row + col * other_stride] * factor;
    }
}

void kernel::matrix::add_scaled(::matrix& mat,
                                const ::matrix& other,
                                const float factor) {
    const size_t total_size = mat.rows * mat.cols;
    const size_t threads_per_block = 256;
    const size_t blocks
        = (total_size + threads_per_block - 1) / threads_per_block;

    matrix_add_value<<<blocks, threads_per_block>>>(
        mat.data, mat.stride, mat.rows, mat.cols, other.data, other.stride,
        factor);
}

static __global__ void matrix_add_value(float* data,
                                        const size_t stride,
                                        const size_t rows,
                                        const size_t cols,
                                        const float value) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_size = rows * cols;

    if (idx < total_size) {
        const size_t row = idx % rows;
        const size_t col = idx / rows;
        data[row + col * stride] += value;
    }
}

void kernel::matrix::add(::matrix& mat, float value) {
    const size_t total_size = mat.rows * mat.cols;
    const size_t threads_per_block = 256;
    const size_t blocks
        = (total_size + threads_per_block - 1) / threads_per_block;

    matrix_add_value<<<blocks, threads_per_block>>>(mat.data, mat.stride,
                                                    mat.rows, mat.cols, value);
}

__global__ void kernel_softmax(float* data,
                               size_t stride,
                               size_t rows,
                               size_t cols) {
    const size_t row = blockIdx.x;

    if (row < rows) {
        // Find max value for numerical stability
        float max_val = data[row + 0 * stride];
        for (size_t j = 1; j < cols; ++j) {
            float val = data[row + j * stride];
            if (val > max_val) {
                max_val = val;
            }
        }

        // Compute exponentials and sum
        float sum_exp = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            float exp_val = expf(data[row + j * stride] - max_val);
            data[row + j * stride] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize to get probabilities
        for (size_t j = 0; j < cols; ++j) {
            float val = data[row + j * stride];
            data[row + j * stride] = val / sum_exp;
        }
    }
}

void kernel::matrix::softmax(::matrix& mat) {
    const size_t threads_per_block = 1;
    const size_t blocks = mat.rows;

    kernel_softmax<<<blocks, threads_per_block>>>(mat.data, mat.stride,
                                                  mat.rows, mat.cols);
}

static __global__ void kernel_backprop_softmax(const float* softmax_output,
                                               const float* gradient,
                                               size_t stride,
                                               size_t rows,
                                               size_t cols,
                                               float* softmax_gradient) {
    const size_t row = blockIdx.x;

    if (row < rows) {
        float s_dot = 0.0f;

        for (size_t c = 0; c < cols; ++c) {
            float s_j
                = kernel::matrix::device_get(softmax_output, stride, row, c);
            float g_j = kernel::matrix::device_get(gradient, stride, row, c);
            s_dot += s_j * g_j;
        }

        for (size_t c = 0; c < cols; ++c) {
            float s_j
                = kernel::matrix::device_get(softmax_output, stride, row, c);
            float g_j = kernel::matrix::device_get(gradient, stride, row, c);
            softmax_gradient[row + c * stride] = s_j * (g_j - s_dot);
        }
    }
}

::matrix kernel::matrix::backprop_softmax(const ::matrix& output,
                                          const ::matrix& gradient) {
    ::matrix softmax_gradient({ gradient.rows, gradient.cols });

    const size_t threads_per_block = 1;
    const size_t blocks = gradient.rows;

    kernel_backprop_softmax<<<blocks, threads_per_block>>>(
        output.data, gradient.data, output.stride, output.rows, output.cols,
        softmax_gradient.data);

    return softmax_gradient;
}

void __global__ kernel_mask_upper_triangular(float* data,
                                             const size_t stride,
                                             const size_t rows,
                                             const size_t cols,
                                             const float mask_value) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_size = rows * cols;

    if (idx < total_size) {
        const size_t row = idx % rows;
        const size_t col = idx / rows;
        if (col > row) {
            data[row + col * stride] = mask_value;
        }
    }
}

void kernel::matrix::mask_upper_triangular(::matrix& mat,
                                           const float mask_value) {
    const size_t total_size = mat.rows * mat.cols;
    const size_t threads_per_block = 256;
    const size_t blocks
        = (total_size + threads_per_block - 1) / threads_per_block;

    kernel_mask_upper_triangular<<<blocks, threads_per_block>>>(
        mat.data, mat.stride, mat.rows, mat.cols, mask_value);
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

    return result;
}

__global__ void element_wise_multiply_kernel(float* a_data,
                                             const size_t a_stride,
                                             const float* b_data,
                                             const size_t b_stride,
                                             const size_t rows,
                                             const size_t cols) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_size = rows * cols;

    if (idx < total_size) {
        const size_t row = idx % rows;
        const size_t col = idx / rows;
        float val_a = a_data[row + col * a_stride];
        float val_b = b_data[row + col * b_stride];
        a_data[row + col * a_stride] = val_a * val_b;
    }
}

void kernel::matrix::element_wise_multiply(::matrix& a, const ::matrix& b) {
    const size_t total_size = a.rows * a.cols;
    const size_t threads_per_block = 256;
    const size_t blocks
        = (total_size + threads_per_block - 1) / threads_per_block;

    element_wise_multiply_kernel<<<blocks, threads_per_block>>>(
        a.data, a.stride, b.data, b.stride, a.rows, a.cols);
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
