#include "matrix.hpp"

#include <cublas_api.h>
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <curand.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>

struct cublas_handle {
    cublasHandle_t handle;

    cublas_handle() {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Failed to create cuBLAS handle" << std::endl;
            std::abort();
        }
    }

    ~cublas_handle() { cublasDestroy(handle); }

    operator cublasHandle_t() const { return handle; }
};

static cublas_handle handle;

static constexpr size_t calculate_stride(const size_t i) {
    // The Stride is Equal to the Least Multiple of (256 / sizeof(float)) Equal
    // to or Greater Than i
    constexpr size_t alignment = 256 / sizeof(float);

    return ((i + alignment - 1) / alignment) * alignment;
}

matrix::matrix(const size_t rows, const size_t cols)
    : rows(rows), cols(cols), stride(calculate_stride(rows)) {
    const auto buffer_size = this->buffer_size();

    cudaMalloc(&data, buffer_size);
    cudaMemset(data, 0, buffer_size);
}

matrix::~matrix() {
    cudaFree(data);
}

size_t matrix::buffer_size() const {
    return stride * cols * sizeof(float);
}

void matrix::verify_bounds(const size_t row, const size_t col) const {
    MATRIX_ASSERT(row < rows && col < cols,
                  "Index out of bounds: (%d, %d) for matrix of size (%d x %d)",
                  row, col, rows, cols);
}


void matrix::randomize(const float min, const float max) {
    static curandGenerator_t gen;
    static bool generator_initialized = false;
    
    if (!generator_initialized) {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
        generator_initialized = true;
    }
    
    curandGenerateUniform(gen, data, buffer_size());

    const auto range = max - min;
    this->map([=](const float val) { return min + val * range; });
}

[[nodiscard]] float matrix::get(const size_t row, const size_t col) const {
    verify_bounds(row, col);
    float value;
    cudaMemcpy(&value, data + row + col * stride, sizeof(float),
               cudaMemcpyDeviceToHost);
    return value;
}

void matrix::set(const size_t row, const size_t col, const float value) {
    verify_bounds(row, col);
    cudaMemcpy(data + row + col * stride, &value, sizeof(float),
               cudaMemcpyHostToDevice);
}

__device__ void kernel_set(float* data,
                           const size_t stride,
                           const size_t rows,
                           const size_t cols,
                           const size_t row,
                           const size_t col,
                           const float value) {
    data[row + col * stride] = value;
}

__device__ float kernel_get(const float* data,
                            const size_t stride,
                            const size_t rows,
                            const size_t cols,
                            const size_t row,
                            const size_t col) {
    return data[row + col * stride];
}

__device__ void kernel_map_single(float* data,
                                  const size_t stride,
                                  const size_t rows,
                                  const size_t cols,
                                  const size_t row,
                                  const size_t col,
                                  float (*func)(float)) {
    kernel_set(data, stride, rows, cols, row, col,
               func(kernel_get(data, stride, rows, cols, row, col)));
}

matrix matrix::clone() const {
    matrix copy{ this->rows, this->cols };
    cudaMemcpy(copy.data_ptr(), this->data_ptr(), this->buffer_size(),
               cudaMemcpyDeviceToDevice);
    return copy;
}

float matrix::sum() const {
    return this->reduce(0, [](float acc, float b) { return acc + b; });
}

float matrix::max() const {
    return this->reduce(0, [](float a, float b) { return std::max(a, b); });
}

float matrix::min() const {
    return this->reduce(0, [](float a, float b) { return std::min(a, b); });
}

float matrix::absmax() const {
    return this->reduce(
        0, [](float a, float b) { return std::max(std::abs(a), std::abs(b)); });
}

float matrix::variance() const {
    float sum = this->reduce(0, [](float acc, float b) { return acc + b; });
    float sum_sq
        = this->reduce(0, [](float acc, float b) { return acc + b * b; });

    return (sum_sq / size()) - (sum / size()) * (sum / size());
}

float matrix::stddev() const {
    return std::sqrt(variance());
}

std::string matrix::header() const {
    std::stringstream ss;
    ss << "Matrix(" << rows << "x" << cols << "):\n";

    return ss.str();
}

std::string matrix::to_string(std::uint8_t precision) const {
    std::stringstream ss;

    ss << header();
    ss << std::fixed << std::setprecision(precision);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            ss << get(i, j) << ' ';
        }
        ss << '\n';
    }

    return ss.str();
}

__global__ void kernel_map(float* data,
                           const size_t stride,
                           const size_t rows,
                           const size_t cols,
                           float (*func)(float)) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_size = rows * cols;

    if (idx < total_size) {
        const size_t row = idx % rows;
        const size_t col = idx / rows;
        data[row + col * stride] = func(data[row + col * stride]);
    }
}

matrix& matrix::softmax() {
    const auto kernel_softmax =
        [] __global__(float* data, size_t stride, size_t rows,
                      size_t cols) static {
            const size_t row = blockIdx.x;

            if (row < rows) {
                float max_val = data[row + 0 * stride];
                for (size_t j = 1; j < cols; ++j) {
                    float val = data[row + j * stride];
                    if (val > max_val) {
                        max_val = val;
                    }
                }

                float sum_exp = 0.0f;
                for (size_t j = 0; j < cols; ++j) {
                    float exp_val = std::expf(data[row + j * stride] - max_val);
                    data[row + j * stride] = exp_val;
                    sum_exp += exp_val;
                }

                for (size_t j = 0; j < cols; ++j) {
                    float val = kernel_get(data, stride, rows, cols, row, j);
                    kernel_set(data, stride, rows, cols, row, j, val / sum_exp);
                }
            }
        };

    kernel_softmax<<<this->stride, 256>>>(this->data_ptr(), this->stride,
                                          this->rows, this->cols);
    kernel_map<<<this->stride, 256>>>(this->data_ptr(), this->stride,
                                      this->rows, this->cols, std::exp);

    for (size_t i = 0; i < rows; ++i) {
        float row_sum = 0;
        for (size_t j = 0; j < cols; ++j) {
            row_sum += get(i, j);
        }
        row_sum += 1e-8f;

        for (size_t j = 0; j < cols; ++j) {
            set(i, j, get(i, j) / row_sum);
        }
    }

    return *this;
}

matrix& matrix::mask_upper_triangular(const float mask_value) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = i + 1; j < cols; ++j) {
            set(i, j, mask_value);
        }
    }
    return *this;
}

float matrix::dot_product(const matrix& other) const {
    MATRIX_ASSERT(this->rows == other.rows && this->cols == other.cols,
                  "Matrix dimensions do not match for dot product");

    float result = 0.0f;
    for (size_t row = 0; row < this->rows; ++row) {
        for (size_t col = 0; col < this->cols; ++col) {
            result += get(row, col) + other.get(row, col);
        }
    }

    return result;
}

matrix matrix::cross_multiplied(const matrix& other) const {
    MATRIX_ASSERT(this->cols == other.rows,
                  "Matrix dimensions do not match for cross multiplication");

    matrix result{ this->rows, other.cols };

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->rows, other.cols,
                this->cols, &alpha, this->data_ptr(), this->stride,
                other.data_ptr(), other.stride, &beta, result.data_ptr(),
                result.stride);

    return result;
}

matrix matrix::cross_t_multiplied(const matrix& other) const {
    MATRIX_ASSERT(this->cols == other.cols,
                  "Matrix dimensions do not match for cross post-transposed "
                  "multiplication");

    matrix result{ this->rows, other.rows };
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, this->rows, other.cols,
                this->cols, &alpha, this->data_ptr(), this->stride,
                other.data_ptr(), other.stride, &beta, result.data_ptr(),
                result.stride);

    return result;
}

matrix matrix::t_cross_multiplied(const matrix& other) const {
    MATRIX_ASSERT(this->rows == other.rows,
                  "Matrix dimensions do not match for pre-transposed cross "
                  "multiplication");

    matrix result = matrix(this->cols, other.cols);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, this->rows, other.cols,
                this->cols, &alpha, this->data_ptr(), this->stride,
                other.data_ptr(), other.stride, &beta, result.data_ptr(),
                result.stride);

    return result;
}

bool matrix::equals(const matrix& other, const float epsilon) const {
    const auto compare
        = [] __global__(const float* a, const float* b, const size_t stride_a,
                        const size_t stride_b, const size_t rows,
                        const size_t cols, const float epsilon, bool* result) static {
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

    bool result;

    compare<<<this->stride, 256>>>(this->data_ptr(), other.data_ptr(),
                                   this->stride, other.stride, this->rows,
                                   this->cols, epsilon, &result);

    return result;
}

matrix matrix::backprop_softmax(const matrix& gradient) const {
    matrix softmax_gradient({ gradient.rows, gradient.cols });

    for (size_t r = 0; r < this->rows; ++r) {
        float s_dot = 0.0f;

        for (size_t c = 0; c < this->cols; ++c) {
            s_dot += this->get(r, c) * gradient.get(r, c);
        }

        for (size_t c = 0; c < this->cols; ++c) {
            const float s_j = this->get(r, c);
            const float g_j = gradient.get(r, c);

            softmax_gradient.set(r, c, s_j * (g_j - s_dot));
        }
    }

    return softmax_gradient;
}

void matrix::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));

    float* buffer = new float[buffer_size()];
    cudaMemcpy(buffer, data_ptr(), buffer_size(), cudaMemcpyDeviceToHost);
    out.write(reinterpret_cast<const char*>(buffer), buffer_size());
    delete[] buffer;
}

matrix matrix::load(std::istream& in) {
    size_t new_rows, new_cols;
    in.read(reinterpret_cast<char*>(&new_rows), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&new_cols), sizeof(size_t));

    matrix new_matrix = matrix(new_rows, new_cols);

    float* buffer_data = new float[new_matrix.buffer_size()];
    in.read(reinterpret_cast<char*>(buffer_data), new_matrix.buffer_size());
    cudaMemcpy(new_matrix.data_ptr(), buffer_data, new_matrix.buffer_size(),
               cudaMemcpyHostToDevice);
    delete[] buffer_data;

    return new_matrix;
}
