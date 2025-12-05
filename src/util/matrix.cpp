//
// Created by user on 7/17/25.
//

#include "matrix.hpp"

#include <blaze/math/CustomMatrix.h>
#include <blaze/math/shims/NextMultiple.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <ostream>
#include <sstream>

static constexpr size_t calculate_padded_row_width(const size_t cols) {
    return (matrix::MATRIX_ELEMENT_ALIGNMENT
            - (cols % matrix::MATRIX_ELEMENT_ALIGNMENT))
               % matrix::MATRIX_ELEMENT_ALIGNMENT
           + cols;
}

matrix::matrix(const size_t rows, const size_t cols)
    : rows(rows), cols(cols), row_width(calculate_padded_row_width(cols)) {
    const auto buffer_size = this->buffer_size();
    
    this->data
        = std::unique_ptr<float[], aligned_deleter>((float *)std::aligned_alloc(
            matrix::MATRIX_ELEMENT_ALIGNMENT * sizeof(float), buffer_size));
    std::memset(this->data_ptr(), 0, buffer_size);
}

size_t matrix::buffer_size() const { return row_width * rows * sizeof(float); }

void matrix::verify_bounds(const size_t row, const size_t col) const {
    MATRIX_ASSERT(row < rows && col < cols,
                  "Index out of bounds: ({}, {}) for matrix of size ({}x{})",
                  row, col, rows, cols);
}

void matrix::randomize(const float min, const float max) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            set(i, j,
                min
                    + static_cast<float>(std::rand())
                          / (static_cast<float>(RAND_MAX) / (max - min)));
        }
    }
}

float matrix::max() const {
    return this->reduce<float>(
        [](const float a, const float b) { return std::max(a, b); },
        std::numeric_limits<float>::lowest());
}

float matrix::min() const {
    return this->reduce<float>(
        [](const float a, const float b) { return std::min(a, b); },
        std::numeric_limits<float>::max());
}

float matrix::absmax() const {
    return this->reduce<float>(
        [](const float a, const float b) { return std::max(a, std::abs(b)); },
        0);
}

float matrix::variance() const {
    const float sum = this->reduce<float>(
        [](const float acc, const float value) { return acc + value; }, 0.0f);
    const float sum_sq = this->reduce<float>(
        [](const float acc, const float value) { return acc + value * value; },
        0.0f);

    return (sum_sq / size()) - (sum / size()) * (sum / size());
}

float matrix::stddev() const { return std::sqrt(variance()); }

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
            ss << get(i, j) << " ";
        }
        ss << "\n";
    }

    return ss.str();
}

matrix &matrix::softmax() {
    for (size_t i = 0; i < rows; ++i) {
        // Find the maximum value in the row
        float max_val = get(i, 0);
        for (size_t j = 1; j < cols; ++j) {
            if (get(i, j) > max_val) {
                max_val = get(i, j);
            }
        }

        // Subtract the max value from each element in the row to prevent
        // overflow
        for (size_t j = 0; j < cols; ++j) {
            set(i, j, get(i, j) - max_val);
        }
    }

    this->map([](const float f) { return std::exp(f); });

    for (size_t i = 0; i < rows; ++i) {
        const auto row_sum
            = this->row_sum(i)
              + 1e-8f;  // Adding a small value to avoid division by zero

        for (size_t j = 0; j < cols; ++j) {
            set(i, j, get(i, j) / row_sum);
        }
    }

    return *this;
}

matrix &matrix::mask_upper_triangular(const float mask_value) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = i + 1; j < cols; ++j) {
            set(i, j, mask_value);
        }
    }
    return *this;
}

float matrix::dot_product(const matrix &other) const {
    MATRIX_ASSERT(this->rows == other.rows && this->cols == other.cols,
                  "Matrix dimensions do not match for dot product");

    float result = 0.0f;

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            result += this->get(i, j) * other.get(i, j);
        }
    }

    return result;
}

using custom_matrix = blaze::CustomMatrix<float, blaze::AlignmentFlag::aligned,
                                          blaze::PaddingFlag::padded>;
                                          
void matrix::cross_multiply_into(const matrix &other, matrix &out) const {
    MATRIX_ASSERT(this->cols == other.rows,
                  "Matrix dimensions do not match for cross multiplication");

    // This const_cast *should* be safe because blaze is not going to modify the
    // buffer and we can safely assume that no matrix is not going to be
    // optimized away by the compiler
    custom_matrix a(const_cast<float *>(this->data_ptr()), this->rows,
                    this->cols, this->row_width);
    custom_matrix b(const_cast<float *>(other.data_ptr()), other.rows,
                    other.cols, other.row_width);
    custom_matrix c(out.data_ptr(), out.rows, out.cols, out.row_width);

    c = a * b;
}

matrix matrix::cross_multiplied(const matrix &other) const {
    MATRIX_ASSERT(this->cols == other.rows,
                  "Matrix dimensions do not match for cross multiplication");

    matrix result{ this->rows, other.cols };

    custom_matrix a(const_cast<float *>(this->data_ptr()), this->rows,
                    this->cols, this->row_width);
    custom_matrix b(const_cast<float *>(other.data_ptr()), other.rows,
                    other.cols, other.row_width);
    custom_matrix c(result.data_ptr(), result.rows, result.cols,
                    result.row_width);

    c = a * b;

    return result;
}

// Self @ Other^T
// [M x N] @ [O x N]^T = [M x N] @ [N x O] = [M x O]
matrix matrix::cross_t_multiplied(const matrix &other) const {
    MATRIX_ASSERT(this->rows == other.rows,
                  "Matrix dimensions do not match for cross transposed "
                  "multiplication");

    matrix result{ this->rows, other.rows };

    custom_matrix a(const_cast<float *>(this->data_ptr()), this->rows,
                    this->cols, this->row_width);
    custom_matrix b(const_cast<float *>(other.data_ptr()), other.rows,
                    other.cols, other.row_width);
    custom_matrix c(result.data_ptr(), result.rows, result.cols,
                    result.row_width);
    
    c = a * blaze::trans(b);
    return result;
}

// Self^T @ Other
// [M x N]^T @ [M x O] = [N x M] @ [M x O] = [N x O]
matrix matrix::t_cross_multiplied(const matrix &other) const {
    MATRIX_ASSERT(this->cols == other.cols,
                  "Matrix dimensions do not match for transposed cross "
                  "multiplication");
    
    matrix result{ this->cols, other.cols };
    
    custom_matrix a(const_cast<float *>(this->data_ptr()), this->rows,
                    this->cols, this->row_width);
    custom_matrix b(const_cast<float *>(other.data_ptr()), other.rows,
                    other.cols, other.row_width);
    custom_matrix c(result.data_ptr(), result.rows, result.cols,
                    result.row_width);
    
    c = blaze::trans(a) * b;
    return result;
}

bool matrix::equals(const matrix &other, const float epsilon) const {
    if (this->rows != other.rows || this->cols != other.cols) {
        return false;
    }

    for (size_t i = 0; i < this->rows; ++i) {
        for (size_t j = 0; j < this->cols; ++j) {
            if (std::abs(this->get(i, j) - other.get(i, j)) > epsilon) {
                return false;
            }
        }
    }

    return true;
}

void matrix::save(std::ostream &out) const {
    out.write(reinterpret_cast<const char *>(&rows), sizeof(size_t));
    out.write(reinterpret_cast<const char *>(&cols), sizeof(size_t));
    out.write(reinterpret_cast<const char *>(data_ptr()), buffer_size());
}

matrix matrix::load(std::istream &in) {
    size_t new_rows, new_cols;
    in.read(reinterpret_cast<char *>(&new_rows), sizeof(size_t));
    in.read(reinterpret_cast<char *>(&new_cols), sizeof(size_t));
    
    matrix new_matrix = matrix(new_rows, new_cols);
    in.read(reinterpret_cast<char *>(new_matrix.data_ptr()),
            new_matrix.buffer_size());

    return new_matrix;
}
