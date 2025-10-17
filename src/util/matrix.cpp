//
// Created by user on 7/17/25.
//

#include "matrix.h"

#include <blaze/math/shims/NextMultiple.h>
#include <blaze/math/CustomMatrix.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <ostream>
#include <sstream>

static constexpr size_t calculate_padded_row_width(const size_t cols) {
    return (matrix::MATRIX_ELEMENT_ALIGNMENT
            - (cols % matrix::MATRIX_ELEMENT_ALIGNMENT))
           % matrix::MATRIX_ELEMENT_ALIGNMENT + cols;
}

matrix::matrix(const size_t rows, const size_t cols)
    : rows(rows), cols(cols), row_width(calculate_padded_row_width(cols)) {
    const auto buffer_size = this->buffer_size();

    this->data = std::unique_ptr<float[], aligned_deleter>((float*) std::aligned_alloc(MATRIX_ELEMENT_ALIGNMENT * sizeof(float),
                                    buffer_size));
    std::memset(this->data_ptr(), 0, buffer_size);
}

size_t matrix::buffer_size() const {
    return row_width * rows * sizeof(float);
}

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
              + 1e-8f; // Adding a small value to avoid division by zero

        for (size_t j = 0; j < cols; ++j) {
            set(i, j, get(i, j) / row_sum);
        }
    }

    return *this;
}

void matrix::cross_multiply_into(const matrix &other, matrix &out) const {   
    using custom_matrix
        = blaze::CustomMatrix<float, blaze::AlignmentFlag::aligned,
                              blaze::PaddingFlag::padded>;
    
    // This const_cast *should* be safe because blaze is not going to modify the buffer
    // and we can safely assume that no matrix is not going to be optimized away by the compiler
    custom_matrix a(const_cast<float*>(this->data_ptr()), this->rows, this->cols,
                    this->row_width);
    custom_matrix b(const_cast<float*>(other.data_ptr()), other.rows, other.cols,
                    other.row_width);
    custom_matrix c(out.data_ptr(), out.rows, out.cols,
                    out.row_width);

    c = a * b;
}

matrix matrix::cross_multiply(const matrix &other) const {
    MATRIX_ASSERT(this->cols == other.rows,
                  "Matrix dimensions do not match for cross multiplication");

    matrix result{ this->rows, other.cols };
    
    this->cross_multiply_into(other, result);
    
    return result;
}
