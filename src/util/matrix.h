#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>

#define MATRIX_CHECKS

#ifdef MATRIX_CHECKS
#include "../util/assert.h"
#include <sstream>
#endif

static float identity(const float f) {
    return f;
}

struct matrix {
    size_t rows, cols;
    std::unique_ptr<float[]> data;

    matrix(const size_t rows, const size_t cols)
        : rows(rows), cols(cols), data(std::make_unique<float[]>(cols * rows)) {
        std::memset(data.get(), 0, cols * rows * sizeof(float));
    }
    matrix(matrix&&) = default;
    matrix(const matrix& other) : matrix(other.rows, other.cols) {
        std::memcpy(data.get(), other.data.get(), cols * rows * sizeof(float));
    }

    matrix& operator=(matrix&&) = default;
    matrix& operator=(const matrix& other) {
        *this = matrix(other);
        return *this;
    }

    void randomize(float min = -1, float max = 1);

    [[nodiscard]] float get(const size_t row, const size_t col) const {
        verify_bounds(row, col);

        const auto value = data[col + row * cols];
        return value;
    }

    void set(const size_t row, const size_t col, const float value) {
        verify_bounds(row, col);

#ifdef MATRIX_CHECKS
        if (std::isnan(value)) {
            throw std::runtime_error("Invalid value: NaN is not allowed in matrix");
        }
#endif

        data[col + row * cols] = value;
    }

    void offset(const size_t row, const size_t col, const float offset) {
        verify_bounds(row, col);
        data[col + row * cols] += offset;
    }

    void set_row_vector(const size_t row, const matrix& row_vector) {
        verify_row_addition(row_vector);

        for (size_t j = 0; j < row_vector.cols; ++j) {
            set(row, j, row_vector.get(0, j));
        }
    }

    void add_row_vector(const size_t row, const matrix &other) {
        verify_row_addition(other);

        for (size_t i = 0; i < cols; ++i) {
            set(row, i, get(row, i) + other.get(0, i));
        }
    }

    matrix& softmax() {
        this->map([](const float f) {
            return std::exp(f);
        });

        for (size_t i = 0; i < rows; ++i) {
            auto row_sum = this->row_sum(i);

            if (row_sum == 0.0f) row_sum += 1e-10f; // Prevent division by zero

            for (size_t j = 0; j < cols; ++j) {
                set(i, j, get(i, j) / row_sum);
            }
        }

        return *this;
    }

    matrix cross_multiply_map(const matrix &other, float(*const mapping)(float)) const {
        verify_cross_multiply(other);

        const matrix other_transposed = other.transposed();
        matrix result { this->rows, other.cols };

        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < other.rows; ++k) {
                    const auto value = this->get(i, k);
                    const auto other_value = other_transposed.get(j, k);
                    const auto prev_sum = sum;

                    sum += value * other_value;

#ifdef MATRIX_CHECKS
                    if (std::isnan(sum) || std::isinf(sum)) {
                        throw std::runtime_error("Invalid value encountered during cross multiplication: \n"
                                               "previous sum = " + std::to_string(prev_sum) +
                                               ",\nvalue = " + std::to_string(value) +
                                               ",\nother_value = " + std::to_string(other_value));
                    }
#endif
                }

                result.set(i, j, mapping(sum));
            }
        }

        return result;
    }

    matrix cross_multiply(const matrix& other) const {
        return cross_multiply_map(other, identity);
    }

//     matrix cross_multiply(const matrix &other) const {
//         verify_cross_multiply(other);
//         matrix result { this->rows, other.cols };
//
// #ifndef MATRIX_CHECKS
// #pragma omp parallel for
// #endif
//         for (size_t i = 0; i < this->rows; ++i) {
//             for (size_t k = 0; k < other.rows; ++k) {
//                 for (size_t j = 0; j < other.cols; ++j) {
//                     const auto value = this->get(i, k);
//                     const auto other_value = other.get(k, j);
//                     const auto product = value * other_value;
//
//                     result.offset(i, j, value * other_value);
//
// #ifdef MATRIX_CHECKS
//                     if (std::isnan(product) || std::isinf(product)) {
//                         throw std::runtime_error("Invalid value encountered during cross multiplication: \n"
//                                                ",\nvalue = " + std::to_string(value) +
//                                                ",\nother_value = " + std::to_string(other_value));
//                     }
// #endif
//                 }
//             }
//         }
//
//         return result;
//     }

    matrix& scale(const float factor) {
        this->map([factor](const float value) {
            return value * factor;
        });

        return *this;
    }

    matrix scaled(const float factor) const {
        matrix copy { *this };
        copy.scale(factor);
        return copy;
    }

    matrix& offset(const matrix &offset) {
#ifdef MATRIX_CHECKS
        llm_assert(this->cols == offset.cols && this->rows == offset.rows,
                   "Matrix dimensions do not match for offset operation");
#endif

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                set(i, j, get(i, j) + offset.get(i, j));
            }
        }

        return *this;
    }

    matrix& map(const auto mapping) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                set(i, j, mapping(get(i, j)));
            }
        }

        return *this;
    }

    matrix mapped(const auto mapping) const {
        matrix copy { *this };
        copy.map(mapping);
        return copy;
    }

    template <typename ret>
    ret reduce(const auto reducer, ret acc = 0) const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                acc = reducer(acc, get(i, j));
            }
        }

        return acc;
    }

    matrix get_row_vector(const size_t row) const {
        matrix row_vector { 1, cols };
        for (size_t j = 0; j < cols; ++j) {
            row_vector.set(0, j, get(row, j));
        }
        return row_vector;
    }

    float row_sum(const size_t row) const {
        float sum = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            sum += get(row, j);
        }
        return sum;
    }

    float col_sum(const size_t col) const {
        float sum = 0.0f;
        for (size_t i = 0; i < rows; ++i) {
            sum += get(i, col);
        }
        return sum;
    }

    float min() const;
    float max() const;
    float variance() const;
    float stddev() const;

    matrix transposed() const {
        matrix transposed { cols, rows };
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                transposed.set(j, i, get(i, j));
            }
        }
        return transposed;
    }

    std::string header() const;
    std::string to_string(std::uint8_t precision = 4) const;

    std::span<float> to_span() const {
        return std::span(data.get(), cols * rows);
    }

    [[nodiscard]] size_t size() const {
        return cols * rows;
    }

private:
    void verify_bounds(const size_t row, const size_t col) const {
#ifdef MATRIX_CHECKS
        if (col >= cols || row >= rows)
            throw std::out_of_range("Index out of bounds: (" + std::to_string(row) + ", " + std::to_string(col) +
                                    ") (rows: " + std::to_string(rows) + ", cols: " + std::to_string(cols) + ")");
#endif
    }

    void verify_cross_multiply(const matrix& other) const {
#ifdef MATRIX_CHECKS
        if (this->cols != other.rows) {
            std::stringstream ss;
            ss << "Invalid cross multiplication: (" << this->rows << "x" << this->cols << ") * ("
               << other.rows << "x" << other.cols << ")";
            throw std::invalid_argument(ss.str());
        }
#endif
    }

    void verify_row_addition(const matrix& other) const {
#ifdef MATRIX_CHECKS
        if (other.rows != 1 || this->cols != other.cols) {
            std::stringstream ss;
            ss << "Invalid row addition: (" << this->rows << "x" << this->cols << ") + ("
               << other.rows << "x" << other.cols << ")";
            throw std::invalid_argument(ss.str());
        }
#endif
    }
};