#include "matrix.hpp"

#include <kernels/matrix_kernels.hpp>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

static constexpr size_t calculate_stride(const size_t i) {
    // The Stride is Equal to the Least Multiple of (256 / sizeof(float)) Equal
    // to or Greater Than i
    constexpr size_t alignment = 256 / sizeof(float);

    return ((i + alignment - 1) / alignment) * alignment;
}

matrix::matrix(const size_t rows, const size_t cols)
    : rows(rows), cols(cols), stride(calculate_stride(rows)) {
    const auto buffer_size = this->buffer_size();

    this->data = kernel::matrix::allocate_buffer(buffer_size);
}

matrix::~matrix() {
    if (data != nullptr) {
        kernel::matrix::free_buffer(data);
    }
    
    this->data = nullptr;
}

void matrix::randomize(const float min, const float max) {
    kernel::matrix::randomize(*this, min, max);
}

size_t matrix::buffer_size() const {
    return stride * cols * sizeof(float);
}

void matrix::verify_bounds(const size_t row, const size_t col) const {
    MATRIX_ASSERT(row < rows && col < cols,
                  "Index out of bounds: (%d, %d) for matrix of size (%d x %d)",
                  row, col, rows, cols);
}

[[nodiscard]] float matrix::get(const size_t row, const size_t col) const {
    verify_bounds(row, col);
    return kernel::matrix::get(*this, row, col);
}

void matrix::set(const size_t row, const size_t col, const float value) {
    verify_bounds(row, col);
    kernel::matrix::set(*this, row, col, value);
}

matrix matrix::clone() const {
    return kernel::matrix::clone(*this);
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
    return this->reduce(0, [](float a, float b) -> float {
        return std::max(std::abs(a), std::abs(b));
    });
}

matrix& matrix::map(float (*func)(float)) {
    kernel::matrix::general_map(*this, func);
    return *this;
}

matrix& matrix::set_all(float value) {
    kernel::matrix::set_all(*this, value);
    return *this;
}

float matrix::reduce(float acc, float (*reducer)(float, float)) const {
    return kernel::matrix::general_reduce(*this, acc, reducer);
}

matrix& matrix::scale(const float factor) {
    kernel::matrix::scale(*this, factor);
    return *this;
}

matrix& matrix::add(const matrix& offset) {
    MATRIX_ASSERT(this->rows == offset.rows && this->cols == offset.cols,
                  "Matrix dimensions do not match for addition");

    kernel::matrix::add(*this, offset);
    return *this;
}

matrix& matrix::add_scaled(const matrix& other, const float factor) {
    MATRIX_ASSERT(this->rows == other.rows && this->cols == other.cols,
                  "Matrix dimensions do not match for scaled addition");

    kernel::matrix::add_scaled(*this, other, factor);
    return *this;
}

matrix& matrix::add(float f) {
    kernel::matrix::add(*this, f);
    return *this;
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

matrix& matrix::softmax() {
    kernel::matrix::softmax(*this);
    return *this;
}

matrix& matrix::mask_upper_triangular(const float mask_value) {
    kernel::matrix::mask_upper_triangular(*this, mask_value);
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

    return kernel::matrix::cross_multiplied(*this, other);
}

matrix matrix::cross_t_multiplied(const matrix& other) const {
    MATRIX_ASSERT(this->cols == other.cols,
                  "Matrix dimensions do not match for cross post-transposed "
                  "multiplication");

    return kernel::matrix::cross_t_multiplied(*this, other);
}

matrix matrix::t_cross_multiplied(const matrix& other) const {
    MATRIX_ASSERT(this->rows == other.rows,
                  "Matrix dimensions do not match for pre-transposed cross "
                  "multiplication");

    return kernel::matrix::t_cross_multiplied(*this, other);
}

bool matrix::equals(const matrix& other, const float epsilon) const {
    return kernel::matrix::is_equal(*this, other, epsilon);
}

matrix matrix::backprop_softmax(const matrix& gradient) const {
    return kernel::matrix::backprop_softmax(*this, gradient);
}

void matrix::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));

    float* buffer = new float[buffer_size()];
    kernel::matrix::store_from(*this, buffer);
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
    kernel::matrix::load_into(new_matrix, buffer_data);
    delete[] buffer_data;

    return new_matrix;
}
