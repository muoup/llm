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
    this->data = kernel::matrix::allocate_buffer(this->buffer_size());
}

matrix::matrix(matrix&& other) {
    this->data = other.data;
    this->rows = other.rows;
    this->cols = other.cols;
    this->stride = other.stride;
    other.data = nullptr;
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
                  "Index out of bounds: (%zu, %zu) for matrix of size (%zu x %zu)",
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
    return kernel::matrix::sum(*this);
}

float matrix::max() const {
    return kernel::matrix::max(*this);
}

float matrix::min() const {
    return kernel::matrix::min(*this);
}

float matrix::absmax() const {
    return kernel::matrix::absmax(*this);
}

matrix& matrix::set_all(float value) {
    kernel::matrix::set_all(*this, value);
    return *this;
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

matrix matrix::get_row_vector(const size_t row) const {
    return kernel::matrix::get_row_vector(*this, row);
}

void matrix::set_row_vector(const size_t row, const matrix& row_vector) {
    MATRIX_ASSERT(row_vector.rows == 1 && row_vector.cols == this->cols,
                  "Row vector dimensions do not match for setting row");

    kernel::matrix::set_row_vector(*this, row, row_vector);
}

void matrix::add_row_vector(const size_t row, const matrix& other) {
    MATRIX_ASSERT(other.rows == 1 && other.cols == this->cols,
                  "Row vector dimensions do not match for adding row");

    kernel::matrix::add_row_vector(*this, row, other);
}

void matrix::set_horizontal_slice(const size_t col_start, const matrix& slice) {
    MATRIX_ASSERT(col_start + slice.cols <= this->cols,
                  "Slice dimensions do not match for setting horizontal slice");

    kernel::matrix::set_horizontal_slice(*this, col_start, slice);
}

matrix matrix::get_horizontal_slice(const size_t col_start,
                                    const size_t slice_cols) const {
    MATRIX_ASSERT(col_start + slice_cols <= this->cols,
                  "Slice dimensions do not match for getting horizontal slice");

    return kernel::matrix::get_horizontal_slice(*this, col_start, slice_cols);
}

matrix& matrix::element_wise_multiply(const matrix& other) {
    MATRIX_ASSERT(this->rows == other.rows && this->cols == other.cols,
                  "Matrix dimensions do not match for element-wise multiplication");

    kernel::matrix::element_wise_multiply(*this, other);
    return *this;
}

float matrix::variance() const {
    return kernel::matrix::variance(*this);
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
