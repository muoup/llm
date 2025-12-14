#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>

#include <inference/feed_forward.hpp>
#include <util/matrix.hpp>
#include <kernels/matrix_kernels.hpp>

// Helper to create a matrix from a vector of vectors
matrix create_matrix(const std::vector<std::vector<float>>& data) {
    if (data.empty() || data[0].empty()) {
        return matrix(0, 0);
    }
    size_t rows = data.size();
    size_t cols = data[0].size();
    matrix m(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m.set(i, j, data[i][j]);
        }
    }
    kernel::matrix::check_errors("create_matrix");
    return m;
}

// Helper to check if two matrices are close enough
bool are_equal(const matrix& a, const matrix& b, float epsilon = 1e-4f) {
    if (a.rows != b.rows || a.cols != b.cols) {
        std::cerr << "Matrix dimension mismatch!" << std::endl;
        a.print_bounds();
        b.print_bounds();
        return false;
    }
    bool equal = true;
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            if (std::abs(a.get(i, j) - b.get(i, j)) > epsilon) {
                std::cerr << "Mismatch at (" << i << ", " << j << "): "
                          << a.get(i, j) << " vs " << b.get(i, j) << " (diff: " << std::abs(a.get(i, j) - b.get(i, j)) << ")" << std::endl;
                equal = false;
            }
        }
    }
    return equal;
}

void print_matrix(const matrix& m, const std::string& name) {
    std::cout << name << ":\n" << m.to_string() << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "--- Running Feed-Forward Layer Test ---" << std::endl;

    // --- Setup ---
    size_t dims = 2;
    size_t proj_dims = 4;
    FeedForwardLayer layer(dims, proj_dims);

    // --- Hardcode weights and biases ---
    layer.w1 = create_matrix({{0.0f, 0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f, 0.7f}});
    layer.b1 = create_matrix({{0.0f, 0.1f, 0.2f, 0.3f}});
    layer.w2 = create_matrix({{0.0f, 0.1f}, {0.2f, 0.3f}, {0.4f, 0.5f}, {0.6f, 0.7f}});
    layer.b2 = create_matrix({{0.0f, 0.1f}});
    
    // --- Forward Pass Test ---
    std::cout << "\n--- Testing Forward Pass ---" << std::endl;
    matrix input = create_matrix({{0.5f, -0.5f}});
    std::vector<matrix> inputs;
    inputs.push_back(std::move(input));

    ForwardingResult fwd_result = layer.forward(inputs);
    const matrix& final_output = fwd_result.outputs[0];

    matrix expected_final_output = create_matrix({{0.0598f, 0.1695f}});

    if (!are_equal(final_output, expected_final_output, 1e-4f)) {
        std::cerr << "Forward pass output is incorrect!" << std::endl;
        print_matrix(expected_final_output, "Expected final_output");
        print_matrix(final_output, "Got final_output");
        return 1;
    }
    std::cout << "Forward pass output is correct." << std::endl;

    // --- Backward Pass Test ---
    std::cout << "\n--- Testing Backward Pass ---" << std::endl;
    matrix post_layer_gradient = create_matrix({{0.1f, -0.1f}});
    float learning_rate = 0.01f;

    std::vector<matrix> backprop_gradients;
    backprop_gradients.emplace_back(std::move(post_layer_gradient));

    std::vector<matrix> input_gradient_vec = layer.backpropogate(fwd_result, inputs, backprop_gradients, learning_rate);
    const matrix& input_gradient = input_gradient_vec[0];

    // Check input gradient
    matrix expected_input_gradient = create_matrix({{-0.00303f, -0.00715f}});
    if (!are_equal(input_gradient, expected_input_gradient, 1e-5f)) {
        std::cerr << "Backward pass input_gradient is incorrect!" << std::endl;
        print_matrix(expected_input_gradient, "Expected input_gradient");
        print_matrix(input_gradient, "Got input_gradient");
        return 1;
    }
    std::cout << "Backward pass input_gradient is correct." << std::endl;
    
    // Check updated weights
    matrix expected_new_w1 = create_matrix({{0.001515f, -0.001995f, -0.003995f, -0.005985f}, {0.007848f, 0.009849f, 0.01185f, 0.01386f}});
    matrix expected_new_b1 = create_matrix({{0.000001f, 0.1f, 0.2f, 0.3001f}});
    matrix expected_new_w2 = create_matrix({{-0.000002f, 0.1002f}, {0.1999f, 0.3001f}, {0.3998f, 0.5002f}, {0.5998f, 0.7002f}});
    matrix expected_new_b2 = create_matrix({{-0.001f, 0.101f}});
    
    if (!are_equal(layer.w1, expected_new_w1, 1e-6f)) {
        std::cerr << "Backward pass updated w1 is incorrect!" << std::endl;
        print_matrix(expected_new_w1, "Expected w1");
        print_matrix(layer.w1, "Got w1");
        return 1;
    }
    std::cout << "Backward pass updated w1 is correct." << std::endl;

    if (!are_equal(layer.b1, expected_new_b1, 1e-6f)) {
        std::cerr << "Backward pass updated b1 is incorrect!" << std::endl;
        print_matrix(expected_new_b1, "Expected b1");
        print_matrix(layer.b1, "Got b1");
        return 1;
    }
    std::cout << "Backward pass updated b1 is correct." << std::endl;
    
    if (!are_equal(layer.w2, expected_new_w2, 1e-6f)) {
        std::cerr << "Backward pass updated w2 is incorrect!" << std::endl;
        print_matrix(expected_new_w2, "Expected w2");
        print_matrix(layer.w2, "Got w2");
        return 1;
    }
    std::cout << "Backward pass updated w2 is correct." << std::endl;

    if (!are_equal(layer.b2, expected_new_b2, 1e-6f)) {
        std::cerr << "Backward pass updated b2 is incorrect!" << std::endl;
        print_matrix(expected_new_b2, "Expected b2");
        print_matrix(layer.b2, "Got b2");
        return 1;
    }
    std::cout << "Backward pass updated b2 is correct." << std::endl;

    std::cout << "\n--- All Feed-Forward Layer Tests Passed! ---" << std::endl;

    return 0;
}