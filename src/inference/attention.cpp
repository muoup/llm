#include "attention.hpp"

#include <cmath>
#include <span>
#include <vector>
#include <training/optimizer.hpp>

NodeType AttentionLayer::getType() const { return NodeType::Attention; }

AttentionLayer::AttentionLayer(size_t dimensions, size_t head_size,
                               size_t head_count)
    : dimensions(dimensions),
      head_size(head_size),
      head_count(head_count),
      wo(matrix(dimensions, dimensions)) {
          
          for (size_t i = 0; i < head_count; ++i) {
              heads.emplace_back(AttentionHead{
                  matrix(dimensions, head_size), // wq
                  matrix(dimensions, head_size), // wk
                  matrix(dimensions, head_size)  // wv
              });
          }
      }

void AttentionLayer::randomize(const float min, const float max) {
    for (auto& head : heads) {
        head.wq.randomize(min, max);
        head.wk.randomize(min, max);
        head.wv.randomize(min, max);
    }
    
    wo.randomize(min, max);
}

std::vector<matrix> AttentionLayer::forward(std::span<const matrix> inputs) {
    const matrix& input = inputs[0];
    
    std::vector<matrix> returns;
    // placeholder for the final output to prevent insert(0) additional overhead
    returns.emplace_back();
    // placeholder for the concatenated heads
    returns.emplace_back(input.rows, input.cols);
    
    for (size_t h = 0; h < head_count; ++h) {
        const AttentionHead& head = heads[h];
        matrix q = input.cross_multiplied(head.wq);
        matrix k = input.cross_multiplied(head.wk);
        matrix v = input.cross_multiplied(head.wv);

        // Attention scores
        matrix scores = q.cross_multiplied(k.transposed());

        // Scale
        const float scale = 1.0f / std::sqrt(static_cast<float>(q.cols));
        scores.scale(scale);

        // Mask & Softmax
        scores.mask_upper_triangular();
        scores.softmax();

        // Weighted sum
        matrix weighted_sum = scores.cross_multiplied(v);
        
        matrix& output = returns[0];
        output.set_row_vector(h, weighted_sum);
        
        // returns is not modified, so the output reference will remain valid until this point
        
        returns.emplace_back(std::move(q));
        returns.emplace_back(std::move(k));
        returns.emplace_back(std::move(v));
        returns.emplace_back(std::move(scores));
        returns.emplace_back(std::move(weighted_sum));
    }

    matrix& output = returns[1];
    matrix final_output = output.cross_multiplied(wo);
    returns[0] = std::move(final_output);

    // Expected returns layout:
    // [0] -> output matrix
    // [1] -> final output
    // [2] -> q1 (queries head 1)
    // [3] -> k1 (keys head 1)
    // [4] -> v1 (values head 2)
    // [5] -> scores1 (attention scores after softmax head 1)
    // [6] -> weighted_sum1 (weighted sum head 1)
    // [7] -> q2 (queries head 2)
    // [8] -> k2 (keys head 2)
    // ...
    
    return returns;
}

std::vector<matrix> AttentionLayer::backpropogate(
    std::span<const matrix> inputs, std::span<const matrix> outputs,
    std::span<const matrix> gradients, float learning_rate) {
    constexpr float regularization_strength = 0.01f;

    const matrix& layer_input = inputs[0];
    const matrix& final_output = outputs[0];
    
    matrix input_gradient(layer_input.rows, layer_input.cols);
    
    for (size_t h = 0; h < head_count; ++h) {
        const matrix& q = outputs[2 + h * 5 + 0];
        const matrix& k = outputs[2 + h * 5 + 1];
        const matrix& v = outputs[2 + h * 5 + 2];
        const matrix& scores = outputs[2 + h * 5 + 3];
        const matrix& weighted_sum = outputs[2 + h * 5 + 4];
        
        const matrix& post_layer_gradient = gradients[0];
        
        // Backprop through output projection (wo)
        matrix wo_gradient
            = weighted_sum.transposed().cross_multiplied(post_layer_gradient);
        matrix weighted_sum_gradient
            = post_layer_gradient.cross_multiplied(wo.transposed());
        
        // Backprop through weighted sum: weighted_sum = scores * v
        matrix v_gradient
            = scores.transposed().cross_multiplied(weighted_sum_gradient);
        matrix scores_gradient_v
            = weighted_sum_gradient.cross_multiplied(v.transposed());
            
        // Backprop through softmax
        matrix softmax_gradient({ scores_gradient_v.rows, scores_gradient_v.cols });
        
#pragma omp parallel for
        for (size_t i = 0; i < scores.rows; ++i) {
            float s_dot = 0.0f;
            for (size_t l = 0; l < scores.cols; ++l) {
                s_dot += scores_gradient_v.get(i, l) * scores.get(i, l);
            }
            for (size_t j = 0; j < scores.cols; ++j) {
                const float s_j = scores.get(i, j);
                const float g_j = scores_gradient_v.get(i, j);
                softmax_gradient.set(i, j, s_j * (g_j - s_dot));
            }
        }
        
        // Backprop through scaling
        const float scale = 1.0f / std::sqrt(static_cast<float>(q.cols));
        softmax_gradient.scale(scale);
        
        // Backprop through attention scores to Q, K
        matrix q_gradient = softmax_gradient.cross_multiplied(k);
        matrix k_gradient = softmax_gradient.transposed().cross_multiplied(q);
        
        // Backprop through Q, K, V projections to their weight matrices
        matrix layer_input_t = layer_input.transposed();
        matrix wq_gradient = layer_input_t.cross_multiplied(q_gradient);
        matrix wk_gradient = layer_input_t.cross_multiplied(k_gradient);
        matrix wv_gradient = layer_input_t.cross_multiplied(v_gradient);
        
        AttentionHead& head = heads[h];
        regularize_weight_gradient(wq_gradient, head.wq, regularization_strength);
        adjust_matrix(head.wq, wq_gradient, learning_rate);
        regularize_weight_gradient(wk_gradient, head.wk, regularization_strength);
        adjust_matrix(head.wk, wk_gradient, learning_rate);
        regularize_weight_gradient(wv_gradient, head.wv, regularization_strength);
        adjust_matrix(head.wv, wv_gradient, learning_rate);
        
        // Gradient of the input: sum of contributions via each projection +
        matrix head_input_gradient = q_gradient.cross_multiplied(head.wq.transposed());
        head_input_gradient.add(k_gradient.cross_multiplied(head.wk.transposed()));
        head_input_gradient.add(v_gradient.cross_multiplied(head.wv.transposed()));
        input_gradient.add(head_input_gradient);
    }
    
    // Backprop through output projection (wo)
    matrix wo_gradient = final_output.transposed().cross_multiplied(gradients[0]);
    regularize_weight_gradient(wo_gradient, wo, regularization_strength);
    adjust_matrix(wo, wo_gradient, learning_rate);

    return matrix::construct_vec(input_gradient);
}

// ---[ Serialization ]---

void AttentionLayer::save(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    out.write(reinterpret_cast<const char*>(&head_size), sizeof(head_size));
    out.write(reinterpret_cast<const char*>(&head_count), sizeof(head_count));
    
    for (const auto& head : heads) {
        head.wq.save(out);
        head.wk.save(out);
        head.wv.save(out);
    }
    
    wo.save(out);
}

AttentionLayer AttentionLayer::load(std::istream& in) {
    size_t dimensions, head_size, head_count;
    in.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
    in.read(reinterpret_cast<char*>(&head_size), sizeof(head_size));
    in.read(reinterpret_cast<char*>(&head_count), sizeof(head_count));
    
    AttentionLayer layer = AttentionLayer();
    layer.dimensions = dimensions;
    layer.head_size = head_size;
    layer.head_count = head_count;

    for (size_t i = 0; i < head_count; ++i) {
        layer.heads.emplace_back(
            /* .wq = */ matrix::load(in),
            /* .wk = */ matrix::load(in),
            /* .wv = */ matrix::load(in)
        );
    }

    return layer;
}
