#include "training.h"

#include <network/neural_net.h>
#include <training/backpropogation.h>

void train(llm& model, const std::span<const token_id_t> input) {
    // since the llm will predict every i + 1 index, we want to uninclude the last token
    // so we can compare the prediction to an actual token without going out of bounds
    const auto truncated_input = std::span { input.begin(), input.end() - 1 };

    training_data data { input, model.m_dimensions };

    matrix acc = model.m_embedding_layer.apply(truncated_input);

    const size_t layer_count = model.m_ff_layer.size();
    data.attention_forward_results.reserve(layer_count);
    data.forward_results.reserve(layer_count);
    data.attention_inputs.reserve(layer_count);

    for (size_t i = 0; i < layer_count; i++) {
        data.attention_inputs.push_back(acc);
        matrix residual = acc;
        auto attention_result = model.m_attention_layers[i].apply(acc);
        data.attention_forward_results.push_back(std::move(attention_result.forward_result));
        acc = std::move(attention_result.output);
        acc.add(residual);

        auto ff_result = model.m_ff_layer[i].apply(acc);
        data.forward_results.push_back(std::move(ff_result.forward_result));
        acc = std::move(ff_result.output);
    }

    data.logit_input = acc;
    data.predictions = model.m_logit_layer.apply(acc).softmax();

    backpropogate(model, data);
}
