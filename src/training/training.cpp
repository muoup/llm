#include "training.h"

#include <network/neural_net.h>
#include <training/backpropogation.h>

void train(llm& model, const std::span<const token_id_t> input) {
    // since the llm will predict every i + 1 index, we want to uninclude the last token
    // so we can compare the prediction to an actual token without going out of bounds
    const auto truncated_input = std::span { input.begin(), input.end() - 1 };

    training_data data { input, model.m_dimensions };

    matrix acc = model.embed_tokens(truncated_input);

    const size_t layer_count = model.m_ff_layer.size();

    for (size_t i = 0; i < layer_count; i++) {
        const matrix l1_output = model.forward_l1(acc, i);
        const matrix activated = model.activate(l1_output);
        const matrix l2_output = model.forward_l2(activated, i);

        data.forward_results.emplace_back(acc, l1_output, activated);

        acc.offset(l2_output);
    }

    data.logit_input = acc;
    data.predictions = model.generate_logits(acc).softmax();

    backpropogate(model, data);
}