#pragma once

#include <util/matrix.hpp>
#include <unordered_map>
#include <memory>
#include <atomic>

struct OptimizerState {
    matrix m;
    matrix v;
    size_t step = 0;
};

class CentralOptimizer {
public:
    CentralOptimizer() = default;

    void update(matrix& parameter, const matrix& gradient, float learning_rate, float weight_decay = 0.01f);
    
    void clear() {
        states.clear();
    }

private:
    std::unordered_map<float*, OptimizerState> states;
};
