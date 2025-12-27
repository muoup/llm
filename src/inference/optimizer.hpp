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
    CentralOptimizer(float learning_rate) : learning_rate(learning_rate) {}

    void update(matrix& parameter, const matrix& gradient, float weight_decay = 0.01f);
    
    void clear() {
        states.clear();
    }
    
    float learning_rate;

private:
    std::unordered_map<float*, OptimizerState> states;
};
