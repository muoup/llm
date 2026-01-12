#include "optimizer.hpp"

#include <kernels/optimizer.hpp>
#include <kernels/matrix.hpp>

void CentralOptimizer::update(matrix& parameter, const matrix& gradient, float weight_decay) {
    auto it = states.find(parameter.data);
    
    if (it == states.end()) {
        OptimizerState state;
        state.m = kernel::matrix::async_allocate(parameter.rows, parameter.cols, parameter.type);
        kernel::matrix::set_all(state.m, 0.0f);
        state.v = kernel::matrix::async_allocate(parameter.rows, parameter.cols, parameter.type);
        kernel::matrix::set_all(state.v, 0.0f);
        state.step = 0;
        
        it = states.emplace(parameter.data, std::move(state)).first;
    }

    OptimizerState& state = it->second;
    state.step++;

    kernel::optimizer::adamw_step(
        parameter,
        gradient,
        state.m,
        state.v,
        state.step,
        learning_rate,
        0.9f,   // beta1
        0.999f, // beta2
        1e-8f,  // epsilon
        weight_decay // weight decay
    );
}
