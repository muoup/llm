#include "logit_layer.h"

// ---[ Layer Operations ]---

void logit_layer::randomize(const float min, const float max) {
    w.randomize(min, max);
    b.randomize(min, max);
}

// ---[ Operations ]---

matrix logit_layer::apply(const matrix& input) const {
    matrix logits = input.cross_multiplied(w);
    for (size_t i = 0; i < logits.rows; ++i) {
        logits.add_row_vector(i, b);
    }
    return logits;
}
