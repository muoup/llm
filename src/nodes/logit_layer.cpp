#include "logit_layer.hpp"

// ---[ Construction ]---

logit_layer::logit_layer(const size_t dimensions, const size_t vocab_size)
    : vocab_size(vocab_size), w(dimensions, vocab_size), b(1, vocab_size) {}

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

// ---[ Serialization ]---

static void write_matrix(std::ostream& out, const matrix& m) {
    uint64_t dims[] = { m.rows, m.cols };
    out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(m.data_ptr()), m.buffer_size());
}

static matrix read_matrix(std::istream& in) {
    uint64_t dims[2];
    in.read(reinterpret_cast<char*>(dims), sizeof(dims));
    matrix m(dims[0], dims[1]);
    in.read(reinterpret_cast<char*>(m.data_ptr()), m.buffer_size());
    return m;
}

void logit_layer::save(std::ostream& out) const {
    write_matrix(out, w);
    write_matrix(out, b);
}

logit_layer logit_layer::load(std::istream& in, size_t dimensions, size_t vocab_size) {
    logit_layer layer(dimensions, vocab_size);
    layer.w = read_matrix(in);
    layer.b = read_matrix(in);
    return layer;
}
