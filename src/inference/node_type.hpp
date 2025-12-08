#pragma once

#include <cstdint>

enum class NodeType : uint32_t {
    Attention,
    LinearizedAttention,
    FeedForward,
    LayerNorm,
};
