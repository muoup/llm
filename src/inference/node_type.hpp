#pragma once

#include <cstdint>

enum class NodeType : uint32_t {
    Attention,
    LinearizedAttention,
    FeedForward,
    LayerNorm,
    Recursion
};

inline const char* node_type_to_string(NodeType type) {
    switch (type) {
        case NodeType::Attention: return "Attention";
        case NodeType::LinearizedAttention: return "LinearizedAttention";
        case NodeType::FeedForward: return "FeedForward";
        case NodeType::LayerNorm: return "LayerNorm";
        case NodeType::Recursion: return "Recursion";
        default: return "Unknown";
    }
}
