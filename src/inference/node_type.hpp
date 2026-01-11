#pragma once

#include <cstdint>

enum class NodeType : uint32_t {
    Attention,
    FeedForward,
    LayerNorm,
    RMSNorm,
    Recursion
};

inline const char* node_type_to_string(NodeType type) {
    switch (type) {
        case NodeType::Attention:
            return "Attention";
        case NodeType::FeedForward:
            return "FeedForward";
        case NodeType::LayerNorm:
            return "LayerNorm";
        case NodeType::RMSNorm:
            return "RMSNorm";
        case NodeType::Recursion:
            return "Recursion";
        default:
            return "Unknown";
    }
}
