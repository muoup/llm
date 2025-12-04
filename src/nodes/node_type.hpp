#pragma once

#include <cstdint>

enum class NodeType : uint32_t {
    Attention = 1,
    FeedForward = 2,
};
