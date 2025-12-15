# Fix for Attention Layer Gradient Vanishing Issue

## Problem Description

The attention layer's `raw_scores_gradient` was zeroing out during backpropagation, causing the `wq` (query weight) and `wk` (key weight) parameter matrices to remain untrained.

From debug.txt:
```
[DEBUG]   Attention Head 0 Gradients:
[DEBUG]     scores_gradient norm: 47.781044
[DEBUG]     raw_scores_gradient norm: 0.000000  <-- ISSUE: Should be non-zero!
[DEBUG]     wq_gradient norm: 0.000219          <-- Too small
[DEBUG]     wk_gradient norm: 0.002710          <-- Too small
[DEBUG]     wv_gradient norm: 23.514482         <-- Normal (this one was working)
```

## Root Cause

In the **forward pass** (attention.cpp:67-82), the code was applying `norm_clip` to attention scores:

```cpp
// Forward pass - BEFORE FIX
matrix scores = q.cross_t_multiplied(k);    // Q * K^T
const float scale = 1.0f / std::sqrt(head_size);

kernel::optimizer::norm_clip(scores);        // <-- PROBLEM: Clips scores
scores.scale(scale);                         // Then scales by 1/sqrt(d_k)
scores.softmax();
```

The `norm_clip` function (optimizer.cu:7-16) scales the matrix down if its absmax exceeds 5.0:
```cpp
void kernel::optimizer::norm_clip(::matrix& gradient) {
    constexpr auto max_magnitude = 5.0f;
    const auto max = gradient.absmax();
    if (max > max_magnitude) {
        float factor = max_magnitude / max;  // Could be very small!
        gradient.scale(factor);
    }
}
```

### Why This Caused Vanishing Gradients

1. With Q and K having large norms (~650 from debug output), their dot product Q*K^T produces values with potentially huge magnitude (e.g., absmax could be 10,000+)

2. `norm_clip` would then scale these down by a factor like `5.0/10000 = 0.0005`

3. In the **backward pass**, there was NO corresponding operation to account for this scaling

4. Result: Gradients flowing backward through the attention scores were scaled down by the same huge factor, effectively vanishing

5. Since `wq` and `wk` gradients depend on `raw_scores_gradient`, they became near-zero, preventing learning

## Solution

**Remove the `norm_clip` operation from the attention forward pass.**

This is the standard approach in transformer implementations:
- Scaled dot-product attention relies on division by √(d_k) for numerical stability
- The softmax implementation already uses the max-subtraction trick to prevent overflow (matrix_kernels.cu:552-558)
- No additional clipping is needed

## Changes Made

File: `src/inference/attention.cpp`

**Before** (lines 67-76):
```cpp
matrix scores = q.cross_t_multiplied(k);
const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
kernel::optimizer::wait_for_operations();

kernel::optimizer::norm_clip(scores);    // REMOVED
kernel::optimizer::wait_for_operations();

scores.scale(scale);
kernel::optimizer::wait_for_operations();
```

**After** (lines 67-73):
```cpp
matrix scores = q.cross_t_multiplied(k);
const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
kernel::optimizer::wait_for_operations();

scores.scale(scale);
kernel::optimizer::wait_for_operations();
```

## Expected Impact

After this fix, you should see:
- Non-zero `raw_scores_gradient` values (likely in the range of 1-100 based on `scores_gradient` magnitude)
- Properly scaled `wq_gradient` and `wk_gradient` values (similar magnitude to `wv_gradient`)
- The query and key projection matrices will now receive gradient updates and train properly
- Overall model training should be more effective

## Technical Notes

### Why Large Q/K Norms?

The debug output shows embedding norms around 1825 for a (32, 128) matrix. This suggests:
- Average element magnitude: ~28.5
- After projection to (32, 16) query/key, norms remain large (~650)
- This might indicate embeddings need better initialization, but the attention mechanism should handle this correctly with proper scaling

### Softmax Numerical Stability

The softmax kernel already handles large values correctly:
```cuda
// matrix_kernels.cu:552-558
float max_val = kernel::matrix::device_get(data, row, 0);
for (size_t j = 1; j < data.cols; ++j) {
    const float val = kernel::matrix::device_get(data, row, j);
    if (val > max_val) max_val = val;
}
// Then: exp(val - max_val) prevents overflow
```

### Alternative Approaches (Not Recommended)

If you wanted to keep norm_clip, you would need to:
1. Store the clipping factor during forward pass
2. Apply the same factor to gradients during backward pass

However, this adds complexity without benefit since standard attention doesn't need it.

## Verification

To verify the fix works, check the debug output after retraining:
1. `raw_scores_gradient norm` should be non-zero (typically 1-100)
2. `wq_gradient norm` should be similar magnitude to `wv_gradient`
3. `wk_gradient norm` should be similar magnitude to `wv_gradient`
4. Training loss should decrease more effectively
5. `wq` and `wk` parameter norms should change across training iterations
