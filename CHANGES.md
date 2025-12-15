# Changes Made to Fix Attention Gradient Vanishing

## Summary

Fixed critical bug where `raw_scores_gradient` in attention layer was becoming zero, preventing query (wq) and key (wk) weight matrices from training.

## Root Cause

The forward pass was applying `norm_clip()` to attention scores before scaling and softmax, but the backward pass did not account for this scaling operation. This caused gradients to be scaled down by potentially huge factors (e.g., 0.0005), effectively vanishing.

## Solution

Removed `norm_clip()` from attention forward pass. Standard scaled dot-product attention doesn't need explicit clipping because:
1. Division by √(d_k) provides numerical scaling
2. Softmax implementation already uses max-subtraction for numerical stability
3. Maintaining symmetry between forward and backward pass

## Files Modified

### src/inference/attention.cpp
- **Lines 72-74**: Removed `kernel::optimizer::norm_clip(scores)` and its wait call
- **Impact**: Forward and backward passes now match, allowing proper gradient flow

### .gitignore  
- **Line 6**: Added `debug*.txt` pattern to ignore debug output files

## Verification Steps

When you retrain with this fix, you should observe:
1. `raw_scores_gradient norm` should be non-zero (typically 1-100 range)
2. `wq_gradient norm` should be comparable to `wv_gradient norm`
3. `wk_gradient norm` should be comparable to `wv_gradient norm`
4. Training loss should decrease more effectively
5. wq and wk parameter values should update across iterations

## Testing

The code compiles successfully. To test on actual hardware with CUDA:
```bash
bash build.bs
build/llm train --data <dataset> --tokenizer <tok_file> --output-model <model_file> --dataset-type row-based -n <num_samples>
```

Monitor the debug output for gradient norms to verify the fix.

## Documentation

See `FIX_SUMMARY.md` for detailed technical explanation of the issue and fix.
