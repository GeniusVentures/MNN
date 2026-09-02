# Vulkan TurboQuant Runtime Configuration

## Overview

TurboQuant is a KV cache compression feature for Vulkan attention. TurboQuant-K compresses key tensors and TurboQuant-V compresses value tensors, reducing memory bandwidth and improving decode throughput for long-context LLM inference.

---

## Runtime Configuration Keys

All keys are set via the LLM config JSON under the `llm_config` or equivalent config object. The engine reads them in `Llm::initRuntime()` (`llm.cpp`).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `vulkan_turboquant_k_enable` | `bool` | `false` | Enable TurboQuant-K (key cache compression) |
| `vulkan_turboquant_v_enable` | `bool` | `false` | Enable TurboQuant-V (value cache compression) |
| `vulkan_turboquant_block_size` | `int` | `32` | Compression block size. Must match internal constant `kTurboQuantKBlockSize = 32` for activation. |
| `vulkan_turboquant_format` | `int` | `0` | Compression format selector. Only `0` (default) is currently supported/activated. |
| `sparse_v_enable` | `bool` | `false` | Enable sparse-V attention (selective value computation). Independent of TurboQuant but often combined. |
| `sparse_v_tau` | `float` | `1.0e-6` | Sparsity threshold for sparse-V attention. |

### Code references

- Config reading: `llm.cpp:164–169`, `llm.cpp:1040–1043`
- Runtime gate checks: `VulkanAttention.cpp:51–58`
- Internal block size: `kTurboQuantKBlockSize = kTurboQuantKBlockD4 (8) × kAttentionVecSize (4) = 32` (`VulkanAttention.cpp:17,32-33`)

---

## Supported Combinations

| Mode | K | V | Sparse-V | Status |
|------|---|---|----------|--------|
| Dense | — | — | — | Baseline (always supported) |
| TurboQuant-K only | ✓ | — | — | Supported |
| TurboQuant-K + Dense-V | ✓ | — | — | Supported |
| TurboQuant-K + TurboQuant-V | ✓ | ✓ | — | Supported (TurboQuant-V uses wider tolerances) |
| TurboQuant-K + Sparse-V | ✓ | — | ✓ | Supported |
| TurboQuant-K + TurboQuant-V + Sparse-V | ✓ | ✓ | ✓ | Supported |

### Activation gate rules

TurboQuant-K activates when ALL of:
1. `turboquant_k_enable == true`
2. `turboquant_format == 0`
3. `turboquant_block_size == kTurboQuantKBlockSize` (32)
4. `headDim > 0`
5. `headDim % kTurboQuantKBlockSize == 0`

TurboQuant-V activates when the same conditions hold with `turboquant_v_enable == true`.

If the config `turboquant_block_size` is not 32, both TurboQuant-K and TurboQuant-V silently disable (no error, just fall through to dense path).

---

## Known Limitations

1. **Block size is fixed at 32.** The `vulkan_turboquant_block_size` config key accepts any int, but only 32 activates TurboQuant. Other values fall through to dense attention with no warning.

2. **Format 0 only.** The `vulkan_turboquant_format` key exists but only value `0` is implemented. Non-zero values deactivate TurboQuant silently.

3. **TurboQuant-V is lossy.** TurboQuant-V uses wider correctness tolerances (diffThreshold=3.0, diffPercentThreshold=0.7 in tests) because compression is intentionally lossy for the value cache.

4. **Head dimension constraint.** The attention head dimension must be a multiple of 32. If it's not, TurboQuant falls back to dense even if enabled.

---

## CPU Fallback Behavior

### Test harness fallback

Vulkan-specific TurboQuant tests detect CPU-only runtimes and **skip gracefully** rather than failing:

```cpp
// Pattern from AttentionTest.cpp:758-771
auto rtInfo = ExecutorScope::Current()->getRuntime().first;
bool cpuInfer = true;
for (auto& rt : rtInfo) {
    if (rt.first != MNN_FORWARD_CPU) {
        cpuInfer = false;
        break;
    }
}
if (cpuInfer) {
    return true;  // Skip — pass without running Vulkan test
}
```

This pattern is used by:
- `AttentionVulkanTurboQuantTest` (line 758)
- `AttentionVulkanTurboQuantVTest` (line 789)
- Additional subclass tests that override `run()` with the same check

### Contract

1. **Test skip, not failure.** CPU-only runs return `true` (pass). Tests never report false Vulkan failures on non-Vulkan systems.

2. **No misreporting.** The check is explicit: if every runtime in `ExecutorScope::Current()->getRuntime().first` is `MNN_FORWARD_CPU`, the test returns immediately. There is no path where Vulkan coverage is falsely claimed.

3. **Silent skip.** Currently the skip is silent (no log message). The test simply passes with no indication that Vulkan was unavailable. This is intentional to avoid noise in CPU-only CI/CD environments, but can make debugging confusing if a user expects TurboQuant coverage.

### Inference fallback

At the inference level (`VulkanAttention.cpp`), TurboQuant is guarded by `_useTurboQuantK()` and `_useTurboQuantV()` checks. If the runtime conditions aren't met (wrong block size, unsupported head dimension, etc.), the Vulkan attention path falls back to dense compute on the Vulkan backend — it does **not** fall back to CPU. TurboQuant is a performance optimization within the Vulkan backend, not a separate backend selection.

---

## Migration Notes: From GENIUS_NEO_SWARM

This configuration was originally developed under the internal codename **GENIUS_NEO_SWARM**. The config keys and semantics are stable, but the codename is not referenced in source code or documentation.
