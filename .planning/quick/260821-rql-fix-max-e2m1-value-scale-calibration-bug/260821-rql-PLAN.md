---
quick_id: 260821-rql
phase: quick-260821-rql
plan: 1
type: execute
wave: 1
depends_on: []
autonomous: true
files_modified:
  - tools/fp4/quantize_fp4.py
  - tools/fp4/test_quantize_fp4.py
must_haves:
  truths:
    - "quantize_channel_weights() no longer saturates any channel's max-magnitude weight to +/-Inf"
    - "MAX_E2M1_VALUE equals 3.0 (the true maximum finite E2M1 magnitude, nibble 0x5/0xD), not 6.0"
    - "A fast, self-contained, stdlib-only regression test exists that fails if MAX_E2M1_VALUE ever regresses away from 3.0"
  artifacts:
    - path: "tools/fp4/quantize_fp4.py"
      provides: "Corrected MAX_E2M1_VALUE scale-calibration constant (3.0) with a rationale comment guarding against recurrence"
      min_lines: 260
    - path: "tools/fp4/test_quantize_fp4.py"
      provides: "Self-contained unittest regression suite proving channel max-magnitude weights round-trip to finite E2M1 values"
      min_lines: 60
  key_links:
    - from: "tools/fp4/test_quantize_fp4.py"
      to: "tools/fp4/quantize_fp4.py"
      via: "import quantize_fp4"
      pattern: "import quantize_fp4"
    - from: "tools/fp4/test_quantize_fp4.py"
      to: "include/MNN/FP4DequantUtils.hpp"
      via: "ground-truth E2M1 test-vector table mirrored into the test file's E2M1_TABLE constant"
      pattern: "E2M1_TABLE"
---

<objective>
Fix the MAX_E2M1_VALUE scale-calibration bug in `tools/fp4/quantize_fp4.py` (the divisor used to compute per-channel scale is 6.0, but the true maximum finite E2M1 magnitude is 3.0), and add a fast, self-contained regression test that proves the fix and guards against it silently recurring.

Purpose: This constant bug currently guarantees that every channel's max-magnitude weight saturates to +/-Inf during quantization (since `max_abs / (max_abs / 6.0) == 6.0`, and 6.0 falls into `encode_fp4_e2m1`'s saturation branch). Per `.planning/quick/260821-p1q-evaluate-current-fp4-ultra-fp4-implement/SGFP4-PIVOT-ANALYSIS.md` Section 1 and Section 6, this is a verified, root-caused defect that corrupts every FP4_ULTRA-quantized weight tensor produced by this tool, including tensors already consumed live by SuperGenius's `SGProcessingManager` (`dequant_fp4_packed_cpu()` call in `processing_processor_mnn_tensor.cpp`). It also violates Phase 4 plan 04-02's own acceptance criterion ("packed FP4 weights ... match original float weights within E2M1 precision, max error <= 0.5"), which has not yet been executed and therefore has not yet caught this. There is currently no automated test anywhere that exercises this code path.

Output: `tools/fp4/quantize_fp4.py` with `MAX_E2M1_VALUE = 3.0` and an accurate saturation-branch comment, plus a new standalone regression test `tools/fp4/test_quantize_fp4.py` that fails if this constant (or the saturation behavior it controls) ever regresses.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/ROADMAP.md
@.planning/quick/260821-p1q-evaluate-current-fp4-ultra-fp4-implement/SGFP4-PIVOT-ANALYSIS.md
@include/MNN/FP4DequantUtils.hpp
@tools/fp4/quantize_fp4.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Fix MAX_E2M1_VALUE scale-calibration constant</name>
  <files>tools/fp4/quantize_fp4.py</files>
  <read_first>
    - tools/fp4/quantize_fp4.py (lines 1-72 — module docstring, MAX_E2M1_VALUE constant at line 27, encode_fp4_e2m1 saturation branch at lines 42-43, quantize_channel_weights scale computation at line 61)
    - include/MNN/FP4DequantUtils.hpp (lines 17-49 — the E2M1 test-vector table: nibble 0x5/0xD = +/-3.0 is the largest finite magnitude; nibble 0x6/0xE = +/-Inf)
  </read_first>
  <action>
    In tools/fp4/quantize_fp4.py, change the `MAX_E2M1_VALUE` constant from 6.0 to 3.0 — this is the true largest finite-magnitude value E2M1 can encode (nibble 0x5/0xD: biased_e=2, m=1, giving 2^(2-1)*(1+0.5)=3.0, per the test-vector table in FP4DequantUtils.hpp). Directly above the constant, add a short comment explaining why 3.0 is correct and why 6.0 was wrong: normalizing a channel's max-magnitude weight by dividing by X makes that weight's normalized value equal exactly X, and encode_fp4_e2m1 saturates any normalized magnitude with biased_e >= 3 (i.e. magnitude >= 4.0) to +/-Inf, so a divisor of 6.0 guaranteed every channel's largest weight saturated to +/-Inf on every quantization run.

    Then update the comment on the saturation-return line inside encode_fp4_e2m1 (currently describing the saturation target using the old incorrect divisor) so it accurately describes the corrected semantics: saturating to +/-Inf applies to magnitudes with biased_e >= 3 (>= 4.0), which have no finite E2M1 encoding — the largest finite-magnitude code remains 3.0 (nibble 0x5/0xD), unchanged by this fix since the fix is in the calibration constant, not the encoder's saturation logic itself.

    Do not modify encode_fp4_e2m1's control flow, pack_fp4_byte, quantize_model, or any other function — this is a single-constant calibration fix plus its explanatory comments.
  </action>
  <verify>
    <automated>python -c "
import sys, math
sys.path.insert(0, 'tools/fp4')
import quantize_fp4 as q
assert q.MAX_E2M1_VALUE == 3.0, ('MAX_E2M1_VALUE must be 3.0, got %r' % q.MAX_E2M1_VALUE)
channel = [1.0, -2.0, 7.5, 0.25]
packed, scale = q.quantize_channel_weights(channel)
max_idx = max(range(len(channel)), key=lambda i: abs(channel[i]))
byte_val = packed[max_idx >> 1]
nibble = (byte_val >> 4) if (max_idx & 1) else (byte_val & 0x0F)
assert nibble not in (0x6, 0x7, 0xE, 0xF), ('max-magnitude weight saturated to non-finite nibble 0x%x' % nibble)
print('PASS: max-magnitude weight nibble = 0x%x (finite)' % nibble)
"
    </automated>
  </verify>
  <done>MAX_E2M1_VALUE equals 3.0 in tools/fp4/quantize_fp4.py with an accurate rationale comment; the saturation-branch comment in encode_fp4_e2m1 no longer describes the old incorrect divisor; a representative channel's max-magnitude weight now encodes to a finite E2M1 nibble (0x5 or 0xD) instead of an Inf nibble (0x6 or 0xE).</done>
</task>

<task type="auto">
  <name>Task 2: Add self-contained regression test for the round-trip finiteness fix</name>
  <files>tools/fp4/test_quantize_fp4.py</files>
  <read_first>
    - tools/fp4/quantize_fp4.py (encode_fp4_e2m1, quantize_channel_weights, MAX_E2M1_VALUE — after Task 1's fix)
    - include/MNN/FP4DequantUtils.hpp (lines 17-49 — the full 16-entry E2M1 test-vector table and dequant_e2m1_cpu's sign/exponent/mantissa decode logic, to mirror in Python)
    - apps/frameworks/sherpa-mnn/sherpa-mnn/python/tests/test_text2token.py (lines 1-20 — this repo's existing convention for standalone stdlib unittest.TestCase test files)
  </read_first>
  <action>
    Create tools/fp4/test_quantize_fp4.py: a standalone, stdlib-only (unittest, math, os, sys — no new dependencies beyond quantize_fp4.py's existing numpy import) regression suite that proves the Task 1 fix and guards against it recurring. Runnable directly via `python3 tools/fp4/test_quantize_fp4.py` with no MNNConvert, no built MNN binaries, and no model files required.

    Add a module docstring stating this is a regression test for the MAX_E2M1_VALUE scale-calibration bug (per SGFP4-PIVOT-ANALYSIS.md Section 1), and how to run it.

    Insert the test file's own directory onto sys.path (os.path.dirname(os.path.abspath(__file__))) before importing quantize_fp4, so the test runs correctly regardless of the caller's working directory.

    Hardcode a module-level E2M1_TABLE dict mapping all 16 nibble values (0x0 through 0xF) to their IEEE-754 float equivalents, transcribed exactly from the doc-comment test-vector table in FP4DequantUtils.hpp's dequant_e2m1_cpu (0x0=+0.0, 0x1=+0.5, 0x2=+1.0, 0x3=+1.5, 0x4=+2.0, 0x5=+3.0, 0x6=+Inf, 0x7=NaN, 0x8=-0.0, 0x9=-0.5, 0xA=-1.0, 0xB=-1.5, 0xC=-2.0, 0xD=-3.0, 0xE=-Inf, 0xF=NaN), using math.inf and math.nan.

    Implement a module-level function dequant_e2m1(nibble) that is a pure-Python port of dequant_e2m1_cpu from FP4DequantUtils.hpp: extract sign bit (bit 3), exponent bits (bits 1-2), mantissa bit (bit 0); for exponent==0 return the subnormal case (sign * mantissa * 0.5); for exponent==3 return the special case (+/-Inf when mantissa==0, NaN when mantissa==1); otherwise return the normal case (sign * 2^(exponent-1) * (1 + mantissa*0.5)). Comment that this is a Python mirror maintained for test purposes only, kept in sync with FP4DequantUtils.hpp's C++ implementation.

    Implement test class MaxE2M1ValueCalibrationTest(unittest.TestCase) with these methods:

    - test_python_decode_mirror_matches_ground_truth_table: for every nibble 0x0 through 0xF, assert dequant_e2m1(nibble) equals E2M1_TABLE[nibble] (compare via math.isnan on both sides when the expected value is NaN, exact equality otherwise). This validates the Python mirror's fidelity before it is trusted to validate quantize_fp4.py's encoder.

    - test_max_e2m1_value_equals_true_max_finite_magnitude: compute the true maximum finite magnitude as max(v for v in E2M1_TABLE.values() if math.isfinite(v)), assert it equals 3.0, then assert quantize_fp4.MAX_E2M1_VALUE equals that same value. This is the direct regression guard on the fixed constant — it fails immediately if MAX_E2M1_VALUE is ever changed back to 6.0 or any other wrong value.

    - test_channel_max_magnitude_weight_round_trips_finite: build a representative channel of floats with a clear max-magnitude element, e.g. [0.5, -1.2, 3.7, -3.7, 2.0, 0.0] (note: two elements tie for max magnitude here — 3.7 and -3.7 — check both). Call quantize_fp4.quantize_channel_weights(channel) to get packed bytes and scale. For every index, unpack its nibble using the same low-nibble-even/high-nibble-odd convention as pack_fp4_byte, decode with dequant_e2m1, and multiply by scale to get the round-tripped value. Assert every round-tripped value is math.isfinite — this is the core defect guard (previously the max-magnitude element always decoded to +/-Inf). Separately and more precisely, assert the max-magnitude element(s) specifically round-trip to within 1e-6 of their original value: by construction scale = max_abs / MAX_E2M1_VALUE, so the max-magnitude element always normalizes to exactly MAX_E2M1_VALUE (3.0) and, encoded via nibble 0x5/0xD and decoded back, exactly reconstructs its original magnitude (up to float rounding). Do not assert a general per-element error bound across the whole channel — E2M1's non-uniform step sizes and encode_fp4_e2m1's exponent-bracket rounding for smaller-magnitude elements are outside this fix's scope; finiteness plus max-magnitude-element precision is the correct and sufficient regression signal for the MAX_E2M1_VALUE defect.

    - test_multiple_channels_never_saturate_to_inf: iterate over a small list of representative channels covering different shapes and magnitudes (a single-element channel, several small positive floats, a mixed-sign channel, and a channel with large-magnitude values), quantize each with quantize_channel_weights, decode every nibble with dequant_e2m1 and multiply by scale, and assert none of the resulting values are ever +/-Inf or NaN.

    End the file with the standard `if __name__ == "__main__": unittest.main()` entry point.
  </action>
  <verify>
    <automated>
      test -f tools/fp4/test_quantize_fp4.py && echo "test file exists"
      grep -c "def test_" tools/fp4/test_quantize_fp4.py | grep -v '^0'
      python tools/fp4/test_quantize_fp4.py -v
    </automated>
  </verify>
  <done>tools/fp4/test_quantize_fp4.py exists, is runnable standalone with no MNNConvert/build/model-file dependency, contains at least 4 test methods on MaxE2M1ValueCalibrationTest, and all tests pass against the Task 1 fix (proving the max-magnitude weight in a representative channel round-trips to a finite value within the 04-02 tolerance, and would fail if MAX_E2M1_VALUE regressed away from 3.0).</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| `quantize_fp4.py` output bytes -> downstream MNN/SuperGenius consumers | Packed FP4 weight bytes and per-channel scale produced by this tool are consumed by MNN's CPU/Vulkan FP4 dequant runtimes and, per SGFP4-PIVOT-ANALYSIS.md Section 6, directly by SuperGenius's `SGProcessingManager` (`dequant_fp4_packed_cpu()`) for `FP4_ULTRA`-format tensors — a calibration bug here silently corrupts live downstream tensor values, not just local test fixtures |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-rql-01 | Tampering (data integrity) | `quantize_channel_weights` / `encode_fp4_e2m1` in `tools/fp4/quantize_fp4.py` | high | mitigate | Fix `MAX_E2M1_VALUE` (6.0 -> 3.0) so the per-channel max-magnitude weight normalizes to the true max finite E2M1 code (3.0) instead of saturating to +/-Inf; add `tools/fp4/test_quantize_fp4.py` asserting round-trip finiteness so a future edit to this constant fails a fast local test run instead of silently corrupting weights again |
| T-rql-SC | Tampering | Package installs | low | accept | No new packages introduced by this fix — the new test file uses only Python stdlib (`unittest`, `math`, `os`, `sys`); `quantize_fp4.py`'s existing `numpy` dependency is unchanged |
</threat_model>

<verification>
1. `python -c "import sys; sys.path.insert(0,'tools/fp4'); import quantize_fp4; assert quantize_fp4.MAX_E2M1_VALUE == 3.0"` passes.
2. `python tools/fp4/test_quantize_fp4.py -v` runs to completion with all tests passing, in well under a few seconds, with no MNNConvert or built MNN binary on PATH.
3. Manually reverting `MAX_E2M1_VALUE` to 6.0 and re-running `tools/fp4/test_quantize_fp4.py` causes `test_max_e2m1_value_equals_true_max_finite_magnitude` (and likely `test_channel_max_magnitude_weight_round_trips_finite`) to fail — confirming the regression test actually catches the original bug.
</verification>

<success_criteria>
1. `tools/fp4/quantize_fp4.py`'s `MAX_E2M1_VALUE` is 3.0, and every channel's max-magnitude weight now normalizes to a finite E2M1 code instead of +/-Inf.
2. A fast (sub-second), self-contained, stdlib-only regression test at `tools/fp4/test_quantize_fp4.py` proves this and would fail if the bug recurred.
3. No files outside `tools/fp4/` were modified; `include/MNN/FP4DequantUtils.hpp`, `CPUFP4Dequant.cpp`, and `VulkanFP4Dequant.cpp` are untouched, since the bug is entirely in the Python encoder's calibration constant.
4. Phase 4 plan 04-02 (not executed as part of this quick task) will no longer be blocked by this defect when it is eventually run.
</success_criteria>

<output>
Create `.planning/quick/260821-rql-fix-max-e2m1-value-scale-calibration-bug/260821-rql-SUMMARY.md` when done
</output>
