---
phase: 04-convert-test-models-mnn-or-onnx-into-ultra-fp4-quantization-
plan: 01
type: execute
tasks: 2
completed: 2
files_created:
  - tools/fp4/quantize_fp4.py
  - source/backend/cpu/CPUFP4Dequant.hpp
  - source/backend/cpu/CPUFP4Dequant.cpp
requirements_met: [PH4-CONV-01, PH4-CONV-02]
---

# Plan 04-01 Summary: FP4 Quantization Tool + CPU Dequant Runtime

## What was built

**Task 1 — Python FP4 quantization tool (`tools/fp4/quantize_fp4.py`):**
- CLI tool: `python tools/fp4/quantize_fp4.py --input model.mnn --output model_fp4.mnn`
- E2M1 encoding verified against all 14 encodable test vectors from `FP4DequantUtils.hpp`
- 2-per-byte little-endian packing (low nibble first)
- Per-channel scale computation (max_abs / 6.0)
- Pipeline: MNNConvert → JSON modification → MNNConvert round-trip
- `symmetricQuan` output: nbits=4, packed weight bytes, per-channel scales, outputDataType=0 (DT_FLOAT)

**Task 2 — CPU FP4 dequant execution class (`CPUFP4Dequant`):**
- `CPUFP4Dequant::onExecute()` calls `dequant_fp4_packed_cpu()` from `FP4DequantUtils.hpp`
- `CPUFP4DequantCreator::onCreate()` detects FP4 by input byte size (`elementSize() == (outputElementCount + 1) / 2`), returns `nullptr` for standard quant types (allowing `CPUDequantizeCreator` fallback)
- Registered via `static bool gResistor` calling `CPUBackend::addCreator(OpType_Dequantize, ...)`
- Co-exists with existing `CPUDequantizeCreator` — no conflicts

## Verification

| Check | Result |
|-------|--------|
| E2M1 encoding (14 test vectors) | All match reference |
| Packing (0x01 low + 0x02 high → 0x21) | Pass |
| `python quantize_fp4.py --help` | Outputs usage |
| C++ build (CPUFP4Dequant) | Compiles clean |
| `libMNN.a` link | Succeeds |

## Self-Check: PASSED
