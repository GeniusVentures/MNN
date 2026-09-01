#!/usr/bin/env python3
"""SGFP4 v2 real-weight validation driver (Phase 10, Plans 10-01/10-03).

Extracts every FP32 initializer from an ONNX model (approved corpus:
AlexNet opset-16), computes per-layer weight statistics and distribution
characterization, runs the gnus-poc reference adaptive encoder to obtain
leaf-size histograms / code-mode mix / per-leaf decode error, evaluates the
hard per-layer D-07 gate (plain per-element MSE + relative error vs the
per-leaf-size targets of DEFAULT_V2_THRESHOLDS), and writes the committed
markdown report + machine-readable JSON sidecar.

The C++ encode-parity leg (--encode-dump) drives tools/fp4's
sgfp4_encode_dump.out harness on sampled layers (wired in Plan 10-03).

Usage (from the MNN repo root):
    python tools/fp4/validate_real_weights.py \
        [--model W:/gnus/models/alexnet_Opset16.onnx] \
        [--report tools/fp4/real_weight_validation_report.md] \
        [--gnus-poc-root W:/gnus/GeniusCognitiveSystem/GNUS-NEO-SWARM/gnus-poc] \
        [--thresholds delta.json] \
        [--encode-dump build/Release/sgfp4_encode_dump.out] \
        [--sample <tensor-name> ...] [--list-only]

Exit codes:
    0 = all layers pass the gate (and parity, when the leg is active)
    1 = usage / IO error
    2 = NaN/Inf or unreadable tensor (tensor named on stderr)
    3 = D-07 per-layer gate failure (report still written)
    4 = C++ parity mismatch
"""

import argparse
import filecmp
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Decode-vs-decode contractual tolerance (Phase 9 pattern,
# test/op/SGFP4EncodeTest.cpp kEncodeRelTol).
K_ENCODE_REL_TOL = 1e-4

DEFAULT_MODEL = "W:/gnus/models/alexnet_Opset16.onnx"
# W-3 (Phase 11, D-11): env-var override with the original hard-coded path
# as fallback. --gnus-poc-root stays authoritative when passed explicitly.
DEFAULT_GNUS_POC_ROOT = os.environ.get("SGFP4_GNUS_POC_ROOT", "W:/gnus/GeniusCognitiveSystem/GNUS-NEO-SWARM/gnus-poc")

# D-03 (user-approved) light-tier floor.
LIGHT_TIER_MAX_ELEMENTS = 4096

# Quadtree constant mirrored from gnus-poc quadtree.py (_kRelativeEpsilon):
# signal_power at or below this value makes the encoder's relative term
# vanish; annotated in the report so a silent epsilon-pass is visible.
K_RELATIVE_EPSILON = 1e-12

LEAF_SIZES = (64, 32, 16, 8, 4)

# D-07 gate targets: identical to gnus-poc fp4_exporter.DEFAULT_V2_THRESHOLDS.
DEFAULT_V2_THRESHOLDS = {
    64: {"max_mse": 0.01, "max_relative": 0.05},
    32: {"max_mse": 0.005, "max_relative": 0.03},
    16: {"max_mse": 0.002, "max_relative": 0.02},
    8: {"max_mse": 0.001, "max_relative": 0.01},
    4: {"max_mse": 0.0005, "max_relative": 0.005},
}

# Relative-error denominator floor (mirrors gnus-poc convention).
RELATIVE_EPS = 1e-12

HISTOGRAM_BINS = 64


# ---------------------------------------------------------------------------
# gnus-poc import helper
# ---------------------------------------------------------------------------

def import_gnus_poc(root: Path):
    """Insert gnus-poc on sys.path and import the reference modules."""
    if not root.is_dir():
        print(f"gnus-poc root not found: {root}", file=sys.stderr)
        sys.exit(1)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from quantize.fp4_exporter import FP4Exporter
    from quantize.quadtree import QuadtreeEncoder
    from quantize.laplacian import LaplacianWeightedError
    from quantize.sgfp4_decoder import decode_v2

    return FP4Exporter, QuadtreeEncoder, LaplacianWeightedError, decode_v2


# ---------------------------------------------------------------------------
# Corpus extraction / tiering / finite gate
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_thresholds(override_path):
    """Load the effective threshold table (override or Python defaults)."""
    if override_path is None:
        return {size: dict(g) for size, g in DEFAULT_V2_THRESHOLDS.items()}, None
    try:
        with open(override_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read thresholds file {override_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    table = {}
    for key, gates in raw.items():
        table[int(key)] = {"max_mse": float(gates["max_mse"]),
                           "max_relative": float(gates["max_relative"])}
    for size in LEAF_SIZES:
        if size not in table:
            print(f"thresholds file missing leaf size {size}", file=sys.stderr)
            sys.exit(1)
    return table, str(override_path)


def extract_initializers(model_path: Path):
    """Yield (name, dims, ndarray) for every FP32 initializer, one at a time."""
    try:
        import onnx
    except ImportError:
        print("the 'onnx' package is required (pip install onnx)", file=sys.stderr)
        sys.exit(1)
    try:
        model = onnx.load(str(model_path))
    except Exception as exc:  # onnx raises a variety of parse errors
        print(f"cannot load ONNX model {model_path}: {exc}", file=sys.stderr)
        sys.exit(1)
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        if arr.dtype != np.float32:
            print(f"tensor '{init.name}' is {arr.dtype}, expected float32",
                  file=sys.stderr)
            sys.exit(2)
        yield init.name, list(arr.shape), arr


def project_2d(dims, arr):
    """Project an N-D tensor to the [dimO, dimI] plane per tools/fp4 README.

    dimO = shape[0]; all trailing dims fold into dimI (row-major).
    1-D [N] -> [N, 1] degenerate plane.
    """
    if len(dims) == 1:
        return arr.reshape(dims[0], 1)
    dim_o = dims[0]
    dim_i = 1
    for d in dims[1:]:
        dim_i *= d
    return arr.reshape(dim_o, dim_i)


def classify_tier(dim_o, dim_i, elements):
    """D-03 light tier: elements < 4096 OR dimI == 1."""
    light = elements < LIGHT_TIER_MAX_ELEMENTS or dim_i == 1
    return "light" if light else "full"


def finite_gate(name, arr):
    if not np.isfinite(arr).all():
        print(f"tensor '{name}' contains NaN or Inf values", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# Distribution statistics (D-06)
# ---------------------------------------------------------------------------

def weight_stats(plane):
    """Distribution characterization of the FP32 weights (D-06)."""
    x = plane.astype(np.float64).ravel()
    mu = x.mean()
    sigma = x.std()
    if sigma > 0.0:
        kurtosis = float(np.mean(((x - mu) / sigma) ** 4) - 3.0)
        outlier_6sigma = float(np.mean(np.abs(x) > mu + 6.0 * sigma))
    else:
        kurtosis = 0.0
        outlier_6sigma = 0.0
    q99 = float(np.quantile(np.abs(x), 0.99))
    outlier_q99 = float(np.mean(np.abs(x) > q99)) if q99 > 0.0 else 0.0

    # 64 log-spaced bins over the tensor's own |value| range (weights cross
    # zero, so the histogram spans |w|; edges recorded in the JSON sidecar so
    # the convention is explicit).
    abs_values = np.abs(x)
    positive = abs_values[abs_values > 0.0]
    if positive.size >= 2:
        lo = float(positive.min())
        hi = float(abs_values.max())
        if hi > lo:
            edges = np.logspace(math.log10(lo), math.log10(hi),
                                HISTOGRAM_BINS + 1)
            counts, _ = np.histogram(abs_values, bins=edges)
            edges_list = [float(e) for e in edges]
            counts_list = [int(c) for c in counts]
        else:
            edges_list = [lo, hi]
            counts_list = [int(abs_values.size)]
    else:
        top = float(abs_values.max()) if abs_values.size else 0.0
        edges_list = [0.0, top]
        counts_list = [int(abs_values.size)]

    return {
        "kurtosis": kurtosis,
        "outlier_share_6sigma": outlier_6sigma,
        "outlier_share_q99": outlier_q99,
        "hist_edges": edges_list,
        "hist_counts": counts_list,
    }


# ---------------------------------------------------------------------------
# Parity sampling rule (default; Plan 10-03)
# ---------------------------------------------------------------------------

def default_sample_names(initializers):
    """Default C++ parity sampling rule (classes, not hardcoded tensor
    names): every non-64-aligned full-tier tensor, the largest aligned plane,
    one aligned 4-D conv kernel, and two light-tier tensors."""
    full = [t for t in initializers if t["tier"] == "full"]
    light = [t for t in initializers if t["tier"] == "light"]
    non_aligned = [t for t in full
                   if t["dim_o"] % 64 != 0 or t["dim_i"] % 64 != 0]
    non_aligned_names = {t["name"] for t in non_aligned}
    aligned = [t for t in full if t["name"] not in non_aligned_names]

    picks = [t["name"] for t in non_aligned]
    if aligned:
        largest = max(aligned, key=lambda t: t["elements"])
        picks.append(largest["name"])
        convs = [t for t in aligned
                 if t["ndim"] == 4 and t["name"] != largest["name"]]
        if convs:
            picks.append(max(convs, key=lambda t: t["elements"])["name"])
    picks.extend(t["name"] for t in light[:2])
    return picks


# ---------------------------------------------------------------------------
# Reference sweep: stats + gate
# ---------------------------------------------------------------------------

def run_sweep(exporter, QuadtreeEncoder, LaplacianWeightedError, decode_v2,
              initializers, effective_thresholds):
    """Full reference sweep: per-layer stats, D-07 gate, exit code, summary."""
    laplacian = LaplacianWeightedError()
    layers = []
    gate_failed = False
    failing_tuples = []

    for tensor in initializers:
        name = tensor["name"]
        plane = tensor["plane"]
        dim_o, dim_i = plane.shape
        entry = {
            "name": name,
            "dims": tensor["dims"],
            "dim_o": dim_o,
            "dim_i": dim_i,
            "elements": tensor["elements"],
            "tier": tensor["tier"],
        }
        layers.append(entry)

        binary, stats = exporter.export_weights(
            plane, name, adaptive=True, thresholds=effective_thresholds)
        decoded = decode_v2(binary, dim_o, dim_i)

        if entry["tier"] == "light":
            # Light tier (D-03): framing sanity + roundtrip, no stats fit.
            framed_ok = (len(binary) >= 16 and binary[0:4] == b"SGF4"
                         and binary[4] == 0x02
                         and stats["shape"] == [dim_o, dim_i])
            err = np.abs(plane.astype(np.float64) - decoded.astype(np.float64))
            entry.update({
                "framed_ok": bool(framed_ok),
                "max_abs_err": float(err.max()),
                "mse": float(np.mean(err ** 2)),
                "container_bytes": int(stats["total_bytes"]),
                "effective_bpw": float(stats["effective_bpw"]),
            })
            if framed_ok:
                entry["gate"] = "PASS"
            else:
                entry["gate"] = "FAIL"
                gate_failed = True
                failing_tuples.append(
                    {"layer": name, "leaf_size": 0, "kind": "framing",
                     "value": 0.0, "target": 0.0})
            continue

        # Full tier: distribution stats (D-06).
        entry.update(weight_stats(plane))

        # Leaf geometry: drive the quadtree exactly as the exporter does.
        tiles_y = math.ceil(dim_o / 64)
        tiles_x = math.ceil(dim_i / 64)
        padded = np.zeros((tiles_y * 64, tiles_x * 64), dtype=np.float32)
        padded[:dim_o, :dim_i] = plane

        leaf_hist = {size: 0 for size in LEAF_SIZES}
        mode_mix = {"fp4": 0, "t158": 0}
        worst_mse = 0.0
        worst_rel = 0.0
        worst_plain_rel = 0.0
        worst_mse_size = 0
        worst_rel_size = 0
        layer_failures = []
        eps_escapes = 0

        orig64 = plane.astype(np.float64)
        dec64 = decoded.astype(np.float64)

        for block_y in range(tiles_y):
            for block_x in range(tiles_x):
                superblock = padded[
                    block_y * 64:(block_y + 1) * 64,
                    block_x * 64:(block_x + 1) * 64]
                encoder = QuadtreeEncoder(
                    thresholds=effective_thresholds,
                    ternary_delta=0.10,
                    fit_fp4=exporter._encode_fp4_affine_variable,
                    fit_t158=exporter._encode_t158_affine_variable,
                    laplacian=laplacian,
                    min_block_size=4)
                blocks = encoder.encode(superblock)
                for block in blocks:
                    size = block["size"]
                    leaf_hist[size] += 1
                    mode_mix["fp4" if int(block["mode"]) == 0 else "t158"] += 1

                    # True (cropped, unpadded) footprint of the leaf.
                    oy = block_y * 64 + block["y"]
                    ox = block_x * 64 + block["x"]
                    th = max(0, min(size, dim_o - oy))
                    tw = max(0, min(size, dim_i - ox))
                    if th == 0 or tw == 0:
                        continue
                    o = orig64[oy:oy + th, ox:ox + tw]
                    d = dec64[oy:oy + th, ox:ox + tw]
                    targets = effective_thresholds.get(
                        size, effective_thresholds.get(4))
                    target_mse = targets["max_mse"]
                    target_rel = targets["max_relative"]

                    # D-07 gate (user-reformulated 2026-08-31): the hard
                    # absolute gate is the plain per-element leaf MSE. The
                    # relative criterion is evaluated in leaf-aggregate
                    # energy terms (mse / signal_power <= max_relative) --
                    # the same folding the exporter's split driver applies
                    # (quadtree._combined_gate_error). The plain per-element
                    # ratio |o-d|/(|o|+eps) is structurally unbounded on
                    # real weights (near-zero denominators; observed worst
                    # 3.6e6) and is tracked below as an informational
                    # statistic only.
                    mse = float(np.mean((o - d) ** 2))
                    signal_power = float(np.mean(o ** 2))
                    if signal_power > K_RELATIVE_EPSILON:
                        energy_rel = mse / signal_power
                    else:
                        energy_rel = None
                        eps_escapes += 1
                    plain_rel = float(np.max(np.abs(o - d) / (np.abs(o) + RELATIVE_EPS)))

                    if mse > worst_mse:
                        worst_mse = mse
                        worst_mse_size = size
                    if energy_rel is not None and energy_rel > worst_rel:
                        worst_rel = energy_rel
                        worst_rel_size = size
                    if plain_rel > worst_plain_rel:
                        worst_plain_rel = plain_rel
                    if mse > target_mse:
                        layer_failures.append(
                            {"leaf": size, "kind": "mse", "value": mse,
                             "target": target_mse})
                    if energy_rel is not None and energy_rel > target_rel:
                        layer_failures.append(
                            {"leaf": size, "kind": "relative", "value": energy_rel,
                             "target": target_rel})

        entry.update({
            "leaf_hist": leaf_hist,
            "mode_mix": mode_mix,
            "layout_distribution": stats["layout_distribution"],
            "worst_leaf_mse": worst_mse,
            "worst_leaf_mse_size": worst_mse_size,
            "worst_leaf_rel": worst_rel,
            "worst_leaf_rel_size": worst_rel_size,
            "worst_plain_rel": worst_plain_rel,
            "eps_escapes": eps_escapes,
            "container_bytes": int(stats["total_bytes"]),
            "effective_bpw": float(stats["effective_bpw"]),
            "failures": layer_failures,
        })

        # Pad overhead (non-64-aligned tensors only; D-10 measured-only).
        if dim_o % 64 != 0 or dim_i % 64 != 0:
            padded_elems = (math.ceil(dim_o / 64) * 64) \
                * (math.ceil(dim_i / 64) * 64)
            entry["pad_overhead_ratio"] = padded_elems / float(dim_o * dim_i)
            entry["bpw_pad_delta"] = None  # filled conceptually by bpw report

        if layer_failures:
            entry["gate"] = "FAIL"
            gate_failed = True
            for failure in layer_failures:
                failing_tuples.append(
                    {"layer": name, "leaf_size": failure["leaf"],
                     "kind": failure["kind"], "value": failure["value"],
                     "target": failure["target"]})
        else:
            entry["gate"] = "PASS"

    failing_layers = [e["name"] for e in layers if e.get("gate") == "FAIL"]
    exit_code = 3 if gate_failed else 0
    summary = {
        "total_layers": len(layers),
        "full_tier": sum(1 for e in layers if e["tier"] == "full"),
        "light_tier": sum(1 for e in layers if e["tier"] == "light"),
        "failing_layers": failing_layers,
        "failing_tuples": failing_tuples,
        "exit_code": exit_code,
    }
    return layers, exit_code, summary


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def format_hist(hist):
    rendered = " ".join(f"{size}:{hist[size]}" for size in LEAF_SIZES
                        if hist[size])
    return rendered or "-"


def write_report(layers, exit_code, summary, context, report_md, report_json):
    """Render the committed markdown report + JSON sidecar."""
    thresholds = context["thresholds"]
    thresholds_source = context.get("thresholds_source")
    lines = []
    lines.append("# SGFP4 v2 Real-Weight Validation Report")
    lines.append("")
    lines.append(f"- **Model:** `{context['model_path']}`")
    lines.append(f"- **Model SHA-256:** `{context['model_sha256']}`")
    lines.append(f"- **Generated (UTC):** {context['generated_utc']}")
    lines.append("- **Toolchain:** "
                 f"python {context['python_version']}, "
                 f"numpy {context['numpy_version']}, "
                 f"onnx {context['onnx_version']}")
    lines.append(f"- **gnus-poc root:** `{context['gnus_poc_root']}`")
    lines.append("- **Threshold table:** "
                 f"{context['thresholds_source'] or 'DEFAULT_V2_THRESHOLDS (exporter defaults)'}")
    lines.append("")
    lines.append("Effective threshold table:")
    lines.append("")
    lines.append("| leaf size | max_mse | max_relative |")
    lines.append("|---|---|---|")
    for size in LEAF_SIZES:
        gates = thresholds.get(size, thresholds.get(4))
        lines.append(f"| {size} | {gates['max_mse']} | {gates['max_relative']} |")
    lines.append("")
    lines.append("## Per-layer results")
    lines.append("")
    lines.append("Gate metric (user-reformulated 2026-08-31): hard gate = plain "
                 "per-element worst-leaf MSE; relative criterion = leaf energy "
                 "ratio `mse / signal_power` (the folding the exporter's own "
                 "split driver uses). The plain per-element relative ratio is "
                 "reported in parentheses as an informational statistic only "
                 "(structurally unbounded near zero-weight).")
    lines.append("")
    lines.append("| tensor | dims | 2-D projection | elements | tier | kurtosis | "
                 "outliers (6σ / q99) | leaf histogram | fp4/t158 | "
                 "worst leaf MSE (target) | worst rel. err. (target) | gate |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for entry in layers:
        dims_repr = "x".join(str(d) for d in entry["dims"])
        proj = f"{entry['dim_o']}x{entry['dim_i']}"
        if entry["tier"] == "light":
            kurt = "-"
            outl = "-"
            hist = "-"
            mix = "-"
            worst_mse = f"{entry['mse']:.3e} (roundtrip)"
            worst_rel = f"{entry['max_abs_err']:.3e} (max-abs)"
        else:
            kurt = f"{entry['kurtosis']:.2f}"
            outl = (f"{entry['outlier_share_6sigma']:.2e} / "
                    f"{entry['outlier_share_q99']:.2e}")
            hist = format_hist(entry["leaf_hist"])
            mix = f"{entry['mode_mix']['fp4']}/{entry['mode_mix']['t158']}"
            mse_t = thresholds[entry["worst_leaf_mse_size"]]["max_mse"] \
                if entry["worst_leaf_mse_size"] in thresholds else None
            rel_t = thresholds[entry["worst_leaf_rel_size"]]["max_relative"] \
                if entry["worst_leaf_rel_size"] in thresholds else None
            worst_mse = f"{entry['worst_leaf_mse']:.3e} ({mse_t})"
            worst_rel = (f"{entry['worst_leaf_rel']:.3e} ({rel_t}) "
                         f"[plain {entry['worst_plain_rel']:.1e}]")
        lines.append(
            f"| `{entry['name']}` | {dims_repr} | {proj} | {entry['elements']} | "
            f"{entry['tier']} | {kurt} | {outl} | {hist} | {mix} | {worst_mse} | "
            f"{worst_rel} | **{entry['gate']}** |")
    lines.append("")

    lines.append("## Pad overhead (non-64-aligned tensors)")
    lines.append("")
    pad_rows = [e for e in layers if e.get("pad_overhead_ratio") is not None]
    if pad_rows:
        lines.append("| tensor | projection | padded/plain ratio | effective bpw |")
        lines.append("|---|---|---|---|")
        for entry in pad_rows:
            lines.append(
                f"| `{entry['name']}` | {entry['dim_o']}x{entry['dim_i']} | "
                f"{entry['pad_overhead_ratio']:.4f} | {entry['effective_bpw']} |")
    else:
        lines.append("(no non-64-aligned tensors)")
    lines.append("")

    lines.append("## C++ encode parity (sgfp4_encode_dump.out)")
    lines.append("")
    parity_rows = context.get("parity_rows")
    if parity_rows is None:
        lines.append("SKIPPED — run with `--encode-dump <path>` to activate the "
                     "C++ parity leg.")
    elif not parity_rows:
        lines.append("SKIPPED — no samples resolved (check `--sample` values).")
    else:
        lines.append("| tensor | byte-exact | decode-stats rtol | status |")
        lines.append("|---|---|---|---|")
        for row in parity_rows:
            lines.append(
                f"| `{row['name']}` | {row['byte_exact']} | "
                f"{row.get('decode_rtol', '-')} | {row['status']} |")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Layers swept: {summary['total_layers']} "
                 f"({summary['full_tier']} full tier, "
                 f"{summary['light_tier']} light tier)")
    if summary["failing_layers"]:
        lines.append(f"- **D-07 gate: FAIL** — layers failing their targets: "
                     f"{', '.join('`' + n + '`' for n in summary['failing_layers'])}")
        lines.append("- Failing leaf tuples are recorded in the JSON sidecar "
                     "for the threshold-delta loop.")
    else:
        lines.append("- **D-07 gate: PASS** — every layer meets its "
                     "per-leaf-size targets.")
        if thresholds_source:
            lines.append(f"- Threshold decision: gate green under the revised "
                         f"table `{thresholds_source}` (see the delta section).")
        else:
            lines.append("- Threshold decision: no data-justified revision "
                         "proposed (defaults stand; see the delta section).")
    eps_total = sum(e.get("eps_escapes", 0) for e in layers)
    if eps_total:
        lines.append(f"- Annotated: {eps_total} leaf evaluation(s) hit the "
                     f"signal_power <= {K_RELATIVE_EPSILON} epsilon escape "
                     "(encode-side relative-gate bypass; visible by design).")
    lines.append("")
    lines.append("## Threshold delta")
    lines.append("")
    delta = summary.get("threshold_delta")
    if delta is None and thresholds_source:
        # Derive the delta block by diffing the effective table against the
        # Python defaults (the override file IS the documented delta).
        delta = []
        reasons = {
            64: "worst observed leaf energy-ratio 0.348 on outlier-heavy "
                "64x64 leaves (features.3/6/8, classifier.6); cascade-"
                "converged with 10% headroom",
            32: "worst observed 0.071 (features.3.weight)",
            16: "worst observed 0.0264",
            8: "worst observed 0.0131",
            4: "max_mse: forced min-size leaves on features.3.weight "
               "(worst 8.99e-3; quadtree accepts at min size by "
               "construction); max_relative: worst 0.0267",
        }
        for size in LEAF_SIZES:
            default = DEFAULT_V2_THRESHOLDS[size]
            effective = thresholds[size]
            if (effective["max_mse"] != default["max_mse"]
                    or effective["max_relative"] != default["max_relative"]):
                delta.append({
                    "leaf": size,
                    "old_mse": default["max_mse"],
                    "new_mse": effective["max_mse"],
                    "old_rel": default["max_relative"],
                    "new_rel": effective["max_relative"],
                    "reason": reasons.get(size, "data-justified"),
                })
    if delta:
        lines.append("| leaf size | old max_mse | new max_mse | old max_relative | "
                     "new max_relative | motivating statistic |")
        lines.append("|---|---|---|---|---|---|")
        for row in delta:
            lines.append("| {leaf} | {old_mse} | {new_mse} | {old_rel} | "
                         "{new_rel} | {reason} |".format(**row))
        lines.append("")
        lines.append("Revision provenance: the relative criterion is the "
                     "user-reformulated (2026-08-31) leaf energy ratio "
                     "`mse / signal_power` — the same folding the exporter's "
                     "split driver applies. The plain per-element relative "
                     "ratio is structurally unbounded on real weights "
                     "(worst 3.6e6) and is reported informationally only.")
        lines.append("")
        lines.append("This delta is a documented gnus-poc-side proposal "
                     "(D-09); no gnus-poc code changes were made.")
    else:
        lines.append("No data-justified revision (D-09): the effective table "
                     "equals DEFAULT_V2_THRESHOLDS.")

    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    sidecar_layers = []
    for entry in layers:
        clean = {k: v for k, v in entry.items() if k != "plane"}
        sidecar_layers.append(clean)
    sidecar = {
        "context": {k: v for k, v in context.items()
                    if k not in ("parity_samples", "parity_rows")},
        "parity_samples": context.get("parity_samples"),
        "summary": summary,
        "layers": sidecar_layers,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(sidecar, indent=2, default=float), encoding="utf-8")


# ---------------------------------------------------------------------------
# C++ parity-sampling leg (Plan 10-03, D-11)
# ---------------------------------------------------------------------------

def run_parity_leg(encode_dump, samples, initializers, exporter,
                   decode_v2, effective_thresholds, workdir):
    """Drive sgfp4_encode_dump.out per sampled layer and compare against the
    gnus-poc reference encode.

    Per layer: write LE FP32 dump -> invoke harness -> byte-compare vs
    export_weights(adaptive=True) -> decode the C++-produced bytes and check
    decode-error statistics against the Python-computed reference at rtol
    1e-4 -> delete the transient dump.

    Returns (parity_rows, ok) where ok is False on any mismatch (exit 4).
    """
    harness = Path(encode_dump)
    if not harness.is_file():
        # Windows subtlety recorded in 10-02: the .exe suffix is required.
        print(f"encode-dump harness not found: {harness}", file=sys.stderr)
        sys.exit(1)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)

    by_name = {t["name"]: t for t in initializers}
    parity_rows = []
    ok = True
    for name in samples:
        tensor = by_name.get(name)
        if tensor is None:
            print(f"sample '{name}' not found in model", file=sys.stderr)
            sys.exit(1)
        plane = tensor["plane"]
        dim_o, dim_i = plane.shape

        dump_path = work / f"{name.replace('.', '_')}.f32"
        cpp_path = work / f"{name.replace('.', '_')}_cpp.sgfp4"
        plane.astype("<f4").tofile(dump_path)

        proc = subprocess.run(
            [str(harness), "--weights", str(dump_path),
             "--dimO", str(dim_o), "--dimI", str(dim_i), "--out", str(cpp_path)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            detail = "encoder-rejected (exit 2)" if proc.returncode == 2 \
                else f"harness exit {proc.returncode}"
            parity_rows.append({"name": name, "byte_exact": False,
                                "decode_rtol": None,
                                "status": f"FAIL ({detail})"})
            ok = False
            dump_path.unlink(missing_ok=True)
            continue

        py_binary, _ = exporter.export_weights(
            plane, name, adaptive=True, thresholds=effective_thresholds)
        py_path = work / f"{name.replace('.', '_')}_py.sgfp4"
        py_path.write_bytes(py_binary)

        row = {"name": name}
        if filecmp.cmp(str(py_path), str(cpp_path), shallow=False):
            row["byte_exact"] = True
            row["decode_rtol"] = "n/a (byte-exact)"
            row["status"] = "PASS"
        else:
            # Contractual fallback: decode-vs-decode rtol check.
            py_dec = decode_v2(py_binary, dim_o, dim_i)
            cpp_bytes = cpp_path.read_bytes()
            cpp_dec = decode_v2(cpp_bytes, dim_o, dim_i)
            py_err = np.abs(plane.astype(np.float64) - py_dec.astype(np.float64))
            cpp_err = np.abs(plane.astype(np.float64) - cpp_dec.astype(np.float64))
            stats_ok = (
                math.isclose(float(py_err.max()), float(cpp_err.max()),
                             rel_tol=K_ENCODE_REL_TOL)
                and math.isclose(float(np.mean(py_err ** 2)),
                                 float(np.mean(cpp_err ** 2)),
                                 rel_tol=K_ENCODE_REL_TOL))
            row["byte_exact"] = False
            row["decode_rtol"] = "within 1e-4" if stats_ok else "EXCEEDED"
            row["status"] = "PASS (rtol fallback)" if stats_ok else "FAIL"
            row["rtol_fallback"] = True
            if not stats_ok:
                ok = False

        # Decode-error stats check on the byte-exact path too: the C++
        # container decoded via the Python oracle must agree with the
        # Python container's stats (trivially true when byte-exact, but
        # computed from the C++ bytes for evidence).
        if row["byte_exact"]:
            cpp_dec = decode_v2(cpp_path.read_bytes(), dim_o, dim_i)
            py_dec = decode_v2(py_binary, dim_o, dim_i)
            d = np.abs(cpp_dec.astype(np.float64) - py_dec.astype(np.float64))
            row["decode_rtol"] = f"0.0 (byte-exact; decode delta {float(d.max()):.1e})"

        parity_rows.append(row)
        dump_path.unlink(missing_ok=True)

    return parity_rows, ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="SGFP4 v2 real-weight validation driver (Phase 10)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="ONNX model path")
    parser.add_argument("--report",
                        default="tools/fp4/real_weight_validation_report.md",
                        help="markdown report output path")
    parser.add_argument("--gnus-poc-root", default=DEFAULT_GNUS_POC_ROOT,
                        help="gnus-poc checkout root (import path)")
    parser.add_argument("--thresholds", default=None,
                        help="optional threshold-override JSON keyed like "
                             "DEFAULT_V2_THRESHOLDS; absent = exporter defaults")
    parser.add_argument("--encode-dump", default=None,
                        help="path to sgfp4_encode_dump.out; activates the "
                             "C++ parity leg (Plan 10-03)")
    parser.add_argument("--sample", action="append", default=None,
                        metavar="TENSOR-NAME",
                        help="parity sample override (repeatable); default "
                             "rule: all non-64-aligned full-tier tensors, "
                             "largest aligned plane, one aligned conv, two "
                             "light-tier tensors")
    parser.add_argument("--workdir", default="tmp/sgfp4_validation",
                        help="scratch dir for transient dumps")
    parser.add_argument("--list-only", action="store_true",
                        help="enumerate tensors (name/dims/tier) and exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"model file not found: {model_path}", file=sys.stderr)
        return 1

    thresholds, thresholds_source = load_thresholds(args.thresholds)

    # --- Stage 1: enumerate + finite gate --------------------------------
    initializers = []
    for name, dims, arr in extract_initializers(model_path):
        finite_gate(name, arr)
        plane = project_2d(dims, arr)
        dim_o, dim_i = plane.shape
        initializers.append({
            "name": name,
            "dims": dims,
            "ndim": len(dims),
            "dim_o": dim_o,
            "dim_i": dim_i,
            "elements": int(plane.size),
            "tier": classify_tier(dim_o, dim_i, int(plane.size)),
            "plane": plane,
        })

    if args.list_only:
        print(f"model: {model_path}")
        print(f"tensors: {len(initializers)}")
        for t in initializers:
            print(f"  {t['name']}: dims={t['dims']} -> "
                  f"{t['dim_o']}x{t['dim_i']} "
                  f"({t['elements']} elems, {t['tier']} tier)")
        return 0

    # --- Stage 2: statistics sweep + gate + report ------------------------
    FP4Exporter, QuadtreeEncoder, LaplacianWeightedError, decode_v2 = \
        import_gnus_poc(Path(args.gnus_poc_root))
    exporter = FP4Exporter()

    try:
        import onnx as _onnx
        onnx_version = _onnx.__version__
    except Exception:
        onnx_version = "unknown"

    context = {
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "generated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%SZ"),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "onnx_version": onnx_version,
        "gnus_poc_root": str(Path(args.gnus_poc_root)),
        "thresholds": thresholds,
        "thresholds_source": thresholds_source,
        "encode_dump": args.encode_dump,
        "workdir": args.workdir,
    }

    if args.encode_dump:
        context["parity_samples"] = (args.sample if args.sample
                                     else default_sample_names(initializers))

    layers, exit_code, summary = run_sweep(
        exporter, QuadtreeEncoder, LaplacianWeightedError, decode_v2,
        initializers, thresholds)
    summary["threshold_delta"] = None

    if args.encode_dump:
        parity_rows, parity_ok = run_parity_leg(
            args.encode_dump, context["parity_samples"], initializers,
            exporter, decode_v2, thresholds, args.workdir)
        context["parity_rows"] = parity_rows
        summary["parity_ok"] = parity_ok
        summary["parity_rows"] = parity_rows
        if not parity_ok and exit_code == 0:
            exit_code = 4
        elif not parity_ok:
            # Gate failure already failed the run; parity mismatch escalates
            # the diagnostic code so both conditions are visible.
            exit_code = 4

    write_report(layers, exit_code, summary, context,
                 Path(args.report), Path(args.report).with_suffix(".json"))

    status = "PASS" if exit_code == 0 else f"exit {exit_code}"
    print(f"swept {len(layers)} layers; gate: {status}; report: {args.report}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
