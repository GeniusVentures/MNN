# MNN fp4 tools

`sgfp4_inject` is the SGFP4 v2 model-artifact injection tool: given a
normally-converted `.mnn` plus one or more gnus-poc `fp4_exporter.py --adaptive`
output directories, it produces a final `.mnn` + external `.mnn.weight`-style
sidecar where each target weight tensor is produced by an
`OpType_SGFP4Dequant` node. The resulting artifact loads and runs via the
classic `Interpreter`/`Session` API.

`sgfp4_encode_dump.out` (Phase 10) is the C++ encode-parity harness: it reads
a raw little-endian FP32 row-major dump (`dimO*dimI` values) plus dims and
writes the SGFP4 v2 container via the shipped `sgfp4_encode::encode`. It
exists so `validate_real_weights.py --encode-dump` can parity-sample the
shipped C++ encoder itself against gnus-poc `fp4_exporter.py --adaptive` on
real weights; standalone smoke verified byte-exact containers on aligned and
non-64-aligned planes. Unlike `sgfp4_inject.out`, it links `sgfp4_encode`.

## Usage

```bash
sgfp4_inject --model <path> --niche-dir <dir> [--niche-dir <dir>...] --output <path>
```

- `--model` — input `.mnn` model (normal converter output, pre-injection).
- `--niche-dir` — one per target weight tensor; repeat the flag for each.
  Each directory is an unmodified `fp4_exporter.py --adaptive` output dir.
- `--output` — output `.mnn` path. The tool also emits the merged sidecar at
  `<output>.weight`.

Build target name: `sgfp4_inject.out`.

## Niche-dir / manifest input contract

Each `--niche-dir` must be an unmodified gnus-poc `fp4_exporter.py --adaptive`
output directory containing exactly one `*.sgfp4` container plus
`manifest.json`. Required manifest fields (the tool reads only these):

- `fp4_binary.path` — basename is cross-checked case-insensitively against the
  discovered `.sgfp4` file; it is never resolved literally (the exporter emits
  a root-relative path with backslashes).
- `fp4_binary.sha256` — SHA-256 integrity over the container bytes; mismatch
  is rejected.
- `fp4_binary.stats.shape` — exactly 2 positive integers `[dimO, dimI]`.

The SGFP4 v2 version gate is **byte-level**: the container's first 4 bytes
must be the `SGF4` magic and byte 4 the version byte `0x02`
(`sgfp4_is_v2_container`). It never consults the manifest's
`fp4_binary.format` label — its `"fp4_ultra_v0.2"` value is a known
terminology trap belonging to an unrelated E2M1 format. The format is called
**SGFP4 v2**.

## dims convention (`dims = {dimO, dimI}`)

- `dimO = shape[0]` (output rows), `dimI = shape[1]` (input columns); the tool
  writes `SGFP4DequantParamT.dims = {dimO, dimI}` on the injected op.
- Weights are 2-D `[out, in]` row matrices only; pairing against the base
  model is an **exact-shape match** on non-input 2-D vars.
- 64-multiple dims only (non-64-multiple tiling/padding is out of scope,
  deferred).
- No transposed-pairing tolerance (rejected by design).
- 2 or more shape matches hard-fail the run (same-shape disambiguation via
  tensor-name keying is deferred).

## Sidecar layout

- One merged sidecar file `<output>.weight`; per-op ranges never overlap.
- Each container is placed at a 16-byte-aligned offset (`sgfp4_align16`) and
  zero-padded to the next multiple of 16.
- Per-op descriptor: `SGFP4DequantParamT{magic = kSGFP4Magic, external =
  {offset, size}, dims = {dimO, dimI}}`; `op->externalPath` is set literally
  on the op (`createExecutionWithExternal` does NOT cover this op type —
  documented gotcha).
- Injected node naming: `weight → weight_sgfp4`.
- Every run performs an unconditional in-tool decode-vs-oracle verification
  (`dequant_sgfp4_container_cpu`) per injected node.

## Failure behavior

Any failure exits non-zero with an `MNN_ERROR` diagnostic and leaves **no**
`<output>` / `<output>.weight` behind — including removal of a previous run's
stale artifact at the same paths. Downstream consumers never see an artifact
that does not correspond to the inputs of the (failed) run.
