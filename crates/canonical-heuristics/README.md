# canonical-heuristics

Neural unification heuristic for Canonical. The runtime path is ONNX imported
with `burn-onnx` (build-time codegen). Weights are a CanonicalHeuristics
PyTorch checkpoint exported by `CanonicalHeuristics/scripts/export_onnx.py`
(and the older `export_for_rust.py` for embeddings / tch / native Burn).

Path embeddings multiply a unit root vector by one orthogonal matrix per
step. Each step is an integer `0..=10` selecting among the 11 trained
matrices (Type/LHS/RHS and right/down for Rule, Param, Let, Arg). A token
is `concat(ast_path, declaration_path)`. Those embeddings live in the ONNX
graph: Rust packs left/right path steps as `[seq, path_len]` int64
(pad `11` = identity; `path_len` is the longest path in the batch) plus
goal/premise indices, and the graph returns the goal×premise submatrix of
unification logits. The path walk is an ONNX `Scan`, so there is no baked-in
maximum depth.

Search starts immediately with uniform `COMPILATION` weights. The model runs
on a background thread; when scores are ready and search is still running,
`compile` swaps them into `COMPILATION` (later unifications see the new
probabilities). `ModelConfig` (temperature, `max_seq_len`, ...) loads from
`CANONICAL_HEURISTICS_WEIGHTS/config.json`, or from
`CanonicalHeuristics/export_rust` next to this repo, at process start. If
neither directory has a `config.json`, compilation stays uniform.

For the ONNX backend specifically, the actual transformer weights are baked
into the binary at *build* time (`build.rs` runs `burn-onnx` codegen against
`onnx/transformer.onnx`); `CANONICAL_HEURISTICS_WEIGHTS` only overrides them
at runtime if that directory also contains a pre-converted `transformer.bpk`
(nothing in this repo produces one today). So switching to a newly exported
ONNX model normally means re-running `export_onnx.py --copy .../onnx/transformer.onnx`
and rebuilding, not just repointing the env var. The `burn_model` / `tch`
backends *do* load their safetensors/TorchScript weights straight from
`CANONICAL_HEURISTICS_WEIGHTS` at runtime, since they aren't code-generated.

Inference uses Burn **NdArray** (CPU) by default. Rebuild with `--features cuda`
to run the same burn-onnx `Model` on `Cuda<f32, i32>` (CubeCL + fusion + autotune).
`canonical_lean` forwards that as `--features cuda`.

The previous backends remain for parity:
- `tch` (`--features tch`): TorchScript `transformer.ts`
- `burn_model`: native Burn encoder + Q/K head, `transformer.safetensors`

```
python CanonicalHeuristics/scripts/export_onnx.py \
  --dir CanonicalHeuristics/export_rust \
  --out Canonical/crates/canonical-heuristics/onnx/transformer.onnx

cargo test -p canonical-heuristics --release --test backend_parity
cargo test -p canonical-heuristics --release --features cuda --test backend_parity
```
