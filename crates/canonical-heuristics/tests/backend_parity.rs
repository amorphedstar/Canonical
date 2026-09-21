//! Compare ONNX (burn-onnx) vs Burn vs the Python export dump. tch tests need `--features tch`.

use std::path::{Path, PathBuf};

use canonical_compat::ai::Example;
use canonical_compat::ir::{BindMap, Position};
use canonical_core::compiler::{compile, CompilationGoals, COMPILATION};
use canonical_core::core::{Type, ES};
use canonical_core::prover::Prover;
use canonical_core::stats::reset;
use canonical_heuristics::encoding::{encode_path, sequence_paths, PositionMatrices};
use canonical_heuristics::weights::{load_file, max_abs, mean_abs};
use canonical_heuristics::{burn_model, HeuristicModel};

fn export_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../CanonicalHeuristics/export_rust")
}

fn example_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../CanonicalHeuristics/tests/abs_cases_425.bin")
}

fn load_example(
    dir: &Path,
) -> (
    Example,
    Vec<Vec<Position>>,
    Vec<f32>,
    Vec<f32>,
    serde_json::Value,
) {
    let example = Example::load(example_path().to_string_lossy().into_owned());
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("parity.json")).unwrap()).unwrap();
    let paths: Vec<Vec<Position>> = meta["bind_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| example.binds[v.as_str().unwrap()].clone())
        .collect();
    let dump = load_file(&dir.join("parity.safetensors")).unwrap();
    (
        example,
        paths,
        dump["x"].data.clone(),
        dump["logits"].data.clone(),
        meta,
    )
}

#[allow(dead_code)]
fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn seeded_steps(n: usize, seed: u64) -> Vec<i64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) % 12) as i64
        })
        .collect()
}

fn embed_packed(
    pm: &PositionMatrices,
    left: &[i64],
    right: &[i64],
    seq: usize,
    path_len: usize,
) -> Vec<f32> {
    let mut rows = Vec::with_capacity(seq * 2 * pm.dim);
    for i in 0..seq {
        let lp: Vec<u8> = left[i * path_len..(i + 1) * path_len]
            .iter()
            .map(|&s| s as u8)
            .collect();
        let rp: Vec<u8> = right[i * path_len..(i + 1) * path_len]
            .iter()
            .map(|&s| s as u8)
            .collect();
        rows.extend(pm.embed_token(&lp, &rp));
    }
    rows
}

fn assert_close(label: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length {} vs {}",
        a.len(),
        b.len()
    );
    let max = max_abs(a, b);
    let mean = mean_abs(a, b);
    println!(
        "{label}: max |Δ| = {max:.3e}, mean |Δ| = {mean:.3e} (n={})",
        a.len()
    );
    assert!(
        max <= tol,
        "{label}: max |Δ| = {max:.3e} (mean {mean:.3e}) exceeds {tol:.3e}"
    );
}

/// Resolve `meta["goals"]`/`meta["premises"]` bind names to indices via `meta["name_to_index"]`.
fn goals_and_premises(meta: &serde_json::Value) -> (Vec<i64>, Vec<i64>) {
    let name_to_index: std::collections::HashMap<&str, usize> = meta["name_to_index"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_u64().unwrap() as usize))
        .collect();
    let indices = |key: &str| -> Vec<i64> {
        meta[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| name_to_index[v.as_str().unwrap()] as i64)
            .collect()
    };
    (indices("goals"), indices("premises"))
}

fn gather_sub(logits: &[f32], n: usize, goals: &[i64], premises: &[i64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(goals.len() * premises.len());
    for &g in goals {
        for &p in premises {
            out.push(logits[(g as usize) * n + p as usize]);
        }
    }
    out
}

fn load_onnx() -> (HeuristicModel, burn_model::HeuristicModel, PathBuf) {
    let dir = export_dir();
    assert!(
        dir.join("transformer.safetensors").exists(),
        "missing export at {}",
        dir.display()
    );
    (
        HeuristicModel::load(&dir).unwrap(),
        burn_model::HeuristicModel::load(&dir).unwrap(),
        dir,
    )
}

#[test]
fn python_dump_matches_onnx_and_burn() {
    let (onnx, burn, dir) = load_onnx();
    let (example, paths, xref, lref, meta) = load_example(&dir);

    let burn_x = burn.embed_vec(&example.tokens, &paths);
    assert_close("burn vs python embedding", &burn_x, &xref, 1e-5);

    let n = burn_x.len() / burn.config.input_dim;
    let (goals, premises) = goals_and_premises(&meta);
    let (left, right) = sequence_paths(&example.tokens, &paths);
    let onnx_sub = onnx.scores_from_paths(&left, &right, &goals, &premises);
    let burn_l = burn.logits_from_slice(&burn_x, n, burn.config.input_dim);
    let onnx_tol = if cfg!(feature = "cuda") { 1e-2 } else { 1e-3 };
    assert_close(
        "onnx vs python submatrix",
        &onnx_sub,
        &gather_sub(&lref, n, &goals, &premises),
        onnx_tol,
    );
    assert_close("burn vs python logits", &burn_l, &lref, 5e-4);
    assert_close(
        "onnx vs burn submatrix",
        &onnx_sub,
        &gather_sub(&burn_l, n, &goals, &premises),
        onnx_tol,
    );
}

#[test]
fn random_sequences_match_onnx_submatrix() {
    let (onnx, burn, dir) = load_onnx();
    let pm = PositionMatrices::load(&dir.join("embedding.safetensors")).unwrap();
    let path_len = 16;
    let dim = onnx.config.input_dim;
    for (seq, seed) in [(4usize, 1u64), (8, 2), (16, 3)] {
        let left = seeded_steps(seq * path_len, seed);
        let right = seeded_steps(seq * path_len, seed.wrapping_add(17));
        let idx: Vec<i64> = (0..seq as i64).collect();
        let a = onnx.scores_from_packed(&left, &right, seq, path_len, &idx, &idx);
        let x = embed_packed(&pm, &left, &right, seq, path_len);
        let b = burn.logits_from_slice(&x, seq, dim);
        let tol = if cfg!(feature = "cuda") { 2e-3 } else { 5e-4 };
        assert_close(&format!("random seq={seq}"), &a, &b, tol);
    }
}

#[test]
fn path_steps_and_indexed_powers_match() {
    let (onnx, burn, _) = load_onnx();
    let cases: Vec<Vec<Position>> = vec![
        vec![],
        vec![Position::Type],
        vec![Position::LHS],
        vec![Position::RHS],
        vec![Position::Type, Position::LHS],
        vec![Position::Rule(0)],
        vec![Position::Rule(1)],
        vec![Position::Param(3), Position::Arg(2)],
        vec![Position::Let(0), Position::Type, Position::Rule(4)],
        (0..12).map(Position::Arg).collect(),
        (0..20).map(Position::Arg).collect(),
    ];
    let onnx_tol = if cfg!(feature = "cuda") { 2e-3 } else { 5e-4 };
    for path in cases {
        let steps = encode_path(&path);
        let steps_ref = std::slice::from_ref(&steps);
        let a = onnx.scores_from_paths(steps_ref, steps_ref, &[0], &[0]);
        let x = burn.embed_vec(&[], std::slice::from_ref(&path));
        let b = burn.logits_from_slice(&x, 1, burn.config.input_dim);
        assert_close(&format!("path {path:?}"), &a, &b, onnx_tol);
    }
}

#[test]
fn bind_softmax_rows_match() {
    let (onnx, burn, dir) = load_onnx();
    let (example, paths, _, _, meta) = load_example(&dir);
    let (goals, premises) = goals_and_premises(&meta);
    let (left, right) = sequence_paths(&example.tokens, &paths);
    let n = left.len();
    let sub = onnx.scores_from_paths(&left, &right, &goals, &premises);
    let x = burn.embed_vec(&example.tokens, &paths);
    let burn_l = burn.logits_from_slice(&x, n, burn.config.input_dim);
    assert!(!goals.is_empty() && !premises.is_empty());
    let n_p = premises.len();
    for (gi, &g) in goals.iter().enumerate().take(32) {
        let a = onnx.softmax_row(&sub, n_p, gi);
        let b = burn.softmax_row(
            &burn_l,
            n,
            g as usize,
            &premises.iter().map(|p| *p as usize).collect::<Vec<_>>(),
        );
        let sm_tol = if cfg!(feature = "cuda") { 2e-3 } else { 1e-5 };
        assert_close(&format!("softmax goal {g}"), &a, &b, sm_tol);
        let sum: f32 = a.iter().sum();
        assert!((sum - 1.0).abs() < sm_tol, "softmax not normalized: {sum}");
    }
}

#[cfg(feature = "tch")]
#[test]
fn gelu_and_linear_primitives_match() {
    let x = seeded(256, 99);
    let xt = tch::Tensor::from_slice(&x);
    let tch_g: Vec<f32> = Vec::try_from(xt.gelu("none")).unwrap();
    let bt = burn::tensor::Tensor::<burn::backend::NdArray<f32>, 1>::from_data(
        burn::tensor::TensorData::new(x.clone(), [x.len()]),
        &Default::default(),
    );
    let burn_g = burn_model::to_vec(burn::tensor::activation::gelu(bt));
    assert_close("gelu", &tch_g, &burn_g, 1e-6);

    let dir = export_dir();
    let w = load_file(&dir.join("transformer.safetensors")).unwrap();
    let weight = &w["input_proj.weight"];
    let bias = &w["input_proj.bias"];
    let (out, inp) = (weight.shape[0], weight.shape[1]);
    let x = seeded(inp, 7);
    let mut y = vec![0.0f32; out];
    for o in 0..out {
        let mut s = bias.data[o];
        for i in 0..inp {
            s += x[i] * weight.data[o * inp + i];
        }
        y[o] = s;
    }
    let xt = tch::Tensor::from_slice(&x).unsqueeze(0);
    let wt = tch::Tensor::from_slice(&weight.data).reshape([out as i64, inp as i64]);
    let bt = tch::Tensor::from_slice(&bias.data);
    let tch_y: Vec<f32> = Vec::try_from(xt.matmul(&wt.tr()).squeeze() + bt).unwrap();
    assert_close("input_proj vs numpy-style", &tch_y, &y, 1e-5);
}

#[test]
fn config_agrees_with_weight_shapes() {
    let (onnx, burn, dir) = load_onnx();
    assert_eq!(onnx.config.d_model, burn.config.d_model);
    assert_eq!(onnx.config.qk_dim, burn.config.qk_dim);
    let w = load_file(&dir.join("transformer.safetensors")).unwrap();
    assert_eq!(
        w["input_proj.weight"].shape,
        vec![onnx.config.d_model, onnx.config.input_dim]
    );
    assert_eq!(
        w["qk_proj.weight"].shape,
        vec![2 * onnx.config.qk_dim, onnx.config.d_model]
    );
    assert_eq!(
        w["encoder.layers.0.self_attn.in_proj_weight"].shape,
        vec![3 * onnx.config.d_model, onnx.config.d_model]
    );
    assert_eq!(w.len(), 77);
}

#[cfg(feature = "cuda")]
#[test]
fn burn_cuda_matches_cpu_logits() {
    use burn::backend::Cuda;
    use burn::tensor::{Tensor, TensorData};

    type Gpu = Cuda<f32, i32>;
    let dir = export_dir();
    let cfg: canonical_heuristics::ModelConfig =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let cpu = burn_model::CanonicalTransformer::<burn::backend::NdArray<f32>>::load(
        &dir.join("transformer.safetensors"),
        &cfg,
    )
    .unwrap();
    let device = Default::default();
    let gpu = burn_model::CanonicalTransformer::<Gpu>::load_on(
        &dir.join("transformer.safetensors"),
        &cfg,
        &device,
    )
    .unwrap();
    let dim = cfg.input_dim;
    for seq in [32usize, 64, 128] {
        let x = seeded(seq * dim, 1);
        let cpu_y = burn_model::to_vec(cpu.forward(Tensor::from_data(
            TensorData::new(x.clone(), [seq, dim]),
            &Default::default(),
        )));
        let gpu_y = burn_model::to_vec(gpu.forward(Tensor::<Gpu, 2>::from_data(
            TensorData::new(x, [seq, dim]),
            &device,
        )));
        assert_close(&format!("cuda vs cpu seq={seq}"), &gpu_y, &cpu_y, 2e-3);
    }
}

#[test]
fn model_scores_replace_uniform_compilation() {
    let dir = export_dir();
    let model = HeuristicModel::load(&dir).unwrap();
    let example = Example::load(example_path().to_string_lossy().into_owned());
    let mut binds = BindMap::default();
    let mut tokens = Vec::new();
    let (tb, problem_bind) =
        example
            .problem
            .to_problem(example.name.clone(), &mut binds, &mut tokens);

    reset();
    let mut owned_linked = Vec::new();
    let _prover = Prover::new(
        tb.downgrade(),
        problem_bind.downgrade(),
        &mut owned_linked,
        None,
    );
    let uniform = COMPILATION.load();
    let n_uniform: usize = uniform.values().map(|v| v.len()).sum();
    assert!(n_uniform > 0, "compile produced no unifications");
    assert!(
        uniform
            .values()
            .flat_map(|v| v.iter())
            .all(|(_, info)| info.weight() == 1.0),
        "Prover::new(..., None) should start with uniform weights"
    );

    let collected =
        CompilationGoals::collect(Type(tb.downgrade(), ES::new(), problem_bind.downgrade()));
    let scores = model.bind_scores(
        &tokens,
        &binds.bind_paths(),
        &collected.goal_binds(),
        &collected.premise_binds(),
    );
    assert!(!scores.is_empty(), "model returned no goal/premise scores");
    compile(
        Type(tb.downgrade(), ES::new(), problem_bind.downgrade()),
        Some(&scores),
    );

    let updated = COMPILATION.load();
    let n_updated: usize = updated.values().map(|v| v.len()).sum();
    assert_eq!(n_uniform, n_updated, "hot-swap must not drop unifications");
    let non_uniform = updated
        .values()
        .flat_map(|v| v.iter())
        .filter(|(_, info)| (info.weight() - 1.0).abs() > 1e-9)
        .count();
    assert!(
        non_uniform > 0,
        "expected model probabilities in COMPILATION, all weights still 1.0"
    );
}
