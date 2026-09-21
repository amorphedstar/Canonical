use std::path::PathBuf;

use canonical_compat::ai::Example as IrExample;
use canonical_heuristics::encoding::{encode_path, sequence_paths, PositionMatrices};
use canonical_heuristics::weights::{load_file, max_abs};
use canonical_heuristics::{Example, HeuristicModel};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .expect("usage: heuristics-parity WEIGHTS_DIR EXAMPLE.bin"),
    );
    let ir = IrExample::load(args.next().unwrap());
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("parity.json")).unwrap()).unwrap();
    let paths: Vec<_> = meta["bind_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| ir.binds[v.as_str().unwrap()].clone())
        .collect();
    let name_to_index: std::collections::HashMap<String, usize> = meta["name_to_index"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_u64().unwrap() as usize))
        .collect();

    let dump = load_file(&dir.join("parity.safetensors")).unwrap();
    let xref = &dump["x"].data;
    let lref = &dump["logits"].data;

    let model = HeuristicModel::load(&dir).unwrap();
    let cpu = PositionMatrices::load(&dir.join("embedding.safetensors")).unwrap();
    let x = cpu.embed_sequence(&ir.tokens, &paths);
    let n = x.len() / model.config.input_dim;
    println!("cpu vs python embedding {:8.3e}", max_abs(&x, xref));

    let goals: Vec<i64> = meta["goals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| name_to_index[g.as_str().unwrap()] as i64)
        .collect();
    let premises: Vec<i64> = meta["premises"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| name_to_index[p.as_str().unwrap()] as i64)
        .collect();
    let (left, right) = sequence_paths(&ir.tokens, &paths);
    let sub = model.scores_from_paths(&left, &right, &goals, &premises);
    let mut lref_sub = Vec::with_capacity(goals.len() * premises.len());
    for &g in &goals {
        for &p in &premises {
            lref_sub.push(lref[(g as usize) * n + p as usize]);
        }
    }
    println!("onnx vs python submatrix {:8.3e}", max_abs(&sub, &lref_sub));

    // Smoke test for the `Example` / `forward_example` convenience path (not used by
    // production `bind_scores`, so this isn't a numeric parity check against `lref_sub`:
    // it appends goal/premise binds as fresh sequence rows instead of indexing into the
    // token/bind sequence already built above, which changes the attention context).
    let example = Example {
        tokens: ir
            .tokens
            .iter()
            .map(|t| (encode_path(&t.position), encode_path(&t.declaration)))
            .collect(),
        goals: meta["goals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| encode_path(&ir.binds[g.as_str().unwrap()]))
            .collect(),
        premises: meta["premises"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| encode_path(&ir.binds[p.as_str().unwrap()]))
            .collect(),
    };
    let n_goals = example.goals.len();
    let n_premises = example.premises.len();
    let fwd = model.forward_example(&example);
    assert_eq!(
        fwd.len(),
        n_goals * n_premises,
        "forward_example: expected {n_goals}x{n_premises} scores, got {}",
        fwd.len()
    );
    println!("forward_example smoke test ok ({n_goals}x{n_premises} scores)");

    if max_abs(&x, xref) > 1e-4 || max_abs(&sub, &lref_sub) > 1e-3 {
        std::process::exit(1);
    }
    println!("parity ok");
}
