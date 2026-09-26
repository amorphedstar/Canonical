//! Fills `canonical_core::heuristic::WEIGHT` from a problem's `Tokenization` with a
//! transformer trained in CanonicalHeuristics. Its `scripts/export_onnx.py` writes
//! `model.onnx`, which takes the tokenization's paths and returns the weight matrix;
//! burn-onnx compiles it in, weights included.

use std::iter;
use std::sync::{Arc, Mutex};
use std::thread;

use burn::tensor::{Int, Tensor, TensorData};
use canonical_compat::ai::Tokenization;
use canonical_core::core::Position;
use canonical_core::heuristic::WEIGHT;

#[allow(dead_code, clippy::all)]
mod model {
    extern crate alloc;
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}

type B = burn::backend::NdArray<f32>;

/// The model saw at most this many rows (tokens plus binds) in training.
const MAX_ROWS: usize = 5000;

static MODEL: Mutex<Option<model::Model<B>>> = Mutex::new(None);
static PROBLEM: Mutex<u64> = Mutex::new(0);

/// A path step as the model reads it: the variant, in declaration order, and its index.
fn step(position: &Position) -> [i64; 2] {
    match *position {
        Position::Type => [0, 0],
        Position::Rule(i) => [1, i as i64],
        Position::LHS => [2, 0],
        Position::RHS => [3, 0],
        Position::Param(i) => [4, i as i64],
        Position::Let(i) => [5, i as i64],
        Position::Arg(i) => [6, i as i64],
    }
}

/// The model's inputs for `tokens`, or `None` if it has no premises or is too big.
fn inputs(tokens: &Tokenization) -> Option<[TensorData; 4]> {
    let rows = tokens.tokens.len() + tokens.goals.len() + tokens.premises.len();
    if tokens.premises.is_empty() || rows > MAX_ROWS {
        return None;
    }
    let paths: [Vec<&[Position]>; 4] = [
        tokens.tokens.iter().map(|(p, _)| &p[..]).collect(),
        tokens.tokens.iter().map(|(_, b)| &b.borrow().position[..]).collect(),
        tokens.goals.iter().map(|b| &b.borrow().position[..]).collect(),
        tokens.premises.iter().map(|b| &b.borrow().position[..]).collect(),
    ];
    let length = paths.iter().flatten().map(|p| p.len()).max().unwrap_or(0).max(1);
    Some(paths.map(|paths| {
        let data: Vec<i64> = paths
            .iter()
            .flat_map(|p| p.iter().map(step).chain(iter::repeat([7, 0])).take(length).flatten())
            .collect();
        TensorData::new(data, [paths.len(), length, 2])
    }))
}

fn run(model: &mut Option<model::Model<B>>, inputs: [TensorData; 4]) -> Vec<Vec<f32>> {
    let [positions, declarations, goals, premises] =
        inputs.map(|d| Tensor::<B, 3, Int>::from_data(d, &Default::default()));
    let weight = model
        .get_or_insert_with(model::Model::default)
        .forward(positions, declarations, goals, premises);
    let [_, n] = weight.dims();
    weight.into_data().to_vec::<f32>().unwrap().chunks(n).map(<[f32]>::to_vec).collect()
}

/// `tokens`' goal × premise weights, indexed by `Bind::index`, or `None` if the model
/// doesn't apply.
pub fn weight(tokens: &Tokenization) -> Option<Vec<Vec<f32>>> {
    inputs(tokens).map(|inputs| run(&mut MODEL.lock().unwrap(), inputs))
}

/// Resets `WEIGHT` to uniform, then sets it to `weight(tokens)` from a background thread
/// unless another problem has started by then. Search reads `WEIGHT` as it goes, so it
/// needn't wait.
pub fn start(tokens: &Tokenization) {
    let problem = {
        let mut problem = PROBLEM.lock().unwrap();
        *problem += 1;
        WEIGHT.store(Arc::new(Vec::new()));
        *problem
    };
    if let Some(inputs) = inputs(tokens) {
        thread::spawn(move || {
            let mut model = MODEL.lock().unwrap();
            // Problems started in quick succession queue up here: only the latest runs.
            if *PROBLEM.lock().unwrap() != problem {
                return;
            }
            let weight = run(&mut model, inputs);
            let latest = PROBLEM.lock().unwrap(); // held so `start` can't reset in between
            if *latest == problem {
                WEIGHT.store(Arc::new(weight));
            }
        });
    }
}
