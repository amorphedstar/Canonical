//! Fills `WEIGHTS` from a problem's `Tokenization` with a transformer trained in
//! CanonicalHeuristics. Its `scripts/export_onnx.py` writes `embedding.bin`, which `run` embeds
//! the tokenization's paths with, and `model.onnx`, which takes the embeddings and returns the
//! weight matrix; burn-onnx compiles it in, weights included (see `build.rs`). It runs on the
//! GPU when there is one it can use.

use std::iter;
use std::panic;
use std::sync::{Arc, Mutex};
use std::thread;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn::{Dispatch, DispatchDevice};
use once_cell::sync::Lazy;

use crate::core::{Bind, Position};
use crate::heuristic::WEIGHTS;
use crate::memory::W;

#[derive(Default)]
pub struct Tokenization {
    pub tokens: Vec<(Vec<Position>, W<Bind>)>,
    pub goals: Vec<W<Bind>>,
    pub premises: Vec<W<Bind>>
}

#[allow(dead_code, clippy::all)]
mod model {
    extern crate alloc;
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}

type Model = model::Model<Dispatch>;

/// The model saw at most this many rows (tokens plus binds) in training.
const MAX_ROWS: usize = 5000;

/// A path step as symbols, the matrices `run` multiplies the root by in turn: `Type`, `LHS`, `RHS`
/// are 0, 1, 2, and `Kind(i)` is Kind's "right" i + 1 times, then its "down" (3 and 7 for `Rule`,
/// 4 and 8 `Param`, 5 and 9 `Let`, 6 and 10 `Arg`). `PAD`, the identity, pads paths to one length.
fn step(position: &Position) -> impl Iterator<Item = i64> {
    let (right, repeats, last) = match *position {
        Position::Type => (0, 0, 0),
        Position::LHS => (0, 0, 1),
        Position::RHS => (0, 0, 2),
        Position::Rule(i) => (3, i + 1, 7),
        Position::Param(i) => (4, i + 1, 8),
        Position::Let(i) => (5, i + 1, 9),
        Position::Arg(i) => (6, i + 1, 10),
    };
    iter::repeat_n(right, repeats).chain([last])
}

const PAD: i64 = 11;

/// The model, on wgpu's GPU (Metal, Vulkan or DX12), loaded when first needed. The lock
/// also keeps problems from running at once.
static MODEL: Lazy<Mutex<(Model, DispatchDevice)>> = Lazy::new(|| {
    let on = |device: DispatchDevice| (Model::from_embedded(&device), device);
    // Burn panics when wgpu has no adapter at all, not even a software one: use the CPU.
    let model = panic::catch_unwind(|| on(DispatchDevice::default()))
        .unwrap_or_else(|_| on(DispatchDevice::NdArray(Default::default())));
    Mutex::new(model)
});

/// The model's input for `tokens`, rows' left then right paths as symbols (a bind's declaration
/// twice, a token's position then its bind's), with the goal and premise counts; or `None` if it
/// has no premises or is too big.
fn inputs(tokens: &Tokenization) -> Option<(TensorData, usize, usize)> {
    let rows = tokens.tokens.len() + tokens.goals.len() + tokens.premises.len();
    if tokens.premises.is_empty() || rows > MAX_ROWS {
        return None;
    }
    let binds = tokens.goals.iter().chain(&tokens.premises).map(|b| &b.borrow().position[..]);
    let left = binds.clone().chain(tokens.tokens.iter().map(|(p, _)| &p[..]));
    let right = binds.chain(tokens.tokens.iter().map(|(_, b)| &b.borrow().position[..]));
    let paths: Vec<&[Position]> = left.chain(right).collect();
    let length = paths.iter().map(|p| p.iter().flat_map(step).count()).max().unwrap_or(0).max(1);
    let data: Vec<i64> =
        paths.iter().flat_map(|p| p.iter().flat_map(step).chain(iter::repeat(PAD)).take(length)).collect();
    Some((TensorData::new(data, [paths.len(), length]), tokens.goals.len(), tokens.premises.len()))
}

/// The embedding's 12 symbol matrices side by side, W [64, 768], then its root vector [64].
static EMBEDDING: Lazy<Vec<f32>> =
    Lazy::new(|| include_bytes!("../embedding.bin").chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect());

fn run((model, device): &(Model, DispatchDevice), (paths, goals, premises): (TensorData, usize, usize)) -> Vec<Vec<f32>> {
    let paths = Tensor::<Dispatch, 2, Int>::from_data(paths, device);
    let n = paths.dims()[0];
    // The root times each symbol's matrix in turn, as `embed` in the export: row 12 i + s of
    // (v W).reshape([12 n, 64]) is path i's v times symbol s's matrix.
    let (w, root) = EMBEDDING.split_at(64 * 768);
    let w = Tensor::<Dispatch, 1>::from_floats(w, device).reshape([64, 768]);
    let mut v = Tensor::<Dispatch, 1>::from_floats(root, device).expand([n, 64]);
    let rows = paths.transpose() + Tensor::arange_step(0..12 * n as i64, 12, device).unsqueeze();
    for r in rows.iter_dim(0) {
        v = v.matmul(w.clone()).reshape([12 * n, 64]).select(0, r.reshape([n]));
    }
    // Each row is its left path's embedding, then its right path's.
    let x = Tensor::cat(v.chunk(2, 0), 1);
    let weight = model.forward(x.clone().slice(0..goals), x.clone().slice(goals..goals + premises), x.slice(goals + premises..));
    let weight = weight.into_data().to_vec::<f32>().unwrap();
    // CubeCL keeps the memory it allocates, so without this the GPU fills up over a few dozen
    // problems and inference then fails. `Dispatch` doesn't forward `memory_cleanup`.
    if let DispatchDevice::Wgpu(device) = device {
        <burn::backend::Wgpu as Backend>::memory_cleanup(device);
    }
    let _ = Dispatch::sync(device);
    weight.chunks(premises).map(<[f32]>::to_vec).collect()
}

/// `tokens`' goal × premise weights, indexed by `Bind::index`, or `None` if the model
/// doesn't apply.
pub fn weight(tokens: &Tokenization) -> Option<Vec<Vec<f32>>> {
    inputs(tokens).map(|inputs| run(&MODEL.lock().unwrap(), inputs))
}

/// Resets `WEIGHTS` to uniform, then sets it to `weight(tokens)` from a background thread
/// unless another problem has reset it since. Search reads `WEIGHTS` as it goes, so it
/// needn't wait.
pub fn start(tokens: &Tokenization) {
    // A new allocation, so the thread can tell whether `WEIGHTS` is still this problem's.
    let uniform = Arc::new(Vec::new());
    WEIGHTS.store(uniform.clone());
    let Some(inputs) = inputs(tokens) else { return };
    thread::spawn(move || {
        let model = MODEL.lock().unwrap();
        // Problems started in quick succession queue up here: only the latest runs.
        if !Arc::ptr_eq(&WEIGHTS.load(), &uniform) {
            return;
        }
        let weight = run(&model, inputs);
        WEIGHTS.compare_and_swap(&uniform, Arc::new(weight));
    });
}
