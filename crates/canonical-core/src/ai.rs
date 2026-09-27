//! Fills `WEIGHTS` from a problem's `Tokenization` with a transformer trained in
//! CanonicalHeuristics. Its `scripts/export_onnx.py` writes `model.onnx`, which takes the
//! tokenization's paths and returns the weight matrix; burn-onnx compiles it in, weights
//! included (see `build.rs`). It runs on the GPU when CUDA is available.

use std::iter;
use std::sync::{Arc, Mutex};
use std::thread;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(not(target_os = "macos"))]
use cudarc::driver::CudaContext;
#[cfg(not(target_os = "macos"))]
use once_cell::sync::Lazy;

use crate::core::{Bind, Polarity, Position};
use crate::heuristic::WEIGHTS;
use crate::memory::W;

pub struct Tokenization {
    pub tokens: Vec<(Vec<Position>, W<Bind>)>,
    pub goals: Vec<W<Bind>>,
    pub premises: Vec<W<Bind>>
}

impl Tokenization {
    pub fn new() -> Tokenization {
        Tokenization { tokens: Vec::new(), goals: Vec::new(), premises: Vec::new() }
    }

    /// Record `bind` as a goal or premise by `polarity`, storing its index in the list on the bind.
    pub fn declare(&mut self, mut bind: W<Bind>, polarity: Polarity) {
        let list = match polarity {
            Polarity::Goal => &mut self.goals,
            Polarity::Premise => &mut self.premises
        };
        bind.borrow_mut().index = list.len();
        list.push(bind);
    }
}

#[allow(dead_code, clippy::all)]
mod model {
    extern crate alloc;
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}

type Cpu = burn::backend::NdArray<f32>;
/// With i32 integers, `Cuda`'s default, the model's output comes out wrong.
#[cfg(not(target_os = "macos"))]
type Gpu = burn::backend::Cuda<f32, i64>;

/// The model saw at most this many rows (tokens plus binds) in training.
const MAX_ROWS: usize = 5000;

/// A path step as the model reads it: the variant, in declaration order, and its index.
/// The model turns each into one of 900 tokens, `offset[variant] + min(index, cap[variant])`.
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

/// The step that pads paths to a common length. The model maps it to a token whose
/// matrix is the identity, so it doesn't change the path's encoding.
const PAD: [i64; 2] = [7, 0];

/// The model on each device, loaded when first needed. The lock also keeps problems
/// from running at once.
struct Models {
    #[cfg(not(target_os = "macos"))]
    gpu: Option<model::Model<Gpu>>,
    cpu: Option<model::Model<Cpu>>,
}

static MODELS: Mutex<Models> = Mutex::new(Models {
    #[cfg(not(target_os = "macos"))]
    gpu: None,
    cpu: None,
});

/// CUDA's primary context, which CubeCL shares, if CUDA works.
#[cfg(not(target_os = "macos"))]
static CUDA: Lazy<Option<Arc<CudaContext>>> = Lazy::new(cuda);

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
            .flat_map(|p| p.iter().map(step).chain(iter::repeat(PAD)).take(length).flatten())
            .collect();
        TensorData::new(data, [paths.len(), length, 2])
    }))
}

#[cfg(not(target_os = "macos"))]
fn cuda() -> Option<Arc<CudaContext>> {
    use cubecl_runtime::config::{cache::CacheConfig, compilation::CompilationConfig};
    use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
    use cudarc::{driver::sys as cu, nvrtc::sys as nvrtc};
    // CUDA is loaded at runtime. CubeCL panics without it, or if the driver is older than the
    // API it uses or than NVRTC, whose code the driver then can't load.
    let (mut driver, mut major, mut minor) = (0, 0, 0);
    if !unsafe {
        cu::is_culib_present()
            && nvrtc::is_culib_present()
            && cu::cuDriverGetVersion(&mut driver) == cu::CUresult::CUDA_SUCCESS
            && nvrtc::nvrtcVersion(&mut major, &mut minor) == nvrtc::nvrtcResult::NVRTC_SUCCESS
    } || driver < cu::CUDA_VERSION as i32
        || driver < 1000 * major + 10 * minor
    {
        return None;
    }
    // Most problems' shapes need new kernels: keep them compiled across runs.
    let compilation = CompilationConfig { cache: Some(CacheConfig::Global), ..Default::default() };
    CubeClRuntimeConfig::set(CubeClRuntimeConfig { compilation, ..Default::default() });
    CudaContext::new(0).ok()
}

fn forward<B: Backend>(model: &model::Model<B>, inputs: [TensorData; 4]) -> Vec<Vec<f32>> {
    let device = Default::default();
    let [positions, declarations, goals, premises] =
        inputs.map(|d| Tensor::<B, 3, Int>::from_data(d, &device));
    let weight = model.forward(positions, declarations, goals, premises);
    let [_, n] = weight.dims();
    let weight = weight.into_data().to_vec::<f32>().unwrap();
    // Leave the GPU's memory to others between problems.
    B::memory_cleanup(&device);
    let _ = B::sync(&device);
    weight.chunks(n).map(<[f32]>::to_vec).collect()
}

fn run(models: &mut Models, inputs: [TensorData; 4]) -> Vec<Vec<f32>> {
    #[cfg(not(target_os = "macos"))]
    if let Some(cuda) = CUDA.as_ref() {
        // CubeCL panics, unrecoverably, when out of memory. It reserves memory in pages of up
        // to a quarter of the GPU's: the attention scores (8 heads × rows² × 4 bytes) must fit
        // in one, and everything together in under half.
        let rows = inputs[0].shape[0] + inputs[2].shape[0] + inputs[3].shape[0];
        if cuda.mem_get_info().is_ok_and(|(free, total)| 2 * free > total && 128 * rows * rows < total) {
            return forward(models.gpu.get_or_insert_with(model::Model::default), inputs);
        }
    }
    forward(models.cpu.get_or_insert_with(model::Model::default), inputs)
}

/// `tokens`' goal × premise weights, indexed by `Bind::index`, or `None` if the model
/// doesn't apply.
pub fn weight(tokens: &Tokenization) -> Option<Vec<Vec<f32>>> {
    inputs(tokens).map(|inputs| run(&mut MODELS.lock().unwrap(), inputs))
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
        let mut models = MODELS.lock().unwrap();
        // Problems started in quick succession queue up here: only the latest runs.
        if !Arc::ptr_eq(&WEIGHTS.load(), &uniform) {
            return;
        }
        let weight = run(&mut models, inputs);
        println!("{:?}", weight);
        WEIGHTS.compare_and_swap(&uniform, Arc::new(weight));
    });
}
