pub mod burn_encoding;
pub mod burn_model;
pub mod encoding;
pub mod model;
pub mod weights;

#[cfg(feature = "tch")]
pub mod tch_encoding;
#[cfg(feature = "tch")]
pub mod tch_model;

use std::path::PathBuf;
use std::sync::Mutex;
use std::thread::{self, JoinHandle};

/// Turn on CubeCL's persistent compiled-kernel (PTX) cache under `~/.config/cuda/`, so a
/// shape seen once on this machine (by any process, ever) skips NVRTC compilation on every
/// later run instead of paying it fresh each time. Measured ~50x on a repeat shape (3.7s ->
/// 71ms cold). Must run before the first CubeCL/burn-cuda call in the process, which is why
/// `model::HeuristicModel::load` calls this as its first line; call it yourself before
/// touching `burn_model::CanonicalTransformer<Cuda<..>>` directly (as heuristics-bench does).
/// A no-op after the first call, and safe (just skipped, not fatal) if some other code in
/// the process already read CubeCL's config before we got here.
#[cfg(feature = "cuda")]
pub fn enable_cubecl_disk_cache() {
    use cubecl_runtime::config::{cache::CacheConfig, compilation::CompilationConfig, CubeClRuntimeConfig, RuntimeConfig};
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let config = CubeClRuntimeConfig {
            compilation: CompilationConfig {
                cache: Some(CacheConfig::Global),
                ..Default::default()
            },
            ..Default::default()
        };
        if std::panic::catch_unwind(|| CubeClRuntimeConfig::set(config)).is_err() {
            eprintln!(
                "canonical-heuristics: could not enable the CubeCL PTX cache (config already \
                 read elsewhere) -- CUDA calls will still work, just without the cross-process \
                 kernel cache"
            );
        }
    });
}

use canonical_compat::ir::{BindMap, Position, Token};
use canonical_core::compiler::{compile, BindScores, CompilationGoals};
use canonical_core::core::{Bind, Type, TypeBase, ES};
use canonical_core::memory::W;
use canonical_core::search::RUN;
use once_cell::sync::OnceCell;
use std::sync::atomic::{AtomicBool, Ordering};

pub use encoding::Example;
pub use model::HeuristicModel;

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ModelConfig {
    pub position_embedding_dim: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dim_feedforward: usize,
    pub qk_dim: usize,
    pub temperature: f64,
    pub max_seq_len: usize,
    pub input_dim: usize,
}

fn default_weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../CanonicalHeuristics/export_rust")
}

fn weights_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CANONICAL_HEURISTICS_WEIGHTS") {
        let p = PathBuf::from(p);
        if p.join("config.json").is_file() {
            return Some(p);
        }
        eprintln!(
            "canonical-heuristics: CANONICAL_HEURISTICS_WEIGHTS is missing config.json: {}",
            p.display()
        );
        return None;
    }
    let baked = default_weights_dir();
    if baked.join("config.json").is_file() {
        return Some(baked);
    }
    None
}

fn global_model() -> &'static Mutex<Option<HeuristicModel>> {
    static MODEL: OnceCell<Mutex<Option<HeuristicModel>>> = OnceCell::new();
    MODEL.get_or_init(|| {
        Mutex::new(weights_dir().and_then(|p| match HeuristicModel::load(&p) {
            Ok(m) => {
                eprintln!(
                    "canonical-heuristics: loaded {} from {}",
                    HeuristicModel::backend_name(),
                    p.display()
                );
                Some(m)
            }
            Err(e) => {
                eprintln!("canonical-heuristics: {e}");
                None
            }
        }))
    })
}

/// Process-global model: `CANONICAL_HEURISTICS_WEIGHTS`, else the sibling `export_rust` dir.
pub fn infer_bind_scores(
    tokens: &[Token],
    bind_paths: &[(W<Bind>, Vec<Position>)],
    goals: &[W<Bind>],
    premises: &[W<Bind>],
) -> Option<BindScores> {
    let guard = global_model().lock().ok()?;
    guard
        .as_ref()
        .map(|m| m.bind_scores(tokens, bind_paths, goals, premises))
}

/// Walk `typ` for its goal/premise binds and score them against `bind_paths`.
fn scores_for_goals(
    tb: W<TypeBase>,
    problem_bind: W<Bind>,
    tokens: &[Token],
    bind_paths: &[(W<Bind>, Vec<Position>)],
) -> Option<BindScores> {
    let collected = CompilationGoals::collect(Type(tb, ES::new(), problem_bind));
    infer_bind_scores(
        tokens,
        bind_paths,
        &collected.goal_binds(),
        &collected.premise_binds(),
    )
}

/// Run the model on this problem and return unification probabilities for `compile` / `Prover::new`.
pub fn bind_scores_for_problem(
    tb: W<TypeBase>,
    problem_bind: W<Bind>,
    tokens: &[Token],
    binds: &BindMap,
) -> Option<BindScores> {
    scores_for_goals(tb, problem_bind, tokens, &binds.bind_paths())
}

/// True after a successful background `compile` with model scores.
pub static HEURISTICS_APPLIED: AtomicBool = AtomicBool::new(false);

/// Infer on a background thread. Call this *after* `Prover::new(..., None)` so
/// the uniform compile cannot overwrite scores. When scores are ready and
/// `RUN` is still true, `compile` swaps them in; later `iter_unify` calls see
/// the new probabilities.
pub fn spawn_heuristic_compilation(
    tb: W<TypeBase>,
    problem_bind: W<Bind>,
    tokens: Vec<Token>,
    bind_paths: Vec<(W<Bind>, Vec<Position>)>,
) -> JoinHandle<()> {
    HEURISTICS_APPLIED.store(false, Ordering::Relaxed);
    thread::spawn(move || {
        let Some(scores) =
            scores_for_goals(tb.clone(), problem_bind.clone(), &tokens, &bind_paths)
        else {
            return;
        };
        if !RUN.load(Ordering::Relaxed) {
            return;
        }
        compile(Type(tb, ES::new(), problem_bind), Some(&scores));
        HEURISTICS_APPLIED.store(true, Ordering::Relaxed);
        eprintln!(
            "canonical-heuristics: applied scores ({} goals)",
            scores.len()
        );
    })
}
