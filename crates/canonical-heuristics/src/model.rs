//! ONNX path embedding + `CanonicalTransformer` via burn-onnx codegen.
//! Packed step indices in, goal×premise logits out.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(not(feature = "cuda"))]
use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use canonical_compat::ir::{Position, Token};
use canonical_core::compiler::BindScores;
use canonical_core::core::Bind;
use canonical_core::memory::W;

use crate::encoding::{pack_pair, sequence_paths, softmax, Example};
use crate::ModelConfig;

#[cfg(feature = "cuda")]
type B = Cuda<f32, i32>;
#[cfg(not(feature = "cuda"))]
type B = NdArray<f32>;

#[allow(dead_code, unused_imports, clippy::all)]
mod onnx_transformer {
    extern crate alloc;
    include!(concat!(env!("OUT_DIR"), "/model/transformer.rs"));
}

pub struct HeuristicModel {
    pub config: ModelConfig,
    net: Mutex<onnx_transformer::Model<B>>,
    device: <B as burn::tensor::backend::BackendTypes>::Device,
}

impl HeuristicModel {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let config: ModelConfig = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let device = Default::default();
        let net = load_onnx_model(&device, dir)?;
        Ok(Self {
            net: Mutex::new(net),
            device,
            config,
        })
    }

    pub fn backend_name() -> &'static str {
        #[cfg(feature = "cuda")]
        {
            "onnx-cuda"
        }
        #[cfg(not(feature = "cuda"))]
        {
            "onnx"
        }
    }

    /// Goal×premise submatrix from packed `[seq, path_len]` path steps.
    pub fn scores_from_packed(
        &self,
        left: &[i64],
        right: &[i64],
        seq: usize,
        path_len: usize,
        goal_index: &[i64],
        premise_index: &[i64],
    ) -> Vec<f32> {
        debug_assert_eq!(left.len(), seq * path_len);
        debug_assert_eq!(right.len(), seq * path_len);
        let lt = Tensor::<B, 2, Int>::from_ints(
            TensorData::new(left.to_vec(), [seq, path_len]),
            &self.device,
        );
        let rt = Tensor::<B, 2, Int>::from_ints(
            TensorData::new(right.to_vec(), [seq, path_len]),
            &self.device,
        );
        let gt = Tensor::<B, 1, Int>::from_ints(
            TensorData::new(goal_index.to_vec(), [goal_index.len()]),
            &self.device,
        );
        let pt = Tensor::<B, 1, Int>::from_ints(
            TensorData::new(premise_index.to_vec(), [premise_index.len()]),
            &self.device,
        );
        let y = self.net.lock().unwrap().forward(lt, rt, gt, pt);
        y.into_data().to_vec::<f32>().unwrap()
    }

    pub fn scores_from_paths(
        &self,
        left_paths: &[Vec<u8>],
        right_paths: &[Vec<u8>],
        goal_index: &[i64],
        premise_index: &[i64],
    ) -> Vec<f32> {
        assert_eq!(left_paths.len(), right_paths.len());
        let seq = left_paths.len();
        let (left, right, path_len) = pack_pair(left_paths, right_paths);
        self.scores_from_packed(&left, &right, seq, path_len, goal_index, premise_index)
    }

    pub fn forward_example(&self, example: &Example) -> Vec<f32> {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (pos, decl) in &example.tokens {
            left.push(pos.clone());
            right.push(decl.clone());
        }
        for g in &example.goals {
            left.push(g.clone());
            right.push(g.clone());
        }
        for p in &example.premises {
            left.push(p.clone());
            right.push(p.clone());
        }
        let n_tokens = example.tokens.len();
        let n_goals = example.goals.len();
        let n_premises = example.premises.len();
        let seq = n_tokens + n_goals + n_premises;
        let goals: Vec<i64> = (n_tokens..n_tokens + n_goals).map(|i| i as i64).collect();
        let premises: Vec<i64> = (n_tokens + n_goals..seq).map(|i| i as i64).collect();
        self.scores_from_paths(&left, &right, &goals, &premises)
    }

    pub fn softmax_row(&self, logits: &[f32], n_premises: usize, goal: usize) -> Vec<f32> {
        let temp = self.config.temperature as f32;
        let start = goal * n_premises;
        let vals: Vec<f32> = logits[start..start + n_premises]
            .iter()
            .map(|v| v / temp)
            .collect();
        softmax(&vals)
    }

    pub fn bind_scores(
        &self,
        tokens: &[Token],
        bind_paths: &[(W<Bind>, Vec<Position>)],
        goals: &[W<Bind>],
        premises: &[W<Bind>],
    ) -> BindScores {
        let n = tokens.len() + bind_paths.len();
        if n == 0 || goals.is_empty() || premises.is_empty() {
            return HashMap::new();
        }
        let uniform = || {
            let p = 1.0 / premises.len() as f64;
            goals
                .iter()
                .map(|g| {
                    (
                        g.clone(),
                        premises.iter().map(|pr| (pr.clone(), p)).collect(),
                    )
                })
                .collect()
        };
        if n > self.config.max_seq_len {
            return uniform();
        }
        let paths: Vec<_> = bind_paths.iter().map(|(_, p)| p.clone()).collect();
        let (left, right) = sequence_paths(tokens, &paths);
        let index: HashMap<_, _> = bind_paths
            .iter()
            .enumerate()
            .map(|(i, (b, _))| (b.clone(), tokens.len() + i))
            .collect();
        let goal_idx: Vec<i64> = goals
            .iter()
            .filter_map(|g| index.get(g).map(|&i| i as i64))
            .collect();
        let premise_idx: Vec<i64> = premises
            .iter()
            .filter_map(|p| index.get(p).map(|&i| i as i64))
            .collect();
        if goal_idx.len() != goals.len() || premise_idx.len() != premises.len() {
            return HashMap::new();
        }
        let sub = self.scores_from_paths(&left, &right, &goal_idx, &premise_idx);
        let n_p = premises.len();
        goals
            .iter()
            .enumerate()
            .map(|(gi, g)| {
                let probs = self.softmax_row(&sub, n_p, gi);
                (
                    g.clone(),
                    premises
                        .iter()
                        .zip(probs)
                        .map(|(p, v)| (p.clone(), v as f64))
                        .collect(),
                )
            })
            .collect()
    }
}

fn load_onnx_model(
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
    dir: &Path,
) -> Result<onnx_transformer::Model<B>, String> {
    let candidates = [
        dir.join("transformer.bpk"),
        PathBuf::from(env!("OUT_DIR")).join("model/transformer.bpk"),
    ];
    for candidate in &candidates {
        if candidate.exists() {
            let path = candidate
                .to_str()
                .ok_or_else(|| format!("non-UTF-8 weights path: {}", candidate.display()))?;
            return Ok(onnx_transformer::Model::from_file(path, device));
        }
    }
    // `onnx_transformer::Model::default()` (the burn-onnx-generated impl) just calls
    // `from_file` on the second candidate above via a path baked in at build time, so if
    // we get here that candidate is already confirmed missing and `default()` would only
    // panic (`.expect("Failed to load burnpack file")`) instead of returning our `Result`.
    Err(format!(
        "no ONNX weights found; tried {} and {}",
        candidates[0].display(),
        candidates[1].display()
    ))
}
