//! Path steps as matrix indices 0–10, plus CPU embedding of those paths.
//!
//! Each `Position` expands to one or more of the 11 orthogonal matrices:
//!   0 Type    1 LHS     2 RHS
//!   3 Rule→   4 Rule↓   5 Param→  6 Param↓
//!   7 Let→    8 Let↓    9 Arg→   10 Arg↓
//! Indexed positions `Kind(i)` become `(i+1)` right multiplies followed by down,
//! matching `right.matrix_power(i+1).matmul(down)`.
//!
//! The ONNX runtime packs these steps into `[seq, path_len]` int tensors
//! (pad = 11, identity), where `path_len` is the longest path in the batch.
//! `PositionMatrices` is the CPU reference used by the native Burn / tch
//! backends, not by ONNX inference.

use std::path::Path;

use canonical_compat::ir::{Position, Token};

use crate::weights::load_file;

pub const NUM_MATRICES: usize = 11;
/// Padding index for packed ONNX path tensors (identity, not a trained matrix).
pub const PAD: u8 = NUM_MATRICES as u8;

/// Already-encoded token/goal/premise paths, ready for [`HeuristicModel::forward_example`]
/// (`crate::model::HeuristicModel`). Built by callers from [`encode_path`]; there is no
/// `from_ir` constructor here because the production path (`HeuristicModel::bind_scores`)
/// goes straight from IR to packed tensors via [`sequence_paths`] without materializing this.
///
/// [`HeuristicModel::forward_example`]: crate::model::HeuristicModel::forward_example
#[derive(Clone, Debug, Default)]
pub struct Example {
    pub tokens: Vec<(Vec<u8>, Vec<u8>)>,
    pub goals: Vec<Vec<u8>>,
    pub premises: Vec<Vec<u8>>,
}

pub struct PositionMatrices {
    pub dim: usize,
    root: Vec<f32>,
    /// Row-major `dim × dim` matrices, indexed 0..11.
    matrices: Vec<Vec<f32>>,
}

pub fn encode_path(path: &[Position]) -> Vec<u8> {
    path.iter().flat_map(|&p| position_to_steps(p)).collect()
}

pub fn position_to_steps(pos: Position) -> Vec<u8> {
    match pos {
        Position::Type => vec![0],
        Position::LHS => vec![1],
        Position::RHS => vec![2],
        Position::Rule(i) => indexed(3, 4, i),
        Position::Param(i) => indexed(5, 6, i),
        Position::Let(i) => indexed(7, 8, i),
        Position::Arg(i) => indexed(9, 10, i),
    }
}

fn indexed(right: u8, down: u8, i: usize) -> Vec<u8> {
    let mut steps = vec![right; i + 1];
    steps.push(down);
    steps
}

/// Token rows `[pos | decl]` and bind rows `[decl | decl]` as step sequences.
pub fn sequence_paths(
    tokens: &[Token],
    bind_paths: &[Vec<Position>],
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut left = Vec::with_capacity(tokens.len() + bind_paths.len());
    let mut right = Vec::with_capacity(tokens.len() + bind_paths.len());
    for t in tokens {
        left.push(encode_path(&t.position));
        right.push(encode_path(&t.declaration));
    }
    for p in bind_paths {
        let e = encode_path(p);
        left.push(e.clone());
        right.push(e);
    }
    (left, right)
}

/// Width of a packed `[seq, path_len]` batch. At least 1 so ONNX Scan has a trip count.
pub fn packed_len(paths: &[Vec<u8>]) -> usize {
    paths.iter().map(|p| p.len()).max().unwrap_or(0).max(1)
}

/// Pack `left` and `right` to the same `[seq, path_len]`, `path_len = max(left, right, 1)`.
pub fn pack_pair(left: &[Vec<u8>], right: &[Vec<u8>]) -> (Vec<i64>, Vec<i64>, usize) {
    let path_len = packed_len(left).max(packed_len(right));
    (
        pack_paths(left, path_len),
        pack_paths(right, path_len),
        path_len,
    )
}

/// Row-major `[seq, path_len]` i64 steps, padded with [`PAD`].
pub fn pack_paths(paths: &[Vec<u8>], path_len: usize) -> Vec<i64> {
    let mut out = vec![i64::from(PAD); paths.len() * path_len];
    for (i, path) in paths.iter().enumerate() {
        for (j, &step) in path.iter().take(path_len).enumerate() {
            out[i * path_len + j] = i64::from(step.min(PAD));
        }
    }
    out
}

impl PositionMatrices {
    pub fn load(path: &Path) -> Result<Self, String> {
        let tensors = load_file(path)?;
        let root = tensors.get("root").ok_or("missing root")?.data.clone();
        let dim = root.len();
        let load2 = |name: &str| -> Result<Vec<f32>, String> {
            let t = tensors.get(name).ok_or_else(|| format!("missing {name}"))?;
            Ok(t.data.clone())
        };
        let matrices = vec![
            load2("unit_matrices.Type")?,
            load2("unit_matrices.LHS")?,
            load2("unit_matrices.RHS")?,
            load2("right_matrices.Rule")?,
            load2("down_matrices.Rule")?,
            load2("right_matrices.Param")?,
            load2("down_matrices.Param")?,
            load2("right_matrices.Let")?,
            load2("down_matrices.Let")?,
            load2("right_matrices.Arg")?,
            load2("down_matrices.Arg")?,
        ];
        Ok(Self {
            dim,
            root,
            matrices,
        })
    }

    pub fn embed_path(&self, steps: &[u8]) -> Vec<f32> {
        steps.iter().fold(self.root.clone(), |v, &s| {
            if s >= PAD {
                v
            } else {
                matvec_left(&v, &self.matrices[s as usize], self.dim)
            }
        })
    }

    pub fn embed_token(&self, pos: &[u8], decl: &[u8]) -> Vec<f32> {
        let mut row = self.embed_path(pos);
        row.extend(self.embed_path(decl));
        row
    }

    pub fn embed_bind(&self, path: &[u8]) -> Vec<f32> {
        let d = self.embed_path(path);
        let mut row = d.clone();
        row.extend(d);
        row
    }

    /// Token rows `[pos | decl]`, then one bind row `[decl | decl]` each.
    pub fn embed_sequence(&self, tokens: &[Token], bind_paths: &[Vec<Position>]) -> Vec<f32> {
        let mut rows = Vec::with_capacity((tokens.len() + bind_paths.len()) * 2 * self.dim);
        for t in tokens {
            rows.extend(self.embed_token(&encode_path(&t.position), &encode_path(&t.declaration)));
        }
        for p in bind_paths {
            rows.extend(self.embed_bind(&encode_path(p)));
        }
        rows
    }
}

/// `out = v @ M` for row-major `M` (`dim × dim`) and vector `v`.
fn matvec_left(v: &[f32], m: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; dim];
    for j in 0..dim {
        let mut s = 0.0;
        for k in 0..dim {
            s += v[k] * m[k * dim + j];
        }
        out[j] = s;
    }
    out
}

pub fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}
