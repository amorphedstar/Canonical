//! Path embedding (LibTorch): `root @ M1 @ M2 @ …`, one orthogonal matrix per `Position`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use canonical_compat::ir::Position;
use tch::Tensor;

use crate::weights::{load_file, NamedTensor};

pub struct PositionMatrices {
    root: Tensor,
    unit: HashMap<&'static str, Tensor>,
    right: HashMap<&'static str, Tensor>,
    down: HashMap<&'static str, Tensor>,
    cache: Mutex<HashMap<Position, Tensor>>,
}

impl PositionMatrices {
    pub fn load(path: &Path) -> Result<Self, String> {
        let tensors = load_file(path)?;
        let get = |name: &str| -> Result<Tensor, String> {
            let t: &NamedTensor = tensors.get(name).ok_or_else(|| format!("missing {name}"))?;
            let dims: Vec<i64> = t.shape.iter().map(|&d| d as i64).collect();
            Ok(Tensor::from_slice(&t.data).reshape(&dims))
        };
        Ok(Self {
            root: get("root")?,
            unit: ["Type", "LHS", "RHS"]
                .into_iter()
                .map(|k| get(&format!("unit_matrices.{k}")).map(|t| (k, t)))
                .collect::<Result<_, _>>()?,
            right: ["Rule", "Param", "Let", "Arg"]
                .into_iter()
                .map(|k| get(&format!("right_matrices.{k}")).map(|t| (k, t)))
                .collect::<Result<_, _>>()?,
            down: ["Rule", "Param", "Let", "Arg"]
                .into_iter()
                .map(|k| get(&format!("down_matrices.{k}")).map(|t| (k, t)))
                .collect::<Result<_, _>>()?,
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn embed_path(&self, path: &[Position]) -> Tensor {
        path.iter()
            .fold(self.root.shallow_clone(), |v, &p| v.matmul(&self.step(p)))
    }

    fn step(&self, pos: Position) -> Tensor {
        if let Some(m) = self.cache.lock().unwrap().get(&pos) {
            return m.shallow_clone();
        }
        let indexed = |k, i| {
            self.right[k]
                .matrix_power(i as i64 + 1)
                .matmul(&self.down[k])
        };
        let m = match pos {
            Position::Type => self.unit["Type"].shallow_clone(),
            Position::LHS => self.unit["LHS"].shallow_clone(),
            Position::RHS => self.unit["RHS"].shallow_clone(),
            Position::Rule(i) => indexed("Rule", i),
            Position::Param(i) => indexed("Param", i),
            Position::Let(i) => indexed("Let", i),
            Position::Arg(i) => indexed("Arg", i),
        };
        self.cache.lock().unwrap().insert(pos, m.shallow_clone());
        m
    }
}
