//! Native Burn path embedding: `root @ M1 @ M2 @ …`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use canonical_compat::ir::Position;

use crate::weights::{load_file, NamedTensor};

pub type B = NdArray<f32>;

pub struct PositionMatrices {
    root: Tensor<B, 1>,
    unit: HashMap<&'static str, Tensor<B, 2>>,
    right: HashMap<&'static str, Tensor<B, 2>>,
    down: HashMap<&'static str, Tensor<B, 2>>,
    cache: Mutex<HashMap<Position, Tensor<B, 2>>>,
}

impl PositionMatrices {
    pub fn load(path: &Path) -> Result<Self, String> {
        let tensors = load_file(path)?;
        let device = Default::default();
        let mat2 = |name: &str| -> Result<Tensor<B, 2>, String> {
            let t = tensors.get(name).ok_or_else(|| format!("missing {name}"))?;
            Ok(from_2d(t, &device))
        };
        let root = tensors.get("root").ok_or("missing root")?;
        Ok(Self {
            root: Tensor::from_data(
                TensorData::new(root.data.clone(), [root.data.len()]),
                &device,
            ),
            unit: ["Type", "LHS", "RHS"]
                .into_iter()
                .map(|k| mat2(&format!("unit_matrices.{k}")).map(|t| (k, t)))
                .collect::<Result<_, _>>()?,
            right: ["Rule", "Param", "Let", "Arg"]
                .into_iter()
                .map(|k| mat2(&format!("right_matrices.{k}")).map(|t| (k, t)))
                .collect::<Result<_, _>>()?,
            down: ["Rule", "Param", "Let", "Arg"]
                .into_iter()
                .map(|k| mat2(&format!("down_matrices.{k}")).map(|t| (k, t)))
                .collect::<Result<_, _>>()?,
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn embed_path(&self, path: &[Position]) -> Tensor<B, 1> {
        path.iter().fold(self.root.clone(), |v, &p| {
            v.unsqueeze_dim(0).matmul(self.step(p)).squeeze_dim(0)
        })
    }

    fn step(&self, pos: Position) -> Tensor<B, 2> {
        if let Some(m) = self.cache.lock().unwrap().get(&pos) {
            return m.clone();
        }
        let indexed = |k, i: usize| pow(self.right[k].clone(), i + 1).matmul(self.down[k].clone());
        let m = match pos {
            Position::Type => self.unit["Type"].clone(),
            Position::LHS => self.unit["LHS"].clone(),
            Position::RHS => self.unit["RHS"].clone(),
            Position::Rule(i) => indexed("Rule", i),
            Position::Param(i) => indexed("Param", i),
            Position::Let(i) => indexed("Let", i),
            Position::Arg(i) => indexed("Arg", i),
        };
        self.cache.lock().unwrap().insert(pos, m.clone());
        m
    }
}

fn from_2d(
    t: &NamedTensor,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(t.data.clone(), [t.shape[0], t.shape[1]]),
        device,
    )
}

fn pow(m: Tensor<B, 2>, n: usize) -> Tensor<B, 2> {
    (1..n).fold(m.clone(), |acc, _| acc.matmul(m.clone()))
}
