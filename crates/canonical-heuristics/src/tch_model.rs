//! LibTorch / TorchScript backend. Kept for parity with the Python export.
//! The encoder inside the `.ts` file is PyTorch's `nn.TransformerEncoder`.

use std::path::Path;
use std::sync::Mutex;

use canonical_compat::ir::{Position, Token};
use tch::{CModule, Tensor};

use crate::tch_encoding::PositionMatrices;
use crate::ModelConfig;

pub struct CanonicalTransformer {
    module: CModule,
}

impl CanonicalTransformer {
    pub fn load(path: impl AsRef<Path>) -> tch::Result<Self> {
        Self::load_on(path, tch::Device::Cpu)
    }

    pub fn load_on(path: impl AsRef<Path>, device: tch::Device) -> tch::Result<Self> {
        let mut module = CModule::load(path.as_ref())?;
        if device != tch::Device::Cpu {
            module.to(device, tch::Kind::Float, false);
        }
        Ok(Self { module })
    }

    /// `x` is `[seq, input_dim]`; returns `[seq, seq]` unification logits.
    pub fn forward(&self, x: &Tensor) -> tch::Result<Tensor> {
        Ok(self.module.forward_ts(&[x.unsqueeze(0)])?.squeeze_dim(0))
    }
}

pub struct HeuristicModel {
    pub config: ModelConfig,
    matrices: PositionMatrices,
    transformer: Mutex<CanonicalTransformer>,
}

impl HeuristicModel {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let config: ModelConfig = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Ok(Self {
            matrices: PositionMatrices::load(&dir.join("embedding.safetensors"))?,
            transformer: Mutex::new(
                CanonicalTransformer::load(dir.join("transformer.ts"))
                    .map_err(|e| e.to_string())?,
            ),
            config,
        })
    }

    /// Token rows `[pos | decl]`, then one bind row `[decl | decl]` each.
    pub fn embed(&self, tokens: &[Token], bind_paths: &[Vec<Position>]) -> Tensor {
        let cat = |a: Tensor, b: Tensor| Tensor::cat(&[a, b], 0);
        let mut rows: Vec<Tensor> = tokens
            .iter()
            .map(|t| {
                cat(
                    self.matrices.embed_path(&t.position),
                    self.matrices.embed_path(&t.declaration),
                )
            })
            .collect();
        rows.extend(bind_paths.iter().map(|p| {
            let d = self.matrices.embed_path(p);
            cat(d.shallow_clone(), d)
        }));
        Tensor::stack(&rows, 0)
    }

    pub fn logits(&self, x: &Tensor) -> Result<Tensor, String> {
        tch::no_grad(|| self.transformer.lock().unwrap().forward(x)).map_err(|e| e.to_string())
    }

    pub fn embed_vec(&self, tokens: &[Token], bind_paths: &[Vec<Position>]) -> Vec<f32> {
        Vec::try_from(self.embed(tokens, bind_paths).flatten(0, -1)).unwrap()
    }

    pub fn logits_from_slice(&self, x: &[f32], seq: usize, dim: usize) -> Vec<f32> {
        let t = Tensor::from_slice(x).reshape([seq as i64, dim as i64]);
        Vec::try_from(self.logits(&t).expect("transformer").flatten(0, -1)).unwrap()
    }

    pub fn softmax_row(
        &self,
        logits: &[f32],
        n: usize,
        goal: usize,
        premises: &[usize],
    ) -> Vec<f32> {
        let temp = self.config.temperature as f32;
        let vals: Vec<f32> = premises
            .iter()
            .map(|&j| logits[goal * n + j] / temp)
            .collect();
        Vec::try_from(Tensor::from_slice(&vals).softmax(-1, tch::Kind::Float)).unwrap()
    }
}
