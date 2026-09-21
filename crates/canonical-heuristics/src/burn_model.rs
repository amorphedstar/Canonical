//! Native Burn encoder (not ONNX). Kept for parity with the safetensors export.

use std::path::Path;

use burn::backend::NdArray;
use burn::module::{Module, Param};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderLayer,
};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{activation::softmax, Tensor, TensorData};
use canonical_compat::ir::{Position, Token};

use crate::burn_encoding::PositionMatrices;
use crate::weights::{load_file, NamedTensor};
use crate::ModelConfig;

pub type B = NdArray<f32>;

#[derive(Module, Debug)]
pub struct CanonicalTransformerNet<B: Backend> {
    input_proj: Linear<B>,
    encoder: TransformerEncoder<B>,
    score_norm: LayerNorm<B>,
    qk_proj: Linear<B>,
}

pub struct CanonicalTransformer<B: Backend> {
    net: CanonicalTransformerNet<B>,
    qk_dim: usize,
}

pub struct HeuristicModel {
    pub config: ModelConfig,
    matrices: PositionMatrices,
    transformer: CanonicalTransformer<B>,
}

impl<B: Backend> CanonicalTransformerNet<B> {
    fn init(cfg: &ModelConfig, device: &B::Device) -> Self {
        Self {
            input_proj: LinearConfig::new(cfg.input_dim, cfg.d_model).init(device),
            encoder: TransformerEncoderConfig::new(
                cfg.d_model,
                cfg.dim_feedforward,
                cfg.num_heads,
                cfg.num_layers,
            )
            .with_dropout(0.0)
            .with_norm_first(true)
            .with_quiet_softmax(false)
            .init(device),
            score_norm: LayerNormConfig::new(cfg.d_model)
                .with_epsilon(1e-5)
                .init(device),
            qk_proj: LinearConfig::new(cfg.d_model, 2 * cfg.qk_dim)
                .with_bias(false)
                .init(device),
        }
    }
}

impl<B: Backend> CanonicalTransformer<B> {
    pub fn load_on(path: &Path, cfg: &ModelConfig, device: &B::Device) -> Result<Self, String> {
        let tensors = load_file(path)?;
        let net = CanonicalTransformerNet::<B>::init(cfg, device);
        let rec = fill_record::<B>(net.clone().into_record(), &tensors, device, cfg)?;
        Ok(Self {
            net: net.load_record(rec),
            qk_dim: cfg.qk_dim,
        })
    }

    /// `x` is `[seq, input_dim]`; returns `[seq, seq]` unification logits.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [seq, dim] = x.dims();
        let mut encoded = self.net.input_proj.forward(x.reshape([1, seq, dim]));
        for layer in self.net.encoder.layers.iter() {
            encoded = encoder_layer_sdpa(layer, encoded);
        }
        let qk = self
            .net
            .qk_proj
            .forward(self.net.score_norm.forward(encoded));
        let q = qk.clone().slice([0..1, 0..seq, 0..self.qk_dim]);
        let k = qk.slice([0..1, 0..seq, self.qk_dim..2 * self.qk_dim]);
        q.matmul(k.swap_dims(1, 2))
            .div_scalar((self.qk_dim as f32).sqrt())
            .reshape([seq, seq])
    }
}

/// Same residual layout as Burn's `TransformerEncoderLayer`, but SDPA goes through
/// `burn::tensor::module::attention` so CubeCL can pick a flash kernel.
fn encoder_layer_sdpa<B: Backend>(
    layer: &TransformerEncoderLayer<B>,
    input: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let mut residual = input.clone();
    if layer.norm_first {
        residual = layer.norm_2.forward(residual);
    }
    let [batch, seq, d_model] = residual.dims();
    let n_heads = layer.mha.n_heads;
    let d_k = layer.mha.d_k;
    let q = layer
        .mha
        .query
        .forward(residual.clone())
        .reshape([batch, seq, n_heads, d_k])
        .swap_dims(1, 2);
    let k = layer
        .mha
        .key
        .forward(residual.clone())
        .reshape([batch, seq, n_heads, d_k])
        .swap_dims(1, 2);
    let v = layer
        .mha
        .value
        .forward(residual)
        .reshape([batch, seq, n_heads, d_k])
        .swap_dims(1, 2);
    let context = attention(
        q,
        k,
        v,
        None,
        None,
        AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: false,
        },
    );
    let context = layer
        .mha
        .output
        .forward(context.swap_dims(1, 2).reshape([batch, seq, d_model]));
    let mut x = input + layer.dropout.forward(context);
    let ffn_in = if layer.norm_first {
        layer.norm_1.forward(x.clone())
    } else {
        x = layer.norm_1.forward(x);
        x.clone()
    };
    x = x + layer.dropout.forward(layer.pwff.forward(ffn_in));
    if !layer.norm_first {
        x = layer.norm_2.forward(x);
    }
    x
}

impl CanonicalTransformer<NdArray<f32>> {
    pub fn load(path: &Path, cfg: &ModelConfig) -> Result<Self, String> {
        Self::load_on(path, cfg, &Default::default())
    }
}

impl HeuristicModel {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let config: ModelConfig = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        Ok(Self {
            matrices: PositionMatrices::load(&dir.join("embedding.safetensors"))?,
            transformer: CanonicalTransformer::load(&dir.join("transformer.safetensors"), &config)?,
            config,
        })
    }

    pub fn embed(&self, tokens: &[Token], bind_paths: &[Vec<Position>]) -> Tensor<B, 2> {
        let cat = |a: Tensor<B, 1>, b: Tensor<B, 1>| Tensor::cat(vec![a, b], 0);
        let mut rows: Vec<Tensor<B, 1>> = tokens
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
            cat(d.clone(), d)
        }));
        Tensor::stack(rows, 0)
    }

    pub fn logits(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.transformer.forward(x)
    }

    pub fn logits_from_slice(&self, x: &[f32], seq: usize, dim: usize) -> Vec<f32> {
        let t =
            Tensor::<B, 2>::from_data(TensorData::new(x.to_vec(), [seq, dim]), &Default::default());
        to_vec(self.logits(t))
    }

    pub fn embed_vec(&self, tokens: &[Token], bind_paths: &[Vec<Position>]) -> Vec<f32> {
        to_vec(self.embed(tokens, bind_paths))
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
        to_vec(softmax(
            Tensor::<B, 1>::from_data(TensorData::new(vals, [premises.len()]), &Default::default()),
            0,
        ))
    }
}

fn fill_record<B: Backend>(
    mut rec: <CanonicalTransformerNet<B> as Module<B>>::Record,
    tensors: &std::collections::HashMap<String, NamedTensor>,
    device: &B::Device,
    cfg: &ModelConfig,
) -> Result<<CanonicalTransformerNet<B> as Module<B>>::Record, String> {
    rec.input_proj.weight = linear_w(tensors, "input_proj.weight", device)?;
    rec.input_proj.bias = Some(bias(tensors, "input_proj.bias", device)?);
    rec.score_norm.gamma = bias(tensors, "score_norm.weight", device)?;
    rec.score_norm.beta = Some(bias(tensors, "score_norm.bias", device)?);
    rec.qk_proj.weight = linear_w(tensors, "qk_proj.weight", device)?;
    rec.qk_proj.bias = None;

    let d = cfg.d_model;
    for (i, layer) in rec.encoder.layers.iter_mut().enumerate() {
        let p = format!("encoder.layers.{i}");
        let (qw, kw, vw) = split3_w(tensors, &format!("{p}.self_attn.in_proj_weight"), d, device)?;
        let (qb, kb, vb) = split3_b(tensors, &format!("{p}.self_attn.in_proj_bias"), d, device)?;
        layer.mha.query.weight = qw;
        layer.mha.query.bias = Some(qb);
        layer.mha.key.weight = kw;
        layer.mha.key.bias = Some(kb);
        layer.mha.value.weight = vw;
        layer.mha.value.bias = Some(vb);
        layer.mha.output.weight =
            linear_w(tensors, &format!("{p}.self_attn.out_proj.weight"), device)?;
        layer.mha.output.bias = Some(bias(
            tensors,
            &format!("{p}.self_attn.out_proj.bias"),
            device,
        )?);
        // PyTorch norm1 = attention, Burn norm_2; PyTorch norm2 = FFN, Burn norm_1.
        layer.norm_2.gamma = bias(tensors, &format!("{p}.norm1.weight"), device)?;
        layer.norm_2.beta = Some(bias(tensors, &format!("{p}.norm1.bias"), device)?);
        layer.norm_1.gamma = bias(tensors, &format!("{p}.norm2.weight"), device)?;
        layer.norm_1.beta = Some(bias(tensors, &format!("{p}.norm2.bias"), device)?);
        layer.pwff.linear_inner.weight = linear_w(tensors, &format!("{p}.linear1.weight"), device)?;
        layer.pwff.linear_inner.bias = Some(bias(tensors, &format!("{p}.linear1.bias"), device)?);
        layer.pwff.linear_outer.weight = linear_w(tensors, &format!("{p}.linear2.weight"), device)?;
        layer.pwff.linear_outer.bias = Some(bias(tensors, &format!("{p}.linear2.bias"), device)?);
    }
    Ok(rec)
}

fn get<'a>(
    tensors: &'a std::collections::HashMap<String, NamedTensor>,
    name: &str,
) -> Result<&'a NamedTensor, String> {
    tensors.get(name).ok_or_else(|| format!("missing {name}"))
}

fn linear_w<B: Backend>(
    tensors: &std::collections::HashMap<String, NamedTensor>,
    name: &str,
    device: &B::Device,
) -> Result<Param<Tensor<B, 2>>, String> {
    let t = get(tensors, name)?;
    let (out, inp) = (t.shape[0], t.shape[1]);
    Ok(Param::from_data(
        TensorData::new(transpose(&t.data, out, inp), [inp, out]),
        device,
    ))
}

fn bias<B: Backend>(
    tensors: &std::collections::HashMap<String, NamedTensor>,
    name: &str,
    device: &B::Device,
) -> Result<Param<Tensor<B, 1>>, String> {
    let t = get(tensors, name)?;
    Ok(Param::from_data(
        TensorData::new(t.data.clone(), [t.data.len()]),
        device,
    ))
}

fn split3_w<B: Backend>(
    tensors: &std::collections::HashMap<String, NamedTensor>,
    name: &str,
    d: usize,
    device: &B::Device,
) -> Result<
    (
        Param<Tensor<B, 2>>,
        Param<Tensor<B, 2>>,
        Param<Tensor<B, 2>>,
    ),
    String,
> {
    let t = get(tensors, name)?;
    let n = d * d;
    let chunk = |off| {
        Param::from_data(
            TensorData::new(transpose(&t.data[off..off + n], d, d), [d, d]),
            device,
        )
    };
    Ok((chunk(0), chunk(n), chunk(2 * n)))
}

fn split3_b<B: Backend>(
    tensors: &std::collections::HashMap<String, NamedTensor>,
    name: &str,
    d: usize,
    device: &B::Device,
) -> Result<
    (
        Param<Tensor<B, 1>>,
        Param<Tensor<B, 1>>,
        Param<Tensor<B, 1>>,
    ),
    String,
> {
    let t = get(tensors, name)?;
    let chunk = |off| Param::from_data(TensorData::new(t.data[off..off + d].to_vec(), [d]), device);
    Ok((chunk(0), chunk(d), chunk(2 * d)))
}

fn transpose(data: &[f32], out_f: usize, in_f: usize) -> Vec<f32> {
    let mut t = vec![0.0; data.len()];
    for o in 0..out_f {
        for i in 0..in_f {
            t[i * out_f + o] = data[o * in_f + i];
        }
    }
    t
}

pub fn to_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}
