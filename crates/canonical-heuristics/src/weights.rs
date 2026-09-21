//! Shared SafeTensors loading (PyTorch export layout).

use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;

#[derive(Clone)]
pub struct NamedTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

pub fn load_file(path: &Path) -> Result<HashMap<String, NamedTensor>, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    load_bytes(&bytes)
}

pub fn load_bytes(bytes: &[u8]) -> Result<HashMap<String, NamedTensor>, String> {
    let t = SafeTensors::deserialize(bytes).map_err(|e| e.to_string())?;
    t.names()
        .into_iter()
        .map(|name| {
            let v = t.tensor(name).map_err(|e| e.to_string())?;
            let data = v
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok((
                name.to_string(),
                NamedTensor {
                    data,
                    shape: v.shape().to_vec(),
                },
            ))
        })
        .collect()
}

pub fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

pub fn mean_abs(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().max(1) as f32;
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / n
}
