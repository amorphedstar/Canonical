//! Isolated transformer timing/RSS (and GPU memory) for tch / Burn. Prints one JSON object.

use std::path::PathBuf;
use std::time::Instant;

use canonical_heuristics::burn_model;
#[cfg(feature = "tch")]
use canonical_heuristics::tch_model;
#[cfg(feature = "tch")]
use canonical_heuristics::tch_model::CanonicalTransformer as TchTransformer;
use canonical_heuristics::HeuristicModel as OnnxModel;
use serde_json::json;
#[cfg(feature = "tch")]
use tch::{Device, Tensor};

fn rss_kb() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        }
    }
    0
}

fn hwm_kb() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        }
    }
    0
}

fn gpu_used_mb() -> Option<f64> {
    let pid = std::process::id();
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    for line in s.lines() {
        let mut parts = line.split(',');
        let p: u32 = parts.next()?.trim().parse().ok()?;
        if p == pid {
            return parts.next()?.trim().parse().ok();
        }
    }
    None
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn parse_flag(args: &[String], name: &str, default: &str) -> String {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn seeded(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn seeded_steps(n: usize, seed: u64) -> Vec<i64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) % 12) as i64
        })
        .collect()
}

fn emit(
    backend: &str,
    seq: usize,
    repeats: usize,
    threads: usize,
    load_ms: f64,
    first_ms: f64,
    times_ms: Vec<f64>,
    rss0: u64,
    rss_load: u64,
) {
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    println!(
        "{}",
        json!({
            "impl": backend,
            "seq": seq,
            "repeats": repeats,
            "threads": threads,
            "load_ms": load_ms,
            "first_ms": first_ms,
            "median_ms": median(times_ms.clone()),
            "min_ms": times_ms.iter().cloned().fold(f64::INFINITY, f64::min),
            "mean_ms": mean,
            "rss_before_kb": rss0,
            "rss_load_kb": rss_load,
            "rss_end_kb": rss_kb(),
            "rss_peak_kb": hwm_kb(),
            "gpu_used_mb": gpu_used_mb(),
        })
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(parse_flag(&args, "--dir", ""));
    let backend = parse_flag(&args, "--backend", "onnx");
    let seq: usize = parse_flag(&args, "--seq", "64").parse().unwrap();
    let seqs: Vec<usize> = parse_flag(&args, "--seqs", "")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let seq_list = if seqs.is_empty() { vec![seq] } else { seqs };
    let repeats: usize = parse_flag(&args, "--repeats", "10").parse().unwrap();
    let threads: usize = parse_flag(&args, "--threads", "8").parse().unwrap();
    let seed: u64 = parse_flag(&args, "--seed", "1").parse().unwrap();

    std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    std::env::set_var("OMP_NUM_THREADS", threads.to_string());
    #[cfg(feature = "tch")]
    tch::set_num_threads(threads as i32);
    #[cfg(feature = "cuda")]
    canonical_heuristics::enable_cubecl_disk_cache();

    let cfg: canonical_heuristics::ModelConfig =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let dim = cfg.input_dim;
    let rss0 = rss_kb();
    let load_t = Instant::now();

    match backend.as_str() {
        #[cfg(feature = "tch")]
        "tch" => {
            let model = tch_model::HeuristicModel::load(&dir).unwrap();
            let rss_load = rss_kb();
            let load_ms = load_t.elapsed().as_secs_f64() * 1e3;
            for seq in seq_list {
                let x = seeded(seq * dim, seed);
                let xt = Tensor::from_slice(&x).reshape([seq as i64, dim as i64]);
                let t0 = Instant::now();
                let y = model.logits(&xt).unwrap();
                let _ = y.double_value(&[0, 0]);
                let first_ms = t0.elapsed().as_secs_f64() * 1e3;
                let mut times = Vec::with_capacity(repeats);
                for _ in 0..repeats {
                    let t = Instant::now();
                    let y = model.logits(&xt).unwrap();
                    let _ = y.double_value(&[0, 0]);
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
                emit(
                    &backend, seq, repeats, threads, load_ms, first_ms, times, rss0, rss_load,
                );
            }
        }
        #[cfg(feature = "tch")]
        "tch-cuda" => {
            let model =
                TchTransformer::load_on(dir.join("transformer.ts"), Device::Cuda(0)).unwrap();
            let rss_load = rss_kb();
            let load_ms = load_t.elapsed().as_secs_f64() * 1e3;
            for seq in seq_list {
                let x = seeded(seq * dim, seed);
                let xt = Tensor::from_slice(&x)
                    .reshape([seq as i64, dim as i64])
                    .to(Device::Cuda(0));
                let t0 = Instant::now();
                let y = model.forward(&xt).unwrap();
                let _ = y.double_value(&[0, 0]);
                let first_ms = t0.elapsed().as_secs_f64() * 1e3;
                let mut times = Vec::with_capacity(repeats);
                for _ in 0..repeats {
                    let t = Instant::now();
                    let y = model.forward(&xt).unwrap();
                    let _ = y.double_value(&[0, 0]);
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
                emit(
                    &backend, seq, repeats, threads, load_ms, first_ms, times, rss0, rss_load,
                );
            }
        }
        "burn" => {
            let model = burn_model::HeuristicModel::load(&dir).unwrap();
            let rss_load = rss_kb();
            let load_ms = load_t.elapsed().as_secs_f64() * 1e3;
            for seq in seq_list {
                let x = seeded(seq * dim, seed);
                let xt = burn::tensor::Tensor::<burn::backend::NdArray<f32>, 2>::from_data(
                    burn::tensor::TensorData::new(x, [seq, dim]),
                    &Default::default(),
                );
                let t0 = Instant::now();
                let y = model.logits(xt.clone());
                let _ = y
                    .clone()
                    .slice([0..1, 0..1])
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()[0];
                let first_ms = t0.elapsed().as_secs_f64() * 1e3;
                let mut times = Vec::with_capacity(repeats);
                for _ in 0..repeats {
                    let t = Instant::now();
                    let y = model.logits(xt.clone());
                    let _ = y
                        .clone()
                        .slice([0..1, 0..1])
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap()[0];
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
                emit(
                    &backend, seq, repeats, threads, load_ms, first_ms, times, rss0, rss_load,
                );
            }
        }
        "burn-cuda" => {
            #[cfg(not(feature = "cuda"))]
            panic!("rebuild with --features cuda");
            #[cfg(feature = "cuda")]
            {
                use burn::backend::Cuda;
                use burn::tensor::backend::Backend;
                type Gpu = Cuda<f32, i32>;
                let device = Default::default();
                let model = burn_model::CanonicalTransformer::<Gpu>::load_on(
                    &dir.join("transformer.safetensors"),
                    &cfg,
                    &device,
                )
                .unwrap();
                let rss_load = rss_kb();
                let load_ms = load_t.elapsed().as_secs_f64() * 1e3;
                for seq in seq_list {
                    let x = seeded(seq * dim, seed);
                    let xt = burn::tensor::Tensor::<Gpu, 2>::from_data(
                        burn::tensor::TensorData::new(x, [seq, dim]),
                        &device,
                    );
                    let _ = Gpu::sync(&device);
                    let t0 = Instant::now();
                    let y = model.forward(xt.clone());
                    let _ = y
                        .clone()
                        .slice([0..1, 0..1])
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap()[0];
                    let first_ms = t0.elapsed().as_secs_f64() * 1e3;
                    let mut times = Vec::with_capacity(repeats);
                    for _ in 0..repeats {
                        let t = Instant::now();
                        let y = model.forward(xt.clone());
                        let _ = y
                            .clone()
                            .slice([0..1, 0..1])
                            .into_data()
                            .to_vec::<f32>()
                            .unwrap()[0];
                        times.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                    emit(
                        &backend, seq, repeats, threads, load_ms, first_ms, times, rss0, rss_load,
                    );
                }
            }
        }
        "onnx" | "onnx-cuda" => {
            #[cfg(not(feature = "cuda"))]
            if backend == "onnx-cuda" {
                panic!("rebuild with --features cuda");
            }
            let model = OnnxModel::load(&dir).unwrap();
            let impl_name = OnnxModel::backend_name();
            let path_len = parse_flag(&args, "--path-len", "16").parse().unwrap();
            let rss_load = rss_kb();
            let load_ms = load_t.elapsed().as_secs_f64() * 1e3;
            for seq in seq_list {
                let left = seeded_steps(seq * path_len, seed);
                let right = seeded_steps(seq * path_len, seed.wrapping_add(17));
                let idx: Vec<i64> = (0..seq as i64).collect();
                let t0 = Instant::now();
                let y = model.scores_from_packed(&left, &right, seq, path_len, &idx, &idx);
                let _ = y[0];
                let first_ms = t0.elapsed().as_secs_f64() * 1e3;
                let mut times = Vec::with_capacity(repeats);
                for _ in 0..repeats {
                    let t = Instant::now();
                    let y = model.scores_from_packed(&left, &right, seq, path_len, &idx, &idx);
                    let _ = y[0];
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
                emit(
                    impl_name, seq, repeats, threads, load_ms, first_ms, times, rss0, rss_load,
                );
            }
        }
        other => panic!("unknown backend {other}"),
    }
}
