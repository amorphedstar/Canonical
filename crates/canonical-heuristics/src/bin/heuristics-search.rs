//! Run Canonical search on exported Example.bin problems.
//!
//! Matches the Lean tactic: compile with uniform weights, spawn the model,
//! search, and adopt scores when they land.
//!
//!   heuristics-search [--uniform] [--timeout-secs 15] [--limit 50000] FILE.bin...

use std::panic;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use canonical_compat::ai::{Example, Inference};
use canonical_compat::ir::{BindMap, IRType};
use canonical_core::core::Term;
use canonical_core::prover::Prover;
use canonical_core::search::RUN;
use canonical_core::stats::{reset, STEP_COUNT};
use canonical_heuristics::{spawn_heuristic_compilation, HEURISTICS_APPLIED};
use serde_json::json;

fn parse_flag(args: &[String], name: &str, default: &str) -> String {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn load_problem(path: &Path) -> (String, IRType, usize, usize) {
    let p = path.to_string_lossy().into_owned();
    let prev = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let example = panic::catch_unwind(|| Example::load(p.clone()));
    panic::set_hook(prev);
    if let Ok(ex) = example {
        return (ex.name, ex.problem, ex.goals.len(), ex.premises.len());
    }
    let inf = Inference::load(p);
    (inf.name, inf.problem, inf.goals.len(), inf.premises.len())
}

fn files(args: &[String]) -> Vec<PathBuf> {
    args.iter()
        .filter(|a| !a.starts_with('-') && !args.windows(2).any(|w| w[0].starts_with('-') && &w[1] == *a))
        .map(PathBuf::from)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let timeout = Duration::from_secs_f64(parse_flag(&args, "--timeout-secs", "20").parse().unwrap());
    let limit: u32 = parse_flag(&args, "--limit", "2000000").parse().unwrap();
    let uniform = has_flag(&args, "--uniform");
    let paths = files(&args);
    if paths.is_empty() {
        eprintln!("usage: heuristics-search [--uniform] [--timeout-secs 15] [--limit 50000] FILE.bin...");
        std::process::exit(2);
    }

    println!(
        "{}",
        json!({
            "mode": if uniform { "uniform" } else { "heuristics" },
            "timeout_secs": timeout.as_secs_f64(),
            "limit": limit,
        })
    );

    for path in paths {
        let started = Instant::now();
        let outcome = panic::catch_unwind(|| run_one(&path, timeout, limit, uniform));
        match outcome {
            Ok(row) => println!("{row}"),
            Err(_) => println!(
                "{}",
                json!({
                    "file": path.display().to_string(),
                    "error": "panic",
                    "elapsed_ms": started.elapsed().as_secs_f64() * 1e3,
                })
            ),
        }
    }
}

fn run_one(path: &Path, timeout: Duration, limit: u32, uniform: bool) -> serde_json::Value {
    let (name, problem, n_goals, n_premises) = load_problem(path);
    let mut binds = BindMap::default();
    let mut tokens = Vec::new();
    let (tb, problem_bind) = problem.to_problem(name.clone(), &mut binds, &mut tokens);

    reset();
    HEURISTICS_APPLIED.store(false, Ordering::Relaxed);
    let mut owned_linked = Vec::new();
    let t_compile = Instant::now();
    let prover = Prover::new(tb.downgrade(), problem_bind.downgrade(), &mut owned_linked, None);
    let compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;

    let infer = if uniform {
        None
    } else {
        Some(spawn_heuristic_compilation(
            tb.downgrade(),
            problem_bind.downgrade(),
            tokens.clone(),
            binds.bind_paths(),
        ))
    };

    let found = Arc::new(AtomicBool::new(false));
    let found_cb = found.clone();
    let cancel_watchdog = Arc::new(AtomicBool::new(false));
    let cancel_w = cancel_watchdog.clone();
    let t_search = Instant::now();
    let stop = thread::spawn(move || {
        let start = Instant::now();
        while !cancel_w.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(50));
            if start.elapsed() >= timeout || STEP_COUNT.load(Ordering::Relaxed) >= limit {
                RUN.store(false, Ordering::Relaxed);
                break;
            }
        }
    });

    let (result, _) = prover.prove(
        &|_term: Term| {
            found_cb.store(true, Ordering::Relaxed);
            RUN.store(false, Ordering::Relaxed);
        },
        false,
    );
    let search_ms = t_search.elapsed().as_secs_f64() * 1e3;
    RUN.store(false, Ordering::Relaxed);
    cancel_watchdog.store(true, Ordering::Relaxed);
    let _ = stop.join();
    if let Some(h) = infer {
        let _ = h.join();
    }

    json!({
        "file": path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string(),
        "name": name,
        "n_tokens": tokens.len(),
        "n_goals": n_goals,
        "n_premises": n_premises,
        "found": found.load(Ordering::Relaxed),
        "steps": result.steps,
        "solutions": result.solution_count,
        "compile_ms": compile_ms,
        "search_ms": search_ms,
        "heuristics_applied": HEURISTICS_APPLIED.load(Ordering::Relaxed),
        "uniform": uniform,
    })
}
