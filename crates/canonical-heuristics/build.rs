use burn_onnx::{LoadStrategy, ModelGen};

fn main() {
    let onnx = "onnx/transformer.onnx";
    println!("cargo:rerun-if-changed={onnx}");
    if !std::path::Path::new(onnx).exists() {
        panic!(
            "missing {onnx}. Export it with:\n  python CanonicalHeuristics/scripts/export_onnx.py --out {onnx}"
        );
    }
    ModelGen::new()
        .input(onnx)
        .out_dir("model/")
        .load_strategy(LoadStrategy::File)
        .run_from_script();
}
