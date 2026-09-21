#!/usr/bin/env python3
"""Run Canonical on Mathlib and keep proofs the model finds that uniform search does not.

Uniform search runs first (no heuristics). Failures are retried with the ONNX
model. New solutions are written to:

  OUT/log.jsonl              every attempt
  OUT/new_solutions.jsonl    model-only finds
  OUT/new_solutions.lean     pretty-printed examples
  OUT/bins/*.bin             MessagePack dumps from the model pass

Requires a heuristics-enabled `libcanonical_lean` (from the Canonical repo:
`python3 build_lean.py` after `cargo build -p canonical_lean --release`).

Examples:

  python3 scripts/run_mathlib_eval.py --timeout 5 Mathlib.Data.Bool.Basic
  python3 scripts/run_mathlib_eval.py --all --timeout 5 --out mathlib_eval_out
  python3 scripts/run_mathlib_eval.py --dry-run Mathlib.Data.Nat.Basic
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
LEAN_DIR = REPO / "lean"
DEFAULT_MATHLIB = LEAN_DIR / ".lake" / "packages" / "mathlib"
DEFAULT_WEIGHTS = REPO.parent / "CanonicalHeuristics" / "export_rust"


def module_from_lean_file(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def list_mathlib_modules(mathlib_src: Path, exclude_prefixes: list[str]) -> list[str]:
    # Walk mathlib_src/Mathlib but compute module names relative to mathlib_src, so
    # e.g. mathlib_src/Mathlib/Data/Nat/Basic.lean becomes "Mathlib.Data.Nat.Basic"
    # (the actual importable module name), not "Data.Nat.Basic".
    files = sorted((mathlib_src / "Mathlib").rglob("*.lean"))
    modules = []
    for f in files:
        if f.name == "lakefile.lean":
            continue
        mod = module_from_lean_file(f, mathlib_src)
        if any(mod == p or mod.startswith(p + ".") for p in exclude_prefixes):
            continue
        modules.append(mod)
    return modules


def already_done(out: Path, module: str) -> bool:
    marker = out / "done" / f"{module}.txt"
    return marker.is_file()


def mark_done(out: Path, module: str, rc: int) -> None:
    d = out / "done"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{module}.txt").write_text(f"{rc}\n")


def run_module(module: str, args: argparse.Namespace, env: dict[str, str]) -> int:
    cmd = [
        "lake",
        "exe",
        "mathlib_eval",
        "--timeout",
        str(args.timeout),
        "--out",
        str(args.out),
        "--mode",
        args.mode,
        "--premises",
        args.premises,
        "--max-premises",
        str(args.max_premises),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.dry_run:
        cmd.append("--dry-run")
    cmd.append(module)
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=LEAN_DIR, env=env)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="*", help="Lean module names, e.g. Mathlib.Data.Nat.Basic")
    parser.add_argument("--all", action="store_true", help="every Mathlib/*.lean module")
    parser.add_argument("--timeout", type=int, default=5, help="seconds per uniform/model attempt")
    parser.add_argument("--out", type=Path, default=Path("mathlib_eval_out"))
    parser.add_argument("--mode", choices=["compare", "uniform", "model"], default="compare")
    parser.add_argument("--premises", choices=["proof", "none", "suggestions"], default="proof")
    parser.add_argument("--max-premises", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="max theorems per module")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip modules with OUT/done/MODULE.txt")
    parser.add_argument(
        "--mathlib-src",
        type=Path,
        default=DEFAULT_MATHLIB,
        help="Mathlib package root (contains Mathlib/*.lean)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="module prefix to skip with --all (repeatable)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="directory with config.json (ONNX weights are compiled into the plugin)",
    )
    args = parser.parse_args()

    if shutil.which("lake") is None:
        print("`lake` not found on PATH (needed to run `lake exe mathlib_eval`); "
              "install elan/lake or add it to PATH", file=sys.stderr)
        return 2

    args.out = args.out if args.out.is_absolute() else (Path.cwd() / args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bins").mkdir(exist_ok=True)
    (args.out / "done").mkdir(exist_ok=True)

    modules = list(args.modules)
    if args.all:
        src = args.mathlib_src / "Mathlib"
        if not src.is_dir():
            print(f"Mathlib sources not found at {src} (lake update in Canonical/lean?)", file=sys.stderr)
            return 2
        modules.extend(list_mathlib_modules(args.mathlib_src, args.exclude))
    # de-dupe, keep order
    seen = set()
    uniq = []
    for m in modules:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    modules = uniq
    if not modules:
        parser.error("pass MODULE... or --all")

    env = os.environ.copy()
    env["CANONICAL_SAVE_RESULTS_DIR"] = str(args.out / "bins")
    if args.weights.is_dir():
        env["CANONICAL_HEURISTICS_WEIGHTS"] = str(args.weights)

    failed = []
    for i, module in enumerate(modules, 1):
        if args.resume and already_done(args.out, module):
            print(f"[{i}/{len(modules)}] skip {module}", flush=True)
            continue
        print(f"[{i}/{len(modules)}] {module}", flush=True)
        rc = run_module(module, args, env)
        mark_done(args.out, module, rc)
        if rc != 0:
            failed.append((module, rc))
            print(f"module failed rc={rc}: {module}", file=sys.stderr, flush=True)

    news = args.out / "new_solutions.jsonl"
    n_new = len(news.read_text().splitlines()) if news.is_file() else 0
    print(f"done. new solutions: {n_new}  failures: {len(failed)}  out: {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
