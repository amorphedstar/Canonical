import Canonical
import Lean
import Lean.Data.Json

open Lean Meta Canonical Json

/-- Lean binding for the FFI flag in `canonical_lean`. Matches the `@[never_extract]`
convention used by the other `Unit`/effect-only externs in `Canonical.Main` (`cancel`,
`refine`), so repeated calls with the same argument aren't merged or dropped by the
compiler -- this is called with alternating 0/1 across theorems and must actually fire
each time. -/
@[never_extract, extern "set_heuristics_enabled"]
opaque setHeuristicsEnabled : UInt8 → IO Unit

structure Args where
  timeout : UInt64 := 5
  outDir : System.FilePath := "mathlib_eval_out"
  mode : String := "compare"
  premises : String := "proof"
  maxPremises : Nat := 64
  limit : Option Nat := none
  dryRun : Bool := false
  modules : Array Name := #[]
  deriving Inhabited

def usage : String :=
  "mathlib_eval [options] MODULE...\n\
   \n\
   Run Canonical on theorems defined in the given Mathlib modules.\n\
   Default mode `compare` tries uniform search first, then the model\n\
   only on failures, and writes proofs the model found that uniform did not.\n\
   \n\
   Options:\n\
     --timeout SECS       per-attempt timeout (default 5)\n\
     --out DIR            output directory (default mathlib_eval_out)\n\
     --mode compare|uniform|model\n\
     --premises proof|none|suggestions\n\
     --max-premises N     cap on proof-used constants (default 64)\n\
     --limit N            max theorems per run\n\
     --dry-run            list theorems, do not search\n"

def toModuleName (s : String) : Name :=
  s.splitOn "." |>.foldl (init := Name.anonymous) fun n p => Name.str n p

partial def parseArgs (args : List String) (acc : Args) : Except String Args :=
  match args with
  | [] => .ok acc
  | "--help" :: _ => .error usage
  | "--timeout" :: n :: rest =>
    match n.toNat? with
    | some t => parseArgs rest { acc with timeout := t.toUInt64 }
    | none => .error s!"bad --timeout {n}"
  | "--out" :: d :: rest => parseArgs rest { acc with outDir := d }
  | "--mode" :: m :: rest =>
    if m == "compare" || m == "uniform" || m == "model" then
      parseArgs rest { acc with mode := m }
    else .error s!"bad --mode {m}"
  | "--premises" :: p :: rest =>
    if p == "proof" || p == "none" || p == "suggestions" then
      parseArgs rest { acc with premises := p }
    else .error s!"bad --premises {p}"
  | "--max-premises" :: n :: rest =>
    match n.toNat? with
    | some t => parseArgs rest { acc with maxPremises := t }
    | none => .error s!"bad --max-premises {n}"
  | "--limit" :: n :: rest =>
    match n.toNat? with
    | some t => parseArgs rest { acc with limit := some t }
    | none => .error s!"bad --limit {n}"
  | "--dry-run" :: rest => parseArgs rest { acc with dryRun := true }
  | s :: rest =>
    if s.startsWith "--" then .error s!"unknown flag {s}\n{usage}"
    else parseArgs rest { acc with modules := acc.modules.push (toModuleName s) }

def pp (e : Expr) : MetaM String := do
  let fmt ← withOptions applyOptions do
    withOptions (fun o => pp.fullNames.set o true) do
      ppExpr e
  return fmt.pretty

def exceptionString (e : Exception) : MetaM String := do
  e.toMessageData.toString

structure Attempt where
  found : Bool := false
  steps : Nat := 0
  proof? : Option Expr := none
  typecheck : Bool := false
  error? : Option String := none
  deriving Inhabited

def tryCanonical (name : Name) (type : Expr) (userPremises : Array Name)
    (timeout : UInt64) (config : Canonical.Config) : MetaM Attempt := do
  let mvar ← mkFreshExprMVar (some type) (userName := name)
  let goal := mvar.mvarId!
  try
    let (premises, structs) ← getPremises goal userPremises config
    let (processedGoal, reconstruct) ← withArityUnfold config.monomorphize do
      preprocess goal config structs
    let typ ← withArityUnfold config.monomorphize do processedGoal.withContext do
      toCanonical (← processedGoal.getType) premises (structs.push ``Pi) config
    let result ← runCanonical typ name.toString timeout config
    if result.terms.isEmpty then
      return { steps := result.steps.toNat }
    let proofs ← postprocess result processedGoal config reconstruct
    match proofs[0]? with
    | none => return { found := true, steps := result.steps.toNat, error? := some "empty postprocess" }
    | some proof =>
      let ok ← isTypeCorrect proof
      return { found := true, steps := result.steps.toNat, proof? := some proof, typecheck := ok }
  catch e =>
    return { error? := some (← exceptionString e) }

def proofPremises (info : ConstantInfo) (us : List Level) (self : Name) (max : Nat) : MetaM (Array Name) := do
  if !info.hasValue (allowOpaque := true) then
    return #[]
  let value := info.instantiateValueLevelParams! us (allowOpaque := true)
  let env ← getEnv
  -- Match the tactic's premise list: lemma names only, not types/constructors/defs.
  return (value.getUsedConstants.filter fun n =>
    n != self && !n.isInternalDetail &&
      match env.find? n with
      | some (.thmInfo _) => true
      | _ => false).take max

def moduleTheorems (env : Environment) (wanted : NameSet) : Array (Name × ConstantInfo) :=
  env.constants.fold (init := #[]) fun acc name info =>
    match info, env.getModuleIdxFor? name with
    | .thmInfo _, some idx =>
      let modName := env.header.moduleNames[idx.toNat]!
      if wanted.contains modName && !name.isInternalDetail then
        acc.push (name, info)
      else acc
    | _, _ => acc

def appendLine (path : System.FilePath) (line : String) : IO Unit := do
  if let some p := path.parent then
    IO.FS.createDirAll p
  let h ← IO.FS.Handle.mk path .append
  h.putStrLn line
  h.flush

def attemptJson (name : Name) (module : Name) (phase : String) (premises : Array Name) (a : Attempt) : MetaM Json := do
  let mut fields : Array (String × Json) := #[
    ("name", toJson name.toString),
    ("module", toJson module.toString),
    ("phase", toJson phase),
    ("found", toJson a.found),
    ("steps", toJson a.steps),
    ("typecheck", toJson a.typecheck),
    ("premises", toJson (premises.map (·.toString)))
  ]
  if let some err := a.error? then
    fields := fields.push ("error", toJson err)
  if let some proof := a.proof? then
    fields := fields.push ("proof", toJson (← pp proof))
  return Json.mkObj fields.toList

def processTheorem (args : Args) (name : Name) (info : ConstantInfo) : MetaM Unit := do
  let env ← getEnv
  let some idx := env.getModuleIdxFor? name | return
  let module := env.header.moduleNames[idx.toNat]!
  let us ← mkFreshLevelMVarsFor info
  let type := info.instantiateTypeLevelParams us
  unless ← isProp type do
    return
  let config : Canonical.Config := {
    count := 1
    suggestions := args.premises == "suggestions"
    simp := true
    monomorphize := true
    destruct := true
  }
  let userPremises ← match args.premises with
    | "proof" => proofPremises info us name args.maxPremises
    | "suggestions" => pure #[]
    | _ => pure #[]
  let typeS ← pp type
  let saved ← saveState
  if args.dryRun then
    let row := Json.mkObj [
      ("name", toJson name.toString),
      ("module", toJson module.toString),
      ("dry_run", toJson true),
      ("type", toJson typeS),
      ("premises", toJson (userPremises.map (·.toString)))
    ]
    IO.println row.compress
    appendLine (args.outDir / "log.jsonl") row.compress
    restoreState saved
    return

  let runUniform := args.mode == "compare" || args.mode == "uniform"
  let runModel := args.mode == "compare" || args.mode == "model"

  let mut uniform : Attempt := {}
  if runUniform then
    setHeuristicsEnabled 0
    uniform ← tryCanonical name type userPremises args.timeout config
    let row ← attemptJson name module "uniform" userPremises uniform
    IO.println row.compress
    appendLine (args.outDir / "log.jsonl") row.compress
    if args.mode == "uniform" then
      restoreState saved
      return

  if runModel && !(args.mode == "compare" && uniform.found) then
    setHeuristicsEnabled 1
    let model ← tryCanonical name type userPremises args.timeout config
    let row ← attemptJson name module "model" userPremises model
    IO.println row.compress
    appendLine (args.outDir / "log.jsonl") row.compress
    let isNew := model.found && (args.mode == "model" || !uniform.found)
    if isNew then
      let mut extra : Array (String × Json) := #[
        ("name", toJson name.toString),
        ("module", toJson module.toString),
        ("type", toJson typeS),
        ("uniform_steps", toJson uniform.steps),
        ("model_steps", toJson model.steps),
        ("typecheck", toJson model.typecheck),
        ("premises", toJson (userPremises.map (·.toString)))
      ]
      if let some err := model.error? then
        extra := extra.push ("error", toJson err)
      let proofS ← match model.proof? with
        | some p => pp p
        | none => pure ""
      extra := extra.push ("proof", toJson proofS)
      let sol := Json.mkObj extra.toList
      appendLine (args.outDir / "new_solutions.jsonl") sol.compress
      let lean :=
        s!"/-- {name} (model found, uniform did not). \
           model_steps={model.steps}, uniform_steps={uniform.steps} -/\n\
           example : {typeS} :=\n  {if proofS.isEmpty then "sorry" else proofS}\n"
      appendLine (args.outDir / "new_solutions.lean") lean
  restoreState saved

unsafe def run (args : Args) : IO UInt32 := do
  if args.modules.isEmpty then
    IO.eprintln usage
    return 2
  initSearchPath (← findSysroot)
  enableInitializersExecution
  let imports := #[{ module := `Canonical }] ++ args.modules.map fun m => { module := m }
  let env ← importModules imports {} (trustLevel := 1024) (loadExts := true)
  let wanted := args.modules.foldl (init := {}) (fun s n => s.insert n)
  let mut thms := moduleTheorems env wanted
  thms := thms.qsort fun a b => Name.lt a.1 b.1
  if let some n := args.limit then
    thms := thms.take n
  IO.eprintln s!"mathlib_eval: {thms.size} theorems in {args.modules.size} module(s), mode={args.mode}, timeout={args.timeout}s"
  let ctx : Core.Context := { fileName := "<mathlib_eval>", fileMap := default, maxHeartbeats := 0 }
  let _ ← MetaM.toIO (do
    let mut i := 0
    for (name, info) in thms do
      i := i + 1
      IO.eprint s!"[{i}/{thms.size}] {name}\n"
      try
        processTheorem args name info
      catch e =>
        let msg ← exceptionString e
        let row := Json.mkObj [
          ("name", toJson name.toString),
          ("error", toJson msg)
        ]
        IO.println row.compress
        appendLine (args.outDir / "log.jsonl") row.compress
    ) ctx { env }
  return 0

unsafe def main (argv : List String) : IO UInt32 := do
  if argv.contains "--help" || argv.contains "-h" then
    IO.println usage
    return 0
  match parseArgs argv {} with
  | .error e =>
    IO.eprintln e
    return 2
  | .ok args =>
    IO.FS.createDirAll args.outDir
    IO.FS.createDirAll (args.outDir / "bins")
    run args
