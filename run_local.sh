#!/usr/bin/env bash
# =============================================================================
# run_local.sh -- run the Basis_Sharing baseline on a single local GPU box.
#                 No Slurm.
#
# This is the Slurm-free equivalent of the Trillium runner (baselines/run.sh in
# the ARKS comparison harness). Same compression, same evaluation, same output
# JSON -- the only thing removed is the batch scheduler.
#
# WHAT THIS MEASURES
#   Basis_Sharing performs the compression. Perplexity comes from
#   local/shared_eval.py, which transcribes the ARKS evaluation recipe, NOT from
#   test.py:compute_ppl -- that one is a strided sliding window over a
#   "\n\n"-joined corpus and is not comparable to the other baselines.
#
# RATIO CONVENTION
#   Basis_Sharing takes its strength from the YAML, not the CLI, and
#   `compression_ratio` there is the percentage of parameters REMOVED
#   (utils.py does `compression_ratio = 1 - compression_ratio / 100`).
#   rho = fraction KEPT, so rho=0.60 -> 40 and rho=0.33 -> 67. Both numbers are
#   written into every generated YAML so the mapping is never in doubt.
#
# SINGLE-GPU / LAYERWISE -- the honest answer for this baseline
#   Basis_Sharing runs on ONE GPU (every reference result up to opt-13b and
#   Llama-3.1-8B was produced on a single H100), but it does so by keeping the
#   whole model resident, not by streaming layers.
#     * Calibration (calib.py) is WHOLE-MODEL. BS_CALIB_GROUPS slices which
#       layers have a hook attached and then runs the ENTIRE 256-sample forward
#       through the ENTIRE model once per group. It bounds the HOST RAM held by
#       the Gram matrices; it is not a layerwise fix. Do not read it as one.
#     * Whitening (Cholesky + inverse of each Gram) IS one group at a time, on
#       the CPU in float64 -- a CPU bottleneck, not a GPU one. Note it does not
#       reduce GPU footprint: the dense reference model is still resident on the
#       card throughout, so this stage frees nothing there.
#     * Decomposition is per-group arithmetic (2 layers at a time for k/q/v/up,
#       1 for o/down) but reads weights straight off a whole-model-resident
#       reference, so residency is still whole-model.
#     * The reference model is loaded with device_map="auto": on a multi-GPU box
#       it will shard. GPUS=0 below pins it to one card, matching the reference
#       runs. During match_state_dict two models briefly coexist.
#     * Evaluation is whole-model. SHARED_EVAL_LAYERWISE=1 exists but is only
#       PARTIALLY effective here, and for BOTH families, not just OPT: the
#       layerwise path detaches `layers` only, so every shared-basis ModuleDict
#       stays pinned on the device -- 42.9% of the compressed linear params on
#       opt-6.7b/13b, 36.6% on Llama-3.1-8B. run_basis_sharing.py also moves the
#       model to cuda before evaluating regardless. Treat layerwise eval as
#       unsupported for this baseline rather than assume it works.
#   NO BACKWARD PASS runs on this path. Two real ones exist -- train.py (full
#   fine-tuning) and lora.py (peft) -- but neither is reachable from this
#   runner, and the generated YAMLs set after_calibration_update_args.update to
#   false, which also disables the closed-form coefficient refit.
#
# USAGE
#   ./run_local.sh setup [--force]        build the venv (transformers 4.45.2)
#   ./run_local.sh yaml                   (re)generate local/yaml/*.yaml
#   ./run_local.sh doctor                 check GPU, venv, imports, yaml
#   ./run_local.sh run MODEL RHO          one (model, rho)
#   ./run_local.sh sweep [MODEL ...]      every model x every rho, serially
#   ./run_local.sh collect                runs/*.json -> results.csv
#
#   ./run_local.sh setup && ./run_local.sh yaml
#   ./run_local.sh run facebook/opt-1.3b 0.60
#
# ENVIRONMENT KNOBS (all optional)
#   VENV=<dir>        venv location                  default ./.venv-basis
#   PYBIN=<python>    interpreter used to build it   default python3.11 else python3
#   RUNS_DIR=<dir>    result JSONs                   default ./local/runs/basis_sharing
#   YAML_DIR=<dir>    generated configs              default ./local/yaml
#   LOG_DIR, RHOS, MODELS, GPUS, OFFLINE, TORCH_SPEC, EXTRA_ARGS
#                     -- as in the other two baselines' run_local.sh
#   BS_MAP_PROCS=1    datasets .map() workers. Upstream's 4 OOMs on big corpora;
#                     prepare_data.py already defaults this to 1.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

METHOD=basis_sharing
VENV="${VENV:-$HERE/.venv-basis}"
RUNS_DIR="${RUNS_DIR:-$HERE/local/runs/$METHOD}"
YAML_DIR="${YAML_DIR:-$HERE/local/yaml}"
LOG_DIR="${LOG_DIR:-$HERE/local/logs}"
RHOS="${RHOS:-0.60 0.33}"
GPUS="${GPUS:-0}"

# Pinned to exactly what the reference runs resolved to. Both exist on PyPI.
# transformers 4.45.2 is NOT negotiable: LlamaSdpaAttention and
# OPTAttention._shape, which models/*.py subclass, are both gone in 4.48+.
TORCH_SPEC="${TORCH_SPEC:-torch==2.14.0}"
TRANSFORMERS_SPEC="transformers==4.45.2"
NUMPY_SPEC="numpy==1.26.4"

MODELS="${MODELS:-facebook/opt-125m facebook/opt-350m facebook/opt-1.3b \
facebook/opt-2.7b meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B \
facebook/opt-6.7b facebook/opt-13b facebook/opt-30b meta-llama/Llama-3.1-8B}"

log() { echo "[run_local] $*"; }
die() { echo "[run_local] ERROR: $*" >&2; exit 1; }
rho_tag() { awk -v r="$1" 'BEGIN{printf "%.0f", r*100}'; }

setup_env() {
  mkdir -p "$RUNS_DIR" "$LOG_DIR" "$YAML_DIR"
  export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  [[ "$GPUS" == "all" ]] || export CUDA_VISIBLE_DEVICES="$GPUS"
  local t="${LOCAL_CPUS:-$(nproc)}"
  export OMP_NUM_THREADS="$t" OPENBLAS_NUM_THREADS="$t" MKL_NUM_THREADS="$t"
  # Basis_Sharing allocates in bursts (a whole compressed stack is materialised
  # on the host, then matched against the reference); expandable segments keep
  # the allocator from fragmenting across that.
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  [[ "${OFFLINE:-0}" == "1" ]] && export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  return 0
}

do_setup() {
  local force=0; [[ "${1:-}" == "--force" ]] && force=1
  local py="${PYBIN:-}"
  if [[ -z "$py" ]]; then
    py=$(command -v python3.11 || command -v python3) || die "no python3 on PATH"
  fi
  if [[ -d "$VENV" && $force -eq 0 ]]; then
    log "$VENV exists -- reusing it (./run_local.sh setup --force to rebuild)"
  else
    (( force )) && rm -rf "$VENV"
    log "building $VENV with $py ($("$py" -V 2>&1))"
    "$py" -m venv "$VENV"
  fi
  "$VENV/bin/pip" install --upgrade pip setuptools wheel
  "$VENV/bin/pip" install "$TORCH_SPEC" "$NUMPY_SPEC" "$TRANSFORMERS_SPEC" \
      scipy scikit_learn safetensors sentencepiece datasets accelerate \
      pyarrow \
      pandas tqdm pyyaml protobuf
  "$VENV/bin/python" - <<'PY'
import torch, transformers, numpy
print(f"  transformers {transformers.__version__}  torch {torch.__version__} "
      f"(cuda {torch.version.cuda})  numpy {numpy.__version__}  "
      f"cuda_available={torch.cuda.is_available()}")
assert transformers.__version__.startswith("4.45"), \
    "models/*.py subclass LlamaSdpaAttention / OPTAttention._shape, gone in 4.48+"
PY
}

do_yaml() {
  mkdir -p "$YAML_DIR"
  BS_YAML_OUT="$YAML_DIR" BS_RUNS_DIR="$RUNS_DIR" python3 "$HERE/local/gen_yaml.py"
}

do_doctor() {
  setup_env
  [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV -- run ./run_local.sh setup"
  log "GPU:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
    || die "nvidia-smi failed -- this script needs a CUDA GPU"
  local n; n=$(ls "$YAML_DIR"/*.yaml 2>/dev/null | wc -l)
  (( n > 0 )) || die "no configs in $YAML_DIR -- run ./run_local.sh yaml"
  log "$n yaml config(s) in $YAML_DIR"
  "$VENV/bin/python" - <<'PY'
import os, sys, torch, transformers
sys.path.insert(0, os.path.join(os.getcwd(), "local"))
sys.path.insert(0, os.getcwd())
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"devices={torch.cuda.device_count()}")
print(f"  transformers {transformers.__version__}")
import shared_eval
from config import ShareConfig, add_args
from model_factory import create_model
print("  imports OK (shared_eval + Basis_Sharing)")
PY
  log "doctor passed"
}

run_one() {
  local model="$1" rho="$2"
  [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV -- run ./run_local.sh setup"
  local tag="${model##*/}_rho$(rho_tag "$rho")"
  local y="$YAML_DIR/${tag}.yaml"
  [[ -f "$y" ]] || die "missing config: $y -- run ./run_local.sh yaml"
  local out="$RUNS_DIR/${tag}.json"
  local logf="$LOG_DIR/${METHOD}_${tag}.log"
  mkdir -p "$RUNS_DIR" "$LOG_DIR"

  # The ONE model-dependent conditional. opt-30b is ~120 GB in fp32 and cannot
  # be placed on one 80 GB card, and hooking all 48 layers at once costs
  # 3.90 GB x 48 = 175 GiB of HOST RAM. Two calibration groups halve that.
  # Calibration is ~11% of runtime, so the extra pass is cheap.
  local bs=()
  [[ "$model" == *30b* ]] && bs=(BS_TORCH_DTYPE=float16 BS_CALIB_GROUPS=2)

  echo "=========================================================="
  echo "[run_local] method=$METHOD model=$model rho=$rho"
  echo "[run_local] yaml=$y"
  echo "[run_local] out=$out"
  echo "[run_local] log=$logf"
  [[ ${#bs[@]} -gt 0 ]] && echo "[run_local] ${bs[*]}  (opt-30b does not fit one card in fp32)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  echo "=========================================================="

  # The cd matters for Basis_Sharing's own relative paths. Imports already
  # resolve via local/run_basis_sharing.py's sys.path, which is checkout-relative.
  ( cd "$HERE" && env "${bs[@]}" "$VENV/bin/python" \
      "$HERE/local/run_basis_sharing.py" --yaml "$y" --out "$out" ${EXTRA_ARGS:-} ) \
    2>&1 | tee "$logf"
}

do_sweep() {
  setup_env
  local models=("$@"); [[ ${#models[@]} -eq 0 ]] && read -r -a models <<< "$MODELS"
  local n=0 fail=0
  for m in "${models[@]}"; do
    for r in $RHOS; do
      n=$((n+1))
      if run_one "$m" "$r"; then log "OK   $m rho=$r"
      else fail=$((fail+1)); log "FAIL $m rho=$r -- continuing"; fi
    done
  done
  log "sweep done: $((n-fail))/$n succeeded"
  (( fail == 0 ))
}

do_collect() {
  "${VENV}/bin/python" - "$RUNS_DIR" <<'PY'
import csv, glob, json, os, re, sys
root = sys.argv[1]
CANON = re.compile(r"^[A-Za-z0-9._-]+_rho\d+\.json$")
rows, skipped = [], []
for f in sorted(glob.glob(os.path.join(root, "*.json"))):
    if not CANON.match(os.path.basename(f)):
        skipped.append(os.path.basename(f)); continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for corpus, ppl in (d.get("ppl") or {}).items():
        rows.append({
            "method": d.get("method"), "model": d.get("model"),
            "rho": d.get("rho_target"),
            "compression_ratio_removed_pct": d.get("compression_ratio_removed_pct"),
            "dense": d.get("dense"), "corpus": corpus, "ppl": ppl,
            "ppl_tokens": (d.get("ppl_tokens") or {}).get(corpus),
            "realised_linear_ratio": d.get("realised_linear_ratio"),
            # realised_total_ratio reads >1 here because the compressed model
            # unties lm_head -- the model grew. Never plot it.
            "realised_total_ratio": d.get("realised_total_ratio"),
            "compress_seconds": d.get("compress_seconds"), "source": f,
        })
if not rows:
    print("no results yet"); raise SystemExit(0)
rows.sort(key=lambda r: (r["model"] or "", str(r["rho"]), r["corpus"]))
out = os.path.join(root, "results.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} row(s) -> {out}")
if skipped:
    print(f"skipped {len(skipped)} non-canonical file(s): {', '.join(skipped)}")
print()
print(f"{'model':<24}{'rho':>6}  {'corpus':<11}{'ppl':>12}")
for r in rows:
    print(f"{(r['model'] or '').split('/')[-1]:<24}"
          f"{(r['rho'] if r['rho'] is not None else 'dense'):>6}  "
          f"{r['corpus']:<11}{r['ppl']:>12.4f}")
PY
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  setup)   do_setup "$@" ;;
  yaml)    do_yaml ;;
  doctor)  do_doctor ;;
  run)     [[ $# -eq 2 ]] || die "usage: ./run_local.sh run MODEL RHO"
           setup_env; run_one "$1" "$2" ;;
  sweep)   do_sweep "$@" ;;
  collect) do_collect ;;
  help|-h|--help) sed -n '2,72p' "$0" | sed 's/^# \?//' ;;
  *) die "unknown command: $cmd (try ./run_local.sh help)" ;;
esac
