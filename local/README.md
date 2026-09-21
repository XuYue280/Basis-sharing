# `local/` — single-GPU runner support files

Everything here exists so `../run_local.sh` can run this baseline on an ordinary
GPU box (e.g. Lambda Labs) with no Slurm and no cluster-specific paths.

| file | origin | change |
|---|---|---|
| `run_basis_sharing.py` | ARKS comparison harness, `baselines/tools/run_basis_sharing.py` | repo root resolved from `__file__` instead of a hard-coded `/scratch` path |
| `shared_eval.py` | ARKS comparison harness, `baselines/tools/shared_eval.py` | `LOCAL_FILES` made environment-driven (`SHARED_EVAL_PTB_FILE`, `SHARED_EVAL_C4_FILE`) instead of hard-coded cluster paths |
| `gen_yaml.py` | ARKS comparison harness, `baselines/tools/gen_yaml.py` | output dir and run root read from `BS_YAML_OUT` / `BS_RUNS_DIR` instead of hard-coded `/scratch` paths |

`gen_yaml.py` emits byte-identical configs to the cluster version apart from
those two paths. Its output lands in `local/yaml/` and is **not** committed — it
embeds absolute paths, so it is machine-specific. Regenerate with
`./run_local.sh yaml`.

## Single-GPU / layerwise / backward — the short answer

Basis_Sharing runs on **one GPU** — every reference result up to opt-13b and
Llama-3.1-8B was produced on a single H100 — but it does so by keeping the whole
model resident, **not** by streaming layers.

* **Backward pass: none on this path.** Two real ones exist, `train.py` (full
  fine-tuning) and `lora.py` (peft), but neither is reachable from this runner,
  and the generated YAMLs set `after_calibration_update_args.update: false`,
  which also disables the closed-form coefficient refit.
* **Calibration is whole-model.** `BS_CALIB_GROUPS` slices which layers have a
  hook attached and then runs the entire 256-sample forward through the entire
  model once per group. It bounds the *host* RAM held by the Gram matrices. It
  is not a layerwise fix — do not read it as one.
* **Whitening** (Cholesky + inverse of each Gram) is one group at a time on the
  CPU in float64 — a CPU bottleneck. It frees nothing on the GPU: the dense
  reference model stays resident throughout.
* **Decomposition** is per-group arithmetic but reads weights off a
  whole-model-resident reference, so residency is still whole-model. During
  `match_state_dict` two models briefly coexist.
* **Evaluation is whole-model.** `SHARED_EVAL_LAYERWISE=1` is only *partially*
  effective here, and for **both** model families: the layerwise path detaches
  `layers` only, so every shared-basis `ModuleDict` stays pinned on the device
  (42.9% of compressed linear params on opt-6.7b/13b, 36.6% on Llama-3.1-8B).
  Treat layerwise evaluation as unsupported for this baseline.

So this is the baseline with the largest single-GPU memory requirement of the
three. See the header of `../run_local.sh` for detail.

## Licence note

Upstream `TUDa-HWAI/Basis_Sharing` ships **no LICENSE file**. This fork adds
none on its behalf; the original authors' terms, whatever they are, still apply.
