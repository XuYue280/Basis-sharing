#!/usr/bin/env python3
"""Compress with Basis_Sharing, then evaluate through the shared harness.

Ratio: Basis_Sharing reads `compression_ratio` from the YAML and treats it as
the percentage REMOVED (utils.py:34 -> `1 - compression_ratio/100`), so the
generated YAMLs carry compression_ratio=40 for rho=0.60 and 67 for rho=0.33.
Both numbers are written into each YAML so the mapping stays visible.

Its own harness (test.py:compute_ppl) is a strided sliding window over a
"\\n\\n"-joined corpus and is not ARKS-comparable; perplexity here comes from
tools/shared_eval.py like every other baseline.
"""
# VENDORED COPY -- the repo root is resolved from this file's location
# instead of the hard-coded /scratch path the Trillium harness used, so
# this runs unchanged from any checkout directory.
import argparse, json, os, sys, time
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)          # <checkout>/local/..  ->  <checkout>
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from shared_eval import (evaluate_all, count_parameters, arks_fields,
                         PeakMemory, dense_cache_meta,
                         load_dense_cache, save_dense_cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dense", action="store_true")
    a = ap.parse_args()

    with open(a.yaml) as fh:
        raw = yaml.safe_load(fh)
    model_id = raw["model_args"]["model_name"]
    rho = raw["model_args"].get("target_keep_ratio")
    cr = raw["model_args"]["compression_ratio"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    t0 = time.perf_counter()
    if a.dense:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        before = after = count_parameters(model)
    else:
        from config import ShareConfig, add_args
        from model_factory import create_model
        # ShareConfig reads the YAML path off the parsed CLI namespace.
        sys.argv = ["run_basis_sharing", "--cf", a.yaml]
        cfg = ShareConfig(add_args())
        from transformers import AutoModelForCausalLM as _AM
        ref = _AM.from_pretrained(model_id, torch_dtype=torch.float16)
        # ARKS bench-record timings / peak memory (shared_eval.PeakMemory)
        _t_wall0 = time.perf_counter()
        _mem = PeakMemory(); _mem.__enter__()
        _t_load = time.perf_counter() - _t_wall0

        before = count_parameters(ref)
        del ref
        model = create_model(cfg)
        model = model.half().to("cuda")
        after = count_parameters(model)
        # ARKS: layers_evaluated / matrices_evaluated (bench record :1043-1044)
        _n_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
        _n_matrices = sum(1 for _n, _m in model.named_modules()
                          if isinstance(_m, torch.nn.Linear) and "lm_head" not in _n)
    compress_s = time.perf_counter() - t0

    _t_eval0 = time.perf_counter()

    res = evaluate_all(model, tok, device="cuda")

    _t_eval1 = time.perf_counter()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    payload = {
        "method": "basis_sharing", "model": model_id,
        "rho_target": None if a.dense else rho,
        "compression_ratio_removed_pct": None if a.dense else cr,
        "dense": a.dense, "yaml": a.yaml,
        "params_before": before, "params_after": after,
        "realised_total_ratio": after["total_params"] / before["total_params"],
        "realised_linear_ratio": after["linear_params"] / before["linear_params"],
        "compress_seconds": compress_s,
        "ppl": {c: v["ppl"] for c, v in res.items()},
        "ppl_tokens": {c: v["ppl_tokens"] for c, v in res.items()},
    }
    # --- ARKS-compatible fields (see shared_eval.arks_fields) ---
    _runs_root = os.path.dirname(a.out)
    if a.dense:
        save_dense_cache(_runs_root, res, inference_seconds=_t_eval1 - _t_eval0)
    _dense_res = load_dense_cache(_runs_root)
    payload.update(arks_fields(
        method="basis_sharing", model=model_id, rho=(None if a.dense else rho),
        res=res, params_before=before, params_after=after,
        # Basis_Sharing exposes NO seed knob -- model_factory.py:41 hard-codes
        # torch.manual_seed(2023) for the randperm that picks calibration
        # samples. Recording 42 here would be a lie (see the seed note in the
        # comparability contract).
        dense=a.dense, seed=2023, dense_res=_dense_res,
        timings={
            "inference_seconds": _t_eval1 - _t_eval0,
            "metric_seconds": _t_eval1 - _t_eval0,
            "model_load_seconds": _t_load,
            "reconstruction_seconds": compress_s if not a.dense else 0.0,
            "artifact_load_seconds": 0.0,
            "dense_inference_seconds": (_dense_res.get("_meta") or {}).get("dense_inference_seconds"),
            "total_seconds": time.perf_counter() - _t_wall0,
        },
        peak_memory=_mem.fields(),
        layers_evaluated=_n_layers, matrices_evaluated=_n_matrices))

    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\n" + json.dumps(payload["ppl"], indent=2))
    print(f"realised linear ratio = {payload['realised_linear_ratio']:.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
