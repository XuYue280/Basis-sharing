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

from shared_eval import evaluate_all, count_parameters


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
        before = count_parameters(ref)
        del ref
        model = create_model(cfg)
        model = model.half().to("cuda")
        after = count_parameters(model)
    compress_s = time.perf_counter() - t0

    res = evaluate_all(model, tok, device="cuda")
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
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\n" + json.dumps(payload["ppl"], indent=2))
    print(f"realised linear ratio = {payload['realised_linear_ratio']:.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
