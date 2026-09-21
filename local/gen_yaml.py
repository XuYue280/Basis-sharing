#!/usr/bin/env python3
"""Generate Basis_Sharing YAML configs for every (model, rho) we need.

Basis_Sharing takes its compression strength from the YAML, not the CLI, and
`compression_ratio` there is the percentage of parameters REMOVED
(utils.py:34 does `compression_ratio = 1 - compression_ratio / 100`).
Our target rho is the fraction KEPT, so rho=0.60 -> 40 and rho=0.33 -> 67.
Both numbers are written into every file so the mapping is never in doubt.
"""
import os

# VENDORED COPY -- the two paths the Trillium version hard-coded are now
# environment-driven, so this generates usable configs from any checkout.
#   BS_YAML_OUT   where the .yaml files land          (default <checkout>/local/yaml)
#   BS_RUNS_DIR   root for calib/ and the saved-model dirs referenced inside them
#                 (default <checkout>/local/runs/basis_sharing)
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("BS_YAML_OUT", os.path.join(_HERE, "yaml"))
RUNS = os.environ.get("BS_RUNS_DIR", os.path.join(_HERE, "runs", "basis_sharing"))
CALIB = "wikitext"          # unified calibration corpus for all three baselines

# model -> (model_type, weight-name map, group_size)
MODELS = {
    "facebook/opt-125m": ("opt", "opt", 2),
    "facebook/opt-350m": ("opt", "opt", 2),
    "facebook/opt-1.3b": ("opt", "opt", 2),
    "facebook/opt-2.7b": ("opt", "opt", 2),
    "meta-llama/Llama-3.2-1B": ("llama2", "llama", 2),
    "meta-llama/Llama-3.2-3B": ("llama2", "llama", 2),
    "facebook/opt-6.7b": ("opt", "opt", 2),
    "facebook/opt-13b":  ("opt", "opt", 2),
    "facebook/opt-30b":  ("opt", "opt", 2),
    "meta-llama/Llama-3.1-8B": ("llama2", "llama", 2),
    "facebook/opt-66b":  ("opt", "opt", 2),
    "meta-llama/Llama-3.1-70B": ("llama2", "llama", 2),
}
NAMES = {
    "opt":   dict(k="self_attn.k_proj", q="self_attn.q_proj", v="self_attn.v_proj",
                  o="self_attn.out_proj", up="fc1", down="fc2"),
    # Llama's MLP is gated: gate_proj is a THIRD matrix with no OPT counterpart.
    # Upstream's own tasks/configs/.../svd_llama_7b_20.yaml lists gate_name and puts
    # "gate" in share_part. Omitting it leaves 58.7M of the 218.1M per-layer target
    # params dense on Llama-3.1-8B (26.9%), so a nominal rho=0.60 would really keep
    # 0.708 -- Basis_Sharing would be plotted against ASVD/SVD-LLM/ARKS at a ratio it
    # never actually reached. It also crashes: model_factory never sets
    # num_basis_gate, so models/llama.py dereferences a None gate_basis.
    "llama": dict(k="self_attn.k_proj", q="self_attn.q_proj", v="self_attn.v_proj",
                  o="self_attn.o_proj", up="mlp.up_proj", down="mlp.down_proj",
                  gate="mlp.gate_proj"),
}
# Matches upstream: the gated MLP's gate travels with up in the shared group.
SHARE = {"opt": ["k", "q", "v", "up"], "llama": ["k", "q", "v", "up", "gate"]}
PRIVATE = {"opt": ["down", "o"], "llama": ["down", "o"]}
RHOS = [0.60, 0.33]
# calib.py:22-25 accumulates `self.calib += (inp.T @ inp).cpu()` over batches, and
# inp is flattened over (batch x seq) first -- so the Gram is a plain sum and the
# micro-batch size cannot change the result, only the peak activation memory.
# That makes this knob numerically free. It is needed because OPT has no SDPA in
# transformers 4.45 (modeling_opt declares only _supports_flash_attn_2), so the
# eager path materialises a full [bsz, heads, 2048, 2048] fp32 score tensor:
# 16*40*2048^2*4 B = 10 GiB on opt-13b, which is what OOM'd bl-942196_4.
# Left at 16 wherever the model already fits, so validated runs are untouched.
# Llama-3.1's vocab is 128256 vs OPT's 50272, and the calibration collator supplies
# labels, so the model computes a loss: logits are [bsz, 2048, 128256] fp32 and
# cross_entropy takes another copy. At bsz=16 that is 16.8 GB x2 on top of the
# 32 GB fp32 model -> OOM (bl-942288_2). Nothing to do with model size.
CALIB_BS = {"facebook/opt-13b": 4, "facebook/opt-30b": 2,
            "meta-llama/Llama-3.1-8B": 2,
            # 66B/70B: 1 is the only safe value. The logits tensor alone is
            # bsz*2048*vocab*4 B -- 0.41 GB per sample for OPT's 50272 vocab,
            # 1.05 GB for Llama-3.1's 128256 -- and cross_entropy takes another
            # copy, on top of a model that already fills most of the card.
            "facebook/opt-66b": 1, "meta-llama/Llama-3.1-70B": 1,
            # Llama-3.2 shares Llama-3.1's 128256 vocab, so the loss tensor is
            # 2.5x an OPT one at the same batch even though the models are tiny.
            "meta-llama/Llama-3.2-1B": 4, "meta-llama/Llama-3.2-3B": 4}


def emit(model_id, rho):
    mtype, nk, gs = MODELS[model_id]
    n = NAMES[nk]
    removed = round((1.0 - rho) * 100)       # rho 0.60 -> 40, rho 0.33 -> 67
    tag = f"{model_id.split('/')[-1]}_rho{int(rho*100)}"
    root = os.path.join(RUNS, tag)
    cbs = CALIB_BS.get(model_id, 16)
    gate_line = f'  gate_name: "{n["gate"]}"\n' if "gate" in n else ""
    share_lines = "".join(f'    - "{x}"\n' for x in SHARE[nk])
    private_lines = "".join(f'    - "{x}"\n' for x in PRIVATE[nk])
    return f"""# Basis_Sharing config -- generated by tools/gen_yaml.py, do not hand-edit.
#
# target_keep_ratio (rho) = {rho}   <- fraction of parameters KEPT
# compression_ratio       = {removed}  <- what Basis_Sharing wants: percent REMOVED
#   utils.py:34  compression_ratio = 1 - compression_ratio / 100
#   so {removed} means it keeps {rho:.2f} of the covered weights.
model_args:
  model_type: "{mtype}"
  model_name: "{model_id}"
  k_name: "{n['k']}"
  q_name: "{n['q']}"
  v_name: "{n['v']}"
  o_name: "{n['o']}"
  up_name: "{n['up']}"
  down_name: "{n['down']}"
{gate_line}  group_size: {gs}
  compression_ratio: {removed}
  target_keep_ratio: {rho}
  context_length: 2048
  stride: 2048
  share_part:
{share_lines}  private_part:
{private_lines}
calibration_args:
  dataset_name: "{CALIB}"
  build_calib: true
  calib_path: "{root}/calib/"
  dataset_cache_dir: null
  calibration_size: 256
  calib_batch_size: {cbs}

after_calibration_update_args:
  update_calib_path: ""
  build_update_calib: false
  update: false

model_saving:
  save_updated_model: false
  updated_model_path: "{root}/updated/"
  save_untrained_model: false
  untrained_model_path: "{root}/untrained/"
"""


os.makedirs(OUT, exist_ok=True)
n = 0
for model_id in MODELS:
    for rho in RHOS:
        tag = f"{model_id.split('/')[-1]}_rho{int(rho*100)}"
        path = os.path.join(OUT, f"{tag}.yaml")
        with open(path, "w") as fh:
            fh.write(emit(model_id, rho))
        print(f"  {path}   rho={rho}  compression_ratio={round((1-rho)*100)}")
        n += 1
print(f"\n{n} yaml file(s) written to {OUT}")
