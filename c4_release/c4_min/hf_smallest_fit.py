"""SMALLEST HF top-10k model that fits the WHOLE C4 ISA — radix vs log-sink DIV,
fp32 vs fp64 (#905).

This module is the ACCOUNTING + FIT front-door for "what is the smallest published
HF model whose (hidden, intermediate, layers) budget can host the full C4 instruction
set", enumerated across the four DIV-strategy × precision points the brief asks for::

    fp32-radix   fp64-radix   fp32-logsink   fp64-logsink

The C4-ISA geometry per point is MEASURED (block-specs only, NO model materialised —
memory-safe) from ``qwen_full_vm._block_specs`` under the two in-model DIV datapaths:

  * **radix**   (``div_radix16``, the fp32-vanilla default) — hidden 3008, inter 7920.
  * **log-sink**(``nibble_logsink_blocks``, ``C4_LOGSINK_DIV=1``) — hidden 2624,
    inter 4320.  NARROWER hidden, but DEEPER (its long-division is traded for a
    reciprocal softmax1 sink + Newton refine, which unrolls to more blocks).

HONEST precision boundary
-------------------------
The in-model log-sink divide carries quotient-scale scalars up to ``2^34`` (beyond
fp32's ``2^24``), so ``C4_LOGSINK_DIV=1`` builds the WHOLE model in **fp64**.  There
is NO in-model fp32 log-sink divide: ``nibble_logsink_fp32`` (the range-reduced fp32
approximate-then-refine reciprocal) exists ONLY as a byte-exact PYTHON REFERENCE, not
wired into ``_block_specs``.  So the ``fp32-logsink`` point is an ACCOUNTING row (the
log-sink shape at fp32 dtype) whose in-model bake does NOT exist on this branch; it is
flagged ``bakeable_in_model=False``.  The radix path is fp32-native, so
``fp{32,64}-radix`` share ONE shape (fp64 is just a wider dtype, same geometry) — the
fp64-vs-fp32 smaller-model delta is ZERO for radix at the selection granularity.

A HF model FITS the C4 ISA at a point iff its config satisfies ALL of::

    hidden_size          >= required_hidden
    intermediate_size    >= required_intermediate
    num_hidden_layers    >= required_stored_layers

``required_stored_layers`` is the FEED-FORWARD-UNROLLED depth for the standard
transformer fit, or the LOOPED / weight-tied-checkpoint (recurrent-divmod) depth for
the Universal-Transformer fit.  "Smallest" = fewest estimated total host params.  The
fit set is restricted to DENSE gated-FFN causal-decoder families (Llama/Qwen/Mistral/
Gemma/Phi/...) — the architectures the ISA actually bakes into — excluding encoders,
MoE, SSM/hybrid and vision/audio configs whose ``intermediate_size`` / param count do
not correspond to a bakeable dense FFN.

All the constants below are MEASURED (see the live reproduction in
``test_hf_smallest_fit.py``); nothing here materialises the (18–70 GB dense) muldiv
model, so the module is memory-safe.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# MEASURED C4 FULL-ISA geometry, from qwen_full_vm._block_specs (SUBSET_FULL,
# efficient_alu=True), unrolled + recurrent, under radix vs log-sink DIV.
# Reproduced live in test_hf_smallest_fit.py::test_measured_geometry_is_reproducible.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IsaGeometry:
    hidden: int
    intermediate: int
    stored_unrolled: int      # feed-forward-unrolled: distinct stored layers
    stored_recurrent: int     # looped / weight-tied checkpoint: reused cells stored
    applied: int              # layers APPLIED per forward (== unrolled applied depth)
    # ISA nnz (nonzero baked FFN weights) + dense host params, per mode.
    nnz_unrolled: int
    nnz_recurrent: int
    host_total_unrolled: int  # dense Qwen2Model params at (hidden, inter, stored_unrolled)
    host_total_recurrent: int


# radix (div_radix16, fp32-native default) — MEASURED.
GEOM_RADIX = IsaGeometry(
    hidden=3008, intermediate=7920, stored_unrolled=123, stored_recurrent=60,
    applied=123, nnz_unrolled=283632, nnz_recurrent=162959,
    host_total_unrolled=9550947456, host_total_recurrent=4659849216)

# log-sink (nibble_logsink_blocks, C4_LOGSINK_DIV=1, fp64) — MEASURED.
GEOM_LOGSINK = IsaGeometry(
    hidden=2624, intermediate=4320, stored_unrolled=221, stored_recurrent=74,
    applied=221, nnz_unrolled=176944, nnz_recurrent=72901,
    host_total_unrolled=8705807488, host_total_recurrent=2916030208)


@dataclass(frozen=True)
class FitPoint:
    name: str
    geom: IsaGeometry
    dtype: str                # "fp32" | "fp64"
    bakeable_in_model: bool   # does an IN-MODEL bake exist at this point?
    note: str


FIT_POINTS: Dict[str, FitPoint] = {
    "fp32-radix": FitPoint(
        "fp32-radix", GEOM_RADIX, "fp32", True,
        "div_radix16 fp32-native (golden default DIV path)"),
    "fp64-radix": FitPoint(
        "fp64-radix", GEOM_RADIX, "fp64", True,
        "same radix SHAPE built as fp64 dtype (geometry identical to fp32-radix)"),
    "fp32-logsink": FitPoint(
        "fp32-logsink", GEOM_LOGSINK, "fp32", False,
        "in-model log-sink forces fp64 (q*b ~2^34 > fp32 2^24); fp32 log-sink exists "
        "ONLY as the byte-exact Python reference nibble_logsink_fp32, NOT in _block_specs"),
    "fp64-logsink": FitPoint(
        "fp64-logsink", GEOM_LOGSINK, "fp64", True,
        "nibble_logsink_blocks (C4_LOGSINK_DIV=1), fp64-native"),
}

# The 4 GB peak-RSS cap: every one of the four FULL-ISA points is a muldiv config whose
# DENSE host model is 18–70 GB (radix fp32 recurrent ~18.6 GB, log-sink fp64 recurrent
# ~23.3 GB, unrolled far more).  ALL FOUR are therefore ACCOUNTED-BUT-OVER-CAP: the
# accounting is memory-safe (block-specs only) but the in-model bake is NOT ≤4 GB.  The
# byte-exact validation is done on the SMALL sub-cap subsets (base/bitwise) + the DIV
# strategy PYTHON references, per the hard memory cap.
MEMORY_CAP_GB = 4.0

# DENSE gated-FFN causal-decoder families the ISA bakes into (standard SwiGLU MLP; the
# per-layer intermediate_size is the true dense FFN width).  Excludes encoders
# (bert/roberta/deberta/electra/mpnet/modernbert), MoE (mixtral/*_moe/deepseek_v3/
# gpt_oss/afmoe), SSM/hybrid (mamba/nemotron_h/qwen3_next/lfm2), and vision/audio
# (vit/clip/siglip/wav2vec2) — architectures whose intermediate/param count do NOT map
# to a bakeable dense FFN.
DENSE_DECODER_FAMILIES = frozenset({
    "llama", "qwen2", "qwen3", "qwen3_5", "mistral", "mistral3", "gemma", "gemma2",
    "gemma3", "gemma3_text", "gemma4", "gemma4_unified", "phi3", "phi", "gpt_neox",
    "granite", "stablelm", "cohere", "cohere2", "starcoder2", "olmo", "olmo2",
    "exaone", "exaone4", "internlm2", "yi", "baichuan", "minicpm", "glm", "glm4",
    "aquila", "xverse", "deci", "persimmon", "gptj", "gpt2", "falcon", "mpt",
    "bloom", "opt", "codegen", "dbrx",
})

C4_VOCAB = 276    # the C4 VM's own vocab (host models use their OWN vocab for fit)

_HERE = os.path.dirname(os.path.abspath(__file__))
HF_CONFIG_CACHE = os.path.join(_HERE, "hf_fit_data", "hf_top10k_configs.json")


# ---------------------------------------------------------------------------
# HF-config fit logic (config-only; no weights).
# ---------------------------------------------------------------------------
def _si(x) -> Optional[int]:
    """Coerce a config field to a positive int scalar, else None (guards list-valued
    per-layer fields + non-numeric junk)."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int) and x > 0:
        return x
    if isinstance(x, float) and x > 0:
        return int(x)
    return None


def estimate_total_params(cfg: dict) -> Optional[int]:
    """Dense Llama/Qwen-style total-param estimate from a config: embed[+head if
    untied] + L*(attn q/k/v/o + gate/up/down MLP + 2 norms).  Unknown vocab assumes a
    large modern vocab (151936) so the estimate is not wildly low."""
    H = _si(cfg.get("hidden_size")); L = _si(cfg.get("num_hidden_layers"))
    I = _si(cfg.get("intermediate_size")); V = _si(cfg.get("vocab_size"))
    if not (H and L and I):
        return None
    nkv = _si(cfg.get("num_key_value_heads")); nh = _si(cfg.get("num_attention_heads"))
    hd = _si(cfg.get("head_dim"))
    tie = cfg.get("tie_word_embeddings", True)
    qd = nh * hd if (nh and hd) else H
    if nkv and nh and hd:
        kvd = nkv * hd
    elif nkv and nh:
        kvd = int((nkv / nh) * H)
    else:
        kvd = H
    attn = H * qd + 2 * H * kvd + qd * H
    mlp = 3 * H * I
    per_layer = attn + mlp + 2 * H
    embed = (H * V * (1 if tie else 2)) if V else H * 151936
    return int(per_layer * L + embed)


def load_hf_configs(path: str = HF_CONFIG_CACHE) -> Dict[str, dict]:
    """Load the cached top-10k config snapshot (id -> config fields)."""
    with open(path) as f:
        return json.load(f)


@dataclass
class HostFit:
    id: str
    hidden: int
    intermediate: int
    layers: int
    model_type: str
    total_params: int


def _usable_dense(configs: Dict[str, dict]) -> List[HostFit]:
    out: List[HostFit] = []
    for mid, cfg in configs.items():
        if not isinstance(cfg, dict) or "_error" in cfg:
            continue
        if cfg.get("model_type") not in DENSE_DECODER_FAMILIES:
            continue
        H = _si(cfg.get("hidden_size")); L = _si(cfg.get("num_hidden_layers"))
        I = _si(cfg.get("intermediate_size"))
        if not (H and L and I):
            continue
        tp = estimate_total_params(cfg)
        if tp is None:
            continue
        out.append(HostFit(mid, H, I, L, cfg["model_type"], tp))
    return out


def smallest_fit(point: FitPoint, looped: bool,
                 configs: Optional[Dict[str, dict]] = None
                 ) -> Tuple[Optional[HostFit], int]:
    """Smallest DENSE-decoder host that fits ``point`` (looped=weight-tied depth,
    else feed-forward-unrolled depth).  Returns (winner-or-None, n_fitting)."""
    if configs is None:
        configs = load_hf_configs()
    g = point.geom
    req_stored = g.stored_recurrent if looped else g.stored_unrolled
    cands = [h for h in _usable_dense(configs)
             if h.hidden >= g.hidden and h.intermediate >= g.intermediate
             and h.layers >= req_stored]
    cands.sort(key=lambda h: h.total_params)
    return (cands[0] if cands else None), len(cands)
