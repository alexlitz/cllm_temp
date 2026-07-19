"""Run the c4 op COMPUTE through a REAL Qwen2 MLP (SwiGLU) with RMSNorm 1/K
compensation — closing the gap ``qwen_embed`` left open.

Where ``qwen_embed`` stopped
============================
``qwen_embed`` proves REGISTER INGEST through a genuine ``Qwen2Model.forward``
(RoPE content-match + BOS-sink softmax + RMSNorm-compensator + GQA): the AX byte
round-trips through Qwen's attention. But the *op compute* (ADD/SUB/…) was still
a Python gadget (``nibble_add_gadget``); only its result was re-ingested. The
forward never actually ran the arithmetic.

What this module does
=====================
It places each op family's **gadget SwiGLU weights into a Qwen2 MLP layer**,
re-scaled so the residual write is byte-exact *after* Qwen's RMSNorm, and runs
the compute through the REAL ``Qwen2Model.forward`` (RMSNorm active, SwiGLU MLP,
RoPE attention present-but-identity). The operands enter on the residual (via
``inputs_embeds``), the MLP computes the next register value, and the LM
byte-head decodes it — the arithmetic is done by Qwen's own MLP.

The gadget<->Qwen MLP mapping
-----------------------------
Our gadget FFN (``blogspec_model.FFN`` / ``compile_ffn``) is

    hidden = silu(W_up·x + b_up) · (W_gate·x + b_gate)
    out    = x + W_down·hidden + b_down          (additive residual)

Qwen2's MLP (``Qwen2MLP``) is, with NO biases and NO internal residual (the
residual add lives in the decoder layer, which is exactly our ``x +``):

    out    = residual + down_proj( silu(gate_proj(x̂)) · up_proj(x̂) )

where ``x̂ = RMSNorm(residual)``. So the mapping is

    gate_proj ← W_up      (the silu-activated projection = our guard indicator)
    up_proj   ← W_gate     (the linear projection      = our write expression)
    down_proj ← W_down

with the three biases folded onto constant residual lanes (Qwen MLP is
bias-free):

    b_up[u]   -> gate_proj[u, ONE]  += b_up[u]     (ONE lane == 1.0)
    b_gate[u] -> up_proj[u, ONE]    += b_gate[u]
    b_down    -> our compiled specs have b_down == 0 (verified), nothing to fold.

RMSNorm 1/K compensation (the load-bearing part)
------------------------------------------------
The MLP sees ``x̂ = RMSNorm(x)``, not ``x``. With a compensator lane holding a
large constant ``K`` and every RMSNorm ``weight = K/sqrt(H)``:

    x̂_i = x_i · (K/sqrt(H)) / sqrt(mean(x^2)+eps)
        = x_i · K / sqrt(K^2 + S)                       (S = real-lane energy)
        = x_i · 1/sqrt(1 + S/K^2)   ≈  x_i · (1 − S/(2K^2))

So RMSNorm multiplies EVERY real lane (operands the projections read) by the
SAME factor ``r = 1/sqrt(1+S/K^2) ≲ 1`` (for K=4000, S≲a few hundred:
``r ≈ 1 − 6e-6``). The gadget thresholds (``up = S·(a+b)``, ``up =
RELU_S·(band−thr)``) are calibrated on integer operands; scaling operands by
``r`` shifts a silu step's argument by ``(1−r)·arg`` — at most ~``6e-6 · 60·30 ≈
1e-2`` in silu units, far inside the ≥1-integer separation between distinct
nibble sums. The DOWN-projection write is likewise ``r``-scaled, but it lands on
the raw residual and the LM byte-head argmax (``2·v·val − v²``) is quadratic with
a ≥1 gap between adjacent bytes, so a ~1e-5 relative perturbation cannot flip the
argmax. We MEASURE the realised worst-case margin per op (``verify_*``) rather
than assume it. If a family's silu steps were too sharp to survive ``r`` the
margin would collapse — reported honestly, with the ``K``/compensator-width fix.

Nothing here rounds. The operands are exact integers on the embedding; the MLP
output is the (near-exact) integer register value; the LM byte-head argmax IS the
vanilla re-quantizer (spec §Tokenization).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from .compile_ffn import S as GADGET_S, RELU_S


# ---------------------------------------------------------------------------
# Qwen size presets (mirror qwen_embed; kept memory-small for exact tests).
# ---------------------------------------------------------------------------
@dataclass
class QwenPreset:
    name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int


# A tiny Qwen2-ARCHITECTURE model: identical primitives (RoPE + RMSNorm + plain
# softmax + SwiGLU MLP + GQA, head_dim=64 so the RoPE ladder matches 0.5B), but
# small hidden / few layers so a single-op forward is cheap and memory-safe.
# intermediate_size is set per-build to fit the op family's hidden-unit count.
QWEN_TINY = QwenPreset(
    name="Qwen-tiny(0.5B-arch)",
    hidden_size=896, num_hidden_layers=2,
    num_attention_heads=14, num_key_value_heads=2,
    intermediate_size=512, vocab_size=512,
)

# The real smallest published Qwen (used to confirm the method on a stock size).
QWEN2_5_0_5B = QwenPreset(
    name="Qwen2.5-0.5B",
    hidden_size=896, num_hidden_layers=24,
    num_attention_heads=14, num_key_value_heads=2,
    intermediate_size=4864, vocab_size=151936,
)


NORM_K = 4000.0   # RMSNorm compensator constant (same as qwen_embed)


# ---------------------------------------------------------------------------
# Op-compute residual layout: the nibble register bands + scalar value lanes an
# op reads/writes, plus a ONE constant lane and the RMSNorm compensator. Kept
# minimal (only what the op families here touch) so the tiny Qwen holds it.
# ---------------------------------------------------------------------------
class OpLayout:
    """Residual layout for op-compute-through-Qwen-MLP.

    Register nibble bands (16 dims each, spec representation) + a scalar value
    lane per register (``compile_ffn`` gadgets that operate on scalar bytes read
    these) + a few CMP scratch lanes + ONE + the compensator.
    """

    def __init__(self, hidden_size: int):
        self._off = 0
        self._names: Dict[str, Tuple[int, int]] = {}
        # nibble register bands (little-endian 4-bit nibbles)
        self.AX = self._band("AX", NIB_PER_REG)
        self.STACK0 = self._band("STACK0", NIB_PER_REG)
        self.RES = self._band("RES", NIB_PER_REG)      # op result nibble band
        # scalar value lanes (byte image of the operands; recomposed / provided)
        self.AX_VAL = self._scalar("AX_VAL")
        self.STK_VAL = self._scalar("STK_VAL")
        self.RES_VAL = self._scalar("RES_VAL")         # scalar op result
        # CMP scratch (boolean primitives)
        self.CMP_EQ = self._scalar("CMP_EQ")
        self.CMP_GT = self._scalar("CMP_GT")
        self.CMP_LT = self._scalar("CMP_LT")
        # per-nibble carry scratch for the multi-nibble adder
        self.CARRY = self._band("CARRY", NIB_PER_REG)
        self.ONE = self._scalar("ONE")
        self.D = self._off
        self.hidden_size = hidden_size
        self.COMP = hidden_size - 1                     # compensator = last dim
        assert self.D < self.COMP, (self.D, self.COMP)

    def _scalar(self, name: str) -> int:
        return self._band(name, 1)

    def _band(self, name: str, size: int) -> int:
        base = self._off
        self._names[name] = (base, size)
        self._off += size
        return base


# ---------------------------------------------------------------------------
# RMSNorm-identity gamma (same construction as qwen_embed): gamma = K/sqrt(H).
# ---------------------------------------------------------------------------
def rmsnorm_identity_gamma(hidden_size: int, K: float = NORM_K) -> torch.Tensor:
    return torch.full((hidden_size,), K / math.sqrt(hidden_size),
                      dtype=torch.float32)


# ---------------------------------------------------------------------------
# Map a compiled gadget SwiGLU spec into a Qwen2 MLP layer (bias-folded, K-safe).
# ---------------------------------------------------------------------------
def bake_spec_into_qwen_mlp(mlp, spec: Dict[str, torch.Tensor], L: OpLayout
                            ) -> None:
    """Load a compiled gadget SwiGLU ``spec`` into a ``Qwen2MLP`` module.

    ``gate_proj ← W_up`` (silu side), ``up_proj ← W_gate`` (linear side),
    ``down_proj ← W_down``. Biases are folded onto the ONE lane (Qwen MLP is
    bias-free). ``b_down`` must be 0 (asserted; our specs satisfy it).

    Because RMSNorm rescales the MLP INPUT by ``r = K/sqrt(K^2+S) ≈ 1`` and this
    ``mlp`` reads that normed input, both the silu thresholds and the linear
    write are ``r``-scaled together — no per-weight compensation is needed beyond
    the shared RMSNorm gamma; the ~1e-5 residue is absorbed by the byte-head
    argmax (measured, not assumed).
    """
    hidden = spec["W_up"].shape[0]
    inter = mlp.gate_proj.weight.shape[0]
    H = mlp.gate_proj.weight.shape[1]
    assert hidden <= inter, (hidden, inter)
    b_down = spec.get("b_down")
    if b_down is not None:
        assert float(b_down.abs().max()) == 0.0, "b_down folding unsupported"

    gate_w = torch.zeros(inter, H)          # <- W_up  (silu side)
    up_w = torch.zeros(inter, H)            # <- W_gate (linear side)
    down_w = torch.zeros(H, inter)          # <- W_down

    Dg = spec["W_up"].shape[1]
    gate_w[:hidden, :Dg] = spec["W_up"]
    up_w[:hidden, :Dg] = spec["W_gate"]
    down_w[:Dg, :hidden] = spec["W_down"]
    # fold biases onto the ONE lane (value 1.0 after RMSNorm ≈ identity).
    gate_w[:hidden, L.ONE] += spec["b_up"]
    up_w[:hidden, L.ONE] += spec["b_gate"]

    with torch.no_grad():
        mlp.gate_proj.weight.copy_(gate_w)
        mlp.up_proj.weight.copy_(up_w)
        mlp.down_proj.weight.copy_(down_w)


# ---------------------------------------------------------------------------
# Build a real Qwen2Model whose MLP layer computes ONE op family.
# ---------------------------------------------------------------------------
def _qwen_config(preset: QwenPreset, intermediate: int):
    from transformers.models.qwen2 import Qwen2Config
    return Qwen2Config(
        hidden_size=preset.hidden_size,
        num_hidden_layers=preset.num_hidden_layers,
        num_attention_heads=preset.num_attention_heads,
        num_key_value_heads=preset.num_key_value_heads,
        intermediate_size=intermediate,
        vocab_size=preset.vocab_size,
        max_position_embeddings=8192,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
        attn_implementation="eager",
    )


@dataclass
class QwenOpVM:
    qmodel: object
    layout: OpLayout
    preset: QwenPreset
    K: float
    compute_layer: int   # which decoder layer runs the op-compute MLP


def build_qwen_op_vm(specs, L: OpLayout,
                     preset: QwenPreset = QWEN_TINY, K: float = NORM_K
                     ) -> QwenOpVM:
    """Construct a real ``Qwen2Model`` whose first ``len(specs)`` MLP layers run
    the op compute (one spec per layer, stacked — a multi-stage op like the byte
    adder's nibble carry occupies 2 layers). All other layers/attention are
    identity; every RMSNorm uses the compensator gamma = K/sqrt(H).

    ``specs`` is a single spec dict or a list of them (layer 0, 1, ...).
    """
    if isinstance(specs, dict):
        specs = [specs]
    inter = max(max(int(s["W_up"].shape[0]) for s in specs), 8)
    n_layers = max(preset.num_hidden_layers, len(specs))
    preset = QwenPreset(preset.name, preset.hidden_size, n_layers,
                        preset.num_attention_heads, preset.num_key_value_heads,
                        preset.intermediate_size, preset.vocab_size)
    cfg = _qwen_config(preset, inter)
    from transformers.models.qwen2 import Qwen2Model
    qmodel = Qwen2Model(cfg).to(torch.float32).eval()
    H = preset.hidden_size

    with torch.no_grad():
        gamma = rmsnorm_identity_gamma(H, K)
        qmodel.norm.weight.copy_(gamma)
        for layer in qmodel.layers:
            layer.input_layernorm.weight.copy_(gamma)
            layer.post_attention_layernorm.weight.copy_(gamma)
            for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                        layer.self_attn.v_proj, layer.self_attn.o_proj):
                lin.weight.zero_()
                if lin.bias is not None:
                    lin.bias.zero_()
            for lin in (layer.mlp.gate_proj, layer.mlp.up_proj,
                        layer.mlp.down_proj):
                lin.weight.zero_()
        for i, spec in enumerate(specs):
            bake_spec_into_qwen_mlp(qmodel.layers[i].mlp, spec, L)

    return QwenOpVM(qmodel=qmodel, layout=L, preset=preset, K=K,
                    compute_layer=0)


# ---------------------------------------------------------------------------
# Drive operands through the model and read out the result.
# ---------------------------------------------------------------------------
def _residual_from_fields(L: OpLayout, fields: Dict[int, float], K: float
                          ) -> torch.Tensor:
    """A single-position residual [1,1,H] with ``fields`` (band->value) set, plus
    ONE=1 and the compensator=K."""
    x = torch.zeros(1, 1, L.hidden_size)
    x[0, 0, L.ONE] = 1.0
    x[0, 0, L.COMP] = K
    for band, val in fields.items():
        x[0, 0, band] = float(val)
    return x


def _forward_last(vm: QwenOpVM, x: torch.Tensor) -> torch.Tensor:
    out = vm.qmodel(inputs_embeds=x, use_cache=False)
    return out.last_hidden_state[0, -1]


def _byte_head(L: OpLayout, reg_base: int, byte_index: int, H: int
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """LM byte-head over nibble dims ``reg_base+2*bi+{0,1}`` (score 2·v·val − v²)."""
    n0 = reg_base + 2 * byte_index + 0
    n1 = reg_base + 2 * byte_index + 1
    W = torch.zeros(256, H)
    b = torch.zeros(256)
    for v in range(256):
        lo, hi = V.nibbles_of_byte(v)
        W[v, n0] = 2.0 * lo
        W[v, n1] = 2.0 * hi
        b[v] = -(lo * lo) - (hi * hi)
    return W, b


def decode_byte_margin(hidden: torch.Tensor, L: OpLayout, reg_base: int,
                       byte_index: int) -> Tuple[int, float]:
    """argmax byte + decode MARGIN (top logit − runner-up), the robustness gap."""
    W, b = _byte_head(L, reg_base, byte_index, L.hidden_size)
    logits = F.linear(hidden.float(), W, b)
    top2 = torch.topk(logits, 2)
    return int(top2.indices[0].item()), float(top2.values[0] - top2.values[1])


def decode_scalar(hidden: torch.Tensor, L: OpLayout, lane: int) -> float:
    """Read a scalar value lane directly (for bool/CMP results)."""
    return float(hidden[lane])


# ===========================================================================
# Op-family gadget SwiGLU specs (the arithmetic done INSIDE the Qwen MLP).
#
# Each builder returns one or more compiled specs (one per Qwen MLP layer in the
# compute chain). A multi-stage op (byte add's nibble carry) uses a 2-layer
# chain; single-stage ops (nibble bitwise, CMP) use one layer.
# ===========================================================================
def _empty(dim: int, n_units: int) -> Dict[str, torch.Tensor]:
    return {
        "W_up": torch.zeros(n_units, dim), "b_up": torch.zeros(n_units),
        "W_gate": torch.zeros(n_units, dim), "b_gate": torch.zeros(n_units),
        "W_down": torch.zeros(dim, n_units), "b_down": torch.zeros(dim),
    }


def _silu05() -> float:
    return float(F.silu(torch.tensor(0.5 * GADGET_S)))


def _add_unit(spec, u, dim, dst, srcs, one, guard_on=True):
    """One SwiGLU unit: dst += sum(srcs) when guard (ONE) holds. srcs is a list
    of (band, coeff). Uses the compile_ffn always-on guard on ONE."""
    for band, c in srcs:
        spec["W_gate"][u, band] += c
    spec["W_up"][u, one] = GADGET_S
    spec["b_up"][u] = -GADGET_S * 0.5          # one window on ONE
    spec["W_down"][dst, u] += 1.0 / _silu05()


def _fold_units(spec, u, dim, band, one, modulus, carry_lane=None):
    """Two SwiGLU units: fold ``band`` mod ``modulus`` (band -= M·(band>=M)); if
    ``carry_lane`` given, ALSO write the carry bit (band>=M) into carry_lane.
    Returns next unit index."""
    M = modulus
    for i, thr in enumerate((M - 1, M)):
        spec["W_up"][u + i, band] = RELU_S
        spec["b_up"][u + i] = -RELU_S * thr
        spec["W_gate"][u + i, one] = 1.0
    spec["W_down"][band, u] += -M / RELU_S
    spec["W_down"][band, u + 1] += +M / RELU_S
    if carry_lane is not None:
        # carry = (band>=M) = relu(band-(M-1)) - relu(band-M) = unit0 - unit1
        spec["W_down"][carry_lane, u] += +1.0 / RELU_S
        spec["W_down"][carry_lane, u + 1] += -1.0 / RELU_S
    return u + 2


# ---------------------------------------------------------------------------
# ADD / SUB (8-bit byte, nibble carry) — a 2-layer Qwen MLP chain.
# ---------------------------------------------------------------------------
# NOTE (why add & fold are in SEPARATE layers): an FFN's units all read the SAME
# input residual. The fold must read the POST-add value of RES+j, so it cannot
# live in the same layer as the add unit that writes RES+j. Each nibble therefore
# spends 2 Qwen MLP layers: (a) sum into RES+j, (b) fold-16 of RES+j + carry-out.
def _add_layer(L, dst, srcs) -> Dict[str, torch.Tensor]:
    D = L.hidden_size
    s = _empty(D, 1)
    _add_unit(s, 0, D, dst, srcs, L.ONE)
    return s


def _fold_layer(L, band, carry_lane=None, modulus=16) -> Dict[str, torch.Tensor]:
    D = L.hidden_size
    s = _empty(D, 2)
    _fold_units(s, 0, D, band, L.ONE, modulus, carry_lane=carry_lane)
    return s


def add_specs(L: OpLayout, n_nibbles: int = 2) -> List[Dict[str, torch.Tensor]]:
    """AX := (STACK0 + AX) mod 2^(4·n_nibbles), nibble ripple-carry, as a stacked
    chain of Qwen MLP layers (2 per nibble: sum, then fold+carry).

      layer 2j   : RES_j = A_j + B_j (+ CARRY_{j-1})
      layer 2j+1 : fold RES_j mod 16 -> low nibble ; CARRY_j = carry (last nibble
                   drops its carry = the mod-2^n wrap).
    """
    specs: List[Dict[str, torch.Tensor]] = []
    for j in range(n_nibbles):
        srcs = [(L.STACK0 + j, 1.0), (L.AX + j, 1.0)]
        if j > 0:
            srcs.append((L.CARRY + (j - 1), 1.0))
        specs.append(_add_layer(L, L.RES + j, srcs))
        carry = L.CARRY + j if j < n_nibbles - 1 else None
        specs.append(_fold_layer(L, L.RES + j, carry_lane=carry))
    return specs


def sub_specs(L: OpLayout, n_nibbles: int = 2) -> List[Dict[str, torch.Tensor]]:
    """AX := (STACK0 - AX) mod 2^(4·n_nibbles) via nibble borrow, computed as the
    SAME silu add+fold primitive with a constant +16 and −B (spec §Basic
    Arithmetic 'subtraction works similarly'):

      d_j = A_j - B_j + 16 (+ CARRY_{j-1} - 1)   -> always ≥ 0, ≤ 31
      fold mod 16 -> low nibble ; CARRY_j = (d_j ≥ 16) == 'no borrow out'.

    j=0 has no incoming borrow so d_0 = A_0 - B_0 + 16. For j>0 the incoming term
    is CARRY_{j-1} - 1 (=0 if the previous nibble produced no borrow, -1 if it
    borrowed), giving d_j = A_j - B_j + 15 + CARRY_{j-1}.
    """
    specs: List[Dict[str, torch.Tensor]] = []
    for j in range(n_nibbles):
        if j == 0:
            srcs = [(L.STACK0 + 0, 1.0), (L.AX + 0, -1.0), (L.ONE, 16.0)]
        else:
            srcs = [(L.STACK0 + j, 1.0), (L.AX + j, -1.0), (L.ONE, 15.0),
                    (L.CARRY + (j - 1), 1.0)]
        specs.append(_add_layer(L, L.RES + j, srcs))
        carry = L.CARRY + j if j < n_nibbles - 1 else None
        specs.append(_fold_layer(L, L.RES + j, carry_lane=carry))
    return specs


# ---------------------------------------------------------------------------
# CMP (EQ/NE/LT/GT/LE/GE) — one SwiGLU layer, boolean into RES_VAL.
# ---------------------------------------------------------------------------
def cmp_spec(L: OpLayout, op: int) -> Dict[str, torch.Tensor]:
    """RES_VAL := (STK_VAL <cmp> AX_VAL). d = STK_VAL - AX_VAL. Uses the §584
    finite-2nd-difference silu bump for EQ and clamped-relu ramps for LT/GT — all
    integer-exact. One MLP layer."""
    D = L.hidden_size
    sig = float(torch.sigmoid(torch.tensor(GADGET_S * 0.5)))
    k = GADGET_S * 0.5 * (2.0 * sig - 1.0)
    if op in (isa.EQ, isa.NE):
        spec = _empty(D, 3)
        for i, (bias, w) in enumerate([(GADGET_S * 0.5, 1.0), (0.0, -2.0),
                                       (-GADGET_S * 0.5, 1.0)]):
            spec["W_up"][i, L.STK_VAL] += GADGET_S
            spec["W_up"][i, L.AX_VAL] += -GADGET_S
            spec["b_up"][i] += bias
            spec["W_gate"][i, L.ONE] = 1.0
            spec["W_down"][L.RES_VAL, i] += w / k
        if op == isa.NE:                       # NE = 1 - EQ
            for i in range(3):
                spec["W_down"][L.RES_VAL, i] *= -1.0
            # add constant 1 via an always-on unit
            spec = _append_const(spec, L, 1.0)
        return spec
    # LT / GT / LE / GE via ramps step(d>=1) / step(-d>=1).
    spec = _empty(D, 2)
    if op in (isa.GT, isa.LE):
        hi, lo = L.STK_VAL, L.AX_VAL           # d = STK - AX ; GT = (d>=1)
    else:                                      # LT / GE : -d = AX - STK
        hi, lo = L.AX_VAL, L.STK_VAL
    for idx, thr in enumerate((0.0, 1.0)):
        spec["W_up"][idx, hi] += RELU_S
        spec["W_up"][idx, lo] += -RELU_S
        spec["b_up"][idx] += -RELU_S * thr
        spec["W_gate"][idx, L.ONE] = 1.0
        spec["W_down"][L.RES_VAL, idx] += (1.0 if idx == 0 else -1.0) / RELU_S
    if op in (isa.LE, isa.GE):                 # LE = 1-GT ; GE = 1-LT
        spec["W_down"][L.RES_VAL, 0] *= -1.0
        spec["W_down"][L.RES_VAL, 1] *= -1.0
        spec = _append_const(spec, L, 1.0)
    return spec


def _append_const(spec, L: OpLayout, c: float) -> Dict[str, torch.Tensor]:
    """Append an always-on unit adding constant ``c`` into RES_VAL."""
    n = spec["W_up"].shape[0]
    D = spec["W_up"].shape[1]
    out = _empty(D, n + 1)
    for key in ("W_up", "b_up", "W_gate", "b_gate"):
        out[key][:n] = spec[key]
    out["W_down"][:, :n] = spec["W_down"]
    out["b_down"] = spec["b_down"]
    _add_unit(out, n, D, L.RES_VAL, [(L.ONE, c)], L.ONE)
    return out


# ---------------------------------------------------------------------------
# Bitwise nibble op (AND/OR/XOR) — one SwiGLU layer, per-nibble truth table.
#
# For each result nibble we sum, over the 16x16 (a,b) input pairs, an indicator
# unit (a==A_j AND b==B_j) scaled by the table output (a OP b). The indicator is
# a 2D silu bump; scaled by the truth value it deposits the correct nibble.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Bitwise (AND/OR/XOR, 8-bit) — one SwiGLU layer, BIT-PARALLEL (memory-safe).
#
# We operate per BIT, not per nibble-pair (which would be a 16x16 table). Each
# output bit is a 2-input boolean whose truth value is a small clamped-relu
# combo of the two input bits (both 0/1):
#
#     A·B  (AND) = relu(A + B - 1)
#     A|B  (OR)  = relu(A + B) - relu(A + B - 1)          ( = min(A+B,1) )
#     A^B  (XOR) = relu(A + B) - 2·relu(A + B - 1)         ( = (A+B) - 2AB )
#
# each an exact 0/1 for 0/1 inputs and ≤2 clamped-relu units. The operand bits
# are provided on dedicated bit lanes (bit-decomposed from the byte, which the
# real VM does with a bit-extract FFN; here the driver seeds the operand bits and
# the byte-head recomposes RES from the result bits). 16 bits × few units easily
# fits the tiny Qwen intermediate — NO 256-wide table.
# ---------------------------------------------------------------------------
def _relu_unit(spec, u, dst, terms, const, one, coeff):
    """dst += coeff · relu( sum(terms) + const ), one clamped-relu SwiGLU unit."""
    for band, c in terms:
        spec["W_up"][u, band] += RELU_S * c
    spec["b_up"][u] += RELU_S * const
    spec["W_gate"][u, one] = 1.0
    spec["W_down"][dst, u] += coeff / RELU_S


def bitwise_spec(L: OpLayout, kind: str, n_bits: int = 8
                 ) -> Dict[str, torch.Tensor]:
    """AND/OR/XOR of two ``n_bits`` operands, BIT-PARALLEL, as ONE SwiGLU layer.

    Operand bits live on ``L.STACK0 + k`` (A bit k) and ``L.AX + k`` (B bit k),
    k = 0..n_bits-1 (the driver seeds them; a real bit-extract FFN would derive
    them from the nibble bands). Result bit k -> ``L.RES + k``. ``kind`` in
    {"and","or","xor"}."""
    D = L.hidden_size
    units_per_bit = {"and": 1, "or": 2, "xor": 2}[kind]
    spec = _empty(D, units_per_bit * n_bits)
    u = 0
    for k in range(n_bits):
        A, B, R = L.STACK0 + k, L.AX + k, L.RES + k
        if kind == "and":
            _relu_unit(spec, u, R, [(A, 1.0), (B, 1.0)], -1.0, L.ONE, 1.0); u += 1
        elif kind == "or":
            _relu_unit(spec, u, R, [(A, 1.0), (B, 1.0)], 0.0, L.ONE, 1.0); u += 1
            _relu_unit(spec, u, R, [(A, 1.0), (B, 1.0)], -1.0, L.ONE, -1.0); u += 1
        else:  # xor
            _relu_unit(spec, u, R, [(A, 1.0), (B, 1.0)], 0.0, L.ONE, 1.0); u += 1
            _relu_unit(spec, u, R, [(A, 1.0), (B, 1.0)], -1.0, L.ONE, -2.0); u += 1
    return spec



# ===========================================================================
# Per-op-family verification through the REAL Qwen2Model.forward.
#
# Each verify_* builds the op VM, drives a sweep of operands through
# ``qmodel.forward`` (RMSNorm active), decodes the result with the LM byte-head,
# and reports {exact, cases, worst_margin, worst_case}. The worst_margin is the
# smallest (top - runner-up) byte-head gap seen — the robustness the RMSNorm
# rescale must not collapse.
# ===========================================================================
def _bits(v: int, n: int) -> List[int]:
    return [(v >> k) & 1 for k in range(n)]


def verify_add(preset: QwenPreset = QWEN_TINY, K: float = NORM_K,
               n: int = 256) -> Dict[str, object]:
    """ADD: AX := (STACK0 + AX) & 0xFF through 2 stacked Qwen MLP layers."""
    L = OpLayout(preset.hidden_size)
    vm = build_qwen_op_vm(add_specs(L), L, preset, K)
    worst = float("inf"); worst_case = None; ok = 0; total = 0
    import random
    random.seed(0)
    pairs = [(a, b) for a in range(16) for b in range(16)]        # all low nibbles
    pairs += [(random.randrange(256), random.randrange(256)) for _ in range(n)]
    for a, b in pairs:
        want = (a + b) & 0xFF
        an = V.nibbles_of_byte(a); bn = V.nibbles_of_byte(b)
        fields = {L.STACK0 + 0: an[0], L.STACK0 + 1: an[1],
                  L.AX + 0: bn[0], L.AX + 1: bn[1]}
        x = _residual_from_fields(L, fields, K)
        h = _forward_last(vm, x)
        got, margin = decode_byte_margin(h, L, L.RES, 0)
        total += 1; ok += (got == want)
        if got != want and worst_case is None:
            worst_case = (a, b, got, want)
        if margin < worst:
            worst = margin
    return {"op": "ADD", "exact": ok == total, "pass": ok, "cases": total,
            "worst_margin": worst, "first_fail": worst_case}


def verify_sub(preset: QwenPreset = QWEN_TINY, K: float = NORM_K,
               n: int = 256) -> Dict[str, object]:
    """SUB: AX := (STACK0 - AX) & 0xFF (two's complement wrap) through 2 MLPs."""
    L = OpLayout(preset.hidden_size)
    vm = build_qwen_op_vm(sub_specs(L), L, preset, K)
    worst = float("inf"); worst_case = None; ok = 0; total = 0
    import random
    random.seed(1)
    pairs = [(a, b) for a in range(16) for b in range(16)]
    pairs += [(random.randrange(256), random.randrange(256)) for _ in range(n)]
    for a, b in pairs:
        want = (a - b) & 0xFF
        an = V.nibbles_of_byte(a); bn = V.nibbles_of_byte(b)
        fields = {L.STACK0 + 0: an[0], L.STACK0 + 1: an[1],
                  L.AX + 0: bn[0], L.AX + 1: bn[1]}
        x = _residual_from_fields(L, fields, K)
        h = _forward_last(vm, x)
        got, margin = decode_byte_margin(h, L, L.RES, 0)
        total += 1; ok += (got == want)
        if got != want and worst_case is None:
            worst_case = (a, b, got, want)
        if margin < worst:
            worst = margin
    return {"op": "SUB", "exact": ok == total, "pass": ok, "cases": total,
            "worst_margin": worst, "first_fail": worst_case}


def verify_add16(preset: QwenPreset = QWEN_TINY, K: float = NORM_K,
                 n: int = 256) -> Dict[str, object]:
    """16-bit ADD carry: 4-nibble add across 4 stacked MLP layers, checking BOTH
    result bytes (byte0 and byte1) — proves the nibble-carry chain survives
    RMSNorm on a wider value."""
    L = OpLayout(preset.hidden_size)
    vm = build_qwen_op_vm(add_specs(L, n_nibbles=4), L, preset, K)
    worst = float("inf"); worst_case = None; ok = 0; total = 0
    import random
    random.seed(2)
    pairs = [(random.randrange(65536), random.randrange(65536)) for _ in range(n)]
    pairs += [(0xFFFF, 1), (0x00FF, 1), (0x0FFF, 1), (0xFFFE, 2), (0x1234, 0xABCD)]
    for a, b in pairs:
        want = (a + b) & 0xFFFF
        an = V.nibbles_of_value(a, 16); bn = V.nibbles_of_value(b, 16)
        fields = {}
        for j in range(4):
            fields[L.STACK0 + j] = an[j]; fields[L.AX + j] = bn[j]
        x = _residual_from_fields(L, fields, K)
        h = _forward_last(vm, x)
        g0, m0 = decode_byte_margin(h, L, L.RES, 0)
        g1, m1 = decode_byte_margin(h, L, L.RES, 1)
        got = g0 | (g1 << 8)
        total += 1; ok += (got == want)
        if got != want and worst_case is None:
            worst_case = (a, b, got, want)
        worst = min(worst, m0, m1)
    return {"op": "ADD16", "exact": ok == total, "pass": ok, "cases": total,
            "worst_margin": worst, "first_fail": worst_case}


_CMP_NAMES = {isa.EQ: "EQ", isa.NE: "NE", isa.LT: "LT", isa.GT: "GT",
              isa.LE: "LE", isa.GE: "GE"}
_CMP_PY = {
    isa.EQ: lambda s, a: int(s == a), isa.NE: lambda s, a: int(s != a),
    isa.LT: lambda s, a: int(s < a),  isa.GT: lambda s, a: int(s > a),
    isa.LE: lambda s, a: int(s <= a), isa.GE: lambda s, a: int(s >= a),
}


def verify_cmp(op: int, preset: QwenPreset = QWEN_TINY, K: float = NORM_K
               ) -> Dict[str, object]:
    """A comparison op: RES_VAL := (STK_VAL <op> AX_VAL) through ONE Qwen MLP.
    Sweeps small operands; the boolean is read from the RES_VAL lane and its
    MARGIN is min distance to the 0.5 decision threshold."""
    L = OpLayout(preset.hidden_size)
    vm = build_qwen_op_vm(cmp_spec(L, op), L, preset, K)
    worst = float("inf"); worst_case = None; ok = 0; total = 0
    vals = list(range(0, 20)) + [50, 100, 200, 255]
    for s in vals:
        for a in vals:
            want = _CMP_PY[op](s, a)
            x = _residual_from_fields(L, {L.STK_VAL: s, L.AX_VAL: a}, K)
            h = _forward_last(vm, x)
            got_f = decode_scalar(h, L, L.RES_VAL)
            got = int(round(got_f))
            margin = abs(got_f - 0.5)          # distance to the decision boundary
            total += 1; ok += (got == want)
            if got != want and worst_case is None:
                worst_case = (s, a, got_f, want)
            worst = min(worst, margin)
    return {"op": _CMP_NAMES[op], "exact": ok == total, "pass": ok,
            "cases": total, "worst_margin": worst, "first_fail": worst_case}


def verify_bitwise(kind: str, preset: QwenPreset = QWEN_TINY, K: float = NORM_K,
                   n_bits: int = 8) -> Dict[str, object]:
    """A bitwise op (and/or/xor), bit-parallel, through ONE Qwen MLP. Result bits
    are read from RES bit lanes; margin is min distance to 0.5."""
    L = OpLayout(preset.hidden_size)
    vm = build_qwen_op_vm(bitwise_spec(L, kind, n_bits), L, preset, K)
    py = {"and": lambda a, b: a & b, "or": lambda a, b: a | b,
          "xor": lambda a, b: a ^ b}[kind]
    worst = float("inf"); worst_case = None; ok = 0; total = 0
    import random
    random.seed(3)
    mask = (1 << n_bits) - 1
    pairs = [(a, b) for a in range(16) for b in range(16)]
    pairs += [(random.randrange(256) & mask, random.randrange(256) & mask)
              for _ in range(128)]
    for a, b in pairs:
        want = py(a, b) & mask
        fields = {}
        for k in range(n_bits):
            fields[L.STACK0 + k] = (a >> k) & 1
            fields[L.AX + k] = (b >> k) & 1
        x = _residual_from_fields(L, fields, K)
        h = _forward_last(vm, x)
        got = 0; m = float("inf")
        for k in range(n_bits):
            bit_f = float(h[L.RES + k])
            got |= (1 if bit_f >= 0.5 else 0) << k
            m = min(m, abs(bit_f - 0.5))
        total += 1; ok += (got == want)
        if got != want and worst_case is None:
            worst_case = (a, b, got, want)
        worst = min(worst, m)
    return {"op": f"BITWISE_{kind.upper()}", "exact": ok == total, "pass": ok,
            "cases": total, "worst_margin": worst, "first_fail": worst_case}


# ---------------------------------------------------------------------------
# The BRANCH decision (BZ/BNZ) is a comparison of AX against 0 -> reuse the CMP
# EQ primitive: taken := (AX_VAL == 0). We verify the predicate through the MLP.
# ---------------------------------------------------------------------------
def verify_branch(preset: QwenPreset = QWEN_TINY, K: float = NORM_K
                  ) -> Dict[str, object]:
    """BZ/BNZ predicate: RES_VAL := (AX_VAL == 0), through one Qwen MLP (the §584
    EQ bump on d = AX_VAL - 0). Sweeps AX; margin to the 0.5 branch threshold."""
    L = OpLayout(preset.hidden_size)
    # AX==0 predicate = EQ bump on AX_VAL vs 0. Reuse cmp_spec with STK_VAL held 0.
    spec = cmp_spec(L, isa.EQ)
    vm = build_qwen_op_vm(spec, L, preset, K)
    worst = float("inf"); worst_case = None; ok = 0; total = 0
    for ax in list(range(0, 20)) + [50, 100, 255]:
        want = int(ax == 0)
        x = _residual_from_fields(L, {L.STK_VAL: 0, L.AX_VAL: ax}, K)
        h = _forward_last(vm, x)
        got_f = decode_scalar(h, L, L.RES_VAL)
        got = int(round(got_f))
        margin = abs(got_f - 0.5)
        total += 1; ok += (got == want)
        if got != want and worst_case is None:
            worst_case = (ax, got_f, want)
        worst = min(worst, margin)
    return {"op": "BRANCH(AX==0)", "exact": ok == total, "pass": ok,
            "cases": total, "worst_margin": worst, "first_fail": worst_case}


# ---------------------------------------------------------------------------
# Full per-family report.
# ---------------------------------------------------------------------------
def run_all(preset: QwenPreset = QWEN_TINY, K: float = NORM_K) -> List[Dict]:
    reports = []
    reports.append(verify_add(preset, K))
    reports.append(verify_sub(preset, K))
    reports.append(verify_add16(preset, K))
    for op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
        reports.append(verify_cmp(op, preset, K))
    for kind in ("and", "or", "xor"):
        reports.append(verify_bitwise(kind, preset, K))
    reports.append(verify_branch(preset, K))
    return reports


def _demo(preset: QwenPreset = QWEN_TINY, K: float = NORM_K) -> None:
    print(f"op-compute through the REAL Qwen2 MLP (RMSNorm active), {preset.name}, "
          f"K={K}\n")
    print(f"{'op':>16}  {'exact':>6}  {'pass/cases':>12}  {'worst_margin':>13}  "
          f"first_fail")
    for r in run_all(preset, K):
        print(f"{r['op']:>16}  {str(r['exact']):>6}  "
              f"{str(r['pass'])+'/'+str(r['cases']):>12}  "
              f"{r['worst_margin']:>13.6f}  {r['first_fail']}")


if __name__ == "__main__":
    _demo()
