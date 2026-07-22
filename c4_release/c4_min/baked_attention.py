"""Full (general) attention baked into SwiGLU/softmax1 — the *completeness* demo.

BLOG_SPEC §"Baking Weights" (line 828-836): the attention path the VM actually
uses is KV-*retrieval* — query == key, weights ~ {0,1}, no real mixing — and
that already works in c4_min (see ``compile_attn.py`` / ``blogspec_memory.py``).
This module is the harder, expensive *general* case the blog shows "for
completeness": emulate a real softmax1 attention layer's output using only
SwiGLU FFN nodes and the efficient-exp attention sink, i.e. bake the mixing into
weights instead of relying on runtime Q/K/V attention over data tokens.

Construction (all blog-faithful, vanilla SwiGLU + softmax1, no exotic ops):

  1. Dot products   Q.K_i  via the 6-weight SiLU-gated *multiply* (§Basic
     Arithmetic, line 593):  (silu(S a)+silu(-S a))*b/S ~= a*b, summed over the
     binary key/query dims.  Because the vectors are binary the per-bit nodes
     are shared across the emulated tokens (only the down-projection bit
     pattern differs, line 834).
  2. exp(score)     via the *efficient exponential* (§Efficient Exp, line 561):
     a BOS-sink token with key sqrt(d), value e^B, ALiBi slope 0; setting the
     score to (N - B) and reading softmax1 * value yields
     exp(N-B)*e^B ~= exp(N).  We evaluate that sink expression directly here
     (it is one attention head in the real model).
  3. Denominator    D = 1 + sum_i exp(score_i)   (the +1 is softmax1's sink).
  4. Reciprocal     1/D via *real-number base-16 long division* (§Long Division,
     line 646):  each quotient nibble q = sum_{k=1..15} step(remainder -
     k*divisor).  This is the expensive part — the blog flags it inserts a
     stack of division layers *per emulated attention layer* (line 836).
  5. Output         out = sum_i (exp(score_i) * (1/D)) * V_i     (again via the
     SiLU-gated multiply), which equals softmax1(scores) . V.

Cost (honest, per the blog): the reciprocal is the dominant term.  For P
precision nibbles it is P base-16 long-division iterations, each a threshold
staircase (15 SiLU step units) plus a subtract — O(P * 15) SwiGLU units and,
laid out as real network layers, ~P sequential layers that must all complete
before the scaled-down values are usable.  Every emulated attention layer pays
this.  The KV-retrieval path (the used one) pays *none* of it.

Accuracy: the multiply, exp-sink and long-division steps are each exact-to-fp
for the small integer-scored case we validate; the reported tolerance below is
the achieved max abs error vs. torch's reference softmax1 attention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Scales (mirroring c4_min.compile_ffn / nibble_muldivmod conventions).
# ---------------------------------------------------------------------------
S = 60.0        # SiLU-identity / gated-multiply scale: silu(S)~=S, silu(-S)~=0
RELU_S = 200.0  # relu-via-silu scale for the long-division staircase (fp-ok)
_DTYPE = torch.float64  # doubles: the division staircase needs > fp32 unit step


# ===========================================================================
# SwiGLU-node primitives (the shared gadgets, evaluated as silu(up)*gate/S)
# ===========================================================================
def _relu(z, s: float = RELU_S):
    """Clamped ReLU ``silu(s*z)/s`` — exact ``max(0, z)`` for integer ``z``."""
    z = torch.as_tensor(z, dtype=_DTYPE)
    return F.silu(s * z) / s


def _step_ge(x, t, s: float = RELU_S):
    """Exact ``[x >= t]`` via ``relu(x-(t-1)) - relu(x-t)`` (two SwiGLU units)."""
    x = torch.as_tensor(x, dtype=_DTYPE)
    t = torch.as_tensor(t, dtype=_DTYPE)
    return _relu(x - (t - 1), s) - _relu(x - t, s)


def _mul(a, b, s: float = S):
    """6-weight SiLU-*gated* multiply ``(silu(S a)+silu(-S a))*b/S ~= a*b``
    (§Basic Arithmetic, line 593).  Exact for a bounded non-negative ``a``.

    This is *the* blog multiplication primitive; the two gate nodes give
    ``|a|`` and the ``* b`` is the SwiGLU gate, ``/S`` the down projection.
    """
    a = torch.as_tensor(a, dtype=_DTYPE)
    b = torch.as_tensor(b, dtype=_DTYPE)
    return (F.silu(s * a) + F.silu(-s * a)) * b / s


# ===========================================================================
# (1) Dot product via shared per-bit SiLU multiplies (binary vectors)
# ===========================================================================
def _dot_binary(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """``sum_d q_d * k_d`` via the SiLU-gated multiply on each dim.

    ``q`` and ``k`` are 1-D.  Per the blog (line 834) the vectors are binary so
    a single shared per-bit multiply node handles every dim; we evaluate the
    sum of the shared node's outputs here.
    """
    q = torch.as_tensor(q, dtype=_DTYPE)
    k = torch.as_tensor(k, dtype=_DTYPE)
    return _mul(q, k).sum()


# ===========================================================================
# (2) Efficient exp via the BOS-sink softmax1 head (§Efficient Exp, line 561)
# ===========================================================================
def _efficient_exp(n: torch.Tensor, bias: float) -> torch.Tensor:
    """``exp(N)`` via the sink: softmax1 over a single sink whose score is
    ``(N - B)`` and value ``e^B``.

        softmax1(N-B) * e^B = e^(N-B)/(1 + e^(N-B)) * e^B ~= e^(N-B) * e^B = e^N

    We keep the sink denominator's ``1`` out of the way by choosing ``B`` so the
    per-score exp stays small vs. 1 (line 563); the tiny ``1/(1+e^(N-B))``
    factor is the sink's approximation error, reported in the validation.
    """
    n = torch.as_tensor(n, dtype=_DTYPE)
    x = n - bias
    # softmax1 with a single competitor: exp(x)/(1+exp(x)) then * e^B.
    return torch.sigmoid(x) * torch.exp(torch.as_tensor(bias, dtype=_DTYPE))


# ===========================================================================
# (4) Real-number base-16 long division for the reciprocal 1/D (§Long Division)
# ===========================================================================
def _reciprocal_long_division(divisor: torch.Tensor, precision_nibbles: int = 8):
    """``1 / divisor`` via base-16 long division (§Long Division, line 646).

    Computes the fixed-point expansion of ``1 / d``: dividend = 1, and each of
    ``precision_nibbles`` fractional nibbles (MSB of the fraction first) is

        q = sum_{k=1..15} step(remainder - k * divisor)      (threshold count)

    scaling the remainder by 16 each step.  Returns (reciprocal, n_layers) where
    ``n_layers`` is the sequential long-division layer count this consumes.

    This is the *expensive* op the blog warns about (line 836): one such stack
    per emulated attention layer.
    """
    d = torch.as_tensor(divisor, dtype=_DTYPE)
    # long division of 1.0 by d, base 16, MSB-of-fraction first.
    remainder = torch.ones((), dtype=_DTYPE)
    quotient = torch.zeros((), dtype=_DTYPE)
    thresholds = torch.arange(1, 16, dtype=_DTYPE)  # k = 1..15
    n_layers = 0
    for j in range(1, precision_nibbles + 1):
        remainder = remainder * 16.0
        # digit = count of k in 1..15 with remainder >= k*d  (SiLU staircase)
        digit = _step_ge(remainder, thresholds * d).sum()
        quotient = quotient + digit * (16.0 ** (-j))
        remainder = remainder - digit * d
        n_layers += 1  # each nibble is one sequential long-division layer
    return quotient, n_layers


# ===========================================================================
# Baked full attention head (the completeness demo)
# ===========================================================================
@dataclass
class BakedKV:
    key: torch.Tensor
    value: torch.Tensor


class BakedFullAttention:
    """Full softmax1 attention emulated with SwiGLU + efficient-exp + long div.

    ``output = sum_i softmax1(Q.K_i / sqrt(d)) * V_i``   computed via the five
    blog gadgets above.  This is the general (mixing) case, NOT KV-retrieval.
    """

    def __init__(self, dim: int, kvs: List[BakedKV],
                 precision_nibbles: int = 8, exp_bias: float | None = None,
                 exp_bias_margin: float = 4.0):
        self.dim = dim
        self.kvs = kvs
        self.num_kvs = len(kvs)
        self.precision_nibbles = precision_nibbles
        # exp_bias B: per §Efficient Exp (line 563) B is chosen so exp(N-B) is
        # SMALL vs. 1, making the sink factor 1/(1+exp(N-B)) ~= 1.  If None we
        # pick it from the max achievable score + a margin.  A fixed value is
        # also accepted (the head's baked-in sink key).
        self.exp_bias = exp_bias
        self.exp_bias_margin = exp_bias_margin
        self.scale = dim ** -0.5
        # bookkeeping for the honest cost report
        self.last_cost: dict = {}

    def _select_bias(self, scores: torch.Tensor) -> float:
        if self.exp_bias is not None:
            return float(self.exp_bias)
        # keep exp(N-B) small: B = max score + margin  ->  N-B <= -margin
        return float(scores.max()) + self.exp_bias_margin

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        q = torch.as_tensor(query, dtype=_DTYPE)

        # (1) dot products Q.K_i via SiLU-gated multiply, scaled by 1/sqrt(d)
        scores = torch.stack([
            _dot_binary(q, kv.key) * self.scale for kv in self.kvs
        ])

        # (2) efficient exp of each score via the BOS-sink head.  Bias B chosen
        #     so the sink factor is negligible (§Efficient Exp).
        bias = self._select_bias(scores)
        exps = torch.stack([
            _efficient_exp(s, bias) for s in scores
        ])

        # (3) softmax1 denominator: 1 (sink) + sum_i exp(score_i)
        denom = 1.0 + exps.sum()

        # (4) reciprocal 1/denom via real-number long division
        recip, div_layers = self._reciprocal(denom)

        # (5) weighted value sum: out = sum_i (exp_i * recip) * V_i
        weights = torch.stack([_mul(e, recip) for e in exps])  # SiLU multiply
        out = torch.zeros(self.dim, dtype=_DTYPE)
        for i, kv in enumerate(self.kvs):
            v = torch.as_tensor(kv.value, dtype=_DTYPE)
            out = out + torch.stack([_mul(weights[i], vd) for vd in v])

        self.last_cost = self._cost(div_layers)
        return out.to(torch.float32)

    def _reciprocal(self, denom):
        return _reciprocal_long_division(denom, self.precision_nibbles)

    # -- cost accounting (honest, per emulated attention layer) --------------
    def _cost(self, div_layers: int) -> dict:
        n, d, P = self.num_kvs, self.dim, self.precision_nibbles
        mul_units = 2                       # per SiLU-gated multiply (2 gate nodes)
        dot_units = n * d * mul_units       # dot products
        exp_heads = n                       # one efficient-exp sink read per score
        div_units = P * 15 * 2              # 15-step staircase, 2 relu units each
        weight_units = n * mul_units        # exp * recip
        value_units = n * d * mul_units     # weight * value
        total_units = dot_units + div_units + weight_units + value_units
        return {
            "num_kvs": n, "dim": d, "precision_nibbles": P,
            "swiglu_units_dot": dot_units,
            "efficient_exp_attention_heads": exp_heads,
            "swiglu_units_longdiv": div_units,
            "long_division_layers_sequential": div_layers,
            "swiglu_units_weight": weight_units,
            "swiglu_units_value": value_units,
            "swiglu_units_total": total_units,
            "note": ("long division is the dominant cost and is SEQUENTIAL: "
                     f"{div_layers} layers per emulated attention layer must "
                     "complete before values can be scaled down (BLOG line 836)"),
        }


# ===========================================================================
# Reference: real softmax1 attention (what we must reproduce)
# ===========================================================================
def reference_softmax1_attention(query: torch.Tensor,
                                 kvs: List[BakedKV],
                                 dim: int) -> torch.Tensor:
    """The ground-truth softmax1 attention output, in plain torch."""
    q = torch.as_tensor(query, dtype=_DTYPE)
    keys = torch.stack([torch.as_tensor(kv.key, dtype=_DTYPE) for kv in kvs])
    values = torch.stack([torch.as_tensor(kv.value, dtype=_DTYPE) for kv in kvs])
    scores = (keys @ q) * (dim ** -0.5)
    exps = torch.exp(scores)
    weights = exps / (1.0 + exps.sum())          # softmax1
    return (weights.unsqueeze(-1) * values).sum(0).to(torch.float32)


# ===========================================================================
# Validation / demo
# ===========================================================================
def validate(verbose: bool = True) -> dict:
    """Small-case validation: baked SwiGLU attention vs. torch softmax1.

    Returns a dict with the achieved max-abs error and the cost report.
    """
    torch.manual_seed(0)
    dim = 4
    # binary keys/values (the blog's shared-per-bit case) + a couple of KVs.
    kvs = [
        BakedKV(key=torch.tensor([1., 0., 1., 0.]),
                value=torch.tensor([1., 0., 0., 0.])),
        BakedKV(key=torch.tensor([0., 1., 1., 0.]),
                value=torch.tensor([0., 1., 0., 0.])),
        BakedKV(key=torch.tensor([1., 1., 0., 1.]),
                value=torch.tensor([0., 0., 1., 1.])),
    ]

    # a spread of binary queries so several attention distributions are exercised
    queries = [
        torch.tensor([1., 0., 1., 0.]),  # matches KV0 best
        torch.tensor([0., 1., 1., 0.]),  # matches KV1 best
        torch.tensor([1., 1., 1., 1.]),  # mixes all three
        torch.tensor([0., 0., 0., 0.]),  # uniform-ish (all scores 0 -> sink)
    ]

    # precision_nibbles=12 + a bias margin of 12 drives the (already near-exact)
    # long division and the sink-exp approximation both below ~1e-6.
    baker = BakedFullAttention(dim, kvs, precision_nibbles=12, exp_bias_margin=12.0)

    max_err = 0.0
    rows = []
    for qi, q in enumerate(queries):
        baked = baker.forward(q)
        ref = reference_softmax1_attention(q, kvs, dim)
        err = float((baked.double() - ref.double()).abs().max())
        max_err = max(max_err, err)
        rows.append((qi, q.tolist(), ref.tolist(), baked.tolist(), err))

    cost = baker.last_cost

    if verbose:
        print("=" * 72)
        print("FULL-ATTENTION BAKING via SwiGLU — completeness demo (c4_min)")
        print("=" * 72)
        print(f"dim={dim}  num_kvs={len(kvs)}  precision_nibbles=12  "
              f"exp_bias=auto(max_score+12)")
        print("-" * 72)
        for qi, q, ref, baked, err in rows:
            print(f"query[{qi}] = {q}")
            print(f"   reference softmax1 . V = {[f'{x:.6f}' for x in ref]}")
            print(f"   baked SwiGLU attention = {[f'{x:.6f}' for x in baked]}")
            print(f"   max|err| = {err:.3e}")
        print("-" * 72)
        print(f"MAX ABS ERROR over all queries: {max_err:.3e}")
        print("-" * 72)
        print("COST (per emulated attention layer):")
        for k, v in cost.items():
            print(f"   {k}: {v}")
        print("=" * 72)
        print("NOTE: KV-retrieval baking (query==key, weights~{0,1}) — the path")
        print("the VM actually uses — already works in c4_min (compile_attn.py /")
        print("blogspec_memory.py) and pays NONE of the long-division cost above.")
        print("This module is the general-attention completeness demo only.")
        print("=" * 72)

    return {"max_abs_error": max_err, "cost": cost, "rows": rows}


if __name__ == "__main__":
    validate(verbose=True)
