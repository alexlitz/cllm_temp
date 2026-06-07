"""Phase R4 prototype: SwiGLU repack + bias fold validation.

Validates the math claim in
``c4_release/docs/QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md`` Phase R4:

  Ours: y = down(silu(W_up x + b_up) * (W_gate x + b_gate)) + b_down
  Qwen: y = down(silu(W_gate' x) * W_up' x)             # no biases

Two transformations are needed.

1. Repack (commutativity of multiplication):
       silu(a) * b == b * silu(a)
   so the gated path is symmetric in (a, b).  Therefore our W_up takes
   the role of Qwen's gate_proj and our W_gate takes Qwen's up_proj.

2. Bias fold via a residual-stream "bias_compensator" dim that is
   always 1 in the input x:
       W x + b == (W with column[bias_compensator] += b) x
   because the bias_compensator column gets multiplied by 1, so its
   accumulation in the output equals b.

   For b_down (an output-side bias) the same trick is applied on the
   *post-FFN* side: we add the bias into the bias_compensator slot of
   the W_down output, by setting
       W_down'[bias_compensator, :] = 0  (zero contribution from hidden)
   and *adding* b_down[bias_compensator] to a downstream layer.
   For this self-contained prototype we verify the FFN identity given
   that the post-FFN bias_compensator value is allowed to be folded
   into the residual stream of the next layer; concretely we compare
   the full output and report the per-dim agreement separately for the
   bias_compensator slot vs the other dims.

Run:
    python -m c4_release.tools.qwen_swiglu_repack_prototype
or
    python c4_release/tools/qwen_swiglu_repack_prototype.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


@dataclass
class SwiGLUDims:
    d_in: int
    d_hidden: int
    bias_compensator: int  # index of the always-1 residual dim


class OurSwiGLU(torch.nn.Module):
    """Our FFN: matches PureFFN in neural_vm/base_layers.py.

    Shapes follow that module:
      W_up:    (hidden, d_in)
      b_up:    (hidden,)
      W_gate:  (hidden, d_in)
      b_gate:  (hidden,)
      W_down:  (d_in, hidden)
      b_down:  (d_in,)

    Forward: ``y = down(silu(W_up x + b_up) * (W_gate x + b_gate)) + b_down``
    NOTE: this prototype only computes the FFN sub-module (no residual add),
    so we can isolate the repack math from the residual stream.
    """

    def __init__(self, dims: SwiGLUDims):
        super().__init__()
        self.dims = dims
        # Random weights and biases — magnitudes deliberately small so
        # silu stays in the smooth regime and float32 round-off doesn't
        # mask a real algebraic mistake.
        g = torch.Generator().manual_seed(0xC4_5101)
        self.W_up = torch.nn.Parameter(
            torch.randn(dims.d_hidden, dims.d_in, generator=g) * 0.05
        )
        self.b_up = torch.nn.Parameter(torch.randn(dims.d_hidden, generator=g) * 0.10)
        self.W_gate = torch.nn.Parameter(
            torch.randn(dims.d_hidden, dims.d_in, generator=g) * 0.05
        )
        self.b_gate = torch.nn.Parameter(
            torch.randn(dims.d_hidden, generator=g) * 0.10
        )
        self.W_down = torch.nn.Parameter(
            torch.randn(dims.d_in, dims.d_hidden, generator=g) * 0.05
        )
        self.b_down = torch.nn.Parameter(torch.randn(dims.d_in, generator=g) * 0.10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = F.linear(x, self.W_up) + self.b_up
        gate = F.linear(x, self.W_gate) + self.b_gate
        hidden = F.silu(up) * gate
        return F.linear(hidden, self.W_down, self.b_down)


class QwenSwiGLU(torch.nn.Module):
    """Qwen-style SwiGLU: ``y = down(silu(gate'(x)) * up'(x))`` — no biases."""

    def __init__(self, dims: SwiGLUDims, gate_proj, up_proj, down_proj):
        super().__init__()
        self.dims = dims
        self.gate_proj = torch.nn.Parameter(gate_proj.clone())
        self.up_proj = torch.nn.Parameter(up_proj.clone())
        self.down_proj = torch.nn.Parameter(down_proj.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, self.gate_proj)
        up = F.linear(x, self.up_proj)
        return F.linear(F.silu(gate) * up, self.down_proj)


# ---------------------------------------------------------------------------
# Repack
# ---------------------------------------------------------------------------


def repack_to_qwen(ours: OurSwiGLU) -> QwenSwiGLU:
    """Apply the R4 repack + bias-fold transformation.

    Transformation summary, with ``c = dims.bias_compensator``:

    *Repack*: gate'(x) = W_up x + b_up,   up'(x) = W_gate x + b_gate
       because silu(a)*b == b*silu(a) (the product is commutative).

    *Bias fold*: assume x[..., c] == 1 always.  Then
       (W x + b)[i] = sum_k W[i,k] x[k] + b[i]
                    = sum_k W'[i,k] x[k]
       where W'[i, k] = W[i, k] for k != c, and W'[i, c] = W[i, c] + b[i].

    For b_down (output bias) we cannot fold into the FFN's own weights
    because there is no "always 1" hidden unit.  Instead we route it
    through the residual stream: the next layer's input will have
    x'[c] = 1 again iff the previous block writes 1 into c.  This
    prototype implements the in-FFN repack and then verifies the FFN
    output equality.  See the doc for the cross-block treatment.
    """
    c = ours.dims.bias_compensator
    d_in = ours.dims.d_in
    d_hidden = ours.dims.d_hidden

    # --- gate' (== our up + b_up folded in) ---
    gate_proj = ours.W_up.detach().clone()
    gate_proj[:, c] = gate_proj[:, c] + ours.b_up.detach()

    # --- up' (== our gate + b_gate folded in) ---
    up_proj = ours.W_gate.detach().clone()
    up_proj[:, c] = up_proj[:, c] + ours.b_gate.detach()

    # --- down (b_down rolled into the column of down that feeds the
    # bias_compensator dim of the *output*).  Concretely we shift the
    # output: y[c] becomes y[c] + b_down[c], etc.  To keep this
    # self-contained we'd need an "always 1" hidden unit, which doesn't
    # exist.  So we leave b_down off and report the bias_compensator
    # slot agreement under the assumption that the next layer's
    # CONST=1 re-injection absorbs it.  See validate_bias_compensator_fold
    # for the inter-block treatment.
    down_proj = ours.W_down.detach().clone()

    return QwenSwiGLU(ours.dims, gate_proj, up_proj, down_proj)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def make_input(dims: SwiGLUDims, batch: int, scale: float, seed: int) -> torch.Tensor:
    """Random input with the bias_compensator slot pinned to 1.

    This is exactly the invariant the plan relies on (BLOG_SPEC's
    CONST=1 residual dim, see ``docs/QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md``).
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, dims.d_in, generator=g) * scale
    x[..., dims.bias_compensator] = 1.0
    return x


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a - b).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff / (a.abs() + 1e-12)).max().item(),
        "byte_identical": bool(torch.equal(a, b)),
        "allclose_1e-5": bool(torch.allclose(a, b, atol=1e-5, rtol=1e-5)),
        "allclose_1e-6": bool(torch.allclose(a, b, atol=1e-6, rtol=1e-6)),
    }


def validate_ffn_identity(dims: SwiGLUDims) -> None:
    """End-to-end: our FFN minus its b_down ≟ repacked Qwen FFN."""
    ours = OurSwiGLU(dims)
    qwen = repack_to_qwen(ours)

    cases = [
        ("random_small", make_input(dims, batch=4, scale=0.5, seed=1)),
        ("random_large", make_input(dims, batch=4, scale=5.0, seed=2)),
        ("zero_input", _zero_with_bias(dims)),
        ("very_large", make_input(dims, batch=2, scale=50.0, seed=3)),
        # Edge case: ALL ones (extreme x means silu saturated to ~x):
        ("ones_input", _ones_with_bias(dims)),
    ]

    print(f"=== FFN identity:  our_y - b_down ≟ qwen_y ===")
    print(f"d_in={dims.d_in}  d_hidden={dims.d_hidden}  "
          f"bias_compensator={dims.bias_compensator}")
    print()
    for name, x in cases:
        with torch.no_grad():
            y_ours = ours(x)
            # b_down is the only piece NOT folded into qwen_layer's weights;
            # subtract it to match.
            y_ours_no_bd = y_ours - ours.b_down
            y_qwen = qwen(x)
        stats = diff_stats(y_ours_no_bd, y_qwen)
        print(f"[{name:>14}]  max_abs={stats['max_abs']:.3e}  "
              f"byte_identical={stats['byte_identical']}  "
              f"allclose_1e-6={stats['allclose_1e-6']}")

    print()


def validate_bias_compensator_fold(dims: SwiGLUDims) -> None:
    """Cross-block treatment of b_down.

    b_down is added to every output dim.  In a residual network the
    output is summed into the next layer's input x_{n+1}.  If the next
    block re-asserts x_{n+1}[c] = 1 (the embedding bake or a
    constant-injection op does this), then b_down values on dims != c
    pass through, but b_down[c] is clobbered.

    Compensation: pre-bake b_down[!=c] into the next block's W_up /
    W_gate via the same trick (add b_down to those weights' c column,
    scaled by the residual contribution).

    For this prototype we just confirm the algebraic identity that
    matters in isolation: subtracting b_down lines up the outputs
    byte-identically.
    """
    ours = OurSwiGLU(dims)
    qwen = repack_to_qwen(ours)

    x = make_input(dims, batch=8, scale=1.0, seed=42)
    with torch.no_grad():
        y_ours = ours(x)
        y_qwen = qwen(x)
        # Mathematically: y_ours[i, d] - b_down[d] == y_qwen[i, d]
        gap = y_ours - y_qwen
    # gap should equal b_down broadcast over batch.
    b_down_broadcast = ours.b_down.unsqueeze(0).expand_as(gap)
    print("=== b_down fold: y_ours - y_qwen ≟ b_down (broadcast) ===")
    stats = diff_stats(gap, b_down_broadcast)
    print(f"  max_abs={stats['max_abs']:.3e}  "
          f"byte_identical={stats['byte_identical']}  "
          f"allclose_1e-6={stats['allclose_1e-6']}")
    print()


def validate_commutativity_microcheck() -> None:
    """Sanity: silu(a) * b == b * silu(a) element-wise.

    Triviality check. If this ever fails we have a torch bug.
    """
    g = torch.Generator().manual_seed(99)
    a = torch.randn(1024, generator=g) * 10
    b = torch.randn(1024, generator=g) * 10
    lhs = F.silu(a) * b
    rhs = b * F.silu(a)
    print("=== Commutativity microcheck: silu(a)*b == b*silu(a) ===")
    print(f"  byte_identical={torch.equal(lhs, rhs)}")
    print()


def validate_byte_identity_matched_order(dims: SwiGLUDims) -> None:
    """Match summation order: compare *qwen forward* against a matched
    "ours-folded" forward that also fuses bias into the weight column.

    The point: F.linear(x, W) + b and F.linear(x, W_with_bias_folded)
    do not summation-order-match.  But Qwen's forward IS the
    folded-version forward.  So to talk about byte-identity, we should
    compare against an *implementation* of ours that runs the same
    matmul Qwen runs.  That implementation is the export target — the
    repack itself is byte-identical to running Qwen on the exported
    weights, by construction.
    """
    ours = OurSwiGLU(dims)
    qwen = repack_to_qwen(ours)

    # "ours-folded" forward: build the same up-projection that Qwen uses,
    # then run silu * gate * down to mimic Qwen's exact op order.  This
    # IS the export — but we run it through our own code path to prove
    # the equivalence is tautological.
    c = dims.bias_compensator
    W_up_folded = ours.W_up.detach().clone()
    W_up_folded[:, c] += ours.b_up.detach()
    W_gate_folded = ours.W_gate.detach().clone()
    W_gate_folded[:, c] += ours.b_gate.detach()

    x = make_input(dims, batch=4, scale=1.0, seed=7)
    with torch.no_grad():
        up = F.linear(x, W_up_folded)
        gate = F.linear(x, W_gate_folded)
        ours_folded_out = F.linear(F.silu(up) * gate, ours.W_down.detach())
        qwen_out = qwen(x)
    stats = diff_stats(ours_folded_out, qwen_out)
    print(f"  ours-folded vs qwen:  byte_identical={stats['byte_identical']}  "
          f"max_abs={stats['max_abs']:.3e}")
    print()


def _zero_with_bias(dims: SwiGLUDims) -> torch.Tensor:
    x = torch.zeros(2, dims.d_in)
    x[..., dims.bias_compensator] = 1.0
    return x


def _ones_with_bias(dims: SwiGLUDims) -> torch.Tensor:
    x = torch.ones(2, dims.d_in)
    # bias_compensator already 1
    return x


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(0)
    # Use d_in=784 (BLOG_SPEC d_model) and d_hidden=4096 to mimic our real shape.
    dims = SwiGLUDims(d_in=784, d_hidden=4096, bias_compensator=0)

    validate_commutativity_microcheck()

    print("########## fp32 ##########")
    validate_ffn_identity(dims)
    validate_bias_compensator_fold(dims)

    # Robustness: a couple of alternative bias_compensator positions to
    # make sure we did not accidentally hard-code index 0 anywhere.
    for c in (1, 17, 783):
        dims2 = SwiGLUDims(d_in=784, d_hidden=4096, bias_compensator=c)
        print(f"--- alt bias_compensator={c} ---")
        validate_ffn_identity(dims2)

    # fp64 demonstrates the math is exact — observed fp32 drift is
    # round-off in the (W_up x) accumulator order, NOT an algebraic
    # error in the repack.
    print("########## fp64 (proves repack is algebraically exact) ##########")
    default_dtype_was = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dims64 = SwiGLUDims(d_in=784, d_hidden=4096, bias_compensator=0)
        validate_ffn_identity(dims64)
        validate_bias_compensator_fold(dims64)
    finally:
        torch.set_default_dtype(default_dtype_was)

    # fp32 byte-identity is achievable when we make the summation
    # order match exactly: instead of letting F.linear separately
    # accumulate (W_up x) and then add b_up, we can pad x with a
    # CONST=1 dim and merge the bias into the column.  The
    # *summation order* of the resulting matmul then matches Qwen's,
    # because both go through the same single linear call.
    print("########## fp32 byte-identity check (matched summation order) ##########")
    dims_byte = SwiGLUDims(d_in=784, d_hidden=4096, bias_compensator=0)
    validate_byte_identity_matched_order(dims_byte)


if __name__ == "__main__":
    main()
