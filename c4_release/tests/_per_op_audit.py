"""Per-op audit primitives for declarative L* FFN rules.

* ``assert_no_drift``: lowers a single ``FFNRule`` (or all rules in a
  ``CompilerIR``) into a stub FFN, runs symbolic and lowered SwiGLU
  forward passes on the shared synthetic state, and asserts agreement
  to ``atol``. The declarative symbolic semantics must match what the
  lowered SwiGLU weights actually compute.

* ``assert_fires_during_bake``: lowers the rules into a stub FFN and
  verifies every rule's W_up / b_up / W_down rows are non-zero, so a
  rule that bakes to all zeros (invisible to the lowered model) is
  caught immediately.

The standalone helper avoids the broken ``c4_release.neural_vm.*`` import
in ``compare_symbolic_to_lowered_ffn``'s default ``PureFFN`` construction
when pytest runs from the ``c4_release`` working directory.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch

from neural_vm.unified_compiler.ir import (
    CompilerIR,
    DimRef,
    FFNRule,
    _synthetic_ffn_state,  # private but stable; reused by the IR comparison helper
)
from neural_vm.unified_compiler.primitives import Primitives


class StubFFN:
    """Minimal SwiGLU-compatible target for L* bake lowering."""

    def __init__(self, *, d_model: int = 512, hidden_dim: int = 4096):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _coerce_ir(rule_or_rules) -> CompilerIR:
    if isinstance(rule_or_rules, CompilerIR):
        return rule_or_rules
    ir = CompilerIR()
    if isinstance(rule_or_rules, FFNRule):
        ir.layer(0).ffn.append(rule_or_rules)
    else:
        ir.layer(0).ffn.rules.extend(tuple(rule_or_rules))
    return ir


def _resolve_key(key: str, dim_positions: Mapping[str, int]) -> Optional[int]:
    ref = DimRef.parse(key)
    if ref.name not in dim_positions:
        return None
    return ref.resolve(dim_positions)


def assert_no_drift(
    rule_or_rules,
    *,
    dim_positions: Mapping[str, int],
    state: Optional[Mapping[str, float]] = None,
    S: float = 1.0,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    msg: str = "",
) -> None:
    """Lower the rule(s), then assert symbolic == lowered forward.

    Accepts a single ``FFNRule``, a sequence of rules, or a ``CompilerIR``.
    Synthesizes a default firing state from the rule conditions when
    ``state`` is omitted, matching ``compare_symbolic_to_lowered_ffn``'s
    own default behaviour.

    ``S`` defaults to ``1.0`` because the synthetic-state margin trick
    relies on ``silu(_SILU_ONE_INPUT) == 1.0`` to normalize the lowered
    SwiGLU hidden against the symbolic write magnitude.
    """

    ir = _coerce_ir(rule_or_rules)
    rules = tuple(ir.layer(0).ffn.rules)
    if state is None:
        state = _synthetic_ffn_state(rules, S=S)
    state = dict(state)

    d_model = max(dim_positions.values()) + 17 if dim_positions else 512
    needed_units = ir.required_ffn_units(layer_idx=0)
    ffn = StubFFN(d_model=d_model, hidden_dim=needed_units + 1)
    ir.lower_ffn(ffn, dim_positions, layer_idx=0, S=S)

    symbolic_out = ir.symbolic_ffn(state, layer_idx=0)

    x = torch.zeros(d_model)
    for key, value in state.items():
        pos = _resolve_key(key, dim_positions)
        if pos is not None:
            x[pos] = float(value)
    up = ffn.W_up @ x + ffn.b_up
    gate = ffn.W_gate @ x + ffn.b_gate
    delta = ffn.W_down @ (torch.nn.functional.silu(up) * gate)

    drift = []
    for key, sym_value in symbolic_out.items():
        if key in state and abs(sym_value - state[key]) < atol:
            continue
        pos = _resolve_key(key, dim_positions)
        if pos is None:
            # Symbolic state can carry NEXT_* sentinels and other keys
            # that have no lowered counterpart; skip them.
            continue
        lowered_value = float(x[pos] + delta[pos])
        if abs(sym_value - lowered_value) > atol + rtol * abs(sym_value):
            drift.append(
                f"  {key}: symbolic={sym_value:.6g}, lowered={lowered_value:.6g}"
            )
    assert not drift, (
        (msg + "\n" if msg else "")
        + "symbolic-vs-lowered drift on rules ["
        + ", ".join(repr(r.name) for r in rules)
        + "]:\n"
        + "\n".join(drift)
    )


def assert_fires_during_bake(
    rules: Sequence[FFNRule],
    *,
    start_unit: int = 0,
    S: float = 100.0,
    BD=None,
) -> StubFFN:
    """Bake ``rules`` into a stub FFN and check each rule's row is non-empty.

    Returns the stub FFN so callers can do further spot-checks.
    """

    if BD is None:
        from neural_vm.vm_step import _SetDim
        BD = _SetDim

    rules = tuple(rules)
    stub = StubFFN(hidden_dim=start_unit + len(rules) + 1)
    dim_names = Primitives.ffn_rule_dim_names(rules)
    dim_positions = Primitives.dim_positions_from_bd(BD, dim_names)
    end = Primitives.lower_ffn_rules(
        stub, rules, dim_positions, start_unit=start_unit, S=S,
    )
    assert end == start_unit + len(rules), (
        f"bake unit count drift: end={end}, expected="
        f"{start_unit + len(rules)}"
    )

    for offset, rule in enumerate(rules):
        unit = start_unit + offset
        assert stub.W_up[unit].abs().sum() > 0.0, (
            f"rule {rule.name!r} (unit {unit}) has empty W_up row"
        )
        assert stub.W_down[:, unit].abs().sum() > 0.0, (
            f"rule {rule.name!r} (unit {unit}) has empty W_down column"
        )
        # b_up encodes the threshold: a non-zero threshold must produce
        # a non-zero b_up after lowering.
        assert stub.b_up[unit].item() != 0.0 or rule.threshold == 0.0, (
            f"rule {rule.name!r} (unit {unit}) has zero b_up "
            f"despite threshold={rule.threshold}"
        )
    return stub
