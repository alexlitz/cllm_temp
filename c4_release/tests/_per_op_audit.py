"""Per-op audit harness used by ``test_l<N>_per_op.py`` files.

Provides three building blocks for cheap per-op declarative tests:

* ``assert_no_drift(op_names, ...)`` -- wraps ``verify_claims_static`` filtered
  to a specific set of op names and raises if any of them are missing a
  declared claim (a.k.a. ``declared_but_not_written``).
* ``assert_fires_during_bake(op_factory, ...)`` -- builds a stub target
  module (FFN or attention block), runs the op's ``bake_fn`` against it,
  and asserts at least one weight tensor was mutated. Catches the regression
  where a bake silently becomes a no-op because it depends on a flag/mode
  that has drifted from the test fixture.
* ``compile_compact_layout(extra_ops=())`` -- compiles a minimal
  ``LayerCompiler`` layout using ``all_core_ops`` for the dim_positions a
  symbolic forward needs. Avoids the ~70s full bake.

The harness is intentionally narrow: per-op tests provide the op factories
and the specific symbolic-forward asserts; the harness handles the boring
plumbing (stub blocks, claim filtering, bake invocation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
from torch import nn

from c4_release.neural_vm.base_layers import PureFFN
from c4_release.neural_vm.unified_compiler.decl_verifier import (
    OpVerificationResult,
    StaticVerificationReport,
    verify_claims_static,
)
from c4_release.neural_vm.unified_compiler.full_vm_compiler import (
    declare_setdim_compat_dims,
)
from c4_release.neural_vm.unified_compiler.layer_compiler import (
    LayerCompiler,
    Operation,
)
from c4_release.neural_vm.unified_compiler.ops.all_core_ops import all_core_ops
from c4_release.neural_vm.vm_step import AutoregressiveAttention


# ---------------------------------------------------------------------------
# Drift checking (Mode A)
# ---------------------------------------------------------------------------


@dataclass
class _StaticVerifyCache:
    """Module-level cache so multiple per-op assertions in one pytest run
    share a single ~70s ``verify_claims_static`` bake.
    """
    report: Optional[StaticVerificationReport] = None
    key: Optional[tuple] = None


_CACHE = _StaticVerifyCache()


def _cached_static_report(
    *,
    alu_mode: str,
    enable_conversational_io: bool,
    enable_tool_calling: bool,
    S: float,
    n_heads: int,
) -> StaticVerificationReport:
    """Run ``verify_claims_static`` once per (alu_mode, flags) tuple per process.

    Tests that share configuration reuse the cached report; tests with a
    different configuration force a fresh bake.
    """
    key = (alu_mode, enable_conversational_io, enable_tool_calling, S, n_heads)
    if _CACHE.report is None or _CACHE.key != key:
        _CACHE.report = verify_claims_static(
            alu_mode=alu_mode,
            enable_conversational_io=enable_conversational_io,
            enable_tool_calling=enable_tool_calling,
            S=S,
            n_heads=n_heads,
        )
        _CACHE.key = key
    return _CACHE.report


def assert_no_drift(
    op_names: Iterable[str],
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    S: float = 100.0,
    n_heads: int = 8,
    allow_missing: bool = False,
) -> List[OpVerificationResult]:
    """Verify that none of the named ops have declared-but-not-written claims.

    Runs the static claim verifier (Mode A) over a full compiled model and
    filters the per-op results to ``op_names``. Asserts that every named op
    either has no claims (and therefore is not in the report at all) OR is
    in the report with no ``declared_but_not_written`` cells.

    Args:
        op_names: the ``Operation.name`` strings to audit.
        alu_mode, enable_conversational_io, enable_tool_calling, S, n_heads:
            forwarded to ``verify_claims_static`` / ``compile_full_vm``.
        allow_missing: when True, ops that the verifier did not exercise
            (e.g. because they declare no claims) are silently allowed. When
            False (default), at least one named op must appear in the report.

    Returns:
        the filtered list of ``OpVerificationResult`` for caller-side asserts.
    """
    targets = set(op_names)
    report = _cached_static_report(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        S=S,
        n_heads=n_heads,
    )
    matched = [r for r in report.results if r.op_name in targets]
    if not allow_missing and not matched:
        raise AssertionError(
            f"assert_no_drift: none of {sorted(targets)!r} appeared in "
            f"verify_claims_static report. Either the ops do not declare "
            f"claims (and the verifier only inspects claim-bearing ops), "
            f"or registration is broken. Pass allow_missing=True to skip."
        )
    drift_msgs: List[str] = []
    for r in matched:
        if r.declared_but_not_written:
            drift_msgs.append(
                f"  [{r.op_name}] declared_but_not_written="
                f"{sorted(r.declared_but_not_written)}"
            )
    if drift_msgs:
        header = (
            f"assert_no_drift: {len(drift_msgs)} op(s) failed declared-claim "
            f"verification:"
        )
        raise AssertionError("\n".join([header, *drift_msgs]))
    return matched


# ---------------------------------------------------------------------------
# Fires-during-bake (per-op smoke)
# ---------------------------------------------------------------------------


class StubBlock(nn.Module):
    """Minimal block stub with ``ffn``, ``attn``, ``post_ops``.

    The bake_fn for a ``kind="block"`` op accesses ``block.ffn`` and/or
    ``block.attn``; ``kind="ffn"`` ops receive the ffn directly; ``kind="attn"``
    ops receive the attn directly. The dispatcher in ``assert_fires_during_bake``
    selects the appropriate target.
    """

    def __init__(self, d_model: int, *, n_heads: int = 8, ffn_hidden: int = 4096):
        super().__init__()
        self.ffn = PureFFN(d_model, ffn_hidden)
        # AutoregressiveAttention assigns alibi_slopes as a buffer when
        # positional_encoding is "alibi", which is what L9 LEV relays touch.
        self.attn = AutoregressiveAttention(
            d_model, num_heads=n_heads, max_seq_len=512,
            positional_encoding="alibi",
            attention_normalization="softmax1",
        )
        self.post_ops = nn.ModuleList()


def _snapshot_module(module: nn.Module) -> dict:
    return {
        name: param.detach().clone()
        for name, param in module.named_parameters()
    }


def _any_param_changed(before: dict, module: nn.Module) -> List[str]:
    diffs: List[str] = []
    for name, param in module.named_parameters():
        if name not in before:
            diffs.append(name)
            continue
        if not torch.equal(before[name], param.data):
            diffs.append(name)
    return diffs


def assert_fires_during_bake(
    op: Operation,
    dim_positions: dict,
    *,
    d_model: int,
    n_heads: int = 8,
    S: float = 100.0,
    ffn_hidden: int = 4096,
    expect_inert: bool = False,
) -> List[str]:
    """Build a stub target, run ``op.bake_fn``, return the names of mutated
    parameter tensors. Asserts at least one tensor changed (unless
    ``expect_inert=True``, used for flag-gated no-op bakes).

    The dispatcher picks the target by ``op.kind``:
      * ``"ffn"``  -> stub.ffn
      * ``"attn"`` -> stub.attn
      * ``"block"`` -> stub (the whole block)
      * ``"model"`` -> raises; model-kind ops need a full ``AutoregressiveVM``
    """
    stub = StubBlock(d_model, n_heads=n_heads, ffn_hidden=ffn_hidden)
    if op.kind == "ffn":
        target = stub.ffn
    elif op.kind == "attn":
        target = stub.attn
    elif op.kind == "block":
        target = stub
    else:
        raise ValueError(
            f"assert_fires_during_bake: unsupported op.kind={op.kind!r} "
            f"(op={op.name}). Use the per-op test directly for model-kind ops."
        )
    before_ffn = _snapshot_module(stub.ffn)
    before_attn = _snapshot_module(stub.attn)
    # Several legacy bake helpers (e.g. ``_set_layer9_alu``) write directly
    # into ``ffn.W_up[...]`` rather than ``ffn.W_up.data[...]``; wrap in
    # ``no_grad`` so the in-place assignment does not trip autograd's leaf
    # guard.
    with torch.no_grad():
        op.bake_fn(target, dim_positions, S)
    changed = _any_param_changed(before_ffn, stub.ffn) + [
        f"attn.{n}" for n in _any_param_changed(before_attn, stub.attn)
    ]
    if expect_inert:
        if changed:
            raise AssertionError(
                f"assert_fires_during_bake: op {op.name!r} declared "
                f"expect_inert=True but mutated: {changed}"
            )
        return changed
    if not changed:
        raise AssertionError(
            f"assert_fires_during_bake: op {op.name!r} did not mutate any "
            f"weight tensor (bake silently inert -- flag gate drift?)"
        )
    return changed


# ---------------------------------------------------------------------------
# Compact layout helper (for symbolic forward tests)
# ---------------------------------------------------------------------------


def compile_compact_layout(
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    extra_ops: Sequence[Operation] = (),
    n_heads: int = 8,
):
    """Compile a minimal layout via ``LayerCompiler``.

    Returns the compiled layout (with ``dim_positions``, ``d_model``, etc.).
    No model bake -- callers should run individual ``op.bake_fn(...)`` against
    a stub block they construct themselves. This is enough to drive a single
    op's symbolic forward without the ~70s full model build.

    ``d_model`` is padded so ``d_model % n_heads == 0`` (mirrors
    ``decl_verifier._build_layout_only``), since ``AutoregressiveAttention``
    asserts head-divisibility at construction time.
    """
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=True)
    for op in all_core_ops(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
    ):
        compiler.add_op(op)
    for op in extra_ops:
        compiler.add_op(op)
    layout = compiler.compile()
    if layout.d_model % n_heads != 0:
        pad = n_heads - (layout.d_model % n_heads)
        compiler.declare_dim("_pad", pad)
        layout = compiler.compile()
    return layout
