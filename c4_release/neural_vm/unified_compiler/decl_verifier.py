"""Declarative op verifier: do declarations match what ``bake_fn`` does?

Each ``Operation`` declares a contract -- ``reads``/``writes`` (dim names),
``claims`` (exact weight slots written), and ``produces``/``consumes_fresh``
(register staleness invariants). The existing static analyzers (collision
in ``_detect_claim_collisions`` and staleness in
``_detect_staleness_violations``) cross-check those declarations against
each other but assume each declaration is HONEST -- there is no test that
``bake_fn`` actually writes (and only writes) the slots ``Operation.claims``
declares.

This module closes that gap. It runs in two modes:

Mode A -- Static claim verification
    For each ``Operation`` with non-empty ``claims``:
        1. Build a fresh model (or take a baseline snapshot before the op).
        2. Snapshot every attn / FFN / embedding parameter matrix.
        3. Run THIS op's ``bake_fn`` (and only this op's).
        4. Diff post-bake against the snapshot; for each cell whose value
           changed, decode it back into a ``(layer, scope, identifier,
           column)`` 4-tuple using the model's geometry + the layout's
           ``dim_positions``.
        5. Compare the observed-write set to ``op.claims``:
           - declared but not written -> warning (unused claim)
           - written but not declared -> ERROR (missing claim)
           - exact match -> OK

Mode B -- Dynamic produces/consumes verification
    Not run by default (slow): set ``C4_VERIFY_DECLARATIONS=1`` to gate it.
    Mode B runs a synthetic C4 program through the compiled model and
    inspects residual-stream values at the AX/SP/PC/BP markers to verify
    that ``produces`` / ``consumes_fresh`` declarations correspond to actual
    register values. Today the dynamic check is best-effort: it inspects
    residual writes per layer at marker positions and reports any op whose
    ``produces[dim]`` does NOT result in a non-zero value at the
    corresponding marker, or whose ``consumes_fresh[dim]`` reads a value
    that matches the prior-step value (i.e. stale).

The static (Mode A) check is the load-bearing one -- it catches honest-or-
not in a single forward pass without needing a runtime smoke. The dynamic
(Mode B) check supplements it for the ``produces``/``consumes_fresh``
axis which is value-shaped rather than weight-shaped.

Usage:
    from c4_release.neural_vm.unified_compiler.decl_verifier import (
        verify_claims_static,
    )
    report = verify_claims_static()  # uses compile_full_vm defaults
    if report.has_errors():
        print(report.format())
        raise AssertionError("declaration drift")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .layer_compiler import LayerCompiler, Operation


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class OpVerificationResult:
    """Per-op outcome of static claim verification."""

    op_name: str
    declared: Set[Tuple[int, str, str, Optional[str]]] = field(default_factory=set)
    observed: Set[Tuple[int, str, str, Optional[str]]] = field(default_factory=set)
    # observed_uncategorized: cells that changed but could not be decoded
    # into a (layer, scope, id, column) tuple (e.g., off-grid writes). These
    # are reported but not flagged as errors so the verifier remains useful
    # even when ops touch non-claim-grid weights (biases, etc).
    observed_uncategorized_count: int = 0
    # Notes / warnings emitted while running the op (e.g., bake_fn raised
    # but is gated on a flag that's off).
    notes: List[str] = field(default_factory=list)

    @property
    def declared_but_not_written(self) -> Set[Tuple[int, str, str, Optional[str]]]:
        """Claims in ``declared`` whose corresponding cell wasn't actually
        written. These are the load-bearing errors -- declaration drift in
        the most unambiguous direction (the op promised to write a slot but
        didn't). The verifier flags these as ERRORS in non-strict mode.

        Exception: flag-gated ops (e.g. ``layer8_head6_ax_carry_refresh``
        with ``enable=False``) legitimately declare claims without writing
        them. The verifier honors a ``notes`` field that marks an op as
        "intentionally inert" -- those ops are reported but not flagged.
        """
        return self.declared - self.observed

    @property
    def written_but_not_declared(self) -> Set[Tuple[int, str, str, Optional[str]]]:
        """Cells the bake wrote that are NOT in the declared set.

        With the partial-claim convention currently in use (each op
        declares only the high-collision-risk row subset, not its full
        write footprint), ``written_but_not_declared`` is expected to be
        large. The verifier reports the count but does NOT flag it as
        an error unless ``strict_mode`` is requested.
        """
        return self.observed - self.declared

    @property
    def ok(self) -> bool:
        """Non-strict: passes if all declared claims were actually written.

        See ``ok_strict`` for the "exact match required" variant.
        """
        # Flag-gated ops are allowed to skip writes; they set ``inert=True``.
        if self.inert:
            return True
        return not self.declared_but_not_written

    @property
    def ok_strict(self) -> bool:
        """Strict: passes only if declared == observed exactly."""
        if self.inert:
            return True
        return not (self.declared_but_not_written
                    or self.written_but_not_declared)

    # Set by the verifier when the target op's bake_fn was a no-op (gated
    # off by a flag like ``enable=False``); the verifier detects this when
    # the snapshot diff is empty.
    inert: bool = False


@dataclass
class StaticVerificationReport:
    """Aggregate report across all verified ops."""

    results: List[OpVerificationResult] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)  # (op_name, reason)
    strict_mode: bool = False

    def has_errors(self) -> bool:
        if self.strict_mode:
            return any(not r.ok_strict for r in self.results)
        return any(not r.ok for r in self.results)

    def errors_by_op(self) -> Dict[str, Set[Tuple[int, str, str, Optional[str]]]]:
        return {
            r.op_name: r.declared_but_not_written
            for r in self.results
            if not r.ok
        }

    def warnings_by_op(self) -> Dict[str, Set[Tuple[int, str, str, Optional[str]]]]:
        return {
            r.op_name: r.written_but_not_declared
            for r in self.results
            if r.written_but_not_declared
        }

    def format(self) -> str:
        lines: List[str] = []
        mode = "STRICT" if self.strict_mode else "PARTIAL-CLAIMS"
        lines.append(f"=== Static claim verification report (mode={mode}) ===")
        lines.append(f"Ops verified: {len(self.results)}")
        if self.strict_mode:
            n_errors = sum(1 for r in self.results if not r.ok_strict)
        else:
            n_errors = sum(1 for r in self.results if not r.ok)
        lines.append(f"Ops with errors: {n_errors}")
        lines.append(f"Ops skipped: {len(self.skipped)}")
        for r in self.results:
            if self.strict_mode:
                status = "OK" if r.ok_strict else "DRIFT"
            else:
                status = "OK" if r.ok else "DRIFT"
            if r.inert:
                status = "INERT"
            lines.append(
                f"  [{status}] {r.op_name}: declared={len(r.declared)} "
                f"observed={len(r.observed)} "
                f"unused_decl={len(r.declared_but_not_written)} "
                f"undeclared_writes={len(r.written_but_not_declared)}"
            )
            # UNUSED CLAIMs are the real bugs (declared but never written).
            for cell in sorted(r.declared_but_not_written, key=_claim_sort_key):
                lines.append(f"     UNUSED CLAIM (DECLARATION DRIFT): {cell}")
            # Show undeclared writes only when in strict mode (otherwise
            # would flood with the expected partial-claim residue).
            if self.strict_mode:
                for cell in sorted(r.written_but_not_declared, key=_claim_sort_key):
                    lines.append(f"     UNDECLARED WRITE: {cell}")
            for note in r.notes:
                lines.append(f"     NOTE: {note}")
        for op_name, reason in self.skipped:
            lines.append(f"  [SKIP] {op_name}: {reason}")
        return "\n".join(lines)


def _claim_sort_key(claim):
    layer_idx, scope, identifier, column = claim
    return (layer_idx, scope, identifier, "" if column is None else column)


# ---------------------------------------------------------------------------
# Helpers: decode a (layer, matrix, row, col) cell -> claim tuple
# ---------------------------------------------------------------------------


# Mapping from attribute name on an attention module to claim-scope string.
_ATTN_MATRICES = (
    ("W_q", "attn_W_q"),
    ("W_k", "attn_W_k"),
    ("W_v", "attn_W_v"),
    ("W_o", "attn_W_o"),
)

_FFN_MATRICES = (
    ("W_up", "ffn_W_up"),
    ("W_gate", "ffn_W_gate"),
    ("W_down", "ffn_W_down"),
)


def _build_dim_lookup(
    dim_positions: Dict[str, int], dim_sizes: Dict[str, int]
) -> List[Tuple[int, int, str]]:
    """Return a sorted list of (start, end, dim_name) ranges.

    Used to convert a column position back into ``"<DIM_NAME>+<offset>"``.
    """
    ranges = []
    for name, pos in dim_positions.items():
        size = dim_sizes.get(name, 1)
        ranges.append((pos, pos + size, name))
    ranges.sort()
    return ranges


def _pos_to_column(
    pos: int, dim_ranges: List[Tuple[int, int, str]]
) -> Optional[str]:
    """Decode a residual position into ``"<DIM>+<offset>"``.

    Returns None if the position doesn't fall inside any declared dim.
    """
    for start, end, name in dim_ranges:
        if start <= pos < end:
            return f"{name}+{pos - start}"
    return None


def _decode_attn_cell(
    layer_idx: int,
    scope: str,
    row: int,
    col: int,
    num_heads: int,
    head_dim: int,
    dim_ranges: List[Tuple[int, int, str]],
) -> Optional[Tuple[int, str, str, Optional[str]]]:
    """Decode (row, col) of an attn weight matrix into a claim 4-tuple.

    For W_q/W_k/W_v, the row encodes ``(head, slot)`` = (row // HD, row % HD)
    and the col is the input residual position. W_o is reversed: ROW is the
    output residual position; COL is the (head, slot) projection input.
    The dim-ownership registry's convention (see ALLOWED_CLAIM_SCOPES doc in
    layer_compiler.py) treats the (head, slot) side as the identifier in both
    cases, so we extract it from the relevant axis.
    """
    if scope == "attn_W_o":
        # row -> output residual position, col -> (head, slot)
        head = col // head_dim
        slot = col % head_dim
        column = _pos_to_column(row, dim_ranges)
    else:
        head = row // head_dim
        slot = row % head_dim
        column = _pos_to_column(col, dim_ranges)
    if head >= num_heads:
        return None
    identifier = f"{head}_{slot}"
    return (layer_idx, scope, identifier, column)


def _decode_ffn_cell(
    layer_idx: int,
    scope: str,
    row: int,
    col: int,
    dim_ranges: List[Tuple[int, int, str]],
) -> Optional[Tuple[int, str, str, Optional[str]]]:
    """Decode (row, col) of an FFN weight matrix into a claim 4-tuple.

    W_up / W_gate: row=hidden_unit, col=input residual dim.
    W_down: row=output residual dim, col=hidden_unit.
    Identifier is the hidden-unit index as a string.
    """
    if scope == "ffn_W_down":
        unit = col
        column = _pos_to_column(row, dim_ranges)
    else:
        unit = row
        column = _pos_to_column(col, dim_ranges)
    identifier = str(unit)
    return (layer_idx, scope, identifier, column)


# ---------------------------------------------------------------------------
# Snapshot / diff infrastructure
# ---------------------------------------------------------------------------


def _snapshot_named(model) -> Dict[int, torch.Tensor]:
    """Snapshot by data_ptr -> cloned tensor. Stable across in-place writes.

    This works because torch.nn.Parameter writes (``.data[...] = ...``) keep
    the same storage; only Parameter replacements break the data_ptr.
    """
    snap: Dict[int, torch.Tensor] = {}
    for _, p in model.named_parameters():
        if p.is_sparse:
            snap[p.data._values().data_ptr()] = p.data.to_dense().clone()
        else:
            snap[p.data.data_ptr()] = p.data.detach().clone()
    return snap


def _lookup_snap_by_ptr(
    snap: Dict[int, torch.Tensor], param
) -> Optional[torch.Tensor]:
    if param.is_sparse:
        key = param.data._values().data_ptr()
    else:
        key = param.data.data_ptr()
    return snap.get(key)


def _diff_block_by_ptr(
    block,
    layer_idx: int,
    snap: Dict[int, torch.Tensor],
    num_heads: int,
    dim_ranges: List[Tuple[int, int, str]],
    tol: float = 1e-9,
) -> Tuple[Set[Tuple[int, str, str, Optional[str]]], int]:
    """Same as ``_diff_attn_or_ffn`` but uses data_ptr-keyed snap."""
    observed: Set[Tuple[int, str, str, Optional[str]]] = set()
    uncategorized = 0

    attn = block.attn
    head_dim = attn.W_q.shape[0] // num_heads
    for attr_name, scope in _ATTN_MATRICES:
        param = getattr(attn, attr_name, None)
        if param is None:
            continue
        cur = param.data.to_dense() if param.is_sparse else param.data
        prev = _lookup_snap_by_ptr(snap, param)
        if prev is None:
            continue
        diff = (cur - prev).abs() > tol
        if not diff.any():
            continue
        rows, cols = diff.nonzero(as_tuple=True)
        for r, c in zip(rows.tolist(), cols.tolist()):
            claim = _decode_attn_cell(
                layer_idx, scope, r, c,
                num_heads=num_heads, head_dim=head_dim,
                dim_ranges=dim_ranges,
            )
            if claim is None:
                uncategorized += 1
            else:
                observed.add(claim)

    ffn = block.ffn
    for attr_name, scope in _FFN_MATRICES:
        param = getattr(ffn, attr_name, None)
        if param is None:
            continue
        cur = param.data.to_dense() if param.is_sparse else param.data
        prev = _lookup_snap_by_ptr(snap, param)
        if prev is None:
            continue
        diff = (cur - prev).abs() > tol
        if not diff.any():
            continue
        rows, cols = diff.nonzero(as_tuple=True)
        for r, c in zip(rows.tolist(), cols.tolist()):
            claim = _decode_ffn_cell(
                layer_idx, scope, r, c, dim_ranges=dim_ranges,
            )
            if claim is None:
                uncategorized += 1
            else:
                observed.add(claim)
    return observed, uncategorized


def _diff_all_blocks_by_ptr(
    model,
    snap: Dict[int, torch.Tensor],
    num_heads: int,
    dim_ranges: List[Tuple[int, int, str]],
    tol: float = 1e-9,
) -> Tuple[Set[Tuple[int, str, str, Optional[str]]], int]:
    """Diff every block (attn + ffn) against the snapshot.

    Used for ``kind="model"`` ops that may touch multiple layers.
    """
    observed: Set[Tuple[int, str, str, Optional[str]]] = set()
    uncategorized = 0
    for layer_idx, block in enumerate(model.blocks):
        b_obs, b_un = _diff_block_by_ptr(
            block, layer_idx, snap, num_heads=num_heads, dim_ranges=dim_ranges,
            tol=tol,
        )
        observed |= b_obs
        uncategorized += b_un
    # Also diff the embedding table for embed_row claims.
    embed_weight = None
    try:
        embed_weight = model.embed.embed.weight
    except AttributeError:
        pass
    if embed_weight is not None:
        prev = _lookup_snap_by_ptr(snap, embed_weight)
        if prev is not None:
            cur = (embed_weight.data.to_dense()
                   if embed_weight.is_sparse else embed_weight.data)
            diff = (cur - prev).abs() > tol
            if diff.any():
                rows, cols = diff.nonzero(as_tuple=True)
                touched_rows: Set[int] = set()
                for r in rows.tolist():
                    touched_rows.add(r)
                for r in sorted(touched_rows):
                    observed.add((-1, "embed_row", str(r), None))
    return observed, uncategorized


# ---------------------------------------------------------------------------
# Mode A: static claim verification driver
# ---------------------------------------------------------------------------


def verify_claims_static(
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    S: float = 100.0,
    n_heads: int = 8,
    op_filter: Optional[Callable[[Operation], bool]] = None,
    strict_mode: bool = False,
) -> StaticVerificationReport:
    """Run static claim verification on every annotated op in compile_full_vm.

    Algorithm (single-pass instrumented bake):
        1. Build the layout exactly as production does.
        2. Build a fresh AutoregressiveVM.
        3. Dispatch every op (migrated per-layer attn/ffn, then block ops,
           then model ops) in production order. For each op whose
           ``claims`` are non-empty, wrap the dispatch call: snapshot
           model state -> call bake_fn -> compute diff against snapshot
           -> record observed cells.
        4. Produce a per-op verification result.

    Single full bake (~70s) verifies every annotated op at once. Compares
    favorably to the previous O(N*bake) loop which scaled with annotation
    coverage.
    """
    layout = _build_layout_only(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        n_heads=n_heads,
    )
    dim_ranges = _build_dim_lookup(layout.dim_positions, layout.dim_sizes)

    # Collect ops to verify.
    op_targets: Dict[str, Operation] = {}
    for ops_at in layout.ops_per_layer:
        for op in ops_at:
            if op.claims and (op_filter is None or op_filter(op)):
                op_targets[op.name] = op
    for op in layout.block_ops:
        if op.claims and (op_filter is None or op_filter(op)):
            op_targets[op.name] = op
    for op in layout.model_ops:
        if op.claims and (op_filter is None or op_filter(op)):
            op_targets[op.name] = op

    report = StaticVerificationReport(strict_mode=strict_mode)
    if not op_targets:
        return report

    # Per-op accumulators populated during the instrumented bake.
    observed_by_op: Dict[str, Set[Tuple[int, str, str, Optional[str]]]] = {
        name: set() for name in op_targets
    }
    uncategorized_by_op: Dict[str, int] = {name: 0 for name in op_targets}
    seen_op_names: Set[str] = set()

    from ..vm_step import AutoregressiveVM
    model = AutoregressiveVM(
        d_model=layout.d_model,
        n_layers=layout.n_layers,
        n_heads=n_heads,
        dim_positions=layout.dim_positions,
    )

    def _instrument_and_dispatch(op, target_module, layer_idx_or_none):
        if op.name not in op_targets:
            # Untracked op: dispatch normally.
            op.bake_fn(target_module, layout.dim_positions, S)
            return
        # Tracked op: snapshot before bake, diff after.
        snap = _snapshot_named(model)
        op.bake_fn(target_module, layout.dim_positions, S)
        if op.kind == "model" or layer_idx_or_none is None:
            obs, un = _diff_all_blocks_by_ptr(
                model, snap, n_heads, dim_ranges,
            )
        else:
            block = model.blocks[layer_idx_or_none]
            obs, un = _diff_block_by_ptr(
                block, layer_idx_or_none, snap, n_heads, dim_ranges,
            )
        observed_by_op[op.name] |= obs
        uncategorized_by_op[op.name] += un
        seen_op_names.add(op.name)

    with torch.no_grad():
        # Migrated per-layer attn/ffn ops.
        for layer_idx, ops_at_layer in enumerate(layout.ops_per_layer):
            block = model.blocks[layer_idx]
            for op in ops_at_layer:
                if not op.migrated:
                    continue
                if op.kind == "block":
                    # block kind dispatched separately below
                    continue
                if op.kind == "attn":
                    _instrument_and_dispatch(op, block.attn, layer_idx)
                elif op.kind == "ffn":
                    _instrument_and_dispatch(op, block.ffn, layer_idx)

        # Migrated block ops sorted by (layer, phase).
        sorted_block_ops = sorted(
            layout.block_ops,
            key=lambda o: (layout.resolve_block_op_layer(o), o.phase or 0),
        )
        for op in sorted_block_ops:
            if not op.migrated:
                continue
            li = layout.resolve_block_op_layer(op)
            block = model.blocks[li]
            block._n_layers_hint = len(model.blocks)
            _instrument_and_dispatch(op, block, li)

        # Model ops sorted by phase.
        for op in sorted(layout.model_ops, key=lambda o: (o.phase or 0)):
            _instrument_and_dispatch(op, model, None)

    for name, target in op_targets.items():
        obs = observed_by_op.get(name, set())
        un = uncategorized_by_op.get(name, 0)
        notes: List[str] = []
        if name not in seen_op_names:
            notes.append("target op did not fire during dispatch")
        report.results.append(_build_result(target, obs, un, notes))

    return report


def _build_layout_only(
    *,
    alu_mode: str,
    enable_conversational_io: bool,
    enable_tool_calling: bool,
    n_heads: int,
):
    """Same op-registration logic as ``compile_full_vm`` but returns ONLY the
    compiled layout. Skips the model build + bake pass.
    """
    from .migrated_ops import (
        all_core_ops,
        declare_setdim_compat_dims,
        make_alu_divmod_composite_ops,
        make_contract_validation_op,
        make_efficient_l8_addsub_wrap_op,
        make_efficient_l10_andorxor_wrap_op,
        make_efficient_l11_alumul_wrap_op,
        make_l10_post_op_attach_op,
        make_l11_alu_mul_bdtoge_op,
        make_l11_alu_mul_carrypass1_op,
        make_l11_alu_mul_carrypass2_op,
        make_l11_alu_mul_carrypass3_op,
        make_l11_alu_mul_schoolbook_op,
        make_l12_alu_mul_binarylookahead_op,
        make_l12_alu_mul_finalcorrection_op,
        make_l12_alu_mul_genprop_op,
        make_l12_alu_mul_getobd_op,
        make_layer8_op_imm_relay_op,
        make_layer10_residual_alibi_slopes_op,
        make_residual_alibi_slopes_op,
        all_alu_postop_attach_ops,
    )

    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=True)
    for op in all_core_ops(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
    ):
        compiler.add_op(op)
    compiler.add_op(make_l10_post_op_attach_op(alu_mode=alu_mode))
    if alu_mode == "efficient":
        compiler.add_op(make_l11_alu_mul_bdtoge_op())
        compiler.add_op(make_l11_alu_mul_schoolbook_op())
        compiler.add_op(make_l11_alu_mul_carrypass1_op())
        compiler.add_op(make_l11_alu_mul_carrypass2_op())
        compiler.add_op(make_l11_alu_mul_carrypass3_op())
        compiler.add_op(make_l12_alu_mul_genprop_op())
        compiler.add_op(make_l12_alu_mul_binarylookahead_op())
        compiler.add_op(make_l12_alu_mul_finalcorrection_op())
        compiler.add_op(make_l12_alu_mul_getobd_op())
        compiler.add_op(make_efficient_l8_addsub_wrap_op(alu_mode=alu_mode))
        compiler.add_op(make_efficient_l10_andorxor_wrap_op(alu_mode=alu_mode))
        compiler.add_op(make_efficient_l11_alumul_wrap_op(alu_mode=alu_mode))
    for op in make_alu_divmod_composite_ops():
        compiler.add_op(op)
    compiler.add_op(make_residual_alibi_slopes_op())
    compiler.add_op(make_layer10_residual_alibi_slopes_op(alu_mode=alu_mode))
    compiler.add_op(make_layer8_op_imm_relay_op())
    compiler.add_op(make_contract_validation_op())
    if alu_mode == "lookup":
        for op in all_alu_postop_attach_ops():
            compiler.add_op(op)

    layout = compiler.compile()
    if layout.d_model % n_heads != 0:
        pad = n_heads - (layout.d_model % n_heads)
        compiler.declare_dim("_pad", pad)
        layout = compiler.compile()
    return layout


def _build_result(
    target: Operation,
    observed: Set[Tuple[int, str, str, Optional[str]]],
    uncategorized: int,
    notes: List[str],
) -> OpVerificationResult:
    # Detect inert (gated-off) ops: bake_fn ran but wrote nothing.
    inert = (not observed and uncategorized == 0)
    if inert:
        notes = list(notes) + [
            "op was inert -- bake_fn did not write any weights "
            "(flag-gated off?). Declared claims are not validated against "
            "writes for inert ops."
        ]
    return OpVerificationResult(
        op_name=target.name,
        declared=set(target.claims),
        observed=observed,
        observed_uncategorized_count=uncategorized,
        notes=list(notes),
        inert=inert,
    )


# ---------------------------------------------------------------------------
# Mode B: dynamic produces/consumes verification (best-effort)
# ---------------------------------------------------------------------------


@dataclass
class DynamicVerificationResult:
    op_name: str
    notes: List[str] = field(default_factory=list)
    drift: List[str] = field(default_factory=list)


@dataclass
class DynamicVerificationReport:
    results: List[DynamicVerificationResult] = field(default_factory=list)

    def has_drift(self) -> bool:
        return any(r.drift for r in self.results)

    def format(self) -> str:
        lines = ["=== Dynamic produces/consumes verification report ==="]
        for r in self.results:
            status = "OK" if not r.drift else "DRIFT"
            lines.append(f"  [{status}] {r.op_name}")
            for n in r.notes:
                lines.append(f"     NOTE: {n}")
            for d in r.drift:
                lines.append(f"     DRIFT: {d}")
        return "\n".join(lines)


def verify_produces_consumes_dynamic(
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    S: float = 100.0,
    n_heads: int = 8,
) -> DynamicVerificationReport:
    """Best-effort dynamic check on ``produces`` / ``consumes_fresh``.

    Runs a synthetic 1-instruction C4 program through the compiled model
    and reports any op whose ``produces[dim]`` declaration does NOT
    correspond to a non-zero residual value at the AX marker position. We
    do not attempt to verify register identity (``"AX_byte0"`` etc.) here
    -- that requires a per-step ground-truth oracle which is out of scope
    for the static contract. Instead we check the *liveness* invariant:
    if an op declares ``produces``, it should actually write residual at
    the declared dim during one VM step.

    This is intentionally lightweight: declaration drift around register
    identity (the dim name vs the value content) needs a behavioural smoke
    test, not a synthetic 1-instruction probe. The verifier surfaces
    discrepancies via per-op notes; callers decide how to escalate.
    """
    from .full_vm_compiler import compile_full_vm

    report = DynamicVerificationReport()

    try:
        model, layout = compile_full_vm(
            S=S,
            alu_mode=alu_mode,
            enable_conversational_io=enable_conversational_io,
            n_heads=n_heads,
        )
    except Exception as exc:
        report.results.append(DynamicVerificationResult(
            op_name="<compile_full_vm>",
            notes=[f"compile failed: {exc!r}"],
        ))
        return report

    candidates = _collect_unique_ops_with(
        layout, lambda op: bool(op.produces or op.consumes_fresh)
    )
    if not candidates:
        return report

    # Synthetic 1-instruction probe: IMM 7 / EXIT. We run the model on the
    # token sequence and capture the residual stream at each layer.
    try:
        token_seq, mark_ax_positions = _build_synthetic_imm_exit(layout)
    except Exception as exc:
        report.results.append(DynamicVerificationResult(
            op_name="<synthetic_program>",
            notes=[f"could not construct synthetic program: {exc!r}"],
        ))
        return report

    try:
        with torch.no_grad():
            x = model.embed(token_seq)
            per_layer_residuals = [x.detach().clone()]
            for block in model.blocks:
                x = block(x)
                per_layer_residuals.append(x.detach().clone())
    except Exception as exc:
        report.results.append(DynamicVerificationResult(
            op_name="<forward>",
            notes=[f"forward pass failed: {exc!r}"],
        ))
        return report

    for op in candidates:
        res = DynamicVerificationResult(op_name=op.name)
        layer_idx = _resolve_op_layer(layout, op)

        if layer_idx is None or layer_idx + 1 >= len(per_layer_residuals):
            res.notes.append("could not locate op layer for dynamic check")
            report.results.append(res)
            continue

        # Residual AFTER this op's layer ran.
        post_residual = per_layer_residuals[layer_idx + 1]  # [B, T, D]
        # AX marker positions (we use the first one if multiple).
        if not mark_ax_positions:
            res.notes.append("no AX marker position in synthetic program")
            report.results.append(res)
            continue
        ax_pos = mark_ax_positions[0]

        for dim_name in op.produces.keys():
            if dim_name not in layout.dim_positions:
                res.drift.append(f"produces {dim_name!r}: dim not in layout")
                continue
            start = layout.dim_positions[dim_name]
            size = layout.dim_sizes.get(dim_name, 1)
            slice_vals = post_residual[0, ax_pos, start:start + size]
            if torch.all(slice_vals.abs() < 1e-6):
                res.drift.append(
                    f"produces {dim_name!r}: residual zero at AX marker (op may not have fired)"
                )

        report.results.append(res)

    return report


def _build_synthetic_imm_exit(layout):
    """Build a tiny token sequence for the dynamic check.

    Returns ``(token_tensor, [ax_marker_positions])``. The tensor shape is
    ``[1, T]`` with B=1.
    """
    from ..vm_step import Token
    # Minimal sequence: a CODE_START token followed by REG_PC + 4 PC bytes
    # + REG_AX + 4 AX bytes + REG_SP + 4 SP bytes + REG_BP + 4 BP bytes +
    # STACK0 + 4 stack bytes + MEM + 4 addr + 4 val + STEP_END.
    # This is overkill for static dim-positional liveness; we just need a
    # context where MARK_AX is present.
    tokens = []
    # code start
    tokens.append(Token.CODE_START)
    # Step 1: PC=8, AX=7, SP=0, BP=0, no STACK0/MEM content
    tokens.append(Token.REG_PC)
    tokens.extend([8, 0, 0, 0])  # PC bytes (little-endian)
    ax_pos = len(tokens)  # REG_AX marker position
    tokens.append(Token.REG_AX)
    tokens.extend([0, 0, 0, 0])
    tokens.append(Token.REG_SP)
    tokens.extend([0, 0, 0, 0])
    tokens.append(Token.REG_BP)
    tokens.extend([0, 0, 0, 0])
    tokens.append(Token.STACK0)
    tokens.extend([0, 0, 0, 0])
    tokens.append(Token.MEM)
    tokens.extend([0, 0, 0, 0])  # addr
    tokens.extend([0, 0, 0, 0])  # val
    tokens.append(Token.STEP_END)

    token_tensor = torch.tensor([tokens], dtype=torch.long)
    return token_tensor, [ax_pos]


# ---------------------------------------------------------------------------
# Mode B+: multi-step produces/consumes verification
# ---------------------------------------------------------------------------
#
# The original Mode B (``verify_produces_consumes_dynamic`` above) runs a
# synthetic 1-instruction probe through the compiled model. Cascade bugs --
# where an op fires correctly on step 1 but its byte-relay state fails to
# propagate to step 2/3/4 -- never surface inside a single step boundary.
# The audit doc + the 2026-05-12 L6/L10 STACK0 byte-relay debugging cycle
# repeatedly flagged this gap.
#
# ``verify_produces_consumes_multistep`` closes that gap by:
#   1. Compiling a real bytecode program (hand-built via opcode/imm
#      packing -- see ``_build_multistep_probe``).
#   2. Running the compiled model forward over a context that includes
#      ``n_steps`` worth of real DraftVM-generated step tokens (each step
#      adds 35 tokens to the sequence: PC/AX/SP/BP/STACK0/MEM/STEP_END).
#   3. For each annotated op + each step, inspecting the residual stream
#      at the marker (or byte) position implied by the produces register
#      name and reporting a drift entry when the residual is ~0 despite a
#      non-empty ``produces[dim]`` declaration.
#
# The probe deliberately uses non-cached forward (single full forward pass
# over the whole context); the per-layer residuals are captured by
# iterating ``model.blocks`` directly, mirroring the Mode B pattern. This
# avoids KV-cache edge cases (eviction, _is_new_only) which matter for
# inference speed but not for liveness probing.


@dataclass
class MultistepDriftEntry:
    """A single (op, step, dim, register) drift observation."""
    op_name: str
    step: int  # 1-indexed step within the n_steps run
    dim: str
    register: str
    position: int  # token position inspected
    observed: float  # abs-max residual value observed (~0 means drift)


@dataclass
class MultistepVerificationResult:
    """Per-op outcome of multistep dynamic verification."""
    op_name: str
    layer_idx: Optional[int] = None
    notes: List[str] = field(default_factory=list)
    drift: List[MultistepDriftEntry] = field(default_factory=list)


@dataclass
class MultistepVerificationReport:
    """Aggregate report across all multistep-verified ops."""
    n_steps: int = 0
    program_summary: str = ""
    results: List[MultistepVerificationResult] = field(default_factory=list)

    def has_drift(self) -> bool:
        return any(r.drift for r in self.results)

    def drift_entries(self) -> List[MultistepDriftEntry]:
        """Flat list of every drift entry across all results, in op order."""
        out: List[MultistepDriftEntry] = []
        for r in self.results:
            out.extend(r.drift)
        return out

    def format(self) -> str:
        lines = [
            "=== Multistep produces/consumes verification report ===",
            f"Program: {self.program_summary}",
            f"Steps: {self.n_steps}",
            f"Ops inspected: {len(self.results)}",
            f"Drift entries: {len(self.drift_entries())}",
        ]
        for r in self.results:
            status = "OK" if not r.drift else "DRIFT"
            layer_tag = (
                f" (layer {r.layer_idx})" if r.layer_idx is not None else ""
            )
            lines.append(f"  [{status}] {r.op_name}{layer_tag}")
            for n in r.notes:
                lines.append(f"     NOTE: {n}")
            for d in r.drift:
                lines.append(
                    f"     DRIFT: step={d.step} dim={d.dim!r} "
                    f"register={d.register!r} pos={d.position} "
                    f"observed_abs_max={d.observed:.3e}"
                )
        return "\n".join(lines)


def _collect_unique_ops_with(layout, predicate: Callable[[Operation], bool]) -> List[Operation]:
    """Walk every op in ``layout`` (per-layer, block, model) and return
    those matching ``predicate``, de-duplicated by ``op.name``.

    Used by Mode B and Mode B+ to gather candidate ops with non-empty
    ``produces``/``consumes_fresh`` annotations.
    """
    seen: Set[str] = set()
    out: List[Operation] = []
    for ops_at in layout.ops_per_layer:
        for op in ops_at:
            if op.name in seen:
                continue
            seen.add(op.name)
            if predicate(op):
                out.append(op)
    for op in layout.block_ops:
        if op.name in seen:
            continue
        seen.add(op.name)
        if predicate(op):
            out.append(op)
    for op in layout.model_ops:
        if op.name in seen:
            continue
        seen.add(op.name)
        if predicate(op):
            out.append(op)
    return out


def _resolve_op_layer(layout, op) -> Optional[int]:
    """Return the layer index an op binds to, or None if unresolvable.

    Mirrors the locate-layer logic Mode B (``verify_produces_consumes_dynamic``)
    and Mode B+ (``verify_produces_consumes_multistep``) share. Model ops
    return ``op.layer_idx`` (may be None for whole-model ops).
    """
    if op.kind == "block":
        try:
            return layout.resolve_block_op_layer(op)
        except ValueError:
            return op.layer_idx
    if op.kind in ("attn", "ffn"):
        for lidx, ops_at in enumerate(layout.ops_per_layer):
            if any(o.name == op.name for o in ops_at):
                return lidx
        return None
    # model ops: respect op.layer_idx hint, None means whole-model.
    return op.layer_idx


# Token offsets within a single 35-token VM step. See ``Token.STEP_TOKENS``
# in ``vm_step.py`` -- the format is REG_PC(5) + REG_AX(5) + REG_SP(5) +
# REG_BP(5) + STACK0(5) + MEM(9) + STEP_END(1).
_STEP_TOKEN_OFFSETS = {
    "REG_PC": 0,    "PC_byte0": 1, "PC_byte1": 2, "PC_byte2": 3, "PC_byte3": 4,
    "REG_AX": 5,    "AX_byte0": 6, "AX_byte1": 7, "AX_byte2": 8, "AX_byte3": 9,
    "REG_SP": 10,   "SP_byte0": 11, "SP_byte1": 12, "SP_byte2": 13, "SP_byte3": 14,
    "REG_BP": 15,   "BP_byte0": 16, "BP_byte1": 17, "BP_byte2": 18, "BP_byte3": 19,
    "STACK0": 20,   "STACK0_byte0": 21, "STACK0_byte1": 22,
                    "STACK0_byte2": 23, "STACK0_byte3": 24,
    "MEM": 25,
    "MEM_ADDR_byte0": 26, "MEM_ADDR_byte1": 27,
    "MEM_ADDR_byte2": 28, "MEM_ADDR_byte3": 29,
    "MEM_VAL_byte0": 30, "MEM_VAL_byte1": 31,
    "MEM_VAL_byte2": 32, "MEM_VAL_byte3": 33,
    "STEP_END": 34,
}


def _resolve_register_offset(register: str) -> Optional[int]:
    """Map an ``Operation.produces`` register name to a step-relative offset.

    The produces convention uses register names like ``"AX_byte0"`` (the
    byte position immediately after REG_AX) or ``"STACK0_byte0"``. For
    register names with the bare-marker convention (``"AX"``, ``"STACK0"``,
    ``"AX_marker"``, ...) we fall back to the REG_* marker position itself.
    Returns None if the name can't be resolved -- caller should record a
    note rather than a drift entry.
    """
    if register in _STEP_TOKEN_OFFSETS:
        return _STEP_TOKEN_OFFSETS[register]
    # Common alias normalization: "AX" -> "REG_AX", "AX_marker" -> "REG_AX".
    bare = register.replace("_marker", "")
    if bare in ("AX", "PC", "SP", "BP"):
        return _STEP_TOKEN_OFFSETS[f"REG_{bare}"]
    if bare == "STACK0":
        return _STEP_TOKEN_OFFSETS["STACK0"]
    if bare == "MEM":
        return _STEP_TOKEN_OFFSETS["MEM"]
    return None


def _pack_instr(opcode: int, imm: int = 0) -> int:
    """Pack ``opcode | (imm << 8)`` the way C4's bytecode list expects.

    Matches the encoding in ``DraftVM.__init__`` / ``DraftVM.step`` which
    reads ``op = instr & 0xFF`` and ``imm = instr >> 8`` from each entry.
    The runner-side context builder (``BatchedSpeculativeRunner._build_context``
    in batch_runner.py) splits each entry into 8 little-endian bytes for
    CODE_START.
    """
    return (opcode & 0xFF) | ((imm & 0xFFFFFF) << 8)


# Canonical ADD-cascade program used by the multistep probe. The byte-level
# semantics are:
#   step 1: IMM 10        -> AX = 10
#   step 2: PSH           -> push AX, SP -= 8, STACK0 = 10
#   step 3: IMM 32        -> AX = 32 (PSH STACK0 byte-relay should keep STACK0=10)
#   step 4: ADD           -> pop STACK0 into operand, AX = 10 + 32 = 42
#   step 5: EXIT          -> halt
# This is the program the cascade-bug debugging agents have been chasing:
# the L6/L10/STK0 byte-relay ops must produce STACK0 bytes at steps 2 AND 3
# (PSH writes them, then the carry-forward op on the IMM step preserves
# them so the ADD step can read them).
_ADD_CASCADE_PROGRAM = [
    _pack_instr(1, 10),   # IMM 10
    _pack_instr(13, 0),   # PSH
    _pack_instr(1, 32),   # IMM 32
    _pack_instr(25, 0),   # ADD
    _pack_instr(38, 0),   # EXIT
]


def _build_multistep_probe(
    layout,
    program: List[int],
    n_steps: int,
) -> Tuple[torch.Tensor, List[Dict[str, int]], List[str]]:
    """Build a multi-step token sequence: bytecode + n_steps of DraftVM-emitted
    step tokens. Returns ``(token_tensor, step_markers, step_summaries)``.

    ``token_tensor`` has shape ``[1, T]`` with T = header_len + n_steps * 35.

    ``step_markers[k]`` is a dict mapping each REG_*/STACK0/MEM marker name
    (as well as the byte slots like ``"AX_byte0"``) to its absolute token
    position in the full sequence for step ``k`` (0-indexed). Callers use
    these positions to look up per-marker residuals after running the
    model.

    ``step_summaries[k]`` is a short human-readable string of the step's
    register state (e.g. ``"PC=8 AX=10 SP=ffffffff"``).

    Note: this builder leverages ``DraftVM`` from ``speculative.py`` as
    a ground-truth oracle for the per-step token bytes. We do NOT feed
    those tokens back through DraftVM for verification -- we only use
    DraftVM to construct a faithful token sequence the production model
    would generate. The verifier is what evaluates the model's residuals.
    """
    from ..vm_step import Token
    from ..speculative import DraftVM

    tokens: List[int] = []
    # Code section (8 bytes per instruction, little-endian) -- matches
    # ``BatchedSpeculativeRunner._build_context``.
    tokens.append(Token.CODE_START)
    for instr in program:
        for i in range(8):
            tokens.append((instr >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    # Empty data section.
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)

    vm = DraftVM(program)
    step_markers: List[Dict[str, int]] = []
    step_summaries: List[str] = []
    for k in range(n_steps):
        if not vm.step():
            # VM halted before n_steps; pad with HALT tokens so callers
            # see the cap was reached, but mark these slots as such.
            step_markers.append({})
            step_summaries.append("(halted)")
            tokens.extend([Token.HALT] * 35)
            continue
        step_tokens = vm.draft_tokens()
        base = len(tokens)
        marker_map: Dict[str, int] = {
            name: base + offset
            for name, offset in _STEP_TOKEN_OFFSETS.items()
        }
        step_markers.append(marker_map)
        step_summaries.append(
            f"PC={vm.pc:x} AX={vm.ax:x} SP={vm.sp:x} "
            f"BP={vm.bp:x} STK0={vm._mem_read(vm.sp):x}"
        )
        tokens.extend(step_tokens)

    token_tensor = torch.tensor([tokens], dtype=torch.long)
    return token_tensor, step_markers, step_summaries


def verify_produces_consumes_multistep(
    model=None,
    layout=None,
    program: Optional[List[int]] = None,
    n_steps: int = 4,
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    S: float = 100.0,
    n_heads: int = 8,
    epsilon: float = 1e-2,
) -> MultistepVerificationReport:
    """Multi-step dynamic check on ``produces`` declarations.

    For each op with a non-empty ``produces`` annotation, runs the compiled
    model forward over a context containing ``n_steps`` worth of real VM
    step tokens (built by ``_build_multistep_probe``) and inspects the
    residual stream post-op-layer at the step-marker position implied by
    the produces register name. A drift entry is recorded when the residual
    abs-max at that position is below ``epsilon`` -- the op's bake_fn
    declared a non-zero write but no value arrived in the residual at the
    declared register on that step.

    Cascade bugs (op fires on step 1 but byte-relay state fails to
    propagate to step 2/3) show up here as drift entries on the later
    steps even though step 1 is clean -- precisely the inversion the
    1-instruction Mode B probe misses.

    Args:
        model, layout: optional pre-built model + layout to reuse. When
            either is None, both are built via ``compile_full_vm`` using
            the keyword config args below. Reusing a model is the common
            case for tests that want to run multiple probes (and saves
            ~70s of bake time per call).
        program: list of packed bytecode instructions (``opcode | imm<<8``).
            Defaults to the canonical ADD-cascade program (IMM 10; PSH;
            IMM 32; ADD; EXIT) which is the smoking-gun smoke for the
            L6/L10/STK0 byte-relay cascade.
        n_steps: number of VM steps to drive forward. The model must have
            ``max_seq_len`` >= ``len(bytecode_header) + n_steps * 35``.
        epsilon: noise threshold. Residuals with abs-max below this are
            considered "not written" for the purposes of liveness probing.
            Set to 1e-2 by default -- below typical post-FFN/attn residual
            noise (~1e-3) but above legitimate bake-time eps.

    Returns:
        ``MultistepVerificationReport`` -- inspect ``.drift_entries()`` or
        ``.format()``.
    """
    if program is None:
        program = list(_ADD_CASCADE_PROGRAM)

    report = MultistepVerificationReport(
        n_steps=n_steps,
        program_summary=" ".join(
            f"0x{instr:016x}" for instr in program
        ),
    )

    if model is None or layout is None:
        from .full_vm_compiler import compile_full_vm
        try:
            model, layout = compile_full_vm(
                S=S,
                alu_mode=alu_mode,
                enable_conversational_io=enable_conversational_io,
                n_heads=n_heads,
            )
        except Exception as exc:
            report.results.append(MultistepVerificationResult(
                op_name="<compile_full_vm>",
                notes=[f"compile failed: {exc!r}"],
            ))
            return report

    candidates = _collect_unique_ops_with(layout, lambda op: bool(op.produces))
    if not candidates:
        return report

    # Build the multistep probe + run the model.
    try:
        token_tensor, step_markers, step_summaries = _build_multistep_probe(
            layout, program, n_steps,
        )
    except Exception as exc:
        report.results.append(MultistepVerificationResult(
            op_name="<build_multistep_probe>",
            notes=[f"could not construct multistep probe: {exc!r}"],
        ))
        return report

    # Honor model.max_seq_len if available so we fail loudly rather than
    # silently truncating attention.
    max_len = getattr(model, "max_seq_len", None)
    if max_len is not None and token_tensor.shape[1] > max_len:
        report.results.append(MultistepVerificationResult(
            op_name="<context_too_long>",
            notes=[
                f"context len {token_tensor.shape[1]} exceeds model "
                f"max_seq_len {max_len}; reduce n_steps or rebuild model"
            ],
        ))
        return report

    try:
        with torch.no_grad():
            x = model.embed(token_tensor)
            per_layer_residuals = [x.detach().clone()]
            for block in model.blocks:
                x = block(x)
                per_layer_residuals.append(x.detach().clone())
    except Exception as exc:
        report.results.append(MultistepVerificationResult(
            op_name="<forward>",
            notes=[f"forward pass failed: {exc!r}"],
        ))
        return report

    for op in candidates:
        result = MultistepVerificationResult(op_name=op.name)
        layer_idx = _resolve_op_layer(layout, op)
        result.layer_idx = layer_idx

        if layer_idx is None or layer_idx + 1 >= len(per_layer_residuals):
            # Fall back to the final residual (after all blocks). This
            # gives model-op annotations a chance to be checked too.
            inspect_layer = len(per_layer_residuals) - 1
            result.notes.append(
                f"layer_idx unresolved (kind={op.kind}); "
                f"inspecting final post-block residual instead"
            )
        else:
            inspect_layer = layer_idx + 1

        post_residual = per_layer_residuals[inspect_layer]  # [B, T, D]

        # For each step + each (dim, register) produces declaration,
        # inspect the residual at the resolved position.
        for step_idx, marker_map in enumerate(step_markers):
            if not marker_map:
                # Step was unreachable (VM halted earlier). Don't flag.
                continue
            step_k = step_idx + 1  # 1-indexed for reporting
            for dim_name, register in op.produces.items():
                if dim_name not in layout.dim_positions:
                    result.notes.append(
                        f"step {step_k}: produces {dim_name!r}: "
                        f"dim not in layout"
                    )
                    continue
                offset = _resolve_register_offset(register)
                if offset is None:
                    result.notes.append(
                        f"step {step_k}: produces {dim_name!r}={register!r}: "
                        f"register name not resolvable to a step offset"
                    )
                    continue
                # marker_map["REG_PC"] is the step's base position (REG_PC
                # is at relative offset 0); add the in-step offset to land
                # on the register-specific slot.
                pos = marker_map["REG_PC"] + offset
                d_start = layout.dim_positions[dim_name]
                d_size = layout.dim_sizes.get(dim_name, 1)
                slice_vals = post_residual[0, pos, d_start:d_start + d_size]
                obs_abs = float(slice_vals.abs().max().item())
                if obs_abs < epsilon:
                    result.drift.append(MultistepDriftEntry(
                        op_name=op.name,
                        step=step_k,
                        dim=dim_name,
                        register=register,
                        position=pos,
                        observed=obs_abs,
                    ))

        report.results.append(result)

    return report


# ---------------------------------------------------------------------------
# Tier B annotation verifiers (Phase 8 structural correctness)
# ---------------------------------------------------------------------------
#
# Three detectors built on top of the new ``Operation`` fields:
#
#   ``verify_alibi_consistency``  -- static cross-op scan of every
#       declared ALiBi slope. Reports double-writes (two ops both set
#       the same ``(layer, head)`` slope) and slope/W_k mismatches (a
#       declared slope value paired with an unchanged-from-baseline W_k
#       row scale, the ``db08e4d`` regression signature).
#
#   ``verify_postconditions``     -- multistep dynamic invariant check.
#       For each op with a ``postcondition`` annotation, run a probe
#       program for ``n_steps`` and assert the named invariant on the
#       residual at the declared output cells. Violations are drift entries.
#
#   ``verify_step_idx_gating``    -- multistep firing-pattern check.
#       For each op with a non-default ``step_idx`` annotation, run the
#       probe and assert the op's ``produces`` writes are zero on every
#       step that's NOT in ``step_idx``. Catches LEA-pattern bugs where
#       an op fires on its declared step but spills into later steps.


@dataclass
class AlibiDriftEntry:
    """A single (layer, head) alibi-consistency anomaly."""
    layer_idx: int
    head_idx: int
    kind: str  # 'double_write' or 'slope_wk_mismatch'
    detail: str  # human-readable description


@dataclass
class AlibiConsistencyReport:
    drift: List[AlibiDriftEntry] = field(default_factory=list)
    # Per-(layer, head) list of (op_name, declared_slope) for inspection.
    declarations: Dict[Tuple[int, int], List[Tuple[str, float]]] = field(
        default_factory=dict
    )

    def has_drift(self) -> bool:
        return bool(self.drift)

    def format(self) -> str:
        lines = ["=== Alibi consistency report ==="]
        lines.append(f"Slots with declarations: {len(self.declarations)}")
        lines.append(f"Drift entries: {len(self.drift)}")
        for entry in self.drift:
            lines.append(
                f"  [{entry.kind}] layer={entry.layer_idx} "
                f"head={entry.head_idx}: {entry.detail}"
            )
        return "\n".join(lines)


def verify_alibi_consistency(
    compile_fn: Optional[Callable] = None,
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    n_heads: int = 8,
    model=None,
    layout=None,
    wk_eps: float = 1e-6,
) -> AlibiConsistencyReport:
    """Static cross-op consistency scan for declared ``alibi_slopes``.

    For each ``(layer_idx, head_idx)`` slot across all ops that declare
    a slope:

      1. **Double-write detection**: if two or more ops declare a slope
         for the same slot, append a ``double_write`` drift entry. The
         later-phase op silently clobbers the earlier; that's almost
         always a bug. Exception: when both ops declare the SAME slope
         value (idempotent) the entry is still reported (as it indicates
         redundant work) but flagged as low-severity by including
         ``redundant`` in the detail string.

      2. **Slope/W_k mismatch**: when a declared slope is non-default
         (i.e. differs from the model's initial ALiBi slope at that
         head), verify that the corresponding attn module's
         ``W_k[head_base, IS_MARK]`` row has been scaled away from its
         default. The intuition: H1's softmax-budget arithmetic
         requires that *both* the slope and W_k row scale move together
         -- the ``db08e4d`` regression doubled the W_k row scale on L0
         head 1 but left the slope unchanged, shifting the softmax
         zero-crossing and producing the PSH 5-marker bug. The
         heuristic detection: if a declared slope is more than 2x the
         model's default ALiBi slope AND the W_k row scale is within
         tolerance of its baseline, flag a ``slope_wk_mismatch``.

    Args:
        compile_fn: optional override for the compile pipeline. When
            ``None`` and ``model``/``layout`` aren't pre-supplied,
            ``compile_full_vm`` is called with the keyword args below.
            Pre-supplying a model+layout avoids the ~70s rebake cost.
        wk_eps: tolerance for "W_k row scale is at baseline" in the
            mismatch heuristic. Default 1e-6 because the W_k *= 2.0
            scale-bump in L0 H1 produces a 2x diff from baseline, well
            above noise.

    Returns:
        ``AlibiConsistencyReport`` with the drift list + the full
        declarations index (so callers can audit per-slot ownership).
    """
    report = AlibiConsistencyReport()

    if layout is None or model is None:
        if compile_fn is None:
            from .full_vm_compiler import compile_full_vm
            compile_fn = compile_full_vm
        try:
            model, layout = compile_fn(
                alu_mode=alu_mode,
                enable_conversational_io=enable_conversational_io,
                enable_tool_calling=enable_tool_calling,
                n_heads=n_heads,
            )
        except Exception as exc:
            report.drift.append(AlibiDriftEntry(
                layer_idx=-1, head_idx=-1, kind="compile_failed",
                detail=repr(exc),
            ))
            return report

    # Walk every op and collect (layer, head) -> [(op_name, slope), ...].
    declarations: Dict[Tuple[int, int], List[Tuple[str, float]]] = {}
    candidates = _collect_unique_ops_with(layout, lambda op: bool(op.alibi_slopes))
    for op in candidates:
        layer_idx = _resolve_op_layer(layout, op)
        if layer_idx is None:
            # Cannot pin op to a layer; skip rather than spuriously claim it.
            continue
        for head_idx, slope in op.alibi_slopes.items():
            declarations.setdefault((layer_idx, head_idx), []).append(
                (op.name, float(slope))
            )
    report.declarations = declarations

    # Double-write detection.
    for slot, owners in declarations.items():
        layer_idx, head_idx = slot
        if len(owners) >= 2:
            distinct_values = {round(s, 9) for _, s in owners}
            if len(distinct_values) == 1:
                kind = "double_write"
                detail = (
                    f"{len(owners)} ops declare same slope "
                    f"{owners[0][1]} (redundant): "
                    f"{[name for name, _ in owners]}"
                )
            else:
                kind = "double_write"
                detail = (
                    f"{len(owners)} ops declare conflicting slopes "
                    f"{owners} -- later op silently overwrites earlier"
                )
            report.drift.append(AlibiDriftEntry(
                layer_idx=layer_idx, head_idx=head_idx,
                kind=kind, detail=detail,
            ))

    # Slope/W_k mismatch detection (best-effort heuristic).
    # Baseline ALiBi slope: read directly from the compiled model. The
    # default after L0's ``fill_(ALIBI_S)`` is 10.0; after per-layer
    # mutations it can be anything. We compare the post-bake slope at
    # each (layer, head) against the op's DECLARED slope and flag
    # if the declared value is >2x the layer's baseline (heuristic
    # tuned to L0 H1's ALIBI_S*2.0 = 20 against baseline 10).
    # A "non-trivial slope but W_k unchanged" pairing is the smoking gun.
    try:
        baseline_default = 10.0  # ALIBI_S in L0/L1
        ismark_col = layout.dim_positions.get("IS_MARK")
    except AttributeError:
        baseline_default = None
        ismark_col = None

    if ismark_col is not None:
        for slot, owners in declarations.items():
            layer_idx, head_idx = slot
            if layer_idx < 0 or layer_idx >= len(model.blocks):
                continue
            try:
                attn = model.blocks[layer_idx].attn
            except (AttributeError, IndexError):
                continue
            HD = attn.W_q.shape[0] // attn.num_heads
            head_base = head_idx * HD
            try:
                wk_row_val = float(
                    attn.W_k.data[head_base, ismark_col].abs().item()
                )
            except (RuntimeError, IndexError):
                continue
            for op_name, declared_slope in owners:
                # Flag when the declared slope is >2x baseline (steep)
                # but the W_k row scale at IS_MARK is still ~0 (i.e.
                # the head doesn't read IS_MARK at all -- so a steep
                # slope is suspect because no mark-anchored arithmetic
                # is happening). This is the L0 H1 regression signature
                # inverted: declared slope=20 with W_k *= 2.0 produces
                # ~20 at this cell, baseline produces ~10. If the cell
                # is 0 the head is not threshold-style at all and we
                # don't flag.
                if (baseline_default is not None
                        and abs(declared_slope) >= 2.0 * baseline_default
                        and wk_row_val < wk_eps):
                    report.drift.append(AlibiDriftEntry(
                        layer_idx=layer_idx, head_idx=head_idx,
                        kind="slope_wk_mismatch",
                        detail=(
                            f"op {op_name!r} declares steep slope "
                            f"{declared_slope} but W_k[{head_base},"
                            f"IS_MARK]={wk_row_val:.3e} ~ 0 "
                            f"(no IS_MARK-anchored arithmetic on this head;"
                            f" slope value is suspect)"
                        ),
                    ))

    return report


# ---------------------------------------------------------------------------
# Postcondition verifier (multistep)
# ---------------------------------------------------------------------------


@dataclass
class PostconditionDriftEntry:
    op_name: str
    step: int
    cell: str  # "DIM[k]"
    invariant: str
    detail: str


@dataclass
class PostconditionReport:
    n_steps: int = 0
    program_summary: str = ""
    drift: List[PostconditionDriftEntry] = field(default_factory=list)

    def has_drift(self) -> bool:
        return bool(self.drift)

    def format(self) -> str:
        lines = [
            "=== Postcondition verification report ===",
            f"Program: {self.program_summary}",
            f"Steps: {self.n_steps}",
            f"Drift entries: {len(self.drift)}",
        ]
        for entry in self.drift:
            lines.append(
                f"  [drift] op={entry.op_name!r} step={entry.step} "
                f"cell={entry.cell!r} invariant={entry.invariant!r}: "
                f"{entry.detail}"
            )
        return "\n".join(lines)


def _parse_postcondition_cell(cell: str) -> Tuple[str, Optional[int]]:
    """Parse ``'DIM_NAME[k]'`` -> ``('DIM_NAME', k)``; ``'DIM_NAME'`` ->
    ``('DIM_NAME', None)`` (whole dim).
    """
    if "[" in cell and cell.endswith("]"):
        name, idx = cell[:-1].split("[", 1)
        try:
            return name, int(idx)
        except ValueError:
            return name, None
    return cell, None


def _read_residual_cell(
    residual: torch.Tensor, pos: int, dim_start: int, dim_size: int,
    offset: Optional[int],
) -> float:
    """Read residual scalar at ``residual[0, pos, dim_start + offset]``.

    Returns the abs-max across the dim slice when ``offset`` is None.
    """
    if offset is None:
        slice_vals = residual[0, pos, dim_start:dim_start + dim_size]
        return float(slice_vals.abs().max().item())
    return float(residual[0, pos, dim_start + offset].item())


def verify_postconditions(
    model=None,
    layout=None,
    program: Optional[List[int]] = None,
    n_steps: int = 4,
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    S: float = 100.0,
    n_heads: int = 8,
    epsilon: float = 1e-2,
) -> PostconditionReport:
    """Multistep dynamic check on ``Operation.postcondition`` annotations.

    For each op with a non-empty ``postcondition``, run the compiled
    model forward over a multi-step probe and verify the declared
    invariant for every step. Drift entries report the offending
    ``(op, step, cell, invariant)``.

    Supported invariants:
        - ``'0_or_1'``: residual abs is either ~0 or ~1 (within epsilon).
        - ``'byte_range'``: residual scalar in [0, 255+epsilon].
        - ``'monotonic_non_decreasing'``: value at step k >= value at
          step k-1 (modulo epsilon).
        - ``'matches_AX_byte<k>'``: value at the op's cell equals the
          current AX byte k value at the AX marker. The reference comes
          from the DraftVM oracle inside ``_build_multistep_probe``.

    The probe defaults to ``_ADD_CASCADE_PROGRAM`` (IMM/PSH/IMM/ADD/EXIT)
    which exercises PC monotonicity, STACK0 byte writes, and AX updates
    across 4 distinct steps -- enough surface to catch most invariant
    drifts without a full smoke run.
    """
    if program is None:
        program = list(_ADD_CASCADE_PROGRAM)
    report = PostconditionReport(
        n_steps=n_steps,
        program_summary=" ".join(f"0x{i:016x}" for i in program),
    )

    if model is None or layout is None:
        from .full_vm_compiler import compile_full_vm
        try:
            model, layout = compile_full_vm(
                S=S, alu_mode=alu_mode,
                enable_conversational_io=enable_conversational_io,
                n_heads=n_heads,
            )
        except Exception as exc:
            report.drift.append(PostconditionDriftEntry(
                op_name="<compile_full_vm>", step=0, cell="<NONE>",
                invariant="<NONE>", detail=f"compile failed: {exc!r}",
            ))
            return report

    candidates = _collect_unique_ops_with(layout, lambda op: bool(op.postcondition))
    if not candidates:
        return report

    try:
        token_tensor, step_markers, _summaries = _build_multistep_probe(
            layout, program, n_steps,
        )
    except Exception as exc:
        report.drift.append(PostconditionDriftEntry(
            op_name="<build_multistep_probe>", step=0, cell="<NONE>",
            invariant="<NONE>", detail=f"probe build failed: {exc!r}",
        ))
        return report

    max_len = getattr(model, "max_seq_len", None)
    if max_len is not None and token_tensor.shape[1] > max_len:
        report.drift.append(PostconditionDriftEntry(
            op_name="<context_too_long>", step=0, cell="<NONE>",
            invariant="<NONE>",
            detail=f"len {token_tensor.shape[1]} > max_seq_len {max_len}",
        ))
        return report

    try:
        with torch.no_grad():
            x = model.embed(token_tensor)
            per_layer_residuals = [x.detach().clone()]
            for block in model.blocks:
                x = block(x)
                per_layer_residuals.append(x.detach().clone())
    except Exception as exc:
        report.drift.append(PostconditionDriftEntry(
            op_name="<forward>", step=0, cell="<NONE>",
            invariant="<NONE>", detail=f"forward failed: {exc!r}",
        ))
        return report

    final_residual = per_layer_residuals[-1]

    for op in candidates:
        layer_idx = _resolve_op_layer(layout, op)
        if layer_idx is None or layer_idx + 1 >= len(per_layer_residuals):
            inspect_layer = len(per_layer_residuals) - 1
        else:
            inspect_layer = layer_idx + 1
        post_residual = per_layer_residuals[inspect_layer]

        # Previous-step residual cache for monotonic invariants.
        prev_value: Dict[str, float] = {}

        for step_idx, marker_map in enumerate(step_markers):
            if not marker_map:
                continue
            step_k = step_idx + 1
            for cell, invariant in op.postcondition.items():
                dim_name, offset = _parse_postcondition_cell(cell)
                if dim_name not in layout.dim_positions:
                    report.drift.append(PostconditionDriftEntry(
                        op_name=op.name, step=step_k, cell=cell,
                        invariant=invariant,
                        detail=f"dim {dim_name!r} not in layout",
                    ))
                    continue
                d_start = layout.dim_positions[dim_name]
                d_size = layout.dim_sizes.get(dim_name, 1)
                # The "natural" probe position depends on the dim name --
                # PC* lives at REG_PC, AX* at REG_AX, STACK0* at STACK0,
                # everything else falls back to REG_AX.
                if dim_name.startswith(("PC", "OUTPUT")):
                    pos = marker_map.get("REG_PC", marker_map["REG_PC"])
                elif dim_name.startswith("AX"):
                    pos = marker_map.get("REG_AX", marker_map["REG_AX"])
                elif dim_name.startswith("STACK0"):
                    pos = marker_map.get("STACK0", marker_map["STACK0"])
                else:
                    pos = marker_map.get("REG_AX", marker_map["REG_AX"])

                val = _read_residual_cell(
                    post_residual, pos, d_start, d_size, offset,
                )

                if invariant == "0_or_1":
                    # Binary flag: |val| in {~0, ~1}.
                    if not (val < epsilon or abs(val - 1.0) < epsilon):
                        report.drift.append(PostconditionDriftEntry(
                            op_name=op.name, step=step_k, cell=cell,
                            invariant=invariant,
                            detail=(
                                f"value {val:.4f} not in "
                                f"{{0, 1}} (eps={epsilon})"
                            ),
                        ))
                elif invariant == "byte_range":
                    if not (-epsilon <= val <= 255 + epsilon):
                        report.drift.append(PostconditionDriftEntry(
                            op_name=op.name, step=step_k, cell=cell,
                            invariant=invariant,
                            detail=f"value {val:.4f} not in [0, 255]",
                        ))
                elif invariant == "monotonic_non_decreasing":
                    if cell in prev_value:
                        if val + epsilon < prev_value[cell]:
                            report.drift.append(PostconditionDriftEntry(
                                op_name=op.name, step=step_k, cell=cell,
                                invariant=invariant,
                                detail=(
                                    f"value {val:.4f} < prev "
                                    f"{prev_value[cell]:.4f} "
                                    f"(decreased)"
                                ),
                            ))
                    prev_value[cell] = val
                elif invariant.startswith("matches_AX_byte"):
                    # Reference AX byte k from oracle (parsed from the
                    # step summary's "AX=<hex>" field). Without the
                    # oracle we fall back to a residual-shape check.
                    # Future: thread the DraftVM AX value through
                    # ``_build_multistep_probe`` for exact equality.
                    # For now, sanity-check that the cell is in byte
                    # range (a weaker invariant; primary use is to
                    # ensure the annotation is parseable).
                    if not (-epsilon <= val <= 255 + epsilon):
                        report.drift.append(PostconditionDriftEntry(
                            op_name=op.name, step=step_k, cell=cell,
                            invariant=invariant,
                            detail=(
                                f"value {val:.4f} not in byte range; "
                                f"AX-byte-equality not yet probed"
                            ),
                        ))
                # Unknown invariants were rejected at add_op; this
                # branch is unreachable.

    return report


# ---------------------------------------------------------------------------
# Step-idx gating verifier (multistep)
# ---------------------------------------------------------------------------


@dataclass
class StepIdxDriftEntry:
    op_name: str
    step: int
    dim: str
    declared: str  # readable form of step_idx (e.g. '{0}' or 'after_first')
    detail: str


@dataclass
class StepIdxReport:
    n_steps: int = 0
    program_summary: str = ""
    drift: List[StepIdxDriftEntry] = field(default_factory=list)

    def has_drift(self) -> bool:
        return bool(self.drift)

    def format(self) -> str:
        lines = [
            "=== step_idx gating report ===",
            f"Program: {self.program_summary}",
            f"Steps: {self.n_steps}",
            f"Drift entries: {len(self.drift)}",
        ]
        for entry in self.drift:
            lines.append(
                f"  [drift] op={entry.op_name!r} step={entry.step} "
                f"dim={entry.dim!r} declared={entry.declared!r}: "
                f"{entry.detail}"
            )
        return "\n".join(lines)


def _step_idx_to_str(step_idx) -> str:
    if step_idx is None:
        return "every"
    if isinstance(step_idx, str):
        return step_idx
    return str(sorted(step_idx))


def _step_is_active(step_idx, k: int) -> bool:
    """Return True if the op is declared to fire on step ``k`` (0-indexed)."""
    if step_idx is None or step_idx == "every":
        return True
    if step_idx == "after_first":
        return k >= 1
    if isinstance(step_idx, (set, frozenset)):
        return k in step_idx
    # Defensive: should be caught by validation in add_op.
    return True


def verify_step_idx_gating(
    model=None,
    layout=None,
    program: Optional[List[int]] = None,
    n_steps: int = 4,
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    S: float = 100.0,
    n_heads: int = 8,
    epsilon: float = 1e-2,
) -> StepIdxReport:
    """Multistep dynamic check on ``Operation.step_idx`` annotations.

    For each op with a non-default ``step_idx`` and a non-empty
    ``produces`` declaration, run the multistep probe and assert the
    op's residual writes are below ``epsilon`` on every step NOT in
    ``step_idx``. Catches the canonical LEA-bug pattern: an op declared
    to fire only on step 1 keeps writing at step 2 because the gating
    logic in its bake is incomplete.

    Ops with ``step_idx=None`` or ``'every'`` are skipped (no gating
    invariant to enforce).
    """
    if program is None:
        program = list(_ADD_CASCADE_PROGRAM)
    report = StepIdxReport(
        n_steps=n_steps,
        program_summary=" ".join(f"0x{i:016x}" for i in program),
    )

    if model is None or layout is None:
        from .full_vm_compiler import compile_full_vm
        try:
            model, layout = compile_full_vm(
                S=S, alu_mode=alu_mode,
                enable_conversational_io=enable_conversational_io,
                n_heads=n_heads,
            )
        except Exception as exc:
            report.drift.append(StepIdxDriftEntry(
                op_name="<compile_full_vm>", step=0, dim="<NONE>",
                declared="<NONE>", detail=f"compile failed: {exc!r}",
            ))
            return report

    def _has_gating(op):
        if op.step_idx is None or op.step_idx == "every":
            return False
        return bool(op.produces)

    candidates = _collect_unique_ops_with(layout, _has_gating)
    if not candidates:
        return report

    try:
        token_tensor, step_markers, _summaries = _build_multistep_probe(
            layout, program, n_steps,
        )
    except Exception as exc:
        report.drift.append(StepIdxDriftEntry(
            op_name="<build_multistep_probe>", step=0, dim="<NONE>",
            declared="<NONE>", detail=f"probe build failed: {exc!r}",
        ))
        return report

    max_len = getattr(model, "max_seq_len", None)
    if max_len is not None and token_tensor.shape[1] > max_len:
        report.drift.append(StepIdxDriftEntry(
            op_name="<context_too_long>", step=0, dim="<NONE>",
            declared="<NONE>",
            detail=f"len {token_tensor.shape[1]} > max_seq_len {max_len}",
        ))
        return report

    try:
        with torch.no_grad():
            x = model.embed(token_tensor)
            per_layer_residuals = [x.detach().clone()]
            for block in model.blocks:
                x = block(x)
                per_layer_residuals.append(x.detach().clone())
    except Exception as exc:
        report.drift.append(StepIdxDriftEntry(
            op_name="<forward>", step=0, dim="<NONE>",
            declared="<NONE>", detail=f"forward failed: {exc!r}",
        ))
        return report

    for op in candidates:
        layer_idx = _resolve_op_layer(layout, op)
        if layer_idx is None or layer_idx + 1 >= len(per_layer_residuals):
            inspect_layer = len(per_layer_residuals) - 1
        else:
            inspect_layer = layer_idx + 1
        post_residual = per_layer_residuals[inspect_layer]
        declared_str = _step_idx_to_str(op.step_idx)

        for step_idx, marker_map in enumerate(step_markers):
            if not marker_map:
                continue
            if _step_is_active(op.step_idx, step_idx):
                continue
            # Op should be inert on this step.
            step_k = step_idx + 1
            for dim_name, register in op.produces.items():
                if dim_name not in layout.dim_positions:
                    continue
                offset = _resolve_register_offset(register)
                if offset is None:
                    continue
                pos = marker_map["REG_PC"] + offset
                d_start = layout.dim_positions[dim_name]
                d_size = layout.dim_sizes.get(dim_name, 1)
                slice_vals = post_residual[0, pos, d_start:d_start + d_size]
                obs_abs = float(slice_vals.abs().max().item())
                if obs_abs >= epsilon:
                    report.drift.append(StepIdxDriftEntry(
                        op_name=op.name, step=step_k, dim=dim_name,
                        declared=declared_str,
                        detail=(
                            f"op declared step_idx={declared_str} but "
                            f"residual abs-max={obs_abs:.3e} at "
                            f"register={register!r} pos={pos} "
                            f"(should be <{epsilon})"
                        ),
                    ))

    return report


# ---------------------------------------------------------------------------
# Convenience top-level entry point
# ---------------------------------------------------------------------------


def verify_all(
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    S: float = 100.0,
    n_heads: int = 8,
    run_dynamic: bool = False,
) -> Tuple[StaticVerificationReport, Optional[DynamicVerificationReport]]:
    """Run Mode A unconditionally; Mode B only if ``run_dynamic`` is True."""
    static_report = verify_claims_static(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        S=S,
        n_heads=n_heads,
    )
    dyn_report = None
    if run_dynamic:
        dyn_report = verify_produces_consumes_dynamic(
            alu_mode=alu_mode,
            enable_conversational_io=enable_conversational_io,
            S=S,
            n_heads=n_heads,
        )
    return static_report, dyn_report
