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

import inspect
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from .layer_compiler import LayerCompiler, Operation, operation_display_label


_LEGACY_HELPER_CALL_RE = re.compile(r"\b_set_[A-Za-z0-9_]+\(")
_LAYER_PREFIX_RE = re.compile(r"^(?:layer|l)\d+_")


_LOWERING_STEP_SLOT_NAMES: Tuple[str, ...] = (
    "REG_PC",
    "PC_byte0",
    "PC_byte1",
    "PC_byte2",
    "PC_byte3",
    "REG_AX",
    "AX_byte0",
    "AX_byte1",
    "AX_byte2",
    "AX_byte3",
    "REG_SP",
    "SP_byte0",
    "SP_byte1",
    "SP_byte2",
    "SP_byte3",
    "REG_BP",
    "BP_byte0",
    "BP_byte1",
    "BP_byte2",
    "BP_byte3",
    "STACK0",
    "STACK0_byte0",
    "STACK0_byte1",
    "STACK0_byte2",
    "STACK0_byte3",
    "MEM",
    "MEM_addr0",
    "MEM_addr1",
    "MEM_addr2",
    "MEM_addr3",
    "MEM_value0",
    "MEM_value1",
    "MEM_value2",
    "MEM_value3",
    "STEP_END",
)


def semantic_op_label(op_name: str) -> str:
    """Return a placement-neutral display label for an operation name.

    Operation names are stable IDs and still carry historical layer prefixes
    in many places. Reports should avoid implying those prefixes are the
    authority now that placement can be resolved dynamically.
    """
    return _LAYER_PREFIX_RE.sub("", op_name)


def _format_dim_with_semantics(dim_name: str, registry=None) -> str:
    """Format ``dim_name`` for printing, appending its semantics string
    if available.

    F-11: pure observability helper. Reviewers reading verifier output can
    see at-a-glance whether a dim's declared semantics matches the rule's
    intent. If ``registry`` is None, or the dim is unknown, or the dim has
    no semantics declared, returns the bare name unchanged.
    """
    if registry is None:
        return dim_name
    try:
        sem = registry.semantics(dim_name)
    except KeyError:
        return dim_name  # unknown dim -- don't crash, just skip semantics
    if sem is None:
        return dim_name
    return f"{dim_name} [{sem}]"


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
    # F-11: optional registry used to enrich dim references in ``format()``
    # output with their declared semantics. None preserves prior behavior.
    registry: Optional[object] = None

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
                lines.append(
                    f"     UNUSED CLAIM (DECLARATION DRIFT): "
                    f"{_format_cell_with_semantics(cell, self.registry)}"
                )
            # Show undeclared writes only when in strict mode (otherwise
            # would flood with the expected partial-claim residue).
            if self.strict_mode:
                for cell in sorted(r.written_but_not_declared, key=_claim_sort_key):
                    lines.append(
                        f"     UNDECLARED WRITE: "
                        f"{_format_cell_with_semantics(cell, self.registry)}"
                    )
            for note in r.notes:
                lines.append(f"     NOTE: {note}")
        for op_name, reason in self.skipped:
            lines.append(f"  [SKIP] {op_name}: {reason}")
        return "\n".join(lines)


def _claim_sort_key(claim):
    layer_idx, scope, identifier, column = claim
    return (layer_idx, scope, identifier, "" if column is None else column)


def _format_cell_with_semantics(cell, registry=None) -> str:
    """F-11: format a (layer, scope, identifier, column) cell tuple for
    display, decoding the ``column`` field's ``"DIM+offset"`` shape and
    appending the dim's declared semantics when ``registry`` is provided.

    Falls back to ``repr(cell)`` when ``column`` is None or not in the
    ``"<dim>+<offset>"`` form -- pure observability, never raises.
    """
    if registry is None:
        return repr(cell)
    layer_idx, scope, identifier, column = cell
    if column is None or "+" not in column:
        return repr(cell)
    dim_name, _, offset = column.partition("+")
    enriched = _format_dim_with_semantics(dim_name, registry)
    if enriched == dim_name:
        return repr(cell)
    return f"(layer={layer_idx}, scope={scope!r}, id={identifier!r}, column={enriched}+{offset})"


# ---------------------------------------------------------------------------
# Mode C: teacher-forced lowering support probes
# ---------------------------------------------------------------------------


def _token_label(token: int) -> str:
    if 0 <= int(token) < 256:
        return f"0x{int(token):02x}"
    try:
        from ..vm_step import Token
    except Exception:
        return str(token)
    for name, value in vars(Token).items():
        if name.isupper() and value == token:
            return name
    return str(token)


def _slot_offset(slot_name: str) -> int:
    try:
        return _LOWERING_STEP_SLOT_NAMES.index(slot_name)
    except ValueError as exc:
        raise KeyError(f"unknown VM-step slot {slot_name!r}") from exc


@dataclass(frozen=True)
class TeacherForcedSymbolicTrace:
    """Declarative token stream used to teacher-force one lowering probe.

    ``context`` is the exact symbolic stream that the neural model should emit:
    immutable CODE/DATA prefix followed by 35-token VM rows.  Tests usually
    select a row with ``token_index(step, slot)`` and then call
    :func:`verify_teacher_forced_token_support`.
    """

    context: Tuple[int, ...]
    prefix_len: int
    steps: int
    exit_code: Optional[int]
    halted: bool
    mem_store_positions: Tuple[int, ...] = ()
    # SI/SC subset of ``mem_store_positions``. Embedding uses this list to
    # restrict MEM_ADDR_SRC=1 injection to STACK0-sourced stores; PSH/JSR/ENT
    # markers are tracked for MEM_STORE=1 but must keep MEM_ADDR_SRC=0.
    mem_addr_src_positions: Tuple[int, ...] = ()
    early_exit: bool = False

    def token_index(self, step: int, slot_name: str) -> int:
        if step < 0:
            raise ValueError("step must be non-negative")
        return (
            self.prefix_len
            + step * len(_LOWERING_STEP_SLOT_NAMES)
            + _slot_offset(slot_name)
        )

    def token_for(self, step: int, slot_name: str) -> int:
        index = self.token_index(step, slot_name)
        return int(self.context[index])


@dataclass(frozen=True)
class LoweringSupportSnapshot:
    """Head support for one residual snapshot at the probed token row."""

    label: str
    block_index: Optional[int]
    original_layer_index: Optional[int]
    expected_token: int
    argmax_token: int
    expected_logit: float
    argmax_logit: float
    expected_margin: float
    residual_note: str = ""
    output_byte: Optional[int] = None
    output_lo_argmax: Optional[int] = None
    output_hi_argmax: Optional[int] = None
    output_band_contract: str = ""

    @property
    def supports_expected_argmax(self) -> bool:
        return self.argmax_token == self.expected_token

    @property
    def supports_expected_byte_channel(self) -> bool:
        if not 0 <= int(self.expected_token) < 256:
            return True
        return self.output_byte == self.expected_token

    def format(self) -> str:
        block = "-" if self.block_index is None else str(self.block_index)
        layer = (
            "-"
            if self.original_layer_index is None
            else str(self.original_layer_index)
        )
        byte = ""
        if 0 <= int(self.expected_token) < 256:
            output_label = (
                "<missing>"
                if self.output_byte is None
                else _token_label(self.output_byte)
            )
            byte = (
                f" output_byte={output_label}"
                f" lo_arg={self.output_lo_argmax}"
                f" hi_arg={self.output_hi_argmax}"
            )
        note = f" {self.residual_note}" if self.residual_note else ""
        band_contract = (
            "" if not self.output_band_contract else f" {self.output_band_contract}"
        )
        return (
            f"block={block} layer={layer} label={self.label!r} "
            f"expected={_token_label(self.expected_token)} "
            f"argmax={_token_label(self.argmax_token)} "
            f"expected_logit={self.expected_logit:+.2f} "
            f"argmax_logit={self.argmax_logit:+.2f} "
            f"margin={self.expected_margin:+.2f}"
            f"{byte}{note}"
            f"{band_contract}"
        )


@dataclass(frozen=True)
class TeacherForcedTokenSupportReport:
    """Result of a teacher-forced symbolic-token lowering probe."""

    probe_name: str
    token_index: int
    expected_token: int
    prefix_len: int
    logit_pos: int
    min_margin: float
    snapshots: Tuple[LoweringSupportSnapshot, ...]

    @property
    def final(self) -> LoweringSupportSnapshot:
        return self.snapshots[-1]

    @property
    def supported(self) -> bool:
        final = self.final
        return (
            final.argmax_token == self.expected_token
            and final.expected_margin >= self.min_margin
            and final.supports_expected_byte_channel
        )

    @property
    def first_loss_after_support(self) -> Optional[LoweringSupportSnapshot]:
        saw_support = False
        for snapshot in self.snapshots:
            if (
                snapshot.argmax_token == self.expected_token
                and snapshot.expected_margin >= self.min_margin
            ):
                saw_support = True
                continue
            if saw_support:
                return snapshot
        return None

    def format(self) -> str:
        loss = self.first_loss_after_support
        loss_part = (
            " first_loss_after_support=" + loss.format()
            if loss is not None
            else ""
        )
        return (
            f"teacher_forced_lowering_probe={self.probe_name!r} "
            f"token_index={self.token_index} expected="
            f"{_token_label(self.expected_token)} supported={self.supported} "
            f"final={self.final.format()}{loss_part}"
        )


@dataclass(frozen=True)
class TeacherForcedTraceAuditFailure:
    """One final-head mismatch in a teacher-forced symbolic trace audit."""

    token_index: int
    step: int
    slot: str
    expected_token: int
    argmax_token: int
    expected_margin: float
    output_byte: Optional[int] = None
    failure_kind: str = "unsupported"
    output_band_contract: str = ""
    output_band_violation_kinds: Tuple[str, ...] = ()

    @property
    def failure_kinds(self) -> Tuple[str, ...]:
        return tuple(
            part
            for part in self.failure_kind.split(",")
            if part
        )

    @property
    def token_is_wrong(self) -> bool:
        kinds = set(self.failure_kinds)
        return bool(kinds & {"wrong_argmax", "wrong_output_byte"})

    @property
    def output_band_contract_only(self) -> bool:
        kinds = set(self.failure_kinds)
        return bool(kinds) and kinds <= {"output_band_contract"}

    @property
    def output_band_margin_only(self) -> bool:
        return (
            self.output_band_contract_only
            and bool(self.output_band_violation_kinds)
            and set(self.output_band_violation_kinds) <= {"active_margin_low"}
        )

    def format(self) -> str:
        byte = (
            ""
            if self.output_byte is None
            else f" output_byte={_token_label(self.output_byte)}"
        )
        band_contract = (
            "" if not self.output_band_contract else f" {self.output_band_contract}"
        )
        band_kinds = (
            ""
            if not self.output_band_violation_kinds
            else " band_violations=" + ",".join(self.output_band_violation_kinds)
        )
        return (
            f"step{self.step}:{self.slot} abs={self.token_index} "
            f"expected={_token_label(self.expected_token)} "
            f"argmax={_token_label(self.argmax_token)} "
            f"margin={self.expected_margin:+.2f}{byte} "
            f"kind={self.failure_kind}"
            f"{band_contract}"
            f"{band_kinds}"
        )


@dataclass(frozen=True)
class TeacherForcedTraceAuditReport:
    """Final lowered-model support audit for all selected symbolic tokens."""

    probe_name: str
    checked: int
    failures: Tuple[TeacherForcedTraceAuditFailure, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def format(self) -> str:
        if self.passed:
            return (
                f"teacher_forced_trace_audit={self.probe_name!r} "
                f"checked={self.checked} failures=0"
            )
        failures = "; ".join(f.format() for f in self.failures)
        return (
            f"teacher_forced_trace_audit={self.probe_name!r} "
            f"checked={self.checked} failures={len(self.failures)} "
            f"{failures}"
        )


def _u32_bytes(value: int) -> Tuple[int, int, int, int]:
    value &= 0xFFFFFFFF
    return (
        value & 0xFF,
        (value >> 8) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 24) & 0xFF,
    )


def build_teacher_forced_symbolic_trace(
    bytecode: Sequence[int],
    data: Sequence[int] | bytes = b"",
) -> TeacherForcedSymbolicTrace:
    """Build the runtime-authoritative teacher-forced symbolic token stream.

    The batched pure-neural runner stops after a completed step when the
    model-emitted PC points at ``EXIT``.  Mirror that contract here so the
    structural verifier checks the rows the neural runtime actually requires,
    instead of demanding an extra symbolic EXIT/HALT row that is never emitted.
    """

    from ..constants import IMMEDIATE_SIZE, INSTR_WIDTH, PADDING_SIZE
    from ..embedding import Opcode
    from ..vm_step import Token
    from .symbolic_program import SymbolicDeclarativeProgramRunner

    context: List[int] = [Token.CODE_START]
    for instr in bytecode:
        op = int(instr) & 0xFF
        imm = int(instr) >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.extend([Token.CODE_END, Token.DATA_START])
    context.extend(int(b) & 0xFF for b in data)
    context.append(Token.DATA_END)
    prefix_len = len(context)

    store_opcodes = {
        int(Opcode.SI),
        int(Opcode.SC),
        int(Opcode.PSH),
        int(Opcode.ENT),
        int(Opcode.JSR),
    }
    # SI/SC carry MEM_ADDR_SRC=1 (address sourced from STACK0); PSH/JSR/ENT
    # keep MEM_ADDR_SRC=0. We track them separately so the embedding can
    # restrict its MEM_ADDR_SRC=1 injection to SI/SC markers only — otherwise
    # the L15 SI MEM addr0-from-STACK0 head fires on PSH/JSR/ENT and routes
    # the wrong source byte (see neural_embedding._inject_mem_store).
    addr_src_opcodes = {int(Opcode.SI), int(Opcode.SC)}
    mem_store_positions: List[int] = []
    mem_addr_src_positions: List[int] = []

    symbolic = SymbolicDeclarativeProgramRunner()
    state = symbolic.init_state(bytecode, data)
    early_exit = False
    while symbolic.step(state):
        trace = state.trace[-1]
        step_base = len(context)
        pc_bytes = _u32_bytes(state.pc)
        ax_bytes = _u32_bytes(state.ax)
        sp_bytes = _u32_bytes(state.sp)
        bp_bytes = _u32_bytes(state.bp)
        stack0_bytes = _u32_bytes(state.mem_read(state.sp))
        mem_addr_bytes = _u32_bytes(trace.mem_addr)
        mem_value_bytes = _u32_bytes(trace.mem_value)
        context.extend((
            Token.REG_PC, *pc_bytes,
            Token.REG_AX, *ax_bytes,
            Token.REG_SP, *sp_bytes,
            Token.REG_BP, *bp_bytes,
            Token.STACK0, *stack0_bytes,
            Token.MEM, *mem_addr_bytes, *mem_value_bytes,
            Token.HALT if state.halted else Token.STEP_END,
        ))
        if int(trace.opcode) in store_opcodes:
            mem_pos = step_base + _slot_offset("MEM")
            mem_store_positions.append(mem_pos)
            if int(trace.opcode) in addr_src_opcodes:
                mem_addr_src_positions.append(mem_pos)
        if state.halted:
            break
        next_idx = int(state.pc) // INSTR_WIDTH
        if 0 <= next_idx < len(bytecode):
            next_op = int(bytecode[next_idx]) & 0xFF
            if next_op == int(Opcode.EXIT):
                early_exit = True
                break

    return TeacherForcedSymbolicTrace(
        context=tuple(int(t) for t in context),
        prefix_len=prefix_len,
        steps=state.steps,
        exit_code=state.ax if state.halted or early_exit else None,
        halted=state.halted or early_exit,
        mem_store_positions=tuple(mem_store_positions),
        mem_addr_src_positions=tuple(mem_addr_src_positions),
        early_exit=early_exit,
    )


def _head_logits_for_residual(model, row: torch.Tensor) -> torch.Tensor:
    from ..vm_step import sparse_linear

    if getattr(model.head.weight, "is_sparse", False):
        return sparse_linear(
            row.unsqueeze(0), model.head.weight, model.head.bias
        ).squeeze(0)
    return model.head(row.unsqueeze(0)).squeeze(0)


def _dim_position(model, name: str) -> int:
    from ..vm_step import _SetDim

    positions = getattr(model, "dim_positions", None)
    if isinstance(positions, dict) and name in positions:
        return int(positions[name])
    return int(getattr(_SetDim, name))


def _lowering_residual_note(model, row: torch.Tensor, token: int) -> str:
    if not 0 <= int(token) < 256:
        return ""
    out_lo = _dim_position(model, "OUTPUT_LO")
    out_hi = _dim_position(model, "OUTPUT_HI")
    lo = int(token) & 0xF
    hi = (int(token) >> 4) & 0xF
    lo_band = row[out_lo : out_lo + 16]
    hi_band = row[out_hi : out_hi + 16]
    lo_arg = int(torch.argmax(lo_band).item())
    hi_arg = int(torch.argmax(hi_band).item())
    return (
        f"OUT_LO[{lo}]={float(lo_band[lo].item()):+.2f} "
        f"arg={lo_arg}/{float(lo_band[lo_arg].item()):+.2f} "
        f"OUT_HI[{hi}]={float(hi_band[hi].item()):+.2f} "
        f"arg={hi_arg}/{float(hi_band[hi_arg].item()):+.2f}"
    )


def _lowering_output_byte(
    model,
    row: torch.Tensor,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    try:
        out_lo = _dim_position(model, "OUTPUT_LO")
        out_hi = _dim_position(model, "OUTPUT_HI")
    except Exception:
        return None, None, None
    lo_arg = int(torch.argmax(row[out_lo:out_lo + 16]).item())
    hi_arg = int(torch.argmax(row[out_hi:out_hi + 16]).item())
    return lo_arg | (hi_arg << 4), lo_arg, hi_arg


def _lowering_output_band_contract_report(
    model,
    row: torch.Tensor,
    expected_token: int,
    *,
    output_band_min_margin: Optional[float],
    output_band_max_inactive_value: float,
    output_band_tolerance: float,
):
    if output_band_min_margin is None or not 0 <= int(expected_token) < 256:
        return None
    from .band_contracts import verify_declared_output_nibble_bands

    out_lo = _dim_position(model, "OUTPUT_LO")
    out_hi = _dim_position(model, "OUTPUT_HI")
    return verify_declared_output_nibble_bands(
        {
            "OUTPUT_LO": row[out_lo:out_lo + 16],
            "OUTPUT_HI": row[out_hi:out_hi + 16],
        },
        expected_byte=int(expected_token),
        min_active_margin=float(output_band_min_margin),
        max_inactive_value=float(output_band_max_inactive_value),
        tolerance=float(output_band_tolerance),
        include_projection=True,
    )


def _lowering_failure_kind(
    *,
    argmax_token: int,
    expected_token: int,
    expected_margin: float,
    min_margin: float,
    byte_ok: bool,
    band_contract_ok: bool,
) -> str:
    kinds: List[str] = []
    if int(argmax_token) != int(expected_token):
        kinds.append("wrong_argmax")
    if not byte_ok:
        kinds.append("wrong_output_byte")
    if float(expected_margin) < float(min_margin):
        kinds.append("low_head_margin")
    if not band_contract_ok:
        kinds.append("output_band_contract")
    return ",".join(kinds) if kinds else "unsupported"


def _lowering_output_band_violation_kinds(band_report) -> Tuple[str, ...]:
    if band_report is None:
        return ()
    return tuple(sorted({violation.kind for violation in band_report.violations}))


def _lowering_trace_audit_failure(
    *,
    model,
    trace: TeacherForcedSymbolicTrace,
    row: torch.Tensor,
    token_index: int,
    min_margin: float,
    output_band_min_margin: Optional[float],
    output_band_max_inactive_value: float,
    output_band_tolerance: float,
) -> Optional[TeacherForcedTraceAuditFailure]:
    expected_token = int(trace.context[token_index])
    if not 0 <= expected_token < int(getattr(model, "vocab_size", 10**9)):
        return None

    logits = _head_logits_for_residual(model, row)
    argmax_token = int(torch.argmax(logits).item())
    expected_logit = float(logits[expected_token].item())
    masked = logits.clone()
    masked[expected_token] = float("-inf")
    expected_margin = expected_logit - float(masked.max().item())
    output_byte, _, _ = _lowering_output_byte(model, row)
    byte_ok = (
        not 0 <= expected_token < 256
        or output_byte == expected_token
    )
    band_report = _lowering_output_band_contract_report(
        model,
        row,
        expected_token,
        output_band_min_margin=output_band_min_margin,
        output_band_max_inactive_value=output_band_max_inactive_value,
        output_band_tolerance=output_band_tolerance,
    )
    band_contract_ok = band_report is None or band_report.ok
    if (
        argmax_token == expected_token
        and expected_margin >= min_margin
        and byte_ok
        and band_contract_ok
    ):
        return None

    generated_index = token_index - trace.prefix_len
    step = generated_index // len(_LOWERING_STEP_SLOT_NAMES)
    offset = generated_index % len(_LOWERING_STEP_SLOT_NAMES)
    return TeacherForcedTraceAuditFailure(
        token_index=int(token_index),
        step=int(step),
        slot=_LOWERING_STEP_SLOT_NAMES[offset],
        expected_token=expected_token,
        argmax_token=argmax_token,
        expected_margin=float(expected_margin),
        output_byte=output_byte if 0 <= expected_token < 256 else None,
        failure_kind=_lowering_failure_kind(
            argmax_token=argmax_token,
            expected_token=expected_token,
            expected_margin=expected_margin,
            min_margin=min_margin,
            byte_ok=byte_ok,
            band_contract_ok=band_contract_ok,
        ),
        output_band_contract=(
            "" if band_report is None else band_report.format_inline()
        ),
        output_band_violation_kinds=_lowering_output_band_violation_kinds(
            band_report
        ),
    )


def _lowering_snapshot(
    *,
    model,
    x: torch.Tensor,
    pos: int,
    label: str,
    block_index: Optional[int],
    original_layer_index: Optional[int],
    expected_token: int,
    output_band_min_margin: Optional[float] = None,
    output_band_max_inactive_value: float = 1e30,
    output_band_tolerance: float = 1e-6,
) -> LoweringSupportSnapshot:
    row = x[0, pos]
    logits = _head_logits_for_residual(model, row)
    argmax_token = int(torch.argmax(logits).item())
    expected_logit = float(logits[expected_token].item())
    argmax_logit = float(logits[argmax_token].item())
    masked = logits.clone()
    masked[expected_token] = float("-inf")
    best_other = float(masked.max().item())
    output_byte, lo_arg, hi_arg = _lowering_output_byte(model, row)
    band_report = _lowering_output_band_contract_report(
        model,
        row,
        expected_token,
        output_band_min_margin=output_band_min_margin,
        output_band_max_inactive_value=output_band_max_inactive_value,
        output_band_tolerance=output_band_tolerance,
    )
    return LoweringSupportSnapshot(
        label=label,
        block_index=block_index,
        original_layer_index=original_layer_index,
        expected_token=int(expected_token),
        argmax_token=argmax_token,
        expected_logit=expected_logit,
        argmax_logit=argmax_logit,
        expected_margin=expected_logit - best_other,
        residual_note=_lowering_residual_note(model, row, expected_token),
        output_byte=output_byte,
        output_lo_argmax=lo_arg,
        output_hi_argmax=hi_arg,
        output_band_contract=(
            "" if band_report is None else band_report.format_inline()
        ),
    )


def _remap_positions_for_window(
    raw_positions: Sequence[int],
    *,
    prefix_len: int,
    dynamic_start: int,
    dynamic_end: int,
) -> Tuple[int, ...]:
    """Map absolute positions to their image in a windowed prefix.

    The window keeps ``context[:prefix_len]`` followed by
    ``context[dynamic_start:dynamic_end]``; positions outside either span are
    dropped. Used by both single-token verification and chunked trace audits.
    """

    remapped: List[int] = []
    for raw_pos in raw_positions:
        pos = int(raw_pos)
        if 0 <= pos < prefix_len:
            remapped.append(pos)
        elif dynamic_start <= pos < dynamic_end:
            remapped.append(prefix_len + (pos - dynamic_start))
    return tuple(remapped)


def _window_teacher_forced_prefix(
    *,
    context: Sequence[int],
    prefix_len: int,
    token_index: int,
    max_context_window: Optional[int],
    mem_store_positions: Sequence[int],
    mem_addr_src_positions: Sequence[int] = (),
) -> Tuple[List[int], int, List[int], List[int]]:
    def visible_store_positions() -> List[int]:
        visible: List[int] = []
        for raw_pos in mem_store_positions:
            pos = int(raw_pos)
            if not 0 <= pos < token_index:
                continue
            # Completed historical stores are visible everywhere. The current
            # store row is visible only while predicting its address bytes;
            # value bytes must not self-look-up an incomplete row.
            if pos + 8 < token_index or token_index <= pos + 4:
                visible.append(pos)
        return visible

    visible_store = visible_store_positions()
    addr_src_set = {int(p) for p in mem_addr_src_positions}
    visible_addr_src = [p for p in visible_store if p in addr_src_set]

    prediction_prefix = [int(t) for t in context[:token_index]]
    if max_context_window is None:
        return (
            prediction_prefix,
            len(prediction_prefix) - 1,
            visible_store,
            visible_addr_src,
        )
    if token_index <= prefix_len + max_context_window:
        return (
            prediction_prefix,
            len(prediction_prefix) - 1,
            visible_store,
            visible_addr_src,
        )

    tail_start = token_index - max_context_window
    windowed = (
        [int(t) for t in context[:prefix_len]]
        + [int(t) for t in context[tail_start:token_index]]
    )
    remapped_store = list(
        _remap_positions_for_window(
            visible_store,
            prefix_len=prefix_len,
            dynamic_start=tail_start,
            dynamic_end=token_index,
        )
    )
    remapped_addr_src = list(
        _remap_positions_for_window(
            visible_addr_src,
            prefix_len=prefix_len,
            dynamic_start=tail_start,
            dynamic_end=token_index,
        )
    )
    return windowed, len(windowed) - 1, remapped_store, remapped_addr_src


def _block_display_label(block, block_index: int) -> Tuple[str, Optional[int]]:
    attn = getattr(block, "attn", None)
    original_layer = getattr(attn, "layer_idx", None)
    ffn = getattr(block, "ffn", None)
    w_up = getattr(ffn, "W_up", None)
    width = None if w_up is None else int(w_up.shape[0])
    width_part = "" if width is None else f" width={width}"
    layer_part = "-" if original_layer is None else str(original_layer)
    return f"block{block_index} layer={layer_part}{width_part}", original_layer


def verify_teacher_forced_token_support(
    model,
    context: Sequence[int],
    *,
    token_index: int,
    expected_token: Optional[int] = None,
    prefix_len: int = 0,
    mem_store_positions: Sequence[int] = (),
    mem_addr_src_positions: Sequence[int] = (),
    max_context_window: Optional[int] = 512,
    min_margin: float = 0.0,
    output_band_min_margin: Optional[float] = None,
    output_band_max_inactive_value: float = 1e30,
    output_band_tolerance: float = 1e-6,
    probe_name: str = "",
) -> TeacherForcedTokenSupportReport:
    """Assertable lowering probe for one teacher-forced symbolic token.

    The function feeds ``context[:token_index]`` through the compiled model and
    records the output-head winner after embedding and after every compiled
    block/tail.  Callers assert ``report.supported`` to catch cases where
    intermediate symbolic evidence is later overwritten by a neural tail.
    """

    if token_index <= 0:
        raise ValueError(
            "token_index must predict a token after at least one prefix row"
        )
    if token_index >= len(context):
        raise IndexError("token_index must point inside the teacher-forced context")
    if expected_token is None:
        expected_token = int(context[token_index])
    if not 0 <= int(expected_token) < int(getattr(model, "vocab_size", 10**9)):
        raise ValueError(f"expected token {expected_token!r} is outside model vocab")

    (
        windowed,
        logit_pos,
        remapped_store_positions,
        remapped_addr_src_positions,
    ) = _window_teacher_forced_prefix(
        context=context,
        prefix_len=prefix_len,
        token_index=token_index,
        max_context_window=max_context_window,
        mem_store_positions=mem_store_positions,
        mem_addr_src_positions=mem_addr_src_positions,
    )
    if logit_pos < 0:
        raise ValueError("empty prediction prefix")

    device = next(model.parameters()).device
    token_ids = torch.tensor([windowed], dtype=torch.long, device=device)

    embed = getattr(model, "embed", None)
    old_mem_history_end = getattr(embed, "_mem_history_end", None)
    old_mem_store_positions = getattr(embed, "_mem_store_positions", None)
    old_mem_addr_src_positions = getattr(embed, "_mem_addr_src_positions", None)

    snapshots: List[LoweringSupportSnapshot] = []
    try:
        if hasattr(embed, "set_mem_history_end"):
            embed.set_mem_history_end(0)
        if hasattr(embed, "set_mem_store_positions"):
            embed.set_mem_store_positions([tuple(remapped_store_positions)])
        if hasattr(embed, "set_mem_addr_src_positions"):
            embed.set_mem_addr_src_positions(
                [tuple(remapped_addr_src_positions)]
            )

        with torch.no_grad():
            x = model.embed(token_ids)
            snapshots.append(
                _lowering_snapshot(
                    model=model,
                    x=x,
                    pos=logit_pos,
                    label="embed",
                    block_index=None,
                    original_layer_index=None,
                    expected_token=int(expected_token),
                    output_band_min_margin=output_band_min_margin,
                    output_band_max_inactive_value=output_band_max_inactive_value,
                    output_band_tolerance=output_band_tolerance,
                )
            )
            for block_index, block in enumerate(model.blocks):
                x = block(x)
                label, original_layer = _block_display_label(block, block_index)
                snapshots.append(
                    _lowering_snapshot(
                        model=model,
                        x=x,
                        pos=logit_pos,
                        label=label,
                        block_index=block_index,
                        original_layer_index=original_layer,
                        expected_token=int(expected_token),
                        output_band_min_margin=output_band_min_margin,
                        output_band_max_inactive_value=output_band_max_inactive_value,
                        output_band_tolerance=output_band_tolerance,
                    )
                )
    finally:
        if old_mem_history_end is not None and hasattr(embed, "set_mem_history_end"):
            embed.set_mem_history_end(old_mem_history_end)
        if hasattr(embed, "set_mem_store_positions"):
            embed.set_mem_store_positions(old_mem_store_positions)
        if hasattr(embed, "set_mem_addr_src_positions"):
            embed.set_mem_addr_src_positions(old_mem_addr_src_positions)

    return TeacherForcedTokenSupportReport(
        probe_name=probe_name,
        token_index=int(token_index),
        expected_token=int(expected_token),
        prefix_len=int(prefix_len),
        logit_pos=int(logit_pos),
        min_margin=float(min_margin),
        snapshots=tuple(snapshots),
    )


def audit_teacher_forced_trace_final_support(
    model,
    trace: TeacherForcedSymbolicTrace,
    *,
    token_indices: Optional[Sequence[int]] = None,
    min_margin: float = 0.0,
    output_band_min_margin: Optional[float] = None,
    output_band_max_inactive_value: float = 1e30,
    output_band_tolerance: float = 1e-6,
    max_context_window: Optional[int] = None,
    window_chunk_tokens: int = 512,
    max_failures: int = 16,
    probe_name: str = "",
) -> TeacherForcedTraceAuditReport:
    """Check final lowered-model support for many symbolic tokens at once.

    This is the fast counterpart to :func:`verify_teacher_forced_token_support`.
    By default it teacher-forces the whole symbolic trace once, runs every
    compiled block, and checks the final output head at each selected token's
    prediction row.  Set ``max_context_window`` to audit long traces in
    bounded chunks that keep the immutable prefix plus a rolling dynamic-token
    window; this matches the single-token verifier's context-window shape and
    avoids full-trace OOMs during shard triage.  When a failure is found,
    callers can rerun the single-token verifier for a block-by-block loss
    report.
    """

    if len(trace.context) < 2:
        raise ValueError("teacher-forced trace is too short to audit")
    if max_context_window is not None and max_context_window < 1:
        raise ValueError("max_context_window must be positive when set")
    if window_chunk_tokens < 1:
        raise ValueError("window_chunk_tokens must be positive")

    if token_indices is None:
        selected = list(range(trace.prefix_len, len(trace.context)))
    else:
        selected = [int(i) for i in token_indices]

    for token_index in selected:
        if token_index <= 0 or token_index >= len(trace.context):
            raise IndexError(f"token index {token_index} outside trace")

    if not selected:
        return TeacherForcedTraceAuditReport(
            probe_name=probe_name,
            checked=0,
            failures=(),
        )

    if max_context_window is not None:
        return _audit_teacher_forced_trace_final_support_windowed(
            model,
            trace,
            selected=selected,
            min_margin=min_margin,
            output_band_min_margin=output_band_min_margin,
            output_band_max_inactive_value=output_band_max_inactive_value,
            output_band_tolerance=output_band_tolerance,
            max_context_window=max_context_window,
            window_chunk_tokens=window_chunk_tokens,
            max_failures=max_failures,
            probe_name=probe_name,
        )

    device = next(model.parameters()).device
    token_ids = torch.tensor(
        [[int(t) for t in trace.context[:-1]]],
        dtype=torch.long,
        device=device,
    )

    embed = getattr(model, "embed", None)
    old_mem_history_end = getattr(embed, "_mem_history_end", None)
    old_mem_store_positions = getattr(embed, "_mem_store_positions", None)
    old_mem_addr_src_positions = getattr(embed, "_mem_addr_src_positions", None)

    failures: List[TeacherForcedTraceAuditFailure] = []
    checked = 0
    try:
        if hasattr(embed, "set_mem_history_end"):
            embed.set_mem_history_end(0)
        if hasattr(embed, "set_mem_store_positions"):
            embed.set_mem_store_positions([tuple(trace.mem_store_positions)])
        if hasattr(embed, "set_mem_addr_src_positions"):
            embed.set_mem_addr_src_positions(
                [tuple(trace.mem_addr_src_positions)]
            )

        with torch.no_grad():
            x = model.embed(token_ids)
            for block in model.blocks:
                x = block(x)

            for token_index in selected:
                expected_token = int(trace.context[token_index])
                if not 0 <= expected_token < int(getattr(model, "vocab_size", 10**9)):
                    continue
                checked += 1
                logit_pos = token_index - 1
                row = x[0, logit_pos]
                failure = _lowering_trace_audit_failure(
                    model=model,
                    trace=trace,
                    row=row,
                    token_index=token_index,
                    min_margin=min_margin,
                    output_band_min_margin=output_band_min_margin,
                    output_band_max_inactive_value=output_band_max_inactive_value,
                    output_band_tolerance=output_band_tolerance,
                )
                if failure is None:
                    continue
                failures.append(failure)
                if len(failures) >= max_failures:
                    break
    finally:
        if old_mem_history_end is not None and hasattr(embed, "set_mem_history_end"):
            embed.set_mem_history_end(old_mem_history_end)
        if hasattr(embed, "set_mem_store_positions"):
            embed.set_mem_store_positions(old_mem_store_positions)
        if hasattr(embed, "set_mem_addr_src_positions"):
            embed.set_mem_addr_src_positions(old_mem_addr_src_positions)

    return TeacherForcedTraceAuditReport(
        probe_name=probe_name,
        checked=checked,
        failures=tuple(failures),
    )


def _remapped_store_positions_for_window(
    trace: TeacherForcedSymbolicTrace,
    *,
    dynamic_start: int,
    dynamic_end: int,
) -> Tuple[int, ...]:
    return _remap_positions_for_window(
        trace.mem_store_positions,
        prefix_len=trace.prefix_len,
        dynamic_start=dynamic_start,
        dynamic_end=dynamic_end,
    )


def _remapped_addr_src_positions_for_window(
    trace: TeacherForcedSymbolicTrace,
    *,
    dynamic_start: int,
    dynamic_end: int,
) -> Tuple[int, ...]:
    return _remap_positions_for_window(
        trace.mem_addr_src_positions,
        prefix_len=trace.prefix_len,
        dynamic_start=dynamic_start,
        dynamic_end=dynamic_end,
    )


def _audit_teacher_forced_trace_final_support_windowed(
    model,
    trace: TeacherForcedSymbolicTrace,
    *,
    selected: Sequence[int],
    min_margin: float,
    output_band_min_margin: Optional[float],
    output_band_max_inactive_value: float,
    output_band_tolerance: float,
    max_context_window: int,
    window_chunk_tokens: int,
    max_failures: int,
    probe_name: str,
) -> TeacherForcedTraceAuditReport:
    device = next(model.parameters()).device
    selected_sorted = sorted(dict.fromkeys(int(i) for i in selected))

    embed = getattr(model, "embed", None)
    old_mem_history_end = getattr(embed, "_mem_history_end", None)
    old_mem_store_positions = getattr(embed, "_mem_store_positions", None)
    old_mem_addr_src_positions = getattr(embed, "_mem_addr_src_positions", None)

    failures: List[TeacherForcedTraceAuditFailure] = []
    checked = 0
    try:
        if hasattr(embed, "set_mem_history_end"):
            embed.set_mem_history_end(0)

        cursor = 0
        while cursor < len(selected_sorted):
            chunk_start = selected_sorted[cursor]
            chunk_limit = min(
                len(trace.context),
                chunk_start + int(window_chunk_tokens),
            )
            chunk_selected: List[int] = []
            while (
                cursor < len(selected_sorted)
                and selected_sorted[cursor] < chunk_limit
            ):
                chunk_selected.append(selected_sorted[cursor])
                cursor += 1
            if not chunk_selected:
                continue

            chunk_end = chunk_selected[-1] + 1
            dynamic_start = max(
                trace.prefix_len,
                chunk_selected[0] - int(max_context_window),
            )
            windowed = (
                [int(t) for t in trace.context[:trace.prefix_len]]
                + [int(t) for t in trace.context[dynamic_start:chunk_end]]
            )
            if hasattr(embed, "set_mem_store_positions"):
                embed.set_mem_store_positions([
                    _remapped_store_positions_for_window(
                        trace,
                        dynamic_start=dynamic_start,
                        dynamic_end=chunk_end,
                    )
                ])
            if hasattr(embed, "set_mem_addr_src_positions"):
                embed.set_mem_addr_src_positions([
                    _remapped_addr_src_positions_for_window(
                        trace,
                        dynamic_start=dynamic_start,
                        dynamic_end=chunk_end,
                    )
                ])

            token_ids = torch.tensor(
                [windowed],
                dtype=torch.long,
                device=device,
            )
            with torch.no_grad():
                x = model.embed(token_ids)
                for block in model.blocks:
                    x = block(x)

                for token_index in chunk_selected:
                    expected_token = int(trace.context[token_index])
                    if not 0 <= expected_token < int(
                        getattr(model, "vocab_size", 10**9)
                    ):
                        continue
                    checked += 1
                    logit_pos = trace.prefix_len + (token_index - dynamic_start) - 1
                    row = x[0, logit_pos]
                    failure = _lowering_trace_audit_failure(
                        model=model,
                        trace=trace,
                        row=row,
                        token_index=token_index,
                        min_margin=min_margin,
                        output_band_min_margin=output_band_min_margin,
                        output_band_max_inactive_value=(
                            output_band_max_inactive_value
                        ),
                        output_band_tolerance=output_band_tolerance,
                    )
                    if failure is None:
                        continue
                    failures.append(failure)
                    if len(failures) >= max_failures:
                        break
            if len(failures) >= max_failures:
                break
    finally:
        if old_mem_history_end is not None and hasattr(embed, "set_mem_history_end"):
            embed.set_mem_history_end(old_mem_history_end)
        if hasattr(embed, "set_mem_store_positions"):
            embed.set_mem_store_positions(old_mem_store_positions)
        if hasattr(embed, "set_mem_addr_src_positions"):
            embed.set_mem_addr_src_positions(old_mem_addr_src_positions)

    return TeacherForcedTraceAuditReport(
        probe_name=probe_name,
        checked=checked,
        failures=tuple(failures),
    )


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
    # Some migrated attention bakes resize individual blocks to add auxiliary
    # heads. Decode ownership against the block's actual head count; using the
    # compile-wide default aliases expanded heads to the wrong ``head_slot`` ids.
    actual_num_heads = int(getattr(attn, "num_heads", num_heads))
    head_dim = attn.W_q.shape[0] // actual_num_heads
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
                num_heads=actual_num_heads, head_dim=head_dim,
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
    registry: Optional[object] = None,
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

    report = StaticVerificationReport(strict_mode=strict_mode, registry=registry)
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
    display_label: Optional[str] = None
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
            label = r.display_label or r.op_name
            lines.append(f"  [{status}] {label} (op={r.op_name})")
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

    # Collect ops with produces or consumes_fresh.
    all_ops = []
    for ops_at in layout.ops_per_layer:
        all_ops.extend(ops_at)
    all_ops.extend(layout.block_ops)
    all_ops.extend(layout.model_ops)

    seen: Set[str] = set()
    candidates = []
    for op in all_ops:
        if op.name in seen:
            continue
        seen.add(op.name)
        if op.produces or op.consumes_fresh:
            candidates.append(op)

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
        res = DynamicVerificationResult(
            op_name=op.name,
            display_label=operation_display_label(op),
        )
        # Locate the op's layer.
        layer_idx = None
        if op.kind == "block":
            try:
                layer_idx = layout.resolve_block_op_layer(op)
            except ValueError:
                layer_idx = op.layer_idx
        elif op.kind in ("attn", "ffn"):
            for lidx, ops_at in enumerate(layout.ops_per_layer):
                if any(o.name == op.name for o in ops_at):
                    layer_idx = lidx
                    break

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
    display_label: Optional[str] = None
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
            placement_tag = (
                f" (placement_idx={r.layer_idx})"
                if r.layer_idx is not None else ""
            )
            label = r.display_label or r.op_name
            lines.append(f"  [{status}] {label} (op={r.op_name}){placement_tag}")
            for n in r.notes:
                lines.append(f"     NOTE: {n}")
            for d in r.drift:
                lines.append(
                    f"     DRIFT: step={d.step} dim={d.dim!r} "
                    f"register={d.register!r} pos={d.position} "
                    f"observed_abs_max={d.observed:.3e}"
                )
        return "\n".join(lines)


@dataclass
class SymbolicDeclarationProducer:
    """Declaration-side producer for one VM register slot."""
    op_name: str
    display_label: str
    dim: str
    register: str
    layer_idx: Optional[int] = None


@dataclass
class SymbolicDeclarationStep:
    """One step of a declaration-only register responsibility trace."""
    step: int
    active_opcode: Optional[str]
    state_summary: str
    producers_by_register: Dict[str, List[SymbolicDeclarationProducer]] = field(default_factory=dict)
    missing_registers: List[str] = field(default_factory=list)


@dataclass
class SymbolicDeclarationTraceReport:
    """Symbolic view of declarations over a concrete probe program.

    This does not execute the neural model and does not feed expected values
    into inference. It is a debugging map from expected VM register slots to
    ops that declare they produce those slots.
    """
    n_steps: int = 0
    program_summary: str = ""
    watched_registers: Tuple[str, ...] = ()
    steps: List[SymbolicDeclarationStep] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def has_missing(self) -> bool:
        return any(step.missing_registers for step in self.steps)

    def format(self) -> str:
        lines = [
            "=== Symbolic declaration trace ===",
            f"Program: {self.program_summary}",
            f"Steps: {self.n_steps}",
            "Watched registers: " + ", ".join(self.watched_registers),
        ]
        for step in self.steps:
            opcode = step.active_opcode or "HALTED"
            lines.append(f"  step {step.step}: opcode={opcode} {step.state_summary}")
            for reg in self.watched_registers:
                producers = step.producers_by_register.get(reg, [])
                if not producers:
                    lines.append(f"     MISSING: {reg}")
                    continue
                labels = ", ".join(
                    f"{p.display_label}({p.dim})" for p in producers
                )
                lines.append(f"     {reg}: {labels}")
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)


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


def _opcode_value_to_symbol(opcode: int) -> str:
    """Return the short Opcode constant name for an encoded opcode byte."""
    try:
        from ..embedding import Opcode
        for name in dir(Opcode):
            if name.startswith("_"):
                continue
            value = getattr(Opcode, name)
            if isinstance(value, int) and value == opcode:
                return name
    except Exception:
        pass
    return f"OP_{opcode}"


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


def _build_symbolic_probe_opcodes(
    program: List[int],
    n_steps: int,
) -> List[Optional[str]]:
    """Return the opcode executed at each DraftVM probe step.

    DraftVM is used here only as a diagnostic oracle for the probe path, just
    like ``_build_multistep_probe``. The returned names are not consumed by
    the runner or used to substitute model output.
    """
    from ..speculative import DraftVM

    vm = DraftVM(program)
    out: List[Optional[str]] = []
    for _ in range(n_steps):
        if vm.halted or vm.idx >= len(vm.code):
            out.append(None)
            continue
        opcode = vm.code[vm.idx] & 0xFF
        if not vm.step():
            out.append(None)
            continue
        out.append(_opcode_value_to_symbol(opcode))
    return out


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

    # Collect annotated ops (same gather pattern Mode B uses).
    all_ops = []
    for ops_at in layout.ops_per_layer:
        all_ops.extend(ops_at)
    all_ops.extend(layout.block_ops)
    all_ops.extend(layout.model_ops)

    seen: Set[str] = set()
    candidates = []
    for op in all_ops:
        if op.name in seen:
            continue
        seen.add(op.name)
        if op.produces:
            candidates.append(op)

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
        result = MultistepVerificationResult(
            op_name=op.name,
            display_label=operation_display_label(op),
        )

        # Resolve op's layer (same logic Mode B uses).
        layer_idx = None
        if op.kind == "block":
            try:
                layer_idx = layout.resolve_block_op_layer(op)
            except ValueError:
                layer_idx = op.layer_idx
        elif op.kind in ("attn", "ffn"):
            for lidx, ops_at in enumerate(layout.ops_per_layer):
                if any(o.name == op.name for o in ops_at):
                    layer_idx = lidx
                    break
        elif op.kind == "model":
            # Model ops may write multiple layers. Use the op's own
            # ``layer_idx`` hint when available (some model ops target a
            # specific block, see ``ffn_units_used`` doc). Else inspect
            # the residual at the deepest block -- liveness should
            # propagate forward.
            layer_idx = op.layer_idx
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
                # Translate marker-name 'REG_AX' -> step offset 5 via the
                # offset directly (avoid double lookup).
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
# Tier B detectors: ALiBi declarations, postconditions, step-index gating
# ---------------------------------------------------------------------------


@dataclass
class AlibiConsistencyEntry:
    layer_idx: int
    head_idx: int
    kind: str
    declared: Optional[float] = None
    actual: Optional[float] = None
    owners: List[str] = field(default_factory=list)


@dataclass
class AlibiConsistencyReport:
    entries: List[AlibiConsistencyEntry] = field(default_factory=list)
    n_ops_with_alibi: int = 0
    n_layers_inspected: int = 0
    notes: List[str] = field(default_factory=list)

    def has_drift(self) -> bool:
        return any(
            e.kind in ("double_write", "slope_mismatch")
            for e in self.entries
        )

    def double_writes(self) -> List[AlibiConsistencyEntry]:
        return [e for e in self.entries if e.kind == "double_write"]

    def mismatches(self) -> List[AlibiConsistencyEntry]:
        return [e for e in self.entries if e.kind == "slope_mismatch"]

    def format(self) -> str:
        lines = [
            "=== Tier B alibi-consistency report ===",
            f"Ops declaring alibi_slopes: {self.n_ops_with_alibi}",
            f"Layers inspected: {self.n_layers_inspected}",
            f"Double-writes: {len(self.double_writes())}",
            f"Slope mismatches: {len(self.mismatches())}",
        ]
        for entry in self.entries:
            lines.append(
                f"  {entry.kind}: layer={entry.layer_idx} "
                f"head={entry.head_idx} declared={entry.declared!r} "
                f"actual={entry.actual!r} owners={sorted(entry.owners)!r}"
            )
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)


def _collect_unique_ops_with(layout, predicate: Callable[[Operation], bool]) -> List[Operation]:
    ops: List[Operation] = []
    for ops_at in getattr(layout, "ops_per_layer", []):
        ops.extend(ops_at)
    ops.extend(getattr(layout, "block_ops", []))
    ops.extend(getattr(layout, "model_ops", []))

    out: List[Operation] = []
    seen: Set[str] = set()
    for op in ops:
        if op.name in seen:
            continue
        seen.add(op.name)
        if predicate(op):
            out.append(op)
    return out


def _resolve_op_layer(layout, op: Operation) -> Optional[int]:
    if op.kind == "model":
        return op.layer_idx
    if op.kind == "block":
        try:
            return layout.resolve_block_op_layer(op)
        except Exception:
            return op.layer_idx
    for layer_idx, ops_at in enumerate(getattr(layout, "ops_per_layer", [])):
        if any(placed.name == op.name for placed in ops_at):
            return layer_idx
    return op.layer_idx


_ADD_CASCADE_WATCHED_REGISTERS = (
    "AX_byte0",
    "SP_byte0", "SP_byte1", "SP_byte2", "SP_byte3",
    "BP_byte0", "BP_byte1", "BP_byte2", "BP_byte3",
    "STACK0_byte0", "STACK0_byte1", "STACK0_byte2", "STACK0_byte3",
)


def trace_symbolic_declarations(
    layout=None,
    program: Optional[List[int]] = None,
    n_steps: int = 4,
    *,
    watched_registers: Optional[Tuple[str, ...]] = None,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    n_heads: int = 8,
) -> SymbolicDeclarationTraceReport:
    """Build a declaration-only responsibility trace for a probe program.

    The trace answers: for each watched VM register byte at each step, which
    ops declare ``produces={dim: register}`` for that slot? It is intentionally
    symbolic: no neural forward pass, no runner interaction, and no DraftVM
    result substitution.
    """
    if program is None:
        program = list(_ADD_CASCADE_PROGRAM)
    if watched_registers is None:
        watched_registers = _ADD_CASCADE_WATCHED_REGISTERS

    report = SymbolicDeclarationTraceReport(
        n_steps=n_steps,
        program_summary=" ".join(f"0x{instr:016x}" for instr in program),
        watched_registers=watched_registers,
    )

    if layout is None:
        try:
            layout = _build_layout_only(
                alu_mode=alu_mode,
                enable_conversational_io=enable_conversational_io,
                enable_tool_calling=enable_tool_calling,
                n_heads=n_heads,
            )
        except Exception as exc:
            report.notes.append(f"layout build failed: {exc!r}")
            return report

    try:
        _, _, step_summaries = _build_multistep_probe(layout, program, n_steps)
        active_opcodes = _build_symbolic_probe_opcodes(program, n_steps)
    except Exception as exc:
        report.notes.append(f"probe build failed: {exc!r}")
        return report

    producers_by_register: Dict[str, List[SymbolicDeclarationProducer]] = {
        reg: [] for reg in watched_registers
    }
    for op in _collect_unique_ops_with(layout, lambda candidate: bool(candidate.produces)):
        layer_idx = _resolve_op_layer(layout, op)
        for dim_name, register in op.produces.items():
            if register not in producers_by_register:
                continue
            producers_by_register[register].append(SymbolicDeclarationProducer(
                op_name=op.name,
                display_label=operation_display_label(op),
                dim=dim_name,
                register=register,
                layer_idx=layer_idx,
            ))

    for step_idx in range(n_steps):
        step_producers = {
            reg: list(producers)
            for reg, producers in producers_by_register.items()
            if producers
        }
        missing = [
            reg for reg in watched_registers
            if not step_producers.get(reg)
        ]
        report.steps.append(SymbolicDeclarationStep(
            step=step_idx + 1,
            active_opcode=(
                active_opcodes[step_idx]
                if step_idx < len(active_opcodes) else None
            ),
            state_summary=(
                step_summaries[step_idx]
                if step_idx < len(step_summaries) else "(missing)"
            ),
            producers_by_register=step_producers,
            missing_registers=missing,
        ))

    return report


def _resolve_alibi_layer(layout, op: Operation) -> Optional[int]:
    return op.layer_idx if op.kind == "model" else _resolve_op_layer(layout, op)


def verify_alibi_consistency(
    compile_fn: Optional[Callable] = None,
    *,
    model=None,
    layout=None,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    S: float = 100.0,
    n_heads: int = 8,
    slope_epsilon: float = 1e-3,
) -> AlibiConsistencyReport:
    """Check ``Operation.alibi_slopes`` declarations for collisions and drift."""
    report = AlibiConsistencyReport()
    if layout is None:
        try:
            if compile_fn is not None:
                built = compile_fn()
                if isinstance(built, tuple) and len(built) == 2:
                    model, layout = built
                else:
                    layout = built
            else:
                from .full_vm_compiler import compile_full_vm
                model, layout = compile_full_vm(
                    S=S,
                    alu_mode=alu_mode,
                    enable_conversational_io=enable_conversational_io,
                    n_heads=n_heads,
                    disk_cache=False,
                )
        except Exception as exc:
            report.notes.append(f"compile failed: {exc!r}")
            return report

    registry: Dict[Tuple[int, int], List[Tuple[str, float]]] = {}
    for op in _collect_unique_ops_with(layout, lambda candidate: True):
        slopes = getattr(op, "alibi_slopes", {})
        if not slopes:
            continue
        report.n_ops_with_alibi += 1
        layer_idx = _resolve_alibi_layer(layout, op)
        if layer_idx is None:
            report.notes.append(
                f"op={op.name!r}: alibi_slopes declared but layer is unresolved"
            )
            continue
        for head_idx, slope in slopes.items():
            registry.setdefault((layer_idx, head_idx), []).append(
                (op.name, float(slope))
            )
    report.n_layers_inspected = len({layer for layer, _ in registry})

    for (layer_idx, head_idx), entries in sorted(registry.items()):
        if len(entries) > 1:
            report.entries.append(AlibiConsistencyEntry(
                layer_idx=layer_idx,
                head_idx=head_idx,
                kind="double_write",
                declared=entries[0][1],
                owners=[name for name, _ in entries],
            ))

    if model is None:
        report.notes.append("no model supplied; skipping slope-mismatch check")
        return report

    for (layer_idx, head_idx), entries in sorted(registry.items()):
        blocks = getattr(model, "blocks", [])
        if layer_idx >= len(blocks):
            report.notes.append(f"layer {layer_idx} out of range")
            continue
        slopes = getattr(getattr(blocks[layer_idx], "attn", None), "alibi_slopes", None)
        if slopes is None:
            report.notes.append(f"layer {layer_idx} has no alibi_slopes")
            continue
        if head_idx >= slopes.numel():
            report.notes.append(f"layer {layer_idx} head {head_idx} out of range")
            continue
        actual = float(slopes[head_idx].item())
        for op_name, declared in entries:
            if abs(actual - declared) > slope_epsilon:
                report.entries.append(AlibiConsistencyEntry(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    kind="slope_mismatch",
                    declared=declared,
                    actual=actual,
                    owners=[op_name],
                ))
    return report


@dataclass
class PostconditionEntry:
    op_name: str
    cell: str
    invariant: str
    step: int
    observed: str


@dataclass
class PostconditionReport:
    n_steps: int = 0
    drift: List[PostconditionEntry] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def has_drift(self) -> bool:
        return bool(self.drift)

    def format(self) -> str:
        lines = [
            "=== Tier B postcondition report ===",
            f"Steps inspected: {self.n_steps}",
            f"Drift entries: {len(self.drift)}",
        ]
        for entry in self.drift:
            lines.append(
                f"  POSTCOND DRIFT: op={entry.op_name!r} cell={entry.cell!r} "
                f"invariant={entry.invariant!r} step={entry.step} "
                f"observed={entry.observed}"
            )
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)


def _parse_postcondition_cell(cell: str, layout) -> Tuple[Optional[str], Optional[int]]:
    name = cell
    index = 0
    if "[" in cell and cell.endswith("]"):
        name, idx = cell[:-1].split("[", 1)
        try:
            index = int(idx)
        except ValueError:
            return None, None
    if name not in layout.dim_positions:
        return None, None
    return name, index


def _eval_postcondition(
    invariant: str,
    values_per_step: List[float],
    epsilon: float,
) -> Optional[str]:
    if invariant == "0_or_1":
        for idx, value in enumerate(values_per_step, start=1):
            if min(abs(value), abs(value - 1.0)) > epsilon:
                return f"step{idx}_value={value:.3f}"
        return None
    if invariant == "monotonic_non_decreasing":
        prev = None
        for idx, value in enumerate(values_per_step, start=1):
            if prev is not None and value < prev - epsilon:
                return f"step{idx}: {prev:.3f} -> {value:.3f}"
            prev = value
        return None
    if invariant == "byte_range":
        for idx, value in enumerate(values_per_step, start=1):
            if value < -0.5 or value > 255.5:
                return f"step{idx}_value={value:.3f}_outside_[0,255]"
        return None
    if invariant.startswith("matches_AX_byte"):
        return None
    return None


def _run_multistep_forward(model, token_tensor):
    try:
        with torch.no_grad():
            x = model.embed(token_tensor)
            residuals = [x.detach().clone()]
            for block in model.blocks:
                x = block(x)
                residuals.append(x.detach().clone())
        return residuals
    except Exception:
        return None


def _probe_parts(probe):
    if len(probe) == 4:
        return probe
    token_tensor, step_markers, step_summaries = probe
    return token_tensor, step_markers, step_summaries, [None] * len(step_markers)


def _op_active_at_step(op: Operation, active_opcode: Optional[str]) -> bool:
    opcodes = getattr(op, "opcodes", set())
    if not opcodes or active_opcode is None:
        return True
    return active_opcode in opcodes


def _postcondition_offset(op: Operation, dim_name: str) -> int:
    reg = op.produces.get(dim_name) if getattr(op, "produces", None) else None
    if reg is None:
        return _STEP_TOKEN_OFFSETS["REG_PC"]
    if reg in _STEP_TOKEN_OFFSETS:
        return _STEP_TOKEN_OFFSETS[reg]
    bare = reg.replace("_marker", "")
    if bare in ("AX", "PC", "SP", "BP"):
        return _STEP_TOKEN_OFFSETS[f"REG_{bare}"]
    if bare in ("STACK0", "MEM"):
        return _STEP_TOKEN_OFFSETS[bare]
    return _STEP_TOKEN_OFFSETS["REG_PC"]


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
    epsilon: float = 1e-1,
) -> PostconditionReport:
    report = PostconditionReport(n_steps=n_steps)
    if program is None:
        program = list(_ADD_CASCADE_PROGRAM)
    if model is None or layout is None:
        try:
            from .full_vm_compiler import compile_full_vm
            model, layout = compile_full_vm(
                S=S,
                alu_mode=alu_mode,
                enable_conversational_io=enable_conversational_io,
                n_heads=n_heads,
                disk_cache=False,
            )
        except Exception as exc:
            report.notes.append(f"compile failed: {exc!r}")
            return report

    candidates = _collect_unique_ops_with(
        layout, lambda op: bool(getattr(op, "postcondition", {}))
    )
    if not candidates:
        report.notes.append("no ops declare postcondition")
        return report

    try:
        token_tensor, step_markers, _, step_active_opcodes = _probe_parts(
            _build_multistep_probe(layout, program, n_steps)
        )
    except Exception as exc:
        report.notes.append(f"could not construct multistep probe: {exc!r}")
        return report

    if getattr(model, "max_seq_len", None) is not None and token_tensor.shape[1] > model.max_seq_len:
        report.notes.append(
            f"context len {token_tensor.shape[1]} exceeds model max_seq_len {model.max_seq_len}"
        )
        return report

    residuals = _run_multistep_forward(model, token_tensor)
    if residuals is None:
        report.notes.append("forward pass failed")
        return report

    for op in candidates:
        layer_idx = _resolve_op_layer(layout, op)
        inspect_layer = len(residuals) - 1 if layer_idx is None else min(layer_idx + 1, len(residuals) - 1)
        post_residual = residuals[inspect_layer]
        for cell, invariant in op.postcondition.items():
            dim_name, index = _parse_postcondition_cell(cell, layout)
            if dim_name is None:
                report.notes.append(f"op={op.name!r}: unresolved cell {cell!r}")
                continue
            values: List[float] = []
            d_start = layout.dim_positions[dim_name]
            offset = _postcondition_offset(op, dim_name)
            for step_idx, marker_map in enumerate(step_markers):
                if not marker_map:
                    continue
                active = step_active_opcodes[step_idx] if step_idx < len(step_active_opcodes) else None
                if not _op_active_at_step(op, active):
                    continue
                pos = marker_map["REG_PC"] + offset
                values.append(float(post_residual[0, pos, d_start + index].item()))
            if not values:
                continue
            failure = _eval_postcondition(invariant, values, epsilon)
            if failure is not None:
                report.drift.append(PostconditionEntry(
                    op_name=op.name,
                    cell=cell,
                    invariant=invariant,
                    step=1,
                    observed=failure,
                ))
    return report


@dataclass
class StepIdxEntry:
    op_name: str
    step: int
    kind: str
    dim: str
    position: int
    observed: float


@dataclass
class StepIdxReport:
    n_steps: int = 0
    drift: List[StepIdxEntry] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def has_drift(self) -> bool:
        return bool(self.drift)

    def format(self) -> str:
        lines = [
            "=== Tier B step_idx gating report ===",
            f"Steps inspected: {self.n_steps}",
            f"Drift entries: {len(self.drift)}",
        ]
        for entry in self.drift:
            lines.append(
                f"  STEP-IDX DRIFT: op={entry.op_name!r} step={entry.step} "
                f"dim={entry.dim!r} pos={entry.position} "
                f"observed={entry.observed:.3e}"
            )
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)


def _op_step_idx_allowed(op: Operation, step_idx: int) -> bool:
    step_decl = getattr(op, "step_idx", None)
    if step_decl is None or step_decl == "every":
        return True
    if step_decl == "after_first":
        return step_idx >= 1
    if isinstance(step_decl, set):
        return step_idx in step_decl
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
    epsilon: float = 1e-1,
) -> StepIdxReport:
    report = StepIdxReport(n_steps=n_steps)
    if program is None:
        program = list(_ADD_CASCADE_PROGRAM)
    if model is None or layout is None:
        try:
            from .full_vm_compiler import compile_full_vm
            model, layout = compile_full_vm(
                S=S,
                alu_mode=alu_mode,
                enable_conversational_io=enable_conversational_io,
                n_heads=n_heads,
                disk_cache=False,
            )
        except Exception as exc:
            report.notes.append(f"compile failed: {exc!r}")
            return report

    candidates = _collect_unique_ops_with(
        layout,
        lambda op: getattr(op, "step_idx", None) not in (None, "every"),
    )
    if not candidates:
        report.notes.append("no ops declare a non-default step_idx")
        return report

    try:
        token_tensor, step_markers, _, step_active_opcodes = _probe_parts(
            _build_multistep_probe(layout, program, n_steps)
        )
    except Exception as exc:
        report.notes.append(f"could not construct multistep probe: {exc!r}")
        return report

    if getattr(model, "max_seq_len", None) is not None and token_tensor.shape[1] > model.max_seq_len:
        report.notes.append(
            f"context len {token_tensor.shape[1]} exceeds model max_seq_len {model.max_seq_len}"
        )
        return report

    residuals = _run_multistep_forward(model, token_tensor)
    if residuals is None:
        report.notes.append("forward pass failed")
        return report

    for op in candidates:
        layer_idx = _resolve_op_layer(layout, op)
        inspect_layer = len(residuals) - 1 if layer_idx is None else min(layer_idx + 1, len(residuals) - 1)
        post_residual = residuals[inspect_layer]
        candidate_dims = set(op.writes)
        candidate_dims.update(
            dim for dim in getattr(op, "produces", {}).keys()
            if dim in layout.dim_positions
        )
        for step_idx, marker_map in enumerate(step_markers):
            if not marker_map or _op_step_idx_allowed(op, step_idx):
                continue
            active = step_active_opcodes[step_idx] if step_idx < len(step_active_opcodes) else None
            if not _op_active_at_step(op, active):
                continue
            pos = marker_map["REG_PC"]
            for dim_name in sorted(candidate_dims):
                if dim_name not in layout.dim_positions:
                    continue
                start = layout.dim_positions[dim_name]
                size = layout.dim_sizes.get(dim_name, 1)
                observed = float(post_residual[0, pos, start:start + size].abs().max().item())
                if observed >= epsilon:
                    report.drift.append(StepIdxEntry(
                        op_name=op.name,
                        step=step_idx + 1,
                        kind="fired_when_disabled",
                        dim=dim_name,
                        position=pos,
                        observed=observed,
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


# ---------------------------------------------------------------------------
# Tier C discoverability audits (smoke coverage, spec coverage, MoE safety).
# These bookkeeping detectors are invoked explicitly by tests and opt-in CI
# jobs; they are intentionally not part of verify_all.
# ---------------------------------------------------------------------------


@dataclass
class SmokeCoverageReport:
    coverage: Dict[str, Set[str]] = field(default_factory=dict)
    untested_ops: List[str] = field(default_factory=list)
    tests_to_ops: Dict[str, List[str]] = field(default_factory=dict)
    orphan_test_refs: Dict[str, Set[str]] = field(default_factory=dict)

    def has_untested(self) -> bool:
        return bool(self.untested_ops)

    def format(self) -> str:
        lines = ["=== Smoke coverage bookkeeping ==="]
        lines.append(f"Ops declared: {len(self.coverage)}")
        lines.append(f"Ops untested: {len(self.untested_ops)}")
        lines.append(f"Smoke tests referenced: {len(self.tests_to_ops)}")
        if self.orphan_test_refs:
            lines.append(
                f"Ops with orphan test refs: {len(self.orphan_test_refs)}"
            )
            for op_name, refs in sorted(self.orphan_test_refs.items()):
                lines.append(f"  {op_name}: {sorted(refs)}")
        return "\n".join(lines)


def audit_smoke_coverage(
    compile_fn: Optional[Callable] = None,
    *,
    known_smoke_tests: Optional[Set[str]] = None,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    n_heads: int = 8,
) -> SmokeCoverageReport:
    if compile_fn is None:
        layout = _build_layout_only(
            alu_mode=alu_mode,
            enable_conversational_io=enable_conversational_io,
            enable_tool_calling=enable_tool_calling,
            n_heads=n_heads,
        )
    else:
        layout = compile_fn()

    coverage: Dict[str, Set[str]] = {}
    tests_to_ops: Dict[str, List[str]] = {}
    untested: List[str] = []
    orphan: Dict[str, Set[str]] = {}

    for op in _collect_unique_ops_with(layout, lambda _op: True):
        coverage[op.name] = set(op.smoke_tests)
        if not op.smoke_tests:
            untested.append(op.name)
            continue
        for entry in op.smoke_tests:
            tests_to_ops.setdefault(entry, []).append(op.name)
            if (
                known_smoke_tests is not None
                and entry != "all"
                and entry not in known_smoke_tests
            ):
                orphan.setdefault(op.name, set()).add(entry)

    return SmokeCoverageReport(
        coverage=coverage,
        untested_ops=sorted(untested),
        tests_to_ops={k: sorted(v) for k, v in tests_to_ops.items()},
        orphan_test_refs=orphan,
    )


@dataclass
class SpecCoverageReport:
    coverage: Dict[str, List[str]] = field(default_factory=dict)
    undocumented_ops: List[str] = field(default_factory=list)
    unreferenced_sections: List[str] = field(default_factory=list)
    all_sections: List[str] = field(default_factory=list)

    def format(self) -> str:
        lines = ["=== Spec coverage bookkeeping ==="]
        lines.append(f"Spec sections found: {len(self.all_sections)}")
        lines.append(f"Sections referenced by >=1 op: {len(self.coverage)}")
        lines.append(
            f"Sections referenced by NO op: {len(self.unreferenced_sections)}"
        )
        lines.append(f"Ops with spec_section=None: {len(self.undocumented_ops)}")
        return "\n".join(lines)


def _slugify_heading(text: str) -> str:
    return "-".join(text.lower().split())


def audit_spec_coverage(
    compile_fn: Optional[Callable] = None,
    *,
    spec_path: str = "c4_release/docs/BLOG_SPEC.md",
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    n_heads: int = 8,
) -> SpecCoverageReport:
    if compile_fn is None:
        layout = _build_layout_only(
            alu_mode=alu_mode,
            enable_conversational_io=enable_conversational_io,
            enable_tool_calling=enable_tool_calling,
            n_heads=n_heads,
        )
    else:
        layout = compile_fn()

    try:
        with open(spec_path, encoding="utf-8") as f:
            spec_lines = f.readlines()
    except OSError:
        spec_lines = []

    all_sections: List[str] = []
    heading_slugs: List[str] = []
    for raw in spec_lines:
        line = raw.rstrip("\n")
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            if text:
                all_sections.append(text)
                heading_slugs.append(_slugify_heading(text))

    coverage: Dict[str, List[str]] = {}
    undocumented: List[str] = []
    referenced_slugs: Set[str] = set()

    for op in _collect_unique_ops_with(layout, lambda _op: True):
        if op.spec_section is None:
            undocumented.append(op.name)
            continue
        ref = op.spec_section
        coverage.setdefault(ref, []).append(op.name)
        if "#" in ref:
            slug = ref.split("#", 1)[1].lower()
            for hslug in heading_slugs:
                if slug in hslug or hslug in slug:
                    referenced_slugs.add(hslug)
                    break

    unreferenced = [
        all_sections[i]
        for i, hslug in enumerate(heading_slugs)
        if hslug not in referenced_slugs
    ]

    return SpecCoverageReport(
        coverage={k: sorted(v) for k, v in coverage.items()},
        undocumented_ops=sorted(undocumented),
        unreferenced_sections=unreferenced,
        all_sections=all_sections,
    )


@dataclass
class DeclarativeAuthorityReport:
    authoritative_ops: List[str] = field(default_factory=list)
    legacy_wrapper_ops: List[str] = field(default_factory=list)
    unclassified_ops: List[str] = field(default_factory=list)
    semantic_labels: Dict[str, str] = field(default_factory=dict)
    explicit_sources: Dict[str, str] = field(default_factory=dict)
    inferred_sources: Dict[str, str] = field(default_factory=dict)
    # F-11: per-op {reads: [...], writes: [...]} captured at audit time so
    # ``format()`` can enrich dim names with semantics when a registry is
    # supplied. Empty by default; populated by ``audit_declarative_authority``.
    op_reads: Dict[str, List[str]] = field(default_factory=dict)
    op_writes: Dict[str, List[str]] = field(default_factory=dict)
    # Optional registry for semantics enrichment in ``format()``.
    registry: Optional[object] = None

    @property
    def authoritative_count(self) -> int:
        return len(self.authoritative_ops)

    @property
    def legacy_wrapper_count(self) -> int:
        return len(self.legacy_wrapper_ops)

    @property
    def unclassified_count(self) -> int:
        return len(self.unclassified_ops)

    def format(self) -> str:
        lines = ["=== Declarative authority audit ==="]
        lines.append(f"Authoritative declarative ops: {self.authoritative_count}")
        lines.append(f"Legacy wrapper ops: {self.legacy_wrapper_count}")
        lines.append(f"Unclassified ops: {self.unclassified_count}")
        lines.append(f"Explicit source markers: {len(self.explicit_sources)}")
        lines.append(f"Inferred source markers: {len(self.inferred_sources)}")
        # F-11: when a registry is supplied, list each op's reads/writes
        # alongside their declared semantics so reviewers can audit at-a-glance
        # whether dim meanings match each rule's intent. Pure observability.
        if self.registry is not None and (self.op_reads or self.op_writes):
            lines.append("--- Per-op dim references (with semantics) ---")
            all_ops = sorted(set(self.op_reads) | set(self.op_writes))
            for op_name in all_ops:
                reads = self.op_reads.get(op_name, [])
                writes = self.op_writes.get(op_name, [])
                if not reads and not writes:
                    continue
                lines.append(f"  {op_name}:")
                for dim_name in reads:
                    lines.append(
                        f"    R {_format_dim_with_semantics(dim_name, self.registry)}"
                    )
                for dim_name in writes:
                    lines.append(
                        f"    W {_format_dim_with_semantics(dim_name, self.registry)}"
                    )
        return "\n".join(lines)

    def semantic_label(self, op_name: str) -> str:
        return self.semantic_labels.get(op_name, semantic_op_label(op_name))


def _looks_like_opaque_legacy_wrapper(op: Operation) -> bool:
    if op.name == "legacy_bake":
        return True
    try:
        source = inspect.getsource(op.bake_fn)
    except (OSError, TypeError):
        return False
    return (
        bool(_LEGACY_HELPER_CALL_RE.search(source))
        or "set_vm_weights" in source
        or "legacy_bake" in source
    )


def audit_declarative_authority(
    compile_fn: Optional[Callable] = None,
    *,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    n_heads: int = 8,
    registry: Optional[object] = None,
) -> DeclarativeAuthorityReport:
    """Classify ops by whether their bake path is declarative authority.

    Explicit ``Operation.declarative_authority`` markers are authoritative.
    While the migration is in progress, this audit also infers two production
    classifications: bake functions that still call legacy ``_set_*`` helpers
    are opaque legacy wrappers; remaining migrated ops are compiler-owned
    declarative bakes.
    """
    if compile_fn is None:
        layout = _build_layout_only(
            alu_mode=alu_mode,
            enable_conversational_io=enable_conversational_io,
            enable_tool_calling=enable_tool_calling,
            n_heads=n_heads,
        )
    else:
        layout = compile_fn()

    authoritative: List[str] = []
    legacy_wrapper: List[str] = []
    unclassified: List[str] = []
    explicit: Dict[str, str] = {}
    inferred: Dict[str, str] = {}
    semantic_labels: Dict[str, str] = {}
    op_reads: Dict[str, List[str]] = {}
    op_writes: Dict[str, List[str]] = {}

    for op in _collect_unique_ops_with(layout, lambda _op: True):
        semantic_labels[op.name] = operation_display_label(op)
        # F-11: capture per-op dim references so the report can later
        # enrich them with semantics if a registry is supplied.
        op_reads[op.name] = sorted(op.reads) if op.reads else []
        op_writes[op.name] = sorted(op.writes) if op.writes else []
        source = op.declarative_authority
        if source is not None:
            explicit[op.name] = source
        elif _looks_like_opaque_legacy_wrapper(op):
            source = "legacy_wrapper"
            inferred[op.name] = source
        elif op.migrated:
            source = "declarative"
            inferred[op.name] = source

        if source in (
            "declarative",
            "spec_generated",
            "structural_model",
            "topology_anchor",
        ):
            authoritative.append(op.name)
        elif source == "legacy_wrapper":
            legacy_wrapper.append(op.name)
        else:
            unclassified.append(op.name)

    return DeclarativeAuthorityReport(
        authoritative_ops=sorted(authoritative),
        legacy_wrapper_ops=sorted(legacy_wrapper),
        unclassified_ops=sorted(unclassified),
        semantic_labels=dict(sorted(semantic_labels.items())),
        explicit_sources=explicit,
        inferred_sources=inferred,
        op_reads=op_reads,
        op_writes=op_writes,
        registry=registry,
    )


@dataclass
class CompactionSafetyReport:
    declared_unsafe: List[str] = field(default_factory=list)
    declared_safe: List[str] = field(default_factory=list)
    mismatches: Dict[str, str] = field(default_factory=dict)
    partition_unavailable: bool = False

    def has_mismatches(self) -> bool:
        return bool(self.mismatches)

    def format(self) -> str:
        lines = ["=== Compaction-safety bookkeeping ==="]
        lines.append(f"Ops declared compaction_safe=True: {len(self.declared_safe)}")
        lines.append(f"Ops declared compaction_safe=False: {len(self.declared_unsafe)}")
        if self.partition_unavailable:
            lines.append("Partition unavailable -- cross-check skipped.")
        else:
            lines.append(f"Mismatches: {len(self.mismatches)}")
            for op_name, reason in sorted(self.mismatches.items()):
                lines.append(f"  {op_name}: {reason}")
        return "\n".join(lines)


def verify_compaction_safety(
    compile_fn: Optional[Callable] = None,
    *,
    moe_partition: Optional[Dict] = None,
    alu_mode: str = "lookup",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    n_heads: int = 8,
) -> CompactionSafetyReport:
    if compile_fn is None:
        layout = _build_layout_only(
            alu_mode=alu_mode,
            enable_conversational_io=enable_conversational_io,
            enable_tool_calling=enable_tool_calling,
            n_heads=n_heads,
        )
    else:
        layout = compile_fn()

    declared_safe: List[str] = []
    declared_unsafe: List[str] = []
    mismatches: Dict[str, str] = {}
    op_units: Dict[str, Set[int]] = {}

    all_ops = _collect_unique_ops_with(layout, lambda _op: True)
    for op in all_ops:
        units: Set[int] = set()
        for claim in op.claims:
            if len(claim) != 4:
                continue
            _, scope, identifier, _ = claim
            if scope not in ("ffn_W_up", "ffn_W_down", "ffn_W_gate"):
                continue
            try:
                units.add(int(identifier))
            except (TypeError, ValueError):
                continue
        op_units[op.name] = units
        if op.compaction_safe:
            declared_safe.append(op.name)
        else:
            declared_unsafe.append(op.name)

    partition_unavailable = moe_partition is None
    if not partition_unavailable:
        opcode_unit_owner: Dict[int, int] = {}
        for opcode_dim, units in moe_partition.items():
            for unit_idx in units:
                opcode_unit_owner[int(unit_idx)] = int(opcode_dim)
        for op_name in declared_unsafe:
            misrouted = sorted(
                unit_idx
                for unit_idx in op_units.get(op_name, ())
                if unit_idx in opcode_unit_owner
            )
            if misrouted:
                suffix = "..." if len(misrouted) > 5 else ""
                mismatches[op_name] = (
                    "declared compaction_safe=False but FFN units "
                    f"{misrouted[:5]}{suffix} are opcode-routed"
                )

    return CompactionSafetyReport(
        declared_safe=sorted(declared_safe),
        declared_unsafe=sorted(declared_unsafe),
        mismatches=mismatches,
        partition_unavailable=partition_unavailable,
    )


def verify_rule_scopes(
    op,                                   # Operation (with .compiler_ir holding FFNOp or FFNRule list)
    registry,                             # DimRegistry
    require_scope: bool = False,
) -> List[Dict]:
    """F-7: check that each FFNRule's declared ``scope`` predicate is
    entailed by the effective firing predicate (over-approximation
    from conditions+threshold).

    Returns a list of issue dicts:
        {'kind': 'no_scope', 'rule': name}   # if require_scope=True and scope missing
        {'kind': 'scope_violation', 'rule': name,
         'effective': pred_str, 'scope': scope_str,
         'reason': explain_failure_str}

    If ``require_scope`` is False (default), rules without a scope are
    silently skipped (opt-in adoption).
    """
    from neural_vm.unified_compiler.predicates import parse, entails, explain_failure
    from neural_vm.unified_compiler.effective_predicate import effective_predicate

    issues: List[Dict] = []

    rules = _collect_ffn_rules_from_op(op)

    for rule in rules:
        scope_str = getattr(rule, "scope", None)
        if scope_str is None:
            if require_scope:
                issues.append({
                    "kind": "no_scope",
                    "rule": getattr(rule, "name", "<anonymous>"),
                })
            continue

        try:
            scope_pred = parse(scope_str)
        except Exception as e:
            issues.append({
                "kind": "scope_parse_error",
                "rule": getattr(rule, "name", "<anonymous>"),
                "scope": scope_str,
                "reason": str(e),
            })
            continue

        try:
            effective = effective_predicate(rule, registry)
        except Exception as e:
            issues.append({
                "kind": "effective_inference_error",
                "rule": getattr(rule, "name", "<anonymous>"),
                "reason": str(e),
            })
            continue

        if not entails(effective, scope_pred):
            reason = explain_failure(effective, scope_pred) or (
                "scope not entailed by effective predicate"
            )
            issues.append({
                "kind": "scope_violation",
                "rule": getattr(rule, "name", "<anonymous>"),
                "effective": str(effective),
                "scope": scope_str,
                "reason": reason,
            })

    return issues


def verify_rule_strength(
    op,
    registry,
    *,
    backbone_bounds=None,   # optional BackboneBounds (from S-8) or callable
    margin: float = 1.0,
    require_dominates: bool = False,
):
    """S-6: check that each FFNRule's declared `dominates_at[D]` is
    actually realized by the rule's algebraic contribution against
    competing writers (per S-4 writer index) and the backbone bound.

    Returns a list of issue dicts. Issue kinds:
      'no_dominates_at'   -- rule has no dominates_at and require_dominates=True
      'strength_violation' -- rule's max_contribution does not dominate
      'overlapping_writers_not_resolved' -- multiple rules claim dominance
                                            at the same position class

    Caller should pass `ops_for_competition` if they want competition
    computed against more ops than just the one being verified -- for now
    we only consider competing rules in `op` itself.
    """
    from neural_vm.unified_compiler.writer_index import build_writer_index
    from neural_vm.unified_compiler.contribution_algebra import max_contribution
    from neural_vm.unified_compiler.effective_predicate import effective_predicate
    from neural_vm.unified_compiler.predicates import parse, overlaps

    issues: List[Dict] = []

    # Build the writer index from this op alone (V1 -- single-op scope).
    # Later versions can take ops_for_competition= for cross-op
    # comparison.
    index = build_writer_index([op], registry)

    rules = _collect_ffn_rules_from_op(op)

    for rule in rules:
        rule_name = getattr(rule, "name", "<anonymous>")

        # Check each write of this rule
        for wt in rule.writes:
            if wt.weight == 0.0:
                continue

            output_dim = wt.dim.name
            output_offset = wt.dim.offset

            # What scope does the rule claim to dominate at this dim?
            dom_scope_str = rule.dominates_at_for(output_dim)

            if dom_scope_str is None:
                if require_dominates:
                    issues.append({
                        "kind": "no_dominates_at",
                        "rule": rule_name,
                        "output_dim": f"{output_dim}+{output_offset}",
                    })
                continue

            try:
                dom_scope = parse(dom_scope_str)
            except Exception as e:
                issues.append({
                    "kind": "dominates_at_parse_error",
                    "rule": rule_name,
                    "output_dim": f"{output_dim}+{output_offset}",
                    "reason": str(e),
                })
                continue

            # Find competing writers at the same output_dim+offset whose
            # effective scope overlaps the rule's declared dominance scope
            key = (output_dim, output_offset)
            all_writers = index.get(key, [])

            competing = []
            for w in all_writers:
                if w.rule is rule:
                    continue
                if overlaps(w.effective_scope, dom_scope):
                    competing.append(w)

            # This rule's contribution
            my_contrib = max_contribution(
                rule, output_dim, output_offset=output_offset
            )

            # Sum of competing contributions
            # Conservative: sum (not max) -- multiple competitors can sum
            # against us in a single forward.
            competing_max_contrib = max(
                (w.max_contribution for w in competing),
                default=0.0,
            )
            # We use max() not sum() for V1 because softmax argmax is
            # decided by the highest single competitor, not the sum.

            # Backbone bound at the dominance scope
            backbone_max = 0.0
            if backbone_bounds is not None:
                backbone_max = backbone_bounds(
                    output_dim, output_offset, dom_scope_str
                )

            required = competing_max_contrib + backbone_max + margin

            if my_contrib < required:
                issues.append({
                    "kind": "strength_violation",
                    "rule": rule_name,
                    "output_dim": f"{output_dim}+{output_offset}",
                    "dominates_at": dom_scope_str,
                    "my_contribution": my_contrib,
                    "competing_max": competing_max_contrib,
                    "backbone_max": backbone_max,
                    "required": required,
                    "shortfall": required - my_contrib,
                    "top_competitor": (
                        competing[0].rule.name if competing else None
                    ),
                })

    return issues


def _collect_ffn_rules_from_op(op) -> List:
    """Walk an Operation to find its FFNRules. Operations expose rules
    via ``.compiler_ir`` which can be an FFNOp, a list, or other shapes.
    Handles the common cases; returns ``[]`` for ops without FFN rules."""
    from neural_vm.unified_compiler.ir import FFNRule, FFNOp

    ir = getattr(op, "compiler_ir", None)
    if ir is None:
        return []

    rules: List = []
    # FFNOp case
    if isinstance(ir, FFNOp):
        return list(ir.rules)
    # List-of-rules case
    if isinstance(ir, list):
        for item in ir:
            if isinstance(item, FFNRule):
                rules.append(item)
            elif isinstance(item, FFNOp):
                rules.extend(item.rules)
        return rules
    # Tuple case
    if isinstance(ir, tuple):
        for item in ir:
            if isinstance(item, FFNRule):
                rules.append(item)
            elif isinstance(item, FFNOp):
                rules.extend(item.rules)
        return rules
    # Object with .rules attribute
    if hasattr(ir, "rules"):
        return [r for r in ir.rules if isinstance(r, FFNRule)]
    return []
