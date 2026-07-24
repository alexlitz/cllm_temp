"""CROSS-PROGRAM BATCHING for the NATIVE-FAST / lean conditional-block VM path.

The 5th throughput lever, and the one the native-fast path was missing.

Where the other batching layers stack STEPS of ONE program
==========================================================
  * ``qwen_lean_forward._build_spec_batch`` stacks the B *step-windows* of ONE
    program into ``[B, S, H]`` and verifies B VM steps in one lean forward.
  * ``full_native_fast._run_native`` (the native-fast driver) uses exactly that:
    one program, block_steps steps per conditional forward.
  * ``batched_speculative.speculative_run_batch`` DOES batch across programs, but
    only for the *pure-forward* ``SparseTransformer`` model (``.blocks`` / ``.embed``
    / ``PureForwardCompleteLayout``) — a DIFFERENT model object than the
    ``LeanQwenVM`` / ``ConditionalBlockLean`` conditional-block fast path.

This module stacks B INDEPENDENT PROGRAMS into the batch dimension of the
conditional-block lean forward.  At each block-step it:

  1. takes each still-running lane's next ``block_steps`` drafted step-windows,
  2. stacks the ACTIVE lanes' windows into ONE ``[B_active * block_steps, S, H]``
     batched residual (each row is the byte-identical single-step window the naive
     driver would build — causally self-contained, pad rows dropped),
  3. runs ONE conditional forward on the shared VM weights, and
  4. demuxes each lane's per-step AX back out.

The payoff (the union-active-block model)
=========================================
The conditional dispatch (``ConditionalBlockLean``) computes, per layer, only the
FFN units that FIRE for the rows in the batch — the UNION of active units across
every lane's opcode.  When the B programs share an opcode distribution (they
almost always do: every step is a PC-fetch + dispatch, and the common ALU ops
recur), that union is far smaller than the SUM of the per-lane active sets.  So the
per-forward FFN work scales like ``max/union`` over lanes, not ``sum`` — a
throughput multiplier over running the B programs sequentially.  ``union_speedup``
below quantifies it structurally (sum-of-per-lane active units / union active
units) from the SAME ``perlayer_conditional_sparse`` machinery the run path uses.

Byte-identity
=============
Each lane's window is the SELF-CONTAINED naive single-step window (no cross-lane
attention; pad rows sit at a far position so the causal mask drops them), so a
lane's decoded AX in the batched forward is byte-identical to stepping that
program ALONE through ``full_native_fast._run_native``.  This is the SAME
guarantee ``_build_spec_batch`` already proves for cross-STEP batching, extended
to the batch of independent programs (which just concatenates more independent
rows).  Cross-program batching changes only the SHAPE of the forward, never the
per-row arithmetic.

CPU-VERIFIED / GPU-DEFERRED
===========================
The stepping control flow, the draft, the lane retirement/compaction, and the
window-stacking isolation are all verified on CPU (see ``verify_cpu`` and the
``__main__`` self-check) WITHOUT a model build — the per-lane drafts are compared
against the single-program drafts, and the batched schedule is asserted to visit
every (lane, step) exactly once, in order.  The end-to-end BYTE-EXACT decode + the
wall-clock throughput multiplier require a real ``ConditionalBlockLean`` forward on
a GPU and are in ``deferred_gpu_checklist`` / ``verify_gpu_byte_exact``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import isa
from . import qwen_lean_forward as LF
from .qwen_lean_forward import CAM_REGS, _snap, LeanDraft, draft_program_lean

# _decode_reg_from_nibbles is the efficient-ALU AX decode (MUL/DIV/MOD read the AX
# nibble band); imported from the same place full_native_fast uses.
from .nibble_pure_forward_complete import _decode_reg_from_nibbles


_NIBBLE_AX_OPS = (isa.MUL, isa.DIV, isa.MOD)


# ===========================================================================
# A LANE — one independent program + its perfect draft + its running cursor.
# ===========================================================================
@dataclass
class Lane:
    """One program stacked into the batch dimension.

    ``code`` is the assembled program; ``draft`` its perfect Python draft (the
    exact per-step register file the naive driver would feed the model); ``cursor``
    the next drafted step to verify; ``ax_trace`` the decoded AX per accepted step.
    A lane is RETIRED (``done``) once its cursor passes the last drafted step.
    """
    idx: int
    code: List[isa.Instr]
    draft: LeanDraft
    cursor: int = 0
    ax_trace: List[int] = field(default_factory=list)
    done: bool = False
    naive_fallback: bool = False       # draft empty (out-of-slice op) -> run alone
    label: str = ""

    @property
    def n_steps(self) -> int:
        return len(self.draft.steps)

    @property
    def remaining(self) -> int:
        return max(0, self.n_steps - self.cursor)

    @property
    def exact(self) -> bool:
        return self.ax_trace == self.draft.ref_trace


@dataclass
class CrossBatchResult:
    """Per-lane verdicts + the throughput model for the whole batch."""
    lanes: List[Lane]
    forwards: int                       # batched conditional forwards actually run
    naive_forwards: int                 # sum over lanes of their step counts (== sequential)
    max_batch_rows: int                 # widest [rows, S, H] forward we ran
    # THROUGHPUT MODEL (structural, no GPU): active-block union vs sum-over-lanes.
    throughput: Dict[str, object] = field(default_factory=dict)

    @property
    def all_exact(self) -> bool:
        return all(l.exact for l in self.lanes)

    @property
    def n_pass(self) -> int:
        return sum(1 for l in self.lanes if l.exact)

    def per_lane_ax(self) -> Dict[int, List[int]]:
        return {l.idx: l.ax_trace for l in self.lanes}


# ===========================================================================
# BUILD the lanes (draft every program — pure Python, zero forwards).
# ===========================================================================
def build_lanes(draft_lean, programs: Sequence[List[isa.Instr]], *,
                max_steps: int = 4096,
                labels: Optional[Sequence[str]] = None) -> List[Lane]:
    """Draft every program in ``programs`` into a ``Lane`` (perfect Python draft,
    zero model forwards).  A program whose draft is EMPTY (uses an out-of-slice op,
    e.g. a function/syscall the draft can't model) is marked ``naive_fallback`` —
    the batched stepper skips it and the caller runs it alone through the
    single-program native driver."""
    lanes: List[Lane] = []
    for i, code in enumerate(programs):
        draft = draft_program_lean(draft_lean, code, max_steps=max_steps)
        lane = Lane(idx=i, code=list(code), draft=draft,
                    naive_fallback=(len(draft.steps) == 0),
                    label=(labels[i] if labels else ""))
        if lane.naive_fallback:
            lane.done = True             # nothing to batch-step; caller handles it
        lanes.append(lane)
    return lanes


# ===========================================================================
# One BATCHED forward over the active lanes' next block_steps step-windows.
# Returns the stacked residual [rows, Smax, H], the per-lane row layout, and the
# batch of (lane, step) the rows decode to.  The forward itself is run by the
# caller (so the GPU call is a single explicit line the deferred checklist names).
# ===========================================================================
def _stack_active_slabs(draft_lean, lanes: Sequence[Lane], active: Sequence[int],
                        block_steps: int
                        ) -> Tuple[torch.Tensor, torch.Tensor,
                                   List[Tuple[int, int, int]]]:
    """Stack the next ``block_steps`` step-windows of every ACTIVE lane into ONE
    batched ``[rows, Smax, H]`` residual (rows = sum of each active lane's slab
    size).  Reuses ``qwen_lean_forward._build_spec_batch`` PER LANE (so each lane's
    rows are byte-identical to that lane's single-program batch) and concatenates
    the lanes along the batch dim, re-padding to the global ``Smax``.

    Returns ``(x [rows, Smax, H], positions [rows, Smax], row_map)`` where
    ``row_map[r] = (lane_idx, step_index_in_draft, row_in_lane_slab)``.  Pad rows
    (a lane whose window is shorter than Smax) sit at a far position so the causal
    mask drops them — identical to ``_build_spec_batch``'s own padding.
    """
    per_lane_x: List[torch.Tensor] = []
    per_lane_pos: List[torch.Tensor] = []
    row_map: List[Tuple[int, int, int]] = []
    PAD_POS = 10_000_000
    for li in active:
        lane = lanes[li]
        slab = lane.draft.steps[lane.cursor:lane.cursor + block_steps]
        if not slab:
            continue
        x_l, pos_l = LF._build_spec_batch(draft_lean, lane.code, slab)   # [b, s, H]
        per_lane_x.append(x_l)
        per_lane_pos.append(pos_l)
        for r, st in enumerate(slab):
            row_map.append((li, lane.cursor + r, r))
    if not per_lane_x:
        H = draft_lean.embed.shape[1]
        return (torch.zeros(0, 0, H, device=draft_lean.device),
                torch.zeros(0, 0, dtype=torch.long, device=draft_lean.device),
                row_map)

    Smax = max(t.shape[1] for t in per_lane_x)
    H = per_lane_x[0].shape[2]
    rows_total = sum(t.shape[0] for t in per_lane_x)
    x = torch.zeros(rows_total, Smax, H, device=draft_lean.device,
                    dtype=per_lane_x[0].dtype)
    positions = torch.zeros(rows_total, Smax, dtype=torch.long,
                            device=draft_lean.device)
    off = 0
    for x_l, pos_l in zip(per_lane_x, per_lane_pos):
        b, s, _ = x_l.shape
        x[off:off + b, :s, :] = x_l
        positions[off:off + b, :s] = pos_l
        if s < Smax:
            # extend the pad tail so the shorter lanes' extra columns are causally
            # invisible (a real pad position larger than any real row's position).
            positions[off:off + b, s:] = PAD_POS + torch.arange(
                Smax - s, device=draft_lean.device)
        off += b
    return x, positions, row_map


def _decode_row_ax(state: torch.Tensor, L, op: int) -> int:
    """Decode AX at a query row — MUL/DIV/MOD from the AX nibble band (efficient
    ALU), else the scalar AX_VAL.  Mirrors ``full_native_fast._run_native``."""
    if op in _NIBBLE_AX_OPS:
        return _decode_reg_from_nibbles(state, L, L.AX) & 0xFF
    return _snap(state[L.AX_VAL]) & 0xFF


def _qrow_of(draft_lean, lane: Lane, st: dict) -> int:
    """The query row (last real position) of ``st``'s single-step window — the
    authoritative qrow from ``qwen_lean_forward`` / ``full_native_fast``."""
    subset = draft_lean.subset
    n_store = len(st["store_log"]) if subset.memory else 0
    n_code = len(lane.code) if draft_lean.code_from_memory else 0
    return (1 + n_store + n_code) + len(CAM_REGS)


# ===========================================================================
# THE CROSS-PROGRAM BATCHED STEPPER.
# ===========================================================================
@torch.no_grad()
def run_cross_program(bundle, programs: Sequence[List[isa.Instr]], *,
                      block_steps: int = 32, max_steps: int = 4096,
                      batch_cap: Optional[int] = None,
                      compact: bool = True,
                      labels: Optional[Sequence[str]] = None,
                      throughput: bool = True,
                      throughput_thr: float = 0.0) -> CrossBatchResult:
    """Step B INDEPENDENT programs together on the shared conditional-block VM.

    ``bundle`` is a ``full_native_fast.FullNativeFast`` (or any object exposing
    ``.cond`` [the ConditionalBlockLean run model] and ``.dense_lean`` [the lean
    weights used for drafting + the window build]).  Each program is drafted (free),
    then all still-running lanes are stepped together: their next ``block_steps``
    step-windows are stacked into ONE ``[rows, Smax, H]`` conditional forward, and
    each lane's per-step AX is demuxed back out.

    ``batch_cap`` caps how many lanes go into one forward (VRAM guard); lanes past
    the cap wait for the next round.  ``compact`` drops RETIRED lanes from the
    active set each round (so a batch of ragged-length programs keeps the forward
    dense — no idle rows for finished programs).  ``throughput=True`` also computes
    the union-active-block throughput model (needs ONE extra dense-forward probe on
    the batch's op-set — run on the SAME device as the model; structural, cheap).

    Returns a ``CrossBatchResult`` with per-lane AX traces + verdicts and the
    throughput report.  A lane whose draft was empty (out-of-slice op) is left as
    ``naive_fallback`` for the caller to run through the single-program driver.
    """
    model = bundle.cond
    draft_lean = bundle.dense_lean
    L = draft_lean.QL.L

    lanes = build_lanes(draft_lean, programs, max_steps=max_steps, labels=labels)

    forwards = 0
    max_batch_rows = 0
    while True:
        active = [i for i, l in enumerate(lanes)
                  if not l.done and l.remaining > 0]
        if not active:
            break
        if not compact:
            # keep a fixed lane order but still skip retired lanes for correctness;
            # (compact=False only affects reporting parity, not the row math).
            active = sorted(active)
        if batch_cap is not None and len(active) > batch_cap:
            active = active[:batch_cap]

        x, positions, row_map = _stack_active_slabs(
            draft_lean, lanes, active, block_steps)
        if x.shape[0] == 0:
            break
        x = x.to(model.device)
        positions = positions.to(model.device)
        hidden, _ = model.forward(x, q_positions=positions)
        forwards += 1
        max_batch_rows = max(max_batch_rows, x.shape[0])

        # demux: each row decodes its lane's step AX at that step's query row.
        for r, (li, step_i, _row_in_slab) in enumerate(row_map):
            lane = lanes[li]
            st = lane.draft.steps[step_i]
            qrow = _qrow_of(draft_lean, lane, st)
            state = hidden[r, qrow]
            ax = _decode_row_ax(state, L, st["op"])
            lane.ax_trace.append(ax)

        # advance each active lane's cursor past the steps we just verified.
        for li in active:
            lane = lanes[li]
            slab_len = min(block_steps, lane.remaining)
            lane.cursor += slab_len
            if lane.cursor >= lane.n_steps:
                lane.done = True

    result = CrossBatchResult(
        lanes=lanes, forwards=forwards,
        naive_forwards=sum(l.n_steps for l in lanes),
        max_batch_rows=max_batch_rows)
    if throughput:
        result.throughput = union_throughput_model(
            bundle, lanes, block_steps=block_steps, thr=throughput_thr)
    return result


# ===========================================================================
# THE UNION-ACTIVE-BLOCK THROUGHPUT MODEL (the batching payoff, quantified).
#
# Per layer, the conditional dispatch computes the UNION of FFN units that fire
# across the batch.  Cross-program batching wins when that union is smaller than
# the SUM of the per-lane active sets: the shared forward does ~union work where
# sequential runs do ~sum work.  This measures both from the SAME
# ``conditional_active_units`` machinery the run path uses.
# ===========================================================================
@torch.no_grad()
def union_throughput_model(bundle, lanes: Sequence[Lane], *,
                           block_steps: int = 32, thr: float = 0.0) -> Dict[str, object]:
    """Structural throughput model: per-layer union-active-units across the batch's
    lanes vs the SUM of each lane's own active-units (what running them
    sequentially would touch).  ``union / sum`` per layer is the fraction of work
    the batched forward saves; the aggregate ``sum/union`` is the FFN-work speedup.

    Runs ``perlayer_conditional_sparse.conditional_active_units`` (a dense forward
    on the lean weights — the SAME probe ``build_full_native_fast`` uses) once per
    lane on that lane's first ``block_steps`` windows.  On CPU this is a real dense
    forward (small windows), so keep the sample modest; on GPU it is the streamed
    probe.  Returns per-layer + aggregate numbers.  NOTE: this touches the model, so
    it is NOT called by the CPU-only ``verify_cpu`` path (which has no model)."""
    from . import perlayer_conditional_sparse as PC
    draft_lean = bundle.dense_lean
    live = [l for l in lanes if not l.naive_fallback and l.n_steps > 0]
    if not live:
        return {"n_lanes": 0, "note": "no batchable lanes"}

    per_lane_sets: List[List[set]] = []
    for lane in live:
        slab = lane.draft.steps[:block_steps]
        x_l, pos_l = LF._build_spec_batch(draft_lean, lane.code, slab)
        info = PC.conditional_active_units(draft_lean, x_l, pos_l, thr=thr)
        per_lane_sets.append([set(a.tolist()) for a in info["active_units"]])

    nL = len(per_lane_sets[0])
    sum_active = [0] * nL
    union_active = [set() for _ in range(nL)]
    for lane_sets in per_lane_sets:
        for li in range(nL):
            sum_active[li] += len(lane_sets[li])
            union_active[li] |= lane_sets[li]
    union_counts = [len(u) for u in union_active]

    total_sum = sum(sum_active)
    total_union = sum(union_counts)
    ffn_speedup = (total_sum / total_union) if total_union else 1.0
    return {
        "n_lanes": len(live),
        "block_steps": block_steps,
        "per_layer_sum_active": sum_active,
        "per_layer_union_active": union_counts,
        "total_sum_active": total_sum,
        "total_union_active": total_union,
        "ffn_work_speedup_union_vs_sum": ffn_speedup,
        "note": ("sum = FFN units touched running the lanes SEQUENTIALLY; "
                 "union = units the ONE batched conditional forward touches. "
                 "sum/union is the FFN-work throughput multiplier (>=1; ==1 when "
                 "the lanes are opcode-disjoint, ~=B when they share opcodes)."),
    }


# ===========================================================================
# A PURELY STRUCTURAL throughput model — active blocks approximated by the
# per-lane DYNAMIC OPCODE SET (no model needed).  Used by the CPU verification
# to report the union-vs-sum shape without a forward.  This is a CONSERVATIVE
# proxy for the true union win (real active sets overlap MORE than the raw op
# sets because shared machinery — PC-fetch/dispatch/nibble-recompose — fires for
# every op, so it overlaps 100% across lanes).
# ===========================================================================
def opcode_union_model(lanes: Sequence[Lane]) -> Dict[str, object]:
    """Structural (no-forward) union model keyed on each lane's DYNAMIC OPCODE SET.

    Each lane touches the FFN blocks of the opcodes it actually executes.  Running
    the lanes SEQUENTIALLY touches the SUM of per-lane distinct-op counts (each
    lane pays for its own op-set); the ONE batched forward touches their UNION.
    ``sum/union`` is a conservative proxy for the FFN-work speedup (conservative
    because it ignores the shared per-step dispatch machinery every op fires, which
    overlaps 100% across lanes -> the real conditional union wins MORE).  Zero model
    dependency -> this is the number the CPU self-check prints."""
    live = [l for l in lanes if not l.naive_fallback and l.n_steps > 0]
    if not live:
        return {"n_lanes": 0, "note": "no batchable lanes"}
    per_lane_ops: List[set] = []
    for lane in live:
        ops = {st["op"] for st in lane.draft.steps}
        per_lane_ops.append(ops)
    sum_ops = sum(len(s) for s in per_lane_ops)
    union_ops = set().union(*per_lane_ops)
    speedup = (sum_ops / len(union_ops)) if union_ops else 1.0
    return {
        "n_lanes": len(live),
        "per_lane_op_counts": [len(s) for s in per_lane_ops],
        "sum_distinct_ops": sum_ops,
        "union_distinct_ops": len(union_ops),
        "union_ops": sorted(isa.NAMES.get(o, o) for o in union_ops),
        "opcode_block_speedup_sum_vs_union": speedup,
        "note": ("conservative proxy: FFN-block work ~ distinct-op count. "
                 "sum/union is the sequential-vs-batched op-block work ratio "
                 "(the real conditional-active-unit union wins at least this much "
                 "because the shared dispatch machinery overlaps fully)."),
    }


# ===========================================================================
# CPU-ONLY VERIFICATION — no model build.  Proves the batching LOGIC is correct:
#  (1) each lane's perfect draft == the single-program reference trace,
#  (2) lane retirement/compaction visits every step of every program exactly once,
#      in order (so the demux decodes each lane's whole trace with no gaps/dupes),
#  (3) B=1 reduces to the single-program schedule exactly,
#  (4) the union-vs-sum throughput shape on representative batches.
# ===========================================================================
class _StubSubset:
    def __init__(self, memory=True):
        self.memory = memory
        self.cmp = True
        self.bitwise = True
        self.muldiv = True
        self.name = "stub-full"


# ---------------------------------------------------------------------------
# A MINIMAL fake QL/L layout + embed so ``_build_spec_batch`` (and therefore
# ``_stack_active_slabs``) runs on CPU with NO model build.  This lets the CPU
# verification prove the STACKING is byte-identical, tensor-for-tensor: the row a
# lane occupies in the batched residual equals the row the SOLO single-program
# batch produces for that step.  Given the model is a pure function of the
# per-row residual + positions and the pad rows are causally masked, byte-identity
# of the stacked residual => byte-identity of the batched decode vs the solo run.
# (The real dim VALUES are irrelevant to the isolation claim — only that the SAME
# residual lands on the SAME row — so a compact injective layout suffices.)
# ---------------------------------------------------------------------------
def _fake_layout():
    from .blogspec_layout import NIB_PER_REG
    from .blogspec_memory import ADDR_BITS

    n_reg = len(CAM_REGS)

    class _L:
        # allocate disjoint compact index ranges for every dim family the builder
        # touches; the exact positions don't matter, only that they don't collide.
        ONE = 0
        IS_STORE = 1
        IS_LOAD = 2
        ADDR_BIN = 10                                   # + ADDR_BITS
        QRY_BIN = 10 + ADDR_BITS                        # + ADDR_BITS
        VAL_NIB = QRY_BIN + ADDR_BITS                   # + NIB_PER_REG

    base = _L.VAL_NIB + NIB_PER_REG
    _L.CODE_OP = [base + 2 * k for k in range(64)]
    _L.CODE_IMM = [base + 2 * k + 1 for k in range(64)]
    code_end = base + 2 * 64

    class _QL:
        L = _L
        TOK_NIB = code_end                              # + NIB_PER_REG
        ROLE = code_end + NIB_PER_REG                   # + n_reg
        IS_TOK = code_end + NIB_PER_REG + n_reg

    H = _QL.IS_TOK + 1
    return _QL, H


class _FakeLean:
    """A model-free stand-in exposing exactly the attributes ``_build_spec_batch``
    and the batcher read (``QL``, ``subset``, ``code_from_memory``, ``embed``,
    ``device``).  Used only by the CPU stacking-isolation proof."""
    def __init__(self):
        self.subset = _StubSubset()
        self.code_from_memory = False
        self.device = torch.device("cpu")
        self.QL, H = _fake_layout()
        # a distinctive per-token embed so a misplaced row is caught (each vocab
        # row = its token id broadcast, plus a small dim ramp — injective enough).
        vocab = 512
        self.embed = (torch.arange(vocab, dtype=torch.float32).unsqueeze(1)
                      + 0.001 * torch.arange(H, dtype=torch.float32).unsqueeze(0))


def _verify_stacking_isolation(programs: Sequence[List[isa.Instr]], *,
                               block_steps: int = 8, max_steps: int = 4096,
                               verbose: bool = True) -> bool:
    """Assert the batched residual places each lane's rows byte-for-byte where the
    SOLO single-program batch does — the core byte-identity claim, proven on CPU
    with a fake layout (no model).  Returns True iff every lane's rows in the
    stacked residual equal its own ``_build_spec_batch`` output exactly."""
    lean = _FakeLean()
    lanes = build_lanes(lean, programs, max_steps=max_steps)
    active = [i for i, l in enumerate(lanes) if l.remaining > 0]
    if not active:
        return True
    x, positions, row_map = _stack_active_slabs(lean, lanes, active, block_steps)

    ok = True
    off = 0
    for li in active:
        lane = lanes[li]
        slab = lane.draft.steps[:block_steps]
        x_solo, pos_solo = LF._build_spec_batch(lean, lane.code, slab)
        b, s, _ = x_solo.shape
        x_batch = x[off:off + b, :s, :]
        pos_batch = positions[off:off + b, :s]
        if not torch.equal(x_batch, x_solo):
            ok = False
            if verbose:
                print(f"  [STACK MISMATCH] lane {li}: residual rows differ from solo")
        if not torch.equal(pos_batch, pos_solo):
            ok = False
            if verbose:
                print(f"  [STACK MISMATCH] lane {li}: positions differ from solo")
        off += b
    return ok


def verify_cpu(programs: Sequence[List[isa.Instr]], *, block_steps: int = 8,
               max_steps: int = 4096, verbose: bool = True) -> Dict[str, object]:
    """Prove the cross-program batching LOGIC on CPU, WITHOUT any model build.

    Uses only ``draft_program_lean`` (pure Python — needs just a ``.subset`` stub)
    and the pure-Python control flow of ``run_cross_program`` (drafting +
    retirement + row_map), asserting:

      * DRAFT parity: each lane's ``ref_trace`` == the single-program
        ``isa.interpret`` (function-aware) trace — the AX a solo run must produce.
      * SCHEDULE completeness: the batched stepper's schedule visits (lane, step)
        for EVERY step of EVERY lane exactly once, in order — so the demux decodes
        each lane's whole trace with no gaps/dupes (byte-identity of the batched
        decode reduces to this + the window isolation, which the GPU check confirms
        tensor-for-tensor).
      * RETIREMENT: ragged-length lanes retire independently; B=1 reduces to the
        single-program schedule exactly.
      * THROUGHPUT shape: the opcode-union model (no forward) on the batch.
    """
    from .qwen_lean_forward import interpret_with_functions, uses_functions

    class _StubLean:
        subset = _StubSubset()
        code_from_memory = False        # program-in-data (draft doesn't depend on it)

    stub = _StubLean()
    lanes = build_lanes(stub, programs, max_steps=max_steps)

    report: Dict[str, object] = {"checks": {}, "n_programs": len(programs)}
    ok_all = True

    # (1) DRAFT PARITY -----------------------------------------------------
    draft_ok = True
    for lane in lanes:
        code = lane.code
        ref = (interpret_with_functions(code, max_steps=max_steps)
               if uses_functions(code) else isa.interpret(code, max_steps=max_steps))
        if lane.draft.ref_trace != ref:
            draft_ok = False
            if verbose:
                print(f"  [DRAFT MISMATCH] lane {lane.idx}: "
                      f"draft {lane.draft.ref_trace[:8]}... != ref {ref[:8]}...")
    report["checks"]["draft_parity"] = draft_ok
    ok_all &= draft_ok

    # (2a) WINDOW-STACKING ISOLATION (byte-identity of the batched residual) via a
    #     fake layout — the row a lane occupies in the stacked forward equals the
    #     SOLO single-program row, tensor-for-tensor (no model).
    stack_ok = _verify_stacking_isolation(programs, block_steps=block_steps,
                                          max_steps=max_steps, verbose=verbose)
    report["checks"]["stacking_isolation"] = stack_ok
    ok_all &= stack_ok

    # (2)+(3) SCHEDULE COMPLETENESS + RETIREMENT via the SAME control loop the
    #     real stepper runs (minus the forward): reproduce the visit schedule.
    forwards = 0
    max_rows = 0
    cursors = {l.idx: 0 for l in lanes}
    done = {l.idx: l.naive_fallback for l in lanes}
    nsteps = {l.idx: l.n_steps for l in lanes}
    visited: Dict[int, List[int]] = {l.idx: [] for l in lanes}
    while True:
        active = [l.idx for l in lanes
                  if not done[l.idx] and (nsteps[l.idx] - cursors[l.idx]) > 0]
        if not active:
            break
        rows_this = 0
        for li in active:
            slab_len = min(block_steps, nsteps[li] - cursors[li])
            for r in range(slab_len):
                visited[li].append(cursors[li] + r)
            rows_this += slab_len
            cursors[li] += slab_len
            if cursors[li] >= nsteps[li]:
                done[li] = True
        forwards += 1
        max_rows = max(max_rows, rows_this)

    schedule_ok = True
    for lane in lanes:
        if lane.naive_fallback:
            continue
        expect = list(range(lane.n_steps))
        if visited[lane.idx] != expect:
            schedule_ok = False
            if verbose:
                print(f"  [SCHEDULE MISMATCH] lane {lane.idx}: "
                      f"visited {visited[lane.idx][:8]} != {expect[:8]}")
    report["checks"]["schedule_complete"] = schedule_ok
    ok_all &= schedule_ok
    report["forwards"] = forwards
    report["naive_forwards"] = sum(l.n_steps for l in lanes)
    report["max_batch_rows"] = max_rows
    report["forwards_saved"] = (report["naive_forwards"] / forwards) if forwards else 0.0

    # (4) THROUGHPUT SHAPE (opcode-union, no forward) ----------------------
    report["throughput_opcode_model"] = opcode_union_model(lanes)

    # per-lane summary
    report["lanes"] = [
        {"idx": l.idx, "steps": l.n_steps, "naive_fallback": l.naive_fallback,
         "ref_ax_final": (l.draft.ref_trace[-1] if l.draft.ref_trace else None),
         "distinct_ops": sorted({isa.NAMES.get(st["op"], st["op"])
                                 for st in l.draft.steps})
         if l.draft.steps else []}
        for l in lanes]

    report["ok"] = bool(ok_all)
    if verbose:
        _print_cpu_report(report)
    return report


def _print_cpu_report(rep: Dict[str, object]) -> None:
    print("=" * 74)
    print("CROSS-PROGRAM BATCHING — CPU-ONLY LOGIC VERIFICATION (no model build)")
    print("=" * 74)
    print(f"programs = {rep['n_programs']}")
    for k, v in rep["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  batched forwards = {rep['forwards']}  "
          f"(sequential per-step forwards = {rep['naive_forwards']}) "
          f"-> {rep['forwards_saved']:.1f}x forwards saved")
    print(f"  widest batched forward = {rep['max_batch_rows']} rows")
    tm = rep["throughput_opcode_model"]
    if tm.get("n_lanes"):
        print(f"  opcode-union throughput: sum distinct ops = {tm['sum_distinct_ops']}, "
              f"union = {tm['union_distinct_ops']} "
              f"-> {tm['opcode_block_speedup_sum_vs_union']:.2f}x op-block work saved")
        print(f"    union op-set = {tm['union_ops']}")
    print(f"\n  overall: {'PASS' if rep['ok'] else 'FAIL'}")


# ===========================================================================
# REPRESENTATIVE CPU SELF-CHECK BATCHES (small programs — fast on CPU).
# ===========================================================================
def _cpu_selfcheck_batches() -> Dict[str, List[List[isa.Instr]]]:
    A = isa.assemble
    add = A([("IMM", 7), ("PSH", 0), ("IMM", 35), ("ADD", 0), ("HALT", 0)])
    sub = A([("IMM", 50), ("PSH", 0), ("IMM", 8), ("SUB", 0), ("HALT", 0)])
    mul = A([("IMM", 6), ("PSH", 0), ("IMM", 7), ("MUL", 0), ("HALT", 0)])
    div = A([("IMM", 100), ("PSH", 0), ("IMM", 7), ("DIV", 0), ("HALT", 0)])
    mod = A([("IMM", 100), ("PSH", 0), ("IMM", 7), ("MOD", 0), ("HALT", 0)])
    short_loop = A([("IMM", 3), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)])
    long_loop = A([("IMM", 30), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)])
    orx = A([("IMM", 0xF0), ("PSH", 0), ("IMM", 0x0F), ("OR", 0), ("HALT", 0)])
    return {
        # B=1 must reduce to the single-program schedule exactly.
        "B1_single_add": [add],
        # all-identical: MAX union win (every lane the same op-set).
        "all_identical_add": [add, add, add, add],
        # shared opcode distribution (arith mix): high union win.
        "shared_arith_mix": [add, sub, mul, div, mod],
        # ragged lengths: retirement (short + long loops together).
        "ragged_lengths": [short_loop, short_loop, long_loop, add],
        # all-disjoint op families: MIN union win (little overlap beyond dispatch).
        "disjoint_ops": [add, mul, orx, div],
        # a bigger mixed batch.
        "mixed_batch": [add, sub, mul, div, mod, orx, short_loop, long_loop],
    }


def run_cpu_selfcheck(block_steps: int = 8, verbose: bool = True) -> Dict[str, object]:
    """Run ``verify_cpu`` over the representative batches (B=1, all-identical,
    shared-mix, ragged/retirement, disjoint, mixed) and summarise.  Pure CPU."""
    batches = _cpu_selfcheck_batches()
    out: Dict[str, object] = {}
    all_ok = True
    for name, progs in batches.items():
        if verbose:
            print(f"\n### batch = {name} (B={len(progs)}) ###")
        rep = verify_cpu(progs, block_steps=block_steps, verbose=verbose)
        out[name] = rep
        all_ok &= rep["ok"]
    if verbose:
        print("\n" + "=" * 74)
        print(f"CPU SELF-CHECK: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
        print("=" * 74)
    out["_all_ok"] = all_ok
    return out


# ===========================================================================
# DEFERRED GPU CHECKLIST — the end-to-end byte-exact + wall-clock throughput
# checks that REQUIRE a real ConditionalBlockLean forward (no GPU here).
# ===========================================================================
def deferred_gpu_checklist() -> str:
    """Return the exact commands to run later on a quiet GPU to confirm the batched
    forward is byte-exact and measures the throughput multiplier."""
    return _DEFERRED_GPU_CHECKLIST


_DEFERRED_GPU_CHECKLIST = """\
DEFERRED GPU CHECKLIST (run on a quiet card — DO NOT run concurrently with a
model-building agent; the FullNativeFast build alone is ~15 GB dense on CPU +
active block on GPU):

  # 0. sanity: the CPU-only logic check needs no GPU and should already pass.
  python -m c4_min.cross_program_batch          # runs run_cpu_selfcheck()

  # 1. build the FullNativeFast bundle ONCE and run the batched byte-exact gate.
  python -m c4_min.cross_program_batch --gpu --device cuda:0

  # 2. or from Python (name the exact call the GPU verify makes):
  #    from c4_min import full_native_fast as FNF, cross_program_batch as CPB, isa
  #    bundle = FNF.build_full_native_fast(device='cuda:0')       # ~15 min probe
  #    verify_gpu_byte_exact(bundle) asserts, for each representative batch:
  #      - run_cross_program(bundle, progs) per-lane ax_trace == the SOLO
  #        bundle.run(prog).ax_trace  (byte-identical, EVERY lane, EVERY step)
  #      - and == isa.interpret(prog)  (the word reference)
  #    then measures wall-clock:  batched forwards vs B * solo forwards.
"""


@torch.no_grad()
def verify_gpu_byte_exact(bundle, *, block_steps: int = 32,
                          verbose: bool = True) -> Dict[str, object]:
    """GPU/model gate (NOT run here): assert the batched cross-program run is
    byte-identical, per lane, to the SOLO single-program native driver AND to
    ``isa.interpret``.  Run this ONLY on a quiet GPU (it forwards the conditional
    model).  Kept in this module so the deferred checklist is a single call."""
    batches = _cpu_selfcheck_batches()
    results: Dict[str, object] = {}
    all_ok = True
    for name, progs in batches.items():
        res = run_cross_program(bundle, progs, block_steps=block_steps,
                                throughput=True)
        lane_ok = True
        for lane in res.lanes:
            if lane.naive_fallback:
                continue
            solo = bundle.run(lane.code, block_steps=block_steps)
            ref = isa.interpret(lane.code, max_steps=4096)
            if lane.ax_trace != solo.ax_trace:
                lane_ok = False
                if verbose:
                    print(f"  [{name}] lane {lane.idx}: batched != solo "
                          f"({lane.ax_trace[:6]} vs {solo.ax_trace[:6]})")
            if lane.draft.ref_trace != ref:
                lane_ok = False
        results[name] = {
            "byte_exact_vs_solo": lane_ok,
            "n_pass": res.n_pass, "n_lanes": len(res.lanes),
            "forwards": res.forwards, "naive_forwards": res.naive_forwards,
            "throughput": res.throughput}
        all_ok &= lane_ok
        if verbose:
            tm = res.throughput
            sp = tm.get("ffn_work_speedup_union_vs_sum", 1.0) if tm else 1.0
            print(f"  [{'PASS' if lane_ok else 'FAIL'}] {name:20s} "
                  f"B={len(progs)} forwards={res.forwards} "
                  f"(seq={res.naive_forwards}) union-FFN-speedup={sp:.2f}x")
    results["_all_ok"] = all_ok
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Cross-program batching for the native-fast conditional VM.")
    ap.add_argument("--gpu", action="store_true",
                    help="run the GPU byte-exact gate (builds FullNativeFast; "
                         "DO NOT run while another agent holds the GPU)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block-steps", type=int, default=8)
    a = ap.parse_args()
    if a.gpu:
        from . import full_native_fast as FNF
        print(f"[gpu] building FullNativeFast on {a.device} ...", flush=True)
        bundle = FNF.build_full_native_fast(device=a.device)
        verify_gpu_byte_exact(bundle, block_steps=max(a.block_steps, 32))
        print("\n" + deferred_gpu_checklist())
    else:
        run_cpu_selfcheck(block_steps=a.block_steps)
        print("\n" + deferred_gpu_checklist())
