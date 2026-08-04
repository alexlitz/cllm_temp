"""STEP 2 — draft-driven per-step BLOCK SKIP for the c4_min pure-forward VM.

The c4_min driver runs each VM step as ONE ``model.forward`` over the growing
token stream, and NOTHING of the residual survives a step except the emitted
token frame (the argmax-decoded PC/AX/SP/BP/STK).  The STEP-1 SOUND greedy
ablation (``_step_block_skip_greedy``) proved that, per opcode, only a SMALL set
of blocks change those decoded registers (min 3 / mean 8.1 / max 16 of 238);
running the rest only scribbles dead scratch that the next step's fresh re-embed
discards.

So: from the DECODED opcode (which the driver ALREADY knows — it fetches
``code[cur_pc].op`` for its own store bookkeeping), run ONLY that opcode's live
blocks and SKIP the rest (pass the residual straight through).  Byte-exact iff the
schedule is a SUPERSET of the op's true (multi-block) decode-live set — VERIFIED
end-to-end against the full 238-block driver on a corpus (``_step_block_skip_verify``).

IMPORTANT (honesty): the SOUND set must be found by CUMULATIVE removal, not
single-block ablation — two blocks can each be individually skippable yet not
jointly (IMM: ``ax-byte-nib`` and ``imm-ax-nib`` both write AX; drop either -> AX
survives, drop both -> AX=0).  The schedule below is the cumulative-sound set.

DIV/MOD are OPERAND-DEPENDENT: base-16 long division fires different per-nibble
iteration blocks per dividend, so a fixed subset is unsafe.  Their static schedule
runs the WHOLE contiguous ``alu-div*`` span (+ the ALU plumbing) — the same span
``block_moe_divmod`` already proves byte-exact to gate.  (Union-live over an
operand battery is only ~23-25 blocks, so a data-dependent DIV/MOD schedule could
skip more; kept conservative here for a STATIC, CUDA-graphable per-op shape.)

Gate: ``C4_STEP_BLOCK_SKIP`` (install-time opt-in, default OFF).  OFF -> the
driver runs the unmodified 238-block forward (golden byte-identical, fingerprint
unchanged).  The schedule is a runtime COMPUTE skip; it changes NO stored weight.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Set

import torch

from . import isa


def step_block_skip_enabled() -> bool:
    return os.environ.get("C4_STEP_BLOCK_SKIP", "0") == "1"


# ---------------------------------------------------------------------------
# #702 KV-BACKED STACK.  A pop-consuming op (ADD/SUB/MUL/DIV/MOD, the bitwise/
# shift/cmp set, SI/SC) reads its stack operand from ``MEM[SP]``.  The model's
# stack-pop CAM head (``_bake_stack_pop_head``, query=SP address, enable=IS_POP,
# dest=STACK0) content-addresses that store row in the persistent KV memory log —
# the SAME log SI/SC/PSH write — and relays it to the STACK0 nibble band the ALU
# reads.  That is a §Memory read at address SP: arbitrary stack DEPTH, latest-
# write-wins, no 1-slot mirror.
#
# The block-skip schedule below historically OMITTED that head from the ALU ops
# (only LEV ran it), so the ALU consumed the teacher-forced STACK0 MIRROR instead
# — which tracks exactly ONE parked cell.  A width>=2 dot parks a partial product
# (stack depth 2) and the mirror loses it, so every intermediate ADD that folds a
# parked partial diverged (the #702 wall; finals happened to re-converge).  Adding
# the KV stack-pop chain (``stack-prep`` arms IS_POP + clears STACK0, ``stack-pop-
# cam`` gathers MEM[SP] into STACK0) to every pop-consumer makes the pop a true
# KV read -> byte-exact at EVERY intermediate for arbitrary stack depth.
#
# Gate: ``C4_KV_STACK`` (default ON — the KV stack is the correct arbitrary-depth
# semantics and is byte-exact on the depth-1 battery too, since a depth-1 mirror
# value EQUALS the MEM[SP] the head recalls).  Set ``C4_KV_STACK=0`` to restore
# the historical 1-slot-mirror schedule (depth-1 byte-identical; depth>=2 walls).
# ---------------------------------------------------------------------------
_KV_STACK_CHAIN = ["stack-prep", "stack-pop-cam"]
_POP_CONSUMER_OPS = (
    isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
    isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR,
    isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE,
    isa.SI, isa.SC,
)


def kv_stack_enabled() -> bool:
    return os.environ.get("C4_KV_STACK", "1") == "1"


# ---------------------------------------------------------------------------
# The SOUND per-op live-block NAME schedule (from the cumulative-greedy ablation,
# ``_step_block_skip_greedy`` on the code_size=32 build).  Resolved BY NAME so it
# is robust to build/dim changes; names absent in a build are ignored.
# ---------------------------------------------------------------------------
_LIVE_NAMES: Dict[int, List[str]] = {
    isa.IMM: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "imm-nib-fetch", "dispatch", "imm-ax-nib"],
    isa.LEA: ["ingest+recompose", "pc-fetch", "opcode-decode", "imm-nib-fetch",
              "dispatch", "lea-q-snap", "lea-addr-nib", "ax-byte-nib"],
    # JMP: PC = IMM_CLEAN (leak-free target) is applied in the DISPATCH block, so
    # a non-zero target needs imm-nib-fetch + imm-clean + dispatch.
    isa.JMP: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "imm-nib-fetch", "imm-clean", "dispatch"],
    isa.PSH: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "dispatch"],
    # branches read IMM_CLEAN for the taken target -> need imm-nib-fetch + imm-clean;
    # dispatch handles the not-taken PC+=1.  (branch-delta applies the taken delta.)
    isa.BZ:  ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "imm-nib-fetch", "imm-clean", "dispatch", "branch-delta"],
    isa.BNZ: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "imm-nib-fetch", "imm-clean", "dispatch", "branch-delta"],
    # ADD/SUB (operand-UNION over carry-triggering pairs): ALL 4 byte lanes can fire.
    isa.ADD: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "alu-expand", "alu-add-b0", "alu-add-b1", "alu-add-b2",
              "alu-add-b3", "ax-mux", "dispatch"],
    isa.SUB: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "alu-expand", "alu-sub-b0", "alu-sub-b1", "alu-sub-b2",
              "alu-sub-b3", "ax-mux", "dispatch"],
    isa.MUL: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "alu-mul-products", "alu-mul-split", "alu-mul-round1",
              "alu-mul-apply", "ax-mux", "dispatch"],
    isa.OR:  ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "bw-bitplanes", "bw-combine", "bw-recompose",
              "dispatch", "ax-nib-split"],
    isa.XOR: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "bw-bitplanes", "bw-combine", "bw-recompose",
              "dispatch", "ax-nib-split"],
    # AND (operand-union): bw-bitplanes + ax-nib-split ARE needed for some patterns.
    isa.AND: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "bw-bitplanes", "bw-combine", "bw-recompose",
              "dispatch", "ax-nib-split"],
    # SHL/SHR (operand-union): SHL also needs ax-byte-nib for some shift amounts.
    isa.SHL: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "bw-tshift-SHL-load", "bw-tshift-SHL-amt1",
              "bw-tshift-SHL-amt2", "bw-tshift-SHL-coarse", "bw-tshift-SHL-fprod",
              "bw-tshift-SHL-fpeel", "bw-tshift-SHL-asm", "bw-tshift-recompose",
              "bw-recompose", "dispatch", "ax-nib-split", "ax-byte-nib"],
    isa.SHR: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "bw-tshift-SHR-load", "bw-tshift-SHR-amt1",
              "bw-tshift-SHR-amt2", "bw-tshift-SHR-coarse", "bw-tshift-SHR-fprod",
              "bw-tshift-SHR-fpeel", "bw-tshift-SHR-asm", "bw-tshift-recompose",
              "bw-recompose", "dispatch", "ax-nib-split", "ax-byte-nib"],
    # compares (operand-union over TRUE/FALSE): all need cmp-compute; the ordering
    # ones (LT/GT/LE/GE) also need cmp-finalize (the signed finalize).
    isa.EQ:  ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "cmp-compute", "dispatch", "ax-nib-split", "ax-byte-nib"],
    isa.NE:  ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "cmp-compute", "dispatch", "ax-nib-split", "ax-byte-nib"],
    isa.LT:  ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "cmp-compute", "cmp-finalize", "dispatch",
              "ax-nib-split", "ax-byte-nib"],
    isa.GT:  ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "cmp-compute", "cmp-finalize", "dispatch",
              "ax-nib-split", "ax-byte-nib"],
    isa.LE:  ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "cmp-compute", "cmp-finalize", "dispatch",
              "ax-nib-split", "ax-byte-nib"],
    isa.GE:  ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "cmp-compute", "cmp-finalize", "dispatch",
              "ax-nib-split", "ax-byte-nib"],
    isa.LI:  ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "mem-prep", "mem-cam", "dispatch"],
    isa.LC:  ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "mem-prep", "mem-cam", "dispatch"],
    isa.SI:  ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "dispatch"],
    isa.SC:  ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "dispatch"],
    # JSR: PC = IMM_CLEAN (call target) applied in dispatch -> needs imm-clean chain.
    isa.JSR: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "imm-nib-fetch", "imm-clean", "dispatch"],
    isa.ENT: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "imm-nib-fetch", "imm-clean", "dispatch"],
    isa.ADJ: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "imm-nib-fetch", "imm-clean", "dispatch"],
    # LEV: SP=BP; BP=MEM[BP]; PC=MEM[BP+4] (return addr).  The two stack loads run
    # through the pop/lev-CAM chain -> needs pop-addr/stack-prep/lev-addr4/
    # stack-pop-cam (the greedy's trivial LEV-0 frame masked these).
    isa.LEV: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
              "pop-addr", "stack-prep", "lev-addr4", "stack-pop-cam", "dispatch"],
    isa.HALT: ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
               "dispatch"],
}
# ops whose live set is OPERAND-DEPENDENT -> run the whole named divmod SPAN plus
# the ALU housekeeping (conservative, static shape).
_SPAN_OPS = {isa.DIV, isa.MOD}
_SPAN_COMMON = ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
                "pop-addr", "ax-mux", "dispatch"]


def build_live_index(model, L) -> Dict[Optional[int], List[int]]:
    """Resolve the per-op live-block NAME schedule into per-op INDEX lists for the
    built model.  Returns ``{op: sorted[block_idx]}`` plus a ``None`` key = the
    FULL block list (every block) for any op not in the table (safe fallback)."""
    names = list(getattr(L, "_block_names", []))
    name_to_idx: Dict[str, List[int]] = {}
    for i, n in enumerate(names):
        name_to_idx.setdefault(n, []).append(i)
    nb = len(model.blocks)

    def idxs(name_list: List[str]) -> Set[int]:
        s: Set[int] = set()
        for nm in name_list:
            s.update(name_to_idx.get(nm, []))
        return s

    div_idx = [i for i, n in enumerate(names) if n.startswith("alu-div")]
    div_span = set(range(min(div_idx), max(div_idx) + 1)) if div_idx else set()

    # #702 KV-backed stack: every pop-consuming op reads MEM[SP] via the stack-pop
    # CAM head (arbitrary depth) rather than the 1-slot STACK0 mirror.  Inject the
    # ``stack-prep`` + ``stack-pop-cam`` chain into those ops' live sets when the
    # gate is on.  ``build_live_index`` already sorts each op's blocks by index, so
    # the chain lands in its correct position (pop-addr(8) -> stack-prep(9) ->
    # stack-pop-cam(11) -> alu-expand(16) ...); the head is IS_POP-gated so it is a
    # no-op on a non-pop step.
    kv_chain = idxs(_KV_STACK_CHAIN) if kv_stack_enabled() else set()

    # WIDE-INGEST (C4_INGEST_WIDE): the 1-query/1-KV wide ingest restructures block 0's
    # frame ingest into three PREPENDED blocks (``wide-preroute`` computes the per-role
    # role⊙nibble product, ``wide-gather`` is the single wide-value head that does the
    # scaled-concat gather, ``wide-snap`` rescales+re-quantises).  They run BEFORE
    # ``ingest+recompose`` and are part of EVERY step's frame ingest, so the per-op
    # block-skip schedule (authored for the stock 20-head ``ingest+recompose`` block)
    # MUST include them or the ingest is skipped and every op decodes garbage.  Injected
    # into every op's live set (and the None/full fallback already covers them).  Absent
    # in a stock build -> ``idxs`` returns {} -> byte-identical to the golden schedule.
    wide_ingest = idxs(["wide-preroute", "wide-gather", "wide-snap"])

    out: Dict[Optional[int], List[int]] = {None: list(range(nb))}
    for op, nms in _LIVE_NAMES.items():
        base = idxs(nms) | wide_ingest
        if op in _POP_CONSUMER_OPS:
            base = base | kv_chain
        out[op] = sorted(base)
    for op in _SPAN_OPS:
        base = idxs(_SPAN_COMMON) | div_span | wide_ingest
        if op in _POP_CONSUMER_OPS:
            base = base | kv_chain
        out[op] = sorted(base)
    return out


class StepBlockSkipRunner:
    """Runs ``model.blocks`` applying ONLY the current opcode's live blocks.

    ``forward(x, op)`` applies, in block order, exactly the blocks in
    ``live_index[op]`` (or the full list when ``op`` is unknown), passing the
    residual straight through every skipped block.  ``x`` is the step's embedded +
    overlaid residual ([1, S, D]); the return is the post-stack residual whose
    query row the driver decodes.  Byte-exact for decode iff the schedule is a
    superset of the op's true DECODE-live set (verified by the corpus gate)."""

    def __init__(self, model, L):
        self.model = model
        self.blocks = model.blocks
        self.live_index = build_live_index(model, L)
        self.n_blocks = len(model.blocks)
        # precompute the per-op live-mask (bool list) so forward is a plain loop.
        self._live_masks: Dict[Optional[int], List[bool]] = {}
        for op, live in self.live_index.items():
            m = [False] * self.n_blocks
            for i in live:
                m[i] = True
            self._live_masks[op] = m

    def live_count(self, op: Optional[int]) -> int:
        return len(self.live_index.get(op, self.live_index[None]))

    def forward(self, x: torch.Tensor, op: Optional[int]) -> torch.Tensor:
        mask = self._live_masks.get(op)
        if mask is None:
            for blk in self.blocks:
                x = blk(x)
            return x
        for bi, blk in enumerate(self.blocks):
            if mask[bi]:
                x = blk(x)
        return x


__all__ = ["step_block_skip_enabled", "StepBlockSkipRunner", "build_live_index"]
