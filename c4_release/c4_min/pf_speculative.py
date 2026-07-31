"""PERFECT-DRAFT SPECULATION for the SPARSE pure-forward whole-VM model.

CHK-1 closes here: the 104 deep-loop programs (gcd, loop_sum, rec_factorial,
loop_countdown, and the nested/rec_* family) that TIMEOUT in the token-by-token
pure-forward driver — not because they diverge (the HYBRID VM halts them
correctly at step-cap 10000) but because a ~500-step loop is intractable
one-``model.forward``-per-step at ~5 steps/sec.

The fix is the BLOG_SPEC §Speculation win, adapted to the pure-forward
STATE-MACHINE model (which decodes the register file from the block-stack hidden
output, NOT from an LM-head next-token predictor like ``nibble_speculative``):

  1. DRAFT (zero model forwards).  The reference ISA (``ref_interpret``'s exact
     transition + the driver's ``_build_frame`` / store-bookkeeping) runs the
     whole program to completion in pure Python — even an 8400-step loop drafts
     in a fraction of a second — and emits the EXACT per-step 30-token frame
     stream the pure-forward driver would emit, plus the store_log (KV memory
     writes) keyed identically.  This is a *perfect* draft: every token is what
     the model, driven token-by-token, would have produced.

  2. VERIFY block-wise (a handful of batched forwards).  We feed the drafted
     stream to the SPARSE model exactly as the KV-cached driver does — a fixed
     per-block incremental KV cache with bounded (softmax1+ALiBi) eviction — BUT
     because the whole stream is known ahead of time we FREEZE the block's frames
     in bulk (their K/V is causally independent of the later step-query rows) and
     run MANY step-query rows in ONE batched ``forward_hidden_cached``, confirming
     at every step-query row that the model's decoded register state == the
     draft's next-step registers.  That equality is EXACTLY what greedy
     token-by-token autoregression would emit at that row (the query row's overlay
     + the cache of the frozen prefix is identical to the naive driver's last-row
     read), so the model does the computing; speculation only verifies it in
     parallel.  A program PASSes iff the model ACCEPTS the full drafted stream
     (every step-query row matches) AND the decoded final AX == expected.

Why this is still 100% autoregressive: the draft is only a guess.  The verify
proves the MODEL ITSELF, given the correct prefix (the cache of the drafted
frozen frames), produces the register state the draft predicted at every step —
byte-for-byte what the token-by-token KV-cached driver produces (spot-checked in
``spotcheck_vs_cached``).  We NEVER accept on the draft alone.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


def _draft_cmp32() -> bool:
    """DRAFT cmp/shift OPERAND width selector.  DEFAULT OFF (``C4_DRAFT_CMP32`` unset)
    -> the historical 8-bit cmp/bitwise draft (``v & 0xFF``), byte-identical to every
    prior corpus draft (the 1096/edge corpus operands are all < 256).  ON
    (``C4_DRAFT_CMP32=1``) -> the draft compares the FULL 32-bit SIGNED operands
    (``s32(pop_val)`` vs ``s32(ax)``), matching the MODEL's 32-bit signed cmp
    (proven byte-exact vs the model in ``_agent_cmp_width_probe`` / ``_agent_cmp_signed_probe``:
    255<256, 2048<=0, x>=0 all model-correct).  REQUIRED for programs whose loop
    counters / operands cross the byte boundary (doom's ``i < CIRC==256`` init loops):
    the 8-bit draft mis-decodes ``2048 <= 0`` as true (2048&0xFF==0) and never exits,
    diverging from the model which correctly halts the loop.  Additive + gated: OFF
    reproduces the exact old draft, so the whole existing corpus is byte-identical."""
    return os.environ.get("C4_DRAFT_CMP32", "0") == "1"


def _s32(v: int) -> int:
    """Interpret ``v`` as a signed 32-bit two's-complement integer (C4's ``int``)."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v

from . import isa
from . import blogspec_vocab as V
from . import nibble_filesys as _FS   # OPEN/READ/CLOS/PRTF via the TOOL_CALL boundary
from .nibble_pure_forward import (
    N_ROLES, _snap_lane, _FRAME_ROLE_SLOTS, _MEM_MARKER_LOCAL, _address_bits,
)
from .nibble_pure_forward_complete import (
    PureForwardCompleteLayout, IMM_NIBS, _build_frame, _decode_reg_from_nibbles,
    _mem_top, ADJ, CODE_ADDR_BITS, _pf_cfm_enabled,
)
import c4_min.nibble_pure_forward_complete as _PFC
from .nibble_pure_forward_cached import (
    apply_overlay_window, BlockKVCacheBatched, evict_all_blocks_fused,
    evict_all_blocks_scheduled,
)
from .nibble_evict_schedule import (
    build_eviction_schedule, sorted_evict_frames, positions_new_at,
)
from .blogspec_layout import NIB_PER_REG


# ===========================================================================
# O(1)-DECODE OVERLAY.  ``apply_overlay_window`` re-writes the PROGRAM-IN-DATA
# (every ``code`` op's CODE_OP/CODE_IMM/CODE_IMM_NIB dims) into EVERY row of the
# span — an O(rows * code_len) scalar-write loop.  Measured: ~1.06 ms/row, of
# which ~97% is that row-INVARIANT code re-scan.  Over a whole program this is
# O(total_rows * code_len) == O(stream * C), the deep-program stall (fib12's 8369
# steps -> ~250k rows -> minutes of pure-Python overlay).
#
# But the code-in-data residual is IDENTICAL for every row (it does not depend on
# the absolute position), so it can be built ONCE as a [D] vector and BROADCAST-
# added to all rows in one tensor op.  Only the per-position frame-role / store
# tags are row-dependent, and those are a handful of scalar writes per row (O(1)
# per row -> O(rows) total, position-independent).  This turns the per-block
# overlay from O(rows * C) into O(rows + C) and the whole-program overlay from
# O(stream * C) into O(stream + blocks * C) — the O(1)-per-row decode.
#
# Byte-identity: the produced residual at every (row, dim) is the SAME float
# ``apply_overlay_window`` writes (same ``float(op)`` / nibble values, same role
# one-hots), just assembled by broadcast + a lean per-row loop instead of a
# per-row full re-scan.  Proven L-inf==0 vs ``apply_overlay_window`` in
# ``spotcheck_vs_cached`` / the runner's ``--check-overlay`` gate.
# ===========================================================================
def build_code_vec(code: List[isa.Instr], L: PureForwardCompleteLayout,
                   D: int, device, dtype=torch.float32
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The ROW-INVARIANT program-in-data + ONE overlay as ``(idx, vals)``.

    ``idx`` [K] long — the residual dims the code-in-data overlay WRITES (ONE +
    every CODE_OP/CODE_IMM/CODE_IMM_NIB dim).  ``vals`` [K] — the values it writes.
    Built ONCE per program (the code never changes across steps); the fast overlay
    ASSIGNS ``x[:, idx] = vals`` on every row (overwriting the embedding at those
    dims, exactly as ``apply_overlay_window`` does with ``x[...] = v`` — NOT ``+=``,
    since the embedding is non-zero at these dims).
    """
    idx: List[int] = [L.ONE]
    vals: List[float] = [1.0]
    if not _pf_cfm_enabled():
        # BAKED path: the program is a row-INVARIANT program-in-data band.
        for k, ins in enumerate(code):
            idx.append(L.CODE_OP[k]); vals.append(float(ins.op))
            idx.append(L.CODE_IMM[k]); vals.append(float(ins.imm))
            for j, nv in enumerate(V.nibbles_of_value(ins.imm & 0xFFFFFFFF, IMM_NIBS)):
                idx.append(L.CODE_IMM_NIB[k] + j); vals.append(float(nv))
    # CODE-FROM-MEMORY: the program lives in the KV as per-position CODE frames (NOT a
    # row-invariant band) — written per-row by ``apply_overlay_window_fast`` — so the
    # row-invariant code_vec carries only ONE.
    return (torch.tensor(idx, device=device, dtype=torch.long),
            torch.tensor(vals, device=device, dtype=dtype))


def apply_overlay_window_fast(x_win: torch.Tensor, w_start: int, L,
                              store_log: Dict[int, Tuple[int, int]],
                              code_vec: Tuple[torch.Tensor, torch.Tensor],
                              query_rows: Optional[List[int]] = None,
                              code=None, code_off: int = 0) -> None:
    """O(rows + code) in-place overlay of ``x_win`` ([1, W, D]) — byte-identical to
    ``apply_overlay_window`` but with the row-invariant code-in-data ASSIGNED by ONE
    broadcast indexed-write instead of a per-row re-scan.

    ``code_vec`` is ``build_code_vec(code, L, D)`` -> ``(idx, vals)`` (precomputed
    once).  ``query_rows`` (window-local indices) are re-tagged with all-ROLE
    one-hots (the per-step query tag); if None, only the LAST row is a query row
    (the single-step window contract of
    ``apply_overlay_window(is_last_row_query=True)``).

    CODE-FROM-MEMORY (``code_off > 0``): positions ``1 .. code_off`` are leading CODE
    frames (written per-row from ``code``) and the register/store frame math shifts by
    ``code_off``.  ``code_vec`` then carries only ONE (no baked program-in-data band).
    """
    _cfm = code_off > 0
    W = x_win.shape[1]
    # 1) row-INVARIANT program-in-data + ONE: ONE broadcast ASSIGN over the code
    # dims (overwrites the embedding at those dims, exactly like the slow overlay's
    # ``x[...] = v``).  O(W*K) writes but as ONE vectorised op, not a Python loop.
    code_idx, code_vals = code_vec
    x_win[0, :, code_idx] = code_vals
    # 2) per-position frame roles / store tags (O(1) per row -> O(W) total).
    for wi in range(W):
        p = w_start + wi
        if p == 0:
            continue                       # BOS row carries no frame roles
        if _cfm and 1 <= p <= code_off:
            # CODE frame for instruction (p-1): the address-keyed KV code memory.
            ins = code[p - 1]
            x_win[0, wi, L.IS_CODE] = 1.0
            x_win[0, wi, L.IS_FRAME_BYTE] = 0.0
            for b in range(CODE_ADDR_BITS):
                x_win[0, wi, L.CODE_KEY_BIN + b] = float(((p - 1) >> b) & 1)
            x_win[0, wi, L.CODE_OPV] = float(ins.op)
            for j, nv in enumerate(V.nibbles_of_value(ins.imm & 0xFFFFFFFF, IMM_NIBS)):
                x_win[0, wi, L.CODE_IMM_NIB_MEM + j] = float(nv)
            continue
        f = (p - 1 - code_off) // V.FRAME_LEN
        local = (p - 1 - code_off) % V.FRAME_LEN
        if local in _FRAME_ROLE_SLOTS:
            role = _FRAME_ROLE_SLOTS[local]
            x_win[0, wi, L.ROLE + role] = 1.0
            x_win[0, wi, L.IS_FRAME_BYTE] = 1.0
        if local == _MEM_MARKER_LOCAL and f in store_log:
            addr, val = store_log[f]
            x_win[0, wi, L.IS_STORE] = 1.0
            x_win[0, wi, L.IS_FRAME_BYTE] = 0.0
            for b, bit in enumerate(_address_bits(addr)):
                x_win[0, wi, L.ADDR_BIN + b] = bit
            for j, nv in enumerate(V.nibbles_of_value(val & 0xFFFFFFFF, NIB_PER_REG)):
                x_win[0, wi, L.VAL_NIB + j] = float(nv)
    # 3) query-row all-ROLE tag (the per-step query overlay).
    if query_rows is None:
        for role in range(N_ROLES):
            x_win[0, -1, L.ROLE + role] = 1.0
    else:
        for wi in query_rows:
            for role in range(N_ROLES):
                x_win[0, wi, L.ROLE + role] = 1.0


# ===========================================================================
# 1. THE PERFECT DRAFT — the reference VM run, materialised as the exact token
#    stream + store_log the pure-forward DRIVER would emit.  ZERO model forwards.
# ===========================================================================
@dataclass
class PFDraft:
    tokens: List[int]                          # BOS + one 30-token frame per step
    frames: List[Dict[str, int]]               # per-step {pc,ax,sp,bp,stk,...}
    store_log: Dict[int, Tuple[int, int]]      # frame_idx -> (addr, val)  (KV writes)
    step_count: int
    halted: bool                               # did the program HALT (vs run off cap)
    final_ax_masked: int                       # AX & mask at the last emitted step
    win_starts: List[int]                      # absolute pos of each step's query row
    out: List[int] = None                      # PRTF visible-output bytes (AX&0xFF)
    prtf_steps: List[int] = None               # step indices that emitted a PRTF byte
    # LIVENESS input (C4_EVICT_SCHEDULE): per-frame LOAD address (LI/LC reads
    # ``mem[addr]``).  ``load_log[frame_idx] = addr`` for every frame that LOADED, so
    # the liveness pass can compute, per store, the step after its LAST load (dead) or
    # its supersession by a same-address store — the deterministic eviction schedule
    # that replaces the O(S^2) content comparison.  None on drafts built without it.
    load_log: Dict[int, int] = None
    # PART B (direct-CAM read): per-frame CAM reads as ``[(head, addr), ...]`` where
    # head is "mem" (§Memory LI/LC), "pop" (stack head, incl. LEV's MEM[BP]) or "lev"
    # (LEV return-PC head, MEM[BP+4]).  ``resolve_load_rows`` turns each into the exact
    # KV store ROW its address resolves to (latest-write-wins), so the fast verify path
    # can DIRECT-GATHER that row instead of running the O(K) softmax CAM score.
    read_log: Dict[int, List[Tuple[str, int]]] = None
    # CODE-FROM-MEMORY (C4_PF_CFM): number of leading CODE-frame rows (one token per
    # instruction) prepended after BOS.  0 on the baked path (byte-identical).  The
    # register/store frames then start at absolute position ``1 + code_off`` instead
    # of ``1`` — the overlay's frame-position math shifts by this offset.
    code_off: int = 0


# The immediate is baked as IMM_NIBS little-endian nibbles (a STATIC re-encoding of
# the literal, gathered at PC), so the model's IMM writes AX = imm & (16**IMM_NIBS-1).
# ``ref_interpret`` masks IMM/LEA to 8 bits, which is WRONG for the model (the model
# keeps the full 20-bit literal — proven: add_0 IMM 654 -> model AX 654, not 142).
# The DRAFT must reproduce the MODEL's per-step register file exactly (else the
# parallel verify correctly rejects a draft that disagrees with the model), so this
# transition mirrors the MODEL, not ``ref_interpret``.
_IMM_MASK = (1 << (4 * IMM_NIBS)) - 1       # 20-bit literal band (IMM_NIBS=5 nibbles)


def draft_pf_program(code: List[isa.Instr], max_steps: int = 300000,
                     mask: int = 0xFFFFFFFF,
                     data_seg: Optional[Dict[int, int]] = None,
                     fio=None) -> PFDraft:
    """Run the MODEL's ISA transition and materialise the per-step 30-token frame
    stream + store_log — the exact token stream the pure-forward DRIVER emits.

    This is the *logical VM* of BLOG_SPEC §Speculation: the plain deterministic
    C4 interpreter, so it is ~free relative to a model forward.  It reproduces the
    driver's own pre-step register bookkeeping (which address a PSH/JSR/ENT/SI
    writes) so the drafted store_log is byte-identical to what the driver records.

    The transition matches the MODEL (not ``ref_interpret``):
      * IMM keeps the full literal ``imm & _IMM_MASK`` (5- or 8-nibble band; see
        ``C4_IMM_NIBS``) — the model's ``compile_imm_ax_nibbles`` writes all fetched
        nibbles;
      * LEA is folded to 8 bits (``compile_ax_byte_to_nibbles`` + ``_fold_ax_gated``);
      * ADD/SUB/MUL/DIV/MOD are 32-bit (``mask``); cmp/bitwise are 8-bit by default,
        or 32-bit signed under ``C4_DRAFT_CMP32`` (matching the model);
      * PSH/JSR/ENT/SI stores carry the full 32-bit AX (the KV memory value band).

    ``data_seg`` ({addr: byte}) is the compiled program's DATA segment (string
    literals etc.).  The driver (``run_pure_forward_cached``) seeds it as LEADING
    KV store frames so a LC/LI from a data address recalls the byte before step 0;
    the draft replicates that exactly (leading store frames + store_log entries)
    so its own ``mem`` and the drafted stream/store_log the verifier feeds the model
    agree.  ``None`` (default) -> no data segment (byte-identical to the old draft).
    """
    SP_INIT = _PFC.SP_INIT
    cmp32 = _draft_cmp32()               # 32-bit signed cmp draft (default OFF)
    mem: Dict[int, int] = {}
    sp = bp = SP_INIT
    ax = pc = 0
    # DATA-SEGMENT seed (leading KV store frames), matching run_pure_forward_cached:
    # each data byte becomes one leading store frame recorded in store_log at frame
    # idx k, and the init frame then sits at frame idx n_seed.  The draft's own `mem`
    # is seeded too so a LC/LI reads the byte.
    seed_frames: List[int] = []
    store_log: Dict[int, Tuple[int, int]] = {}
    for k, (addr, val) in enumerate(sorted((data_seg or {}).items())):
        a32, v32 = addr & 0xFFFFFFFF, val & 0xFFFFFFFF
        seed_frames += _build_frame(0, 0, SP_INIT, SP_INIT, 0,
                                    mem_addr=a32, mem_val=v32)
        store_log[k] = (a32, v32)
        mem[a32] = v32
    n_seed = len(store_log)
    # The driver's stream is [BOS] + seed_frames + init_frame, then one appended
    # frame per step EXCEPT the HALT step's frame (the driver breaks BEFORE the
    # append).  The init frame IS a real stream frame (frame_idx n_seed), so tokens
    # must include it or every later position shifts by 30.
    init_frame = _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    # CODE-FROM-MEMORY (C4_PF_CFM): prepend ONE token per instruction as leading CODE
    # frames (the program lives in the KV, fetched at PC by the address CAM).  The
    # register/store frames then begin at absolute position ``1 + code_off``.  The
    # code-frame tokens are neutral MEM markers (the overlay writes their CODE bands;
    # IS_FRAME_BYTE=0 -> ingest ignores them; the store CAM excludes IS_CODE rows).
    code_off = len(code) if _pf_cfm_enabled() else 0
    code_frame_toks: List[int] = [V.MEM] * code_off
    tokens: List[int] = [V.BOS] + code_frame_toks + seed_frames + init_frame
    frames: List[Dict[str, int]] = []
    load_log: Dict[int, int] = {}               # frame_idx -> loaded address (LI/LC)
    win_starts: List[int] = []
    out: List[int] = []                         # PRTF visible-output bytes
    prtf_steps: List[int] = []                  # step indices emitting a PRTF byte
    read_log: Dict[int, List[Tuple[str, int]]] = {}  # frame_idx -> [(head, addr), ...]
    stk = 0                                     # STACK0 mirror (MEM_VAL of a frame)
    stream_len = len(tokens)                    # == 1 + n_seed*30 + 30
    frame_idx = n_seed                          # init frame is frame n_seed
    halted = False
    steps = 0
    cur_pc, cur_sp, cur_bp = 0, SP_INIT, SP_INIT
    while steps < max_steps:
        if not (0 <= pc < len(code)):
            break
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        i = pc
        pc += 1
        step_halted = False
        pop_val = stk                        # STACK0 default (unchanged if not a pop)
        load_addr = None                     # set by LI/LC (the address recalled)
        # PART B (direct-CAM read): the address(es) this step CONTENT-ADDRESSES via a
        # global CAM head, tagged by which head reads it — ``(head, addr)`` pairs.
        # "mem" = the §Memory LI/LC head (query=AX), "pop" = the stack head
        # (query=SP, or BP on LEV), "lev" = the LEV return-PC head (query=BP+4).  The
        # resolver maps each to its latest superseding store row for the direct gather.
        read_addrs: List[Tuple[str, int]] = []
        # --- FILE OP (OPEN/READ/CLOS/PRTF): §Tool Use Mode.  When a ``fio`` is given
        # the DRIVER services the whole op via the TOOL_CALL runner (the compiler-ABI
        # dispatcher for the c4-compiler stack layout), exactly as run_pure_forward_cached
        # does: it reads the args from the store_log, runs the op (printf appends the
        # FORMATTED bytes to fio.runner.stdout; read lays the input bytes into memory),
        # and returns (new_ax, new_sp, byte_stores).  The byte_stores become their OWN
        # leading §Memory KV store frames (a later LC(addr) recalls the read byte).  This
        # makes the draft reproduce the driver's I/O stream byte-for-byte — needed for
        # doom's multi-arg printf("%c[2J%c[H", ESC, ESC) (7 stdout bytes, NOT 1). ------
        if fio is not None and op in _FS.FILE_OPCODES:
            new_ax, new_sp, byte_stores = _FS.dispatch_file_op_driver(
                op, ax & 0xFFFFFFFF, imm, cur_sp, store_log, fio,
                data_seg=data_seg, slot=4)
            pc = cur_pc + 1                       # file ops advance PC by one (no branch)
            sp = new_sp
            bp = cur_bp
            ax = new_ax & 0xFFFFFFFF
            if op == isa.PRTF:
                prtf_steps.append(len(frames))
            # the primary register frame for this step, then one store frame per byte
            # the op wrote to VM memory (READ's input bytes).
            frames.append({"pc": pc, "ax": ax & mask, "sp": sp & 0xFFFFFFFF,
                           "bp": bp & 0xFFFFFFFF, "stk": stk & 0xFFFFFFFF,
                           "op": isa.NAMES.get(op, op), "is_store": False,
                           "s_addr": 0, "s_val": 0, "is_halt": False,
                           # §Tool Use Mode: the model's registers at a FILE-op row are
                           # MEANINGLESS (the DRIVER services the op + overrides them), so
                           # verify_blocks must NOT check them against these driver-side
                           # post-dispatch values (e.g. PRTF's return = #bytes printed).
                           "is_file": True})
            win_starts.append(stream_len - 1)
            frame_idx += 1
            tokens += _build_frame(pc, ax, sp, bp, stk)
            stream_len += V.FRAME_LEN
            for (baddr, bval) in byte_stores:
                frame_idx += 1
                store_log[frame_idx] = (baddr & 0xFFFFFFFF, bval & 0xFF)
                tokens += _build_frame(pc, ax, sp, bp, stk,
                                       mem_addr=baddr & 0xFFFFFFFF, mem_val=bval & 0xFF)
                stream_len += V.FRAME_LEN
            cur_pc, cur_sp, cur_bp = pc, sp, bp
            if not (0 <= pc < len(code)):
                halted = (pc < 0 or pc >= len(code))
                break
            continue
        # --- the MODEL's transition (32-bit; IMM keeps the full nibble literal) --
        if op == isa.IMM:
            ax = imm & _IMM_MASK              # full 20-bit literal (model, not &0xFF)
        elif op == isa.LEA:
            ax = (bp + 4 * imm) & 0xFF
        elif op == isa.PSH:
            sp -= 4; mem[sp] = ax & mask
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
            read_addrs.append(("pop", sp))    # stack head reads MEM[sp]
            v = mem.get(sp, 0) & mask; sp += 4; pop_val = v
            if op == isa.ADD:
                ax = (v + ax) & mask
            elif op == isa.SUB:
                ax = (v - ax) & mask
            elif op == isa.MUL:
                ax = (v * ax) & mask
            elif op == isa.DIV:
                ax = ((v // ax) if ax else 0) & mask
            else:
                ax = ((v % ax) if ax else 0) & mask
        elif op in (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR):
            read_addrs.append(("pop", sp))    # stack head reads MEM[sp]
            pop_val = mem.get(sp, 0); v = pop_val & 0xFF; sp += 4
            if op == isa.OR:
                ax = (v | ax) & 0xFF
            elif op == isa.XOR:
                ax = (v ^ ax) & 0xFF
            elif op == isa.AND:
                ax = (v & ax) & 0xFF
            elif op == isa.SHL:
                ax = (v << ax) & 0xFF
            else:
                ax = (v >> ax) & 0xFF
        elif op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            read_addrs.append(("pop", sp))    # stack head reads MEM[sp]
            pop_val = mem.get(sp, 0); sp += 4
            if cmp32:
                # 32-bit SIGNED cmp (matches the model; needed for operands > 255,
                # e.g. doom's `i < CIRC==256` loop counters).  EQ/NE compare the full
                # 32-bit words; the ordering ops compare the two's-complement signed
                # integers.
                a32, v32 = ax & 0xFFFFFFFF, pop_val & 0xFFFFFFFF
                sa, sv = _s32(a32), _s32(v32)
                r = {isa.EQ: v32 == a32, isa.NE: v32 != a32, isa.LT: sv < sa,
                     isa.GT: sv > sa, isa.LE: sv <= sa, isa.GE: sv >= sa}[op]
            else:
                v = pop_val & 0xFF
                r = {isa.EQ: v == ax, isa.NE: v != ax, isa.LT: v < ax,
                     isa.GT: v > ax, isa.LE: v <= ax, isa.GE: v >= ax}[op]
            ax = 1 if r else 0
        elif op in (isa.LI, isa.LC):
            # The §Memory KV head relays ALL value nibbles the matching store wrote
            # (NIB_PER_REG = the full register width), and the store_log KV entry the
            # model recalls carries the full ``ax & mask`` (see the store bookkeeping
            # below).  So a load recalls the FULL stored value — NOT ``& 0xFF``.  The
            # old truncation drafted 222 for a >8-bit stored value (var_simple x=990)
            # while the model correctly recalled 990, and speculation then wrongly
            # rejected the model's CORRECT load.  Keep the draft consistent with the
            # store_log (the actual KV the model reads).
            load_addr = ax                     # the address this LI/LC recalled (KV read)
            read_addrs.append(("mem", ax))     # §Memory head reads MEM[ax]
            ax = mem.get(ax, 0) & mask
        elif op in (isa.SI, isa.SC):
            read_addrs.append(("pop", sp))     # stack head reads MEM[sp] (the store addr)
            addr = mem.get(sp, 0); pop_val = addr; sp += 4
            mem[addr] = ax & mask
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.JSR:
            # The return PC is stored to KV memory (store_log s_val) at the FULL
            # 32-bit width (see the store bookkeeping below), and the model's LEV
            # recalls it from that KV row — so the draft's own ``mem`` must carry the
            # full PC too, else a LEV returning to a >255 address (any program with
            # >256 instructions, e.g. doom) reads a truncated PC and diverges.  The
            # 8-bit form is preserved when cmp32 is OFF (the corpus is < 256 instrs,
            # so ``& 0xFF`` == the full PC there -> byte-identical).
            sp -= 4
            mem[sp] = (i + 1) & (0xFFFFFFFF if cmp32 else 0xFF)
            pc = imm
        elif op == isa.ENT:
            mem[sp - 4] = bp & 0xFFFFFFFF; sp -= 4; bp = sp; sp -= 4 * imm
        elif op == ADJ:
            sp += 4 * imm
        elif op == isa.LEV:
            sp = bp
            read_addrs.append(("pop", sp))     # stack head reads MEM[bp]   (saved BP)
            read_addrs.append(("lev", sp + 4)) # lev head reads MEM[bp+4]   (return PC)
            bp = mem.get(sp, 0); pc = mem.get(sp + 4, 0); sp += 8
        elif op == isa.PRTF:
            # I/O op (printf("%c", AX)): PC += 1 only, registers UNCHANGED — the
            # driver decodes this step's AX from the SAME KV-cached model row and
            # appends AX&0xFF as the visible output byte.  So the drafted frame is a
            # normal register frame (AX unchanged) and the byte is verified for free:
            # verify_blocks confirms the model's decoded AX == this frame's AX at the
            # PRTF query row, and out.append(AX&0xFF) is that verified byte.
            out.append(ax & 0xFF)
            prtf_steps.append(len(frames))       # this step's index (frame position)
        elif op == isa.NOP:
            pass
        elif op == isa.HALT:
            step_halted = True
        else:
            raise NotImplementedError(f"op {isa.NAMES.get(op, op)} not in ref ISA")

        # --- the driver's store bookkeeping (which address/value this step wrote) --
        # The driver decides the store target from the PRE-step registers it tracks
        # (cur_pc/cur_sp/cur_bp) and the emitted AX — identical to the model's own
        # code-as-data fetch.  We replicate it so the drafted store_log matches.
        s_addr = s_val = 0
        is_store = False
        if op in (isa.SI, isa.SC):
            is_store = True; s_addr = _mem_top(store_log, cur_sp); s_val = ax & mask
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & mask
        elif op == isa.JSR:
            is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
        elif op == isa.ENT:
            is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
        # STACK0 mirror (the non-store MEM_VAL the driver carries as `stk`): a POP
        # op loads MEM[sp_at_pop] into STACK0, so the NEXT emitted frame's MEM_VAL
        # is that popped value; otherwise STACK0 persists.  This value is never
        # CONSUMED by the model's transition (a pop reads the address-keyed KV store,
        # not the ingested STACK0), but tracking it keeps the drafted stream
        # byte-identical to the driver's emitted stream.
        if op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
                  isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR,
                  isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE,
                  isa.SI, isa.SC):
            stk = pop_val    # the value this pop loaded (captured in the transition)

        # the query row for THIS step is the row the driver decodes it from: the LAST
        # row of the stream BEFORE this step's frame is appended, matching
        # run_pure_forward_cached's win_start bookkeeping.
        win_starts.append(stream_len - 1)
        frames.append({"pc": pc, "ax": ax & mask, "sp": sp & 0xFFFFFFFF,
                       "bp": bp & 0xFFFFFFFF, "stk": stk & 0xFFFFFFFF,
                       "op": isa.NAMES.get(op, op),
                       "is_store": is_store, "s_addr": s_addr, "s_val": s_val,
                       "is_halt": step_halted})
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        if load_addr is not None:
            load_log[frame_idx] = load_addr & 0xFFFFFFFF
        if read_addrs:
            read_log[frame_idx] = [(hd, a & 0xFFFFFFFF) for hd, a in read_addrs]
        cur_pc, cur_sp, cur_bp = pc, sp, bp
        if step_halted or not (0 <= pc < len(code)):
            halted = step_halted or (pc < 0 or pc >= len(code))
            break                                 # HALT frame is NOT appended
        # append this (non-halt) step's frame — the driver appends AFTER the break
        frame = _build_frame(pc, ax, sp, bp, stk,
                             mem_addr=(s_addr if is_store else 0),
                             mem_val=(s_val if is_store else 0))
        tokens += frame
        stream_len += V.FRAME_LEN

    final_ax = frames[-1]["ax"] if frames else 0
    return PFDraft(tokens=tokens, frames=frames, store_log=store_log,
                   step_count=len(frames), halted=halted,
                   final_ax_masked=final_ax, win_starts=win_starts,
                   out=out, prtf_steps=prtf_steps, load_log=load_log,
                   read_log=read_log, code_off=code_off)


# ===========================================================================
# BLOCK-MoE skip forward: run the KV-cached block stack but SKIP the [start, end)
# block range (the divmod span), returning None kv for the skipped blocks.  Valid
# only when the span has no divmod step (the caller gates this): every divmod
# block's attention is all-zero (identity) so it produces identity output AND its
# K/V is never read, and its FFN writes only dead scratch — so dropping the span's
# COMPUTE is byte-identical to running it, at ~7x fewer blocks per forward.
# ===========================================================================
def _forward_hidden_cached_skip(model, x, past_key_values, q_positions, skip_range):
    start, end = skip_range
    n = len(model.blocks)
    if past_key_values is None:
        past_key_values = [None] * n
    new_caches = [None] * n
    hidden = x
    for b in range(n):
        if start <= b < end:
            continue                        # identity expert: skip the block compute
        hidden, kv = model.blocks[b](
            hidden, past_kv=past_key_values[b], q_positions=q_positions, use_cache=True)
        new_caches[b] = kv
    return hidden, new_caches


# ===========================================================================
# 2. THE BLOCK-WISE PARALLEL VERIFIER — run the SPARSE model over the drafted
#    stream in blocks, against the per-block KV cache + bounded eviction, and
#    confirm the decoded register state at every step-query row == the draft.
# ===========================================================================
@dataclass
class VerifyResult:
    accepted_steps: int                 # step-query rows the model confirmed
    total_steps: int                    # step-query rows checked
    all_matched: bool
    forwards: int                       # batched model.forward passes run
    first_mismatch: Optional[dict] = None
    max_seq_len: int = 0
    max_cache_size: int = 0
    total_evicted: int = 0
    decoded_final_ax: Optional[int] = None
    peak_vram_gb: float = 0.0           # peak CUDA allocated during verify
    evict_rounds: int = 0               # #(eviction sweeps) run over all blocks
    effective_block_steps: int = 0      # smallest K actually run (after OOM backoff)


def verify_blocks(model, L: PureForwardCompleteLayout, code: List[isa.Instr],
                  draft: PFDraft, *, block_steps: int = 64, device: str = "cpu",
                  evict: bool = True, cos_threshold: float = 0.99,
                  prune_interval: int = 120, zero_eps: float = 1e-9,
                  recency_eps: float = 1e-6, mask: int = 0xFFFFFFFF,
                  stats: Optional[dict] = None, fast: bool = True,
                  collect_out: Optional[List[int]] = None,
                  block_moe: bool = False,
                  evict_interval_steps: Optional[int] = None,
                  oom_backoff: bool = True,
                  min_block_steps: int = 4,
                  evict_schedule: Optional[bool] = None,
                  exact_evict: Optional[bool] = None,
                  prime_chunk: int = 2048) -> VerifyResult:
    """Verify the whole drafted stream on the SPARSE model in BLOCKS.

    Processes ``block_steps`` (== K) VM steps per batched ``forward_hidden_cached``.
    For each block we forward the CONTIGUOUS token span covering those steps against
    the growing per-block KV cache, decode the register state at each step-query
    row, and confirm it equals the draft's next-step registers.  Frozen frame rows
    (all non-query rows of the span) are committed to the cache.  The first
    mismatch aborts and is reported (a genuine fail: model argmax != draft).

    Because softmax1 is causal and each step-query row sits at the END of its own
    frame, its block output is identical whether computed in this bulk forward or
    in the per-step 31-row window (every row it attends to — its frame + all
    earlier frozen frames — is present with the same K/V).  This is the
    speculative collapse: N one-step forwards -> a few batched forwards.

    BIG-K + EVICTION-INTERVAL (2026-07-20).  ``block_steps`` may be cranked to
    ~1000 (~30k tokens/forward): the whole program is drafted ahead (free) and
    verified in the FEWEST, LARGEST batched forwards, so the per-forward Python /
    dispatch / eviction overhead amortizes over K steps.  The K ceiling is the
    attention score matrix ``[H, Sq, Sk]`` (Sq == K*30, Sk == cache+K*30) which
    grows ~O(K^2) in VRAM; ``oom_backoff`` HALVES the block on a CUDA-OOM and
    retries (so a caller can request a big K and the verifier finds the largest
    span that fits, down to ``min_block_steps``).  The K vs VRAM curve is reported
    in ``stats`` (``peak_vram_gb`` / ``effective_block_steps``).

    ``evict_interval_steps`` (default None == once per verify block, tuned to the
    block boundary) decouples the EVICTION cadence from the OLD per-``prune_interval``
    -token trigger: eviction runs once per this many VM steps, at block boundaries.
    This is the user's lever — with a big K it is ONE eviction ROUND per ~30k-token
    block, ~250x fewer eviction rounds than the old per-120-token cadence, which is
    the CPU wall (each round = 306 per-block GPU-decisions + host syncs).  Because a
    block is one forward, this can only make eviction LESS frequent than per-block;
    the WITHIN-block cache growth is bounded by K (the span), which — with the
    O(K^2) score matrix — is what caps K via VRAM.  For a FLAT-cache program a
    moderate K keeps the cache small and the OOM backoff finds the fitting K; for a
    GROWING-heap program (malloc) the cache tracks the live heap regardless of K, so
    K is genuinely cache-capped and the backoff finds the real ceiling.
    """
    n_blocks = len(model.blocks)
    H = model.blocks[0].attn.n_heads
    HD = model.blocks[0].attn.head_dim
    caches = [BlockKVCacheBatched(H, HD, model.blocks[b].attn.alibi_slopes)
              for b in range(n_blocks)]
    # DROP-KV local attention: if ``install_local_attention(..., drop_local_kv=True)``
    # tagged the blocks, tell each block's cache which heads are GLOBAL (keep full KV)
    # vs LOCAL (keep only the last W positions — OLD rows DROPPED, not just masked).
    # This is the actual VRAM lever: the ~7357 local head-slots keep ~W rows instead of
    # the whole S-row history; eviction then only manages the ~3 global heads.
    for b in range(n_blocks):
        at = model.blocks[b].attn
        if getattr(at, "_drop_local_kv", False) and getattr(at, "_local_window", None):
            gmask = at._global_head_mask
            g_idx = [int(h) for h in range(H) if bool(gmask[h])]
            caches[b].set_head_groups(
                g_idx, int(at._local_window),
                content_bound=getattr(at, "_content_bound_global", False),
                content_cR=getattr(at, "_store_gate_channel", None))
    store_log = draft.store_log
    n_steps = draft.step_count
    # SCHEDULE-DRIVEN eviction (C4_EVICT_SCHEDULE): precompute the deterministic
    # per-store eviction step off the perfect draft ONCE, replacing the per-round
    # O(S^2) content comparison with an O(dropped) position drop.  Default OFF
    # (env / explicit arg) so the proven content-bound path stays the fallback until
    # byte-identity is established.  When None, read the env flag.
    if evict_schedule is None:
        import os
        evict_schedule = os.environ.get("C4_EVICT_SCHEDULE", "0") not in ("0", "", "false", "False")
    # EXACT O(steps) EVICTION (C4_EXACT_EVICT): unify supersession + free (pop-free +
    # heap zero-tombstone) + last-read+1 into ONE liveness schedule off the draft, and
    # drive ALL drops through it — NO O(S^2) content prune (evict_all_blocks_fused) at
    # all.  Implies evict_schedule (the schedule is the drop authority).  Default OFF.
    if exact_evict is None:
        import os
        exact_evict = os.environ.get("C4_EXACT_EVICT", "0") not in ("0", "", "false", "False")
    if exact_evict:
        evict_schedule = True                       # exact-evict IS a schedule mode
    sched = None
    if evict and evict_schedule:
        # the widest (smallest-slope) GLOBAL head sets the recency horizon for the
        # freed (zero-value) rows, so the freed-row drop is never earlier than any
        # head would drop it — conservative + byte-identical to the content horizon.
        slope_min = None
        for b in range(n_blocks):
            c = caches[b]
            if getattr(c, "split", False) and c._global_head_idx is not None \
                    and c._global_head_idx.numel() > 0:
                sl = c.slopes.index_select(0, c._global_head_idx)
            else:
                sl = c.slopes
            sl = sl[sl > 0]
            if sl.numel() == 0:
                continue
            m = float(sl.min())
            slope_min = m if slope_min is None else min(slope_min, m)
        # In EXACT mode the last-read+1 pass is the SOLE liveness authority and
        # subsumes the pop-free and freed-recency heuristics EXACTLY (a pop / tombstone
        # consumer IS a resolved read, so ``last_read+1`` == the pop-free frame and
        # correctly keeps a tombstone a later LI re-reads).  So we DISABLE the freed
        # (``supersession_only=True`` skips the recency heuristic) and pop passes and
        # let the exact fold compute min(supersession, last_read+1) per row.  In plain
        # schedule mode only supersession is encoded (the content ``skip_mech1`` pass
        # completes the O(S) zero-value/recency per-head).
        sched = build_eviction_schedule(
            draft, slope_min=slope_min, recency_eps=recency_eps, zero_eps=zero_eps,
            supersession_only=True,
            pop_free=(False if exact_evict else None),
            exact_evict=exact_evict)
    sched_frames = sorted_evict_frames(sched) if sched is not None else []
    sched_ptr = 0                       # frontier into sched_frames (O(steps) walk)
    forwards = 0
    accepted = 0
    steps_since_evict = 0
    # EVICTION cadence in VM STEPS (the user's lever).  Default: once per verify
    # block (K steps).  The OLD ``prune_interval`` was a 120-TOKEN trigger that
    # fired every ~4 steps regardless of K; ``evict_interval_steps`` fires at most
    # once per this many steps, so a big-K block evicts ONCE at its boundary.
    evict_every = (evict_interval_steps if evict_interval_steps is not None
                   else block_steps)
    evict_every = max(1, int(evict_every))
    max_seq = 1 + n_steps * V.FRAME_LEN
    max_cache = 0
    evict_rounds = 0
    peak_vram = 0
    Hf = model.blocks[0].attn.n_heads
    # DROP-KV split peak accounting (measured at the max total-footprint moment, so
    # the reduction is honest even though eviction later shrinks the global cache).
    split_on = any(getattr(c, "split", False) for c in caches)
    peak_split_rows = 0          # sum_b (Hg_b*global_b + Hl_b*local_b) at its peak
    peak_gmax = 0                # high-water GLOBAL cache size (pre-eviction); the
                                 # classic full-H cache would have held H*this per block
                                 # (the global heads see FULL history == what a classic
                                 # LOCAL head would also hold).
    is_cuda = device.startswith("cuda")
    min_k = max(1, int(min_block_steps))
    import time as _time
    t_evict = 0.0                           # wall spent in the fused eviction
    last_got_ax = None                      # the MODEL's decoded AX at the last step
    prtf_set = set(draft.prtf_steps or ())  # steps whose model AX is a PRTF byte
    # BLOCK-MoE divmod-skip: when block_moe and a whole block-verify span has NO
    # DIV/MOD step, skip the ~262-block divmod span for that forward.  Safe because
    # every divmod block's attention is all-zero (identity) — its K/V is never read
    # — and its FFN writes only dead scratch bands (re-derived each step, consumed
    # only by the post-span ax-mux, which reads them only on a DIV/MOD step).  So a
    # span with no divmod step is byte-identical with the divmod blocks skipped, and
    # their absent cache is never referenced (identity attention).  A span WITH a
    # divmod step runs the full stack (the ax-mux needs the RES bands that step).
    moe_span = None
    divmod_step = None
    blocks_run_total = 0
    blocks_full_total = 0
    if block_moe:
        from .block_moe_divmod import resolve_divmod_span
        moe_span = resolve_divmod_span(L)
        _DM = {"DIV", "MOD"}
        divmod_step = [(draft.frames[s]["op"] in _DM) for s in range(n_steps)]
    dev = torch.device(device)
    if is_cuda:
        # Reset the peak so ``peak_vram_gb`` reflects THIS verify's largest span
        # (the K-vs-VRAM datapoint), not any earlier allocation.
        torch.cuda.reset_peak_memory_stats(dev)
    # O(1)-decode: build the row-invariant program-in-data vector ONCE (on the
    # embed's device/dtype so the broadcast-add is a device-resident op) instead of
    # re-writing the whole code into every span row each block.
    code_vec = (build_code_vec(code, L, model.embed.shape[1], dev,
                               dtype=model.embed.dtype) if fast else None)

    # Holder for the primed leading-context boundary: step 0's span starts here (0 if
    # no priming ran).  Set after the priming loop below.
    _primed_start = [0]

    # --- one block-verify forward over steps [step, end) --------------------
    # Returns (hidden, new_kv, span_start, S, blocks_run).  Raises
    # torch.cuda.OutOfMemoryError (or RuntimeError with 'out of memory') so the
    # caller can halve K and retry; the caches are NOT mutated here (commit happens
    # after), so a retry is safe.  Block counts are RETURNED (not accumulated into
    # the running totals) so an OOM-retried block is not double-counted.
    def _forward_span(step, end):
        # When the leading context was PRIMED (cached in chunks), step 0's span starts
        # AFTER the leading rows (they are already in the cache) instead of at 0.
        if step == 0:
            span_start = _primed_start[0]
        else:
            span_start = draft.win_starts[step]
        span_end = (draft.win_starts[end] + 1) if end < n_steps \
            else len(draft.tokens)
        span_toks = draft.tokens[span_start:span_end]
        S = len(span_toks)
        win_toks = torch.tensor([span_toks], device=dev)
        q_positions = torch.arange(span_start, span_start + S, device=dev)
        with torch.no_grad():
            x = model.embed[win_toks].clone()
            # overlay the span: program-in-data + frame roles + store KV entries.
            # The span's last row is NOT necessarily a query row, so overlay with
            # is_last_row_query=False, then explicitly re-tag EACH step's query row
            # with all-ROLE one-hots (the driver's per-step query tag).
            q_local = [draft.win_starts[s] - span_start for s in range(step, end)]
            if fast:
                # O(rows + code): broadcast the code-in-data, per-row roles, then
                # tag exactly this block's query rows (byte-identical residual).
                apply_overlay_window_fast(x, span_start, L, store_log, code_vec,
                                          query_rows=q_local, code=code,
                                          code_off=draft.code_off)
            else:
                apply_overlay_window(x, span_start, code, L, store_log,
                                     is_last_row_query=False,
                                     code_off=draft.code_off)
                for wi in q_local:
                    for role in range(N_ROLES):
                        x[0, wi, L.ROLE + role] = 1.0
            past = [caches[b].as_past_kv() for b in range(n_blocks)]
            # BLOCK-MoE: skip the divmod span iff NO step in [step, end) is DIV/MOD.
            span_has_divmod = (moe_span is not None
                               and any(divmod_step[s] for s in range(step, end)))
            skip_range = (moe_span if (moe_span is not None and not span_has_divmod)
                          else None)
            if skip_range is not None:
                hidden, new_kv = _forward_hidden_cached_skip(
                    model, x, past, q_positions, skip_range)
                blocks_run = n_blocks - (skip_range[1] - skip_range[0])
            else:
                hidden, new_kv = model.forward_hidden_cached(
                    x, past_key_values=past, q_positions=q_positions, use_cache=True)
                blocks_run = n_blocks
        return hidden, new_kv, span_start, S, blocks_run

    # --- commit the FROZEN (non-query) rows of the span to the caches --------
    # A step's query row carries the all-ROLE query overlay (wrong K/V for a frozen
    # context token), so we commit exactly the non-query rows — whose overlay IS the
    # plain context overlay computed above.  The NEXT block's first window re-reads
    # from the last query row's position (span_start), which we therefore leave
    # un-cached until it is committed here as a frozen row of the current span.
    def _commit_span(new_kv, span_start, S, step, end):
        q_rows_in_span = {draft.win_starts[s] - span_start
                          for s in range(step, end)}
        keep = [p for p in range(S) if p not in q_rows_in_span]
        if not keep:
            return
        keep_idx = torch.tensor(keep, device=dev, dtype=torch.long)
        for b in range(n_blocks):
            if new_kv[b] is None:
                continue                    # block-MoE skipped block (identity
                # attention -> its cache is never read, so a gap is harmless).
            K_all, V_all, pos_all = new_kv[b]
            K_span = K_all[:, :, -S:, :]
            V_span = V_all[:, :, -S:, :]
            pos_span = pos_all[-S:]
            caches[b].commit(K_span[:, :, keep_idx, :],
                             V_span[:, :, keep_idx, :],
                             pos_span[keep_idx])

    def _cache_sizes():
        return [c.size() for c in caches]

    def _rollback_caches(sizes):
        # Truncate any PARTIAL commit back to the pre-block sizes so an OOM mid-way
        # through the per-block commit loop leaves the caches in the clean pre-block
        # state — the retry then re-commits from scratch at the smaller K.
        for c, n0 in zip(caches, sizes):
            n_now = c.size()
            if n_now > n0 and c.K is not None:
                c.K = c.K[:, :, :n0, :]
                c.V = c.V[:, :, :n0, :]
                c.pos = c.pos[:n0]

    # ---- LEADING-CONTEXT PRIMING (large seed / code frames) -----------------
    # The first step's query row sits at ``win_starts[0]`` — with a big DATA-SEGMENT
    # seed (doom: 984 store frames ≈ 29.5k tokens) or CODE-FROM-MEMORY code frames
    # (3976 rows) the leading FROZEN context is tens of thousands of rows.  Forwarding
    # it as ONE span makes the local heads' dense score matrix ``[H, Sq, Sk]`` blow
    # VRAM (O(lead^2)).  Instead prime the KV cache by forwarding the leading rows in
    # ``prime_chunk``-sized causal chunks, committing each chunk's frozen KV — so the
    # step spans below start with the cache already populated and no single forward
    # sees the whole leading block.  These rows are all frozen context (no query row),
    # byte-identical to committing them as part of the first span.
    lead_end = draft.win_starts[0] if n_steps > 0 else len(draft.tokens)

    def _prime_leading():
        pos = 0
        pc_ = int(max(1, prime_chunk))
        while pos < lead_end:
            ce = min(pos + pc_, lead_end)
            span_toks = draft.tokens[pos:ce]
            S = len(span_toks)
            win_toks = torch.tensor([span_toks], device=dev)
            q_positions = torch.arange(pos, pos + S, device=dev)
            with torch.no_grad():
                x = model.embed[win_toks].clone()
                if fast:
                    apply_overlay_window_fast(x, pos, L, store_log, code_vec,
                                              query_rows=[], code=code,
                                              code_off=draft.code_off)
                else:
                    apply_overlay_window(x, pos, code, L, store_log,
                                         is_last_row_query=False,
                                         code_off=draft.code_off)
                past = [caches[b].as_past_kv() for b in range(n_blocks)]
                _, new_kv = model.forward_hidden_cached(
                    x, past_key_values=past, q_positions=q_positions, use_cache=True)
                # commit ALL rows of this chunk (every row is frozen context).
                for b in range(n_blocks):
                    if new_kv[b] is None:
                        continue
                    K_all, V_all, pos_all = new_kv[b]
                    caches[b].commit(K_all[:, :, -S:, :], V_all[:, :, -S:, :],
                                     pos_all[-S:])
            pos = ce
        return pos

    if lead_end > int(max(1, prime_chunk)):
        n_primed = _prime_leading()
        _primed_start[0] = lead_end          # step-0 span starts past the primed rows
        if stats is not None:
            stats["primed_leading_rows"] = n_primed
        if is_cuda:
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated(dev))

    cur_k = int(block_steps)
    eff_min_k = cur_k                       # smallest K actually run (OOM backoff)
    step = 0
    while step < n_steps:
        end = min(step + cur_k, n_steps)
        # OOM-adaptive block: run forward + commit; on CUDA-OOM (the K^2 score
        # matrix or the commit concat overflowing VRAM) roll back any partial
        # commit, HALVE K, empty the cache and retry the WHOLE block from the same
        # step.  ``cur_k`` stays dropped so later blocks also fit — this is how a
        # caller can request a big K and the verifier finds the largest span that
        # fits (the K vs VRAM ceiling), down to ``min_block_steps``.
        pre_sizes = _cache_sizes()
        while True:
            try:
                hidden, new_kv, span_start, S, blk_run = _forward_span(step, end)
                # commit INSIDE the retry so a commit-time OOM (the concat) also
                # backs off; the query-row verify below allocates nothing.
                _commit_span(new_kv, span_start, S, step, end)
                if is_cuda:
                    peak_vram = max(peak_vram, torch.cuda.max_memory_allocated(dev))
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                oom = ("out of memory" in str(e).lower()
                       or isinstance(e, torch.cuda.OutOfMemoryError))
                span_k = end - step
                if not (oom_backoff and is_cuda and oom and span_k > min_k):
                    raise
                _rollback_caches(pre_sizes)
                torch.cuda.empty_cache()
                cur_k = max(min_k, span_k // 2)
                end = min(step + cur_k, n_steps)
                eff_min_k = min(eff_min_k, cur_k)
        forwards += 1
        blocks_run_total += blk_run       # count only the SUCCEEDED block
        blocks_full_total += n_blocks

        # decode + verify each step-query row of the block against the draft.
        for s in range(step, end):
            wi = draft.win_starts[s] - span_start
            state = hidden[0, wi]
            got_pc = _snap_lane(state[L.PC_VAL])
            got_sp = _snap_lane(state[L.SP_VAL])
            got_bp = _snap_lane(state[L.BP_VAL])
            got_ax = _decode_reg_from_nibbles(state, L, L.AX)
            fr = draft.frames[s]
            want_pc = fr["pc"]
            want_ax = fr["ax"] & mask
            want_sp = fr["sp"] & 0xFFFFFFFF
            want_bp = fr["bp"] & 0xFFFFFFFF
            # The HALT step emits NO frame (the driver breaks before the append) — it
            # only reads the final AX (the answer).  HALT leaves pc un-incremented in
            # the model (the draft advanced pc past it), so verify ONLY the AX there.
            if fr.get("is_file"):
                # §Tool Use Mode: the DRIVER services OPEN/READ/CLOS/PRTF and OVERRIDES
                # the model's registers for this row (the model never runs the printf /
                # read).  The draft's frame carries the DRIVER's post-dispatch registers
                # (e.g. PRTF's AX = #bytes printed = 7, not the on-top ESC=27 the model
                # still shows), so verifying the model's decoded registers here is a
                # false mismatch.  The output byte(s) are already produced by the draft's
                # own compiler-ABI FileRunner (fio.runner.stdout), so this row is a
                # driver-accepted no-op for speculation.
                bad = False
            elif fr.get("is_halt"):
                bad = ((got_ax & mask) != want_ax)
            else:
                bad = (got_pc != want_pc or (got_ax & mask) != want_ax
                       or got_sp != want_sp or got_bp != want_bp)
            if bad:
                # WALL#6 DIAG (C4_WALL6_DIAG=1, additive/inert by default): decode the
                # model's STACK0 (pop operand) + all mem/pop register bands at the
                # mismatch so we can localize the address-CAM stale read.
                import os as _osd
                if _osd.environ.get("C4_WALL6_DIAG", "0") == "1" and stats is not None:
                    try:
                        stk = _decode_reg_from_nibbles(state, L, L.STACK0)
                    except Exception:
                        stk = None
                    fr_ = draft.frames[s]
                    stats["diag_stack0_model"] = stk
                    stats["diag_stk_draft"] = fr_.get("stk")
                    stats["diag_op"] = fr_.get("op")
                cache_now = max(max_cache, caches[0].size())
                evicted_now = sum(c.total_evicted for c in caches)
                vram_gb = peak_vram / (1024 ** 3)
                if stats is not None:
                    stats["max_seq_len"] = max_seq
                    stats["max_cache_size"] = cache_now
                    stats["total_evicted"] = evicted_now
                    stats["forwards"] = forwards
                    stats["peak_vram_gb"] = vram_gb
                    stats["evict_rounds"] = evict_rounds
                    stats["effective_block_steps"] = eff_min_k
                return VerifyResult(
                    accepted_steps=accepted, total_steps=n_steps,
                    all_matched=False, forwards=forwards,
                    first_mismatch={
                        "step": s, "query_pos": draft.win_starts[s],
                        "got": {"pc": got_pc, "ax": got_ax & mask,
                                "sp": got_sp, "bp": got_bp},
                        "want": {"pc": want_pc, "ax": want_ax,
                                 "sp": want_sp, "bp": want_bp}},
                    max_seq_len=max_seq, max_cache_size=cache_now,
                    total_evicted=evicted_now, decoded_final_ax=None,
                    peak_vram_gb=vram_gb, evict_rounds=evict_rounds,
                    effective_block_steps=eff_min_k)
            accepted += 1
            if collect_out is not None and s in prtf_set:
                # PRTF visible byte = the MODEL's decoded AX byte-0 at this row (a
                # verified byte: got_ax == want_ax just passed the accept check), so
                # this is byte-identical to run_pure_forward_cached's out.append.
                collect_out.append(got_ax & 0xFF)
            if s == n_steps - 1:
                last_got_ax = got_ax & mask     # the model's actual final AX

        # (the FROZEN non-query rows were committed to the caches inside the
        # OOM-guarded block above.)
        if device.startswith("cuda"):
            torch.cuda.synchronize(dev)

        # DROP-KV peak accounting — measured HERE (post-commit, PRE-eviction) so the
        # global-cache high-water reflects what the classic full-H cache would have
        # reached before eviction (the fair "avoided cost"); the split's own local
        # rows are already window-bounded so their peak is stable.
        if split_on:
            g_max = max((c.size() for c in caches), default=0)
            split_rows = sum(c.n_global_heads() * c.size()
                             + (Hf - c.n_global_heads()) * c.local_size()
                             for c in caches)
            peak_split_rows = max(peak_split_rows, split_rows)
            peak_gmax = max(peak_gmax, g_max)

        # EVICTION-INTERVAL (user's lever): evict ONCE per ``evict_every`` VM steps
        # (default: once per verify block).  Fires at the block boundary, so with a
        # big K this is ONE eviction sweep per ~30k-token block — ~250x fewer
        # eviction ROUNDS than the old per-120-token trigger, which is the CPU wall.
        # Note: because a block is one forward, this can only make eviction LESS
        # frequent than per-block (evict_every > K prunes every ceil(evict_every/K)
        # blocks); it does NOT prune within a big-K forward, so the within-block
        # cache growth is bounded by K (the span), not by evict_every.
        steps_since_evict += (end - step)
        if evict and steps_since_evict >= evict_every:
            # FUSED eviction (#667/#670): ONE batched on-GPU decision for ALL
            # n_blocks caches (replaces the per-block Python loop of host-synced
            # ``evict``), then a boolean-mask compaction per block.  Byte-identical
            # survivor set to the per-block ``caches[b].evict`` loop; de-syncs the
            # deep-loop prune so the GPU stays busy (GPU util was 41% under the old
            # per-block host-synced eviction).  Timed with a drain around the prune
            # (a correctness barrier the next forward needs anyway); the forward /
            # overlay wall is the remainder (t_fast - t_evict).
            if is_cuda:
                torch.cuda.synchronize(dev)
            _t0 = _time.perf_counter()
            if sched is not None:
                # SCHEDULE-DRIVEN: drop exactly the store rows the liveness pass
                # marked dead through the highest committed store frame (frame_idx
                # ``end`` — step ``s`` commits ``store_log[s+1]`` so committing steps
                # [step, end) has laid down store frames up to frame_idx ``end``).
                # Walk the ASCENDING eviction-frame FRONTIER: emit only the NEW dead
                # positions this round (true O(steps) total — no cumulative rescan).
                # Already-dropped rows are gone from the cache, so incremental ==
                # cumulative, byte-identical.
                drop_pos: List[int] = []
                while sched_ptr < len(sched_frames) and sched_frames[sched_ptr] <= end:
                    drop_pos.extend(positions_new_at(sched, sched_frames[sched_ptr]))
                    sched_ptr += 1
                # SUPERSESSION (+ exact-mode free/pop/last-read) drops from the draft.
                keep_masks = (evict_all_blocks_scheduled(caches, drop_pos)
                              if drop_pos else [None] * n_blocks)
                for b in range(n_blocks):
                    if keep_masks[b] is not None:
                        caches[b].apply_keep_mask(keep_masks[b])
                if exact_evict:
                    # EXACT mode: the liveness schedule is the SOLE drop authority.
                    # The unified min(supersession, free, last-read+1) already dropped
                    # every provably-inert store row (n_read_after_free == 0 gate), so
                    # there is NOTHING left for the O(S) content pass to do — and the
                    # O(S^2) content prune (evict_all_blocks_fused) is dropped ENTIRELY.
                    # This is what removes the ~40% eviction wall.
                    keep_masks = [None] * n_blocks
                else:
                    # HYBRID schedule mode: the O(S) zero-value/recency mechanisms
                    # (2a/2b/3) — per-head, byte-exact, NO cdist/cosine (mech-1 skipped:
                    # the schedule already did supersession).  Handles the cross-head
                    # zero-value/recency decision the address-only schedule can't see.
                    keep_masks = evict_all_blocks_fused(
                        caches, cos_threshold, zero_eps, recency_eps, skip_mech1=True)
            else:
                # FUSED CONTENT eviction (#667/#670): ONE batched on-GPU decision for
                # ALL n_blocks caches (the O(S^2) near-dup cdist/cosine over the live
                # cache), then a boolean-mask compaction per block.
                keep_masks = evict_all_blocks_fused(
                    caches, cos_threshold, zero_eps, recency_eps)
            for b in range(n_blocks):
                if keep_masks[b] is not None:
                    caches[b].apply_keep_mask(keep_masks[b])
            if is_cuda:
                torch.cuda.synchronize(dev)
            t_evict += _time.perf_counter() - _t0
            steps_since_evict = 0
            evict_rounds += 1
        if split_on:
            max_cache = max(max_cache, max((c.size() for c in caches), default=0))
        else:
            max_cache = max(max_cache, caches[0].size())
        step = end

    # the answer is the MODEL's decoded AX at the last step (verify PROVED it equals
    # the draft's, so this is the token-by-token autoregressive final byte).
    decoded_final = last_got_ax
    vram_gb = peak_vram / (1024 ** 3)
    if stats is not None:
        stats["max_seq_len"] = max_seq
        _final_cache = (max((c.size() for c in caches), default=0) if split_on
                        else caches[0].size())
        stats["max_cache_size"] = max(max_cache, _final_cache)
        stats["total_evicted"] = sum(c.total_evicted for c in caches)
        stats["forwards"] = forwards
        # EXACT-EVICT liveness-schedule report (C4_EXACT_EVICT): the drop breakdown
        # and the READ-AFTER-FREE audit — n_read_after_free MUST be 0 for byte-exact.
        if sched is not None:
            stats["sched_stores"] = sched.n_stores
            stats["sched_superseded"] = sched.n_superseded
            stats["sched_freed"] = sched.n_freed
            stats["sched_popped"] = sched.n_popped
            stats["sched_dead_unread"] = sched.n_dead_unread
            stats["sched_live"] = sched.n_live
            stats["sched_read_after_free"] = sched.n_read_after_free
            stats["exact_evict"] = bool(exact_evict)
        stats["peak_vram_gb"] = vram_gb
        stats["evict_rounds"] = evict_rounds
        stats["effective_block_steps"] = eff_min_k
        # block-MoE accounting: total blocks executed vs the full-stack equivalent.
        stats["blocks_run"] = blocks_run_total
        stats["blocks_full"] = blocks_full_total
        stats["block_moe_speedup"] = (blocks_full_total / max(blocks_run_total, 1)
                                      if block_moe else 1.0)
        # eviction wall (the #667 bottleneck instrumentation): the fused on-GPU
        # prune should be a SMALL fraction of the fast wall (it was the dominant
        # cost with the per-block host-synced loop).  ``t_evict`` is measured with a
        # drain around the prune (correctness barrier the next forward needs anyway);
        # the forward/overlay wall is the remainder (t_fast - t_evict), NOT separately
        # drained so the block forwards still pipeline.
        stats["t_evict"] = t_evict
        stats["n_prunes"] = evict_rounds       # #fused-eviction rounds (== evict_rounds)
        # DROP-KV split accounting: how many caches actually manage a GLOBAL cache
        # (the only ones eviction touches) vs the local-head window rows.  Reports
        # the eviction-overhead reduction (only ~2 blocks have a global cache).
        if split_on:
            g_caches = [c for c in caches if getattr(c, "split", False)
                        and c.n_global_heads() > 0]
            stats["split_active"] = True
            # #caches that manage a GLOBAL cache (the ONLY ones eviction touches) —
            # the eviction-overhead reduction: ~2 of ~320 blocks, vs all 320 classic.
            stats["n_global_caches"] = len(g_caches)
            stats["max_local_cache"] = max((c.local_size() for c in caches), default=0)
            # KV-row footprint: the split's peak rows vs the classic full-H cache the
            # split AVOIDS.  Classic == n_blocks * H * (peak global cache high-water),
            # since the global heads see FULL history so a classic LOCAL head would
            # hold the SAME rows.  On a growing-cache program this is the big win; on a
            # flat-cache (well-evicted) program the local window rows dominate.
            classic_rows = n_blocks * Hf * peak_gmax
            stats["kv_rows_split"] = peak_split_rows
            stats["kv_rows_classic"] = classic_rows
            stats["peak_global_cache"] = peak_gmax
            stats["kv_row_reduction"] = (classic_rows / max(peak_split_rows, 1))
    return VerifyResult(
        accepted_steps=accepted, total_steps=n_steps,
        all_matched=(accepted == n_steps), forwards=forwards,
        max_seq_len=max_seq, max_cache_size=max(max_cache, caches[0].size()),
        total_evicted=sum(c.total_evicted for c in caches),
        decoded_final_ax=decoded_final, peak_vram_gb=vram_gb,
        evict_rounds=evict_rounds, effective_block_steps=eff_min_k)


# ===========================================================================
# 3. THE SPECULATIVE DRIVER — draft -> block-verify -> PASS iff model accepts the
#    full stream AND the decoded final AX == expected.
# ===========================================================================
@dataclass
class SpecResult:
    status: str                          # PASS | FAIL | TIMEOUT
    decoded_final_ax: Optional[int]
    expected: int
    step_count: int
    forwards: int                        # batched forwards run (speculative cost)
    naive_forwards: int                  # token-by-token forwards (== step_count)
    speedup: float                       # naive / speculative forwards
    accepted_steps: int
    all_matched: bool
    detail: str = ""
    first_mismatch: Optional[dict] = None
    max_seq_len: int = 0
    max_cache_size: int = 0
    total_evicted: int = 0


def speculative_run(model, L: PureForwardCompleteLayout, code: List[isa.Instr],
                    expected: int, *, block_steps: int = 64,
                    max_steps: int = 300000, device: str = "cpu",
                    evict: bool = True, prune_interval: int = 120,
                    mask: int = 0xFFFFFFFF, fast: bool = True,
                    block_moe: bool = False,
                    collect_out: Optional[List[int]] = None,
                    evict_interval_steps: Optional[int] = None,
                    oom_backoff: bool = True,
                    min_block_steps: int = 4,
                    evict_schedule: Optional[bool] = None) -> SpecResult:
    """Full speculative decode of ONE pure-forward program.

    1) draft the whole stream with the reference VM (zero forwards);
    2) verify it block-wise on the SPARSE model (a handful of forwards);
    3) PASS iff the model ACCEPTS the full drafted stream (every step-query row's
       decoded register state matches the draft) AND the decoded final AX ==
       expected.  A verify mismatch is a genuine FAIL (model argmax != draft),
       reported with the step/position.  A draft that never HALTs within
       ``max_steps`` is a TIMEOUT.

    ``block_steps`` (== K), ``evict_interval_steps``, ``oom_backoff`` and
    ``min_block_steps`` are passed straight to ``verify_blocks`` (see its docstring
    for the big-K + eviction-interval levers).
    """
    exp = expected & 0xFFFFFFFF
    draft = draft_pf_program(code, max_steps=max_steps, mask=mask)
    if not draft.halted:
        return SpecResult(
            status="TIMEOUT", decoded_final_ax=None, expected=exp,
            step_count=draft.step_count, forwards=0,
            naive_forwards=draft.step_count, speedup=0.0,
            accepted_steps=0, all_matched=False,
            detail=f"draft did not HALT within {max_steps} steps")
    stats: dict = {}
    vr = verify_blocks(model, L, code, draft, block_steps=block_steps,
                       device=device, evict=evict, prune_interval=prune_interval,
                       mask=mask, stats=stats, fast=fast, block_moe=block_moe,
                       collect_out=collect_out,
                       evict_interval_steps=evict_interval_steps,
                       oom_backoff=oom_backoff, min_block_steps=min_block_steps,
                       evict_schedule=evict_schedule)
    naive = draft.step_count
    speedup = (naive / vr.forwards) if vr.forwards else float("inf")
    if not vr.all_matched:
        return SpecResult(
            status="FAIL", decoded_final_ax=None, expected=exp,
            step_count=draft.step_count, forwards=vr.forwards,
            naive_forwards=naive, speedup=speedup,
            accepted_steps=vr.accepted_steps, all_matched=False,
            detail=(f"model argmax != draft at step {vr.first_mismatch['step']} "
                    f"pos {vr.first_mismatch['query_pos']}: "
                    f"got {vr.first_mismatch['got']} want {vr.first_mismatch['want']}"),
            first_mismatch=vr.first_mismatch,
            max_seq_len=vr.max_seq_len, max_cache_size=vr.max_cache_size,
            total_evicted=vr.total_evicted)
    got = vr.decoded_final_ax & mask
    status = "PASS" if got == exp else "FAIL"
    detail = "" if status == "PASS" else f"exit mismatch: exp {exp} got {got}"
    return SpecResult(
        status=status, decoded_final_ax=got, expected=exp,
        step_count=draft.step_count, forwards=vr.forwards,
        naive_forwards=naive, speedup=speedup,
        accepted_steps=vr.accepted_steps, all_matched=True, detail=detail,
        max_seq_len=vr.max_seq_len, max_cache_size=vr.max_cache_size,
        total_evicted=vr.total_evicted)


# ===========================================================================
# 4. BYTE-IDENTITY SPOT-CHECK vs the token-by-token KV-cached driver.
# ===========================================================================
def spotcheck_vs_cached(model, L: PureForwardCompleteLayout,
                        code: List[isa.Instr], *, max_steps: int = 4096,
                        device: str = "cpu", evict: bool = False,
                        prune_interval: int = 120, block_steps: int = 64,
                        mask: int = 0xFFFFFFFF) -> dict:
    """Prove the speculative decode is byte-identical to the token-by-token
    KV-cached driver: run BOTH and assert the per-step AX trace matches the
    driver's emitted frames position-by-position.

    The driver is the authoritative token-by-token autoregressive decode; the
    speculative verify only ACCEPTS what the model itself produces, so on the
    perfect draft the two traces are identical byte-for-byte.
    """
    from .nibble_pure_forward_cached import run_pure_forward_cached
    drv_trace = run_pure_forward_cached(
        model, L, code, max_steps=max_steps, mask=mask, evict=evict,
        prune_interval=prune_interval)
    draft = draft_pf_program(code, max_steps=max_steps, mask=mask)
    spec_trace = [fr["ax"] & mask for fr in draft.frames]
    # Verify BOTH overlay paths (slow apply_overlay_window vs the O(1) fast one) and
    # assert they accept the same stream + decode the same final AX — the byte-
    # identity gate for the O(1)-decode optimization.
    vr = verify_blocks(model, L, code, draft, block_steps=block_steps,
                       device=device, evict=evict, prune_interval=prune_interval,
                       mask=mask, fast=True)
    vr_slow = verify_blocks(model, L, code, draft, block_steps=block_steps,
                            device=device, evict=evict, prune_interval=prune_interval,
                            mask=mask, fast=False)
    fast_matches_slow = (
        vr.all_matched == vr_slow.all_matched
        and vr.decoded_final_ax == vr_slow.decoded_final_ax
        and vr.accepted_steps == vr_slow.accepted_steps)
    identical = (drv_trace == spec_trace) and vr.all_matched and fast_matches_slow
    first_div = None
    for i, (a, b) in enumerate(zip(drv_trace, spec_trace)):
        if a != b:
            first_div = {"step": i, "driver_ax": a, "spec_ax": b}
            break
    if first_div is None and len(drv_trace) != len(spec_trace):
        first_div = {"len_driver": len(drv_trace), "len_spec": len(spec_trace)}
    return {
        "identical": identical,
        "n_steps_driver": len(drv_trace),
        "n_steps_spec": len(spec_trace),
        "verify_all_matched": vr.all_matched,
        "verify_forwards": vr.forwards,
        "fast_matches_slow": fast_matches_slow,
        "first_divergence": first_div,
        "driver_final_ax": drv_trace[-1] if drv_trace else None,
        "spec_final_ax": spec_trace[-1] if spec_trace else None,
    }
