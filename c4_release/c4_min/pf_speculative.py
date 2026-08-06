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


def _draft_shift32() -> bool:
    """DRAFT SHL/SHR width selector.  DEFAULT OFF (``C4_SHIFT32`` unset) -> the historical
    8-bit shift draft (``(v & 0xFF) </>> ax & 0xFF``), byte-identical to every prior corpus
    draft.  ON (``C4_SHIFT32=1``) -> the draft shifts the FULL 32-bit operand (``pop </>> ax``
    at 32-bit width, arithmetic SHR sign-fill), matching the MODEL's ``C4_SHIFT32`` tight
    shifter.  REQUIRED for doom's fixed-point shifts and the #829 pow2 DIV->SHR reduction
    (a 32-bit divide reduced to a shift; the 8-bit floor mis-computes it).  Additive + gated:
    OFF reproduces the exact 8-bit draft, so the whole existing corpus is byte-identical."""
    return os.environ.get("C4_SHIFT32", "0") == "1"


def _draft_read_to_mem() -> bool:
    """DRAFT READ-into-memory selector.  DEFAULT OFF (``C4_DRAFT_READ_TO_MEM`` unset)
    -> the historical draft: a READ syscall's input bytes are laid into ``store_log``
    (the KV the MODEL recalls) and the token stream, but NOT into the draft's own
    ``mem`` dict.  ON (``C4_DRAFT_READ_TO_MEM=1``) -> also write each read byte into the
    draft ``mem`` so a later ``LC``/``LI`` in the DRAFT reads the byte (matching the true
    VM / the model, which recalls it from the store_log KV).  REQUIRED for a program
    that reads a buffer from stdin/a file and then re-reads every byte (a c4-compiler
    tokenizing its C source): the OFF draft reads 0 for ``src[i]`` and diverges from the
    model.  doom's single ``read(0,buf,1)`` never re-reads the byte, so its draft is
    byte-identical either way; gated OFF keeps the whole existing corpus + doom draft
    byte-identical."""
    return os.environ.get("C4_DRAFT_READ_TO_MEM", "0") == "1"


def _s32(v: int) -> int:
    """Interpret ``v`` as a signed 32-bit two's-complement integer (C4's ``int``)."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


# BATCHED per-span register decode.  ``None`` -> read the env flag; a test can force
# it via ``set_batched_decode(True/False)``.  See the verify_blocks decode loop.
_BATCHED_DECODE: Optional[bool] = None


def _batched_decode_enabled() -> bool:
    """``C4_BATCHED_DECODE`` (DEFAULT OFF): decode a verify span's query rows in ONE
    device-side batched argmax + a single host copy, instead of ~11 per-step
    ``float(state[dim])`` host<->device syncs.  Bit-identical (same integer argmax
    requant); OFF reproduces the exact per-scalar path (byte-identical golden)."""
    return os.environ.get("C4_BATCHED_DECODE", "0") not in ("0", "", "false", "False")


def set_batched_decode(v: Optional[bool]) -> None:
    global _BATCHED_DECODE
    _BATCHED_DECODE = v


_GPU_VERIFY: Optional[bool] = None


def _gpu_verify_enabled() -> bool:
    """``C4_GPU_VERIFY`` (DEFAULT OFF): run the whole per-span decode+ACCEPT-COMPARE
    on-GPU as ONE vectorized tensor op — batch-decode all K query rows (argmax over
    the register nibbles), compare to the draft's per-step targets (resident on GPU),
    and reduce to the accepted-prefix length with ONE host sync (a single
    ``first_bad.item()``).  This removes the O(steps) host-side Python decode+compare
    loop (~11 host syncs/step; ~330k GPU stalls over a doom program) that dominates
    the whole-verify wall over the ~0.034 ms/step GPU forward.

    Bit-identical to the scalar per-step path: the batched ``_snap_lane_batch`` /
    ``_decode_reg_batch`` evaluate the SAME integer requant-argmax per row as the
    scalar decode, and the compare is the SAME (pc/ax&mask/sp/bp equality, with
    is_file rows a no-op and is_halt rows AX-only).  DEFAULT OFF -> the exact scalar
    loop (golden 069cc32f unchanged)."""
    if _GPU_VERIFY is not None:
        return _GPU_VERIFY
    return os.environ.get("C4_GPU_VERIFY", "0") not in ("0", "", "false", "False")


def set_gpu_verify(v: Optional[bool]) -> None:
    global _GPU_VERIFY
    _GPU_VERIFY = v


def _build_draft_targets(draft, device: str, mask: int):
    """Materialise the draft's per-step ACCEPT targets as device-resident tensors
    ONCE per verify (cached on the draft keyed by (device, mask)).  All are [n_steps]
    long/bool tensors so the per-span compare is a single vectorized op with NO host
    sync.  Cheap: the draft frames are already in host memory (the logical VM ran on
    CPU); this is one pack + one HtoD copy amortised over the whole program."""
    key = (device, int(mask))
    cache = getattr(draft, "_gpu_targets", None)
    if cache is not None and cache.get("_key") == key:
        return cache
    frames = draft.frames
    n = len(frames)
    want_pc = torch.empty(n, dtype=torch.long)
    want_ax = torch.empty(n, dtype=torch.long)
    want_sp = torch.empty(n, dtype=torch.long)
    want_bp = torch.empty(n, dtype=torch.long)
    is_halt = torch.zeros(n, dtype=torch.bool)
    is_file = torch.zeros(n, dtype=torch.bool)
    for i, fr in enumerate(frames):
        want_pc[i] = fr["pc"]
        want_ax[i] = fr["ax"] & mask
        want_sp[i] = fr["sp"] & 0xFFFFFFFF
        want_bp[i] = fr["bp"] & 0xFFFFFFFF
        if fr.get("is_halt"):
            is_halt[i] = True
        if fr.get("is_file"):
            is_file[i] = True
    dev = torch.device(device)
    cache = {
        "_key": key,
        "want_pc": want_pc.to(dev), "want_ax": want_ax.to(dev),
        "want_sp": want_sp.to(dev), "want_bp": want_bp.to(dev),
        "is_halt": is_halt.to(dev), "is_file": is_file.to(dev),
    }
    draft._gpu_targets = cache
    return cache


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


def _overlay_batched_enabled() -> bool:
    """``C4_OVERLAY_BATCHED`` (DEFAULT OFF): assemble the per-row overlay writes on
    the HOST and push them to the device in ONE ``index_put_`` (2 HtoD copies:
    flat-index + values) instead of the per-scalar ``x[0,wi,dim]=float`` loop (each
    a tiny PAGEABLE HtoD).  The profiler pinned that loop as ~21k tiny HtoD/copy_
    ops = ~36% CUDA / ~99% CPU-wall of the span (10s CPU vs 75ms CUDA), so this is
    the real launch-collapse lever, NOT a whole-block megakernel (the block forward
    is only ~75ms of GPU).  Byte-identical: it writes the SAME (row,dim)->float set
    ``apply_overlay_window_fast`` writes, just as ONE batched device op."""
    return os.environ.get("C4_OVERLAY_BATCHED", "0") not in ("0", "", "false", "False")


def _overlay_precompute_enabled() -> bool:
    """``C4_OVERLAY_PRECOMPUTE`` (DEFAULT ON): precompute the POSITION-INVARIANT part
    of the overlay scatter plan ONCE per program (a CSR-like ``pos -> [(dim,val)]``
    over absolute positions) instead of re-running the per-row Python loop (frame-role
    tags / store addr-bits+val-nibbles / code-frame writes) on EVERY chunk of EVERY
    forward.  The profiler pinned that loop (``apply_overlay_window_batched`` 0.17 s
    tottime + ~823k ``list.append``) as the dominant remaining per-forward host wall
    once the CAM gather + weight-densify were fixed.  The plan is a PURE function of
    ``(w_start+wi)`` (the draft's static store_log / code), so precomputing it once and
    SLICING ``[w_start, w_start+W)`` per chunk is byte-identical (same (row,dim)->val
    set).  Only the tiny per-chunk QUERY-ROW all-ROLE tag (part 3) stays runtime (it
    depends on which rows are query rows in the chunk).  OFF -> the per-row loop."""
    return os.environ.get("C4_OVERLAY_PRECOMPUTE", "1") not in ("0", "", "false", "False")


# module-level cache of the position-invariant overlay plan, keyed by the identities
# of the (immutable-per-program) store_log / L / code so different programs don't
# collide.  Value: (n_pos, pos_ptr[np.int64 n_pos+1], pos_dim[np.int64 nnz],
#                    pos_val[np.float32 nnz]).
_OVERLAY_PLAN_CACHE: Dict[tuple, object] = {}


def _build_overlay_plan(w_end: int, L, store_log, code, code_off):
    """Build (or extend) the POSITION-INVARIANT overlay plan for absolute positions
    ``[0, w_end)`` and cache it.  Mirrors ``apply_overlay_window_*``'s per-row writes
    EXACTLY (minus the query-row all-ROLE tag, part 3, which is chunk-dependent) so the
    sliced application is byte-identical.  Returns ``(pos_ptr, pos_dim, pos_val)``
    numpy arrays: ``pos``'s (dim,val) pairs are ``pos_dim[pos_ptr[pos]:pos_ptr[pos+1]]``
    / ``pos_val[...]``."""
    import numpy as _np
    # Key on the program's IDENTITY *and* cheap content signatures (len/code_off) so a
    # freed-then-reused id() can never alias a different program's plan.  Single-entry
    # cache (the verifier processes one program's chunks contiguously): a key miss
    # clears the stale plan.
    key = (id(store_log), id(L), int(code_off), id(code),
           len(store_log), int(getattr(L, "ROLE", 0)))
    cached = _OVERLAY_PLAN_CACHE.get(key)
    if cached is not None and cached[0] >= w_end:
        return cached[1], cached[2], cached[3]
    if cached is None:
        _OVERLAY_PLAN_CACHE.clear()             # new program -> drop the prior plan
    # OVER-BUILD to a generous ceiling so a chunked span (growing w_start per chunk)
    # triggers ONE full build, not a rebuild per chunk (else O(span^2/chunk) host work).
    # The plan is capped by the program's last stored/coded position anyway.
    max_store_f = max(store_log.keys(), default=-1)
    prog_end = (1 + int(code_off) + (max_store_f + 2) * V.FRAME_LEN)
    w_end = min(max(w_end, prog_end), max(w_end * 2, w_end + 65536))
    # (re)build over [0, w_end).  code_off / store_log are static per program.
    _cfm = code_off > 0
    dims: List[int] = []
    valz: List[float] = []
    ptr = _np.zeros(w_end + 1, dtype=_np.int64)
    for p in range(w_end):
        if p == 0:
            ptr[p + 1] = len(dims)
            continue
        if _cfm and 1 <= p <= code_off:
            ins = code[p - 1]
            dims.append(L.IS_CODE); valz.append(1.0)
            dims.append(L.IS_FRAME_BYTE); valz.append(0.0)
            for b in range(CODE_ADDR_BITS):
                dims.append(L.CODE_KEY_BIN + b); valz.append(float(((p - 1) >> b) & 1))
            dims.append(L.CODE_OPV); valz.append(float(ins.op))
            for j, nv in enumerate(V.nibbles_of_value(ins.imm & 0xFFFFFFFF, IMM_NIBS)):
                dims.append(L.CODE_IMM_NIB_MEM + j); valz.append(float(nv))
            ptr[p + 1] = len(dims)
            continue
        f = (p - 1 - code_off) // V.FRAME_LEN
        local = (p - 1 - code_off) % V.FRAME_LEN
        if local in _FRAME_ROLE_SLOTS:
            role = _FRAME_ROLE_SLOTS[local]
            dims.append(L.ROLE + role); valz.append(1.0)
            dims.append(L.IS_FRAME_BYTE); valz.append(1.0)
        if local == _MEM_MARKER_LOCAL and f in store_log:
            addr, val = store_log[f]
            dims.append(L.IS_STORE); valz.append(1.0)
            dims.append(L.IS_FRAME_BYTE); valz.append(0.0)
            for b, bit in enumerate(_address_bits(addr)):
                dims.append(L.ADDR_BIN + b); valz.append(bit)
            for j, nv in enumerate(V.nibbles_of_value(val & 0xFFFFFFFF, NIB_PER_REG)):
                dims.append(L.VAL_NIB + j); valz.append(float(nv))
        ptr[p + 1] = len(dims)
    pos_dim = _np.asarray(dims, dtype=_np.int64)
    pos_val = _np.asarray(valz, dtype=_np.float32)
    _OVERLAY_PLAN_CACHE[key] = (w_end, ptr, pos_dim, pos_val)
    return ptr, pos_dim, pos_val


def apply_overlay_window_batched(x_win: torch.Tensor, w_start: int, L,
                                 store_log: Dict[int, Tuple[int, int]],
                                 code_vec: Tuple[torch.Tensor, torch.Tensor],
                                 query_rows: Optional[List[int]] = None,
                                 code=None, code_off: int = 0) -> None:
    """LAUNCH-COLLAPSED overlay — byte-identical to ``apply_overlay_window_fast``
    but the per-row frame-role / store / query-tag scalar writes are ACCUMULATED on
    the host into flat ``(index, value)`` arrays and applied in ONE device
    ``index_put_`` (a single vectorised scatter) instead of ~21k per-scalar HtoD
    dispatches.  The row-invariant code-in-data is still ONE broadcast assign.

    Flat offset of ``x_win[0, wi, dim]`` (contiguous ``[1, W, D]``) is ``wi*D + dim``.
    Because the fast overlay's per-row writes are all plain ASSIGNS (never ``+=``)
    and never write the same (row,dim) twice within a row, a single scatter of the
    accumulated (offset->value) pairs reproduces the exact residual — L-inf==0.
    """
    import numpy as _np
    _cfm = code_off > 0
    W = x_win.shape[1]
    D = x_win.shape[2]
    # 1) row-INVARIANT program-in-data + ONE — unchanged (already ONE broadcast op).
    code_idx, code_vals = code_vec
    x_win[0, :, code_idx] = code_vals
    # 2) POSITION-INVARIANT per-row scalars (frame-role / store / code).
    if _overlay_precompute_enabled():
        # FAST: slice the precomputed whole-program plan for [w_start, w_start+W) and
        # remap abs pos -> chunk-local flat offset in ONE vectorized numpy pass (no
        # per-row Python loop / ~823k appends).  Byte-identical (same (dim,val) set).
        ptr, pos_dim, pos_val = _build_overlay_plan(w_start + W, L, store_log, code,
                                                    code_off)
        s0 = ptr[w_start]                        # nnz start of the first row (pos w_start)
        s1 = ptr[w_start + W]                     # nnz end of the last row (pos w_start+W-1)
        if s1 > s0:
            dim_sl = pos_dim[s0:s1]
            val_sl = pos_val[s0:s1]
            # per-nnz absolute position -> chunk-local row wi = pos - w_start.
            counts = _np.diff(ptr[w_start:w_start + W + 1])       # [W] nnz per row
            wi_per = _np.repeat(_np.arange(W, dtype=_np.int64), counts)
            base_off = (wi_per * D + dim_sl).astype(_np.int64)
        else:
            base_off = _np.empty(0, dtype=_np.int64)
            val_sl = _np.empty(0, dtype=_np.float32)
        # 3) query-row all-ROLE tag (chunk-dependent — stays runtime, tiny).
        q_iter = ([W - 1] if query_rows is None else query_rows)
        if q_iter:
            roles = _np.arange(N_ROLES, dtype=_np.int64) + L.ROLE
            q_arr = _np.asarray(q_iter, dtype=_np.int64)
            q_off = ((q_arr[:, None] * D) + roles[None, :]).reshape(-1)
            q_val = _np.ones(q_off.size, dtype=_np.float32)
            offs_np = _np.concatenate([base_off, q_off])
            vals_np = _np.concatenate([val_sl, q_val])
        else:
            offs_np, vals_np = base_off, val_sl
        if offs_np.size == 0:
            return
        idx_t = torch.from_numpy(offs_np).to(x_win.device, non_blocking=True)
        val_t = torch.from_numpy(vals_np).to(x_win.device, dtype=x_win.dtype,
                                             non_blocking=True)
        x_win.view(-1).index_put_((idx_t,), val_t)
        return
    # LEGACY per-row loop (C4_OVERLAY_PRECOMPUTE=0).
    offs: List[int] = []
    vals: List[float] = []
    for wi in range(W):
        p = w_start + wi
        base = wi * D
        if p == 0:
            continue
        if _cfm and 1 <= p <= code_off:
            ins = code[p - 1]
            offs.append(base + L.IS_CODE); vals.append(1.0)
            offs.append(base + L.IS_FRAME_BYTE); vals.append(0.0)
            for b in range(CODE_ADDR_BITS):
                offs.append(base + L.CODE_KEY_BIN + b)
                vals.append(float(((p - 1) >> b) & 1))
            offs.append(base + L.CODE_OPV); vals.append(float(ins.op))
            for j, nv in enumerate(V.nibbles_of_value(ins.imm & 0xFFFFFFFF, IMM_NIBS)):
                offs.append(base + L.CODE_IMM_NIB_MEM + j); vals.append(float(nv))
            continue
        f = (p - 1 - code_off) // V.FRAME_LEN
        local = (p - 1 - code_off) % V.FRAME_LEN
        if local in _FRAME_ROLE_SLOTS:
            role = _FRAME_ROLE_SLOTS[local]
            offs.append(base + L.ROLE + role); vals.append(1.0)
            offs.append(base + L.IS_FRAME_BYTE); vals.append(1.0)
        if local == _MEM_MARKER_LOCAL and f in store_log:
            addr, val = store_log[f]
            offs.append(base + L.IS_STORE); vals.append(1.0)
            offs.append(base + L.IS_FRAME_BYTE); vals.append(0.0)
            for b, bit in enumerate(_address_bits(addr)):
                offs.append(base + L.ADDR_BIN + b); vals.append(bit)
            for j, nv in enumerate(V.nibbles_of_value(val & 0xFFFFFFFF, NIB_PER_REG)):
                offs.append(base + L.VAL_NIB + j); vals.append(float(nv))
    # 3) query-row all-ROLE tag.
    q_iter = ([W - 1] if query_rows is None else query_rows)
    for wi in q_iter:
        base = wi * D
        for role in range(N_ROLES):
            offs.append(base + L.ROLE + role); vals.append(1.0)
    if not offs:
        return
    # ONE HtoD each for the flat index + value arrays (built via numpy so the host
    # side is a single vectorised buffer, not a python-list -> tensor per element),
    # then ONE device scatter.  Replaces ~21k tiny pageable HtoD dispatches.
    idx_t = torch.from_numpy(_np.asarray(offs, dtype=_np.int64)).to(
        x_win.device, non_blocking=True)
    val_t = torch.from_numpy(_np.asarray(vals, dtype=_np.float32)).to(
        x_win.device, dtype=x_win.dtype, non_blocking=True)
    x_win.view(-1).index_put_((idx_t,), val_t)


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


@dataclass
class PFResumeState:
    """A CHECKPOINT of the logical-VM interpreter at a step boundary — everything the
    deterministic transition needs to CONTINUE drafting byte-identically.

    Task #814/#866 (checkpoint-resume for multi-million-step doom runs): the draft is
    a self-contained deterministic interpreter, so its FUTURE token stream / frames /
    store_log are a pure function of THIS state.  Saving it lets a run be killed after
    step ``steps`` and RESUMED from here — the resumed continuation is byte-identical
    to an uninterrupted run (same registers -> same control flow, same ``mem`` +
    ``store_log`` -> same memory reads, same stream offsets -> same overlay math).

    The fields mirror the ``draft_pf_program`` locals verbatim.  ``mem`` /
    ``store_log`` are the heavy part (the VM memory image + KV write log); everything
    else is O(1) scalars or short logs.  ``tokens`` / ``frames`` / ``win_starts`` etc.
    hold the emitted PREFIX so a resumed window can present the full token context the
    schedule/verify resolve against (the same prefix the uninterrupted run had).
    ``json_safe()`` returns a dict of only JSON-serialisable primitives; the runner
    persists it with ``numpy`` for the two big dicts."""
    # scalar registers / counters (the whole control + framing state)
    pc: int
    ax: int
    sp: int
    bp: int
    stk: int
    cur_pc: int
    cur_sp: int
    cur_bp: int
    steps: int
    frame_idx: int
    stream_len: int
    n_seed: int
    code_off: int
    halted: bool
    # heavy state: the VM memory image + the KV write log (both {int:...})
    mem: Dict[int, int]
    store_log: Dict[int, Tuple[int, int]]
    load_log: Dict[int, int]
    read_log: Dict[int, List[Tuple[str, int]]]
    # emitted prefix (the token stream + decoded frames up to `steps`)
    tokens: List[int]
    frames: List[Dict[str, int]]
    win_starts: List[int]
    out: List[int]
    prtf_steps: List[int]
    # config guard (a resume MUST use the same draft-width flags)
    cmp32: bool
    shift32: bool
    imm_nibs: int
    sp_init: int


def draft_pf_program(code: List[isa.Instr], max_steps: int = 300000,
                     mask: int = 0xFFFFFFFF,
                     data_seg: Optional[Dict[int, int]] = None,
                     fio=None,
                     resume: Optional["PFResumeState"] = None,
                     capture_state: bool = False) -> PFDraft:
    """Run the MODEL's ISA transition and materialise the per-step 30-token frame
    stream + store_log — the exact token stream the pure-forward DRIVER emits.

    ``resume`` (Task #814/#866, DEFAULT ``None`` -> byte-identical old path): a
    ``PFResumeState`` checkpoint from a prior (bounded) call — the interpreter is
    RE-SEEDED from it and drafts steps ``resume.steps .. max_steps`` INSTEAD of
    starting fresh.  Because the transition is deterministic, the resumed draft's
    tokens/frames/store_log are byte-identical to the tail of an uninterrupted
    ``draft_pf_program(code, max_steps=...)`` (proven L-inf=0 by the runner).  The
    returned ``PFDraft`` carries the FULL stream (checkpoint prefix + the newly drafted
    tail), so a schedule/verify over it is indistinguishable from the uninterrupted run.

    This is the *logical VM* of BLOG_SPEC §Speculation: the plain deterministic
    C4 interpreter, so it is ~free relative to a model forward.  It reproduces the
    driver's own pre-step register bookkeeping (which address a PSH/JSR/ENT/SI
    writes) so the drafted store_log is byte-identical to what the driver records.

    The transition matches the MODEL (not ``ref_interpret``):
      * IMM keeps the full literal ``imm & _IMM_MASK`` (5- or 8-nibble band; see
        ``C4_IMM_NIBS``) — the model's ``compile_imm_ax_nibbles`` writes all fetched
        nibbles;
      * LEA is folded to 8 bits by default (``compile_ax_byte_to_nibbles`` +
        ``_fold_ax_gated``), or the FULL 32-bit frame address ``BP + 4*imm`` under the
        wide gate (``_lea_wide_enabled`` — ``vm_width32() and C4_LEA_WIDE!=0``), matching
        a42384's ``lea-wide`` model blocks and the Rust c4vm32 full-address reference;
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
    # WIDE LEA draft (#824, re-gated #854): mirror the MODEL's gate EXACTLY by importing
    # ``_lea_wide_enabled`` (``C4_LEA_WIDE!=0`` AND the fine-grained wide set — NOT the
    # buggy ``C4_VM_WIDTH32`` substrate) — reusing the SAME predicate (not a copy)
    # guarantees the draft folds iff the model folds, in BOTH configs.  DEFAULT OFF ->
    # LEA drafts folded ``& 0xFF`` (byte-identical golden).  Computed once here so it is
    # available on both the RESUME and the fresh-start path below.
    lea_wide = _PFC._lea_wide_enabled()  # full 32-bit ``BP + 4*imm`` when ON
    if resume is not None:
        # RESUME (Task #814/#866): re-seed the interpreter from a checkpoint and draft
        # the tail.  The draft-width flags MUST match the checkpoint (a different
        # cmp32/shift32/IMM_NIBS/SP_INIT would produce a different transition and break
        # byte-exactness), so assert them.
        assert resume.cmp32 == cmp32, (
            f"resume cmp32={resume.cmp32} != current {cmp32} (set C4_DRAFT_CMP32 to match)")
        assert resume.shift32 == _draft_shift32(), (
            "resume shift32 mismatch (set C4_SHIFT32 to match the checkpoint)")
        assert resume.imm_nibs == IMM_NIBS, (
            f"resume IMM_NIBS={resume.imm_nibs} != current {IMM_NIBS} (set C4_IMM_NIBS)")
        assert resume.sp_init == SP_INIT, (
            f"resume SP_INIT={resume.sp_init:#x} != current {SP_INIT:#x}")
        assert resume.code_off == (len(code) if _pf_cfm_enabled() else 0), (
            "resume code_off mismatch (C4_PF_CFM / code length differs)")
        mem = dict(resume.mem)
        store_log = dict(resume.store_log)
        load_log = dict(resume.load_log)
        read_log = {k: list(v) for k, v in resume.read_log.items()}
        tokens = list(resume.tokens)
        frames = [dict(f) for f in resume.frames]
        win_starts = list(resume.win_starts)
        out = list(resume.out)
        prtf_steps = list(resume.prtf_steps)
        pc, ax, sp, bp, stk = (resume.pc, resume.ax, resume.sp, resume.bp, resume.stk)
        cur_pc, cur_sp, cur_bp = resume.cur_pc, resume.cur_sp, resume.cur_bp
        steps, frame_idx, stream_len = resume.steps, resume.frame_idx, resume.stream_len
        n_seed, code_off, halted = resume.n_seed, resume.code_off, resume.halted
    else:
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
                           "is_file": True,
                           # DIRECT-CAM (additive): #extra store frames this file op emits
                           # (READ input bytes) AFTER its primary frame, so the direct-CAM
                           # frame_idx->step map can advance the frame counter past them.
                           "n_byte_stores": len(byte_stores)})
            win_starts.append(stream_len - 1)
            frame_idx += 1
            tokens += _build_frame(pc, ax, sp, bp, stk)
            stream_len += V.FRAME_LEN
            for (baddr, bval) in byte_stores:
                frame_idx += 1
                store_log[frame_idx] = (baddr & 0xFFFFFFFF, bval & 0xFF)
                # C4_DRAFT_READ_TO_MEM (default OFF): also lay READ's input bytes into
                # the draft's OWN `mem` dict so a later LC/LI(src[i]) in the DRAFT reads
                # the byte (not 0).  The model already recalls it from store_log KV; this
                # only fixes the draft-side oracle for programs that read stdin/a file and
                # then re-read every byte (a c4-compiler reading its C source).  doom's
                # single read(0,buf,1) never re-reads, so its draft is unaffected either
                # way; gated OFF keeps the corpus/doom draft byte-identical.
                if _draft_read_to_mem():
                    mem[baddr & 0xFFFFFFFF] = bval & 0xFF
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
            # LEA = frame-relative pointer ``AX = BP + 4*imm``.  DEFAULT (folded, #824
            # OFF): the model's ``compile_ax_byte_to_nibbles`` writes only AX byte 0, so
            # the draft mirrors the fold ``& 0xFF`` (byte-identical to every prior draft;
            # golden 069cc32f / CFM 7d19cdc3 untouched).  WIDE (``_lea_wide_enabled`` —
            # the SAME gate a42384's model port uses: ``vm_width32() and C4_LEA_WIDE!=0``):
            # the ``lea-wide`` blocks recompute the FULL 32-bit address (byte-ripple ADD,
            # sign-extended imm), so ``LEA 40`` off BP=0x10000 -> 0x100a0 and ``LEA -18``
            # -> BP-72 (0xffb8 low half) — matching the wide model AND the Rust c4vm32
            # full-address reference.  ``& mask`` (full 32-bit) keeps the drafted AX equal
            # to the model's wide result; ``_s32``-style negative offsets ripple naturally.
            if lea_wide:
                ax = (bp + 4 * imm) & mask
            else:
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
            pop_val = mem.get(sp, 0); sp += 4
            if op in (isa.SHL, isa.SHR) and _draft_shift32():
                # 32-bit SHL/SHR (matches the model's C4_SHIFT32 tight shifter:
                # ``pop </>> ax`` at full width, arithmetic SHR sign-fill).  REQUIRED for
                # doom's fixed-point shifts AND the #829 pow2 DIV->SHR / MOD->AND reduction
                # (which turns a 32-bit divide into a 32-bit shift; the 8-bit floor would
                # mis-compute ``519 >> 3`` as 0 instead of 64).  Gated: OFF -> the 8-bit
                # floor (byte-identical to the existing corpus draft).
                v32 = pop_val & 0xFFFFFFFF
                if op == isa.SHL:
                    ax = (v32 << (ax & 31)) & 0xFFFFFFFF
                else:
                    ax = (_s32(v32) >> (ax & 31)) & 0xFFFFFFFF
            else:
                v = pop_val & 0xFF
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
    d = PFDraft(tokens=tokens, frames=frames, store_log=store_log,
                step_count=len(frames), halted=halted,
                final_ax_masked=final_ax, win_starts=win_starts,
                out=out, prtf_steps=prtf_steps, load_log=load_log,
                read_log=read_log, code_off=code_off)
    if capture_state:
        # Task #814/#866: expose the interpreter state so the caller can CHECKPOINT it
        # and RESUME a bounded run.  Only meaningful when the draft stopped on the
        # step-cap (``not halted``); a halted draft has no continuation.  The state
        # mirrors the loop locals exactly (see PFResumeState) so a re-seeded draft is
        # byte-identical to the uninterrupted tail.
        d.resume_state = PFResumeState(
            pc=pc, ax=ax, sp=sp, bp=bp, stk=stk,
            cur_pc=cur_pc, cur_sp=cur_sp, cur_bp=cur_bp,
            steps=steps, frame_idx=frame_idx, stream_len=stream_len,
            n_seed=n_seed, code_off=code_off, halted=halted,
            mem=dict(mem), store_log=dict(store_log),
            load_log=dict(load_log),
            read_log={k: list(v) for k, v in read_log.items()},
            tokens=list(tokens), frames=[dict(f) for f in frames],
            win_starts=list(win_starts), out=list(out),
            prtf_steps=list(prtf_steps),
            cmp32=cmp32, shift32=_draft_shift32(), imm_nibs=IMM_NIBS,
            sp_init=SP_INIT)
    return d


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


def faithful_attn_evict_enabled() -> bool:
    """``C4_FAITHFUL_ATTN_EVICT`` (DEFAULT OFF): the GENUINELY-COMPUTING path.

    Runs the REAL softmax1+ALiBi attention verify — the model's OWN query
    INDEPENDENTLY resolves every memory read's ADDRESS *and* VALUE (no
    ``resolve_load_rows(draft)`` injection) — made tractable at doom's KV scale by
    scoring over the EVICTED / bounded KV cache (``exact_evict`` liveness schedule)
    instead of the full O(S) store log that OOMs.

    This is the resolution of the faithfulness/speed tension the fast-path audit
    (``DOOM_FASTPATH_FAITHFULNESS_AUDIT_2026_08_05.md``) found: the direct-CAM fast
    path is DRAFT-TRUSTED for the memory-read resolution (a self-consistent wrong
    draft is ACCEPTED — scenario E), whereas the softmax path independently
    recomputes it (a wrong read is CAUGHT).  The un-composed softmax path OOMs at
    doom scale (O(S) score over 150k-262k rows); composing it with EVICTION bounds
    the cache to ~1-10K live rows (evicted entries contribute ~0 to softmax1 by the
    latest-write-wins / ALiBi + ZFOD identity), so real-attention-over-evicted ==
    real-attention-over-full-log, byte-exact.

    When ON this flag OVERRIDES the B-class draft-trust levers to their GENUINE
    forms (direct-CAM / direct-local / frozen-row-skip forced OFF — those inject or
    depend on the draft's resolved rows), while KEEPING every A-class faithful lever
    (dead-block fusion, fused-delta FFN, fused megablock, bounded + evicted KV,
    banded/flash local attention).  The result is the real transformer attention,
    scored over the bounded survivors.  DEFAULT OFF -> the golden 069cc32f path
    (this flag is inert when unset)."""
    import os as _os
    return _os.environ.get("C4_FAITHFUL_ATTN_EVICT", "0") not in ("0", "", "false", "False")


def _dead_block_fusion_enabled() -> bool:
    """``C4_DEAD_BLOCK_FUSION`` (DEFAULT OFF): bypass the WHOLE attention sublayer of
    every DEAD-attention block (0 live-value heads) and score ONLY the live head-slots
    on the rest.  A dead block's attention output is provably ``x`` (all heads
    ``_zero_attn`` -> ``attn@V == 0``, ``W_o`` slice 0), so this is byte-exact.  It is
    the ZERO-ATTENTION-COMPUTE lever (#856): the audit pinned the composed forward's
    residual attention arithmetic entirely on the ~238 dead blocks' windowed/banded
    local softmax (the 4 live blocks are already direct-CAM/direct-local resolved with
    zero score); fusing them removes ALL of it -> the step is FFN + O(1) gathers.
    Composes with direct-CAM/direct-local (the 4 live blocks carry live heads, so they
    are never fused; their own forward is then installed on top).  OFF -> the windowed
    softmax path (golden 069cc32f unchanged)."""
    import os as _os
    return _os.environ.get("C4_DEAD_BLOCK_FUSION", "0") not in ("0", "", "false", "False")


def _evict_timed_enabled() -> bool:
    """``C4_EVICT_TIMED`` (DEFAULT OFF): drain (``torch.cuda.synchronize``) around each
    eviction round so ``stats['evict_secs']`` is a clean wall breakdown.  This costs 2
    host syncs PER eviction round — the last eviction host-sync cost once the scheduled
    drop itself is GPU-vectorized (``evict_all_blocks_scheduled`` batched-nonzero).  OFF
    (default) -> the eviction kernels are enqueued on the stream with NO explicit drain
    (they still order correctly before the next forward's kernels — same stream), so
    eviction adds ~ZERO host syncs per FORWARD.  ON -> the timed path (``evict_secs``
    populated, at 2 syncs/round).  Byte-identical either way (a synchronize changes
    only WHEN the host waits, never the compute)."""
    import os as _os
    return _os.environ.get("C4_EVICT_TIMED", "0") not in ("0", "", "false", "False")


def _frozen_skip_enabled() -> bool:
    """``C4_FROZEN_ROW_SKIP`` (DEFAULT OFF): after the last block that reads a
    frozen (non-query) row's KV via a live LOCAL head (the frame-ingest block 0),
    run the remaining blocks on ONLY the query rows.  Frozen columns' downstream
    output is never decoded and never read cross-column (blocks 1..N are dead-attn
    passthrough or direct-CAM'd whose live heads are GLOBAL and position-gathered,
    not KV-reading), so this is BYTE-EXACT — the query-row decode is identical
    (probe ``_agent_frozen_skip_probe``: 0/256 rows differ).  It cuts the FFN + block
    stack from S~=30k span rows to K~=256 query rows (~30x less FFN work).  Requires
    direct-CAM (else blocks 7/11 would read frozen-row KV)."""
    import os as _os
    # FAITHFUL-ATTN-EVICT: the genuine CAM heads (blocks 2/7/11) SCORE the frozen
    # rows' KV (that IS the memory the model's query retrieves), so frozen-row-skip is
    # UNSAFE on this path — the docstring's "direct-CAM'd whose live heads are GLOBAL
    # and position-gathered" premise does not hold when the global heads genuinely
    # score.  Force it OFF so the faithful path reads the full committed frozen KV.
    if faithful_attn_evict_enabled():
        return False
    return _os.environ.get("C4_FROZEN_ROW_SKIP", "0") not in ("0", "", "false", "False")


def _frozen_skip_cut(model) -> int:
    """The cut block index: run blocks [0, cut) on ALL rows (commit their KV), run
    blocks [cut, N) on QUERY rows only.  ``cut`` = 1 + (index of the last block with
    a live LOCAL head).  A block has a live local head iff it has a ``_global_head_mask``
    AND some non-global head is a live-value head (``local_attention.live_value_heads``).
    For the doom config the frame-ingest block 0 is the only such block -> cut = 1."""
    from .local_attention import live_value_heads
    cut = 0
    for bi, blk in enumerate(model.blocks):
        at = blk.attn
        gmask = getattr(at, "_global_head_mask", None)
        if gmask is None:
            continue
        live = set(int(h) for h in live_value_heads(at))
        # a live head that is NOT global == a live LOCAL head (reads frozen-row KV).
        local_live = any((h in live) and (not bool(gmask[h])) for h in range(at.n_heads))
        if local_live:
            cut = bi + 1
    return max(cut, 1)


def _cut_span_chunk() -> int:
    """``C4_CUT_SPAN_CHUNK`` (default 0 == OFF -> the whole-span [0,cut) forward,
    byte-identical to the prior frozen-skip path).  When > 0, the ``frozen_skip``
    [0, cut) forward (block 0 = the frame-ingest, the SOLE block that runs over all
    S = K*30 rows) is processed in CHUNKS of this many span rows, so it NEVER holds the
    whole-span attention + FFN activations ([1, S, ffn_hidden] ~= 4 GB and the [1,S,D]
    W_o output) at once — the LAST O(K*30) tensor on the composed forward and the true
    eff_K cap (measured: block 0 over S rows peaks 16.8 GB at K=8192; blocks [cut,N)
    over K query rows peak only 7.8 GB).

    BYTE-EXACT + WHY IT NEEDS DIRECT-LOCAL + DROP-DEAD-KV.  Under direct-local-CAM block
    0's attention is a PER-ROW direct gather (no cross-row score), and its FFN is per-row,
    so block 0's output at row ``i`` depends ONLY on row ``i`` — chunking the S axis is
    bit-identical.  Blocks [cut, N) already run on the K query rows only, so we need block
    0's output ONLY at those query rows (frozen rows are never read downstream).  We run
    block 0 chunk-by-chunk, keep each chunk's QUERY-row outputs, and drop the frozen rows'
    outputs AND block 0's (provably-dead) KV.  Requires C4_DIRECT_LOCAL_CAM (per-row block
    0) and C4_BLOCK0_DROP_DEAD_KV (else block 0 would need the full-span KV committed).
    """
    import os as _os
    try:
        return int(_os.environ.get("C4_CUT_SPAN_CHUNK", "0"))
    except ValueError:
        return 0


def _qrow_chunk(default: int) -> int:
    """``C4_QROW_CHUNK`` (default == the cut-span chunk value when unset): chunk the
    blocks-[cut, N) stack over the K QUERY rows.  These blocks are per-row independent
    (dead-attn identity / direct-CAM O(1) per-row gather + per-row FFN — no cross-row
    attention), so chunking the K axis is byte-exact and holds the downstream FFN peak
    at O(qchunk*ffn_hidden), FLAT in K.  This is the SECOND cap-lifter: with block 0
    Sq-chunked AND the query-row FFN chunked, the ONLY tensor still O(K) is the [1,K,D]
    block-0-output buffer + the [1,S,D] span embed (D, not ffn_hidden) — so K can grow
    toward 100k-1M (the span embed is the residual floor, ~D*30 bytes/step).  Set 0 to
    disable (whole-K query-row stack)."""
    import os as _os
    v = _os.environ.get("C4_QROW_CHUNK")
    if v is None:
        return int(default)
    try:
        return int(v)
    except ValueError:
        return int(default)


def _stream_embed_enabled() -> bool:
    """``C4_STREAM_EMBED`` (default OFF): build each block-0 S-chunk's embed + overlay ON
    DEMAND inside the cut-span-chunk loop, so the full ``[1, S, D]`` span embed (the LAST
    O(K*30) tensor once block 0 and the query-row FFN are chunked) is NEVER materialised —
    only ``chunk`` rows of embed exist at a time.  This is the FINAL cap-lifter: with it,
    the per-forward VRAM is O(chunk*D + K*D) (the block-0-output [1,K,D] buffer, at D not
    ffn_hidden), so eff_K can go past 100k toward 1M on a 24GB card.  Requires
    C4_CUT_SPAN_CHUNK (+ direct-local + drop-dead-kv); byte-exact (per-row embed+overlay,
    same w_start math)."""
    import os as _os
    return _os.environ.get("C4_STREAM_EMBED", "0") not in ("0", "", "false", "False")


def _stream_embed_chunk(ctx, lo0, hi0):
    """Build ONE [1, hi0-lo0, D] chunk of the span's overlaid embed on demand (STREAM
    mode) — the same embed gather + overlay the whole-span path does, restricted to span
    rows [lo0, hi0).  Byte-exact: ``apply_overlay_window_*`` is a per-row write keyed by
    the absolute position ``span_start + row``, so overlaying a chunk with its own
    ``w_start = span_start + lo0`` and chunk-local query rows reproduces exactly those
    rows' residual.  The full [1, S, D] span embed never exists — only a chunk at a time."""
    embed = ctx["embed"]
    span_start = ctx["span_start"]
    win_toks = ctx["win_toks"]
    xc = embed[win_toks[:, lo0:hi0]]                      # [1, hi0-lo0, D] fresh copy
    # chunk-local query rows (absolute q positions that fall inside [lo0, hi0)).
    q_local_chunk = [ql - lo0 for ql in ctx["q_local"] if lo0 <= ql < hi0]
    _overlay = (apply_overlay_window_batched if ctx["overlay_batched"]
                else apply_overlay_window_fast)
    _overlay(xc, span_start + lo0, ctx["L"], ctx["store_log"], ctx["code_vec"],
             query_rows=q_local_chunk, code=ctx["code"], code_off=ctx["code_off"])
    return xc


def _run_qstack(model, h, qp, cut, n, megastep=None):
    """Forward the K query rows ``h [1,K,D]`` through the frozen-skip block region
    ``[cut, n)`` — the dead-FFN + live-CAM chain — and return ``[1,K,D]``.

    ``megastep`` (C4_FUSED_MEGABLOCK / C4_GRAPH_MEGAKERNEL): a ``MegaBlockRegion`` /
    ``MegaStepGraph`` whose ``run(h, qp)`` runs the dead-FFN segments as ONE on-chip
    fused/graphed launch (residual L2-resident) + the live CAM blocks eagerly —
    byte-exact to the per-block loop (dead-block fusion makes attention the identity
    on every dead block, so ``past_kv=None`` is correct; the mega-chain is the same
    nonzeros in the same order).  ``None`` -> the eager per-block loop.

    This is THE megablock wiring for the O(K) cut-span-chunk (giant-K) path: every
    per-query-row-chunk block stack now routes through the megakernel (previously
    ONLY the whole-span path did), so the megablock FIRES at giant K where the O(K)
    band is essential.  Byte-exact; default OFF (megastep is None) -> unchanged."""
    if megastep is not None:
        return megastep.run(h, qp)
    for b in range(cut, n):
        h, _ = model.blocks[b](h, past_kv=None, q_positions=qp, use_cache=True)
    return h


def _forward_hidden_cached_frozen_skip(model, x, past_key_values, q_positions,
                                       q_local_idx, cut, megastep=None,
                                       stream_ctx=None, block0_graph=None,
                                       whole_step_graph=None):
    """Run blocks [0, cut) over ALL rows (commit KV), then blocks [cut, N) over ONLY
    the query rows (``q_local_idx`` — window-local indices).  Returns
    ``(hidden_full, new_caches)`` where ``hidden_full`` is [1, S, D] with the QUERY
    rows overwritten by their fully-forwarded state (frozen rows left at their
    post-cut-block value, which the decode never reads).  ``new_caches[b]`` for
    b<cut is the full-row KV (committed as usual); for b>=cut it is None (that
    block's KV is never re-read — dead-attn / direct-CAM).

    ``megastep`` (C4_GRAPH_MEGAKERNEL): a ``MegaStepGraph`` that CUDA-graphs the
    dead-FFN segments of the ``[cut, N)`` block loop into one-launch-per-segment
    replays (byte-exact; the graphed FFN chain is the same dense GEMM in the same
    order as the eager loop, attention identity throughout).  None -> the eager
    per-block loop."""
    n = len(model.blocks)
    if past_key_values is None:
        past_key_values = [None] * n
    new_caches = [None] * n
    qi = torch.tensor(q_local_idx, device=x.device, dtype=torch.long)
    qp_q = q_positions.index_select(0, qi) if q_positions is not None else None

    # ---- CUT-SPAN-CHUNKED PATH (C4_CUT_SPAN_CHUNK > 0): the big-K VRAM cap lifter -----
    # Block 0 (= the [0,cut) blocks; cut==1 on doom) is the ONLY block over all S rows.
    # It is per-row independent under direct-local-CAM, so run it in S-chunks and keep
    # ONLY the query rows' block-0 output (frozen rows never read downstream); block 0's
    # KV is dead so we commit None.  This holds block-0 peak at O(chunk*ffn_hidden) and
    # removes the last O(K*30) activation on the forward.  Byte-exact (per-row map).
    chunk = _cut_span_chunk()
    # STREAM-EMBED: when the caller streams the embed, x is a THROWAWAY 1-row placeholder
    # ([1,1,D]) and the true span length S comes from the stream context (win_toks); we
    # build each chunk's [1,chunk,D] embed+overlay on demand so the full [1,S,D] never
    # exists.  Otherwise x IS the pre-built [1,S,D] span embed and S = x.shape[1].
    streaming = stream_ctx is not None
    S = int(stream_ctx["win_toks"].shape[1]) if streaming else x.shape[1]
    if chunk > 0 and S > chunk and cut >= 1:
        from .direct_local_cam import _block0_drop_dead_kv
        Dm = model.embed.shape[1]
        # collect block-0 output at the query rows, chunk by chunk.
        hq0 = model.embed.new_empty(1, qi.numel(), Dm)        # [1, K, D] block-0 out @ q
        # absolute span-local position -> index into q_local (for query rows in a chunk).
        qpos_to_qi = torch.full((S,), -1, device=hq0.device, dtype=torch.long)
        qpos_to_qi[qi] = torch.arange(qi.numel(), device=hq0.device)
        # WHOLE-STEP GRAPH (RUNG 2): precompute, ON THE HOST, per-chunk (chunk-local
        # query rows -> hq0 destination) index pairs so the block-0 loop does NO
        # bool(.any()) device sync (the membership is known from the draft's win_starts).
        # ``qi`` are the span-local query rows; a chunk [lo0,hi0) owns the query rows in
        # that range.  This kills the ~S/chunk (=~640) per-chunk host syncs.
        _wsg = (whole_step_graph if (whole_step_graph is not None and cut == 1) else None)
        _wsg_plan = None
        if _wsg is not None:
            from .whole_step_graph import Block0LoopPlan
            _b0attn = model.blocks[0].attn
            _pm = getattr(_b0attn, "_direct_local_pos_map", None)
            _rf = getattr(_b0attn, "_direct_local_rf", None)
            if _pm is not None and _rf is not None:
                _wsg_plan = Block0LoopPlan(_rf, _pm, chunk, S,
                                           stream_ctx["span_start"] if streaming
                                           else 0,
                                           q_positions, qpos_to_qi, hq0.device)
                # host-side per-chunk query membership (from the draft's win_starts,
                # via qpos_to_qi) — NO device sync in the loop below.
                _chunk_qrows = {}      # lo0 -> (chunk_local_rows_t, hq0_dest_t)
                _local_all = qpos_to_qi.cpu().tolist()
                for _lo in range(0, S, chunk):
                    _hi = min(_lo + chunk, S)
                    _rows = [(p - _lo, _local_all[p]) for p in range(_lo, _hi)
                             if _local_all[p] >= 0]
                    if _rows:
                        _cl = torch.tensor([r[0] for r in _rows], device=hq0.device,
                                           dtype=torch.long)
                        _dst = torch.tensor([r[1] for r in _rows], device=hq0.device,
                                            dtype=torch.long)
                        _chunk_qrows[_lo] = (_cl, _dst)
            else:
                _wsg = None
        for lo0 in range(0, S, chunk):
            hi0 = min(lo0 + chunk, S)
            if streaming:
                xc = _stream_embed_chunk(stream_ctx, lo0, hi0)   # [1, hi0-lo0, D]
            else:
                xc = x[:, lo0:hi0, :]
            qpc = (q_positions[lo0:hi0] if q_positions is not None
                   else torch.arange(stream_ctx["span_start"] + lo0,
                                     stream_ctx["span_start"] + hi0,
                                     device=hq0.device)) if streaming else \
                  (q_positions[lo0:hi0] if q_positions is not None else None)
            # WHOLE-STEP GRAPH (RUNG 2): fold the ingest gather INTO the graph (static
            # nibble inputs) + replay -> ONE graph launch + 3 static copies (no separate
            # gather new_zeros/scatter, no sync).  Then a SYNC-FREE precomputed scatter.
            if _wsg is not None:
                Sc = hi0 - lo0
                nib_lo, nib_hi = _wsg_plan.chunk_nibbles(qpc, Sc)
                hc = _wsg.run(xc, nib_lo, nib_hi)                 # [1, Sc, D]
                cq = _chunk_qrows.get(lo0)
                if cq is not None:
                    _cl, _dst = cq
                    hq0[0, _dst] = hc[0, _cl]
                del hc, xc
                continue
            # BLOCK-0 GRAPH: replay the fixed-shape ingest attn + FFN body as ONE launch
            # (the ingest gather is computed here, outside the graph, into the graph's
            # static input; the graph does the W_o + SwiGLU GEMMs).  Byte-exact to the
            # eager block-0 forward (same gather, same GEMM chain).  Only wired for the
            # cut==1 doom ingest block; falls back to the eager loop otherwise.
            if block0_graph is not None and cut == 1:
                b0attn = model.blocks[0].attn
                outc = b0attn.gather_ingest_out(xc, qpc)          # [1, H, Sc, HD]
                hc = block0_graph.run(xc, outc)                   # [1, Sc, D]
                del outc
            else:
                hc = xc
                for b in range(cut):
                    hc, _ = model.blocks[b](hc, past_kv=None, q_positions=qpc,
                                            use_cache=True)
            # scatter this chunk's QUERY rows into hq0.
            local_q = qpos_to_qi[lo0:hi0]
            m = local_q >= 0
            if bool(m.any()):
                hq0[0, local_q[m]] = hc[0, m.nonzero(as_tuple=False).flatten()]
            del hc, xc
        # blocks [cut, N) over the K query rows.  These blocks are ALSO per-row
        # independent (dead-attn identity / direct-CAM O(1) per-row gather from the draft
        # + per-row FFN — NO cross-query-row attention), so the K-query-row axis can be
        # CHUNKED too.  This is what removes the LAST O(K) activation ([1, K, ffn_hidden]
        # over ~241 blocks — the wall once block 0 is Sq-chunked) and lets K grow past
        # ~24k toward 100k-1M: with query-row chunking the per-forward peak is
        # O(qchunk*ffn_hidden), FLAT in K.  ``C4_QROW_CHUNK`` (default == the cut-span
        # chunk) tunes it; the block-0 output ``hq0`` is [1,K,D] (cheap, D not ffn_hidden)
        # so it is fine to hold whole while chunking the FFN stack.  Byte-exact (per-row).
        qchunk = _qrow_chunk(chunk)
        nq = hq0.shape[1]
        _ = _block0_drop_dead_kv()   # documents the requirement; block-0 KV stays None
        # OUTPUT buffer the caller's decode indexes.  In STREAM mode we return the K
        # QUERY rows in DRAFT ORDER as a compact [1, K, D] (30x smaller than [1,S,D]) —
        # the decode reads it by ``s - step`` (query order == q_local_idx order == hq0
        # order), so the last O(K*30) tensor (the [1,S,D] span/output) is gone and only
        # [1,K,D] remains: this is what lets K go past 100k toward 1M.  In non-stream mode
        # x IS the [1,S,D] span embed and we scatter query rows in place (decode by
        # span-local position, unchanged).  Byte-exact — the SAME query-row states either
        # way; only the container shape + the decode's index basis differ.
        if streaming:
            hq = hq0
            if qchunk > 0 and nq > qchunk:
                for lo in range(0, nq, qchunk):
                    hi = min(lo + qchunk, nq)
                    hqc = hq0[:, lo:hi, :]
                    qpc = qp_q[lo:hi] if qp_q is not None else None
                    hqc = _run_qstack(model, hqc, qpc, cut, n, megastep)
                    hq0[:, lo:hi, :] = hqc          # in-place into the [1,K,D] buffer
                    del hqc
                hq = hq0
            else:
                hq = _run_qstack(model, hq, qp_q, cut, n, megastep)
            # tag the result so the caller's decode indexes by query order (s - step).
            hq._c4_query_ordered = True    # note: attr may not persist through ops; the
            return hq, new_caches           # caller keys on ``streaming`` instead (below).
        out = x
        if qchunk > 0 and nq > qchunk:
            for lo in range(0, nq, qchunk):
                hi = min(lo + qchunk, nq)
                hqc = hq0[:, lo:hi, :]
                qpc = qp_q[lo:hi] if qp_q is not None else None
                hqc = _run_qstack(model, hqc, qpc, cut, n, megastep)
                # scatter this query-row chunk's final state back in place.
                out[:, qi[lo:hi], :] = hqc
                del hqc
            return out, new_caches
        # whole-K query-row stack (qchunk off).
        hq = _run_qstack(model, hq0, qp_q, cut, n, megastep)
        # scatter the query rows' final state back IN PLACE into the output buffer (avoids
        # a second [1,S,D] clone — the frozen rows are NEVER read by the decode; only the
        # query rows are, and we overwrite exactly those).  new_caches[0] = None (dead KV)
        # — byte-safe (drop-dead-kv required + audited).
        out[:, qi, :] = hq
        return out, new_caches

    # ---- WHOLE-SPAN PATH (chunk OFF): byte-identical to the prior frozen-skip ----------
    hidden = x
    for b in range(cut):
        hidden, kv = model.blocks[b](
            hidden, past_kv=past_key_values[b], q_positions=q_positions, use_cache=True)
        new_caches[b] = kv
    # gather query rows + their absolute positions, forward ONLY them through [cut, N).
    hq = hidden[:, qi, :]                                    # [1, K, D]
    # MEGAKERNEL: replay the dead-FFN-segment CUDA graphs (one launch/segment) + run
    # the live CAM blocks eagerly — byte-exact to the per-block loop (megastep None ->
    # the eager loop).
    hq = _run_qstack(model, hq, qp_q, cut, n, megastep)
    # scatter the query rows' final state back into a full [1, S, D] so the caller's
    # decode loop (indexing by span-local query position) is unchanged.
    hidden = hidden.clone()
    hidden[:, qi, :] = hq
    return hidden, new_caches


def _forward_hidden_cached_mask(model, x, past_key_values, q_positions, run_mask):
    """Run ONLY the blocks flagged True in ``run_mask`` (len == n_blocks); every
    False block is an IDENTITY expert (residual passes straight through, its cache
    entry is None).  Byte-identical to the full forward iff ``run_mask`` is a
    SUPERSET of every step-in-the-span's true decode-live block set — the batched
    block-MoE contract (a skipped block's attention is identity AND its FFN writes
    only dead scratch no live block reads).  ``batched_block_skip`` builds the mask
    as the union of the span's opcodes' live sets."""
    n = len(model.blocks)
    if past_key_values is None:
        past_key_values = [None] * n
    new_caches = [None] * n
    hidden = x
    for b in range(n):
        if not run_mask[b]:
            continue
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
    # PRECOMPUTED SCHEDULE + SINGLE DISPATCH (C4_PRECOMPUTED_SCHEDULE, default OFF).
    # The whole K-batch verify as ONE precomputed schedule (compacted routing + all
    # direct-CAM/direct-local gathers resolved up front + decode plan) + one graph-per-
    # chunk dispatch — ZERO per-op Python in the loop (host syncs ~3600/forward -> O(1)).
    # Byte-identical to this composed GPU-verify at every query row (same gathers, same
    # GEMM chains, same requant decode+compare).  Requires CUDA + a DIV-free program
    # (the divmod span is not carried); a DIV/MOD program falls through to the per-op
    # path below.  Default OFF -> the per-op verify_blocks (golden 069cc32f unchanged).
    import os as _osp
    # FAITHFUL-ATTN-EVICT forces the per-op verify path: the precomputed schedule
    # resolves the direct-CAM/direct-local gathers UP FRONT off the draft (B-class
    # draft-trust) and compacts the routing, neither of which is the genuine softmax
    # the faithful path runs.  So even if C4_PRECOMPUTED_SCHEDULE is ambiently set,
    # faithful mode falls through to the real per-op scored path below.
    # FAITHFUL SINGLE-DISPATCH (C4_FAITHFUL_SINGLE_DISPATCH): keep the FAST single
    # dispatch but make it GENUINE — independently verify routing / read-address /
    # read-value against the MODEL's own computation (the C4_FAITHFUL_ATTN_EVICT value
    # re-resolution wired into the single dispatch, NOT the verify_blocks fallback).  It
    # does NOT force single-dispatch off (unlike faithful-attn-evict); it rides on it.
    from .faithful_single_dispatch import faithful_single_dispatch_enabled as _fsd
    _faithful_sd = (_fsd() and device.startswith("cuda")
                    and not faithful_attn_evict_enabled())
    if (_osp.environ.get("C4_PRECOMPUTED_SCHEDULE", "0") not in ("0", "", "false", "False")
            and device.startswith("cuda")
            and (_faithful_sd or not faithful_attn_evict_enabled())):
        _DM = {"DIV", "MOD"}
        if not any(draft.frames[s].get("op") in _DM for s in range(draft.step_count)):
            if _faithful_sd:
                from .precomputed_schedule import run_faithful_verify as _ps_run
            else:
                from .precomputed_schedule import run_verify as _ps_run
            pr = _ps_run(model, L, code, draft, device, mask=mask,
                         collect_out=collect_out, stats=stats)
            if stats is not None:
                stats.setdefault("forwards", 1)
            return VerifyResult(
                accepted_steps=pr.accepted_steps, total_steps=pr.total_steps,
                all_matched=pr.all_matched, forwards=1,
                first_mismatch=pr.first_mismatch,
                max_seq_len=1 + draft.step_count * V.FRAME_LEN,
                decoded_final_ax=pr.decoded_final_ax)
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
    # DIRECT-CAM (C4_DIRECT_CAM_BATCHED): the GLOBAL memory/stack/LEV/code heads
    # direct-gather the draft-resolved value per query row (no O(S) global score).
    # DIRECT-LOCAL-CAM (C4_DIRECT_LOCAL_CAM): the block-0 register-ingest LOCAL heads
    # direct-gather their resolved frame byte per query row (no windowed score) — the
    # last local-attention floor.  Both DEFAULT OFF -> byte-identical to the softmax /
    # banded path (golden 069cc32f unchanged).  Kept alive on locals so the installed
    # forwards' resolved tables persist for the whole verify.
    # ZERO-ATTENTION-COMPUTE (C4_DEAD_BLOCK_FUSION): the audit (#856) found the
    # residual attention ARITHMETIC in the composed forward is NOT on the 4 live
    # blocks (0=ingest, 2=code-select, 7=mem-cam, 11=stack-pop-cam — all fully
    # resolved by direct-local / direct-CAM below, ZERO score/softmax) but on the
    # ~238 DEAD-attention blocks (pc-fetch, opcode-decode, alu-*, the divmod span,
    # …).  ``install_local_attention`` WINDOWS those blocks' zero-value local heads
    # and routes them through the banded/masked softmax — but never SKIPS them, so
    # each dead block still pays a full O(S·W) banded score+softmax+ctx over heads
    # whose output is provably 0.  MEASURED: 33 GFLOP / 1666 softmax1 calls on a 402-
    # step malloc verify — 100% of it dead-block waste.  ``install_dead_block_fusion``
    # bypasses the ENTIRE attention sublayer of a 0-live-head block (output = x, no
    # Q/K/V/W_o linear, no softmax, no KV write — proven L-inf=0), and
    # ``install_live_head_attention`` scores ONLY the live head-slots on the rest.
    # Composed with direct-CAM/direct-local (installed AFTER, on the 4 live blocks
    # which are never fused since they carry live heads) this drives the composed
    # step's attention-compute to EXACTLY ZERO — the step becomes FFN + O(1) gathers.
    # Byte-exact (a dead block IS the identity on the residual); default OFF ->
    # golden 069cc32f unchanged.
    if _dead_block_fusion_enabled():
        from .live_head_attention import (install_live_head_attention,
                                          install_dead_block_fusion)
        install_live_head_attention(model, verbose=False)
        install_dead_block_fusion(model, verbose=False)
    _dcam_tbl = _dlocal_tbl = None
    from .direct_cam_batched import (direct_cam_batched_enabled,
                                     install_direct_cam_batched)
    if direct_cam_batched_enabled():
        _dcam_tbl = install_direct_cam_batched(model, L, draft, code, verbose=False)
    from .direct_local_cam import (direct_local_cam_enabled,
                                    install_direct_local_cam)
    if direct_local_cam_enabled():
        _dlocal_tbl = install_direct_local_cam(model, L, draft, verbose=False)
    # FUSED-DELTA SPARSE FFN (C4_FUSED_DELTA_FFN): with attention driven to ~ZERO
    # by dead-block-fusion + direct-CAM/local, the composed step is FFN-bound (#869:
    # the dense SwiGLU GEMM ``ampere_sgemm`` over the 99.9%-zero weights is ~67% of
    # the 2.27 ms step, multiplying all the zeros — 0.167 GFLOP where the live nnz
    # are 5.6K FLOP).  ``install_fused_delta_ffn`` swaps every non-routed block's
    # dense SwiGLU for the #808/#841 fused-delta sparse-COO kernel (kernel-1 fuses
    # up+gate+silu; kernel-2 adds W_down@hidden ONLY to the residual rows W_down
    # writes, in place — touching only the ~1.3-nnz-per-unit weights).  It touches
    # ONLY ``block.ffn`` (leaves the 4 live blocks' attention forwards installed
    # above untouched), so it COMPOSES with dead-block-fusion + direct-CAM/local +
    # bounded-KV.  Byte-exact at the nibble-snap margin (same nonzeros, fp-accum-
    # order residue only — #808/#841 proved L-inf=0 at doom scale standalone).
    # DEFAULT OFF -> the dense/COO golden path (069cc32f unchanged).
    #
    # PRECEDENCE vs the FUSED MEGABLOCK (C4_FUSED_MEGABLOCK).  The megablock is the
    # STRONGER on-chip drop-in for the SAME dead-FFN [cut, N) region: it builds its own
    # ``_MegaFFN`` from each block's ORIGINAL ``W_up/W_gate/W_down`` and runs the whole
    # chain in-place, L2-resident.  ``install_fused_delta_ffn`` REPLACES ``block.ffn``
    # with a ``FusedUpGateSiluDeltaFFN`` (no ``W_up`` attribute), which the megablock
    # then can't read.  So when the megablock WILL fire (frozen-skip + dead-block-fusion
    # + CUDA + flag on) it OWNS the dead-FFN region and we SKIP fused-delta entirely (it
    # would only accelerate the same blocks the megablock already fuses more tightly, and
    # would break the megablock's original-weight read).  Both are byte-exact dead-FFN
    # accelerators; the megablock subsumes fused-delta on the query-row path.
    try:
        from .fused_megablock import fused_megablock_enabled as _fme
    except ImportError:
        _fme = lambda: False
    _mega_will_fire = (_fme() and _frozen_skip_enabled()
                       and device.startswith("cuda") and _dead_block_fusion_enabled())
    from .fused_sparse_ffn import (fused_delta_ffn_enabled,
                                   install_fused_delta_ffn)
    if fused_delta_ffn_enabled() and not _mega_will_fire:
        install_fused_delta_ffn(model, device=torch.device(device), verbose=False)
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
        # FAITHFUL-ATTN-EVICT implies the exact-evict liveness schedule: the whole
        # point is to run the GENUINE softmax over the EVICTED / bounded cache, so the
        # eviction policy is the schedule drop.  A caller may still pass exact_evict
        # explicitly to override.
        if faithful_attn_evict_enabled():
            exact_evict = True
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
    # BATCHED BLOCK-SKIP (C4_BATCHED_BLOCK_SKIP): per-span union block-MoE.  For each
    # forward span, run only the UNION of that span's opcodes' live blocks (in
    # model.blocks application coords) and skip the rest as identity passthrough.
    # Byte-exact iff the union is a superset of every step's true decode-live set.
    bbs_plan = None
    bbs_stats = {"skip_blocks_run": 0, "skip_blocks_full": 0, "spans": 0}
    from .batched_block_skip import batched_block_skip_enabled, BatchedBlockSkipPlan
    if batched_block_skip_enabled():
        bbs_plan = BatchedBlockSkipPlan(model, L)
    # FROZEN-ROW SKIP (C4_FROZEN_ROW_SKIP): after the last live-LOCAL-head block, run
    # the remaining blocks on QUERY rows only (~30x less FFN).  Byte-exact with
    # direct-CAM (blocks past the cut don't read frozen-row KV).  Computed ONCE.
    frozen_skip = _frozen_skip_enabled()
    frozen_cut = _frozen_skip_cut(model) if frozen_skip else 0
    # MEGAKERNEL (C4_GRAPH_MEGAKERNEL): CUDA-graph the post-cut dead-FFN segments of
    # the frozen-skip query-row block loop (one graph launch per contiguous dead
    # segment, live CAM blocks eager).  Requires dead-block-fusion (dead blocks are
    # attention-identity) + frozen-skip (the region runs over K query rows) + CUDA.
    # Byte-exact; default OFF -> the eager per-block loop.
    _megastep = None
    # MEGAKERNEL / FUSED MEGABLOCK install.  The two megakernel modules are
    # lazy-imported (guarded) so a missing session artifact never breaks the default
    # (flag-OFF) verify; both are now committed but the guard keeps the default path
    # robust.  Byte-identical: both megakernels are default OFF.
    #
    # FUSED MEGABLOCK (C4_FUSED_MEGABLOCK): the on-chip in-place-delta dead-FFN
    # megakernel — a stronger drop-in for the megastep graph (no per-block full-D
    # residual copy, no hidden HBM buffer; the [D,K] residual stays L2-resident across
    # the whole dead-FFN chain, all launches collapsed into one CUDA-graph replay).
    # Takes PRECEDENCE over C4_GRAPH_MEGAKERNEL.  Byte-exact (nibble-snap margin).
    try:
        from .fused_megablock import (fused_megablock_enabled,
                                      install_fused_megablock)
        _fused_mega_on = fused_megablock_enabled()
    except ImportError:
        install_fused_megablock = None
        _fused_mega_on = False
    try:
        from .megastep_graph import (megastep_graph_enabled,
                                     install_megastep_graph)
        _mega_on = megastep_graph_enabled()
    except ImportError:
        install_megastep_graph = None
        _mega_on = False
    if (frozen_skip and is_cuda and _dead_block_fusion_enabled()
            and _fused_mega_on and install_fused_megablock is not None):
        # CARRY SET.  Default = the FULL [cut, N) region (byte-identical superset: every
        # dead block is fused, so the mega-chain reproduces the eager loop exactly, incl.
        # the recurrent-divmod-tied blocks, which are NOT identity on the residual and so
        # must NOT be dropped).  ``C4_MEGABLOCK_DOOM_LEAN=1`` opts into the DIV-free lean
        # carry (drops the 179-block divmod span) — VALID ONLY on a build where the divmod
        # blocks are provably identity on a DIV-free step (they are NOT on the recurrent-
        # divmod build: measured Linf 116 >> the nibble decode margin -> garbage).  So the
        # lean carry is OFF by default; the full carry is byte-exact (Linf ~0.09).
        import os as _os
        _lean = _os.environ.get("C4_MEGABLOCK_DOOM_LEAN", "0") not in (
            "0", "", "false", "False")
        _megastep = install_fused_megablock(model, device, frozen_cut,
                                            L=(L if _lean else None), verbose=False)
    elif (frozen_skip and is_cuda and _dead_block_fusion_enabled()
            and _mega_on and install_megastep_graph is not None):
        _megastep = install_megastep_graph(model, device, frozen_cut, verbose=False)
    # BLOCK-0 FUSED FFN (C4_BLOCK0_FUSED_FFN, RUNG 3): fold block-0's DENSE SwiGLU FFN
    # into the fused-delta COO sparse kernel (block-0's FFN is Dff=33, 6 active down rows
    # -> the dense GEMM wastes ~99.98% of its FLOPs on zeros).  MUST install BEFORE the
    # block-0 graph captures so the graph bakes the sparse Triton kernels, not the dense
    # cuBLAS GEMM.  Byte-exact at the nibble margin; default OFF.
    _block0_fused_ffn = None
    try:
        from .block0_fused_ffn import (block0_fused_ffn_enabled,
                                       install_block0_fused_ffn)
        _b0ffn_on = block0_fused_ffn_enabled()
    except ImportError:
        install_block0_fused_ffn = None
        _b0ffn_on = False
    if (frozen_skip and is_cuda and _b0ffn_on
            and install_block0_fused_ffn is not None):
        _block0_fused_ffn = install_block0_fused_ffn(model, device, 0, verbose=False)
    # BLOCK-0 GRAPH (C4_GRAPH_BLOCK0): CUDA-graph block-0's S-chunked ingest attention +
    # FFN per-chunk body into ONE fixed-shape replay per chunk.  Block 0 is the SOLE block
    # over all S = K*30 span rows (the cut-span-chunk loop processes them in fixed-size
    # chunks, launching block-0's whole forward per chunk on the host — the ~640-707 us
    # host-dispatch wall).  Under direct-local-CAM block 0 is a per-row independent map, so
    # its per-chunk body (gather -> W_o GEMM -> dense SwiGLU FFN) is fixed-shape and captures
    # cleanly; one replay per chunk instead of ~13 launches.  Requires direct-local-CAM +
    # cut-span-chunking (fixed chunk size) + CUDA.  Byte-exact; default OFF.
    _block0_graph = None
    try:
        from .block0_graph import block0_graph_enabled, install_block0_graph
        _b0g_on = block0_graph_enabled()
    except ImportError:
        install_block0_graph = None
        _b0g_on = False
    _cutc0 = _cut_span_chunk()
    if (frozen_skip and is_cuda and _b0g_on and install_block0_graph is not None
            and direct_local_cam_enabled() and _cutc0 > 0
            and getattr(model.blocks[0].attn, "_direct_local_installed", False)):
        _block0_graph = install_block0_graph(model, device, _cutc0, frozen_cut,
                                             verbose=False)
    # WHOLE-STEP GRAPH (C4_WHOLE_STEP_GRAPH, RUNG 2): collapse the block-0 S-chunk loop's
    # per-chunk host dispatch (ingest gather + graph + bool(.any()) SYNC + hq0 scatter)
    # into ONE CUDA-graph replay (gather+W_o+FFN fused) + a precomputed SYNC-FREE scatter
    # per chunk.  Kills the ~640 per-chunk host syncs and the gather/scatter launch storm
    # — the dominant (81%) block-0 host-dispatch wall.  Requires direct-local-CAM (the
    # resolved-frame gather internals) + cut-span-chunk + CUDA.  Byte-exact; default OFF.
    # SUPERSEDES the plain block-0 graph (this one also folds the gather + kills the sync).
    _whole_step_graph = None
    try:
        from .whole_step_graph import (whole_step_graph_enabled,
                                       install_whole_step_graph)
        _wsg_on = whole_step_graph_enabled()
    except ImportError:
        install_whole_step_graph = None
        _wsg_on = False
    if (frozen_skip and is_cuda and _wsg_on and install_whole_step_graph is not None
            and direct_local_cam_enabled() and _cutc0 > 0 and frozen_cut == 1
            and getattr(model.blocks[0].attn, "_direct_local_installed", False)):
        _b0attn = model.blocks[0].attn
        _rf = getattr(_b0attn, "_direct_local_rf", None)
        _ing = getattr(_b0attn, "_direct_local_ing_heads", None)
        if _rf is not None and _ing is not None:
            _whole_step_graph = install_whole_step_graph(
                model, device, _cutc0, _rf, _ing, int(_rf.nib_lo.shape[1]),
                cut=frozen_cut, verbose=False)
    # LAUNCH-COLLAPSE (C4_OVERLAY_BATCHED): assemble the overlay's per-row scalar
    # writes on the host and push them in ONE index_put_ (kills the ~21k tiny
    # pageable HtoD dispatches the profiler pinned as the span wall).
    _overlay_batched = _overlay_batched_enabled()
    # EVICTION host-sync policy: the scheduled drop is now GPU-vectorized
    # (evict_all_blocks_scheduled batched-nonzero, ~1 sync/size-group), so the only
    # remaining per-round syncs are the OPTIONAL timing drains — gated OFF by default.
    _evict_timed = _evict_timed_enabled()
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

    # STREAM-EMBED signal: set True by _forward_span when it returns the compact
    # query-ORDERED [1,K,D] hidden (decode indexes by ``s - step`` instead of the
    # span-local ``win_starts[s] - span_start``).  Reset each span.
    _qordered = [False]

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
        q_local = [draft.win_starts[s] - span_start for s in range(step, end)]
        # STREAM-EMBED (big-K): frozen_skip will build each block-0 S-chunk's embed +
        # overlay on demand, so the full [1, S, D] span embed is NEVER built here.  Only
        # for the frozen_skip path (block-MoE / block-skip paths still need the whole span).
        _cutc = _cut_span_chunk()
        _use_stream = (fast and frozen_skip and _stream_embed_enabled() and _cutc > 0
                       and S > _cutc and bbs_plan is None and moe_span is None)
        _qordered[0] = _use_stream
        with torch.no_grad():
            if _use_stream:
                x = model.embed[win_toks[:, :1]]              # [1,1,D] throwaway placeholder
                _overlay = None                               # overlay done per-chunk
            else:
                # ``model.embed[win_toks]`` (advanced indexing) ALREADY returns a fresh
                # copy (never a view of the embed table), so the ``.clone()`` is a
                # redundant second [1, S, D] allocation — at big K (S = K*30) it DOUBLES the
                # span-embed peak (measured: the clone transient is 5.6GB at K=16384) and is
                # the top VRAM wall once block 0 is Sq-chunked.  Drop it on the big-K
                # cut-span-chunk path (byte-identical — the gather copy is already private);
                # keep the explicit clone on the default path so the golden memory behaviour
                # is unchanged.
                x = model.embed[win_toks]
                if _cutc <= 0:
                    x = x.clone()
            # overlay the span: program-in-data + frame roles + store KV entries.
            # The span's last row is NOT necessarily a query row, so overlay with
            # is_last_row_query=False, then explicitly re-tag EACH step's query row
            # with all-ROLE one-hots (the driver's per-step query tag).
            if _use_stream:
                pass                                          # per-chunk overlay in frozen_skip
            elif fast:
                # O(rows + code): broadcast the code-in-data, per-row roles, then
                # tag exactly this block's query rows (byte-identical residual).
                _overlay = (apply_overlay_window_batched if _overlay_batched
                            else apply_overlay_window_fast)
                _overlay(x, span_start, L, store_log, code_vec,
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
            # BATCHED BLOCK-SKIP: run only the union of this span's ops' live blocks.
            if bbs_plan is not None:
                span_ops = [draft.frames[s]["op"] for s in range(step, end)]
                run_mask = bbs_plan.span_live_mask(span_ops).tolist()
                hidden, new_kv = _forward_hidden_cached_mask(
                    model, x, past, q_positions, run_mask)
                blocks_run = int(sum(run_mask))
                bbs_stats["skip_blocks_run"] += blocks_run
                bbs_stats["skip_blocks_full"] += n_blocks
                bbs_stats["spans"] += 1
                return hidden, new_kv, span_start, S, blocks_run
            # BLOCK-MoE: skip the divmod span iff NO step in [step, end) is DIV/MOD.
            span_has_divmod = (moe_span is not None
                               and any(divmod_step[s] for s in range(step, end)))
            skip_range = (moe_span if (moe_span is not None and not span_has_divmod)
                          else None)
            if skip_range is not None:
                hidden, new_kv = _forward_hidden_cached_skip(
                    model, x, past, q_positions, skip_range)
                blocks_run = n_blocks - (skip_range[1] - skip_range[0])
            elif frozen_skip:
                # blocks [0, cut) over all S rows; blocks [cut, N) over the K query
                # rows only.  ~30x less FFN.  Byte-exact under direct-CAM.
                # STREAM-EMBED (C4_STREAM_EMBED, big-K only): pass the chunk-builder so
                # the block-0 S-chunk loop builds each chunk's embed+overlay ON DEMAND —
                # the full [1, S, D] span embed (the last O(K*30) tensor) never exists.
                stream_ctx = None
                if _use_stream:
                    stream_ctx = dict(
                        win_toks=win_toks, span_start=span_start, code_vec=code_vec,
                        q_local=q_local, code=code, code_off=draft.code_off,
                        store_log=store_log, overlay_batched=_overlay_batched,
                        embed=model.embed, L=L)
                hidden, new_kv = _forward_hidden_cached_frozen_skip(
                    model, x, past, q_positions, q_local, frozen_cut,
                    megastep=_megastep, stream_ctx=stream_ctx,
                    block0_graph=_block0_graph, whole_step_graph=_whole_step_graph)
                # cost accounting: cut blocks over S rows + (N-cut) over K query rows.
                blocks_run = n_blocks
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
                    _overlay = (apply_overlay_window_batched if _overlay_batched
                                else apply_overlay_window_fast)
                    _overlay(x, pos, L, store_log, code_vec,
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

        # ============================================================================
        # DIRECT-CAM ADDRESS VERIFY (C4_DIRECT_CAM_VERIFY_ADDR): the direct-CAM forward
        # decoded the model's OWN queried address for every CAM read in this span and
        # reported any mismatch vs the draft's resolved address into ``_dcam_tbl.addr_sink``.
        # An address the MODEL would NOT have selected is a genuine divergence — turn it
        # into the SAME terminal FAIL the register/token compare uses (first-divergence
        # stop).  This closes the "address draft-TRUSTED" gap (scenario E of the audit):
        # a self-consistent wrong-address draft is now CAUGHT, not rubber-stamped.  O(1)
        # per read, so the large-KV speedup is untouched.
        _addr_sink = getattr(_dcam_tbl, "addr_sink", None) if _dcam_tbl is not None else None
        if _addr_sink is not None and _addr_sink.hit is not None:
            _h = _addr_sink.hit
            _qpos = int(_h["query_pos"])
            # map the divergent read's query position back to its step (its step's own
            # query row is at draft.win_starts[s]).  Resolve it robustly by inverse
            # lookup over ALL steps (a span's forward can
            # touch a read's query row before that step is register-verified, so the
            # divergent step may be >= the current ``end``; search the whole win_starts).
            _ws_to_step = getattr(draft, "_ws_to_step_cache", None)
            if _ws_to_step is None:
                _ws_to_step = {ws: s for s, ws in enumerate(draft.win_starts)}
                try:
                    draft._ws_to_step_cache = _ws_to_step
                except Exception:
                    pass
            _s_bad = _ws_to_step.get(_qpos)
            if _s_bad is None:
                _s_bad = min(step, n_steps - 1)   # defensive: never mis-report OK
            accepted = min(accepted, _s_bad)      # accept only up to the divergence
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
                stats["cam_addr_divergence"] = dict(_h)
            _fr = draft.frames[_s_bad]
            return VerifyResult(
                accepted_steps=accepted, total_steps=n_steps,
                all_matched=False, forwards=forwards,
                first_mismatch={
                    "step": _s_bad, "query_pos": _qpos, "kind": "cam_addr",
                    "cam_head": _h["head"], "cam_kind": _h["kind"],
                    "got": {"addr": _h["model_addr"]},
                    "want": {"addr": _h["draft_addr"]},
                    "detail": (f"direct-CAM read at step {_s_bad}: model query "
                               f"addr={_h['model_addr']} != draft resolved "
                               f"addr={_h['draft_addr']} (head {_h['head']} "
                               f"kind={_h['kind']})")},
                max_seq_len=max_seq, max_cache_size=cache_now,
                total_evicted=evicted_now, decoded_final_ax=None,
                peak_vram_gb=vram_gb, evict_rounds=evict_rounds,
                effective_block_steps=eff_min_k)

        # BATCHED DECODE (C4_BATCHED_DECODE, default OFF): decode ALL query rows of
        # this span in ONE device-side gather + argmax + a SINGLE host copy, instead
        # of the per-step ``float(state[dim])`` scalar path (which host<->device syncs
        # ~11 times PER STEP -> ~330k GPU stalls over a program).  Bit-identical: the
        # batched ``_snap_lane_batch`` / ``_decode_reg_batch`` evaluate the SAME integer
        # ``argmax_v (2vx - v^2)`` requant per row as the scalar ``_snap_lane`` /
        # ``_decode_reg_from_nibbles`` (probe: byte-exact, 15x faster on the decode).
        batched_decode = (_batched_decode_enabled() if _BATCHED_DECODE is None
                          else _BATCHED_DECODE)
        # ============================================================================
        # GPU-VERIFY (C4_GPU_VERIFY, default OFF): batch-decode ALL K query rows AND
        # compare to the draft targets ENTIRELY on-GPU, reducing to the accepted-prefix
        # length with ONE host sync per forward.  Replaces the O(K) host-side Python
        # decode+compare loop (the ~0.2-0.32 ms/step wall over the 0.034 ms/step GPU
        # forward).  Bit-identical to the scalar loop (same requant-argmax, same
        # pc/ax&mask/sp/bp compare, is_file no-op, is_halt AX-only).
        # ============================================================================
        if _gpu_verify_enabled():
            from .nibble_pure_forward_gpu import _decode_reg_batch, _snap_lane_batch
            tgt = _build_draft_targets(draft, device, mask)
            K_span = end - step
            if _qordered[0]:
                wi_idx = torch.arange(K_span, device=hidden.device, dtype=torch.long)
            else:
                wi_idx = torch.tensor(
                    [draft.win_starts[s] - span_start for s in range(step, end)],
                    device=hidden.device, dtype=torch.long)
            qs_all = hidden[0].index_select(0, wi_idx)      # [K, D] device-resident
            # DECODE-CHUNK (big-K): the per-row [rows, vocab] requant tensor is O(K)
            # VRAM; decode in qrow-sized chunks (byte-exact per row) into device buffers.
            _dchunk = _qrow_chunk(_cut_span_chunk() or K_span)
            Kd = qs_all.shape[0]
            _stepc = _dchunk if (_dchunk > 0 and Kd > _dchunk) else Kd
            got_pc_t = torch.empty(Kd, dtype=torch.long, device=hidden.device)
            got_sp_t = torch.empty(Kd, dtype=torch.long, device=hidden.device)
            got_bp_t = torch.empty(Kd, dtype=torch.long, device=hidden.device)
            got_ax_t = torch.empty(Kd, dtype=torch.long, device=hidden.device)
            _lo = 0
            while _lo < Kd:
                _hi = min(_lo + _stepc, Kd)
                qs = qs_all[_lo:_hi]
                got_pc_t[_lo:_hi] = _snap_lane_batch(qs[:, L.PC_VAL])
                got_sp_t[_lo:_hi] = _snap_lane_batch(qs[:, L.SP_VAL])
                got_bp_t[_lo:_hi] = _snap_lane_batch(qs[:, L.BP_VAL])
                got_ax_t[_lo:_hi] = _decode_reg_batch(qs, L.AX)
                _lo = _hi
            got_ax_m = got_ax_t & mask
            w_pc = tgt["want_pc"][step:end]
            w_ax = tgt["want_ax"][step:end]
            w_sp = tgt["want_sp"][step:end]
            w_bp = tgt["want_bp"][step:end]
            f_file = tgt["is_file"][step:end]
            f_halt = tgt["is_halt"][step:end]
            # normal-row mismatch: any of pc/ax/sp/bp differs; halt-row: AX only;
            # file-row: never bad (driver overrides the model's registers).
            bad_normal = ((got_pc_t != w_pc) | (got_ax_m != w_ax)
                          | (got_sp_t != w_sp) | (got_bp_t != w_bp))
            bad = torch.where(f_halt, got_ax_m != w_ax, bad_normal)
            bad = bad & (~f_file)                            # file rows are accepted
            # first divergence: argmax over the bad mask (0 if none) — but distinguish
            # "no bad" from "bad at index 0" via any().  ONE host sync for both.
            any_bad_t = bad.any()
            first_bad_t = torch.argmax(bad.to(torch.uint8))  # 0 when all-False
            # accepted count within this span = first_bad if any bad else K.
            n_ok_span = torch.where(any_bad_t, first_bad_t,
                                    torch.tensor(K_span, device=bad.device))
            # PRTF visible bytes for the ACCEPTED prefix of this span (vectorized): the
            # model's decoded AX byte-0 at each accepted prtf step (byte-identical to the
            # scalar path's collect_out.append).  Gather on GPU, one small copy.
            prtf_bytes = None
            if collect_out is not None and prtf_set:
                _prtf_local = [s - step for s in range(step, end) if s in prtf_set]
                if _prtf_local:
                    _pl = torch.tensor(_prtf_local, device=bad.device, dtype=torch.long)
                    prtf_bytes = (_pl, (got_ax_t & 0xFF).index_select(0, _pl))
            # last-step AX: if this span covers the final step AND it is accepted.
            last_ax_dev = None
            if step <= n_steps - 1 < end:
                last_ax_dev = got_ax_m[n_steps - 1 - step]
            # ---- the SINGLE host sync per forward -------------------------------
            n_ok = int(n_ok_span.item())
            accepted += n_ok
            if collect_out is not None and prtf_bytes is not None:
                _pl, _pb = prtf_bytes
                _pl_h = _pl.tolist()
                _pb_h = _pb.tolist()
                for _li, _lv in zip(_pl_h, _pb_h):
                    if _li < n_ok:
                        collect_out.append(int(_lv) & 0xFF)
            if n_ok < K_span:
                # MISMATCH inside this span: report the first divergence (byte-exact to
                # the scalar path's first-mismatch abort).  Decode the exact got/want
                # for the reporting dict (one small gather + sync — only on a real fail).
                s_bad = step + n_ok
                gp = int(got_pc_t[n_ok].item()); ga = int(got_ax_m[n_ok].item())
                gs = int(got_sp_t[n_ok].item()); gb = int(got_bp_t[n_ok].item())
                import os as _osd
                if _osd.environ.get("C4_WALL6_DIAG", "0") == "1" and stats is not None:
                    try:
                        _st = hidden[0, draft.win_starts[s_bad] - span_start]
                        stats["diag_stack0_model"] = _decode_reg_from_nibbles(
                            _st, L, L.STACK0)
                        # raw lane / nibble dump for the BP-restore high-byte probe
                        stats["diag_bp_val_lane"] = float(_st[L.BP_VAL].item())
                        stats["diag_stk_val_lane"] = float(_st[L.STK_VAL].item())
                        stats["diag_stack0_nibs"] = [
                            round(float(_st[L.STACK0 + j].item()), 4)
                            for j in range(16)]
                        stats["diag_lev_ret_val"] = float(_st[L.LEV_RET_VAL].item())
                        stats["diag_bp_nibs"] = [
                            round(float(_st[L.BP + j].item()), 4)
                            for j in range(16)]
                    except Exception as _e:
                        stats["diag_stack0_model"] = None
                        stats["diag_exc"] = repr(_e)
                    stats["diag_stk_draft"] = draft.frames[s_bad].get("stk")
                    stats["diag_op"] = draft.frames[s_bad].get("op")
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
                fr = draft.frames[s_bad]
                return VerifyResult(
                    accepted_steps=accepted, total_steps=n_steps,
                    all_matched=False, forwards=forwards,
                    first_mismatch={
                        "step": s_bad, "query_pos": draft.win_starts[s_bad],
                        "got": {"pc": gp, "ax": ga, "sp": gs, "bp": gb},
                        "want": {"pc": fr["pc"], "ax": fr["ax"] & mask,
                                 "sp": fr["sp"] & 0xFFFFFFFF,
                                 "bp": fr["bp"] & 0xFFFFFFFF}},
                    max_seq_len=max_seq, max_cache_size=cache_now,
                    total_evicted=evicted_now, decoded_final_ax=None,
                    peak_vram_gb=vram_gb, evict_rounds=evict_rounds,
                    effective_block_steps=eff_min_k)
            if last_ax_dev is not None:
                last_got_ax = int(last_ax_dev.item())
            # the whole span is verified on-GPU; skip the scalar per-step loop below and
            # fall through to the SHARED post-span tail (sync + accounting + eviction).
            _gpu_verified = True
        else:
            _gpu_verified = False
        pre_pc = pre_sp = pre_bp = pre_ax = None
        if batched_decode and not _gpu_verified:
            from .nibble_pure_forward_gpu import _decode_reg_batch, _snap_lane_batch
            # STREAM mode: hidden is the compact query-ORDERED [1,K,D] (row j == step
            # step+j), so the gather is the identity 0..K-1; otherwise index by the
            # span-local query position.  Both select the SAME K query-row states.
            if _qordered[0]:
                wi_idx = torch.arange(end - step, device=hidden.device, dtype=torch.long)
            else:
                wi_idx = torch.tensor(
                    [draft.win_starts[s] - span_start for s in range(step, end)],
                    device=hidden.device, dtype=torch.long)
            qs_all = hidden[0].index_select(0, wi_idx)  # [K, D] device-resident
            # DECODE-CHUNK (big-K): ``_snap_lane_batch`` / ``_decode_reg_batch`` build a
            # per-row ``[rows, vocab]`` fp64 requant-logit tensor (vocab~256) — O(K) VRAM
            # that OOMs at big K (58GB at K=131072).  It is per-ROW independent, so chunk
            # the K query rows through the decode too (default == the qrow chunk).  Byte-
            # exact: each row's argmax-requant is unchanged by the chunk boundary.
            _dchunk = _qrow_chunk(_cut_span_chunk() or (end - step))
            Kd = qs_all.shape[0]
            pre_pc, pre_sp, pre_bp, pre_ax = [], [], [], []
            _lo = 0
            _stepc = _dchunk if (_dchunk > 0 and Kd > _dchunk) else Kd
            while _lo < Kd:
                _hi = min(_lo + _stepc, Kd)
                qs = qs_all[_lo:_hi]
                pc_b = _snap_lane_batch(qs[:, L.PC_VAL])
                sp_b = _snap_lane_batch(qs[:, L.SP_VAL])
                bp_b = _snap_lane_batch(qs[:, L.BP_VAL])
                ax_b = _decode_reg_batch(qs, L.AX)
                dec = torch.stack([pc_b, sp_b, bp_b, ax_b], dim=1).cpu().tolist()
                pre_pc.extend(d[0] for d in dec)
                pre_sp.extend(d[1] for d in dec)
                pre_bp.extend(d[2] for d in dec)
                pre_ax.extend(d[3] for d in dec)
                _lo = _hi

        # decode + verify each step-query row of the block against the draft.
        # (SKIPPED when the GPU-verify path above already verified the whole span.)
        for s in (range(step, end) if not _gpu_verified else ()):
            if batched_decode:
                _i = s - step
                got_pc, got_sp, got_bp, got_ax = (
                    pre_pc[_i], pre_sp[_i], pre_bp[_i], pre_ax[_i])
            else:
                wi = (s - step) if _qordered[0] else (draft.win_starts[s] - span_start)
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
                        _st = hidden[0, draft.win_starts[s] - span_start]
                        stk = _decode_reg_from_nibbles(_st, L, L.STACK0)
                        stats["diag_bp_val_lane"] = float(_st[L.BP_VAL].item())
                        stats["diag_stk_val_lane"] = float(_st[L.STK_VAL].item())
                        stats["diag_lev_ret_val"] = float(_st[L.LEV_RET_VAL].item())
                        stats["diag_stack0_nibs"] = [
                            round(float(_st[L.STACK0 + j].item()), 4) for j in range(16)]
                        stats["diag_bp_nibs"] = [
                            round(float(_st[L.BP + j].item()), 4) for j in range(16)]
                    except Exception as _e:
                        stk = None
                        stats["diag_exc"] = repr(_e)
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
            # per-block host-synced eviction).  OPTIONALLY timed with a drain around
            # the prune (C4_EVICT_TIMED; 2 syncs/round) — OFF by default so eviction
            # adds ~ZERO host syncs per forward (the scheduled drop is GPU-vectorized;
            # the enqueued kernels order on the stream before the next forward with no
            # explicit drain needed).  The forward/overlay wall is the remainder.
            if is_cuda and _evict_timed:
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
            if is_cuda and _evict_timed:
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
        if bbs_plan is not None:
            stats["bbs_blocks_run"] = bbs_stats["skip_blocks_run"]
            stats["bbs_blocks_full"] = bbs_stats["skip_blocks_full"]
            stats["bbs_spans"] = bbs_stats["spans"]
            stats["bbs_block_reduction"] = (
                bbs_stats["skip_blocks_full"] / max(bbs_stats["skip_blocks_run"], 1))
            stats["bbs_mean_blocks_per_span"] = (
                bbs_stats["skip_blocks_run"] / max(bbs_stats["spans"], 1))
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
