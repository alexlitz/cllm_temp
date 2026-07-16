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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import blogspec_vocab as V
from .nibble_pure_forward import N_ROLES, _snap_lane
from .nibble_pure_forward_complete import (
    PureForwardCompleteLayout, IMM_NIBS, _build_frame, _decode_reg_from_nibbles,
    _mem_top, ADJ,
)
import c4_min.nibble_pure_forward_complete as _PFC
from .nibble_pure_forward_cached import apply_overlay_window, BlockKVCacheBatched


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


# The immediate is baked as IMM_NIBS little-endian nibbles (a STATIC re-encoding of
# the literal, gathered at PC), so the model's IMM writes AX = imm & (16**IMM_NIBS-1).
# ``ref_interpret`` masks IMM/LEA to 8 bits, which is WRONG for the model (the model
# keeps the full 20-bit literal — proven: add_0 IMM 654 -> model AX 654, not 142).
# The DRAFT must reproduce the MODEL's per-step register file exactly (else the
# parallel verify correctly rejects a draft that disagrees with the model), so this
# transition mirrors the MODEL, not ``ref_interpret``.
_IMM_MASK = (1 << (4 * IMM_NIBS)) - 1       # 20-bit literal band (IMM_NIBS=5 nibbles)


def draft_pf_program(code: List[isa.Instr], max_steps: int = 300000,
                     mask: int = 0xFFFFFFFF) -> PFDraft:
    """Run the MODEL's ISA transition and materialise the per-step 30-token frame
    stream + store_log — the exact token stream the pure-forward DRIVER emits.

    This is the *logical VM* of BLOG_SPEC §Speculation: the plain deterministic
    C4 interpreter, so it is ~free relative to a model forward.  It reproduces the
    driver's own pre-step register bookkeeping (which address a PSH/JSR/ENT/SI
    writes) so the drafted store_log is byte-identical to what the driver records.

    The transition matches the MODEL (not ``ref_interpret``):
      * IMM keeps the full literal ``imm & _IMM_MASK`` (20-bit nibble band), NOT
        ``imm & 0xFF`` — the model's ``compile_imm_ax_nibbles`` writes all fetched
        nibbles;
      * LEA is folded to 8 bits (``compile_ax_byte_to_nibbles`` + ``_fold_ax_gated``);
      * ADD/SUB/MUL/DIV/MOD are 32-bit (``mask``); cmp/bitwise are 8-bit;
      * PSH/JSR/ENT/SI stores carry the full 32-bit AX (the KV memory value band).
    """
    SP_INIT = _PFC.SP_INIT
    mem: Dict[int, int] = {}
    sp = bp = SP_INIT
    ax = pc = 0
    # The driver's stream is [BOS] + init_frame, then one appended frame per step
    # EXCEPT the HALT step's frame (the driver breaks BEFORE the append).  The init
    # frame IS a real stream frame (frame_idx 0), so tokens must include it or every
    # later position shifts by 30.
    init_frame = _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    tokens: List[int] = [V.BOS] + init_frame
    frames: List[Dict[str, int]] = []
    store_log: Dict[int, Tuple[int, int]] = {}
    win_starts: List[int] = []
    stk = 0                                     # STACK0 mirror (MEM_VAL of a frame)
    stream_len = len(tokens)                    # == 31 (BOS + init frame)
    frame_idx = 0                               # init frame is frame 0
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
        # --- the MODEL's transition (32-bit; IMM keeps the full nibble literal) --
        if op == isa.IMM:
            ax = imm & _IMM_MASK              # full 20-bit literal (model, not &0xFF)
        elif op == isa.LEA:
            ax = (bp + 4 * imm) & 0xFF
        elif op == isa.PSH:
            sp -= 4; mem[sp] = ax & mask
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
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
            pop_val = mem.get(sp, 0); v = pop_val & 0xFF; sp += 4
            r = {isa.EQ: v == ax, isa.NE: v != ax, isa.LT: v < ax,
                 isa.GT: v > ax, isa.LE: v <= ax, isa.GE: v >= ax}[op]
            ax = 1 if r else 0
        elif op in (isa.LI, isa.LC):
            ax = mem.get(ax, 0) & 0xFF
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp, 0); pop_val = addr; sp += 4
            mem[addr] = ax & 0xFF
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.JSR:
            sp -= 4; mem[sp] = (i + 1) & 0xFF; pc = imm
        elif op == isa.ENT:
            mem[sp - 4] = bp & 0xFFFFFFFF; sp -= 4; bp = sp; sp -= 4 * imm
        elif op == ADJ:
            sp += 4 * imm
        elif op == isa.LEV:
            sp = bp; bp = mem.get(sp, 0); pc = mem.get(sp + 4, 0); sp += 8
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
                   final_ax_masked=final_ax, win_starts=win_starts)


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


def verify_blocks(model, L: PureForwardCompleteLayout, code: List[isa.Instr],
                  draft: PFDraft, *, block_steps: int = 64, device: str = "cpu",
                  evict: bool = True, cos_threshold: float = 0.99,
                  prune_interval: int = 120, zero_eps: float = 1e-9,
                  recency_eps: float = 1e-6, mask: int = 0xFFFFFFFF,
                  stats: Optional[dict] = None) -> VerifyResult:
    """Verify the whole drafted stream on the SPARSE model in BLOCKS.

    Processes ``block_steps`` VM steps per batched ``forward_hidden_cached``.  For
    each block we forward the CONTIGUOUS token span covering those steps against
    the growing per-block KV cache, decode the register state at each step-query
    row, and confirm it equals the draft's next-step registers.  Frozen frame rows
    (all non-query rows of the span) are committed to the cache and pruned every
    ``prune_interval`` tokens so VRAM stays flat over deep loops.  The first
    mismatch aborts and is reported (a genuine fail: model argmax != draft).

    Because softmax1 is causal and each step-query row sits at the END of its own
    frame, its block output is identical whether computed in this bulk forward or
    in the per-step 31-row window (every row it attends to — its frame + all
    earlier frozen frames — is present with the same K/V).  This is the
    speculative collapse: N one-step forwards -> a few batched forwards.
    """
    n_blocks = len(model.blocks)
    H = model.blocks[0].attn.n_heads
    HD = model.blocks[0].attn.head_dim
    caches = [BlockKVCacheBatched(H, HD, model.blocks[b].attn.alibi_slopes)
              for b in range(n_blocks)]
    store_log = draft.store_log
    n_steps = draft.step_count
    forwards = 0
    accepted = 0
    tokens_since_prune = 0
    max_seq = 1 + n_steps * V.FRAME_LEN
    max_cache = 0
    last_got_ax = None                      # the MODEL's decoded AX at the last step
    dev = torch.device(device)

    step = 0
    while step < n_steps:
        end = min(step + block_steps, n_steps)
        # The contiguous token span covering steps [step, end): from this block's
        # first step-query row to the last frame's end.  Step 0's span starts at 0
        # (it must include BOS + init frame, whose query row is win_starts[0]).
        span_start = 0 if step == 0 else draft.win_starts[step]
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
            apply_overlay_window(x, span_start, code, L, store_log,
                                 is_last_row_query=False)
            for s in range(step, end):
                wi = draft.win_starts[s] - span_start
                for role in range(N_ROLES):
                    x[0, wi, L.ROLE + role] = 1.0
            past = [caches[b].as_past_kv() for b in range(n_blocks)]
            hidden, new_kv = model.forward_hidden_cached(
                x, past_key_values=past, q_positions=q_positions, use_cache=True)
        forwards += 1

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
            if fr.get("is_halt"):
                bad = ((got_ax & mask) != want_ax)
            else:
                bad = (got_pc != want_pc or (got_ax & mask) != want_ax
                       or got_sp != want_sp or got_bp != want_bp)
            if bad:
                cache_now = max(max_cache, caches[0].size())
                evicted_now = sum(c.total_evicted for c in caches)
                if stats is not None:
                    stats["max_seq_len"] = max_seq
                    stats["max_cache_size"] = cache_now
                    stats["total_evicted"] = evicted_now
                    stats["forwards"] = forwards
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
                    total_evicted=evicted_now, decoded_final_ax=None)
            accepted += 1
            if s == n_steps - 1:
                last_got_ax = got_ax & mask     # the model's actual final AX

        # commit the FROZEN (non-query) rows of the span to the caches.  A step's
        # query row carries the all-ROLE query overlay (wrong K/V for a frozen
        # context token), so we commit exactly the non-query rows — whose overlay
        # IS the plain context overlay computed above.  The NEXT block's first
        # window re-reads from the last query row's position (span_start), which we
        # therefore leave un-cached until it is committed here as a frozen row of
        # the current span (it is a non-query row of THIS span).
        q_rows_in_span = {draft.win_starts[s] - span_start
                          for s in range(step, end)}
        keep = [p for p in range(S) if p not in q_rows_in_span]
        if keep:
            keep_idx = torch.tensor(keep, device=dev, dtype=torch.long)
            for b in range(n_blocks):
                K_all, V_all, pos_all = new_kv[b]
                K_span = K_all[:, :, -S:, :]
                V_span = V_all[:, :, -S:, :]
                pos_span = pos_all[-S:]
                caches[b].commit(K_span[:, :, keep_idx, :],
                                 V_span[:, :, keep_idx, :],
                                 pos_span[keep_idx])
        if device.startswith("cuda"):
            torch.cuda.synchronize(dev)

        tokens_since_prune += (end - step) * V.FRAME_LEN
        if evict and tokens_since_prune >= prune_interval:
            for b in range(n_blocks):
                caches[b].evict(cos_threshold, prune_interval, zero_eps, recency_eps)
            tokens_since_prune = 0
        max_cache = max(max_cache, caches[0].size())
        step = end

    # the answer is the MODEL's decoded AX at the last step (verify PROVED it equals
    # the draft's, so this is the token-by-token autoregressive final byte).
    decoded_final = last_got_ax
    if stats is not None:
        stats["max_seq_len"] = max_seq
        stats["max_cache_size"] = max(max_cache, caches[0].size())
        stats["total_evicted"] = sum(c.total_evicted for c in caches)
        stats["forwards"] = forwards
    return VerifyResult(
        accepted_steps=accepted, total_steps=n_steps,
        all_matched=(accepted == n_steps), forwards=forwards,
        max_seq_len=max_seq, max_cache_size=max(max_cache, caches[0].size()),
        total_evicted=sum(c.total_evicted for c in caches),
        decoded_final_ax=decoded_final)


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
                    mask: int = 0xFFFFFFFF) -> SpecResult:
    """Full speculative decode of ONE pure-forward program.

    1) draft the whole stream with the reference VM (zero forwards);
    2) verify it block-wise on the SPARSE model (a handful of forwards);
    3) PASS iff the model ACCEPTS the full drafted stream (every step-query row's
       decoded register state matches the draft) AND the decoded final AX ==
       expected.  A verify mismatch is a genuine FAIL (model argmax != draft),
       reported with the step/position.  A draft that never HALTs within
       ``max_steps`` is a TIMEOUT.
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
                       mask=mask, stats=stats)
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
    vr = verify_blocks(model, L, code, draft, block_steps=block_steps,
                       device=device, evict=evict, prune_interval=prune_interval,
                       mask=mask)
    identical = (drv_trace == spec_trace) and vr.all_matched
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
        "first_divergence": first_div,
        "driver_final_ax": drv_trace[-1] if drv_trace else None,
        "spec_final_ax": spec_trace[-1] if spec_trace else None,
    }
