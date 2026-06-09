"""Blog-spec-grounded isolation tests for L15 memory_lookup.

Codifies the BLOG_SPEC.md §408-461 ("Memory") contract as unit tests
that pin down which slot in L15 is actually load-bearing for SI/LI
roundtrip — without depending on full-model smoke runs (~60s each).

Spec summary (BLOG_SPEC.md §408-461):
  1. SI/SC stores: K-row at MEM section = +/- scale per address bit
     (binary encoding of 32-bit address). V-row = stored value bytes.
     ALiBi gives latest-write-wins.
  2. LI/LC loads: Q-row at AX marker = +/- scale per address bit of AX.
     softmax picks the matching past K-row. V at that position = stored
     value.
  3. softmax1 gives ZFOD — unmapped addresses return 0.
  4. 24 binary address bits x scale=10 -> match score = 24*100 / sqrt(HD)
     = ~300 for HD=64. Random ~ 0.

These tests build a minimal residual + run JUST L15's heads_0_3 setter
(plus the l15_ops overlay where applicable) directly. Each test runs in
<5 seconds, with no full smoke run and no program execution.

Test status at current main:
  test_zfod_via_softmax1_on_unmapped_address ............... PASS
  test_latest_write_wins_via_alibi .......................... PASS
  test_blog_spec_binary_address_match_dominates ............. FAIL (bug)
  test_l15_head_0_row_36_does_not_alias_stack0 .............. FAIL (bug)
  test_per_head_byte_index_correct .......................... FAIL (bug)
  test_si_si_li_e2e_isolated ................................ FAIL (bug)
"""
from __future__ import annotations

import math
import os
import sys

import torch

# Force CPU-only execution to avoid OOM with parallel agents.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.vm_step import (  # noqa: E402
    _SetDim,
    _set_layer15_memory_lookup_heads_0_3,
)

# d_model + scale constants matching the production bake.
D_MODEL = 512
S = 100.0
HD = 64  # head dim; L15 heads_0_3 setter uses rows 0..63 per head
NUM_HEADS = 4  # heads_0_3
BD = _SetDim


class _StubAttn:
    """Bare attention module exposing W_q/W_k/W_v/W_o tensors.

    Mirrors the shape used by ``_set_layer15_memory_lookup_heads_0_3``:
    weights are (n_heads * head_dim, d_model) — head h slot s lives at
    row ``h * HD + s``.  No biases, no alibi_slopes (tests apply ALiBi
    explicitly when needed).
    """

    def __init__(self, *, d_model: int = D_MODEL, num_heads: int = NUM_HEADS, head_dim: int = HD):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = d_model
        dim_out = num_heads * head_dim
        self.W_q = torch.zeros(dim_out, d_model)
        self.W_k = torch.zeros(dim_out, d_model)
        self.W_v = torch.zeros(dim_out, d_model)
        self.W_o = torch.zeros(d_model, dim_out)
        self.alibi_slopes = torch.zeros(num_heads)


def _baked_l15_attn() -> _StubAttn:
    """Return a fresh L15 attn module with the heads_0_3 setter applied."""
    attn = _StubAttn()
    _set_layer15_memory_lookup_heads_0_3(attn, S, BD, HD)
    return attn


def _addr_bits(addr: int, num_bits: int = 24) -> list[int]:
    """Return list of 0/1 bits for an address (LSB first).

    Layout matches L15's binary-address dims: 3 bytes x (lo nibble, hi
    nibble) x 4 bits = 24 bits, lo-byte-lo-nibble-bit-0 first.
    """
    bits = []
    for byte_i in range(3):
        b = (addr >> (8 * byte_i)) & 0xFF
        lo = b & 0xF
        hi = (b >> 4) & 0xF
        for nibble in (lo, hi):
            for bit in range(4):
                bits.append((nibble >> bit) & 1)
    return bits


def _write_addr_to_row(row: torch.Tensor, addr: int) -> None:
    """Write a 24-bit address into the row's nibble one-hot slots.

    The L15 binary-address dims read from ADDR_B{0,1,2}_{LO,HI} as
    one-hot 16-wide nibbles. Setting nibble = k populates k as one-hot
    so that the bit-extraction Q/K rows recover bit_val = +1/-1 per bit.
    """
    nibble_bases = [
        BD.ADDR_B0_LO, BD.ADDR_B0_HI,
        BD.ADDR_B1_LO, BD.ADDR_B1_HI,
        BD.ADDR_B2_LO, BD.ADDR_B2_HI,
    ]
    for byte_i in range(3):
        b = (addr >> (8 * byte_i)) & 0xFF
        lo = b & 0xF
        hi = (b >> 4) & 0xF
        row[nibble_bases[byte_i * 2] + lo] = 1.0
        row[nibble_bases[byte_i * 2 + 1] + hi] = 1.0


def _li_query_row() -> torch.Tensor:
    """AX-marker Q row at an LI step.

    Mirrors the production residual at the AX marker token of an LI/LC
    step: CONST=1, MARK_AX=1 (marker identity flag), OP_LI_RELAY=1
    (carried from the AX marker), H1[AX]=1 (L0 threshold head 1 fires
    within 4.5 from AX). The H1[AX] flag is critical — without it, the
    slot-36 overlay's `W_q[..., H1+ax_i] = +10*s` doesn't fire and the
    BYTE_INDEX_0 alias bug stays masked.
    """
    row = torch.zeros(D_MODEL)
    row[BD.CONST] = 1.0
    row[BD.MARK_AX] = 1.0
    row[BD.OP_LI_RELAY] = 1.0
    # L0 H1 fires within distance 4.5 from each marker — at the AX marker
    # itself, distance is 0 so H1[AX] is on.
    AX_I = 1
    row[BD.H1 + AX_I] = 1.0
    return row


def _li_query_row_with_addr(addr: int) -> torch.Tensor:
    row = _li_query_row()
    _write_addr_to_row(row, addr)
    return row


def _mem_val_byte0_key_row(addr: int, val_byte0: int = 0) -> torch.Tensor:
    """K row at MEM val byte 0 position of a SI/SC store.

    Contains: MARK_MEM-adjacent threshold flags (L2H0[MEM]=1, H1[MEM]=0),
    MEM_STORE=1, the 24-bit binary address, and the val byte 0 nibbles.
    """
    row = torch.zeros(D_MODEL)
    row[BD.CONST] = 1.0
    row[BD.IS_BYTE] = 1.0
    row[BD.MEM_STORE] = 1.0
    # Head 0 byte-selection: L2H0[MEM]=1, H1[MEM]=0 selects val byte 0.
    MEM_I = 4
    row[BD.L2H0 + MEM_I] = 1.0
    # H1[MEM] left at 0 to satisfy the differential threshold.
    _write_addr_to_row(row, addr)
    # Value: split into low/high nibble at CLEAN_EMBED for V projection.
    lo = val_byte0 & 0xF
    hi = (val_byte0 >> 4) & 0xF
    row[BD.CLEAN_EMBED_LO + lo] = 1.0
    row[BD.CLEAN_EMBED_HI + hi] = 1.0
    return row


def _stack0_byte0_key_row() -> torch.Tensor:
    """K row at a STACK0 byte 0 position (NOT a MEM store).

    BYTE_INDEX_0=1, MARK_STACK0=0 (byte position, not marker).
    No MEM_STORE flag. No address.

    Includes the L0/L1 threshold-head signatures that fire at a STACK0
    byte-0 position in production (within range of the STACK0 marker
    immediately preceding it). The probe (probe_l15_lookup_li_step.py)
    confirms dims like H1+10 (the H2 band) are active at K@81 (STACK0
    byte 0). Without those flags the slot-36 alias bug is masked by
    the K[CONST]=-2*s baseline in the overlay; with them, the bug
    manifests in isolation tests.
    """
    row = torch.zeros(D_MODEL)
    row[BD.CONST] = 1.0
    row[BD.IS_BYTE] = 1.0
    row[BD.BYTE_INDEX_0] = 1.0
    # L0/L1 threshold heads firing at STACK0 byte 0 (d=1 from STACK0 marker).
    # The slot-36 overlay reads from H1+10 specifically (an H2 alias slot).
    # In production, the STACK0 byte 0 position has L0 head H2 fire for
    # the STACK0 marker (within 5.5 distance threshold).
    STACK0_I = 10  # the 10th index past H1's 7 marker slots
    # Set H1+10 = dim 77 = inside H2 band (the dim slot-36 K reads).
    row[BD.H1 + 10] = 1.0
    return row


def _ax_byte_n_key_row(byte_idx: int) -> torch.Tensor:
    """K row at AX byte N position (not a MEM store; not a target)."""
    row = torch.zeros(D_MODEL)
    row[BD.CONST] = 1.0
    row[BD.IS_BYTE] = 1.0
    AX_I = 1
    row[BD.H1 + AX_I] = 1.0
    byte_idx_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][byte_idx]
    row[byte_idx_dim] = 1.0
    return row


def _per_head_qk_scores(
    attn: _StubAttn,
    q_row: torch.Tensor,
    k_rows: list[torch.Tensor],
    head: int,
) -> list[float]:
    """Return raw Q . K^T / sqrt(HD) for a single head across k_rows."""
    base = head * HD
    Wq = attn.W_q[base : base + HD]    # (HD, d_model)
    Wk = attn.W_k[base : base + HD]
    q_vec = q_row @ Wq.T  # (HD,)
    scores = []
    for k_row in k_rows:
        k_vec = k_row @ Wk.T  # (HD,)
        s = float((q_vec * k_vec).sum().item()) / math.sqrt(HD)
        scores.append(s)
    return scores


def _softmax1(scores: torch.Tensor) -> torch.Tensor:
    """softmax with an implicit +1 baseline in the denominator (softmax1).

    softmax1(x_i) = exp(x_i) / (1 + sum_j exp(x_j))

    Numerically stable via the standard max-shift trick, extended for the
    implicit +1: shift by max(0, max(scores)) so neither the numerator nor
    the denominator overflows.  Without this, scores > ~88 (float32) or
    > ~709 (float64) produce ``inf/inf = nan``.

    BLOG_SPEC §410: 'softmax1' gives ZFOD — unmapped addresses return 0.
    """
    # Finite-only max so -inf-masked positions don't contaminate the shift.
    finite = scores[torch.isfinite(scores)]
    m_scores = float(finite.max().item()) if finite.numel() > 0 else 0.0
    m = max(0.0, m_scores)
    e = torch.exp(scores - m)
    return e / (math.exp(-m) + e.sum())


# ----------------------------------------------------------------------
# Test 1: smoking-gun — binary address match SHOULD beat BYTE_INDEX_0.
# ----------------------------------------------------------------------

def test_blog_spec_binary_address_match_dominates():
    """BLOG_SPEC §410: binary address bits dominate other Q-K signals.

    The legitimate +300 match should win against any non-target K row
    that has BYTE_INDEX_0=1 (which is fundamentally NOT a memory store).
    Currently FAILS: l15_ops.py:1300+ writes slot 36 W_q[OP_LI_RELAY]=
    100000 x W_k[BYTE_INDEX_0]=20000 giving +1.9G score >> +300.
    """
    attn = _baked_l15_attn()
    # Apply the l15_ops overlay that introduces the slot-36 override
    # (so we test the production bake, not just the setter).
    _apply_l15_ops_head_0_overlay(attn)

    addr = 0x200
    q = _li_query_row_with_addr(addr)
    k_real_target = _mem_val_byte0_key_row(addr, val_byte0=42)
    k_byte_index_0 = _stack0_byte0_key_row()

    head = 0
    scores = _per_head_qk_scores(attn, q, [k_real_target, k_byte_index_0], head)
    score_target, score_byte_index_0 = scores

    # Real address-matching MEM store should outscore a row whose only
    # flag is BYTE_INDEX_0 (and which has no MEM_STORE flag, no address).
    assert score_target > score_byte_index_0, (
        f"BLOG_SPEC §410 binary-address match must dominate BYTE_INDEX_0 alias.\n"
        f"  score(real MEM store @ addr={hex(addr)}) = {score_target:.2f}\n"
        f"  score(STACK0 BYTE_INDEX_0 row)          = {score_byte_index_0:.2f}\n"
        f"Currently FAILS because slot-36 (l15_ops.py:1300) has"
        f" Q[OP_LI_RELAY]*K[BYTE_INDEX_0] = ~2e9 / sqrt(HD)."
    )


# ----------------------------------------------------------------------
# Test 2: softmax1 ZFOD for unmapped addresses.
# ----------------------------------------------------------------------

def test_zfod_via_softmax1_on_unmapped_address():
    """BLOG_SPEC §410: softmax1's '1' trivializes unmapped lookups.

    For a Q address that no past K row matches, softmax1's denominator
    is dominated by +1, so attention probabilities ~= 0 and the V/O
    output ~= 0.
    """
    attn = _baked_l15_attn()

    # Q at unmapped address.
    q = _li_query_row_with_addr(0xDEAD)
    # K positions: only non-store rows with no matching address.
    k_rows = [
        _stack0_byte0_key_row(),
        _ax_byte_n_key_row(0),
        _ax_byte_n_key_row(1),
    ]
    head = 0
    scores = _per_head_qk_scores(attn, q, k_rows, head)
    scores_t = torch.tensor(scores)
    probs = _softmax1(scores_t)
    total_attention = float(probs.sum().item())

    # softmax1 ZFOD: with no matching K, total attention << 1 means the
    # output is dominated by the implicit zero (sink).
    assert total_attention < 0.1, (
        f"softmax1 should give ZFOD on unmapped addresses.\n"
        f"  Q address: 0xDEAD (no matching K rows)\n"
        f"  scores: {scores}\n"
        f"  softmax1 sum: {total_attention:.4f}\n"
        f"Expected total_attention << 1 (sink dominates)."
    )


# ----------------------------------------------------------------------
# Test 3: ALiBi recency — newer stores beat older stores at same addr.
# ----------------------------------------------------------------------

def test_latest_write_wins_via_alibi():
    """BLOG_SPEC §410: positional bias via ALiBi upweights recent writes.

    Two SI stores to the same address; the more recent one should have
    higher attention than the older one.
    """
    attn = _baked_l15_attn()

    addr = 0x200
    q = _li_query_row_with_addr(addr)
    # Both K rows have matching addr + MEM_STORE.
    k_older = _mem_val_byte0_key_row(addr, val_byte0=10)
    k_newer = _mem_val_byte0_key_row(addr, val_byte0=42)

    head = 0
    raw_scores = _per_head_qk_scores(attn, q, [k_older, k_newer], head)
    # Apply ALiBi: older position gets larger negative bias relative to
    # the query position. Use a representative slope (head 0 default).
    # Place: older at pos 0, newer at pos 10, query at pos 20.
    alibi_slope = 0.25  # typical head-0 slope
    q_pos = 20
    older_pos = 0
    newer_pos = 10
    alibi_bias_older = -alibi_slope * (q_pos - older_pos)
    alibi_bias_newer = -alibi_slope * (q_pos - newer_pos)

    adj_older = raw_scores[0] + alibi_bias_older
    adj_newer = raw_scores[1] + alibi_bias_newer

    assert adj_newer > adj_older, (
        f"BLOG_SPEC §410: ALiBi should prefer recent writes.\n"
        f"  older (pos={older_pos}): score={raw_scores[0]:.2f} + alibi"
        f"={alibi_bias_older:.2f} = {adj_older:.2f}\n"
        f"  newer (pos={newer_pos}): score={raw_scores[1]:.2f} + alibi"
        f"={alibi_bias_newer:.2f} = {adj_newer:.2f}"
    )


# ----------------------------------------------------------------------
# Test 4: direct codification of the bug — slot 36 alias.
# ----------------------------------------------------------------------

def test_l15_head_0_row_36_does_not_alias_stack0():
    """Slot 36 must not score > +300 (the binary-match scale) for a K row
    whose only marker is BYTE_INDEX_0 (a non-store STACK0 byte position).

    Currently FAILS: l15_ops.py:1300+ writes
        W_q[base+36, OP_LI_RELAY] = 100000
        W_k[base+36, BYTE_INDEX_0] = 20000
    giving slot-36 score = 100000 * 20000 / sqrt(HD) = ~2.5e8.
    """
    attn = _baked_l15_attn()
    _apply_l15_ops_head_0_overlay(attn)

    # Q row identical to LI-step at AX marker.
    q = _li_query_row()
    # K row: STACK0 byte 0 position (BYTE_INDEX_0=1).
    k = _stack0_byte0_key_row()

    head = 0
    base = head * HD
    # Read slot 36 in isolation.
    Wq = attn.W_q[base + 36]
    Wk = attn.W_k[base + 36]
    q_slot = float(q @ Wq)
    k_slot = float(k @ Wk)
    slot_36_score = q_slot * k_slot / math.sqrt(HD)

    binary_addr_scale = 300.0  # 24 bits * 10 * 10 / sqrt(64) ~= 300
    assert abs(slot_36_score) <= binary_addr_scale, (
        f"L15 head-0 slot-36 must not dwarf the binary-address scale.\n"
        f"  Q[slot 36] for OP_LI_RELAY=1 row: {q_slot:.2f}\n"
        f"  K[slot 36] for BYTE_INDEX_0=1 row: {k_slot:.2f}\n"
        f"  slot 36 score contribution: {slot_36_score:.2f}\n"
        f"  binary-address scale: +-{binary_addr_scale}\n"
        f"FAILS due to l15_ops.py:1300+ writing W_q[36, OP_LI_RELAY]=100000"
        f" and W_k[36, BYTE_INDEX_0]=20000."
    )


# ----------------------------------------------------------------------
# Test 5: per-head byte selection — heads 1-3 read MEM val bytes 1-3.
# ----------------------------------------------------------------------

def test_per_head_byte_index_correct():
    """Heads 0-3 in L15 must attend to MEM val byte h, NOT to their own
    AX byte position.
    """
    attn = _baked_l15_attn()
    _apply_l15_ops_head_0_overlay(attn)

    addr = 0x200
    failures = []
    for h in range(4):
        # Q at head h's target AX-byte position.
        q = _li_query_row_with_addr(addr)
        if h == 0:
            # Head 0 query rides on MARK_AX (already set) + OP_LI_RELAY.
            pass
        else:
            # Heads 1-3 fire at AX byte positions (BYTE_INDEX_{h-1}).
            q[BD.MARK_AX] = 0.0
            q[BD.IS_BYTE] = 1.0
            q[BD.H1 + 1] = 1.0  # H1[AX_I]
            byte_q_flag = [None, BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][h]
            q[byte_q_flag] = 1.0

        # K rows: MEM val byte h, STACK0 byte h, AX byte h.
        # MEM val byte h key row matches BLOG_SPEC: MEM_STORE, addr bits,
        # AND the correct byte-selection threshold flag.
        MEM_I = 4
        mem_val_row = torch.zeros(D_MODEL)
        mem_val_row[BD.CONST] = 1.0
        mem_val_row[BD.IS_BYTE] = 1.0
        mem_val_row[BD.MEM_STORE] = 1.0
        _write_addr_to_row(mem_val_row, addr)
        if h == 0:
            mem_val_row[BD.L2H0 + MEM_I] = 1.0  # selects val byte 0
        elif h == 1:
            mem_val_row[BD.MEM_VAL_B1] = 1.0
        elif h == 2:
            mem_val_row[BD.MEM_VAL_B2] = 1.0
        elif h == 3:
            mem_val_row[BD.MEM_VAL_B3] = 1.0
        # CLEAN_EMBED nibble for the read value.
        mem_val_row[BD.CLEAN_EMBED_LO + 2] = 1.0
        mem_val_row[BD.CLEAN_EMBED_HI + 0] = 1.0

        stack0_row = _ax_byte_n_key_row(h if h < 3 else 0).clone()
        # STACK0_BYTE{h} dim is also exposed; set the appropriate flag.
        stack_dim = [BD.STACK0_BYTE0, BD.STACK0_BYTE1, BD.STACK0_BYTE2, BD.STACK0_BYTE3][h]
        stack0_row[stack_dim] = 1.0

        ax_byte_row = _ax_byte_n_key_row(h if h < 3 else 0)

        rows = [mem_val_row, stack0_row, ax_byte_row]
        scores = _per_head_qk_scores(attn, q, rows, h)
        winner = scores.index(max(scores))
        if winner != 0:
            failures.append(
                f"head {h}: scores={[f'{s:.1f}' for s in scores]}"
                f" winner_idx={winner} (expected 0=mem_val_row)"
            )

    assert not failures, (
        "Per-head byte selection violated BLOG_SPEC §410:\n"
        + "\n".join(failures)
    )


# ----------------------------------------------------------------------
# Test 6: end-to-end SI/LI roundtrip via residual synthesis.
# ----------------------------------------------------------------------

def test_si_si_li_e2e_isolated():
    """End-to-end roundtrip: program IMM 0x200; PSH; IMM 42; SI; IMM 0x200; LI; EXIT.

    Synthesize the residual at the LI step's AX marker position with
    one past SI to (addr=0x200, value=42). Run forward through L15
    attention. Assert OUTPUT_LO/HI carries the nibbles of 42.

    Currently FAILS due to slot-36 override picking the wrong K row.
    """
    attn = _baked_l15_attn()
    _apply_l15_ops_head_0_overlay(attn)

    addr = 0x200
    value = 42  # 0x2A = nibbles (0xA, 0x2)
    seq_len = 10

    # Residual: (seq_len, d_model)
    x = torch.zeros(seq_len, D_MODEL)
    # Most positions are uninteresting; we mark CONST=1 everywhere so
    # the L15 negative-bias mechanics work.
    for i in range(seq_len):
        x[i, BD.CONST] = 1.0

    # Past store: one SI to (0x200, 42) at position 5 (a MEM val byte 0 row).
    si_pos = 5
    x[si_pos] = _mem_val_byte0_key_row(addr, val_byte0=value)
    x[si_pos, BD.CONST] = 1.0  # re-assert CONST

    # LI step query at position 9: AX marker, OP_LI_RELAY=1, addr=0x200.
    q_pos = seq_len - 1
    x[q_pos] = _li_query_row_with_addr(addr)
    x[q_pos, BD.CONST] = 1.0

    # Forward through L15 attention (head 0 only — handles byte 0).
    head = 0
    base = head * HD
    Wq = attn.W_q[base : base + HD]
    Wk = attn.W_k[base : base + HD]
    Wv = attn.W_v[base : base + HD]

    Q = x @ Wq.T  # (seq, HD)
    K = x @ Wk.T
    V = x @ Wv.T

    q_vec = Q[q_pos]  # (HD,)
    scores = (K @ q_vec) / math.sqrt(HD)  # (seq,)
    # Causal mask: positions > q_pos forbidden.
    mask = torch.full((seq_len,), float("-inf"))
    mask[: q_pos + 1] = 0.0
    scores_masked = scores + mask
    # softmax1 via the stable helper (handles exp overflow for large scores).
    probs = _softmax1(scores_masked)
    head_out = probs @ V  # (HD,)
    # W_o projects head 0's slots back to OUTPUT_LO/HI.
    Wo_h = attn.W_o[:, base : base + HD]  # (d_model, HD)
    output = head_out @ Wo_h.T  # (d_model,)

    out_lo = output[BD.OUTPUT_LO : BD.OUTPUT_LO + 16]
    out_hi = output[BD.OUTPUT_HI : BD.OUTPUT_HI + 16]
    lo_idx = int(torch.argmax(out_lo).item())
    hi_idx = int(torch.argmax(out_hi).item())

    expected_lo = value & 0xF       # 0xA
    expected_hi = (value >> 4) & 0xF  # 0x2
    assert lo_idx == expected_lo and hi_idx == expected_hi, (
        f"E2E SI/LI roundtrip failed.\n"
        f"  Program: IMM 0x200; PSH; IMM 42; SI; IMM 0x200; LI; EXIT\n"
        f"  Expected OUTPUT_LO nibble = {expected_lo:#x} (got {lo_idx:#x})\n"
        f"  Expected OUTPUT_HI nibble = {expected_hi:#x} (got {hi_idx:#x})\n"
        f"  scores: {scores.tolist()}\n"
        f"  probs:  {probs.tolist()}\n"
        f"FAILS because slot-36 override + other overlays make the"
        f" wrong K row win attention."
    )


# ----------------------------------------------------------------------
# Helper: apply the l15_ops slot-36 overlay (production bake addition).
# ----------------------------------------------------------------------

def _apply_l15_ops_head_0_overlay(attn: _StubAttn) -> None:
    """Apply the specific slot-36 writes from l15_ops.py:1300+ for head 0.

    These are the "stack0_preserve_row" writes that override the
    baseline heads_0_3 setter and create the slot-36 BYTE_INDEX_0 alias
    bug. The test imports them as data (mirroring the production overlay)
    so that the bug-pinning tests fail at current main.
    """
    base = 0  # head 0
    row = base + 36
    # Mirrors l15_ops.py:1306 after the BYTE_INDEX_0-alias fix
    # (was 10000.0; lowered to keep slot 36 within the L15 binary-address
    # match scale).
    stack0_preserve_s = 1.0
    ax_i = 1
    attn.W_q[row, BD.CONST] = -1.0 * stack0_preserve_s
    attn.W_q[row, BD.MARK_STACK0] = 3.0 * stack0_preserve_s
    attn.W_q[row, BD.HAS_SE] = 1.0 * stack0_preserve_s
    attn.W_q[row, BD.CMP + 3] = -5.0 * stack0_preserve_s
    attn.W_q[row, BD.MEM_STORE] = -10.0 * stack0_preserve_s
    attn.W_q[row, BD.IS_BYTE] = -10.0 * stack0_preserve_s
    attn.W_q[row, BD.MARK_AX] = -10.0 * stack0_preserve_s
    attn.W_q[row, BD.H1 + ax_i] = 10.0 * stack0_preserve_s
    attn.W_q[row, BD.OP_LI_RELAY] = 10.0 * stack0_preserve_s
    attn.W_q[row, BD.OP_LC_RELAY] = 10.0 * stack0_preserve_s
    attn.W_q[row, BD.MARK_PC] = -10.0 * stack0_preserve_s
    attn.W_q[row, BD.MARK_SP] = -10.0 * stack0_preserve_s
    attn.W_q[row, BD.MARK_BP] = -10.0 * stack0_preserve_s
    attn.W_q[row, BD.MARK_MEM] = -10.0 * stack0_preserve_s
    attn.W_k[row, BD.CONST] = -2.0 * stack0_preserve_s
    attn.W_k[row, BD.H1 + 10] = 2.0 * stack0_preserve_s
    attn.W_k[row, BD.BYTE_INDEX_0] = 2.0 * stack0_preserve_s
