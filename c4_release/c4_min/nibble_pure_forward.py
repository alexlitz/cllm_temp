"""PURE-FORWARD C4 VM step — one VM step = one ``model.forward``, compute in weights.

This is the mission's *true form*: the whole VM runs through the **vanilla
autoregressive forward**. The token stream + KV cache carry ALL state; the ALU /
dispatch / control live in FFN weights; memory is softmax1-KV attention over the
emitted MEM tokens; PC/registers live in the emitted 30-token frames. The ONLY
Python on the compute path is the standard generation loop — ``argmax`` the next
token and append it, then forward again. There is **no**
``blogspec_run._apply_op`` (python if/elif + integer VMState), **no**
``DictMemStack`` (python dict memory), and **no** functional torch gadget built
per call.

Contrast with the recurrent step-loop (``nibble_vm.run_program`` /
``verify_unified.step_run``): that carries the VM state in a persistent RESIDUAL
vector and snaps lanes in Python between steps. Here the state instead round-trips
through the **token stream**: each step re-reads the register file out of the
previously-emitted 30-token frame by ATTENTION (the spec's "we write the registers
each step ... retrieve by attending"), the baked FFN blocks compute the next step
in the SAME forward, and the LM head emits the next frame's bytes. State lives in
the sequence, exactly as a real decoder-only transformer.

The pure-forward step mechanism
===============================
The token stream is ``BOS`` then a sequence of 30-token register frames
(``blogspec_vocab.build_step_frame``). One VM step is ONE ``model.forward`` over
the whole stream so far; block 0's attention reconstructs the register nibble
state from the most-recent frame, the baked step blocks compute the next state on
that state, and the register value lanes at the LAST position are the next-step
registers. The driver decodes those four register values with the LM byte-head's
value argmax (the spec's own re-quantiser — no ``torch.round``) and APPENDS the
next 30-token frame, closing the loop through the token stream.

  block 0   FRAME-INGEST attention — a softmax1 + ALiBi content-addressable read
            keyed on a per-(register, byte) ROLE (a rigid structural tag of the
            frame slot, like a positional encoding), with a query-exclusion
            penalty and recency ALiBi so the LATEST frame's bytes are gathered
            into this position's register NIBBLE bands. State comes from the
            SEQUENCE, not a python variable.
  block 0 FFN + blocks 1..k  the baked VM STEP — recompose nibbles→scalar lanes,
            fetch@PC over the program (code-as-data), opcode decode, dispatch of
            the op (IMM/LEA/PSH/ADD/SUB/JMP/BZ/BNZ + cmp/bitwise experts; MUL/DIV/MOD
            run via the efficient nibble_alu32 ALU in the Qwen build, not here),
            branch delta, mod-256 fold. Identical persistent weights to
            ``nibble_vm.build_step_model`` — the op RESULT is computed by these
            FFN weights inside ``model.forward``.

The register-nibble ingest is proven byte-exact (``prove_ingest``); the whole step
is proven byte-exact vs ``isa.interpret`` by ``run_pure_forward`` under a trace
guard (``assert_no_python_compute``) that fails if ``_apply_op`` / ``DictMemStack``
/ a per-call gadget is ever entered.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import blogspec_vocab as V
from .nibble_vm_layout import NibbleVMLayout
from .nibble_vm import (
    S, RELU_S, SILU_S, build_step_model, load_program,
    compile_nibble_to_scalar, compile_pc_fetch, compile_code_select,
    compile_opcode_decode, base_dispatch_rules, compile_branch_delta,
    compile_fold, compile_ffn, _empty_spec, _load_ffn, _zero_attn,
    _snap_lane, VALVOCAB,
)
from .blogspec_model import Transformer, Attn, softmax1


def _snap_nib(x: float) -> int:
    """The nibble re-quantiser: ``argmax_n (2*n*x - n^2)`` over n in 0..15 — the
    same vanilla LM-head argmax the register decode uses (``_snap_lane`` for a full
    value, this for a single 4-bit nibble), NOT ``torch.round``.  Residue-immune:
    an argmax over the 16 discrete candidates, so it snaps a nibble dim to its exact
    integer regardless of any O(1e-5) fp residue."""
    best, bn = -1e30, 0
    for n in range(16):
        s = 2.0 * n * x - n * n
        if s > best:
            best, bn = s, n
    return bn


# The registers the ingest reconstructs from the prior frame, in frame order.
# Each is 4 bytes; STACK0 is carried in the STACK0 slot of the frame (extended
# frame). The base 30-token frame carries PC/AX/SP/BP; STACK0 is carried in the
# frame's MEM value slot for the pure-forward stack-top mirror.
INGEST_REGS = ["PC", "AX", "SP", "BP", "STACK0"]
BYTES_PER_REG = 4
N_ROLES = len(INGEST_REGS) * BYTES_PER_REG          # 20 (register, byte) roles


# ===========================================================================
# The pure-forward layout: the baked VM step-block bands + the ingest role bands.
# ===========================================================================
class PureForwardLayout(NibbleVMLayout):
    """``NibbleVMLayout`` (the baked step bands) + the frame-ingest role bands.

    The extra bands are set on the residual by the driver as it embeds each frame
    (a rigid structural tag of the 30-token frame slot — like a positional
    encoding — NOT computed VM state):

      ``ROLE`` (N_ROLES)   — per (register, byte) one-hot; a frame byte token in
                             slot (r, bi) carries ROLE[r*4+bi]=1 (its KEY), the
                             ingest query for that slot carries the same one-hot.
      ``IS_FRAME_BYTE`` (1)— 1.0 on real frame byte tokens (KV candidates); 0 on
                             markers / the query row (query-exclusion penalty).

    KV-memory bands (§Memory softmax1 CAM over the emitted store frames):
      ``ADDR_BIN`` (32)    — store address bits (KEY) on a store frame's MEM token.
      ``QRY_BIN``  (32)    — load address bits (QUERY) on the load step's query row.
      ``VAL_NIB``  (16)    — store value nibbles (VALUE) on a store frame's MEM token.
      ``IS_STORE`` (1) / ``IS_LOAD`` (1) — the store/load role flags.
    """

    def __init__(self, code_size: int, n_heads: int,
                 include_memory: bool = True, include_cmp: bool = True,
                 include_muldiv: bool = False):
        # SUBSET-AWARE ALLOCATION.  The optional op families each own a chunk of the
        # residual (memory KV = 82 dims, cmp = 11 dims).  The ``include_*`` flags (which
        # the subset builders thread) GATE the allocation: a family whose block is not
        # baked no longer reserves its band.  A skipped family's band attributes stay
        # ``None``.
        #
        # ``include_muldiv`` (default False) is now a LEGACY no-op knob.  The 256x256x3
        # MUL/DIV/MOD lookup table has been REMOVED entirely (it was the ~45 GB wall),
        # so this LEAN blogspec model no longer carries a MUL/DIV/MOD path at all — the
        # ONLY MUL/DIV/MOD implementation is the efficient ``nibble_alu32`` ALU used by
        # the Qwen build (``qwen_full_vm``).  The flag is kept so existing callers that
        # pass ``include_muldiv=`` keep working; the value no longer allocates any band
        # or bakes any block (the table is gone).
        super().__init__(code_size, n_heads=n_heads)
        self._off = self.D
        self.ROLE = self._band("ROLE", N_ROLES)
        self.IS_FRAME_BYTE = self._scalar("IS_FRAME_BYTE")
        # KV-memory bands (only when the memory op family is baked).
        from .blogspec_memory import ADDR_BITS as _AB
        from .blogspec_layout import NIB_PER_REG as _NR
        self.ADDR_BIN = self.QRY_BIN = self.VAL_NIB = None
        self.IS_STORE = self.IS_LOAD = None
        if include_memory:
            self.ADDR_BIN = self._band("ADDR_BIN", _AB)
            self.QRY_BIN = self._band("QRY_BIN", _AB)
            self.VAL_NIB = self._band("VAL_NIB", _NR)
            self.IS_STORE = self._scalar("IS_STORE")
            self.IS_LOAD = self._scalar("IS_LOAD")
        # CMP result scratch lanes (computed ungated from d = STK - AX each step).
        # GT/LT are SIGNED (two's-complement, C4-faithful): the magnitude order is
        # corrected by the two operands' sign bits so cross-sign pairs compare
        # correctly (see ``compile_cmp_compute`` + ``compile_cmp_signed_finalize``).
        self.CMP_EQ = self.CMP_GT = self.CMP_LT = None
        self.MAG_GT = self.MAG_LT = self.SGN_STK = self.SGN_AX = None
        if include_cmp:
            self.CMP_EQ = self._scalar("CMP_EQ")     # 1 iff STK == AX
            self.CMP_GT = self._scalar("CMP_GT")     # 1 iff STK  > AX (SIGNED, final)
            self.CMP_LT = self._scalar("CMP_LT")     # 1 iff STK  < AX (SIGNED, final)
            # signed-compare intermediates (cmp-compute writes; cmp-finalize consumes).
            self.MAG_GT = self._scalar("MAG_GT")     # UNSIGNED (STK>AX), raw ramp
            self.MAG_LT = self._scalar("MAG_LT")     # UNSIGNED (STK<AX), raw ramp
            self.SGN_STK = self._scalar("SGN_STK")   # 1 iff STK bit31 set (negative)
            self.SGN_AX = self._scalar("SGN_AX")     # 1 iff AX  bit31 set (negative)
        # (The 256x256x3 MUL/DIV/MOD lookup-table operand one-hot bands MDM_A_OH/
        # MDM_B_OH/MDM_RES have been removed — the table is gone; MUL/DIV/MOD run only
        # through the efficient nibble_alu32 ALU in the Qwen build.)
        while self._off % n_heads != 0:
            self._scalar(f"_pfpad{self._off}")
        self.D = self._off


# ===========================================================================
# FRAME-INGEST attention — the content-addressable read of the prior frame.
#
# One head per (register, byte) role gathers that slot's two nibbles from the
# latest frame into the register's nibble band. Each head is the same CAM: KEY =
# +smag on the role dim the token holds, QUERY = +smag on the role this head
# reconstructs, a query-exclusion penalty drives non-frame-byte rows to -PEN, and
# a recency ALiBi picks the LATEST frame among equal roles (loops re-emit the same
# roles every step). The VALUE is the token's CUR_NIB nibbles; W_o writes them
# into the register nibble band dims 2*bi+0 / 2*bi+1.
# ===========================================================================
INGEST_EFF = 4000.0                 # per-role match contribution (huge; §Memory)
INGEST_RECENCY = 6.0                # ALiBi recency slope (latest frame wins)


def bake_frame_ingest(attn, L: PureForwardLayout, reg_bases: Dict[str, int]) -> None:
    """Bake the N_ROLES-head frame-ingest CAM. ``reg_bases`` maps register name ->
    its nibble-band base. head ``h = r*4+bi`` gathers register ``r``'s byte ``bi``.
    Requires ``attn.n_heads >= N_ROLES`` and ``attn.head_dim >= 3`` (2 CAM channels
    + 1 penalty; value uses 2 local channels)."""
    hs = attn.scale
    smag = (INGEST_EFF / hs) ** 0.5
    PEN = 100.0 * INGEST_EFF
    p = (PEN / hs) ** 0.5
    HD = attn.head_dim
    for w in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
        w.zero_()
    for h in range(N_ROLES):
        attn.alibi_slopes[h] = INGEST_RECENCY
        base = h * HD
        r_idx, bi = divmod(h, BYTES_PER_REG)
        reg_name = INGEST_REGS[r_idx]
        reg_base = reg_bases[reg_name]
        # CAM channel 0: role match (key = the token's role dim, query = this role).
        attn.W_k[base + 0, L.ROLE + h] = smag
        attn.W_q[base + 0, L.ROLE + h] = smag
        # penalty channel 1: non-frame-byte rows keyed -p (via ONE), byte rows 0.
        # The query keys +p via ONE (constant), so a non-frame-byte candidate (incl.
        # the query row itself, IS_FRAME_BYTE=0) scores -p*hs = -PEN and can never
        # win; a real frame byte row (IS_FRAME_BYTE=1) keys 0 here.
        attn.W_q[base + 1, L.ONE] = p
        attn.W_k[base + 1, L.ONE] = -p
        attn.W_k[base + 1, L.IS_FRAME_BYTE] = p
        # value: the two nibbles carried by the byte token in CUR_NIB.
        attn.W_v[base + 0, L.CUR_NIB + 0] = 1.0
        attn.W_v[base + 1, L.CUR_NIB + 1] = 1.0
        # write into the register nibble band's two dims for byte bi.
        attn.W_o[reg_base + 2 * bi + 0, base + 0] = 1.0
        attn.W_o[reg_base + 2 * bi + 1, base + 1] = 1.0


# ===========================================================================
# GQA frame-ingest — 20 query heads sharing ONE KV head (the KV-cache reduction).
#
# The stock ``bake_frame_ingest`` gives each of the 20 heads a DISTINCT K slice (it
# keys on ``ROLE+h``, a different role dim per head), so the KV cache stores 20
# distinct K/V heads even though the 20 heads all read the SAME previous frame.  As
# in stock Qwen's GQA (14 query / 2 KV), the query heads can share ONE KV head: the
# shared K encodes ALL 20 role identities (so it records WHICH role each frame byte
# holds) and the shared V is the token's two nibbles (already identical across the 20
# heads).  Each of the 20 QUERY heads then sets its own Q to match the shared-K
# channel for ITS role and routes the selected byte's nibbles to its register band.
# Every head's K/V slice is byte-identical ⇒ the cache stores ONE KV head's worth
# (20 KV heads → 1), while the 20 query reads stay byte-EXACT (proven vs the stock
# ingest + ``isa.interpret`` through the real ``model.forward``).
#
# WHY QUERY HEADS STAY AT 20:  softmax1 selects ONE row per head (the role-CAM argmax),
# and the 20 register bytes live at 20 DISTINCT frame-token positions.  A head whose
# query matches SEVERAL roles splits its softmax weight across those byte tokens and
# returns their AVERAGE — it cannot separate them (empirically: a 5-head "one per
# register, 4-role query" gather decodes garbage).  So each distinct byte position
# needs its own query-head selection; 20 is the minimum for full 5-register (×4-byte)
# frame reconstruction.  (STACK0's 4 heads are load-bearing in the base pure-forward
# model — the pushed ALU operand is reconstructed from the STACK0 frame mirror — so
# 16 heads mis-reads ADD/SUB; the pure-forward-COMPLETE model instead supplies STACK0
# via a separate stack-pop KV head, where the 4 ingest STACK0 heads could be elided.)
# ===========================================================================
def bake_frame_ingest_gqa(attn, L: PureForwardLayout,
                          reg_bases: Dict[str, int]) -> None:
    """GQA form of :func:`bake_frame_ingest`: 20 query heads, ONE shared KV head.

    All 20 heads carry an IDENTICAL K/V projection (the shared KV head); only the
    per-head Q + O differ.  Byte-exact frame reconstruction, KV-cache content
    collapsed 20 → 1 distinct KV head.  Requires ``attn.head_dim >= N_ROLES + 3``.
    """
    hs = attn.scale
    smag = (INGEST_EFF / hs) ** 0.5
    PEN = 100.0 * INGEST_EFF
    p = (PEN / hs) ** 0.5
    HD = attn.head_dim
    assert HD >= N_ROLES + 3, (
        f"GQA ingest needs head_dim >= {N_ROLES + 3} (all role keys + penalty + "
        f"2 value channels), got {HD}")
    for w in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
        w.zero_()
    # SHARED K/V channel layout inside every head slice (identical across heads):
    #   0..N_ROLES-1 : role-r match key (a byte token keys +smag on the role dim it holds)
    #   N_ROLES      : query-exclusion penalty (non-frame-byte rows keyed -p; bytes 0)
    #   N_ROLES+1/+2 : the token's two value nibbles (CUR_NIB+0 / CUR_NIB+1)
    cPEN, cV0, cV1 = N_ROLES, N_ROLES + 1, N_ROLES + 2
    for h in range(N_ROLES):
        attn.alibi_slopes[h] = INGEST_RECENCY
        base = h * HD
        r_idx, bi = divmod(h, BYTES_PER_REG)
        reg_base = reg_bases[INGEST_REGS[r_idx]]
        # --- SHARED K (byte-identical for every head): all 20 role keys + penalty --
        for r in range(N_ROLES):
            attn.W_k[base + r, L.ROLE + r] = smag
        attn.W_k[base + cPEN, L.ONE] = -p
        attn.W_k[base + cPEN, L.IS_FRAME_BYTE] = p
        # --- SHARED V (byte-identical for every head): the token's two nibbles -----
        attn.W_v[base + cV0, L.CUR_NIB + 0] = 1.0
        attn.W_v[base + cV1, L.CUR_NIB + 1] = 1.0
        # --- PER-HEAD Q: select THIS head's role via the shared-K channel h + penalty
        attn.W_q[base + h, L.ROLE + h] = smag
        attn.W_q[base + cPEN, L.ONE] = p
        # --- PER-HEAD O: write the selected byte's two nibbles into the register band
        attn.W_o[reg_base + 2 * bi + 0, base + cV0] = 1.0
        attn.W_o[reg_base + 2 * bi + 1, base + cV1] = 1.0


def ingest_gqa_enabled() -> bool:
    """``C4_INGEST_GQA`` (default OFF): use the 1-KV-head GQA ingest bake instead of
    the stock 20-distinct-KV-head bake.  OFF ⇒ byte-identical to the golden build."""
    import os
    return os.environ.get("C4_INGEST_GQA", "0") not in ("0", "", "false", "False")


# ===========================================================================
# WIDE-VALUE single-head ingest — 1 QUERY head + 1 KV head (down from 20+1).
#
# CONFIRMS the user's hypothesis (refutes agent a8c09504's "20 query heads are the
# minimum"): a8c09504's 5-head/4-role gather decoded garbage ONLY because it kept the
# NARROW value projection — all registers write the SAME CUR_NIB band, so a multi-role
# query averages them destructively.  The WIDE-VALUE case routes each (register, byte)
# to its OWN band (a concatenation), so a SINGLE role-agnostic query attending to all
# 20 frame-byte tokens returns  out = Σ_i w_i·V_i = [w_0·v_0 | w_1·v_1 | ... ] — a
# SCALED CONCATENATION, not a blend.  A fixed downstream FFN rescales lane r by 1/wtot_r
# (wtot_r = Σ_frames w_{f,r}, the FIXED positional weight fraction — length-invariant
# because the huge match logit makes softmax1 scale-free in n_frames), and the nibble
# argmax re-quant snaps the fp residue.  PROVEN byte-EXACT through the real FFN.forward +
# Attn.forward (see ``c4_min/_wide_ingest_integrated.py``) on the a8c09504 battery incl.
# multi-frame loops, both fp32 and fp64.
#
# Three in-weight stages (NO python compute): (A) a PRE-ROUTE SwiGLU FFN computes the
# per-role product PREROUTE[2r+b] = CUR_NIB[b]·[ROLE==r] (the role⊙nibble gate a LINEAR
# W_v cannot form — role & nibble are ADDED on the residual, never multiplied);
# (B) the WIDE-value single head (role-agnostic query; V = identity copy of the 40
# PREROUTE dims -> the scaled concat in a GATHER band); (C) a RESCALE FFN lane*=1/wtot_r.
#
# HONEST CAVEATS (both proven, both measured):
#   * band WIDTH: PREROUTE(40)+GATHER(40) = 80 fresh dims (vs the 20-head bake's 0), and
#     the wide head needs head_dim >= 2+40 = 42.  In the base build (dim=700, n_heads=20
#     -> head_dim=35) the 40 value lanes DO NOT FIT one 35-ch head slice; the wide head
#     needs n_heads=1 for block 0 (head_dim=dim) OR a widen — a genuine restructure, so
#     this lands as a validated builder + probe, NOT yet swapped into build_pure_forward.
#   * w_i degeneracy (fp32): the fixed 1/wtot rescale + latest-frame selection conflict
#     bounds the ALiBi recency to a WINDOW.  fp32 byte-exact (8 DIFFERING loop frames):
#     recency ~[0.12, 1.0], sweet spot ~0.5 (worst nibble residue ~0.002, margin ~0.498).
#     recency > 1.0 fails (1/wtot too large x fp32-quantized value -> residue > 0.3);
#     recency < 0.12 fails (earlier differing frames leak past the total-weight rescale).
#     fp64 is byte-exact across the whole range (residue ~1.8e-15); the window is an fp32
#     artifact.  The stock 20-head bake uses recency 6.0 (per-head role-CAM, no rescale),
#     which is OUTSIDE this window — the wide head must use ~0.5.
# ===========================================================================
def ingest_wide_enabled() -> bool:
    """``C4_INGEST_WIDE`` (default OFF): use the 1-query + 1-KV wide-value ingest.

    OFF is a strict no-op (the builders skip the whole wide path) ⇒ the golden
    ``_fingerprint_build`` hash ``8f4dd780`` is unchanged.  ON is WIRED into the
    production ``build_pure_forward_model`` AND ``build_pure_forward_complete_model``:
    it allocates the 80-dim PREROUTE/GATHER band, prepends the
    ``wide-preroute | wide-gather | wide-snap`` blocks in front of block 0, and swaps
    the wide-gather block's attention for a 1-head Attn carrying the single wide gather
    head (block-0 ingest heads 20/21 → 1).  Byte-EXACT through ``run_pure_forward`` +
    ``run_pure_forward_complete`` vs the stock 20-head build (the reconstruction is a
    scaled concat the fixed 1/wtot rescale + nibble re-quant recover to the bit).  The
    flag-ON fingerprint MOVES (intended); flag-OFF stays ``8f4dd780``."""
    import os
    return os.environ.get("C4_INGEST_WIDE", "0") not in ("0", "", "false", "False")


# Recommended ALiBi recency for the wide head (inside the fp32 byte-exact window).
WIDE_INGEST_RECENCY = 0.5


# ===========================================================================
# WIDE-INGEST BUILDERS (production form of ``_wide_ingest_integrated.py``).
#
# The three stages that replace the 20-head (or 1-KV GQA) role-CAM ingest with a
# SINGLE query head + SINGLE KV head, wired into ``build_pure_forward_model`` /
# ``build_pure_forward_complete_model`` behind ``C4_INGEST_WIDE`` (default OFF):
#
#   (A) ``compile_wide_preroute`` — a SwiGLU FFN block (attn zeroed) that computes
#       ``PREROUTE[2r+b] = CUR_NIB[b]·[ROLE==r]`` on every frame-byte token.
#   (B) ``bake_wide_ingest_head`` — the block-0 attention: ONE role-AGNOSTIC query
#       head + ONE KV head (``V`` = identity copy of the 40 PREROUTE dims) whose
#       ``W_o`` writes the scaled concat into the fresh GATHER band.  Needs
#       ``head_dim >= 2 + 2*N_ROLES`` (= 42), which a 1-head block-0 (head_dim=dim)
#       trivially satisfies.  ALiBi recency = ``WIDE_INGEST_RECENCY`` (0.5).
#   (C) ``compile_wide_rescale`` — a SwiGLU FFN block that writes
#       ``reg_nib[r,2*bi+b] = GATHER[2r+b] / wtot_r`` (self-clearing the reg band
#       first), with ``wtot_r`` the FIXED positional weight fraction (baked at
#       n_frames=1, length-invariant because the huge match logit makes softmax1
#       scale-free in n_frames).
#   (D) ``compile_wide_nibble_snap`` — an in-weight integer re-quant of the reg
#       nibble band (the argmax the spec applies at decode).  LOAD-BEARING: the
#       recompose weights nibble j by ``16^j``, so a ~1e-8 gather residue on a high
#       nibble becomes ``~1.5e-3`` in the recomposed scalar — enough to shift the
#       triangular-pulse opcode decode off its integer.  Snapping first keeps the
#       recompose bit-clean.  The stock 20-head ingest writes exact nibbles so it
#       never needs this.  The downstream recompose (``compile_nibble_to_scalar`` on
#       the existing ingest+recompose block) then folds clean nibbles -> scalar.
#
# The block ORDER when the flag is ON becomes (in front of the stock block 0):
#   wide-preroute (A) | wide-gather (B attn=wide head; C rescale FFN) | wide-snap (D)
#   | ingest+recompose (the STOCK block 0, attn now ZEROED — gather moved to B).
# So block 0's original 20 (or 21) ingest heads collapse to the ONE wide head on the
# wide-gather block; block 0 itself carries no ingest attention.  +3 stored blocks
# (flag-ON only); flag-OFF is a strict no-op (golden ``8f4dd780`` unchanged).
# ===========================================================================
def extend_layout_for_wide_ingest(L) -> None:
    """Allocate the 80 fresh dims the wide ingest needs (flag-ON only):
    ``PREROUTE`` (2*N_ROLES = 40, the per-(role, byte-half) role⊙nibble gate) +
    ``GATHER`` (2*N_ROLES = 40, where the wide head's ``W_o`` writes the scaled
    concat — separate from PREROUTE so the query row's own PREROUTE is never summed
    into its gather).  Must be called BEFORE ``L.D`` is fixed by the head-dim pad."""
    L.PREROUTE = L._band("PREROUTE", 2 * N_ROLES)
    L.GATHER = L._band("GATHER", 2 * N_ROLES)


def _wide_wtot_fractions(recency: float):
    """``wtot_r = Σ_frames w_{f,r}`` for each role r — the FIXED rescale constants,
    computed at n_frames=1 (length-invariant: the huge match logit ``INGEST_EFF``
    makes softmax1 scale-free in n_frames, so the per-frame weight fraction of role r
    is the same at any sequence length).  Same construction as
    ``_wide_ingest_integrated._wtot_fractions`` at the bake length (BOS + one frame),
    so the wired build reuses the reference physics exactly."""
    M = INGEST_EFF
    S_len = 1 + V.FRAME_LEN
    qpos = S_len - 1
    role_to_local = {rr: lc for lc, rr in _FRAME_ROLE_SLOTS.items()}
    scores = torch.full((S_len,), -1e30, dtype=torch.float64)
    for local in _FRAME_ROLE_SLOTS:
        p = 1 + local
        scores[p] = M - recency * abs(qpos - p)
    wsm = softmax1(scores.unsqueeze(0)).squeeze(0)
    return {r: float(wsm[1 + role_to_local[r]]) for r in range(N_ROLES)}


def compile_wide_preroute(L, dim: int) -> Dict[str, torch.Tensor]:
    """(A) SwiGLU: ``PREROUTE[2r+b] = CUR_NIB[b]·[ROLE==r]`` (the per-role gate a
    LINEAR W_v cannot form — role & nibble are ADDED on the residual, never
    multiplied).  One hidden unit per (role r, byte-half b):
      gate = CUR_NIB[b]                     (the nibble value 0..15)
      up   = S·(ROLE+r) - 0.5·S             (>0 iff ROLE==r; silu -> S else ~0)
      down = 1/silu(0.5S) into PREROUTE[2r+b]
    => hidden = silu(up)·gate = (ROLE==r ? S : 0)·nib ; down scales silu(0.5S)->1."""
    n_units = 2 * N_ROLES
    spec = _empty_spec(dim, n_units)
    u = 0
    for r in range(N_ROLES):
        for b in range(2):
            spec["W_gate"][u, L.CUR_NIB + b] = 1.0    # gate = the nibble value
            spec["W_up"][u, L.ROLE + r] = S           # up = S iff ROLE==r ...
            spec["b_up"][u] = -0.5 * S                # ... (silu(0.5S) on, ~0 off)
            spec["W_down"][L.PREROUTE + 2 * r + b, u] = 1.0 / SILU_HALF
            u += 1
    return spec


def compile_wide_rescale(L, reg_bases: Dict[str, int], dim: int,
                         recency: float = WIDE_INGEST_RECENCY
                         ) -> Dict[str, torch.Tensor]:
    """(C) SwiGLU: ``reg_nib[r,2*bi+b] = GATHER[2r+b] / wtot_r`` (self-clearing the
    register nibble band first, SET semantics).  ``wtot_r`` is the fixed positional
    weight fraction from :func:`_wide_wtot_fractions`."""
    wtot = _wide_wtot_fractions(recency)
    reads = []                                        # (dst_nib_dim, src_gather_dim, inv)
    for r in range(N_ROLES):
        r_idx, bi = divmod(r, BYTES_PER_REG)
        reg_base = reg_bases[INGEST_REGS[r_idx]]
        inv = 1.0 / wtot[r]
        for b in range(2):
            reads.append((reg_base + 2 * bi + b, L.GATHER + 2 * r + b, inv))
    clears = sorted({dst for dst, _, _ in reads})
    n_units = len(clears) + len(reads)
    spec = _empty_spec(dim, n_units)
    u = 0
    for dst in clears:                                # clear each reg nibble dim (SET)
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, dst] = 1.0
        spec["W_down"][dst, u] += -1.0 / SILU_S
        u += 1
    for dst, src, inv in reads:                       # add GATHER[src]/wtot_r
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, src] = 1.0
        spec["W_down"][dst, u] += inv / SILU_S
        u += 1
    return spec


def compile_wide_nibble_snap(L, reg_bases: Dict[str, int], dim: int,
                             n_nib: int = 8) -> Dict[str, torch.Tensor]:
    """Re-quant (SNAP) each register nibble to its exact integer 0..15 IN WEIGHTS —
    the argmax re-quant the spec applies at decode, done here on the residual so the
    downstream recompose (``Σ 16^j·nib_j``) reads CLEAN nibbles.

    WHY THIS IS LOAD-BEARING FOR THE WIDE INGEST: the wide gather leaves a ~1e-8
    residue on the (all-zero) HIGH nibbles of a small register (the cross-frame
    softmax weight the fixed 1/wtot rescale cannot cancel to the bit).  At the
    nibble level that residue is far inside the 0.5 argmax-snap margin (the ingest is
    byte-exact), but the recompose amplifies nibble j by ``16^j`` — nibble 4's 1e-8
    residue becomes ``65536·1e-8 ≈ 1.5e-3`` in PC_VAL, enough to shift the
    triangular-pulse opcode decode off its integer (OP_IS -> 0.98) and corrupt the
    op.  The stock 20-head ingest writes exact nibbles so it never needs this; the
    wide ingest does.  Snapping the nibble band BEFORE the recompose restores exact
    integers, so PC_VAL/AX_VAL/… are bit-clean and every op decodes byte-identically.

    Snap = a monotone 16-cell STAIRCASE: ``snap(v) = Σ_{t=1}^{15} ramp(v-(t-0.5))``
    where each ``ramp`` is a sharp relu step (width ``w``) rising 0->1 across the
    half-integer boundary.  For an integer-ish ``v ≈ n`` (n in 0..15) exactly ``n``
    of the 15 steps are on, so the sum is ``n``.  SET semantics (self-clear first).
    Residue-immune to any |residue| < 0.5 - w/2.  ``n_nib`` nibbles per register."""
    from .nibble_vm import RELU_S
    w = 0.2
    regs = sorted({reg_bases[r] for r in INGEST_REGS})
    dsts = [(rb + j) for rb in regs for j in range(n_nib)]
    # per dst: 1 self-clear + 15 steps * 2 relu (rise/fall of each staircase step).
    n_units = len(dsts) * (1 + 15 * 2)
    spec = _empty_spec(dim, n_units)
    u = 0
    for dst in dsts:
        spec["W_up"][u, L.ONE] = S                    # self-clear the nibble (SET)
        spec["W_gate"][u, dst] = 1.0
        spec["W_down"][dst, u] += -1.0 / SILU_S
        u += 1
        for t in range(1, 16):                        # 15 staircase steps at t-0.5
            lo = t - 0.5
            for idx, thr in enumerate((lo, lo + w)):
                spec["W_up"][u, dst] = RELU_S
                spec["b_up"][u] = -RELU_S * thr
                spec["W_gate"][u, L.ONE] = 1.0
                spec["W_down"][dst, u] += (1.0 if idx == 0 else -1.0) / (RELU_S * w)
                u += 1
    return spec


def bake_wide_ingest_head(attn, L, recency: float = WIDE_INGEST_RECENCY) -> None:
    """(B) ONE query head + ONE KV head on a 1-head block-0 (head_dim = dim).
      K: match ch keys +smag on ONE (role-AGNOSTIC); penalty ch keys -p·ONE +
         p·IS_FRAME_BYTE (a non-frame-byte / query row scores -p·hs, can never win).
      Q: match ch +smag on ONE; penalty ch +p·ONE.
      V: identity copy of the 40 PREROUTE dims into local value channels 2..41.
      O: copy those value channels -> the FRESH GATHER band (0 on every row, so the
         gather is the pure scaled concat with no query-row PREROUTE self-pollution).
    Requires ``attn.n_heads == 1`` and ``head_dim >= 2 + 2*N_ROLES``."""
    assert attn.n_heads == 1, "wide ingest head requires a 1-head block-0 attn"
    HD = attn.head_dim
    assert HD >= 2 + 2 * N_ROLES, (
        f"wide ingest needs head_dim >= {2 + 2 * N_ROLES} (match + penalty + 40 "
        f"value channels), got {HD}")
    hs = attn.scale
    smag = (INGEST_EFF / hs) ** 0.5
    PEN = 100.0 * INGEST_EFF
    p = (PEN / hs) ** 0.5
    for w in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
        w.zero_()
    attn.alibi_slopes[0] = recency
    cMATCH, cPEN = 0, 1
    attn.W_k[cMATCH, L.ONE] = smag
    attn.W_q[cMATCH, L.ONE] = smag
    attn.W_k[cPEN, L.ONE] = -p
    attn.W_k[cPEN, L.IS_FRAME_BYTE] = p
    attn.W_q[cPEN, L.ONE] = p
    for k in range(2 * N_ROLES):
        vch = 2 + k
        attn.W_v[vch, L.PREROUTE + k] = 1.0
        attn.W_o[L.GATHER + k, vch] = 1.0


def _make_one_head_attn(dim: int, max_seq_len: int):
    """A fresh 1-head ``Attn`` (softmax1 + ALiBi) to swap into block 0 for the wide
    ingest — head_dim = dim, so the 40 value channels fit.  Swapping a SINGLE block's
    Attn keeps every OTHER block byte-identical (they retain their own multi-head
    Attn with the build-wide n_heads)."""
    return Attn(dim, n_heads=1, max_seq_len=max_seq_len,
                positional="alibi", sink="softmax1")


# ===========================================================================
# BUILD the pure-forward model: ingest attention on block 0 + the baked step.
# ===========================================================================
MEM_HEAD_CHANNELS = 51               # 32 addr + ZFOD + penalty + load-enable + 16 value


# Ops the pure-forward interpreter decodes (base + memory when included).
PF_MEM_OPS = [isa.LI, isa.LC, isa.SI, isa.SC]


def compile_opcode_decode_pf(L, dim: int) -> Dict[str, torch.Tensor]:
    """``OP_IS[op] = (OP_VAL == op)`` for the base ops + memory ops LI/LC/SI/SC +
    the comparisons EQ/NE/LT/GT/LE/GE (the triangular-pulse decode)."""
    from .nibble_vm import BASE_OPS
    from .nibble_unified import compile_opcode_decode_ops
    ops = sorted(set(BASE_OPS + PF_MEM_OPS +
                     [isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE] +
                     [isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR] +
                     [isa.MUL, isa.DIV, isa.MOD]))
    return compile_opcode_decode_ops(L, dim, ops)


def compile_mem_prep(L, dim: int) -> Dict[str, torch.Tensor]:
    """The pre-CAM memory prep FFN (runs after opcode decode, before the mem-cam
    attention). On a LOAD (OP_IS[LI] or OP_IS[LC]): set IS_LOAD=1, expand the
    ingested AX_VAL -> QRY_BIN (the load address), and CLEAR the AX nibble band so
    the CAM's additive write lands on exactly the loaded value. On a STORE (OP_IS[SI]
    /OP_IS[SC]): set IS_STORE=1 (this position becomes the store KV row) and expand
    the address (STK_VAL, the popped address) -> ADDR_BIN and copy AX nibbles ->
    VAL_NIB. Everything gated on the decoded opcode one-hot — in weights, no python."""
    specs = []
    # IS_LOAD = OP_IS[LI]+OP_IS[LC] (SET, opcode-gated). This is the ONLY store/load
    # flag the model sets on the CURRENT row; the STORE-side bands (IS_STORE,
    # ADDR_BIN, VAL_NIB) live on PAST store-frame MEM tokens and are set by the
    # driver's overlay from the emitted MEM bytes (the §Memory write log in the token
    # stream), so the current-step FFN must NOT touch them (it would clobber a
    # store row's key when it runs at that row).
    specs.append(_flag_from_ops(L, L.IS_LOAD, [isa.LI, isa.LC], dim))
    # QRY_BIN <- AX_VAL bits (the load address). Ungated: QRY_BIN is only READ by the
    # CAM when IS_LOAD is set (the load-enable channel gates the whole head).
    specs.append(compile_addr_expand(L, L.AX_VAL, L.QRY_BIN, dim))
    # clear AX nibble band on a load (so the CAM's additive write is a clean SET).
    specs.append(_clear_band_gated(L, L.AX, 8, dim, gate_ops=[isa.LI, isa.LC]))
    return _concat_specs(specs, dim)


def _concat_specs(specs, dim):
    tot = sum(s["W_up"].shape[0] for s in specs)
    out = _empty_spec(dim, tot)
    u = 0
    for s in specs:
        h = s["W_up"].shape[0]
        out["W_up"][u:u + h] = s["W_up"]; out["b_up"][u:u + h] = s["b_up"]
        out["W_gate"][u:u + h] = s["W_gate"]; out["b_gate"][u:u + h] = s["b_gate"]
        out["W_down"][:, u:u + h] = s["W_down"]
        out["b_down"] += s["b_down"]
        u += h
    return out


def _flag_from_ops(L, flag_band, ops, dim, sharpen: bool = True):
    """flag_band := (a THRESHOLDED) indicator of ``any OP_IS[op] active``.

    ``sharpen=True`` (default) writes a clean STEP: ``flag = 1`` when the summed
    op-hot ``g = Σ OP_IS[op]`` exceeds ~0.5, else ``0`` — a saturating relu ramp
    (0 for g<0.4, 1 for g>0.6, linear between) so a sub-permille opcode-decode
    RESIDUE (g = 1-ε at a large PC) snaps to an EXACT 1.0 before it reaches the
    §Memory CAM's role-gate channel.  Without this, the gate penalty
    ``-PEN·(1-flag)`` (PEN huge, to dominate the address separation) turned a tiny
    ``ε`` into a catastrophic ``-PEN·ε`` that sank an EXACT-address load to ZFOD 0
    — the malloc_printf frame-pointer read-back bug (and the deep-frame ceiling on
    mandelbrot / self-emulation, where PC/SP grow large).  ``sharpen=False`` keeps
    the legacy linear copy (``flag = Σ OP_IS[op]``).

    SET semantics: self-clear ``flag_band`` first, then write the (sharpened) sum.
    """
    if not sharpen:
        spec = _empty_spec(dim, 1 + len(ops))
        spec["W_up"][0, L.ONE] = S; spec["W_gate"][0, flag_band] = 1.0
        spec["W_down"][flag_band, 0] += -1.0 / SILU_S
        for i, op in enumerate(ops, start=1):
            spec["W_up"][i, L.ONE] = S; spec["W_gate"][i, L.OP_IS + op] = 1.0
            spec["W_down"][flag_band, i] += 1.0 / SILU_S
        return spec
    # sharpened step: 3 units — self-clear + two relu ramps (lo=0.4, hi=0.6) of the
    # summed op-hot ``g`` giving a saturating 0..1 indicator, residue-immune.
    spec = _empty_spec(dim, 3)
    # unit 0: self-clear flag_band (SET) — silu(S)/SILU_S · flag_band = flag_band.
    spec["W_up"][0, L.ONE] = S; spec["W_gate"][0, flag_band] = 1.0
    spec["W_down"][flag_band, 0] += -1.0 / SILU_S
    # units 1,2: relu(g - lo) - relu(g - hi), scaled to a 0..1 ramp.  up = RELU_S·g
    # (shifted); gate = constant 1 (b_gate=1) so silu(up)·1 ≈ RELU_S·relu(shift).
    RAMP = 1.0 / (0.2 * RELU_S)          # (relu(g-.4) - relu(g-.6)) * 5 -> 0..1
    for k, (lo, sign) in enumerate(((0.4, +1.0), (0.6, -1.0)), start=1):
        for op in ops:
            spec["W_up"][k, L.OP_IS + op] = RELU_S
        spec["b_up"][k] = -RELU_S * lo
        spec["b_gate"][k] = 1.0          # gate = 1 (silu(up)·1)
        spec["W_down"][flag_band, k] += sign * RAMP
    return spec


def _clear_band_gated(L, base, n, dim, gate_ops):
    """Clear band dims base..base+n-1 when any OP_IS[gate_ops] is active (SET -old)."""
    spec = _empty_spec(dim, n * len(gate_ops))
    u = 0
    for op in gate_ops:
        g = L.OP_IS + op
        for j in range(n):
            spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
            spec["W_gate"][u, base + j] = 1.0
            spec["W_down"][base + j, u] += -1.0 / SILU_HALF
            u += 1
    return spec


def _copy_nibbles(L, src_base, dst_base, dim, n=16):
    """dst nibbles := src nibbles (SET, ungated)."""
    spec = _empty_spec(dim, 2 * n)
    u = 0
    for j in range(n):
        spec["W_up"][u, L.ONE] = S                            # clear dst
        spec["W_gate"][u, dst_base + j] = 1.0
        spec["W_down"][dst_base + j, u] += -1.0 / SILU_S; u += 1
        spec["W_up"][u, L.ONE] = S                            # + src
        spec["W_gate"][u, src_base + j] = 1.0
        spec["W_down"][dst_base + j, u] += 1.0 / SILU_S; u += 1
    return spec


from .nibble_vm import SILU_HALF


# ---------------------------------------------------------------------------
# AX nibbles -> QRY_BIN (32 per-bit dims) — a fixed FFN, not python.
# bit b of the address = bit (b%4) of NIBBLE (b//4). The address already lives as
# NIBBLES in the residual (L.AX+j holds nibble j, value 0..15), so we expand it
# PER-NIBBLE: a 16-cell one-hot of each queried nibble (16 relu instead of 256),
# then bit-select the 4 bits out of that nibble. For an 8-bit address that is 2
# nibbles × ~18 relu instead of a dense 256-cell scalar one-hot (~8× smaller).
# ---------------------------------------------------------------------------
# Which nibble band feeds each scalar value lane (so an ``AX_VAL`` query is
# expanded from the AX nibble band, ``STK_VAL`` from STACK0, etc.). Auto-derived
# from the layout's ``reg_pairs`` — the per-nibble path uses the band; a caller
# whose scalar lane has no nibble band (rare) falls back to the scalar one-hot.
def _nib_base_for_lane(L, src_lane: int) -> Optional[int]:
    for nib_base, val_lane in L.reg_pairs():
        if val_lane == src_lane:
            return nib_base
    return None


def compile_addr_expand(L, src_lane: int, bin_base: int, dim: int,
                        n_bits: int = 8, clear_bits: Optional[int] = None,
                        src_nib_base: Optional[int] = None
                        ) -> Dict[str, torch.Tensor]:
    """``QRY_BIN[b] = bit b of the address`` for b < n_bits (the low byte load
    address), expanded PER-NIBBLE from the address's NIBBLE band.

    bit ``b`` of the address is bit ``b%4`` of nibble ``b//4`` (each nibble is
    4 bits). The address already lives as nibbles in the residual (``src_nib_base
    + nib`` holds nibble ``nib``, an integer 0..15), so instead of the old dense
    256-cell scalar one-hot on ``src_lane`` we expand each QUERIED nibble to a
    16-cell one-hot (the proven triangular pulse, thresholds -1..16 → ~18 shared
    relu units per nibble) and bit-SELECT: ``QRY_BIN[nib*4+k] = Σ_{v:(v>>k)&1}
    onehot(nib==v)``. For an 8-bit address that is ``ceil(8/4)=2`` nibbles instead
    of a 256-cell table — ~8× fewer weights, same output contract.

    ``src_nib_base`` is the base of that nibble band; when ``None`` it is
    auto-derived from ``src_lane`` (``AX_VAL`` → ``L.AX``, ``STK_VAL`` → ``L.STACK0``,
    …). If no nibble band maps to ``src_lane`` the function transparently falls
    back to the legacy scalar 256-cell one-hot on ``src_lane`` (so any caller that
    passes a scalar lane with no backing nibble band still works).

    ``clear_bits`` (default ``ADDR_BITS``=32) is how many QRY_BIN lanes are
    zero-CLEARED before the low ``n_bits`` are (re)set from the address. This MUST
    cover the whole address width the §Memory CAM queries: the store keys expand the
    FULL 32-bit ``ADDR_BIN`` (high bits = 0 for a ≤8-bit stack address), so the load
    QUERY's high bits (8..31) must be an explicit 0 too — otherwise a single stale
    high query bit disagrees with every store's 0 high bit and turns a ``+EFF``
    exact-address match into a ``-EFF`` mismatch (the LI→0 deep-loop fade: the query
    high bits were left uncleared, so a residual QRY_BIN[8]=1 from an earlier step
    destroyed the address match and the load faded to ZFOD).  We only compute the
    low ``n_bits`` from the value (stack addresses fit a byte); bits ``n_bits..
    clear_bits`` are held at 0."""
    from .blogspec_memory import ADDR_BITS
    if clear_bits is None:
        clear_bits = ADDR_BITS
    clear_bits = max(clear_bits, n_bits)
    if src_nib_base is None:
        src_nib_base = _nib_base_for_lane(L, src_lane)
    if src_nib_base is None:
        # no backing nibble band for this scalar lane — legacy scalar one-hot path.
        return _compile_addr_expand_scalar(L, src_lane, bin_base, dim,
                                           n_bits, clear_bits)

    # PER-NIBBLE expand. Queried bits 0..n_bits-1 span nibbles 0..n_nib-1; each
    # nibble owns 4 consecutive bits (the last may be partial when n_bits%4).
    n_nib = (n_bits + 3) // 4
    THR = list(range(-1, 17))          # 18 thresholds -> 16-cell one-hot per nibble
    n_thr = len(THR)
    n_units = n_nib * n_thr + clear_bits   # per-nibble relu banks + one clear/bit
    spec = _empty_spec(dim, n_units)
    # per-nibble 16-cell one-hot relu banks (one bank of n_thr units per nibble).
    for nib in range(n_nib):
        base_u = nib * n_thr
        for jj, t in enumerate(THR):
            u = base_u + jj
            spec["W_up"][u, src_nib_base + nib] = RELU_S
            spec["b_up"][u] = -RELU_S * t
            spec["W_gate"][u, L.ONE] = 1.0
    # self-clear EVERY queried QRY_BIN[b] (SET 0) across the full CAM address width.
    clear0 = n_nib * n_thr
    for b in range(clear_bits):
        uu = clear0 + b
        spec["W_up"][uu, L.ONE] = S
        spec["W_gate"][uu, bin_base + b] = 1.0
        spec["W_down"][bin_base + b, uu] += -1.0 / SILU_S
    # bit-select: QRY_BIN[nib*4+k] += onehot(nib==v) for every v with (v>>k)&1.
    # onehot(nib==v) via the triangular pulse over thresholds v-1,v,v+1 (+1,-2,+1).
    for nib in range(n_nib):
        base_u = nib * n_thr
        tu = {t: base_u + jj for jj, t in enumerate(THR)}
        for k in range(4):
            b = nib * 4 + k
            if b >= n_bits:
                break
            for v in range(16):
                if (v >> k) & 1:
                    spec["W_down"][bin_base + b, tu[v - 1]] += 1.0 / RELU_S
                    spec["W_down"][bin_base + b, tu[v]] += -2.0 / RELU_S
                    spec["W_down"][bin_base + b, tu[v + 1]] += 1.0 / RELU_S
    return spec


def _compile_addr_expand_scalar(L, src_lane: int, bin_base: int, dim: int,
                                n_bits: int, clear_bits: int
                                ) -> Dict[str, torch.Tensor]:
    """Legacy 256-cell scalar one-hot expand (fallback when a scalar ``src_lane``
    has no backing nibble band). Kept for full caller compatibility; the live
    §Memory query uses the per-nibble path above."""
    thresholds = list(range(-1, 257))
    n_thr = len(thresholds)
    tu = {t: j for j, t in enumerate(thresholds)}
    n_units = n_thr + clear_bits       # relu bank + self-clear per (queried) bit
    spec = _empty_spec(dim, n_units)
    for t, j in tu.items():
        spec["W_up"][j, src_lane] = RELU_S
        spec["b_up"][j] = -RELU_S * t
        spec["W_gate"][j, L.ONE] = 1.0
    clear0 = n_thr
    for b in range(clear_bits):        # self-clear EVERY queried QRY_BIN[b] (SET 0)
        uu = clear0 + b
        spec["W_up"][uu, L.ONE] = S
        spec["W_gate"][uu, bin_base + b] = 1.0
        spec["W_down"][bin_base + b, uu] += -1.0 / SILU_S
    # one-hot cell a via the triangular pulse; add 1 to QRY_BIN[b] for each a with bit b.
    for a in range(256):
        for b in range(n_bits):
            if (a >> b) & 1:
                spec["W_down"][bin_base + b, tu[a - 1]] += 1.0 / RELU_S
                spec["W_down"][bin_base + b, tu[a]] += -2.0 / RELU_S
                spec["W_down"][bin_base + b, tu[a + 1]] += 1.0 / RELU_S
    return spec




def _bake_pf_memory_head(attn, L, head: int) -> None:
    """Bake the §Memory KV head on head ``head`` of ``attn`` (pure-forward layout).
    Same CAM as ``blogspec_memory.bake_memory_head`` but on THIS layout's bands and
    a chosen head index; writes the loaded value nibbles into the AX nibble band."""
    from .blogspec_memory import ADDR_BITS, EFF, BIAS, MEM_ALIBI_SLOPE, PEN_GATE
    from .blogspec_layout import NIB_PER_REG
    hs = attn.scale
    smag = (EFF / hs) ** 0.5
    qb = (BIAS / hs) ** 0.5
    kb = (BIAS / hs) ** 0.5
    # Role-gate penalty (must dominate the worst-case partial address match, so it
    # stays huge = 100·ADDR_BITS·EFF).  Robustness to a flag RESIDUE at a large PC
    # comes from THRESHOLDING the IS_LOAD flag clean (``_flag_from_ops`` step), not
    # from shrinking this gate — see the PEN_GATE note (frame-pointer read-back fix).
    PEN = PEN_GATE
    p = (PEN / hs) ** 0.5
    attn.alibi_slopes[head] = MEM_ALIBI_SLOPE
    HD = attn.head_dim
    base = head * HD
    for b in range(ADDR_BITS):
        attn.W_k[base + b, L.ADDR_BIN + b] = 2.0 * smag
        attn.W_k[base + b, L.ONE] = -smag
        attn.W_q[base + b, L.QRY_BIN + b] = 2.0 * smag
        attn.W_q[base + b, L.ONE] = -smag
    cB = base + ADDR_BITS
    attn.W_q[cB, L.IS_LOAD] = -qb
    attn.W_k[cB, L.IS_STORE] = kb
    cR = base + ADDR_BITS + 1
    attn.W_q[cR, L.IS_LOAD] = p
    attn.W_k[cR, L.ONE] = -p
    attn.W_k[cR, L.IS_STORE] = p
    # channel n+2: LOAD-ENABLE. A NON-load query (IS_LOAD=0) must attend to nothing
    # (the softmax1 sink) so the CAM contributes 0 on non-memory steps. KEY = -c·ONE
    # on every row; QUERY = c·(ONE - IS_LOAD). For IS_LOAD=1 the query is 0 (no
    # penalty); for IS_LOAD=0 the query is c·ONE so EVERY candidate scores -c²·hs
    # (far below the sink) ⇒ the head outputs ~0 on every non-load step.
    cL = base + ADDR_BITS + 2
    c = (PEN / hs) ** 0.5
    attn.W_q[cL, L.ONE] = c
    attn.W_q[cL, L.IS_LOAD] = -c
    attn.W_k[cL, L.ONE] = -c
    for j in range(NIB_PER_REG):
        attn.W_v[base + ADDR_BITS + 3 + j, L.VAL_NIB + j] = 1.0
        attn.W_o[L.AX + j, base + ADDR_BITS + 3 + j] = 1.0


# ---------------------------------------------------------------------------
# LI / SI dispatch on the value lanes (§dispatch interface). The memory EFFECT
# (the CAM read for LI, the store-token emit for SI) happens via the KV head +
# the driver's overlay; here we only do the register/PC/SP housekeeping.
# ---------------------------------------------------------------------------
def memory_dispatch_rules(L) -> List:
    """LI: AX := loaded value (written by the memory CAM into the AX nibble band;
    the value-lane AX_VAL is refreshed by the mem-cam block's recompose FFN, so the
    LI expert must NOT overwrite AX_VAL) — only PC += 1. SI: PC += 1, SP += 4 (pop
    consumed the address); the store addr/val are laid into the emitted MEM token by
    the driver (a store frame), becoming a KV entry. IS_LOAD / IS_STORE flags are
    set by the driver on the query/store rows."""
    from .dsl import FFNRule, LinearExpr
    pc, sp = L.PC_VAL, L.SP_VAL

    def G(op):
        return [(L.OP_IS + op, 0.5, 1.5)]

    rules = []
    rules.append(FFNRule(G(isa.LI), {pc: LinearExpr.c(1.0)}))       # AX from CAM
    rules.append(FFNRule(G(isa.LC), {pc: LinearExpr.c(1.0)}))
    rules.append(FFNRule(G(isa.SI), {pc: LinearExpr.c(1.0), sp: LinearExpr.c(4.0)}))
    rules.append(FFNRule(G(isa.SC), {pc: LinearExpr.c(1.0), sp: LinearExpr.c(4.0)}))
    return rules


# ---------------------------------------------------------------------------
# COMPARISONS (EQ/NE/LT/GT/LE/GE) — the §Comparisons zero-detector + sign-of-diff
# on the byte value lanes, as SwiGLU weights. Split into (a) an UNGATED compute of
# the 3 primitives CMP_EQ/CMP_GT/CMP_LT from d = STK - AX (runs every step), and
# (b) an opcode-gated WRITE of the boolean into AX_VAL (product units: contributes
# OP_IS[op]·result). This keeps the arbitrary-d gadget out of the per-op gate.
# ---------------------------------------------------------------------------
CMP_OPS = [isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE]


def compile_cmp_compute(L, dim: int) -> Dict[str, torch.Tensor]:
    """Ungated comparison PRIMITIVES from STK_VAL / AX_VAL each step (a companion
    ``compile_cmp_signed_finalize`` block turns them into the SIGNED verdict):

      CMP_EQ = (STK == AX)                       (sign-agnostic; final)
      MAG_GT = (STK > AX)   as UNSIGNED magnitudes (raw ramp)
      MAG_LT = (STK < AX)   as UNSIGNED magnitudes (raw ramp)
      SGN_STK / SGN_AX = the operands' 32-bit sign bit (bit 31)

    The value lanes hold the UNSIGNED magnitude of the two's-complement WORD
    (nibbles recompose to Σ 16^j·nib_j ≥ 0), so the magnitude difference
    ``d = STK_VAL - AX_VAL`` orders the operands correctly ONLY when they share a
    sign.  C4's ``int`` is 32-bit and LT/GT/LE/GE are SIGNED; the finalize block
    reconstructs the signed verdict from these primitives:

        signed_GT = clamp01(MAG_GT) + (SGN_AX - SGN_STK)
        signed_LT = clamp01(MAG_LT) + (SGN_STK - SGN_AX)

    (derivation: the naive ``same_sign·mag ± cross_sign`` products cancel to this
    linear correction, which flips exactly the two cross-sign cases and leaves
    same-sign pairs unchanged; the result is always exactly 0 or 1).

    ``sign(v)`` = clamped ``step(nib7 >= 8)`` on the value's TOP NIBBLE (nibble 7 of
    the register nibble band), written to its OWN ``SGN_*`` dim.  The nibble is a
    SMALL operand (0..15), so the step's threshold (``7.5``) and silu intermediates
    are tiny and fp-EXACT — reading the sign from the ~2^31 recomposed SCALAR instead
    would need a ``-RELU_S·(2^31-0.5)`` bias that is not fp32-representable (the specs
    are built fp32, then cast fp64), corrupting the threshold; the nibble sidesteps
    that.  ``MAG_GT``/``MAG_LT`` are the raw ``step(d)`` ramps and can read NOISY when
    ``|d| ≈ 2^32`` (a cross-sign pair, where two ~2^32-scale relus cancel imperfectly
    under the recompose's ~4-unit fp error) — the finalize block ``clamp01``s them
    (relu(m) - relu(m-1)) so a noisy ~95 collapses to the correct unsigned 1, then
    applies the sign correction.

    Under the DEFAULT 8-bit fold the recompose/ingest carry only nibbles 0..4, so
    nibble 7 is always 0, ``SGN_*`` are 0, ``MAG_*`` never exceed the 8-bit range (no
    noise), and the finalize's ``clamp01(MAG) + 0`` is BYTE-IDENTICAL to the pre-fix
    unsigned ``CMP_GT``/``CMP_LT`` (no 8-bit-corpus regression).  Under
    ``C4_VM_WIDTH32`` a genuine two's-complement negative has nibble 7 >= 8, so the
    correction fires and LT/GT/LE/GE become gcc-exact signed."""
    SIGN_NIB = 7                         # top nibble of the 32-bit word (bit 28..31)
    stk_sign_nib = L.STACK0 + SIGN_NIB   # STK top nibble (>= 8 iff STK negative)
    ax_sign_nib = L.AX + SIGN_NIB        # AX top nibble  (>= 8 iff AX  negative)
    W = 0.2                              # clamp-ramp width (matches compile_fold)
    # 5 clears + EQ(3) + MAG_GT(2) + MAG_LT(2) + SGN_STK(2) + SGN_AX(2)
    spec = _empty_spec(dim, 5 + 3 + 2 + 2 + 2 + 2)
    u = 0
    for lane in (L.CMP_EQ, L.MAG_GT, L.MAG_LT, L.SGN_STK, L.SGN_AX):  # self-clear
        spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, lane] = 1.0
        spec["W_down"][lane, u] += -1.0 / SILU_S; u += 1
    # EQ = Z(d): the §584 finite-2nd-difference silu bump (peak 1 at d==0).
    sig = float(torch.sigmoid(torch.tensor(S * 0.5)))
    k = S * 0.5 * (2.0 * sig - 1.0)
    for bias, w in [(S * 0.5, 1.0), (0.0, -2.0), (-S * 0.5, 1.0)]:
        spec["W_up"][u, L.STK_VAL] += S; spec["W_up"][u, L.AX_VAL] += -S
        spec["b_up"][u] += bias
        spec["W_gate"][u, L.ONE] = 1.0
        spec["W_down"][L.CMP_EQ, u] += w / k; u += 1
    # MAG_GT = step(d >= 1) = relu(d) - relu(d-1)  (integer d), UNSIGNED.
    for idx, thr in enumerate((0.0, 1.0)):
        spec["W_up"][u, L.STK_VAL] += RELU_S; spec["W_up"][u, L.AX_VAL] += -RELU_S
        spec["b_up"][u] += -RELU_S * thr; spec["W_gate"][u, L.ONE] = 1.0
        spec["W_down"][L.MAG_GT, u] += (1.0 if idx == 0 else -1.0) / RELU_S; u += 1
    # MAG_LT = step(-d >= 1), UNSIGNED.
    for idx, thr in enumerate((0.0, 1.0)):
        spec["W_up"][u, L.AX_VAL] += RELU_S; spec["W_up"][u, L.STK_VAL] += -RELU_S
        spec["b_up"][u] += -RELU_S * thr; spec["W_gate"][u, L.ONE] = 1.0
        spec["W_down"][L.MAG_LT, u] += (1.0 if idx == 0 else -1.0) / RELU_S; u += 1
    # SGN_STK / SGN_AX = clamped step(nib7 >= 8) on the SMALL top nibble, each into
    # its OWN dim (fp-exact: tiny threshold, no 2^31-scale bias).
    lo = 7.5
    for sign_nib, sgn_lane in ((stk_sign_nib, L.SGN_STK), (ax_sign_nib, L.SGN_AX)):
        for idx, thr in enumerate((lo, lo + W)):
            spec["W_up"][u, sign_nib] += RELU_S
            spec["b_up"][u] += -RELU_S * thr
            spec["W_gate"][u, L.ONE] = 1.0
            spec["W_down"][sgn_lane, u] += (1.0 if idx == 0 else -1.0) / (RELU_S * W)
            u += 1
    return spec


def compile_cmp_signed_finalize(L, dim: int) -> Dict[str, torch.Tensor]:
    """Combine the ``compile_cmp_compute`` primitives into the SIGNED verdict:

        CMP_GT = clamp01(MAG_GT) + (SGN_AX - SGN_STK)
        CMP_LT = clamp01(MAG_LT) + (SGN_STK - SGN_AX)

    ``clamp01(m) = relu(m) - relu(m-1)`` collapses a NOISY cross-sign magnitude
    ramp (~95, from the ~2^32-scale relu cancellation) to the correct unsigned 0/1
    while leaving a clean same-sign 0/1 unchanged; the ``SGN_AX - SGN_STK`` term is
    0 for same-sign pairs and ±1 across a sign boundary, so the sum is exactly the
    signed verdict (0 or 1).  SET (self-clears CMP_GT/CMP_LT first)."""
    spec = _empty_spec(dim, 2 + 2 + 2 + 2 + 2)  # 2 clears + clamp_GT(2)+clamp_LT(2)+sgn(2+2)
    u = 0
    for lane in (L.CMP_GT, L.CMP_LT):           # self-clear (SET)
        spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, lane] = 1.0
        spec["W_down"][lane, u] += -1.0 / SILU_S; u += 1
    # clamp01(MAG_GT) = relu(MAG_GT) - relu(MAG_GT - 1)  -> CMP_GT
    for dst, mag in ((L.CMP_GT, L.MAG_GT), (L.CMP_LT, L.MAG_LT)):
        for idx, thr in enumerate((0.0, 1.0)):
            spec["W_up"][u, mag] += RELU_S; spec["b_up"][u] += -RELU_S * thr
            spec["W_gate"][u, L.ONE] = 1.0
            spec["W_down"][dst, u] += (1.0 if idx == 0 else -1.0) / RELU_S; u += 1
    # sign correction: +SGN_AX - SGN_STK -> CMP_GT ; +SGN_STK - SGN_AX -> CMP_LT.
    # SGN_* are already clean 0/1, so a silu-identity read routes them verbatim.
    for sgn, gt_c, lt_c in ((L.SGN_AX, +1.0, -1.0), (L.SGN_STK, -1.0, +1.0)):
        spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, sgn] = 1.0
        spec["W_down"][L.CMP_GT, u] += gt_c / SILU_S
        spec["W_down"][L.CMP_LT, u] += lt_c / SILU_S
        u += 1
    return spec


def cmp_dispatch_rules(L) -> List:
    """The opcode-gated CMP write: AX := boolean (SET) + PC+=1, SP+=4. The boolean
    is a linear combo of the CMP_EQ/GT/LT scratch lanes:
      EQ=CMP_EQ  NE=1-CMP_EQ  LT=CMP_LT  GT=CMP_GT  LE=1-CMP_GT  GE=1-CMP_LT."""
    from .dsl import FFNRule, LinearExpr
    ax, sp, pc = L.AX_VAL, L.SP_VAL, L.PC_VAL

    def G(op):
        return [(L.OP_IS + op, 0.5, 1.5)]

    bool_of = {
        isa.EQ: LinearExpr.of(L.CMP_EQ, 1.0),
        isa.NE: LinearExpr.c(1.0) + LinearExpr.of(L.CMP_EQ, -1.0),
        isa.LT: LinearExpr.of(L.CMP_LT, 1.0),
        isa.GT: LinearExpr.of(L.CMP_GT, 1.0),
        isa.LE: LinearExpr.c(1.0) + LinearExpr.of(L.CMP_GT, -1.0),
        isa.GE: LinearExpr.c(1.0) + LinearExpr.of(L.CMP_LT, -1.0),
    }
    rules = []
    for op in CMP_OPS:
        rules.append(FFNRule(G(op), {
            ax: bool_of[op] + LinearExpr.of(ax, -1.0),   # SET AX = boolean
            pc: LinearExpr.c(1.0), sp: LinearExpr.c(4.0)}))
    return rules


# (``muldiv_dispatch_rules`` + the 8-bit MUL/DIV/MOD lookup table it dispatched have
# been removed — the table is gone; MUL/DIV/MOD run only through the efficient
# ``nibble_alu32`` ALU in the Qwen build.)


BITWISE_OPS = [isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR]


def bitwise_dispatch_rules(L) -> List:
    """OR/XOR/AND/SHL/SHR housekeeping: PC += 1 ; SP += 4 (pop consumed). The RESULT
    is written to the AX nibble band by the bw-select block and recomposed to AX_VAL
    by bw-recompose; the dispatch only does the PC/SP update (like the unified
    model's bitwise experts)."""
    from .dsl import FFNRule, LinearExpr
    pc, sp = L.PC_VAL, L.SP_VAL
    rules = []
    for op in BITWISE_OPS:
        rules.append(FFNRule([(L.OP_IS + op, 0.5, 1.5)],
                             {pc: LinearExpr.c(1.0), sp: LinearExpr.c(4.0)}))
    return rules


def build_pure_forward_model(code_size: int = 32, include_memory: bool = True,
                             include_cmp: bool = True, include_bitwise: bool = True,
                             include_muldiv: bool = False):
    """Assemble the pure-forward VM: block-0 attention = the frame-ingest CAM, an
    optional §Memory KV head, and the SAME baked step FFN blocks as
    ``nibble_vm.build_step_model`` (recompose / fetch / code-select / decode /
    dispatch / branch / fold), with LI/SI dispatch when ``include_memory``.

    ``n_heads = N_ROLES (+1 for the memory head)`` — one gather head per register
    byte plus the memory CAM. ``head_dim`` is forced ≥ ``MEM_HEAD_CHANNELS`` (via
    dim padding) so the memory head's 50 local channels fit.

    Returns ``(model, L)``. The op result is computed by the FFN weights inside
    ``model.forward``; the register state + memory are reconstructed from the token
    stream by attention. Nothing is computed in Python.
    """
    n_heads = N_ROLES + (1 if include_memory else 0)
    # SUBSET-AWARE: skip the memory / cmp residual bands a build does not bake (a base
    # arith/func build no longer carries the dead dims of the unused op families).
    # ``include_bitwise`` extends the layout separately below.  ``include_muldiv`` is a
    # legacy no-op (the lookup table is gone; MUL/DIV/MOD is efficient-ALU only).
    L = PureForwardLayout(code_size, n_heads=n_heads,
                          include_memory=include_memory, include_cmp=include_cmp,
                          include_muldiv=include_muldiv)
    _wide = ingest_wide_enabled()
    if _wide:
        # 80 fresh dims (PREROUTE 40 + GATHER 40) for the 1-query/1-KV wide ingest.
        # Allocated BEFORE dim is fixed; they fit in the d_model headroom.  Flag-OFF
        # this is never called ⇒ the golden layout / hash is unchanged.  Re-fix L.D
        # (pad to n_heads) so ``dim`` below includes the wide bands even when no other
        # layout extension (bitwise) runs after this.
        extend_layout_for_wide_ingest(L)
        while L._off % n_heads != 0:
            L._scalar(f"_widepad{L._off}")
        L.D = L._off
    if include_bitwise:
        from . import nibble_bitwise as _bw
        _bw.extend_layout_for_bitwise(L)          # A_OH/B_OH/SHIFT_* bands
        # SHIFTER scratch: allocate the active shifter's private per-op scratch bands
        # NOW, before ``dim`` is fixed below, so the shift blocks (compiled inside
        # build_bitwise_blocks at the fixed ``dim``) address valid dims.  BARREL
        # (C4_BARREL_SHIFT=1) takes precedence over the TIGHT shifter (default ON).
        if _bw.barrel_shift_enabled():
            for _op in (isa.SHL, isa.SHR):
                _bw.extend_layout_for_barrel_shift(L, _op)
        elif _bw.tight_shift_enabled():
            for _op in (isa.SHL, isa.SHR):
                _bw.extend_layout_for_tight_shift(L, _op)
        while L._off % n_heads != 0:
            L._scalar(f"_bwpad{L._off}")
        L.D = L._off
    # force head_dim >= MEM_HEAD_CHANNELS by padding dim up to a multiple of n_heads.
    min_dim = n_heads * MEM_HEAD_CHANNELS if include_memory else L.D
    if L.D < min_dim:
        target = -(-min_dim // n_heads) * n_heads
        while L._off < target:
            L._scalar(f"_hdpad{L._off}")
        L.D = L._off
    dim = L.D
    # Block order. The memory read has an intra-forward data dependency: the load
    # ADDRESS is the ingested AX, and the CAM must know it is a LOAD. So fetch+decode
    # run FIRST (to know the opcode), THEN a mem-prep FFN sets IS_LOAD from OP_IS[LI/
    # LC], expands the ingested AX_VAL -> QRY_BIN, and clears the AX nibble band on a
    # load; THEN the mem-cam block's ATTENTION runs the §Memory CAM (writes the loaded
    # value nibbles into the cleared AX band); a recompose refreshes AX_VAL; dispatch
    # does the PC/SP housekeeping. All inside ONE model.forward.
    reg_bases = {"PC": L.PC, "AX": L.AX, "SP": L.SP, "BP": L.BP, "STACK0": L.STACK0}
    block_specs = []
    if _wide:
        # WIDE INGEST (C4_INGEST_WIDE): replace the 20/21-head role-CAM ingest with a
        # SINGLE query + SINGLE KV head.  Three blocks IN FRONT of the stock block 0:
        #   wide-preroute : attn zeroed, FFN = the ROLE⊙CUR_NIB per-role gate (A).
        #   wide-gather   : attn = the 1-head wide gather (B, baked below); FFN = the
        #                   1/wtot rescale GATHER -> reg nibble band (C).
        #   wide-snap     : attn zeroed, FFN = integer re-quant of the reg nibbles so
        #                   the recompose's 16^j amplification of a ~1e-8 gather residue
        #                   can't shift PC_VAL/AX_VAL off their integer (opcode-decode).
        # The stock "ingest+recompose" block then keeps its recompose FFN but its attn
        # is ZEROED (the gather moved to wide-gather).
        block_specs += [
            ("wide-preroute", compile_wide_preroute(L, dim)),
            ("wide-gather", compile_wide_rescale(L, reg_bases, dim)),
            ("wide-snap", compile_wide_nibble_snap(L, reg_bases, dim)),
        ]
    block_specs += [
        ("ingest+recompose", compile_nibble_to_scalar(L, dim)),   # block 0 FFN
        ("pc-fetch",    compile_pc_fetch(L, dim)),
        ("code-select", compile_code_select(L, dim)),
        ("opcode-decode", compile_opcode_decode_pf(L, dim)),
    ]
    if include_memory:
        block_specs += [
            ("mem-prep", compile_mem_prep(L, dim)),               # IS_LOAD + QRY_BIN + clear AX
            ("mem-cam",  compile_nibble_to_scalar(L, dim)),       # ATTN=CAM; FFN: refresh AX_VAL
        ]
    if include_cmp:
        block_specs += [
            ("cmp-compute", compile_cmp_compute(L, dim)),        # MAG/SGN primitives
            ("cmp-finalize", compile_cmp_signed_finalize(L, dim)),  # signed CMP_GT/LT
        ]
    if include_bitwise:
        from .nibble_unified import build_bitwise_blocks, _bw_recompose_spec
        for name, spec in build_bitwise_blocks(L, dim):     # bw-expand/bit4/select
            block_specs.append((name, spec))
        block_specs.append(("bw-recompose", _bw_recompose_spec(
            L, dim, (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR))))  # AX nibbles -> AX_VAL
    disp_rules = base_dispatch_rules(L)
    if include_memory:
        disp_rules = disp_rules + memory_dispatch_rules(L)
    if include_cmp:
        disp_rules = disp_rules + cmp_dispatch_rules(L)
    if include_bitwise:
        disp_rules = disp_rules + bitwise_dispatch_rules(L)     # PC+1 ; SP+4
    block_specs += [
        ("dispatch", compile_ffn(disp_rules, dim)),
        ("branch-delta", compile_branch_delta(L, dim)),
        ("fold", compile_fold(L.AX_VAL, L.ONE, dim, modulus=256)),
    ]
    n_blocks = len(block_specs)
    hidden = max(f["W_up"].shape[0] for _, f in block_specs)
    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, vocab=V.VOCAB, max_seq_len=8192)
    with torch.no_grad():
        _bake_pure_embedding(model, L)
        for bi, (name, spec) in enumerate(block_specs):
            _zero_attn(model.blocks[bi].attn)
            _load_ffn(model.blocks[bi].ffn, spec, hidden)
        if _wide:
            # WIDE INGEST: swap the "wide-gather" block's Attn for a 1-head Attn
            # (head_dim = dim) and bake the single wide gather head.  The stock
            # "ingest+recompose" block's attn stays ZEROED (gather moved to wide-
            # gather).  Swapping ONLY this block's Attn leaves every OTHER block's
            # multi-head Attn (mem-cam, etc.) byte-identical.
            gi = [i for i, (nm, _) in enumerate(block_specs) if nm == "wide-gather"][0]
            model.blocks[gi].attn = _make_one_head_attn(dim, model.max_seq_len)
            bake_wide_ingest_head(model.blocks[gi].attn, L)
        else:
            # block 0 attention = the frame-ingest CAM.  C4_INGEST_GQA (default OFF)
            # swaps in the 1-KV-head GQA bake (byte-exact, 20 KV heads → 1).
            _ingest = (bake_frame_ingest_gqa if ingest_gqa_enabled()
                       else bake_frame_ingest)
            _ingest(model.blocks[0].attn, L, reg_bases)
        if include_memory:
            # the §Memory KV head is the ATTENTION of the "mem-cam" block (index 2),
            # AFTER addr-expand has set QRY_BIN: it reads the load query, content-
            # addresses the store rows in the stream, and writes the loaded value
            # into the AX nibble band (its FFN then refreshes AX_VAL).
            mem_block = [i for i, (nm, _) in enumerate(block_specs) if nm == "mem-cam"][0]
            _bake_pf_memory_head(model.blocks[mem_block].attn, L, head=N_ROLES)
    L._block_names = [n for n, _ in block_specs]
    return model, L


def _bake_pure_embedding(model, L: PureForwardLayout) -> None:
    """Byte tokens embed their two nibbles into CUR_NIB; ONE=1 in every row. The
    ROLE / IS_FRAME_BYTE tags are set by the driver's per-position frame overlay
    (structural frame-slot tags), so the embedding table itself stays universal."""
    E = torch.zeros(V.VOCAB, model.dim)
    E[:, L.ONE] = 1.0
    for b in range(256):
        lo, hi = V.nibbles_of_byte(b)
        E[b, L.CUR_NIB + 0] = float(lo)
        E[b, L.CUR_NIB + 1] = float(hi)
    model.embed.copy_(E)


# ===========================================================================
# The token stream + the per-position overlay (program-in-data + frame roles).
#
# The stream is BOS then one 30-token frame per emitted step. The overlay carries
# two structural things (NOT computed VM state): (1) the PROGRAM in the DATA bands
# on the BOS position (universal fetch — the code is INPUT), and (2) the per-frame
# ROLE / IS_FRAME_BYTE tags of each byte token (a rigid function of frame slot,
# like a positional encoding). The register VALUES ride in the byte-token
# embeddings (CUR_NIB) — the real state, in the token stream.
# ===========================================================================
# Frame slot -> (register_index, byte_index) for the 20 register byte tokens.
# Frame layout: [REG_PC, pc0..3, REG_AX, ax0..3, REG_SP, sp0..3, REG_BP, bp0..3,
#                MEM, addr0..3, val0..3, STEP_END]  (STACK0 rides the MEM val slot)
_FRAME_ROLE_SLOTS = {}
def _init_frame_role_slots():
    # marker positions and their following 4 byte slots, in frame-local index.
    reg_marker_pos = {"PC": 0, "AX": 5, "SP": 10, "BP": 15}
    for r_idx, name in enumerate(["PC", "AX", "SP", "BP"]):
        m = reg_marker_pos[name]
        for bi in range(4):
            _FRAME_ROLE_SLOTS[m + 1 + bi] = r_idx * 4 + bi
    # STACK0 rides the MEM value bytes (frame slots 25..28).
    for bi in range(4):
        _FRAME_ROLE_SLOTS[25 + bi] = 4 * 4 + bi       # register index 4 = STACK0
_init_frame_role_slots()


def build_frame_tokens(pc: int, ax: int, sp: int, bp: int, stack0: int,
                       mem_addr: int = 0, mem_val: int = 0) -> List[int]:
    """The 30-token frame carrying the five registers; STACK0 in the MEM value
    slot (so the pure-forward stack-top mirror round-trips through the stream).
    On a STORE step the MEM slot instead carries the store's ``addr``/``val`` (the
    KV entry) — the store DATA rides in the emitted token stream, per §Memory."""
    if mem_addr or mem_val:
        return V.build_step_frame(pc, ax, sp, bp, mem_addr=mem_addr, mem_val=mem_val)
    return V.build_step_frame(pc, ax, sp, bp, mem_addr=0, mem_val=stack0 & 0xFFFFFFFF)


# frame-local index of the MEM marker and its addr/val byte slots.
_MEM_MARKER_LOCAL = 20
_MEM_ADDR_LOCAL = [21, 22, 23, 24]
_MEM_VAL_LOCAL = [25, 26, 27, 28]


def _lay_store_row(x, L, p: int, addr: int, val: int) -> None:
    """Tag stream position ``p`` as a §Memory store KV row: IS_STORE=1, the token
    dropped from the ingest (IS_FRAME_BYTE=0), ADDR_BIN expanded from ``addr`` and
    VAL_NIB from ``val``.  A ``val==0`` row is a ZFOD TOMBSTONE (a vanilla store of
    0 — the model op the blog spec's free() performs, §689-691).  The CAM keys only
    on these overlay bands (independent of the token id at ``p``), so a store row
    can ride ANY position — this is what lets one step carry several store rows."""
    x[0, p, L.IS_STORE] = 1.0
    x[0, p, L.IS_FRAME_BYTE] = 0.0
    for b, bit in enumerate(_address_bits(addr)):
        x[0, p, L.ADDR_BIN + b] = bit
    for j, nv in enumerate(V.nibbles_of_value(val & 0xFFFFFFFF, 16)):
        x[0, p, L.VAL_NIB + j] = float(nv)


def make_overlay(code: List[isa.Instr], L: PureForwardLayout, store_frames=None,
                 frame_spare_stores=None):
    """Return an ``overlay(x)`` that writes, in-place on the embedded stream ``x``
    ([1,S,D]): the PROGRAM into the DATA bands at every position (so fetch@PC works
    at the last position), and the ROLE / IS_FRAME_BYTE frame-slot tags on each
    30-token frame. Everything here is structural (program = input; roles = frame
    layout).

    ``store_frames`` (optional set of frame indices, 0 = the init frame) marks which
    emitted frames were STORE steps: their MEM token is turned into a KV entry —
    ``IS_STORE=1`` + ``ADDR_BIN`` expanded from the frame's MEM addr bytes +
    ``VAL_NIB`` from the MEM val bytes. This is the §Memory store log riding in the
    emitted MEM tokens; marking which MEM token is a store is the driver's routing
    bookkeeping (it fetched the op), exactly the ``KVMemory`` contract.

    ``frame_spare_stores`` (optional ``{frame_idx: {local_pos: (addr, val)}}``) lays
    EXTRA §Memory store rows on SPARE token positions of a frame — the VANILLA
    IN-STEP TOMBSTONE path (``C4_VANILLA_TOMBSTONE``).  Each ``(addr, 0)`` is a real
    zero-write tombstone freeing ``addr``; distributing N of them across the free /
    return step's spare positions frees an N-slot frame in ONE step (0 extra STEPS).
    ``None`` ⇒ byte-identical to the stock overlay (golden ``8f4dd780`` unchanged)."""
    store_frames = store_frames or set()
    frame_spare_stores = frame_spare_stores or {}

    def overlay(x: torch.Tensor) -> None:
        Sn = x.shape[1]
        for i in range(Sn):
            x[0, i, L.ONE] = 1.0
            for k, ins in enumerate(code):
                x[0, i, L.CODE_OP[k]] = float(ins.op)
                x[0, i, L.CODE_IMM[k]] = float(ins.imm)
        pos = 1
        frame_idx = 0
        while pos + V.FRAME_LEN <= Sn:
            for local, role in _FRAME_ROLE_SLOTS.items():
                p = pos + local
                x[0, p, L.ROLE + role] = 1.0
                x[0, p, L.IS_FRAME_BYTE] = 1.0
            # If this emitted frame was a STORE, make its MEM token a KV entry.
            if frame_idx in store_frames:
                mem_pos = pos + _MEM_MARKER_LOCAL
                addr = 0
                for bi, a in enumerate(_MEM_ADDR_LOCAL):
                    # decode the address byte from its two nibble dims via the
                    # vanilla LM-head argmax (``_snap_nib``), NOT a python round —
                    # ONE re-quant mechanism (the argmax) on the whole exec path.
                    byte = _snap_nib(float(x[0, pos + a, L.CUR_NIB + 0])) \
                        + (_snap_nib(float(x[0, pos + a, L.CUR_NIB + 1])) << 4)
                    addr |= byte << (8 * bi)
                x[0, mem_pos, L.IS_STORE] = 1.0
                x[0, mem_pos, L.IS_FRAME_BYTE] = 0.0     # the store token is not a role byte
                for b, bit in enumerate(_address_bits(addr)):
                    x[0, mem_pos, L.ADDR_BIN + b] = bit
                # value nibbles from the MEM val bytes (2 nibbles per byte token).
                for bi, vloc in enumerate(_MEM_VAL_LOCAL):
                    lo = int(x[0, pos + vloc, L.CUR_NIB + 0])
                    hi = int(x[0, pos + vloc, L.CUR_NIB + 1])
                    x[0, mem_pos, L.VAL_NIB + 2 * bi + 0] = float(lo)
                    x[0, mem_pos, L.VAL_NIB + 2 * bi + 1] = float(hi)
            # VANILLA IN-STEP TOMBSTONES: extra store rows on this frame's SPARE
            # positions (the free/return step's zero-writes, distributed across the
            # token budget the register updates leave unused).
            for local, (saddr, sval) in frame_spare_stores.get(frame_idx, {}).items():
                _lay_store_row(x, L, pos + local, saddr, sval)
            pos += V.FRAME_LEN
            frame_idx += 1
        for role in range(N_ROLES):
            x[0, -1, L.ROLE + role] = 1.0
    return overlay


def _address_bits(addr: int):
    from .blogspec_memory import ADDR_BITS
    return [float((addr >> b) & 1) for b in range(ADDR_BITS)]


# ===========================================================================
# THE PURE-FORWARD DRIVER — one VM step = one model.forward; argmax + append only.
# ===========================================================================
# Spec register init (§C4 Registers): PC=AX=0, SP=BP at the stack top, STACK0=0.
SP_INIT = 0x10000


def _emit_frame_from_state(state: torch.Tensor, L: PureForwardLayout,
                           is_store: bool = False, store_addr: int = 0,
                           store_val: int = 0) -> Tuple[List[int], int, int, bool]:
    """Decode the model's computed next-state (the value lanes at the last
    position) into the next 30-token frame via the LM byte-head's value argmax —
    the spec's own re-quantiser (no ``torch.round``). On a STORE step the MEM slot
    carries ``store_addr``/``store_val`` (the KV entry). Returns
    ``(frame_tokens, ax_value, next_pc, halted)``."""
    pc = _snap_lane(state[L.PC_VAL])
    ax = _snap_lane(state[L.AX_VAL])
    sp = _snap_lane(state[L.SP_VAL])
    bp = _snap_lane(state[L.BP_VAL])
    stk = _snap_lane(state[L.STK_VAL])
    halted = float(state[L.HALTED]) > 0.5
    if is_store:
        frame = build_frame_tokens(pc, ax, sp, bp, stk,
                                   mem_addr=store_addr, mem_val=store_val)
    else:
        frame = build_frame_tokens(pc, ax, sp, bp, stk)
    return frame, ax & 0xFF, pc, halted


def run_pure_forward(model, L: PureForwardLayout, code: List[isa.Instr],
                     max_steps: int = 512, verbose: bool = False,
                     collect_tokens: bool = False, fio=None, data_seg=None,
                     mask: int = 0xFF, frame_tombstones=None, report=None):
    """Execute ``code`` with the PURE-FORWARD step: every VM step is ONE
    ``model.forward`` over the growing token stream (state read from the prior
    frame by the block-0 attention; the op computed by the FFN weights), and the
    only Python is the LM value-argmax emit + append. Returns the per-step AX trace
    (matching ``isa.interpret``); with ``collect_tokens`` also the flat token
    stream.

    The stream starts ``[BOS] + init_frame`` where ``init_frame`` is the spec
    register init (PC=AX=0, SP=BP=0x10000, STACK0=0) — the "step 0" frame the first
    real step ingests. Each iteration appends exactly one 30-token frame.

    FILE OPS (OPEN/READ/CLOS/PRTF) — the ONE class NOT computed neurally (§Tool Use
    Mode).  When ``fio`` (a ``nibble_filesys.FileOpState``) is given, an op in
    ``FILE_OPCODES`` is dispatched via the TOOL_CALL protocol exactly as in the
    ``_complete`` driver: the runner marshals the args off the KV store log,
    performs the real I/O (a READ(fd=0) is served from the neural-stdin
    ``InputKVStream`` — the SAME path argv/stdin use, §"Reading Arguments"), and
    the integer result re-enters AX.  READ's bytes re-enter the token stream as
    their OWN §Memory KV store frames so a later LC reads them back byte-exact.
    ``data_seg`` (``{byte_addr: byte}``) seeds the read-only data segment.  When
    ``fio`` is None this argument is inert and the driver is byte-identical to the
    original.

    VANILLA IN-STEP TOMBSTONE (``C4_VANILLA_TOMBSTONE``): ``frame_tombstones`` maps
    ``emitted_frame_idx -> [free_addr, ...]`` — free addresses to tombstone (real
    zero-writes) on that step's SPARE token positions (see
    ``nibble_vanilla_tombstone``).  The driver distributes each frame's frees across
    the slack the register updates leave unused (0 extra STEPS); it records each
    zero-write in ``store_log`` so the cache manager / eviction schedule reclaims the
    zeroed rows.  ``None`` (or the flag OFF) ⇒ byte-identical to the stock driver."""
    from . import nibble_filesys as _FS
    from . import nibble_vanilla_tombstone as _TS
    frame_tombstones = frame_tombstones or {}
    _tomb_on = bool(frame_tombstones) and _TS.vanilla_tombstone_enabled()
    frame_spare_stores: Dict[int, Dict[int, Tuple[int, int]]] = {}
    tomb_spill: Dict[int, List[int]] = {}           # honest report of un-fit frees
    init_frame = build_frame_tokens(0, 0, SP_INIT, SP_INIT, 0)
    stream: List[int] = [V.BOS] + init_frame
    trace: List[int] = []
    store_frames = set()                            # emitted frame indices that are stores
    store_log: Dict[int, Tuple[int, int]] = {}      # frame_idx -> (addr, val) for file marshalling
    cur_pc = 0                                       # PC of the step about to run
    cur_sp = SP_INIT
    prev_regs = {"PC": 0, "AX": 0, "SP": SP_INIT, "BP": SP_INIT, "STACK0": 0}
    frame_idx = 0                                    # emitted-frame counter (0 = init)
    for _ in range(max_steps):
        # the overlay marks past store frames (KV entries) + program + roles + the
        # in-step tombstones distributed onto spare frame positions (VANILLA free).
        overlay = make_overlay(code, L, store_frames=store_frames,
                               frame_spare_stores=(frame_spare_stores or None))
        toks = torch.tensor([stream])
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)                              # program-in-data + frame roles + KV
            for blk in model.blocks:                # == model.forward minus LM head
                x = blk(x)
        state = x[0, -1]
        # Is the step that just ran a STORE? (the driver fetches the op at cur_pc —
        # the same code-as-data fetch the model does; this is routing bookkeeping.)
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        imm = code[cur_pc].imm if 0 <= cur_pc < len(code) else 0
        sp = _snap_lane(state[L.SP_VAL])
        # -- FILE OP: not computed neurally (§Tool Use Mode).  The driver performs
        #    the whole op via the TOOL_CALL runner and overrides the registers;
        #    READ's bytes re-enter as their own §Memory KV frames (LC reads back).
        if fio is not None and op in _FS.FILE_OPCODES:
            cur_ax = _snap_lane(state[L.AX_VAL])
            new_ax, new_sp, byte_stores = _FS.dispatch_file_op_driver(
                op, cur_ax & 0xFFFFFFFF, imm, cur_sp, store_log, fio,
                data_seg=data_seg, slot=4)
            npc = cur_pc + 1                         # file ops advance PC by one
            frame = build_frame_tokens(npc, new_ax & 0xFFFFFFFF, new_sp, SP_INIT, 0)
            trace.append(new_ax & mask)
            frame_idx += 1
            stream += frame
            for (baddr, bval) in byte_stores:       # READ bytes -> KV store frames
                bframe = build_frame_tokens(npc, new_ax & 0xFFFFFFFF, new_sp,
                                            SP_INIT, 0, mem_addr=baddr,
                                            mem_val=bval & 0xFF)
                frame_idx += 1
                store_frames.add(frame_idx)
                store_log[frame_idx] = (baddr, bval & 0xFF)
                stream += bframe
            if verbose:
                print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op)} -> "
                      f"pc_next={npc} ax={new_ax & 0xFFFFFFFF} (FILE, "
                      f"{len(byte_stores)} byte-stores)")
            cur_pc, cur_sp = npc, new_sp
            if npc < 0 or npc >= len(code):
                break
            continue
        is_store = op in (isa.SI, isa.SC)
        s_addr = s_val = 0
        if is_store:
            # SI: *pop = AX. The store ADDRESS is the popped stack top (STK_VAL, the
            # STACK0 the model ingested) and the VALUE is AX — both model value lanes,
            # decoded by the same LM value-argmax as the registers. They ride in the
            # emitted frame's MEM addr/val slot (the §Memory write log token).
            s_addr = _snap_lane(state[L.STK_VAL])
            s_val = _snap_lane(state[L.AX_VAL]) & mask
        frame, ax_byte, npc, halted = _emit_frame_from_state(
            state, L, is_store=is_store, store_addr=s_addr, store_val=s_val)
        trace.append(ax_byte)
        frame_idx += 1
        # VANILLA IN-STEP TOMBSTONE: if frees are requested at this emitted frame,
        # distribute them across the frame's SPARE token positions (0 extra steps).
        # ``changed_regs`` = which registers moved this step -> which role-byte slots
        # are NOT slack (a store row there would drop the byte from ingest; unchanged
        # bytes fall back to the prior frame, so they ARE slack).
        if _tomb_on and frame_idx in frame_tombstones:
            cur_regs = {
                "PC": npc, "AX": _snap_lane(state[L.AX_VAL]) & 0xFFFFFFFF,
                "SP": sp, "BP": _snap_lane(state[L.BP_VAL]) & 0xFFFFFFFF,
                "STACK0": _snap_lane(state[L.STK_VAL]) & 0xFFFFFFFF,
            }
            changed = {r for r in cur_regs if cur_regs[r] != prev_regs.get(r)}
            # the frame's own MEM token is a live store this step -> reserve it.
            spare, spill = _TS.distribute_tombstones(
                list(frame_tombstones[frame_idx]), changed_regs=changed,
                reserve_mem=is_store)
            if spare:
                frame_spare_stores[frame_idx] = spare
                for _lp, (taddr, tval) in spare.items():
                    store_log[frame_idx] = (taddr, tval)   # last-write bookkeeping
            if spill:
                tomb_spill[frame_idx] = spill
            prev_regs = cur_regs
        else:
            prev_regs = {
                "PC": npc, "AX": _snap_lane(state[L.AX_VAL]) & 0xFFFFFFFF,
                "SP": sp, "BP": _snap_lane(state[L.BP_VAL]) & 0xFFFFFFFF,
                "STACK0": _snap_lane(state[L.STK_VAL]) & 0xFFFFFFFF,
            }
        if is_store:
            store_frames.add(frame_idx)             # this emitted frame is a KV entry
            store_log[frame_idx] = (s_addr, s_val)
        elif fio is not None and op == isa.PSH:
            # A PSH writes AX to MEM[cur_sp-4] — the c4 stack IS memory (§Stack).
            # The lean model handles the stack neurally via the STACK0 mirror, so a
            # PSH is not marked as a KV store frame; but file-op ARG MARSHALLING
            # (dispatch_file_op_driver) reads the pushed syscall args back off the
            # KV store log, so when a file op is in play we record the push there
            # (marshalling bookkeeping only — no effect on the neural stack path).
            store_log[frame_idx] = (cur_sp - 4, _snap_lane(state[L.AX_VAL]) & 0xFFFFFFFF)
        stream += frame                             # APPEND the emitted frame
        if verbose:
            print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op)} -> "
                  f"pc_next={npc} ax={ax_byte} store={is_store} halted={halted}")
        cur_pc = npc
        cur_sp = sp
        if halted or npc < 0 or npc >= len(code):
            break
    if report is not None:
        # honest tombstone report: which frame carried which zero-writes, and any
        # frees that did NOT fit the step's slack (spill = would need a next step).
        report["frame_spare_stores"] = frame_spare_stores
        report["tomb_spill"] = tomb_spill
        report["n_tombstones"] = sum(len(v) for v in frame_spare_stores.values())
        report["n_spill"] = sum(len(v) for v in tomb_spill.values())
    if collect_tokens:
        return trace, stream
    return trace


# ===========================================================================
# TRACE GUARD — prove NO python compute (no _apply_op / DictMemStack / gadget).
# ===========================================================================
_FORBIDDEN_QUALNAMES = {
    "_apply_op",                       # blogspec_run python if/elif dispatch
    "DictMemStack.__init__", "DictMemStack.store_int", "DictMemStack.load_int",
    "nibble_add_gadget", "nibble_sub_gadget",   # per-call ALU gadgets
    "mul32", "div32", "mod32",         # per-call muldiv gadgets
    "compare", "to_bit",               # per-call cmp gadget
    "or_gadget", "xor_gadget", "and_gadget", "shl_gadget", "shr_gadget",
}


class _NoPythonComputeGuard:
    """A settrace guard that raises if any forbidden compute-path function is
    entered while it is active — the machine proof that the VM ran purely in
    ``model.forward`` (no python if/elif dispatch, no python memory dict, no
    per-call gadget)."""

    def __init__(self):
        self.violations: List[str] = []
        self._prev = None

    def _tracer(self, frame, event, arg):
        if event == "call":
            name = frame.f_code.co_name
            qual = frame.f_code.co_qualname if hasattr(frame.f_code, "co_qualname") else name
            if name in _FORBIDDEN_QUALNAMES or qual in _FORBIDDEN_QUALNAMES or \
               any(qual.endswith("." + f) for f in _FORBIDDEN_QUALNAMES):
                self.violations.append(qual)
        return None                                 # do not trace lines (fast)

    def __enter__(self):
        self._prev = sys.gettrace()
        sys.settrace(self._tracer)
        return self

    def __exit__(self, *a):
        sys.settrace(self._prev)
        return False


def assert_no_python_compute(fn, *args, **kwargs):
    """Run ``fn`` under the guard; assert it entered NO forbidden compute-path
    function. Returns ``fn``'s result. This is the mission's make-or-break proof:
    the VM step ran entirely in ``model.forward``."""
    guard = _NoPythonComputeGuard()
    with guard:
        result = fn(*args, **kwargs)
    assert not guard.violations, \
        f"python compute leaked into the step: {sorted(set(guard.violations))}"
    return result
