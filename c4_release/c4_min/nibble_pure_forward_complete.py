"""PURE-FORWARD C4 VM — the LAST boundary closed: multi-slot stack via the KV head
+ the fp32-exact 32-bit ALU folded in.  EVERYTHING through the vanilla forward.

This extends ``nibble_pure_forward`` (the pure-forward VM: one VM step = one
``model.forward``, state round-tripping through the emitted 30-token frames, ALU /
dispatch / control in FFN weights, memory as softmax1-KV attention over the token
stream, argmax-generate-append the only Python on the compute path) to remove the
two remaining honest-boundary items the pure-forward deliverable named:

  1. **MULTI-SLOT STACK via the KV head.**  The pure-forward VM carried a single
     ``STACK0`` mirror slot, so ``PSH`` depth>1 (and JSR/ENT/LEV/ADJ) collided.
     Here the stack **is memory** (BLOG_SPEC §Memory / §Stack): ``SP``/``BP`` are
     byte addresses, and push/pop go through the **same softmax1-KV memory head**
     that already runs LI/SI byte-exact — a store keyed on the binary address, a
     load whose query is that address.

        PSH v      :  MEM[SP-4] = v ; SP -= 4                 (store @ new top)
        <consume>  :  STK = MEM[SP] ; op(STK, AX) ; SP += 4   (load @ top, then op)
        JSR t      :  MEM[SP-4] = pc+1 ; SP -= 4 ; PC = t
        ENT n      :  MEM[SP-4] = BP ; SP -= 4 ; BP = SP ; SP -= 4n
        ADJ n      :  SP += 4n
        LEV        :  SP=BP ; BP=MEM[SP] ; PC=MEM[SP+4] ; SP+=8
        LEA o      :  AX = BP + 4o                             (frame-relative addr)

     A push lays a store MEM token in the emitted frame (a KV entry keyed on the
     32-bit address, EXACTLY as SI does); a pop is a load whose QUERY is ``SP``
     instead of ``AX`` and whose retrieved value lands in the ``STACK0`` nibble
     band (the operand the ALU pops).  Depth-N stacks + the full calling
     convention run in-forward because each push is its own KV entry in the token
     stream and each pop content-addresses the newest write at that address.

  2. **32-bit ALU folded into the pure-forward model.**  The 8-bit MUL/DIV/MOD
     lookup table is replaced by ``nibble_alu32``'s persistent fp32 FFN blocks
     (ADD/SUB per-byte carry chain, MUL nibble schoolbook, DIV/MOD base-16 long
     division — all fp32-EXACT for the full 32-bit range, no lookup table, no
     fp64).  The op result is computed by ``model.forward`` on the register nibble
     bands and multiplexed into AX by an opcode-gated mux.

Everything runs through the vanilla autoregressive forward, proven byte-exact vs
:func:`ref_interpret` (the SP-addressed memory-stack reference) under the
``assert_no_python_compute`` guard (re-exported from ``nibble_pure_forward``).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import blogspec_vocab as V
from . import nibble_filesys as _FS   # OPEN/READ/CLOS/PRTF via the TOOL_CALL boundary
from .nibble_vm import (
    S, RELU_S, SILU_S, SILU_HALF,
    compile_nibble_to_scalar, compile_pc_fetch, compile_code_select,
    base_dispatch_rules, compile_branch_delta, compile_fold, compile_ffn,
    _empty_spec, _load_ffn, _zero_attn, _snap_lane, _recompose_hi_nibbles,
)
from .blogspec_model import Transformer
from . import nibble_pure_forward as PF
from .nibble_pure_forward import (
    PureForwardLayout, N_ROLES, bake_frame_ingest, _bake_pf_memory_head,
    compile_opcode_decode_pf, compile_cmp_compute, compile_cmp_signed_finalize,
    cmp_dispatch_rules,
    compile_mem_prep, memory_dispatch_rules, MEM_HEAD_CHANNELS,
    _address_bits, _FRAME_ROLE_SLOTS, _MEM_MARKER_LOCAL,
    _MEM_ADDR_LOCAL, _MEM_VAL_LOCAL, SP_INIT, _flag_from_ops, _concat_specs,
    ingest_wide_enabled, extend_layout_for_wide_ingest, compile_wide_preroute,
    compile_wide_rescale, compile_wide_nibble_snap, bake_wide_ingest_head,
    _make_one_head_attn,
)
from . import nibble_alu32 as A
from .dsl import FFNRule, LinearExpr
from .blogspec_memory import ADDR_BITS

# opcode value for ADJ / LC / SC / NOP (canonical C4 values; base isa lacks ADJ).
ADJ = isa.ADJ if hasattr(isa, "ADJ") else 7


def _unify_cam_one_enabled() -> bool:
    """PART A+ (``C4_UNIFY_CAM_ONE``, DEFAULT OFF): collapse the global address-CAM to
    a SINGLE head INDEX — fold the LEV return-PC read onto the SAME merged head.

    a17bf3b1 (``C4_UNIFY_CAM_HEAD``) merged the §Memory LI read + the stack-pop read
    into ONE head, leaving 2 global heads (merged + a dedicated LEV ret-PC head).  It
    concluded 2 is the floor because LEV reads ``MEM[BP]`` AND ``MEM[BP+4]`` in the
    same forward — two distinct addresses — and it tied both to the merged head's
    SINGLE query row.

    The per-query-position insight (the same mechanism as the wide-value ingest): a
    head has a query at EVERY row/block, and LEV's two reads land in DIFFERENT
    registers.  Reuse the SAME head INDEX in TWO cam BLOCKS: the stack-pop-cam block
    reads ``MEM[BP]`` (query=UNI_QRY_BIN=BP, dest=STACK0 -> BP) exactly as a17bf3b1
    does, and a SECOND cam block (``lev-cam2``) RE-FIRES the SAME head index with
    query=LEV_QRY_BIN (BP+4), enable=IS_LEV, dest=LEV_RET (the return PC).  The two
    reads are now SEQUENTIAL BLOCKS, not simultaneous — one softmax per (head, row)
    still holds — so the LEV return-PC head INDEX is DROPPED and the distinct
    global-CAM head count is 1 (n_heads N_ROLES+3 -> N_ROLES+2).  On a non-LEV step
    IS_LEV=0, so the second cam block's head is a pure softmax1 sink (contributes 0),
    exactly as the standalone LEV head was on non-LEV steps.

    BP+4 needs NO extra frame plumbing: LEV_QRY_BIN is already computed (pop-addr sets
    LEV_ADDR=BP, and the ``lev-addr4`` block ripple-carries +4 into LEV_QRY_BIN) for
    the a17bf3b1 build; ``C4_UNIFY_CAM_ONE`` reuses that exact band on the second block.

    Implies the ``C4_UNIFY_CAM_HEAD`` merged path (UNI_* bands, mux/demux).  DEFAULT
    OFF -> byte-IDENTICAL to golden (no UNI_* bands, n_heads unchanged)."""
    import os
    return os.environ.get("C4_UNIFY_CAM_ONE", "0") not in ("0", "", "false", "False")


def _unify_cam_head_enabled() -> bool:
    """PART A: unify the §Memory LI read into the stack-pop CAM head (``C4_UNIFY_CAM_HEAD``,
    DEFAULT OFF).

    The three global attention heads all run the SAME binary-address softmax1 CAM
    read (``_bake_cam_head``); the §Memory LI head (query=QRY_BIN, enable=IS_LOAD)
    and the stack-pop head (query=SP_QRY_BIN, enable=IS_POP) are MUTUALLY EXCLUSIVE
    per step (LI/LC ∉ POP_OPS ∪ {LEV}), so they can share ONE head slot with the
    query address MUXed by the opcode.  When ON: a mux FFN builds ``UNI_QRY_BIN``
    (= QRY_BIN on a load, SP_QRY_BIN on a pop/LEV) and ``IS_MEMREAD`` (load OR pop);
    the merged head at the stack-pop-cam block reads ``UNI_QRY_BIN`` -> ``UNI_VAL``,
    and a demux FFN copies ``UNI_VAL`` -> AX (loads) / STACK0 (pops/LEV).  The
    standalone mem-cam head is then DROPPED -> the live global-head count falls 3->2.
    (a17bf3b1 held that it cannot reach 1: LEV reads MEM[BP] AND MEM[BP+4] in the SAME
    forward.  ``C4_UNIFY_CAM_ONE`` refutes that by reusing the merged head INDEX in a
    second cam block for the BP+4 read — see ``_unify_cam_one_enabled``.)

    DEFAULT OFF -> layout, blocks and weights are byte-IDENTICAL to the golden
    3-head build; the golden/integer VM is unchanged with the gate off.
    ``C4_UNIFY_CAM_ONE`` IMPLIES this (it needs the UNI_* bands + mux/demux)."""
    import os
    if _unify_cam_one_enabled():
        return True
    return os.environ.get("C4_UNIFY_CAM_HEAD", "0") not in ("0", "", "false", "False")


def _bp_restore_hibyte_nibs() -> int:
    """``C4_BP_RESTORE_HIBYTE`` (DEFAULT ON): how many nibbles the LEV/pop STK_VAL /
    BP_VAL / LEV_RET_VAL scalar recomposes read.

    The frame-pointer-restore path (``compile_stk_recompose`` / ``compile_unify_stk_recompose``
    / ``compile_lev_ret_recompose``) recomposes the saved BP / return-PC popped from the
    KV store into its wide scalar lane as ``VAL = Σ_j 16^j·nibble_j``.  Historically it
    HARDCODED 8 nibbles — the ``C4_VM_WIDTH32`` fp64 count, where ``16^7`` is exact — but
    the doom/mandelbrot build runs the recompose in **fp32**, where ``16^7 = 2^28`` is far
    past fp32's 2^24 unit precision.  A saved BP like ``0x10000`` (SP_INIT) arrives with a
    clean nibble-4 but a ~1e-6 CAM-read RESIDUE on the high nibbles j=5,6,7 (which SHOULD
    be 0).  The 8-nibble recompose weights that residue by ``16^7 ≈ 2.7e8`` → a −256/−16/−1
    error, dragging ``0x10000`` down to ``0xFEEF`` (65536 → 65263, the mandelbrot boundary
    step-688 LEV divergence).  The residue-immune per-byte-argmax decode
    (``_decode_reg_from_nibbles``) is CORRECT on the SAME nibbles — only the wide-scalar
    recompose amplifies the residue.

    DEFAULT (ON): read ``_recompose_hi_nibbles()`` nibbles — the SAME fp32-safe count the
    base ``compile_nibble_to_scalar`` uses for EVERY other register (5 in fp32: coeff ≤
    16^4 = 2^16 < 2^24 fp32-exact; 8 in fp64/width32).  The dropped high nibbles j∈[5,7]
    carry only residue (a real saved-BP / return-PC in these programs fits bits 0..19), so
    the value is UNCHANGED for a clean value and residue-immune for a residue-laden one.
    This is the general-program-correctness default: MANDELBROT is fully byte-exact
    (interior/escape/boundary all match oracle) and DOOM stays byte-exact (its saved BPs
    decode clean at both 5 and 8 nibbles).  DEFAULT-ON golden ``_fingerprint_build`` =
    ``7d4afe61`` (fewer FFN units: the recompose reads 5 nibbles not 8, so each of the 3
    recompose ops drops 3 hi-nibble units).

    ESCAPE HATCH (``C4_BP_RESTORE_HIBYTE=0``): revert to the old ``hi_nibbles=8``
    recompose — the pre-fix golden ``069cc32f`` (byte-identical to the historical
    doom-build golden), for rollback + old-byte-identity checks."""
    import os
    if os.environ.get("C4_BP_RESTORE_HIBYTE", "1") in ("0", "false", "False"):
        return 8
    from .nibble_vm import _recompose_hi_nibbles
    return _recompose_hi_nibbles()

# Number of nibbles the STATIC immediate-nibble program encoding carries per slot.
# The corpus's largest VALUE literal is < 10^4 (< 16^4); 5 nibbles (< 16^5 ≈ 1.05M)
# is generous headroom while keeping the CODE_IMM_NIB band (code_size × IMM_NIBS)
# small (each extra nibble/slot inflates the model residual dim — the wall-time and
# memory driver).  A COMPUTED result (e.g. factorial) uses the full 8 AX nibbles via
# the ALU; only the IMM LITERAL is bounded here.
#
# ``C4_IMM_NIBS`` widens this to the FULL 32-bit literal (8 nibbles) for programs
# carrying a >20-bit / negative constant baked as an unsigned 32-bit word — e.g.
# doom's ``IMM -1`` (== 0xFFFFFFFF), which at 5 nibbles materializes as 0xFFFFF and
# corrupts the sign of every ``expr * -1`` / ``- term`` in its fixed-point math.
# DEFAULT 5 -> the corpus geometry + golden are byte-identical (every corpus literal
# is < 16^5, so the extra-nibble bands are all-zero anyway when widened).  Widening
# grows the CODE_IMM_NIB band (code_size × IMM_NIBS) and the residual dim; it is the
# cheapest correct route for doom's negative literals (one gated constant, no
# per-op weight rewrite — ``compile_imm_ax_nibbles`` already writes ``min(8, IMM_NIBS)``
# nibbles verbatim into AX).
import os as _os
IMM_NIBS = int(_os.environ.get("C4_IMM_NIBS", "5"))


# ===========================================================================
# CODE-FROM-MEMORY (C4_PF_CFM, DEFAULT OFF) — the program lives in the KV as
# address-keyed CODE frames, fetched at PC by a 12-bit-address softmax1 CAM,
# so the per-slot residual bands (CODE_OP[i]/CODE_IMM[i]/CODE_IMM_LO/HI[i]/
# PC_IS[i]/CODE_IMM_NIB[i]) are NOT baked — the residual dim becomes INDEPENDENT
# of code_size (the doom scale wall).  CRITICAL: the CAM value copies the code
# frame's op scalar -> OP_VAL AND its IMM_NIBS immediate NIBBLES -> IMM_NIB
# (nibble-wise, NOT a mod-256 scalar), so the pure-forward 20-bit-IMM
# materialization survives (literals > 255 / negative sign survive).  DEFAULT
# OFF -> the baked-table build is byte-identical (golden 069cc32f unchanged).
# ===========================================================================
# CODE_ADDR_BITS = number of code-address bits the softmax1 CODE CAM keys on (max
# addressable program = 2^CODE_ADDR_BITS instructions).  DEFAULT 12 (4096; doom =
# 3976 instrs < 2^12) keeps the historical build byte-identical (golden 069cc32f).
# WIDEN via ``C4_CODE_ADDR_BITS=<n>`` (mirrors ``C4_MEM_ADDR_BITS`` / wall #1) for
# programs with PC >> 4096 (e.g. a large Doom port).
#
# CEILING (this cfm path only): the code CAM head uses ALiBi slope 0.0 (see
# ``_bake_code_cam_head`` — the code address is UNIQUE per PC so no recency decay)
# and NO RoPE on its address lanes, so it is POSITION-INVARIANT by construction —
# a distant code frame scores the SAME as a near one (UNLIKE the RoPE
# ``qwen_full_vm._bake_code_cam``, whose fast rotary lanes DO cap the bits at a
# fidelity ceiling).  So the ONLY limits here are (a) the head_dim budget
# ``head_dim >= CODE_ADDR_BITS + 4 + (1+IMM_NIBS)`` (~48 bits at head_dim=58) and
# (b) the softmax1 discriminability margin (the ZFOD bias, scaled per width in
# ``_bake_code_cam_head``, must keep an exact match above the +1 sink and every
# 1-bit mismatch below it).  Measured BYTE-EXACT through PC 99,999 at
# ``C4_CODE_ADDR_BITS=20`` (see _agent_code_addr_bits.py).  Clamped by head_dim.
CODE_ADDR_BITS = int(_os.environ.get("C4_CODE_ADDR_BITS", "12"))


def _pf_cfm_enabled() -> bool:
    """``C4_PF_CFM`` (DEFAULT OFF): fetch the instruction at PC from an
    address-keyed §Memory CODE-frame CAM instead of the baked CODE_OP[i]/PC_IS[i]
    table, so the residual dim no longer scales with ``code_size``.  OFF -> the
    baked table build is byte-IDENTICAL to golden."""
    return _os.environ.get("C4_PF_CFM", "0") not in ("0", "", "false", "False")


def _cfm_emit_enabled() -> bool:
    """``C4_CFM_EMIT`` (DEFAULT OFF): the in-transformer JIT on the DIRECT-CAM code
    path.  The driver services ``isa.EMIT`` (value 45) as a control op (no FFN
    dispatch rule, like JSR/ENT/LEV): the model FETCHES the EMIT instruction from
    the code CAM, the register CAM reconstructs the input state, and the driver
    reads the produced word from the input registers (op<-AX, imm<-STACK0), overlays
    a runtime CODE frame at the EMIT immediate keyed bits(addr) on the WIDE
    (``CODE_ADDR_BITS``, position-invariant) direct-CAM lanes, and bumps PC.  EMIT
    adds NO baked weights, so flag-ON is byte-identical to golden 069cc32f."""
    return _os.environ.get("C4_CFM_EMIT", "0") not in ("0", "", "false", "False")


# ===========================================================================
# RUN-PHASE 32-bit gap closers (#789 PC-arith, #790 32-bit op gaps).  Every one
# is a GATED env flag DEFAULT-OFF: with all off the build is byte-IDENTICAL to
# golden ``069cc32f``.  The Doom port (552,698 instrs, full 32-bit ops) needs
# them ON; the 1096 corpus (8-bit, code_size < few-hundred) never trips them.
# ===========================================================================
def _shift32_enabled() -> bool:
    """``C4_SHIFT32`` (DEFAULT OFF): keep the FULL 32-bit SHL/SHR result the tight
    shifter already computes in the AX nibble band, instead of truncating it to the
    low byte (the ``ax-byte-nib`` writeback + the low-byte ``bw-recompose``).  The
    shifter (``shift_tight_nibble.tight_out_to_ax_rules``) writes all 8 AX nibbles
    with ``pop </>> ax`` (32-bit, arithmetic SHR sign-fill) — the byte truncation was
    the ONLY thing losing bits.  Doom's fixed-point ``>>`` is the #1 fast-path fix.
    OFF -> SHL/SHR stay in ``byte_ax_ops`` (byte result) -> golden byte-IDENTICAL."""
    return _os.environ.get("C4_SHIFT32", "0") not in ("0", "", "false", "False")


def _cmp32_enabled() -> bool:
    """``C4_CMP32`` (DEFAULT OFF): make the EQ/NE/LT/GT/LE/GE verdict robust to the
    full 32-bit operand width by deciding equality from a PER-NIBBLE all-8-nibble
    AND (``STACK0[j]==AX[j]`` for every j) rather than the noisy wide-scalar bump
    ``Z(STK_VAL-AX_VAL)`` (whose ~1e-6 relative recompose residue reaches ~±1 at a
    ~10^6-scale operand, mis-firing the tie).  On an exact 32-bit tie it forces
    ``CMP_EQ=1`` and ``CMP_GT=CMP_LT=0``; the sign-corrected magnitude order (already
    32-bit-exact for the DECISIVE sign/low-byte cases) is untouched off the tie.  OFF
    -> the cmp blocks are byte-IDENTICAL to golden."""
    return _os.environ.get("C4_CMP32", "0") not in ("0", "", "false", "False")


def _divmod_signed_enabled() -> bool:
    """``C4_DIVMOD_SIGNED`` (DEFAULT ON — golden 3cabef64): compute DIV/MOD with C4/native-c4 SIGNED
    truncation-toward-zero semantics (quotient sign = sign(a)^sign(b), remainder sign
    = sign(a), magnitudes from the unsigned base-16 long division) instead of the
    golden's UNSIGNED floor.  Native c4 ``int`` is signed, so Doom's signed ``/`` /
    ``%`` need this.  C4_DIVMOD_SIGNED=0 -> unsigned-floor rollback == the pre-migration golden 7d4afe61."""
    return _os.environ.get("C4_DIVMOD_SIGNED", "1") not in ("0", "", "false", "False")


# ===========================================================================
# The op groups (the stack contract).
# ===========================================================================
# POP-consuming ops: pop the stack top MEM[SP] into STACK0, op(STACK0, AX), SP+=4.
POP_OPS = [isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
           isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR,
           isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE,
           isa.SI, isa.SC]
# PUSH ops (allocate a stack slot): the driver lays a store MEM token.
PUSH_OPS = [isa.PSH, isa.JSR, isa.ENT]
# ALU ops that read STACK0 as operand A and AX as operand B (the 32-bit ALU set).
ALU_OPS = [isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD]


# ===========================================================================
# LAYOUT: PureForwardLayout + the stack-pop KV query band + the 32-bit ALU bands.
# ===========================================================================
class PureForwardCompleteLayout(PureForwardLayout):
    """``PureForwardLayout`` + the stack-pop KV query band + the ALU-32 scratch.

      ``IS_POP`` (1)       — set on a pop-consuming op; enables the stack KV head.
      ``SP_QRY_BIN`` (32)  — the SP address bits (the stack-pop QUERY).
      ``ALU_*`` (via ``extend_layout_for_alu32``) — the 32-bit ALU scratch bands.
    """

    def __init__(self, code_size: int, n_heads: int):
        super().__init__(code_size, n_heads=n_heads)
        from .blogspec_layout import NIB_PER_REG
        self._off = self.D
        self.IS_POP = self._scalar("IS_POP")        # enables the stack KV head
        self.IS_LEV = self._scalar("IS_LEV")        # enables the LEV return-PC head
        self.POP_ADDR = self._band("POP_ADDR", NIB_PER_REG)   # = SP (or BP on LEV)
        self.LEV_ADDR = self._band("LEV_ADDR", NIB_PER_REG)   # = BP+4 (LEV ret-PC)
        self.SP_QRY_BIN = self._band("SP_QRY_BIN", ADDR_BITS)  # pop address bits
        self.LEV_QRY_BIN = self._band("LEV_QRY_BIN", ADDR_BITS)  # LEV ret-PC bits
        self.LEV_RET = self._band("LEV_RET", NIB_PER_REG)     # loaded return-PC nibbles
        self.LEV_RET_VAL = self._scalar("LEV_RET_VAL")        # its scalar image
        self.AXB_LO = self._scalar("AXB_LO")                  # AX_VAL low nibble scratch
        self.AXB_HI = self._scalar("AXB_HI")                  # AX_VAL high nibble scratch
        # LEA frame quantity = BP_low + 4*imm (SIGNED imm), computed EXACT-INTEGER from
        # BP + IMM nibbles in ``compile_lea_q_reduce`` (no silu-recompose scalar residue)
        # and kept in the SMALL signed range [-256,511] so the address decode fires few
        # steps -- avoids the large-q GPU sparse accumulation error (the #680 deep-read-
        # back miss).  The address decode takes its low byte via ``a & 0xFF``.
        self.LEA_Q = self._scalar("LEA_Q")                    # signed BP_low+4*imm [-256,511]
        # WIDE LEA (C4_LEA_WIDE, DEFAULT OFF — see ``_lea_wide_enabled``): the full
        # 32-bit frame address ``AX = BP + 4*imm`` computed byte-by-byte with a carry
        # chain, so a frame-relative pointer keeps its high bytes (``LEA 40`` off
        # BP=0x10000 -> 0x100a0, not the folded 0xa0).  Allocated ONLY under the wide
        # flag; OFF -> these bands are absent -> the layout / golden hash is byte-
        # identical (the golden 069cc32f flags-off build never widens the substrate).
        self.LEAW_A = self.LEAW_B = self.LEAW_C = self.LEAW_U16 = None
        if _lea_wide_enabled():
            self.LEAW_A = self._band("LEAW_A", 4)    # operand A bytes = BP bytes 0..3
            self.LEAW_B = self._band("LEAW_B", 4)    # operand B bytes = (4*imm) bytes 0..3
            self.LEAW_C = self._band("LEAW_C", 5)    # carry chain C[0]=0 .. C[4]=drop
            self.LEAW_U16 = self._scalar("LEAW_U16")  # u16 = (4*imm) mod 2^16 scratch
        # FULL 32-bit IMM: the immediate's 8 nibbles are part of the STATIC program
        # encoding (CODE_IMM_NIB[i][j], written by the overlay from the bytecode —
        # a pure re-encoding of the constant, no runtime compute), gathered at PC
        # into IMM_NIB by the same PC-one-hot product-select as the scalar IMM, and
        # written verbatim into the 8 AX nibbles on IMM so literals > 255 survive.
        self.CODE_IMM_NIB = [self._band(f"CODE_IMM_NIB_{i}", IMM_NIBS)
                             for i in range(code_size)]
        self.IMM_NIB = self._band("IMM_NIB", IMM_NIBS)        # fetched immediate nibbles
        # CLEAN signed immediate scalar, reconstructed from the leak-free IMM_NIB
        # nibbles (each rounded to its exact 0..15 cell) + a two's-complement sign
        # correction.  Written ONCE into this DEDICATED never-share scratch dim (never
        # an in-place overwrite of the leaky ``IMM`` scalar), so the frame-offset ops
        # (LEA/ENT/ADJ/JSR) read an EXACT integer offset instead of ``IMM``'s
        # large-literal PC-one-hot leak (#648/#660).  See ``compile_imm_clean``.
        self.IMM_CLEAN = self._scalar("IMM_CLEAN")
        # LC SIGNED CHAR (c4 ``a = *(char *)a``): a scratch flag = OP_IS[LC] AND the
        # loaded byte's bit 7 (AX nibble 1 >= 8).  When set, the LC sign-extend block
        # fills AX nibbles 2..7 with 0xF (byte >= 0x80 -> negative char, sign-extended
        # to the 32-bit register).  LI stays an unsigned word load.
        self.LC_SIGN = self._scalar("LC_SIGN")
        # C4_CMP32 (#790, default OFF): a scratch scalar = 1 iff STACK0 and AX are
        # EQUAL across ALL 8 nibbles (a full 32-bit tie).  Used by the cmp override to
        # make EQ/NE/GT/LE/GE robust to the wide-scalar recompose noise.  Allocated
        # ONLY when the flag is on, so the flag-OFF layout / golden hash is unchanged.
        self.NIB_EQ = None
        # C4_CMP32 ORDER extension (#825): the per-nibble lexicographic-ORDER scratch.
        # ``NGT[j]``/``NLT[j]``/``NEQ[j]`` are the per-nibble compare lanes (each nibble
        # a small 0..15 integer, so every relu is fp-EXACT), and the override rewrites
        # the UNSIGNED magnitude order ``MAG_GT``/``MAG_LT`` from a MSB-first
        # lexicographic scan of those lanes — dodging the ~10^9-scale wide-scalar
        # ``step(STK_VAL-AX_VAL)`` recompose that collapses to 0 past 2^24.  All under
        # the flag, so the flag-OFF layout / golden hash is unchanged.
        self.NGT = self.NLT = self.NEQ = None
        if _cmp32_enabled():
            self.NIB_EQ = self._scalar("NIB_EQ")
            self.NGT = self._band("CMP_NGT", 8)   # [STACK0[j] > AX[j]] per nibble j
            self.NLT = self._band("CMP_NLT", 8)   # [STACK0[j] < AX[j]] per nibble j
            self.NEQ = self._band("CMP_NEQ", 8)   # [STACK0[j] == AX[j]] per nibble j
        # PART A (C4_UNIFY_CAM_HEAD, default OFF): the merged-CAM-head scratch bands.
        # Allocated ONLY when the flag is on, so the flag-OFF layout (and thus every
        # baked weight / the golden hash) is byte-identical to the 3-head build.
        if _unify_cam_head_enabled():
            from .blogspec_layout import NIB_PER_REG as _NPR
            self.UNI_QRY_BIN = self._band("UNI_QRY_BIN", ADDR_BITS)  # muxed read addr
            self.UNI_VAL = self._band("UNI_VAL", _NPR)               # merged read value
            self.IS_MEMREAD = self._scalar("IS_MEMREAD")            # IS_LOAD OR IS_POP
        # CODE-FROM-MEMORY (C4_PF_CFM, DEFAULT OFF): the FIXED (code_size-independent)
        # code-fetch CAM key/query + value bands.  When on, the layout is built with a
        # SINGLE vestigial per-slot table (code_size=1) and the program lives in these
        # code frames instead (the fetch@PC CAM), so the residual dim no longer scales
        # with the program length.  OFF -> these bands are absent -> byte-identical.
        self.CODE_KEY_BIN = self.CODE_QRY_BIN = None
        self.CODE_OPV = self.IS_CODE = self.IS_FETCH = None
        self.CODE_IMM_NIB_MEM = None
        self.cfm = _pf_cfm_enabled()
        if self.cfm:
            self.CODE_KEY_BIN = self._band("CODE_KEY_BIN", CODE_ADDR_BITS)  # instr addr i (KEY)
            self.CODE_QRY_BIN = self._band("CODE_QRY_BIN", CODE_ADDR_BITS)  # current PC (QUERY)
            self.CODE_OPV = self._scalar("CODE_OPV")     # code-frame op (VALUE -> OP_VAL)
            # the code frame's IMMEDIATE as IMM_NIBS nibbles (VALUE -> IMM_NIB): the
            # nibble-wise fetch that KEEPS the 20-bit IMM (NOT a mod-256 scalar).
            self.CODE_IMM_NIB_MEM = self._band("CODE_IMM_NIB_MEM", IMM_NIBS)
            self.IS_CODE = self._scalar("IS_CODE")       # code-frame flag (KEY gate)
            self.IS_FETCH = self._scalar("IS_FETCH")     # fetch-query flag (QUERY gate)
        while self._off % n_heads != 0:
            self._scalar(f"_pfcpad{self._off}")
        self.D = self._off


# ===========================================================================
# NIBBLE -> ADDRESS BITS: expand a register's low bytes into 32 per-bit dims by
# reading its NIBBLE band directly (each nibble 0..15 -> its 4 bits via a 16-cell
# one-hot).  EXACT and O(1) per nibble — no scalar magnitude, no 256-cell table.
# Bit (4*j + t) = bit t of nibble j = sum over cells a of nibble j with bit t set.
# ===========================================================================
def compile_nibble_addr_expand(L, reg_base: int, bin_base: int, dim: int,
                               n_nibbles: int = 8,
                               n_bits: int = None) -> Dict[str, torch.Tensor]:
    """``bin_base[4*j + t] = bit t of nibble j of the register at ``reg_base```` for
    j < n_nibbles, t in 0..3.  A nibble is 0..15; its one-hot over the 16 cells is
    the triangular pulse, and bit t is the sum of the cells whose index has bit t.
    Self-clears each written bit lane first (SET).

    ``n_bits`` (default ``4*n_nibbles``) CAPS how many low bits are written into
    ``bin_base`` — needed when the destination band is NARROWER than a nibble
    boundary (e.g. a 13-bit ``CODE_QRY_BIN``: 4 nibbles decode PC but only 13 bits
    may be written, else bits 13..15 spill into the NEXT band and corrupt it)."""
    thr = list(range(-1, 17))
    tu = {t: k for k, t in enumerate(thr)}
    n_thr = len(thr)
    n_bits = (4 * n_nibbles) if n_bits is None else min(n_bits, 4 * n_nibbles)
    spec = _empty_spec(dim, n_nibbles * n_thr + n_bits)
    u = 0
    # relu bank per nibble (16-cell one-hot thresholds).
    relu0 = {}
    for j in range(n_nibbles):
        relu0[j] = u
        for t, k in tu.items():
            spec["W_up"][u, reg_base + j] = RELU_S
            spec["b_up"][u] = -RELU_S * t
            spec["W_gate"][u, L.ONE] = 1.0
            u += 1
    # self-clear each output bit lane (SET).
    for b in range(n_bits):
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, bin_base + b] = 1.0
        spec["W_down"][bin_base + b, u] += -1.0 / SILU_S
        u += 1
    # bit t of nibble j = sum over cells a in 0..15 with (a>>t)&1 of one-hot(a).
    for j in range(n_nibbles):
        r0 = relu0[j]
        for a in range(16):
            # one-hot(a) coefficient = tri pulse: +cell(a-1) -2cell(a) +cell(a+1).
            for t in range(4):
                b = 4 * j + t
                if b >= n_bits:        # NARROW destination band: don't spill
                    continue
                if (a >> t) & 1:
                    spec["W_down"][bin_base + b, r0 + tu[a - 1]] += 1.0 / RELU_S
                    spec["W_down"][bin_base + b, r0 + tu[a]] += -2.0 / RELU_S
                    spec["W_down"][bin_base + b, r0 + tu[a + 1]] += 1.0 / RELU_S
    return spec


def _clear_band_gated_ops(L, base, n, dim, gate_ops):
    """Clear band dims base..base+n-1 when any OP_IS[gate_ops] active (SET -old)."""
    spec = _empty_spec(dim, n * len(gate_ops))
    u = 0
    for op in gate_ops:
        g = L.OP_IS + op
        for j in range(n):
            spec["W_up"][u, g] = S
            spec["b_up"][u] = -S * 0.5
            spec["W_gate"][u, base + j] = 1.0
            spec["W_down"][base + j, u] += -1.0 / SILU_HALF
            u += 1
    return spec


# ===========================================================================
# STACK-POP PREP FFN: on a pop op set IS_POP, expand the SP nibble band ->
# SP_QRY_BIN (the load address), and clear the STACK0 nibble band so the stack KV
# head's additive write is a clean SET of the popped value.  All opcode-gated.
# ===========================================================================
def compile_pop_addr(L: PureForwardCompleteLayout, dim: int) -> Dict[str, torch.Tensor]:
    """POP_ADDR nibbles := SP nibbles by default, but := BP nibbles on a LEV (LEV's
    first load is MEM[BP], not MEM[SP]).  LEV_ADDR nibbles := BP+4 (LEV's second
    load, the return PC) — computed as BP scalar + 4 recomposed to nibbles is
    awkward; instead we set LEV_ADDR = BP nibbles and add 4 in the address bits by
    querying BP and letting the LEV head's query be BP with bit-2 forced.  Simpler
    & exact: LEV_ADDR carries BP's nibbles and we OR bit 2 (value 4) — but BP is
    4-aligned so bit 2 = 4 flips cleanly.  We build LEV_QRY_BIN = SP_QRY_BIN(BP)
    with bit 2 set, done in the expand.  Here we only fill POP_ADDR / LEV_ADDR
    nibble bands.  IS_POP / IS_LEV flags are set in compile_stack_prep."""
    a = L
    n = 8
    # POP_ADDR = SP (default), overwritten to BP on LEV.
    spec = _empty_spec(dim, n * 3 + n * 3)
    u = 0
    for j in range(n):                          # POP_ADDR := SP  (clear + copy)
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, L.POP_ADDR + j] = 1.0
        spec["W_down"][L.POP_ADDR + j, u] += -1.0 / SILU_S; u += 1
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, L.SP + j] = 1.0
        spec["W_down"][L.POP_ADDR + j, u] += 1.0 / SILU_S; u += 1
        # on LEV: POP_ADDR += (BP - SP)  (gated), so POP_ADDR becomes BP.
        g = L.OP_IS + isa.LEV
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.BP + j] = 1.0
        spec["W_gate"][u, L.SP + j] = -1.0
        spec["W_down"][L.POP_ADDR + j, u] += 1.0 / SILU_HALF; u += 1
    for j in range(n):                          # LEV_ADDR := BP  (clear + copy)
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, L.LEV_ADDR + j] = 1.0
        spec["W_down"][L.LEV_ADDR + j, u] += -1.0 / SILU_S; u += 1
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, L.BP + j] = 1.0
        spec["W_down"][L.LEV_ADDR + j, u] += 1.0 / SILU_S; u += 1
        u += 1                                   # (spare unit, keeps width uniform)
    return spec


def compile_stack_prep(L: PureForwardCompleteLayout, dim: int) -> Dict[str, torch.Tensor]:
    specs = []
    specs.append(_flag_from_ops(L, L.IS_POP, POP_OPS + [isa.LEV], dim))
    specs.append(_flag_from_ops(L, L.IS_LEV, [isa.LEV], dim))
    # SP_QRY_BIN <- bits of POP_ADDR (SP for pops, BP for LEV).
    specs.append(compile_nibble_addr_expand(L, L.POP_ADDR, L.SP_QRY_BIN, dim, n_nibbles=8))
    # LEV_QRY_BIN <- bits of LEV_ADDR (BP).  The +4 (BP -> BP+4, the return-PC slot)
    # is a SEPARATE block (``lev-addr4``) because the +4 ripple-carry reads
    # LEV_QRY_BIN at its block INPUT — it must run AFTER this expand writes it.
    specs.append(compile_nibble_addr_expand(L, L.LEV_ADDR, L.LEV_QRY_BIN, dim, n_nibbles=8))
    # clear STACK0 (pop dest) + LEV_RET (LEV ret-PC dest) so head writes are clean.
    specs.append(_clear_band_gated_ops(L, L.STACK0, 8, dim, POP_OPS + [isa.LEV]))
    specs.append(_clear_band_gated_ops(L, L.LEV_RET, 8, dim, [isa.LEV]))
    return _concat_specs(specs, dim)


def _force_bit2(L, bin_base, dim):
    """Add 4 to the binary address band (BP+4 = LEV return-PC slot).  BP is
    4-aligned (bits 0,1 = 0) but bit 2 is NOT necessarily 0 (e.g. BP=228=0xE4 has
    bit 2 set) — so a bit-force would be WRONG.  This is a true ripple-carry +1 at
    bit position 2, expressed ENTIRELY on the block-input bits (FFN units all read
    the block input, so no relu chaining):

        carry_k  = AND(bits 2..k-1)  = [ sum(bits 2..k-1) == k-2 ]     (carry into k)
        andk     = AND(bits 2..k)    = [ sum(bits 2..k)   == k-1 ]     ( = bit_k·carry_k)
        new_bit_k = bit_k XOR carry_k = bit_k + carry_k - 2·andk

    carry_2 = 1 (the injected +4).  Bits 0,1 are untouched.  Each ``[sum==n]``
    prefix-AND is one relu bump (relu(x-(n-0.5)) - relu(x-(n+0.5)))."""
    hi = ADDR_BITS

    def and_bits(spec, u, lo, k, out_base, coeff):
        """Add ``coeff · AND(bits lo..k)`` to residual dim ``out_base``.  x = Σ bits
        (an integer in 0..cnt), so AND == [x >= cnt] — a SHARP unit step at cnt
        realised as a narrow ramp relu(x-(cnt-w)) - relu(x-cnt) of amplitude 1 at
        x=cnt (w=0.5, so the step is 1.0 exactly at the integer x=cnt, 0 at cnt-1)."""
        cnt = k - lo + 1
        w = 0.5
        for i, thr in enumerate((cnt - w, cnt)):
            spec["W_up"][u, L.ONE] = -RELU_S * thr
            for j in range(lo, k + 1):
                spec["W_up"][u, bin_base + j] = RELU_S
            spec["W_gate"][u, L.ONE] = 1.0
            spec["W_down"][out_base, u] += (coeff if i == 0 else -coeff) / (RELU_S * w)
            u += 1
        return u

    # new_bit_k = bit_k XOR carry_k, so the FFN DELTA (added to the residual bit_k)
    # is exactly  carry_k - 2·(bit_k AND carry_k) = carry_k - 2·AND(bits 2..k).  No
    # clear/restore needed — the residual already carries bit_k.  carry_2 = 1.
    spec = _empty_spec(dim, hi * 6)
    u = 0
    for k in range(2, hi):
        if k == 2:                                   # carry_2 = 1 (the injected +4)
            spec["W_up"][u, L.ONE] = S
            spec["W_gate"][u, L.ONE] = 1.0
            spec["W_down"][bin_base + k, u] += 1.0 / SILU_S; u += 1
        else:                                        # carry_k = AND(bits 2..k-1)
            u = and_bits(spec, u, 2, k - 1, bin_base + k, +1.0)
        # - 2·(bit_k AND carry_k) = -2·AND(bits 2..k)
        u = and_bits(spec, u, 2, k, bin_base + k, -2.0)
    return spec


# ===========================================================================
# ONE address-CAM read head (the unification of the three §Memory KV heads).
# ===========================================================================
# The model's three GLOBAL attention heads (§Memory KV head, stack-pop head, LEV
# return-PC head) all compute the SAME operation — a binary-address softmax1 CAM
# read of ``mem[addr]`` (latest-write-wins) — differing ONLY in three parameters:
#
#     head          query band     enable flag   value dest
#     -----------   ------------   -----------   ----------
#     §Memory LI    QRY_BIN        IS_LOAD       AX      (the loaded value)
#     stack-pop     SP_QRY_BIN     IS_POP        STACK0  (the popped operand)
#     LEV ret-PC    LEV_QRY_BIN    IS_LEV        LEV_RET (the saved return PC)
#
# So ONE parameterised bake (:func:`_bake_cam_head`) authors all three: the 32
# ±smag address channels, the ZFOD bias, the store-role penalty, the read-enable
# channel and the value relay are IDENTICAL; only ``qry_band`` / ``enable_flag`` /
# ``value_dest`` change.  The three thin wrappers below select the trio.  This is a
# pure structural dedup — the emitted weights are byte-IDENTICAL to the three
# former hand-copied bakes (proven by the golden-hash gate).
def _bake_cam_head(attn, L, head: int, qry_band: int, enable_flag: int,
                   value_dest: int) -> None:
    """Bake attention head ``head`` as a §Memory binary-address CAM read.

    ``qry_band``    — the residual band holding the 32 query address bits.
    ``enable_flag`` — the scalar flag (1.0) that arms the read for this op class.
    ``value_dest``  — the nibble band the retrieved value is written into.

    The KEY is always the store address (``ADDR_BIN``, ±smag); the store-flag /
    ZFOD-bias / store-role-penalty channels are exactly ``_bake_pf_memory_head``'s
    (§Memory) — only the query source, arm flag and value destination differ.
    """
    from .blogspec_memory import EFF, BIAS, MEM_ALIBI_SLOPE, PEN_GATE
    from .blogspec_layout import NIB_PER_REG
    hs = attn.scale
    smag = (EFF / hs) ** 0.5
    qb = (BIAS / hs) ** 0.5
    kb = (BIAS / hs) ** 0.5
    # Role-gate penalty (stays huge = 100·ADDR_BITS·EFF); the enable query flag is
    # THRESHOLDED clean (``_flag_from_ops`` step) so a residue at a large SP/BP/PC
    # cannot swamp the exact-address match (see PEN_GATE note).
    PEN = PEN_GATE
    p = (PEN / hs) ** 0.5
    attn.alibi_slopes[head] = MEM_ALIBI_SLOPE
    HD = attn.head_dim
    base = head * HD
    for b in range(ADDR_BITS):
        attn.W_k[base + b, L.ADDR_BIN + b] = 2.0 * smag
        attn.W_k[base + b, L.ONE] = -smag
        attn.W_q[base + b, qry_band + b] = 2.0 * smag       # QUERY = this op's addr
        attn.W_q[base + b, L.ONE] = -smag
    cB = base + ADDR_BITS
    attn.W_q[cB, enable_flag] = -qb         # ZFOD bias enabled by the read flag
    attn.W_k[cB, L.IS_STORE] = kb
    cR = base + ADDR_BITS + 1
    attn.W_q[cR, enable_flag] = p           # store-role penalty
    attn.W_k[cR, L.ONE] = -p
    attn.W_k[cR, L.IS_STORE] = p
    cL = base + ADDR_BITS + 2               # READ-enable: non-read query -> sink
    c = (PEN / hs) ** 0.5
    attn.W_q[cL, L.ONE] = c
    attn.W_q[cL, enable_flag] = -c
    attn.W_k[cL, L.ONE] = -c
    for j in range(NIB_PER_REG):
        attn.W_v[base + ADDR_BITS + 3 + j, L.VAL_NIB + j] = 1.0
        attn.W_o[value_dest + j, base + ADDR_BITS + 3 + j] = 1.0


# ===========================================================================
# CODE-FROM-MEMORY (C4_PF_CFM): the code-fetch CAM head.  The SAME softmax1 binary-
# address CAM as ``_bake_cam_head``, but keyed on the CODE frames (KEY =
# ``CODE_KEY_BIN`` = instruction address i, gated ``IS_CODE``; QUERY =
# ``CODE_QRY_BIN`` = current PC, gated ``IS_FETCH``) and delivering the code frame's
# op scalar -> ``OP_VAL`` and its IMM_NIBS immediate NIBBLES -> ``IMM_NIB`` (the
# nibble-wise fetch that PRESERVES the 20-bit IMM — NO mod-256 scalar fold).  Every
# code address appears at MOST once, so no recency tiebreak is needed.  Store frames
# (``IS_STORE``) are pushed FAR below the sink so the two logs never collide.
# ===========================================================================
def _bake_code_cam_head(attn, L: PureForwardCompleteLayout, head: int) -> None:
    """Bake the code-fetch CAM on ``attn`` head ``head`` (fetch mem_code[PC]).

    Structurally identical to ``_bake_cam_head`` (per-bit address agreement +
    ZFOD-bias + read-enable gate), on the CODE_ADDR_BITS-wide code address, with a
    store-frame exclusion channel.  VALUE copies ``CODE_OPV`` -> ``OP_VAL`` and each
    ``CODE_IMM_NIB_MEM[j]`` -> ``IMM_NIB[j]`` (nibble-wise -> 20-bit IMM survives)."""
    from .blogspec_memory import EFF, BIAS, MEM_ALIBI_SLOPE, PEN_GATE
    hs = attn.scale
    smag = (EFF / hs) ** 0.5
    # ZFOD bias for the CODE_ADDR_BITS-wide match (BIAS is authored for ADDR_BITS=32;
    # scale it to the 12-bit code address so an EXACT match nets +EFF and any 1-bit
    # mismatch nets < 0 -> softmax1's +1 sink gives 0 on a non-fetch / no-such-addr).
    qb = kb = (BIAS * (CODE_ADDR_BITS - 1) / (ADDR_BITS - 1) / hs) ** 0.5
    PEN = PEN_GATE
    p = (PEN / hs) ** 0.5
    # RECALL HORIZON (wall #6): the §Memory ALiBi recency imposes a horizon of
    # ``EFF/slope`` (= 500000) tokens — a far-back exact match scores ``EFF - slope·dist``
    # and fades to the softmax1 sink once ``dist > EFF/slope``.  For the STACK / LI heads
    # that horizon is load-bearing (latest-write-wins among SAME-address stores).  But the
    # CODE frames are STATIC: exactly ONE code frame per PC (never re-emitted, never
    # superseded), so there are NO same-address ties for recency to break — the code-CAM
    # needs a PURE address match with NO distance decay.  With slope=1.0 the code fetch
    # FADED once the query row crossed ~500000 tokens (doom step ~15,574 at token 500,746),
    # so ``EFF - dist ≈ 0`` tipped below the sink -> the op didn't decode -> IS_POP unset ->
    # the stack-pop CAM applied its -PEN gate and returned ZFOD -> the SI desynced.  Setting
    # the code-CAM ALiBi slope to 0 removes the horizon (byte-exact: the address match is
    # unique per PC), so the program fetches cleanly at ANY stream depth.  Kill-switch:
    # ``C4_CODE_CAM_SLOPE`` overrides the slope (default 0.0); set to MEM_ALIBI_SLOPE to
    # restore the pre-fix behaviour.
    import os as _oscs
    _cs = _oscs.environ.get("C4_CODE_CAM_SLOPE")
    attn.alibi_slopes[head] = float(_cs) if _cs is not None else 0.0
    HD = attn.head_dim
    base = head * HD
    n_val = 1 + IMM_NIBS
    assert HD >= CODE_ADDR_BITS + 4 + n_val, (
        f"code CAM needs head_dim >= {CODE_ADDR_BITS + 4 + n_val}, got {HD}")
    # per-bit address agreement: q_b = smag*(2*QRY_BIN[b]-ONE), k_b = smag*(2*KEY_BIN[b]-ONE)
    # gated so a NON-fetch query (IS_FETCH=0) and a non-code frame (IS_CODE=0) score 0.
    for b in range(CODE_ADDR_BITS):
        attn.W_k[base + b, L.CODE_KEY_BIN + b] = 2.0 * smag
        attn.W_k[base + b, L.IS_CODE] = -smag
        attn.W_q[base + b, L.CODE_QRY_BIN + b] = 2.0 * smag
        attn.W_q[base + b, L.IS_FETCH] = -smag
    cB = base + CODE_ADDR_BITS               # ZFOD bias (needs BOTH flags)
    attn.W_q[cB, L.IS_FETCH] = -qb
    attn.W_k[cB, L.IS_CODE] = kb
    cR = base + CODE_ADDR_BITS + 1           # code-frame role penalty (non-code -> sink)
    attn.W_q[cR, L.IS_FETCH] = p
    attn.W_k[cR, L.ONE] = -p
    attn.W_k[cR, L.IS_CODE] = p
    cL = base + CODE_ADDR_BITS + 2           # FETCH-enable: non-fetch query -> sink
    attn.W_q[cL, L.ONE] = p
    attn.W_q[cL, L.IS_FETCH] = -p
    attn.W_k[cL, L.ONE] = -p
    # STORE-FRAME EXCLUSION: a store frame (IS_STORE=1) shares the token stream and
    # would otherwise score at the sink (0); push it FAR below so store rows are
    # INVISIBLE to the code CAM (the code and store logs never collide).
    cX = base + CODE_ADDR_BITS + 3
    if getattr(L, "IS_STORE", None) is not None:
        attn.W_q[cX, L.ONE] = p
        attn.W_k[cX, L.IS_STORE] = -p
    # VALUE: op scalar -> OP_VAL ; immediate nibbles -> IMM_NIB (20-bit IMM survives).
    v0 = base + CODE_ADDR_BITS + 4
    attn.W_v[v0, L.CODE_OPV] = 1.0
    attn.W_o[L.OP_VAL, v0] = 1.0
    for j in range(IMM_NIBS):
        attn.W_v[v0 + 1 + j, L.CODE_IMM_NIB_MEM + j] = 1.0
        attn.W_o[L.IMM_NIB + j, v0 + 1 + j] = 1.0


# ===========================================================================
# The STACK KV head: identical §Memory CAM as the LI/LC head, but keyed on the
# STACK query band SP_QRY_BIN (address = SP) and enabled by IS_POP, writing the
# retrieved value nibbles into the STACK0 band (the operand the ALU pops).
# ===========================================================================
def _bake_stack_pop_head(attn, L: PureForwardCompleteLayout, head: int) -> None:
    """Bake the stack-pop KV head: query = SP_QRY_BIN, enable = IS_POP, value ->
    STACK0 nibble band.  Same address CAM + ZFOD-bias + store-role + pop-enable
    channels as ``_bake_pf_memory_head``, on a different head/query/dest."""
    _bake_cam_head(attn, L, head, qry_band=L.SP_QRY_BIN,
                   enable_flag=L.IS_POP, value_dest=L.STACK0)


# ===========================================================================
# STACK-POP recompose: refresh STK_VAL from the STACK0 nibble band (the KV head
# just wrote the popped value there), so the dispatch/ALU read the fresh operand.
# (Same nibble->scalar recompose the base block does, but only STACK0.)
# ===========================================================================
def compile_stk_recompose(L, dim: int, hi_nibbles: int = 8) -> Dict[str, torch.Tensor]:
    """Refresh STK_VAL from STACK0 and LEV_RET_VAL from LEV_RET (both nibble bands
    the KV heads just wrote), so the LEV dispatch reads the loaded scalars."""
    spec = _empty_spec(dim, 2 * (1 + hi_nibbles))
    u = 0
    for nib_base, val_lane in ((L.STACK0, L.STK_VAL), (L.LEV_RET, L.LEV_RET_VAL)):
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, val_lane] = 1.0
        spec["W_down"][val_lane, u] += -1.0 / SILU_S
        u += 1
        for j in range(hi_nibbles):
            spec["W_up"][u, L.ONE] = S
            spec["W_gate"][u, nib_base + j] = 1.0
            spec["W_down"][val_lane, u] += (16.0 ** j) / SILU_S
            u += 1
    return spec


def compile_lev_ret_recompose(L, dim: int, hi_nibbles: int = 8) -> Dict[str, torch.Tensor]:
    """Refresh ``LEV_RET_VAL`` from the ``LEV_RET`` nibble band (SET).  The
    ``C4_UNIFY_CAM_ONE`` FFN of the SECOND cam block (``lev-cam2``): its attention
    (the RE-FIRED merged head, query=BP+4) has just written the return-PC nibbles into
    ``LEV_RET``, so recompose the scalar the LEV dispatch reads (``compile_stk_recompose``
    at the stack-pop-cam block SET it to 0 because ``LEV_RET`` was still empty there —
    the read is issued a block LATER on the reused head index).  Idempotent on a
    non-LEV step (``LEV_RET`` stays 0 -> ``LEV_RET_VAL`` = 0)."""
    spec = _empty_spec(dim, 1 + hi_nibbles)
    u = 0
    spec["W_up"][u, L.ONE] = S
    spec["W_gate"][u, L.LEV_RET_VAL] = 1.0
    spec["W_down"][L.LEV_RET_VAL, u] += -1.0 / SILU_S
    u += 1
    for j in range(hi_nibbles):
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, L.LEV_RET + j] = 1.0
        spec["W_down"][L.LEV_RET_VAL, u] += (16.0 ** j) / SILU_S
        u += 1
    return spec


def _bake_lev_ret_head(attn, L: PureForwardCompleteLayout, head: int) -> None:
    """The LEV return-PC KV head: query = LEV_QRY_BIN (address = BP+4), enable =
    IS_LEV, value -> LEV_RET nibble band.  Same §Memory CAM as the stack head."""
    _bake_cam_head(attn, L, head, qry_band=L.LEV_QRY_BIN,
                   enable_flag=L.IS_LEV, value_dest=L.LEV_RET)


# ===========================================================================
# PART A (C4_UNIFY_CAM_HEAD): merge the §Memory LI read into the stack-pop head.
# ===========================================================================
def compile_unify_cam_prep(L: PureForwardCompleteLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Build the MERGED-CAM inputs (``C4_UNIFY_CAM_HEAD``): the muxed query address
    ``UNI_QRY_BIN`` and the merged read-enable ``IS_MEMREAD``.

    ``UNI_QRY_BIN = QRY_BIN`` on a load, ``= SP_QRY_BIN`` on a pop/LEV.  The two
    query bands are set UNGATED (QRY_BIN = AX bits, SP_QRY_BIN = POP_ADDR bits every
    step), so we do a FLAG-GATED copy of each: ``+ QRY_BIN·IS_LOAD`` and
    ``+ SP_QRY_BIN·IS_POP``.  IS_LOAD and IS_POP are mutually exclusive, so exactly
    one term is live and ``UNI_QRY_BIN`` holds the correct address.  ``IS_MEMREAD =
    IS_LOAD OR IS_POP`` arms the merged head.  Runs AFTER stack-prep (both flags +
    both query bands are set by then).  Self-clears ``UNI_QRY_BIN`` first (SET)."""
    n = ADDR_BITS
    # per output bit: 1 self-clear + 2 gated adds ; + 2 for IS_MEMREAD (clear + 2 ORs)
    spec = _empty_spec(dim, n * 3 + 3)
    u = 0
    for b in range(n):
        # self-clear UNI_QRY_BIN[b] (SET)
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, L.UNI_QRY_BIN + b] = 1.0
        spec["W_down"][L.UNI_QRY_BIN + b, u] += -1.0 / SILU_S
        u += 1
        # + QRY_BIN[b] gated on IS_LOAD:  silu(S·IS_LOAD - S/2)·QRY_BIN[b]
        spec["W_up"][u, L.IS_LOAD] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.QRY_BIN + b] = 1.0
        spec["W_down"][L.UNI_QRY_BIN + b, u] += 1.0 / SILU_HALF
        u += 1
        # + SP_QRY_BIN[b] gated on IS_POP
        spec["W_up"][u, L.IS_POP] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.SP_QRY_BIN + b] = 1.0
        spec["W_down"][L.UNI_QRY_BIN + b, u] += 1.0 / SILU_HALF
        u += 1
    # IS_MEMREAD := IS_LOAD OR IS_POP  (mutually exclusive so the sum is a clean 0/1).
    spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.IS_MEMREAD] = 1.0
    spec["W_down"][L.IS_MEMREAD, u] += -1.0 / SILU_S; u += 1
    spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.IS_LOAD] = 1.0
    spec["W_down"][L.IS_MEMREAD, u] += 1.0 / SILU_S; u += 1
    spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.IS_POP] = 1.0
    spec["W_down"][L.IS_MEMREAD, u] += 1.0 / SILU_S; u += 1
    return spec


def compile_unify_cam_demux(L: PureForwardCompleteLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Demux the merged-head value ``UNI_VAL`` into the right destination band
    (``C4_UNIFY_CAM_HEAD``): ``AX += UNI_VAL`` on a load, ``STACK0 += UNI_VAL`` on a
    pop/LEV.  Gated on IS_LOAD / IS_POP (mutually exclusive), so exactly one
    destination is written and it receives the same nibbles the standalone §Memory
    / stack head would have written directly.  ADD semantics (not SET): the merged
    head is enabled only on a read, and the destination band was cleared by the same
    upstream FFN (mem-prep clears AX on a load; stack-prep clears STACK0 on a pop),
    exactly as in the 3-head build.

    Then recompose ``AX_VAL`` from the (now updated) AX nibble band so the scalar
    lane the dispatch reads reflects a loaded value — the exact role the mem-cam
    block's ``compile_nibble_to_scalar`` recompose played after the standalone
    §Memory head.  Idempotent on a non-load step (AX nibble band == step-input AX)."""
    from .blogspec_layout import NIB_PER_REG
    hi = _recompose_hi_nibbles()
    spec = _empty_spec(dim, NIB_PER_REG * 2 + (1 + hi))
    u = 0
    for j in range(NIB_PER_REG):
        # AX[j] += UNI_VAL[j] gated on IS_LOAD
        spec["W_up"][u, L.IS_LOAD] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.UNI_VAL + j] = 1.0
        spec["W_down"][L.AX + j, u] += 1.0 / SILU_HALF
        u += 1
        # STACK0[j] += UNI_VAL[j] gated on IS_POP
        spec["W_up"][u, L.IS_POP] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.UNI_VAL + j] = 1.0
        spec["W_down"][L.STACK0 + j, u] += 1.0 / SILU_HALF
        u += 1
    # AX_VAL := Σ 16^j · AX_nibble_j  (SET: self-clear then recompose)
    spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.AX_VAL] = 1.0
    spec["W_down"][L.AX_VAL, u] += -1.0 / SILU_S; u += 1
    for j in range(hi):
        spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.AX + j] = 1.0
        spec["W_down"][L.AX_VAL, u] += (16.0 ** j) / SILU_S; u += 1
    return spec


def compile_lc_sign_detect(L: PureForwardCompleteLayout, dim: int) -> Dict[str, torch.Tensor]:
    """``LC_SIGN = [OP_IS[LC] == 1  AND  AX_nib1 >= 8]`` — a signed-char LC whose
    loaded byte has bit 7 set (c4 ``a = *(char *)a``, byte >= 0x80 -> negative).

    Realised as a SINGLE sharp step over the combined form ``f = 16*OP_IS[LC] +
    AX_nib1`` (AX_nib1 in 0..15, OP_IS[LC] in {0,1}):  ``f >= 24`` iff OP_IS[LC]==1
    AND AX_nib1 >= 8  (LC + nib1<8 -> f<=23; LI (OP_IS[LC]=0) -> f<=15).  Its OWN
    block, so the fill block (next) reads a materialised LC_SIGN."""
    w = 0.5
    spec = _empty_spec(dim, 3)
    u = 0
    # SET LC_SIGN := 0 first (self-clear), then the ramp step.
    spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.LC_SIGN] = 1.0
    spec["W_down"][L.LC_SIGN, u] += -1.0 / SILU_S; u += 1
    for i, thr in enumerate((24 - w, 24)):
        spec["W_up"][u, L.OP_IS + isa.LC] = RELU_S * 16.0
        spec["W_up"][u, L.AX + 1] = RELU_S * 1.0
        spec["b_up"][u] = -RELU_S * thr
        spec["W_gate"][u, L.ONE] = 1.0
        spec["W_down"][L.LC_SIGN, u] += (1.0 if i == 0 else -1.0) / (RELU_S * w)
        u += 1
    return spec


def compile_lc_sign_extend(L: PureForwardCompleteLayout, dim: int) -> Dict[str, torch.Tensor]:
    """SIGN-EXTEND a signed-char LC: when ``LC_SIGN`` is set, fill AX nibbles 2..7
    with 0xF (byte >= 0x80 -> the char is negative, sign-extended to the 32-bit
    register: 0x80 -> 0xFFFFFF80, 0xFF -> 0xFFFFFFFF).  LI is untouched (LC_SIGN is
    OP_IS[LC]-gated).  Then re-recompose ``AX_VAL`` from the sign-extended nibbles so
    the scalar lane the dispatch/emit reads reflects the signed value.

    The low 2 nibbles (the loaded byte) are already correct; nibbles 2..7 were 0 from
    a byte-wide store, so SETting them to 15 under LC_SIGN is a clean fill."""
    hi = _recompose_hi_nibbles()
    spec = _empty_spec(dim, (8 - 2) + 1 + hi)
    u = 0
    for j in range(2, 8):
        # AX[j] := 15 gated on LC_SIGN.  SET: the nibble is 0 for a byte load, so a
        # gated +15 (silu-gate on LC_SIGN, value 15) sets it; on a non-LC-sign step
        # (LC_SIGN=0) the gate is ~0 -> untouched.
        spec["W_up"][u, L.LC_SIGN] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.ONE] = 15.0
        spec["W_down"][L.AX + j, u] += 1.0 / SILU_HALF
        u += 1
    # AX_VAL := Σ 16^j · AX_nibble_j  (SET: self-clear then recompose) — reflect the
    # now sign-extended high nibbles.
    spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.AX_VAL] = 1.0
    spec["W_down"][L.AX_VAL, u] += -1.0 / SILU_S; u += 1
    for j in range(hi):
        spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.AX + j] = 1.0
        spec["W_down"][L.AX_VAL, u] += (16.0 ** j) / SILU_S; u += 1
    return spec


def compile_unify_stk_recompose(L, dim: int, hi_nibbles: int = 8) -> Dict[str, torch.Tensor]:
    """Refresh ``STK_VAL`` from the ``STACK0`` nibble band AFTER the demux wrote it
    (``C4_UNIFY_CAM_HEAD`` / ``C4_UNIFY_CAM_ONE``).  In the merged path the popped
    value lands in ``STACK0`` only at the ``unify-cam-demux`` block (one block AFTER
    the stack-pop-cam block whose FFN ``compile_stk_recompose`` runs), so the STK_VAL
    that block recomposed is STALE (0).  The ALU reads ``STACK0`` nibbles directly
    (so pop_add etc. are unaffected), but **LEV reads ``STK_VAL``** for ``BP=MEM[BP]``
    — a stale STK_VAL made nested LEV set BP=0 (invisible on a single-frame return
    that halts right after, but WRONG once an outer frame's BP must survive).  This
    block re-recomposes STK_VAL from the demux-updated STACK0 (SET).  Idempotent on a
    non-pop step (STACK0 == its step-input, so STK_VAL is unchanged)."""
    spec = _empty_spec(dim, 1 + hi_nibbles)
    u = 0
    spec["W_up"][u, L.ONE] = S
    spec["W_gate"][u, L.STK_VAL] = 1.0
    spec["W_down"][L.STK_VAL, u] += -1.0 / SILU_S
    u += 1
    for j in range(hi_nibbles):
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, L.STACK0 + j] = 1.0
        spec["W_down"][L.STK_VAL, u] += (16.0 ** j) / SILU_S
        u += 1
    return spec


def _bake_unified_cam_head(attn, L: PureForwardCompleteLayout, head: int) -> None:
    """The MERGED CAM head (``C4_UNIFY_CAM_HEAD``): query = UNI_QRY_BIN (muxed load /
    pop address), enable = IS_MEMREAD, value -> UNI_VAL.  ONE head serving both the
    §Memory LI read and the stack-pop read (never co-occur)."""
    _bake_cam_head(attn, L, head, qry_band=L.UNI_QRY_BIN,
                   enable_flag=L.IS_MEMREAD, value_dest=L.UNI_VAL)


def bake_global_cam_heads(blocks, L: PureForwardCompleteLayout, block_specs) -> None:
    """Bake the global address-CAM head(s) onto ``blocks`` per the active unify flag.
    ONE authority shared by ALL builders (the dense builder here + the two streaming
    builders in ``compact_alloc``) so the head layout is identical across build paths.

    ``blocks``       — the built block list (each ``.attn`` already zeroed).
    ``block_specs``  — the ``(name, spec)`` list, for ``_find`` block-name lookup.

    Three configurations (governed by the flags, all default OFF -> the 3-head build):
      * ``C4_UNIFY_CAM_ONE``  : ONE head INDEX (N_ROLES+1) for EVERY global CAM read.
        stack-pop-cam: merged head (UNI_QRY_BIN -> UNI_VAL: LI/LC + pop + LEV MEM[BP]).
        lev-cam2: the SAME index re-fires (LEV_QRY_BIN=BP+4, IS_LEV -> LEV_RET).
        Global-CAM head count = 1.  (n_heads = N_ROLES+2, so N_ROLES+1 is the last head.)
      * ``C4_UNIFY_CAM_HEAD`` : merged head (N_ROLES+1) + dedicated LEV head (N_ROLES+2)
        on stack-pop-cam.  Standalone mem-cam head dropped.  Count = 2.
      * neither               : the golden 3-head build (mem-cam LI head N_ROLES +
        stack-pop head N_ROLES+1 + LEV head N_ROLES+2).  Count = 3."""
    stk = _find(block_specs, "stack-pop-cam")
    if _unify_cam_one_enabled():
        _bake_unified_cam_head(blocks[stk].attn, L, head=N_ROLES + 1)
        lev2 = _find(block_specs, "lev-cam2")
        _bake_cam_head(blocks[lev2].attn, L, head=N_ROLES + 1,
                       qry_band=L.LEV_QRY_BIN, enable_flag=L.IS_LEV,
                       value_dest=L.LEV_RET)
    elif _unify_cam_head_enabled():
        _bake_unified_cam_head(blocks[stk].attn, L, head=N_ROLES + 1)
        _bake_lev_ret_head(blocks[stk].attn, L, head=N_ROLES + 2)
    else:
        mem = _find(block_specs, "mem-cam")
        _bake_pf_memory_head(blocks[mem].attn, L, head=N_ROLES)
        _bake_stack_pop_head(blocks[stk].attn, L, head=N_ROLES + 1)
        _bake_lev_ret_head(blocks[stk].attn, L, head=N_ROLES + 2)
    # CODE-FROM-MEMORY: bake the code-fetch CAM on the "code-select" block, on the
    # LAST head index (past the 3 global CAM heads).  attn=code CAM; its FFN is a
    # no-op (the CAM writes OP_VAL + IMM_NIB).
    if _pf_cfm_enabled():
        csel = _find(block_specs, "code-select")
        _bake_code_cam_head(blocks[csel].attn, L, head=blocks[csel].attn.n_heads - 1)


def cam_baked_blocks(block_specs) -> set:
    """The set of block indices that get a global CAM head baked (so the streaming
    builder marks them as dense-attn).  Depends on the active unify flag."""
    stk = _find(block_specs, "stack-pop-cam")
    if _unify_cam_one_enabled():
        base = {stk, _find(block_specs, "lev-cam2")}
    elif _unify_cam_head_enabled():
        base = {stk}
    else:
        base = {_find(block_specs, "mem-cam"), stk}
    if _pf_cfm_enabled():
        base = base | {_find(block_specs, "code-select")}
    return base


# ===========================================================================
# CALLING-CONVENTION dispatch (JSR/ENT/ADJ/LEV/LEA/PSH) on the value lanes.  The
# stack MOTION (SP/BP/PC changes) is register arithmetic done here in weights; the
# stack STORE/LOAD (the pushed/popped values) rides the KV head + the driver's
# store-token overlay.  All gated on OP_IS[op].
# ===========================================================================
def callconv_dispatch_rules(L) -> List[FFNRule]:
    ax, sp, bp, stk, pc = L.AX_VAL, L.SP_VAL, L.BP_VAL, L.STK_VAL, L.PC_VAL
    # Frame-offset ops (JSR/ENT/ADJ/LEA) read the CLEAN reconstructed immediate, not
    # the leaky scalar ``L.IMM``: the leak (a fraction of nearby large literals via
    # the imperfect PC one-hot) scaled by 4 rounds the frame byte to the wrong
    # integer (#648/#660).  ``IMM_CLEAN`` is exact for the signed slot offsets and
    # the small positive PC targets these ops carry.
    imm = L.IMM_CLEAN

    def G(op):
        return [(L.OP_IS + op, 0.5, 1.5)]

    rules: List[FFNRule] = []
    # PSH: SP -= 4 ; PC += 1.  The pushed value (AX) rides the driver's store token
    # at MEM[SP-4].  STACK0 mirror is not used for the stack any more.
    rules.append(FFNRule(G(isa.PSH), {sp: LinearExpr.c(-4.0), pc: LinearExpr.c(1.0)}))
    # JSR: SP -= 4 ; PC = imm.  The return PC (old_pc+1) rides the store token.
    rules.append(FFNRule(G(isa.JSR), {
        sp: LinearExpr.c(-4.0),
        pc: LinearExpr.of(imm, 1.0) + LinearExpr.of(pc, -1.0)}))
    # ENT n: MEM[SP-4]=BP (store token) ; SP -= 4 ; BP = SP ; SP -= 4*imm ; PC += 1.
    # Net: BP = SP_old - 4 ; SP = SP_old - 4 - 4*imm.
    rules.append(FFNRule(G(isa.ENT), {
        bp: LinearExpr.of(sp, 1.0) + LinearExpr.of(bp, -1.0) + LinearExpr.c(-4.0),
        sp: LinearExpr.c(-4.0) + LinearExpr.of(imm, -4.0),
        pc: LinearExpr.c(1.0)}))
    # ADJ n: SP += 4*imm ; PC += 1.
    rules.append(FFNRule([(L.OP_IS + ADJ, 0.5, 1.5)], {
        sp: LinearExpr.of(imm, 4.0), pc: LinearExpr.c(1.0)}))
    # NOP: PC += 1 (no state change).  The compiler emits NOP for alignment/padding;
    # without this rule the PC never advances past a NOP and the VM spins forever.
    rules.append(FFNRule([(L.OP_IS + isa.NOP, 0.5, 1.5)], {pc: LinearExpr.c(1.0)}))
    # PRTF: PC += 1 (I/O only, no register state change).  The visible output byte
    # is printf(AX & 0xFF): the driver reads it from the model's decoded AX byte-0
    # and emits it via the think-tag protocol (§Printing, BLOG_SPEC line 851).  AX
    # and the stack are untouched, exactly like NOP for the register transition.
    rules.append(FFNRule([(L.OP_IS + isa.PRTF, 0.5, 1.5)], {pc: LinearExpr.c(1.0)}))
    # LEA o: AX = (BP + 4*imm) & 0xFF ; PC += 1.  imm is the SLOT offset; *4 gives a
    # byte address matching the SP/BP byte-addressing.  Add BP's LOW BYTE only (the
    # 8-bit op masks &0xFF, and BP's higher bytes vanish mod 256) so the downstream
    # single mod-256 fold keeps AX a byte.
    rules.append(FFNRule(G(isa.LEA), {
        ax: LinearExpr.of(L.BP_LOW, 1.0) + LinearExpr.of(imm, 4.0) + LinearExpr.of(ax, -1.0),
        pc: LinearExpr.c(1.0)}))
    # LEV: SP = BP + 8 ; BP = MEM[BP] (loaded into STK_VAL by the stack head) ;
    # PC = MEM[BP+4] (loaded into LEV_RET_VAL by the lev head).  Both loads done
    # in-forward by the two KV heads; here we just route the loaded scalars.
    rules.append(FFNRule(G(isa.LEV), {
        sp: LinearExpr.of(bp, 1.0) + LinearExpr.of(sp, -1.0) + LinearExpr.c(8.0),
        bp: LinearExpr.of(stk, 1.0) + LinearExpr.of(bp, -1.0),
        # PC = LEV_RET_VAL ; the base PC+1 must be cancelled -> write PC delta =
        # (LEV_RET_VAL - pc).  (No base rule fires for LEV, so PC is untouched here
        # except this write.)
        pc: LinearExpr.of(L.LEV_RET_VAL, 1.0) + LinearExpr.of(pc, -1.0)}))
    return rules


# ===========================================================================
# ALU-32 dispatch: on an ALU op the result is written to AX by the ax-mux from the
# op's dedicated RES band; here we just do SP += 4 (pop consumed) + PC += 1.  The
# ax-mux (nibble_alu32.compile_ax_mux) writes AX = RES nibbles; we must NOT also
# write AX here.  So the base ADD/SUB rules are REPLACED by SP/PC-only rules.
# ===========================================================================
def alu32_housekeeping_rules(L) -> List[FFNRule]:
    sp, pc = L.SP_VAL, L.PC_VAL
    rules = []
    ops = list(ALU_OPS)
    if isa.float_ops_enabled():
        # C4_FLOAT_OPS: the float ops (F_ADD/F_SUB/F_MUL/F_DIV) also pop one 32-bit
        # stack operand and advance PC — same housekeeping (SP += 4, PC += 1) as the
        # integer ALU ops; the F_RES -> AX write is done by the float latch.
        ops += [isa.F_ADD, isa.F_SUB, isa.F_MUL, isa.F_DIV]
    for op in ops:
        rules.append(FFNRule([(L.OP_IS + op, 0.5, 1.5)],
                             {sp: LinearExpr.c(4.0), pc: LinearExpr.c(1.0)}))
    return rules


def compile_opcode_decode_pfc(L, dim: int) -> Dict[str, torch.Tensor]:
    """Opcode decode for the COMPLETE VM: the pure-forward decode set PLUS the
    calling-convention ops (JSR/ENT/ADJ/LEV).  ``compile_opcode_decode_pf`` decodes
    BASE_OPS + memory + cmp + bitwise + muldiv, but NOT JSR/ENT/ADJ/LEV — so their
    dispatch rules (all gated on ``OP_IS[op]``) never fire and JSR/ENT/LEV stay
    no-ops.  This widens the decode so the callconv one-hots light up."""
    from .nibble_vm import BASE_OPS
    from .nibble_unified import compile_opcode_decode_ops
    ops = set(BASE_OPS + PF.PF_MEM_OPS +
              [isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE] +
              [isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR] +
              [isa.MUL, isa.DIV, isa.MOD] +
              [isa.JSR, isa.ENT, ADJ, isa.LEV] +      # + calling convention
              [isa.PRTF] +                            # + PRTF (I/O: PC += 1)
              [isa.NOP])                              # + NOP (PC += 1 no-op)
    if isa.float_ops_enabled():
        # C4_FLOAT_OPS: decode the IEEE-754 single ops so OP_IS[F_ADD/F_SUB/F_MUL/
        # F_DIV] light up for the float megablocks + latches.
        ops |= {isa.F_ADD, isa.F_SUB, isa.F_MUL, isa.F_DIV}
    return compile_opcode_decode_ops(L, dim, sorted(ops))


def compile_ax_nib_split(L, dim: int) -> Dict[str, torch.Tensor]:
    """AXB_LO = low nibble of AX_VAL's byte, AXB_HI = high nibble — via ONE ungated
    256-cell one-hot of AX_VAL (triangular pulse) summed with the nibble weights.
    Runs in its OWN block so the writeback (next block) can read AXB_LO/AXB_HI (FFN
    units all see the block INPUT, so producer and consumer must be different
    blocks)."""
    thr = list(range(-1, 257))
    tu = {t: k for k, t in enumerate(thr)}
    n_thr = len(thr)
    axb_lo, axb_hi = L.AXB_LO, L.AXB_HI
    spec = _empty_spec(dim, n_thr + 2)
    u = 0
    relu0 = u
    for t, k in tu.items():
        spec["W_up"][u, L.AX_VAL] = RELU_S
        spec["b_up"][u] = -RELU_S * t
        spec["W_gate"][u, L.ONE] = 1.0
        u += 1
    for lane in (axb_lo, axb_hi):                # self-clear (SET)
        spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, lane] = 1.0
        spec["W_down"][lane, u] += -1.0 / SILU_S; u += 1
    for a in range(256):
        lo, hi = a & 0xF, (a >> 4) & 0xF
        for lane, val in ((axb_lo, lo), (axb_hi, hi)):
            if val:
                spec["W_down"][lane, relu0 + tu[a - 1]] += val / RELU_S
                spec["W_down"][lane, relu0 + tu[a]] += -2.0 * val / RELU_S
                spec["W_down"][lane, relu0 + tu[a + 1]] += val / RELU_S
    return spec


def compile_lea_q_reduce(L, dim: int) -> Dict[str, torch.Tensor]:
    """Compute the EXACT-INTEGER LEA frame quantity ``LEA_Q = BP_low + 4*imm`` (with a
    SIGNED immediate) -- the producer for ``compile_lea_addr_nib`` (the #680
    generalization fix).

    Both parts are read as EXACT INTEGERS from the canonical NIBBLE bands (a pure
    LINEAR read of the residual -- NO silu-recompose scalar residue, which is what
    ``BP_LOW`` / ``IMM_CLEAN`` carried and drifted 4x across a cell boundary at a deep
    read-back, #680):

        LEA_Q = BP_nib0 + 16*BP_nib1                     (BP's low byte, [0,255])
              + 4*IMM_NIB0 + 64*IMM_NIB1                 (|4*imm| low part)
              - 1024*(IMM_NIB1 >= 8)                     (two's-complement SIGN: makes
                                                          4*imm SIGNED so LEA_Q stays
                                                          small, in [-256, 511])

    The SIGN step ``(IMM_NIB1>=8)`` is a clean 0/1 of the exact-integer nibble (its silu
    argument is bounded by ``RELU_S*7.5 = 1500``, so ``1024*silu(...) < 2^24`` is fp32-
    EXACT -- no catastrophic-cancellation error, unlike a mod-256 subtraction whose
    ``256 * silu(RELU_S*sum)`` overflows 2^24).  Keeping ``LEA_Q`` in the small signed
    range means the downstream address decode fires only ~O(few-hundred) saturated silu
    steps, so the GPU-sparse accumulation error (which made the wide-range/large-q
    decode blend AXB_LO to a mid-nibble 11.5 -> byte 235 not 236 -> a0 lost) is
    negligible.  SET (self-clears), gated on OP_IS[LEA] so ``LEA_Q`` is 0 on every
    non-LEA op (byte-identical).

    Units: 1 self-clear + 1 gated-sum + 2 for the sign step (a UNIT silu step pair)."""
    g = L.OP_IS + isa.LEA
    q = L.LEA_Q
    spec = _empty_spec(dim, 1 + 1 + 2)
    u = 0
    # self-clear LEA_Q (SET) so recurrent steps are idempotent.
    spec["W_up"][u, L.ONE] = S
    spec["W_gate"][u, q] = 1.0
    spec["W_down"][q, u] += -1.0 / SILU_S
    u += 1
    # LEA_Q += (BP_low + 4*imm low part), gated on LEA: the GATE carries the exact
    # linear nibble sum; up ~ +S/2 on LEA (silu -> SILU_HALF via the -S/2 bias), ~ -S/2
    # off-LEA (silu ~ 0) so LEA_Q stays 0 on non-LEA ops.
    spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
    spec["W_gate"][u, L.BP + 0] = 1.0
    spec["W_gate"][u, L.BP + 1] = 16.0
    spec["W_gate"][u, L.IMM_NIB + 0] = 4.0
    spec["W_gate"][u, L.IMM_NIB + 1] = 64.0
    spec["W_down"][q, u] += 1.0 / SILU_HALF
    u += 1
    # SIGN: LEA_Q -= 1024*(IMM_NIB1 >= 8).  UNIT step silu(z)-silu(z-1) at edge 7.5 of
    # the exact-integer nibble; gated on LEA (W_gate = g) so it is 0 on non-LEA ops.
    for jj in range(2):
        spec["W_up"][u, L.IMM_NIB + 1] = RELU_S
        spec["W_up"][u, L.ONE] = -RELU_S * 7.5 - (0.0 if jj == 0 else 1.0)
        spec["W_gate"][u, g] = 1.0
        spec["W_down"][q, u] += (-1024.0 if jj == 0 else 1024.0)
        u += 1
    return spec


def _fine_grained_wide_enabled() -> bool:
    """The DOOM-PROVEN fine-grained full-32-bit config: every per-concern wide flag ON
    EXCEPT the buggy two-limb ``C4_VM_WIDTH32`` value substrate.  DEFAULT OFF (all six
    flags off -> golden ``069cc32f`` byte-identical).

    Umbrella = ``C4_CMP32 ∧ C4_SHIFT32 ∧ C4_PC_WIDE ∧ C4_GLOBAL_ADDR32 ∧ C4_SP_WIDE ∧
    C4_DIVMOD_SIGNED``.  This is the config that gives 7/7 signed compares, 0 CMP
    divergences, and correct signed DIV/MOD/SHL/SHR (evidence: /tmp/c4_stage1
    ``regcheck_doomcfg.txt`` 7/7 vs the ``C4_VM_WIDTH32``-substrate ``regcheck_*``
    2/7 — the two-limb value substrate REGRESSES signed CMP ``GT(5,-5)→0`` /
    ``NE(5,-5)→0`` and SHIFT because it recomposes a wide scalar with ~1e-6 residue
    that reaches ~±1 at ~10^6-scale operands and mis-fires the sign/tie).  It is the
    #854 Stage-2 target and the substrate the wide-LEA is now gated onto (below)."""
    from .nibble_vm import _pc_wide_enabled, sp_wide_enabled
    from .nibble_pure_forward import global_addr32_enabled
    return (_cmp32_enabled() and _shift32_enabled() and _pc_wide_enabled()
            and global_addr32_enabled() and sp_wide_enabled()
            and _divmod_signed_enabled())


def _lea_wide_enabled() -> bool:
    """WIDE (full-32-bit) LEA on the CFM/lean path (#824 ``C4_LEA_WIDE`` ported into
    the ``build_pure_forward_complete_model`` builder).  DEFAULT OFF.

    The CFM LEA pipeline (dispatch ``AX = BP_LOW + 4·imm`` -> ``fold-lea`` &0xFF ->
    ``lea-q-reduce`` -> ``lea-addr-nib`` -> ``ax-byte-nib``) UNCONDITIONALLY folds the
    frame address to its LOW BYTE (``AX = (BP + 4·imm) & 0xFF``), regardless of the
    wide flags — the wide-LEA branch that the UNIFIED builder carries
    (``nibble_vm.base_dispatch_rules``: ``lea_bp = full BP if w32``, and the two-limb
    ``AX_LO = BP + IMM_LO``) was never ported here.  So a frame-relative pointer
    ``LEA 40`` off ``BP = 0x10000`` folds to ``0xa0`` instead of the full ``0x100a0``,
    and any load through it reads the wrong cell.

    ON the ``lea-wide`` blocks below recompute the FULL address ``AX = BP + 4·imm``
    across all 4 bytes via a nibble-domain byte-ripple-add (every intermediate < 512 ->
    fp32-EXACT, no wide-scalar recompose), matching the unified builder and the
    full-address Rust c4vm32 reference.  The byte-0 result is identical to the folded
    path (``ax-byte-nib`` already wrote AX nibbles 0,1 = ``(BP_low+4·imm)&0xFF``); the
    wide blocks OVERWRITE AX nibbles 2..7 with bytes 1..3 of the full sum
    (carry-propagated from byte 0), gated on OP_IS[LEA].

    GATE (#854 re-gate, decoupled from the buggy substrate): the wide-LEA activates iff
    ``C4_LEA_WIDE!=0`` AND the DOOM-PROVEN fine-grained wide set is enabled
    (``_fine_grained_wide_enabled()``: ``C4_CMP32 ∧ C4_SHIFT32 ∧ C4_PC_WIDE ∧
    C4_GLOBAL_ADDR32 ∧ C4_SP_WIDE ∧ C4_DIVMOD_SIGNED``).  It requires ``C4_SP_WIDE`` for
    the full BP recompose and ``C4_GLOBAL_ADDR32`` for the wide load address to be
    MEANINGFUL, and it must NOT require the two-limb ``C4_VM_WIDTH32`` value substrate —
    that substrate REGRESSES signed CMP (``GT(5,-5)→0`` / ``NE(5,-5)→0``) and SHIFT
    (see ``_fine_grained_wide_enabled``), while the fine-grained set gives 7/7 signed
    compares and 0 CMP divergences.  This decouples the CORRECT wide-LEA from the broken
    substrate: the folded golden path is unchanged, and the doom fine-grained config now
    gets BOTH correct signed CMP AND the wide frame address.

    OFF (the golden 069cc32f flags-off build, or any config missing a fine-grained
    flag) the ``lea-wide`` scratch bands are never allocated and the blocks are never
    appended, so the layout / every baked weight / the golden fingerprint are
    byte-identical.  Kill-switch ``C4_LEA_WIDE=0`` forces the fold even under the full
    fine-grained set (escape hatch)."""
    if not _fine_grained_wide_enabled():
        return False
    return _os.environ.get("C4_LEA_WIDE", "1") != "0"


def _lea_q_snap_enabled() -> bool:
    """The LEA_Q integer-SNAP (deep-context residue fix, #705).  DEFAULT ON.

    ``compile_lea_q_reduce`` sums the RAW ``IMM_NIB`` nibbles (``4*IMM_NIB0 +
    64*IMM_NIB1``).  Those nibbles are fetched via the PC one-hot, which at DEEP
    context (large emitted token stream, e.g. the 2x2 self-emulation matmul at
    ~6.5k tokens) carries a sub-permille residue: measured ``IMM_NIB0 = 11.9997``
    / ``IMM_NIB1 = 14.9996`` (should be 12 / 15).  Amplified 64x in the sum, this
    drifts ``LEA_Q`` to ``227.875`` (should be an exact ``228``).  The downstream
    ``compile_lea_addr_nib`` decode is only residue-IMMUNE when ``LEA_Q`` is a TRUE
    float32 INTEGER (its half-integer edges then cleanly separate cells); a
    non-integer ``LEA_Q`` sitting ~0.1 below an integer BLENDS the decode across
    cells (``AXB_LO = 4.557`` -> byte 229 instead of 228), so the LEA reads the
    wrong frame address and the following ``LI`` loads ZFOD 0 from the un-stored
    address (the 2x2 matmul step-225 divergence).  Snapping ``LEA_Q`` to the nearest
    integer BEFORE the decode restores the exact-integer invariant the decode relies
    on.  Escape hatch ``C4_LEA_Q_SNAP=0``."""
    import os
    return os.environ.get("C4_LEA_Q_SNAP", "1") != "0"


def compile_lea_q_snap(L, dim: int) -> Dict[str, torch.Tensor]:
    """RECOMPUTE ``LEA_Q`` from INTEGER-ROUNDED ``IMM_NIB`` nibbles (LEA-gated), #705.

    ``compile_lea_q_reduce`` builds ``LEA_Q = BP0 + 16*BP1 + 4*IMM0 + 64*IMM1 - 1024*
    sign`` from the RAW ``IMM_NIB`` nibbles.  Those nibbles are fetched via the PC
    one-hot and at DEEP context carry a sub-permille residue (measured ``IMM0=11.9997``
    / ``IMM1=14.9996``); the ``64x`` amplification drifts ``LEA_Q`` to ``227.875``
    (target ``228``), and the downstream ``compile_lea_addr_nib`` decode — residue-
    IMMUNE only for an EXACT-integer ``LEA_Q`` — then BLENDS across cells (byte 229
    not 228), so the LEA reads the wrong frame address and the next ``LI`` loads ZFOD
    0 (the 2x2 self-emulation matmul step-225 divergence).

    Fix: re-derive ``LEA_Q`` here with each ``IMM_NIB`` nibble first ROUNDED to its
    nearest integer via a NEAREST-INTEGER ROUND ``rnd(nib) = Σ_{v=1..15} step(nib >=
    v-0.5)`` — a staircase of SHARP UNIT silu-steps at the half-integer edges.  Unlike
    a triangular-pulse one-hot (which INTERPOLATES near an integer and would reproduce
    the residue), the half-integer step staircase SNAPS: a nibble with residue << 0.5
    sits >> 1/RELU_S from every edge, so each step is a clean 0/1 and the sum is the
    exact nearest integer.  The edges are LOCAL to a single nibble (0..15) so there is
    NO wide-range silu-tail accumulation, and the coefficient per nibble stays <= 64.
    ``BP0/BP1`` are read directly (measured exact-integer; a heap BP is a clean nibble
    already).  All units are gated on ``OP_IS[LEA]`` so the write is 0 on non-LEA ops
    (``LEA_Q`` is a SET target, so an ungated add would leak) -> byte-identical off-LEA.
    Runs BETWEEN ``lea-q-reduce`` and ``lea-addr-nib``; the decode then sees a TRUE
    integer:

        LEA_Q := BP0 + 16*BP1 + 4*rnd(IMM0) + 64*rnd(IMM1) - 1024*(rnd(IMM1) >= 8)
    """
    g = L.OP_IS + isa.LEA
    q = L.LEA_Q
    # rnd(nib) = Σ_{v=1..15} step(nib >= v-0.5), a nearest-integer ROUND (each step is
    # a SHARP UNIT silu-step at a half-integer edge; a nibble with residue << 0.5 sits
    # >> 1/RELU_S from every edge -> each step is a clean 0/1, so the sum is the exact
    # nearest integer, unlike the triangular pulse which INTERPOLATES near an integer).
    # A step at edge e needs TWO relu units: silu(RELU_S*(nib-e)) - silu(...-1).
    edges = [v - 0.5 for v in range(1, 16)]     # 15 half-integer edges per nibble
    n_step = len(edges)
    # units: 2 nibbles * (15 edges * 2 relu) + clear + BP-add.
    spec = _empty_spec(dim, 2 * (n_step * 2) + 1 + 1)
    u = 0
    step_base = {}
    for bi, src in enumerate((L.IMM_NIB + 0, L.IMM_NIB + 1)):
        step_base[bi] = {}
        for e in edges:
            base = u
            for j in range(2):
                spec["W_up"][u, src] = RELU_S
                spec["b_up"][u] = -RELU_S * e - (0.0 if j == 0 else 1.0)
                spec["W_gate"][u, g] = 1.0        # gate on LEA -> 0 off-LEA
                u += 1
            step_base[bi][e] = base               # (base, base+1) = the +/- relu pair
    # SET: clear LEA_Q (gated on LEA).
    clr = u
    spec["W_up"][clr, g] = S; spec["b_up"][clr] = -S * 0.5
    spec["W_gate"][clr, q] = 1.0
    spec["W_down"][q, clr] += -1.0 / SILU_HALF
    u += 1
    # LEA-gated linear BP low-byte re-add (BP nibbles are clean integers).
    add = u
    spec["W_up"][add, g] = S; spec["b_up"][add] = -S * 0.5
    spec["W_gate"][add, L.BP + 0] = 1.0
    spec["W_gate"][add, L.BP + 1] = 16.0
    spec["W_down"][q, add] += 1.0 / SILU_HALF
    u += 1
    # rounded-nibble re-add: LEA_Q += 4*rnd(IMM0) + 64*rnd(IMM1).  rnd(nib) is the sum
    # of unit steps; each edge v-0.5 contributes ``scale`` to LEA_Q iff nib >= v-0.5.
    # (The sign fold -1024*(IMM1>=8) is applied SEPARATELY below as one extra step.)
    for bi, scale in ((0, 4.0), (1, 64.0)):
        for e in edges:                            # e = v-0.5 for v=1..15
            p, m = step_base[bi][e], step_base[bi][e] + 1
            spec["W_down"][q, p] += scale          # +step (silu@e)
            spec["W_down"][q, m] += -scale         # -step (silu@e-1) -> UNIT step
    # two's-complement sign for IMM1: subtract 1024 when rnd(IMM1) >= 8, i.e. when the
    # step at edge 7.5 is ON.  Reuse the bi=1 step at e=7.5 (v=8 edge) already built.
    p, m = step_base[1][7.5], step_base[1][7.5] + 1
    spec["W_down"][q, p] += -1024.0
    spec["W_down"][q, m] += 1024.0
    return spec


def compile_lea_addr_nib(L, dim: int) -> Dict[str, torch.Tensor]:
    """LEA-ONLY, residue-immune frame-address nibbles (the #648 + #680 fix).

    ``compile_ax_nib_split`` derives AXB_LO/AXB_HI from the SCALAR ``AX_VAL``.  On a
    LEA the frame byte is ``(BP_low + 4*imm) mod 256``.  The naive scalar path is
    corrupted by fp residues that surface when the frame byte is 16-aligned (a malloc'd
    pointer stored to such a frame local is read back from the wrong cell and lost,
    #648) OR sits NEXT TO a rounding edge (the self-emul matvec reads a0 from BP-2 =
    addr 236 = 0xEC, adjacent to edge 235.5 -> decoded 235 -> a0 lost, #680):

      (a) the dispatch SET ``AX_VAL += BP_LOW + 4*imm - AX_VAL_old`` scaled by the
          opcode gate leaves ``(1-k)*AX_VAL_old`` when the prior AX held a heap pointer;
      (b) the ``IMM`` scalar leaks a fraction of a nearby large literal via the PC
          one-hot; and (c) even ``IMM_CLEAN`` / ``BP_LOW`` (silu-recomposed scalars)
          carry a sub-milli residue that, amplified 4x, drifts q across a cell boundary
          at a DEEP read-back (matvec LEA -2 at BP=244: IMM_CLEAN=-1.988 -> q=236.05).

    Fix (#680 generalization): the frame byte is computed EXACT-INTEGER and REDUCED to
    the SIGNED integer ``LEA_Q = BP_low + 4*imm`` upstream (``compile_lea_q_reduce``)
    entirely from the canonical NIBBLE bands (a pure linear read, no silu-recompose
    residue), so this block just does a nearest-integer decode of ``LEA_Q`` (taking the
    low byte via ``a & 0xFF``).  Two reasons this GENERALIZES where the old path did not:
      * INTEGER INPUT: ``LEA_Q`` is a true float32 integer, so every ``|LEA_Q - e|``
        is EXACTLY >= 0.5 at a half-integer edge -> the clamp is a clean 0/1 (no residue
        can cross the edge -- kills the #680 deep-read-back miss AT SOURCE); and
      * SMALL q: the SIGNED immediate keeps ``LEA_Q`` in ``[-256, 511]`` (not the raw
        two's-complement ~1260), so only ~O(few-hundred) saturated silu steps sum on the
        GPU sparse path -- the accumulation error that grows with q (and blended AXB_LO
        to a mid-nibble 11.5) stays negligible.
    OVERWRITES AXB_LO/AXB_HI, gated on OP_IS[LEA] (0 on every non-LEA op ->
    byte-identical).  Runs AFTER ``lea-q-reduce`` and BEFORE ``ax-byte-nib``."""
    g = L.OP_IS + isa.LEA
    axb_lo, axb_hi = L.AXB_LO, L.AXB_HI
    q = L.LEA_Q                            # exact-integer signed frame qty in [-256, 511]
    q_lo, q_hi = -256, 512
    edges = [a - 0.5 for a in range(q_lo, q_hi + 1)]
    eu = {e: k for k, e in enumerate(edges)}
    n_edge = len(edges)
    spec = _empty_spec(dim, 2 * n_edge + 2)
    u = 0
    step0 = u
    for e in edges:                       # two relu units -> one clamped step per edge
        for j in range(2):
            # decode the EXACT-INTEGER LEA_Q (small range): cell a = s_{a-.5} - s_{a+.5}.
            spec["W_up"][u, q] = RELU_S
            spec["b_up"][u] = -RELU_S * e - (0.0 if j == 0 else 1.0)
            spec["W_gate"][u, g] = 1.0    # gate on OP_IS[LEA] (0/1) so the step is
            u += 1                        # zero on non-LEA steps
    def step_units(e):                    # (idx_of_+relu, idx_of_-relu) for edge e
        base = step0 + 2 * eu[e]
        return base, base + 1
    for lane in (axb_lo, axb_hi):         # SET: clear AXB (gated on LEA) before re-add
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, lane] = 1.0
        spec["W_down"][lane, u] += -1.0 / SILU_HALF; u += 1
    for a in range(q_lo, q_hi):           # cell a = step(a-0.5) - step(a+0.5)
        byte = a & 0xFF
        lo, hi = byte & 0xF, (byte >> 4) & 0xF
        p_lo, m_lo = step_units(a - 0.5)  # +step at lower edge
        p_hi, m_hi = step_units(a + 0.5)  # -step at upper edge
        # step(q>=e) = silu(RELU_S*(q-e)) - silu(RELU_S*(q-e)-1) saturates to UNIT
        # height, so route with coefficient ``val`` directly (no /RELU_S).
        for lane, val in ((axb_lo, lo), (axb_hi, hi)):
            if val:
                spec["W_down"][lane, p_lo] += val
                spec["W_down"][lane, m_lo] += -val
                spec["W_down"][lane, p_hi] += -val
                spec["W_down"][lane, m_hi] += val
    return spec


# LEA immediate window: IMM_CLEAN is exact in [-2048, 2047] (CLEAN_NIBS=3), so
# 4*imm is in [-8192, 8188].  ``LEA_IMM4_BIAS`` shifts 4*imm into the non-negative
# range [0, 16380] so a SMALL-coefficient (<=256) staircase computes floor(4*imm/256)
# WITHOUT the fp32-lethal 65536-scale two's-complement indicator (a 65536*[.] step's
# ~1e-5 silu residue amplifies to ~0.8 absolute — the off-by-one seen with the naive
# u16 form).  Every coefficient here stays <= 256, so the residue amplification is
# < 3e-3 << 0.5 and the split is fp32-EXACT.
_LEA_IMM4_BIAS = 8192
_LEA_F_OFF = _LEA_IMM4_BIAS // 256          # 32 : floor((4*imm+BIAS)/256) - 32 = floor(4*imm/256)


def compile_lea_wide_operands(L, dim: int) -> Dict[str, torch.Tensor]:
    """WIDE LEA, block 1/3 (C4_LEA_WIDE): set operand A = BP bytes and the SIGNED
    ``f = floor(4*imm / 256)`` scratch (the low addend split runs the next block),
    LEA-gated (every band 0 on a non-LEA op -> byte-identical off-LEA).

      LEAW_A[i] = BP byte i = BP_nib[2i] + 16*BP_nib[2i+1]   (i=0..3, clean nibbles)
      LEAW_U16  = f = floor(4*imm / 256)  in [-32, 31]  (signed; computed via the
                  bias-shifted small-coefficient staircase, see ``_LEA_IMM4_BIAS``).

    ``4*imm`` from the EXACT signed ``IMM_CLEAN`` (leak-free, reconstructed in the
    ``imm-clean`` block; |4*imm| <= 8188).  The two low addend bytes (B0/B1) and the
    sign-fill bytes (B2/B3) are finished in ``compile_lea_wide_split`` from this f +
    4*imm, all with coefficients <= 256 (fp32-EXACT — no 65536-scale indicator)."""
    g = L.OP_IS + isa.LEA
    a, f = L.LEAW_A, L.LEAW_U16
    imm4b = {L.IMM_CLEAN: 4.0, L.ONE: float(_LEA_IMM4_BIAS)}   # (4*imm + BIAS) in [0,16380]
    spec = _empty_spec(dim, 4 * 2 + 2 + 64 * 2)
    u = 0

    def clear(band):
        nonlocal u
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, band] = 1.0
        spec["W_down"][band, u] += -1.0 / SILU_HALF; u += 1

    def add_lin(terms, const, band, scale):
        nonlocal u
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        for bd, c in terms.items():
            spec["W_gate"][u, bd] += c
        spec["b_gate"][u] += const
        spec["W_down"][band, u] += scale / SILU_HALF; u += 1

    def floor_div_ge(terms, m, kmax, band, scale):    # band += scale*floor((terms)/m)
        nonlocal u
        for k in range(1, kmax + 1):
            c, w = k * m - 0.5, 0.2
            for tt, tc in ((c - w, scale), (c, -scale)):
                for bd, cf in terms.items():
                    spec["W_up"][u, bd] += RELU_S * cf
                spec["b_up"][u] += -RELU_S * tt
                spec["W_gate"][u, g] = 1.0
                spec["W_down"][band, u] += tc / (RELU_S * w); u += 1

    # A[i] = BP byte i (clean BP nibbles).
    for i in range(4):
        clear(a + i)
        add_lin({L.BP + 2 * i: 1.0, L.BP + 2 * i + 1: 16.0}, 0.0, a + i, 1.0)
    # f = floor(4*imm/256) = floor((4*imm+BIAS)/256) - BIAS/256.  (4*imm+BIAS) in [0,16380]
    # -> floor <= 63 (kmax=64).  All coefficients <= 256 -> fp32-exact.
    clear(f)
    add_lin({L.ONE: 1.0}, 0.0, f, -float(_LEA_F_OFF))          # - 32
    floor_div_ge(imm4b, 256, 64, f, 1.0)                      # + floor((4*imm+BIAS)/256)
    return spec


def compile_lea_wide_split(L, dim: int) -> Dict[str, torch.Tensor]:
    """WIDE LEA, block 2/3 (C4_LEA_WIDE): finish the SIGN-EXTENDED addend bytes from
    ``4*imm`` and the signed ``f = floor(4*imm/256)`` scratch (computed the previous
    block), LEA-gated.  All coefficients <= 256 (fp32-EXACT):

        neg     = [4*imm < 0] = [f < 0]                       (sharp 0/1 step)
        LEAW_B0 = 4*imm - 256*f          in [0, 255]          (low byte)
        LEAW_B1 = f + 256*neg            in [0, 255]          (second byte, two's-compl)
        LEAW_B2 = LEAW_B3 = 255*neg      (sign fill: 0x00 for imm>=0, 0xFF for imm<0)
    """
    g = L.OP_IS + isa.LEA
    b, f = L.LEAW_B, L.LEAW_U16
    imm4 = {L.IMM_CLEAN: 4.0}
    spec = _empty_spec(dim, 3 + 2 + 2 + 2 * 2 + 2 * 3)
    u = 0

    def clear(band):
        nonlocal u
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, band] = 1.0
        spec["W_down"][band, u] += -1.0 / SILU_HALF; u += 1

    def add_lin(terms, const, band, scale):
        nonlocal u
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        for bd, c in terms.items():
            spec["W_gate"][u, bd] += c
        spec["b_gate"][u] += const
        spec["W_down"][band, u] += scale / SILU_HALF; u += 1

    def step_lt0(terms, band, scale):              # band += scale*[form < 0] = scale*(1-[form>=0])
        nonlocal u
        add_lin({L.ONE: 1.0}, 0.0, band, scale)    # + scale
        c, w = -0.5, 0.2                           # [form >= 0] ramp at -0.5
        for k, t in enumerate((c - w, c)):
            for bd, cf in terms.items():
                spec["W_up"][u, bd] += RELU_S * cf
            spec["b_up"][u] += -RELU_S * t
            spec["W_gate"][u, g] = 1.0
            spec["W_down"][band, u] += (-scale if k == 0 else scale) / (RELU_S * w)
            u += 1

    # B0 = 4*imm - 256*f  (f = floor(4*imm/256), so this is 4*imm mod 256 in [0,255]).
    clear(b + 0)
    add_lin(imm4, 0.0, b + 0, 1.0)                 # + 4*imm
    add_lin({f: 1.0}, 0.0, b + 0, -256.0)          # - 256*f
    # B1 = f + 256*neg   (neg = [4*imm<0]; f in [-32,31] -> B1 in [0,255]).
    clear(b + 1)
    add_lin({f: 1.0}, 0.0, b + 1, 1.0)             # + f
    step_lt0(imm4, b + 1, 256.0)                   # + 256*[4*imm<0]
    # B2 = B3 = 255*neg  (sign fill).
    for hb in (b + 2, b + 3):
        clear(hb)
        step_lt0(imm4, hb, 255.0)
    return spec


def compile_lea_wide_byte(L, dim: int, i: int) -> Dict[str, torch.Tensor]:
    """WIDE LEA, byte ``i`` of the ripple ADD ``AX = BP + 4*imm`` (C4_LEA_WIDE),
    LEA-gated.  ONE block per byte (like the ALU add chain) because the carry lane
    ``LEAW_C[i+1]`` written here is READ as ``LEAW_C[i]`` by the NEXT byte's block —
    the carry ripples through the residual across blocks (a unit only sees the block
    INPUT, so producer and consumer must be different blocks).

        sum_i = A[i] + B[i] + carry_i        (carry_0 = 0)
        AX_nib[2i]   = sum_i mod 16
        AX_nib[2i+1] = floor(sum_i/16) mod 16
        LEAW_C[i+1]  = [sum_i >= 256]         (carry-out)
    Every sum_i < 512 -> fp32-EXACT.  SET (clears each dst first), gated on OP_IS[LEA]
    so a non-LEA op is untouched (byte-identical off-LEA)."""
    g = L.OP_IS + isa.LEA
    a, b, cc = L.LEAW_A, L.LEAW_B, L.LEAW_C
    spec = _empty_spec(dim, 200)
    u = 0
    terms = {a + i: 1.0, b + i: 1.0}
    if i > 0:
        terms[cc + i] = 1.0                         # + carry_in from previous byte's block

    def clear(band):
        nonlocal u
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, band] = 1.0
        spec["W_down"][band, u] += -1.0 / SILU_HALF; u += 1

    def step_ge(thr, band, scale):                  # band += scale*[sum >= thr]
        nonlocal u
        c, w = thr - 0.5, 0.2
        for k, t in enumerate((c - w, c)):
            for bd, cf in terms.items():
                spec["W_up"][u, bd] += RELU_S * cf
            spec["b_up"][u] += -RELU_S * t
            spec["W_gate"][u, g] = 1.0
            spec["W_down"][band, u] += (scale if k == 0 else -scale) / (RELU_S * w)
            u += 1

    def floor_div(m, kmax, band, scale):            # band += scale*floor(sum/m)
        for k in range(1, kmax + 1):
            step_ge(k * m, band, scale)

    # carry-out first (needed for the next byte): [sum >= 256]
    clear(cc + i + 1)
    step_ge(256, cc + i + 1, 1.0)
    # low nibble = sum - 16*floor(sum/16)   (sum in [0,511])
    clear(L.AX + 2 * i)
    for bd, cf in terms.items():                    # + sum
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, bd] = cf
        spec["W_down"][L.AX + 2 * i, u] += 1.0 / SILU_HALF; u += 1
    floor_div(16, 32, L.AX + 2 * i, -16.0)
    # high nibble = floor(sum/16) - 16*floor(sum/256)
    clear(L.AX + 2 * i + 1)
    floor_div(16, 32, L.AX + 2 * i + 1, 1.0)
    floor_div(256, 2, L.AX + 2 * i + 1, -16.0)
    return spec


def compile_ax_byte_to_nibbles(L, dim: int, ops) -> Dict[str, torch.Tensor]:
    """When any op in ``ops`` (byte-producing: IMM/LEA/CMP/bitwise/LI) is active,
    SET AX nibble 0 = AXB_LO, nibble 1 = AXB_HI (computed by ``compile_ax_nib_split``
    in the previous block), and clear AX nibbles 2..7 — so the AX nibble band is the
    canonical 8-bit result and the driver decodes AX from it uniformly."""
    axb_lo, axb_hi = L.AXB_LO, L.AXB_HI
    spec = _empty_spec(dim, len(ops) * 8 + len(ops) * 2)
    u = 0
    for op in ops:
        g = L.OP_IS + op
        for j in range(8):                       # clear AX[0..7] gated
            spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
            spec["W_gate"][u, L.AX + j] = 1.0
            spec["W_down"][L.AX + j, u] += -1.0 / SILU_HALF; u += 1
        for nb, src in ((0, axb_lo), (1, axb_hi)):   # nib0=AXB_LO, nib1=AXB_HI
            spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
            spec["W_gate"][u, src] = 1.0
            spec["W_down"][L.AX + nb, u] += 1.0 / SILU_HALF; u += 1
    return spec


def _noop_spec(L, dim: int) -> Dict[str, torch.Tensor]:
    """A single-unit no-op FFN (writes nothing) — used as the FFN of a cfm block
    whose only job is to carry the code-fetch CAM head."""
    spec = _empty_spec(dim, 1)
    spec["W_up"][0, L.ONE] = 0.0
    spec["W_gate"][0, L.ONE] = 0.0
    return spec


def compile_pc_fetch_cfm(L, dim: int) -> Dict[str, torch.Tensor]:
    """CODE-FROM-MEMORY (C4_PF_CFM) pc-fetch: the fetch@PC *query* setup, replacing
    the O(code_size) PC-one-hot table (``compile_pc_fetch``).

    Computes:
      * ``AX_ZERO = (AX == 0)`` (the BZ/BNZ predicate — still needed).
      * ``CODE_QRY_BIN`` = bits(PC) from the PC nibble band (the CAM query address).
      * ``IS_FETCH = 1`` (arms the code CAM every step).
      * clears ``OP_VAL`` and ``IMM_NIB`` (SET), so the code CAM's additive write
        (next block) lands cleanly on a zeroed lane.
    NO PC one-hot is baked, so the residual dim is INDEPENDENT of code_size."""
    from .nibble_vm import vm_two_limb
    # CODE_QRY_BIN <- PC nibbles (first ceil(CODE_ADDR_BITS/4) nibbles suffice).  Cap
    # the written bits at CODE_ADDR_BITS so a non-nibble-aligned width (e.g. 13, 17,
    # 18) does not spill the high bits of the top nibble into the NEXT band.
    n_pc_nib = (CODE_ADDR_BITS + 3) // 4
    qexp = compile_nibble_addr_expand(L, L.PC, L.CODE_QRY_BIN, dim,
                                      n_nibbles=n_pc_nib, n_bits=CODE_ADDR_BITS)
    base_units = qexp["W_up"].shape[0]
    # units: AX_ZERO(2: clear+ramp) + IS_FETCH(2: clear+set1) + clear OP_VAL(1)
    #        + clear IMM_NIB band(IMM_NIBS)
    spec = _empty_spec(dim, base_units + 2 + 2 + 1 + IMM_NIBS)
    for k in ("W_up", "W_gate", "b_up"):
        spec[k][:base_units] = qexp[k]
    spec["W_down"][:, :base_units] = qexp["W_down"]
    u = base_units
    # AX_ZERO = relu(1 - AX_VAL)  (two-limb: relu(1 - AX_LO - AX_HI)).  Self-cleared
    # first so it is idempotent across recurrent steps.
    spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.AX_ZERO] = 1.0
    spec["W_down"][L.AX_ZERO, u] += -1.0 / SILU_S; u += 1
    if vm_two_limb():
        spec["W_up"][u, L.AX_LO] = -RELU_S
        spec["W_up"][u, L.AX_HI] = -RELU_S
    else:
        spec["W_up"][u, L.AX_VAL] = -RELU_S
    spec["b_up"][u] = RELU_S * 1.0
    spec["W_gate"][u, L.ONE] = 1.0
    spec["W_down"][L.AX_ZERO, u] += 1.0 / RELU_S; u += 1
    # IS_FETCH := 1 (SET = self-clear then +1).
    spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.IS_FETCH] = 1.0
    spec["W_down"][L.IS_FETCH, u] += -1.0 / SILU_S; u += 1
    spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.ONE] = 1.0
    spec["W_down"][L.IS_FETCH, u] += 1.0 / SILU_S; u += 1
    # clear OP_VAL + IMM_NIB band (SET, so the CAM writes cleanly next block).
    spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.OP_VAL] = 1.0
    spec["W_down"][L.OP_VAL, u] += -1.0 / SILU_S; u += 1
    for j in range(IMM_NIBS):
        spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.IMM_NIB + j] = 1.0
        spec["W_down"][L.IMM_NIB + j, u] += -1.0 / SILU_S; u += 1
    return spec


def compile_imm_nib_fetch(L, dim: int) -> Dict[str, torch.Tensor]:
    """IMM_NIB[j] = Σ_i PC_IS[i]·CODE_IMM_NIB[i][j]  (j=0..7) — the same bilinear
    PC-one-hot product-select the scalar-IMM fetch uses, applied to the immediate's
    8 static program-data nibbles.  Self-clears IMM_NIB first (SET)."""
    n = L.code_size
    spec = _empty_spec(dim, IMM_NIBS + n * IMM_NIBS)
    u = 0
    for j in range(IMM_NIBS):                        # self-clear IMM_NIB (SET)
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, L.IMM_NIB + j] = 1.0
        spec["W_down"][L.IMM_NIB + j, u] += -1.0 / SILU_S
        u += 1
    for i in range(n):
        for j in range(IMM_NIBS):
            spec["W_up"][u, L.PC_IS[i]] = S
            spec["W_gate"][u, L.CODE_IMM_NIB[i] + j] = 1.0
            spec["W_down"][L.IMM_NIB + j, u] += 1.0 / SILU_S
            u += 1
    return spec


def compile_imm_clean(L, dim: int) -> Dict[str, torch.Tensor]:
    """IMM_CLEAN = the EXACT signed immediate, reconstructed from ``IMM_NIB`` (the
    leak-free per-nibble fetch) — the ROOT fix for the frame-address IMM leak
    (#648/#660).

    The scalar ``IMM = Σ_i PC_IS[i]·CODE_IMM[i]`` leaks a fraction of every NEARBY
    large literal (e.g. ``0x20000``) through the imperfect PC one-hot; with the
    corpus's ~43 big literals in flight the accumulated residue exceeds 0.5, so any
    op scaling it by 4 (``AX/SP += 4·imm``) rounds the frame byte to the wrong
    integer.  ``IMM_NIB[j]`` is fetched the SAME way, but each nibble is a small
    integer 0..15, so a 16-cell one-hot (triangular pulse, exact at each integer)
    ROUNDS each nibble to its nearest cell and kills the leak per-nibble.

    We reconstruct a bounded-width SIGNED value from the low ``CLEAN_NIBS`` nibbles
    (two's-complement, ``CLEAN_NIBS·4`` bits): the top of those nibbles is treated as
    SIGNED (cell ``a>=8`` -> ``a-16``), so a negative slot offset ``-k`` (stored as
    ``0xF..FC``) reconstructs to exactly ``-k`` and a small positive PC target /
    offset to itself.  ``CLEAN_NIBS`` is chosen so every frame-offset op's immediate
    (LEA/ENT/ADJ signed slot counts, |·| <= a few; JSR positive PC targets < code
    size) fits its signed range — and, crucially, so the recompose coefficients
    (``<= 16^CLEAN_NIBS``) stay SMALL: a large coefficient (e.g. ``16^4``) amplifies
    the silu unit's ~1e-6 relative error to ~0.5 ABSOLUTE at the cell, which would
    re-introduce a half-integer error.  With ``CLEAN_NIBS=3`` (signed range
    ``[-2048, 2047]``) the max coefficient is ``16^3=4096`` and the result is
    fp32-EXACT for every corpus immediate (unit-tested).

    Result written ONCE into the DEDICATED never-share ``IMM_CLEAN`` scalar (NOT an
    in-place overwrite of the leaky ``IMM`` dim — a fresh dim written once sidesteps
    the compact_alloc liveness interaction that neutralised the in-place attempt).
    Ungated (IMM_CLEAN is only READ by the frame-offset ops, which gate their own
    writes on OP_IS)."""
    # 3 low nibbles: signed 12-bit range [-2048, 2047] covers every corpus LEA/ENT/
    # ADJ slot offset (|·| <= a few) and every JSR PC target (< code_size).  Kept
    # small so the recompose coefficients (<= 16^3) do not amplify the silu residual.
    #
    # LARGE-CODE (doom, code_size 3976): a JMP/JSR/branch PC target can EXCEED the
    # 3-nibble signed range 2047 (doom's ``JMP 3652`` -> its top nibble 0xE>=8 would be
    # mis-read as SIGN -> -444, corrupting the jump).  Widen ``n_nib`` so the SIGNED
    # range ``[-2^(4n-1), 2^(4n-1))`` covers ``code_size`` positively — one more nibble
    # per 4 address bits.  The corpus (code_size<=few-hundred, all targets<2047) keeps
    # ``n_nib=3`` -> the golden IMM_CLEAN block is byte-IDENTICAL.  The extra-nibble
    # coefficient (<= 16^3=4096) amplifies the silu ~1e-6 rel error to ~4e-3 absolute
    # per cell (<< 0.5), so the recompose stays fp32-exact for the integer nibbles.
    _tcs = getattr(L, "true_code_size", L.code_size)
    _need = 1
    while _tcs >= (1 << (4 * _need - 1)):        # signed top nibble -> range 2^(4n-1)
        _need += 1
    n_nib = min(max(3, _need), IMM_NIBS)
    thr = list(range(-1, 17))
    tu = {t: k for k, t in enumerate(thr)}
    n_thr = len(thr)
    # one relu bank per nibble (16-cell one-hot) + one self-clear unit for IMM_CLEAN.
    spec = _empty_spec(dim, n_nib * n_thr + 1)
    u = 0
    relu0 = {}
    for j in range(n_nib):
        relu0[j] = u
        for t, k in tu.items():
            spec["W_up"][u, L.IMM_NIB + j] = RELU_S
            spec["b_up"][u] = -RELU_S * t
            spec["W_gate"][u, L.ONE] = 1.0
            u += 1
    # self-clear IMM_CLEAN (SET) so recurrent steps are idempotent.
    spec["W_up"][u, L.ONE] = S
    spec["W_gate"][u, L.IMM_CLEAN] = 1.0
    spec["W_down"][L.IMM_CLEAN, u] += -1.0 / SILU_S
    u += 1
    # recompose: IMM_CLEAN += Σ_j Σ_a one-hot_j(a)·coeff(j,a).  The TOP of the
    # reconstructed nibbles is SIGNED (a-16 when a>=8), so ``0xF..FC`` -> ``-k``.
    base = 1
    top = n_nib - 1
    for j in range(n_nib):
        r0 = relu0[j]
        for a in range(16):
            av = (a - 16) if (j == top and a >= 8) else a   # signed top nibble
            coeff = av * base
            if coeff:
                spec["W_down"][L.IMM_CLEAN, r0 + tu[a - 1]] += coeff / RELU_S
                spec["W_down"][L.IMM_CLEAN, r0 + tu[a]] += -2.0 * coeff / RELU_S
                spec["W_down"][L.IMM_CLEAN, r0 + tu[a + 1]] += coeff / RELU_S
        base *= 16
    return spec


def compile_imm_ax_nibbles(L, dim: int) -> Dict[str, torch.Tensor]:
    """On IMM, SET all 8 AX nibbles = IMM_NIB (the fetched immediate nibbles), so a
    full 32-bit literal lands in the canonical AX nibble band.  Gated on OP_IS[IMM].
    Runs AFTER the byte-nib writeback (which only wrote nibbles 0,1) and overwrites
    all 8 with the full value."""
    n_ax = min(8, IMM_NIBS)                          # AX carries at most 8 nibbles (32 bits)
    spec = _empty_spec(dim, 8 + n_ax)
    u = 0
    g = L.OP_IS + isa.IMM
    for j in range(8):                              # clear ALL 8 AX nibbles gated
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.AX + j] = 1.0
        spec["W_down"][L.AX + j, u] += -1.0 / SILU_HALF; u += 1
    for j in range(n_ax):                            # + IMM_NIB[j] into AX[j] (rest 0)
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.IMM_NIB + j] = 1.0
        spec["W_down"][L.AX + j, u] += 1.0 / SILU_HALF; u += 1
    return spec


def _fold_ax_gated(L, dim: int, ops) -> Dict[str, torch.Tensor]:
    """Fold AX_VAL mod 256 ONLY when one of ``ops`` is active (LEA masks &0xFF).
    The 32-bit ALU result must NOT be folded, so the base global fold is replaced
    by this opcode-gated one.  Sharp ramp AX_VAL -= 256·[AX_VAL>=256], gated."""
    M, w = 256, 0.2
    lo = M - 0.5
    spec = _empty_spec(dim, len(ops) * 2)
    u = 0
    for op in ops:
        gu = L.OP_IS + op
        for i, thr in enumerate((lo, lo + w)):
            spec["W_up"][u, L.AX_VAL] = RELU_S
            spec["b_up"][u] = -RELU_S * thr
            spec["W_gate"][u, gu] = 1.0          # gate = OP_IS[op] (0 or 1)
            spec["W_down"][L.AX_VAL, u] += (-M if i == 0 else M) / (RELU_S * w)
            u += 1
    return spec


# ===========================================================================
# BUILD the complete pure-forward model.
# ===========================================================================
def build_pure_forward_complete_model(code_size: int = 32,
                                      recurrent_divmod: bool = False):
    """Assemble the complete pure-forward VM: frame ingest + LI/LC KV head +
    stack-pop KV head + the 32-bit ALU FFN blocks + callconv + bitwise + the
    base-16 long-division DIV/MOD blocks, all as persistent weights applied by
    ``model.forward``.  Returns ``(model, L)``.

    This is the SINGLE canonical full-VM interpreter: it ALWAYS folds the
    complete op set (ADD/SUB/MUL + DIV/MOD + OR/XOR/AND/SHL/SHR + cmp + memory +
    callconv).  There is no reduced / "lean" op-subset construction — the former
    ``include_bitwise`` / ``include_divmod`` split flags have been removed so the
    build is one byte-identical model with every opcode present.  (Skipping the
    ~300 divmod blocks when the *step's* opcode is not DIV/MOD is a runtime
    block-dispatch concern — see the ``_apply_order`` seam and the block-MoE
    integration note below — NOT a build-time op-subset toggle.)

    ``recurrent_divmod`` (default False) folds DIV/MOD as a RECURRENT step: the
    8 long-division iterations are unrolled into ONE reused iteration BODY (21
    blocks) instead of 168 distinct blocks, and the block-application loop applies
    that body 8 times (threading the running remainder R + digit-index counter IT
    through the residual across the reused applications — exactly as the VM step
    itself is recurrent, threading PC/AX/SP).  ``L._apply_order`` then indexes the
    physical blocks in the FULL application sequence (the body indices repeat);
    the physical block count (what the model STORES) drops 262 -> 115 for divmod,
    while the forward applies the same 262-long sequence.  Byte-identical DIV/MOD
    results to the unrolled build (gadget gate: 24/24).  This is a compute-SHAPE
    toggle (same op set, fewer stored blocks), NOT an op-subset toggle."""
    # 20 ingest + LI head + stack-pop head + lev head = N_ROLES+3.  With
    # C4_UNIFY_CAM_ONE the LEV ret-PC read rides the SAME merged head INDEX in a
    # second cam block (see _unify_cam_one_enabled), so the dedicated LEV head index
    # is dropped: N_ROLES+2.  (C4_UNIFY_CAM_HEAD alone keeps N_ROLES+3 — it drops the
    # standalone mem-cam head but reuses its freed slot layout byte-identically.)
    _one = _unify_cam_one_enabled()
    _wide = ingest_wide_enabled()
    _cfm = _pf_cfm_enabled()
    # CODE-FROM-MEMORY adds ONE head (the code-fetch CAM); the per-slot table is
    # vestigial (1 slot) since the program lives in the KV code frames.
    n_heads = (N_ROLES + 2) if _one else (N_ROLES + 3)
    if _cfm:
        n_heads += 1
    pf_code_size = 1 if _cfm else code_size
    L = PureForwardCompleteLayout(pf_code_size, n_heads=n_heads)
    L.true_code_size = code_size            # the real program length (CAM addr space)
    if _wide:
        # 80 fresh dims (PREROUTE 40 + GATHER 40) for the 1-query/1-KV wide ingest.
        # Allocated BEFORE dim is fixed; flag-OFF this is never called ⇒ the golden
        # ``_fingerprint_build`` hash 069cc32f is unchanged (was 8f4dd780 before the
        # 2026-07 c4-faithful SHR-arithmetic + signed-LC re-baseline).
        extend_layout_for_wide_ingest(L)
    A.extend_layout_for_alu32(L, recurrent_divmod=recurrent_divmod)  # ALU scratch bands
    from . import nibble_bitwise as _bw
    _bw.extend_layout_for_bitwise(L)
    # NATIVE FLOAT (C4_FLOAT_OPS, default OFF): allocate the IEEE-754 single ALU
    # scratch bands.  Flag OFF -> extend_layout_for_fp32 is never called ⇒ L.D and
    # every band offset are byte-identical to the golden 069cc32f build.
    _float = isa.float_ops_enabled()
    if _float:
        from . import nibble_fp32 as _fp
        _fp.extend_layout_for_fp32(L)
    # SHIFTER scratch: allocate the active shifter's private per-op scratch bands NOW,
    # before ``dim`` is fixed, so the shift blocks (compiled inside build_bitwise_blocks
    # at the fixed ``dim``) address valid residual dims.  BARREL (C4_BARREL_SHIFT=1)
    # takes precedence over the TIGHT shifter (default ON).  BARREL OFF -> the tight
    # pre-extension is UNCHANGED, so the golden fingerprint 069cc32f is untouched.
    if _bw.barrel_shift_enabled():
        if _bw.barrel_unify_enabled():
            _bw.extend_layout_for_barrel_unify(L)       # ONE shared scratch set
        else:
            for _op in (isa.SHL, isa.SHR):
                _bw.extend_layout_for_barrel_shift(L, _op)
    elif _bw.tight_shift_enabled():
        for _op in (isa.SHL, isa.SHR):
            _bw.extend_layout_for_tight_shift(L, _op)
    while L._off % n_heads != 0:
        L._scalar(f"_bwpad{L._off}")
    L.D = L._off
    # force head_dim >= MEM_HEAD_CHANNELS so both KV heads' local channels fit.
    min_dim = n_heads * MEM_HEAD_CHANNELS
    if L.D < min_dim:
        target = -(-min_dim // n_heads) * n_heads
        while L._off < target:
            L._scalar(f"_hdpad{L._off}")
        L.D = L._off
    dim = L.D
    A._ONE = L.ONE

    # Block order (all one model.forward):
    #   ingest+recompose | pc-fetch | code-select | opcode-decode
    #   | mem-prep (LI addr) | mem-cam (LI/LC KV head -> AX) + recompose
    #   | stack-prep (pop addr) | stack-pop-cam (stack KV head -> STACK0) + recompose
    #   | cmp-compute | alu-expand | addsub | mul | divmod | ax-mux
    #   | [bitwise blocks] | dispatch | branch-delta | fold
    reg_bases = {"PC": L.PC, "AX": L.AX, "SP": L.SP, "BP": L.BP, "STACK0": L.STACK0}
    wide_blocks: List[Tuple[str, Dict]] = []
    if _wide:
        # WIDE INGEST (C4_INGEST_WIDE): replace the 20/21-head role-CAM ingest with a
        # SINGLE query + SINGLE KV head — three blocks IN FRONT of the stock block 0
        # (pre-route gate | wide gather+rescale | nibble snap), then the stock
        # "ingest+recompose" keeps its recompose FFN with its attn ZEROED.
        wide_blocks = [
            ("wide-preroute", compile_wide_preroute(L, dim)),
            ("wide-gather", compile_wide_rescale(L, reg_bases, dim)),
            ("wide-snap", compile_wide_nibble_snap(L, reg_bases, dim)),
        ]
    # CODE-FROM-MEMORY (C4_PF_CFM): the fetch@PC is a KV code-frame CAM, NOT the
    # O(code_size) baked PC-one-hot table.  pc-fetch sets the CAM query (CODE_QRY_BIN
    # = bits(PC)) + AX_ZERO; code-select carries the code CAM head (attn) whose value
    # writes OP_VAL + IMM_NIB (baked in ``bake_global_cam_heads``); imm-nib-fetch is a
    # no-op (the CAM already delivered IMM_NIB).  Residual dim is code_size-independent.
    if _cfm:
        _pcf = compile_pc_fetch_cfm(L, dim)
        _csel = _noop_spec(L, dim)          # attn = code CAM head; FFN no-op
        _inf = _noop_spec(L, dim)           # CAM delivered IMM_NIB -> no fetch FFN
    else:
        _pcf = compile_pc_fetch(L, dim)
        _csel = compile_code_select(L, dim)
        _inf = compile_imm_nib_fetch(L, dim)
    block_specs: List[Tuple[str, Dict]] = wide_blocks + [
        ("ingest+recompose", compile_nibble_to_scalar(L, dim)),
        ("pc-fetch",    _pcf),
        ("code-select", _csel),
        ("opcode-decode", compile_opcode_decode_pfc(L, dim)),  # + JSR/ENT/ADJ/LEV
        ("imm-nib-fetch", _inf),                               # IMM_NIB <- CODE_IMM_NIB@PC
        ("imm-clean", compile_imm_clean(L, dim)),              # IMM_CLEAN <- round(IMM_NIB)
        ("mem-prep", compile_mem_prep(L, dim)),
        ("mem-cam",  compile_nibble_to_scalar(L, dim)),          # ATTN=LI head (dropped if UNIFY)
        ("pop-addr", compile_pop_addr(L, dim)),                  # POP_ADDR/LEV_ADDR
        ("stack-prep", compile_stack_prep(L, dim)),
        ("lev-addr4", _force_bit2(L, L.LEV_QRY_BIN, dim)),       # LEV_QRY_BIN += 4
    ]
    _unify = _unify_cam_head_enabled()
    if _unify:
        # PART A: build the muxed read address + merged enable BEFORE the merged head.
        block_specs.append(("unify-cam-prep", compile_unify_cam_prep(L, dim)))
    block_specs += [
        ("stack-pop-cam", compile_stk_recompose(L, dim, hi_nibbles=_bp_restore_hibyte_nibs())),  # ATTN=stack+lev(+merged) heads
    ]
    if _unify:
        # PART A: demux the merged-head value -> AX (load) / STACK0 (pop) + AX_VAL recompose.
        block_specs.append(("unify-cam-demux", compile_unify_cam_demux(L, dim)))
        # PART A fix: re-recompose STK_VAL from the demux-updated STACK0 (the demux
        # wrote STACK0 one block AFTER stack-pop-cam's stale STK_VAL recompose).  LEV
        # reads STK_VAL for BP=MEM[BP]; without this a nested LEV sets BP=0.
        block_specs.append(("unify-stk-recompose",
                            compile_unify_stk_recompose(L, dim, hi_nibbles=_bp_restore_hibyte_nibs())))
    if _one:
        # PART A+ (C4_UNIFY_CAM_ONE): the SECOND cam block.  The SAME merged head INDEX
        # re-fires here with query=LEV_QRY_BIN (BP+4), enable=IS_LEV -> LEV_RET; its FFN
        # recomposes LEV_RET_VAL.  This is LEV's SECOND read (the return PC) on the SAME
        # head as the first (MEM[BP]) — one head index, two sequential blocks.
        block_specs.append(("lev-cam2",
                            compile_lev_ret_recompose(L, dim, hi_nibbles=_bp_restore_hibyte_nibs())))
    # LC SIGNED CHAR (c4 ``a = *(char *)a``): after the §Memory read has laid the
    # loaded byte into AX, sign-extend it for LC (byte >= 0x80 -> negative char ->
    # AX nibbles 2..7 filled with 0xF).  Two blocks: detect (LC_SIGN) then fill.  LI
    # is untouched (LC_SIGN is OP_IS[LC]-gated); a non-LC step is a strict no-op.
    block_specs += [
        ("lc-sign-detect", compile_lc_sign_detect(L, dim)),
        ("lc-sign-extend", compile_lc_sign_extend(L, dim)),
    ]
    block_specs += [
        ("cmp-compute", compile_cmp_compute(L, dim)),
    ]
    if _cmp32_enabled():
        # C4_CMP32 ORDER extension (#825): recompute the UNSIGNED magnitude order
        # (MAG_GT/MAG_LT) LEXICOGRAPHICALLY from the per-nibble bands BEFORE the
        # signed finalize, so the finalize's sign correction reuses the (now
        # fp-EXACT-at-any-magnitude) order.  This dodges the ~10^9-scale wide-scalar
        # ``step(STK_VAL-AX_VAL)`` recompose that collapses past fp32's 2^24 range.
        block_specs += [
            ("cmp-nib-order-lanes", compile_cmp_nib_order_lanes(L, dim)),
            ("cmp-nib-order-mag", compile_cmp_nib_order_mag(L, dim)),
        ]
    block_specs += [
        ("cmp-finalize", compile_cmp_signed_finalize(L, dim)),
    ]
    if _cmp32_enabled():
        # C4_CMP32 (#790): make EQ/NE/LT/GT/LE/GE robust to the full 32-bit tie by
        # deciding equality from the residue-immune NIBBLE bands (NDIFF -> tie flag ->
        # verdict override).  Three additive, UNGATED blocks touching only cmp scratch.
        block_specs += [
            ("cmp-nib-ndiff", compile_cmp_nib_ndiff(L, dim)),
            ("cmp-nib-tie", compile_cmp_nib_tie(L, dim)),
            ("cmp-nib-override", compile_cmp_nib_override(L, dim)),
        ]
    if _divmod_signed_enabled():
        # C4_DIVMOD_SIGNED (#790): replace STACK0/AX with |a|/|b| (op-gated DIV|MOD)
        # BEFORE alu-expand, so the UNSIGNED divider sees the magnitudes.  The signs
        # (SGN_A/SGN_B/RES_SGN) are recorded here and consumed by the post-divide
        # negate (sd-neg* below).  A non-DIV/MOD step is a strict no-op.
        for name, spec in A.compile_divmod_sign_prep(L, dim):
            block_specs.append((name, spec))
    block_specs += [
        ("alu-expand", A.compile_expand(L, dim)),
    ]
    for name, spec in A.compile_addsub_blocks(L, dim):
        block_specs.append((name, spec))
    for name, spec in A.compile_mul_blocks(L, dim):
        block_specs.append((name, spec))
    # ``divmod_apply_order`` records the physical-block indices to apply for the
    # DIV/MOD span, in application order.  For the unrolled path it is just the
    # divmod blocks' own indices (identity); for the recurrent path the reused
    # iteration-body block indices REPEAT 8x, so the model STORES 115 blocks but
    # APPLIES 262 — the recurrence.  The final full ``L._apply_order`` is assembled
    # after all blocks are appended (below).
    divmod_apply_order: List[int] = []
    # ---- BLOCK-MoE SEAM (integration point for the block-level identity/null
    # expert + block dispatch) -------------------------------------------------
    # ``[divmod_start, divmod_end)`` is the contiguous span of the ~262 DIV/MOD
    # blocks — the dominant physical cost and the block-MoE's primary skip target.
    # The full op set is ALWAYS built (no op-subset flag); making it CHEAP for a
    # non-DIV/MOD step is a RUNTIME block-dispatch concern.  #628's block-MoE plugs
    # in here: it can read ``L._divmod_span = (divmod_start, divmod_end)`` (recorded
    # below) to gate this contiguous block range behind an identity/null expert when
    # the step's decoded opcode is not DIV/MOD, without changing any stored weight.
    # Do NOT convert this span back into a build-time op-subset — the model must
    # STORE every op; block-MoE only SKIPS its application per step.
    divmod_start = len(block_specs)                 # first physical divmod block idx
    if recurrent_divmod:
        unique, apply_names = A.compile_divmod_blocks_recurrent(L, dim)
        # place the UNIQUE blocks physically, remember each name's physical index,
        # then map the (repeating) apply_names to those indices.
        name_to_idx: Dict[str, int] = {}
        for name, spec in unique:
            name_to_idx[name] = len(block_specs)
            block_specs.append((name, spec))
        divmod_apply_order = [name_to_idx[n] for n in apply_names]
    else:
        for name, spec in A.compile_divmod_blocks(L, dim):
            divmod_apply_order.append(len(block_specs))
            block_specs.append((name, spec))
    divmod_end = len(block_specs)                   # one past the last physical divmod block
    L._divmod_span = (divmod_start, divmod_end)     # block-MoE skip target (seam)
    if _divmod_signed_enabled():
        # C4_DIVMOD_SIGNED (#790): the unsigned divide wrote |a|//|b| into DIV_RES and
        # |a|%|b| into MOD_RES; apply the C-truncation result signs (quotient <-
        # RES_SGN on DIV, remainder <- SGN_A on MOD) BEFORE the ax-mux copies the
        # active op's result into AX.
        for name, spec in A.compile_divmod_sign_apply(L, dim):
            block_specs.append((name, spec))
    block_specs.append(("ax-mux", A.compile_ax_mux(L, dim, ops=ALU_OPS)))
    # NATIVE FLOAT (C4_FLOAT_OPS, default OFF): the IEEE-754 single F_ADD/F_SUB/
    # F_MUL/F_DIV megablocks compute their result into F_RES and a per-op latch copies
    # it into AX gated on OP_IS[F_*].  Runs AFTER the integer ax-mux so a float step's
    # AX is set by the float latch.  Flag OFF -> no blocks added -> byte-identical.
    if _float:
        from . import nibble_fp32 as _fp
        for name, spec in _fp.compile_all_fp_blocks(L, dim):
            block_specs.append((name, spec))
    from .nibble_unified import build_bitwise_blocks, _bw_recompose_spec
    for name, spec in build_bitwise_blocks(L, dim):
        block_specs.append((name, spec))
    # C4_SHIFT32 (#790): recompose the FULL 32-bit SHL/SHR result into AX_VAL (the
    # shifter already wrote all 8 AX nibbles) instead of the low byte, so the 32-bit
    # shift survives.  OR/XOR/AND stay 8-bit (per-nibble table over the loaded byte,
    # the golden semantics).  Flag OFF -> wide_ops empty -> byte-IDENTICAL.
    _shift_wide = (isa.SHL, isa.SHR) if _shift32_enabled() else ()
    block_specs.append(("bw-recompose", _bw_recompose_spec(
        L, dim, (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR),
        wide_ops=_shift_wide)))

    # Dispatch: base ops MINUS the ALU ops (ax-mux writes AX) + memory + cmp +
    # callconv + ALU housekeeping + bitwise housekeeping.
    disp_rules = _base_rules_minus_alu(L)
    disp_rules += memory_dispatch_rules(L)
    disp_rules += cmp_dispatch_rules_pop(L)
    disp_rules += callconv_dispatch_rules(L)
    disp_rules += alu32_housekeeping_rules(L)
    disp_rules += _bitwise_pop_rules(L)
    # ops whose AX result is a BYTE recomposed from AX_VAL by the byte-nib writeback
    # (so the AX nibble band is canonical, the driver decodes AX from it): IMM/LEA +
    # cmp + bitwise.  NOTE: LI/LC are EXCLUDED — the memory KV head writes their AX
    # nibbles DIRECTLY with the full 32-bit loaded value, and the byte-nib block reads
    # AXB_LO/HI off the STEP-INPUT AX_VAL (the OLD AX, recomposed at block 0 before
    # the head ran), so including LI/LC would clobber the loaded value with the stale
    # low byte.  A loaded 32-bit int (e.g. a variable holding 1000) must survive whole.
    byte_ax_ops = [isa.IMM, isa.LEA] + \
                  [isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE] + \
                  [isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR]
    if _shift32_enabled():
        # C4_SHIFT32 (#790): SHL/SHR keep their FULL 32-bit AX nibble result — the
        # ``ax-byte-nib`` writeback must NOT clear nibbles 2..7 for them.  (OR/XOR/AND
        # stay byte ops.)  The full result already lives in the AX nibble band from
        # the tight shifter; the byte-nib block only clobbered it.
        byte_ax_ops = [op for op in byte_ax_ops if op not in (isa.SHL, isa.SHR)]
    block_specs += [
        ("dispatch", compile_ffn(disp_rules, dim)),
        ("branch-delta", compile_branch_delta_clean(L, dim)),  # BZ/BNZ via IMM_CLEAN
        ("fold-lea", _fold_ax_gated(L, dim, [isa.LEA])),  # LEA masks &0xFF; ALU 32-bit
        ("ax-nib-split", compile_ax_nib_split(L, dim)),   # AX_VAL byte -> AXB_LO/HI
        # LEA frame address is residue-immune (#648 + #680): the frame byte is computed
        # EXACT-INTEGER + reduced to [0,256) from the BP/IMM NIBBLES (lea-q-reduce ->
        # LEA_Q), then decoded in a TINY 256-cell range (lea-addr-nib) -> no fp residue
        # can round the byte to the wrong 16-aligned/edge-adjacent cell, at any read-back
        # depth.  LEA-gated; else no-op (LEA_Q stays 0, AXB untouched).
        ("lea-q-reduce", compile_lea_q_reduce(L, dim)),   # LEA_Q = (BP_low+4*imm) mod 256
    ]
    if _lea_q_snap_enabled():
        # #705: integer-SNAP LEA_Q before the decode (deep-context IMM_NIB residue
        # fix).  DEFAULT ON; C4_LEA_Q_SNAP=0 reverts to the pre-#705 build (the
        # 2x2 self-emulation matmul then diverges at step 225 as documented).
        block_specs += [("lea-q-snap", compile_lea_q_snap(L, dim))]
    block_specs += [
        ("lea-addr-nib", compile_lea_addr_nib(L, dim)),   # AXB_LO/HI <- decode(LEA_Q)
        ("ax-byte-nib", compile_ax_byte_to_nibbles(L, dim, byte_ax_ops)),
        # FULL 32-bit IMM: overwrite ALL 8 AX nibbles with the fetched immediate
        # nibbles (the byte-nib block above only set nibbles 0,1) so literals > 255
        # (corpus values up to ~10000) survive into the canonical AX nibble band.
        ("imm-ax-nib", compile_imm_ax_nibbles(L, dim)),
    ]
    if _lea_wide_enabled():
        # WIDE LEA (C4_LEA_WIDE, #824 ported): overwrite the folded byte-0-only AX with
        # the FULL 32-bit frame address ``AX = BP + 4*imm`` via a nibble-domain
        # byte-ripple ADD (operands -> low-byte split -> 4 per-byte add blocks).  Runs
        # AFTER ``ax-byte-nib`` (which wrote the folded byte 0); byte 0 is re-derived
        # identically here and bytes 1..3 are widened.  All LEA-gated -> non-LEA ops
        # untouched.  Flag OFF -> none of these blocks exist -> golden 069cc32f build
        # byte-identical.
        block_specs += [
            ("lea-wide-operands", compile_lea_wide_operands(L, dim)),  # A=BP bytes, B2/B3, u16
            ("lea-wide-split", compile_lea_wide_split(L, dim)),        # B0=u16%256, B1=u16//256
        ]
        block_specs += [(f"lea-wide-b{i}", compile_lea_wide_byte(L, dim, i))
                        for i in range(4)]                             # ripple add -> AX nibbles
    n_blocks = len(block_specs)
    # ---- application order (recurrence) ----------------------------------
    # The model STORES ``n_blocks`` distinct blocks but APPLIES them in
    # ``L._apply_order``: identity everywhere except the DIV/MOD span, where the
    # recurrent path repeats the reused iteration-body indices.  A None means "no
    # remapping" (the driver falls back to range(n_blocks)) so the unrolled build
    # is byte-identical to before.
    if recurrent_divmod:
        L._apply_order = (list(range(divmod_start)) + divmod_apply_order +
                          list(range(divmod_end, n_blocks)))
    else:
        L._apply_order = None
    hidden = max(f["W_up"].shape[0] for _, f in block_specs)
    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, vocab=V.VOCAB, max_seq_len=16384)
    with torch.no_grad():
        _bake_pure_embedding(model, L)
        for bi, (name, spec) in enumerate(block_specs):
            _zero_attn(model.blocks[bi].attn)
            _load_ffn(model.blocks[bi].ffn, spec, hidden)
        if _wide:
            # WIDE INGEST: swap the "wide-gather" block's Attn for a 1-head Attn
            # (head_dim = dim) and bake the single wide gather head; the stock
            # "ingest+recompose" attn stays ZEROED (gather moved to wide-gather).
            # Every OTHER block's multi-head Attn is byte-identical.
            gi = [i for i, (nm, _) in enumerate(block_specs) if nm == "wide-gather"][0]
            model.blocks[gi].attn = _make_one_head_attn(dim, model.max_seq_len)
            bake_wide_ingest_head(model.blocks[gi].attn, L)
        else:
            # C4_INGEST_GQA (default OFF): 1-KV-head GQA ingest (byte-exact, 20 KV → 1).
            from .nibble_pure_forward import bake_frame_ingest_gqa, ingest_gqa_enabled
            (bake_frame_ingest_gqa if ingest_gqa_enabled()
             else bake_frame_ingest)(model.blocks[0].attn, L, reg_bases)
        bake_global_cam_heads(model.blocks, L, block_specs)
    L._block_names = [n for n, _ in block_specs]
    # RECURRENCE: re-point ``model.blocks`` through the application order so the
    # forward applies the reused divmod iteration body N times (the stored blocks
    # are the DISTINCT set; ``model.blocks`` now holds repeated references — shared
    # nn.Module weights, identical math).  ``_phys_blocks`` keeps the distinct set.
    if L._apply_order is not None:
        import torch.nn as _nn
        phys = list(model.blocks)
        model._phys_blocks = phys
        model.blocks = _nn.ModuleList([phys[i] for i in L._apply_order])
    return model, L


def _bake_pure_embedding(model, L) -> None:
    E = torch.zeros(V.VOCAB, model.dim)
    E[:, L.ONE] = 1.0
    for b in range(256):
        lo, hi = V.nibbles_of_byte(b)
        E[b, L.CUR_NIB + 0] = float(lo)
        E[b, L.CUR_NIB + 1] = float(hi)
    model.embed.copy_(E)


def _find(block_specs, name):
    return [i for i, (nm, _) in enumerate(block_specs) if nm == name][0]


def _base_rules_minus_alu(L) -> List[FFNRule]:
    """The base dispatch rules with ADD/SUB REMOVED (the 32-bit ALU + ax-mux own
    AX for ADD/SUB/MUL/DIV/MOD), PSH/LEA REMOVED (callconv owns them), and JMP
    REMOVED (re-added below reading the CLEAN immediate ``IMM_CLEAN`` for its target,
    so a large-literal PC-one-hot leak cannot shift the jump target — the memset
    ``JMP 134`` was leaking to 133, corrupting the fill loop; #648/#660)."""
    rules = base_dispatch_rules(L)
    drop = {isa.ADD, isa.SUB, isa.PSH, isa.LEA, isa.JMP}
    out = []
    for r in rules:
        # a rule's gate is [(OP_IS+op, ...)]; find which op.
        op = r.when[0][0] - L.OP_IS if r.when else None
        if op in drop:
            continue
        out.append(r)
    # JMP: PC = IMM_CLEAN (leak-free target).  PC += (IMM_CLEAN - PC).
    out.append(FFNRule([(L.OP_IS + isa.JMP, 0.5, 1.5)], {
        L.PC_VAL: LinearExpr.of(L.IMM_CLEAN, 1.0) + LinearExpr.of(L.PC_VAL, -1.0)}))
    return out


def compile_branch_delta_clean(L, dim: int) -> Dict[str, torch.Tensor]:
    """``compile_branch_delta`` (BZ/BNZ bilinear PC update) but reading the CLEAN
    immediate ``IMM_CLEAN`` for the taken target instead of the leaky ``IMM`` — same
    root fix as JMP: a large-literal PC-one-hot leak must not shift a branch target
    (BZ/BNZ targets are small positive PC values < code_size, exact in IMM_CLEAN's
    12-bit signed range).  Structurally identical to ``compile_branch_delta`` with
    ``imm`` rebound; kept here (not a flag on the base) so non-complete builds — which
    have no ``IMM_CLEAN`` dim — are untouched."""
    import torch.nn.functional as _Fnn
    pc, imm, azero, one = L.PC_VAL, L.IMM_CLEAN, L.AX_ZERO, L.ONE
    BZ, BNZ = L.OP_IS + isa.BZ, L.OP_IS + isa.BNZ
    spec = _empty_spec(dim, 4)
    BIG = 200.0
    silu_big = float(_Fnn.silu(torch.tensor(0.5 * BIG)))

    def _and_unit(u, op_band, bool_terms, gate_terms):
        spec["W_up"][u, op_band] += BIG
        for band, coeff, const in bool_terms:
            if band is not None:
                spec["W_up"][u, band] += BIG * coeff
            spec["b_up"][u] += BIG * const
        spec["b_up"][u] += -BIG * 1.5
        for band, coeff in gate_terms:
            spec["W_gate"][u, band] += coeff

    _and_unit(0, BZ,  [(azero, 1.0, 0.0)],  [(imm, 1.0), (pc, -1.0)])
    spec["W_down"][pc, 0] += 1.0 / silu_big
    _and_unit(1, BZ,  [(azero, -1.0, 1.0)], [(one, 1.0)])
    spec["W_down"][pc, 1] += 1.0 / silu_big
    _and_unit(2, BNZ, [(azero, -1.0, 1.0)], [(imm, 1.0), (pc, -1.0)])
    spec["W_down"][pc, 2] += 1.0 / silu_big
    _and_unit(3, BNZ, [(azero, 1.0, 0.0)],  [(one, 1.0)])
    spec["W_down"][pc, 3] += 1.0 / silu_big
    return spec


def cmp_dispatch_rules_pop(L) -> List[FFNRule]:
    """The CMP write rules (from ``nibble_pure_forward.cmp_dispatch_rules``) but
    with the SP += 4 pop already done — cmp_dispatch_rules already adds SP+=4, so
    reuse it directly."""
    return cmp_dispatch_rules(L)


def _bitwise_pop_rules(L) -> List[FFNRule]:
    from .nibble_pure_forward import bitwise_dispatch_rules
    return bitwise_dispatch_rules(L)


# ===========================================================================
# C4_CMP32 (#790): PER-NIBBLE 32-bit equality, robust to wide-scalar noise.
# The default cmp reads STK_VAL/AX_VAL (a 5-nibble ~20-bit recompose whose ~1e-6
# relative silu residue reaches ~±1 at a ~10^6-scale operand), so an EXACT 32-bit
# TIE (STK==AX) mis-fires the Z(STK_VAL-AX_VAL) bump / the MAG ramps.  These two
# ADDITIVE blocks decide equality from the residue-immune NIBBLE bands directly
# (each nibble is a small 0..15 integer, exact) and override the verdict on a tie.
# Both are UNGATED FFNs (no OP_IS) — they only touch cmp scratch dims, which are
# read only by the OP_IS-gated cmp writeback, so a non-cmp step is a strict no-op.
# ===========================================================================
def compile_cmp_nib_ndiff(L, dim: int) -> Dict[str, torch.Tensor]:
    """``NIB_EQ := NDIFF = Σ_{j<8} |STACK0[j] - AX[j]|`` — the nibble-wise L1 distance
    (a non-negative integer, 0 iff the two words are equal in EVERY nibble = a full
    32-bit tie).  Each nibble is a small 0..15 integer so every ``relu`` is fp-exact.
    SET (self-clears NIB_EQ first)."""
    n = 8                                              # 8 nibbles = 32 bits
    spec = _empty_spec(dim, 1 + 2 * n)                 # self-clear + 2 relu/nibble
    u = 0
    spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.NIB_EQ] = 1.0
    spec["W_down"][L.NIB_EQ, u] += -1.0 / SILU_S; u += 1
    for j in range(n):
        s, a = L.STACK0 + j, L.AX + j
        spec["W_up"][u, s] = RELU_S; spec["W_up"][u, a] = -RELU_S     # +relu(s-a)
        spec["W_gate"][u, L.ONE] = 1.0
        spec["W_down"][L.NIB_EQ, u] += 1.0 / RELU_S; u += 1
        spec["W_up"][u, a] = RELU_S; spec["W_up"][u, s] = -RELU_S     # +relu(a-s)
        spec["W_gate"][u, L.ONE] = 1.0
        spec["W_down"][L.NIB_EQ, u] += 1.0 / RELU_S; u += 1
    return spec


def compile_cmp_nib_tie(L, dim: int) -> Dict[str, torch.Tensor]:
    """``NIB_EQ := relu(1 - NIB_EQ)`` — collapse the NDIFF distance into the clean
    0/1 TIE flag (1 iff NDIFF==0, since NDIFF is a non-negative integer).  In place
    (reads the NDIFF written by :func:`compile_cmp_nib_ndiff`, self-clears, rewrites
    the clamped flag)."""
    spec = _empty_spec(dim, 2)
    u = 0
    # self-clear NIB_EQ (subtract the OLD NDIFF, read from the block input).
    spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.NIB_EQ] = 1.0
    spec["W_down"][L.NIB_EQ, u] += -1.0 / SILU_S; u += 1
    # NIB_EQ += relu(1 - NDIFF_old).  Both units read the block INPUT (NDIFF_old), so
    # the self-clear (also reading input) and this add compose to SET NIB_EQ = tie.
    spec["W_up"][u, L.NIB_EQ] = -RELU_S; spec["b_up"][u] = RELU_S * 1.0
    spec["W_gate"][u, L.ONE] = 1.0
    spec["W_down"][L.NIB_EQ, u] += 1.0 / RELU_S; u += 1
    return spec


def compile_cmp_nib_override(L, dim: int) -> Dict[str, torch.Tensor]:
    """Override the cmp verdict on an exact 32-bit tie using the clean ``NIB_EQ``
    tie flag (0/1, from :func:`compile_cmp_nib_tie`):

        CMP_EQ := NIB_EQ                    (clean 0/1, replaces the noisy Z(d) bump)
        CMP_GT := relu(CMP_GT_old - NIB_EQ)  (0 on a tie, unchanged otherwise)
        CMP_LT := relu(CMP_LT_old - NIB_EQ)  (0 on a tie, unchanged otherwise)

    ``CMP_*_old`` and ``NIB_EQ`` are both clean 0/1 read from the block INPUT, so
    ``relu(old - tie)`` is exactly ``old AND NOT tie``.  SET (self-clears the three
    verdict lanes first)."""
    spec = _empty_spec(dim, 3 + 1 + 2)                  # 3 clears + EQ(1) + GT/LT(2)
    u = 0
    for lane in (L.CMP_EQ, L.CMP_GT, L.CMP_LT):         # self-clear (SET)
        spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, lane] = 1.0
        spec["W_down"][lane, u] += -1.0 / SILU_S; u += 1
    # CMP_EQ := NIB_EQ (silu-identity read of the clean 0/1 flag).
    spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.NIB_EQ] = 1.0
    spec["W_down"][L.CMP_EQ, u] += 1.0 / SILU_S; u += 1
    # CMP_GT := relu(CMP_GT_old - NIB_EQ) ; CMP_LT := relu(CMP_LT_old - NIB_EQ).
    for old in (L.CMP_GT, L.CMP_LT):
        spec["W_up"][u, old] = RELU_S; spec["W_up"][u, L.NIB_EQ] = -RELU_S
        spec["W_gate"][u, L.ONE] = 1.0
        spec["W_down"][old, u] += 1.0 / RELU_S; u += 1
    return spec


# ===========================================================================
# C4_CMP32 ORDER extension (#825): the MSB-first LEXICOGRAPHIC magnitude order.
# The golden MAG_GT/MAG_LT read ``step(STK_VAL - AX_VAL)`` on the WIDE-SCALAR
# recompose (``Σ 16^j·nib_j``), which at a ~10^9-scale operand exceeds fp32's 2^24
# exact range — a 1-ULP magnitude gap at 10^9 is unrepresentable, so the ramp
# collapses to 0 and GT/LT/LE/GE mis-fire (the doom step-30,850 wall: GT(2^30 heap
# ptr, 4096) gives 0 not 1).  These two ADDITIVE blocks recompute the UNSIGNED
# magnitude order from the PER-NIBBLE bands (each nibble a small 0..15 integer, so
# every relu is fp-EXACT) via a lexicographic scan from the MSB, then OVERWRITE
# MAG_GT/MAG_LT.  They are inserted BEFORE ``compile_cmp_signed_finalize`` so the
# EXISTING sign correction (``CMP_GT = clamp01(MAG_GT) + (SGN_AX - SGN_STK)``, and
# the symmetric LT) reuses them verbatim — the signed verdict is unchanged in
# structure, only the (now exact) magnitude order flows in.  Both are UNGATED FFNs
# touching only cmp scratch (read only by the OP_IS-gated cmp writeback), so a
# non-cmp step is a strict no-op.
# ===========================================================================
def compile_cmp_nib_order_lanes(L, dim: int) -> Dict[str, torch.Tensor]:
    """Per-nibble compare lanes, MSB-first-ready:

        NGT[j] = [STACK0[j] >  AX[j]]   NLT[j] = [STACK0[j] <  AX[j]]
        NEQ[j] = [STACK0[j] == AX[j]]                          (j = 0..7)

    Each ``d = STACK0[j] - AX[j]`` is in [-15,15], so every relu staircase argument
    is a small integer -> fp-EXACT.  ``NGT = relu(d) - relu(d-1) = [d>=1]``,
    ``NLT = relu(-d) - relu(-d-1) = [d<=-1]``, ``NEQ = 1 - NGT - NLT = [d==0]``.
    SET (self-clears each lane first)."""
    n = 8
    spec = _empty_spec(dim, n * (1 + 2 + 2 + 1 + 1 + 1))   # per nibble: clear*3 + gt(2)+lt(2)+eq(2)... generous
    u = 0
    for j in range(n):
        s, a = L.STACK0 + j, L.AX + j
        gt, lt, eq = L.NGT + j, L.NLT + j, L.NEQ + j
        # self-clear the three lanes (SET).
        for lane in (gt, lt, eq):
            spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, lane] = 1.0
            spec["W_down"][lane, u] += -1.0 / SILU_S; u += 1
        # NGT = relu(d) - relu(d-1)   (d = s-a); also feed NEQ -= NGT.
        for idx, thr in enumerate((0.0, 1.0)):
            spec["W_up"][u, s] = RELU_S; spec["W_up"][u, a] = -RELU_S
            spec["b_up"][u] = -RELU_S * thr; spec["W_gate"][u, L.ONE] = 1.0
            c = (1.0 if idx == 0 else -1.0)
            spec["W_down"][gt, u] += c / RELU_S
            spec["W_down"][eq, u] += -c / RELU_S      # NEQ -= NGT contribution
            u += 1
        # NLT = relu(-d) - relu(-d-1); also feed NEQ -= NLT.
        for idx, thr in enumerate((0.0, 1.0)):
            spec["W_up"][u, a] = RELU_S; spec["W_up"][u, s] = -RELU_S
            spec["b_up"][u] = -RELU_S * thr; spec["W_gate"][u, L.ONE] = 1.0
            c = (1.0 if idx == 0 else -1.0)
            spec["W_down"][lt, u] += c / RELU_S
            spec["W_down"][eq, u] += -c / RELU_S      # NEQ -= NLT contribution
            u += 1
        # NEQ += 1  ( => NEQ = 1 - NGT - NLT = [d==0] ).
        spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, L.ONE] = 1.0
        spec["W_down"][eq, u] += 1.0 / SILU_S; u += 1
    return spec


def compile_cmp_nib_order_mag(L, dim: int) -> Dict[str, torch.Tensor]:
    """Assemble the lexicographic UNSIGNED-magnitude order and OVERWRITE MAG_GT/MAG_LT:

        MAG_GT = Σ_{i=7..0} ( NGT[i] AND NEQ[i+1] AND ... AND NEQ[7] )
        MAG_LT = Σ_{i=7..0} ( NLT[i] AND NEQ[i+1] AND ... AND NEQ[7] )

    The first (highest) nibble where STACK0 differs from AX decides the order; all
    higher nibbles must tie (the NEQ window).  Exactly one product term fires (or
    none, on a full tie), so each sum is a clean 0/1.  Each product is a multi-way
    AND over 0/1 indicators, emitted as one guarded silu unit (``up = S·(Σw-(n-0.5))``
    fires iff all windows are 1).  SET (self-clears MAG_GT/MAG_LT first).  These
    read the residue-immune PER-NIBBLE lanes only, so at any magnitude the order is
    fp-EXACT."""
    n = 8
    n_terms = n                         # one AND-product per nibble position i
    spec = _empty_spec(dim, 2 + 2 * n_terms)   # 2 clears + n GT-products + n LT-products
    u = 0
    for lane in (L.MAG_GT, L.MAG_LT):           # self-clear (SET)
        spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, lane] = 1.0
        spec["W_down"][lane, u] += -1.0 / SILU_S; u += 1
    # MAG_GT / MAG_LT products, MSB-first.
    for dst, decide in ((L.MAG_GT, L.NGT), (L.MAG_LT, L.NLT)):
        for i in range(n - 1, -1, -1):
            windows = [decide + i] + [L.NEQ + k for k in range(i + 1, n)]
            m = len(windows)
            for w in windows:
                spec["W_up"][u, w] += S
            spec["b_up"][u] += -S * (m - 0.5)
            spec["W_gate"][u, L.ONE] = 1.0
            spec["W_down"][dst, u] += 1.0 / SILU_HALF
            u += 1
    return spec


# ===========================================================================
# SP-ADDRESSED MEMORY-STACK REFERENCE (the byte-exact oracle).  Stack + frame +
# heap share ONE byte-addressed region; SP/BP are byte addresses descending from
# SP_INIT.  Matches the neural build's per-step AX emit.
# ===========================================================================
def ref_interpret(code: List[isa.Instr], max_steps: int = 512,
                  mask: int = 0xFF, out: List[int] = None) -> List[int]:
    """Reference interpreter with the SP-addressed memory stack.  ``mask`` = 0xFF
    (8-bit AX trace, the default; matches the neural byte frame) or 0xFFFFFFFF (the
    full 32-bit AX for the 32-bit-arithmetic proof: ADD/SUB/MUL wrap mod 2^32,
    DIV/MOD unsigned floor with b==0 -> 0, ISA_SPEC 4.2).

    If ``out`` is a list, PRTF appends ``AX & 0xFF`` to it — the visible stdout
    byte a ``printf("%c", AX)`` would emit (§System / op 33)."""
    mem: Dict[int, int] = {}
    sp = bp = SP_INIT
    ax = pc = 0
    trace: List[int] = []
    steps = 0
    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        i = pc
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.LEA:
            ax = (bp + 4 * imm) & 0xFF
        elif op == isa.PSH:
            sp -= 4; mem[sp] = ax & mask
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
            v = mem.get(sp, 0) & mask; sp += 4
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
            # OR/XOR/AND stay 8-bit (per-nibble table over the loaded byte). SHL/SHR
            # honour ``mask``: the neural model computes them via the NATIVE 32-bit
            # MUL/DIV gadgets (``x*2**n`` / ``x//2**n``, shift-via-mul).  SHL stays
            # logical (c4 ``a = *sp++ << a`` matches at the observable byte).  SHR is
            # ARITHMETIC in c4 (``a = *sp++ >> a`` on a SIGNED ``long long``): read
            # the popped operand as SIGNED at the value width and sign-fill on the
            # shift (Python ``>>`` on a negative already sign-extends), then re-mask.
            if op in (isa.SHL, isa.SHR):
                v = mem.get(sp, 0) & mask; sp += 4
                if op == isa.SHL:
                    ax = (v << ax) & mask
                else:
                    _sign = (mask >> 1) + 1                  # 0x80 / 2^31 at the width
                    sv = v - (mask + 1) if v & _sign else v  # signed operand
                    ax = (sv >> ax) & mask
            else:
                v = mem.get(sp, 0) & 0xFF; sp += 4
                if op == isa.OR:
                    ax = (v | ax) & 0xFF
                elif op == isa.XOR:
                    ax = (v ^ ax) & 0xFF
                else:
                    ax = (v & ax) & 0xFF
        elif op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            # C4's ordering comparisons (LT/GT/LE/GE) are SIGNED two's-complement
            # on the 32-bit word (sign bit 31 — matching the neural model, whose
            # signed-compare gadget uses a FIXED 2^31 sign boundary); EQ/NE are
            # bit-equality (sign-agnostic).  Operands are read at the value width
            # ``mask``: under the 8-bit fold they are < 2^31, so the sign bit is
            # never set and the comparison is the unsigned byte order (unchanged);
            # under the 32-bit proof a genuine negative has bit 31 set.
            v = mem.get(sp, 0) & mask; av = ax & mask
            sv = v - (1 << 32) if v & (1 << 31) else v      # signed 32-bit STK
            sax = av - (1 << 32) if av & (1 << 31) else av  # signed 32-bit AX
            r = {isa.EQ: v == av, isa.NE: v != av, isa.LT: sv < sax,
                 isa.GT: sv > sax, isa.LE: sv <= sax, isa.GE: sv >= sax}[op]
            ax = 1 if r else 0
        elif op == isa.LI:
            ax = mem.get(ax, 0) & 0xFF
        elif op == isa.LC:
            # c4: ``a = *(char *)a`` -> SIGNED char load.  A byte >= 0x80 is a
            # negative char, sign-extended to the register (value) width; LI stays
            # an unsigned byte load.  At mask=0xFF the byte is unchanged; at the
            # 32-bit width 0x80 -> 0xFFFFFF80, 0xFF -> 0xFFFFFFFF.
            b = mem.get(ax, 0) & 0xFF
            ax = (b - 0x100 if b & 0x80 else b) & mask
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp, 0); sp += 4
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
        elif op == isa.PRTF:
            if out is not None:
                out.append(ax & 0xFF)   # printf visible byte; registers unchanged
        elif op == isa.NOP:
            pass
        elif op == isa.HALT:
            trace.append(ax & mask); break
        else:
            raise NotImplementedError(f"op {isa.NAMES.get(op, op)} not in ref ISA")
        trace.append(ax & mask)
    return trace


# ===========================================================================
# THE OVERLAY: program-in-data + frame ROLE tags + the KV store log.  A store
# frame's MEM token becomes a KV entry keyed on its 32-bit address.  This carries
# BOTH the SI/SC program stores AND the stack pushes (PSH/JSR/ENT) — they are the
# same kind of address-keyed write; the driver records (frame_idx -> (addr, val)).
# ===========================================================================
def _overlay_pf_code_frames(x, L, code: List[isa.Instr], base_pos: int = 1,
                            row: int = 0) -> None:
    """CODE-FROM-MEMORY: write the program into the KV as address-keyed CODE frames.

    One frame per instruction i at ``x[row, base_pos + i]``: KEY = ``CODE_KEY_BIN``
    = bits(i), VALUE = ``CODE_OPV`` (op) + ``CODE_IMM_NIB_MEM`` (the IMM_NIBS
    immediate NIBBLES — 20-bit IMM preserved), gated ``IS_CODE=1``.  These PERSIST
    (the address-keyed code memory the fetch CAM reads at PC), exactly like the
    store log persists for LI/SI.  Guarded to the rows that actually exist so a
    caller whose window omits the code rows (a short probe) is a no-op for the
    missing frames."""
    Sn = x.shape[1]
    for i, ins in enumerate(code):
        p = base_pos + i
        if p >= Sn:
            break
        x[row, p, L.IS_CODE] = 1.0
        x[row, p, L.IS_FRAME_BYTE] = 0.0
        for b in range(CODE_ADDR_BITS):
            x[row, p, L.CODE_KEY_BIN + b] = float((i >> b) & 1)
        x[row, p, L.CODE_OPV] = float(ins.op)
        for j, nv in enumerate(V.nibbles_of_value(ins.imm & 0xFFFFFFFF, IMM_NIBS)):
            x[row, p, L.CODE_IMM_NIB_MEM + j] = float(nv)


def _overlay_pf_runtime_code_frames(x, L, runtime_frames, base_pos: int = 1,
                                    row: int = 0) -> int:
    """RUNTIME (EMIT-emitted) code frames on the DIRECT-CAM path.

    ``runtime_frames`` maps ``addr -> Instr`` for every instruction the running
    program produced with ``isa.EMIT`` at a code ADDRESS that is NOT one of the
    original ``1..len(code)`` positional rows (i.e. a HIGH address, up to
    ``2^CODE_ADDR_BITS``).  Each is written as ONE COMPACT physical code-frame row
    (KEY = ``CODE_KEY_BIN`` = bits(addr), VALUE = op + imm nibbles, IS_CODE=1) —
    the physical row index is independent of the (possibly huge) logical address:
    the fetch@PC CAM (``_bake_code_cam_head``, ALiBi slope 0) matches PURELY on the
    ``CODE_ADDR_BITS``-wide address key, so a frame keyed bits(440000) resolves
    fetch@PC=440000 at a compact row with NO positional decay.  THIS is what lifts
    the RoPE ~12-16-bit ceiling: the address key rides the position-invariant
    direct-CAM lanes, not the slow rotary ones.  Returns the number of rows written."""
    Sn = x.shape[1]
    n = 0
    for addr, ins in runtime_frames.items():
        p = base_pos + n
        if p >= Sn:
            break
        x[row, p, L.IS_CODE] = 1.0
        x[row, p, L.IS_FRAME_BYTE] = 0.0
        for b in range(CODE_ADDR_BITS):
            x[row, p, L.CODE_KEY_BIN + b] = float((addr >> b) & 1)
        x[row, p, L.CODE_OPV] = float(ins.op)
        for j, nv in enumerate(V.nibbles_of_value(ins.imm & 0xFFFFFFFF, IMM_NIBS)):
            x[row, p, L.CODE_IMM_NIB_MEM + j] = float(nv)
        n += 1
    return n


def make_overlay_complete(code: List[isa.Instr], L: PureForwardCompleteLayout,
                          store_log=None, frame_start: int = None,
                          runtime_frames=None, code_rows: int = None):
    """``overlay(x)`` writes the program into the DATA bands at every position, the
    ROLE/IS_FRAME_BYTE frame-slot tags, and turns each stored frame's MEM token
    into a KV entry.  ``store_log`` maps ``frame_idx -> (addr, val)`` for every
    emitted frame that wrote memory (a program store OR a stack push).

    CODE-FROM-MEMORY (C4_PF_CFM): the program lives in the KV as CODE frames on the
    positions ``1 .. 1+len(code)`` (not in the per-position DATA bands), so the first
    register/store frame begins at ``frame_start = 1 + len(code)``.  ``frame_start``
    defaults to ``1`` (baked path, byte-identical) or ``1+len(code)`` under cfm.

    ``runtime_frames`` (in-transformer JIT, ``C4_CFM_EMIT``): an ``{addr -> Instr}``
    map of instructions the running program produced with ``isa.EMIT`` at a HIGH code
    address (past the ``1..len(code)`` positional rows).  They are overlaid as COMPACT
    address-keyed code frames right after the positional code rows — one physical row
    per distinct high address, keyed bits(addr) on the position-invariant direct-CAM
    lanes — so a HIGH-address emit stays fetchable with NO positional decay.  The
    register frames then begin AFTER these runtime rows (``frame_start`` moves past
    ``1 + len(code) + len(runtime_frames)``)."""
    store_log = store_log or {}
    runtime_frames = runtime_frames or {}
    from .blogspec_layout import NIB_PER_REG
    _cfm = getattr(L, "cfm", False)
    # ``code_rows`` = how many POSITIONAL code rows exist physically (defaults to
    # len(code)).  Under EMIT, ``code`` (= run_code) may GROW past the reserved
    # positional rows for a high-address emit — those high entries live in
    # ``runtime_frames`` as compact address-keyed rows, NOT positional ones — so the
    # positional overlay is capped at ``code_rows`` (the original program length).
    _n_pos = code_rows if code_rows is not None else len(code)
    _n_rt = len(runtime_frames) if _cfm else 0
    fstart = (frame_start if frame_start is not None
              else (1 + _n_pos + _n_rt if _cfm else 1))

    def overlay(x: torch.Tensor) -> None:
        Sn = x.shape[1]
        for i in range(Sn):
            x[0, i, L.ONE] = 1.0
            if not _cfm:
                for k, ins in enumerate(code):
                    x[0, i, L.CODE_OP[k]] = float(ins.op)
                    x[0, i, L.CODE_IMM[k]] = float(ins.imm)
                    # the immediate's low IMM_NIBS nibbles are part of the STATIC program
                    # encoding (a pure re-encoding of the constant, gathered at PC).
                    for j, nv in enumerate(V.nibbles_of_value(ins.imm & 0xFFFFFFFF, IMM_NIBS)):
                        x[0, i, L.CODE_IMM_NIB[k] + j] = float(nv)
        if _cfm:
            # CODE-FROM-MEMORY: the program lives in the KV as address-keyed CODE
            # frames on the leading positions after BOS.  Each is ONE row: IS_CODE=1,
            # CODE_KEY_BIN=bits(addr i), CODE_OPV=op, CODE_IMM_NIB_MEM=imm nibbles.
            # The fetch@PC query (IS_FETCH/CODE_QRY_BIN) is set by the pc-fetch FFN
            # from the PC register — not here.  (Guarded to the available leading rows.)
            _overlay_pf_code_frames(x, L, code[:_n_pos], base_pos=1)
            # RUNTIME (EMIT) code frames: compact address-keyed rows for HIGH-address
            # emits, right after the positional code rows.  The fetch@PC CAM matches on
            # the CODE_ADDR_BITS-bit address key alone (ALiBi slope 0), so their
            # physical row is position-invariant -> a 440000-address emit fetches at a
            # compact row with no positional decay.
            if runtime_frames:
                _overlay_pf_runtime_code_frames(x, L, runtime_frames,
                                                base_pos=1 + _n_pos)
        pos = fstart
        frame_idx = 0
        while pos + V.FRAME_LEN <= Sn:
            for local, role in _FRAME_ROLE_SLOTS.items():
                p = pos + local
                x[0, p, L.ROLE + role] = 1.0
                x[0, p, L.IS_FRAME_BYTE] = 1.0
            if frame_idx in store_log:
                addr, val = store_log[frame_idx]
                mem_pos = pos + _MEM_MARKER_LOCAL
                x[0, mem_pos, L.IS_STORE] = 1.0
                x[0, mem_pos, L.IS_FRAME_BYTE] = 0.0
                for b, bit in enumerate(_address_bits(addr)):
                    x[0, mem_pos, L.ADDR_BIN + b] = bit
                for j, nv in enumerate(V.nibbles_of_value(val & 0xFFFFFFFF, NIB_PER_REG)):
                    x[0, mem_pos, L.VAL_NIB + j] = float(nv)
            pos += V.FRAME_LEN
            frame_idx += 1
        # the query row (the last position) carries all ROLE one-hots for ingest.
        for role in range(N_ROLES):
            x[0, -1, L.ROLE + role] = 1.0
    return overlay


# ===========================================================================
# THE DRIVER — one VM step = one model.forward; argmax-generate-append only.  The
# only Python beyond that is the driver's store-token BOOKKEEPING (it fetches the
# op at PC — the same code-as-data fetch the model does — to know WHICH address a
# push/store writes, then lays that KV entry into the emitted frame).  No VM
# transition is computed in Python: the op RESULT, the register motion, the pop
# value, all come out of ``model.forward``.
# ===========================================================================
def _build_frame(pc, ax, sp, bp, stk, mem_addr=0, mem_val=0):
    if mem_addr or mem_val:
        return V.build_step_frame(pc, ax, sp, bp, mem_addr=mem_addr, mem_val=mem_val)
    return V.build_step_frame(pc, ax, sp, bp, mem_addr=0, mem_val=stk & 0xFFFFFFFF)


def _seed_frames(seed_mem):
    """Turn a ``{addr: value}`` data segment into a list of leading STORE frames.

    Each entry becomes one MEM-store frame (PC/AX/SP/BP left at their init values)
    that seeds ``mem[addr] = value`` into the model's KV memory BEFORE the program
    runs — the classic quine's "string literal in the data segment", materialised
    as the §Memory store-log rows the loads content-address.  Returns
    ``(frames, store_log)`` where ``store_log`` maps each seed frame's index to its
    ``(addr, val)`` so the overlay tags its MEM token as a KV entry."""
    frames: List[int] = []
    store_log: Dict[int, Tuple[int, int]] = {}
    for k, (addr, val) in enumerate(sorted(seed_mem.items())):
        frames += _build_frame(0, 0, SP_INIT, SP_INIT, 0,
                               mem_addr=addr & 0xFFFFFFFF, mem_val=val & 0xFFFFFFFF)
        store_log[k] = (addr & 0xFFFFFFFF, val & 0xFFFFFFFF)   # frame idx k
    return frames, store_log


def run_pure_forward_complete(model, L: PureForwardCompleteLayout,
                              code: List[isa.Instr], max_steps: int = 512,
                              verbose: bool = False, collect_tokens: bool = False,
                              mask: int = 0xFF, fio=None, data_seg=None,
                              out: List[int] = None, seed_mem=None):
    """Execute ``code`` with the complete pure-forward step: every VM step is ONE
    ``model.forward`` over the growing token stream.  Returns the per-step AX
    trace (matching :func:`ref_interpret`).  ``mask`` (0xFF = 8-bit AX trace, or
    0xFFFFFFFF = the full 32-bit AX for the 32-bit-arithmetic proof) is applied to
    the emitted AX only; the model always carries the full 32-bit AX in its nibble
    bands (the ALU is 32-bit-exact) regardless of ``mask``.

    FILE OPS (OPEN/READ/CLOS/PRTF) are the ONE class not computed neurally (§Tool
    Use Mode): when ``fio`` (a ``nibble_filesys.FileOpState``) is passed, an op in
    ``FILE_OPCODES`` is dispatched via the TOOL_CALL protocol — the driver marshals
    the args off the KV store log, the runner performs the real I/O against the
    stub filesystem / stdin, and the integer result re-enters AX (READ also lays
    its bytes back into the store log as fresh §Memory KV frames so LC reads them).
    ``data_seg`` (``{byte_addr: byte}``) seeds the read-only data segment (the
    filename / format string literals the c4 loader places).

    If ``out`` is a list, a PRTF step (in the ``out=`` string-quine mode, no
    ``fio``) appends its VISIBLE output byte — the AX byte-0 the model itself
    decoded from its nibble band (a genuine LM-head argmax, NOT a python copy) —
    and a SEPARATE ``vis_stream`` carries ``THINK_END, <byte>, THINK_START`` around
    that byte so ``blogspec_vocab.visible_output(vis_stream)`` recovers exactly
    ``out`` (the think-tag stdout protocol, §Printing).  ``seed_mem``
    (``{addr: value}``) seeds the data segment as leading MEM-store frames (the
    quine's bundled string literal).

    LEV is expanded to its two frame loads (BP then return-PC) by re-running the
    forward with the appropriate stack query — but because the whole state lives
    in the stream, LEV is handled as a NORMAL step: the model reads MEM[BP] into
    STACK0 via the stack head (query = BP) ... which the single stack head cannot
    do together with reading MEM[BP+4].  So LEV is realised as the model computing
    SP=BP and the driver issuing the two frame reads across the SAME KV log — see
    below (the two loads are two model.forwards, still pure-forward)."""
    init_frame = _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    # The MODEL's token stream stays the pure contiguous 30-token frame stream (no
    # in-stream think tags) so the overlay/frame geometry is byte-identical to the
    # non-PRTF path.  The think-tag protocol (THINK_END, <visible byte>, THINK_START)
    # is materialised in a SEPARATE ``vis_stream`` the driver returns, from which
    # ``blogspec_vocab.visible_output`` recovers exactly the printed bytes — the
    # user-facing view, decoupled from the internal execution stream.
    # Prepend the seeded data segment (if any) as leading STORE frames so the KV
    # memory holds it before step 0; then BOS + the seed frames + the init frame.
    seed_frames, store_log = _seed_frames(seed_mem or {})
    n_seed = len(store_log)                        # number of leading data frames
    # CODE-FROM-MEMORY (C4_PF_CFM): the program lives in the KV as persistent CODE
    # frames (one token per instruction, right after BOS).  The overlay writes their
    # CODE_KEY_BIN/CODE_OPV/CODE_IMM_NIB_MEM bands; the token is a neutral MEM marker
    # (IS_FRAME_BYTE=0 -> ingest ignores it; store CAM excludes IS_CODE rows).
    _cfm = getattr(L, "cfm", False)
    code_frame_toks: List[int] = [V.MEM] * len(code) if _cfm else []
    stream: List[int] = [V.BOS] + code_frame_toks + seed_frames + init_frame
    vis_stream: List[int] = [V.BOS, V.THINK_START]
    trace: List[int] = []
    cur_pc = 0
    # pre-step register bookkeeping (mirrors the state the model reads; the driver
    # only uses these to know WHICH address a push/store writes — the code-as-data
    # fetch the model itself does — never to compute the VM transition).
    cur_sp = cur_bp = SP_INIT
    cur_ax = 0
    # the init frame sits at frame index ``n_seed`` (after the seed data frames);
    # the first emitted step frame is ``n_seed + 1``.
    frame_idx = n_seed
    for _ in range(max_steps):
        overlay = make_overlay_complete(code, L, store_log=store_log)
        toks = torch.tensor([stream])
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)
            for blk in model.blocks:
                x = blk(x)
        state = x[0, -1]
        pc = _snap_lane(state[L.PC_VAL])
        sp = _snap_lane(state[L.SP_VAL])
        bp = _snap_lane(state[L.BP_VAL])
        stk = _snap_lane(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        imm = code[cur_pc].imm if 0 <= cur_pc < len(code) else 0
        # AX is ALWAYS the canonical AX nibble band: the ax-mux writes the full 4-byte
        # 32-bit ALU result there, the memory head writes the loaded byte, and the
        # ax-byte-nib writeback puts every byte-producing op's AX_VAL into it.  Decode
        # it per-byte with the LM byte-head argmax (the spec re-quantiser; residue-
        # immune, no torch.round).  One uniform path — no python if/elif on the op.
        ax = _decode_reg_from_nibbles(state, L, L.AX)
        # -- FILE OP: the ONE class not computed neurally (§Tool Use Mode).  The
        # model has no rules for OPEN/READ/CLOS/PRTF, so its PC/SP/AX for this row
        # are meaningless — the DRIVER performs the whole op via the TOOL_CALL
        # runner and overrides the registers.  READ's bytes re-enter the token
        # stream as their OWN §Memory KV frames so LC reads them back byte-exact.
        if fio is not None and op in _FS.FILE_OPCODES:
            new_ax, new_sp, byte_stores = _FS.dispatch_file_op_driver(
                op, cur_ax & 0xFFFFFFFF, imm, cur_sp, store_log, fio,
                data_seg=data_seg, slot=4)
            pc = cur_pc + 1                     # file ops advance PC by one (no branch)
            sp = new_sp
            bp = cur_bp
            ax = new_ax & 0xFFFFFFFF
            frame = _build_frame(pc, ax, sp, bp, stk)
            trace.append(ax & mask)
            frame_idx += 1
            stream += frame
            # lay READ's bytes into the KV log as one store frame per byte, so a
            # later LC(addr) attends to the byte the file delivered.
            for (baddr, bval) in byte_stores:
                bframe = _build_frame(pc, ax, sp, bp, stk,
                                      mem_addr=baddr, mem_val=bval & 0xFF)
                frame_idx += 1
                store_log[frame_idx] = (baddr, bval & 0xFF)
                stream += bframe
            if verbose:
                print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):4s} -> "
                      f"pc'={pc} ax={ax&0xFFFFFFFF} sp={sp} (FILE, "
                      f"{len(byte_stores)} byte-stores)")
            cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
            if pc < 0 or pc >= len(code):
                break
            continue
        # store bookkeeping: which address does this op write, and what value?  All
        # addresses/values are derived from the PRE-step registers the driver
        # tracks (the store target is decided before the op runs; the value is a
        # register the model already emitted).  This is the SI store contract.
        s_addr = s_val = 0
        is_store = False
        if op in (isa.SI, isa.SC):
            # SI: *pop = AX.  The popped stack top MEM[cur_sp] is the target ADDRESS.
            is_store = True; s_addr = _mem_top(store_log, cur_sp); s_val = ax & mask
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & mask
        elif op == isa.JSR:
            is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
        elif op == isa.ENT:
            is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
        frame = _build_frame(pc, ax, sp, bp, stk,
                             mem_addr=(s_addr if is_store else 0),
                             mem_val=(s_val if is_store else 0))
        trace.append(ax & mask)
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        stream += frame
        vis_stream += frame
        # --- I/O: PRTF emits a VISIBLE byte via the think-tag protocol -----------
        # The printed byte is the model's OWN decoded AX byte-0 (a genuine LM-head
        # argmax over the nibble band, not a python copy).  PRTF leaves AX/SP/BP
        # unchanged (its dispatch rule only does PC += 1), so this decoded AX is the
        # value ``printf("%c", AX)`` prints.  We exit the think block, emit the byte,
        # re-enter — so ``visible_output(vis_stream)`` yields exactly ``out``.
        if op == isa.PRTF:
            emit_b = ax & 0xFF
            if out is not None:
                out.append(emit_b)
            vis_stream += [V.THINK_END, emit_b, V.THINK_START]
        if verbose:
            print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):4s} -> "
                  f"pc'={pc} ax={ax&0xFF} sp={sp} bp={bp} stk={stk} "
                  f"store={'Y' if is_store else '.'}@{s_addr}={s_val} halt={halted}")
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        if halted or pc < 0 or pc >= len(code):
            break
    vis_stream += [V.THINK_END, V.HALT]            # close think, then terminate
    if out is not None:
        return (trace, vis_stream) if collect_tokens else trace
    if collect_tokens:
        return trace, stream
    return trace


def run_pure_forward_complete_emit(model, L: PureForwardCompleteLayout,
                                   code: List[isa.Instr], max_steps: int = 512,
                                   verbose: bool = False, seed_mem=None,
                                   mask: int = 0xFF, n_rt_pool: int = 32):
    """DIRECT-CAM in-transformer JIT: ``run_pure_forward_complete`` + ``isa.EMIT``.

    Same APPEND-ONLY pure-forward step (one ``model.forward`` per VM step, state read
    from the register CAM at the latest frame, the store log persistent), but
    ``isa.EMIT`` is serviced as a DRIVER control op (no FFN dispatch rule) exactly as
    JSR/ENT/LEV are: the model FETCHES the EMIT instruction from the code CAM (proving
    fetch@PC serves runtime-appended frames), the register CAM reconstructs the input
    state, and the driver reads the produced word off the model-reconstructed INPUT
    registers (op<-AX, imm<-STACK0), writes it into a private mutable code copy
    ``run_code[target]`` AND — when ``target`` is a HIGH address past the
    ``1..len(code)`` positional rows — into ``runtime_frames``, a COMPACT
    ``{addr -> Instr}`` map overlaid as address-keyed code frames on the
    position-invariant direct-CAM lanes.

    THE CEILING LIFT: the fetch@PC CAM (``_bake_code_cam_head``, ALiBi slope 0,
    ``CODE_ADDR_BITS`` wide) matches on the ``CODE_ADDR_BITS``-bit address key ALONE,
    so a runtime frame keyed bits(440000) resolves fetch@PC=440000 at a COMPACT
    physical row with NO positional decay — the RoPE ~12-16-bit ceiling (slow rotary
    lanes) is lifted to the full ``CODE_ADDR_BITS`` range.  ``n_rt_pool`` compact
    runtime code-frame rows are RESERVED up-front (right after the positional code
    rows) so the append-only frame geometry stays stable as EMIT fills them.

    Same-address re-emit OVERWRITES the slot (``run_code`` in place; ``runtime_frames``
    is a dict) -> latest-write-wins by CONSTRUCTION (one frame per address), so no CAM
    recency tiebreak is needed — the SAME contract 8e6946e7 used on the RoPE path.

    Returns ``{"ax_trace", "ref_trace", "exact", "steps", "produced_code",
    "runtime_frames", "max_emit_addr"}``.  ``ref_trace`` is ``isa.interpret`` (the c4
    oracle) over the initial ``code`` with the ``seed_mem`` data segment preloaded."""
    if not _cfm_emit_enabled():
        raise RuntimeError("run_pure_forward_complete_emit requires C4_CFM_EMIT=1")
    if not getattr(L, "cfm", False):
        raise RuntimeError("EMIT needs the CFM (C4_PF_CFM=1) direct-CAM code path")

    mem_init = dict(seed_mem or {})
    ref_trace = isa.interpret(list(code), max_steps=max_steps, mem_init=mem_init or None)

    # PRE-SEED the data §Memory (the "input file" a loaded compiler reads via LI/LC)
    # as leading STORE frames — the same seed-frame mechanism the read-only path uses.
    seed_frames, store_log = _seed_frames(seed_mem or {})
    n_seed = len(store_log)
    init_frame = _build_frame(0, 0, SP_INIT, SP_INIT, 0)

    n_code = len(code)                             # positional code rows never shrink
    # STREAM (append-only): BOS | code rows | RESERVED runtime code rows | seed data
    # frames | init frame | ... step frames.  The reserved runtime rows are neutral MEM
    # tokens until EMIT fills them (the overlay writes IS_CODE + bits(addr) into the
    # used ones); reserving them up-front keeps every later position STABLE across
    # steps (the register CAM latest-write-wins; the store CAM persistent).
    code_frame_toks: List[int] = ([V.MEM] * n_code + [V.MEM] * n_rt_pool)
    stream: List[int] = [V.BOS] + code_frame_toks + seed_frames + init_frame

    run_code = list(code)                          # mutable positional code (low-addr)
    runtime_frames: Dict[int, isa.Instr] = {}      # HIGH-address emits: addr -> Instr
    # A HIGH-address emit lives ONLY in ``runtime_frames`` (a compact dict), never in a
    # giant ``run_code`` list — the direct-CAM address key decouples logical address
    # from physical row, so a 2^20 target costs one dict slot + one KV row, not 2^20
    # list entries.  ``_fetch`` mirrors the model's code CAM: positional rows for
    # addr < n_code, the runtime dict for high addrs (else NOP == "no such frame").
    def _fetch(pc: int) -> isa.Instr:
        if 0 <= pc < len(run_code):
            return run_code[pc]
        return runtime_frames.get(pc, isa.Instr(isa.NOP, 0))

    def _reachable(pc: int) -> bool:
        return (0 <= pc < len(run_code)) or (pc in runtime_frames)

    trace: List[int] = []
    cur_pc = 0
    cur_sp = cur_bp = SP_INIT
    cur_ax = 0
    frame_idx = n_seed
    max_emit_addr = -1

    for _ in range(max_steps):
        if len(runtime_frames) > n_rt_pool:
            raise RuntimeError(f"EMIT produced {len(runtime_frames)} runtime frames "
                               f"> reserved pool {n_rt_pool}; raise n_rt_pool")
        overlay = make_overlay_complete(run_code, L, store_log=store_log,
                                        runtime_frames=runtime_frames,
                                        code_rows=n_code,
                                        frame_start=1 + n_code + n_rt_pool)
        toks = torch.tensor([stream])
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)
            for blk in model.blocks:
                x = blk(x)
        state = x[0, -1]

        ins = _fetch(cur_pc)
        op, imm = ins.op, ins.imm
        pc = _snap_lane(state[L.PC_VAL])
        sp = _snap_lane(state[L.SP_VAL])
        bp = _snap_lane(state[L.BP_VAL])
        stk = _snap_lane(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5
        ax = _decode_reg_from_nibbles(state, L, L.AX)

        # HIGH-ADDRESS CONTROL FLOW (driver bookkeeping, keeps the model build byte-
        # identical to golden 069cc32f): a JMP/BZ/BNZ TARGET rides the fetched
        # instruction's immediate (IMM_CLEAN), whose SIGNED nibble width is sized for
        # the corpus code_size — a target above that range folds in the model's PC
        # decode (a 4096-target -> 0).  The BRANCH DECISION is the model's (its AX
        # predicate + PC transition are correct); only the target VALUE needs the wide
        # immediate.  So the driver recomputes the taken target from the fetched imm
        # (exactly as it already derives EMIT's target + store addresses from the
        # fetched instruction) — no model weight changes, so C4_CFM_EMIT stays golden.
        # The model still FETCHES the branch instruction from the code CAM and computes
        # AX; this only widens the driver's PC bookkeeping past IMM_CLEAN's range.
        if op == isa.JMP:
            pc = imm & ((1 << CODE_ADDR_BITS) - 1)
        elif op == isa.BZ:
            pc = (imm & ((1 << CODE_ADDR_BITS) - 1)) if (ax & mask) == 0 else (cur_pc + 1)
        elif op == isa.BNZ:
            pc = (imm & ((1 << CODE_ADDR_BITS) - 1)) if (ax & mask) != 0 else (cur_pc + 1)

        # ---- EMIT: driver-serviced control op (no FFN rule) --------------------
        # The model FETCHED this EMIT from the code CAM; the register CAM reconstructed
        # AX + STACK0.  op<-AX, imm<-STACK0(popped); target = EMIT imm.  We use the
        # PRE-step (driver-tracked) registers for the produced word — the driver-service
        # contract 8e6946e7 uses (the model's PC/AX for the EMIT row itself are the
        # control op's, not a computed transition).
        if op == isa.EMIT:
            produced_op = cur_ax & 0xFF            # AX rides the produced op
            produced_imm = _mem_top(store_log, cur_sp) & 0xFF   # STACK0 rides imm
            target = imm
            produced = isa.Instr(produced_op, produced_imm)
            if target < n_code:                    # LOW addr -> positional row (in place)
                while target >= len(run_code):
                    run_code.append(isa.Instr(isa.NOP, 0))
                run_code[target] = produced
            else:                                  # HIGH addr -> position-invariant frame
                runtime_frames[target] = produced  # latest-write-wins (dict compaction)
                max_emit_addr = max(max_emit_addr, target)
            # EMIT pops the immediate it consumed off the KV-backed stack: drop the
            # LATEST (highest frame_idx == the current top-of-stack, latest-write-wins)
            # store frame at MEM[cur_sp], SP += 4 (the pop).
            _pop_fi = None
            for fi in sorted(store_log):
                if fi > n_seed and store_log[fi][0] == cur_sp:
                    _pop_fi = fi
            if _pop_fi is not None:
                del store_log[_pop_fi]
            new_sp = (cur_sp + 4) & 0xFFFFFFFF
            new_pc = cur_pc + 1
            trace.append(cur_ax & mask)
            # the EMIT step still emits a (neutral) frame so the stream stays a clean
            # frame grid: PC advances by one, registers otherwise unchanged.
            frame_idx += 1
            stream += _build_frame(new_pc, cur_ax, new_sp, cur_bp, 0)
            if verbose:
                print(f"  step pc={cur_pc} op=EMIT -> code[{target}]="
                      f"({isa.NAMES.get(produced_op, produced_op)} {produced_imm}) "
                      f"pc={new_pc} (runtime_frame={target >= n_code})", flush=True)
            cur_pc, cur_sp = new_pc, new_sp
            # cur_ax/cur_bp unchanged by EMIT.
            if not _reachable(cur_pc):
                break
            continue

        # store bookkeeping (same SI store contract as run_pure_forward_complete).
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
        frame = _build_frame(pc, ax, sp, bp, stk,
                             mem_addr=(s_addr if is_store else 0),
                             mem_val=(s_val if is_store else 0))
        trace.append(ax & mask)
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        stream += frame
        if verbose:
            print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):4s} -> "
                  f"pc'={pc} ax={ax & 0xFF} sp={sp} bp={bp} stk={stk} "
                  f"store={'Y' if is_store else '.'}@{s_addr}={s_val} halt={halted}",
                  flush=True)
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        if halted or not _reachable(pc):
            break

    return {"ax_trace": trace, "ref_trace": ref_trace,
            "exact": trace == ref_trace, "steps": len(trace),
            "produced_code": run_code, "runtime_frames": dict(runtime_frames),
            "max_emit_addr": max_emit_addr}


def _decode_reg_from_nibbles(state: torch.Tensor, L, reg_base: int) -> int:
    """Decode a 4-byte register value from its 8 nibble dims via the LM byte-head
    argmax per byte (the spec re-quantiser: ``argmax_v (2·v·nib − v²)``, no round).
    Residue-immune — the argmax over the 256 byte candidates snaps each byte to the
    exact integer regardless of the O(1e-6) fp residue on the nibble dims."""
    val = 0
    for bi in range(4):
        lo = float(state[reg_base + 2 * bi + 0])
        hi = float(state[reg_base + 2 * bi + 1])
        # per-byte argmax over 256 candidates: v = lo' + 16*hi' where lo',hi' are the
        # nibble argmaxes (each argmax_n (2·n·x − n²) over 0..15).
        best_lo = _snap_nib(lo)
        best_hi = _snap_nib(hi)
        val |= (best_lo + 16 * best_hi) << (8 * bi)
    return val


def _snap_nib(x: float) -> int:
    """argmax_n (2·n·x − n²) over n in 0..15 — the nibble re-quantiser (no round)."""
    best, bn = -1e30, 0
    for n in range(16):
        s = 2.0 * n * x - n * n
        if s > best:
            best, bn = s, n
    return bn


def _mem_top(store_log: Dict[int, Tuple[int, int]], sp: int) -> int:
    """The address value that lives at MEM[sp] = the most recent store to address
    ``sp`` (latest-write-wins), i.e. the popped stack top's VALUE.  For SI the
    popped top is an ADDRESS the store writes to.  The driver reads it from the
    KV log (the same store stream the model attends to)."""
    val = 0
    for fi in sorted(store_log):
        a, v = store_log[fi]
        if a == sp:
            val = v
    return val
