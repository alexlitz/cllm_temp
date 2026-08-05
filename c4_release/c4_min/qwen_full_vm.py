"""FUSED FULL VM through a genuine ``Qwen2Model.forward`` — compute AND control.

This closes the four gaps ``qwen_embed`` / ``qwen_mlp_ops`` left open:

  1. **Compute + control FUSED.**  ``qwen_embed.run_program_through_qwen`` computed
     the ALU transition in Python and only round-tripped AX through Qwen. Here the
     WHOLE VM step — recompose, PC-fetch-from-data, opcode decode, dispatch, branch
     delta, mod-fold, AND the register CAM that reconstructs state from the prior
     frame — runs inside ONE ``Qwen2Model.forward`` per program step. The only
     Python on the compute path is the standard autoregressive emit+append (argmax
     the next register values, append the next frame). This is the
     ``nibble_pure_forward`` mechanism PORTED onto a real Qwen2.

  2. **Functions (JSR/ENT/ADJ/LEV).**  JSR/ENT push (return-PC / saved-BP), ADJ/LEV
     unwind. The register housekeeping is FFN dispatch (``_call_dispatch_rules``);
     the return-PC / saved-BP push+pop rides a small explicit CALL STACK the driver
     re-embeds into the next frame (the same "state lives in the token stream"
     contract the register frame uses). All the arithmetic (SP-=imm, BP:=SP, PC set)
     is computed in the Qwen forward.

  3. **MUL/DIV/MOD/SHL/SHR.**  Wired in from the c4_min nibble gadgets: SHL/SHR +
     bitwise as SwiGLU FFN blocks (Qwen's MLP), 8-bit MUL/DIV/MOD as the table
     expert. The table's hidden width is the D-budget wall (see ``build`` /
     ``fits_stock``).

  4. **Corpus sample.**  ``run_corpus`` runs a stratified op-family set through the
     real Qwen forward and reports the argmax-exact pass fraction vs
     ``isa.interpret``.

How the blogspec block stack maps onto Qwen2
============================================
A ``nibble_pure_forward`` model is a stack of blogspec ``Block``s, each
``ffn(attn(x))`` (softmax1 + ALiBi attn, then SwiGLU FFN, both norm-free additive
residual). A Qwen2 ``Qwen2DecoderLayer`` is ``x + attn(RMSNorm(x))`` then
``x + mlp(RMSNorm(x))`` — the SAME two additive sub-layers, wrapped in RMSNorm.
So block ``i``'s ``(attn, ffn)`` ports directly onto Qwen layer ``i``'s
``(self_attn, mlp)``:

  * **MLP** — gate←W_up (silu side), up←W_gate (linear side), down←W_down; biases
    fold onto the ONE lane. RMSNorm-K makes the normed input ``r·x`` (r≈1); the
    byte-head argmax absorbs the ~1e-5 residue (``qwen_mlp_ops`` proof).
  * **attention** — the register/memory CAM, re-expressed with Qwen's RoPE recency
    (fast-lane distance decay picks the LATEST frame, replacing ALiBi) + a BOS sink
    (plain softmax + a content-free sink row == softmax1's ZFOD). The value copy
    (register nibbles) is untouched by RoPE.
  * **RMSNorm** — every layernorm weight = ``K/sqrt(H)`` on a compensator lane = K.

Frame / head budget
--------------------
The per-BYTE ingest CAM (``nibble_pure_forward``: 20 heads) does NOT fit Qwen's 14
query heads. So each register's WHOLE value rides on ONE frame token whose residual
carries all its nibbles (the driver overlay writes them; the embedding table stays
universal). ONE head per register (PC/AX/SP/BP/STACK0) then gathers that register's
16 nibbles with a single RoPE content-match + recency. 5 register heads + 1 memory
head = 6 ≤ 14, all in KV-group 0 of the 14/2 GQA (``repeat_kv`` broadcasts it).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from . import nibble_pure_forward as PF
from .nibble_pure_forward import (
    PureForwardLayout, INGEST_REGS, build_pure_forward_model, SP_INIT,
)
from .nibble_pure_forward_complete import _decode_reg_from_nibbles
from .nibble_vm import S, RELU_S, SILU_S, SILU_HALF, _empty_spec


# RMSNorm compensator. It must DOMINATE every residual value so RMSNorm ~ identity
# (r = K/sqrt(K^2 + S) ~ 1 with S the real-lane energy). Unlike qwen_embed (which
# only carried a single AX byte, S ~ a few hundred, K=4000 sufficed), the FULL VM
# carries SP/BP ~ 0x10000 = 65536 on the recompose value lanes, so S ~ 65536^2 per
# lane. The recompose value lane must be within 0.5 of the true SP/BP for the LM
# value-argmax to snap correctly, i.e. 65536*(1-r) < 0.5 -> 1-r < 7.6e-6. With
# 1-r ~ S/(2K^2) and S ~ 5*65536^2 that needs K > ~4e7. K=1e8 gives 1-r ~ 1.1e-6,
# so SP/BP recompose error < 0.1 (snaps exact) and all small lanes are unperturbed.
NORM_K = 1e8
ROPE_THETA = 1_000_000.0


# ---------------------------------------------------------------------------
# ALU wiring flags (all default to the SHALLOWER, byte-exact modules; each is a
# kill-switch back to the older/general path).  See docs/FLAG_REGISTRY.md.
# ---------------------------------------------------------------------------
def _div_longdiv() -> bool:
    """DIV/MOD path selector.  DEFAULT OFF -> the fp32-byte-exact radix-16
    digit-recurrence divide (``div_radix16_hardened``, 88 blocks).  ON
    (``C4_DIV_LONGDIV=1``) -> the base-16 long-division fallback
    (``nibble_alu32.compile_divmod_blocks``, ~262 blocks) — retained as an escape
    hatch for the retired log-sink regression."""
    return os.environ.get("C4_DIV_LONGDIV", "0") == "1"


def _cfm_emit_enabled() -> bool:
    """``C4_CFM_EMIT`` (DEFAULT OFF): the in-transformer JIT on the GOLDEN CFM path.

    ON -> the CFM driver (``run_program`` / ``qwen_lean_forward.run_program_lean``)
    services the ``isa.EMIT`` opcode (value 45, ABOVE the OP_IS band so the golden
    069cc32f build is byte-identical) by APPENDING / rewriting a CODE frame into the
    SAME KV §Memory the fetch@PC CAM (``_bake_code_cam``) reads: the produced
    instruction word is read from the reconstructed input registers (op<-AX,
    imm<-STACK0), the target code address is the EMIT immediate, and the new frame
    joins the runtime code list ``_overlay_code_frames`` writes -> fetchable by PC
    next step with NO residual-width (D) growth (program-length INDEPENDENT, unlike
    the bespoke fixed-width ``CODE_WORD`` band).  A same-address re-emit OVERWRITES
    the slot (latest-write-wins by construction — one code frame per address, so no
    CAM recency tiebreak is needed).  OFF -> the read-only code path is byte-
    identical to the historical CFM driver."""
    return os.environ.get("C4_CFM_EMIT", "0") not in ("0", "", "false", "False")


def _div_lean() -> bool:
    """Radix-16 divide VARIANT selector (only takes effect on the radix-16 path,
    i.e. when NOT ``C4_DIV_LONGDIV``).  DEFAULT ON (``C4_DIV_LEAN`` unset or ``1``)
    -> the LEAN radix-16 digit-recurrence divide (``div_radix16_lean``, 80 blocks:
    3-limb ``16^p``-recompose borrow) — the SHALLOWEST byte-exact divide.  OFF
    (``C4_DIV_LEAN=0``) -> the HARDENED radix-16 variant (``div_radix16_hardened``,
    88 blocks: nibble-lane Kogge-Stone borrow), retained as the escape hatch.

    The DEFAULT was flipped from hardened to lean after the lean variant was
    proven byte-exact IN-VM through the real fused Qwen forward: the 8-bit-IMM
    DIV/MOD grid (288/288, dividends over byte edges + spread, divisors incl b==0)
    matched the hardened control 288/288, and ``pytest -k muldiv`` passed under
    both.  The c4 ISA feeds 8-bit immediates, so the lean variant's documented
    fp32 residue floor on the LARGE-divisor 32-bit adversarial classes (why the
    +8-block hardening exists standalone) is NOT exercised on this path; the
    escape hatch (``C4_DIV_LEAN=0``) restores the hardened variant for any wide
    divisor use.

    The two variants allocate DIFFERENT LEANDIV scratch bands (lean B0/B1/RV vs
    hardened G0/P0/G1/P1), so this flag MUST be read identically at layout-extend
    time and builder time (both in ``QwenFullLayout`` and ``build``/``_block_specs``)
    or ``L.D`` and the built blocks disagree."""
    return os.environ.get("C4_DIV_LEAN", "1") != "0"


def _kb_batched_prefix() -> bool:
    """KB-precompute (``k*b`` normalise) resolve selector, shared by the
    long-division fallback + the KB threshold table.  DEFAULT ON -> the batched
    base-16 Kogge-Stone parallel prefix (``div_kb_lookahead``, ~9 blocks).  OFF
    (``C4_KB_BATCHED=0``) -> the historical 15x6 serial ripple (~92 blocks)."""
    return os.environ.get("C4_KB_BATCHED", "1") != "0"


def _width_narrow() -> bool:
    """Static width-narrowing of ALU ops (``width_narrow.infer_widths`` +
    narrowed builders).  DEFAULT OFF (``C4_WIDTH_NARROW=1`` to enable) — the
    biggest behaviour change; wired but deferred to the GPU byte-exact run."""
    return os.environ.get("C4_WIDTH_NARROW", "0") == "1"


def _const_operand() -> bool:
    """Constant-operand fast paths for MUL/DIV/MOD (``const_mul`` / const-divisor
    ``const_divmod_digitrec``).  DEFAULT OFF (``C4_CONST_OPERAND=1`` to enable) —
    wired as a detection hook; deferred to the GPU byte-exact run."""
    return os.environ.get("C4_CONST_OPERAND", "0") == "1"


def _mem_addr_bits(default: Optional[int] = None) -> int:
    """Number of §Memory address bits ``_bake_memory_cam`` KEYS onto RoPE lanes.

    DEFAULT ``MEM_ADDR_BITS`` = 8 (byte-identical golden build; the 1096 corpus +
    edge_corpus use 8-bit data addresses, so the ADDR_BIN upper bits are 0 and never
    contribute — widening past 8 with all-zero upper bits is byte-neutral for those
    programs too, but 8 is the historical default and keeps the weight matrices
    literally identical).  Set ``C4_MEM_ADDR_BITS=<n>`` (or pass ``mem_addr_bits=`` to
    ``build``) to WIDEN for 32-bit programs like doom (needs 18 bits; see the
    ``MEM_ADDR_BITS`` module comment for the fidelity ceiling).  Clamped to the
    physical band width (``ADDR_BITS``=32) and the lane budget in ``_bake_memory_cam``."""
    if default is None:
        default = MEM_ADDR_BITS
    v = os.environ.get("C4_MEM_ADDR_BITS")
    return default if v is None else int(v)


def _pack_memcam() -> bool:
    """Drop the DEAD head-dim padding the pure-forward layout carries so the Qwen
    residual floor is not inflated by it.  DEFAULT OFF (``C4_PACK_MEMCAM=1`` to
    enable).

    ``build_pure_forward_model`` pads its layout up to ``n_heads *
    MEM_HEAD_CHANNELS`` (= 21*51 = 1071) so that the pure-forward ``Transformer``'s
    §Memory head — which slices ``head_dim = dim // n_heads`` channels — has room
    for its 51 local channels.  But ``QwenFullLayout`` only KEEPS the layout ``L``
    from that call and DISCARDS the pure-forward ``Transformer`` model (it re-bakes
    every attention head on the Qwen geometry ``head_dim=64, n_heads=14``).  So the
    ~800 head-dim ``_hdpad`` scalars (and the other anonymous ``_*pad`` fill) are
    NEVER read or written by any Qwen weight matrix or by the driver's per-position
    overlay — they are pure dead residual width that pushes ``D_used`` (hence the
    ``hidden_for`` floor) up unnecessarily (base+mem 1121 -> 1152, an 18-head model
    instead of the 14-head stock).

    When ON, ``QwenFullLayout`` COMPACTS the layout by removing every anonymous
    ``_*pad`` scalar and renumbering all named bands + the ``L.<band>`` position
    attributes densely.  The Qwen build reads the compacted positions, so the built
    weights are byte-identical in VALUE (same magnitudes at the same, now-denser
    dims) — only the residual is narrower.  A final pad to a multiple of
    ``head_dim`` keeps the Qwen attention head partition valid.  Verified: LI/LC/SI/
    SC + var_* round-trip byte-exact ON vs OFF; the ``_fingerprint_build`` golden (a
    SEPARATE compact-alloc path) is unchanged flag-OFF."""
    return os.environ.get("C4_PACK_MEMCAM", "0") == "1"


# ``_*pad`` scalar-name prefixes the various layout builders use for anonymous
# head-dim / alignment fill.  These carry NO role: never read/written by any Qwen
# weight or the driver overlay (audited).  ``_pack_memcam`` drops exactly these.
_PAD_NAME_PREFIXES = ("_hdpad", "_pfpad", "_widepad", "_bwpad", "_alupad",
                      "_leanpad", "_pad")


def _compact_layout_drop_pad(L, head_dim: int) -> int:
    """Remove every anonymous ``_*pad`` scalar from ``L`` and renumber all remaining
    named bands + every int position attribute (on ``L`` AND on its nested band-
    holder objects like ``L.ALU32`` / ``L.LEANDIV``) densely, order-preserving.

    Returns the number of dims removed.  ``L.D`` / ``L._off`` are reset to the
    compacted extent, then padded up to a multiple of ``head_dim`` (so the Qwen
    attention head partition stays valid).  Only the anonymous pad scalars move
    OUT; every meaningful band keeps its width and relative order, so the Qwen
    build addresses byte-identical weights at a narrower width.

    A position attribute is remapped iff its value equals a band BASE in the OLD
    layout (``old_bases``).  That discriminates a genuine dim offset from a same-
    typed COUNT/width field (e.g. ``ALU32.RN`` / ``LEANDIV.RN``, a nibble count that
    is provably NOT any band's base) — so counts are left untouched while every real
    offset — including the ones the nested holders store under a DIFFERENT attribute
    name than their band (``ALU32.A`` -> ``ALU_A``, ``LEANDIV.DIV_RES`` -> ``LR_DIV``)
    — is moved."""
    items = sorted(L._names.items(), key=lambda kv: kv[1][0])
    old_bases = {b for (b, _w) in L._names.values()}
    remap: Dict[int, int] = {}
    new_names: Dict[str, Tuple[int, int]] = {}
    new_off = 0
    for name, (base, width) in items:
        if any(name.startswith(p) for p in _PAD_NAME_PREFIXES):
            continue                       # drop the dead pad scalar
        remap[base] = new_off
        new_names[name] = (new_off, width)
        new_off += width
    removed = L._off - new_off
    L._names = new_names

    def _remap_int_attrs(obj):
        # Remap every int attr whose value is a KNOWN OLD BAND BASE (a real dim
        # offset).  ``bool`` is excluded (it is an int subclass); a count/width that
        # is not a band base (``RN``) is never in ``old_bases`` so it is left alone.
        for attr, val in list(vars(obj).items()):
            if attr.startswith("_") or isinstance(val, bool) or not isinstance(val, int):
                continue
            if val in old_bases:
                setattr(obj, attr, remap[val])

    _remap_int_attrs(L)
    # Nested band-holder objects (ALU32, LEANDIV, and any other object attr that
    # carries dim offsets) — remap their offsets too.
    for _a, _v in list(vars(L).items()):
        if _a.startswith("_") or not hasattr(_v, "__dict__"):
            continue
        if isinstance(_v, (int, float, str, bytes, list, dict, tuple, set)):
            continue
        _remap_int_attrs(_v)
    # Reset extent + pad to a multiple of head_dim for the Qwen head partition.
    L._off = new_off
    while L._off % head_dim != 0:
        L._scalar(f"_qpad{L._off}")
    L.D = L._off
    return removed


QWEN_HEAD_DIM = 64

# stock Qwen2.5-0.5B residual / FFN / layer / head budget (the fit ceiling).
STOCK_HIDDEN = 896
STOCK_INTERMEDIATE = 4864
STOCK_LAYERS = 24
STOCK_QHEADS = 14


# The registers the CAM reconstructs each step, in frame order. Each rides on ONE
# frame token carrying all 16 of its nibbles (compact frame — fits the head budget).
CAM_REGS = ["PC", "AX", "SP", "BP", "STACK0"]

# ---------------------------------------------------------------------------
# CODE-FROM-MEMORY (blogspec "Universal = bytecode fetched by PC").
#
# The baked CODE_OP[k]/CODE_IMM[k] table caps program size (PC >= code_size ->
# IndexError) AND makes D scale ~3 per instruction (the ~3*code_size PC_IS +
# CODE_OP + CODE_IMM residual bands), pushing hidden_size past stock-896 for a
# long program.  The spec's true mechanism is the SAME as LI/SI: the program
# lives in the KV §Memory, one CODE frame per instruction keyed on its address i.
# Each step FETCHES the instruction at PC via the address-keyed CAM (like a load of
# mem[PC]), delivering op -> OP_VAL and imm -> IMM.  This makes the fetch
# PROGRAM-LENGTH-INDEPENDENT: no code_size table, no per-instruction residual band.
#
# The address bits key on the SLOWEST rotary lanes (near-identity RoPE) so the dot
# is POSITION-INVARIANT: unlike the store log (whose loads are near their stores),
# the code frames span the WHOLE window — a distant frame is up to ~program-length
# positions from the fetch query, so a bit on a fast-rotating lane loses its match
# on that frame and mis-fetches (frame 0 out-scoring frame 256 — a real off-by-
# address bug).  The lowest-freq 12 rotary lanes (theta=1e6) stay cos>=0.997 even at
# Δpos~500, so 12 bits key a PC up to 4095 (covers the full ~4000-instr c4 compiler
# and every muldiv subroutine, code_size- and width-independent).
#
# CONFIGURABLE (``C4_CODE_ADDR_BITS=<n>``, DEFAULT 12 -> byte-identical golden
# 069cc32f; mirrors ``C4_MEM_ADDR_BITS`` / wall #1).  ``_bake_code_cam`` places the
# n bits on the SLOWEST n rotary pairs (lanes ``half-1-b``) + 4 control lanes, so a
# widen keys a longer program.  Because this is the RoPE path, the fidelity ceiling
# is the rotary-lane one: a bit ``b`` on lane ``half-1-b`` rotates with a freq that
# rises with ``b``, so a distant code frame's high address bits decay past cos~0 at a
# Δpos that shrinks with b (the same curve wall #1 measured for the MEM CAM: ~12
# reliable bits at Δ~program-length, up to ~16-18 when the fetch stays near the
# frame).  The cfm pure-forward CODE CAM (``nibble_pure_forward_complete``) has NO
# such ceiling — it uses ALiBi slope 0.0 + no RoPE, so it is position-invariant by
# construction (that is the path the batched/doom verifier + direct-CAM run).
CODE_ADDR_BITS = int(os.environ.get("C4_CODE_ADDR_BITS", "12"))


# ---------------------------------------------------------------------------
# WIDE §Memory address CAM (WALL #1: doom's 32-bit addresses).
#
# ``_bake_memory_cam`` KEYS only ``MEM_ADDR_BITS`` of the 32 physically-carried
# ``ADDR_BIN``/``QRY_BIN`` residual bits onto RoPE lanes.  The corpus + edge_corpus
# use 8-bit data addresses, so 8 bits (256 keyspace) suffice and are the DEFAULT
# (byte-identical to the historical build).  doom is 32-bit: its live addresses in
# the first ~30k executed steps span 0xffb4..0x20a15 (data segment DATA_BASE=0x10000
# heap + a descending stack near SP_INIT=0x10000), which need 18 bits to distinguish
# without aliasing — under the 8-bit CAM ALL ~1088 distinct addresses fold to
# addr&0xFF and collide (doom diverges at its first out-of-256 memory op).
#
# WIDENING is opt-in via ``C4_MEM_ADDR_BITS=<n>`` (or the ``mem_addr_bits=`` build
# kwarg): it places ``n`` address bits on the SLOWEST ``n`` rotary pairs exactly like
# the CODE CAM (lanes ``half-1-b``, near-identity RoPE), plus 3 control lanes (bias /
# gate / code-exclusion) below them, keeping the fixed recency lane 3.  The physical
# ``ADDR_BITS=32`` residual bands already carry every bit, so widening the KEYED count
# is a pure ``_bake_memory_cam`` change — no D/param cost (the bands exist either way).
#
# FIDELITY CEILING (MEASURED, not projected — see _agent_doom_mem_addr_resolve.py):
# the per-bit address dot decays with the store->load position distance Δ as
# ``cos(freq_lane · Δ)``.  A bit ``b`` sits on the lane ``half-1-b`` whose RoPE freq
# rises with ``b``; the SLOWEST lanes (theta=1e6) hold cos>=0.997 out to Δ~500, but a
# high address bit (b>=14, freq>=2.4e-3) rotates PAST cos~0 by Δ~700.  So the reliably-
# keyable width is a function of the recall DEPTH (how far back the matching store is):
#
#   depth<=50  : 8..18 bits all resolve doom's real loads 100% (18 covers the full
#                18-bit span with ZERO aliasing — the clean target WITH eviction).
#   depth<=213 : 14..16 bits resolve 100%; 18 drops to ~84%, 20 to ~48% (high lanes
#                rotated); 8 bits aliases to ~89% (the WALL #1 bug).
#   depth~4000 : only ~12 bits stay position-invariant (the CODE-CAM ceiling) — a raw
#                store-log-before-code-frames window (Δ = n_code + n_store) is NOT
#                wide-keyable for doom; the store log must sit ADJACENT to the query
#                (reorder / eviction) so recalls stay shallow (doom median depth 3,
#                p90 213).
#
# So: 16 bits is the widest reliable keying at doom's p90 recall depth (213); 18 bits
# reaches the full zero-alias span ONLY when an evicting driver keeps the live store
# log shallow (<=~50, well within the lean-evict cap).  Full 32 bits is NOT reliably
# keyable at any non-trivial Δ (the fast address lanes rotate past cos~0).  doom needs
# only 18 (its bits above 18 are constant 0).  The DRIVER side must ALSO widen (write
# full-width addresses + reorder stores adjacent to the query); the corpus drivers keep
# their 8-bit mask (byte-neutral for <256 addresses), and the doom harnesses
# (_agent_doom_*.py) drive the wide path.
MEM_ADDR_BITS = 8   # DEFAULT keyed width (byte-identical golden build; corpus 8-bit)


@dataclass
class QwenArch:
    name: str = "Qwen2.5-0.5B-arch"
    num_attention_heads: int = STOCK_QHEADS
    num_key_value_heads: int = 2
    head_dim: int = QWEN_HEAD_DIM

    def hidden_for(self, dim_needed: int) -> int:
        hd = self.head_dim
        return max(self.num_attention_heads * hd, -(-dim_needed // hd) * hd)


QWEN2_5_ARCH = QwenArch()


def rmsnorm_identity_gamma(hidden_size: int, K: float = NORM_K) -> torch.Tensor:
    return torch.full((hidden_size,), K / math.sqrt(hidden_size), dtype=torch.float32)


def _rope_lane_pair(head_dim: int, slow: bool) -> Tuple[int, int]:
    half = head_dim // 2
    j = (half - 1) if slow else 0
    return j, j + half


# ===========================================================================
# Layout: the pure-forward VM bands + per-register CAM role/value bands + sink +
# compensator. We EXTEND PureForwardLayout so the ported FFN specs address the
# identical dims, then add the CAM tokens' bands.
# ===========================================================================
class QwenFullLayout:
    """PureForwardLayout bands + CAM bands (role marker, per-reg value token nibble
    band, is-frame flag) + BOS sink + RMSNorm compensator, placed in a Qwen hidden
    vector.

    ``efficient_alu`` (default False) attaches the ``nibble_alu32`` scratch bands
    (``L.ALU32``: the operand-byte, MUL column, DIV/MOD nibble long-division and
    per-op RES bands) BEFORE the CAM bands, so MUL/DIV/MOD run as the spec's
    genuine 32-bit fp32 FFN gadgets INSTEAD of the 256x256 lookup table (the
    ~45 GB intermediate wall).  ``recurrent_divmod`` folds the 8 long-division
    iterations into ONE reused iteration body (adds the IT counter band)."""

    def __init__(self, code_size: int, subset: "Subset",
                 efficient_alu: bool = True, recurrent_divmod: bool = False,
                 code_from_memory: bool = True, shift_via_mul: bool = True,
                 div_logsink: bool = False):
        # Build the pure-forward layout (all VM compute bands live here).  MUL/DIV/MOD
        # ALWAYS run through the ``nibble_alu32`` fp32 FFN gadgets (the 256x256x3
        # lookup table has been removed entirely), so the layout carries the ALU32
        # scratch bands, never the (deleted) mdm one-hot bands.
        # CODE-FROM-MEMORY: the program lives in the KV §Memory (code frames), NOT in
        # the CODE_OP[k]/PC_IS[k] residual table, so the vestigial table is sized to a
        # SINGLE slot (~3 dims) instead of ~3*code_size — removes the program-length
        # dependence of the residual width.  ``code_size`` no longer gates the program
        # that can run (the CAM fetches at ANY PC).
        pf_code_size = 1 if code_from_memory else code_size
        self.pf = build_pure_forward_model(
            code_size=pf_code_size, include_memory=subset.memory,
            include_cmp=subset.cmp, include_bitwise=subset.bitwise,
            include_muldiv=False)[1]
        L = self.pf
        self.efficient_alu = efficient_alu
        self.recurrent_divmod = recurrent_divmod
        self.code_from_memory = code_from_memory
        # SHIFT-VIA-MUL/DIV is available only when BOTH the MUL/DIV gadgets (muldiv)
        # and the shift ops (bitwise) are present; a muldiv-less bitwise subset keeps
        # the barrel shifter.  ``shift_via_mul`` requests it; ``self.shift_via_mul`` is
        # the RESOLVED flag the block builder reads.
        self.shift_via_mul = bool(shift_via_mul and subset.muldiv and subset.bitwise
                                  and efficient_alu)
        # DIV/MOD always route through the fp32 base-16 recurrent/unrolled long
        # division (byte-exact in fp32).  The fp64 log-sink reciprocal divide has been
        # removed from the production path so the default build is genuinely fp32.
        # ``div_logsink`` is retained (always resolves False) only so callers passing
        # the historical kwarg do not error.
        self.div_logsink = False
        # DIV/MOD default = the fp32-byte-exact radix-16 digit-recurrence divide
        # (``div_radix16_lean``, 80 blocks — the DEFAULT since the lean grid was
        # proven byte-exact in-VM; ``C4_DIV_LEAN=0`` selects the 88-block hardened
        # variant); ``C4_DIV_LONGDIV=1`` falls back to the base ALU long division
        # (~262 blocks).  The radix-16 divide reads the SAME operands (AX=B,
        # STACK0=A) and (via a copy block) writes the SAME ``L.ALU32.DIV_RES /
        # MOD_RES`` the ax-mux reads, so the ax-mux is unchanged.
        self.div_radix16 = bool(efficient_alu and subset.muldiv and not _div_longdiv())
        if efficient_alu and subset.muldiv:
            from . import nibble_alu32 as _A
            _A.extend_layout_for_alu32(L, recurrent_divmod=recurrent_divmod,
                                       shift_via_mul=self.shift_via_mul)
            if self.div_radix16:
                # LEAN (80 blocks, DEFAULT) vs HARDENED (88 blocks, C4_DIV_LEAN=0).
                # The two allocate DIFFERENT LEANDIV bands, so the SAME module's
                # extend_layout must run here AND at builder time (build/_block_specs)
                # or L.D and the built blocks disagree.
                if _div_lean():
                    from . import div_radix16_lean as _DR
                else:
                    from . import div_radix16_hardened as _DR
                _DR.extend_layout(L)   # allocate the LEANDIV scratch bands (pads L.D)
        # CODE-FROM-MEMORY bands: attach a fixed-width (code_size-INDEPENDENT) code
        # §Memory CAM key/query + value onto the pure-forward layout, BEFORE the CAM
        # bands.  On (DEFAULT) the program lives in these code frames (fetch@PC);
        # off (explicit opt-in) reverts to the baked CODE_OP[k]/PC_IS[k] table.
        L.CODE_KEY_BIN = L.CODE_QRY_BIN = None
        L.CODE_OPV = L.CODE_IMMV = None
        L.IS_CODE = L.IS_FETCH = None
        if code_from_memory:
            L.CODE_KEY_BIN = L._band("CODE_KEY_BIN", CODE_ADDR_BITS)  # instr addr i (KEY)
            L.CODE_QRY_BIN = L._band("CODE_QRY_BIN", CODE_ADDR_BITS)  # current PC (QUERY)
            L.CODE_OPV = L._scalar("CODE_OPV")     # instr op    (VALUE -> OP_VAL)
            L.CODE_IMMV = L._scalar("CODE_IMMV")   # instr imm   (VALUE -> IMM)
            L.IS_CODE = L._scalar("IS_CODE")       # code-frame flag (KEY gate)
            L.IS_FETCH = L._scalar("IS_FETCH")     # fetch-query flag (QUERY gate)
            L.D = L._off                           # extend the pure-forward width
        # PACK-MEMCAM (``C4_PACK_MEMCAM``, DEFAULT OFF): the pure-forward layout pads
        # itself up to ``n_heads*MEM_HEAD_CHANNELS`` for ITS OWN §Memory head — but this
        # Qwen build DISCARDED that pure-forward Transformer and re-bakes attention on
        # the Qwen geometry, so those ~800 head-dim ``_hdpad`` dims are dead weight
        # inflating the residual floor.  Drop the anonymous pad scalars (renumbering
        # every named band densely) BEFORE the CAM bands are appended, so the whole
        # layout — and hence every block that reads ``L.<band>`` — addresses the same
        # weights at a narrower width.  OFF -> byte-identical wider layout.
        self.pack_memcam = _pack_memcam()
        if self.pack_memcam:
            _compact_layout_drop_pad(L, head_dim=QWEN_HEAD_DIM)
        # Append the CAM bands past the (possibly ALU-/code-extended) pure-forward bands.
        off = L.D
        self._names: Dict[str, Tuple[int, int]] = {}
        # per-register ROLE one-hot (which register a frame token carries): 5.
        self.ROLE = off; off += len(CAM_REGS)
        # the per-token 16-nibble VALUE band the CAM reads (a frame reg token puts
        # its register's nibbles here; the CAM copies them into the reg nibble band).
        self.TOK_NIB = off; off += NIB_PER_REG
        # is-frame-token flag (recency key + query exclusion).
        self.IS_TOK = off; off += 1
        self.D_used = off
        self.subset = subset
        self.L = L


# ---------------------------------------------------------------------------
@dataclass
class Subset:
    memory: bool = False
    cmp: bool = False
    bitwise: bool = False
    muldiv: bool = False
    name: str = "base"


SUBSET_BASE = Subset(name="base")
SUBSET_MEM = Subset(memory=True, name="base+mem")
SUBSET_MEM_CMP = Subset(memory=True, cmp=True, name="mem+cmp")
SUBSET_BITWISE = Subset(memory=True, cmp=True, bitwise=True, name="+bitwise")
SUBSET_MULDIV = Subset(memory=True, cmp=True, muldiv=True, name="+muldiv")  # no bitwise
SUBSET_FULL = Subset(memory=True, cmp=True, bitwise=True, muldiv=True, name="full")


# ===========================================================================
# Qwen config + build.
# ===========================================================================
def _qwen_config(hidden_size, intermediate_size, n_layers, vocab_size, arch):
    from transformers.models.qwen2 import Qwen2Config
    return Qwen2Config(
        hidden_size=hidden_size, num_hidden_layers=n_layers,
        num_attention_heads=arch.num_attention_heads,
        num_key_value_heads=arch.num_key_value_heads, head_dim=arch.head_dim,
        intermediate_size=intermediate_size, vocab_size=vocab_size,
        max_position_embeddings=32768, rope_theta=ROPE_THETA, rms_norm_eps=1e-6,
        hidden_act="silu", attention_dropout=0.0, tie_word_embeddings=False,
        use_cache=False, use_sliding_window=False, sliding_window=None,
        attn_implementation="eager",
    )


@dataclass
class QwenFullVM:
    qmodel: object
    QL: QwenFullLayout
    subset: Subset
    K: float
    block_names: List[str]
    cam_layers: Dict[str, int]
    arch: QwenArch
    hidden_size: int
    intermediate_size: int
    n_layers: int                 # STORED physical layers (Qwen num_hidden_layers)
    fits_stock: bool
    embed: torch.Tensor           # [vocab, hidden] token -> residual (injected)
    efficient_alu: bool = True    # MUL/DIV/MOD via nibble_alu32 gadgets (the ONLY path)
    n_applied: int = 0            # layers APPLIED per forward (>= n_layers if recurrent)
    code_from_memory: bool = True   # program in KV §Memory (fetch@PC), not a baked table
    shift_via_mul: bool = False   # SHL/SHR via native MUL/DIV (x*2^n / x//2^n), no barrel
    div_logsink: bool = False     # retired: DIV/MOD are fp32 long division (always False)
    model_dtype: object = torch.float32   # the model is always fp32 (0 fp64 params)


def build(code_size: int = 24, subset: Subset = SUBSET_BASE,
          arch: QwenArch = QWEN2_5_ARCH, K: float = NORM_K,
          efficient_alu: bool = True,
          recurrent_divmod: bool = False, pad_to_stock: bool = False,
          code_from_memory: bool = True, shift_via_mul: bool = True,
          div_logsink: bool = False,
          mem_addr_bits: Optional[int] = None) -> QwenFullVM:
    """Construct a genuine ``Qwen2Model`` whose layers ARE the fused VM step.

    The model is ALWAYS fp32 (``model_dtype == torch.float32``, zero fp64 params).
    DIV/MOD route through the ``nibble_alu32`` fp32 base-16 recurrent/unrolled long
    division (byte-exact in fp32).  The historical fp64 LOG-SINK reciprocal divide
    has been removed from the production path; ``div_logsink`` is retained as a no-op
    kwarg (always resolves False) only so legacy callers do not error.

    ``efficient_alu`` (default True) bakes MUL/DIV/MOD as the ``nibble_alu32``
    spec-faithful 32-bit fp32 FFN gadgets (byte MUL schoolbook + base-16 long
    division).  This is now the ONLY MUL/DIV/MOD path — the 256x256x3 lookup table
    (intermediate ~160465 -> ~45 GB fp32, unbuildable in RAM) has been removed
    entirely.  The efficient ALU has no ~45 GB intermediate wall, full 32-bit-exact
    operands, and needs no operand pruning.  It trades the table WIDTH for DEPTH: MUL
    is ~10 blocks, DIV/MOD ~262 (the 8 long-division iterations).  ``recurrent_divmod``
    folds those 8 iterations into ONE reused layer body (Qwen ``layers`` gets repeated
    module references, so the model STORES ~115 divmod layers but APPLIES 262) — a
    Universal-Transformer-style recurrence exactly mirroring
    ``nibble_pure_forward_complete``.

    ``pad_to_stock`` (default ``False``) builds the model at the EXACT stock
    Qwen2.5-0.5B checkpoint shape — ``hidden_size=896``, ``intermediate_size=4864``,
    ``num_hidden_layers=24`` — instead of the tight compacted shape. The VM occupies
    the first ``len(block_specs)`` layers; the remaining layers are IDENTITY (their
    self-attn AND MLP projections stay zeroed, so ``x + attn(RMSNorm(x)) + mlp(...)``
    = ``x``), and the residual bands past the VM's ``D_used`` stay 0 on every token.
    This proves the SAME weights load and run byte-exact through a config that is
    shape-identical to the released 0.5B (only a subset that already ``fits_stock``
    can be padded; a wider subset raises).

    UNIFIED ALU UNIT SELECTION: the per-op ``C4_ADD_UNIT`` / ``C4_MUL_UNIT`` /
    ``C4_DIV_UNIT`` registry flags (see ``c4_min.alu_units``) are projected onto the
    legacy builder flags (``C4_DIV_LEAN`` / ``C4_DIV_LONGDIV`` / ``C4_MUL_LOOKAHEAD``)
    HERE, before the layout + block builders read them, so the registry is the single
    interface without touching the proven flag-read sites. With NO unit flag set this
    is a NO-OP (byte-identical to golden)."""
    from transformers.models.qwen2 import Qwen2Model

    # Non-invasive: project any C4_*_UNIT registry selection onto the legacy flags
    # the layout/block builders already read. No-op (mutates nothing) when unset.
    from . import alu_units as _AU
    _AU.apply_to_env()

    QL = QwenFullLayout(code_size, subset, efficient_alu=efficient_alu,
                        recurrent_divmod=recurrent_divmod,
                        code_from_memory=code_from_memory,
                        shift_via_mul=shift_via_mul, div_logsink=div_logsink)
    L = QL.L
    block_specs = _block_specs(L, code_size, subset,
                              efficient_alu=efficient_alu,
                              recurrent_divmod=recurrent_divmod,
                              code_from_memory=code_from_memory,
                              shift_via_mul=QL.shift_via_mul,
                              div_logsink=QL.div_logsink)
    # The model is genuinely fp32-vanilla — every param is float32, zero fp64.  DIV/MOD
    # are the fp32 base-16 long division (recurrent or unrolled), byte-exact, so no
    # doubles and no raised K_DIV compensator are needed.
    model_dtype = torch.float32
    block_names = [nm for nm, _ in block_specs]
    apply_order = getattr(L, "_qwen_apply", None)

    dim_needed = QL.D_used + 1
    hidden_size = arch.hidden_for(dim_needed)
    intermediate = max(int(s["W_up"].shape[0]) for _, s in block_specs)
    intermediate = max(intermediate, arch.num_attention_heads * arch.head_dim, 8)
    n_layers = len(block_specs)            # STORED (distinct) physical layers
    n_applied = len(apply_order) if apply_order is not None else n_layers
    comp = QL.D_used                       # compensator lane

    fits_stock = (hidden_size <= STOCK_HIDDEN and intermediate <= STOCK_INTERMEDIATE
                  and n_layers <= STOCK_LAYERS)

    if pad_to_stock:
        if not fits_stock:
            raise ValueError(
                f"subset {subset.name!r} does not fit the stock 0.5B budget "
                f"(hidden={hidden_size} inter={intermediate} layers={n_layers}); "
                "cannot pad to stock")
        # Grow to the EXACT released Qwen2.5-0.5B shape; the VM fills the first
        # n_layers, the rest are identity, the residual past D_used stays 0.
        hidden_size = STOCK_HIDDEN
        intermediate = STOCK_INTERMEDIATE
        n_layers = STOCK_LAYERS

    cfg = _qwen_config(hidden_size, intermediate, n_layers, V.VOCAB, arch)
    qmodel = Qwen2Model(cfg).to(model_dtype).eval()

    embed = _build_embedding(L, hidden_size, comp, K).to(model_dtype)

    with torch.no_grad():
        gamma = rmsnorm_identity_gamma(hidden_size, K)
        qmodel.norm.weight.copy_(gamma)
        # embed_tokens is VESTIGIAL — the VM always feeds inputs_embeds (the driver
        # calls qmodel(inputs_embeds=...)), so this random nn.Embedding never fires.
        # Zero it so the state_dict reflects the actual sparse working model (drops
        # ~240-308K random non-zeros); the forward output is byte-identical.
        qmodel.embed_tokens.weight.zero_()
        for layer in qmodel.layers:
            layer.input_layernorm.weight.copy_(gamma)
            layer.post_attention_layernorm.weight.copy_(gamma)
            for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                        layer.self_attn.v_proj, layer.self_attn.o_proj):
                lin.weight.zero_()
                if lin.bias is not None:
                    lin.bias.zero_()
            for lin in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                lin.weight.zero_()
        for i, (_, spec) in enumerate(block_specs):
            _bake_ffn(qmodel.layers[i].mlp, spec, L, comp)
        cam_layers: Dict[str, int] = {}
        _bake_register_cam(qmodel.layers[0].self_attn, QL, arch, comp, K)
        cam_layers["ingest"] = 0
        if code_from_memory:
            # bake the code-fetch CAM onto the code-cam block's self_attn (fetch@PC).
            code_idx = block_names.index("code-cam")
            _bake_code_cam(qmodel.layers[code_idx].self_attn, QL, arch, comp, K)
            cam_layers["code-cam"] = code_idx
        if subset.memory:
            mem_idx = block_names.index("mem-cam")
            _bake_memory_cam(qmodel.layers[mem_idx].self_attn, QL, arch, comp, K,
                             mem_addr_bits=mem_addr_bits)
            cam_layers["mem-cam"] = mem_idx

    # RECURRENCE (recurrent divmod): re-point qmodel.layers through the apply order
    # so the forward applies the reused iteration-body layers 8x.  The STORED set
    # (the distinct nn.Modules, what the model holds) stays n_layers; qmodel.layers
    # holds repeated references (shared weights, identical math) — exactly the
    # nibble_pure_forward_complete recurrence, ported onto Qwen2's ModuleList.
    if apply_order is not None:
        import torch.nn as _nn
        phys = list(qmodel.layers)
        qmodel._phys_layers = phys
        qmodel.layers = _nn.ModuleList([phys[i] for i in apply_order])
        # Qwen2Model.forward uses self.config.num_hidden_layers for the rotary/mask
        # loop bound in some versions; iterating self.layers directly is the norm,
        # but keep the config in sync with the APPLIED length for safety.
        qmodel.config.num_hidden_layers = len(qmodel.layers)

    return QwenFullVM(qmodel=qmodel, QL=QL, subset=subset, K=K,
                      block_names=block_names, cam_layers=cam_layers, arch=arch,
                      hidden_size=hidden_size, intermediate_size=intermediate,
                      n_layers=n_layers, fits_stock=fits_stock, embed=embed,
                      efficient_alu=efficient_alu, n_applied=n_applied,
                      code_from_memory=code_from_memory,
                      shift_via_mul=QL.shift_via_mul,
                      div_logsink=QL.div_logsink, model_dtype=model_dtype)


# ---------------------------------------------------------------------------
# Block-spec list — mirrors nibble_pure_forward.build_pure_forward_model's FFN
# side but with FUNCTION dispatch rules added.
# ---------------------------------------------------------------------------
def compile_ax_zero(L, dim: int) -> Dict[str, torch.Tensor]:
    """``AX_ZERO = relu(1 - AX_VAL)`` (1 iff AX==0) — the BZ/BNZ branch predicate.

    The code-from-memory path drops ``compile_pc_fetch`` (whose PC one-hot scaled
    with ``code_size``); ``AX_ZERO`` is the only OTHER lane it produced, so this
    ``code_size``-INDEPENDENT FFN recomputes just it (SET, self-clearing).  Same
    relu ramp as ``compile_pc_fetch``'s AX_ZERO unit."""
    spec = _empty_spec(dim, 2)
    # unit 0: self-clear AX_ZERO (SET).
    spec["W_up"][0, L.ONE] = S
    spec["W_gate"][0, L.AX_ZERO] = 1.0
    spec["W_down"][L.AX_ZERO, 0] += -1.0 / SILU_S
    # unit 1: relu(1 - AX_VAL) -> AX_ZERO (1 iff AX==0, 0 for AX>=1).
    spec["W_up"][1, L.AX_VAL] = -RELU_S
    spec["b_up"][1] = RELU_S * 1.0
    spec["W_gate"][1, L.ONE] = 1.0
    spec["W_down"][L.AX_ZERO, 1] += 1.0 / RELU_S
    return spec


def _block_specs(L, code_size, subset, efficient_alu=True,
                 recurrent_divmod=False, code_from_memory=False,
                 shift_via_mul=False, div_logsink=False):
    """The fused-VM FFN block list (one Qwen layer per block).

    Returns ``specs`` (list of ``(name, ffn_spec)``); when the efficient-ALU
    recurrent path is active it ALSO returns an ``apply_order`` on ``L._qwen_apply``
    so ``build`` can point ``qmodel.layers`` at the reused iteration-body layers.
    MUL/DIV/MOD ALWAYS run through the ``nibble_alu32`` fp32 FFN gadgets
    (spec-faithful 32-bit MUL schoolbook + base-16 long division) — the 256x256
    lookup table has been removed, so ``efficient_alu=True`` is the only muldiv
    path (``efficient_alu=False`` simply omits the muldiv blocks).

    ``code_from_memory`` swaps the baked ``pc-fetch`` + ``code-select`` blocks (which
    scaled with ``code_size``) for a single ``code-cam`` attention block that FETCHES
    the instruction at PC out of the KV §Memory (one code frame per instruction),
    delivering ``OP_VAL``/``IMM`` — the "Universal = bytecode fetched by PC"
    mechanism.  ``build`` bakes the code CAM onto that block's ``self_attn``.  Fetch
    is then program-length-independent (no ``CODE_OP[k]`` table)."""
    from .nibble_vm import (
        compile_nibble_to_scalar, compile_pc_fetch, compile_code_select,
        base_dispatch_rules, compile_branch_delta, compile_fold, compile_ffn,
    )
    dim = L.D
    L._qwen_apply = None                   # default: identity apply order
    if code_from_memory:
        # FETCH FROM §MEMORY: the driver overlay writes CODE_QRY_BIN = bits(PC) on the
        # query row; the code CAM (baked on this block's self_attn) selects the code
        # frame whose CODE_KEY_BIN == PC and copies its op/imm into OP_VAL/IMM.  A
        # tiny ax-zero FFN recomputes the only OTHER lane the dropped pc-fetch made.
        specs = [
            ("ingest+recompose", compile_nibble_to_scalar(L, dim)),
            ("code-cam",  compile_ax_zero(L, dim)),   # FFN=AX_ZERO; attn=code fetch CAM
            ("opcode-decode", compile_opcode_decode_full(L, dim)),
        ]
    else:
        specs = [
            ("ingest+recompose", compile_nibble_to_scalar(L, dim)),
            ("pc-fetch",    compile_pc_fetch(L, dim)),
            ("code-select", compile_code_select(L, dim)),
            ("opcode-decode", compile_opcode_decode_full(L, dim)),
        ]
    if subset.memory:
        specs += [("mem-prep", PF.compile_mem_prep(L, dim)),
                  ("mem-cam",  compile_nibble_to_scalar(L, dim))]
    if subset.cmp:
        # cmp-compute emits the UNGATED primitives (CMP_EQ + the raw unsigned ramps
        # MAG_GT/MAG_LT + the operands' sign bits SGN_STK/SGN_AX); cmp-finalize
        # combines them into the SIGNED verdict lanes CMP_GT/CMP_LT that
        # cmp_dispatch_rules reads (LT=CMP_LT, GT=CMP_GT, LE=1-CMP_GT, GE=1-CMP_LT).
        # WITHOUT the finalize block CMP_GT/CMP_LT are never written and stay 0, so
        # LT/GT degenerate to 0 and LE/GE to 1 for ALL operands (#691 BUG 1). The deep
        # nibble_pure_forward.build_pure_forward_model / nibble_pure_forward_complete
        # both append cmp-finalize right after cmp-compute (#673); the compacted Qwen
        # bake was missing it. clamp01(MAG)+0 is byte-identical to the pre-fix unsigned
        # verdict at the 8-bit fold (SGN_*=0 there), so this is signed-correct and
        # regression-free for the unsigned <2^31 corpus.
        specs += [("cmp-compute", PF.compile_cmp_compute(L, dim)),
                  ("cmp-finalize", PF.compile_cmp_signed_finalize(L, dim))]
    eff_mdm = subset.muldiv and efficient_alu
    if subset.bitwise:
        from .nibble_unified import build_bitwise_blocks, _bw_recompose_spec
        # SHIFT-VIA-MUL/DIV: when on, the barrel shifter DROPS SHL/SHR (routed through
        # the native MUL/DIV gadgets below), so build_bitwise_blocks keeps only the
        # OR/XOR/AND per-bit combine and bw-recompose only recomposes those (SHL/SHR
        # write AX via the ax-mux like MUL/DIV, not via the bitwise nibble path).
        barrel_ops = () if shift_via_mul else (isa.SHL, isa.SHR)
        recompose_ops = (isa.OR, isa.XOR, isa.AND) + barrel_ops
        for name, spec in build_bitwise_blocks(L, dim, barrel_shift_ops=barrel_ops):
            specs.append((name, spec))
        specs.append(("bw-recompose", _bw_recompose_spec(L, dim, recompose_ops)))
    # -- EFFICIENT ALU: the nibble_alu32 fp32 FFN gadgets (§Basic Arithmetic /
    #    §Multiplication / §Division) replacing the mdm table.  Each block writes its
    #    op's dedicated RES nibble band UNCONDITIONALLY; the ax-mux copies the active
    #    op's result into the AX nibble band gated on OP_IS[op].  The recurrent
    #    divmod stores ONE reused iteration body (repeated 8x by _qwen_apply). ---
    use_logsink = False   # retired: DIV/MOD are always fp32 long division
    if eff_mdm:
        from . import nibble_alu32 as A
        A._ONE = L.ONE
        alu_ops = [isa.MUL, isa.DIV, isa.MOD]
        # SHIFT-VIA-MUL/DIV: SHL reuses MUL (x * 2^n), SHR reuses DIV (x // 2^n).  The
        # pow2-route block (BEFORE the operand expand + MUL/DIV read AX) overwrites AX
        # with 2^n gated on SHL|SHR; the ax-mux then delivers MUL_RES/DIV_RES into AX
        # on SHL/SHR.  No new VM steps — the shift rides the muldiv blocks already run.
        mux_ops = list(alu_ops)
        if shift_via_mul:
            mux_ops += [isa.SHL, isa.SHR]
        specs.append(("alu-psh-nib", A.compile_psh_nibble_copy(L, dim)))
        if shift_via_mul:
            # one-hot(n) FIRST (its own block: the pow2 write's guard must read it as
            # block input), then the pow2 route overwrites AX with 2^n before expand.
            specs.append(("alu-shift-onehot", A.compile_shift_onehot(L, dim)))
            specs.append(("alu-shift-pow2", A.compile_shift_pow2_route(L, dim)))
        specs.append(("alu-expand", A.compile_expand(L, dim)))
        for name, spec in A.compile_mul_blocks(L, dim):
            specs.append((name, spec))
        # DIV/MOD builder selection.  When the layout carries the LEANDIV bands
        # (radix-16 default, allocated by QwenFullLayout unless C4_DIV_LONGDIV=1)
        # route DIV/MOD through the fp32 radix-16 digit-recurrence divide instead
        # of the base ALU long division (~262).  Two radix-16 variants: LEAN
        # (80 blocks, DEFAULT) or HARDENED (88 blocks, C4_DIV_LEAN=0).  The variant MUST
        # match the one QwenFullLayout used to extend_layout (its bands differ), so
        # the same _div_lean() gate selects the builder module here.  Both write the
        # shared L.ALU32.DIV_RES/MOD_RES the ax-mux reads (via their result-copy
        # block), so the ax-mux + housekeeping are unchanged.
        if getattr(L, "LEANDIV", None) is not None and not _div_longdiv():
            if _div_lean():
                from . import div_radix16_lean as DIVMOD
            else:
                from . import div_radix16_hardened as DIVMOD
        else:
            DIVMOD = A
        div_start = len(specs)
        div_apply = []
        if recurrent_divmod:
            unique, apply_names = DIVMOD.compile_divmod_blocks_recurrent(L, dim)
            name_to_idx = {}
            for name, spec in unique:
                name_to_idx[name] = len(specs)
                specs.append((name, spec))
            div_apply = [name_to_idx[n] for n in apply_names]
        else:
            for name, spec in DIVMOD.compile_divmod_blocks(L, dim):
                div_apply.append(len(specs))
                specs.append((name, spec))
        div_end = len(specs)
        specs.append(("alu-ax-mux", A.compile_ax_mux(L, dim, ops=mux_ops)))
    disp = base_dispatch_rules(L) + _call_dispatch_rules(L)
    if subset.memory:
        disp = disp + PF.memory_dispatch_rules(L)
    if subset.cmp:
        disp = disp + PF.cmp_dispatch_rules(L)
    if subset.bitwise:
        disp = disp + PF.bitwise_dispatch_rules(L)
    if eff_mdm:
        # ALU housekeeping: SP += 4 (popped operand) + PC += 1, gated per ALU op.
        # (The ax-mux owns the AX write; no muldiv_dispatch_rules AX write here.)
        disp = disp + _alu_housekeeping_rules(L, alu_ops)
    specs += [
        ("dispatch", compile_ffn(disp, dim)),
        ("branch-delta", compile_branch_delta(L, dim)),
        # width-aware fold modulus (256 at the 8-bit fold; a no-op 2^32 ramp under
        # C4_VM_WIDTH32, where the per-byte requant carries the wrap). The prior
        # hardcoded modulus=256 mod-clamped a genuine two's-complement negative SUB
        # result under WIDTH32; matching the deep pure-forward path (compile_fold's
        # default) keeps the 8-bit fold BYTE-IDENTICAL (modulus resolves to 256 when
        # WIDTH32 is off) while not corrupting a wide/negative value when it is on.
        ("fold", compile_fold(L.AX_VAL, L.ONE, dim)),
    ]
    # Record the apply-order for the recurrent divmod (identity elsewhere): the
    # physical layers before div_start, then the repeated iteration-body layer
    # indices, then the rest — so build() points qmodel.layers at the reused layers.
    # The log-sink divide is UNROLLED (no reused body), so it keeps identity apply.
    if eff_mdm and recurrent_divmod and not use_logsink:
        n = len(specs)
        L._qwen_apply = (list(range(div_start)) + div_apply + list(range(div_end, n)))
    return specs


def _alu_housekeeping_rules(L, ops):
    """SP += 4 (the popped operand consumed a stack slot) + PC += 1, gated on each
    efficient-ALU op.  The ax-mux writes AX; these rules only advance SP/PC."""
    from .dsl import FFNRule, LinearExpr
    sp, pc = L.SP_VAL, L.PC_VAL
    return [FFNRule([(L.OP_IS + op, 0.5, 1.5)],
                    {sp: LinearExpr.c(4.0), pc: LinearExpr.c(1.0)}) for op in ops]


# ---------------------------------------------------------------------------
# (3) WIDTH-NARROWING HOOK  (flag: C4_WIDTH_NARROW, default OFF)
# (4) CONSTANT-OPERAND FAST-PATH HOOK  (flag: C4_CONST_OPERAND, default OFF)
#
# These two are PROGRAM-SPECIFIC optimisations: they need a compile-time-known
# operand width / constant operand, which the UNIVERSAL qwen_full_vm build (the
# program lives in KV §Memory, fetched at PC; the ALU blocks are baked ONCE,
# program-independent) does not have.  So they cannot be a drop-in swap in the
# universal ``_block_specs`` divmod path the way (1)/(2) are — they belong to a
# PROGRAM-SPECIFIC depth-unrolled compile.
#
# Per the integration brief ("if too invasive to wire cleanly without a model
# build, write the hook + the flag and flag it as needing the deferred test"),
# these are wired as DETECTION + ROUTING-PLAN hooks: given a program (and, for a
# per-site const operand, its constant-propagated operands), they return the list
# of ALU sites to route to the narrowed / const-operand gadget and the builder to
# call.  A program-specific compiler consumes the plan; the universal build leaves
# the general ALU blocks in place.  Both DEFAULT OFF -> zero effect until the
# deferred GPU byte-exact run wires the plan into the unrolled compile.
# ---------------------------------------------------------------------------
def plan_narrow_ops(program, mask_nibbles: int = 8) -> Optional[dict]:
    """(3) Width-narrowing routing plan for a program (flag ``C4_WIDTH_NARROW``).

    Returns ``None`` when the flag is OFF (the default).  When ON, runs
    ``width_narrow.infer_widths`` and returns, per narrowable ALU site, the
    inferred ``(wa, wb, w_result)`` and the ``width_narrow`` builder to call so an
    op with statically-bounded operands is built at its inferred width instead of
    the full 8-nibble gadget::

        {"sites": [ {pc, op, wa, wb, w_result, builder}, ... ],
         "ax_widths": [...], "mask_nibbles": mask_nibbles}

    ``builder`` is a ``(op, wa, wb) -> callable(L, dim) -> (blocks, res_band, info)``
    selector over ``width_narrow.build_{mul,add,sub,div,mod,cmp}``.  DEFERRED: a
    program-specific unrolled compile consumes this plan; the universal build does
    not (its ALU blocks are program-independent)."""
    if not _width_narrow():
        return None
    from . import width_narrow as WN
    wi = WN.infer_widths(program, mask_nibbles=mask_nibbles)

    def _builder_for(op_name):
        return {
            "MUL": lambda L, dim, wa, wb: WN.build_mul(L, dim, wa, wb),
            "ADD": lambda L, dim, wa, wb: WN.build_add(L, dim, max(wa, wb)),
            "SUB": lambda L, dim, wa, wb: WN.build_sub(L, dim, max(wa, wb)),
            "DIV": lambda L, dim, wa, wb: WN.build_div(L, dim, wa),
            "MOD": lambda L, dim, wa, wb: WN.build_mod(L, dim, wb),
        }.get(op_name)

    sites = []
    for s in wi["alu_sites"]:
        b = _builder_for(s["op"])
        if b is None:
            continue                            # cmp/bitwise: not narrowed here
        sites.append({**s, "builder": b})
    return {"sites": sites, "ax_widths": wi["ax_widths"],
            "mask_nibbles": wi["mask_nibbles"]}


def plan_const_operand_ops(program) -> Optional[dict]:
    """(4) Constant-operand fast-path routing plan (flag ``C4_CONST_OPERAND``).

    Returns ``None`` when the flag is OFF (the default).  When ON, scans the
    bytecode for MUL/DIV/MOD sites whose second operand (AX = ``b``) is a
    compile-time constant (an ``IMM k`` immediately before the op — the simplest,
    always-safe const-propagation window) and routes them to the specialised
    gadget: const-multiplier (``const_mul.build_const_mul``) for MUL, const-divisor
    digit-recurrence (``const_divmod_digitrec.build_const_divmod_digitrec``) for
    DIV/MOD.  Returns::

        {"sites": [ {pc, op, const, builder}, ... ]}

    ``builder`` closes over the constant so the caller only supplies the layout.
    DEFERRED: a program-specific unrolled compile consumes the plan; the universal
    build leaves the general MUL/DIV/MOD blocks in place."""
    if not _const_operand():
        return None
    from . import const_mul as CM
    from . import const_divmod_digitrec as CD
    code = isa.assemble(program) if program and not hasattr(program[0], "op") else list(program)
    sites = []
    prev_imm = None
    for pc, ins in enumerate(code):
        op = ins.op
        if op == isa.IMM:
            prev_imm = int(ins.imm)
        elif op in (isa.MUL, isa.DIV, isa.MOD) and prev_imm is not None:
            k = prev_imm
            if op == isa.MUL:
                builder = (lambda L, dim, kk=k: CM.build_const_mul(L, dim, kk))
            else:                               # DIV / MOD share the const-divisor circuit
                builder = (lambda kk=k: CD.build_const_divmod_digitrec(kk))
            sites.append({"pc": pc, "op": isa.NAMES.get(op, op), "const": k,
                          "builder": builder})
            prev_imm = None
        else:
            # any op that overwrites AX (or a PSH consuming AX) breaks the IMM->op
            # const window; conservatively drop the tracked constant.
            if op not in (isa.PSH,):
                prev_imm = None
    return {"sites": sites}


def compile_opcode_decode_full(L, dim):
    """Opcode decode covering the pure-forward ops + the FUNCTION ops JSR/ENT/ADJ/
    LEV (so their OP_IS[op] one-hots materialise for dispatch)."""
    from .nibble_vm import BASE_OPS
    from .nibble_unified import compile_opcode_decode_ops
    ops = sorted(set(BASE_OPS + PF.PF_MEM_OPS +
                     [isa.JSR, isa.ENT, isa.ADJ, isa.LEV] +
                     [isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE] +
                     [isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR] +
                     [isa.MUL, isa.DIV, isa.MOD]))
    return compile_opcode_decode_ops(L, dim, ops)


# ---------------------------------------------------------------------------
# FUNCTION dispatch (register housekeeping in the FFN; the return-PC / saved-BP
# push+pop rides the driver's explicit call frame — see run_program).
# ---------------------------------------------------------------------------
def _call_dispatch_rules(L) -> List:
    from .dsl import FFNRule, LinearExpr
    ax, sp, bp, pc = L.AX_VAL, L.SP_VAL, L.BP_VAL, L.PC_VAL
    IMM = L.IMM

    def G(op):
        return [(L.OP_IS + op, 0.5, 1.5)]

    rules: List[FFNRule] = []
    # JSR: PC := imm.
    rules.append(FFNRule(G(isa.JSR), {pc: LinearExpr.of(IMM, 1.0) + LinearExpr.of(pc, -1.0)}))
    # ENT: BP := SP ; SP -= imm ; PC += 1.
    rules.append(FFNRule(G(isa.ENT), {
        bp: LinearExpr.of(sp, 1.0) + LinearExpr.of(bp, -1.0),
        sp: LinearExpr.of(IMM, -1.0), pc: LinearExpr.c(1.0)}))
    # ADJ: SP += imm ; PC += 1.
    rules.append(FFNRule(G(isa.ADJ), {sp: LinearExpr.of(IMM, 1.0), pc: LinearExpr.c(1.0)}))
    # LEV: SP := BP (unwind). BP/PC are restored from the popped call frame by the
    # driver's re-embed; here we only unwind SP.
    rules.append(FFNRule(G(isa.LEV), {sp: LinearExpr.of(bp, 1.0) + LinearExpr.of(sp, -1.0)}))
    return rules


# ===========================================================================
# FFN bake: pure-forward SwiGLU spec -> Qwen2 MLP (RMSNorm-K compensated).
# ===========================================================================
def _bake_ffn(mlp, spec, L, comp):
    hidden = spec["W_up"].shape[0]
    inter = mlp.gate_proj.weight.shape[0]
    H = mlp.gate_proj.weight.shape[1]
    assert hidden <= inter, (hidden, inter)
    bd = spec.get("b_down")
    if bd is not None and float(bd.abs().max()) != 0.0:
        raise AssertionError("nonzero b_down not supported")
    # Match the MLP weight dtype (always fp32 now — the model is genuinely fp32).
    dt = mlp.gate_proj.weight.dtype
    gate_w = torch.zeros(inter, H, dtype=dt); up_w = torch.zeros(inter, H, dtype=dt)
    down_w = torch.zeros(H, inter, dtype=dt)
    Dg = spec["W_up"].shape[1]
    gate_w[:hidden, :Dg] = spec["W_up"].to(dt)
    up_w[:hidden, :Dg] = spec["W_gate"].to(dt)
    down_w[:Dg, :hidden] = spec["W_down"].to(dt)
    gate_w[:hidden, L.ONE] += spec["b_up"].to(dt)
    up_w[:hidden, L.ONE] += spec["b_gate"].to(dt)
    mlp.gate_proj.weight.copy_(gate_w)
    mlp.up_proj.weight.copy_(up_w)
    mlp.down_proj.weight.copy_(down_w)


# ===========================================================================
# Register value-copy CAM (frame ingest): one head per register.
#
# Each register rides on ONE frame token carrying all 16 nibbles in TOK_NIB and a
# ROLE one-hot. Head h (register h):
#   * CONTENT — slow RoPE lane keyed on ROLE[h] (position-invariant match).
#   * RECENCY — fast RoPE lane keyed on IS_TOK for every frame token (latest wins).
#   * SINK    — BOS row is content-free (ROLE/IS_TOK all 0) -> logit 0 sink (ZFOD).
#   * VALUE   — copy TOK_NIB[0..15] into register h's nibble band (RoPE-free).
# All heads share KV-group 0; each query head lights only ITS role.
# ===========================================================================
# CONTENT must DOMINATE recency: within a frame each register appears once, so the
# role match alone must select the token; recency only breaks ties among the SAME
# role ACROSS frames (loop re-emissions), where content is equal. So content^2 >>
# recency's max RoPE-boosted contribution.
# CONTENT selects the register (role match, position-invariant on the slow lane).
# The windowed stream keeps only the LATEST register frame, so there is exactly ONE
# token per role and content alone selects it — no cross-frame recency needed for
# the register CAM (the window IS the recency). A small RECENCY_GAIN keeps a mild
# nearest-token preference (harmless; the frame has one token per role).
CONTENT_GAIN = 26.0
RECENCY_GAIN = 3.0


def _bake_register_cam(attn, QL, arch, comp, K):
    L = QL.L
    hd = arch.head_dim
    slow_lo, _ = _rope_lane_pair(hd, slow=True)
    fast_lo, _ = _rope_lane_pair(hd, slow=False)
    q_w = attn.q_proj.weight; k_w = attn.k_proj.weight
    v_w = attn.v_proj.weight; o_w = attn.o_proj.weight
    reg_bases = {"PC": L.PC, "AX": L.AX, "SP": L.SP, "BP": L.BP, "STACK0": L.STACK0}

    # KV-group 0 keys (shared): content on slow lane (one per register role), recency
    # on fast lane (ONE-of-IS_TOK). value copies TOK_NIB[j] on value lanes j.
    # query-exclusion penalty lane: valid keys carry IS_TOK=1 (register tokens); the
    # STEP_END query row and BOS carry IS_TOK=0 and must score hugely negative as
    # keys so a head never attends to a non-register row (the query self-match /
    # sink). ZFOD is unneeded here (every register always has a token). pen_lane is a
    # NON-rotary lane (mid-band, RoPE ~ identity) shared by all heads.
    PEN = 20.0
    pen_lane = slow_lo - len(CAM_REGS)      # a spare slow lane below the content lanes
    for h in range(len(CAM_REGS)):
        base = h * hd
        q_w[base + pen_lane, L.ONE] = PEN
    k_w[pen_lane, L.ONE] = -PEN
    k_w[pen_lane, QL.IS_TOK] = PEN          # IS_TOK=1 -> key 0 ; IS_TOK=0 -> -PEN

    for h, reg in enumerate(CAM_REGS):
        base = h * hd
        # content: query head h keys ROLE[h] on a distinct slow lane; kv key matches.
        c_lane = slow_lo - h            # distinct near-identity slow lane per register
        q_w[base + c_lane, QL.ROLE + h] = CONTENT_GAIN
        k_w[c_lane, QL.ROLE + h] = CONTENT_GAIN
        # recency: fast lane, query on ONE, key on IS_TOK (every frame token).
        q_w[base + fast_lo, L.ONE] = RECENCY_GAIN
    k_w[fast_lo, QL.IS_TOK] = RECENCY_GAIN
    # value copy: kv-group-0 value lanes 0..15 carry TOK_NIB; o_proj writes head h's
    # value lanes into register h's nibble band.
    for j in range(NIB_PER_REG):
        v_w[j, QL.TOK_NIB + j] = 1.0
    for h, reg in enumerate(CAM_REGS):
        base = h * hd
        reg_base = reg_bases[reg]
        for j in range(NIB_PER_REG):
            o_w[reg_base + j, base + j] = 1.0


# ===========================================================================
# Memory / call CAM: RoPE address-match + BOS sink over the store frames.
#
# Ported from nibble_pure_forward._bake_pf_memory_head but onto the Qwen attn +
# RoPE. Store rows key ADDR_BIN (32 bits) on slow lanes; the load query keys
# QRY_BIN; recency picks the latest store to the same address (latest-write-wins).
# The loaded value nibbles (VAL_NIB) are copied into the AX nibble band.
# ===========================================================================
def _bake_memory_cam(attn, QL, arch, comp, K, mem_addr_bits=None):
    """Address-matching KV read that is SINK-CLEAN under PLAIN Qwen softmax.

    Design constraint (differs from the softmax1 pure-forward): the BOS sink is a
    REAL token at logit 0, so a store row must score STRICTLY POSITIVE only when it
    is a LOAD to the store's exact address, and score <= 0 (below the sink) on every
    other step / address. We therefore build the score so it has NO positive
    baseline:

      per address bit b, on a store row (ADDR_BIN[b], IS_STORE=1) vs a load query
      (QRY_BIN[b], IS_LOAD=1):  agreement_b = 1 iff QRY_BIN[b]==ADDR_BIN[b], else 0.
      We realise ``sum_b agreement_b`` and subtract ``n_bits`` so the score is
      ``-hamming(addr,qry)`` (<=0, ==0 iff exact match). Then a load-enable term
      adds ``+MATCH`` only when BOTH IS_LOAD and IS_STORE hold, lifting an exact
      match to ``+MATCH`` (> sink 0) and leaving non-load / non-store rows at <=0.
    The BOS sink (no IS_STORE, no address) sits at exactly 0 -> ZFOD (unwritten
    address reads 0). RoPE recency on the fast lane breaks ties among repeated
    stores to the same address (latest-write-wins).
    """
    from .blogspec_memory import ADDR_BITS
    L = QL.L
    hd = arch.head_dim
    q_w = attn.q_proj.weight; k_w = attn.k_proj.weight
    v_w = attn.v_proj.weight; o_w = attn.o_proj.weight
    half = hd // 2
    # The corpus uses 8-bit data addresses, so 8 address bits suffice (ADDR_BIN's
    # upper bits are 0 for both key and query and never contribute). Placing all
    # match + control lanes on the SLOWEST rotary pairs (near-identity RoPE) keeps
    # the address dot position-invariant and the control gates un-rotated — the
    # fast rotary lanes (near lane 0) rotate too much to carry a stable content dot.
    # WIDE-MEM-CAM (WALL #1, doom 32-bit): ``mem_addr_bits`` KEYS more of the 32
    # physically-carried ADDR_BIN/QRY_BIN bits onto the next-slowest lanes (below the
    # 8-bit set), pushing the control lanes down accordingly.  DEFAULT 8 -> the lanes
    # + weights below are LITERALLY unchanged (byte-identical golden build); doom
    # passes 18.  Clamp to the physical band width and leave >=1 lane above rec_lane 3
    # for the 3 control lanes.  ``rec_lane=3`` (fixed) is NEVER an address lane, so the
    # widened set must not reach it: cap so the lowest control lane (excl) stays > 3.
    req = _mem_addr_bits() if mem_addr_bits is None else int(mem_addr_bits)
    # Lane map: addr lanes = half-1 .. half-n_bits ; bias=half-1-n_bits,
    # gate=half-2-n_bits, excl=half-3-n_bits ; rec_lane=3 (fixed).  Keep excl>3 (no
    # collision with the recency lane): half-3-n_bits > 3  =>  n_bits <= half-7.
    n_bits = min(req, ADDR_BITS, half - 7)  # half-7 = 25 for head_dim 64 (excl_lane >= 4)
    # Qwen attention divides the QK dot by sqrt(head_dim)=8, so the raw gains must be
    # large enough that the POST-scale exact-match margin dominates the sink. The
    # exact-match net score is (n_bits*G^2 - (n_bits-0.5)*G^2)/sqrt(hd) = 0.5*G^2/8;
    # G=16 -> net ~16 logits (softmax weight ~1.0), and a 1-bit mismatch is
    # ~-2*G^2/8 = -64 (far below the sink) -> exact-match-only, clean ZFOD.
    G = 16.0                                 # per-bit agreement gain
    # agreement_b = qb*ab + (1-qb)*(1-ab) = 1 - qb - ab + 2*qb*ab. The cross term
    # qb*ab is bilinear (RoPE dot of a query lane and a key lane). We realise the
    # score directly as the dot of vectors that make matching bits ADD and
    # mismatching bits CANCEL: put on lane b  q = G*(2*QRY_BIN[b]-IS_LOAD),
    # k = G*(2*ADDR_BIN[b]-IS_STORE). On a load (IS_LOAD=1) vs a store (IS_STORE=1):
    #   q_b*k_b = G^2*(2qb-1)(2ab-1) = +G^2 iff bits agree, -G^2 iff disagree.
    # On a NON-load query (IS_LOAD=0): q_b = G*2*QRY_BIN[b] = 0 (QRY_BIN=0) -> 0
    # contribution. On the BOS sink (IS_STORE=0, ADDR=0): k_b = 0 -> 0 contribution.
    for b in range(n_bits):
        lane = half - 1 - b
        q_w[lane, L.QRY_BIN + b] = 2.0 * G
        q_w[lane, L.IS_LOAD] = -G
        k_w[lane, L.ADDR_BIN + b] = 2.0 * G
        k_w[lane, L.IS_STORE] = -G
    # Now a store row on a load scores sum_b (+/-G^2) = G^2*(n_bits - 2*hamming).
    # An EXACT match scores +G^2*n_bits; every 1-bit mismatch costs 2*G^2. We must
    # push a NON-exact match below the sink (0) and keep the exact match above it.
    # Subtract (n_bits-1)*G^2 via a load*store bias lane so exact = +G^2, 1-mismatch
    # = -G^2 (< sink), and NON-load / NON-store rows stay 0 (the bias needs both).
    bias_lane = half - 1 - n_bits
    B = math.sqrt((n_bits - 0.5)) * G        # sqrt so B^2 = (n_bits-0.5)*G^2
    q_w[bias_lane, L.IS_LOAD] = -B
    k_w[bias_lane, L.IS_STORE] = B           # load*store -> -B^2 = -(n_bits-0.5)G^2
    # LOAD-ENABLE gate: on a NON-load step every store row must sit FAR below the
    # sink (0) so the CAM outputs exactly 0 (no leak into AX). Query keys +P on ONE
    # always, -P on IS_LOAD (so query=0 on a load, +P on a non-load); store rows key
    # -P on IS_STORE. Product on a non-load store = -P^2 (huge negative); on a load,
    # query is 0 -> no penalty. The sink (IS_STORE=0) is untouched -> stays 0.
    gate_lane = half - 1 - n_bits - 1
    P = 60.0                                 # gate penalty (P^2/8 ~ 450 >> match)
    q_w[gate_lane, L.ONE] = P
    q_w[gate_lane, L.IS_LOAD] = -P
    k_w[gate_lane, L.IS_STORE] = -P
    # CODE-FRAME EXCLUSION (code-from-memory): the program lives in the SAME token
    # stream as the store log (as ~N extra CODE frames, IS_CODE=1).  A code frame has
    # IS_STORE=0 so it scores at the SINK (0) for this CAM — but with MANY code frames
    # their softmax weight SUMS (each e^0=1), stealing a few % off an exact store
    # match (whose net score ~8 logits is only ~e^8 vs ~N·e^0) and diluting the loaded
    # value toward 0 (VAL_NIB=0 on code frames) -> an off-by-one load.  Push every
    # IS_CODE row FAR below the sink (query +Pc·ONE, key -Pc·IS_CODE -> -Pc^2 on a
    # code frame, 0 on store/BOS) so the code frames are INVISIBLE to the memory CAM.
    # Present only when the layout carries code frames (byte-identical otherwise).
    if getattr(L, "IS_CODE", None) is not None:
        excl_lane = half - 1 - n_bits - 2
        Pc = 60.0
        q_w[excl_lane, L.ONE] = Pc
        k_w[excl_lane, L.IS_CODE] = -Pc
    # recency on a MEDIUM rotary lane (lane 3: monotone decay over ~11 positions).
    # Two stores to the SAME address tie on the address match, so recency ALONE must
    # pick the latest (§Memory latest-write-wins). It only acts among stores on a
    # load (query IS_LOAD). MEM_RECENCY must be strong enough to fully resolve a
    # same-address re-store tie, yet WEAK enough not to override an exact-address
    # match for an OLDER, distinct address (the exact-match margin post-scale is
    # ~0.5*G^2/8 = 16 logits; MEM_RECENCY=14 gives up to ~14^2/8*rope ~ 24 logits,
    # which OUTWEIGHS the exact match and made a load of the OLDEST of several
    # distinct-address stores return garbage — the ELIZA multi-address failure: a
    # keyword-table byte read after later buffer stores). MEM_RECENCY=8.0 is the
    # measured sweet spot: it still snaps a same-address re-store to the latest
    # (softmax ~1.0) but stays comfortably below the exact-match margin, so loading
    # any older distinct address is exact. Verified across the corpus + ELIZA.
    rec_lane = 3
    MEM_RECENCY = 8.0
    q_w[rec_lane, L.IS_LOAD] = MEM_RECENCY
    k_w[rec_lane, L.IS_STORE] = MEM_RECENCY
    # value: copy VAL_NIB into the AX nibble band.
    for j in range(NIB_PER_REG):
        v_w[j, L.VAL_NIB + j] = 1.0
        o_w[L.AX + j, j] = 1.0


# ===========================================================================
# CODE §Memory CAM (fetch@PC): the "Universal = bytecode fetched by PC" mechanism.
#
# The SAME address-keyed CAM as _bake_memory_cam, applied to the CODE frames (one
# per instruction, keyed on its address i) instead of the STORE frames.  A fetch
# query keyed on PC selects the code frame whose address == PC and copies its op ->
# OP_VAL and imm -> IMM (the scalars the opcode-decode + dispatch read).  This is
# the address-keyed §Memory read (LI/SI machinery) reused to fetch the instruction
# at PC — no baked CODE_OP[k] table, so the fetch is program-length-independent.
# ===========================================================================
def _bake_code_cam(attn, QL, arch, comp, K):
    """Bake the code-fetch CAM on ``attn`` head 0 (fetch mem_code[PC]).

    Structurally identical to ``_bake_memory_cam`` (per-bit agreement on slow RoPE
    lanes + a bias lane so a NON-exact match sinks below the BOS 0, + a fetch-enable
    gate), but keyed on the CODE frames: KEY = ``CODE_KEY_BIN`` (instruction address
    i) gated by ``IS_CODE``; QUERY = ``CODE_QRY_BIN`` (current PC) gated by
    ``IS_FETCH``; VALUE copies the frame's ``CODE_OPV`` -> ``OP_VAL`` and
    ``CODE_IMMV`` -> ``IMM``.  Every code address i appears at MOST once, so no
    recency tiebreak is needed (unlike the store log's latest-write-wins)."""
    L = QL.L
    hd = arch.head_dim
    q_w = attn.q_proj.weight; k_w = attn.k_proj.weight
    v_w = attn.v_proj.weight; o_w = attn.o_proj.weight
    half = hd // 2
    # CODE_ADDR_BITS (DEFAULT 12, ``C4_CODE_ADDR_BITS``) address bits; keep them on the
    # slowest rotary pairs (near-identity RoPE) so the address dot is position-invariant,
    # exactly as the memory CAM does.  half=32 rotary pairs hold up to 29 bits + 3
    # control lanes; the RoPE fidelity ceiling (not the lane count) is the real cap on
    # how far a distant frame stays keyable (see the CODE_ADDR_BITS module comment).
    n_bits = min(CODE_ADDR_BITS, half - 3)
    G = 16.0                                  # per-bit agreement gain (same as mem CAM)
    # q = G*(2*QRY_BIN[b]-IS_FETCH), k = G*(2*KEY_BIN[b]-IS_CODE): on a fetch
    # (IS_FETCH=1) vs a code frame (IS_CODE=1), q_b*k_b = +G^2 iff bits agree, -G^2
    # iff disagree.  A non-fetch query (IS_FETCH=0) -> q_b=0; the BOS sink (IS_CODE=0,
    # KEY=0) -> k_b=0.
    for b in range(n_bits):
        lane = half - 1 - b
        q_w[lane, L.CODE_QRY_BIN + b] = 2.0 * G
        q_w[lane, L.IS_FETCH] = -G
        k_w[lane, L.CODE_KEY_BIN + b] = 2.0 * G
        k_w[lane, L.IS_CODE] = -G
    # bias lane: subtract (n_bits-0.5)*G^2 on a fetch*code pair so an EXACT match =
    # +G^2 (> sink 0) and every 1-bit mismatch = -G^2 (< sink).  Needs BOTH flags.
    bias_lane = half - 1 - n_bits
    B = math.sqrt((n_bits - 0.5)) * G
    q_w[bias_lane, L.IS_FETCH] = -B
    k_w[bias_lane, L.IS_CODE] = B
    # FETCH-ENABLE gate: on a NON-fetch step every code frame must sit FAR below the
    # sink so the CAM outputs 0 (no leak).  Query +P on ONE, -P on IS_FETCH (=0 on a
    # fetch, +P otherwise); code rows key -P on IS_CODE.  (In practice every step IS
    # a fetch, but this keeps the head sink-clean if a non-fetch query ever runs.)
    gate_lane = half - 1 - n_bits - 1
    P = 60.0
    q_w[gate_lane, L.ONE] = P
    q_w[gate_lane, L.IS_FETCH] = -P
    k_w[gate_lane, L.IS_CODE] = -P
    # STORE-FRAME EXCLUSION (symmetric to the mem CAM's code exclusion): store frames
    # (IS_STORE=1) share the token stream and score at the SINK (0) for this CAM; with
    # many stores their summed softmax weight would steal a few % off the exact fetch
    # match (net ~16 logits) and dilute the delivered OP_VAL/IMM toward 0.  Push every
    # IS_STORE row FAR below the sink so store frames are INVISIBLE to the code CAM.
    if getattr(L, "IS_STORE", None) is not None:
        excl_lane = half - 1 - n_bits - 2
        q_w[excl_lane, L.ONE] = P
        k_w[excl_lane, L.IS_STORE] = -P
    # value: copy the selected code frame's op/imm scalars into OP_VAL / IMM.  The
    # softmax weight on the exact-PC-match frame is ~1.0, so OP_VAL = CODE_OPV and
    # IMM = CODE_IMMV of the fetched instruction (the bilinear code-select, but from
    # §Memory instead of the baked table).
    v_w[0, L.CODE_OPV] = 1.0
    o_w[L.OP_VAL, 0] = 1.0
    v_w[1, L.CODE_IMMV] = 1.0
    o_w[L.IMM, 1] = 1.0


# ===========================================================================
# Embedding table (token -> residual). Byte tokens embed their two nibbles; the
# ROLE / TOK_NIB / IS_TOK / memory bands are written by the driver overlay.
# ===========================================================================
def _build_embedding(L, hidden_size, comp, K):
    E = torch.zeros(V.VOCAB, hidden_size)
    E[:, L.ONE] = 1.0
    for b in range(256):
        lo, hi = V.nibbles_of_byte(b)
        E[b, L.CUR_NIB + 0] = float(lo)
        E[b, L.CUR_NIB + 1] = float(hi)
    E[:, comp] = K                          # compensator on every live token
    # BOS sink: content-free (no ROLE/IS_TOK/CUR_NIB), compensator only.
    E[V.BOS, :] = 0.0
    E[V.BOS, comp] = K
    return E


# ===========================================================================
# THE DRIVER — one program step = one Qwen2Model.forward; argmax + append only.
#
# The token stream each step is a WINDOW:
#   [BOS sink] + [persistent STORE frames (memory KV log)] + [the LATEST register
#   frame] + [STEP_END query].
# A REGISTER frame is 5 tokens (PC/AX/SP/BP/STACK0): the overlay writes each
# register's ROLE one-hot + its 16 nibbles into TOK_NIB + IS_TOK=1. Only the LATEST
# register frame is in the window, so the register CAM's content match selects each
# register with ZERO cross-frame recency ambiguity (state fully round-trips through
# the token stream every step — the window is the spec's KV memory, pruned to what
# a step reads). A STORE frame is 1 MEM token (ADDR_BIN/VAL_NIB/IS_STORE) that
# PERSISTS (the §Memory write log the memory CAM content-addresses).
# The program lives in the DATA bands (CODE_OP/CODE_IMM) at every position (fetch@PC
# works at the last position). The CAM reconstructs the register state; the FFN
# blocks compute the next step; the value lanes at the last position are the next
# registers, decoded by the LM value-argmax (the vanilla re-quantiser).
# ===========================================================================
_REG_TOKEN = {"PC": V.REG_PC, "AX": V.REG_AX, "SP": V.REG_SP, "BP": V.REG_BP,
              "STACK0": V.MEM}     # STACK0 rides a spare marker id (overlay-tagged)


def _snap(x: float) -> int:
    """LM-head value-argmax requant (argmax_v 2·v·x − v²), no round. Vectorised.

    This is a RUNTIME DECODE helper (Python-side, not a model parameter): the
    genuinely-fp32 model produces the lane ``x``, and this argmax reads it back to an
    integer register value.  The value vocab reaches ``VALVOCAB ≈ 0x10100`` (SP/BP =
    0x10000), where ``v²`` ≈ 4.3e9 exceeds fp32's 2^24 integer precision, so the
    argmax computation stays fp64 to keep the SP/BP snap byte-exact.  The MODEL is
    still fully fp32 (zero fp64 params); only this local Python argmax uses doubles."""
    from .nibble_vm import VALVOCAB
    v = torch.arange(VALVOCAB, dtype=torch.float64)
    logits = 2.0 * v * float(x) - v * v
    return int(logits.argmax().item())


def _address_bits(addr: int, n: int) -> List[float]:
    return [float((addr >> b) & 1) for b in range(n)]


def _signed_imm(imm: int) -> int:
    """Interpret a 32-bit-wrapped immediate as its signed image (so a small negative
    like LEA -8 stays -8, not 0xFFFFFFF8) to keep |imm| << the RMSNorm compensator."""
    imm &= 0xFFFFFFFF
    return imm - (1 << 32) if imm >= (1 << 31) else imm


def _overlay_code_frames(x, L, code: List[isa.Instr], base_pos: int, row: int = 0):
    """Write the program into the KV §Memory as CODE frames (code-from-memory).

    One frame per instruction i at ``x[row, base_pos + i]``: KEY = ``CODE_KEY_BIN`` =
    bits(i), VALUE = ``CODE_OPV`` (op) + ``CODE_IMMV`` (signed imm), gated by
    ``IS_CODE=1``.  These PERSIST (the address-keyed code memory the fetch CAM reads
    at PC), exactly like the store log persists for LI/SI.  Immediates are the SIGNED
    image so |imm| stays O(hundreds) << the RMSNorm compensator (as CODE_IMM was)."""
    for i, ins in enumerate(code):
        p = base_pos + i
        x[row, p, L.IS_CODE] = 1.0
        for b, bit in enumerate(_address_bits(i, CODE_ADDR_BITS)):
            x[row, p, L.CODE_KEY_BIN + b] = bit
        x[row, p, L.CODE_OPV] = float(ins.op)
        x[row, p, L.CODE_IMMV] = float(_signed_imm(ins.imm))


def _overlay_fetch_query(x, L, pc: int, row: int = 0, col: int = -1):
    """Write the fetch@PC query (code-from-memory) at ``x[row, col]``: ``IS_FETCH=1``
    + ``CODE_QRY_BIN`` = bits(PC).  The code CAM selects the frame whose address ==
    PC and delivers its op/imm into OP_VAL/IMM."""
    x[row, col, L.IS_FETCH] = 1.0
    for b, bit in enumerate(_address_bits(pc & ((1 << CODE_ADDR_BITS) - 1), CODE_ADDR_BITS)):
        x[row, col, L.CODE_QRY_BIN + b] = bit


def _seed_logsink_rows(x, L, base_pos: int, row: int = 0):
    """SEED the reciprocal sink head's reserved KV rows every step (position-indep):
      rows base_pos..base_pos+7 : the 8 log-key rows (LOGKEY[j]=1, IS_RECIP_ROW=1)
      row  base_pos+8           : the sink row (IS_SINK=1, IS_RECIP_ROW=1, value 1)
    Seeding EVERY step (not just at t=0) is what makes a LEADING DIV work — the head
    finds its log-keys regardless of how many program tokens precede the query."""
    a = L.LOGSINK
    for j in range(8):
        p = base_pos + j
        x[row, p, a.LOGKEY + j] = 1.0
        x[row, p, a.IS_RECIP_ROW] = 1.0
    sink = base_pos + 8
    x[row, sink, a.IS_SINK] = 1.0
    x[row, sink, a.IS_RECIP_ROW] = 1.0


def _build_stream_and_overlay(vm: QwenFullVM, code: List[isa.Instr],
                              reg_state: dict, store_log: List[dict],
                              load_addr: Optional[int]) -> torch.Tensor:
    """Build the windowed token stream + its overlay for ONE forward. Returns the
    embedded+overlaid residual [1,S,H]. Window:
      pos 0                : BOS sink
      1 .. len(store_log)  : one MEM token per persistent store (KV memory log)
      [code frames]        : one per instruction (code-from-memory), else in-data table
      next 5               : the LATEST register frame (PC/AX/SP/BP/STACK0 tokens)
      last                 : STEP_END query row (register-role query + load query)."""
    QL, L = vm.QL, vm.QL.L
    subset = vm.subset
    cfm = vm.code_from_memory
    from .blogspec_memory import ADDR_BITS

    n_store = len(store_log) if subset.memory else 0
    n_code = len(code) if cfm else 0
    dls = getattr(vm, "div_logsink", False)
    # LOG-SINK reciprocal: 8 pre-seeded reserved-KV log-key rows + 1 sink row, placed
    # right after BOS.  They are SEEDED EVERY STEP at fixed content (position-indep on
    # the slow rotary lanes), so a program whose FIRST executed instruction is DIV
    # finds them — the "leading-DIV" guarantee.
    n_logsink = 9 if dls else 0
    stream: List[int] = [V.BOS]
    stream += [V.MEM] * n_logsink                     # log-key rows + sink (recip head)
    stream += [V.MEM] * n_store                       # store KV tokens
    stream += [V.MEM] * n_code                        # code KV tokens (fetch@PC)
    stream += [_REG_TOKEN[r] for r in CAM_REGS]       # register frame
    stream += [V.STEP_END]                            # query row
    x = vm.embed[torch.tensor([stream])].clone()
    Sn = x.shape[1]

    for i in range(Sn):
        x[0, i, L.ONE] = 1.0
    if dls:
        _seed_logsink_rows(x, L, base_pos=1)
    if cfm:
        # CODE-FROM-MEMORY: the program lives in the KV §Memory as code frames (one
        # per instruction), fetched by PC via the address CAM — no baked CODE_OP[k]
        # table, so no code_size limit and no program-length width dependence.
        _overlay_code_frames(x, L, code, base_pos=1 + n_logsink + n_store)
    else:
        # program-in-data at every position (fetch@PC universal). Immediates are stored
        # as SIGNED small values (an assembler stores imm & 0xFFFFFFFF, so a small
        # negative like LEA -8 would be 0xFFFFFFF8 = 4.3e9 — that would blow up the
        # RMSNorm compensator, which must dominate every residual value; store the
        # signed image so |imm| stays O(hundreds), well under K).
        for i in range(Sn):
            for k, ins in enumerate(code):
                x[0, i, L.CODE_OP[k]] = float(ins.op)
                x[0, i, L.CODE_IMM[k]] = float(_signed_imm(ins.imm))

    # persistent store frames (memory KV log).
    if subset.memory:
        for si, st in enumerate(store_log):
            p = 1 + n_logsink + si
            x[0, p, L.IS_STORE] = 1.0
            for b, bit in enumerate(_address_bits(st["addr"], ADDR_BITS)):
                x[0, p, L.ADDR_BIN + b] = bit
            for j, nv in enumerate(V.nibbles_of_value(st["val"], NIB_PER_REG)):
                x[0, p, L.VAL_NIB + j] = float(nv)

    # the latest register frame: one token per register carrying its nibbles+role.
    reg0 = 1 + n_logsink + n_store + n_code
    for h, reg in enumerate(CAM_REGS):
        p = reg0 + h
        for j, nv in enumerate(V.nibbles_of_value(reg_state[reg], NIB_PER_REG)):
            x[0, p, QL.TOK_NIB + j] = float(nv)
        x[0, p, QL.ROLE + h] = 1.0
        x[0, p, QL.IS_TOK] = 1.0

    # QUERY row (last position): every register role one-hot (the register CAM query)
    # + the load address query (memory CAM) on a LOAD step + the fetch@PC query.
    for h in range(len(CAM_REGS)):
        x[0, -1, QL.ROLE + h] = 1.0
    if cfm:
        _overlay_fetch_query(x, L, reg_state["PC"])
    if subset.memory and load_addr is not None:
        x[0, -1, L.IS_LOAD] = 1.0
        for b, bit in enumerate(_address_bits(load_addr, ADDR_BITS)):
            x[0, -1, L.QRY_BIN + b] = float(bit)
    return x


def _forward(vm: QwenFullVM, x: torch.Tensor) -> torch.Tensor:
    out = vm.qmodel(inputs_embeds=x, use_cache=False)
    return out.last_hidden_state[0, -1]


def run_program(vm: QwenFullVM, code: List[isa.Instr], max_steps: int = 64,
                verbose: bool = False, mask: int = 0xFF,
                spill_stack_to_kv: bool = False) -> Dict[str, object]:
    """Execute ``code`` on the fused Qwen VM. One VM step = one Qwen2Model.forward
    over the windowed token stream (state read from the latest frame by the register
    CAM; the op computed by the SwiGLU MLPs; control-flow — PC update / branch /
    call — all inside the forward). Returns
    ``{"ax_trace": [...], "ref_trace": [...], "exact": bool, "steps": int}``.

    ``mask`` (default ``0xFF``) is the value width the emitted AX is decoded at for
    the efficient-ALU ops (MUL/DIV/MOD).  The default 8-bit matches ``isa.interpret``
    (the 8-bit reference used across c4_min and the 8-bit operand-load corpus path),
    so ``ax_trace == ref_trace`` is byte-exact.  The EFFICIENT ALU computes the FULL
    32-bit result; pass ``mask=0xFFFFFFFF`` to keep the 32-bit result in the trace and
    compare it against a 32-bit reference (``nibble_muldivmod`` mul32/divmod32/mod32).
    NB: the 32-bit result lives in the AX nibble band only at the ALU-op step — a
    trailing non-ALU step (e.g. HALT) reads the scalar ``AX_VAL``, folded mod 256.

    Functions: JSR/ENT push (return-PC / saved-BP), LEV pops them. The push+pop
    rides an explicit CALL STACK (state that lives alongside the token stream); all
    the SP/BP/PC arithmetic is done inside the Qwen forward."""
    QL, L = vm.QL, vm.QL.L
    subset = vm.subset
    # Run the reference oracle to the SAME step budget the model driver uses. The
    # default isa.interpret cap (max_steps=256) TRUNCATES the reference for a loop
    # longer than 256 VM steps — e.g. a countdown from n takes 4n+2 steps, so any
    # n >= 64 (256 steps) truncated the golden while the driver ran the loop to
    # completion, making the correct model trace mismatch a short reference (#691
    # BUG 2: "countdown >= 100 diverges"). The model arithmetic is exact across the
    # nibble-carry boundaries (100/200); the divergence was purely the capped oracle.
    ref_trace = isa.interpret(code, max_steps=max_steps)

    reg_state = {"PC": 0, "AX": 0, "SP": SP_INIT, "BP": SP_INIT, "STACK0": 0}
    store_log: List[dict] = []                       # persistent memory KV entries
    ax_trace: List[int] = []
    cur_pc = 0
    call_stack: List[Tuple[Optional[int], Optional[int]]] = []
    # KV-MEMORY-BACKED operand stack (#692/#702 depth-1 wall fix, opt-in). The
    # register CAM tracks exactly ONE live top-of-stack cell (STACK0) and a bare
    # PSH does NOT spill to the KV log, so a 2nd push loses the 1st cell -> a
    # width>=2 dot (needs depth 2 to park a partial) diverges. When ON, every PSH
    # MIRRORS its value into the persistent KV memory log at the SP-relative stack
    # address (the SAME log SI/SC write and the mem CAM content-addresses), and the
    # next STACK0 is reconstructed from that log after each pop -> arbitrary depth,
    # byte-exact. OFF (default) == the historical byte-identical 1-slot path.
    stack_kv: List[Tuple[int, int]] = []
    _POP_OPS = {isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD, isa.AND, isa.OR,
                isa.XOR, isa.SHL, isa.SHR, isa.EQ, isa.NE, isa.LT, isa.GT,
                isa.LE, isa.GE}
    # RUNTIME code list — the in-transformer JIT's writable code §Memory (a private
    # mutable copy EMIT rewrites; ``_build_stream_and_overlay`` overlays THIS as the
    # code frames the fetch@PC CAM reads).  Flag-off it equals ``code`` (byte-exact).
    emit_on = _cfm_emit_enabled() and vm.code_from_memory
    run_code = list(code)

    for _ in range(max_steps):
        op = run_code[cur_pc].op if 0 <= cur_pc < len(run_code) else None
        prev = dict(reg_state)
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = prev["AX"] & 0xFF            # 8-bit load address (corpus slice)

        x = _build_stream_and_overlay(vm, run_code, reg_state, store_log, load_addr)
        state = _forward(vm, x)

        # EMIT (in-transformer JIT, C4_CFM_EMIT): a driver-serviced control op (like
        # JSR/ENT/LEV — EMIT has no FFN dispatch rule).  The model FETCHED this EMIT
        # from the code CAM (proving fetch@PC serves runtime-appended frames) + the
        # register CAM reconstructed the input state; the driver reads the produced
        # word from the input registers (op<-AX, imm<-STACK0), writes/overwrites the
        # code frame at the EMIT immediate, and bumps PC.  Byte-exact vs isa.interpret.
        if emit_on and op == isa.EMIT:
            produced_op = prev["AX"] & 0xFF
            produced_imm = prev["STACK0"] & 0xFF
            target = run_code[cur_pc].imm
            while target >= len(run_code):
                run_code.append(isa.Instr(isa.NOP, 0))
            run_code[target] = isa.Instr(produced_op, produced_imm)
            # EMIT pops the immediate it consumed off the KV-backed stack.
            new_stk = 0
            if spill_stack_to_kv and stack_kv:
                _psaddr, _ = stack_kv.pop()
                store_log = [s for s in store_log
                             if (s["addr"] & 0xFF) != (_psaddr & 0xFF)]
                new_stk = stack_kv[-1][1] if stack_kv else 0
            pc = cur_pc + 1
            reg_state = {"PC": pc, "AX": prev["AX"], "SP": (prev["SP"] + 1) & 0xFFFFFFFF,
                         "BP": prev["BP"], "STACK0": new_stk}
            ax_trace.append(prev["AX"])
            if verbose:
                print(f"  step pc={cur_pc} op=EMIT  -> code[{target}]="
                      f"({isa.NAMES.get(produced_op, produced_op)} {produced_imm}) pc={pc}")
            cur_pc = pc
            if cur_pc < 0 or cur_pc >= len(run_code):
                break
            continue

        pc = _snap(state[L.PC_VAL])
        # AX decode: for the efficient-ALU ops (MUL/DIV/MOD) the ax-mux writes the
        # FULL 32-bit result into the AX NIBBLE band, so decode it per-byte with the
        # LM byte-head argmax (residue-immune, 32-bit-exact); every other op's AX is
        # the scalar AX_VAL byte (the 8-bit fold path / loaded byte / cmp result).
        # ``mask`` narrows the emitted AX to the compare width (8-bit vs isa.interpret,
        # 32-bit vs the nibble_muldivmod reference).
        # The nibble-decoded AX ops: MUL/DIV/MOD always, and SHL/SHR when they are
        # routed through the native MUL/DIV gadgets (shift-via-mul writes the result to
        # the AX NIBBLE band via the ax-mux, exactly like MUL/DIV/MOD).
        _nib_ax_ops = {isa.MUL, isa.DIV, isa.MOD}
        if vm.shift_via_mul:
            _nib_ax_ops |= {isa.SHL, isa.SHR}
        if vm.efficient_alu and op in _nib_ax_ops:
            ax = _decode_reg_from_nibbles(state, L, L.AX) & mask
        else:
            ax = _snap(state[L.AX_VAL]) & 0xFF
        sp = _snap(state[L.SP_VAL])
        bp = _snap(state[L.BP_VAL])
        stk = _snap(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5

        # KV-backed stack maintenance (the depth-1 wall fix).
        if spill_stack_to_kv:
            if op == isa.PSH:
                saddr = sp & 0xFF                       # SP-relative stack address
                stack_kv.append((saddr, prev["AX"] & 0xFF))
                store_log = [s for s in store_log if (s["addr"] & 0xFF) != saddr]
                store_log.append({"addr": saddr, "val": prev["AX"] & 0xFF})
            elif op in _POP_OPS and stack_kv:
                _psaddr, _ = stack_kv.pop()            # ALU pop freed the top cell
                store_log = [s for s in store_log
                             if (s["addr"] & 0xFF) != (_psaddr & 0xFF)]

        # function control that spans the token stream (push/pop of ret-PC/saved-BP).
        if op == isa.JSR:
            call_stack.append((cur_pc + 1, bp))       # return PC = after JSR
        elif op == isa.ENT:
            call_stack.append((None, prev["BP"]))     # save caller BP
        elif op == isa.LEV:
            saved_bp = ret_pc = None
            if call_stack:
                _, saved_bp = call_stack.pop()
            if call_stack:
                ret_pc, _ = call_stack.pop()
            if saved_bp is not None:
                bp = saved_bp
            if ret_pc is not None:
                pc = ret_pc
        elif subset.memory and op in (isa.SI, isa.SC):
            if spill_stack_to_kv:
                store_addr = stack_kv[-1][1] if stack_kv else 0   # KV-backed top-of-stack
                if stack_kv:
                    _psaddr, _ = stack_kv.pop()
                    store_log = [s for s in store_log
                                 if (s["addr"] & 0xFF) != (_psaddr & 0xFF)]
            else:
                store_addr = _snap(state[L.STK_VAL])   # popped address (1-slot path)
            store_val = ax if op == isa.SI else (ax & 0xFF)
            # §Memory latest-write-wins as KV-log COMPACTION: a re-store to an
            # address SUPERSEDES the prior write, so drop the earlier frame and keep
            # only the latest. This makes each address appear at MOST once in the
            # window, so the address CAM is a clean one-frame content-address (the
            # RoPE recency lane is then only a safety tiebreak, never forced to
            # split two frames at the SAME sharp address match — the resolution
            # limit that otherwise pits same-address recency against a distinct
            # older-address exact match). Compaction also shrinks the window.
            store_log = [s for s in store_log
                         if (s["addr"] & 0xFF) != (store_addr & 0xFF)]
            store_log.append({"addr": store_addr, "val": store_val})

        if spill_stack_to_kv:
            stk = stack_kv[-1][1] if stack_kv else 0   # next STACK0 = KV top-of-stack
        reg_state = {"PC": pc, "AX": ax, "SP": sp, "BP": bp, "STACK0": stk}
        ax_trace.append(ax)
        if verbose:
            print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):5s} -> "
                  f"pc={pc} ax={ax} sp={sp} bp={bp} stk={stk} halt={halted}")
        cur_pc = pc
        if halted or cur_pc < 0 or cur_pc >= len(run_code):
            break
    return {"ax_trace": ax_trace, "ref_trace": ref_trace,
            "exact": ax_trace == ref_trace, "steps": len(ax_trace)}


# ===========================================================================
# D-BUDGET REPORT — which subset fits which Qwen2 residual/FFN/layer budget.
#
# Sizes each subset WITHOUT building the (possibly huge) model, so the wall is
# visible cheaply. The stock Qwen2.5-0.5B budget is hidden_size 896, intermediate
# 4864, 24 layers, 14 query heads.
#
# The 256x256x3 dense MUL/DIV/MOD LOOKUP TABLE has been REMOVED entirely (it was the
# unbuildable ~45 GB wall: intermediate ~160465 -> ~45 GB fp32). MUL/DIV/MOD now run
# ONLY through the efficient ``nibble_alu32`` fp32 FFN gadgets, so the muldiv/full
# rows here report the EFFICIENT intermediate (~1124 / ~2608), NOT the table width.
# ``_MDM_TABLE_WOULD_BE`` keeps the historical dense-table width as a documented
# ANALYTIC constant (computed WITHOUT building the table) for the docs/accounting.
# ===========================================================================
# The removed dense-table width: one FFN hidden unit per NONZERO op(a,b) over all
# 256x256 pairs of MUL/DIV/MOD, + 1 self-clear unit.  Materialising that was the
# ~45 GB wall; kept here only as the accounting reference the fit configurator and
# docs cite.  Computed once, tensor-free, from the tiny ``_MDM_FN`` truth table.
def _mdm_table_would_be() -> int:
    from .nibble_unified import _MDM_FN
    return 1 + sum(1 for op in (isa.MUL, isa.DIV, isa.MOD)
                   for a in range(256) for b in range(256) if _MDM_FN[op](a, b) != 0)


_MDM_TABLE_WOULD_BE = _mdm_table_would_be()   # == 160465 (the removed dense-table width)


def fit_report(code_size: int = 24, arch: QwenArch = QWEN2_5_ARCH) -> List[Dict]:
    """Width/depth fit per subset for the PRODUCTION build (efficient ALU is the only
    MUL/DIV/MOD path, so the muldiv/full rows show the ~1124/~2608 efficient
    intermediate — NEVER the removed ~160465 dense table)."""
    rows = []
    for sub in (SUBSET_BASE, SUBSET_MEM_CMP, SUBSET_BITWISE, SUBSET_MULDIV, SUBSET_FULL):
        eff = sub.muldiv                       # efficient ALU whenever muldiv is present
        QL = QwenFullLayout(code_size, sub, efficient_alu=eff)
        specs = _block_specs(QL.L, code_size, sub, efficient_alu=eff)
        inter = max(int(s["W_up"].shape[0]) for _, s in specs)
        inter = max(inter, arch.num_attention_heads * arch.head_dim, 8)
        hidden = arch.hidden_for(QL.D_used + 1)
        n_layers = len(specs)
        fits = (hidden <= STOCK_HIDDEN and inter <= STOCK_INTERMEDIATE
                and n_layers <= STOCK_LAYERS)
        rows.append({
            "subset": sub.name, "hidden_size": hidden, "intermediate_size": inter,
            "n_layers": n_layers, "query_heads": arch.num_attention_heads,
            "fits_stock_0_5b": fits,
            "over_hidden": max(0, hidden - STOCK_HIDDEN),
            "over_intermediate": max(0, inter - STOCK_INTERMEDIATE),
        })
    return rows


def fit_report_efficient(code_size: int = 24, arch: QwenArch = QWEN2_5_ARCH) -> List[Dict]:
    """Depth/width fit for the EFFICIENT-ALU MUL/DIV/MOD (nibble_alu32) — the table
    WIDTH (intermediate ~160465 -> ~45 GB) is traded for DEPTH (schoolbook rounds +
    long-division iterations).  Reports both the unrolled and the recurrent
    (layer-reuse) layer counts so the honest depth cost is visible.  ``n_stored`` is
    Qwen ``num_hidden_layers`` (distinct nn.Modules); ``n_applied`` is layers run per
    forward (== n_stored unless recurrent)."""
    rows = []
    for sub in (SUBSET_MULDIV, SUBSET_FULL):
        for recurrent in (False, True):
            QL = QwenFullLayout(code_size, sub, efficient_alu=True,
                                recurrent_divmod=recurrent)
            specs = _block_specs(QL.L, code_size, sub, efficient_alu=True,
                                 recurrent_divmod=recurrent)
            apply_order = getattr(QL.L, "_qwen_apply", None)
            inter = max(int(s["W_up"].shape[0]) for _, s in specs)
            inter = max(inter, arch.num_attention_heads * arch.head_dim, 8)
            hidden = arch.hidden_for(QL.D_used + 1)
            n_stored = len(specs)
            n_applied = len(apply_order) if apply_order is not None else n_stored
            rows.append({
                "subset": sub.name, "mode": "recurrent" if recurrent else "unrolled",
                "hidden_size": hidden, "intermediate_size": inter,
                "n_stored_layers": n_stored, "n_applied_layers": n_applied,
                "query_heads": arch.num_attention_heads,
                "fits_stock_0_5b_24L": n_stored <= STOCK_LAYERS,
                "fits_stock_intermediate": inter <= STOCK_INTERMEDIATE,
            })
    return rows
