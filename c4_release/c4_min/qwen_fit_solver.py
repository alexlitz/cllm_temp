"""MODEL-FIT CONFIGURATOR — pick the width↔depth↔steps↔precision lever combo that
fits the C4 VM into a constraint box, then bake it.

Generalises ``qwen_full_vm.fit_report`` / ``fit_report_efficient`` from a fixed set
of printed rows into a *solver*: given a constraint box (max depth / width /
precision / a named stock Qwen target) and an op-set to cover, it enumerates the
feasible ``(op-subset, muldiv-strategy, precision, recurrence)`` lever combos,
accounts each one's exact ``(hidden, intermediate, stored_layers, applied_depth,
steps_per_op)`` by reusing the REAL ``qwen_full_vm._block_specs`` builders, ranks by
an objective, and returns the best fit + a tradeoff table. If the box is infeasible
it reports the *binding constraint* and the closest relaxation.

Where the cost of each op goes
==============================
Every op's cost flows into one of four axes, and precision scales all of them:

  * **WIDTH**  (``intermediate_size``) — a lookup table (the 256×256×3 MUL/DIV/MOD
    table = ``intermediate 160465``) buys zero depth for a huge width.
  * **DEPTH**  (``num_hidden_layers``) — the efficient ALU (``nibble_alu32``) buys a
    small width (``intermediate 11272``) for a large depth (schoolbook carry rounds +
    long-division iterations: MUL +10 layers, DIV/MOD +262 layers).
  * **STEPS**  (``steps_per_op``) — a bytecode SUBROUTINE (JSR/LEV into a baked
    library routine) buys near-zero *extra* width/depth for many *program steps* per
    op (each program step = one ``Qwen2Model.forward``).
  * **PRECISION** (8/16/32 bits) — the nibble granularity scales the ALU depth
    (fewer/more div iterations + carry rounds) and the lookup-table width.

Lever ↔ cost axis
-----------------
    muldiv-strategy = lookup-table            -> WIDTH  (huge intermediate)
    muldiv-strategy = efficient-ALU-unrolled  -> DEPTH  (many stored layers)
    muldiv-strategy = efficient-ALU-recurrent -> DEPTH applied, ~half stored (reuse)
    muldiv-strategy = subroutine              -> STEPS  (per-op program steps)  [EST]
    precision 8/16/32                          -> scales all three

#913 fp32-FIT levers (FitConfig.pack_memcam / overlap_scratch / bit_level_bitwise)
-----------------------------------------------------------------------------------
Three MEASURED reductions of the residual-width / depth floor, each DEFAULT OFF (so
the base accounting is byte-identical to before) and each a delta read from the REAL
layout (``_packmemcam_saving`` / ``_alu_scratch_overlap`` / ``_bitwise_barrel_saving``):

  * **pack_memcam** — drop the ~330-780 DEAD head-dim pad dims the pure-forward
    §Memory head allocated but this Qwen build (re-bakes attention on Qwen geometry)
    never uses (``C4_PACK_MEMCAM``).  Collapses the subroutine hidden 1152 -> 896.
  * **overlap_scratch** — THE op-overlap lever: the inline-ALU scratch bands are
    stored side-by-side (SUM) but only ONE op fires per step, so under an opcode-gated
    parallel-scratch mux the residual holds operand + MAX-over-families, not the SUM.
    Measured: sum 1335 -> operand(12)+MAX(divmod 1165) = 1177.  It NARROWS the inline
    ALU but divmod's own 1165-dim scratch still exceeds 896, so the inline ALU does
    NOT fit even fully overlapped — subroutine (ALU out of the residual) is the lever.
  * **bit_level_bitwise** — #911 per-bit polynomial AND/OR/XOR (depth-1 combine, no
    attention) replaces the 256-entry nibble-LUT, and SHL/SHR leave the persistent
    stack (routed to subroutine / native MUL-DIV), retiring the ~17 barrel-shifter
    blocks: subroutine stored 31 -> 14.

  RESULT (#913): subroutine + pack_memcam + bit_level_bitwise FITS stock Qwen2.5-0.5B
  at fp32 — hidden 896 (<=896), intermediate 896 (<=4864), stored 14 (<=24).

#916-continued INLINE + FEED-FORWARD fit (attn_cam_div LR-dup / bit_level_shift / overlap_depth)
------------------------------------------------------------------------------------------------
The #916 goal: fit the FULL fp32 C4 ISA into stock 0.5B INLINE (DIV/MOD stay in the
weights, NOT exported to program STEPS like subroutine) and FEED-FORWARD (no loop/tie).
The prior pass reached hidden 1472 / inter 9710 / stored 115 with attn_cam_div (#908
R256E attention-CAM DIV); the three remaining binders are cracked here (all DEFAULT OFF):

  * **attn_cam_div (LR-dup removal)** — the production divmod allocates the residual
    TWICE: the ``ALU_*`` nibble_alu32 divmod scratch (1156) AND the compact radix-16
    lean-radix ``LR_*`` datapath (480, the ``div_radix16_lean`` KB-lookahead/GE-threshold
    quotient-estimate).  The ``LR_*`` band is touched ONLY by the ``lean-*`` blocks the
    attention-CAM swap removes, so it is DEAD once R256E replaces the radix-16 division —
    the divmod residual is 277 TOTAL (R256E running state), and BOTH the 1156 scratch and
    the 480 LR datapath leave.  MEASURED: hidden 1472 -> 960.
  * **bit_level_shift** — SHL/SHR run as native MUL/DIV (x*2^n / x//2^n, shift_via_mul,
    the fit regime), which drops the barrel-shifter BLOCKS but leaves the layout still
    ALLOCATING the dead ``SH_STAGE_*`` (160) + ``TS_*`` (118) = 278 residual bands that NO
    block touches — the shift analog of bit_level_bitwise retires them.  hidden 960 -> 896.
  * **narrower R256E multiply** (in div_radix256_est) — the R256E per-digit qhat*dn and
    MOD q*d carry-normalise used a kmax=255 staircase (9710/7666-wide blocks, the INTER
    binder); a nibble-SPLIT of the quotient byte (qlo/qhi) + a TWO-LEVEL lane-floor make
    every carry kmax<=112 (mostly 15).  MEASURED byte-exact (fp64 6084/6084, fp32
    single-row 1195/1200, a hair BETTER than the base 1188) — inter 9710 -> 3376.
  * **overlap_depth** — op-overlap: only ONE opcode fires per step, so the per-op-family
    machinery is opcode-gated onto SHARED physical layers -> depth = base pipeline + the
    DEEPEST single op cascade (max-op), NOT the SUM.  MEASURED: stored 123 (sum) -> 108
    (max-op = base 10 + divmod 98).

  RESULT (#916): inline + feed-forward FITS stock 0.5B on WIDTH — hidden 896 (<=896) AND
  intermediate 3376 (<=4864) — but NOT on DEPTH: the DIV/MOD unrolled digit-recurrence is
  ~98 sequential layers (each block reads the prior block's residual, no fusion), so the
  op-overlap max-op depth is 108 > 24.  DIV/MOD only reaches ~48 via RECURRENCE (weight-tied
  loop = Universal-Transformer, divmod body stored once) — still > 24, and looping is
  excluded for stock feed-forward Qwen2.  IRREDUCIBLE BINDER: divmod unrolled depth.

Honesty (verified vs estimated)
-------------------------------
  * **lookup-table** width, **efficient-ALU** unrolled/recurrent depth+width — VERIFIED:
    computed live from ``qwen_full_vm._block_specs`` (the same builders ``build``
    bakes), reproducing #698's ``fit_report_efficient`` numbers exactly (MUL +10 →
    ~23 layers, DIV/MOD 262 unrolled / 138 stored recurrent, intermediate 11272).
  * **precision != 32** on the efficient ALU — ESTIMATED. The baked ``nibble_alu32``
    is 32-bit-exact (fixed 8-nibble DIV / 4-byte ADD-SUB). The 8/16-bit *depths* here
    are a linear-in-nibble accounting PROJECTION of what a precision-parameterised ALU
    would cost — they are NOT a config you can bake on this branch (``verified=False``).
  * **subroutine** steps_per_op — ESTIMATED (labelled ``[EST]``). The bytecode
    MUL/DIV/MOD subroutine step-counts come from #699 and are pending #699-reverify;
    treated as an estimate until confirmed.

Everything except the actual bake runs on CPU with no model materialised (the
accounting sizes the layout WITHOUT building the possibly-huge model, exactly like
``fit_report``).  MEMORY-SAFE: the 256x256x3 dense lookup table (``intermediate
160465``, the historical ~45 GB / ~55 GB-peak-RSS wall) has been REMOVED — the only
MUL/DIV/MOD path is now the efficient nibble_alu32 ALU.  The "lookup-table" strategy
is therefore a pure ACCOUNTING/tradeoff row whose width is the ANALYTIC count of the
removed table (``qwen_full_vm._MDM_TABLE_WOULD_BE`` == 160465, tensor-free); nothing
ever materialises a table, so the solver is memory-safe unconditionally (peaks
~4.7 GB from the muldiv-OFF reference build only).  Spec builds are
``lru_cache``-memoized on the spec-determining levers, so the P8/P16/P32 variants
share one build.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from . import isa
from . import qwen_full_vm as Q
from .qwen_full_vm import QwenArch, QWEN2_5_ARCH, QwenFullLayout, Subset, _block_specs


# ===========================================================================
# Named stock Qwen2.5 targets (hidden, intermediate, layers, arch heads).
# ``arch`` fixes the GQA head geometry; only 0.5B's 14/2 GQA is what qwen_full_vm's
# CAM was built for, but the depth/width *budgets* of the larger stocks are still
# valid fit ceilings.
# ===========================================================================
@dataclass(frozen=True)
class StockTarget:
    name: str
    hidden: int
    intermediate: int
    layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int = 64

    @property
    def arch(self) -> QwenArch:
        return QwenArch(
            name=f"{self.name}-arch",
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
        )


STOCK_TARGETS: Dict[str, StockTarget] = {
    "stock-qwen2.5-0.5b": StockTarget("stock-qwen2.5-0.5b", 896, 4864, 24, 14, 2, 64),
    "stock-1.5b": StockTarget("stock-1.5b", 1536, 8960, 28, 12, 2, 128),
    "stock-7b": StockTarget("stock-7b", 3584, 18944, 28, 28, 4, 128),
}
STOCK_TARGETS["stock-0.5b"] = STOCK_TARGETS["stock-qwen2.5-0.5b"]     # alias


# ===========================================================================
# Op set → Subset capability flags.
#
# qwen_full_vm composes op families at the granularity of the four Subset flags
# (memory / cmp / bitwise / muldiv); the base decode+CAM+arith+control stack is
# always present.  Given a set of opcodes to COVER we light exactly the flags that
# family requires.  MUL/DIV/MOD light ``muldiv``.  We track whether DIV/MOD are
# actually in the op-set (a MUL-only set skips the 262 long-division layers).
# ===========================================================================
BASE_OPS = {isa.IMM, isa.LEA, isa.JMP, isa.BZ, isa.BNZ, isa.PSH, isa.ADD, isa.SUB,
            isa.JSR, isa.ENT, isa.ADJ, isa.LEV, isa.NOP, isa.HALT}
MEM_OPS = {isa.LI, isa.LC, isa.SI, isa.SC}
CMP_OPS = {isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE}
BITWISE_OPS = {isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR}
MUL_OPS = {isa.MUL}
DIVMOD_OPS = {isa.DIV, isa.MOD}
MULDIV_OPS = MUL_OPS | DIVMOD_OPS

# "FULL" = every op family.
FULL = frozenset(BASE_OPS | MEM_OPS | CMP_OPS | BITWISE_OPS | MULDIV_OPS)


def subset_for_ops(ops: Sequence[int]) -> Subset:
    """Light the minimal Subset capability flags that COVER ``ops``."""
    ops = set(ops)
    memory = bool(ops & MEM_OPS)
    cmp = bool(ops & CMP_OPS)
    bitwise = bool(ops & BITWISE_OPS)
    muldiv = bool(ops & MULDIV_OPS)
    name = "+".join(
        [t for t, on in (("mem", memory), ("cmp", cmp), ("bit", bitwise),
                         ("muldiv", muldiv)) if on] or ["base"])
    return Subset(memory=memory, cmp=cmp, bitwise=bitwise, muldiv=muldiv, name=name)


# ===========================================================================
# Levers.
# ===========================================================================
MULDIV_STRATEGIES = (
    "lookup-table",              # WIDTH  (verified)
    "efficient-ALU-unrolled",    # DEPTH  (verified)
    "efficient-ALU-recurrent",   # DEPTH applied, ~half stored (verified)
    "subroutine",                # STEPS  (estimated — #699 pending reverify)
)
PRECISIONS = (8, 16, 32)
GRANULARITIES = ("nibble", "byte")

# Verified 32-bit DIV/MOD long-division anchors (compile_divmod_blocks does range(8);
# 262 total unrolled blocks: KB-precompute + init + 8×(≈30-block iteration) + finalize).
_DIV_ITERS_32 = 8
_DIVMOD_UNROLLED_LAYERS = 262
_DIV_FIXED_LAYERS = 22                 # KB-precompute (93→incl in stored) + init + finalize
_DIV_LAYERS_PER_ITER = (_DIVMOD_UNROLLED_LAYERS - _DIV_FIXED_LAYERS) // _DIV_ITERS_32  # ≈30

# Estimated bytecode-subroutine program-step counts per op (#699 lean muldiv library,
# branch verify-lean-muldiv-mandelbrot — LABELLED [EST], pending #699 re-verify).
#
# A subroutine MUL/DIV/MOD is a baked bytecode library routine reached by JSR/LEV: it
# adds ~zero persistent width/depth but costs many PROGRAM STEPS (each step = one
# Qwen forward).  The cost is O(precision):
#   * schoolbook MUL  ~ O(bytes^2) partial-products with a per-byte carry inner loop;
#   * base-2^8 long DIV/MOD ~ O(bytes) outer digits x an inner compare/subtract loop.
# We anchor the 8-bit points on #699's observed lean-stack step counts (an 8-bit
# ``program_mul8`` runs ~1.5e2 steps incl operand setup; DIV/MOD a bit more) and scale
# per BYTE of precision.  Conservative + clearly ESTIMATED.
_SUB_STEP_MUL_PER_BYTE2 = 150      # ~O(bytes^2): 8-bit(1 byte) ~= 150 steps (#699 obs.)
_SUB_STEP_DIVMOD_PER_BYTE = 200    # ~O(bytes) outer x inner loop: 8-bit ~= 200 steps
_SUBROUTINE_NOTE = ("subroutine steps_per_op ESTIMATED "
                    "(#699 lean-muldiv, branch verify-lean-muldiv-mandelbrot; "
                    "pending re-verify)")


def _est_subroutine_steps(precision: int, divmod_: bool) -> int:
    """Precision-scaled bytecode-subroutine step estimate per op ([EST], #699)."""
    nbytes = max(1, -(-precision // 8))            # ceil(P/8) bytes
    if divmod_:
        return _SUB_STEP_DIVMOD_PER_BYTE * nbytes
    return _SUB_STEP_MUL_PER_BYTE2 * nbytes * nbytes


def _div_iters(precision: int) -> int:
    return max(1, -(-precision // 4))          # ceil(P/4) nibbles


# ===========================================================================
# Config schema.
# ===========================================================================
@dataclass(frozen=True)
class FitConfig:
    """A concrete lever combo the solver can account + (for the buildable ones) bake."""
    ops: frozenset
    muldiv_strategy: str = "efficient-ALU-recurrent"
    precision: int = 32                          # 8 / 16 / 32
    granularity: str = "nibble"
    code_size: int = 24
    arch: QwenArch = field(default_factory=lambda: QWEN2_5_ARCH)
    # ---- #913 fp32-fit levers (all default OFF -> byte-identical to the prior
    #      accounting; each is a MEASURED reduction of the residual/depth) --------
    pack_memcam: bool = False        # drop the ~800 dead head-dim pad dims that this
                                     #   Qwen build never uses (C4_PACK_MEMCAM). The
                                     #   pure-forward layout pads up to n_heads*MEM_HEAD
                                     #   CHANNELS for its OWN §Memory head, but this build
                                     #   re-bakes attention on the Qwen geometry, so those
                                     #   dims are dead weight. OFF -> byte-identical wider.
    overlap_scratch: bool = False    # THE op-overlap lever: the inline-ALU scratch bands
                                     #   are allocated side-by-side (SUM) but only ONE op
                                     #   fires per step. Under an opcode-gated parallel-
                                     #   scratch mux the residual holds base + operand +
                                     #   MAX-over-families(private scratch), not the SUM.
    bit_level_bitwise: bool = False  # #911 bit-level (per-bit polynomial) AND/OR/XOR:
                                     #   depth-1-combine, no attention, ~narrow; replaces
                                     #   the 256-entry nibble-LUT. SHL/SHR leave the
                                     #   persistent stack (routed to subroutine/MUL-DIV),
                                     #   dropping the ~17 barrel-shifter blocks.
    bit_level_shift: bool = False    # the SHIFT analog of bit_level_bitwise: SHL/SHR run
                                     #   as native MUL/DIV (x*2^n / x//2^n, "bit-level
                                     #   shift" per-bit position map on the muldiv path,
                                     #   shift_via_mul).  The barrel-shifter BLOCKS already
                                     #   drop, but the layout still ALLOCATES the dead
                                     #   SH_STAGE_* (160) + TS_* (118) = 278 residual bands
                                     #   that NO block touches — this retires them.
    overlap_depth: bool = False      # DEPTH op-overlap: only ONE opcode fires per step, so
                                     #   the per-op-family machinery is opcode-gated and
                                     #   SHARES the same physical layers -> the feed-forward
                                     #   network depth is the SHARED base pipeline + the
                                     #   DEEPEST single op-family cascade (max-op), NOT the
                                     #   SUM of every family's blocks.  MEASURED from the
                                     #   live block stack (_maxop_overlap_depth).
    # ---- #916 INLINE-fit lever (attention-CAM DIV, #908) — DEFAULT OFF -----------
    attn_cam_div: bool = False       # #908 monotone-GE threshold-CAM DIV/MOD: the wide
                                     #   radix-16 KB-lookahead / GE-threshold quotient-
                                     #   estimate bank (8 ALU_ bands == 1089 of the 1165
                                     #   divmod RESIDUAL dims + the 4x 7920-wide lean-kb
                                     #   FFN estimate blocks) moves OUT of FFN/residual and
                                     #   INTO ONE softmax1 attention head with 255 SHARED,
                                     #   divisor-INDEPENDENT KV rows.  Measured (#908
                                     #   div_radix256_est / attn_radix_div @ 43b5cd74):
                                     #   the divmod residual scratch drops 1165 -> 277
                                     #   (the R256E running remainder/divisor/digit/result/
                                     #   SRT-lane state, NOT the bank), and the max FFN
                                     #   intermediate the divmod family drives drops from
                                     #   7920 (lean-kb-c*) to the radix-256 SRT lane
                                     #   multiplies.  This is the INLINE (in-weights, feed-
                                     #   forward) muldiv fit lever — DIV/MOD stay on the
                                     #   persistent stack (UNLIKE subroutine, which exports
                                     #   them to program STEPS).

    @property
    def subset(self) -> Subset:
        return subset_for_ops(self.ops)

    @property
    def has_muldiv(self) -> bool:
        return bool(set(self.ops) & MULDIV_OPS)

    @property
    def has_divmod(self) -> bool:
        return bool(set(self.ops) & DIVMOD_OPS)

    @property
    def efficient_alu(self) -> bool:
        return self.muldiv_strategy.startswith("efficient-ALU") and self.has_muldiv

    @property
    def recurrent_divmod(self) -> bool:
        return self.muldiv_strategy == "efficient-ALU-recurrent" and self.has_divmod

    @property
    def subroutine_muldiv(self) -> bool:
        return self.muldiv_strategy == "subroutine" and self.has_muldiv

    @property
    def label(self) -> str:
        strat = self.muldiv_strategy if self.has_muldiv else "n/a"
        return f"[{self.subset.name}] muldiv={strat} P{self.precision} {self.granularity}"


# ===========================================================================
# ACCOUNTING — size a config WITHOUT baking the (possibly huge) model, by calling
# the real qwen_full_vm._block_specs builders.  Precision-scales the result.
# ===========================================================================
@dataclass
class Accounting:
    config: FitConfig
    hidden: int
    intermediate: int
    stored_layers: int              # Qwen num_hidden_layers (distinct nn.Modules)
    applied_depth: int              # layers run per forward (>= stored if recurrent)
    steps_per_op: int               # program steps (Qwen forwards) per op
    query_heads: int
    verified: bool                  # accounting VERIFIED-buildable at this config?
    params_estimate: int            # rough dense param count (for smallest-params rank)
    notes: str = ""


# The dense 8-bit MUL/DIV/MOD LOOKUP-TABLE width, computed ANALYTICALLY (no tensors).
# The table was ONE hidden unit per NONZERO op(a,b) over all 256x256 pairs of
# MUL/DIV/MOD, + 1 self-clear unit; MATERIALISING it was the ~45 GB / ~55 GB-RSS wall.
# The dense table has been REMOVED entirely (the only MUL/DIV/MOD path is now the
# efficient nibble_alu32 ALU), so the "lookup-table" strategy here is a pure
# ACCOUNTING/tradeoff row: its width is the analytic count of the removed table (the
# nonzero ``_MDM_FN`` entries, tensor-free) — nothing ever builds a table.  This is
# always analytic now (there is no table to build), so the strategy is memory-safe
# unconditionally.
_ANALYTIC_LOOKUP_TABLE = True


@lru_cache(maxsize=None)
def _mdm_lookup_width() -> int:
    """The analytic width the REMOVED dense MUL/DIV/MOD table would have had: nonzero
    op(a,b) entries over all 256x256 pairs + 1 self-clear unit (== 160465).  Sourced
    from ``qwen_full_vm._MDM_TABLE_WOULD_BE`` (the documented analytic constant),
    tensor-free — nothing builds a table."""
    return Q._MDM_TABLE_WOULD_BE


@lru_cache(maxsize=None)
def _raw_spec_sizes(code_size: int, memory: bool, cmp: bool, bitwise: bool,
                    muldiv: bool, efficient_alu: bool, recurrent_divmod: bool
                    ) -> Tuple[int, int, int, int]:
    """(dim_used, intermediate, stored_layers, applied_depth) from the REAL
    ``qwen_full_vm._block_specs`` builders — MEMOIZED on the spec-determining levers.

    Precision is a POST-HOC projection (§account) and does NOT change the baked
    specs, so the three (P8/P16/P32) accountings of e.g. the 160465-wide MUL/DIV/MOD
    lookup-table TRADEOFF ROW share one accounting.  ``arch`` (head geometry) is
    folded in by the caller — it only rescales ``hidden``/``intermediate`` floors,
    never the spec shapes — so it stays out of the cache key.  This is the whole
    reason the CPU solver is fast + memory-safe: only the distinct (subset x strategy)
    spec builds ever materialise, once.

    The dense MUL/DIV/MOD lookup table has been REMOVED (the ~45 GB wall is gone), so
    the "lookup-table" strategy is a pure ANALYTIC tradeoff row: we account it from the
    memory-LIGHT muldiv-OFF build (~4.5 GB) plus the analytic removed-table width
    (``_mdm_lookup_width`` == 160465) + the 2 lookup blocks it would have added — i.e.
    what a dense byte×byte table WOULD have cost.  Nothing builds a table."""
    lookup_table = muldiv and not efficient_alu
    if lookup_table and _ANALYTIC_LOOKUP_TABLE:
        # analytic: size from the muldiv-OFF build + the removed-table's analytic width.
        sub = Subset(memory=memory, cmp=cmp, bitwise=bitwise, muldiv=False)
        QL = QwenFullLayout(code_size, sub)
        specs = _block_specs(QL.L, code_size, sub)
        base_w = max(int(s["W_up"].shape[0]) for _, s in specs)
        inter = max(base_w, _mdm_lookup_width())    # the removed mdm-select dominated
        n_stored = len(specs) + 2                    # + mdm-expand + mdm-select (removed)
        return QL.D_used, inter, n_stored, n_stored
    sub = Subset(memory=memory, cmp=cmp, bitwise=bitwise, muldiv=muldiv)
    QL = QwenFullLayout(code_size, sub, efficient_alu=efficient_alu,
                        recurrent_divmod=recurrent_divmod)
    specs = _block_specs(QL.L, code_size, sub, efficient_alu=efficient_alu,
                         recurrent_divmod=recurrent_divmod)
    apply_order = getattr(QL.L, "_qwen_apply", None)
    inter = max(int(s["W_up"].shape[0]) for _, s in specs)
    n_stored = len(specs)
    n_applied = len(apply_order) if apply_order is not None else n_stored
    return QL.D_used, inter, n_stored, n_applied


# ===========================================================================
# #913 fp32-fit levers — MEASURED reductions of the residual width + depth.
# Each helper reads the REAL layout/blocks (no model built), so the reductions
# are numbers the build path could actually realize, not hand-waves.
# ===========================================================================

# The dead head-dim pad the pure-forward §Memory head allocated but this Qwen build
# never uses.  Measured directly as D_used(pad-off) - D_used(pad-on).  See
# QwenFullLayout._compact_layout_drop_pad + the C4_PACK_MEMCAM comment (qwen_full_vm).
@lru_cache(maxsize=None)
def _packmemcam_saving(code_size: int, memory: bool, cmp: bool, bitwise: bool,
                       muldiv: bool, efficient_alu: bool, recurrent_divmod: bool) -> int:
    """D_used dims removed by dropping the dead §Memory head pad (C4_PACK_MEMCAM).
    Measured as the difference between the pad-off and pad-on layouts of the SAME
    subset — a genuine layout collapse the build already supports."""
    import os as _os
    from . import qwen_full_vm as _Q
    sub = Subset(memory=memory, cmp=cmp, bitwise=bitwise, muldiv=muldiv)
    prev = _os.environ.get("C4_PACK_MEMCAM")
    try:
        _os.environ["C4_PACK_MEMCAM"] = "0"
        _Q._pack_memcam.cache_clear() if hasattr(_Q._pack_memcam, "cache_clear") else None
        off = _Q.QwenFullLayout(code_size, sub, efficient_alu=efficient_alu,
                                recurrent_divmod=recurrent_divmod).D_used
        _os.environ["C4_PACK_MEMCAM"] = "1"
        _Q._pack_memcam.cache_clear() if hasattr(_Q._pack_memcam, "cache_clear") else None
        on = _Q.QwenFullLayout(code_size, sub, efficient_alu=efficient_alu,
                               recurrent_divmod=recurrent_divmod).D_used
    finally:
        if prev is None:
            _os.environ.pop("C4_PACK_MEMCAM", None)
        else:
            _os.environ["C4_PACK_MEMCAM"] = prev
        _Q._pack_memcam.cache_clear() if hasattr(_Q._pack_memcam, "cache_clear") else None
    return max(0, off - on)


# The inline-ALU scratch bands, tallied per machinery FAMILY from the LIVE layout.
# Op-overlap replaces the SUM of the private per-family scratch with operand(shared)
# + MAX-over-families, because only ONE op fires per step (an opcode-gated parallel-
# scratch mux, the nibble_alu32 pattern).  Returned as (sum, overlapped, families).
_ALU_FAMILY = {  # ALU_<name> -> machinery family (operand bands are SHARED across ops)
    "A": "operand", "B": "operand", "NOTB": "operand",
    "ADD_C": "addsub", "SUB_C": "addsub", "ADD_RES": "addsub", "SUB_RES": "addsub",
    "PP": "mul", "MCOL": "mul", "MC1": "mul", "MUL_RES": "mul",
    "MUL_T": "mul", "MUL_G0": "mul", "MUL_G1": "mul", "MUL_P0": "mul", "MUL_P1": "mul",
    "R": "divmod", "R2": "divmod", "KB": "divmod", "GT": "divmod", "EQ_N": "divmod",
    "QD": "divmod", "QB": "divmod", "SUBB": "divmod", "DIV_RES": "divmod",
    "MOD_RES": "divmod", "BZ": "divmod", "IT": "divmod", "IT_OH": "divmod",
    "KB_T": "divmod", "KB_G0": "divmod", "KB_G1": "divmod", "KB_P0": "divmod",
    "KB_P1": "divmod", "SGN_A": "divmod", "SGN_B": "divmod", "RES_SGN": "divmod",
    "ZCUM": "divmod", "SH_N_OH": "shift",
}


@lru_cache(maxsize=None)
def _alu_scratch_overlap(code_size: int, recurrent_divmod: bool) -> Tuple[int, int, dict]:
    """(sum_scratch, overlapped_scratch, per_family) for the inline-ALU scratch bands,
    read from the LIVE layout's named ALU_ bands.  overlapped = operand + MAX(family)
    (one op active per step); sum = every family's scratch side-by-side (the SUM the
    current build stores)."""
    from . import qwen_full_vm as _Q
    sub = Subset(memory=True, cmp=True, bitwise=True, muldiv=True)
    QL = _Q.QwenFullLayout(code_size, sub, efficient_alu=True,
                           recurrent_divmod=recurrent_divmod)
    names = getattr(QL.L, "_names", {})
    from collections import defaultdict
    fw: Dict[str, int] = defaultdict(int)
    for k, (off, w) in names.items():
        if k.startswith("ALU_"):
            base = k[len("ALU_"):]
            fam = _ALU_FAMILY.get(base)
            if fam:
                fw[fam] += w
    total = sum(fw.values())
    op_families = [v for f, v in fw.items() if f != "operand"]
    overlapped = fw.get("operand", 0) + (max(op_families) if op_families else 0)
    return total, overlapped, dict(fw)


# The barrel-shifter blocks + residual the #911 bit-level bitwise retires (SHL/SHR
# leave the persistent stack; only the depth-1 OR/XOR/AND combine remains).  Measured
# as the block/residual delta between the barrel-shift-on and barrel-shift-off (shift_
# via_mul) bitwise builds of the same subset.
@lru_cache(maxsize=None)
def _bitwise_barrel_saving(code_size: int, pack_memcam: bool) -> Tuple[int, int]:
    """(blocks_removed, residual_dims_removed) when the 17 barrel-shifter blocks leave
    the bitwise stack (SHL/SHR routed to subroutine / native MUL-DIV, keeping only the
    depth-1 #911 per-bit OR/XOR/AND combine).

    ``pack_memcam`` selects the residual regime: with the dead §Memory pad PRESENT
    (pad-on) the barrel-shifter bands sit INSIDE the dead pad so removing them frees
    NO net residual (the pad already dominates the floor); with the pad DROPPED
    (pack_memcam, the fit regime) the barrel shifter's SH_KEEP band is REAL residual
    that the bit-level construction retires.  The block count is pad-invariant."""
    import os as _os
    from . import qwen_full_vm as _Q
    sub = Subset(memory=True, cmp=True, bitwise=True, muldiv=False)
    sub_nobit = Subset(memory=True, cmp=True, bitwise=False, muldiv=False)
    prev = _os.environ.get("C4_PACK_MEMCAM")
    try:
        _os.environ["C4_PACK_MEMCAM"] = "1" if pack_memcam else "0"
        QL_full = _Q.QwenFullLayout(code_size, sub, efficient_alu=False)
        specs_full = _block_specs(QL_full.L, code_size, sub, efficient_alu=False)
        # bit-level combine keeps only the non-tshift bitwise blocks.
        shift_blocks = sum(1 for n, _ in specs_full if "tshift" in n)
        d_bit = QL_full.D_used
        d_nobit = _Q.QwenFullLayout(code_size, sub_nobit, efficient_alu=False).D_used
    finally:
        if prev is None:
            _os.environ.pop("C4_PACK_MEMCAM", None)
        else:
            _os.environ["C4_PACK_MEMCAM"] = prev
    # residual freed = the bitwise residual delta minus the bit-level combine floor.
    # The bit-level combine (A_BIT/B_BIT planes + recompose) still needs its bit-plane
    # bands; the SH_KEEP barrel-shift keep-band is the removable part.  In the pad-on
    # regime the bitwise residual delta is 0 (bands absorbed by the pad) -> 0 freed.
    bitwise_residual = d_bit - d_nobit
    L = QL_full.L
    bands = sorted((getattr(L, nm), nm) for nm in dir(L)
                   if not nm.startswith("_") and isinstance(getattr(L, nm, None), int))
    sh_keep_w = 0
    for i, (o, nm) in enumerate(bands):
        if nm == "SH_KEEP":
            nxt = bands[i + 1][0] if i + 1 < len(bands) else o
            sh_keep_w = nxt - o
            break
    residual_removed = min(sh_keep_w, max(0, bitwise_residual))
    return shift_blocks, residual_removed


# The DEAD barrel-shifter residual the shift-via-MUL/DIV route leaves behind.  In the
# fit regime the production layout runs ``shift_via_mul=True`` (SHL = x*2^n via MUL,
# SHR = x//2^n via DIV — the "bit-level shift", per-bit position map riding the muldiv
# datapath), which DROPS the 17 barrel-shifter (tshift) BLOCKS, but the layout still
# ALLOCATES the barrel-shifter RESIDUAL bands (the 5-stage ``SH_STAGE_*`` shifter, 160,
# + the ``TS_*`` two-shift datapath, 118 = 278 dims).  With shift-via-mul on, NOTHING
# reads or writes those bands (verified: 0 blocks touch them), so they are DEAD residual
# a bit-level-shift construction retires — the shift analog of ``bit_level_bitwise``.
@lru_cache(maxsize=None)
def _bitlevel_shift_saving(code_size: int, pack_memcam: bool) -> int:
    """Dead ``SH_STAGE_*`` (barrel-shifter) + ``TS_*`` (two-shift) residual dims freed
    when SHL/SHR route through native MUL/DIV (``shift_via_mul``, the fit regime) — the
    barrel-shifter bands are allocated but touched by NO block, MEASURED live."""
    import os as _os
    from . import qwen_full_vm as _Q
    sub = Subset(memory=True, cmp=True, bitwise=True, muldiv=True)
    prev = _os.environ.get("C4_PACK_MEMCAM")
    try:
        _os.environ["C4_PACK_MEMCAM"] = "1" if pack_memcam else "0"
        if hasattr(_Q, "_pack_memcam") and hasattr(_Q._pack_memcam, "cache_clear"):
            _Q._pack_memcam.cache_clear()
        QL = _Q.QwenFullLayout(code_size, sub, efficient_alu=True,
                               recurrent_divmod=False, shift_via_mul=True)
        names = getattr(QL.L, "_names", {})
        dead = sum(w for k, (off, w) in names.items()
                   if k.startswith("SH_STAGE") or k.startswith("TS_"))
    finally:
        if prev is None:
            _os.environ.pop("C4_PACK_MEMCAM", None)
        else:
            _os.environ["C4_PACK_MEMCAM"] = prev
        if hasattr(_Q, "_pack_memcam") and hasattr(_Q._pack_memcam, "cache_clear"):
            _Q._pack_memcam.cache_clear()
    return dead


# ===========================================================================
# #916 INLINE-fit lever — attention-CAM DIV/MOD (#908).  The wide radix-16
# KB-lookahead / GE-threshold quotient-estimate BANK moves OUT of FFN/residual
# and INTO ONE softmax1 attention head with 255 SHARED, divisor-independent KV
# rows.  Every number below is MEASURED from the real layouts (no model built):
#
#   * the removed divmod THRESHOLD bank + the KEPT running state, from the
#     production nibble_alu32 ALU_ bands (``_alu_scratch_overlap`` families);
#   * the replacement residual + FFN + depth, from the on-branch #908 SRT
#     pipeline ``div_radix256_est`` (the estimate block is what the attention
#     head replaces; the rest of the R256E pipeline is byte-exact-validated).
# ===========================================================================

# The 8 wide radix-16 divmod bands that ARE the GE-threshold / KB-lookahead
# quotient-estimate bank (moved to the 255 shared attention KV rows).  The
# REMAINING divmod ALU_ bands are the running remainder/divisor/digit/result/
# sign state, which stays in the residual.  (Measured split: bank 1089, running
# 76, of the 1165 recurrent divmod scratch — see docs/INLINE_FEEDFORWARD_FIT_916.md.)
_DIVMOD_THRESHOLD_BANK_BANDS = frozenset(
    {"KB", "GT", "EQ_N", "KB_T", "KB_G0", "KB_P0", "KB_G1", "KB_P1"})


@lru_cache(maxsize=None)
def _attn_cam_div_geometry(code_size: int, recurrent_divmod: bool) -> dict:
    """MEASURED geometry of the attention-CAM DIV/MOD swap (#908), all read live:

      * ``divmod_residual_before`` — the production radix-16 divmod family
        residual scratch (the ALU_ divmod bands: 1165 recurrent / 1156 unrolled).
      * ``divmod_bank`` / ``divmod_running`` — that scratch split into the wide
        GE-threshold/KB bank (moved to attention KV) and the kept running state.
      * ``divmod_residual_after`` — the #908 R256E residual scratch (the running
        remainder/divisor/digit/result/SRT-lane state) that REPLACES it: 277.
      * ``kv_rows`` — the shared, divisor-independent GE-threshold KV rows (255).
      * ``r256e_stored`` / ``r256e_applied`` — the R256E SRT pipeline's block
        count (36 unique / 90 applied at radix-256, 4 digit iterations).
      * ``r256e_ffn_max`` — the widest FFN block the R256E pipeline keeps (the
        radix-256 qhat*divisor lane multiply, 9710 — this is the honest INTER
        tension the radix-256 estimate-CAM trades the wide threshold bank for).
      * ``inter_without_leankb`` — the inline-ALU max FFN intermediate once the
        radix-16 lean-kb estimate FFN blocks (7920) leave (the next-widest block).
      * ``leankb_blocks`` — the radix-16 estimate FFN blocks removed (lean-kb*).
    """
    from . import qwen_full_vm as _Q
    sub = Subset(memory=True, cmp=True, bitwise=True, muldiv=True)
    QL = _Q.QwenFullLayout(code_size, sub, efficient_alu=True,
                           recurrent_divmod=recurrent_divmod)
    names = getattr(QL.L, "_names", {})
    bank = running = 0
    for k, (off, w) in names.items():
        if not k.startswith("ALU_"):
            continue
        base = k[len("ALU_"):]
        if _ALU_FAMILY.get(base) != "divmod":
            continue
        if base in _DIVMOD_THRESHOLD_BANK_BANDS:
            bank += w
        else:
            running += w
    divmod_residual_before = bank + running

    # The SECOND, duplicate divmod residual: the radix-16 lean-radix datapath
    # (``LR_*`` bands, ``div_radix16_lean`` — the KB[k]=k*b lookahead table +
    # GE-threshold/EQ bank + running remainder + quotient result).  The production
    # divmod allocates the wide nibble_alu32 ``ALU_*`` divmod scratch (1156)
    # AND this compact lean-radix ``LR_*`` datapath (480), and the ``LR_*`` band is
    # touched ONLY by the ``lean-*`` divmod blocks (verified: nothing else reads or
    # writes it).  When ``attn_cam_div`` swaps the whole radix-16 lean-radix division
    # for the R256E SRT pipeline, the ``lean-*`` blocks LEAVE, so the ``LR_*`` residual
    # (like the ``ALU_*`` divmod scratch) is DEAD and its dims free.  Measured live.
    lr_dup = sum(w for k, (off, w) in names.items() if k.startswith("LR_"))

    # inline-ALU max FFN intermediate with the radix-16 lean-kb estimate FFN gone.
    specs = _block_specs(QL.L, code_size, sub, efficient_alu=True,
                         recurrent_divmod=recurrent_divmod)
    leankb_blocks = [n for n, _ in specs if n.startswith("lean-kb")]
    non_kb = [int(s["W_up"].shape[0]) for n, s in specs if not n.startswith("lean-kb")]
    inter_without_leankb = max(non_kb) if non_kb else 0

    # the #908 R256E replacement geometry (residual + depth + widest FFN), read
    # live from the on-branch div_radix256_est SRT pipeline (byte-exact validated).
    from . import div_radix256_est as _E
    L2 = _E._new_layout()
    a = L2.R256E
    RN, NLANE, NITERS = _E._RN, _E._NLANE, _E._NITERS
    r256e_residual = (32 + 9 + RN + RN + 1 + 1 + RN + RN + RN + 1 + 1 + RN + RN
                      + NLANE * RN + NLANE * RN + NLANE + NLANE + NLANE + NLANE
                      + NLANE * RN + NLANE * RN + 1 + 8 + 8 + 8 + 1 + NITERS)
    unique, apply_names = _E.compile_blocks_recurrent(L2, L2.D)
    r256e_ffn_max = max((int(s["W_up"].shape[0]) for n, s in unique
                         if isinstance(s, dict) and "W_up" in s), default=0)
    est_units = next((int(s["W_up"].shape[0]) for n, s in unique
                      if n == "r256e-est" and isinstance(s, dict) and "W_up" in s), 0)
    kv_rows = 255                        # R-1 shared GE-threshold rows (radix 256)

    return {
        "divmod_residual_before": divmod_residual_before,
        "divmod_bank": bank,
        "divmod_running": running,
        "lr_dup_residual": lr_dup,                     # the duplicate lean-radix LR_* datapath
        "divmod_residual_after": r256e_residual,      # 277 (the R256E running state)
        "kv_rows": kv_rows,
        "est_units_replaced": est_units,              # the FFN GE bank -> KV rows
        "r256e_stored": len(unique),
        "r256e_applied": len(apply_names),
        "r256e_ffn_max": r256e_ffn_max,
        "inter_without_leankb": inter_without_leankb,
        "leankb_blocks": len(leankb_blocks),
    }


def _spec_sizes(config: FitConfig,
                subset: Optional[Subset] = None) -> Tuple[int, int, int, int]:
    """(hidden, intermediate, stored_layers, applied_depth) at the config's op-subset
    + efficient/recurrent flags (32-bit, the baked path), arch-rescaled.  ``subset``
    overrides ``config.subset`` (used by the subroutine lever, which accounts the
    stack WITHOUT muldiv baked in).

    The #913 fit levers (``pack_memcam`` / ``overlap_scratch`` / ``bit_level_bitwise``)
    apply MEASURED reductions to ``d_used`` (and ``bit_level_bitwise`` to the depth):
    each is a delta computed from the real layout, so the reported geometry is one the
    build path could realize.

    The #916 fit lever (``attn_cam_div``) is the INLINE DIV/MOD one: it swaps the wide
    radix-16 GE-threshold BANK for the #908 attention-CAM head (255 shared KV rows), so
    the divmod residual scratch drops 1165 -> 277, the lean-kb estimate FFN (7920) leaves
    the intermediate, and the radix-16 long-division blocks are replaced by the R256E SRT
    pipeline's blocks — all MEASURED from the on-branch layouts."""
    sub = subset if subset is not None else config.subset
    eff = config.efficient_alu and (subset is None)   # subroutine drops the baked ALU
    rec = config.recurrent_divmod and (subset is None)
    d_used, inter, n_stored, n_applied = _raw_spec_sizes(
        config.code_size, sub.memory, sub.cmp, sub.bitwise, sub.muldiv, eff, rec)

    # ---- #913 measured residual/depth reductions -----------------------------
    if config.pack_memcam:
        d_used -= _packmemcam_saving(config.code_size, sub.memory, sub.cmp,
                                     sub.bitwise, sub.muldiv, eff, rec)
    if config.overlap_scratch and eff:
        total, overlapped, _ = _alu_scratch_overlap(config.code_size, rec)
        d_used -= max(0, total - overlapped)
    if config.bit_level_bitwise and sub.bitwise:
        blocks_removed, res_removed = _bitwise_barrel_saving(
            config.code_size, config.pack_memcam)
        d_used -= res_removed
        n_stored -= blocks_removed
        n_applied -= blocks_removed
    if config.bit_level_shift and sub.bitwise and eff:
        # SHL/SHR via native MUL/DIV (shift_via_mul): the barrel-shifter RESIDUAL
        # (SH_STAGE_* + TS_*, 278) is allocated but dead — retire it.  Depth is
        # UNCHANGED (the 17 tshift blocks are already gone in the shift_via_mul base;
        # the +2 alu-shift-onehot/pow2 blocks are already counted in n_stored).
        d_used -= _bitlevel_shift_saving(config.code_size, config.pack_memcam)

    # ---- #916 INLINE attention-CAM DIV/MOD lever (only for a baked inline ALU) --
    #   The divmod family's residual scratch drops from its production width to the
    #   #908 R256E running-state width (1165 -> 277: -888), the radix-16 lean-kb
    #   estimate FFN (7920) leaves the intermediate, and the radix-16 long-division
    #   depth is replaced by the R256E SRT pipeline's depth.  All MEASURED live.
    if config.attn_cam_div and eff and (sub.muldiv):
        g = _attn_cam_div_geometry(config.code_size, rec)
        # RESIDUAL: divmod scratch was the MAX-over-families term (overlap) and a SUM
        #   summand (non-overlap); either way the divmod family shrinks by the same
        #   (before - after) delta (277 stays the widest op-family, so it is still the
        #   overlap MAX) -> a single measured d_used reduction.
        d_used -= max(0, g["divmod_residual_before"] - g["divmod_residual_after"])
        # RESIDUAL (2nd term): the DUPLICATE radix-16 lean-radix datapath (``LR_*``,
        #   480) is a SEPARATE always-summed residual band (NOT an ``ALU_*`` overlap
        #   family), touched ONLY by the ``lean-*`` blocks that this lever removes.  Once
        #   the R256E pipeline replaces the whole radix-16 division, ``LR_*`` is dead, so
        #   its dims free.  (The R256E running state is already accounted above as the
        #   277 that replaces the ``ALU_*`` divmod scratch — the divmod family is 277 TOTAL,
        #   so both the ``ALU_*`` 1156 scratch AND this ``LR_*`` 480 datapath leave.)
        d_used -= g.get("lr_dup_residual", 0)
        # INTERMEDIATE: the radix-16 lean-kb GE bank (7920) leaves FFN -> the next
        #   widest inline block, BUT the radix-256 R256E SRT keeps its own widest FFN
        #   (the qhat*divisor lane multiply, r256e_ffn_max) — the honest intermediate is
        #   the wider of the two (the radix-256 estimate-CAM trades the threshold bank
        #   for a wide per-digit multiply; see the report's intermediate caveat).
        inter = max(g["inter_without_leankb"], g["r256e_ffn_max"])
        # DEPTH: the radix-16 long-division blocks (lean-*) are replaced by the R256E
        #   SRT pipeline blocks.  Recurrent divmod stores the SRT cell once (r256e_stored
        #   == 36); unrolled feed-forward stores every applied step (r256e_applied == 90).
        d_before, r256e = _divmod_block_delta(
            config.code_size, rec, (g["r256e_stored"], g["r256e_applied"]))
        n_stored += r256e - d_before
        n_applied += r256e - d_before
        # DEPTH op-overlap: replace the SUMMED stored depth with the op-overlapped
        #   max-op depth (shared base pipeline + deepest single op-family cascade),
        #   since only one opcode fires per step and the machinery is opcode-gated onto
        #   shared physical layers.  MEASURED live.  ``applied`` (per-forward) already
        #   equals the deepest op cascade, so it is set to the same max-op depth.
        if config.overlap_depth:
            depth, _fam = _maxop_overlap_depth(
                config.code_size, g["r256e_applied"], g["r256e_stored"],
                rec, config.bit_level_shift)
            n_stored = depth
            n_applied = depth

    arch = config.arch
    inter = max(inter, arch.num_attention_heads * arch.head_dim, 8)
    hidden = arch.hidden_for(d_used + 1)
    return hidden, inter, max(1, n_stored), max(1, n_applied)


@lru_cache(maxsize=None)
def _divmod_block_delta(code_size: int, recurrent_divmod: bool, g_key) -> Tuple[int, int]:
    """(radix16_divmod_blocks, r256e_replacement_blocks) — the #stored/applied block
    swap for the attention-CAM DIV/MOD lever.  ``radix16_divmod_blocks`` is the count
    of the production nibble long-division blocks (``lean-*``, MEASURED live), replaced
    by the R256E SRT pipeline (recurrent -> its 36 unique cells; unrolled -> its 90
    applied steps).  ``g_key`` is the #908 geometry dict's (stored, applied) tuple so the
    lru_cache stays hashable (dicts aren't)."""
    from . import qwen_full_vm as _Q
    sub = Subset(memory=True, cmp=True, bitwise=True, muldiv=True)
    QL = _Q.QwenFullLayout(code_size, sub, efficient_alu=True,
                           recurrent_divmod=recurrent_divmod)
    specs = _block_specs(QL.L, code_size, sub, efficient_alu=True,
                         recurrent_divmod=recurrent_divmod)
    radix16_divmod_blocks = sum(1 for n, _ in specs if n.startswith("lean-"))
    r256e_stored, r256e_applied = g_key
    r256e_blocks = r256e_stored if recurrent_divmod else r256e_applied
    return radix16_divmod_blocks, r256e_blocks


# The op families whose block PREFIXES partition the inline-ALU block stack (op-overlap:
# only ONE opcode fires per step, so per-family private cascades SHARE the same physical
# layers, gated on the opcode — the feed-forward network's depth is the SHARED base
# pipeline + the DEEPEST single op-family cascade, NOT the SUM of all families).
def _block_family(name: str) -> str:
    if name.startswith("lean-"):
        return "divmod"                       # replaced by R256E when attn_cam_div on
    if "tshift" in name:
        return "barrel"                       # dead when shift_via_mul on
    if name.startswith("alu-mul"):
        return "mul"
    if name.startswith("alu-shift"):
        return "shift"
    if "bitwise" in name or "bw-" in name:
        return "bitwise"
    if name.startswith("alu-"):
        return "shared-alu"                   # expand/psh-nib: shared by mul/div/shift
    if "cmp" in name:
        return "cmp"
    if "mem" in name.lower() or "load" in name.lower():
        return "memory"
    return "base"                             # fetch/decode/frame/writeback (shared)


@lru_cache(maxsize=None)
def _maxop_overlap_depth(code_size: int, r256e_unrolled: int, r256e_recurrent: int,
                         divmod_recurrent: bool, bit_level_shift: bool) -> Tuple[int, dict]:
    """The OP-OVERLAP max-op stored depth for the inline feed-forward ALU: the SHARED
    pipeline (``base`` + ``shared-alu``) + the DEEPEST single op-family private cascade.
    Because exactly one opcode fires per step, the per-op-family machinery is gated on
    the opcode and shares the SAME physical layers, so the network depth is the max-op
    cascade, NOT the SUM of all families.  Read live from the block stack; the divmod
    family is the R256E pipeline (unrolled or recurrent).  Returns (depth, per_family)."""
    from . import qwen_full_vm as _Q
    sub = Subset(memory=True, cmp=True, bitwise=True, muldiv=True)
    QL = _Q.QwenFullLayout(code_size, sub, efficient_alu=True,
                           recurrent_divmod=False, shift_via_mul=True)
    specs = _block_specs(QL.L, code_size, sub, efficient_alu=True,
                         recurrent_divmod=False, shift_via_mul=True)
    from collections import defaultdict
    fam: Dict[str, int] = defaultdict(int)
    for n, _s in specs:
        f = _block_family(n)
        if f == "divmod":
            continue                          # divmod is the R256E replacement below
        if f == "barrel" and bit_level_shift:
            continue                          # dead barrel-shift blocks (shift-via-mul)
        fam[f] += 1
    # divmod family = R256E (unrolled feed-forward, or recurrent tied-loop).
    fam["divmod"] = r256e_recurrent if divmod_recurrent else r256e_unrolled
    shared = fam.get("base", 0) + fam.get("shared-alu", 0)
    op_private = {k: v for k, v in fam.items() if k not in ("base", "shared-alu", "barrel")}
    max_op = max(op_private.values()) if op_private else 0
    return shared + max_op, dict(fam)


def account(config: FitConfig) -> Accounting:
    """Full accounting for ``config`` (CPU, no model built)."""
    arch = config.arch
    verified = True
    notes = ""
    steps_per_op = 1                     # one Qwen forward per op on the persistent stack

    if config.subroutine_muldiv:
        # SUBROUTINE lever: MUL/DIV/MOD leave the persistent stack and run as a baked
        # bytecode library routine — many PROGRAM STEPS per op, near-zero extra
        # persistent depth/width.  Re-account WITHOUT muldiv baked into the stack.
        sub_no_md = _drop_muldiv(config.subset)
        hidden, inter, n_stored, n_applied = _spec_sizes(config, subset=sub_no_md)
        steps_per_op = _est_subroutine_steps(config.precision,
                                             divmod_=config.has_divmod)
        verified = False
        notes = _SUBROUTINE_NOTE
    else:
        hidden, inter, n_stored, n_applied = _spec_sizes(config)

    # ---- PRECISION scaling.  The baked ALU is 32-bit; other precisions are a
    #      linear-in-nibble PROJECTION of the DIV/MOD iteration depth.
    if config.efficient_alu and config.precision != 32 and config.has_divmod:
        delta_iters = _div_iters(config.precision) - _DIV_ITERS_32
        n_applied = max(1, n_applied + delta_iters * _DIV_LAYERS_PER_ITER)
        if not config.recurrent_divmod:
            n_stored = max(1, n_stored + delta_iters * _DIV_LAYERS_PER_ITER)
        verified = False
        notes = _join(notes, f"P{config.precision} ALU depth ESTIMATE (branch bakes 32-bit only)")
    elif (config.muldiv_strategy == "lookup-table" and config.has_muldiv
          and config.precision != 8):
        # the lookup table is inherently 8-bit (256×256); 16/32-bit is unbuildable.
        verified = False
        notes = _join(notes, f"P{config.precision} lookup table UNBUILDABLE (256^{config.precision//8} domain)")

    params = _param_estimate(hidden, inter, n_stored, arch)
    return Accounting(
        config=config, hidden=hidden, intermediate=inter, stored_layers=n_stored,
        applied_depth=n_applied, steps_per_op=steps_per_op,
        query_heads=arch.num_attention_heads, verified=verified,
        params_estimate=params, notes=notes)


def _drop_muldiv(sub: Subset) -> Subset:
    return Subset(memory=sub.memory, cmp=sub.cmp, bitwise=sub.bitwise, muldiv=False,
                  name=sub.name.replace("+muldiv", "").strip("+") or "base")


def _join(a: str, b: str) -> str:
    return f"{a}; {b}" if a else b


def _param_estimate(hidden: int, inter: int, n_layers: int, arch: QwenArch) -> int:
    """Rough dense fp32 param count for the smallest-params ranking (Qwen2 layer:
    3 MLP mats hidden*inter + q/k/v/o attn + 2 RMSNorm + embed/lm-head)."""
    hd = arch.head_dim
    q = hidden * arch.num_attention_heads * hd
    kv = hidden * arch.num_key_value_heads * hd * 2
    o = arch.num_attention_heads * hd * hidden
    attn = q + kv + o
    mlp = 3 * hidden * inter
    per_layer = attn + mlp + 2 * hidden
    embed = hidden * Q.V.VOCAB * 2       # embed + lm head (untied)
    return per_layer * n_layers + embed


# ===========================================================================
# Constraint box + feasibility.
# ===========================================================================
@dataclass
class ConstraintBox:
    max_depth: Optional[int] = None          # num_hidden_layers (STORED)
    max_applied_depth: Optional[int] = None  # layers run per forward
    max_hidden: Optional[int] = None
    max_intermediate: Optional[int] = None
    max_precision: Optional[int] = None
    max_steps_per_op: Optional[int] = None
    target: Optional[str] = None             # a named stock target
    require_buildable: bool = False          # only accept VERIFIED-buildable configs

    def resolved_arch(self) -> QwenArch:
        if self.target:
            return STOCK_TARGETS[_canon_target(self.target)].arch
        return QWEN2_5_ARCH

    def effective(self) -> "ConstraintBox":
        """Fold a named ``target`` into concrete depth/width/precision caps."""
        if not self.target:
            return self
        st = STOCK_TARGETS[_canon_target(self.target)]
        return ConstraintBox(
            max_depth=_min_opt(self.max_depth, st.layers),
            max_applied_depth=self.max_applied_depth,
            max_hidden=_min_opt(self.max_hidden, st.hidden),
            max_intermediate=_min_opt(self.max_intermediate, st.intermediate),
            max_precision=self.max_precision, max_steps_per_op=self.max_steps_per_op,
            target=self.target, require_buildable=self.require_buildable)


def _canon_target(name: str) -> str:
    if name not in STOCK_TARGETS:
        raise KeyError(f"unknown target {name!r}; known: {sorted(STOCK_TARGETS)}")
    return name


def _min_opt(a, b):
    return b if a is None else min(a, b)


def violations(acc: Accounting, box: ConstraintBox) -> List[Tuple[str, int, int]]:
    """The list of VIOLATED axes (empty == fits)."""
    box = box.effective()
    v: List[Tuple[str, int, int]] = []
    if box.max_depth is not None and acc.stored_layers > box.max_depth:
        v.append(("depth (stored layers)", acc.stored_layers, box.max_depth))
    if box.max_applied_depth is not None and acc.applied_depth > box.max_applied_depth:
        v.append(("applied depth", acc.applied_depth, box.max_applied_depth))
    if box.max_hidden is not None and acc.hidden > box.max_hidden:
        v.append(("hidden", acc.hidden, box.max_hidden))
    if box.max_intermediate is not None and acc.intermediate > box.max_intermediate:
        v.append(("intermediate", acc.intermediate, box.max_intermediate))
    if box.max_precision is not None and acc.config.precision > box.max_precision:
        v.append(("precision", acc.config.precision, box.max_precision))
    if box.max_steps_per_op is not None and acc.steps_per_op > box.max_steps_per_op:
        v.append(("steps_per_op", acc.steps_per_op, box.max_steps_per_op))
    if box.require_buildable and not acc.verified:
        v.append(("buildable", 0, 1))
    return v


# ===========================================================================
# SOLVER — enumerate feasible configs, rank by objective.
# ===========================================================================
OBJECTIVES = ("depth", "width", "steps", "params", "fits-named")


def _candidate_strategies(ops) -> List[str]:
    if not (set(ops) & MULDIV_OPS):
        return ["n/a"]
    return list(MULDIV_STRATEGIES)


def _candidate_precisions(box: ConstraintBox) -> List[int]:
    box = box.effective()
    ps = [p for p in PRECISIONS if (box.max_precision is None or p <= box.max_precision)]
    return ps or [min(PRECISIONS)]


def enumerate_configs(ops, box: ConstraintBox, code_size: int = 24) -> List[FitConfig]:
    ops = frozenset(ops)
    arch = box.resolved_arch()
    cfgs = []
    for strat in _candidate_strategies(ops):
        for prec in _candidate_precisions(box):
            cfgs.append(FitConfig(
                ops=ops,
                muldiv_strategy=(strat if strat != "n/a" else "efficient-ALU-recurrent"),
                precision=prec, granularity="nibble", code_size=code_size, arch=arch))
    return cfgs


def _objective_key(acc: Accounting, objective: str, box: ConstraintBox):
    """Sort key (ascending == better)."""
    if objective == "depth":
        return (acc.stored_layers, acc.intermediate, acc.steps_per_op, acc.params_estimate)
    if objective == "width":
        return (acc.intermediate, acc.stored_layers, acc.steps_per_op, acc.params_estimate)
    if objective == "steps":
        return (acc.steps_per_op, acc.stored_layers, acc.intermediate, acc.params_estimate)
    if objective == "params":
        return (acc.params_estimate, acc.stored_layers, acc.intermediate)
    if objective == "fits-named":
        st = STOCK_TARGETS[_canon_target(box.target)] if box.target else None
        slack = 0
        if st:
            slack = -min(st.layers - acc.stored_layers, st.hidden - acc.hidden,
                         st.intermediate - acc.intermediate)
        return (slack, acc.params_estimate)
    raise ValueError(objective)


@dataclass
class SolveResult:
    best: Optional[Accounting]
    feasible: List[Accounting]                            # all feasible, ranked
    infeasible: List[Tuple[Accounting, List[Tuple[str, int, int]]]]
    objective: str
    box: ConstraintBox
    binding: Optional[str] = None                         # constraint blocking all configs
    relaxation: Optional[str] = None                      # closest relaxation to feasibility

    @property
    def ok(self) -> bool:
        return self.best is not None


def solve(ops, box: ConstraintBox, objective: str = "depth",
          code_size: int = 24) -> SolveResult:
    """Enumerate feasible configs for ``ops`` in ``box``, rank by ``objective``.

    Returns the best fit + the ranked feasible table; if INFEASIBLE, the binding
    constraint + the closest relaxation."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective {objective!r} not in {OBJECTIVES}")
    cfgs = enumerate_configs(ops, box, code_size=code_size)
    feasible, infeasible = [], []
    for cfg in cfgs:
        acc = account(cfg)
        vv = violations(acc, box)
        (infeasible.append((acc, vv)) if vv else feasible.append(acc))
    feasible.sort(key=lambda a: _objective_key(a, objective, box))
    best = feasible[0] if feasible else None
    binding, relaxation = (None, None)
    if best is None and infeasible:
        binding, relaxation = _binding_and_relaxation(infeasible, box)
    return SolveResult(best=best, feasible=feasible, infeasible=infeasible,
                       objective=objective, box=box, binding=binding, relaxation=relaxation)


def _binding_and_relaxation(infeasible, box: ConstraintBox):
    """The axis that blocks the MOST configs is the binding constraint; the closest
    relaxation is the smallest cap bump that makes the least-overshooting config fit."""
    from collections import Counter
    axis_counts = Counter()
    for _, vv in infeasible:
        for axis, _got, _cap in vv:
            axis_counts[axis] += 1
    binding = axis_counts.most_common(1)[0][0] if axis_counts else None

    def overshoot(item):
        acc, vv = item
        b = next((g - c for a, g, c in vv if a == binding), 0)
        return (len(vv), b)
    acc, vv = min(infeasible, key=overshoot)
    parts = [f"{axis}: need <= {got} (cap {cap})" for axis, got, cap in vv]
    relaxation = f"closest config {acc.config.label}: relax " + "; ".join(parts)
    return binding, relaxation


# ===========================================================================
# FORMATTING — a tradeoff table + a one-line best summary.
# ===========================================================================
def tradeoff_table(result: SolveResult, show: int = 12) -> str:
    rows = result.feasible[:show]
    if not rows and result.infeasible:
        rows = [a for a, _ in sorted(result.infeasible, key=lambda it: len(it[1]))][:show]
    hdr = (f"{'config':52s} {'hidden':>6s} {'inter':>7s} {'stored':>6s} "
           f"{'applied':>7s} {'stp/op':>6s} {'params':>8s} {'ok':>3s} {'vfd':>3s}")
    lines = [hdr, "-" * len(hdr)]
    feas_set = {id(a) for a in result.feasible}
    for acc in rows:
        ok = "yes" if id(acc) in feas_set else "no"
        vfd = "V" if acc.verified else "E"
        lines.append(
            f"{acc.config.label:52s} {acc.hidden:6d} {acc.intermediate:7d} "
            f"{acc.stored_layers:6d} {acc.applied_depth:7d} {acc.steps_per_op:6d} "
            f"{_h(acc.params_estimate):>8s} {ok:>3s} {vfd:>3s}")
    if result.best is None and result.binding:
        lines += ["", f"INFEASIBLE — binding constraint: {result.binding}",
                  f"  {result.relaxation}"]
    return "\n".join(lines)


def _h(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n/div:.1f}{unit}"
    return str(n)


def best_summary(result: SolveResult) -> str:
    if result.best is None:
        return (f"INFEASIBLE for objective={result.objective}. "
                f"Binding: {result.binding}. {result.relaxation}")
    a = result.best
    return (f"BEST (min {result.objective}): {a.config.label} -> "
            f"hidden={a.hidden} intermediate={a.intermediate} stored_layers={a.stored_layers} "
            f"applied_depth={a.applied_depth} steps/op={a.steps_per_op} params~{_h(a.params_estimate)} "
            f"[{'VERIFIED-buildable' if a.verified else 'ESTIMATED'}]"
            + (f" — {a.notes}" if a.notes else ""))


# ===========================================================================
# Top-level fit() — the one-call configurator front door.
# ===========================================================================
def fit(target: Optional[str] = None, *, ops=FULL, max_depth: Optional[int] = None,
        max_width: Optional[int] = None, max_hidden: Optional[int] = None,
        max_intermediate: Optional[int] = None, precision: Optional[int] = None,
        max_steps_per_op: Optional[int] = None, minimize: str = "depth",
        require_buildable: bool = False, code_size: int = 24) -> SolveResult:
    """Configurator front door.

    ``target``  — a named stock ('stock-0.5b'/'stock-1.5b'/'stock-7b') fixes the
                  depth/width/head budget as the box.
    ``ops``     — the op-set to COVER (default FULL).
    ``max_*``   — explicit caps (intersected with the named target if given).
    ``max_width`` — caps BOTH hidden and intermediate (the brief's ``max_width``).
    ``precision`` — max precision bits (8/16/32).
    ``minimize`` — objective: 'depth' | 'width' | 'steps' | 'params' | 'fits-named'.
    """
    mh = max_hidden if max_hidden is not None else max_width
    mi = max_intermediate if max_intermediate is not None else max_width
    box = ConstraintBox(
        max_depth=max_depth, max_hidden=mh, max_intermediate=mi,
        max_precision=precision, max_steps_per_op=max_steps_per_op,
        target=target, require_buildable=require_buildable)
    if minimize == "fits-named" and target is None:
        raise ValueError("minimize='fits-named' needs a target=")
    return solve(ops, box, objective=minimize, code_size=code_size)


# ===========================================================================
# BAKE the chosen config through qwen_full_vm.build, and measure it fits the box.
# ===========================================================================
def bake(config: FitConfig, K: float = Q.NORM_K):
    """Bake ``config`` into a genuine ``Qwen2Model`` via ``qwen_full_vm.build``.

    Only the VERIFIED-buildable configs (efficient-ALU or 8-bit lookup-table at
    32-bit) bake here; the subroutine + non-32-bit-ALU accountings are ESTIMATES and
    raise.  Returns the ``QwenFullVM``."""
    if config.subroutine_muldiv:
        raise NotImplementedError(
            "subroutine muldiv is an ESTIMATE lever (#699 pending reverify); not bakeable here")
    if config.efficient_alu and config.precision != 32:
        raise NotImplementedError(
            f"efficient ALU is baked 32-bit; P{config.precision} is an accounting estimate")
    return Q.build(code_size=config.code_size, subset=config.subset, arch=config.arch,
                   K=K, efficient_alu=config.efficient_alu,
                   recurrent_divmod=config.recurrent_divmod)


def measure_built(vm) -> Dict[str, int]:
    """Read back the ACTUAL geometry of a baked ``QwenFullVM`` (what the model IS)."""
    cfg = vm.qmodel.config
    return {
        "hidden_size": int(cfg.hidden_size),
        "intermediate_size": int(cfg.intermediate_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),   # == applied (recurrent repoints)
        "stored_layers": int(vm.n_layers),
        "applied_depth": int(vm.n_applied or cfg.num_hidden_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
    }


# ===========================================================================
# OPCONFIG-AWARE GEOMETRY — size ANY per-op {precision, radix, extraction,
# recurrence} config (c4_min.opconfig.OpConfig) into (n_layers, hidden,
# intermediate, stored, applied, fits-stock).  This is the FITTER wiring the
# toggle system needs: the DEFAULT (nibble/fp32) config routes to the REAL
# nibble solver above (byte-exact golden geometry); the non-default (clever
# digit-extract / whole-value) configs are accounted from the completed
# min-param construction (ALL_OPS_MINPARAM.md + CLEVER_MINPARAM_ALU.md), whose
# cost is a NARROW reused cell paid in APPLIED depth, not width.
# ===========================================================================
# Stock Qwen2.5-0.5B budget (the fits-stock gate).
_STOCK_0_5B = STOCK_TARGETS["stock-0.5b"]

# The clever construction's per-op APPLIED depth (digit-extract places / limb
# steps), from the completed docs.  Keyed by (extraction, op-class).  Decimal
# digit-extract: ADD/SUB 11, DIV/MOD 10, MUL 20, CMP/frame 1.  Nibble-serial /
# radix-limb: ~8 base-16 limbs.  The DEEPEST op sets the reused-cell APPLIED
# depth reported (the cost of a single op's forward-unroll).
_CLEVER_DEPTH_DIGIT = {           # decimal MSB-first digit-extract (whole-value)
    "add": 11, "cmp": 1, "div": 10, "mul": 20,
}


def _limb_depth(radix: int, bits: int = 32) -> int:
    """#base-`radix` limbs for a `bits`-bit value (radix-limb / nibble depth)."""
    import math
    if radix < 2:
        return bits
    return max(1, math.ceil(bits / math.log2(radix)))


def _clever_applied_depth(op_class: str, axes) -> int:
    """APPLIED per-op depth (reused-cell unroll count) for one op's axes."""
    if axes.extraction == "whole_value":
        # MSB-first decimal digit-extract; deepest op sets the cell depth.
        return _CLEVER_DEPTH_DIGIT.get(op_class, 1)
    if axes.extraction == "digit_extract":
        # radix-limb decode: #base-`radix` limbs of the RESULT width.
        result_bits = 64 if op_class == "mul" else 32
        return _limb_depth(axes.radix, result_bits)
    # nibble (the production build): the nibble solver owns depth; sentinel 0.
    return 0


# ---------------------------------------------------------------------------
# MACHINERY families for the STANDARD (feed-forward, UNROLLED) accounting.
#
# In a STANDARD feed-forward transformer the FFN stack must CONTAIN every op's
# machinery — the active op is data-dependent (conditionally applied at run time),
# so a layer cannot be re-used across ops the way a LOOPED / Universal-Transformer
# re-applies one cell.  Each op's digit-extraction places are therefore DISTINCT
# stored layers, and the network's n_layers is the SUMMED unrolled depth across
# all the distinct machinery families.
#
# We group the ops into machinery families (finer than the accumulator-bound
# _OP_CLASS, which coarsely maps bitwise/memory/trivial into "add"): each family
# has its own datapath, so each family's unrolled place-depth is stored ONCE.
# Ops WITHIN a family (e.g. ADD and SUB, or the six CMPs) share the family's
# decode structurally, so they are counted once — but the arith / mul / div /
# bitwise / memory / trivial FAMILIES are all genuinely distinct machinery that
# must all be present, so their depths SUM.  This is the honest floor for "the
# feed-forward network contains every op".
#
#   arith  — the whole-value / radix-limb ingest + difference-min digit-extract
#            (ADD/SUB/CMP/SHL/SHR + frame-adds + memory-address adds share it).
#            depth = the arith family's decode places (whole-value 11 / limb 8).
#   div    — long-division digit-extract (DIV/MOD).  depth 10 (whole) / 8 (limb).
#   mul    — the widest: 64-bit product digit-extract.  depth 20 (whole) / 16 (limb).
#   bitwise— a per-nibble 16x16 LUT applied once per nibble (OR/XOR/AND).  depth 8.
#   memory — the shared content-addressed CAM read/write (LI/LC/SI/SC).  depth 1.
#   trivial— register move / stack write / no-op (IMM/PSH/NOP/HALT).  depth 1.
#
# NB the +1 finalize/writeback per family is folded into the place-depths (the
# doc numbers already include ingest+finalize), so the family depth IS the stored
# layer count that family contributes.
_MACHINERY_FAMILY: Dict[str, str] = {}     # op -> machinery family
for _o in ("ADD", "SUB"):
    _MACHINERY_FAMILY[_o] = "arith"
for _o in ("EQ", "NE", "LT", "GT", "LE", "GE"):
    _MACHINERY_FAMILY[_o] = "arith"        # CMP is a sign read on the shared decode
for _o in ("SHL", "SHR"):
    _MACHINERY_FAMILY[_o] = "arith"
for _o in ("LEA", "JMP", "JSR", "BZ", "BNZ", "ENT", "ADJ", "LEV"):
    _MACHINERY_FAMILY[_o] = "arith"        # frame ops are address adds on the decode
for _o in ("MUL",):
    _MACHINERY_FAMILY[_o] = "mul"
for _o in ("DIV", "MOD"):
    _MACHINERY_FAMILY[_o] = "div"
for _o in ("OR", "XOR", "AND"):
    _MACHINERY_FAMILY[_o] = "bitwise"
for _o in ("LI", "LC", "SI", "SC"):
    _MACHINERY_FAMILY[_o] = "memory"
for _o in ("IMM", "PSH", "NOP", "HALT"):
    _MACHINERY_FAMILY[_o] = "trivial"


def _family_unrolled_depth(family: str, axes) -> int:
    """The #DISTINCT stored layers one machinery FAMILY contributes to a STANDARD
    (unrolled) feed-forward network, at the family's clever axes.

    arith / div / mul: the digit-extract place count (whole-value decimal or
    radix-limb).  bitwise: the per-nibble LUT depth (8).  memory: the shared CAM
    read/write (1).  trivial: a register move (1)."""
    if family == "arith":
        return _clever_applied_depth("add", axes)
    if family == "div":
        return _clever_applied_depth("div", axes)
    if family == "mul":
        return _clever_applied_depth("mul", axes)
    if family == "bitwise":
        # OR/XOR/AND: a 16x16 nibble LUT applied once per nibble; depth 8 (32-bit).
        return 8
    if family == "memory":
        return 1                            # one shared CAM touch
    if family == "trivial":
        return 1                            # register move / stack write
    return 1


def summed_unrolled_depth(config) -> Tuple[int, Dict[str, int]]:
    """The STANDARD feed-forward transformer's n_layers for a clever ``OpConfig``:
    the SUMMED unrolled depth across all DISTINCT machinery families (each family's
    digit-extraction places are stored distinctly — the active op is data-dependent,
    so the network must contain every op's machinery, and no cell is re-used the way
    a LOOPED model re-applies one).  Returns ``(total, per_family_depth)``."""
    from . import opconfig as OC
    per_family: Dict[str, int] = {}
    for op in OC.ALL_OPS:
        fam = _MACHINERY_FAMILY[op]
        if fam in per_family:
            continue                        # count each family ONCE (deepest member)
        per_family[fam] = _family_unrolled_depth(fam, config.for_op(op))
    total = sum(per_family.values())
    return total, per_family


@dataclass
class OpConfigGeometry:
    """The geometry an ``OpConfig`` implies, reported for the fitter table."""
    label: str
    precision_set: str              # e.g. "fp64/fp128" or "fp32(nibble)"
    extraction: str
    recurrence: str
    hidden: int
    intermediate: int
    stored_layers: int              # distinct stored layers (STANDARD: n_layers;
                                    #   LOOPED: the few reused cells)
    applied_depth: int              # deepest per-op unroll (per forward)
    params_estimate: int
    fits_stock_0_5b: bool
    binding: str                    # "" if fits, else the binding axis
    looped_transformer: bool = False  # model MODE (True == UT / looped)
    model_mode: str = "standard-feedforward"  # human label for the mode
    note: str = ""


def account_opconfig(config, code_size: int = 24, arch: QwenArch = QWEN2_5_ARCH
                     ) -> OpConfigGeometry:
    """Size an ``opconfig.OpConfig`` into a geometry row, HONESTLY per model mode.

    The critical distinction (see the module docstring + TOGGLE_SCHEMA.md §honesty):

    * DEFAULT (nibble/fp32/unrolled): routes to the REAL nibble solver's FULL
      geometry (the golden build width/depth) — hidden ~3008, does NOT fit 0.5B.

    * clever, ``looped_transformer=False`` (STANDARD feed-forward): weight-TIED
      recurrence is ILLEGAL for a standard transformer, so the tied axis is
      DOWNGRADED to unrolled and the network must CONTAIN every op's machinery as
      DISTINCT stored layers.  ``stored_layers`` = the SUMMED unrolled depth across
      all machinery families (``summed_unrolled_depth``); it is narrow + shallow
      vs nibble (digit-extract << the 189-block nibble long-division) but its
      ~tens-to-hundreds of DISTINCT layers STILL exceed stock 0.5B's 24 → fits
      WIDTH, does NOT fit DEPTH → does NOT fit stock 0.5B as a standard transformer.

    * clever, ``looped_transformer=True`` (LOOPED / Universal-Transformer): the few
      reused cells are STORED (~6) and re-applied ``depth`` times.  This fits a
      0.5B-WIDTH checkpoint, but as a DIFFERENT (UT) architecture — NOT stock
      feed-forward Qwen2.  Labelled Universal-Transformer.
    """
    from . import opconfig as OC

    # --- DEFAULT (nibble) path: the REAL nibble solver FULL geometry ---------
    if config.is_default():
        acc = account(FitConfig(ops=FULL, muldiv_strategy="efficient-ALU-unrolled",
                                precision=32, code_size=code_size, arch=arch))
        fits = (acc.hidden <= _STOCK_0_5B.hidden
                and acc.intermediate <= _STOCK_0_5B.intermediate
                and acc.stored_layers <= _STOCK_0_5B.layers)
        binding = ""
        if not fits:
            if acc.hidden > _STOCK_0_5B.hidden:
                binding = "hidden"
            elif acc.stored_layers > _STOCK_0_5B.layers:
                binding = "depth (stored layers)"
            else:
                binding = "intermediate"
        return OpConfigGeometry(
            label="nibble-fp32-FULL (DEFAULT / golden 174ece66)",
            precision_set="fp32 (nibble 4-bit lanes)", extraction="nibble",
            recurrence="unrolled", hidden=acc.hidden, intermediate=acc.intermediate,
            stored_layers=acc.stored_layers, applied_depth=acc.applied_depth,
            params_estimate=acc.params_estimate, fits_stock_0_5b=fits,
            binding=binding, looped_transformer=False,
            model_mode="standard-feedforward",
            note="the production nibble build (routes through the real fit solver)")

    # --- CLEVER path -----------------------------------------------------------
    # hidden floored to the Qwen GQA head partition (14 q-heads x 64 = 896); the
    # clever cell's residual is a handful of dims (value axis + flags + work lanes).
    HEAD_DIM = arch.head_dim
    QHEADS = arch.num_attention_heads
    hidden_floor = QHEADS * HEAD_DIM
    d_used = 4 + 8                                   # value/flag axis + ~8 work lanes
    hidden = max(hidden_floor, -(-(d_used + 1) // HEAD_DIM) * HEAD_DIM)

    # Precision set across the op-set + deepest single-op applied depth + decode fan.
    precs = set()
    max_applied = 1
    max_decode_fan = 1
    extraction = config.base.extraction
    for op in OC.ALL_OPS:
        ax = config.for_op(op)
        cls = OC._OP_CLASS[op]
        precs.add(ax.precision)
        max_applied = max(max_applied, _clever_applied_depth(cls, ax))
        fan = ax.radix if ax.extraction == "digit_extract" else 10
        max_decode_fan = max(max_decode_fan, fan)
    # bitwise LUT rides alongside (256-unit FFN table); floor the intermediate to
    # the head partition, cap by the widest block (bitwise 256 or the decode fan).
    intermediate = max(max_decode_fan, 256, QHEADS * HEAD_DIM, 8)
    prec_set = "/".join(sorted(precs, key=lambda p: PRECISIONS_ORDER.index(p)))

    if config.looped_transformer:
        # ===== LOOPED / Universal-Transformer: few STORED cells, applied N ======
        recurrence = "tied"
        stored_cells = 6        # ingest / ADD·CMP·shift / DIV / MUL / bitwise-LUT / CAM
        # A UT checkpoint fits the stock 0.5B WIDTH+SHAPE, but it is a DIFFERENT
        # architecture (a loop), not stock feed-forward Qwen2.  We record that it
        # fits the checkpoint WIDTH; the honest verdict is labelled UT, not "stock".
        fits = (hidden <= _STOCK_0_5B.hidden
                and intermediate <= _STOCK_0_5B.intermediate
                and stored_cells <= _STOCK_0_5B.layers)
        binding = "" if fits else "hidden"
        params = _param_estimate(hidden, intermediate, stored_cells, arch)
        note = (f"LOOPED / Universal-Transformer (NOT stock feed-forward Qwen2): "
                f"{stored_cells} reused cells re-applied per forward (deepest single "
                f"op {max_applied}). Fits a 0.5B-WIDTH UT checkpoint (hidden {hidden}"
                f"<=896, inter {intermediate}<=4864, stored {stored_cells}<=24), but "
                f"the STANDARD feed-forward version must UNROLL and does NOT fit.")
        return OpConfigGeometry(
            label=_opcfg_label(config, prec_set, looped=True),
            precision_set=prec_set, extraction=extraction, recurrence=recurrence,
            hidden=hidden, intermediate=intermediate, stored_layers=stored_cells,
            applied_depth=max_applied, params_estimate=params,
            fits_stock_0_5b=fits, binding=binding, looped_transformer=True,
            model_mode="LOOPED / Universal-Transformer", note=note)

    # ===== STANDARD feed-forward (unrolled): n_layers = SUMMED unrolled depth =====
    recurrence = "unrolled"
    # tied on a standard model is illegal — force-downgrade so the reported geometry
    # is what a standard transformer would ACTUALLY have to store.
    ff_config = OC.force_standard_feedforward(config)
    n_layers, per_family = summed_unrolled_depth(ff_config)
    stored_layers = n_layers
    # The active op is data-dependent, so ALL family machinery is present, but only
    # ONE op fires per forward → applied depth per forward = the deepest single op.
    applied = max_applied
    fits = (hidden <= _STOCK_0_5B.hidden
            and intermediate <= _STOCK_0_5B.intermediate
            and stored_layers <= _STOCK_0_5B.layers)
    # binding: depth first (it is the expected blocker — tens-to-hundreds > 24).
    if fits:
        binding = ""
    elif stored_layers > _STOCK_0_5B.layers:
        binding = "depth (stored layers)"
    elif hidden > _STOCK_0_5B.hidden:
        binding = "hidden"
    else:
        binding = "intermediate"
    params = _param_estimate(hidden, intermediate, stored_layers, arch)
    fam_str = " + ".join(f"{k} {v}" for k, v in per_family.items())
    note = (f"STANDARD feed-forward (tied illegal -> UNROLLED): the FFN stack must "
            f"CONTAIN every op's machinery as DISTINCT layers. n_layers = summed "
            f"unrolled depth = {fam_str} = {n_layers}. NARROWER + SHALLOWER than "
            f"nibble (hidden {hidden}<=896; digit-extract << the 189-block nibble "
            f"long-division) but {n_layers} distinct layers still EXCEED stock 0.5B's "
            f"24 -> fits WIDTH, NOT DEPTH -> does NOT fit stock 0.5B as a standard "
            f"transformer. Only the LOOPED (UT) variant fits a 0.5B-width checkpoint.")
    return OpConfigGeometry(
        label=_opcfg_label(config, prec_set, looped=False),
        precision_set=prec_set, extraction=extraction, recurrence=recurrence,
        hidden=hidden, intermediate=intermediate, stored_layers=stored_layers,
        applied_depth=applied, params_estimate=params,
        fits_stock_0_5b=fits, binding=binding, looped_transformer=False,
        model_mode="standard-feedforward", note=note)


PRECISIONS_ORDER = ("int8", "fp16", "bf16", "fp32", "fp64", "fp128")


def _opcfg_label(config, prec_set: str, looped: bool = False) -> str:
    base = config.base
    if looped:
        recur = "tied-LOOPED-UT"
    else:
        recur = "unrolled-stdFF"
    return (f"opcfg {prec_set}-{base.extraction}-{recur}-FULL")


def opconfig_geometry_table(configs, code_size: int = 24,
                            arch: QwenArch = QWEN2_5_ARCH) -> str:
    """Render the geometry table for a list of (name, OpConfig) pairs.

    The ``fit0.5B`` column is the HONEST per-mode verdict: a STANDARD feed-forward
    clever config reports its SUMMED unrolled ``stored`` layer count (does NOT fit
    on DEPTH); a LOOPED (UT) config reports its few reused cells (fits a 0.5B-WIDTH
    UT checkpoint, marked ``UT`` — NOT stock feed-forward)."""
    rows = [(name, account_opconfig(cfg, code_size=code_size, arch=arch))
            for name, cfg in configs]
    hdr = (f"{'config':40s} {'mode':>10s} {'prec':>12s} {'extract':>13s} "
           f"{'recur':>16s} {'hidden':>6s} {'inter':>6s} {'stored':>6s} "
           f"{'applied':>7s} {'params':>8s} {'fit0.5B':>10s}")
    lines = [hdr, "-" * len(hdr)]
    for name, g in rows:
        if g.fits_stock_0_5b:
            fit = "UT-width" if g.looped_transformer else "YES"
        else:
            fit = f"no({g.binding.split()[0]})"
        mode = "loop/UT" if g.looped_transformer else "std-FF"
        lines.append(
            f"{name[:40]:40s} {mode:>10s} {g.precision_set[:12]:>12s} "
            f"{g.extraction[:13]:>13s} {g.recurrence[:16]:>16s} {g.hidden:6d} "
            f"{g.intermediate:6d} {g.stored_layers:6d} {g.applied_depth:7d} "
            f"{_h(g.params_estimate):>8s} {fit:>10s}")
    return "\n".join(lines)


# ===========================================================================
# JOINT HARD-CONSTRAINT SOLVER — precision + depth + width + KV-cache budget, all
# at once, wired to opconfig's per-op AxisConfig.
#
# The four axes above (WIDTH / DEPTH / STEPS / PRECISION) size a geometry; this
# section takes a concrete ``opconfig.OpConfig`` (per-op {precision, radix,
# extraction, recurrence}) + a ``FitConstraints`` box and checks EVERY hard
# constraint SIMULTANEOUSLY, reporting the binding constraint (and the axis to
# relax) on infeasibility.  See docs/SOLVER_CONSTRAINTS.md.
# ===========================================================================

# --- KV-cache size.  THE nuance to get right (see docs/SOLVER_CONSTRAINTS.md) ---
#
#   KV_bytes = 2 (K+V) * n_layers_kv * n_heads * head_dim * seq_len * batch
#              * precision_bytes
#
# CRITICAL: the KV cache grows with the number of layer-APPLICATIONS AT INFERENCE,
# NOT the number of DISTINCT STORED cells.  A LOOPED / Universal-Transformer stores
# few cells (~6) but RE-APPLIES them ``applied_depth`` times per forward; each
# application writes its OWN K and V into the cache (the loop UNROLLS into the
# cache at run time).  So:
#
#     n_layers_kv = APPLIED depth   (in BOTH standard-FF and looped/UT modes)
#
# The stored-param reduction from looping does NOT reduce KV.  Weight-tying shrinks
# the PARAMETER footprint (fewer distinct cells) but the KV footprint is set by how
# many times a cell is APPLIED, which is unchanged.  This is why a looped and an
# unrolled model with the same applied depth pay IDENTICAL KV but different stored
# params (test_looped_vs_unrolled_same_kv_diff_params).
#
# n_heads defaults to the arch's KEY-VALUE head count (GQA: Qwen2.5-0.5B caches 2 KV
# heads x 64 dim, NOT the 14 query heads) — the honest KV-cache head count — but is
# overridable in FitConstraints for a non-GQA accounting.


def kv_cache_bytes(n_layers_kv: int, n_heads: int, head_dim: int, seq_len: int,
                   batch: int, precision: str) -> int:
    """KV-cache size in bytes.

    ``KV = 2 (K+V) * n_layers_kv * n_heads * head_dim * seq_len * batch *
    precision_bytes``.  ``n_layers_kv`` MUST be the APPLIED depth (layer
    applications at inference), NOT the stored-cell count — a looped/UT model
    unrolls into the cache and pays KV for its applied depth (see the module
    header + docs/SOLVER_CONSTRAINTS.md)."""
    from . import opconfig as OC
    pb = OC.precision_bytes(precision)
    return 2 * n_layers_kv * n_heads * head_dim * seq_len * batch * pb


@dataclass
class FitConstraints:
    """A JOINT hard-constraint box: precision + depth + width + KV-cache budget.

    All caps are optional (``None`` == no cap on that axis).  ``kv_budget_bytes``
    with ``seq_len`` / ``batch`` bounds the KV-cache footprint; ``n_heads`` /
    ``head_dim`` default to the arch's KEY-VALUE head geometry (the honest GQA KV
    head count) when left ``None``.

      * ``precision``       — the REQUIRED datapath precision (int8/fp16/bf16/fp32/
                              fp64/fp128).  Used both to VALIDATE the config's radix
                              (via opconfig.max_safe_radix — too-low precision at a
                              given radix is an INVALID-RADIX bind) and to size KV
                              bytes/elem.  ``None`` == accept the config's own per-op
                              precisions.
      * ``max_layers``      — cap on n_layers (STORED distinct decoder layers).
      * ``max_hidden``      — cap on hidden_size.
      * ``max_intermediate``— cap on intermediate_size (FFN width).
      * ``kv_budget_bytes`` — cap on the KV-cache footprint in bytes.
      * ``seq_len`` / ``batch`` — the KV-cache sizing context (default 2048 / 1).
      * ``n_heads`` / ``head_dim`` — KV-cache head geometry override (default from
                              the arch's num_key_value_heads / head_dim).
      * ``forwards_per_step`` — the DEPTH lever: run the SAME network ``F`` forward
                              passes per VM step (the autoregressive digit-per-token
                              loop) so the required depth D is met by only
                              ``ceil(D/F)`` STORED physical layers.  KV-neutral here
                              (applied depth spread over F forwards => seq*F, and
                              (D/F)*(seq*F) == D*seq).  Default 1 == distinct-layers.
                              See ``forwards_per_step.py`` for the full (L,F) model.
    """
    precision: Optional[str] = None
    max_layers: Optional[int] = None
    max_hidden: Optional[int] = None
    max_intermediate: Optional[int] = None
    kv_budget_bytes: Optional[int] = None
    seq_len: int = 2048
    batch: int = 1
    n_heads: Optional[int] = None
    head_dim: Optional[int] = None
    forwards_per_step: int = 1

    def kv_heads(self, arch: QwenArch) -> int:
        """The KV-cache head count (override or the arch's GQA KV heads)."""
        return self.n_heads if self.n_heads is not None else arch.num_key_value_heads

    def kv_head_dim(self, arch: QwenArch) -> int:
        return self.head_dim if self.head_dim is not None else arch.head_dim


@dataclass
class ConstraintSlack:
    """Per-constraint slack (cap - required); negative == VIOLATED."""
    precision: Optional[str] = None            # required vs config precision note
    layers: Optional[int] = None
    hidden: Optional[int] = None
    intermediate: Optional[int] = None
    kv_bytes: Optional[int] = None
    radix_valid: Optional[bool] = None         # None == not checked; False == invalid


@dataclass
class JointFitResult:
    """The joint-constraint solve outcome."""
    fits: bool
    geometry: OpConfigGeometry                 # required (n_layers, hidden, inter, ...)
    kv_bytes: int
    n_layers_kv: int                           # APPLIED depth (what KV is sized on)
    binding_constraint: Optional[str]          # None if fits; else the tightest bind
    relax_axis: Optional[str]                  # the axis to relax (None if fits)
    slack: ConstraintSlack
    constraints: FitConstraints
    notes: str = ""

    def summary(self) -> str:
        if self.fits:
            return (f"FITS: n_layers={self.geometry.stored_layers} "
                    f"hidden={self.geometry.hidden} "
                    f"intermediate={self.geometry.intermediate} "
                    f"KV={_h(self.kv_bytes)}B (applied depth {self.n_layers_kv})")
        return (f"INFEASIBLE — binding constraint: {self.binding_constraint}. "
                f"Relax: {self.relax_axis}. {self.notes}")


def _radix_valid_under(config, precision: Optional[str]) -> Tuple[bool, str]:
    """Is every op's radix EXACT under ``precision`` (the precision<->radix
    coupling)?  For a whole_value op the radix does not bound the datapath, so it is
    always valid.  Returns (ok, note).  ``precision=None`` checks each op's OWN
    precision instead."""
    from . import opconfig as OC
    for op in OC.ALL_OPS:
        ax = config.for_op(op)
        prec = precision if precision is not None else ax.precision
        if ax.extraction == "whole_value":
            continue                              # radix labels readout, not datapath
        ceiling = OC.PRECISION_CEILING[prec]
        am = OC.acc_max(op, ax.radix)
        if am > ceiling:
            return (False,
                    f"{op}: radix {ax.radix} overflows {prec} exact-int ceiling "
                    f"{ceiling} (accMax {am} > {ceiling}) -> lower radix or raise "
                    f"precision")
    return (True, "")


def _kv_precision_for(config, override: Optional[str]) -> str:
    """The precision KV bytes/elem are sized on: the FitConstraints override if
    given, else the config's DEEPEST op precision (the widest dtype the cache must
    hold — MUL's fp128 etc.)."""
    if override is not None:
        return override
    from . import opconfig as OC
    precs = {config.for_op(op).precision for op in OC.ALL_OPS}
    return max(precs, key=lambda p: OC.precision_bytes(p))


def solve_opconfig(config, constraints: FitConstraints,
                   code_size: int = 24, arch: QwenArch = QWEN2_5_ARCH
                   ) -> JointFitResult:
    """JOINTLY solve/validate ``config`` (an opconfig.OpConfig) against ALL of
    ``constraints`` — precision, depth (max_layers), width (max_hidden +
    max_intermediate) AND the KV-cache budget — at once.

    Returns a ``JointFitResult`` with the required geometry, the KV bytes (sized on
    the APPLIED depth — the looped/UT loop unrolls into the cache), each
    constraint's slack, and (on infeasibility) the BINDING constraint plus the axis
    to relax.  When several constraints are violated the binding one is the axis
    with the LARGEST relative overshoot (most-binding-first).
    """
    g = account_opconfig(config, code_size=code_size, arch=arch)

    # --- forwards-per-step: re-book stored depth into the autoregressive loop ----
    # Running the SAME network F forwards/VM-step needs only ceil(D/F) PHYSICAL
    # stored layers, so the DEPTH constraint is checked against that (F=1 == the
    # distinct-layer default, byte-identical to before).  KV is CONSERVED: applied
    # depth spread over F forwards => seq*F, and (D/F)*(seq*F) == D*seq, so KV is
    # sized on the applied depth exactly as before (forwards_per_step-invariant).
    fps = max(1, constraints.forwards_per_step)
    effective_stored_layers = -(-g.stored_layers // fps)   # ceil(stored_layers / F)

    # --- KV bytes, sized on the APPLIED depth (the loop unrolls into the cache) ---
    kv_prec = _kv_precision_for(config, constraints.precision)
    n_heads = constraints.kv_heads(arch)
    head_dim = constraints.kv_head_dim(arch)
    n_layers_kv = g.applied_depth
    kv_bytes = kv_cache_bytes(n_layers_kv, n_heads, head_dim,
                              constraints.seq_len, constraints.batch, kv_prec)

    # --- precision<->radix validity (a too-low precision at a given radix binds) ---
    radix_ok, radix_note = _radix_valid_under(config, constraints.precision)

    # --- per-constraint slack (cap - required); negative == violated -------------
    slack = ConstraintSlack()
    viol: List[Tuple[str, float, str]] = []   # (axis, relative_overshoot, relax_axis)

    if constraints.max_layers is not None:
        slack.layers = constraints.max_layers - effective_stored_layers
        if slack.layers < 0:
            viol.append(("depth (max_layers)",
                         effective_stored_layers / max(1, constraints.max_layers),
                         "max_layers (raise) / forwards_per_step (re-book depth into "
                         "the AR loop) / recurrence=tied+looped_transformer (fewer "
                         "stored cells) / shallower extraction"))
    if constraints.max_hidden is not None:
        slack.hidden = constraints.max_hidden - g.hidden
        if slack.hidden < 0:
            viol.append(("width (max_hidden)",
                         g.hidden / max(1, constraints.max_hidden),
                         "max_hidden (raise) / narrower extraction"))
    if constraints.max_intermediate is not None:
        slack.intermediate = constraints.max_intermediate - g.intermediate
        if slack.intermediate < 0:
            viol.append(("width (max_intermediate)",
                         g.intermediate / max(1, constraints.max_intermediate),
                         "max_intermediate (raise) / smaller radix/LUT extraction"))
    if constraints.kv_budget_bytes is not None:
        slack.kv_bytes = constraints.kv_budget_bytes - kv_bytes
        if slack.kv_bytes < 0:
            viol.append(("kv_cache (kv_budget_bytes)",
                         kv_bytes / max(1, constraints.kv_budget_bytes),
                         "kv_budget_bytes (raise) / lower precision (fewer bytes/"
                         "elem) / shallower applied depth / smaller seq_len,batch"))
    slack.radix_valid = radix_ok
    if constraints.precision is not None:
        slack.precision = constraints.precision
        if not radix_ok:
            # a hard bind: the required precision cannot hold the op's radix exactly.
            viol.append(("precision (invalid radix)", float("inf"),
                         "precision (raise) / radix (lower) — " + radix_note))

    fits = not viol
    binding, relax, note = None, None, ""
    if not fits:
        # most-binding-first: the axis with the LARGEST relative overshoot.
        viol.sort(key=lambda v: v[1], reverse=True)
        binding, _over, relax = viol[0]
        note = radix_note if binding.startswith("precision") else ""
        if binding == "precision (invalid radix)":
            note = radix_note

    if fps > 1:
        note = (f"forwards_per_step={fps}: depth met by {effective_stored_layers} "
                f"stored layers x {fps} forwards/step (re-booked from "
                f"{g.stored_layers} distinct; KV-neutral). " + note).strip()

    return JointFitResult(
        fits=fits, geometry=g, kv_bytes=kv_bytes, n_layers_kv=n_layers_kv,
        binding_constraint=binding, relax_axis=relax, slack=slack,
        constraints=constraints, notes=note)


# ===========================================================================
# PRECISION <-> KV <-> DEPTH coupling: min_kv_precision.
#
# The three-way coupling the search must HONESTLY account:
#
#   * LOWER precision  -> FEWER bytes/elem in the KV cache (int8=1 vs fp64=8),
#   * BUT lower precision -> a LOWER exact-int radix CEILING (opconfig.max_safe_radix)
#     -> MORE digits/limbs per value -> MORE applied depth -> MORE layer-applications
#     -> MORE KV (and more stored params in an unrolled model).
#
# So dropping precision does NOT monotonically shrink KV: it trades bytes/elem
# against applied depth.  ``min_kv_precision`` searches the precisions, builds the
# HONEST geometry for each (radix pinned to that precision's max-safe value, so the
# depth reflects the ceiling), and returns the precision minimizing the TOTAL
# inference-memory footprint = stored-param bytes + KV bytes.  Because KV scales
# with seq_len*batch but the stored params do NOT, the winner FLIPS with the
# seq_len/batch regime: at small seq/batch the (seq-independent) stored-param depth
# dominates and the SHALLOW fp64 whole-value config wins; at large seq/batch the KV
# term dominates and the tiny-bytes/elem int8 config wins — honestly accounting BOTH
# effects.  See docs/SOLVER_CONSTRAINTS.md §coupling for the worked table.
# ===========================================================================
_KV_SEARCH_PRECISIONS = ("int8", "bf16", "fp16", "fp32", "fp64")

# The whole-value precisions (a single 64-bit-mantissa scalar holds the value, no
# limb decomposition); their NATURAL model mode is LOOPED/UT (the min-params corner:
# few reused cells).  The finite-radix precisions limb-decompose (digit_extract);
# their NATURAL "many layers" mode is the STANDARD-FF unrolled stack.
_WHOLE_VALUE_PRECISIONS = ("fp64", "fp128")


def _natural_looped(precision: str) -> bool:
    """The NATURAL model mode for a precision when comparing precisions honestly:
    the whole-value precisions (fp64/fp128) are looped/UT (few cells — the
    min-params corner, "FEW LAYERS"); the finite-radix precisions are standard-FF
    unrolled ("MANY LAYERS", one distinct stored layer per limb place)."""
    return precision in _WHOLE_VALUE_PRECISIONS


def _honest_geometry_at_precision(precision: str, looped: bool,
                                  code_size: int, arch: QwenArch) -> OpConfigGeometry:
    """Build the HONEST geometry for an all-ops config at ``precision``: radix pinned
    to that precision's max-safe value (so applied depth reflects the exact-int
    ceiling), digit_extract for the finite-radix precisions, whole_value for the
    64-bit-mantissa precisions (fp64/fp128) where a single scalar holds the value."""
    from . import opconfig as OC
    if precision in ("fp64", "fp128"):
        # whole-value: one high-precision scalar per value, fixed decimal digit depth.
        ov = {op: dict(precision=precision, extraction="whole_value",
                       recurrence=("tied" if looped else "unrolled"))
              for op in OC.ALL_OPS}
        ov["MUL"] = dict(precision="fp128", extraction="whole_value",
                         recurrence=("tied" if looped else "unrolled"))
        base = OC.AxisConfig(precision=precision, radix=10, extraction="whole_value",
                             recurrence=("tied" if looped else "unrolled"))
        cfg = OC.OpConfig(base=base, overrides=ov, looped_transformer=looped)
    else:
        # digit_extract at the max-safe radix for the DIV bound (the r^2 coupling is
        # the tightest of the ALU ops that limb-decompose).
        r = OC.max_safe_radix("DIV", precision)
        ov = {op: dict(precision=precision, radix=r, extraction="digit_extract",
                       recurrence=("tied" if looped else "unrolled"))
              for op in OC.ALL_OPS}
        base = OC.AxisConfig(precision=precision, radix=r, extraction="digit_extract",
                             recurrence=("tied" if looped else "unrolled"))
        cfg = OC.OpConfig(base=base, overrides=ov, looped_transformer=looped)
    return account_opconfig(cfg, code_size=code_size, arch=arch)


@dataclass
class KvPrecisionRow:
    """One precision's honest cost on the precision<->KV<->depth frontier."""
    precision: str
    applied_depth: int          # layer-applications (what KV is sized on)
    stored_layers: int
    max_safe_radix_div: Optional[int]  # the radix ceiling driving depth (None=whole)
    bytes_per_elem: int
    kv_bytes: int               # KV cache at (seq_len, batch)
    param_bytes: int            # stored-param footprint at this precision
    total_bytes: int            # param_bytes + kv_bytes (the ranked objective)


@dataclass
class KvPrecisionResult:
    """``min_kv_precision`` outcome: the winner + the full frontier."""
    best_precision: str
    frontier: List[KvPrecisionRow]      # ranked ascending by total_bytes
    seq_len: int
    batch: int
    objective: str                       # "total" (params+KV) or "kv" (KV only)
    mode: str = "natural"                # "natural" | "looped" | "unrolled"

    def table(self) -> str:
        hdr = (f"{'prec':>6s} {'bytes/e':>7s} {'radixDIV':>8s} {'applied':>7s} "
               f"{'stored':>6s} {'KV':>9s} {'params':>9s} {'total':>9s}")
        lines = [f"seq_len={self.seq_len} batch={self.batch} "
                 f"objective={self.objective} mode={self.mode}", hdr, "-" * len(hdr)]
        for r in self.frontier:
            rad = "whole" if r.max_safe_radix_div is None else str(r.max_safe_radix_div)
            star = "  <== WIN" if r.precision == self.best_precision else ""
            lines.append(
                f"{r.precision:>6s} {r.bytes_per_elem:7d} {rad:>8s} "
                f"{r.applied_depth:7d} {r.stored_layers:6d} {_h(r.kv_bytes):>9s} "
                f"{_h(r.param_bytes):>9s} {_h(r.total_bytes):>9s}{star}")
        return "\n".join(lines)


def min_kv_precision(constraints: FitConstraints, mode: str = "natural",
                     looped: Optional[bool] = None,
                     objective: str = "total", code_size: int = 24,
                     arch: QwenArch = QWEN2_5_ARCH,
                     precisions: Sequence[str] = _KV_SEARCH_PRECISIONS
                     ) -> KvPrecisionResult:
    """Search precisions and return the one minimizing KV cost, HONESTLY accounting
    BOTH the precision<->KV coupling (lower precision = fewer bytes/elem) AND the
    precision<->depth coupling (lower precision = lower radix ceiling = more applied
    depth = more layer-applications = more KV + more stored params).

    ``objective``:
      * ``"total"`` (default) — minimize stored-param bytes + KV bytes (the total
        inference-memory footprint).  The winner FLIPS with the seq_len/batch regime
        (small -> fp64/whole-value's FEW-LAYER stored depth wins; large -> int8's
        tiny bytes/elem wins), because KV scales with seq*batch but stored params do
        not.
      * ``"kv"`` — minimize the KV bytes ALONE (applied_depth * bytes/elem * the
        seq*batch factor); the seq*batch factor is common so this ranks by
        applied_depth * bytes/elem, the pure per-token KV cost.

    ``mode`` selects the model MODE per precision:
      * ``"natural"`` (default) — each precision at its HONEST natural mode: the
        whole-value precisions (fp64/fp128) as LOOPED/UT (the min-params corner:
        FEW stored cells, big bytes/elem), the finite-radix precisions as STANDARD-FF
        UNROLLED (MANY distinct stored layers, small bytes/elem).  This is the
        fp64(few-layers) vs int8(many-layers) tradeoff the brief asks for; the
        total-footprint winner flips with seq_len/batch.
      * ``"looped"`` — force every precision LOOPED/UT (few cells).
      * ``"unrolled"`` — force every precision STANDARD-FF unrolled.
    ``looped=True/False`` is a back-compat override for ``mode`` (True==looped,
    False==unrolled).  Returns the winner + the full ranked frontier
    (``KvPrecisionResult.table()`` renders it)."""
    from . import opconfig as OC
    if looped is not None:
        mode = "looped" if looped else "unrolled"
    if mode not in ("natural", "looped", "unrolled"):
        raise ValueError(f"mode {mode!r} not in ('natural','looped','unrolled')")
    rows: List[KvPrecisionRow] = []
    n_heads = constraints.kv_heads(arch)
    head_dim = constraints.kv_head_dim(arch)
    for prec in precisions:
        if mode == "natural":
            prec_looped = _natural_looped(prec)
        else:
            prec_looped = (mode == "looped")
        g = _honest_geometry_at_precision(prec, prec_looped, code_size, arch)
        radix_div = None if prec in ("fp64", "fp128") else OC.max_safe_radix("DIV", prec)
        kv = kv_cache_bytes(g.applied_depth, n_heads, head_dim,
                            constraints.seq_len, constraints.batch, prec)
        pbytes = g.params_estimate * OC.precision_bytes(prec)
        total = pbytes + kv
        rows.append(KvPrecisionRow(
            precision=prec, applied_depth=g.applied_depth,
            stored_layers=g.stored_layers, max_safe_radix_div=radix_div,
            bytes_per_elem=OC.precision_bytes(prec), kv_bytes=kv,
            param_bytes=pbytes, total_bytes=total))
    if objective == "total":
        rows.sort(key=lambda r: (r.total_bytes, r.kv_bytes, r.bytes_per_elem))
    elif objective == "kv":
        rows.sort(key=lambda r: (r.kv_bytes, r.total_bytes, r.bytes_per_elem))
    else:
        raise ValueError(f"objective {objective!r} not in ('total', 'kv')")
    return KvPrecisionResult(
        best_precision=rows[0].precision, frontier=rows,
        seq_len=constraints.seq_len, batch=constraints.batch, objective=objective,
        mode=mode)
