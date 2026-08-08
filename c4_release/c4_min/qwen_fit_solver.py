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


def _spec_sizes(config: FitConfig,
                subset: Optional[Subset] = None) -> Tuple[int, int, int, int]:
    """(hidden, intermediate, stored_layers, applied_depth) at the config's op-subset
    + efficient/recurrent flags (32-bit, the baked path), arch-rescaled.  ``subset``
    overrides ``config.subset`` (used by the subroutine lever, which accounts the
    stack WITHOUT muldiv baked in)."""
    sub = subset if subset is not None else config.subset
    eff = config.efficient_alu and (subset is None)   # subroutine drops the baked ALU
    rec = config.recurrent_divmod and (subset is None)
    d_used, inter, n_stored, n_applied = _raw_spec_sizes(
        config.code_size, sub.memory, sub.cmp, sub.bitwise, sub.muldiv, eff, rec)
    arch = config.arch
    inter = max(inter, arch.num_attention_heads * arch.head_dim, 8)
    hidden = arch.hidden_for(d_used + 1)
    return hidden, inter, n_stored, n_applied


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
