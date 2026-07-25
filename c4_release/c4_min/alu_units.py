"""UNIFIED per-op ALU implementation registry — one interface to select the ADD /
MUL / DIV *unit* used by the fused Qwen VM, exposing the **depth<->weight** tradeoff
so arithmetic can be FIT to a target model's layer / width budget.

Why
===
The three arithmetic families each grew their OWN scattered selection switch:

  * DIV / MOD:  ``C4_DIV_LONGDIV`` (radix-16 vs base-16 long division),
                ``C4_DIV_LEAN`` (lean-80 vs hardened-88 radix-16 variant),
                and ``recurrent_divmod=`` (unroll 8 iters vs fold to 1 body).
  * MUL:        ``C4_MUL_LOOKAHEAD`` (Kogge-Stone prefix resolve vs serial ripple).
  * ADD / SUB:  no explicit per-op unit toggle — the fused VM's default path folds
                ADD/SUB into ONE scalar dispatch FFN rule; the byte-carry-chain
                gadget (``compile_addsub_blocks``, 8 blocks) is the deep pure-forward
                build's path, and ``C4_VM_WIDTH32`` / ``C4_VM_TWO_LIMB`` swap the
                *value substrate* (8-bit-fold vs fp32 two-limb) underneath both.

This module is a THIN dispatch/metadata layer: it does NOT reimplement any
arithmetic. It names the EXISTING implementations, records each one's depth
(blocks), nz-weight cost, width needs, and const-vs-variable applicability, exposes
``C4_ADD_UNIT`` / ``C4_MUL_UNIT`` / ``C4_DIV_UNIT`` env flags that DEFAULT to the
current production unit (so an unset environment is byte-identical to golden), and
provides ``select_units_for_model(layers, d_model)`` to auto-pick a unit per op to
fit a layer/width budget.

Honesty notes (do NOT invent variants)
=====================================
  * ADD / SUB has essentially ONE production unit in the fused VM (the scalar
    dispatch rule). The byte-carry-chain is a REAL alternative unit but is only
    wired on the deep ``nibble_pure_forward_complete`` path today; it is registered
    here as ``add=byte_chain`` for completeness + the fit selector, but selecting it
    in the fused VM is a NO-OP unless that path is used. This is called out in the
    unit's ``note``.
  * MUL has TWO real resolve variants (ripple vs Kogge-Stone lookahead) that are
    byte-identical in RESULT and differ only in block depth. The ``const`` multiply
    (``const_mul``) is a genuine third unit but only applies when an operand is a
    compile-time constant (``const_only=True``).
  * DIV has THREE production-wireable variants sharing the ``compile_divmod_blocks``
    interface (base long division, radix-16 lean, radix-16 hardened), PLUS the
    ``recurrent`` fold (orthogonal: any radix-16 variant can be unrolled OR folded),
    PLUS a ``const`` digit-recurrence divmod (``const_divmod_digitrec``, const
    divisor only). The attention-select / automaton / microcode divides
    (``div_radix16_attn`` / ``div_automaton_attn`` / ``microcode_div``) are
    MEASURE-ONLY bakeoffs — NOT wired into the fused VM builder — so they are listed
    with ``wired=False`` and MUST NOT be silently selected. The sibling "sub-40
    divide" agent owns adding a new *wired* variant; when it lands it registers here
    via :func:`register_div_unit` and the fit selector will pick it up.

Everything is GATED and DEFAULTS to the current production unit, so with no
``C4_*_UNIT`` set the fused VM build is byte-identical to golden.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Unit metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AluUnit:
    """One arithmetic implementation choice for an op family.

    Attributes
    ----------
    op        : "add" | "mul" | "div" — which op family this unit implements.
    name      : the ``C4_<OP>_UNIT`` selector token (e.g. ``"radix16_lean"``).
    depth     : the number of FFN/attn *blocks* the unit APPLIES per divide (the
                UNROLLED depth == total forward compute). ``None`` when the unit
                rides an existing block and adds no depth (the scalar ADD rule).
    stored_blocks : distinct blocks physically STORED (== depth for the unrolled
                wiring; << depth for the RECURRENT fold that stores ONE iteration
                body and reuses it 8x via the layer apply-order). THIS is what must
                fit the model's LAYER COUNT (a recurrent divide reuses layers, so it
                fits a 24-layer model even though it APPLIES 80 blocks). ``None`` ==
                same as depth.
    recurrent_blocks : distinct blocks STORED when the divide is wired via the
                ``recurrent_divmod`` fold (KB precompute + init + ONE iteration body
                + finalize). ``None`` for units with no recurrent form. This is the
                number the fit selector charges against the layer budget when it
                chooses the recurrent fold.
    nz        : approximate nonzero-weight count the unit bakes (order-of-magnitude
                cost signal; ``None`` when it rides shared weights).
    min_d_model : minimum residual width the unit's scratch bands need on top of the
                base VM layout, or ``None`` if it adds no bands beyond the family's
                default scratch. (Advisory — the real allocator pads L.D.)
    const_only : True == unit ONLY applies when the relevant operand is a
                compile-time constant.
    wired     : True == selectable in the fused VM builder TODAY. False == a
                measure-only bakeoff not yet on the build path (never auto-selected).
    module    : the implementing module (dotted, relative to ``c4_min``) — the source
                of truth. ``None`` for the ride-along scalar rule.
    builder   : the callable NAME the ``qwen_full_vm`` builder dispatches to (for the
                wired units), documentation-only here.
    note      : honest caveats.
    """

    op: str
    name: str
    depth: Optional[int]
    nz: Optional[int] = None
    stored_blocks: Optional[int] = None
    recurrent_blocks: Optional[int] = None
    min_d_model: Optional[int] = None
    const_only: bool = False
    wired: bool = True
    module: Optional[str] = None
    builder: Optional[str] = None
    note: str = ""

    @property
    def applied_depth(self) -> int:
        """Total forward compute (unrolled) the unit APPLIES (0 for a ride-along)."""
        return 0 if self.depth is None else self.depth

    def layer_cost(self, recurrent: bool = False) -> int:
        """Distinct STORED layers the unit needs (what the layer budget must hold).
        ``recurrent=True`` and a recurrent form present -> the folded stored count;
        else the stored (== applied) count. 0 for a ride-along unit."""
        if recurrent and self.recurrent_blocks is not None:
            return self.recurrent_blocks
        if self.stored_blocks is not None:
            return self.stored_blocks
        return self.applied_depth


# ---------------------------------------------------------------------------
# The registry — the EXISTING implementations, named + measured.
#
# Depths / nz are taken from each module's own validated docstring header
# (block counts the module reports for its 32-bit runtime path). They are the
# fit-selector's cost model, NOT re-derived here.
# ---------------------------------------------------------------------------

# --- ADD / SUB -------------------------------------------------------------
_ADD_UNITS: Dict[str, AluUnit] = {
    # PRODUCTION DEFAULT in the fused VM: ADD/SUB are a single scalar dispatch FFN
    # rule (base_dispatch_rules), riding the shared ``dispatch`` block — adds NO
    # dedicated depth. Value substrate (8-bit fold vs fp32 two-limb) is orthogonal
    # (C4_VM_WIDTH32 / C4_VM_TWO_LIMB), not an ADD *unit*.
    "scalar": AluUnit(
        op="add", name="scalar", depth=None, stored_blocks=0, nz=None,
        min_d_model=None, wired=True, module=None,
        builder="nibble_vm.base_dispatch_rules",
        note="Production default; ADD/SUB fold into the shared dispatch FFN rule. "
             "No dedicated blocks. Substrate width via C4_VM_WIDTH32/two-limb."),
    # REAL alternative unit: the per-byte carry chain (4 add bytes + 4 sub bytes).
    # Only wired on the deep nibble_pure_forward_complete path — selecting it in the
    # fused VM is inert (that build uses the scalar rule). Registered for the fit
    # selector + honest completeness.
    "byte_chain": AluUnit(
        op="add", name="byte_chain", depth=8, stored_blocks=8, nz=None,
        min_d_model=None, wired=False,
        module="nibble_alu32", builder="compile_addsub_blocks",
        note="Per-byte carry chain (4 ADD + 4 SUB byte blocks). Wired on the deep "
             "pure-forward path only; NOT on the fused-VM builder — do not expect "
             "byte-identity change in the fused VM when selected."),
}

# --- MUL -------------------------------------------------------------------
_MUL_UNITS: Dict[str, AluUnit] = {
    # PRODUCTION DEFAULT: schoolbook products + Kogge-Stone parallel-prefix resolve.
    # C4_MUL_LOOKAHEAD default ON == this unit. 8 blocks (products+split=2, then the
    # KS resolve). Byte-identical RESULT to the ripple variant.
    "lookahead": AluUnit(
        op="mul", name="lookahead", depth=8, stored_blocks=8, nz=11430,
        min_d_model=None, wired=True,
        module="nibble_alu32", builder="compile_mul_blocks (mul_lookahead ON)",
        note="Kogge-Stone parallel-prefix carry resolve (3 log-depth stages). "
             "C4_MUL_LOOKAHEAD default ON. Shallowest general 32-bit multiply."),
    # The historical serial-ripple resolve — byte-identical result, +2 blocks deeper.
    "ripple": AluUnit(
        op="mul", name="ripple", depth=10, stored_blocks=10, nz=11430,
        min_d_model=None, wired=True,
        module="nibble_alu32", builder="compile_mul_blocks (mul_lookahead OFF)",
        note="7 serial ripple carry rounds. C4_MUL_LOOKAHEAD=0. Byte-identical to "
             "lookahead, 2 blocks deeper, fewer scratch bands."),
    # Constant-multiplier fast path (b a compile-time constant): weighted copies,
    # no gated multiply. Only applies when an operand is constant.
    "const": AluUnit(
        op="mul", name="const", depth=None, stored_blocks=None, nz=None,
        min_d_model=None, const_only=True, wired=False,
        module="const_mul", builder="build_const_mul",
        note="Constant-multiplier weighted-copy fold (const operand only). Much "
             "shallower/cheaper than the general multiply but const-only; wired as "
             "a detection hook (C4_CONST_OPERAND), not the default path."),
}

# --- DIV / MOD -------------------------------------------------------------
# The three variants below SHARE the ``compile_divmod_blocks`` /
# ``compile_divmod_blocks_recurrent`` interface and are all production-wired in
# qwen_full_vm._block_specs. The ``recurrent`` fold is ORTHOGONAL (any radix-16
# variant can unroll OR fold-to-one-body); it is expressed via ``recurrent=`` on the
# resolved plan, not a separate unit here.
_DIV_UNITS: Dict[str, AluUnit] = {
    # PRODUCTION DEFAULT: lean radix-16 digit-recurrence. 80 blocks unrolled
    # (5 blocks/iter x 8 + KB precompute + init/finalize). Shallowest byte-exact
    # divide on the fused VM. C4_DIV_LEAN unset/1 == this.
    "radix16_lean": AluUnit(
        op="div", name="radix16_lean", depth=80, stored_blocks=80,
        recurrent_blocks=17, nz=None,
        min_d_model=None, wired=True,
        module="div_radix16_lean", builder="compile_divmod_blocks",
        note="Lean radix-16 digit-recurrence (9 blocks/iter x 8 + KB precompute + "
             "init/finalize = 80 applied). Production DEFAULT (C4_DIV_LEAN unset/1). "
             "RECURRENT fold stores ~17 layers (1 body reused 8x) -> fits shallow "
             "models. Shallowest byte-exact variable divide."),
    # Hardened radix-16 — +8 blocks (10 blocks/iter: KS-prefix borrow) for fp32
    # robustness on large-divisor 32-bit adversarial classes. C4_DIV_LEAN=0.
    "radix16_hardened": AluUnit(
        op="div", name="radix16_hardened", depth=88, stored_blocks=88,
        recurrent_blocks=18, nz=None,
        min_d_model=None, wired=True,
        module="div_radix16_hardened", builder="compile_divmod_blocks",
        note="Hardened radix-16 (10 blocks/iter, Kogge-Stone borrow). +8 blocks over "
             "lean for fp32 robustness on wide divisors. C4_DIV_LEAN=0 escape hatch. "
             "RECURRENT fold stores ~18 layers."),
    # Base-16 long division fallback (~262 blocks): the general base ALU path,
    # retained as the escape hatch. C4_DIV_LONGDIV=1.
    "longdiv": AluUnit(
        op="div", name="longdiv", depth=262, stored_blocks=262,
        recurrent_blocks=42, nz=None,
        min_d_model=None, wired=True,
        module="nibble_alu32", builder="compile_divmod_blocks",
        note="Base-16 long division fallback (~21 blocks/iter). Deepest; the general "
             "escape hatch, C4_DIV_LONGDIV=1. Prefer radix16 variants."),
    # Constant-divisor digit-recurrence divmod (const b): baked k*b thresholds,
    # q is a parallel threshold count — ~1-2 blocks/dividend-nibble. Const-only.
    "const": AluUnit(
        op="div", name="const", depth=None, stored_blocks=None, nz=None,
        min_d_model=None, const_only=True, wired=False,
        module="const_divmod_digitrec", builder="build_const_divmod_digitrec",
        note="Constant-divisor digit-recurrence (const divisor only, baked "
             "thresholds). Very shallow but const-only; wired as C4_CONST_OPERAND "
             "detection hook, not the default variable-divisor path."),
    # ---- Measure-only bakeoffs (NOT wired into the fused VM builder). Listed so the
    #      inventory is complete and honest; never auto-selected (wired=False). The
    #      sibling 'sub-40 divide' agent owns promoting one of these (or a new build)
    #      to wired via register_div_unit(). ------------------------------------
    "radix16_attn": AluUnit(
        op="div", name="radix16_attn", depth=None, nz=None, wired=False,
        module="div_radix16_attn", builder="build (measure-only)",
        note="Attention-SELECT radix-16: moves the per-iteration quotient-digit "
             "SELECT into a softmax1 head. MEASURE-ONLY bakeoff, not builder-wired. "
             "Sibling sub-40 agent territory."),
    "automaton_attn": AluUnit(
        op="div", name="automaton_attn", depth=None, nz=None, const_only=True,
        wired=False, module="div_automaton_attn", builder="build_div_automaton_attn",
        note="Const-divisor divmod as an attention-CAM automaton (transition table "
             "as KV memory). MEASURE-ONLY bakeoff, const-only. Sibling territory."),
    "microcode": AluUnit(
        op="div", name="microcode", depth=None, nz=None, wired=False,
        module="microcode_div", builder="(compact micro-frame)",
        note="Microcoded divide: 8 DIV_STEP micro-frames re-embedded per step "
             "(no layer looping). MEASURE-ONLY, not builder-wired. Sibling territory."),
}

_REGISTRY: Dict[str, Dict[str, AluUnit]] = {
    "add": _ADD_UNITS,
    "mul": _MUL_UNITS,
    "div": _DIV_UNITS,
}

# The current PRODUCTION default per op (what an unset environment selects). These
# MUST match the fused-VM builder's own default so a bare build stays golden.
_PRODUCTION_DEFAULT = {
    "add": "scalar",
    "mul": "lookahead",
    "div": "radix16_lean",
}

_ENV_FLAG = {"add": "C4_ADD_UNIT", "mul": "C4_MUL_UNIT", "div": "C4_DIV_UNIT"}


# ---------------------------------------------------------------------------
# Registration API (for the sibling divide agent / future units)
# ---------------------------------------------------------------------------
def register_unit(unit: AluUnit) -> None:
    """Add / replace a unit in the registry. Used by a sibling module (e.g. the
    'sub-40 divide' build) to make a NEW variant selectable without editing this
    file. Does NOT change any op's default."""
    _REGISTRY.setdefault(unit.op, {})[unit.name] = unit


def register_div_unit(unit: AluUnit) -> None:
    """Convenience for the sibling divide agent — asserts op=='div'."""
    if unit.op != "div":
        raise ValueError(f"register_div_unit expects op='div', got {unit.op!r}")
    register_unit(unit)


# ---------------------------------------------------------------------------
# Selection: env flag -> resolved unit, defaulting to production.
# ---------------------------------------------------------------------------
def units_for(op: str) -> Dict[str, AluUnit]:
    """All registered units for an op family."""
    if op not in _REGISTRY:
        raise KeyError(f"unknown op {op!r}; expected one of {list(_REGISTRY)}")
    return dict(_REGISTRY[op])


def default_unit(op: str) -> AluUnit:
    """The current PRODUCTION unit for an op (what an unset env selects)."""
    return _REGISTRY[op][_PRODUCTION_DEFAULT[op]]


def selected_unit(op: str, env: Optional[Dict[str, str]] = None) -> AluUnit:
    """Resolve the unit for an op from the ``C4_<OP>_UNIT`` env flag, defaulting to
    the production unit when unset/blank. An unknown or non-wired token raises (a
    non-wired unit is a measure-only bakeoff and must be promoted via
    ``register_unit`` before selection).

    NOTE on interop with the legacy per-op flags: this registry is the *intended*
    single interface, but the fused-VM builder still reads the legacy flags
    (``C4_DIV_LEAN`` / ``C4_DIV_LONGDIV`` / ``C4_MUL_LOOKAHEAD``) directly. Callers
    that want the registry choice honoured by the builder should call
    :func:`apply_to_env` to project the selection onto those legacy flags before
    building. With NOTHING set, both agree on the production default (golden)."""
    env = os.environ if env is None else env
    flag = _ENV_FLAG[op]
    tok = (env.get(flag) or "").strip()
    if not tok:
        return default_unit(op)
    units = _REGISTRY[op]
    if tok not in units:
        raise ValueError(
            f"{flag}={tok!r} is not a registered {op} unit; choose from "
            f"{sorted(units)}")
    unit = units[tok]
    if not unit.wired:
        raise ValueError(
            f"{flag}={tok!r} names a MEASURE-ONLY unit ({unit.module}); it is not "
            f"wired into the fused-VM builder. Promote it via register_unit() first.")
    return unit


def is_default_env(env: Optional[Dict[str, str]] = None) -> bool:
    """True iff NO C4_*_UNIT flag is set (build is the registry's golden default)."""
    env = os.environ if env is None else env
    return all(not (env.get(f) or "").strip() for f in _ENV_FLAG.values())


def apply_to_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Project the registry's per-op ``C4_<OP>_UNIT`` selection onto the LEGACY
    builder flags (``C4_DIV_LEAN`` / ``C4_DIV_LONGDIV`` / ``C4_MUL_LOOKAHEAD``) so
    the existing ``qwen_full_vm`` builder honours the registry choice WITHOUT
    editing it. With no unit flag set, returns {} and mutates nothing (golden).

    This is the bridge that keeps the registry non-invasive: the builder keeps its
    proven flag reads; the registry drives them. When ``env is os.environ`` the
    projected flags ARE written so a subsequent build picks them up."""
    env = os.environ if env is None else env
    if is_default_env(env):
        return {}
    overrides: Dict[str, str] = {}

    # DIV -> C4_DIV_LONGDIV / C4_DIV_LEAN
    div = selected_unit("div", env)
    if div.name == "longdiv":
        overrides["C4_DIV_LONGDIV"] = "1"
    elif div.name == "radix16_hardened":
        overrides["C4_DIV_LONGDIV"] = "0"
        overrides["C4_DIV_LEAN"] = "0"
    elif div.name == "radix16_lean":
        overrides["C4_DIV_LONGDIV"] = "0"
        overrides["C4_DIV_LEAN"] = "1"
    # const/measure-only div units cannot be projected onto the legacy flags; they
    # require the const-operand hook (C4_CONST_OPERAND) or a new builder wire — left
    # to the owning agent. selected_unit() already rejects non-wired tokens.

    # MUL -> C4_MUL_LOOKAHEAD
    mul = selected_unit("mul", env)
    if mul.name == "lookahead":
        overrides["C4_MUL_LOOKAHEAD"] = "1"
    elif mul.name == "ripple":
        overrides["C4_MUL_LOOKAHEAD"] = "0"

    # ADD has no legacy builder flag (scalar rule is the only wired fused-VM path);
    # byte_chain is not fused-VM-wired, so nothing to project. selected_unit() would
    # reject byte_chain (wired=False) before reaching here.

    for k, v in overrides.items():
        env[k] = v
    return overrides


# ---------------------------------------------------------------------------
# Fit-to-model selector.
# ---------------------------------------------------------------------------
@dataclass
class FitPlan:
    """The chosen unit per op for a (layers, d_model) budget + the accounting."""

    layers: int
    d_model: int
    add: AluUnit
    mul: AluUnit
    div: AluUnit
    div_recurrent: bool = False
    const_operands: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def arith_applied_depth(self) -> int:
        """Total forward compute (unrolled) across add+mul+div — the compute the
        model APPLIES per arithmetic op (a divide reuses layers, so this can exceed
        the layer count)."""
        return (self.add.applied_depth + self.mul.applied_depth
                + self.div.applied_depth)

    @property
    def arith_layer_cost(self) -> int:
        """Distinct STORED layers the arithmetic units occupy — what the model's
        layer count must actually hold (the recurrent divide folds to ~17)."""
        return (self.add.layer_cost() + self.mul.layer_cost()
                + self.div.layer_cost(recurrent=self.div_recurrent))

    def as_env(self) -> Dict[str, str]:
        """The C4_*_UNIT env the plan corresponds to (wired units only)."""
        out = {}
        for op, u in (("add", self.add), ("mul", self.mul), ("div", self.div)):
            if u.name != _PRODUCTION_DEFAULT[op] and u.wired:
                out[_ENV_FLAG[op]] = u.name
        return out


# Coarse reserve for the non-arithmetic VM machinery (fetch / decode / dispatch /
# mem-cam / cmp / branch / fold) that shares the layer budget with the ALU units.
_MACHINERY_RESERVE = 16


def select_units_for_model(layers: int, d_model: int,
                           const_operands: bool = False,
                           divmod_recurrent: Optional[bool] = None) -> FitPlan:
    """Pick a unit per op to satisfy a ``layers`` / ``d_model`` budget.

    The binding constraint is the DIVIDE: it is by far the deepest op (80 applied
    blocks for the lean radix-16). Two mechanisms let it fit a model with far fewer
    layers, and the selector chooses between them by the STORED layer cost:

      * UNROLLED: 80 distinct layers. Only fits genuinely deep models
        (``layers >= reserve + 80``); prefer the HARDENED variant when the model is
        deep AND wide enough for the robustness headroom.
      * RECURRENT fold: stores ONE iteration body (~17 layers) reused 8x by the
        layer apply-order (the production ``recurrent_divmod`` path). This is how a
        24-layer model runs an 80-block divide — it REUSES layers, so the ~17 STORED
        layers are what must fit, not the 80 APPLIED. The selector picks this
        automatically when the unrolled divide overflows the layer budget.
      * const_operands=True: prefer the CONST units (magic-multiply / const-divisor
        divmod) where an operand is a compile-time constant — the shallowest of all,
        near-zero added depth — falling back to the variable unit otherwise.

    ADD/MUL keep the shallow scalar/lookahead defaults (already the cheapest wired
    units). This is a COST-MODEL heuristic over the registry metadata, NOT a
    rebuild; callers apply the result via ``plan.as_env()`` / ``apply_to_env`` (and
    pass ``recurrent_divmod=plan.div_recurrent`` to the builder) before building.
    """
    notes: List[str] = []

    # ADD: the scalar rule is depth-0 and the only fused-VM-wired unit — always it.
    add = _ADD_UNITS["scalar"]

    # MUL: const magic-multiply if const operands available, else the shallow
    # lookahead (already the cheapest general unit).
    if const_operands:
        mul = _MUL_UNITS["const"]
        notes.append("mul=const: compile-time-constant multiplier -> magic-multiply "
                     "weighted-copy fold (near-zero added depth); variable b falls "
                     "back to lookahead.")
    else:
        mul = _MUL_UNITS["lookahead"]
        notes.append("mul=lookahead: shallowest general 32-bit multiply "
                     "(Kogge-Stone resolve, 8 blocks).")

    div_recurrent = bool(divmod_recurrent)

    if const_operands:
        # Constant divisor -> the const-divisor magic divide, shallowest of all.
        div = _DIV_UNITS["const"]
        div_recurrent = False
        notes.append("div=const: compile-time-constant divisor -> magic divide "
                     "(~1-2 blocks/nibble, near-zero added depth); a variable divisor "
                     "falls back to radix16_lean (recurrent if shallow).")
    else:
        budget_for_div = layers - _MACHINERY_RESERVE - mul.layer_cost()
        lean = _DIV_UNITS["radix16_lean"]
        hard = _DIV_UNITS["radix16_hardened"]
        if budget_for_div >= hard.stored_blocks and d_model >= 2048:
            # deep + wide: room to UNROLL the robust variant as distinct layers.
            div = hard
            div_recurrent = False
            notes.append(
                f"div=radix16_hardened UNROLLED: deep/wide model (budget "
                f"{budget_for_div} layers for divide >= {hard.stored_blocks}, "
                f"d_model {d_model}) has headroom for the fp32-robust +8-block "
                f"variant as distinct layers.")
        elif budget_for_div >= lean.stored_blocks:
            div = lean
            div_recurrent = False
            notes.append(
                f"div=radix16_lean UNROLLED: budget {budget_for_div} layers >= "
                f"{lean.stored_blocks}; the divide fits as distinct layers.")
        else:
            # Too shallow to unroll -> RECURRENT fold (stores ~17, reuses 8x). This
            # is the PRODUCTION mechanism for the 24-layer Qwen-0.5B.
            div = lean
            div_recurrent = True
            if budget_for_div >= lean.recurrent_blocks:
                notes.append(
                    f"div=radix16_lean RECURRENT: unrolled 80 overflows the "
                    f"{budget_for_div}-layer divide budget, so fold to the reused "
                    f"iteration body (~{lean.recurrent_blocks} stored layers, applied "
                    f"8x). This is how a {layers}-layer model runs an 80-block divide.")
            else:
                notes.append(
                    f"div=radix16_lean RECURRENT but even the folded "
                    f"~{lean.recurrent_blocks} stored layers exceed the "
                    f"{budget_for_div}-layer divide budget: the model is too shallow "
                    f"for a variable divide at all. Restrict to const divisors "
                    f"(const_operands=True -> magic divide) or grow the layer count.")

    return FitPlan(layers=layers, d_model=d_model, add=add, mul=mul, div=div,
                   div_recurrent=div_recurrent, const_operands=const_operands,
                   notes=notes)


# ---------------------------------------------------------------------------
# Human-readable inventory / reporting.
# ---------------------------------------------------------------------------
def format_inventory() -> str:
    lines = ["ALU UNIT REGISTRY — per-op implementation inventory", "=" * 68]
    for op in ("add", "mul", "div"):
        dflt = _PRODUCTION_DEFAULT[op]
        lines.append(f"\n[{op.upper()}]  env flag: {_ENV_FLAG[op]}  "
                     f"(default -> {dflt})")
        lines.append(f"  {'unit':<18}{'applied':>8}{'stored':>7}{'recur':>6}"
                     f"{'nz':>8}  {'wired':<6}{'const':<6} module")
        for name, u in _REGISTRY[op].items():
            depth = "-" if u.depth is None else str(u.depth)
            stored = "-" if u.stored_blocks is None else str(u.stored_blocks)
            recur = "-" if u.recurrent_blocks is None else str(u.recurrent_blocks)
            nz = "-" if u.nz is None else str(u.nz)
            mark = " *DEFAULT" if name == dflt else ""
            lines.append(
                f"  {name:<18}{depth:>8}{stored:>7}{recur:>6}{nz:>8}"
                f"  {str(u.wired):<6}{str(u.const_only):<6} {u.module or '(rule)'}{mark}")
    lines.append("\n  applied = unrolled forward blocks; stored = distinct layers "
                 "unrolled; recur = distinct layers when the divide is folded "
                 "recurrently (reused 8x).")
    return "\n".join(lines)


def format_fit_examples() -> str:
    """Worked fit examples across the Qwen sizes."""
    lines = ["FIT-TO-MODEL SELECTOR — worked examples", "=" * 68]
    examples = [
        ("Qwen2.5-0.5B", 24, 896),
        ("Qwen2.5-1.5B", 28, 1536),
        ("Qwen2.5-7B", 28, 3584),
        ("Qwen2.5-72B", 80, 8192),
    ]
    for label, layers, d_model in examples:
        for const in (False, True):
            plan = select_units_for_model(layers, d_model, const_operands=const)
            tag = " (const operands)" if const else ""
            recur = " +recurrent-fold" if plan.div_recurrent else ""
            lines.append(f"\n{label}: {layers} layers / d_model {d_model}{tag}")
            lines.append(f"  add={plan.add.name}  mul={plan.mul.name}  "
                         f"div={plan.div.name}{recur}")
            lines.append(
                f"  arith STORED layer cost = {plan.arith_layer_cost} "
                f"(add {plan.add.layer_cost()} + mul {plan.mul.layer_cost()} + div "
                f"{plan.div.layer_cost(recurrent=plan.div_recurrent)}) "
                f"vs {layers} model layers  "
                f"[applied forward = {plan.arith_applied_depth} blocks]")
            env = plan.as_env()
            recur_env = "  recurrent_divmod=True" if plan.div_recurrent else ""
            lines.append(f"  env: {env if env else '{} (all production defaults)'}"
                         f"{recur_env}")
            for n in plan.notes:
                lines.append(f"    - {n}")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - manual/CLI reporting
    print(format_inventory())
    print()
    print(format_fit_examples())


if __name__ == "__main__":  # pragma: no cover
    main()
