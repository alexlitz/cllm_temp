"""PER-OP CONFIG TOGGLE SYSTEM — a first-class {precision, radix, extraction,
recurrence} config per c4 opcode, wired into the size-fitter + the ``C4_*`` flag
registry.  DEFAULT-PRESERVING: the global DEFAULT config reproduces the current
nibble / fp32 build (golden ``174ece66``); every non-default axis value is a
documented golden-MOVING toggle.

This is the CONFIG / TOGGLE + FITTER layer that consolidates the completed
min-param / precision / depth work
(``examples/{clever_minparam_alu, lowprec_radix_alu}.py`` +
``docs/{CLEVER_MINPARAM_ALU, ALL_OPS_MINPARAM, PRECISION_RADIX_SURFACE,
CLEVER_DOOM_REALTIME}.md``) into ONE resolvable schema.  It does NOT rebuild the
byte-exact doom VM with the clever ops — that is separate future work (see
``docs/TOGGLE_SCHEMA.md`` §honesty).

The four axes (per opcode)
==========================
  * **precision**  — the datapath dtype.  Its *exact-integer ceiling* (largest M
    with every integer in ``[0, M]`` exactly representable) BOUNDS the radix:
      int8 2^7-1=127 · bf16 2^8 · fp16 2^11 · fp32 2^24 · fp64 2^53 · fp128 2^64.
  * **radix**      — any base ``r``; each value/limb lives in ``[0, r)``.  Bounded
    by the precision's ceiling for the op via the op's accumulator-max formula:
      ADD/SUB ~2r · CMP ~r · MUL ~r^2*L (L = #operand limbs) · DIV/MOD ~r^2.
  * **extraction** — how a value is read out: ``nibble`` (the production 4-bit
    lanes), ``digit_extract`` (MSB-first one-digit-per-layer difference-min
    decode — the clever-minparam form), or ``whole_value`` (one high-precision
    scalar, no decomposition).
  * **recurrence** — ``unrolled`` (each output place a distinct stored layer) or
    ``tied`` (one reused cell applied ``depth`` times — STORED shrinks, APPLIED
    unchanged; the Universal-Transformer fold).

    HONESTY CONSTRAINT (``looped_transformer``): weight-TIED recurrence is a
    *looped* / Universal-Transformer implementation — the SAME stored cell is
    re-applied ``depth`` times per forward.  A STANDARD feed-forward transformer
    (like the released Qwen2.5-0.5B: 24 DISTINCT decoder layers, each applied
    exactly once) has NO loop, so it CANNOT weight-tie: it must UNROLL every place
    into a distinct stored layer.  ``recurrence='tied'`` is therefore ONLY legal
    when the model is declared ``looped_transformer=True``; on a standard
    feed-forward model the validator rejects (or, with ``force=True``, downgrades)
    ``tied`` → ``unrolled``.  Counting ``tied`` as a param-win for a STOCK
    feed-forward checkpoint is dishonest — see ``TOGGLE_SCHEMA.md`` §honesty.

The MODEL MODE (``looped_transformer``)
=======================================
``OpConfig.looped_transformer`` (default ``False`` = a STANDARD feed-forward
transformer, matching stock Qwen2) declares whether the target architecture is a
LOOPED / Universal-Transformer (a single stored cell re-applied ``depth`` times)
or a standard feed-forward stack (distinct layer per place, each applied once).
It is the gate that makes ``tied`` legal.

The DEFAULT
===========
Every op defaults to ``precision=fp32, radix=16 (nibble base), extraction=nibble,
recurrence=unrolled`` — i.e. the production nibble-c4 build — and the model mode
defaults to ``looped_transformer=False`` (standard feed-forward).  ``DEFAULT`` (a
plain ``OpConfig`` with an empty per-op override dict) resolves to exactly that
for every op, so ``DEFAULT`` == the golden ``174ece66`` build config (the resolver
never touches the build unless a non-default override is present).

The ``C4_OPCFG_*`` env flags
============================
Following the ``C4_*`` convention (see ``docs/DOOM_FLAG_REGISTRY.md``), each axis
is overridable per op::

    C4_OPCFG_DIV_PRECISION=fp64
    C4_OPCFG_DIV_RADIX=16
    C4_OPCFG_MUL_EXTRACTION=digit_extract
    C4_OPCFG_ADD_RECURRENCE=tied

A bare ``C4_OPCFG_ALL_PRECISION=bf16`` (op = ``ALL``) sets the axis for every op
(individual per-op flags override it).  With NO ``C4_OPCFG_*`` flag set, the
resolver returns ``DEFAULT`` and the build is byte-identical to golden.

``C4_OPCFG_LOOPED_TRANSFORMER=1`` declares the MODEL MODE as a LOOPED /
Universal-Transformer (making ``recurrence=tied`` legal); unset/0 == a STANDARD
feed-forward transformer (the default), on which ``tied`` is rejected — see
``validate``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple


# =========================================================================== #
# Axis value vocabularies.
# =========================================================================== #
PRECISIONS: Tuple[str, ...] = ("int8", "fp16", "bf16", "fp32", "fp64", "fp128")
EXTRACTIONS: Tuple[str, ...] = ("nibble", "digit_extract", "whole_value")
RECURRENCES: Tuple[str, ...] = ("unrolled", "tied")

# exact-integer CEILING per precision: the largest integer M such that every
# integer in [0, M] is representable EXACTLY.  For an IEEE float with m stored
# mantissa bits that is 2^(m+1) (implicit leading bit).  int8 keys off its signed
# magnitude 2^7-1 = 127 (we compute magnitudes).  fp128 = x86-64 longdouble
# (80-bit ext, 64-bit effective mantissa) holds every integer < 2^64 exactly.
# Sourced from lowprec_radix_alu.PRECISIONS + CLEVER_MINPARAM_ALU.md.
PRECISION_CEILING: Dict[str, int] = {
    "int8": (1 << 7) - 1,   # 127  (signed magnitude)
    "bf16": 1 << 8,         # 256    (7 stored mantissa bits)
    "fp16": 1 << 11,        # 2048   (10 stored mantissa bits)
    "fp32": 1 << 24,        # 16.7M  (23 stored)
    "fp64": 1 << 53,        # 9.0e15 (52 stored)
    "fp128": 1 << 64,       # 1.8e19 (x86-64 longdouble, 64-bit effective mantissa)
}

# =========================================================================== #
# Op names (the config keys).  Mirror c4_min.isa; kept as plain strings so
# opconfig has no heavy import + so ``ALL`` is a valid pseudo-op key for flags.
# =========================================================================== #
ALU_OPS: Tuple[str, ...] = ("ADD", "SUB", "MUL", "DIV", "MOD")
CMP_OPS: Tuple[str, ...] = ("EQ", "NE", "LT", "GT", "LE", "GE")
SHIFT_OPS: Tuple[str, ...] = ("SHL", "SHR")
BITWISE_OPS: Tuple[str, ...] = ("OR", "XOR", "AND")
FRAME_OPS: Tuple[str, ...] = ("LEA", "JMP", "JSR", "BZ", "BNZ", "ENT", "ADJ", "LEV")
MEM_OPS: Tuple[str, ...] = ("LI", "LC", "SI", "SC")
TRIVIAL_OPS: Tuple[str, ...] = ("IMM", "PSH", "NOP", "HALT")

# The full ordered op vocabulary the config covers.
ALL_OPS: Tuple[str, ...] = (
    FRAME_OPS + MEM_OPS + ("PSH",) + BITWISE_OPS + CMP_OPS + SHIFT_OPS
    + ALU_OPS + ("IMM", "NOP", "HALT")
)
# de-dup while preserving order (PSH appears in both FRAME-adjacent and TRIVIAL).
_seen: set = set()
ALL_OPS = tuple(o for o in ALL_OPS if not (o in _seen or _seen.add(o)))

# Which accumulator-max formula each op uses (from PRECISION_RADIX_SURFACE.md §1).
# accMax at radix r must stay <= the precision ceiling.
#   add/sub : 2r      cmp : r       mul : r^2 * L (L = #operand limbs)   div/mod : r^2
# ops with no radix-coupled accumulator (trivial / bitwise LUT / memory-CAM /
# whole-value frame adds that never limb-decompose) use the "add"-class 2r bound
# as a conservative default (it is the weakest constraint that still rejects an
# overflowing radix).
_OP_CLASS: Dict[str, str] = {}
for _o in ALU_OPS:
    _OP_CLASS[_o] = {"ADD": "add", "SUB": "add", "MUL": "mul",
                     "DIV": "div", "MOD": "div"}[_o]
for _o in CMP_OPS:
    _OP_CLASS[_o] = "cmp"
for _o in SHIFT_OPS:
    _OP_CLASS[_o] = "add"            # <<n / >>n whole-value scale + mask, ~2r ripple
for _o in FRAME_OPS:
    _OP_CLASS[_o] = "add"           # address adds are ADD-class
for _o in BITWISE_OPS + MEM_OPS + TRIVIAL_OPS:
    _OP_CLASS[_o] = "add"           # no radix-coupled accumulator -> loosest bound


def _operand_limbs(radix: int, bits: int = 32) -> int:
    """#base-`radix` limbs to hold a `bits`-bit unsigned operand (for the MUL
    r^2*L coupling)."""
    import math
    if radix < 2:
        return bits
    return max(1, math.ceil(bits / math.log2(radix)))


def _mul_peak_column(radix: int, bits: int = 32) -> int:
    """Exact worst-case schoolbook-multiply column accumulator (incl. carry) for a
    ``bits``x``bits`` base-``radix`` multiply, all operand limbs = radix-1 (the
    true maximum the low dtype must hold).  Mirrors
    lowprec_radix_alu.mul_peak_column so the surface's MUL rows match exactly.
    The loose ``r^2 * L`` bound (PRECISION_RADIX_SURFACE.md §1) over-counts (not
    every column is full width); this is the genuine peak §2 reports."""
    L = _operand_limbs(radix, bits)
    lo = radix - 1
    carry = 0
    peak = 0
    for k in range(2 * L):
        col = carry
        for i in range(max(0, k - (L - 1)), min(L - 1, k) + 1):
            col += lo * lo
        peak = max(peak, col)
        carry = col // radix
    return peak


# the accumulator CLASS labels (also accepted directly by acc_max/max_safe_radix
# so the surface doc's per-class rows — ADD/SUB, CMP, MUL, DIV/MOD — resolve).
_CLASS_ALIASES = {
    "ADD/SUB": "add", "ADD": "add", "SUB": "add",
    "CMP": "cmp",
    "MUL": "mul",
    "DIV/MOD": "div", "DIV": "div", "MOD": "div",
}


def acc_max(op: str, radix: int) -> int:
    """The op's worst-case intermediate accumulator at ``radix`` (the value the
    precision must hold EXACTLY).  Matches PRECISION_RADIX_SURFACE.md §1:

        add/sub : 2r      cmp : r       mul : r^2 * L      div/mod : r^2

    ``op`` may be a concrete opcode (``ADD``, ``EQ``, ...) OR a surface class
    label (``ADD/SUB``, ``CMP``, ``DIV/MOD``, ``MUL``)."""
    key = op.upper()
    cls = _OP_CLASS.get(key) or _CLASS_ALIASES.get(key)
    if cls is None:
        raise KeyError(f"unknown op {op!r}; known: {ALL_OPS} + class labels "
                       f"{sorted(set(_CLASS_ALIASES))}")
    r = int(radix)
    if cls == "add":
        return 2 * r
    if cls == "cmp":
        # one limb difference ~ r; the surface's max_safe_radix keys off r+1 (the
        # inclusive bound a_j - b_j spans), so we match it exactly.
        return r + 1
    if cls == "div":
        return r * r
    if cls == "mul":
        # exact worst-case column peak (matches the surface §2); the loose
        # r^2*L bound over-counts and would spuriously reject radix 2 at int8.
        return _mul_peak_column(r)
    raise AssertionError(cls)


# =========================================================================== #
# The per-op config.
# =========================================================================== #
@dataclass(frozen=True)
class AxisConfig:
    """The four toggle axes for ONE opcode."""
    precision: str = "fp32"
    radix: int = 16                  # nibble base = radix 16
    extraction: str = "nibble"
    recurrence: str = "unrolled"

    def label(self) -> str:
        return (f"prec={self.precision} radix={self.radix} "
                f"extract={self.extraction} recur={self.recurrence}")


# The DEFAULT axis config == the production nibble / fp32 build.
DEFAULT_AXES = AxisConfig(precision="fp32", radix=16,
                          extraction="nibble", recurrence="unrolled")


@dataclass(frozen=True)
class OpConfig:
    """A per-op toggle configuration.  ``overrides`` maps an op name -> a partial
    dict of axis overrides; ``base`` is the fallback axis config for any op NOT in
    ``overrides``.  ``DEFAULT`` == ``OpConfig()`` == the golden build.

    ``looped_transformer`` (default ``False``) is the MODEL-MODE field: ``False``
    declares a STANDARD feed-forward transformer (distinct layer per place, each
    applied once — stock Qwen2), ``True`` declares a LOOPED / Universal-Transformer
    (one stored cell re-applied ``depth`` times).  It gates whether the
    weight-TIED recurrence axis is legal: ``recurrence='tied'`` requires
    ``looped_transformer=True`` (see ``validate``).
    """
    base: AxisConfig = field(default_factory=lambda: DEFAULT_AXES)
    overrides: Dict[str, Dict[str, object]] = field(default_factory=dict)
    looped_transformer: bool = False

    def for_op(self, op: str) -> AxisConfig:
        """The fully-resolved ``AxisConfig`` for ``op`` (base + any override)."""
        op = op.upper()
        if op not in ALL_OPS:
            raise KeyError(f"unknown op {op!r}; known: {ALL_OPS}")
        ov = self.overrides.get(op, {})
        if not ov:
            return self.base
        return replace(self.base, **ov)

    def is_default(self) -> bool:
        """True iff every op resolves to ``DEFAULT_AXES`` AND the model is a
        standard feed-forward transformer (== golden build)."""
        if self.looped_transformer:
            return False
        if self.base != DEFAULT_AXES:
            return False
        return all(self.for_op(op) == DEFAULT_AXES for op in ALL_OPS)

    def non_default_ops(self) -> Dict[str, AxisConfig]:
        """Every op whose resolved axes differ from the DEFAULT (the golden-MOVING
        set)."""
        return {op: self.for_op(op) for op in ALL_OPS
                if self.for_op(op) != DEFAULT_AXES}


# The global DEFAULT config: the current nibble / fp32 build.  Building from
# DEFAULT is byte-identical to golden 174ece66 (no override touches the build).
DEFAULT = OpConfig()


# =========================================================================== #
# VALIDATOR — reject a radix that overflows the precision's exact-int ceiling for
# the op, per the accMax formulas.
# =========================================================================== #
class OpConfigError(ValueError):
    """A per-op axis config is invalid (bad axis value or radix overflow)."""


def validate_axes(op: str, axes: AxisConfig) -> None:
    """Raise ``OpConfigError`` if ``axes`` is invalid for ``op``.

    Checks:
      1. precision / extraction / recurrence are in the allowed vocabularies.
      2. radix >= 2.
      3. the op's accumulator max at this radix stays <= the precision ceiling
         (the precision <-> radix COUPLING TABLE), per PRECISION_RADIX_SURFACE.md.

    The radix<->precision coupling is SKIPPED for ``whole_value`` extraction: the
    whole-value form holds the *whole* operand/result in one scalar and never
    limb-decomposes, so radix is not a datapath bound (it only labels the readout
    base).  The precision must instead hold the whole value, which the fitter/doc
    layer tracks (e.g. MUL needs fp128 for the 64-bit product); the validator only
    rejects a radix that would OVERFLOW under a limb-decomposing extraction.
    """
    op = op.upper()
    if op not in ALL_OPS:
        raise OpConfigError(f"unknown op {op!r}; known: {ALL_OPS}")
    if axes.precision not in PRECISIONS:
        raise OpConfigError(
            f"{op}: precision {axes.precision!r} not in {PRECISIONS}")
    if axes.extraction not in EXTRACTIONS:
        raise OpConfigError(
            f"{op}: extraction {axes.extraction!r} not in {EXTRACTIONS}")
    if axes.recurrence not in RECURRENCES:
        raise OpConfigError(
            f"{op}: recurrence {axes.recurrence!r} not in {RECURRENCES}")
    if not isinstance(axes.radix, int) or axes.radix < 2:
        raise OpConfigError(f"{op}: radix {axes.radix!r} must be an int >= 2")

    if axes.extraction == "whole_value":
        # whole-value: radix does not bound the datapath (no limb decomposition).
        return
    ceiling = PRECISION_CEILING[axes.precision]
    am = acc_max(op, axes.radix)
    if am > ceiling:
        raise OpConfigError(
            f"{op}: radix {axes.radix} overflows {axes.precision} exact-int "
            f"ceiling {ceiling} (accMax {am} > {ceiling}; op-class "
            f"'{_OP_CLASS[op]}' bound). Lower the radix or raise precision.")


def _tied_ops(config: OpConfig) -> Tuple[str, ...]:
    """The ops whose resolved recurrence is ``tied`` (incl. the base)."""
    tied = [op for op in ALL_OPS if config.for_op(op).recurrence == "tied"]
    return tuple(tied)


def validate(config: OpConfig) -> None:
    """Validate every op's resolved axes in ``config`` (raise on the first bad
    one), AND enforce the MODEL-MODE honesty constraint:

      * ``recurrence='tied'`` (weight-tied recurrence) is a LOOPED /
        Universal-Transformer implementation and is ONLY legal when the model is
        declared ``looped_transformer=True``.  A STANDARD feed-forward transformer
        (``looped_transformer=False``, e.g. stock Qwen2.5-0.5B — 24 distinct
        layers each applied once) has no loop to re-apply a tied cell, so it MUST
        unroll.  A ``tied`` axis on a standard model is REJECTED here (use
        ``force_standard_feedforward()`` to downgrade ``tied`` → ``unrolled``, or
        set ``looped_transformer=True`` to declare a UT model explicitly).
    """
    validate_axes("ADD", config.base)  # base must itself be a coherent config
    for op in ALL_OPS:
        validate_axes(op, config.for_op(op))
    if not config.looped_transformer:
        tied = _tied_ops(config)
        if tied:
            raise OpConfigError(
                "recurrence='tied' requires a LOOPED / Universal-Transformer model "
                "(looped_transformer=True): a standard feed-forward transformer "
                "cannot re-apply a weight-tied cell, it must UNROLL. Offending ops: "
                f"{tied}. Either set looped_transformer=True (declare a UT model) or "
                "call force_standard_feedforward(config) to downgrade tied->unrolled.")


def force_standard_feedforward(config: OpConfig) -> OpConfig:
    """Return a copy of ``config`` coerced to a STANDARD feed-forward model:
    ``looped_transformer=False`` and every ``recurrence='tied'`` axis downgraded to
    ``'unrolled'`` (the only honest recurrence for a stock feed-forward checkpoint).
    Use this to turn a UT/looped config into what a standard transformer would
    actually have to store (distinct layer per place)."""
    new_base = replace(config.base, recurrence="unrolled") \
        if config.base.recurrence == "tied" else config.base
    new_over: Dict[str, Dict[str, object]] = {}
    for op, ov in config.overrides.items():
        ov2 = dict(ov)
        if ov2.get("recurrence") == "tied":
            ov2["recurrence"] = "unrolled"
        new_over[op] = ov2
    return OpConfig(base=new_base, overrides=new_over, looped_transformer=False)


def max_safe_radix(op: str, precision: str) -> int:
    """The largest power-of-two radix (>= 2) whose accMax stays <= the precision
    ceiling for ``op`` (the exact bound the validator enforces).  Handy for the
    fitter + the doc coupling table."""
    ceiling = PRECISION_CEILING[precision]
    if acc_max(op, 2) > ceiling:
        raise OpConfigError(
            f"{op}: even radix 2 overflows {precision} (accMax "
            f"{acc_max(op, 2)} > {ceiling})")
    r = 2
    while acc_max(op, r * 2) <= ceiling:
        r *= 2
    return r


# =========================================================================== #
# RESOLVER — read ``C4_OPCFG_*`` env flags into an ``OpConfig``.
# =========================================================================== #
_AXIS_ENV = {
    "PRECISION": "precision",
    "RADIX": "radix",
    "EXTRACTION": "extraction",
    "RECURRENCE": "recurrence",
}


def _coerce(axis: str, raw: str) -> object:
    if axis == "radix":
        try:
            return int(raw)
        except ValueError as e:
            raise OpConfigError(f"C4_OPCFG_*_RADIX={raw!r} is not an int") from e
    return raw


def resolve(env: Optional[Dict[str, str]] = None,
            validate_result: bool = True) -> OpConfig:
    """Build an ``OpConfig`` from ``C4_OPCFG_*`` env flags (``os.environ`` by
    default).

    Flag grammar (per the ``C4_*`` convention):
        C4_OPCFG_<OP>_<AXIS> = <value>
      where <OP> is an op name (``DIV``, ``MUL``, ...) or ``ALL`` (sets the axis
      for the base, i.e. every op), and <AXIS> is ``PRECISION`` / ``RADIX`` /
      ``EXTRACTION`` / ``RECURRENCE``.

    With NO ``C4_OPCFG_*`` flag present the result is ``DEFAULT`` (the golden
    build config).  ``ALL`` writes the ``base``; per-op flags write ``overrides``
    (which win over ``ALL``).  Validated unless ``validate_result=False``.
    """
    env = os.environ if env is None else env
    base = dict(precision=DEFAULT_AXES.precision, radix=DEFAULT_AXES.radix,
                extraction=DEFAULT_AXES.extraction,
                recurrence=DEFAULT_AXES.recurrence)
    overrides: Dict[str, Dict[str, object]] = {}
    # MODEL-MODE flag: C4_OPCFG_LOOPED_TRANSFORMER=1 declares a LOOPED / UT model
    # (makes recurrence='tied' legal); unset/0 == standard feed-forward (default).
    looped_raw = env.get("C4_OPCFG_LOOPED_TRANSFORMER", "0")
    looped = looped_raw not in ("0", "", "false", "False", "no", "off")
    for key, val in env.items():
        if not key.startswith("C4_OPCFG_"):
            continue
        if key == "C4_OPCFG_LOOPED_TRANSFORMER":
            continue                             # handled above (not an <OP>_<AXIS>)
        rest = key[len("C4_OPCFG_"):]           # e.g. "DIV_PRECISION"
        # split on the LAST underscore so multi-word ops would still parse (all
        # current op names are single-token, but ALL/axis split is unambiguous).
        if "_" not in rest:
            raise OpConfigError(f"malformed flag {key!r} (want C4_OPCFG_<OP>_<AXIS>)")
        op_part, _, axis_part = rest.rpartition("_")
        axis = _AXIS_ENV.get(axis_part.upper())
        if axis is None:
            raise OpConfigError(
                f"{key!r}: unknown axis {axis_part!r}; known: {sorted(_AXIS_ENV)}")
        value = _coerce(axis, val)
        op = op_part.upper()
        if op == "ALL":
            base[axis] = value
        elif op in ALL_OPS:
            overrides.setdefault(op, {})[axis] = value
        else:
            raise OpConfigError(f"{key!r}: unknown op {op!r}; known: ALL, {ALL_OPS}")

    cfg = OpConfig(base=AxisConfig(**base), overrides=overrides,
                   looped_transformer=looped)
    if validate_result:
        validate(cfg)
    return cfg


# =========================================================================== #
# The two PARETO-corner named configs (from PRECISION_RADIX_SURFACE.md §6).
# =========================================================================== #
def min_params_config() -> OpConfig:
    """MIN-PARAMS corner (clever_minparam_alu): whole-value fp64 (fp128 for MUL),
    difference-min digit-extract, tied recurrence — ~4 scalars/op, slow fp64
    datapath.  ADD/SUB/DIV/MOD/CMP/frame fp64; MUL fp128.

    ``tied`` recurrence is a LOOPED / Universal-Transformer implementation, so this
    config is declared ``looped_transformer=True``.  Its ~4-scalar param win is a
    UT-checkpoint claim, NOT a stock feed-forward Qwen2 claim — a standard
    feed-forward model must UNROLL (see ``force_standard_feedforward`` /
    ``qwen_fit_solver.account_opconfig``)."""
    ov: Dict[str, Dict[str, object]] = {}
    for op in ALL_OPS:
        ov[op] = dict(precision="fp64", extraction="whole_value",
                      recurrence="tied")
    ov["MUL"] = dict(precision="fp128", extraction="whole_value",
                     recurrence="tied")
    return OpConfig(base=AxisConfig(precision="fp64", radix=10,
                                    extraction="whole_value", recurrence="tied"),
                    overrides=ov, looped_transformer=True)


def min_walltime_config() -> OpConfig:
    """MIN-WALLTIME corner (lowprec_radix_alu): bf16 radix-16, digit-extract, tied
    recurrence — the measured tensor-core sweet spot (~13x faster than the fp64
    cell on MUL/DIV).  radix 16 keeps MUL depth 16 / DIV depth 8 while staying
    exact under bf16's 2^8 ceiling (DIV boundary r^2=256=ceiling; MUL uses fp16 to
    hold its 1904 column peak).

    ``tied`` recurrence is a LOOPED / Universal-Transformer implementation, so this
    config is declared ``looped_transformer=True`` (a UT-style checkpoint, not
    stock feed-forward Qwen2)."""
    ov: Dict[str, Dict[str, object]] = {}
    for op in ALL_OPS:
        ov[op] = dict(precision="bf16", radix=16, extraction="digit_extract",
                      recurrence="tied")
    # MUL's column accumulator (1904 at radix 16) needs fp16's 2^11 ceiling.
    ov["MUL"] = dict(precision="fp16", radix=16, extraction="digit_extract",
                     recurrence="tied")
    return OpConfig(base=AxisConfig(precision="bf16", radix=16,
                                    extraction="digit_extract", recurrence="tied"),
                    overrides=ov, looped_transformer=True)
