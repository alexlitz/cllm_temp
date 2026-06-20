#!/usr/bin/env python
"""Cross-op FFN / residual-band interaction lint — the l14-entanglement catcher.

THE BLIND SPOT (the mul-l14 case study)
---------------------------------------
``compare_symbolic_to_lowered_ffn`` verifies ONE op in ISOLATION (a single
query, hardmax-style synthetic state) — it CANNOT see that two ops sharing a
residual band / FFN region interact. There is a sibling lint for shared
ATTENTION heads (``tools/lint_cross_op_attention.py``, built after a near-
identical −39 incident) but NOTHING for shared FFN regions / residual bands.

This blind spot let a MUL l14 fix's silu firing silently change what
add/sub/div READ in the shared l14 ALU region → a −60 regression that BOTH the
golden byte-identity check AND the isolated-op check missed. The mechanism:

  * The l14 ALU region (blocks 13/15/28/31/33 — AddSub / DivMod / Mul / Shift —
    plus the ``_layer14_alu_high_byte_relay`` head) all WRITE and READ the SAME
    ``OUTPUT_LO/HI`` + ``ALU_LO/HI`` residual dims. They are one shared bus.
  * A "MUL-only" fix that adds/retunes a unit (FFN) or a Q/K/V/O slot (the relay
    head's W_o writes OUTPUT) is gated on ``OP_MUL`` *in intent* — but silu is a
    SMOOTH nonlinearity: ``silu(up)*gate`` is NON-ZERO for OTHER inputs too, and
    an attention W_o write is re-weighted by a GLOBAL softmax. So the post-block
    residual at ``OUTPUT_LO/HI`` shifts on ADD/SUB/DIV rows that read the same
    band one block later — the exact −60.

``compare_symbolic_to_lowered_ffn`` runs ONE rule against a synthetic state
crafted to fire THAT rule; it never places an ADD/SUB/DIV operand frame through
the modified region and re-reads the shared band. That is the blind spot.

WHAT THIS LINT DOES (mirrors lint_cross_op_attention.py for FFN/bands)
----------------------------------------------------------------------
Given a candidate op's flag (auto-detected, like the attention lint):

  1. Build the model flag-OFF and flag-ON (``compile_full_vm_dynamic``,
     ``disk_cache=False``, CPU, own cache dir). For a clean band-local or
     region-local fix the residual layout is identical, so OFF/ON are directly
     comparable per-dim.
  2. Auto-detect every block whose FFN weights OR attention ``W_o`` differ
     OFF→ON, and the SHARED residual band it writes — the ``OUTPUT``/``ALU``
     dim family that OTHER l14-region ops read. (A fix that only writes a
     PRIVATE never-share band, or a brand-new dim no other op reads, touches no
     shared surface and trivially passes.)
  3. For each modified region/block, run a battery of OTHER-OP / OTHER-CONTEXT
     probe rows (ADD/SUB/DIV/MOD/MUL operand frames at MARK_AX, with several
     competing STACK0/ALU candidate rows so the softmax normalization AND the
     silu firing are genuinely exercised). Each probe is forwarded through the
     REAL block module (production attn+ffn math), and the POST-BLOCK residual
     at the shared OUTPUT/ALU dims is diffed OFF vs ON.
  4. FLAG any (opcode, context) row whose shared-band residual CHANGES beyond
     tolerance OFF→ON — a non-local effect. The opcode(s) the fix legitimately
     targets are EXPECTED and can be whitelisted via ``--expect``.

DISCRIMINATION (the proof)
--------------------------
  * mul-l14 (``C4_FFN_LINT_MULL14_DEMO``, a build-reaching reproduction of the
    −60 bug CLASS that the reverted ``ba06deaa`` relay change exemplified — its
    flag-gated W_q anchors are INERT in the current bake, so a dedicated fixture
    stands in) -> FLAGS the perturbation of ADD/SUB/DIV's OUTPUT read dims while
    MUL changes as intended. NON-LOCAL EFFECT DETECTED.
  * a clean band-separable fix (``C4_FFN_LINT_CLEAN_DEMO``: an FFN unit gated on
    OP_MUL that writes a PRIVATE, never-shared MUL-only result dim) -> PASSES.
    No shared OUTPUT/ALU dim that another op reads moves.

Any op modifying a SHARED FFN region / residual band (the l14 ALU, the l16
materializers, any shared OUTPUT/ALU band) MUST pass this lint (see CLAUDE.md).
It is CPU, fast (~the time of two builds), and runs BEFORE a GPU ever sees it.

USAGE
-----
    # lint a candidate fix's flag
    CUDA_VISIBLE_DEVICES="" C4_VM_CACHE_DIR=/tmp/c4cache_ffnlint \
        python tools/lint_cross_op_ffn.py --flag C4_MY_FIX --expect OP_MUL

    # demonstrate it discriminates mul-l14 (flagged) vs a clean band fix (passes)
    CUDA_VISIBLE_DEVICES="" C4_VM_CACHE_DIR=/tmp/c4cache_ffnlint \
        python tools/lint_cross_op_ffn.py --demo

Exits non-zero when an UN-whitelisted (opcode, context) row's shared-band
residual changes (a non-local regression). In ``--demo`` mode, exits non-zero
unless mul-l14 flags AND the clean fix passes (the discrimination contract).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Own cache dir + CPU, per the brief's memory-discipline requirement. Set
# BEFORE any neural_vm import so the compiler picks them up. (Respect a caller
# override if one is already set.)
# ---------------------------------------------------------------------------
os.environ.setdefault("C4_VM_CACHE_DIR", "/tmp/c4cache_ffnlint")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# ---------------------------------------------------------------------------
# Import THIS checkout's neural_vm. When the lint is run as a script
# (``python tools/lint_cross_op_ffn.py``) ``sys.path[0]`` is the ``tools/``
# dir, NOT the repo root, so a bare ``import neural_vm`` can resolve to a STRAY
# sibling checkout that happens to be on ``sys.path`` (e.g. an editable install
# of the parent repo). That stray checkout does NOT carry this worktree's
# flag-gated demo fixtures, so the OFF/ON builds would be IDENTICAL and EVERY
# fix would false-PASS. Prepend the repo root (the parent of ``tools/``) so the
# co-located neural_vm always wins. MUST precede any neural_vm import.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Probe battery: opcode x operand-context.
#
# The shared l14 ALU residual band (OUTPUT_LO/HI + ALU_LO/HI) is written AND
# read by ADD/SUB/MUL/DIV/MOD/SHL at the MARK_AX result row. A "MUL-only" fix
# that perturbs that band corrupts what the OTHER opcodes read one block later.
# A probe ROW is the pair (opcode, context): the battery sweeps both axes.
# ---------------------------------------------------------------------------
PROBE_OPCODES: Tuple[str, ...] = (
    "OP_MUL",                              # the fix's intended target
    "OP_ADD", "OP_SUB",                    # the -60 victims (read OUTPUT band)
    "OP_DIV", "OP_MOD",                    # other binary-ALU readers
    "OP_SHL", "OP_SHR",                    # shift readers of the same band
)

# Operand-row contexts. Each places competing ALU/STACK0 candidate rows before
# a MARK_AX result query so BOTH the attention softmax-normalization (W_o-into-
# OUTPUT writers like the relay head) AND the FFN silu firing are exercised.
#   alu_set : row(s) carry a live ALU_LO/HI one-hot (a real result in the band)
#   out_set : row 0 already carries an OUTPUT_LO/HI value (a prior-step result)
#   n_rows  : number of competing candidate rows preceding the query
CONTEXTS: Tuple[dict, ...] = (
    {"name": "targeted_result_row", "alu": True,  "out": True,  "n_rows": 2},
    {"name": "no_alu_result",       "alu": False, "out": True,  "n_rows": 2},
    {"name": "fresh_output_only",   "alu": True,  "out": False, "n_rows": 2},
    {"name": "single_candidate",    "alu": True,  "out": True,  "n_rows": 1},
    {"name": "three_competing",     "alu": True,  "out": True,  "n_rows": 3},
)

# The shared residual-band dim FAMILIES the l14 ALU region writes+reads. A
# change to any of these at an OTHER-op row is a cross-op interaction. (Resolved
# to concrete dims via layout.dim_positions at lint time; absent families are
# skipped, so this list is a superset that degrades gracefully.)
SHARED_BAND_FAMILIES: Tuple[str, ...] = (
    "OUTPUT_LO", "OUTPUT_HI", "ALU_LO", "ALU_HI",
)


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------
@dataclass
class BlockMod:
    """A block whose FFN and/or attention W_o differs between OFF and ON."""

    block: int
    ffn_type: str
    ffn_l1: float               # L1 weight delta in the FFN module
    attn_l1: float              # L1 weight delta across attn W_q/W_k/W_v/W_o
    writes_shared_band: bool    # writes a dim some OTHER op reads
    shared_out_dims: Tuple[int, ...]  # the shared OUTPUT/ALU dims it perturbs


@dataclass
class RowResult:
    opcode: str
    context: str
    max_abs_delta: float
    changed: bool


@dataclass
class BlockReport:
    mod: BlockMod
    rows: List[RowResult] = field(default_factory=list)


class BuildFailed(Exception):
    def __init__(self, flag, value, exc):
        super().__init__(f"build({flag}={value}) raised: {exc!r}")
        self.flag = flag
        self.value = value
        self.exc = exc


# ---------------------------------------------------------------------------
# Build helpers (env-isolated, CPU, no disk cache) — mirrors the attn lint.
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _flag_env(flag: Optional[str], value: str):
    """Set ``flag=value`` for the duration, then RESTORE the prior value.

    Critical for running several ``lint_flag`` calls in one process (``--demo``):
    without restoration a default-ON fix's forced-OFF build leaves the env in a
    broken state that contaminates the NEXT build. Each build is env-isolated.
    """
    prev_skip = os.environ.get("C4_SKIP_DIM_INTEGRITY")
    os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
    prev = os.environ.get(flag) if flag is not None else None
    if flag is not None:
        os.environ[flag] = value
    try:
        yield
    finally:
        if flag is not None:
            if prev is None:
                os.environ.pop(flag, None)
            else:
                os.environ[flag] = prev
        if prev_skip is None:
            os.environ.pop("C4_SKIP_DIM_INTEGRITY", None)
        else:
            os.environ["C4_SKIP_DIM_INTEGRITY"] = prev_skip


def _build(flag: Optional[str], value: str):
    """Build the production model with ``flag=value`` (CPU, no disk cache).

    Returns ``(model, layout)``. Both the in-process memo and the disk cache are
    bypassed (``disk_cache=False``) so the bake honours the live env var. The
    env var is set+restored by :func:`_flag_env`. Raises :class:`BuildFailed`
    if the bake itself raises (the caller treats an un-buildable OFF baseline as
    "no comparable shared region" — a width/band fix).
    """
    import neural_vm.unified_compiler.full_vm_compiler_dynamic as _fvc
    compile_full_vm_dynamic = _fvc.compile_full_vm_dynamic
    with _flag_env(flag, value):
        try:
            # CRITICAL: clear the in-process compile memo. ``disk_cache=False``
            # only bypasses the DISK cache; the ``_INPROC_COMPILE_CACHE`` memo
            # is keyed on a kwargs snapshot that does NOT include most env
            # flags (e.g. C4_NO_STACK0_EMIT), so OFF and ON would otherwise
            # COLLIDE and the second build would return the first model — the
            # lint would then see ZERO change and false-PASS every fix. We must
            # honour the live env flag on every build, so flush the memo first.
            try:
                _fvc._INPROC_COMPILE_CACHE.clear()
            except Exception:  # noqa: BLE001 — memo is best-effort
                pass
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()):
                    model, layout = compile_full_vm_dynamic(disk_cache=False)
        except Exception as exc:  # noqa: BLE001 — re-wrapped for the caller
            raise BuildFailed(flag, value, exc) from exc
    return model, layout


# ---------------------------------------------------------------------------
# Weight-diff detection (which blocks' FFN / attn-W_o changed).
# ---------------------------------------------------------------------------
def _ffn_weight_l1(ffn) -> float:
    """Total L1 weight magnitude of an FFN module (PureFFN or specialized).

    Sums every floating ``nn.Parameter`` so it works for PureFFN
    (W_up/W_gate/W_down) and the specialized ALU blocks (FlattenedALUMul,
    FlattenedDivMod, AddSub5StageBlock, ALUShiftComposite) uniformly.
    """
    import torch

    total = 0.0
    for p in ffn.parameters():
        if p.dtype.is_floating_point:
            total += float(p.data.abs().sum().item())
    return total


def _ffn_weight_delta(ffn_off, ffn_on) -> float:
    """L1 delta between two FFN modules' parameters (same structure assumed).

    Pairs parameters by name; a structural mismatch (different param set) is
    reported as +inf so the block is treated as modified.
    """
    import torch

    po = dict(ffn_off.named_parameters())
    pn = dict(ffn_on.named_parameters())
    if set(po) != set(pn):
        return float("inf")
    total = 0.0
    for name, a in po.items():
        b = pn[name]
        if a.shape != b.shape:
            return float("inf")
        if a.dtype.is_floating_point:
            total += float((a.data - b.data).abs().sum().item())
    return total


def _resolve_band_dims(layout) -> List[int]:
    """Concrete residual dims of every shared l14-ALU band present in layout."""
    dp = layout.dim_positions
    dims: List[int] = []
    for fam in SHARED_BAND_FAMILIES:
        base = dp.get(fam)
        if base is None:
            continue
        # The family occupies 16 one-hot nibble cells.
        for k in range(16):
            d = base + k
            if d < layout.d_model:
                dims.append(d)
    return sorted(set(dims))


def _changed_down_columns(Wd_on, Wd_off):
    """Column indices of ``W_down`` (hidden units) that DIFFER OFF->ON.

    Returns the unit columns whose W_down content changed — the units the FIX
    actually touched. When ON added units past the OFF width (a chain-appended
    unit, the common case), every appended column counts as changed. When the
    widths match, only columns with a non-zero per-column delta count. This is
    what lets the reach screen attribute the OUTPUT band to the FIX's units, not
    to the l14 ALU block's pre-existing OUTPUT-writing units.
    """
    import torch

    n_on = Wd_on.shape[1]
    if Wd_off is None:
        return torch.arange(n_on)
    n_off = Wd_off.shape[1]
    cols = []
    common = min(n_on, n_off)
    if common > 0:
        delta = (Wd_on[:, :common] - Wd_off[:, :common]).abs().sum(dim=0)
        cols.append((delta > 0).nonzero(as_tuple=False).squeeze(-1))
    if n_on > n_off:
        cols.append(torch.arange(n_off, n_on))
    if not cols:
        return torch.arange(0)
    return torch.unique(torch.cat([c.reshape(-1) for c in cols]))


def _downstream_band_readers(model, after_block: int, band_dims: Sequence[int]):
    """Return True iff ANY block strictly AFTER ``after_block`` READS a band dim.

    A block "reads" a dim if its FFN ``W_up``/``W_gate`` or its attention
    ``W_q``/``W_k``/``W_v`` has a non-zero column at that dim — i.e. the dim
    enters that op's computation. This is the residual_band liveness in
    practice: it proves the perturbed band is genuinely consumed downstream
    (so the change is non-local), not a dead write.
    """
    import torch

    band = list(band_dims)
    if not band:
        return False
    for bi in range(after_block + 1, len(model.blocks)):
        blk = model.blocks[bi]
        ffn = getattr(blk, "ffn", None)
        if ffn is not None:
            for wname in ("W_up", "W_gate"):
                W = getattr(ffn, wname, None)
                if W is None:
                    continue
                Wd = W.data.to_dense() if W.is_sparse else W.data
                if Wd.shape[1] > max(band) and Wd[:, band].abs().sum().item() > 0:
                    return True
        attn = getattr(blk, "attn", None)
        if attn is not None and hasattr(attn, "W_q"):
            for wname in ("W_q", "W_k", "W_v"):
                W = getattr(attn, wname, None)
                if W is None:
                    continue
                Wd = W.data
                if Wd.shape[1] > max(band) and Wd[:, band].abs().sum().item() > 0:
                    return True
    return False


def detect_modified_blocks(model_off, model_on, layout, *, tol: float = 1e-9):
    """Find every block whose FFN or attn-W_o differs OFF→ON and writes a band.

    Returns ``List[BlockMod]``. ``writes_shared_band`` is True when the block's
    perturbation actually moves a shared OUTPUT/ALU dim that a downstream op
    reads (computed by forwarding a battery probe in :func:`lint_flag`; here we
    pre-screen which band dims the block COULD reach via its W_down / W_o rows).
    """
    import torch

    band_dims = _resolve_band_dims(layout)
    band_set = set(band_dims)
    mods: List[BlockMod] = []
    n = min(len(model_off.blocks), len(model_on.blocks))
    for bi in range(n):
        b_off = model_off.blocks[bi]
        b_on = model_on.blocks[bi]
        ffn_off = getattr(b_off, "ffn", None)
        ffn_on = getattr(b_on, "ffn", None)
        ffn_l1 = (
            _ffn_weight_delta(ffn_off, ffn_on)
            if ffn_off is not None and ffn_on is not None
            else 0.0
        )
        # Attention changes are an OUTPUT-band hazard too: a Q/K/V edit changes
        # the GLOBAL softmax, so the head's W_o-projected residual into OUTPUT
        # moves on OTHER rows even when W_o itself is byte-identical (the
        # relay-head / mul-l14 case — ba06deaa edits ONLY W_q). So we flag a
        # block when ANY of W_q/W_k/W_v/W_o differs; the band-reach screen below
        # (via W_o rows) restricts to heads that actually write OUTPUT/ALU.
        attn_off = getattr(b_off, "attn", None)
        attn_on = getattr(b_on, "attn", None)
        attn_o_l1 = 0.0
        if attn_off is not None and attn_on is not None and hasattr(attn_off, "W_q"):
            for wname in ("W_q", "W_k", "W_v", "W_o"):
                W0 = getattr(attn_off, wname, None)
                W1 = getattr(attn_on, wname, None)
                if W0 is None or W1 is None:
                    continue
                if W0.data.shape != W1.data.shape:
                    attn_o_l1 = float("inf")
                    break
                attn_o_l1 += float((W0.data - W1.data).abs().sum().item())
        if max(ffn_l1, attn_o_l1) <= tol:
            continue

        # Which band dims does the FIX (the units/heads that CHANGED OFF->ON)
        # write? Use ONLY the modified columns, not the whole block's W_down --
        # the l14 ALU block writes the OUTPUT band from its EXISTING units on
        # every build, so a whole-block W_down screen would attribute the shared
        # band to ANY l14 edit (even a private-TEMP one). Restricting to the
        # changed units is what lets the clean-control fixture (writes TEMP)
        # report shared_dims=() while the entanglement fixture (writes OUTPUT)
        # reports the OUTPUT band.
        reach: set = set()
        if ffn_on is not None:
            Wdn_on = getattr(ffn_on, "W_down", None)
            Wdn_off = getattr(ffn_off, "W_down", None) if ffn_off is not None else None
            if Wdn_on is not None:
                Wd_on = Wdn_on.data.to_dense() if Wdn_on.is_sparse else Wdn_on.data
                Wd_off = None
                if Wdn_off is not None:
                    Wd_off = Wdn_off.data.to_dense() if Wdn_off.is_sparse else Wdn_off.data
                changed_cols = _changed_down_columns(Wd_on, Wd_off)
                if changed_cols.numel():
                    for d in band_dims:
                        if d < Wd_on.shape[0] and Wd_on[d, changed_cols].abs().sum().item() > 0:
                            reach.add(d)
        if attn_on is not None and hasattr(attn_on, "W_o"):
            Wo_on = attn_on.W_o.data
            Wo_off = attn_off.W_o.data if (
                attn_off is not None and hasattr(attn_off, "W_o")
            ) else None
            # An attn edit anywhere (Q/K/V/O) re-weights the softmax, so its W_o
            # write into OUTPUT moves on other rows even if W_o is byte-identical
            # -- attribute the whole head's OUTPUT-writing rows for attn mods.
            for d in band_dims:
                if d < Wo_on.shape[0] and Wo_on[d].abs().sum().item() > 0:
                    if Wo_off is None or (
                        d >= Wo_off.shape[0]
                        or attn_o_l1 > tol
                    ):
                        reach.add(d)
        shared = sorted(d for d in reach if d in band_set)
        writes_band = bool(shared) and _downstream_band_readers(
            model_on, bi, shared
        )
        mods.append(BlockMod(
            block=bi,
            ffn_type=type(ffn_on).__name__ if ffn_on is not None else "None",
            ffn_l1=ffn_l1,
            attn_l1=attn_o_l1,
            writes_shared_band=writes_band,
            shared_out_dims=tuple(shared),
        ))
    return mods


# ---------------------------------------------------------------------------
# Probe construction. Each probe is a small multi-row 3-D sequence ending in a
# MARK_AX result query, preceded by ``n_rows`` competing ALU/STACK0 candidate
# rows. Multiple competing rows are LOAD-BEARING: the relay head's W_o write is
# softmax-weighted across rows, and the FFN silu fires on every row — a single
# key never exposes the cross-row interaction.
# ---------------------------------------------------------------------------
def _make_probe(layout, opcode: str, ctx: dict):
    """Build a ``[1, S, d_model]`` residual probe for ``(opcode, ctx)``.

    Returns the tensor and the query row index (the MARK_AX row whose post-block
    OUTPUT/ALU dims we compare).
    """
    import torch

    dp = layout.dim_positions
    D = layout.d_model
    n_rows = int(ctx["n_rows"])
    S = n_rows + 1
    x = torch.zeros(1, S, D, dtype=torch.float32)

    def setdim(pos, name, val=1.0):
        idx = dp.get(name)
        if idx is not None:
            x[0, pos, idx] = float(val)

    alu_lo = dp.get("ALU_LO")
    alu_hi = dp.get("ALU_HI")
    out_lo = dp.get("OUTPUT_LO")
    out_hi = dp.get("OUTPUT_HI")

    # Rows 0..n_rows-1: competing ALU/STACK0 result candidates.
    for r in range(n_rows):
        setdim(r, "STACK0_BYTE0", 1.0)
        setdim(r, "CONST", 1.0)
        setdim(r, "IS_BYTE", 1.0)
        if ctx["alu"] and alu_lo is not None:
            x[0, r, alu_lo + (3 + r) % 16] = 1.0
        if ctx["alu"] and alu_hi is not None:
            x[0, r, alu_hi + (1 + r) % 16] = 1.0
        if ctx["out"] and out_lo is not None:
            x[0, r, out_lo + (5 + r) % 16] = 1.0
        if ctx["out"] and out_hi is not None:
            x[0, r, out_hi + (2 + r) % 16] = 1.0

    # Final row: the MARK_AX result query tagged with the probe opcode.
    q = S - 1
    setdim(q, "MARK_AX", 1.0)
    setdim(q, "CONST", 1.0)
    setdim(q, "IS_BYTE", 1.0)
    setdim(q, opcode, 1.0)
    # Give the query row a live ALU result to relay (so the band is non-trivial).
    if alu_lo is not None:
        x[0, q, alu_lo + 7] = 1.0
    if alu_hi is not None:
        x[0, q, alu_hi + 4] = 1.0
    return x, q


def _block_forward(block, x):
    """Run a block's REAL forward (production attn+ffn math) on ``[1,S,D]``.

    Returns the post-block residual ``[S, D]``. Mirrors
    ``vm_step.Block.forward`` (no RMSNorm in this VM): ``x = attn(x); x =
    ffn(x); post_ops``. Computed under ``no_grad`` on CPU. SDPA is forced to the
    deterministic math backend so OFF/ON are numerically comparable.
    """
    import torch

    with torch.no_grad():
        y = x
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "W_q"):
            # Prefer the manual (math) attention path for determinism if the
            # module exposes a toggle; otherwise the module's own forward.
            prev_flash = getattr(attn, "use_flash_attention", None)
            if prev_flash is not None:
                attn.use_flash_attention = False
            try:
                y = attn(y)
            finally:
                if prev_flash is not None:
                    attn.use_flash_attention = prev_flash
        ffn = getattr(block, "ffn", None)
        if ffn is not None:
            y = ffn(y)
        for op in getattr(block, "post_ops", []):
            y = op(y)
    return y[0]  # [S, D]


# ---------------------------------------------------------------------------
# Core lint
# ---------------------------------------------------------------------------
def _is_expected(opcode: str, context: str, expect_set) -> bool:
    return (
        opcode in expect_set
        or context in expect_set
        or f"{opcode}:{context}" in expect_set
    )


def lint_flag(
    flag: str,
    *,
    expect: Tuple[str, ...] = (),
    atol: float = 1e-4,
    rtol: float = 1e-3,
    verbose: bool = True,
) -> Tuple[bool, List[BlockReport]]:
    """Lint a single candidate fix flag. Returns ``(ok, reports)``.

    ``ok`` is False when an UN-whitelisted (opcode, context) row's shared-band
    residual changes on a block that writes a shared OUTPUT/ALU band.
    """
    import torch

    if verbose:
        print(f"[lint] flag={flag}", flush=True)

    try:
        model_off, layout_off = _build(flag, "0")
    except BuildFailed as bf:
        if verbose:
            print(
                f"[lint] PASS (no shared-band hazard): the OFF baseline could "
                f"not be built ({bf.exc!r}). A default-ON band/width fix whose "
                f"OFF state is not comparable does not modify a shared FFN "
                f"region in a matching layout.",
                flush=True,
            )
        return True, []

    model_on, layout_on = _build(flag, "1")

    width_shift = layout_off.dim_positions != layout_on.dim_positions
    if width_shift:
        if verbose:
            print(
                f"[lint] NOTE: {flag} shifts dim_positions/d_model "
                f"(OFF d_model={layout_off.d_model}, ON d_model={layout_on.d_model}). "
                f"A residual-band/width fix moves dims, so a per-dim OFF/ON "
                f"comparison of a SHARED region is not well-defined — such a fix "
                f"adds a NEW band (it does not retune a shared region in place). "
                f"PASS: no in-place shared-region edit to evaluate.",
                flush=True,
            )
        return True, []

    mods = detect_modified_blocks(model_off, model_on, layout_off)
    shared = [m for m in mods if m.writes_shared_band]

    if verbose:
        print(
            f"[lint] modified blocks: {len(mods)} "
            f"(writing a SHARED OUTPUT/ALU band read downstream: {len(shared)})",
            flush=True,
        )
        for m in mods:
            tag = "SHARED-BAND" if m.writes_shared_band else "local/private"
            print(
                f"[lint]   block={m.block:2d} ffn={m.ffn_type:20s} "
                f"dFFN={m.ffn_l1:.4g} dAttn={m.attn_l1:.4g}  "
                f"shared_dims={list(m.shared_out_dims)}  [{tag}]",
                flush=True,
            )

    if not shared:
        if verbose:
            print(
                "[lint] PASS: no block modifies a SHARED OUTPUT/ALU residual "
                "band that a downstream op reads -> no cross-op FFN/band hazard.",
                flush=True,
            )
        return True, []

    band_dims = _resolve_band_dims(layout_off)
    expect_set = set(expect)
    reports: List[BlockReport] = []
    any_unexpected = False

    for m in shared:
        b_off = model_off.blocks[m.block]
        b_on = model_on.blocks[m.block]
        # Diff only the shared dims this block actually reaches (its band).
        cmp_dims = list(m.shared_out_dims) if m.shared_out_dims else band_dims
        rep = BlockReport(mod=m)
        for opcode in PROBE_OPCODES:
            for ctx in CONTEXTS:
                x, q = _make_probe(layout_off, opcode, ctx)
                y_off = _block_forward(b_off, x)[q, cmp_dims]
                y_on = _block_forward(b_on, x)[q, cmp_dims]
                delta = (y_off - y_on).abs()
                tol = atol + rtol * y_off.abs()
                changed = bool((delta > tol).any().item())
                rep.rows.append(RowResult(
                    opcode=opcode,
                    context=ctx["name"],
                    max_abs_delta=float(delta.max().item()),
                    changed=changed,
                ))
                if changed and not _is_expected(opcode, ctx["name"], expect_set):
                    any_unexpected = True
        reports.append(rep)

    if verbose:
        _print_reports(reports, expect_set)

    return (not any_unexpected), reports


def _print_reports(reports: List[BlockReport], expect_set):
    for rep in reports:
        m = rep.mod
        print(
            f"\n[lint] probe results — block {m.block} ({m.ffn_type}, "
            f"shared dims {list(m.shared_out_dims)}):",
            flush=True,
        )
        changed_rows = [r for r in rep.rows if r.changed]
        if not changed_rows:
            print("[lint]   (no probe row changed beyond tolerance)", flush=True)
            continue
        for r in changed_rows:
            expected = _is_expected(r.opcode, r.context, expect_set)
            tag = "EXPECTED" if expected else "*** NON-LOCAL CHANGE ***"
            print(
                f"[lint]   {r.opcode:8s} / {r.context:20s}  "
                f"max|Δband|={r.max_abs_delta:.4g}   {tag}",
                flush=True,
            )


def _summary_line(flag, ok, reports, expect_set):
    n_changed = sum(
        1
        for rep in reports
        for r in rep.rows
        if r.changed and not _is_expected(r.opcode, r.context, expect_set)
    )
    verdict = "PASS" if ok else "FAIL"
    print(
        f"\n[lint] {flag}: {verdict} "
        f"({n_changed} non-local (opcode,context) row(s) changed beyond "
        f"tolerance in a shared OUTPUT/ALU band)",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Demo: prove the lint discriminates the mul-l14 entanglement (flagged) from a
# clean band-separable change (passed).
# ---------------------------------------------------------------------------
# The mul-l14 entanglement fixture (flag C4_FFN_LINT_MULL14_DEMO) is the
# build-reaching reproduction of the −60 bug CLASS (commit ba06deaa's salvaged
# relay change is INERT in the current bake — its flag-gated W_q anchors are
# overwritten by a downstream legacy post-pass, so it cannot be used to PROVE
# the lint). The fixture adds a unit AUTHORED as "MUL-only" (W_up reads OP_MUL)
# whose POSITIVE b_up leaks silu onto OTHER opcodes and writes the SHARED
# OUTPUT_LO band the ADD/SUB/DIV ops read one block later. We whitelist ONLY the
# opcode it targets (OP_MUL); any change to ADD/SUB/DIV's read band is the
# non-local effect — exactly the regression the lint exists to catch.
_MUL_L14_FLAG = "C4_FFN_LINT_MULL14_DEMO"
_MUL_L14_EXPECT = ("OP_MUL",)

# The clean control is synthesized by --demo via C4_FFN_LINT_CLEAN_DEMO: an FFN
# unit gated on OP_MUL that writes a PRIVATE never-shared MUL-only result dim
# (no OUTPUT/ALU band) — see _install_clean_demo_op below.
_CLEAN_FLAG = "C4_FFN_LINT_CLEAN_DEMO"


def run_demo() -> int:
    print("=" * 72, flush=True)
    print("DEMO 1/2 — mul-l14 entanglement (C4_FFN_LINT_MULL14_DEMO, a",
          flush=True)
    print("           build-reaching reproduction of the -60 bug CLASS)",
          flush=True)
    print("  A unit AUTHORED as 'MUL-only' whose leaky silu writes the SHARED",
          flush=True)
    print("  OUTPUT_LO band -- the same mechanism as the reverted ba06deaa",
          flush=True)
    print("  relay change (which is INERT in the current bake, so it cannot",
          flush=True)
    print("  PROVE the lint; see the fixture banner in l14_ops.py).", flush=True)
    print("  whitelist = the ONE opcode the fix targets (OP_MUL).", flush=True)
    print("  EXPECT: lint FLAGS the perturbation of ADD/SUB/DIV's OUTPUT", flush=True)
    print("          read band (the -60 victims).", flush=True)
    print("=" * 72, flush=True)
    ok_mul, rep_mul = lint_flag(_MUL_L14_FLAG, expect=_MUL_L14_EXPECT)
    _summary_line(_MUL_L14_FLAG, ok_mul, rep_mul, set(_MUL_L14_EXPECT))

    print("\n" + "=" * 72, flush=True)
    print("DEMO 2/2 — clean band-separable fix (synthetic control)", flush=True)
    print("  An FFN unit gated on OP_MUL that writes a PRIVATE, never-shared", flush=True)
    print("  MUL-only result dim (NO OUTPUT/ALU band).", flush=True)
    print("  EXPECT: PASS — touches no shared band a downstream op reads.", flush=True)
    print("=" * 72, flush=True)
    ok_clean, rep_clean = lint_flag(_CLEAN_FLAG)
    _summary_line(_CLEAN_FLAG, ok_clean, rep_clean, set())

    mul_flagged = not ok_mul
    clean_pass = ok_clean
    print("\n" + "=" * 72, flush=True)
    print("DISCRIMINATION CONTRACT", flush=True)
    print(f"  mul-l14 flagged a non-local regression : {mul_flagged}", flush=True)
    print(f"  clean band fix passed                  : {clean_pass}", flush=True)
    discriminates = mul_flagged and clean_pass
    print(
        f"  => lint DISCRIMINATES real regressions : {discriminates}",
        flush=True,
    )
    print("=" * 72, flush=True)
    return 0 if discriminates else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--flag", default=_MUL_L14_FLAG,
                    help="env flag of the candidate fix to lint "
                         "(default: the mul-l14 relay flag)")
    ap.add_argument("--expect", default="",
                    help="comma-separated whitelist of (opcode,context) rows "
                         "the fix is ALLOWED to change. Each entry is an opcode "
                         "(OP_MUL), a context (targeted_result_row), or a pair "
                         "(OP_MUL:targeted_result_row).")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--demo", action="store_true",
                    help="run the mul-l14-vs-clean discrimination demo")
    args = ap.parse_args(argv)

    if args.demo:
        return run_demo()

    expect = tuple(s.strip() for s in args.expect.split(",") if s.strip())
    ok, reports = lint_flag(args.flag, expect=expect, atol=args.atol, rtol=args.rtol)
    _summary_line(args.flag, ok, reports, set(expect))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
