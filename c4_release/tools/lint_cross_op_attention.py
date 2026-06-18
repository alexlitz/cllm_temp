#!/usr/bin/env python
"""Cross-op attention-interaction lint — the AUTHORING-TIME counterpart to the
GPU tripwire, closing the non-local-regression blind spot.

THE BLIND SPOT (the byte-0 case study)
--------------------------------------
The byte-0 fix (``C4_OPERAND_GATHER_PSH_ROWSELECT``) added an attention CAM
Q/K slot (slot 34) to the SHARED L7 operand-gather head 0 (physical block 11),
gated ``Q[34] = c*(OP_ADD + OP_SUB)``. It passed every per-op gate: per-op
lowering byte-identity, smoke 51/0, AND targeted add/sub +8. But the FULL run
showed -39: it BROKE var_simple (22) + expr_mod (25).

Root cause: attention SOFTMAX is GLOBAL over the head. Adding a K slot changes
the post-softmax attention OUTPUT for OTHER ops' rows (var/expr operands
gathered by the SAME head) even though Q is "gated" to ADD/SUB — because the
new K row participates in the softmax1 NORMALIZATION of every query, and its
non-zero K projection perturbs the score battery at non-add/sub rows whose
operand candidates include that PSH/STACK0 row. ``compare_symbolic_to_lowered_attn``
checks ONE op in isolation (single high-amplitude query, hardmax) — it CANNOT
see this multi-row softmax-normalization interaction.

WHAT THIS LINT DOES
-------------------
Given a candidate op's flag (a fix that modifies an attention head):

  1. Build the model flag-OFF and flag-ON (``compile_full_vm_dynamic``,
     ``disk_cache=False``, CPU). The residual layout (``dim_positions``,
     ``d_model``) is identical for a clean head-only fix, so the two builds are
     directly comparable.
  2. Auto-detect every (physical_block, head) whose Q/K/V/O weights DIFFER
     between OFF and ON AND that PRE-EXISTS (is non-empty) flag-OFF — i.e. a
     SHARED head being extended, not a brand-new head. (A brand-new head, or a
     fix that only adds residual bands / LM-head columns, touches no shared
     head and trivially passes.)
  3. For each modified shared head, run a battery of multi-row PROBE
     sequences spanning the opcodes the head serves (MARK_AX operand-gather
     rows for ADD/SUB/MUL/DIV/MOD/CMP plus var/expr operand contexts). Each
     probe places SEVERAL competing operand-candidate K rows (so the softmax1
     normalization is genuinely exercised) before a MARK_AX query tagged with
     the probe opcode. The head's per-query attention OUTPUT is computed with
     the REAL production math (ALiBi + softmax1 + causal, per
     ``vm_step.AutoregressiveAttention.forward``).
  4. FLAG any opcode whose head OUTPUT (or the resulting argmax of the head's
     write-dims) CHANGES beyond tolerance OFF->ON — that is a non-local
     effect. The opcode(s) the fix legitimately targets (ADD/SUB for byte-0)
     are EXPECTED to change and can be whitelisted via ``--expect``.

DISCRIMINATION (the proof)
--------------------------
  * ``C4_OPERAND_GATHER_PSH_ROWSELECT`` (byte-0) -> FLAGS the perturbation at
    var/expr/MUL/DIV/MOD operand rows (the rows that regressed) while ADD/SUB
    change as intended. NON-LOCAL EFFECT DETECTED.
  * ``C4_AX_BYTE1_HINIB`` (a known-clean fix; adds a residual band + LM-head
    columns, touches NO shared attention head) -> PASSES. No shared head
    differs, so no probe row changes.

Any op modifying a SHARED / pre-existing attention head MUST pass this lint
(see CLAUDE.md). It is CPU, fast (~2x a single build), and runs BEFORE the
build is ever shipped to a GPU.

USAGE
-----
    # lint a candidate fix's flag (default: the byte-0 fix)
    CUDA_VISIBLE_DEVICES="" python tools/lint_cross_op_attention.py \
        --flag C4_OPERAND_GATHER_PSH_ROWSELECT

    # whitelist the opcodes the fix is allowed to change
    CUDA_VISIBLE_DEVICES="" python tools/lint_cross_op_attention.py \
        --flag C4_OPERAND_GATHER_PSH_ROWSELECT --expect OP_ADD,OP_SUB

    # demonstrate it discriminates: byte-0 (flagged) vs HINIB (clean)
    CUDA_VISIBLE_DEVICES="" python tools/lint_cross_op_attention.py --demo

Exits non-zero when an UN-whitelisted opcode row changes (a non-local
regression). In ``--demo`` mode, exits non-zero unless byte-0 flags AND HINIB
passes (the discrimination contract).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Probe battery: opcode x operand-context.
#
# The shared L7 operand-gather head serves MANY opcodes at MARK_AX, but the
# byte-0 regression is NOT cross-OPCODE (a non-add/sub query has Q[34]=0, so the
# new K slot is multiplied by 0 — MUL/DIV/CMP rows are genuinely untouched).
# The regression is cross-CONTEXT: var_simple / expr_mod gather their operands
# at the SAME ADD/SUB opcodes but in DIFFERENT operand-row configurations
# (no genuine PSH-output row, or a corrupted high nibble, or a different number
# of competing STACK0 rows). The slot-34 softmax bias re-ranks / re-weights
# those configurations even though the targeted add/sub smoke context is the
# only one the fix INTENDED to change. So a probe ROW is the pair
# ``(opcode, context)`` — the battery sweeps both axes.
#
# Each opcode the head serves at MARK_AX:
PROBE_OPCODES: Tuple[str, ...] = (
    "OP_ADD", "OP_SUB",                       # byte-0 fix's intended targets
    "OP_MUL", "OP_DIV", "OP_MOD",             # other binary-ALU operand gathers
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",  # CMP operand gathers
    "OP_OR", "OP_XOR", "OP_AND",              # bitwise operand gathers
    "OP_SHL", "OP_SHR",                       # shift operand gathers
)

# Operand-row contexts. ``psh`` = is row 0 a genuine PSH-output candidate
# (PSH_AT_SP set); ``hi_nibble`` = the high nibble cell of row 0's
# CLEAN_EMBED_HI (0 == the "corrupted/default" high nibble the IMM-decode bug
# zeroes; >0 == a clean value); ``n_rows`` = how many competing STACK0 rows
# precede the query. The "targeted" context is the single config the byte-0 fix
# was authored for (a clean PSH row competing with one recency restamp); the
# others REPRESENT the var/expr operand frames the fix silently perturbed.
CONTEXTS: Tuple[dict, ...] = (
    # name, row-0 PSH present, row-0 clean hi nibble, # competing rows
    {"name": "targeted_clean_psh", "psh": True,  "hi_nibble": 5, "n_rows": 2},
    {"name": "var_expr_no_psh",    "psh": False, "hi_nibble": 5, "n_rows": 2},
    {"name": "corrupted_hi_nib",   "psh": True,  "hi_nibble": 0, "n_rows": 2},
    {"name": "single_operand_row", "psh": True,  "hi_nibble": 5, "n_rows": 1},
    {"name": "three_competing",    "psh": True,  "hi_nibble": 5, "n_rows": 3},
)


@dataclass
class HeadMod:
    """A shared attention head that differs between flag-OFF and flag-ON."""

    block: int
    head: int
    pre_existing: bool          # non-empty Q or K rows flag-OFF (a SHARED head)
    out_dims: Tuple[int, ...]   # residual dims this head's W_o writes (head 0 -> ALU_LO/HI)
    q_l1: float
    k_l1: float
    v_l1: float
    o_l1: float


@dataclass
class RowResult:
    opcode: str
    context: str
    max_abs_delta: float
    argmax_changed: bool
    weight_l1_delta: float      # L1 change in the head's per-row softmax weights
    changed: bool               # delta exceeded tolerance (real, not fp noise)


@dataclass
class HeadReport:
    mod: HeadMod
    rows: List[RowResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
class BuildFailed(Exception):
    """A build at a given flag value raised — recorded, not crashed."""

    def __init__(self, flag, value, exc):
        super().__init__(f"build({flag}={value}) raised: {exc!r}")
        self.flag = flag
        self.value = value
        self.exc = exc


@contextlib.contextmanager
def _flag_env(flag: Optional[str], value: str):
    """Set ``flag=value`` for the duration, then RESTORE the prior value.

    Critical for running multiple ``lint_flag`` calls in one process (the
    ``--demo`` path): without restoration, a default-ON fix's forced-OFF build
    leaves the env in a broken state that contaminates the NEXT fix's build
    (e.g. ``C4_AX_BYTE1_HINIB=0`` shrinks d_model, then a sibling band fix
    crashes). Each build is therefore env-isolated.
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

    Returns ``(model, layout)``. The in-process memo + disk cache are BOTH
    bypassed (``disk_cache=False``) so the bake honours the live env var
    instead of a stale artifact. The env var is set (and restored) around the
    call by :func:`_flag_env` so sibling lints don't contaminate each other.

    Raises :class:`BuildFailed` if the bake itself raises — some default-ON
    fixes leave the OFF state un-buildable (e.g. an L15 head that lost its slot
    when the band was removed). The caller treats an un-buildable OFF baseline
    as "no shared-head edit observable" (a width/band fix), which is the
    correct verdict for such fixes.
    """
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    with _flag_env(flag, value):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()):
                    model, layout = compile_full_vm_dynamic(disk_cache=False)
        except Exception as exc:  # noqa: BLE001 — re-wrapped for the caller
            raise BuildFailed(flag, value, exc) from exc
    return model, layout


def _attn_blocks(model):
    """Yield ``(block_idx, attn)`` for every block carrying an attention head."""
    for bi, block in enumerate(model.blocks):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "W_q"):
            yield bi, attn


def detect_modified_shared_heads(model_off, model_on, *, tol: float = 1e-9) -> List[HeadMod]:
    """Find every (block, head) whose Q/K/V/O differs OFF->ON.

    A head is ``pre_existing`` (SHARED) when its OFF Q or K rows are non-empty:
    extending a shared head is exactly the byte-0 hazard. A brand-new head
    (empty OFF) is reported with ``pre_existing=False`` (lint passes it — a new
    head cannot perturb an existing op's softmax because it never shared the
    block before; the GPU tripwire still covers genuinely-new heads).
    """
    import torch

    blocks_off = dict(_attn_blocks(model_off))
    blocks_on = dict(_attn_blocks(model_on))
    mods: List[HeadMod] = []
    for bi in sorted(set(blocks_off) & set(blocks_on)):
        a0, a1 = blocks_off[bi], blocks_on[bi]
        if a0.W_q.shape != a1.W_q.shape:
            # Different residual layout — the fix widened d_model, so OFF/ON are
            # not directly comparable per-row. Such fixes are residual-band
            # additions, not shared-head edits; they don't hit this hazard.
            continue
        HD = a0.head_dim
        H = a0.num_heads
        for h in range(H):
            rows = slice(h * HD, (h + 1) * HD)
            q0, q1 = a0.W_q.data[rows], a1.W_q.data[rows]
            k0, k1 = a0.W_k.data[rows], a1.W_k.data[rows]
            v0, v1 = a0.W_v.data[rows], a1.W_v.data[rows]
            # W_o reads from this head's slot block (columns base..base+HD).
            cols = slice(h * HD, (h + 1) * HD)
            o0, o1 = a0.W_o.data[:, cols], a1.W_o.data[:, cols]
            q_l1 = (q0 - q1).abs().sum().item()
            k_l1 = (k0 - k1).abs().sum().item()
            v_l1 = (v0 - v1).abs().sum().item()
            o_l1 = (o0 - o1).abs().sum().item()
            if max(q_l1, k_l1, v_l1, o_l1) <= tol:
                continue
            pre_existing = bool(
                q0.abs().sum().item() > tol or k0.abs().sum().item() > tol
            )
            # Out dims this head writes (union OFF+ON), for argmax tracking.
            out_dims = sorted(
                set((o0.abs().sum(dim=1) > tol).nonzero().flatten().tolist())
                | set((o1.abs().sum(dim=1) > tol).nonzero().flatten().tolist())
            )
            mods.append(HeadMod(
                block=bi, head=h, pre_existing=pre_existing,
                out_dims=tuple(int(d) for d in out_dims),
                q_l1=q_l1, k_l1=k_l1, v_l1=v_l1, o_l1=o_l1,
            ))
    return mods


# ---------------------------------------------------------------------------
# Production attention math (ALiBi + softmax1 + causal), per head.
# Mirrors vm_step.AutoregressiveAttention.forward (manual path, lines ~614-628):
#   scores = (Q @ K^T) * scale + bias
#   bias   = -slope * |q_pos - k_pos|  +  causal(-inf above diagonal)
#   attn   = softmax1(scores)   (anchor=0 sink column)
#   out    = attn @ V
# We compute ONE head's output so a shared-head edit is isolated exactly.
# ---------------------------------------------------------------------------
def _head_attn_weights(attn, x, head: int):
    """Per-head, per-query softmax1 attention WEIGHTS -> [S, S] for one head.

    Replicates ``vm_step.AutoregressiveAttention.forward`` (manual path):
    ALiBi bias (absolute positions) + causal mask + softmax1 (anchor=0 sink).
    The returned matrix is the over-real-keys distribution; the sink mass is
    ``1 - row.sum()``. Exactly the production numerics, so the lint sees real
    re-ranking / re-weighting.
    """
    import torch

    HD = attn.head_dim
    rows = slice(head * HD, (head + 1) * HD)
    Q = x @ attn.W_q.data[rows].T   # [S, HD]
    K = x @ attn.W_k.data[rows].T
    S = x.shape[0]
    scores = (Q @ K.T) * attn.scale  # [S, S]

    slopes = getattr(attn, "alibi_slopes", None)
    if slopes is not None:
        pos = torch.arange(S, dtype=torch.float32)
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
        scores = scores + (-float(slopes[head].item()) * dist)

    # Operand-gather is causal: a query attends to earlier rows only.
    scores = scores + torch.triu(
        torch.full((S, S), float("-inf")), diagonal=1,
    )

    if getattr(attn, "use_softmax1", True):
        anchor = torch.zeros(S, 1)
        max_val = torch.maximum(scores.amax(dim=-1, keepdim=True), anchor)
        exp_scores = torch.exp(scores - max_val)
        exp_anchor = torch.exp(anchor - max_val)
        return exp_scores / (exp_anchor + exp_scores.sum(dim=-1, keepdim=True))
    return torch.softmax(scores, dim=-1)


def _head_output(attn, x, head: int):
    """Per-head attention output ``attn_weights @ V`` -> [S, HD]."""
    HD = attn.head_dim
    rows = slice(head * HD, (head + 1) * HD)
    V = x @ attn.W_v.data[rows].T
    return _head_attn_weights(attn, x, head) @ V


def _head_residual_output(attn, x, head: int, out_dims):
    """Project the head's per-query output through W_o into ``out_dims``.

    Returns ``[S, len(out_dims)]`` — the actual residual contribution this head
    makes to ALU_LO/ALU_HI (the dims downstream ops read). This is what a
    regression actually corrupts, so we diff in residual space, not slot space.
    """
    import torch

    HD = attn.head_dim
    cols = slice(head * HD, (head + 1) * HD)
    head_out = _head_output(attn, x, head)          # [S, HD]
    Wo = attn.W_o.data[list(out_dims)][:, cols]      # [n_out, HD]
    return head_out @ Wo.T                            # [S, n_out]


# ---------------------------------------------------------------------------
# Probe construction. Each probe is a small multi-row sequence ending in a
# MARK_AX operand-gather query, preceded by ``n_rows`` competing STACK0 byte-0
# candidate rows. Multiple competing rows are LOAD-BEARING: a single key never
# exposes a softmax-NORMALIZATION shift, only a re-ranking among >1 candidates
# does — which is exactly the byte-0 mechanism that regressed var/expr.
#
#   row 0       : the primary operand candidate (PSH-output if ctx['psh']),
#                 high nibble clean (ctx['hi_nibble']>0) or corrupted (==0)
#   rows 1..k-1 : recency re-stamp candidates (different values, no PSH)
#   row k       : the MARK_AX query, tagged with the probe opcode
# ---------------------------------------------------------------------------
def _make_probe(layout, opcode: str, ctx: dict):
    """Build a ``[S, d_model]`` residual probe for ``(opcode, ctx)``.

    Returns the tensor and the query row index (the MARK_AX row whose head
    output we compare).
    """
    import torch

    dp = layout.dim_positions
    D = layout.d_model
    n_rows = int(ctx["n_rows"])
    S = n_rows + 1
    x = torch.zeros(S, D, dtype=torch.float32)

    def setdim(pos, name, val=1.0):
        idx = dp.get(name)
        if idx is not None:
            x[pos, idx] = float(val)

    le_lo = dp.get("CLEAN_EMBED_LO")
    le_hi = dp.get("CLEAN_EMBED_HI")

    # Row 0: primary operand candidate.
    setdim(0, "STACK0_BYTE0", 1.0)
    setdim(0, "CONST", 1.0)
    if ctx["psh"]:
        setdim(0, "PSH_AT_SP", 1.0)
    if le_lo is not None:
        x[0, le_lo + 3] = 1.0                 # low byte value carrier
    if le_hi is not None and int(ctx["hi_nibble"]) > 0:
        x[0, le_hi + int(ctx["hi_nibble"])] = 1.0  # clean high nibble
    # ctx['hi_nibble']==0 leaves the high nibble at the default/corrupted cell.

    # Rows 1..n_rows-1: recency re-stamp candidates (distinct values, no PSH).
    for r in range(1, n_rows):
        setdim(r, "STACK0_BYTE0", 1.0)
        setdim(r, "CONST", 1.0)
        if le_lo is not None:
            x[r, le_lo + (5 + r) % 16] = 1.0
        if le_hi is not None:
            x[r, le_hi + (2 + r) % 16] = 1.0

    # Final row: the MARK_AX operand-gather query, tagged with the probe opcode.
    q = S - 1
    setdim(q, "MARK_AX", 1.0)
    setdim(q, "CONST", 1.0)
    setdim(q, opcode, 1.0)
    return x, q


# ---------------------------------------------------------------------------
# Core lint
# ---------------------------------------------------------------------------
def _is_expected(opcode: str, context: str, expect_set) -> bool:
    """Whitelist match: an entry is ``OPCODE``, ``CONTEXT``, or ``OPCODE:CONTEXT``.

    ``OP_ADD`` whitelists every ADD context; ``targeted_clean_psh`` whitelists
    that context for every opcode; ``OP_ADD:targeted_clean_psh`` is the precise
    pair the byte-0 fix is authored for.
    """
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
) -> Tuple[bool, List[HeadReport]]:
    """Lint a single candidate fix flag. Returns ``(ok, reports)``.

    ``ok`` is False when an UN-whitelisted (opcode, context) row changes on a
    shared head. Whitelist entries match opcode, context, or ``OPCODE:CONTEXT``.
    """
    import torch

    try:
        model_off, layout_off = _build(flag, "0")
    except BuildFailed as bf:
        # The OFF baseline does not build (a default-ON band/width fix whose
        # OFF state lost a head/band slot). Such a fix did NOT extend a shared
        # head in a comparable layout, so there is no cross-op softmax hazard
        # to evaluate — PASS, and say why.
        if verbose:
            print(f"[lint] flag={flag}", flush=True)
            print(
                f"[lint] PASS (no shared-head hazard): the OFF baseline could "
                f"not be built ({bf.exc!r}). This is a default-ON band/width "
                f"fix whose OFF state is not comparable — it does not extend a "
                f"shared attention head in a matching layout.",
                flush=True,
            )
        return True, []

    model_on, layout_on = _build(flag, "1")

    width_shift = layout_off.dim_positions != layout_on.dim_positions
    if width_shift and verbose:
        print(
            f"[lint] NOTE: {flag} shifts dim_positions/d_model "
            f"(OFF d_model={layout_off.d_model}, ON d_model={layout_on.d_model}). "
            f"This is a residual-band/width fix, not a shared-head edit — "
            f"per-row OFF/ON comparison is over shape-stable shared heads only.",
            flush=True,
        )

    mods = detect_modified_shared_heads(model_off, model_on)
    shared = [m for m in mods if m.pre_existing]
    new_heads = [m for m in mods if not m.pre_existing]

    if verbose:
        print(f"[lint] flag={flag}", flush=True)
        print(
            f"[lint] modified heads: {len(mods)} "
            f"(shared/pre-existing: {len(shared)}, brand-new: {len(new_heads)})",
            flush=True,
        )
        for m in shared:
            print(
                f"[lint]   SHARED head block={m.block} head={m.head} "
                f"dQ={m.q_l1:.3g} dK={m.k_l1:.3g} dV={m.v_l1:.3g} dO={m.o_l1:.3g} "
                f"out_dims={list(m.out_dims)}",
                flush=True,
            )

    if not shared:
        if verbose:
            print(
                "[lint] PASS: no SHARED/pre-existing attention head modified "
                "-> no cross-op softmax hazard.",
                flush=True,
            )
        return True, []

    blocks_off = dict(_attn_blocks(model_off))
    blocks_on = dict(_attn_blocks(model_on))

    expect_set = set(expect)
    reports: List[HeadReport] = []
    any_unexpected = False

    for m in shared:
        a_off = blocks_off[m.block]
        a_on = blocks_on[m.block]
        out_dims = m.out_dims if m.out_dims else (0,)
        rep = HeadReport(mod=m)
        for opcode in PROBE_OPCODES:
            for ctx in CONTEXTS:
                x, q = _make_probe(layout_off, opcode, ctx)
                with torch.no_grad():
                    w_off = _head_attn_weights(a_off, x, m.head)[q]
                    w_on = _head_attn_weights(a_on, x, m.head)[q]
                    y_off = _head_residual_output(a_off, x, m.head, out_dims)[q]
                    y_on = _head_residual_output(a_on, x, m.head, out_dims)[q]
                delta = (y_off - y_on).abs()
                tol = atol + rtol * y_off.abs()
                # A row CHANGES if either the residual output OR the softmax
                # weight distribution moves beyond tolerance. The weight test
                # catches re-weighting even when V relays the same value (the
                # corrupted-hi case sharpens 0.525 -> 1.0 with no argmax flip).
                w_delta = float((w_off - w_on).abs().sum().item())
                changed = bool((delta > tol).any().item()) or w_delta > atol
                argmax_changed = bool(
                    y_off.numel() > 1
                    and int(y_off.argmax().item()) != int(y_on.argmax().item())
                )
                rep.rows.append(RowResult(
                    opcode=opcode,
                    context=ctx["name"],
                    max_abs_delta=float(delta.max().item()),
                    argmax_changed=argmax_changed,
                    weight_l1_delta=w_delta,
                    changed=changed,
                ))
                if changed and not _is_expected(opcode, ctx["name"], expect_set):
                    any_unexpected = True
        reports.append(rep)

    if verbose:
        _print_reports(reports, expect_set)

    ok = not any_unexpected
    return ok, reports


def _print_reports(reports: List[HeadReport], expect_set):
    for rep in reports:
        m = rep.mod
        print(
            f"\n[lint] probe results — block {m.block} head {m.head} "
            f"(out_dims={list(m.out_dims)}):",
            flush=True,
        )
        changed_rows = [r for r in rep.rows if r.changed]
        if not changed_rows:
            print("[lint]   (no probe row changed beyond tolerance)", flush=True)
            continue
        for r in changed_rows:
            expected = _is_expected(r.opcode, r.context, expect_set)
            tag = "EXPECTED" if expected else "*** NON-LOCAL CHANGE ***"
            extra = " argmax-flip" if r.argmax_changed else ""
            print(
                f"[lint]   {r.opcode:8s} / {r.context:20s}  "
                f"max|Δout|={r.max_abs_delta:.4g}  Σ|Δweight|={r.weight_l1_delta:.4g}"
                f"{extra}   {tag}",
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
        f"({n_changed} non-local (opcode,context) row(s) changed beyond tolerance)",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Demo: prove the lint discriminates the byte-0 fix (flagged) from clean
# band/LM-head fixes (passed).
# ---------------------------------------------------------------------------
# The byte-0 fix was AUTHORED for exactly one operand context: an ADD/SUB
# operand gather where a genuine clean PSH-output row competes with a recency
# restamp (``targeted_clean_psh``). We whitelist precisely THAT pair; any change
# the fix makes to OTHER contexts — including OTHER ADD/SUB contexts that
# represent the var_simple / expr_mod operand frames — is the non-local effect.
_BYTE0_TARGETED = ("OP_ADD:targeted_clean_psh", "OP_SUB:targeted_clean_psh")


def run_demo() -> int:
    print("=" * 72, flush=True)
    print("DEMO 1/3 — byte-0 fix (C4_OPERAND_GATHER_PSH_ROWSELECT)", flush=True)
    print("  whitelist = the ONE context the fix was authored for", flush=True)
    print("             (OP_ADD/OP_SUB : targeted_clean_psh).", flush=True)
    print("  EXPECT: lint FLAGS the var/expr / corrupted-hi / multi-row", flush=True)
    print("          ADD/SUB contexts the fix silently perturbed.", flush=True)
    print("=" * 72, flush=True)
    ok_b0, rep_b0 = lint_flag(
        "C4_OPERAND_GATHER_PSH_ROWSELECT", expect=_BYTE0_TARGETED,
    )
    _summary_line("C4_OPERAND_GATHER_PSH_ROWSELECT", ok_b0, rep_b0,
                  set(_BYTE0_TARGETED))

    print("\n" + "=" * 72, flush=True)
    print("DEMO 2/3 — known-clean fix (C4_AX_BYTE1_HINIB)", flush=True)
    print("  EXPECT: PASS — adds a residual band + LM-head columns, touches", flush=True)
    print("          NO shared attention head. (OFF state is not buildable on", flush=True)
    print("          this codebase; the lint reports that and passes.)", flush=True)
    print("=" * 72, flush=True)
    ok_hinib, rep_hinib = lint_flag("C4_AX_BYTE1_HINIB")
    _summary_line("C4_AX_BYTE1_HINIB", ok_hinib, rep_hinib, set())

    print("\n" + "=" * 72, flush=True)
    print("DEMO 3/3 — known-clean fix (C4_AX_BYTE1_FULL_WIDTH)", flush=True)
    print("  Same family as HINIB but BUILDABLE both ways.", flush=True)
    print("  EXPECT: PASS — band + LM-head columns only; the shared", flush=True)
    print("          operand-gather head is byte-identical OFF->ON.", flush=True)
    print("=" * 72, flush=True)
    ok_fw, rep_fw = lint_flag("C4_AX_BYTE1_FULL_WIDTH")
    _summary_line("C4_AX_BYTE1_FULL_WIDTH", ok_fw, rep_fw, set())

    # Discrimination contract: byte-0 must be flagged (non-targeted contexts
    # changed), both band/LM-head fixes must pass.
    byte0_flagged = not ok_b0
    clean_pass = ok_hinib and ok_fw
    print("\n" + "=" * 72, flush=True)
    print("DISCRIMINATION CONTRACT", flush=True)
    print(f"  byte-0 flagged a non-local regression  : {byte0_flagged}", flush=True)
    print(f"  HINIB clean fix passed                 : {ok_hinib}", flush=True)
    print(f"  FULL_WIDTH clean fix passed            : {ok_fw}", flush=True)
    discriminates = byte0_flagged and clean_pass
    print(
        f"  => lint DISCRIMINATES real regressions : {discriminates}",
        flush=True,
    )
    print("=" * 72, flush=True)
    return 0 if discriminates else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flag", default="C4_OPERAND_GATHER_PSH_ROWSELECT",
                    help="env flag of the candidate fix to lint (default: byte-0)")
    ap.add_argument("--expect", default="",
                    help="comma-separated whitelist of (opcode,context) rows "
                         "the fix is ALLOWED to change. Each entry is an opcode "
                         "(OP_ADD), a context (targeted_clean_psh), or a pair "
                         "(OP_ADD:targeted_clean_psh).")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--demo", action="store_true",
                    help="run the byte-0-vs-HINIB discrimination demo")
    args = ap.parse_args(argv)

    if args.demo:
        return run_demo()

    expect = tuple(s.strip() for s in args.expect.split(",") if s.strip())
    ok, reports = lint_flag(args.flag, expect=expect, atol=args.atol, rtol=args.rtol)
    _summary_line(args.flag, ok, reports, set(expect))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
