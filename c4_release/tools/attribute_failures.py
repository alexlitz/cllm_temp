#!/usr/bin/env python3
"""Bridge 1096-sweep per-row failures to candidate-rule lists.

Given one or more 1096 sweep shard logs (with ``[1096-diag] id=...`` lines)
this tool:

    1. Parses failing rows from the sweep log(s).
    2. Joins each row to first-fatal-slot data already captured in the
       ``.agent-logs/lowering-audit/`` audit logs
       (``[1096-lowering] id=... first_fatal=severity=... stepN:SLOT ...
       output_byte=0xXX lo_arg=N hi_arg=M ... OUTPUT_LO[expected_low]:
       expected=A winner=B ... OUTPUT_HI[expected_high]:expected=C winner=D
       ...``).
    3. Compiles the unified-VM layout (``decl_verifier._build_layout_only``)
       and builds the per-(dim_name, offset) writer index
       (``writer_index.build_writer_index``).
    4. For each row's (winner_lo, winner_hi, expected_lo, expected_hi)
       lanes, looks up rules contesting that output byte and filters them
       to those whose name / scope match the slot register family
       (SP_*, PC_*, MEM_*, STACK0_*, AX_*, BP_*).
    5. Emits a Markdown report (or CSV) listing per-row first_fatal_slot,
       candidate rule names, and candidate op names. Top of the report
       includes attribution-rate aggregates and a slot histogram.

Usage:

    python tools/attribute_failures.py \\
        .agent-logs/sweep-2026-06-01/shard_0.log \\
        .agent-logs/sweep-2026-06-01/shard_548.log \\
        --audit-glob '.agent-logs/lowering-audit/**/*.log' \\
        --out .agent-logs/attribution_2026_06_01.md

    python tools/attribute_failures.py shard_0.log --format csv

    # Reverse lookup -- which rows would change if we touched a rule?
    python tools/attribute_failures.py shard_0.log \\
        --rule-impact tail_sp_byte1_ff_from_initial_stack

The tool is diagnostic-only; no other source files are modified and the
output report is intended to be gitignored (write under .agent-logs/).
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import io
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# Make ``import neural_vm`` work whether the tool is run from
# c4_release/tools or from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)


# ---------------------------------------------------------------------------
# Sweep + audit log parsing
# ---------------------------------------------------------------------------


_DIAG_RE = re.compile(
    r"\[1096-diag\]\s+"
    r"id=(?P<id>\d+)\s+"
    r"mode=(?P<mode>\S+)\s+"
    r"status=(?P<status>\S+)\s+"
    r"suite_decl=(?P<suite_decl>\S+)\s+"
    r"desc=(?P<desc>'(?:[^'\\]|\\.)*'|\S+)\s+"
    r"expected=(?P<expected>\S+)\s+"
    r"decl=(?P<decl>\S+)\s+"
    r"decl_steps=(?P<decl_steps>\S+)\s+"
    r"neural=(?P<neural>\S+)"
)


_LOWERING_HEADER_RE = re.compile(
    r"\[1096-lowering\]\s+id=(?P<id>\d+)\b"
)


_FIRST_FATAL_STEP_SLOT_RE = re.compile(
    r"first_fatal=[^\n]*?step(?P<step>\d+):(?P<slot>[A-Za-z_][A-Za-z0-9_]*)"
)


_FIRST_FATAL_BYTES_RE = re.compile(
    r"first_fatal=[^\n]*?abs=(?P<abs>\d+)\s+"
    r"expected=(?P<expected>0x[0-9a-fA-F]+)\s+"
    r"argmax=(?P<argmax>0x[0-9a-fA-F]+)"
)


_FIRST_FATAL_OUTPUT_BYTE_RE = re.compile(
    r"first_fatal=[^\n]*?output_byte=(?P<output_byte>0x[0-9a-fA-F]+)"
)


_LO_HI_ARG_RE = re.compile(
    r"lo_arg=(?P<lo>\d+)\s+hi_arg=(?P<hi>\d+)"
)


_OUTPUT_LO_EXPECTED_RE = re.compile(
    r"OUTPUT_LO\[expected_low\]:expected=(?P<expected>\d+)\s+winner=(?P<winner>\d+)"
)


_OUTPUT_HI_EXPECTED_RE = re.compile(
    r"OUTPUT_HI\[expected_high\]:expected=(?P<expected>\d+)\s+winner=(?P<winner>\d+)"
)


_DESC_SHORT_DESCS = re.compile(r"'((?:[^'\\]|\\.)*)'")


@dataclass(frozen=True)
class DiagRow:
    """One parsed ``[1096-diag]`` line."""

    test_idx: int
    mode: str
    status: str
    suite_decl: str
    description: str
    expected: str
    declarative: str
    decl_steps: str
    neural: str

    @property
    def is_failure(self) -> bool:
        return self.status not in {"ok", "strict-ok"}


@dataclass(frozen=True)
class LoweringRow:
    """One parsed ``[1096-lowering]`` line capturing first-fatal slot."""

    test_idx: int
    description: str
    step: int
    slot: str
    abs_pos: int
    expected_byte: int
    argmax_byte: int
    output_byte: Optional[int]
    lo_arg: Optional[int]  # winner_lo lane in OUTPUT_LO
    hi_arg: Optional[int]  # winner_hi lane in OUTPUT_HI
    expected_lo: Optional[int]
    expected_hi: Optional[int]


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] == "'":
        return s[1:-1]
    return s


def parse_diag_rows(text: str) -> List[DiagRow]:
    rows: List[DiagRow] = []
    for m in _DIAG_RE.finditer(text):
        rows.append(
            DiagRow(
                test_idx=int(m.group("id")),
                mode=m.group("mode"),
                status=m.group("status"),
                suite_decl=m.group("suite_decl"),
                description=_strip_quotes(m.group("desc")),
                expected=m.group("expected"),
                declarative=m.group("decl"),
                decl_steps=m.group("decl_steps"),
                neural=m.group("neural"),
            )
        )
    return rows


def parse_lowering_rows(text: str) -> List[LoweringRow]:
    """Parse ``[1096-lowering] id=... first_fatal=...`` records.

    Each record is one line in the audit logs. We extract the first-fatal
    step/slot plus the output-band winner/expected lanes.
    """
    out: List[LoweringRow] = []
    # The audit log puts everything on one line per row, but we don't rely
    # on it -- we walk line-by-line and match self-contained patterns.
    for line in text.splitlines():
        head = _LOWERING_HEADER_RE.search(line)
        if not head:
            continue
        test_idx = int(head.group("id"))
        desc_m = re.search(r"desc=('(?:[^'\\]|\\.)*'|\S+)", line)
        description = _strip_quotes(desc_m.group(1)) if desc_m else ""
        slot_m = _FIRST_FATAL_STEP_SLOT_RE.search(line)
        if not slot_m:
            continue
        step = int(slot_m.group("step"))
        slot = slot_m.group("slot")
        bytes_m = _FIRST_FATAL_BYTES_RE.search(line)
        if not bytes_m:
            continue
        abs_pos = int(bytes_m.group("abs"))
        expected_byte = int(bytes_m.group("expected"), 16)
        argmax_byte = int(bytes_m.group("argmax"), 16)
        out_byte_m = _FIRST_FATAL_OUTPUT_BYTE_RE.search(line)
        output_byte = (
            int(out_byte_m.group("output_byte"), 16) if out_byte_m else None
        )
        lohi_m = _LO_HI_ARG_RE.search(line)
        lo_arg = int(lohi_m.group("lo")) if lohi_m else None
        hi_arg = int(lohi_m.group("hi")) if lohi_m else None
        lo_exp_m = _OUTPUT_LO_EXPECTED_RE.search(line)
        expected_lo = int(lo_exp_m.group("expected")) if lo_exp_m else None
        hi_exp_m = _OUTPUT_HI_EXPECTED_RE.search(line)
        expected_hi = int(hi_exp_m.group("expected")) if hi_exp_m else None
        out.append(
            LoweringRow(
                test_idx=test_idx,
                description=description,
                step=step,
                slot=slot,
                abs_pos=abs_pos,
                expected_byte=expected_byte,
                argmax_byte=argmax_byte,
                output_byte=output_byte,
                lo_arg=lo_arg,
                hi_arg=hi_arg,
                expected_lo=expected_lo,
                expected_hi=expected_hi,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Layout + writer-index helpers
# ---------------------------------------------------------------------------


_SLOT_FAMILIES: Mapping[str, Tuple[str, ...]] = {
    "PC": ("PC", "pc", "MARK_PC"),
    "AX": ("AX", "ax", "MARK_AX"),
    "SP": ("SP", "sp", "MARK_SP"),
    "BP": ("BP", "bp", "MARK_BP"),
    "STACK0": ("STACK0", "stack0", "MARK_STACK0"),
    "MEM_addr": ("MEM_ADDR", "mem_store", "mem_addr", "MARK_MEM"),
    "MEM_value": ("MEM_VAL", "mem_value", "mem_val", "MARK_MEM"),
    "REG_PC": ("PC", "pc", "MARK_PC"),
    "REG_AX": ("AX", "ax", "MARK_AX"),
    "REG_SP": ("SP", "sp", "MARK_SP"),
    "REG_BP": ("BP", "bp", "MARK_BP"),
    "MEM": ("MEM", "MARK_MEM"),
}


_SLOT_SCOPE_HINTS: Mapping[str, Tuple[str, ...]] = {
    "PC": ("mark == PC",),
    "AX": ("mark == AX",),
    "SP": ("mark == SP",),
    "BP": ("mark == BP",),
    "STACK0": ("mark == STACK0",),
    "MEM": ("mark == MEM",),
    "MEM_addr": ("mark == MEM",),
    "MEM_value": ("mark == MEM",),
    "REG_PC": ("mark == PC",),
    "REG_AX": ("mark == AX",),
    "REG_SP": ("mark == SP",),
    "REG_BP": ("mark == BP",),
}


def _slot_family(slot: str) -> str:
    """Map a slot name like ``SP_byte0`` to a coarse family key
    (``SP``, ``MEM_addr``, ``REG_PC``...).  Falls back to the slot itself
    when no prefix matches."""
    for prefix in ("MEM_addr", "MEM_value", "REG_PC", "REG_AX", "REG_SP",
                   "REG_BP", "STACK0", "STEP_END"):
        if slot.startswith(prefix):
            return prefix
    for prefix in ("PC", "AX", "SP", "BP", "MEM"):
        if slot.startswith(prefix + "_") or slot == prefix:
            return prefix
    return slot


def _byte_index_from_slot(slot: str) -> Optional[int]:
    m = re.search(r"_byte(\d+)$", slot)
    if m:
        return int(m.group(1))
    m = re.search(r"_addr(\d+)$", slot)
    if m:
        return int(m.group(1))
    m = re.search(r"_value(\d+)$", slot)
    if m:
        return int(m.group(1))
    return None


@dataclass
class LayoutBundle:
    layout: object  # CompiledLayout
    ops: List[object]  # all ops with .name (+ optional .compiler_ir)
    writer_index: Mapping[Tuple[str, int], List[object]]  # WriterEntry list
    # Helper indices.
    rules_by_op: Dict[str, List[object]] = field(default_factory=dict)
    # Reverse index: rule (id) -> set of (dim_name, offset) it writes.
    rule_writes: Dict[int, Set[Tuple[str, int]]] = field(default_factory=dict)


def build_layout_bundle(
    *,
    alu_mode: str = "efficient",
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    n_heads: int = 8,
) -> LayoutBundle:
    """Build the production layout, collect every op carrying FFNRules,
    and assemble the writer index over them.
    """
    from neural_vm.dim_registry import DimRegistry
    from neural_vm.verification.decl_verifier import _build_layout_only
    from neural_vm.verification.writer_index import (
        _collect_ffn_rules_from_op,
        build_writer_index,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layout = _build_layout_only(
            alu_mode=alu_mode,
            enable_conversational_io=enable_conversational_io,
            enable_tool_calling=enable_tool_calling,
            n_heads=n_heads,
        )

    ops: List[object] = []
    for ops_at in layout.ops_per_layer:
        for op in ops_at:
            ops.append(op)
    ops.extend(layout.block_ops)
    ops.extend(layout.model_ops)

    # Build a minimal registry from layout.  build_writer_index calls
    # effective_predicate which needs semantics for many dims; rules whose
    # eff_predicate fails are skipped silently.  We deliberately tolerate
    # that -- the writer_index still indexes the rule writes themselves.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        registry = DimRegistry(d_model=layout.d_model)
        for name, pos in layout.dim_positions.items():
            size = layout.dim_sizes.get(name, 1)
            try:
                registry.alloc(name, pos, size, name, semantics=None)
            except Exception:
                pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            widx = build_writer_index(ops, registry)
        except Exception:
            widx = {}

    # The fancier writer index may drop rules whose effective_predicate
    # fails.  As a fallback for candidate lookup we also build a raw index
    # straight off the writes -- no semantics required.
    raw_index: Dict[Tuple[str, int], List[object]] = collections.defaultdict(list)
    rules_by_op: Dict[str, List[object]] = {}
    rule_writes: Dict[int, Set[Tuple[str, int]]] = {}
    for op in ops:
        op_name = getattr(op, "name", "<anonymous>")
        rules = _collect_ffn_rules_from_op(op)
        if not rules:
            continue
        rules_by_op[op_name] = list(rules)
        for r in rules:
            wset: Set[Tuple[str, int]] = set()
            for wt in getattr(r, "writes", ()):
                if wt.weight == 0.0:
                    continue
                key = (wt.dim.name, wt.dim.offset)
                wset.add(key)
                raw_index[key].append((op_name, r))
            rule_writes[id(r)] = wset

    # Merge: prefer raw_index entries so rules without registry semantics
    # are still considered.  Each candidate is stored as (op_name, rule).
    merged: Dict[Tuple[str, int], List[Tuple[str, object]]] = {
        k: list(v) for k, v in raw_index.items()
    }

    return LayoutBundle(
        layout=layout,
        ops=ops,
        writer_index=merged,
        rules_by_op=rules_by_op,
        rule_writes=rule_writes,
    )


_FAMILY_NAME_PATTERN = {
    # Family -> compiled regex matching rule names that target that
    # register family.  Patterns are intentionally narrow: they require
    # word-boundary-ish separators so e.g. ``sp`` does not match
    # ``alu_carry_lo_sp_only`` unintentionally and ``ax`` does not match
    # ``max``.
    "PC": re.compile(r"(?:^|_)pc(?:_|$)|pc_byte\d|pc_marker"),
    "AX": re.compile(r"(?:^|_)ax(?:_|$)|ax_byte\d|ax_lo|ax_hi"),
    "SP": re.compile(r"(?:^|_)sp(?:_|$)|sp_byte\d|sp_marker"),
    "BP": re.compile(r"(?:^|_)bp(?:_|$)|bp_byte\d|bp_marker"),
    "STACK0": re.compile(r"stack0(?:_|$)"),
    "MEM": re.compile(r"(?:^|_)mem(?:_|$)|mem_store|mem_addr|mem_value|mem_val"),
}


_LANE_DOT_RE = re.compile(r"\.lane_(\d+)$")
_LANE_LO_RE = re.compile(r"_lo_(\d+)$")
_LANE_HI_RE = re.compile(r"_hi_(\d+)$")
_LANE_HILO_RE = re.compile(r"_hi(\d+)_lo(\d+)_([0-9a-f])$")


def _rule_lanes(rule_name: str) -> List[Tuple[str, int]]:
    """Best-effort lane extraction from rule names.

    Returns a list of ``(kind, lane_index)`` for the rule's OUTPUT lane
    targets (``kind`` in ``{'lo', 'hi'}``).  The empty list means the
    rule is not lane-split (we can't filter it by lane).

    Recognised suffix patterns (lower-cased):
        * ``.lo.lane_N``  -> [(lo, N)]
        * ``.hi.lane_N``  -> [(hi, N)]
        * ``_lo_N`` (where N is hex digit) -> [(lo, N)]
        * ``_hi_N`` -> [(hi, N)]
        * ``_hiK_loJ_M`` (used by some L16 frame rules) -> [(hi, K), (lo, M)]
          We treat the trailing digit as the lo lane index because the
          rule is split per lo-lane.
    """
    if not rule_name:
        return []
    n = rule_name.lower()
    out: List[Tuple[str, int]] = []
    if ".lo.lane_" in n:
        m = re.search(r"\.lo\.lane_(\d+)$", n)
        if m:
            return [("lo", int(m.group(1)))]
    if ".hi.lane_" in n:
        m = re.search(r"\.hi\.lane_(\d+)$", n)
        if m:
            return [("hi", int(m.group(1)))]
    # _hiX_loY_Z form (X, Y, Z are hex single chars used by L16 frame
    # rules: hi-lane X with low-lane base Y and individualized lane Z).
    m = _LANE_HILO_RE.search(n)
    if m:
        out.append(("hi", int(m.group(1))))
        out.append(("lo", int(m.group(3), 16)))
        return out
    m = _LANE_LO_RE.search(n)
    if m:
        out.append(("lo", int(m.group(1))))
    m = _LANE_HI_RE.search(n)
    if m:
        out.append(("hi", int(m.group(1))))
    if out:
        return out
    m = _LANE_DOT_RE.search(n)
    if m:
        # Bare ``.lane_N`` without a lo/hi tag -- ambiguous; report both.
        idx = int(m.group(1))
        return [("lo", idx), ("hi", idx)]
    return []


def candidate_rules_for(
    bundle: LayoutBundle,
    audit: LoweringRow,
) -> List[Tuple[str, object, str]]:
    """Return the candidate rules competing for ``audit``'s first-fatal byte.

    Returns a list of ``(op_name, rule, reason)`` where ``reason`` notes
    the OUTPUT lane / family match.  Rules are de-duplicated across lane
    hits.  Filtered to rules whose name or scope plausibly target the
    slot's register family AND the specific byte index inside that family
    (e.g. ``SP_byte1`` only matches names containing ``sp_byte1``,
    ``sp_marker_byte1``, ``stack0_byte1`` -- not ``sp_byte0``).
    """
    if audit.lo_arg is None or audit.hi_arg is None:
        return []

    lanes_lo: Set[int] = set()
    lanes_hi: Set[int] = set()
    if audit.lo_arg is not None:
        lanes_lo.add(audit.lo_arg)
    if audit.hi_arg is not None:
        lanes_hi.add(audit.hi_arg)
    if audit.expected_lo is not None:
        lanes_lo.add(audit.expected_lo)
    if audit.expected_hi is not None:
        lanes_hi.add(audit.expected_hi)

    family = _slot_family(audit.slot)
    family_pat = _FAMILY_NAME_PATTERN.get(family)
    scope_hints = _SLOT_SCOPE_HINTS.get(family, ())
    byte_idx = _byte_index_from_slot(audit.slot)

    # Byte-index discriminator: when the slot specifies a byte index
    # (e.g. SP_byte1, MEM_addr0, STACK0_byte2), the rule must reference
    # the same index in its name OR have an empty/no byte-tag (we
    # tolerate the latter as a soft match).
    if byte_idx is not None:
        # Allow byte0 in mem_value/mem_addr forms as well.
        byte_pat = re.compile(
            rf"(?:byte|addr|value|val|b){byte_idx}(?:[^0-9]|$)"
        )
    else:
        byte_pat = None

    seen: Dict[int, Tuple[str, object, str]] = {}

    def _maybe_add(op_name: str, rule: object, reason: str) -> None:
        rn = (getattr(rule, "name", None) or "").lower()
        sc = (getattr(rule, "scope", None) or "")
        # Family match (name or scope).
        family_match = False
        if family_pat is not None and family_pat.search(rn):
            family_match = True
        elif scope_hints and any(h in sc for h in scope_hints):
            family_match = True
        # If we don't know the family at all, accept all writers.
        if family_pat is None and not scope_hints:
            family_match = True
        if not family_match:
            return
        # Byte-index discriminator (soft: keep rules with no byte hint).
        if byte_pat is not None:
            rn_has_byte_tag = bool(
                re.search(r"(byte|addr|value|val|b)\d", rn)
            )
            if rn_has_byte_tag and not byte_pat.search(rn):
                return
        # Lane consistency: lane-split rules only fire at specific
        # OUTPUT lanes.  Drop rules whose declared lanes don't hit the
        # investigated lane set.
        lanes = _rule_lanes(getattr(rule, "name", "") or "")
        if lanes:
            ok = False
            for kind, idx in lanes:
                if kind == "lo" and idx in lanes_lo:
                    ok = True
                    break
                if kind == "hi" and idx in lanes_hi:
                    ok = True
                    break
            if not ok:
                return
        key = id(rule)
        if key in seen:
            return
        seen[key] = (op_name, rule, reason)

    for off in lanes_lo:
        for op_name, rule in bundle.writer_index.get(("OUTPUT_LO", off), []):
            _maybe_add(op_name, rule, f"OUTPUT_LO[{off}]")
    for off in lanes_hi:
        for op_name, rule in bundle.writer_index.get(("OUTPUT_HI", off), []):
            _maybe_add(op_name, rule, f"OUTPUT_HI[{off}]")

    return list(seen.values())


def _base_rule_name(name: Optional[str]) -> str:
    """Strip lane suffixes from a rule name so a lane-split family
    collapses to one display string.  Recognised forms:
        ``.lo.lane_N``  ``.hi.lane_N``  ``.lane_N``
        ``_lo_N`` ``_hi_N``
        ``_hiK_loJ_M`` (L16 frame rules) -> drop the trailing ``_M``
    """
    if not name:
        return "<unnamed>"
    n = name
    n = re.sub(r"\.(lo|hi)\.lane_\d+$", "", n)
    n = re.sub(r"\.lane_\d+$", "", n)
    n = re.sub(r"_hi(\d+)_lo(\d+)_[0-9a-f]$", r"_hi\1_lo\2", n)
    n = re.sub(r"_(lo|hi)_\d+$", "", n)
    return n


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


@dataclass
class AttributionRecord:
    test_idx: int
    description: str
    status: str
    expected: str
    decl_steps: str
    neural: str
    audit: Optional[LoweringRow]
    candidates: List[Tuple[str, object, str]]

    @property
    def first_fatal_slot(self) -> str:
        if self.audit is None:
            return "<no-audit>"
        return f"step{self.audit.step}:{self.audit.slot}"

    @property
    def candidate_rule_names(self) -> List[str]:
        """Distinct rule names after collapsing lane-split suffixes."""
        return sorted({
            _base_rule_name(getattr(r, "name", None))
            for _op, r, _why in self.candidates
        })

    @property
    def candidate_rule_names_full(self) -> List[str]:
        return sorted({
            getattr(r, "name", None) or "<unnamed>"
            for _op, r, _why in self.candidates
        })

    @property
    def candidate_op_names(self) -> List[str]:
        return sorted({op for op, _r, _why in self.candidates})


def _summary_lines(records: Sequence[AttributionRecord]) -> List[str]:
    by_slot = collections.Counter()
    n_attributed = 0
    n_unique = 0
    n_big = 0
    n_no_audit = 0
    for r in records:
        if r.audit is None:
            n_no_audit += 1
            by_slot["<no-audit>"] += 1
            continue
        n_attributed += 1
        by_slot[f"step{r.audit.step}:{r.audit.slot}"] += 1
        if len(r.candidates) == 1:
            n_unique += 1
        if len(r.candidates) > 5:
            n_big += 1

    total = len(records)
    lines = [
        "## Attribution summary",
        "",
        f"- total rows analysed: **{total}**",
        f"- rows with audit data (first-fatal slot known): **{n_attributed}**",
        f"- rows with unique candidate rule (==1): **{n_unique}**",
        f"- rows with >5 candidate rules: **{n_big}**",
        f"- rows without audit (no first-fatal slot): **{n_no_audit}**",
        "",
        "## First-fatal slot histogram (rows per slot)",
        "",
        "| slot | rows |",
        "|---|---:|",
    ]
    for slot, n in by_slot.most_common():
        lines.append(f"| `{slot}` | {n} |")
    lines.append("")
    return lines


def _records_to_markdown(records: Sequence[AttributionRecord]) -> str:
    buf = io.StringIO()
    buf.write("# 1096 failure attribution\n\n")
    for line in _summary_lines(records):
        buf.write(line + "\n")
    buf.write("\n## Per-row attribution\n\n")
    buf.write("| id | desc | first_fatal | n_cand | candidate_rules | candidate_ops |\n")
    buf.write("|---:|---|---|---:|---|---|\n")
    for r in records:
        cand_names = r.candidate_rule_names
        cand_ops = r.candidate_op_names
        rules_disp = ", ".join(f"`{n}`" for n in cand_names[:8])
        if len(cand_names) > 8:
            rules_disp += f", ... (+{len(cand_names) - 8})"
        ops_disp = ", ".join(f"`{n}`" for n in cand_ops[:6])
        if len(cand_ops) > 6:
            ops_disp += f", ... (+{len(cand_ops) - 6})"
        desc = r.description.replace("|", "\\|")
        buf.write(
            f"| {r.test_idx:04d} "
            f"| {desc} "
            f"| `{r.first_fatal_slot}` "
            f"| {len(cand_names)} "
            f"| {rules_disp or '-'} "
            f"| {ops_disp or '-'} |\n"
        )
    return buf.getvalue()


def _records_to_csv(records: Sequence[AttributionRecord]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "row_id", "description", "status", "expected", "decl_steps",
        "neural", "first_fatal_slot", "first_fatal_step",
        "expected_byte", "argmax_byte", "output_byte",
        "lo_arg", "hi_arg", "expected_lo", "expected_hi",
        "n_candidate_rules", "candidate_rules", "candidate_ops",
    ])
    for r in records:
        a = r.audit
        w.writerow([
            f"{r.test_idx:04d}",
            r.description,
            r.status,
            r.expected,
            r.decl_steps,
            r.neural,
            r.first_fatal_slot,
            a.step if a else "",
            f"0x{a.expected_byte:02x}" if a else "",
            f"0x{a.argmax_byte:02x}" if a else "",
            f"0x{a.output_byte:02x}" if a and a.output_byte is not None else "",
            a.lo_arg if a and a.lo_arg is not None else "",
            a.hi_arg if a and a.hi_arg is not None else "",
            a.expected_lo if a and a.expected_lo is not None else "",
            a.expected_hi if a and a.expected_hi is not None else "",
            len(r.candidate_rule_names),
            ";".join(r.candidate_rule_names),
            ";".join(r.candidate_op_names),
        ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _load_audit_index(audit_paths: Iterable[str]) -> Dict[int, LoweringRow]:
    audit_by_id: Dict[int, LoweringRow] = {}
    for path in audit_paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                data = f.read()
        except OSError:
            continue
        for row in parse_lowering_rows(data):
            # Keep the earliest occurrence per id; audit logs are mostly
            # deterministic but multiple files may overlap.
            audit_by_id.setdefault(row.test_idx, row)
    return audit_by_id


def _default_audit_paths() -> List[str]:
    # Best-effort defaults: walk .agent-logs/lowering-audit/ under repo root.
    # Use this only when no --audit-glob is given.
    repo_root = os.path.abspath(os.path.join(_PKG, ".."))
    default_dir = os.path.join(
        repo_root, ".agent-logs", "lowering-audit",
    )
    if not os.path.isdir(default_dir):
        return []
    out: List[str] = []
    for root, _dirs, files in os.walk(default_dir):
        for fn in files:
            if fn.endswith(".log"):
                out.append(os.path.join(root, fn))
    return out


def _expand_globs(patterns: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        if any(ch in p for ch in "*?[") or "**" in p:
            out.extend(sorted(glob.glob(p, recursive=True)))
        else:
            out.append(p)
    return out


def _attribute(
    sweep_paths: Sequence[str],
    audit_paths: Sequence[str],
    *,
    include_passing: bool = False,
) -> Tuple[List[AttributionRecord], LayoutBundle]:
    # 1. Parse sweep diag rows.
    rows: List[DiagRow] = []
    seen_ids: Set[int] = set()
    for path in sweep_paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                data = f.read()
        except OSError as e:
            sys.stderr.write(f"[attribute_failures] WARN: cannot read {path}: {e}\n")
            continue
        for row in parse_diag_rows(data):
            if row.test_idx in seen_ids:
                continue
            seen_ids.add(row.test_idx)
            rows.append(row)

    if not include_passing:
        rows = [r for r in rows if r.is_failure]

    # 2. Build audit index.
    audit_idx = _load_audit_index(audit_paths)

    # 3. Build layout + writer index.
    bundle = build_layout_bundle()

    # 4. Per row, fetch audit + candidates.
    records: List[AttributionRecord] = []
    for r in rows:
        audit = audit_idx.get(r.test_idx)
        cands = candidate_rules_for(bundle, audit) if audit is not None else []
        records.append(
            AttributionRecord(
                test_idx=r.test_idx,
                description=r.description,
                status=r.status,
                expected=r.expected,
                decl_steps=r.decl_steps,
                neural=r.neural,
                audit=audit,
                candidates=cands,
            )
        )

    return records, bundle


def _rule_impact_report(
    records: Sequence[AttributionRecord],
    rule_query: str,
) -> str:
    """Reverse-lookup: rows whose candidate set includes ``rule_query``."""
    matches: List[AttributionRecord] = []
    for rec in records:
        for _op, rule, _why in rec.candidates:
            rn = getattr(rule, "name", None) or ""
            if rule_query == rn or rule_query in rn:
                matches.append(rec)
                break
    buf = io.StringIO()
    buf.write(f"# Rule impact: `{rule_query}`\n\n")
    buf.write(f"- matched query in **{len(matches)}** failing rows\n\n")
    by_slot = collections.Counter(
        rec.first_fatal_slot for rec in matches
    )
    if by_slot:
        buf.write("## Slot breakdown\n\n")
        buf.write("| slot | rows |\n|---|---:|\n")
        for k, v in by_slot.most_common():
            buf.write(f"| `{k}` | {v} |\n")
        buf.write("\n")
    buf.write("## Rows\n\n")
    buf.write("| id | desc | first_fatal |\n|---:|---|---|\n")
    for rec in matches[:200]:
        desc = rec.description.replace("|", "\\|")
        buf.write(
            f"| {rec.test_idx:04d} | {desc} | `{rec.first_fatal_slot}` |\n"
        )
    if len(matches) > 200:
        buf.write(f"\n... ({len(matches) - 200} more rows)\n")
    return buf.getvalue()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge 1096-sweep failures to first-fatal-slot candidate "
            "rule lists."
        ),
    )
    parser.add_argument(
        "sweep_logs",
        nargs="+",
        help="1096 sweep shard log file(s) carrying [1096-diag] lines",
    )
    parser.add_argument(
        "--audit-glob",
        action="append",
        default=[],
        help=(
            "Glob (recursive **) for [1096-lowering] audit logs. Repeat "
            "for multiple globs. Default: "
            "<repo>/.agent-logs/lowering-audit/**/*.log"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("md", "csv"),
        default="md",
        help="Report format (default md)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write report to this path. Default: stdout.",
    )
    parser.add_argument(
        "--include-passing",
        action="store_true",
        help="Include rows whose status is ok / strict-ok in the report.",
    )
    parser.add_argument(
        "--rule-impact",
        default=None,
        help=(
            "Reverse-lookup mode: given a rule name (substring match), "
            "list every failing row whose candidate set includes that "
            "rule. Overrides the regular per-row report."
        ),
    )
    args = parser.parse_args(argv)

    sweep_paths = _expand_globs(args.sweep_logs)
    if args.audit_glob:
        audit_paths = _expand_globs(args.audit_glob)
    else:
        audit_paths = _default_audit_paths()

    records, _bundle = _attribute(
        sweep_paths,
        audit_paths,
        include_passing=args.include_passing,
    )

    if args.rule_impact:
        report = _rule_impact_report(records, args.rule_impact)
    elif args.format == "csv":
        report = _records_to_csv(records)
    else:
        report = _records_to_markdown(records)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        sys.stderr.write(
            f"[attribute_failures] wrote {len(records)} rows -> {args.out}\n"
        )
    else:
        sys.stdout.write(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
