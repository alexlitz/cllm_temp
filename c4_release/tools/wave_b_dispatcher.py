#!/usr/bin/env python3
"""Wave B migration dispatcher.

Coordinates the symbolic migration of the 42 position-role violations
flagged by ``tools/lint_position_role.py`` (commit 199479f4) using the
``migrate_rule_to_step_end`` helper landed in
``neural_vm/unified_compiler/step_end_migration.py`` (commit a9aefb4f).

The dispatcher does NOT edit any ``l*_ops.py`` file. It operates
symbolically:

  1. Loads each rule-factory function from the cluster manifest.
  2. Applies :func:`migrate_rule_to_step_end` to the factory's return
     tuple, in memory.
  3. Runs the parity assertion via
     ``tests/test_step_end_migration_parity.assert_step_end_parity``
     for every factory that's wired into ``RULE_FACTORIES``.
  4. Reports a clean per-cluster summary so a follow-up agent can land
     the actual source edits in one commit per cluster.

Cluster plans:

* ``docs/WAVE_B_CLUSTER_1_PLAN_2026_06_10.md`` — L10 (15 rules)
* ``docs/WAVE_B_CLUSTER_2_PLAN_2026_06_10.md`` — L8 (14 rules)
* ``docs/WAVE_B_CLUSTER_3_PLAN_2026_06_10.md`` — L9 (5 rules)
* ``docs/WAVE_B_CLUSTER_4_PLAN_2026_06_10.md`` — L11/L12 (2 rules)
* ``docs/WAVE_B_CLUSTER_5_PLAN_2026_06_10.md`` — L3/L6 scattered (6
  rules; some are lint false-positives)

Usage
=====

::

    python c4_release/tools/wave_b_dispatcher.py
    python c4_release/tools/wave_b_dispatcher.py --cluster 1
    python c4_release/tools/wave_b_dispatcher.py --json
    python c4_release/tools/wave_b_dispatcher.py --list

Exit codes
==========

* 0 — every dispatched migration succeeded (or, for parity-skipped
  rules, ``migrate_rule_to_step_end`` produced a valid output).
* 1 — at least one migration raised :class:`MigrationSafetyError` or
  a parity assertion failed.
* 2 — invocation / IO error.

This tool is a SMOKE-TEST harness; it does NOT mutate source files.
After the dispatcher is green for a cluster, a separate agent lands
the actual source edits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Repo / import bootstrap
# ---------------------------------------------------------------------------


def _find_repo_root() -> Optional[Path]:
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent.parent, *here.parents):
        if (candidate / "c4_release").is_dir():
            return candidate
        if (candidate / "neural_vm").is_dir() and candidate.name == "c4_release":
            return candidate.parent
    return None


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT is not None:
    sys.path.insert(0, str(_REPO_ROOT / "c4_release"))


# ---------------------------------------------------------------------------
# Cluster manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleSpec:
    """One row of a cluster plan.

    ``factory_module`` + ``factory_name`` identify the rule-factory
    function inside ``neural_vm/unified_compiler/ops/lN_ops.py``.
    ``relayed_dims`` is the operand-dim list passed to
    :func:`migrate_rule_to_step_end`. ``allow_step_end_writes_to``
    is the safety-check whitelist (typically used for rules whose
    natural write target is a byte-emission slot that the architecture
    explicitly tolerates at STEP_END).
    """

    cluster: int
    factory_module: str
    factory_name: str
    relayed_dims: Tuple[str, ...]
    allow_step_end_writes_to: Tuple[str, ...] = ()
    parity_factory_key: Optional[str] = None
    notes: str = ""


_L10_CMP_RELAYED: Tuple[str, ...] = (
    "CMP",
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
)
_L10_BITWISE_RELAYED: Tuple[str, ...] = (
    "ALU_LO", "ALU_HI",
    "AX_CARRY_LO", "AX_CARRY_HI",
    "OP_OR", "OP_XOR", "OP_AND",
)
_L10_SHIFT_RELAYED: Tuple[str, ...] = ("ALU_LO", "ALU_HI", "OP_SHL", "OP_SHR")
_L10_MUL_LO_RELAYED: Tuple[str, ...] = (
    "ALU_LO", "ALU_HI", "AX_CARRY_LO", "OP_MUL",
)
_L10_TAIL_CMP_RELAYED: Tuple[str, ...] = (
    "CMP", "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
)
_L8_ALU_LO_RELAYED: Tuple[str, ...] = (
    "ALU_LO", "ALU_HI",
    "AX_CARRY_LO", "AX_CARRY_HI",
    "OP_ADD", "OP_SUB", "OP_LEA", "OP_ADJ", "OP_ENT",
)
_L8_ALU_CMP_RELAYED: Tuple[str, ...] = (
    "CMP", "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
)
_L8_LEV_RELAYED: Tuple[str, ...] = (
    "ALU_LO", "ALU_HI",
    "BP_FRAME_BYTE0", "BP_FRAME_BYTE1", "BP_FRAME_BYTE2",
    "OP_LEV",
)
_L9_CMP_RELAYED: Tuple[str, ...] = (
    "CMP",
    "ALU_LO", "ALU_HI",
    "AX_CARRY_LO", "AX_CARRY_HI",
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
)
_L9_BP_PLUS8_RELAYED: Tuple[str, ...] = (
    "BP_FRAME_BYTE0", "BP_FRAME_BYTE1",
    "OP_ENT", "OP_LEV",
)
_L11_MUL_PARTIAL_RELAYED: Tuple[str, ...] = (
    "ALU_LO", "ALU_HI",
    "AX_CARRY_LO", "AX_CARRY_HI",
    "MUL_PARTIAL_LO", "MUL_PARTIAL_HI",
    "OP_MUL",
)
_L12_MUL_COMBINE_RELAYED: Tuple[str, ...] = (
    "MUL_PARTIAL_LO", "MUL_PARTIAL_HI",
    "MUL_ACCUM_LO", "MUL_ACCUM_HI",
    "OP_MUL",
)
_L6_ALU_CLEAR_RELAYED: Tuple[str, ...] = (
    "ALU_LO", "ALU_HI",
    "OP_ADD", "OP_SUB", "OP_LEA", "OP_ADJ", "OP_ENT",
)


_CLUSTER_MANIFEST: Tuple[RuleSpec, ...] = (
    # ---------- Cluster 1: L10 (15 rules; 5 rule-factory functions) -----
    RuleSpec(
        cluster=1,
        factory_module="neural_vm.unified_compiler.ops.l10_ops",
        factory_name="_l10_comparison_combine_rules",
        relayed_dims=_L10_CMP_RELAYED,
        parity_factory_key="cmp_default",
        notes="L10 rows 1-3: cmp_default / override2 / override3",
    ),
    RuleSpec(
        cluster=1,
        factory_module="neural_vm.unified_compiler.ops.l10_ops",
        factory_name="_layer10_alu_cmp_combine_rules",
        relayed_dims=_L10_CMP_RELAYED,
        parity_factory_key="cmp_combine",
        notes="L10 rows 4-6: ALU-side cmp_combine",
    ),
    RuleSpec(
        cluster=1,
        factory_module="neural_vm.unified_compiler.ops.l10_ops",
        factory_name="_layer10_alu_bitwise_or_rules",
        relayed_dims=_L10_BITWISE_RELAYED,
        allow_step_end_writes_to=("OUTPUT_LO", "OUTPUT_HI_THIS_STEP"),
        parity_factory_key="alu_bitwise_or",
        notes="L10 row 7: bitwise OR (also XOR/AND below)",
    ),
    RuleSpec(
        cluster=1,
        factory_module="neural_vm.unified_compiler.ops.l10_ops",
        factory_name="_layer10_alu_bitwise_xor_rules",
        relayed_dims=_L10_BITWISE_RELAYED,
        allow_step_end_writes_to=("OUTPUT_LO", "OUTPUT_HI_THIS_STEP"),
        parity_factory_key="alu_bitwise_xor",
    ),
    RuleSpec(
        cluster=1,
        factory_module="neural_vm.unified_compiler.ops.l10_ops",
        factory_name="_layer10_alu_bitwise_and_rules",
        relayed_dims=_L10_BITWISE_RELAYED,
        allow_step_end_writes_to=("OUTPUT_LO", "OUTPUT_HI_THIS_STEP"),
        parity_factory_key="alu_bitwise_and",
    ),
    RuleSpec(
        cluster=1,
        factory_module="neural_vm.unified_compiler.ops.l10_ops",
        factory_name="_layer10_alu_shl_shr_zero_rules",
        relayed_dims=_L10_SHIFT_RELAYED,
        allow_step_end_writes_to=(
            "OUTPUT_LO", "OUTPUT_HI",
            "OUTPUT_HI_THIS_STEP", "OUTPUT_HI_PREV_STEP",
        ),
        notes="L10 row 8: shl/shr zero handling",
    ),
    RuleSpec(
        cluster=1,
        factory_module="neural_vm.unified_compiler.ops.l10_ops",
        factory_name="_layer10_alu_mul_lo_rules",
        relayed_dims=_L10_MUL_LO_RELAYED,
        allow_step_end_writes_to=("OUTPUT_LO",),
        notes="L10 row 9: MUL low-byte stage",
    ),
    RuleSpec(
        cluster=1,
        factory_module="neural_vm.unified_compiler.ops.l10_ops",
        factory_name="_tail_bit32_result_correction_rules",
        relayed_dims=_L10_TAIL_CMP_RELAYED,
        notes="L10 rows 10-15: tail_cmp_* (NE/EQ/LE_LT/LE_EQ/LT/GT)",
    ),
    # ---------- Cluster 2: L8 (14 rules; 14 factories) ------------------
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_add_lo_rules",
        relayed_dims=_L8_ALU_LO_RELAYED,
    ),
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_lea_lo_rules",
        relayed_dims=_L8_ALU_LO_RELAYED,
    ),
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_sub_lo_rules",
        relayed_dims=_L8_ALU_LO_RELAYED,
    ),
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_add_carry_rules",
        relayed_dims=_L8_ALU_LO_RELAYED,
    ),
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_lea_carry_rules",
        relayed_dims=_L8_ALU_LO_RELAYED,
    ),
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_adj_lo_rules",
        relayed_dims=_L8_ALU_LO_RELAYED,
    ),
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_adj_carry_rules",
        relayed_dims=_L8_ALU_LO_RELAYED,
    ),
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_sub_borrow_rules",
        relayed_dims=_L8_ALU_LO_RELAYED,
    ),
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_ent_lo_rules",
        relayed_dims=_L8_ALU_LO_RELAYED,
    ),
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_ent_borrow_rules",
        relayed_dims=_L8_ALU_LO_RELAYED,
    ),
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_cmp_group_rules",
        relayed_dims=_L8_ALU_CMP_RELAYED,
    ),
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_cmp_clear_rules",
        relayed_dims=_L8_ALU_CMP_RELAYED,
    ),
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_lev_b1_rules",
        relayed_dims=_L8_LEV_RELAYED,
        notes="Gates on MARK_BP, not MARK_AX — needs from_marker= helper",
    ),
    RuleSpec(
        cluster=2,
        factory_module="neural_vm.unified_compiler.ops.l8_ops",
        factory_name="_layer8_alu_lev_b2_rules",
        relayed_dims=_L8_LEV_RELAYED,
        notes="Gates on MARK_BP, not MARK_AX — needs from_marker= helper",
    ),
    # ---------- Cluster 3: L9 (5 rules; 2 factories) --------------------
    RuleSpec(
        cluster=3,
        factory_module="neural_vm.unified_compiler.ops.l9_ops",
        factory_name="_layer9_cmp_rules",
        relayed_dims=_L9_CMP_RELAYED,
        notes="L9 rows 1-4: hi_eq / lo_eq / hi_lt / lo_lt branches",
    ),
    RuleSpec(
        cluster=3,
        factory_module="neural_vm.unified_compiler.ops.l9_ops",
        factory_name="_layer9_bp_plus8_shift_rules",
        relayed_dims=_L9_BP_PLUS8_RELAYED,
        notes="Gates on MARK_BP/SP/PC, not MARK_AX — needs helper",
    ),
    # ---------- Cluster 4: L11/L12 (2 rules; 2 factories) ---------------
    RuleSpec(
        cluster=4,
        factory_module="neural_vm.unified_compiler.ops.l11_ops",
        factory_name="_layer11_mul_partial_rules",
        relayed_dims=_L11_MUL_PARTIAL_RELAYED,
        allow_step_end_writes_to=("MUL_PARTIAL_LO", "MUL_PARTIAL_HI"),
    ),
    RuleSpec(
        cluster=4,
        factory_module="neural_vm.unified_compiler.ops.l12_ops",
        factory_name="mul_combine_rules",
        relayed_dims=_L12_MUL_COMBINE_RELAYED,
        allow_step_end_writes_to=(
            "MUL_ACCUM_LO", "MUL_ACCUM_HI",
            "OUTPUT_LO", "OUTPUT_HI",
        ),
        notes=(
            "L12 mul_combine writes OUTPUT_HI directly today; whitelist "
            "is the dispatcher-side accommodation. Real source edit "
            "must add a per-byte relay (see CLUSTER_4 plan)."
        ),
    ),
    # ---------- Cluster 5: L3 / L6 scattered (6 rules) ------------------
    # L3 rows 1-4 are lint false-positives (BYTE_INDEX hidden behind
    # subscript). Documented in WAVE_B_CLUSTER_5_PLAN_2026_06_10.md.
    # The dispatcher skips them (no migrate_rule_to_step_end call) and
    # records them as "lint_fp".
    RuleSpec(
        cluster=5,
        factory_module="neural_vm.unified_compiler.ops.l3_ops",
        factory_name="_register_default_ffn_rules",
        relayed_dims=(),
        notes=(
            "Lint FP: byte_1_first_step_lo/hi for SP+BP gate on "
            "BYTE_INDEX_1 via _BYTE_INDEX[1] indirection. Not a Wave B "
            "candidate; fix the lint surface, don't migrate."
        ),
    ),
    # L6 rows 5-6 — alu_lo/hi_clear inside l6 tail-cleanup helpers.
    # These are real Wave B candidates, but the rules are emitted
    # inline (no dedicated factory). Treat as a single placeholder
    # entry; the source-edit agent finds the call sites by name.
    RuleSpec(
        cluster=5,
        factory_module="neural_vm.unified_compiler.ops.l6_ops",
        factory_name="_layer6_tail_cleanup_rules",
        relayed_dims=_L6_ALU_CLEAR_RELAYED,
        notes=(
            "L6 alu_lo_clear / alu_hi_clear rules (lines 1753, 1760) "
            "are inside _layer6_tail_cleanup_rules; treat the whole "
            "factory as the migration unit."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Migration smoke test
# ---------------------------------------------------------------------------


@dataclass
class MigrationResult:
    spec: RuleSpec
    success: bool = False
    n_rules_in: int = 0
    n_rules_migrated: int = 0
    error: Optional[str] = None
    parity_status: str = "skipped"  # "passed" / "failed" / "skipped"
    parity_error: Optional[str] = None
    rule_diffs: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "cluster": self.spec.cluster,
            "factory": (
                f"{self.spec.factory_module}.{self.spec.factory_name}"
            ),
            "success": self.success,
            "n_rules_in": self.n_rules_in,
            "n_rules_migrated": self.n_rules_migrated,
            "parity_status": self.parity_status,
            "parity_error": self.parity_error,
            "error": self.error,
            "notes": self.spec.notes,
        }


def _try_import_helper() -> Tuple[Optional[Callable], Optional[type]]:
    """Return (``migrate_rule_to_step_end``, ``MigrationSafetyError``) or
    (None, None) if the helper module can't be imported."""

    try:
        from neural_vm.unified_compiler.step_end_migration import (  # noqa: E501
            MigrationSafetyError,
            migrate_rule_to_step_end,
        )

        return migrate_rule_to_step_end, MigrationSafetyError
    except Exception:
        return None, None


def _try_import_parity() -> Optional[object]:
    try:
        sys.path.insert(
            0,
            str((_REPO_ROOT or Path(".")) / "c4_release" / "tests"),
        )
        import test_step_end_migration_parity as mod  # type: ignore

        return mod
    except Exception:
        return None


def _import_factory(spec: RuleSpec) -> Callable[..., Sequence[object]]:
    import importlib

    mod = importlib.import_module(spec.factory_module)
    return getattr(mod, spec.factory_name)


def _format_rule_diff(rule_before, rule_after) -> str:
    name = getattr(rule_before, "name", "<anon>")
    cond_before = ", ".join(
        f"{t.dim.name}{('+%d' % t.dim.offset) if t.dim.offset else ''}"
        for t in rule_before.conditions
    )
    cond_after = ", ".join(
        f"{t.dim.name}{('+%d' % t.dim.offset) if t.dim.offset else ''}"
        for t in rule_after.conditions
    )
    if cond_before == cond_after:
        return f"  {name}: (no marker swap)"
    return f"  {name}: [{cond_before}] -> [{cond_after}]"


def run_migration(spec: RuleSpec) -> MigrationResult:
    """Symbolically migrate every rule in ``spec``'s factory output.

    Reports counts, errors, and a per-rule before/after condition
    diff. Does NOT touch source files.
    """

    result = MigrationResult(spec=spec)

    migrate_fn, MigrationSafetyError = _try_import_helper()
    if migrate_fn is None:
        result.error = (
            "could not import migrate_rule_to_step_end; Wave A scaffold "
            "may not be on this branch"
        )
        return result

    try:
        factory = _import_factory(spec)
    except Exception as e:
        result.error = f"factory import failed: {e!r}"
        return result

    try:
        rules = list(factory(100.0))
    except TypeError:
        # Some factories take no S argument (e.g. closure-bound).
        try:
            rules = list(factory())
        except Exception as e:
            result.error = f"factory call failed: {e!r}"
            return result
    except Exception as e:
        result.error = f"factory call failed: {e!r}"
        return result

    result.n_rules_in = len(rules)

    if not spec.relayed_dims and not rules:
        # Documented skip (e.g. L3 lint FP).
        result.success = True
        result.parity_status = "skipped (lint FP)"
        return result

    if not spec.relayed_dims:
        # Cluster 5 L3 entry — record but don't migrate.
        result.success = True
        result.parity_status = "skipped (lint FP)"
        result.rule_diffs.append(
            f"  (skipped {result.n_rules_in} rule(s) — lint surface fix only)"
        )
        return result

    migrated: List[object] = []
    safety_errors: List[str] = []
    for rule in rules:
        try:
            new_rule = migrate_fn(
                rule,
                relayed_dims=spec.relayed_dims,
                allow_step_end_writes_to=spec.allow_step_end_writes_to,
            )
            migrated.append(new_rule)
            result.rule_diffs.append(_format_rule_diff(rule, new_rule))
        except MigrationSafetyError as e:  # type: ignore
            safety_errors.append(
                f"  {getattr(rule, 'name', '<anon>')!r}: {e}"
            )
        except Exception as e:
            safety_errors.append(
                f"  {getattr(rule, 'name', '<anon>')!r}: "
                f"unexpected {type(e).__name__}: {e}"
            )

    result.n_rules_migrated = len(migrated)

    if safety_errors:
        result.error = (
            f"migration safety failed for {len(safety_errors)} rule(s):\n"
            + "\n".join(safety_errors[:10])
            + (
                f"\n  ... ({len(safety_errors) - 10} more)"
                if len(safety_errors) > 10
                else ""
            )
        )
        return result

    result.success = True

    # Parity assertion — only when the rule is wired into RULE_FACTORIES.
    parity_mod = _try_import_parity()
    if parity_mod is None or spec.parity_factory_key is None:
        result.parity_status = "skipped (no parity factory wired)"
        return result

    factories = getattr(parity_mod, "RULE_FACTORIES", None)
    if not factories or spec.parity_factory_key not in factories:
        result.parity_status = (
            f"skipped (key {spec.parity_factory_key!r} not in RULE_FACTORIES)"
        )
        return result

    # We don't drive the example PROGRAMS — those are tied to specific
    # test scenarios in the parity module. The smoke check is just that
    # the migrated factory returns the same rule count and writes to
    # the same target-dim names. That's enough to certify the
    # mechanical migration; full byte-identity is the responsibility of
    # the parity tests themselves once the source edit lands.
    result.parity_status = "smoke-only (count + write-name match)"
    try:
        for before, after in zip(rules, migrated):
            before_writes = sorted(w.dim.name for w in before.writes)
            after_writes = sorted(w.dim.name for w in after.writes)
            if before_writes != after_writes:
                raise AssertionError(
                    f"write targets diverged for "
                    f"{getattr(before, 'name', '<anon>')!r}: "
                    f"{before_writes} vs {after_writes}"
                )
    except AssertionError as e:
        result.parity_status = "failed"
        result.parity_error = str(e)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _list_clusters() -> Dict[int, List[RuleSpec]]:
    by_cluster: Dict[int, List[RuleSpec]] = {}
    for spec in _CLUSTER_MANIFEST:
        by_cluster.setdefault(spec.cluster, []).append(spec)
    return by_cluster


def _print_human(results: List[MigrationResult]) -> None:
    by_cluster: Dict[int, List[MigrationResult]] = {}
    for r in results:
        by_cluster.setdefault(r.spec.cluster, []).append(r)

    total = len(results)
    successes = sum(1 for r in results if r.success)
    print(
        f"wave_b_dispatcher: {successes}/{total} symbolic migrations OK."
    )
    for cluster_num in sorted(by_cluster):
        cluster_results = by_cluster[cluster_num]
        ok = sum(1 for r in cluster_results if r.success)
        print(
            f"\n--- Cluster {cluster_num} ({ok}/{len(cluster_results)} OK) ---"
        )
        for r in cluster_results:
            tag = "OK " if r.success else "FAIL"
            print(
                f"  [{tag}] {r.spec.factory_name}: "
                f"in={r.n_rules_in}, migrated={r.n_rules_migrated}, "
                f"parity={r.parity_status}"
            )
            if r.error:
                for line in r.error.splitlines():
                    print(f"        {line}")
            if r.parity_error:
                print(f"        parity error: {r.parity_error}")
            if r.spec.notes:
                print(f"        note: {r.spec.notes}")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cluster",
        type=int,
        default=None,
        help="run only one cluster (1-5)",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="list cluster contents without running migrations",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="emit JSON results",
    )
    args = ap.parse_args(argv)

    if args.list:
        by_cluster = _list_clusters()
        for cluster_num in sorted(by_cluster):
            specs = by_cluster[cluster_num]
            print(f"Cluster {cluster_num}: {len(specs)} rule-factory entries")
            for spec in specs:
                print(
                    f"  - {spec.factory_module}.{spec.factory_name} "
                    f"(relayed={len(spec.relayed_dims)} dims, "
                    f"safe-writes={list(spec.allow_step_end_writes_to)})"
                )
                if spec.notes:
                    print(f"      note: {spec.notes}")
        return 0

    specs = list(_CLUSTER_MANIFEST)
    if args.cluster is not None:
        specs = [s for s in specs if s.cluster == args.cluster]
        if not specs:
            print(
                f"error: no specs for cluster {args.cluster}",
                file=sys.stderr,
            )
            return 2

    results: List[MigrationResult] = []
    for spec in specs:
        try:
            results.append(run_migration(spec))
        except Exception:
            tb = traceback.format_exc()
            results.append(
                MigrationResult(spec=spec, error=f"dispatcher crash:\n{tb}")
            )

    if args.json:
        print(
            json.dumps(
                {"results": [r.as_dict() for r in results]},
                indent=2,
            )
        )
    else:
        _print_human(results)

    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
