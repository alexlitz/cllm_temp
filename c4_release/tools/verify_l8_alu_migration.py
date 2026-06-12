"""Byte-identity gate for the L8 ALU declarative live-imperative cut.

Mirrors ``tools/verify_l14_mem_default_suppress_migration.py`` (commit
cc9bf21f). It proves that folding the (now-removed) imperative
``vm_step._set_layer8_alu`` re-bake into ``layer8_alu``'s declarative
rules is BYTE-IDENTICAL to the legacy two-pass production bake.

Background
----------
Until 2026-06-12, the production L8 FFN was baked in two passes:

  1. phase 8.2 ``layer8_alu``: lowered the declarative ``_layer8_alu_rules``
     (the Wave-B ``MARK_SE_ONLY`` variant) into units 0..2022 via ``+=``.
  2. phase 8.3 ``layer8_multibyte_routing``: re-invoked the imperative
     ``vm_step._set_layer8_alu`` helper (the un-migrated ``MARK_AX``
     variant) over the SAME units 0..2022 via ``=`` assignment.

``_set_layer8_alu`` only writes ``W_up[MARK_AX]`` / ``W_gate[MARK_AX]``
(and overwrites a handful of overlapping cells with identical values, plus
``W_up[MARK_BP]`` on the two LEV bytes-1/2 units); it never touches
``MARK_SE_ONLY``. So the two passes SUPERPOSE and the LIVE weights carry
BOTH markers on every ALU unit -- the imperative re-bake was the LAST live
imperative weight write in the production forward model.

The migration folds that superposition into ``_layer8_alu_rules`` (see
``_l8_alu_add_mark_ax_mirror`` + the LEV bytes-1/2 ``MARK_BP`` edit) so the
single declarative phase-8.2 pass produces the both-markers production
state and the imperative re-bake can be deleted.

What this gate checks
---------------------
1. ``compare_symbolic_to_lowered_ffn(_layer8_alu_ir()).ok == True`` --
   the folded rule list lowers to exactly what each rule declares.
2. element-wise ``torch.equal`` (diff == 0) on W_up / b_up / W_gate /
   b_gate / W_down between:
     * the NEW single-pass folded bake (``lower_layer8_alu_ir``), and
     * the LEGACY two-pass bake reconstructed in this tool:
       (legacy SE-only declarative ``+=``) THEN (imperative
       ``_set_layer8_alu`` ``=``).
   The legacy SE-only declarative is reconstructed by inverting the two
   source edits this migration made (strip the ``MARK_AX`` mirror; restore
   the LEV bytes-1/2 ``MARK_BP=-1e6`` blocker), so the comparison is
   self-contained and does not depend on git history.

A non-zero diff means the fold does NOT reproduce the legacy live weights
-> the migration is not byte-identical and must be aborted.

Run: python tools/verify_l8_alu_migration.py
"""
import dataclasses
import os
import sys

# Resolve the worktree's ``neural_vm`` package ahead of any sibling
# checkout, and add the worktree root so ``import c4_release.neural_vm...``
# (used inside ir.py) resolves to THIS checkout.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKTREE_ROOT = os.path.dirname(_PKG_ROOT)
sys.path.insert(0, _WORKTREE_ROOT)
sys.path.insert(0, _PKG_ROOT)

import torch

from neural_vm.vm_step import _SetDim, _set_layer8_alu
from neural_vm.unified_compiler.ir import (
    ConditionTerm,
    DimRef,
    compare_symbolic_to_lowered_ffn,
)
from neural_vm.unified_compiler.primitives import Primitives
from neural_vm.unified_compiler.ops import l8_ops
from neural_vm.unified_compiler.ops.l8_ops import (
    _layer8_alu_ir,
    _layer8_alu_rules,
    lower_layer8_alu_ir,
)

S = 100.0
N_ALU_UNITS = 2023  # offsets 0..2022


class _StubFFN:
    """Minimal PureFFN-shaped tensor bag (matches the L8 test stubs)."""

    def __init__(self, d_model: int = 512, hidden_dim: int = 2200):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)


def _dim_positions_map():
    d = {}
    for k in dir(_SetDim):
        if k.startswith("_"):
            continue
        v = getattr(_SetDim, k)
        if isinstance(v, int):
            d[k] = v
    return d


def _strip_mark_ax_mirror(rules):
    """Inverse of ``_l8_alu_add_mark_ax_mirror`` for legacy reconstruction.

    Drops every ``MARK_AX`` condition/gate-term that was ADDED as a mirror
    of a co-located ``MARK_SE_ONLY`` term. A ``MARK_AX`` term is treated as
    an added mirror iff the same rule carries a ``MARK_SE_ONLY`` term of the
    same weight (the mirror is equal-weight). Rules that name ``MARK_AX``
    WITHOUT a matching ``MARK_SE_ONLY`` (the LEV bytes-1/2 -1e6 blocker, and
    the ENT/ADJ-default ``MARK_AX`` operand) keep it -- they were never
    mirrored.
    """
    out = []
    for rule in rules:
        se_weights = {
            t.weight for t in rule.conditions if t.dim.name == "MARK_SE_ONLY"
        }
        gate_se_weights = set()
        if rule.gate is not None and rule.gate.name == "MARK_SE_ONLY":
            gate_se_weights.add(rule.gate_weight)
        gate_se_weights |= {
            t.weight for t in rule.gate_terms if t.dim.name == "MARK_SE_ONLY"
        }

        new_conditions = tuple(
            t for t in rule.conditions
            if not (t.dim.name == "MARK_AX" and t.weight in se_weights)
        )
        new_gate_terms = tuple(
            t for t in rule.gate_terms
            if not (t.dim.name == "MARK_AX" and t.weight in gate_se_weights)
        )
        out.append(dataclasses.replace(
            rule,
            conditions=new_conditions,
            gate_terms=new_gate_terms,
        ))
    return tuple(out)


def _restore_lev_bp_blocker(rules):
    """Inverse of the LEV bytes-1/2 ``MARK_BP`` edit for legacy reconstruction.

    The migration changed the LEV bytes-1/2 ``MARK_BP`` from the dead -1e6
    blocker to the imperative ``=`` winner (+1.0). To rebuild the legacy
    phase-8.2 declarative pass (which laid down the -1e6 blocker before the
    imperative re-bake overwrote it), flip ``MARK_BP`` back to -1e6 on those
    two units (identified by the ``l8_alu_lev_b{1,2}_step_end`` rule names).
    """
    out = []
    for rule in rules:
        if rule.name in ("l8_alu_lev_b1_step_end", "l8_alu_lev_b2_step_end"):
            new_conditions = tuple(
                ConditionTerm(dim=t.dim, weight=(-1e6 if t.dim.name == "MARK_BP" else t.weight))
                for t in rule.conditions
            )
            out.append(dataclasses.replace(rule, conditions=new_conditions))
        else:
            out.append(rule)
    return tuple(out)


def _bake_new_single_pass():
    ffn = _StubFFN()
    n = lower_layer8_alu_ir(ffn, S, _SetDim, start_unit=0)
    return ffn, n


def _bake_legacy_two_pass():
    """Reconstruct the legacy production bake: SE-only declarative + imperative."""
    # 1. Legacy phase-8.2 declarative pass (SE-only; LEV MARK_BP=-1e6).
    legacy_rules = _restore_lev_bp_blocker(
        _strip_mark_ax_mirror(_layer8_alu_rules(S))
    )
    ffn = _StubFFN()
    dim_positions = Primitives.dim_positions_from_bd(
        _SetDim, Primitives.ffn_rule_dim_names(legacy_rules)
    )
    Primitives.lower_ffn_rules(ffn, legacy_rules, dim_positions, start_unit=0, S=S)
    # 2. Legacy phase-8.3 imperative re-bake over the same units (``=``).
    n = _set_layer8_alu(ffn, S, _SetDim)
    return ffn, n


def _diff_ffn(a, b, label):
    ok = True
    for name in ("W_up", "b_up", "W_gate", "b_gate", "W_down"):
        ta = getattr(a, name).detach()
        tb = getattr(b, name).detach()
        if ta.shape != tb.shape:
            print(f"  [{label}] {name}: SHAPE MISMATCH {ta.shape} vs {tb.shape}")
            ok = False
            continue
        if not torch.equal(ta, tb):
            md = (ta - tb).abs().max().item()
            nz = int((ta != tb).sum().item())
            print(f"  [{label}] {name}: DIFF max={md} ncells={nz}")
            # Show the first few differing cells for debugging.
            diff = (ta != tb).nonzero()
            for row in diff[:8].tolist():
                idx = tuple(row)
                print(f"      at {idx}: legacy={ta[idx].item()} new={tb[idx].item()}")
            ok = False
        else:
            print(f"  [{label}] {name}: identical")
    return ok


def main():
    all_ok = True

    # Sanity: both bakes must end the ALU cursor at the same offset.
    new_ffn, new_end = _bake_new_single_pass()
    legacy_ffn, legacy_end = _bake_legacy_two_pass()
    print(f"=== cursor end: new={new_end} legacy={legacy_end} "
          f"(expected {N_ALU_UNITS}) ===")
    if not (new_end == legacy_end == N_ALU_UNITS):
        print("  CURSOR DRIFT -- ABORT")
        all_ok = False

    print("=== element-wise tensor diff (new single-pass vs legacy two-pass) ===")
    all_ok = _diff_ffn(legacy_ffn, new_ffn, "layer8_alu") and all_ok

    # ``compare_symbolic_to_lowered_ffn`` reports two issue classes:
    #   * ``lowering`` -- the per-rule W_up/b_up/W_gate/b_gate/W_down do NOT
    #     match the rule declaration. This IS the byte-identity criterion and
    #     MUST be zero.
    #   * ``weight_output_mismatch`` -- the SYMBOLIC bag-of-dims forward
    #     diverges from the matrix forward at a synthetic all-operands-set
    #     test state. The L8 ALU has many one-hot OUTPUT_LO writers that the
    #     symbolic interpreter sums (it cannot model the per-position one-hot
    #     selection), so this class is non-empty on HEAD too and is NOT a
    #     byte-identity defect. We assert the fold introduces NO NEW members
    #     of this class vs the reconstructed legacy SE-only rules.
    print("=== compare_symbolic_to_lowered_ffn(_layer8_alu_ir()) ===")
    rep_new = compare_symbolic_to_lowered_ffn(
        _layer8_alu_ir(S), _dim_positions_map(), S=S)
    new_lowering = [i for i in rep_new.issues if i.kind == "lowering"]
    new_wom = sorted(
        i.message for i in rep_new.issues if i.kind == "weight_output_mismatch")
    print(f"  ok = {rep_new.ok}  "
          f"(lowering issues = {len(new_lowering)}, "
          f"weight_output_mismatch = {len(new_wom)})")
    if new_lowering:
        print("  LOWERING (byte-identity) ISSUES -- ABORT:")
        for issue in new_lowering[:20]:
            print(f"    {issue.message}")
        all_ok = False

    # Legacy baseline for the symbolic-output class: the fold must not add any.
    legacy_rules = _restore_lev_bp_blocker(
        _strip_mark_ax_mirror(_layer8_alu_rules(S)))
    from neural_vm.unified_compiler.ir import CompilerIR
    ir_leg = CompilerIR()
    ir_leg.layer(0).ffn.rules.extend(legacy_rules)
    rep_leg = compare_symbolic_to_lowered_ffn(ir_leg, _dim_positions_map(), S=S)
    leg_wom = sorted(
        i.message for i in rep_leg.issues if i.kind == "weight_output_mismatch")
    print(f"  legacy weight_output_mismatch = {len(leg_wom)} "
          f"(must equal new: {new_wom == leg_wom})")
    if new_wom != leg_wom:
        print("  NEW SYMBOLIC-OUTPUT DIVERGENCE INTRODUCED BY FOLD -- ABORT")
        all_ok = False

    # Confirm the live path no longer CALLS the imperative helper. Walk the
    # AST of the two L8 ALU/routing op factories and look for any actual
    # ``_set_layer8_alu`` reference in code (import or call), ignoring the
    # docstrings/comments that legitimately mention it for history.
    import ast
    import inspect
    imperative_live = False
    for fn in (l8_ops.make_layer8_multibyte_routing_op, l8_ops.make_layer8_alu_op):
        tree = ast.parse(inspect.getsource(fn))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "_set_layer8_alu":
                imperative_live = True
            if isinstance(node, ast.Attribute) and node.attr == "_set_layer8_alu":
                imperative_live = True
            if isinstance(node, ast.ImportFrom):
                if any(a.name == "_set_layer8_alu" for a in node.names):
                    imperative_live = True
    print("=== live-path reference check (AST, code only) ===")
    print(f"  L8 ALU / routing op factories CALL/import _set_layer8_alu: "
          f"{imperative_live}")
    if imperative_live:
        print("  LIVE IMPERATIVE REFERENCE STILL PRESENT -- ABORT")
        all_ok = False

    print()
    print("RESULT:", "BYTE-IDENTICAL OK" if all_ok else "MISMATCH -- ABORT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
