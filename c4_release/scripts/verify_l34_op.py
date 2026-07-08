"""Verify post_l9_bz_bnz_pc_override (L34.ffn, 192 units) using the
strength + scope verifier. Read-only attribution helper.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from neural_vm.dim_registry import build_default_registry
from neural_vm.verification.decl_verifier import (
    verify_rule_strength,
    verify_rule_scopes,
    _collect_ffn_rules_from_op,
)
from neural_vm.unified_compiler.ops.l6_ops import (
    make_post_l9_bz_bnz_pc_override_op,
)

# Build competition set: scan known L6/L10/L14 ops that also write to OUTPUT_LO/HI
from neural_vm.unified_compiler.ops.l10_ops import (
    make_tail_bit32_result_correction_op,
)
from neural_vm.unified_compiler.ops.l14_ops import (
    make_layer14_temp_clear_op,
    make_layer14_clear_output_corruption_op,
    make_layer14_clear_mem_marker_output_op,
)

try:
    from neural_vm.verification.backbone_bounds import load_default_bounds
    bounds = load_default_bounds()
    bb = bounds.as_strength_bound if bounds is not None else None
except Exception:
    bb = None

registry = build_default_registry()

op = make_post_l9_bz_bnz_pc_override_op()
n_rules = len(_collect_ffn_rules_from_op(op))
print(f"op=post_l9_bz_bnz_pc_override rules={n_rules}")

competition = [
    make_tail_bit32_result_correction_op(),
    make_layer14_temp_clear_op(),
    make_layer14_clear_output_corruption_op(),
    make_layer14_clear_mem_marker_output_op(),
]

t0 = time.time()
s_issues = verify_rule_strength(
    op, registry, backbone_bounds=bb, ops_for_competition=competition,
)
t_s = time.time() - t0

t0 = time.time()
sc_issues = verify_rule_scopes(op, registry, require_scope=False)
t_sc = time.time() - t0

sv = [i for i in s_issues if i.get("kind") == "strength_violation"]
nd = [i for i in s_issues if i.get("kind") == "no_dominates_at"]
print(f"strength_violation: {len(sv)} (runtime {t_s:.1f}s)")
print(f"no_dominates_at: {len(nd)}")
print(f"scope_violation: {len(sc_issues)} (runtime {t_sc:.1f}s)")
print()
if sv:
    print("--- strength_violation (top 20) ---")
    for i in sv[:20]:
        print(
            f"  rule={i.get('rule')!r} dim={i.get('output_dim')!r} "
            f"my={i.get('my_contribution', 0):.2f} "
            f"comp={i.get('competing_max', 0):.2f} "
            f"shortfall={i.get('shortfall', 0):.2f} "
            f"top={i.get('top_competitor')!r}"
        )
if nd:
    print()
    print("--- no_dominates_at (top 20) ---")
    for i in nd[:20]:
        print(
            f"  rule={i.get('rule')!r} reason={i.get('reason')!r}"
        )
if sc_issues:
    print()
    print("--- scope_violation (top 20) ---")
    for i in sc_issues[:20]:
        reason = str(i.get("reason", ""))
        if len(reason) > 200:
            reason = reason[:200] + "..."
        print(f"  rule={i.get('rule')!r} reason={reason}")
