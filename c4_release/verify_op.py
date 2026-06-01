"""Single-op verifier: run strength + scope on one op, print counts and details.

Usage:
    CUDA_VISIBLE_DEVICES="" python verify_op.py <op_name>

Where op_name is one of:
    phase_a_ffn
    layer8_multibyte_routing
    layer15_nibble_copy
    layer16_lev_routing
    tail_bit32_result_correction
"""
import sys
import time

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(1)
target = sys.argv[1]

from neural_vm.dim_registry import build_default_registry
from neural_vm.unified_compiler.decl_verifier import (
    verify_rule_strength,
    verify_rule_scopes,
    _collect_ffn_rules_from_op,
)

try:
    from neural_vm.unified_compiler.backbone_bounds import load_default_bounds
    bounds_obj = load_default_bounds()
    bb = bounds_obj.as_strength_bound if bounds_obj is not None else None
except Exception:
    bb = None

registry = build_default_registry()

FACTORIES = {}
def _add(name, modpath, factory_name):
    try:
        mod = __import__(modpath, fromlist=[factory_name])
        FACTORIES[name] = getattr(mod, factory_name)
    except Exception:
        pass

_add("phase_a_ffn", "neural_vm.unified_compiler.ops.l0_ops", "make_phase_a_ffn_op")
_add("layer8_multibyte_routing", "neural_vm.unified_compiler.ops.l8_ops",
     "make_layer8_multibyte_routing_op")
_add("layer15_nibble_copy", "neural_vm.unified_compiler.ops.l15_ops",
     "make_layer15_nibble_copy_op")
_add("layer16_lev_routing", "neural_vm.unified_compiler.ops.l16_ops",
     "make_layer16_lev_routing_op")
_add("tail_bit32_result_correction", "neural_vm.unified_compiler.ops.l10_ops",
     "make_tail_bit32_result_correction_op")

if target not in FACTORIES:
    print(f"unknown op: {target!r}")
    print(f"available: {sorted(FACTORIES)}")
    sys.exit(2)

op = FACTORIES[target]()
n_rules = len(_collect_ffn_rules_from_op(op))
print(f"op={target} rules={n_rules}")

competition = [f() for n, f in FACTORIES.items() if n != target]

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
            f"my={i.get('my_contribution', 0):.1f} comp={i.get('competing_max', 0):.1f} "
            f"shortfall={i.get('shortfall', 0):.1f} top={i.get('top_competitor')!r}"
        )
if sc_issues:
    print()
    print("--- scope_violation (top 20) ---")
    for i in sc_issues[:20]:
        reason = str(i.get("reason", ""))
        if len(reason) > 200:
            reason = reason[:200] + "..."
        print(f"  rule={i.get('rule')!r} reason={reason}")
