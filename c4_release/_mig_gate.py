"""Per-op migration gate: symbolic==lowered FFN + validate_requires_op_refs."""
import importlib
import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from neural_vm.vm_step import _SetDim
from neural_vm.unified_compiler.ir import compare_symbolic_to_lowered_ffn
from neural_vm.unified_compiler.layer_compiler import validate_requires_op_refs

def _dim_positions_map():
    d = {}
    for k in dir(_SetDim):
        if k.startswith("_"): continue
        v = getattr(_SetDim, k)
        if isinstance(v, int): d[k] = v
    return d

# Pre-existing f32-rounding tolerance: some lowered weights (e.g. L12's
# 5.12 == 16 * 0.32 == S*0.0512) are not exactly representable in float32,
# so the lowered forward reads e.g. 5.119885. These are NOT semantic
# mismatches (the weight matches the lowering contract); a rename never
# touches a numeric value. Use a relative tolerance so the gate flags only
# REAL drift, not the inherent f32 representation error.
_ATOL = 1e-3
_RTOL = 1e-3

def check_ir(spec):
    mod_name, func_name = spec.split(":")
    mod = importlib.import_module(mod_name)
    ir = getattr(mod, func_name)()
    dp = _dim_positions_map(); ok_all = True; found = False
    for li in range(0, 60):
        try: rules = ir.layer(li).ffn.rules
        except Exception: continue
        if not rules: continue
        found = True
        rep = compare_symbolic_to_lowered_ffn(ir, dp, layer_idx=li,
                                              atol=_ATOL, rtol=_RTOL)
        st = "OK" if rep.ok else "FAIL"
        print(f"  [{st}] {spec} layer={li} ({len(rules)} rules)")
        if not rep.ok:
            ok_all = False
            for iss in rep.issues[:8]: print(f"        issue: {iss}")
    if not found: print(f"  [SKIP] {spec}: no ffn rules")
    return ok_all

# Pre-existing dangling ref on the clean committed tree (HEAD b9d8861f):
# layer9_se_relay_slope requires['after']=layer10_residual_alibi_slopes,
# an ATTENTION op the (FFN-op-indexed) validator doesn't see. It is
# unrelated to the rename pass; baselined so the gate flags only NEW refs.
_KNOWN_PREEXISTING_DANGLING = (
    "op 'layer9_se_relay_slope' requires['after']="
    "'layer10_residual_alibi_slopes' references an unknown op",
)

def check_requires():
    from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops as make_all_core_ops
    ops = make_all_core_ops()
    errs = validate_requires_op_refs(ops)
    new_errs = [e for e in errs if str(e) not in _KNOWN_PREEXISTING_DANGLING]
    if new_errs:
        print(f"  [FAIL] validate_requires_op_refs: {len(new_errs)} NEW dangling refs")
        for e in new_errs[:20]: print(f"        {e}")
        return False
    base = len(errs) - len(new_errs)
    print(f"  [OK] validate_requires_op_refs: {len(ops)} ops, "
          f"0 new dangling refs ({base} pre-existing baselined)")
    return True

if __name__ == "__main__":
    all_ok = True
    print("=== compare_symbolic_to_lowered_ffn ===")
    for spec in sys.argv[1:]: all_ok &= check_ir(spec)
    print("=== validate_requires_op_refs ===")
    all_ok &= check_requires()
    print("RESULT:", "ALL OK" if all_ok else "FAILURES")
    sys.exit(0 if all_ok else 1)
