"""List all ops, count their FFNRules, identify which carry compiler_ir."""
from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops

ops = all_core_ops()
print(f"total ops: {len(ops)}")

buckets = {"has_rules": [], "no_rules": [], "no_ir": []}
for op in ops:
    name = getattr(op, "name", repr(op))
    ir = getattr(op, "compiler_ir", None)
    if ir is None:
        buckets["no_ir"].append((name, 0))
        continue
    n_rules = 0
    if hasattr(ir, "rules"):
        n_rules = len(ir.rules)
    elif hasattr(ir, "layers"):
        for layer in ir.layers:
            ffn = getattr(layer, "ffn", None)
            if ffn is not None:
                n_rules += len(getattr(ffn, "rules", []))
    bucket = "has_rules" if n_rules > 0 else "no_rules"
    buckets[bucket].append((name, n_rules))

print(f"has_rules: {len(buckets['has_rules'])} ops")
print(f"no_rules: {len(buckets['no_rules'])} ops")
print(f"no_ir: {len(buckets['no_ir'])} ops")
print()
print("Top 25 by rule count:")
top = sorted(buckets["has_rules"], key=lambda x: -x[1])[:25]
for name, n in top:
    print(f"  {n:>5}  {name}")
print()
print("Total rules across all ops:", sum(n for _, n in buckets["has_rules"]))
