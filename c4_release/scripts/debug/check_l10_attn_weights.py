"""Check which heads of block 11 attn have nonzero weights writing to OUTPUT_LO/HI/CARRY."""
import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD

runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
model = runner.model
model.eval()

attn = model.blocks[11].attn
W_o = attn.W_o.data
if W_o.is_sparse:
    W_o = W_o.to_dense()

print(f"W_o shape: {W_o.shape}")
print(f"num_heads: {attn.num_heads}, head_dim: {attn.head_dim}")

# For each head, check which output dims it writes to
HD = W_o.shape[1] // attn.num_heads
print(f"HD = {HD}")

for h in range(attn.num_heads):
    base = h * HD
    out_writes = []
    for out_dim in range(W_o.shape[0]):
        # Check if any column in this head's range is non-zero
        if W_o[out_dim, base:base+HD].abs().sum().item() > 0:
            # Get names
            name = ""
            for attr in dir(BD):
                if attr.startswith("_"):
                    continue
                val_attr = getattr(BD, attr)
                if isinstance(val_attr, int) and val_attr == out_dim:
                    name = attr
                    break
            if not name:
                for base_attr, count in [("OUTPUT_LO", 16), ("OUTPUT_HI", 16), ("CARRY", 16),
                                          ("EMBED_LO", 16), ("EMBED_HI", 16), ("TEMP", 16)]:
                    if hasattr(BD, base_attr):
                        base_val = getattr(BD, base_attr)
                        if base_val <= out_dim < base_val + count:
                            name = f"{base_attr}+{out_dim - base_val}"
                            break
            out_writes.append((out_dim, name, W_o[out_dim, base:base+HD].abs().sum().item()))
    print(f"\nHead {h} (slots {base}:{base+HD}) writes to {len(out_writes)} output dims:")
    for d, n, val in out_writes[:20]:
        print(f"  out_dim {d} ({n}): total |W_o| = {val:.2f}")
    if len(out_writes) > 20:
        print(f"  ... and {len(out_writes) - 20} more")

# Also check V weights
W_v = attn.W_v.data
if W_v.is_sparse:
    W_v = W_v.to_dense()
print(f"\n\nW_v shape: {W_v.shape}")
for h in range(attn.num_heads):
    base = h * HD
    # Which input dims are read?
    reads = (W_v[base:base+HD].abs().sum(dim=0) > 0).nonzero(as_tuple=True)[0].tolist()
    print(f"\nHead {h} V reads from dims: {len(reads)} dims, top by |sum|:")
    sums = W_v[base:base+HD].abs().sum(dim=0)
    topk = sums.topk(min(10, sums.shape[0]))
    for i in topk.indices:
        d = i.item()
        name = ""
        for attr in dir(BD):
            if attr.startswith("_"):
                continue
            val_attr = getattr(BD, attr)
            if isinstance(val_attr, int) and val_attr == d:
                name = attr
                break
        if not name:
            for base_attr, count in [("OUTPUT_LO", 16), ("OUTPUT_HI", 16), ("CARRY", 16),
                                      ("EMBED_LO", 16), ("EMBED_HI", 16), ("CLEAN_EMBED_LO", 16),
                                      ("CLEAN_EMBED_HI", 16), ("TEMP", 16)]:
                if hasattr(BD, base_attr):
                    base_val = getattr(BD, base_attr)
                    if base_val <= d < base_val + count:
                        name = f"{base_attr}+{d - base_val}"
                        break
        print(f"  dim {d} ({name}): |sum|={sums[d].item():.2f}")
