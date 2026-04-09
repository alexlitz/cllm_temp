"""Check if Layer 1 attention weights are actually set."""

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim
import torch
import sys

BD = _SetDim

print("Creating model and setting weights...", file=sys.stderr)
model = AutoregressiveVM()
set_vm_weights(model)

# Get Layer 1 attention
attn1 = model.blocks[1].attn

print("\nLayer 1 Attention Weight Analysis:", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Check total non-zero weights
total_weights = attn1.W_q.numel() + attn1.W_k.numel() + attn1.W_v.numel() + attn1.W_o.numel()
nonzero_q = (attn1.W_q.data.abs() > 1e-6).sum().item()
nonzero_k = (attn1.W_k.data.abs() > 1e-6).sum().item()
nonzero_v = (attn1.W_v.data.abs() > 1e-6).sum().item()
nonzero_o = (attn1.W_o.data.abs() > 1e-6).sum().item()
nonzero_total = nonzero_q + nonzero_k + nonzero_v + nonzero_o

print(f"Total weights: {total_weights}", file=sys.stderr)
print(f"Non-zero weights:", file=sys.stderr)
print(f"  W_q: {nonzero_q} / {attn1.W_q.numel()}", file=sys.stderr)
print(f"  W_k: {nonzero_k} / {attn1.W_k.numel()}", file=sys.stderr)
print(f"  W_v: {nonzero_v} / {attn1.W_v.numel()}", file=sys.stderr)
print(f"  W_o: {nonzero_o} / {attn1.W_o.numel()}", file=sys.stderr)
print(f"  Total: {nonzero_total} / {total_weights} ({100.0 * nonzero_total / total_weights:.2f}%)", file=sys.stderr)
print("", file=sys.stderr)

if nonzero_total == 0:
    print("❌ CRITICAL: Layer 1 attention has NO non-zero weights!", file=sys.stderr)
    print("   Weights were not set by set_vm_weights()", file=sys.stderr)
    sys.exit(1)

# Check specifically for threshold head configuration (heads 0, 1, 2)
# Threshold heads should have:
# - W_q[base, BD.CONST] = 8.0 * slope
# - W_k[base, BD.IS_MARK] = threshold
# - W_v and W_o configured for output

HD = 64  # head_dim for 512 / 8 heads
print("Checking threshold heads (heads 0, 1, 2):", file=sys.stderr)
print("", file=sys.stderr)

# Expected thresholds
thresholds = [0.5, 1.5, 2.5]
out_bases = [BD.L1H0, BD.L1H1, BD.L1H2]

for i, (h, t, out_base) in enumerate(zip([0, 1, 2], thresholds, out_bases)):
    base = h * HD

    # Check W_q[base, BD.CONST]
    wq_const = attn1.W_q[base, BD.CONST].item()

    # Check W_k[base, BD.IS_MARK]
    wk_is_mark = attn1.W_k[base, BD.IS_MARK].item()

    # Check W_v and W_o for output
    wv_nonzero = (attn1.W_v[base:base+HD, :].abs() > 1e-6).sum().item()
    wo_nonzero = (attn1.W_o[out_base:out_base+7, base:base+HD].abs() > 1e-6).sum().item()

    print(f"Head {h} (threshold {t}):", file=sys.stderr)
    print(f"  W_q[{base}, CONST={BD.CONST}] = {wq_const:.3f} (expected: ~80.0)", file=sys.stderr)
    print(f"  W_k[{base}, IS_MARK={BD.IS_MARK}] = {wk_is_mark:.3f} (expected: {t})", file=sys.stderr)
    print(f"  W_v non-zero: {wv_nonzero} / {HD} rows", file=sys.stderr)
    print(f"  W_o[{out_base}:{out_base+7}, {base}:{base+HD}] non-zero: {wo_nonzero} / {7 * HD}", file=sys.stderr)

    if abs(wq_const) < 0.1:
        print(f"  ⚠ WARNING: W_q[CONST] not set!", file=sys.stderr)
    if abs(wk_is_mark - t) > 0.1:
        print(f"  ⚠ WARNING: W_k[IS_MARK] = {wk_is_mark:.3f}, expected {t}", file=sys.stderr)
    if wv_nonzero == 0:
        print(f"  ⚠ WARNING: W_v not configured!", file=sys.stderr)
    if wo_nonzero == 0:
        print(f"  ⚠ WARNING: W_o not configured!", file=sys.stderr)

    print("", file=sys.stderr)

# Check for marker index outputs in W_o
print("Checking W_o outputs to L1H1[AX] (dim {})".format(BD.L1H1 + 1), file=sys.stderr)

AX_I = 1
base_h1 = 1 * HD
wo_l1h1_ax = attn1.W_o[BD.L1H1 + AX_I, base_h1:base_h1+HD].abs().sum().item()

print(f"  W_o[L1H1+AX_I={BD.L1H1 + AX_I}, head1_range] sum: {wo_l1h1_ax:.3f}", file=sys.stderr)

if wo_l1h1_ax < 0.1:
    print(f"  ⚠ WARNING: No W_o weights writing to L1H1[AX]!", file=sys.stderr)
else:
    print(f"  ✓ W_o configured for L1H1[AX]", file=sys.stderr)

print("", file=sys.stderr)
print("=" * 70, file=sys.stderr)
