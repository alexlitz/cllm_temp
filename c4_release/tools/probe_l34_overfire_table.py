"""Wave 1 Cluster C3 — per-rule contribution table for L34
`tail_bit32_result_correction` over-fire at IMM step of
`IMM 0xFF; PSH; IMM 0xD5; XOR; EXIT` (XOR_BASIC shape).

The L34 FFN block (block 34 in the 36-block model) is a SwiGLU PureFFN
with hidden_dim == len(rules) == 2059. Each hidden unit corresponds to
exactly one FFNRule. Per-rule contribution at output dim d:

    h_u = silu(W_up[u] @ x + b_up[u]) * (W_gate[u] @ x + b_gate[u])
    c[d, u] = W_down[d, u] * h_u

We aggregate per rule over the 32 output dims OUTPUT_LO[0..15] and
OUTPUT_HI_THIS_STEP[0..15] and rank by absolute magnitude. Top-10 rules
are reported alongside total destructive sum.
"""
import os
import sys
import json

# Make the package importable regardless of CWD when invoked as a
# script. Walks up from this file's parent (tools/) to find c4_release.
_HERE = os.path.dirname(os.path.abspath(__file__))
_C4_ROOT = os.path.dirname(_HERE)
if _C4_ROOT not in sys.path:
    sys.path.insert(0, _C4_ROOT)

import torch
import torch.nn.functional as F

from neural_vm.constants import INSTR_WIDTH, IMMEDIATE_SIZE, PADDING_SIZE
from neural_vm.embedding import Opcode
from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
from neural_vm.unified_compiler.ops.l10_ops import _tail_bit32_result_correction_rules


def encode_instr(op, imm):
    out = [op & 0xFF]
    for i in range(IMMEDIATE_SIZE):
        out.append((imm >> (i * 8)) & 0xFF)
    for _ in range(PADDING_SIZE):
        out.append(0)
    return out


def classify_rule(name: str) -> str:
    """Bucket rule names into 'legit' (LI/LC/SI load paths) vs 'leaked' (IMM/PSH)."""
    n = name.lower()
    if "stack0_pop_loaded" in n:
        return "leaked"  # OUTPUT_LO/HI-weighted 0.1 family — the over-fire target
    if "stack0_store_loaded" in n or "addr_from_l13" in n or "addr0" in n:
        return "legit_load"
    if "stack0_store_top" in n or "stack0_store_nonzero" in n:
        return "legit_store"
    if "lea" in n or "ax_lea" in n:
        return "legit_lea"
    if "sp_pop" in n or "stack0_pushed_addr" in n:
        return "legit_sp"
    if "exact_output_byte" in n or "clear_output" in n:
        return "legit_exact"
    return "other"


def main():
    runner = BatchedPureNeuralRunner(max_seq_len=2048)
    model = runner.model
    device = next(model.parameters()).device
    dp = model.dim_positions
    OUTPUT_LO_BASE = dp["OUTPUT_LO"]
    OUTPUT_HI_BASE = dp["OUTPUT_HI"]

    # XOR_BASIC: IMM 0xFF; PSH; IMM 0xD5; XOR; EXIT
    bc = []
    bc += encode_instr(Opcode.IMM, 0xFF)
    bc += encode_instr(Opcode.PSH, 0)
    bc += encode_instr(Opcode.IMM, 0xD5)
    bc += encode_instr(Opcode.XOR, 0)
    bc += encode_instr(Opcode.EXIT, 0)
    instrs = []
    for i in range(0, len(bc), INSTR_WIDTH):
        op = bc[i]
        imm = 0
        for j in range(IMMEDIATE_SIZE):
            imm |= bc[i + 1 + j] << (j * 8)
        instrs.append(op | (imm << 8))

    # Run program, capture the model context at each step.
    contexts = []
    original = runner._dispatch_pure_neural
    def patched(s):
        contexts.append(list(s.context))
        return original(s)
    runner._dispatch_pure_neural = patched
    runner.run_batch([instrs])
    runner._dispatch_pure_neural = original

    print(f"# steps captured: {len(contexts)}")
    for i, ctx in enumerate(contexts):
        print(f"  step {i}: ctx len = {len(ctx)}")

    # The IMM step where the f3342968 override fires is step 2 (0-indexed)
    # — second IMM in IMM,PSH,IMM,binop,EXIT.
    # Per /tmp/probe_block_chain.py the probe_pos is 132 (STACK0 marker of
    # step 2). Use the same setup.
    full_ctx = contexts[2]
    probe_pos = 132
    prefix = full_ctx[: probe_pos + 1]
    print(f"# Using step 2 context, probe_pos={probe_pos}, prefix_len={len(prefix)}")

    # Hook block 34 to capture its input residual at probe_pos
    L34_IDX = 34
    block34 = model.blocks[L34_IDX]
    ffn34 = block34.ffn  # PureFFN with 2059 hidden units
    print(f"# L34 ffn type: {type(ffn34).__name__}, hidden_dim={ffn34.hidden_dim}")

    # Capture the input to the FFN module specifically (post-attn residual,
    # post ffn_norm if rms_norm enabled). PureFFN.forward is:
    #     up = F.linear(x, W_up) + b_up
    #     gate = F.linear(x, W_gate) + b_gate
    #     hidden = F.silu(up) * gate
    #     out = x + F.linear(hidden, W_down, b_down)
    # We need `x` as passed into ffn34.forward.
    captured = {}

    def ffn_pre_hook(m, inputs):
        captured["ffn_in"] = inputs[0].detach().clone()

    h = ffn34.register_forward_pre_hook(ffn_pre_hook)

    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    model.embed.set_mem_history_end(0)
    with torch.no_grad():
        _ = model(input_ids)
    h.remove()

    ffn_in = captured["ffn_in"]  # shape [1, seq, d_model]
    print(f"# ffn_in shape: {tuple(ffn_in.shape)}")
    x = ffn_in[0, probe_pos]  # [d_model]
    print(f"# x norm: {float(x.norm()):.4e}")
    print(f"# x[OUTPUT_LO+2]: {float(x[OUTPUT_LO_BASE+2]):.4e}")
    print(f"# x[OUTPUT_LO+0]: {float(x[OUTPUT_LO_BASE+0]):.4e}")
    print(f"# x[OUTPUT_HI+3]: {float(x[OUTPUT_HI_BASE+3]):.4e}")

    # Densify weights if sparse (handles both COO and CSR layouts).
    def _dense(t):
        if t.layout != torch.strided:
            return t.to_dense()
        return t

    W_up = _dense(ffn34.W_up.data).to(device)
    b_up = ffn34.b_up.data.to(device)
    W_gate = _dense(ffn34.W_gate.data).to(device)
    b_gate = ffn34.b_gate.data.to(device)
    W_down = _dense(ffn34.W_down.data).to(device)

    H = W_up.shape[0]
    print(f"# W_up shape: {tuple(W_up.shape)}, W_down shape: {tuple(W_down.shape)}")

    # Per-unit hidden activation at the probe position.
    with torch.no_grad():
        up = W_up @ x + b_up        # [H]
        gate = W_gate @ x + b_gate  # [H]
        hidden = F.silu(up) * gate  # [H]
        # Per-unit contribution at OUTPUT_LO[0..15] and OUTPUT_HI[0..15]
        OUT_DIMS = list(range(OUTPUT_LO_BASE, OUTPUT_LO_BASE + 16)) + \
                   list(range(OUTPUT_HI_BASE, OUTPUT_HI_BASE + 16))
        W_out = W_down[OUT_DIMS, :]  # [32, H]
        per_unit_out = W_out * hidden.unsqueeze(0)  # [32, H]
        # Total OUTPUT writes per rule (signed sum across 32 dims)
        per_unit_sum = per_unit_out.sum(dim=0)  # [H] signed
        per_unit_abs = per_unit_out.abs().sum(dim=0)  # [H] magnitude
        per_unit_max_abs = per_unit_out.abs().max(dim=0).values  # [H]

    # Pull rule names (in unit order).
    rules = _tail_bit32_result_correction_rules()
    assert len(rules) == H, f"len(rules)={len(rules)} != H={H}"

    # Rank by |sum| of OUTPUT contributions.
    sums = per_unit_sum.cpu().tolist()
    abs_sums = per_unit_abs.cpu().tolist()
    max_abs = per_unit_max_abs.cpu().tolist()
    hidden_vals = hidden.cpu().tolist()

    records = []
    for u in range(H):
        records.append({
            "unit": u,
            "name": rules[u].name or f"rule_{u}",
            "hidden": hidden_vals[u],
            "signed_sum": sums[u],
            "abs_sum": abs_sums[u],
            "max_abs": max_abs[u],
            "klass": classify_rule(rules[u].name or f"rule_{u}"),
        })

    # Filter to firing rules (|hidden| > 0 effectively).
    firing = [r for r in records if abs(r["abs_sum"]) > 1e-3]
    print(f"# total rules: {H}, firing (|abs_sum|>1e-3): {len(firing)}")

    # Sort by |abs_sum| desc.
    firing.sort(key=lambda r: -abs(r["abs_sum"]))

    # Aggregates
    total_signed = sum(r["signed_sum"] for r in firing)
    total_abs = sum(r["abs_sum"] for r in firing)
    destructive = sum(r["signed_sum"] for r in firing if r["signed_sum"] < 0)
    constructive = sum(r["signed_sum"] for r in firing if r["signed_sum"] > 0)

    print(f"# Total signed sum across firing rules: {total_signed:+.4e}")
    print(f"# Total |abs| sum: {total_abs:+.4e}")
    print(f"# Destructive (negative) sum: {destructive:+.4e}")
    print(f"# Constructive (positive) sum: {constructive:+.4e}")

    # Per-class breakdown
    from collections import defaultdict
    by_class = defaultdict(lambda: {"count": 0, "signed": 0.0, "abs": 0.0})
    for r in firing:
        c = by_class[r["klass"]]
        c["count"] += 1
        c["signed"] += r["signed_sum"]
        c["abs"] += r["abs_sum"]
    print("\n# Per-class breakdown:")
    for k, v in by_class.items():
        print(f"  {k}: count={v['count']} signed={v['signed']:+.4e} |abs|={v['abs']:+.4e}")

    # Dump top 30
    print("\n# Top-30 firing rules by |abs_sum|:")
    print(f"{'rank':>4} {'unit':>5} {'hidden':>14} {'signed_sum':>14} {'abs_sum':>14} {'klass':<12} name")
    for i, r in enumerate(firing[:30]):
        print(f"{i+1:>4} {r['unit']:>5} {r['hidden']:>+14.4e} {r['signed_sum']:>+14.4e} "
              f"{r['abs_sum']:>+14.4e} {r['klass']:<12} {r['name']}")

    # Save full table + a top-50 markdown table
    out = {
        "x_OUTPUT_LO": [float(x[OUTPUT_LO_BASE + k]) for k in range(16)],
        "x_OUTPUT_HI": [float(x[OUTPUT_HI_BASE + k]) for k in range(16)],
        "total_signed_sum": total_signed,
        "total_abs_sum": total_abs,
        "destructive_sum": destructive,
        "constructive_sum": constructive,
        "by_class": {k: dict(v) for k, v in by_class.items()},
        "firing_count": len(firing),
        "total_rule_count": H,
        "top50": firing[:50],
    }
    with open("/tmp/probe_l34_overfire_table.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n# Wrote /tmp/probe_l34_overfire_table.json")


if __name__ == "__main__":
    main()
