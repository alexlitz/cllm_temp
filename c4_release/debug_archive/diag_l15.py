#!/usr/bin/env python3
"""Focused diagnostic for L15 ZFOD leakage at AX marker during LI step."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from neural_vm.vm_step import AutoregressiveVM, Token, _SetDim, set_vm_weights
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

torch.set_printoptions(precision=4, linewidth=200)

BD = _SetDim

# Use runner to execute program properly
runner = AutoregressiveVMRunner()
model = runner.model
set_vm_weights(model)

bytecode = [Opcode.IMM | (42 << 8), Opcode.PSH, Opcode.LI, Opcode.EXIT]
context = runner._build_context(bytecode, b'', [])
context_len_before = len(context)

# Generate tokens step by step, tracking step boundaries
step_num = 0
step_tokens = []
all_step_starts = []  # position in context where each step starts

for i in range(4 * Token.STEP_TOKENS + 10):
    if step_tokens == []:
        all_step_starts.append(len(context))

    tok = model.generate_next(context)
    context.append(tok)
    step_tokens.append(tok)
    if i < 80:
        print(f"  gen[{i}]={tok}", end=" ")
        if (i+1) % 10 == 0:
            print()

    if tok in (Token.STEP_END, Token.HALT):
        # Extract PC for this step
        pc = runner._extract_register(context, Token.REG_PC)
        instr_idx = pc // 5 if pc is not None else None
        op = bytecode[instr_idx] & 0xFF if instr_idx is not None and 0 <= instr_idx < len(bytecode) else None
        op_name = {v: k for k, v in vars(Opcode).items() if isinstance(v, int)}.get(op, '?')

        ax = runner._extract_register(context, Token.REG_AX)
        sp = runner._extract_register(context, Token.REG_SP)
        sp_str = f"{sp:#010x}" if sp else "0"
        print(f"Step {step_num} ({op_name}): AX={ax} SP={sp_str}")

        # Run syscall handler
        if op is not None:
            handler = runner._syscall_handlers.get(op)
            if handler:
                handler(context, [])
            # SP corrections
            if not runner.pure_attention_memory:
                if op == Opcode.PSH:
                    new_sp = (runner._last_sp - 8) & 0xFFFFFFFF
                    runner._override_register_in_last_step(context, Token.REG_SP, new_sp)
                    sp = new_sp
                    print(f"  Runner corrected SP to {sp:#010x}")
            # Track memory
            runner._track_memory_write(context, op)

        # Update SP/BP/PC tracking
        sp_now = runner._extract_register(context, Token.REG_SP)
        runner._last_sp = sp_now if sp_now is not None else 0
        bp_now = runner._extract_register(context, Token.REG_BP)
        runner._last_bp = bp_now if bp_now is not None else 0

        step_num += 1
        step_tokens = []

        if tok == Token.HALT:
            break

# Now shadow memory state
print(f"\nRunner shadow memory: {dict(runner._memory)}")

# Now we need to trace L15 at the LI step (step 2).
# The LI step is step 2. AX marker position = step_start + 5.
# But we already generated all tokens. We need to re-run the forward pass
# at the AX marker position of step 2 to see what L15 does.

li_step_start = all_step_starts[2]
ax_marker_pos = li_step_start + 5

print(f"\nLI step starts at context pos {li_step_start}")
print(f"AX marker at pos {ax_marker_pos}")
print(f"Token at AX marker pos: {context[ax_marker_pos]} (should be {Token.REG_AX})")
assert context[ax_marker_pos] == Token.REG_AX

# Build context up to and including AX marker
ctx_slice = context[:ax_marker_pos + 1]
x_tensor = torch.tensor([ctx_slice], dtype=torch.long)

# Hook L15 attention to capture pre-L15 residual
residuals = {}

def make_pre_hook(name):
    def hook(module, input):
        residuals[name] = input[0].detach().clone()
    return hook

def make_post_hook(name):
    def hook(module, input, output):
        residuals[name] = output.detach().clone()
    return hook

# Register hooks
h1 = model.blocks[15].attn.register_forward_pre_hook(make_pre_hook('pre_l15_attn'))
h2 = model.blocks[15].attn.register_forward_hook(make_post_hook('post_l15_attn'))

with torch.no_grad():
    _ = model(x_tensor)  # hooks fire during forward pass

h1.remove()
h2.remove()

pre_l15 = residuals['pre_l15_attn']
post_l15 = residuals['post_l15_attn']

# L15 attention contribution
l15_delta = post_l15[0, -1] - pre_l15[0, -1]  # at last position (AX marker)

print(f"\n=== L15 attention delta at AX marker ===")
for k in range(16):
    v = l15_delta[BD.OUTPUT_LO + k].item()
    if abs(v) > 0.001:
        print(f"  OUTPUT_LO[{k}] = {v:.6f}")
for k in range(16):
    v = l15_delta[BD.OUTPUT_HI + k].item()
    if abs(v) > 0.001:
        print(f"  OUTPUT_HI[{k}] = {v:.6f}")

# Now compute Q, K, scores manually for L15 head 0
ax_res = pre_l15[0, -1]  # residual at AX marker before L15

print(f"\n=== Key dims at AX marker (pre-L15) ===")
print(f"MARK_AX:      {ax_res[BD.MARK_AX]:.4f}")
print(f"OP_LI_RELAY:  {ax_res[BD.OP_LI_RELAY]:.4f}")
print(f"OP_LC_RELAY:  {ax_res[BD.OP_LC_RELAY]:.4f}")
print(f"CONST:        {ax_res[BD.CONST]:.4f}")
print(f"CMP[0] (PSH): {ax_res[BD.CMP]:.4f}")
print(f"MEM_STORE:    {ax_res[BD.MEM_STORE]:.4f}")
print(f"IS_BYTE:      {ax_res[BD.IS_BYTE]:.4f}")
print(f"MARK_STACK0:  {ax_res[BD.MARK_STACK0]:.4f}")

# ADDR dims at AX marker
print(f"\nADDR at AX marker (gathered prev AX bytes):")
for name, base, count in [("B0_LO", BD.ADDR_B0_LO, 16), ("B0_HI", BD.ADDR_B0_HI, 16),
                           ("B1_LO", BD.ADDR_B1_LO, 16), ("B1_HI", BD.ADDR_B1_HI, 16),
                           ("B2_LO", BD.ADDR_B2_LO, 16), ("B2_HI", BD.ADDR_B2_HI, 16)]:
    vals = [ax_res[base + k].item() for k in range(count)]
    nz = [(k, v) for k, v in enumerate(vals) if abs(v) > 0.01]
    if nz:
        print(f"  {name}: {nz}")

# Compute Q/K/scores
l15_attn = model.blocks[15].attn
HD = 64
W_q = l15_attn.W_q.data
W_k = l15_attn.W_k.data
W_v = l15_attn.W_v.data
W_o = l15_attn.W_o.data

seq_len = pre_l15.shape[1]

# Full Q and K
q_full = pre_l15[0] @ W_q.T  # [seq, dim]
k_full = pre_l15[0] @ W_k.T

# Head 0
q_h0_ax = q_full[-1, :HD]  # Q at AX marker
k_h0 = k_full[:, :HD]  # K for all positions

print(f"\n=== L15 Head 0 Q at AX marker ===")
print(f"Q[0] (bias):   {q_h0_ax[0]:.2f}")
print(f"Q[1] (store):  {q_h0_ax[1]:.2f}")
print(f"Q[2] (ZFOD):   {q_h0_ax[2]:.2f}")
print(f"Q[3] (byte):   {q_h0_ax[3]:.2f}")
print(f"Q[28] (gate):  {q_h0_ax[28]:.2f}")
addr_dims = [d for d in range(4, 28) if abs(q_h0_ax[d]) > 1]
print(f"Non-zero addr dims: {len(addr_dims)} (expected 24)")

# Compute raw scores
scores_raw = (q_h0_ax.unsqueeze(0) * k_h0[:-1]).sum(dim=-1) / (HD ** 0.5)

# ALiBi
alibi_slope = l15_attn.alibi_slopes[0].item()
dists = torch.arange(seq_len - 2, -1, -1, dtype=torch.float32)
alibi = -alibi_slope * dists
scores = scores_raw + alibi

# Find MEM marker positions
mem_marker_positions = [i for i in range(seq_len - 1) if ctx_slice[i] == Token.MEM]
print(f"\nMEM markers at positions: {mem_marker_positions}")

# Show scores at MEM section positions
for mp in mem_marker_positions:
    print(f"\n--- MEM at pos {mp} ---")
    for offset in range(min(9, seq_len - 1 - mp)):
        pos = mp + offset
        tok = ctx_slice[pos]
        s = scores[pos].item()
        raw = scores_raw[pos].item()

        # Break down score
        k_vec = k_h0[pos]
        dim_scores = {}
        dim_scores['bias'] = (q_h0_ax[0] * k_vec[0]).item() / (HD**0.5)
        dim_scores['store'] = (q_h0_ax[1] * k_vec[1]).item() / (HD**0.5)
        dim_scores['zfod'] = (q_h0_ax[2] * k_vec[2]).item() / (HD**0.5)
        dim_scores['byte'] = (q_h0_ax[3] * k_vec[3]).item() / (HD**0.5)
        dim_scores['gate'] = (q_h0_ax[28] * k_vec[28]).item() / (HD**0.5)
        dim_scores['addr'] = sum((q_h0_ax[d] * k_vec[d]).item() for d in range(4, 28)) / (HD**0.5)

        # Key dims
        ms = pre_l15[0, pos, BD.MEM_STORE].item()

        print(f"  d={offset} pos={pos} tok={tok:3d} | total={s:+8.1f} "
              f"(bias={dim_scores['bias']:+7.1f} store={dim_scores['store']:+7.1f} "
              f"zfod={dim_scores['zfod']:+7.1f} byte={dim_scores['byte']:+7.1f} "
              f"addr={dim_scores['addr']:+7.1f} gate={dim_scores['gate']:+7.1f}) "
              f"MEM_STORE={ms:.3f}")

# Top scores
print(f"\n=== Top 10 scores ===")
top_vals, top_idx = scores.topk(min(10, len(scores)))
for s, idx in zip(top_vals, top_idx):
    tok = ctx_slice[idx.item()]
    ms = pre_l15[0, idx.item(), BD.MEM_STORE].item()
    print(f"  pos={idx.item()} tok={tok:3d} score={s.item():+.2f} MEM_STORE={ms:.3f}")

# softmax1 computation
print(f"\nMax score: {scores.max():.4f}")
exp_scores = torch.exp(scores)
denom = 1.0 + exp_scores.sum()
weights = exp_scores / denom
total_w = weights.sum().item()
print(f"Total attention weight (1-p_null): {total_w:.8f}")

if total_w > 0.001:
    # Show what value is being read
    v_full = pre_l15[0, :-1] @ W_v.T  # [seq-1, dim]
    v_h0 = v_full[:, :HD]
    weighted_v = (weights.unsqueeze(1) * v_h0).sum(dim=0)
    out_head = W_o[:, :HD] @ weighted_v
    print(f"\nL15 head 0 weighted output:")
    for k in range(16):
        v = out_head[BD.OUTPUT_LO + k].item()
        if abs(v) > 0.001:
            print(f"  OUTPUT_LO[{k}] = {v:.6f}")
    for k in range(16):
        v = out_head[BD.OUTPUT_HI + k].item()
        if abs(v) > 0.001:
            print(f"  OUTPUT_HI[{k}] = {v:.6f}")

    # Also show top contributing positions
    top_w, top_wi = weights.topk(min(5, len(weights)))
    print(f"\nTop 5 attention weights:")
    for w, idx in zip(top_w, top_wi):
        tok = ctx_slice[idx.item()]
        print(f"  pos={idx.item()} tok={tok:3d} w={w.item():.8f}")

# Also check: OUTPUT at AX marker BEFORE L15
print(f"\n=== OUTPUT at AX marker before L15 ===")
for k in range(16):
    v = pre_l15[0, -1, BD.OUTPUT_LO + k].item()
    if abs(v) > 0.01:
        print(f"OUTPUT_LO[{k}] = {v:.4f}")
for k in range(16):
    v = pre_l15[0, -1, BD.OUTPUT_HI + k].item()
    if abs(v) > 0.01:
        print(f"OUTPUT_HI[{k}] = {v:.4f}")

# AX_CARRY at AX marker
print(f"\nAX_CARRY at AX marker before L15:")
for k in range(16):
    v = pre_l15[0, -1, BD.AX_CARRY_LO + k].item()
    if abs(v) > 0.01:
        print(f"AX_CARRY_LO[{k}] = {v:.4f}")
for k in range(16):
    v = pre_l15[0, -1, BD.AX_CARRY_HI + k].item()
    if abs(v) > 0.01:
        print(f"AX_CARRY_HI[{k}] = {v:.4f}")
