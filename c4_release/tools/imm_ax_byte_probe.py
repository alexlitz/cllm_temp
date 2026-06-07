#!/usr/bin/env python3
"""Probe AX_CARRY/FETCH/OUTPUT at AX byte rows (NOT MARK_AX) for IMM 0xFF.

The b5cf7099 override patches REG_AX bytes 1,2,3 in last step. If MARK_AX
byte 0 emit is fine (per docs/IMM_OVERRIDE_REAL_SURFACE_2026_06_07.md),
then the bug is downstream at AX byte 1/2/3 rows.

This probe hooks every block to capture the per-position residual values
for FETCH_LO, FETCH_HI, AX_CARRY_LO, AX_CARRY_HI, OUTPUT_LO at:
  - MARK_AX row
  - AX byte 1/2/3 rows (IS_BYTE + H1+AX_I + BYTE_INDEX_{1,2,3})

For IMM 0xFF, expected:
  - All four bytes should have OUTPUT carrying 0xFF (low nibble=F, hi nibble=F)
  - AX_CARRY_LO+15 = high, AX_CARRY_HI+15 = high (the F nibble bit)

If AX byte 1/2/3 rows have wrong AX_CARRY, the head 3 multibyte_fetch
is mis-routing for high-bit IMMs.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from neural_vm.embedding import Opcode  # noqa: E402
from neural_vm.batched_pure_neural import BatchedPureNeuralRunner  # noqa: E402
from neural_vm.dim_registry import build_default_registry  # noqa: E402


def make_bytecode(ops):
    bytecode = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bytecode.append(opcode | (imm << 8))
        else:
            bytecode.append(op)
    return bytecode


def main():
    # Optionally patch out override temporarily by setting env flag we read in batched_pure_neural
    disable_override = os.environ.get("DISABLE_IMM_OVERRIDE", "0") == "1"

    print(f"[script] DISABLE_IMM_OVERRIDE={disable_override}", flush=True)
    print(f"[script] building runner...", flush=True)

    if disable_override:
        # Monkey-patch: replace the override block with a noop
        from neural_vm import batched_pure_neural as bpn
        orig_apply = bpn.BatchedPureNeuralRunner._apply_runner_overrides if hasattr(bpn.BatchedPureNeuralRunner, '_apply_runner_overrides') else None
        # simpler: monkey patch _override_register_in_last_step to noop for REG_AX *only when called from IMM*
        # Actually, easier: don't patch; just run with override on first, then off.
        pass

    runner = BatchedPureNeuralRunner()
    model = runner.model

    if disable_override:
        # Patch out the IMM override by replacing the offending lines via a runtime flag
        # We'll modify the runner method directly:
        import types
        orig_finalize = runner._after_step if hasattr(runner, '_after_step') else None
        # Instead: monkey-patch _override_register_in_last_step to ignore IMM-context calls
        # Easier: just modify s.last_ax computation. We won't fully suppress; the probe needs the data flow as-is.
        # SIMPLER: leave the override on. We'll probe residuals which the override doesn't affect.
        pass

    registry = build_default_registry()

    # Capture dims of interest
    dim_names = [
        "MARK_AX", "MARK_PC",
        "IS_BYTE",
        "H1+1",  # AX_I=1
        "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
        "OP_IMM",
    ]
    for k in range(16):
        dim_names.extend([
            f"FETCH_LO+{k}", f"FETCH_HI+{k}",
            f"AX_CARRY_LO+{k}", f"AX_CARRY_HI+{k}",
            f"OUTPUT_LO+{k}",
        ])

    def dim_pos(name):
        if "+" in name:
            base, offset = name.split("+")
            return registry.slots[base].start + int(offset)
        return registry.slots[name].start

    positions = {n: dim_pos(n) for n in dim_names}

    captured = []  # list of (block_idx, residual_tensor) per forward call

    def make_hook(block_idx):
        def hook(module, args):
            # args[0] is the residual input to this block
            t = args[0].detach().cpu().clone()
            captured.append((block_idx, t))
        return hook

    # Register hook on every block FFN (post-attn, pre-FFN — the residual after attention)
    handles = []
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'ffn'):
            handles.append(block.ffn.register_forward_pre_hook(make_hook(i)))

    # Probe: IMM 0xFF
    bytecode = make_bytecode([
        (Opcode.IMM, 0xFF), Opcode.PSH,
        (Opcode.IMM, 0xD5), Opcode.XOR,
        Opcode.EXIT,
    ])

    captured.clear()
    print(f"[probe] running IMM 0xFF / PSH / IMM 0xD5 / XOR / EXIT", flush=True)
    results = runner.run_batch([bytecode], max_steps=20, bucket_by_predicted_length=False)
    out, exit_code = results[0]
    print(f"[probe] exit_code={hex(exit_code) if isinstance(exit_code, int) else exit_code}", flush=True)
    print(f"[probe] expected 0x2A (= 0xFF XOR 0xD5)", flush=True)
    print(f"[probe] total hook calls: {len(captured)}", flush=True)

    # Find first forward (step 0)
    # Group by forward index (each forward populates len(model.blocks) entries)
    n_blocks = len(model.blocks)
    forward_groups = []
    cur = []
    last_block_idx = -1
    for block_idx, t in captured:
        if block_idx <= last_block_idx and cur:
            forward_groups.append(cur)
            cur = []
        cur.append((block_idx, t))
        last_block_idx = block_idx
    if cur:
        forward_groups.append(cur)
    print(f"[probe] forward groups: {len(forward_groups)}", flush=True)

    # List all forwards with MARK_AX at the last position (the emit-driving position).
    print(f"[probe] === MARK_AX at last position by forward ===", flush=True)
    mark_ax_fwds = []
    for fi, fg in enumerate(forward_groups):
        _, last_t = fg[-1]
        sl = last_t.shape[1]
        if sl == 0:
            continue
        row = last_t[0, sl-1, :].numpy()
        if float(row[positions["MARK_AX"]]) > 0.5:
            mark_ax_fwds.append(fi)
    print(f"[probe] MARK_AX at last-pos in forwards: {mark_ax_fwds}", flush=True)
    candidate_fwd_idx = mark_ax_fwds[0] if mark_ax_fwds else None
    print(f"[probe] first forward containing MARK_AX: fwd_idx={candidate_fwd_idx}", flush=True)

    # Process all MARK_AX forwards
    print(f"[probe] === Processing each MARK_AX forward ===", flush=True)
    for tf in mark_ax_fwds[:6]:
        print(f"\n[probe] >>> forward_idx={tf} <<<", flush=True)
        fg = forward_groups[tf]
        _, t0 = fg[0]
        seq_len = t0.shape[1]
        print(f"[probe] seq_len={seq_len}", flush=True)
        # Just dump MARK_AX row across layers
        pos = seq_len - 1
        for layer in [5, 6, 7, 8, 9, 10, len(model.blocks)-1]:
            entry = None
            for b_idx, t in fg:
                if b_idx == layer:
                    entry = (b_idx, t)
                    break
            if entry is None:
                continue
            _, t = entry
            row = t[0, pos, :].numpy()
            vals = {}
            for n in dim_names:
                v = float(row[positions[n]])
                if abs(v) > 1e-2:
                    vals[n] = round(v, 3)
            print(f"  L{layer}: {vals}", flush=True)
    return

    # For each position, check if it's MARK_AX or AX byte k
    # Use the residual AFTER the last block as the discriminator
    _, last_t = fwd0[-1]

    def classify(pos):
        row = last_t[0, pos, :].numpy()
        mark_ax = float(row[positions["MARK_AX"]])
        is_byte = float(row[positions["IS_BYTE"]])
        h1_ax = float(row[positions["H1+1"]])
        bi = [float(row[positions[f"BYTE_INDEX_{i}"]]) for i in range(4)]
        if mark_ax > 0.5:
            return "MARK_AX"
        if is_byte > 0.5 and h1_ax > 0.5:
            for i, b in enumerate(bi):
                if b > 0.5:
                    return f"AX_BYTE_{i}"
            return "AX_BYTE_?"
        return None

    # Show MARK_AX and AX byte rows
    interesting = []
    for pos in range(seq_len):
        c = classify(pos)
        if c:
            interesting.append((pos, c))
    print(f"[probe] interesting positions: {interesting}", flush=True)
    # Scan all forwards for AX_BYTE_1/2/3 markers
    print(f"[probe] === scanning all forwards for AX_BYTE_1/2/3 ===", flush=True)
    for fi, fg in enumerate(forward_groups):
        _, last_t = fg[-1]
        sl = last_t.shape[1]
        for pos in range(max(0, sl-10), sl):
            row = last_t[0, pos, :].numpy()
            ib = float(row[positions["IS_BYTE"]])
            hax = float(row[positions["H1+1"]])
            if ib > 0.5 and hax > 0.5:
                bi = [float(row[positions[f"BYTE_INDEX_{i}"]]) for i in range(4)]
                hi_idx = -1
                for i, b in enumerate(bi):
                    if b > 0.5:
                        hi_idx = i
                        break
                if hi_idx >= 1:
                    print(f"  fwd={fi} pos={pos} AX_BYTE_{hi_idx}", flush=True)
    # If no AX_BYTE_1..3 found, also dump every position with IS_BYTE + H1+1 in the last 35 positions
    if not any("AX_BYTE_" in lbl for _, lbl in interesting):
        print(f"[probe] no AX_BYTE_k rows found in target_fwd; scanning IS_BYTE+H1+1 rows...", flush=True)
        _, last_t = fwd0[-1]
        for pos in range(max(0, seq_len-35), seq_len):
            row = last_t[0, pos, :].numpy()
            ib = float(row[positions["IS_BYTE"]])
            hax = float(row[positions["H1+1"]])
            if ib > 0.5 and hax > 0.5:
                bi = [float(row[positions[f"BYTE_INDEX_{i}"]]) for i in range(4)]
                print(f"  pos={pos}: IS_BYTE={ib} H1+1={hax} BYTE_INDEX={bi}", flush=True)

    # For each interesting position, dump residual values across layers
    for pos, label in interesting:
        print(f"\n[probe] === pos={pos} ({label}) ===", flush=True)
        # Sample at L5, L6, L7, L8, L9, L10, L14, last block
        sample_layers = [5, 6, 7, 8, 9, 10, 14, len(model.blocks)-1]
        for layer in sample_layers:
            # Find the entry where block_idx == layer
            entry = None
            for b_idx, t in fwd0:
                if b_idx == layer:
                    entry = (b_idx, t)
                    break
            if entry is None:
                continue
            _, t = entry
            row = t[0, pos, :].numpy()
            vals = {}
            for n in dim_names:
                v = float(row[positions[n]])
                if abs(v) > 1e-3:
                    vals[n] = round(v, 3)
            print(f"  L{layer}: {vals}", flush=True)

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
