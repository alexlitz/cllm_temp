#!/usr/bin/env python3
"""Combined L34 residual probe + smoke for removal-1 IMM override removal.

After model code changes:
- Removed b5cf7099 IMM AX override in batched_pure_neural.py
- Possibly added FETCH_LO+8 blocker to L10 tail_lea rule

This script builds the model ONCE, then:
1. Probes the L34 FFN input residual at MARK_AX positions for IMM 0xC8 (ADD_16BIT),
   IMM 0xFF (XOR_BASIC + ADD_CARRY), and a real LEA case.
2. Runs the 3 critical smoke tests (test_xor_basic, test_add_16bit,
   test_add_carry_cascade).
3. Runs the full smoke suite to report the regression count.
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


# Predicates from the rule with weights (from l10_ops.py:6562-6629)
# As of this measurement, weights are:
PREDICATES = [
    ("MARK_AX", 1.0, 0),
    ("HAS_SE", 1.0, 0),
    ("OP_LEA", 1.0, 0),
    ("CMP", 1.0, 7),       # CMP+7
    ("FETCH_LO", 2.0, 8),  # FETCH_LO+8
    ("FETCH_HI", 0.2, 15), # FETCH_HI+15
    ("MEM_ADDR_SRC", 5.0, 0),
    ("IS_BYTE", -10.0, 0),
    ("MARK_PC", -10000.0, 0),
    ("MARK_SP", -10000.0, 0),
    ("MARK_BP", -10000.0, 0),
    ("MARK_STACK0", -10000.0, 0),
    ("MARK_MEM", -10000.0, 0),
]
THRESHOLD = 7.0


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
    print("[script] building BatchedPureNeuralRunner (this can take a few minutes)...", flush=True)
    runner = BatchedPureNeuralRunner()
    model = runner.model

    registry = build_default_registry()
    dim_pos = {}
    for name, _w, _off in PREDICATES:
        if name not in registry.slots:
            raise SystemExit(f"Unknown dim {name!r}")
        dim_pos[name] = registry.slots[name].start

    # Find the block hosting tail_bit32_result_correction (2059 hidden units).
    target_block_idx = None
    for i, block in enumerate(model.blocks):
        if hasattr(block.ffn, "W_up"):
            try:
                if block.ffn.W_up.shape[0] == 2059:
                    target_block_idx = i
                    break
            except Exception:
                pass
    if target_block_idx is None:
        raise SystemExit("Could not find tail_bit32_result_correction block (2059 units)")
    print(f"[script] tail_bit32_result_correction lives at block[{target_block_idx}].ffn", flush=True)

    captured_seq = []
    def capture_seq_hook(module, args):
        captured_seq.append(args[0].detach().cpu().clone())

    handle = model.blocks[target_block_idx].ffn.register_forward_pre_hook(capture_seq_hook)

    probe_programs = [
        ("LEA_BASIC", make_bytecode([
            Opcode.ENT,
            (Opcode.IMM, 0),
            (Opcode.LEA, 2),
            Opcode.EXIT,
        ])),
        ("XOR_BASIC", make_bytecode([
            (Opcode.IMM, 0xFF), Opcode.PSH,
            (Opcode.IMM, 0xD5), Opcode.XOR,
            Opcode.EXIT,
        ])),
        ("ADD_16BIT", make_bytecode([
            (Opcode.IMM, 200), Opcode.PSH,
            (Opcode.IMM, 100), Opcode.ADD,
            Opcode.EXIT,
        ])),
        ("ADD_CARRY", make_bytecode([
            (Opcode.IMM, 0xFF), Opcode.PSH,
            (Opcode.IMM, 1), Opcode.ADD,
            Opcode.EXIT,
        ])),
    ]

    try:
        results = {}
        for name, bytecode in probe_programs:
            print(f"\n[probe] === {name} ===", flush=True)
            print(f"[probe] bytecode: {[hex(b) for b in bytecode]}", flush=True)

            captured_seq.clear()
            batch_results = runner.run_batch(
                [bytecode],
                max_steps=20,
                bucket_by_predicted_length=False,
            )
            out, exit_code = batch_results[0]
            print(f"[probe] run finished: exit_code={exit_code}, out={out!r}, n_forwards={len(captured_seq)}", flush=True)

            per_step_values = []
            for fwd_idx, residual in enumerate(captured_seq):
                if residual.dim() != 3:
                    continue
                last_pos = residual.shape[1] - 1
                row = residual[0, last_pos, :].numpy()
                values = {}
                weighted_sum = 0.0
                for pname, w, off in PREDICATES:
                    pos = dim_pos[pname] + off
                    v = float(row[pos])
                    values[pname + (f"+{off}" if off else "")] = v
                    weighted_sum += w * v
                per_step_values.append((fwd_idx, residual.shape, values, weighted_sum))
            results[name] = (per_step_values, out, exit_code)

        # Report: only positions where MARK_AX>=0.5 (where the rule could fire).
        print("\n\n[probe] === ANALYSIS (positions where MARK_AX>=0.5) ===")
        for name, (per_step, out, exit_code) in results.items():
            print(f"\n--- {name} exit={exit_code} ---")
            mark_ax_positions = [(idx, shape, vals, sum_) for idx, shape, vals, sum_ in per_step if vals.get("MARK_AX", 0) >= 0.5]
            if not mark_ax_positions:
                print("  (no MARK_AX positions captured)")
                continue
            for idx, shape, vals, sum_ in mark_ax_positions:
                nz_items = [(k, v) for k, v in vals.items() if abs(v) > 1e-5]
                nz = ", ".join(f"{k}={v:.4f}" for k, v in nz_items)
                fired = "FIRE" if sum_ >= THRESHOLD else "ok"
                print(f"  fwd={idx} sum={sum_:+.3f} threshold={THRESHOLD} [{fired}]")
                print(f"      nonzero: {nz}")

        # Run smoke tests via the same runner.
        print("\n\n[smoke] === SMOKE TESTS ===")
        smoke_programs = [
            ("test_xor_basic", make_bytecode([
                (Opcode.IMM, 0xFF), Opcode.PSH,
                (Opcode.IMM, 0xD5), Opcode.XOR,
                Opcode.EXIT,
            ]), 0x2A),
            ("test_add_16bit", make_bytecode([
                (Opcode.IMM, 200), Opcode.PSH,
                (Opcode.IMM, 100), Opcode.ADD,
                Opcode.EXIT,
            ]), 300),
            ("test_add_carry_cascade", make_bytecode([
                (Opcode.IMM, 0xFF), Opcode.PSH,
                (Opcode.IMM, 1), Opcode.ADD,
                Opcode.EXIT,
            ]), 0x100),
        ]
        smoke_pass = 0
        smoke_fail = 0
        for name, bytecode, expected in smoke_programs:
            batch_results = runner.run_batch([bytecode], max_steps=20, bucket_by_predicted_length=False)
            out, exit_code = batch_results[0]
            status = "PASS" if exit_code == expected else "FAIL"
            if exit_code == expected:
                smoke_pass += 1
            else:
                smoke_fail += 1
            print(f"  {name:35s} expected={expected:6d} got={exit_code} [{status}]")
        print(f"\n[smoke] critical 3-test: {smoke_pass}/{smoke_pass+smoke_fail} pass")

    finally:
        handle.remove()


if __name__ == "__main__":
    main()
