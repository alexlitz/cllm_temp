#!/usr/bin/env python3
"""Probe the L10 FFN input residual for tail_lea_local_ax_marker_byte0_e8 rule.

Captures the residual at each step's emit position for two programs:
1. test_lea_basic:  ENT, IMM 0, LEA 2, EXIT
2. test_xor_basic:  IMM 0xFF, PSH, IMM 0xD5, XOR, EXIT (spurious IMM 0xFF fire path)

For each candidate emit position, reads the dim values for the rule's predicates
and computes the weighted sum (vs threshold 14).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from neural_vm.embedding import Opcode  # noqa: E402
from neural_vm.batched_pure_neural import BatchedPureNeuralRunner  # noqa: E402
from neural_vm.dim_registry import build_default_registry  # noqa: E402
from neural_vm.vm_step import Token  # noqa: E402


# Predicates from the rule with weights (from l10_ops.py:6477-6502)
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
    print("[probe] building BatchedPureNeuralRunner (this can take a few minutes)...", flush=True)
    runner = BatchedPureNeuralRunner()
    model = runner.model

    # Resolve dim positions. Use build_default_registry — these are
    # the historical pin positions that the layout's pin= arguments
    # preserve byte-identically.
    registry = build_default_registry()
    dim_pos = {}
    for name, _w, _off in PREDICATES:
        if name not in registry.slots:
            raise SystemExit(f"Unknown dim {name!r}")
        dim_pos[name] = registry.slots[name].start
    for name, pos in sorted(dim_pos.items(), key=lambda x: x[1]):
        print(f"[probe] dim {name:14s} -> position {pos}")

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
    print(f"[probe] tail_bit32_result_correction lives at block[{target_block_idx}].ffn", flush=True)

    captured_seq = []
    def capture_seq_hook(module, args):
        captured_seq.append(args[0].detach().cpu().clone())

    handle = model.blocks[target_block_idx].ffn.register_forward_pre_hook(capture_seq_hook)

    try:
        results = {}
        for name, bytecode in [
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
        ]:
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

            # For each forward, examine EVERY position (not just the last),
            # since the rule fires at the AX-marker position of each step.
            # Take the LAST captured forward, which has the full S, and walk
            # every position in it. Compute the rule's contribution at each
            # position. Also report token at that position (we need the context
            # but we have the residual; for context we re-derive from runner).
            #
            # Actually, the simpler approach: walk per-forward and capture the
            # LAST position's predicate values. As the run progresses, each
            # forward emits one new token, so the LAST position of each forward
            # is the residual at one specific token in the trace.
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
            results[name] = per_step_values

        # Analysis: focus on positions where MARK_AX > 0.5 (those are the AX marker
        # emit positions, where the rule's `dominates_at == mark == AX` is true).
        print("\n\n[probe] === ANALYSIS (positions where MARK_AX>=0.5) ===")
        for name, per_step in results.items():
            print(f"\n--- {name} ---")
            for idx, shape, vals, sum_ in per_step:
                if vals.get("MARK_AX", 0) >= 0.5:
                    nz_items = [(k, v) for k, v in vals.items() if abs(v) > 1e-5]
                    nz = ", ".join(f"{k}={v:.4f}" for k, v in nz_items)
                    print(f"  fwd={idx} S={shape[1]} sum={sum_:.3f}")
                    print(f"      all_nonzero: {nz}")
                    # Detailed by predicate, showing residual value × weight contribution
                    print(f"      contributions:")
                    for pname, w, off in PREDICATES:
                        v = vals[pname + (f"+{off}" if off else "")]
                        if abs(v) > 1e-5:
                            print(f"          {pname}{'+'+str(off) if off else '':<3s} v={v:+.4f} × w={w:+.1f} = {v*w:+.3f}")

        # Top-sum analysis: which positions across the run have the highest
        # weighted sum (= where the rule is closest to firing)?
        print("\n\n[probe] === TOP-15 POSITIONS BY WEIGHTED SUM (any position) ===")
        for name, per_step in results.items():
            print(f"\n--- {name} ---")
            top = sorted(per_step, key=lambda x: -x[3])[:15]
            for idx, shape, vals, sum_ in top:
                nz_items = [(k, v) for k, v in vals.items() if abs(v) > 1e-5]
                nz = ", ".join(f"{k}={v:.3f}" for k, v in nz_items)
                print(f"  fwd={idx} S={shape[1]} sum={sum_:+.3f}  nz: {nz}")

        # Also dump ALL positions on the last forward (gives full picture)
        print("\n\n[probe] === FULL TRACE (all positions of last forward, only MARK_AX positions) ===")
        for name, per_step in results.items():
            if not per_step:
                continue
            print(f"\n--- {name} (last forward) ---")
            # Re-capture: we have residual stored in captured_seq, but cleared.
            # Skip this for brevity; per_step already has the per-fwd data.

    finally:
        handle.remove()


if __name__ == "__main__":
    main()
