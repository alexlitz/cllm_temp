#!/usr/bin/env python3
"""Verify shr result (clear ON vs OFF) + isolate the block-17 OUTPUT+0 source.

Part 1: emitted result of shr 84>>1 under C4_SHIFT_OUTPUT_B0_CLEAR ON vs OFF,
using the SAME smoke runner path (BatchedPureNeuralRunner.run_batch, spec_k=0).

Part 2 (only meaningful with clear OFF): on the SHR MARK_AX row, dump the FULL
non-zero residual delta between block 16 (input) and block 17 (output), so we
see EXACTLY which dims block 17 writes (attn+ffn+post together) — pinpointing the
OUTPUT_LO+0/HI+0 = 2.0 source and any co-written discriminators.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
# Caller sets C4_SHIFT_OUTPUT_B0_CLEAR (0 or 1) BEFORE running.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.embedding import Opcode  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _mk(items):
    out = []
    for it in items:
        if isinstance(it, tuple):
            out.append(int(it[0])); out.append(int(it[1]))
        else:
            out.append(int(it))
    return out


SHR_PROG = _mk([(Opcode.IMM, 84), Opcode.PSH, (Opcode.IMM, 1), Opcode.SHR, Opcode.EXIT])


def main():
    clear = os.environ.get("C4_SHIFT_OUTPUT_B0_CLEAR", "1")
    print(f"### C4_SHIFT_OUTPUT_B0_CLEAR={clear} "
          f"C4_OUTPUT_B0_NOLEAK={os.environ.get('C4_OUTPUT_B0_NOLEAK','(unset)')}")
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)

    # Part 1: actual emitted result via the smoke runner path.
    res = runner.run_batch([bytes(SHR_PROG)], spec_k=0,
                           expected_steps_list=[5])
    r0 = res[0]
    print(f"  run_batch shr 84>>1: exit_code={getattr(r0,'exit_code',None)} "
          f"stdout={getattr(r0,'stdout',None)!r} -> "
          f"{'PASS(42)' if getattr(r0,'exit_code',None)==42 else 'FAIL'}")

    # Part 2: block16 -> block17 delta on the SHR row.
    _, oracle_tokens = runner._oracle_pc_ax_steps(
        SHR_PROG, b"", "", expected_steps=None, with_tokens=True)
    prompt = runner._build_element(SHR_PROG, b"", [], "", spec_k=1,
                                   adaptive_start_k=0, expected_steps=None)
    prefix = list(prompt.context)
    tape = list(prefix)
    for stp in oracle_tokens:
        tape.extend(stp)
    padded = torch.tensor([tape], dtype=torch.long, device=dev)
    ax_base = dp["MARK_AX"]
    with torch.no_grad():
        emb = model.embed(padded)[0]
        r11 = model.forward(padded, stop_after_block=11)[0]
    shift_row = None
    for stp_idx in range(len(oracle_tokens)):
        ss = len(prefix) + stp_idx * STEP
        for off in range(STEP):
            r = ss + off
            if r < len(tape) and emb[r, ax_base].abs().item() > 0.5 \
               and r11[r, dp["OP_SHR"]].item() > 0.5:
                shift_row = r
                break
        if shift_row is not None:
            break
    print(f"  SHR MARK_AX row = {shift_row}")
    with torch.no_grad():
        r16 = model.forward(padded, stop_after_block=16)[0][shift_row]
        r17 = model.forward(padded, stop_after_block=17)[0][shift_row]
    inv = {int(v): k for k, v in dp.items() if isinstance(v, int)}
    # map dim -> nearest band label + offset
    def label(d):
        best = None
        for name, base in dp.items():
            b = int(base)
            if b <= d < b + 16 and (best is None or b > best[1]):
                best = (name, b)
        if best:
            return f"{best[0]}+{d-best[1]}"
        return f"dim{d}"
    delta = (r17 - r16)
    changed = [(i, float(r16[i]), float(r17[i]))
               for i in range(delta.shape[0]) if abs(float(delta[i])) > 0.2]
    print(f"  block16->17 changed dims ({len(changed)}):")
    for d, a, b in sorted(changed, key=lambda x: -abs(x[2]-x[1]))[:40]:
        print(f"    {label(d):<22} {a:8.2f} -> {b:8.2f}")


if __name__ == "__main__":
    main()
