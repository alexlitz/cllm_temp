#!/usr/bin/env python3
"""Verify the L11 OUTPUT byte-0 no-leak root + ShiftOutputClear inertness.

Run with C4_OUTPUT_B0_NOLEAK=1 (and the campaign flags):
  1. shr 84>>1 -> emitted result (should be 42 with the root ON).
  2. OUTPUT band on the SHR MARK_AX row at the shift-composite block is empty.
  3. INERTNESS: find the ShiftOutputClearFFN wrapper in the model, feed it the
     actual block-input residual for the SHR row, and assert
     wrap.forward(x) == wrap.inner(x) (max-abs-diff) -> the wrap is now a no-op.

Requires the wrapper to still be installed (i.e. run this BEFORE deleting it,
with BOTH C4_SHIFT_OUTPUT_B0_CLEAR=1 and C4_OUTPUT_B0_NOLEAK=1) to prove the
new root subsumes the old corrector.
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
# Caller sets C4_OUTPUT_B0_NOLEAK and C4_SHIFT_OUTPUT_B0_CLEAR.
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


SHR = _mk([(Opcode.IMM, 84), Opcode.PSH, (Opcode.IMM, 1), Opcode.SHR, Opcode.EXIT])
SHL = _mk([(Opcode.IMM, 21), Opcode.PSH, (Opcode.IMM, 1), Opcode.SHL, Opcode.EXIT])


def fmt(row, base):
    return "[" + ", ".join(f"{row[base+i].item():.2f}@{i}" for i in range(16)
                           if abs(row[base+i].item()) > 0.3) + "]"


def find_shift_clear_block(model):
    for phys, blk in enumerate(model.blocks):
        ffn = getattr(blk, "ffn", None)
        if ffn is not None and getattr(ffn, "_is_shift_output_clear_wrap", False):
            return phys, ffn
    return None, None


def main():
    print(f"### C4_OUTPUT_B0_NOLEAK={os.environ.get('C4_OUTPUT_B0_NOLEAK')} "
          f"C4_SHIFT_OUTPUT_B0_CLEAR={os.environ.get('C4_SHIFT_OUTPUT_B0_CLEAR')}")
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)

    # (1) emitted results
    for label, prog in (("shr 84>>1", SHR), ("shl 21<<1", SHL)):
        res = runner.run_batch([bytes(prog)], spec_k=0, expected_steps_list=[5])
        r0 = res[0]
        ec = getattr(r0, "exit_code", None)
        print(f"  {label}: exit_code={ec} -> {'PASS(42)' if ec == 42 else 'FAIL'}")

    # locate the ShiftOutputClear wrapper block
    clear_phys, clear_ffn = find_shift_clear_block(model)
    print(f"  ShiftOutputClear wrapper at physical block: {clear_phys}")

    # Build the SHR tape + find shift MARK_AX row.
    _, otoks = runner._oracle_pc_ax_steps(SHR, b"", "", expected_steps=None, with_tokens=True)
    prompt = runner._build_element(SHR, b"", [], "", spec_k=1, adaptive_start_k=0, expected_steps=None)
    prefix = list(prompt.context)
    tape = list(prefix)
    for stp in otoks:
        tape.extend(stp)
    padded = torch.tensor([tape], dtype=torch.long, device=dev)
    ax = dp["MARK_AX"]
    with torch.no_grad():
        emb = model.embed(padded)[0]
        r11 = model.forward(padded, stop_after_block=11)[0]
    row = None
    for si in range(len(otoks)):
        ss = len(prefix) + si * STEP
        for off in range(STEP):
            rr = ss + off
            if rr < len(tape) and emb[rr, ax].abs().item() > 0.5 and r11[rr, dp["OP_SHR"]].item() > 0.5:
                row = rr; break
        if row is not None:
            break
    print(f"  SHR MARK_AX row = {row}")

    # (2) OUTPUT band at L11 (block 17) and at the ShiftOutputClear block input.
    with torch.no_grad():
        r17 = model.forward(padded, stop_after_block=17)[0][row]
    print(f"  @blk17 (L11) OUTPUT_LO={fmt(r17, dp['OUTPUT_LO'])} "
          f"OUTPUT_HI={fmt(r17, dp['OUTPUT_HI'])}  (should be empty)")

    # (3) INERTNESS: feed the ShiftOutputClear block's INPUT residual to the
    # wrapper and compare forward vs inner on ALL rows of the sequence.
    if clear_ffn is not None:
        with torch.no_grad():
            x_in = model.forward(padded, stop_after_block=clear_phys - 1)  # [1,S,D]
            fwd = clear_ffn.forward(x_in)
            inner = clear_ffn.inner(x_in)
            diff = (fwd - inner).abs().max().item()
            # also isolate the shr row
            row_diff = (fwd[0, row] - inner[0, row]).abs().max().item()
        print(f"  INERTNESS: ShiftOutputClear.forward vs .inner  "
              f"max-abs-diff (all rows) = {diff:.6g};  (shr row) = {row_diff:.6g}")
        print(f"  -> {'INERT (deletable)' if diff < 1e-6 else 'STILL ACTIVE'}")
    else:
        print("  ShiftOutputClear wrapper NOT found (already deleted?)")


if __name__ == "__main__":
    main()
