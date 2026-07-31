#!/usr/bin/env python3
"""ROOT-CAUSE probe: why does the CFM recurrent-divmod LEAN forward get DIV/MUL wrong?

Runs `12 DIV 7` (+ MUL/MOD) through:
  (A) HF Qwen2Model.forward driver  (qwen_full_vm.run_program)  -- brief says byte-exact
  (B) LEAN forward driver           (qwen_lean_forward.run_program_lean) -- brief says WRONG

For the DIV step it dumps BOTH the scalar AX_VAL and the nibble-decoded AX from the
SAME lean hidden state, to distinguish two hypotheses:
  H1 (recurrence): the lean forward does not replay _apply_order -> divmod runs 1 iter
     -> the NIBBLE band itself is wrong.
  H2 (decode)    : the lean forward computes the right NIBBLES but the lean DRIVER reads
     the scalar AX_VAL (which is stale for MUL/DIV/MOD) instead of the nibble band.

If the nibble-decoded AX at the lean DIV step == correct, root == H2 (decode), and the
fix is a one-line driver change (decode MUL/DIV/MOD AX from nibbles, exactly as
run_program does).  If the nibbles are ALSO wrong, root == H1 (recurrence).
"""
from __future__ import annotations

import argparse
import warnings

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from c4_min import isa
    from c4_min import qwen_full_vm as Q
    from c4_min import qwen_lean_forward as LF
    from c4_min.nibble_pure_forward_complete import _decode_reg_from_nibbles
    from c4_min.qwen_full_vm import _snap
    from _agent_lean_recurrent import build_lean_recurrent

    dev = torch.device(args.device)
    subset = Q.SUBSET_MULDIV

    print("[build] CFM recurrent-divmod muldiv model (code_from_memory=True) ...", flush=True)
    vm = Q.build(code_size=24, subset=subset, recurrent_divmod=True, code_from_memory=True)
    L = vm.QL.L
    print(f"[build] stored={vm.n_layers}L applied={vm.n_applied} hidden={vm.hidden_size} "
          f"cfm={vm.code_from_memory}", flush=True)

    vm.embed = vm.embed.to(dev)
    vm.qmodel = vm.qmodel.to(dev)
    if dev.type == "cuda":
        print(f"[vram] after HF on device: {torch.cuda.memory_allocated(dev)/1e9:.2f} GB", flush=True)

    battery = [
        ("mul_12x7",  [("IMM", 12), ("PSH", 0), ("IMM", 7), ("MUL", 0), ("HALT", 0)], 84),
        ("div_100_7", [("IMM", 100), ("PSH", 0), ("IMM", 7), ("DIV", 0), ("HALT", 0)], 14),
        ("mod_100_7", [("IMM", 100), ("PSH", 0), ("IMM", 7), ("MOD", 0), ("HALT", 0)], 2),
    ]

    print("\n=== (A) HF Qwen2Model.forward driver (run_program) ===", flush=True)
    for name, prog, want in battery:
        code = isa.assemble(prog)
        r = Q.run_program(vm, code, max_steps=64)
        print(f"  {name:10s} exact={r['exact']} ax_trace={r['ax_trace']} ref={r['ref_trace']}", flush=True)

    print("\n[build] extracting distinct layers into RecurrentLeanQwenVM ...", flush=True)
    lean = build_lean_recurrent(vm, device=str(dev))
    print(f"[build] lean: n_layers(applied)={lean.n_layers} apply_order_len={len(lean.apply_order)}", flush=True)

    print("\n=== (B) LEAN forward driver (run_program_lean, scalar AX_VAL decode) ===", flush=True)
    for name, prog, want in battery:
        code = isa.assemble(prog)
        r = LF.run_program_lean(lean, code, max_steps=64)
        print(f"  {name:10s} exact={r['exact']} ax_trace={r['ax_trace']} ref={r['ref_trace']}", flush=True)

    print("\n=== (C) decisive probe: lean hidden at the ALU step -- scalar vs nibble decode ===", flush=True)
    from c4_min.nibble_pure_forward import SP_INIT
    for name, prog, want in battery:
        code = isa.assemble(prog)
        op = code[3].op          # the ALU op is at pc=3 in each battery program
        reg_state = {"PC": 0, "AX": 0, "SP": SP_INIT, "BP": SP_INIT, "STACK0": 0}
        store_log = []
        cur_pc = 0
        alu_state = None
        for _ in range(64):
            cur_op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
            x, positions = LF._build_stream_and_overlay(lean, code, reg_state, store_log, None)
            with torch.no_grad():
                hidden, _ = lean.forward(x, past=None, q_positions=positions)
            state = hidden[0, -1]
            pc = _snap(state[L.PC_VAL])
            ax_scalar = _snap(state[L.AX_VAL]) & 0xFF
            ax_nib = _decode_reg_from_nibbles(state, L, L.AX)
            if cur_op == op:
                alu_state = (ax_scalar, ax_nib)
                break
            sp = _snap(state[L.SP_VAL]); bp = _snap(state[L.BP_VAL]); stk = _snap(state[L.STK_VAL])
            reg_state = {"PC": pc, "AX": ax_scalar, "SP": sp, "BP": bp, "STACK0": stk}
            cur_pc = pc
        if alu_state is not None:
            ax_scalar, ax_nib = alu_state
            print(f"  {name:10s} want={want:4d}  scalar_AX_VAL={ax_scalar:4d}  "
                  f"nibble_decode={ax_nib:6d} (nib&0xFF={ax_nib & 0xFF})  "
                  f"scalar_ok={ax_scalar == want}  nib_ok={(ax_nib & 0xFF) == want}", flush=True)
    print("\n[verdict] If nib_ok==True but scalar_ok==False -> ROOT = LEAN DRIVER AX DECODE "
          "(H2); the nibble band (recurrence) is correct.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
