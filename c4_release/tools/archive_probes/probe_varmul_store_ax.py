#!/usr/bin/env python3
"""Probe: var_mul SI-store AX-carry vs OUTPUT materialization (step-9 root).

Runs the AUTOREGRESSIVE faithful decode for `int a; int b; a=A; b=B; return a*b`
and dumps, at EVERY step's AX marker, AX_CARRY_LO/HI vs OUTPUT_LO/HI plus the
active op flags and emitted AX bytes. Pinpoints whether the SI-store of `b`
delivers 0 because OUTPUT fails to materialize the carried AX (the brief's
step-9 root: value in AX_CARRY, OUTPUT empty/zero-default).

Runs CAMPAIGN config by default (the production default). Set A/B via env.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.compiler import compile_c
from neural_vm.token_layout import Token
from neural_vm.unified_compiler.faithful_autoregressive import (
    FaithfulAutoregressiveRunner, _ContextShim,
)
from tools.faithful_interpreter_validate import _faithful_residual_pre_head

A = int(os.environ.get("PROBE_A", "17"))
B = int(os.environ.get("PROBE_B", "3"))
SRC = f"int main() {{ int a; int b; a = {A}; b = {B}; return a * b; }}"
N_STEPS = int(os.environ.get("PROBE_NSTEPS", "16"))


def main():
    print(f"# var_mul A={A} B={B} expect={A*B}  campaign={os.environ.get('C4_NO_STACK0_EMIT','1')!='0'}")
    bc, data = compile_c(SRC)
    STEP = Token.STEP_TOKENS
    runner = FaithfulAutoregressiveRunner(disk_cache=True)
    model = runner.model
    layout = runner.layout
    dp = layout.dim_positions
    inner = runner._inner

    shim = _ContextShim(model)
    ctx = list(shim._build_context(bc, data, [], ""))
    prefix_len = len(ctx)

    tape = list(ctx)
    for s in range(N_STEPS):
        for _ in range(STEP):
            logits = inner._faithful.forward(tape)
            tape.append(int(torch.argmax(logits[-1]).item()))

    x = _faithful_residual_pre_head(model, tape)

    def nib(row, name):
        if name not in dp:
            return -1
        base = dp[name]
        return int(torch.argmax(row[base:base + 16]).item())

    op_flags = ["OP_IMM", "OP_PSH", "OP_SI", "OP_SC", "OP_LI", "OP_LC",
                "OP_LEA", "OP_ENT", "OP_LEV", "OP_MUL", "OP_ADJ", "OP_EXIT"]
    ohi_name = 'OUTPUT_HI_THIS_STEP' if 'OUTPUT_HI_THIS_STEP' in dp else 'OUTPUT_HI'
    for s in range(N_STEPS):
        start = prefix_len + s * STEP
        slc = tape[start:start + STEP]
        ax_off = None
        for i, tk in enumerate(slc):
            if tk == Token.REG_AX:
                ax_off = i
                break
        if ax_off is None:
            print(f"step {s}: no AX marker in slice")
            continue
        axpos = start + ax_off
        row = x[axpos]
        c_lo = nib(row, "AX_CARRY_LO"); c_hi = nib(row, "AX_CARRY_HI")
        o_lo = nib(row, "OUTPUT_LO");  o_hi = nib(row, ohi_name)
        ax_byte0 = (c_hi << 4) | c_lo
        out_byte0 = (o_hi << 4) | o_lo
        emit_ax = [int(slc[ax_off + 1 + j]) & 0xFF for j in range(4)]
        active = [f for f in op_flags if f in dp and float(row[dp[f]]) > 0.5]
        print(f"step {s:2d}: AX_CARRY b0=0x{ax_byte0:02x} OUTPUT b0=0x{out_byte0:02x} "
              f"| emit_AX={emit_ax} | ops@AX={active}")
        # On the SI-store steps, dump the raw OUTPUT_LO band to see the zero-default
        if "OP_SI" in active:
            olo = [round(float(row[dp['OUTPUT_LO'] + k]), 2) for k in range(16)]
            clo = [round(float(row[dp['AX_CARRY_LO'] + k]), 2) for k in range(16)]
            print(f"        OUTPUT_LO raw: {olo}")
            print(f"        AX_CARRY_LO  : {clo}")


if __name__ == "__main__":
    main()
