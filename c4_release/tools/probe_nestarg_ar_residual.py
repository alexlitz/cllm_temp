#!/usr/bin/env python3
"""Probe: AR-emitted step-3 AX-marker residual for nested_quad id950.

Runs the AUTOREGRESSIVE faithful decode (model feeds its own emitted tokens),
then reads the residual bands (AX_CARRY_LO/HI, OUTPUT_LO/HI, the OP flags) at
each step's AX marker. Pinpoints whether AX_CARRY itself is lost (cross-step
carry break) or whether OUTPUT just fails to materialize it.
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
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
from neural_vm.unified_compiler.faithful_autoregressive import (
    FaithfulAutoregressiveRunner, _ContextShim,
)
from tools.faithful_interpreter_validate import _faithful_residual_pre_head

SRC = '''int double_it(int x) { return x * 2; }
            int quad(int x) { return double_it(double_it(x)); }
            int main() { return quad(10); }'''
N_STEPS = 4


def main():
    bc, data = compile_c(SRC)
    STEP = Token.STEP_TOKENS
    runner = FaithfulAutoregressiveRunner(disk_cache=False)
    model = runner.model
    layout = runner.layout
    dp = layout.dim_positions
    inner = runner._inner

    shim = _ContextShim(model)
    ctx = list(shim._build_context(bc, data, [], ""))
    prefix_len = len(ctx)

    # AR emit.
    tape = list(ctx)
    for s in range(N_STEPS):
        for _ in range(STEP):
            logits = inner._faithful.forward(tape)
            tape.append(int(torch.argmax(logits[-1]).item()))

    # Faithful residual on the full AR tape.
    x = _faithful_residual_pre_head(model, tape)

    def nibs(row, name):
        base = dp[name]
        vals = [round(float(row[base + k]), 1) for k in range(16)]
        # argmax nibble
        amax = int(torch.argmax(row[base:base + 16]).item())
        return amax, vals

    op_flags = ["OP_IMM", "OP_PSH", "OP_JSR", "OP_ENT", "OP_LEV", "OP_LEA", "OP_LI", "OP_ADJ", "OP_EXIT"]
    for s in range(N_STEPS):
        start = prefix_len + s * STEP
        slc = tape[start:start + STEP]
        ax_off = None
        for i, tk in enumerate(slc):
            if tk == Token.REG_AX:
                ax_off = i
                break
        axpos = start + ax_off
        row = x[axpos]
        c_lo, _ = nibs(row, "AX_CARRY_LO")
        c_hi, _ = nibs(row, "AX_CARRY_HI")
        o_lo, _ = nibs(row, "OUTPUT_LO")
        try:
            o_hi, _ = nibs(row, "OUTPUT_HI_THIS_STEP")
        except KeyError:
            o_hi, _ = nibs(row, "OUTPUT_HI")
        ax_byte0 = (c_hi << 4) | c_lo
        out_byte0 = (o_hi << 4) | o_lo
        emit_ax = [int(slc[ax_off + 1 + j]) & 0xFF for j in range(4)]
        active_ops = [f for f in op_flags if f in dp and float(row[dp[f]]) > 0.5]
        print(f"step {s}: PC_lo={slc[1]} | AX_CARRY byte0=0x{ax_byte0:02x} OUTPUT byte0=0x{out_byte0:02x} | emit_AX={emit_ax} | ops@AXmark={active_ops}")
        if s == 3:
            olo = [round(float(row[dp['OUTPUT_LO'] + k]), 3) for k in range(16)]
            ohi_name = 'OUTPUT_HI_THIS_STEP' if 'OUTPUT_HI_THIS_STEP' in dp else 'OUTPUT_HI'
            ohi = [round(float(row[dp[ohi_name] + k]), 3) for k in range(16)]
            print(f"   step3 OUTPUT_LO raw: {olo}")
            print(f"   step3 {ohi_name} raw: {ohi}")


if __name__ == "__main__":
    main()
