"""Run lea autoregressively through 872 vs 981/9 models, same process.

Builds both models (efficient mode) via the env-gated widen scaffold,
swaps them into one AutoregressiveVMRunner, and runs serial .run() on
each. Compares exit codes + the generated context token-by-token.

Run: CUDA_VISIBLE_DEVICES=1 python -m tools.probe_widen_run
"""
from __future__ import annotations

import os

from neural_vm.embedding import Opcode


def _make_bytecode(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


def build(target=None, nheads=None):
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
    if target:
        os.environ["C4_FORCE_WIDEN_DMODEL"] = str(target)
        if nheads:
            os.environ["C4_FORCE_WIDEN_NHEADS"] = str(nheads)
    else:
        os.environ.pop("C4_FORCE_WIDEN_DMODEL", None)
        os.environ.pop("C4_FORCE_WIDEN_NHEADS", None)
    m, l = compile_full_vm_dynamic(
        disk_cache=False, strict=False, alu_mode="efficient", max_seq_len=4096,
    )
    return m.cuda().eval()


def main():
    import torch
    from neural_vm.run_vm import AutoregressiveVMRunner

    serial = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0)
    serial._func_call_handlers = {}
    serial._syscall_handlers = {}

    progs = {
        "lea_basic": [Opcode.ENT, (Opcode.IMM, 0), (Opcode.LEA, 2), Opcode.EXIT],
        "bnz_branch": [(Opcode.IMM, 1), (Opcode.BNZ, 3), (Opcode.IMM, 99),
                       (Opcode.IMM, 42), Opcode.EXIT],
        "imm_exit": [(Opcode.IMM, 42), Opcode.EXIT],
    }

    m872 = build(None)
    m981 = build(981, 9)
    print(f"m872 d_model={m872.d_model} hd={m872.blocks[0].attn.head_dim}")
    print(f"m981 d_model={m981.d_model} hd={m981.blocks[0].attn.head_dim}")

    for name, instrs in progs.items():
        bc = _make_bytecode(instrs)
        serial.model = m872
        out0, ec0 = serial.run(bc, max_steps=20)
        serial.model = m981
        out1, ec1 = serial.run(bc, max_steps=20)
        print(f"{name:12s}  872->{ec0}   981->{ec1}   match={ec0==ec1}")


if __name__ == "__main__":
    main()
