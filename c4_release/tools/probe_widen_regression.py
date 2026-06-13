"""Probe: reproduce the d_model-widen regression on test_bnz_branch.

Builds the production model at its natural d_model (872) and a widened
variant (default 920) with NO behavioural change other than the pool
width, then runs a handful of smoke programs through each and prints the
exit codes. Demonstrates the head-repartitioning regression.

Run: CUDA_VISIBLE_DEVICES=1 python tools/probe_widen_regression.py [target_d_model]
"""
from __future__ import annotations

import os
import sys

import torch

from neural_vm.unified_compiler import full_vm_compiler_dynamic as fvc
from neural_vm.unified_compiler.layer_compiler import ModelLayout
from neural_vm.embedding import Opcode  # noqa: F401  (re-exported below)


def _make_bytecode(ops):
    """Build a packed bytecode list (mirror of tests.test_smoke._make_bytecode)."""
    bytecode = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bytecode.append(opcode | (imm << 8))
        else:
            bytecode.append(op)
    return bytecode


PROGRAMS = {
    "bnz_branch": (
        [(Opcode.IMM, 1), (Opcode.BNZ, 3), (Opcode.IMM, 99),
         (Opcode.IMM, 42), Opcode.EXIT],
        15, 42,
    ),
    "bz_branch": (
        [(Opcode.IMM, 0), (Opcode.BZ, 3), (Opcode.IMM, 99),
         (Opcode.IMM, 42), Opcode.EXIT],
        15, 42,
    ),
    "jmp_forward": (
        [(Opcode.JMP, 2), (Opcode.IMM, 99), (Opcode.IMM, 42), Opcode.EXIT],
        15, 42,
    ),
    "imm_exit": (
        [(Opcode.IMM, 42), Opcode.EXIT],
        10, 42,
    ),
    "add_basic": (
        [(Opcode.IMM, 20), Opcode.PSH, (Opcode.IMM, 22), Opcode.ADD, Opcode.EXIT],
        20, 42,
    ),
    "lea_basic": (
        [Opcode.ENT, (Opcode.IMM, 0), (Opcode.LEA, 2), Opcode.EXIT],
        20, None,  # _ne(0): expected nonzero
    ),
    "si_li_roundtrip": (
        [(Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 42), Opcode.SI,
         (Opcode.IMM, 0x200), Opcode.LI, Opcode.EXIT],
        30, 42,
    ),
}


def build_model(target_d_model, n_heads=None):
    """Compile the production VM, optionally padded to target_d_model.

    Uses the C4_FORCE_WIDEN_DMODEL / C4_FORCE_WIDEN_NHEADS env scaffold in
    full_vm_compiler_dynamic so the widen exercises the exact production
    compile path.
    """
    if target_d_model is None:
        os.environ.pop("C4_FORCE_WIDEN_DMODEL", None)
        os.environ.pop("C4_FORCE_WIDEN_NHEADS", None)
    else:
        os.environ["C4_FORCE_WIDEN_DMODEL"] = str(target_d_model)
        if n_heads is not None:
            os.environ["C4_FORCE_WIDEN_NHEADS"] = str(n_heads)
        else:
            os.environ.pop("C4_FORCE_WIDEN_NHEADS", None)
    model, layout = fvc.compile_full_vm_dynamic(
        disk_cache=False, strict=False,
        alu_mode="efficient", max_seq_len=4096,
    )
    return model, layout


def run_programs(model):
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from neural_vm.run_vm import AutoregressiveVMRunner

    # Build a real runner (for _build_context + handlers), then swap its
    # model with the freshly-compiled one so we test THAT model's behaviour.
    serial = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True,
                                    spec_k=0)
    serial._func_call_handlers = {}
    serial._syscall_handlers = {}
    serial.model = model.cuda() if torch.cuda.is_available() else model
    serial.model.eval()
    runner = BatchedPureNeuralRunner(model_runner=serial, csr_inference=False)

    out = {}
    for name, (instrs, max_steps, expected) in PROGRAMS.items():
        bc = _make_bytecode(instrs)
        results = runner.run_batch([bc], max_steps=max_steps, spec_k=0)
        # results: list of (output_str, exit_code) tuples
        r = results[0]
        exit_code = r[1] if isinstance(r, (list, tuple)) else r
        if expected is None:
            ok = exit_code != 0  # _ne(0) check
        else:
            ok = exit_code == expected
        out[name] = (exit_code, expected, ok)
    return out


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 920
    nheads = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print("=== Building baseline (natural d_model) ===")
    m0, l0 = build_model(None)
    print(f"baseline d_model = {l0.d_model}, head_dim = {l0.d_model // 8}")
    r0 = run_programs(m0)
    for name, (ec, exp, ok) in r0.items():
        print(f"  {name:16s} got={ec} expected={exp} {'PASS' if ok else 'FAIL'}")

    hd = (target // nheads) if nheads else (target // 8)
    print(f"\n=== Building widened d_model = {target}, n_heads={nheads or 8}, head_dim={hd} ===")
    m1, l1 = build_model(target, nheads)
    print(f"widened d_model = {l1.d_model}")
    r1 = run_programs(m1)
    for name, (ec, exp, ok) in r1.items():
        flag = "PASS" if ok else "FAIL"
        regressed = " <-- REGRESSED" if (r0[name][2] and not ok) else ""
        print(f"  {name:16s} got={ec} expected={exp} {flag}{regressed}")


if __name__ == "__main__":
    main()
