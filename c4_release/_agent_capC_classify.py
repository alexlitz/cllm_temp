#!/usr/bin/env python3
"""CAPSTONE PHASE C — classify oracle<->draft divergences.

Runs c4vm32 (independent 32-bit oracle) and the draft over N steps and, at EACH
step, records whether op/PC-index/AX agree.  Crucially it separates:
  * ADDRESS-typed AX (LEA produces a frame ADDRESS; the two VMs use different
    stack BASES by design -> AX differs by a constant base offset, isomorphic),
  * VALUE-typed AX (IMM/ALU/CMP/LI-of-a-value produce architectural VALUES that
    are base-INDEPENDENT and MUST match byte-exact).

A VALUE-op divergence is a REAL model op-gap.  An address-op divergence is the
expected /8->*4 base isomorphism (still reported, counted separately).

Also tests the ADDRESS-BASE-ALIGNED hypothesis: re-run c4vm32 with the draft's
compact stack base (STACK_TOP=SP_INIT, STRIDE aligned) is NOT directly possible
(c4vm32 STRIDE is baked 8), so instead we check: for LEA, does the LOW BYTE of the
oracle frame address == the draft's (bp+4*imm)&0xFF re-based?  If the ONLY
difference is the constant base, the model's compact-image execution is a faithful
isomorph; if the low bytes disagree, it's a true truncation gap.
"""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["C4_PF_CFM"] = "1"
os.environ.setdefault("C4_DRAFT_CMP32", "1")
os.environ.setdefault("C4_IMM_NIBS", "6")
os.environ.setdefault("C4_PC_WIDE", "1")
os.environ.setdefault("C4_SHIFT32", "1")
os.environ.setdefault("C4_CMP32", "1")
os.environ.setdefault("C4_DIVMOD_SIGNED", "1")
os.environ.setdefault("C4_CODE_ADDR_BITS", "20")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, "/home/alexlitz/Documents/misc/c4_doom")
sys.path.insert(0, "/home/alexlitz/Documents/misc/c4_doom/id_port")
import numpy as np  # noqa: E402
from _agent_capC_oracle_trace import traced_run, build_syscall_argc  # noqa: E402


def main():
    N_CAP = int(os.environ.get("N_STEPS_CAP", "2000"))
    from c4_min import isa
    from c4_min import nibble_filesys as FS
    from c4_min.pf_speculative import draft_pf_program
    from run_c4_min import (tag_compiler_syscalls, install_compiler_abi_file_dispatcher,
                            data_segment)
    import c4vm32 as VM32

    snap = np.load(os.path.join(_HERE, "_doom_bytecode_snapshot.npz"))
    ops, imms, data = snap["ops"], snap["imms"], snap["data"]
    n_instr = len(ops)

    code32 = []
    for i in range(n_instr):
        op = int(ops[i]); im = int(imms[i])
        if op in (VM32.LEA, VM32.ENT, VM32.ADJ):
            im = im * 8
        code32.append((op, im))
    argc_map = build_syscall_argc(code32, {VM32.OPEN, VM32.READ, VM32.CLOS, VM32.PRTF,
                                           VM32.LSEEK, VM32.FSTAT, VM32.PUTCHAR,
                                           VM32.GETCHAR}, VM32.ADJ)
    vm = VM32.C4VM32(code32, [int(b) for b in data], stdin=b"q")
    trace = traced_run(vm, code32, argc_map, N_CAP, VM32)

    install_compiler_abi_file_dispatcher()
    code = [isa.Instr(int(ops[i]), int(imms[i])) for i in range(n_instr)]
    code = tag_compiler_syscalls(code, isa)
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=N_CAP, mask=0xFFFFFFFF,
                             data_seg=data_segment([int(b) for b in data]), fio=fio)

    # VALUE ops: AX is a base-independent architectural value.
    VALUE_OPS = {isa.IMM, isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD, isa.OR,
                 isa.XOR, isa.AND, isa.SHL, isa.SHR, isa.EQ, isa.NE, isa.LT,
                 isa.GT, isa.LE, isa.GE, isa.LC}
    # ADDRESS ops: AX is a frame/heap address (base-dependent by isomorphism).
    ADDR_OPS = {isa.LEA}
    # LI: AX = MEM[addr] -> value IF the loaded cell holds a value, address if it
    # holds a pointer; we classify by whether oracle & draft low-byte agree.

    n = min(len(trace), draft.step_count)
    val_ok = val_tot = 0
    addr_lowbyte_ok = addr_tot = 0
    li_val_ok = li_tot = 0
    op_mismatch = pc_mismatch = 0
    first_value_div = None
    op_counts_div = {}
    for s in range(n):
        o = trace[s]; d = draft.frames[s]
        d_pc_before = 0 if s == 0 else draft.frames[s - 1]["pc"]
        o_op = o["op"]; o_ax = o["ax"] & 0xFFFFFFFF; d_ax = d["ax"] & 0xFFFFFFFF
        o_name = isa.NAMES.get(o_op, str(o_op)); d_name = d["op"]
        if o_name != d_name:
            op_mismatch += 1
            continue
        if o["pc_before_idx"] != d_pc_before:
            pc_mismatch += 1
            continue
        if d.get("is_file"):
            continue
        if o_op in VALUE_OPS:
            val_tot += 1
            if o_ax == d_ax:
                val_ok += 1
            else:
                op_counts_div[o_name] = op_counts_div.get(o_name, 0) + 1
                if first_value_div is None:
                    first_value_div = {"step": s, "pc_idx": o["pc_before_idx"],
                                       "op": o_name, "oracle_ax": o_ax, "draft_ax": d_ax}
        elif o_op in ADDR_OPS:
            addr_tot += 1
            # low byte of the frame address must match (the model keeps AX&0xFF).
            if (o_ax & 0xFF) == (d_ax & 0xFF):
                addr_lowbyte_ok += 1
        elif o_op == isa.LI:
            li_tot += 1
            if o_ax == d_ax:
                li_val_ok += 1

    print(f"[capC-classify] compared {n} steps", flush=True)
    print(f"  op-name mismatches:  {op_mismatch}", flush=True)
    print(f"  pc-index mismatches: {pc_mismatch}", flush=True)
    print(f"  VALUE-op AX exact:   {val_ok}/{val_tot}  "
          f"(IMM/ALU/CMP/LC — base-independent, MUST match)", flush=True)
    print(f"  ADDR-op(LEA) lowbyte:{addr_lowbyte_ok}/{addr_tot}  "
          f"(full addr base-dependent; low byte = model's AX&0xFF)", flush=True)
    print(f"  LI (load) AX exact:  {li_val_ok}/{li_tot}  "
          f"(matches only if loaded cell + base align)", flush=True)
    if first_value_div:
        print(f"  FIRST VALUE-OP DIVERGENCE (REAL gap): {json.dumps(first_value_div)}",
              flush=True)
        print(f"  value-op divergence counts by op: {op_counts_div}", flush=True)
    else:
        print(f"  NO VALUE-OP DIVERGENCE — every base-independent architectural VALUE "
              f"(IMM/ALU/CMP/LC) is byte-exact vs the independent c4vm32 oracle over "
              f"{n} steps.", flush=True)
    print("RESULT " + json.dumps({
        "compared": n, "op_mismatch": op_mismatch, "pc_mismatch": pc_mismatch,
        "value_ax_exact": [val_ok, val_tot],
        "lea_lowbyte_exact": [addr_lowbyte_ok, addr_tot],
        "li_exact": [li_val_ok, li_tot],
        "first_value_divergence": first_value_div}))


if __name__ == "__main__":
    main()
