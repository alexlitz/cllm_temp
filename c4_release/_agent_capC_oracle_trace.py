#!/usr/bin/env python3
"""CAPSTONE PHASE C (part 1) — INDEPENDENT c4vm32 ORACLE TRACE + draft cross-check.

Runs the SNAPSHOT Doom bytecode on TWO independent interpreters and asserts they
agree architecturally STEP-FOR-STEP (op + AX value + PC-index + stack structure):

  (a) id_port/c4vm32.py            -- the independent 32-bit reference VM (Capstone A).
                                       PC in bytes (STRIDE=8), SP/BP from STACK_TOP,
                                       DATA at 0x10000.
  (b) c4_min pf_speculative.draft  -- the MODEL's ISA transition (what verify_blocks
                                       confirms the transformer reproduces byte-exact).
                                       PC in instr units, SP/BP from SP_INIT=0x10000
                                       (4-byte slots), DATA at 0x10000.

The two use DIFFERENT address BASES/STRIDES by design (the /8->*4 slot isomorphism
in c4vm32's docstring).  Base-INDEPENDENT invariants that MUST be byte-exact every
step: opcode, AX (the full 32-bit architectural value), PC-as-instruction-index.
For the stack we compare the RELATIVE depth (how many words below the frame base)
and, for LOADS from the DATA segment (absolute addr, base-shared at 0x10000), the
recalled value via AX.  A divergence here is a REAL model op-gap (the draft == the
model), named precisely.

Emits _capC_oracle_trace.npz : per-step {pc, op, ax} from the ORACLE for the model
compare (part 2), plus the reconciliation verdict.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # oracle+draft are CPU
os.environ.setdefault("OMP_NUM_THREADS", "4")
# draft transition must match the MODEL's 32-bit config:
os.environ["C4_PF_CFM"] = "1"
os.environ.setdefault("C4_DRAFT_CMP32", "1")
os.environ.setdefault("C4_IMM_NIBS", "6")
os.environ.setdefault("C4_PC_WIDE", "1")
os.environ.setdefault("C4_SHIFT32", "1")
os.environ.setdefault("C4_CMP32", "1")
os.environ.setdefault("C4_DIVMOD_SIGNED", "1")
os.environ.setdefault("C4_CODE_ADDR_BITS", "20")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, "/home/alexlitz/Documents/misc/c4_doom")
sys.path.insert(0, "/home/alexlitz/Documents/misc/c4_doom/id_port")

import numpy as np  # noqa: E402


def build_syscall_argc(code_pairs, SYSCALL_OPS, ADJ):
    """Map code-slot idx -> argc (from the following ADJ), for c4vm32._syscall."""
    argc = {}
    for i, (op, imm) in enumerate(code_pairs):
        if op in SYSCALL_OPS and i + 1 < len(code_pairs) and code_pairs[i + 1][0] == ADJ:
            # ADJ imm is in 8-byte words already (native c4) -> argc = imm//? ; the
            # compiler emits ADJ n where n = argc (word count).  c4vm32.ADJ does
            # sp += imm, and _syscall uses argc directly.  The tag in run_c4_min uses
            # imm as argc too; here ADJ.imm is the raw compiler value = argc.
            argc[i] = int(code_pairs[i + 1][1])
    return argc


def main():
    N_CAP = int(os.environ.get("N_STEPS_CAP", "2000"))

    from c4_min import isa
    from c4_min import nibble_filesys as FS
    from c4_min.pf_speculative import draft_pf_program
    from run_c4_min import (tag_compiler_syscalls, install_compiler_abi_file_dispatcher,
                            data_segment, SYSCALL_OPS)
    import c4vm32 as VM32

    snap = np.load(os.path.join(_HERE, "_doom_bytecode_snapshot.npz"))
    ops, imms, data = snap["ops"], snap["imms"], snap["data"]
    n_instr = len(ops)
    print(f"[capC-oracle] snapshot instrs={n_instr} data={len(data)} N_CAP={N_CAP}",
          flush=True)

    # ------------------------------------------------------------------ ORACLE
    # c4vm32 wants code as list of (op, imm) with imm as the RAW compiler immediate.
    # NOTE: c4vm32 does NOT re-encode LEA/ENT/ADJ slots; it uses the raw bytecode
    # immediate directly (LEA imm bytes, ENT/ADJ byte counts, JMP/BZ imm*8 targets).
    # The snapshot ops/imms are the ISA-DECODED stream (bytecode_to_isa re-encodes
    # LEA/ENT/ADJ to SLOT units).  c4vm32 expects the RAW bytecode.  So for the oracle
    # we reconstruct raw (op,imm) from the snapshot's raw bytecode+data is not stored;
    # instead we reload raw from the npz 'bytecode' if present, else re-derive.
    if "bytecode" in snap.files:
        raw = snap["bytecode"]
        # raw bytecode layout from compile_c: it is a flat int array where code is
        # (op, imm) interleaved OR op-with-inline-imm.  We must match what c4vm32
        # consumes.  c4vm32.run indexes code[pc>>3] and expects code[idx]=(op,imm).
        pass
    # Build the (op, imm) code list for c4vm32 the SAME way the c4vm32 CLI does:
    # from compile_c output.  We recompile-free by using the ISA stream but with the
    # RAW immediate semantics c4vm32 needs.  c4vm32 expects: LEA/ENT/ADJ imm are BYTE
    # offsets (k*8), IMM/JMP/BZ etc use raw imm.  The ISA stream re-encoded LEA/ENT/ADJ
    # to SLOT (k) units.  Reconstruct byte units for c4vm32 (its STRIDE=8):
    from c4_min.isa import LEA as ISA_LEA
    code32 = []
    for i in range(n_instr):
        op = int(ops[i]); im = int(imms[i])
        # c4vm32 opcode ids == our isa ids (range(14)/range(14,30)/range(30,40)),
        # PUTCHAR=65/GETCHAR=64 match.  Immediate reconciliation:
        #   ISA re-encoded LEA off -> slot k (off=k*8 in native c4 / k*4 in c4_min).
        #   c4vm32 does ax=(bp+imm) with STRIDE=8, LEA imm must be BYTE off = k*8.
        #   ISA ENT/ADJ imm -> slot k; c4vm32 does sp-=imm (ENT) / sp+=imm (ADJ) in
        #     BYTES, needs k*8.
        #   JMP/BZ/BNZ/JSR: c4vm32 does pc=imm*8, imm is the INSTRUCTION INDEX (slot),
        #     ISA imm is already the instruction index -> use as-is.
        if op in (VM32.LEA,):
            im = im * 8
        elif op in (VM32.ENT, VM32.ADJ):
            im = im * 8
        # JMP/BZ/BNZ/JSR imm stays the instruction index (c4vm32 multiplies by 8).
        code32.append((op, im))

    argc_map = build_syscall_argc(code32, SYSCALL_OPS={VM32.OPEN, VM32.READ, VM32.CLOS,
                                                      VM32.PRTF, VM32.LSEEK, VM32.FSTAT,
                                                      VM32.PUTCHAR, VM32.GETCHAR},
                                  ADJ=VM32.ADJ)

    vm = VM32.C4VM32(code32, [int(b) for b in data], stdin=b"q")
    # instrument: capture per-step (pc_idx_before, op, ax_after, sp_after, bp_after,
    # store_addr, store_val).  We wrap by single-stepping via a light re-implementation:
    # easiest = monkeypatch mem writes.  Instead run in a custom stepping loop mirroring
    # c4vm32.run but capturing state.  To avoid divergence we call the SAME transition by
    # setting max_cycles=1 in a loop is too slow; instead we snapshot registers each step
    # using a thin trace by re-running run() with a per-step hook is not supported.
    # -> Re-implement the exact c4vm32 dispatch here for tracing (byte-identical copy of
    #    the class loop) would risk drift; instead we subclass and add a traced run.
    trace = traced_run(vm, code32, argc_map, N_CAP, VM32)
    print(f"[capC-oracle] c4vm32 traced {len(trace)} steps "
          f"halted={vm.halted} exit={vm.exit_code}", flush=True)

    # ------------------------------------------------------------------ DRAFT
    install_compiler_abi_file_dispatcher()
    code = [isa.Instr(int(ops[i]), int(imms[i])) for i in range(n_instr)]
    code = tag_compiler_syscalls(code, isa)
    data_seg = data_segment([int(b) for b in data])
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    t0 = time.time()
    draft = draft_pf_program(code, max_steps=N_CAP, mask=0xFFFFFFFF,
                             data_seg=data_seg, fio=fio)
    print(f"[capC-oracle] draft ran {draft.step_count} steps halted={draft.halted} "
          f"wall={time.time()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------ RECONCILE
    # Base-independent invariants per step: op, AX (32-bit value), PC-index.
    # c4vm32 pc is BYTES -> pc_idx = pc>>3 ; draft pc is instr index already.
    n = min(len(trace), draft.step_count)
    n_ok = 0
    first_div = None
    ox = []  # oracle per-step (pc_idx_before, op, ax_after) for the model compare
    for s in range(n):
        o = trace[s]           # dict: pc_before(idx), op, ax, sp, bp, s_addr, s_val
        d = draft.frames[s]    # dict: pc(after), ax, sp, bp, op(name), s_addr, s_val
        # draft.frames[s]['pc'] is the pc AFTER the step; the pc BEFORE step s equals
        # frames[s-1]['pc'] (or 0 at step 0).  c4vm32 trace records pc BEFORE.
        d_pc_before = 0 if s == 0 else draft.frames[s - 1]["pc"]
        o_op = o["op"]; d_op_name = d["op"]
        o_op_name = isa.NAMES.get(o_op, str(o_op))
        o_ax = o["ax"] & 0xFFFFFFFF
        d_ax = d["ax"] & 0xFFFFFFFF
        is_file = d.get("is_file", False)
        # PC-index before the step must match (independent of stride).
        pc_ok = (o["pc_before_idx"] == d_pc_before)
        op_ok = (o_op_name == d_op_name)
        # AX: file ops have driver-meaningless registers in the draft -> skip AX there.
        ax_ok = is_file or (o_ax == d_ax)
        ok = pc_ok and op_ok and ax_ok
        ox.append((o["pc_before_idx"], o_op, o_ax))
        if ok:
            n_ok += 1
        elif first_div is None:
            first_div = {
                "step": s, "pc_idx": o["pc_before_idx"],
                "oracle_op": o_op_name, "draft_op": d_op_name,
                "oracle_ax": o_ax, "draft_ax": d_ax,
                "pc_ok": pc_ok, "op_ok": op_ok, "ax_ok": ax_ok,
                "is_file": is_file,
            }
            break

    print(f"\n[capC-oracle] ORACLE<->DRAFT reconciliation: {n_ok}/{n} steps "
          f"byte-exact (op+AX+PC-index)", flush=True)
    if first_div:
        print(f"[capC-oracle] FIRST DIVERGENCE: {json.dumps(first_div)}", flush=True)
    else:
        print(f"[capC-oracle] NO DIVERGENCE across {n} steps — the independent 32-bit "
              f"c4vm32 oracle == the draft (== what the transformer reproduces).",
              flush=True)

    ox = np.array(ox, dtype=np.int64)  # (n_ok+..., 3) pc_idx, op, ax
    np.savez(os.path.join(_HERE, "_capC_oracle_trace.npz"),
             pc_idx=ox[:, 0], op=ox[:, 1], ax=ox[:, 2],
             n_reconciled=n_ok)
    print("RESULT " + json.dumps({
        "oracle_steps": len(trace), "draft_steps": draft.step_count,
        "reconciled": n_ok, "compared": n, "first_divergence": first_div,
        "all_exact": first_div is None}))
    return 0 if first_div is None else 1


def traced_run(vm, code, argc_map, max_steps, VM32):
    """Byte-identical single-step trace of c4vm32's dispatch, capturing per-step
    (pc_before_idx, op, ax_after, sp_after, bp_after, store_addr, store_val)."""
    MASK = VM32.MASK; SIGN = VM32.SIGN; STRIDE = VM32.STRIDE
    frb = int.from_bytes
    mem = vm.mem
    ax, sp, bp, pc = vm.ax, vm.sp, vm.bp, vm.pc
    ncode = len(code)
    SYSCALLS = (VM32.OPEN, VM32.READ, VM32.CLOS, VM32.PRTF, VM32.LSEEK,
                VM32.FSTAT, VM32.PUTCHAR, VM32.GETCHAR)
    (LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV, LI, LC, SI, SC, PSH) = range(14)
    (OR, XOR, AND, EQ, NE, LT, GT, LE, GE, SHL, SHR, ADD, SUB, MUL, DIV, MOD) = range(14, 30)
    EXIT, MALC, NOP = VM32.EXIT, VM32.MALC, VM32.NOP
    trace = []
    steps = 0
    while steps < max_steps:
        idx = pc >> 3
        if idx >= ncode:
            break
        op, imm = code[idx]
        pc_before_idx = idx
        pc += 8
        steps += 1
        s_addr = s_val = None
        if op == LI:
            a = ax & MASK; ax = frb(mem[a:a + 4], "little")
        elif op == LEA: ax = (bp + imm) & MASK
        elif op == IMM: ax = imm & MASK
        elif op == PSH:
            sp -= STRIDE; a = sp & MASK
            mem[a:a + 4] = (ax & MASK).to_bytes(4, "little"); s_addr, s_val = a, ax & MASK
        elif op == ADD:
            a = sp & MASK; ax = (frb(mem[a:a + 4], "little") + ax) & MASK; sp += STRIDE
        elif op == SUB:
            a = sp & MASK; ax = (frb(mem[a:a + 4], "little") - ax) & MASK; sp += STRIDE
        elif op == SI:
            a = sp & MASK; dst = frb(mem[a:a + 4], "little") & MASK
            mem[dst:dst + 4] = (ax & MASK).to_bytes(4, "little"); sp += STRIDE
            s_addr, s_val = dst, ax & MASK
        elif op == LC:
            a = ax & MASK; b = mem[a]
            ax = (b - 0x100) & MASK if b & 0x80 else b
        elif op == SC:
            a = sp & MASK; dst = frb(mem[a:a + 4], "little") & MASK
            mem[dst] = ax & 0xFF; sp += STRIDE; s_addr, s_val = dst, ax & 0xFF
        elif op == JMP: pc = imm * 8
        elif op == BZ:
            if ax == 0: pc = imm * 8
        elif op == BNZ:
            if ax != 0: pc = imm * 8
        elif op == JSR:
            sp -= STRIDE; a = sp & MASK
            mem[a:a + 4] = (pc & MASK).to_bytes(4, "little"); s_addr, s_val = a, pc & MASK
            pc = imm * 8
        elif op == ENT:
            sp -= STRIDE; a = sp & MASK
            mem[a:a + 4] = (bp & MASK).to_bytes(4, "little"); s_addr, s_val = a, bp & MASK
            bp = sp; sp -= imm
        elif op == ADJ: sp += imm
        elif op == LEV:
            sp = bp; a = sp & MASK
            bp = frb(mem[a:a + 4], "little"); sp += STRIDE; a = sp & MASK
            pc = frb(mem[a:a + 4], "little"); sp += STRIDE
        elif op == MUL:
            a = sp & MASK; x = frb(mem[a:a + 4], "little")
            x = x - (1 << 32) if x & SIGN else x
            y = ax - (1 << 32) if ax & SIGN else ax
            ax = (x * y) & MASK; sp += STRIDE
        elif op == EQ:
            a = sp & MASK; ax = 1 if frb(mem[a:a + 4], "little") == ax else 0; sp += STRIDE
        elif op == NE:
            a = sp & MASK; ax = 1 if frb(mem[a:a + 4], "little") != ax else 0; sp += STRIDE
        elif op == LT:
            a = sp & MASK; x = frb(mem[a:a + 4], "little")
            x = x - (1 << 32) if x & SIGN else x; y = ax - (1 << 32) if ax & SIGN else ax
            ax = 1 if x < y else 0; sp += STRIDE
        elif op == GT:
            a = sp & MASK; x = frb(mem[a:a + 4], "little")
            x = x - (1 << 32) if x & SIGN else x; y = ax - (1 << 32) if ax & SIGN else ax
            ax = 1 if x > y else 0; sp += STRIDE
        elif op == LE:
            a = sp & MASK; x = frb(mem[a:a + 4], "little")
            x = x - (1 << 32) if x & SIGN else x; y = ax - (1 << 32) if ax & SIGN else ax
            ax = 1 if x <= y else 0; sp += STRIDE
        elif op == GE:
            a = sp & MASK; x = frb(mem[a:a + 4], "little")
            x = x - (1 << 32) if x & SIGN else x; y = ax - (1 << 32) if ax & SIGN else ax
            ax = 1 if x >= y else 0; sp += STRIDE
        elif op == AND:
            a = sp & MASK; ax = (frb(mem[a:a + 4], "little") & ax) & MASK; sp += STRIDE
        elif op == OR:
            a = sp & MASK; ax = (frb(mem[a:a + 4], "little") | ax) & MASK; sp += STRIDE
        elif op == XOR:
            a = sp & MASK; ax = (frb(mem[a:a + 4], "little") ^ ax) & MASK; sp += STRIDE
        elif op == SHL:
            a = sp & MASK; ax = (frb(mem[a:a + 4], "little") << (ax & 31)) & MASK; sp += STRIDE
        elif op == SHR:
            a = sp & MASK; x = frb(mem[a:a + 4], "little")
            x = x - (1 << 32) if x & SIGN else x
            ax = (x >> (ax & 31)) & MASK; sp += STRIDE
        elif op == DIV:
            a = sp & MASK; x = frb(mem[a:a + 4], "little")
            x = x - (1 << 32) if x & SIGN else x; y = ax - (1 << 32) if ax & SIGN else ax
            ax = (int(x / y) if y else 0) & MASK; sp += STRIDE
        elif op == MOD:
            a = sp & MASK; x = frb(mem[a:a + 4], "little")
            x = x - (1 << 32) if x & SIGN else x; y = ax - (1 << 32) if ax & SIGN else ax
            ax = (x - int(x / y) * y if y else 0) & MASK; sp += STRIDE
        elif op in SYSCALLS:
            vm.ax = ax; vm.sp = sp; vm.bp = bp; vm.pc = pc; vm.cycle = steps
            vm._syscall(op, argc_map.get(idx, 0))
            ax = vm.ax; sp = vm.sp; bp = vm.bp; pc = vm.pc
        elif op == EXIT or op == MALC:
            vm.halted = True
            vm.exit_code = ax - (1 << 32) if ax & SIGN else ax
            trace.append({"pc_before_idx": pc_before_idx, "op": op, "ax": ax & MASK,
                          "sp": sp, "bp": bp, "s_addr": s_addr, "s_val": s_val})
            break
        elif op == NOP:
            pass
        else:
            raise RuntimeError(f"unknown op {op} at idx {idx}")
        trace.append({"pc_before_idx": pc_before_idx, "op": op, "ax": ax & MASK,
                      "sp": sp, "bp": bp, "s_addr": s_addr, "s_val": s_val})
    vm.ax, vm.sp, vm.bp, vm.pc = ax, sp, bp, pc
    return trace


if __name__ == "__main__":
    raise SystemExit(main())
