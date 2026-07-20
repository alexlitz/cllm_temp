"""Run REAL malloc_printf through the model (bounded), capturing the FULL 32-bit
per-step AX (the driver's ``trace`` at mask=0xFFFFFFFF), and compare EACH LI/LC
step's model AX against a word-width python-truth interpreter run in lock-step on
the SAME control flow the model takes.  Reports the FIRST LI/LC whose model AX
disagrees with truth — the memory-CAM read failure, with full 32-bit values.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from c4_min import isa
from c4_min import libprog_corpus as LC

STEPS = int(os.environ.get("C4_STEPS", "130"))


def word_ref_trace(code, max_steps, sp_init):
    """Word-width reference: (pc,op,ax,sp,bp, li_addr, li_val) per step.  li_addr /
    li_val are set only on LI/LC (the loaded address + true loaded value)."""
    mem = {}
    sp = bp = sp_init
    ax = pc = 0
    out = []
    steps = 0
    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]; op, imm = ins.op, ins.imm
        i = pc; pc += 1
        li_addr = li_val = None
        if op == isa.IMM: ax = imm & 0xFFFFFFFF
        elif op == isa.LEA: ax = (bp + 4 * imm) & 0xFFFFFFFF
        elif op == isa.PSH: sp -= 4; mem[sp] = ax
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
            v = mem.get(sp, 0); sp += 4
            ax = {isa.ADD: v + ax, isa.SUB: v - ax, isa.MUL: v * ax,
                  isa.DIV: (v // ax if ax else 0),
                  isa.MOD: (v % ax if ax else 0)}[op] & 0xFFFFFFFF
        elif op in (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR):
            v = mem.get(sp, 0) & 0xFF; sp += 4
            ax = {isa.OR: v | ax, isa.XOR: v ^ ax, isa.AND: v & ax,
                  isa.SHL: v << ax, isa.SHR: v >> ax}[op] & 0xFF
        elif op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            v = mem.get(sp, 0) & 0xFF; sp += 4
            ax = 1 if {isa.EQ: v == ax, isa.NE: v != ax, isa.LT: v < ax,
                       isa.GT: v > ax, isa.LE: v <= ax, isa.GE: v >= ax}[op] else 0
        elif op in (isa.LI, isa.LC):
            li_addr = ax
            ax = mem.get(ax, 0) & (0xFFFFFFFF if op == isa.LI else 0xFF)
            li_val = ax
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp, 0); sp += 4
            mem[addr] = ax & (0xFFFFFFFF if op == isa.SI else 0xFF)
        elif op == isa.JMP: pc = imm
        elif op == isa.BZ: pc = imm if ax == 0 else pc
        elif op == isa.BNZ: pc = imm if ax != 0 else pc
        elif op == isa.JSR: sp -= 4; mem[sp] = i + 1; pc = imm
        elif op == isa.ENT: mem[sp - 4] = bp; sp -= 4; bp = sp; sp -= 4 * imm
        elif op == isa.ADJ: sp += 4 * imm
        elif op == isa.LEV: sp = bp; bp = mem.get(sp, 0); pc = mem.get(sp + 4, 0); sp += 8
        elif op == isa.PRTF: pass
        elif op == isa.HALT: out.append((i, op, ax, sp, bp, li_addr, li_val)); break
        out.append((i, op, ax, sp, bp, li_addr, li_val))
    return out


def main():
    os.system("free -g | head -2")
    entry = [e for e in LC.CORPUS if e.name == "malloc_printf"][0]
    raw, data = LC.compile_entry(entry)
    instrs = LC.retarget_to_neural_abi(raw)
    print("n instrs:", len(instrs), flush=True)

    from c4_min.lib_neural import build_lib_model_streaming
    print("building STREAMING model ...", flush=True)
    sparse, L, _ = build_lib_model_streaming(
        code_size=len(instrs) + 2, recurrent_divmod=True, addr32=True)
    print("built; dim", L.D, flush=True)
    os.system("free -g | head -2")

    from c4_min import nibble_filesys as FS
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    from c4_min.nibble_pure_forward import SP_INIT as _  # noqa
    import c4_min.nibble_pure_forward_cached as pfcache

    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem(dict(entry.files)),
        stdin=FS.InputKVStream(entry.stdin)))

    evict = os.environ.get("C4_EVICT", "1") not in ("0", "")
    pi = int(os.environ.get("C4_PRUNE_INTERVAL", "60"))
    print(f"evict={evict} prune_interval={pi} steps={STEPS}", flush=True)
    with LC._low_stack_sp():
        sp_init = pfcache.SP_INIT
        ref = word_ref_trace(instrs, STEPS, sp_init)
        # per-step live comparison: patch _decode_reg_from_nibbles to log step-by-step.
        import c4_min.nibble_pure_forward_cached as _pfc
        _orig_dec = _pfc._decode_reg_from_nibbles
        step = {"i": 0}

        def _traced(state, L_, reg_base):
            v = _orig_dec(state, L_, reg_base)
            # only the AX decode (reg_base == L.AX) drives the per-step AX; log it.
            if reg_base == L_.AX:
                i = step["i"]
                if i < len(ref):
                    pcv, op, rax, rsp, rbp, li_addr, li_val = ref[i]
                    tag = ""
                    if op in (isa.LI, isa.LC):
                        tag = (f"  {isa.NAMES[op]} @0x{li_addr:X} truth={li_val} "
                               f"model={v}" + ("  <<< WRONG" if v != li_val else ""))
                    print(f"step {i:3d} pc={pcv:3d} {isa.NAMES.get(op, op):4s} "
                          f"truth_ax={rax} model_ax={v}{tag}", flush=True)
                step["i"] += 1
            return v
        _pfc._decode_reg_from_nibbles = _traced
        try:
            with LC._install_fileop_marshalling():
                got = run_pure_forward_cached(
                    sparse, L, instrs, max_steps=STEPS, mask=0xFFFFFFFF, verbose=False,
                    fio=fio, data_seg=LC._bytes_to_seg(data),
                    evict=evict, prune_interval=pi)
        finally:
            _pfc._decode_reg_from_nibbles = _orig_dec
    print("\nstdout so far:", repr(bytes(fio.runner.stdout)), flush=True)
    print(f"ref steps={len(ref)}  model AX-trace len={len(got)}", flush=True)

    n = min(len(ref), len(got))
    first_ax_div = None
    print("\n--- LI/LC reads (model AX vs word-truth) ---", flush=True)
    for i in range(n):
        pcv, op, rax, rsp, rbp, li_addr, li_val = ref[i]
        max_ = got[i]
        if op in (isa.LI, isa.LC):
            ok = (max_ == li_val)
            mark = "" if ok else "   <-- WRONG READ"
            print(f"  step {i:3d} pc={pcv:3d} {isa.NAMES[op]:3s} @0x{li_addr:X} "
                  f"truth={li_val} model={max_}{mark}", flush=True)
        if rax != max_ and first_ax_div is None:
            first_ax_div = (i, pcv, isa.NAMES.get(op, op), rax, max_)
    if first_ax_div:
        i, pcv, opn, rax, mx = first_ax_div
        print(f"\nFIRST AX divergence: step {i} pc={pcv} {opn} truth={rax} model={mx}",
              flush=True)
    else:
        print(f"\nNO AX divergence in first {n} steps.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
