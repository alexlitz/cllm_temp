"""Full malloc_printf per-step PC/SP/BP/AX trace vs ref_interpret, to localize the
empty-output failure.  Runs to completion (or max_steps) and reports the FIRST step
where the model's PC diverges from the reference (the control-flow break), plus the
trace around the first printf (pc reaching the PRTF op).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from c4_min import isa
from c4_min import libprog_corpus as LC
from c4_min import nibble_pure_forward_complete as C
from c4_min.nibble_pure_forward import _snap_lane

STEPS = int(os.environ.get("C4_DIAG_STEPS", "260"))


def ref_pc_trace(code, max_steps):
    """Reference (pc, op, ax, sp, bp) per step under low stack."""
    from c4_min.nibble_pure_forward import SP_INIT
    mem = {}
    sp = bp = SP_INIT
    ax = pc = 0
    out = []
    steps = 0
    ADJ = isa.ADJ
    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]; op, imm = ins.op, ins.imm
        i = pc; pc += 1
        if op == isa.IMM: ax = imm & 0xFFFFFFFF
        elif op == isa.LEA: ax = (bp + 4 * imm) & 0xFFFFFFFF
        elif op == isa.PSH: sp -= 4; mem[sp] = ax
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
            v = mem.get(sp, 0); sp += 4
            ax = {isa.ADD: v+ax, isa.SUB: v-ax, isa.MUL: v*ax,
                  isa.DIV: (v//ax if ax else 0), isa.MOD: (v%ax if ax else 0)}[op] & 0xFFFFFFFF
        elif op in (isa.OR,isa.XOR,isa.AND,isa.SHL,isa.SHR):
            v = mem.get(sp,0)&0xFF; sp+=4
            ax = {isa.OR:v|ax,isa.XOR:v^ax,isa.AND:v&ax,isa.SHL:v<<ax,isa.SHR:v>>ax}[op]&0xFF
        elif op in (isa.EQ,isa.NE,isa.LT,isa.GT,isa.LE,isa.GE):
            v=mem.get(sp,0)&0xFF; sp+=4
            ax=1 if {isa.EQ:v==ax,isa.NE:v!=ax,isa.LT:v<ax,isa.GT:v>ax,isa.LE:v<=ax,isa.GE:v>=ax}[op] else 0
        elif op in (isa.LI, isa.LC): ax = mem.get(ax, 0) & (0xFFFFFFFF if op==isa.LI else 0xFF)
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp,0); sp += 4; mem[addr] = ax & (0xFFFFFFFF if op==isa.SI else 0xFF)
        elif op == isa.JMP: pc = imm
        elif op == isa.BZ: pc = imm if ax == 0 else pc
        elif op == isa.BNZ: pc = imm if ax != 0 else pc
        elif op == isa.JSR: sp -= 4; mem[sp] = i + 1; pc = imm
        elif op == isa.ENT: mem[sp-4]=bp; sp-=4; bp=sp; sp-=4*imm
        elif op == ADJ: sp += 4*imm
        elif op == isa.LEV: sp=bp; bp=mem.get(sp,0); pc=mem.get(sp+4,0); sp+=8
        elif op == isa.PRTF: pass
        elif op == isa.HALT: out.append((i, op, ax&0xFF, sp, bp)); break
        out.append((i, op, ax & 0xFF, sp, bp))
    return out


def main():
    os.system("free -g | head -2")
    entry = [e for e in LC.CORPUS if e.name == "malloc_printf"][0]
    raw, data = LC.compile_entry(entry)
    instrs = LC.retarget_to_neural_abi(raw)
    from c4_min.lib_neural import build_lib_model_streaming
    print("building STREAMING model ...", flush=True)
    sparse, L, _ = build_lib_model_streaming(
        code_size=len(instrs) + 2, recurrent_divmod=True, addr32=True)
    print("built; dim", L.D, flush=True)

    from c4_min import nibble_filesys as FS
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached

    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem(dict(entry.files)),
        stdin=FS.InputKVStream(entry.stdin)))

    with LC._low_stack_sp():
        ref = ref_pc_trace(instrs, STEPS)
        with LC._install_fileop_marshalling():
            got = run_pure_forward_cached(
                sparse, L, instrs, max_steps=STEPS, mask=0xFFFFFFFF, verbose=True,
                fio=fio, data_seg=LC._bytes_to_seg(data),
                evict=True, prune_interval=60)

    print("\nstdout:", repr(bytes(fio.runner.stdout)), flush=True)
    print("ref steps:", len(ref), " model AX-trace len:", len(got), flush=True)
    # ref AX per step (index-aligned with got which is per-step AX)
    ref_ax = [r[2] for r in ref]
    n = min(len(ref_ax), len(got))
    div = next((i for i in range(n) if ref_ax[i] != got[i]), None)
    if div is None:
        print(f"NO AX divergence in first {n} steps; ref pc-op tail:")
        for r in ref[max(0, n-8):n]:
            print("   ", r)
    else:
        print(f"FIRST AX divergence at step {div}: ref_ax={ref_ax[div]} got={got[div]}")
        print("  around (ref pc,op,ax,sp,bp):")
        for r in ref[max(0, div-4):div+2]:
            print("   ", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
