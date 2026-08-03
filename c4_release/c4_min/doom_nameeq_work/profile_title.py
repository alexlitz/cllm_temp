#!/usr/bin/env python3
"""profile_title.py -- profile the Doom TITLE-frame op stream on the 32-bit c4 VM.

Compiles doom_run.c (READ-ONLY) via compile_c, caches the bytecode, and runs an
instrumented VM that builds a per-PC-index execution histogram to the first
I_FinishUpdate (the title frame).  Reports:
  * total VM steps (should ~match #803's 6.89M),
  * the hottest PC-index ranges (contiguous basic blocks) and their step %,
  * a per-instruction disassembly of the hot region so we can identify the
    V_DrawPatch inner blit loop precisely.

Durable: writes the histogram + hot-loop report to disk (JSON) so re-runs are
free.  doom_run.c / c4vm32.py are NOT modified.
"""
import sys, os, argparse, time, json, pickle, hashlib
from pathlib import Path

ROOT = Path("/home/alexlitz/Documents/misc/c4_doom/id_port")
# THIS worktree's c4_release checkout (so the compiler + memory.c4 that produce the
# bytecode match the branch's doom_blit — keeps entry PCs consistent across both).
PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(ROOT))

from src.compiler import Compiler, Op, Symbol, INT
import c4vm32 as c4vm

WORK = Path(__file__).resolve().parent
LSEEK, FSTAT = 40, 41

OPNAMES = {
    c4vm.LEA:"LEA", c4vm.IMM:"IMM", c4vm.JMP:"JMP", c4vm.JSR:"JSR", c4vm.BZ:"BZ",
    c4vm.BNZ:"BNZ", c4vm.ENT:"ENT", c4vm.ADJ:"ADJ", c4vm.LEV:"LEV", c4vm.LI:"LI",
    c4vm.LC:"LC", c4vm.SI:"SI", c4vm.SC:"SC", c4vm.PSH:"PSH", c4vm.OR:"OR",
    c4vm.XOR:"XOR", c4vm.AND:"AND", c4vm.EQ:"EQ", c4vm.NE:"NE", c4vm.LT:"LT",
    c4vm.GT:"GT", c4vm.LE:"LE", c4vm.GE:"GE", c4vm.SHL:"SHL", c4vm.SHR:"SHR",
    c4vm.ADD:"ADD", c4vm.SUB:"SUB", c4vm.MUL:"MUL", c4vm.DIV:"DIV", c4vm.MOD:"MOD",
    c4vm.OPEN:"OPEN", c4vm.READ:"READ", c4vm.CLOS:"CLOS", c4vm.PRTF:"PRTF",
    c4vm.MALC:"MALC/EXIT?", c4vm.FREE:"FREE", c4vm.MSET:"MSET", c4vm.MCMP:"MCMP",
    c4vm.EXIT:"EXIT", c4vm.NOP:"NOP", c4vm.GETCHAR:"GETCHAR", c4vm.PUTCHAR:"PUTCHAR",
    LSEEK:"LSEEK", FSTAT:"FSTAT",
}


def _register_op(value):
    if value in Op._value2member_map_:
        return
    pseudo = int.__new__(Op, value)
    pseudo._name_ = f"SYS{value}"
    pseudo._value_ = value
    Op._value2member_map_[value] = pseudo
_register_op(LSEEK); _register_op(FSTAT)
SYSCALL_OPS = {c4vm.OPEN, c4vm.READ, c4vm.CLOS, c4vm.PRTF, LSEEK, FSTAT,
               c4vm.PUTCHAR, c4vm.GETCHAR}


class DoomCompiler(Compiler):
    def __init__(self):
        super().__init__()
        self.symbols['lseek'] = Symbol('lseek', 'Sys', INT, LSEEK)
        self.symbols['fstat'] = Symbol('fstat', 'Sys', INT, FSTAT)


def decode(bytecode):
    code = []
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        code.append((op, imm))
    return code


def compute_argc(code):
    argc_at = {}
    for i, (op, imm) in enumerate(code):
        if op in SYSCALL_OPS and i + 1 < len(code) and code[i + 1][0] == 7:
            argc_at[i] = code[i + 1][1] // 8
    return argc_at


def compile_doom():
    """Compile doom_run.c (cached)."""
    cache = WORK / "_doom_bytecode.pkl"
    src_hash = hashlib.sha256((ROOT / "doom_run.c").read_bytes()).hexdigest()[:16]
    if cache.exists():
        with open(cache, "rb") as f:
            d = pickle.load(f)
        if d.get("src_hash") == src_hash:
            print(f"  using cached bytecode ({len(d['code']):,} instrs)", flush=True)
            return d["code"], d["data"], d["argc_at"]
    os.chdir(ROOT)
    src = (ROOT / "doom_run.c").read_text()
    stdlib_path = PARENT / "src" / "compiler" / "stdlib" / "memory.c4"
    if not stdlib_path.exists():
        stdlib_path = PARENT / "src" / "stdlib" / "memory.c4"
    stdlib = stdlib_path.read_text()
    sys.setrecursionlimit(1_000_000)
    print("compiling doom_run.c ...", flush=True)
    t0 = time.time()
    probe = DoomCompiler()
    probe.compile(src + "\n" + stdlib)
    gtop = max((s.value for s in probe.symbols.values()
                if getattr(s, "sclass", "") == "Glo"), default=0x20000)
    heap_base = ((gtop + 0x10000) + 0xFFFF) & ~0xFFFF
    heap_base = max(heap_base, 0x100000)
    stdlib = stdlib.replace("__heap_ptr = 0x20000;", f"__heap_ptr = {hex(heap_base)};")
    bytecode, data = DoomCompiler().compile(src + "\n" + stdlib)
    print(f"  compiled: {len(bytecode):,} instrs ({time.time()-t0:.1f}s)", flush=True)
    code = decode(bytecode)
    argc_at = compute_argc(code)
    with open(cache, "wb") as f:
        pickle.dump({"src_hash": src_hash, "code": code, "data": data,
                     "argc_at": argc_at}, f)
    return code, data, argc_at


class ProfVM(c4vm.C4VM32):
    """C4VM32 + a per-PC-index execution histogram, stopping after the first frame."""
    def run_profile(self, argc_at, max_cycles):
        import array
        code = self.code
        ncode = len(code)
        hist = array.array("Q", bytes(8 * ncode))   # per-idx exec count
        mem = self.mem
        MASK_ = c4vm.MASK; SIGN_ = c4vm.SIGN; STRIDE_ = c4vm.STRIDE
        frb = int.from_bytes
        ax = self.ax; sp = self.sp; bp = self.bp; pc = self.pc
        cycle = self.cycle
        SYSCALLS = (c4vm.OPEN, c4vm.READ, c4vm.CLOS, c4vm.PRTF, LSEEK, FSTAT,
                    c4vm.PUTCHAR, c4vm.GETCHAR)
        frame_done_cycle = None
        # detect frame end: the 128000-hex-char line ends with a newline via PUTCHAR
        newline_run = [0]
        try:
            while cycle < max_cycles:
                idx = pc >> 3
                if idx >= ncode:
                    break
                hist[idx] += 1
                op = code[idx]; imm = op[1]; op = op[0]
                pc += 8; cycle += 1
                if op == c4vm.LI:
                    a = ax & MASK_; ax = frb(mem[a:a+4], "little")
                elif op == c4vm.LEA: ax = (bp + imm) & MASK_
                elif op == c4vm.IMM: ax = imm & MASK_
                elif op == c4vm.PSH:
                    sp -= STRIDE_; a = sp & MASK_
                    mem[a:a+4] = (ax & MASK_).to_bytes(4, "little")
                elif op == c4vm.ADD:
                    a = sp & MASK_; ax = (frb(mem[a:a+4],"little")+ax)&MASK_; sp+=STRIDE_
                elif op == c4vm.SUB:
                    a = sp & MASK_; ax = (frb(mem[a:a+4],"little")-ax)&MASK_; sp+=STRIDE_
                elif op == c4vm.SI:
                    a = sp & MASK_; dst = frb(mem[a:a+4],"little")&MASK_
                    mem[dst:dst+4] = (ax & MASK_).to_bytes(4,"little"); sp+=STRIDE_
                elif op == c4vm.LC:
                    a = ax & MASK_; b = mem[a]
                    ax = (b - 0x100) & MASK_ if b & 0x80 else b
                elif op == c4vm.SC:
                    a = sp & MASK_; dst = frb(mem[a:a+4],"little")&MASK_
                    mem[dst] = ax & 0xFF; sp+=STRIDE_
                elif op == c4vm.JMP: pc = imm * 8
                elif op == c4vm.BZ:
                    if ax == 0: pc = imm*8
                elif op == c4vm.BNZ:
                    if ax != 0: pc = imm*8
                elif op == c4vm.JSR:
                    sp -= STRIDE_; a = sp & MASK_
                    mem[a:a+4] = (pc & MASK_).to_bytes(4,"little"); pc = imm*8
                elif op == c4vm.ENT:
                    sp -= STRIDE_; a = sp & MASK_
                    mem[a:a+4] = (bp & MASK_).to_bytes(4,"little"); bp = sp; sp -= imm
                elif op == c4vm.ADJ: sp += imm
                elif op == c4vm.LEV:
                    sp = bp; a = sp & MASK_
                    bp = frb(mem[a:a+4],"little"); sp+=STRIDE_; a = sp & MASK_
                    pc = frb(mem[a:a+4],"little"); sp+=STRIDE_
                elif op == c4vm.MUL:
                    a = sp & MASK_; x = frb(mem[a:a+4],"little")
                    x = x-(1<<32) if x & SIGN_ else x
                    y = ax-(1<<32) if ax & SIGN_ else ax
                    ax = (x*y)&MASK_; sp+=STRIDE_
                elif op == c4vm.EQ:
                    a = sp & MASK_; ax = 1 if frb(mem[a:a+4],"little")==ax else 0; sp+=STRIDE_
                elif op == c4vm.NE:
                    a = sp & MASK_; ax = 1 if frb(mem[a:a+4],"little")!=ax else 0; sp+=STRIDE_
                elif op == c4vm.LT:
                    a = sp & MASK_; x = frb(mem[a:a+4],"little")
                    x = x-(1<<32) if x&SIGN_ else x; y = ax-(1<<32) if ax&SIGN_ else ax
                    ax = 1 if x<y else 0; sp+=STRIDE_
                elif op == c4vm.GT:
                    a = sp & MASK_; x = frb(mem[a:a+4],"little")
                    x = x-(1<<32) if x&SIGN_ else x; y = ax-(1<<32) if ax&SIGN_ else ax
                    ax = 1 if x>y else 0; sp+=STRIDE_
                elif op == c4vm.LE:
                    a = sp & MASK_; x = frb(mem[a:a+4],"little")
                    x = x-(1<<32) if x&SIGN_ else x; y = ax-(1<<32) if ax&SIGN_ else ax
                    ax = 1 if x<=y else 0; sp+=STRIDE_
                elif op == c4vm.GE:
                    a = sp & MASK_; x = frb(mem[a:a+4],"little")
                    x = x-(1<<32) if x&SIGN_ else x; y = ax-(1<<32) if ax&SIGN_ else ax
                    ax = 1 if x>=y else 0; sp+=STRIDE_
                elif op == c4vm.AND:
                    a = sp & MASK_; ax = (frb(mem[a:a+4],"little")&ax)&MASK_; sp+=STRIDE_
                elif op == c4vm.OR:
                    a = sp & MASK_; ax = (frb(mem[a:a+4],"little")|ax)&MASK_; sp+=STRIDE_
                elif op == c4vm.XOR:
                    a = sp & MASK_; ax = (frb(mem[a:a+4],"little")^ax)&MASK_; sp+=STRIDE_
                elif op == c4vm.SHL:
                    a = sp & MASK_; ax = (frb(mem[a:a+4],"little")<<(ax&31))&MASK_; sp+=STRIDE_
                elif op == c4vm.SHR:
                    a = sp & MASK_; x = frb(mem[a:a+4],"little")
                    x = x-(1<<32) if x&SIGN_ else x; ax = (x>>(ax&31))&MASK_; sp+=STRIDE_
                elif op == c4vm.DIV:
                    a = sp & MASK_; x = frb(mem[a:a+4],"little")
                    x = x-(1<<32) if x&SIGN_ else x; y = ax-(1<<32) if ax&SIGN_ else ax
                    ax = (int(x/y) if y else 0)&MASK_; sp+=STRIDE_
                elif op == c4vm.MOD:
                    a = sp & MASK_; x = frb(mem[a:a+4],"little")
                    x = x-(1<<32) if x&SIGN_ else x; y = ax-(1<<32) if ax&SIGN_ else ax
                    ax = (x-int(x/y)*y if y else 0)&MASK_; sp+=STRIDE_
                elif op in SYSCALLS:
                    self.ax=ax; self.sp=sp; self.bp=bp; self.pc=pc; self.cycle=cycle
                    self._syscall(op, argc_at.get(idx,0) if argc_at else 0)
                    ax=self.ax; sp=self.sp; bp=self.bp; pc=self.pc
                    if op == c4vm.PUTCHAR:
                        # frame-end detection: the hex frame is 128000 chars then '\n'
                        c = self.stdout[-1] if self.stdout else 0
                        if c == 10:  # newline
                            # was it a long hex line? check tail
                            so = self.stdout
                            # find last newline before this one
                            if len(so) >= 2:
                                prevn = so.rfind(b"\n", 0, len(so)-1)
                                linelen = len(so)-1 - (prevn+1)
                                if linelen >= 100000:
                                    frame_done_cycle = cycle
                                    break
                elif op == c4vm.EXIT or op == c4vm.MALC:
                    self.halted = True
                    self.exit_code = ax-(1<<32) if ax & SIGN_ else ax
                    break
                elif op == c4vm.NOP:
                    pass
                else:
                    raise RuntimeError(f"unknown op {op} at idx {idx}")
        finally:
            self.ax=ax; self.sp=sp; self.bp=bp; self.pc=pc; self.cycle=cycle
        return hist, frame_done_cycle


def disasm(code, i0, i1):
    lines = []
    for i in range(i0, i1):
        op, imm = code[i]
        lines.append(f"    [{i}] {OPNAMES.get(op,'?'+str(op))} {imm}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cycles", type=int, default=2_000_000_000)
    args = ap.parse_args()

    code, data, argc_at = compile_doom()
    ncode = len(code)
    print(f"code: {ncode:,} instrs", flush=True)

    vm = ProfVM(code, data, stdin=b"")
    print("profiling to first (title) frame ...", flush=True)
    t0 = time.time()
    hist, frame_cycle = vm.run_profile(argc_at, args.max_cycles)
    dt = time.time() - t0
    total = sum(hist)
    print(f"  ran {vm.cycle:,} steps in {dt:.1f}s; frame_done_cycle={frame_cycle}; "
          f"stdout={len(vm.stdout):,} bytes", flush=True)
    print(f"  TOTAL VM STEPS (title frame) = {total:,}", flush=True)

    # top hot PC indices
    ranked = sorted(range(ncode), key=lambda i: hist[i], reverse=True)
    print("\n==== TOP 40 HOTTEST PC INDICES ====", flush=True)
    print(f"{'idx':>8} {'op':>12} {'imm':>10} {'count':>14} {'pct':>7}")
    for i in ranked[:40]:
        op, imm = code[i]
        print(f"{i:>8} {OPNAMES.get(op,'?'+str(op)):>12} {imm:>10} "
              f"{hist[i]:>14,} {100.0*hist[i]/total:>6.2f}%")

    # cluster the hottest indices into contiguous basic-block ranges (a hot loop
    # is a run of adjacent indices all executed a similar # of times).
    hot = sorted(i for i in range(ncode) if hist[i] > total * 0.001)
    ranges = []
    if hot:
        s = hot[0]; prev = hot[0]
        for i in hot[1:]:
            if i - prev <= 3:
                prev = i
            else:
                ranges.append((s, prev)); s = i; prev = i
        ranges.append((s, prev))
    # score each range by total steps
    rng_scored = []
    for (a, b) in ranges:
        cnt = sum(hist[i] for i in range(a, b+1))
        rng_scored.append((cnt, a, b))
    rng_scored.sort(reverse=True)
    print("\n==== HOT LOOP RANGES (contiguous, >0.1% each idx) ====", flush=True)
    for cnt, a, b in rng_scored[:12]:
        print(f"  idx [{a}..{b}]  steps={cnt:,}  ({100.0*cnt/total:.2f}%)  "
              f"width={b-a+1}")

    # disassemble the top 3 ranges
    print("\n==== DISASSEMBLY OF TOP 3 HOT RANGES ====", flush=True)
    for cnt, a, b in rng_scored[:3]:
        print(f"\n-- range [{a}..{b}]  ({100.0*cnt/total:.2f}% of steps) --")
        print(disasm(code, max(0, a-2), min(ncode, b+3)))

    # durable
    out = {
        "total_steps": total,
        "frame_done_cycle": frame_cycle,
        "top_indices": [{"idx": i, "op": OPNAMES.get(code[i][0], str(code[i][0])),
                         "imm": code[i][1], "count": int(hist[i]),
                         "pct": 100.0*hist[i]/total} for i in ranked[:60]],
        "hot_ranges": [{"start": a, "end": b, "steps": int(cnt),
                        "pct": 100.0*cnt/total,
                        "disasm": disasm(code, a, b+1)} for cnt, a, b in rng_scored[:12]],
    }
    with open(WORK / "title_profile.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {WORK/'title_profile.json'}", flush=True)


if __name__ == "__main__":
    main()
