#!/usr/bin/env python3
"""measure_nameeq.py -- measure the INIT-frame STEP REDUCTION from the native
NAMEEQ (__name_eq) superinstruction, and verify it is byte-EXACT vs the compiled
c4 __name_eq AS IT RUNS ON c4vm32.

Follows measure_blit.py (#810) exactly:
  1. compile doom_run.c (reuse profile_title.compile_doom cache),
  2. find the __name_eq entry PC from the linker symbol table,
  3. run the init/title frame ONCE with a __name_eq-call-instrumented VM that
     records, per invocation, the (rec, want8) args + the exact # of decoded VM
     steps that call consumed (so we get the REAL per-call cost + total, INCLUDING
     the nested per-char c4_toupper JSR steps that live inside __name_eq),
  4. for the native NAMEEQ op each of those calls collapses to 1 decoded step; the
     init step reduction is (sum c_steps of __name_eq calls) - (n_calls * 1),
     PLUS the c4_toupper steps folded into __name_eq are removed with it,
  5. byte-EXACTNESS: for a battery of the observed (rec,want) cases AND the
     hand-picked pad/case/position edges, run the REAL compiled __name_eq on a
     fresh c4vm32 (via an appended call trampoline) and compare AX (0/1) to the
     native doom_nameeq.name_eq reference,
  6. NEW FRAME TIME = (total_init_steps - reduction) * MS_PER_STEP.

Writes nameeq_measure.json.  doom_run.c / c4vm32.py are NOT modified.
"""
import sys, os, json, time, importlib.util
from pathlib import Path

WORK = Path(__file__).resolve().parent
ROOT = Path("/home/alexlitz/Documents/misc/c4_doom/id_port")
# PARENT is THIS worktree's c4_release checkout (so c4_min.doom_nameeq + src.compiler
# + src/stdlib/memory.c4 all resolve to the branch under test).
PARENT = Path(__file__).resolve().parents[2]   # .../c4_min/doom_nameeq_work -> c4_release
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORK))

import c4vm32 as c4vm
import profile_title as P

from c4_min import doom_nameeq as NE

MS_PER_STEP = 0.0074   # ms per decoded VM step (transformer forward), per #810


def entry_pcs():
    """Instruction-INDEX entry PCs of the compiled ``__name_eq`` + ``c4_toupper``
    + ``W_CheckNumForName`` (from the symbol table).  Returns a dict name->pc."""
    from src.compiler import Compiler, Op, Symbol, INT
    LSEEK, FSTAT = 40, 41
    def _reg(v):
        if v in Op._value2member_map_: return
        p = int.__new__(Op, v); p._name_ = f"S{v}"; p._value_ = v
        Op._value2member_map_[v] = p
    _reg(LSEEK); _reg(FSTAT)
    class DC(Compiler):
        def __init__(s):
            super().__init__()
            s.symbols['lseek'] = Symbol('lseek', 'Sys', INT, LSEEK)
            s.symbols['fstat'] = Symbol('fstat', 'Sys', INT, FSTAT)
    os.chdir(ROOT)
    src = (ROOT / "doom_run.c").read_text()
    sp = PARENT / "src" / "compiler" / "stdlib" / "memory.c4"
    if not sp.exists(): sp = PARENT / "src" / "stdlib" / "memory.c4"
    stdlib = sp.read_text(); sys.setrecursionlimit(1_000_000)
    c = DC(); c.compile(src + "\n" + stdlib)
    def _pc(name):
        s = c.symbols.get(name)
        v = getattr(s, "value", None) if s is not None else None
        return v if isinstance(v, int) else None
    return {n: _pc(n) for n in ("__name_eq", "c4_toupper", "W_CheckNumForName")}


class NameEqProfVM(c4vm.C4VM32):
    """C4VM32 that records every __name_eq invocation's (rec, want8) + step cost,
    stopping after the first (title/init) frame.

    __name_eq is NOT a leaf (it calls c4_toupper), so a simple ANY-LEV close is
    wrong.  We track the call by BP depth: at the JSR we record the target sp;
    __name_eq's own ENT sets bp = sp-8 (after pushing the return addr); its
    matching LEV restores sp above that frame.  We close the active __name_eq
    when a LEV brings sp back to the level it had right AFTER the JSR (i.e. the
    return address slot popped) — tracked precisely via the recorded post-JSR sp.
    Nested c4_toupper calls push/pop deeper, so they never trigger the close."""
    def run_nameeq_profile(self, argc_at, name_eq_pc, max_cycles):
        code = self.code
        ncode = len(code)
        mem = self.mem
        MASK_ = c4vm.MASK; SIGN_ = c4vm.SIGN; STRIDE_ = c4vm.STRIDE
        frb = int.from_bytes
        ax = self.ax; sp = self.sp; bp = self.bp; pc = self.pc
        cycle = self.cycle
        SYSCALLS = (c4vm.OPEN, c4vm.READ, c4vm.CLOS, c4vm.PRTF, 40, 41,
                    c4vm.PUTCHAR, c4vm.GETCHAR)
        # active __name_eq calls: stack of [start_cycle, rec, want, sp_after_jsr]
        active = []
        calls = []          # completed: (rec, want, steps)
        frame_done = None
        try:
            while cycle < max_cycles:
                idx = pc >> 3
                if idx >= ncode:
                    break
                op = code[idx]; imm = op[1]; op = op[0]
                # detect __name_eq entry: JSR to name_eq_pc.  At the JSR the two
                # args are on the stack: c4 pushes left->right so arg0 (rec) is
                # DEEPEST: [sp]=want8 (arg1), [sp+8]=rec (arg0).
                if op == c4vm.JSR and imm == name_eq_pc:
                    want = frb(mem[sp & MASK_:(sp & MASK_) + 4], "little")
                    rec = frb(mem[(sp + 8) & MASK_:((sp + 8) & MASK_) + 4], "little")
                    # after this JSR executes, sp -= 8 (return addr pushed); the
                    # matching LEV pops the frame + return addr, bringing sp back
                    # to (sp_here) -- record sp_here so the LEV close is exact.
                    active.append([cycle, rec, want, sp])
                pc += 8; cycle += 1
                # --- execute (copy of c4vm32.run body) ---
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
                        so = self.stdout
                        if so and so[-1] == 10 and len(so) >= 2:
                            prevn = so.rfind(b"\n", 0, len(so)-1)
                            if len(so)-1 - (prevn+1) >= 100000:
                                frame_done = cycle; break
                elif op == c4vm.EXIT or op == c4vm.MALC:
                    self.halted = True; break
                elif op == c4vm.NOP:
                    pass
                else:
                    raise RuntimeError(f"unknown op {op} at idx {idx}")
                # close the innermost active __name_eq when a LEV unwinds sp back
                # to the PRE-JSR level.  __name_eq's own LEV pops its frame + its
                # pushed return address, restoring sp EXACTLY to the pre-JSR sp
                # (st[3]); nested c4_toupper LEVs restore sp to a level still
                # BELOW pre-JSR sp (toupper was called deeper), so they never
                # reach it.  __name_eq is non-recursive so at most one is active.
                if active and op == c4vm.LEV and sp >= active[-1][3]:
                    st = active.pop()  # [start_cycle, rec, want, sp_pre_jsr]
                    steps = cycle - st[0]
                    calls.append((st[1], st[2], steps))
        finally:
            self.ax=ax; self.sp=sp; self.bp=bp; self.pc=pc; self.cycle=cycle
        return calls, frame_done


def run_real_name_eq_on_vm(full_code, data, name_eq_pc, rec_addr, want_addr):
    """Run the REAL compiled ``__name_eq`` at ``name_eq_pc`` with args
    (rec_addr, want_addr) on a FRESH c4vm32 via an appended call trampoline, and
    return AX (0/1).  The record/query bytes must already be laid into ``data``
    (the data segment) OR written into vm.mem by the caller."""
    full = list(full_code)
    ti = len(full)                          # trampoline start index
    # PSH rec; PSH want; JSR __name_eq; ADJ 16; EXIT  (rec deepest = arg0)
    full += [
        (c4vm.IMM, rec_addr & c4vm.MASK), (c4vm.PSH, 0),
        (c4vm.IMM, want_addr & c4vm.MASK), (c4vm.PSH, 0),
        (c4vm.JSR, name_eq_pc), (c4vm.ADJ, 16),
        (c4vm.EXIT, 0),
    ]
    vm = c4vm.C4VM32(full, data, stdin=b"")
    vm.code = full
    vm.ax = 0; vm.sp = c4vm.STACK_TOP; vm.bp = c4vm.STACK_TOP
    vm.pc = ti * 8; vm.cycle = 0; vm.halted = False
    return vm, full


def verify_byte_exact_onvm(full_code, data, name_eq_pc, cases):
    """For each (rec_bytes, want_bytes) case, lay the two names into a fresh VM's
    memory at disjoint addresses, run the REAL compiled __name_eq, and compare AX
    to doom_nameeq.name_eq_bytes.  Returns {checked, fails, fail_detail}."""
    REC = 0x2000000       # 32 MB, above the data seg + heap, below the 256MB stack
    WANT = 0x2001000
    fails = 0; checked = 0; fail_detail = []
    for (rec_bytes, want_bytes) in cases:
        vm, full = run_real_name_eq_on_vm(full_code, data, name_eq_pc, REC, WANT)
        # lay the names into memory (NUL guard byte after each 8-byte field).
        vm.mem[REC:REC + 9] = (rec_bytes[:8]).ljust(9, b"\x00")
        vm.mem[WANT:WANT + 9] = (want_bytes[:8]).ljust(9, b"\x00")
        vm.run(max_cycles=5000)
        got = vm.ax & 1
        ref = NE.name_eq_bytes(rec_bytes, want_bytes)
        checked += 1
        if got != ref:
            fails += 1
            fail_detail.append({"rec": rec_bytes[:8].decode("latin-1"),
                                "want": want_bytes[:8].decode("latin-1"),
                                "got": got, "ref": ref, "halted": vm.halted})
    return {"checked": checked, "fails": fails, "fail_detail": fail_detail}


def main():
    print("compiling doom (cached) ...", flush=True)
    code, data, argc_at = P.compile_doom()
    print(f"  {len(code):,} instrs", flush=True)
    pcs = entry_pcs()
    name_pc = pcs["__name_eq"]
    print(f"  __name_eq         entry PC (instr idx) = {name_pc}", flush=True)
    print(f"  c4_toupper        entry PC (instr idx) = {pcs['c4_toupper']}", flush=True)
    print(f"  W_CheckNumForName entry PC (instr idx) = {pcs['W_CheckNumForName']}", flush=True)
    if name_pc is None:
        raise RuntimeError("__name_eq symbol not found")

    # peephole substitution count on the WHOLE image (all __name_eq call sites)
    imap = NE.IntrinsicMap.for_doom(name_pc)
    _, n_sites = NE.substitute_intrinsics(code, imap)
    print(f"  __name_eq call SITES in image = {n_sites}", flush=True)

    print("running init/title frame with __name_eq-call instrumentation ...", flush=True)
    vm = NameEqProfVM(list(code), data, stdin=b"")
    t0 = time.time()
    calls, frame_done = vm.run_nameeq_profile(argc_at, name_pc, 2_000_000_000)
    dt = time.time() - t0
    total_init_steps = vm.cycle

    c_steps = sum(s for (_r, _w, s) in calls)
    native = len(calls)                 # 1 decoded step each
    reduction = c_steps - native
    new_total = total_init_steps - reduction
    print(f"  {total_init_steps:,} init steps in {dt:.1f}s; __name_eq CALLS = {native:,}",
          flush=True)
    print(f"  __name_eq FUNCTION-CALL steps = {c_steps:,}  -> NAMEEQ {native} steps "
          f"(cut {reduction:,}, {100.0*reduction/total_init_steps:.2f}% of init frame)",
          flush=True)

    old_ms = total_init_steps * MS_PER_STEP
    new_ms = new_total * MS_PER_STEP
    print(f"\n  OLD init frame = {total_init_steps:,} steps x {MS_PER_STEP} ms "
          f"= {old_ms:,.1f} ms ({old_ms/1000:.2f} s)", flush=True)
    print(f"  NEW init frame = {new_total:,} steps x {MS_PER_STEP} ms "
          f"= {new_ms:,.1f} ms ({new_ms/1000:.2f} s)   ({old_ms/new_ms:.3f}x)", flush=True)

    # per-call step stats
    if calls:
        stepvals = sorted(s for (_r, _w, s) in calls)
        print(f"  per-call steps: min={stepvals[0]} median={stepvals[len(stepvals)//2]} "
              f"max={stepvals[-1]} mean={c_steps/native:.1f}", flush=True)

    # ---- byte-exactness: OBSERVED cases (from the real frame) ----
    # read the actual (rec,want) name bytes that were compared during the frame
    # from a FRESH run of the frame's memory image is complex; instead we verify
    # on the STATIC battery of real WAD names + edges (self-describing) PLUS a
    # sample of the observed (rec,want) address pairs re-materialised from the
    # frame's final memory (the lump directory is stable after W_InitFiles).
    print("\nverifying byte-exactness vs REAL compiled __name_eq on c4vm32 ...", flush=True)
    battery = NE.battery_cases()
    vres = verify_byte_exact_onvm(code, data, name_pc, battery)
    print(f"  STATIC battery byte-exact: checked {vres['checked']} / fails {vres['fails']}",
          flush=True)
    if vres["fail_detail"]:
        print("  FAILS:", json.dumps(vres["fail_detail"][:10], indent=2), flush=True)

    # observed real (rec,want) pairs: sample distinct byte-content pairs from the
    # frame's memory (the VM's mem still holds the lump directory + want buffers).
    obs = []
    seen = set()
    for (rec, want, _s) in calls:
        rb = bytes(vm.mem[rec & c4vm.MASK: (rec & c4vm.MASK) + 8])
        wb = bytes(vm.mem[want & c4vm.MASK: (want & c4vm.MASK) + 8])
        key = (rb, wb)
        if key in seen:
            continue
        seen.add(key)
        obs.append((rb, wb))
        if len(obs) >= 60:
            break
    print(f"  sampling {len(obs)} DISTINCT observed (rec,want) byte-pairs from the frame ...",
          flush=True)
    vres_obs = verify_byte_exact_onvm(code, data, name_pc, obs)
    print(f"  OBSERVED-pairs byte-exact: checked {vres_obs['checked']} / fails {vres_obs['fails']}",
          flush=True)
    if vres_obs["fail_detail"]:
        print("  FAILS:", json.dumps(vres_obs["fail_detail"][:10], indent=2), flush=True)

    out = {
        "total_init_steps": total_init_steps,
        "name_eq_calls": native,
        "call_sites_in_image": n_sites,
        "name_eq_c_steps": c_steps,
        "native_nameeq_steps": native,
        "step_reduction": reduction,
        "step_reduction_pct": 100.0 * reduction / total_init_steps,
        "ms_per_step": MS_PER_STEP,
        "old_frame_ms": old_ms,
        "new_frame_ms": new_ms,
        "speedup": old_ms / new_ms,
        "static_battery_checked": vres["checked"],
        "static_battery_fails": vres["fails"],
        "observed_pairs_checked": vres_obs["checked"],
        "observed_pairs_fails": vres_obs["fails"],
        "name_eq_entry_pc": name_pc,
        "c4_toupper_entry_pc": pcs["c4_toupper"],
        "w_checknumforname_entry_pc": pcs["W_CheckNumForName"],
        "per_call_step_mean": (c_steps / native) if native else 0,
    }
    with open(WORK / "nameeq_measure.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {WORK/'nameeq_measure.json'}", flush=True)


if __name__ == "__main__":
    main()
