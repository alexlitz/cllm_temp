"""#853 native oracle for dmod.c — the INDEPENDENT byte-exact reference the
transformer run is checked against.

Two independent confirmations that the whole-module compilation of dmod.c returns
40 (exit(40)):

  (A) NATIVE ./c4orig: the classic native c4 binary run directly on dmod.c.
      ``/home/alexlitz/Documents/misc/c4_doom/c4 dmod.c`` -> ``exit(40)`` (exit code 40).

  (B) EMITTED-BYTECODE oracle: run minic_mod.c (the whole-module compiler) as a
      program on the c4_min reference draft (== native c4 semantics) over dmod.c,
      capture the emitted bytecode (137 instrs), prepend a [JSR main; PSH; EXIT]
      thunk (main is the 5th emitted function, ENT @ instr 97), and EXECUTE the
      emitted bytecode -> main() returns AX=40.

Both agree on 40, and the transformer accepts EVERY one of the 89 741 draft steps
byte-exact (run_module_neural_chunked.py: matched=True, 89741/89741) to program HALT.

Usage:
  python -m c4_min.task853_compiler_cfm.native_ref_dmod
"""
import os, sys, subprocess

os.environ['C4_DRAFT_READ_TO_MEM'] = '1'
os.environ['C4_DRAFT_CMP32'] = '1'
os.environ['C4_SHIFT32'] = '1'
os.environ.setdefault('C4_MEM_ADDR_BITS', '18')

WT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(WT))
sys.path.insert(0, REPO)
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_doom')

NATIVE_C4 = '/home/alexlitz/Documents/misc/c4_doom/c4'
MINIC = os.path.join(WT, 'minic_mod.c')
MODULE = os.path.join(WT, 'dmod.c')
MAIN_ENT_IDX = 97   # main() is the 5th emitted function (iabs, FixedMul, FixedDiv, clamp, main)


def native_exit_code():
    """(A) run the native c4 binary on dmod.c; return its exit code (== the C
    program's exit value)."""
    r = subprocess.run([NATIVE_C4, MODULE], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def emitted_bytecode_result():
    """(B) run minic_mod on dmod, exec the emitted bytecode via a JSR-main thunk,
    return main()'s AX (signed)."""
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0x10000
    from pathlib import Path
    from c4_min import isa
    from c4_min import nibble_filesys as FS
    from c4_min.pf_speculative import draft_pf_program
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from run_c4_min import (tag_compiler_syscalls, install_compiler_abi_file_dispatcher,
                            data_segment)
    mc = Path(MINIC).read_text()
    bc, data = compile_c(mc)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    inp = Path(MODULE).read_bytes()
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                                              stdin=FS.InputKVStream(inp, neural=True)))
    d = draft_pf_program(code, max_steps=300000, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
    out = bytes(fio.runner.stdout).decode('latin1')
    pairs = []
    for ln in out.strip().split('\n'):
        ln = ln.strip()
        if ln:
            o, m = ln.split()
            pairs.append((int(o), int(m)))
    IMM, JSR, PSH, EXIT = isa.IMM, isa.JSR, isa.PSH, isa.HALT
    THUNK = 3
    branch_ops = {JSR, isa.JMP, isa.BZ, isa.BNZ}
    body = [(o, (m + THUNK) if o in branch_ops else m) for (o, m) in pairs]
    prog_pairs = [(JSR, MAIN_ENT_IDX + THUNK), (PSH, 0), (EXIT, 0)] + body
    prog = tag_compiler_syscalls([isa.Instr(o, m & 0xFFFFFFFF) for (o, m) in prog_pairs], isa)
    install_compiler_abi_file_dispatcher()
    fio2 = FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                                               stdin=FS.InputKVStream(b"", neural=True)))
    d2 = draft_pf_program(prog, max_steps=300000, mask=0xFFFFFFFF, data_seg=[], fio=fio2)
    ax = d2.frames[-1]['ax'] if d2.frames else None
    def s32(v):
        return v - (1 << 32) if v is not None and (v & (1 << 31)) else v
    return len(pairs), d.step_count, s32(ax)


def main():
    ec, sout = native_exit_code()
    print(f"[native] ./c4orig ({NATIVE_C4}) dmod.c -> exit({ec})  stdout={sout!r}")
    n_emit, draft_steps, ax = emitted_bytecode_result()
    print(f"[emitted] minic_mod emitted {n_emit} instrs (draft {draft_steps} steps); "
          f"exec -> main()->AX={ax}")
    ok = (ec == 40) and (ax == 40)
    print(f"[oracle] native exit(40) == emitted-bytecode result(40): {ok}")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
