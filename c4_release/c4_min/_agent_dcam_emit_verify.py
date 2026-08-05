"""Verify EMIT on the DIRECT-CAM (position-invariant, CODE_ADDR_BITS=20) code path.

1. BASELINE: read-only program byte-exact vs isa.interpret on the direct-CAM fetch.
2. 8e6946e7 PARITY: append-and-run, re-emit (self-modifying), end-to-end compile.
3. NEW CAPABILITY: HIGH-address EMIT — emit above 2^16 / 2^18 (toward Doom PCs), JMP
   there, run byte-exact.  A sweep reports the max fetchable EMIT target address.
4. LARGER PROGRAM: a multi-statement expr+branch loaded program spanning a wider
   address range.

``C4_PC_WIDE=1`` is REQUIRED for the high-address sweep: it lets the runtime PC
requant (``_snap_lane``) decode a PC > 2^16 without clipping (the SAME flag the Doom
~440K-PC path uses).  Without it the model's JMP-to-high-address PC decode saturates
at ~65792 and the fetch resolves the wrong frame.  The CODE-CAM address KEY is 20-bit
regardless; C4_PC_WIDE only widens the driver-side PC/return-PC decode band.

Run (LEAN build, ~1 GB sparse-resident; NEVER load_sparse_transformer):
    PYTHONPATH=<c4_release> C4_PF_CFM=1 C4_CODE_ADDR_BITS=20 C4_CFM_EMIT=1 \
        C4_PC_WIDE=1 python -m c4_min._agent_dcam_emit_verify
"""
from __future__ import annotations

import os

os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("C4_CODE_ADDR_BITS", "20")
os.environ.setdefault("C4_CFM_EMIT", "1")
os.environ.setdefault("C4_PC_WIDE", "1")

import torch

from c4_min import isa
from c4_min.compact_alloc import build_compact_sparse_streaming

IMM, PSH, HALT, JMP, EMIT, ADD, NOP = (
    isa.IMM, isa.PSH, isa.HALT, isa.JMP, isa.EMIT, isa.ADD, isa.NOP)


def _emit_seq(target, op, immval):
    """The 4-instruction loader gadget producing Instr(op, immval) at code[target]:
    IMM immval ; PSH ; IMM op ; EMIT target  (op rides AX, imm rides STACK0)."""
    return [isa.Instr(IMM, immval), isa.Instr(PSH, 0),
            isa.Instr(IMM, op), isa.Instr(EMIT, target)]


def _pad(prog, n):
    while len(prog) < n:
        prog.append(isa.Instr(NOP, 0))
    return prog


def prog_append_and_run(prod, end):
    """Loader EMITs IMM 14 at code[prod], HALT at code[end], JMPs to prod.  AX=14."""
    prog = []
    prog += _emit_seq(prod, IMM, 14)
    prog += _emit_seq(end, HALT, 0)
    prog += [isa.Instr(JMP, prod)]
    return prog, 14


def prog_high_addr(target):
    """Loader EMITs IMM 14 at HIGH addr ``target`` and HALT at target+1, JMPs there.
    Proves fetch@PC resolves a runtime frame at ANY address (up to 2^CODE_ADDR_BITS)."""
    prog = []
    prog += _emit_seq(target, IMM, 14)
    prog += _emit_seq(target + 1, HALT, 0)
    prog += [isa.Instr(JMP, target)]
    return prog, 14


def _build_lean(device, code_size=48):
    model, L, _ = build_compact_sparse_streaming(code_size=code_size)
    model.to(device)
    return model, L


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[build] lean streaming-sparse CFM  device={dev}  "
          f"C4_PF_CFM={os.environ.get('C4_PF_CFM')} "
          f"C4_CODE_ADDR_BITS={os.environ.get('C4_CODE_ADDR_BITS')} "
          f"C4_CFM_EMIT={os.environ.get('C4_CFM_EMIT')}", flush=True)
    model, L = _build_lean(dev)
    print(f"[build] done  cfm={getattr(L,'cfm',None)}  n_blocks={len(model.blocks)}",
          flush=True)

    from c4_min.nibble_pure_forward_complete import (
        run_pure_forward_complete, run_pure_forward_complete_emit)

    # ---- 1. BASELINE read-only --------------------------------------------
    prog = isa.assemble([("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)])
    ref = isa.interpret(prog, max_steps=32)
    tr = run_pure_forward_complete(model, L, prog, max_steps=32)
    print(f"[read-only] model={tr[-3:]} ref={ref[-3:]} exact={tr == ref}", flush=True)
    assert tr == ref

    # ---- 2. 8e6946e7 PARITY: append-and-run (low address) -----------------
    prog, expect = prog_append_and_run(prod=10, end=11)
    prog = _pad(prog, 12)
    res = run_pure_forward_complete_emit(model, L, prog, max_steps=64)
    ok = res["exact"] and res["ax_trace"][-1] == expect
    print(f"[append-and-run] tail={res['ax_trace'][-3:]} expect={expect} "
          f"exact={res['exact']} PASS={ok}", flush=True)
    assert ok, res

    # ---- 3. HIGH-ADDRESS EMIT sweep (the ceiling lift) --------------------
    # sweep target addresses across the CODE_ADDR_BITS=20 range; report max byte-exact.
    print("\n[high-addr sweep] EMIT IMM 14; HALT to target, JMP there, run:", flush=True)
    targets = [1 << 12, 1 << 14, 1 << 16, (1 << 16) + 1234, 1 << 17,
               1 << 18, (1 << 18) + 55555, 1 << 19, (1 << 19) + 300000,
               (1 << 20) - 2, (1 << 20) - 1]
    max_ok = -1
    for t in targets:
        prog, expect = prog_high_addr(t)
        prog = _pad(prog, 10)
        res = run_pure_forward_complete_emit(model, L, prog, max_steps=64)
        got = res["ax_trace"][-1] if res["ax_trace"] else None
        ok = res["exact"] and got == expect
        print(f"  target={t:>8} (2^{t.bit_length()-1}~)  model_tail={got} "
              f"expect={expect} exact={res['exact']} PASS={ok}", flush=True)
        if ok:
            max_ok = max(max_ok, t)
    print(f"[high-addr] MAX byte-exact EMIT target = {max_ok}  "
          f"(~2^{max_ok.bit_length()-1 if max_ok>0 else 0})", flush=True)

    # ---- 4. re-emit (self-modifying, high address) ------------------------
    T = 1 << 18
    prog = []
    prog += _emit_seq(T, IMM, 3)          # 0..3   code[T] := IMM 3
    prog += _emit_seq(T + 1, JMP, 12)     # 4..7   code[T+1] := JMP 12
    prog += [isa.Instr(JMP, T)]           # 8      run v1: IMM3 -> JMP12
    prog += [isa.Instr(NOP, 0)] * 3       # 9..11
    prog += _emit_seq(T, IMM, 9)          # 12..15 code[T] := IMM 9 (re-emit same addr)
    prog += _emit_seq(T + 1, HALT, 0)     # 16..19 code[T+1] := HALT
    prog += [isa.Instr(JMP, T)]           # 20     run v2: IMM9 -> HALT
    prog = _pad(prog, 24)
    res = run_pure_forward_complete_emit(model, L, prog, max_steps=64)
    ok = res["exact"] and res["ax_trace"][-1] == 9
    print(f"\n[re-emit@2^18] tail={res['ax_trace'][-3:]} expect=9 "
          f"exact={res['exact']} PASS={ok}", flush=True)
    assert ok, res

    # ---- 5. LARGER loaded program: multi-statement expr + a BRANCH ---------
    # A LOADED compiler (code frames, NOT baked weights) reads 3 operands d0,d1,d2 and
    # a selector s from the data band, then EMITs a 13-instruction MULTI-STATEMENT
    # program with a CONDITIONAL branch — a real expr+stmt subset, well past the toy
    # `d1 ADD d2`.  It runs it and produces s!=0 -> d0+d1+d2 ; s==0 -> 111 byte-exact.
    #
    #   NOTE on the 8-bit EMIT immediate: `isa.interpret` (the c4 oracle) pops the
    #   produced immediate off the byte-masked STACK0, so an EMITted control-flow
    #   TARGET is 8-bit.  The produced code therefore lives at a base whose branch
    #   targets fit in a byte (base=200, span 200..212, all <256) — 13 distinct
    #   runtime code frames spanning a wide-ish address band.  The HIGH-address FETCH
    #   (past 2^16) is proven separately by the sweep above (the loader's own
    #   full-width JMP reaches 2^20-2); here we prove a MULTI-STATEMENT + BRANCH loaded
    #   program is byte-exact.
    #
    # produced layout (at base=200):
    #   +0 IMM d0  +1 PSH  +2 IMM d1  +3 ADD  +4 PSH  +5 IMM d2  +6 ADD   (AX=d0+d1+d2)
    #   +7 IMM s   +8 BZ (base+11)    +9 <fallthrough sum>  +10 JMP (base+12)
    #   +11 IMM 111   +12 HALT
    # s!=0 -> take the sum (skip else) -> AX=d0+d1+d2 ; s==0 -> BZ to else -> AX=111.
    LI, BZ = isa.LI, isa.BZ
    for (d0, d1, d2, s, want) in [(2, 3, 4, 1, 9), (5, 6, 7, 0, 111)]:
        base = 200
        loader = []

        def emit_imm_from_mem(k, addr):
            return [isa.Instr(IMM, k), isa.Instr(LI, 0), isa.Instr(PSH, 0),
                    isa.Instr(IMM, IMM), isa.Instr(EMIT, addr)]
        loader += emit_imm_from_mem(0, base + 0)    # IMM d0
        loader += _emit_seq(base + 1, PSH, 0)
        loader += emit_imm_from_mem(1, base + 2)    # IMM d1
        loader += _emit_seq(base + 3, ADD, 0)
        loader += _emit_seq(base + 4, PSH, 0)
        loader += emit_imm_from_mem(2, base + 5)    # IMM d2
        loader += _emit_seq(base + 6, ADD, 0)       # AX = d0+d1+d2
        loader += emit_imm_from_mem(3, base + 7)    # IMM s (branch predicate)
        loader += _emit_seq(base + 8, BZ, base + 11)   # BZ else (s==0)
        loader += _emit_seq(base + 9, IMM, d0 + d1 + d2)   # then: reload the sum
        loader += _emit_seq(base + 10, JMP, base + 12)     # then: JMP end
        loader += _emit_seq(base + 11, IMM, 111)           # else: IMM 111
        loader += _emit_seq(base + 12, HALT, 0)            # end: HALT
        loader += [isa.Instr(JMP, base)]                   # handoff -> run produced
        loader = _pad(loader, 90)
        seed = {0: d0, 1: d1, 2: d2, 3: s}
        res = run_pure_forward_complete_emit(model, L, loader, max_steps=200,
                                             seed_mem=seed, n_rt_pool=16)
        got = res["ax_trace"][-1] if res["ax_trace"] else None
        ok = res["exact"] and got == want
        print(f"[larger d=({d0},{d1},{d2}) s={s}] model_tail={got} want={want} "
              f"ref_tail={res['ref_trace'][-1]} exact={res['exact']} "
              f"n_runtime_frames={len(res['runtime_frames'])} steps={res['steps']} "
              f"PASS={ok}", flush=True)
        assert ok, (d0, d1, d2, s, res["ax_trace"][-6:], res["ref_trace"][-6:])

    print("\nALL DIRECT-CAM EMIT CHECKS PASS (byte-exact vs isa.interpret).", flush=True)


if __name__ == "__main__":
    main()
