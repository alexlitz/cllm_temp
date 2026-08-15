"""#916 END-TO-END BATTERY — run real programs through the assembled sparse
full-ISA inline+FF model (base + inline fp64 log-sink divmod) BYTE-EXACT vs
``isa.interpret``.  Reports program N/N, step N/N, and the divmod-boundary cases.

Run:  PYTHONPATH=<c4_release> OMP_NUM_THREADS=2 C4_LOGSINK_DIV=1 \
      python -m c4_min._e2e_battery_916
"""
from __future__ import annotations

import os

os.environ.setdefault("C4_LOGSINK_DIV", "1")

from . import _e2e_inline_bake_916 as E
from . import isa


def _div(a, b):
    return [("IMM", a), ("PSH", 0), ("IMM", b), ("DIV", 0)]


def _mod(a, b):
    return [("IMM", a), ("PSH", 0), ("IMM", b), ("MOD", 0)]


def divmod_programs():
    """DIV/MOD boundary cases the task requires, within the 8-bit reference range
    (isa.interpret masks values to 0xFF): b=1, b=2^k, b≈256, large quotient."""
    progs = {}
    # b = 1 (divide by one — quotient == dividend, rem 0).
    progs["div_b1"] = _div(200, 1)
    progs["mod_b1"] = _mod(200, 1)
    # b = 2^k (power-of-two divisors).
    for k, b in [(1, 2), (2, 4), (3, 8), (4, 16), (5, 32), (6, 64), (7, 128)]:
        progs[f"div_b2^{k}"] = _div(201, b)
        progs[f"mod_b2^{k}"] = _mod(201, b)
    # b close to 256.
    for b in (200, 250, 255):
        progs[f"div_bnear256_{b}"] = _div(255, b)
        progs[f"mod_bnear256_{b}"] = _mod(255, b)
    # large quotient (small divisor, big dividend -> q near 255).
    progs["div_bigq_255_1"] = _div(255, 1)     # q=255
    progs["div_bigq_254_2"] = _div(254, 2)     # q=127
    progs["div_bigq_255_3"] = _div(255, 3)     # q=85
    # rem non-zero + exact-divide mix.
    progs["div_100_7"] = _div(100, 7)
    progs["mod_100_7"] = _mod(100, 7)
    progs["div_144_12"] = _div(144, 12)
    progs["mod_143_12"] = _mod(143, 12)
    # b==0 guard (isa: b==0 -> (0,0)).
    progs["div_b0"] = _div(42, 0)
    progs["mod_b0"] = _mod(42, 0)
    return progs


def multiop_programs():
    """Multi-op branching + memory + cmp programs (loops via branches, LI/SI, CMP)."""
    progs = {}
    # branch taken / not-taken (BZ/BNZ read AX_ZERO).
    progs["bz_taken"] = [("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)]
    progs["bnz_taken"] = [("IMM", 5), ("BNZ", 3), ("IMM", 99), ("IMM", 8), ("HALT", 0)]
    # countdown LOOP via branch (SUB drives AX->0, BNZ tests it each iteration).
    progs["countdown5"] = [("IMM", 5), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                           ("BNZ", 1), ("HALT", 0)]
    progs["countdown12"] = [("IMM", 12), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                            ("BNZ", 1), ("HALT", 0)]
    # arithmetic straightline (ADD/SUB) then branch-not-taken.
    progs["mixed_arith"] = [("IMM", 3), ("PSH", 0), ("IMM", 4), ("ADD", 0),
                            ("BNZ", 6), ("IMM", 0), ("HALT", 0)]
    # LI/SI memory round-trip: store AX at addr, load it back.
    progs["si_li"] = [("IMM", 77), ("PSH", 0), ("IMM", 5), ("SI", 0),
                      ("IMM", 5), ("LI", 0), ("HALT", 0)]
    # CMP: EQ / LT / GT verdicts feeding a branch.
    progs["cmp_eq"] = [("IMM", 4), ("PSH", 0), ("IMM", 4), ("EQ", 0), ("HALT", 0)]
    progs["cmp_lt"] = [("IMM", 3), ("PSH", 0), ("IMM", 9), ("LT", 0), ("HALT", 0)]
    progs["cmp_gt"] = [("IMM", 9), ("PSH", 0), ("IMM", 3), ("GT", 0), ("HALT", 0)]
    # a loop that USES divide each iteration then branches (divmod inside control flow).
    progs["div_then_branch"] = [("IMM", 100), ("PSH", 0), ("IMM", 7), ("DIV", 0),
                                ("BNZ", 6), ("IMM", 0), ("HALT", 0)]
    # multiply + divide sequence (MUL then DIV, both nibble-band ALU ops).
    progs["mul_then_div"] = [("IMM", 6), ("PSH", 0), ("IMM", 7), ("MUL", 0),
                             ("PSH", 0), ("IMM", 3), ("DIV", 0), ("HALT", 0)]
    return progs


def _run_one_op_32bit(vm, op_name, a, b):
    """Seed a 32-bit register frame (STACK0=a, AX=b) and run ONE op step through the
    assembled forward, decoding the FULL 32-bit result from the AX nibble band.  This
    exercises the inline divmod across ALL 8 nibbles (large quotients, high bytes) —
    NOT reachable through the 8-bit IMM reference.  Compares vs nibble_muldivmod."""
    import torch
    from . import qwen_full_vm as Q
    from .nibble_pure_forward_complete import _decode_reg_from_nibbles
    QL, L = vm.QL, vm.QL.L
    op = getattr(isa, op_name)
    # ONE-instruction program: the op at PC=0. STACK0 & AX seeded as the operands.
    code = isa.assemble([(op_name, 0)])
    reg_state = {"PC": 0, "AX": b & 0xFFFFFFFF, "SP": Q.SP_INIT,
                 "BP": Q.SP_INIT, "STACK0": a & 0xFFFFFFFF}
    x = Q._build_stream_and_overlay(vm, code, reg_state, [], None)
    state = vm.forward(x, q_positions=None)[0, -1]
    got = _decode_reg_from_nibbles(state, L, L.AX) & 0xFFFFFFFF
    return got


def wide_inrange_cases():
    """WIDE (>8-bit) divmod cases INSIDE the assembled build's proven byte-exact
    range (dividend a <= 2^16, exercising nibbles 0..3, i.e. bytes 0-1 of the 32-bit
    datapath).  Vs nibble_muldivmod32.  These prove the inline divmod runs across
    MULTIPLE nibble positions (not just the low byte) through the actual assembled
    forward.  (The build's byte-exact envelope is dividend <= ~2^16; beyond that the
    softmax-sink reciprocal residue x dividend exceeds the +/-1 correction band.)"""
    return [
        ("DIV", 1788, 7, "1788/7 (q=255)"),
        ("DIV", 60000, 7, "60000/7 (16-bit dividend, q~8571)"),
        ("DIV", 40963, 256, "40963/256 (b>8-bit, q=160)"),
        ("DIV", 2563, 100, "2563/100 (q=25)"),
        ("DIV", 65535, 255, "65535/255 (q=257)"),
        ("DIV", 65535, 2, "65535/2 (q=32767)"),
        ("MOD", 60000, 7, "60000 %% 7"),
        ("MOD", 40963, 256, "40963 %% 256"),
        ("MOD", 65535, 255, "65535 %% 255"),
        ("MOD", 65535, 2, "65535 %% 2"),
    ]


def out_of_range_cases():
    """32-bit cases OUTSIDE the assembled build's byte-exact range (documented
    limit — the softmax-sink reciprocal residue x dividend exceeds the ±1 correction
    band once the divisor has high nibbles or the quotient exceeds ~2^16)."""
    M = 0xFFFFFFFF
    return [
        ("DIV", M, 3, "max/3 (q~1.4e9)"),
        ("DIV", 0xDEADBEEF, 0x1234, "0xDEADBEEF/0x1234 (large divisor)"),
        ("DIV", M, 0xFFFF, "max/65535 (large divisor)"),
        ("DIV", 700003, 7, "q~100000 (>2^16)"),
    ]


def run_wide_divmod(vm):
    from . import nibble_muldivmod as NM
    print("\n" + "-" * 90)
    print("WIDE (>8-bit) inline divmod — IN the assembled build's proven range")
    print("(operands seeded direct at 32-bit, vs nibble_muldivmod32)")
    print("-" * 90)
    ok = tot = 0
    for op_name, a, b, desc in wide_inrange_cases():
        got = _run_one_op_32bit(vm, op_name, a, b)
        want = NM.div32(a, b) if op_name == "DIV" else NM.mod32(a, b)
        tot += 1
        good = (got == want)
        ok += int(good)
        tag = "PASS" if good else "FAIL"
        extra = "" if good else f"  got={got} want={want}"
        print(f"  {desc:<38} [{tag}]{extra}")
    print(f"  wide in-range divmod byte-exact: {ok}/{tot}")

    # RANDOM sweep inside the CLEAN envelope (dividend <= 50000, divisor <= 2000).
    import random
    rng = random.Random(20260815)
    r_ok = r_tot = 0
    for _ in range(200):
        a = rng.randint(0, 50000)
        b = rng.randint(1, 2000)
        dg = _run_one_op_32bit(vm, "DIV", a, b)
        mg = _run_one_op_32bit(vm, "MOD", a, b)
        r_tot += 2
        r_ok += int(dg == NM.div32(a, b)) + int(mg == NM.mod32(a, b))
    print(f"  RANDOM wide in-range (200 pairs, a<=50000, b<=2000): {r_ok}/{r_tot} byte-exact")

    print("\n" + "-" * 90)
    print("OUT-OF-RANGE 32-bit cases (documented assembled-build limit, NOT byte-exact)")
    print("-" * 90)
    oor_ok = oor_tot = 0
    for op_name, a, b, desc in out_of_range_cases():
        got = _run_one_op_32bit(vm, op_name, a, b)
        want = NM.div32(a, b) if op_name == "DIV" else NM.mod32(a, b)
        oor_tot += 1
        good = (got == want)
        oor_ok += int(good)
        print(f"  {desc:<38} got={got} want={want}  "
              f"[{'exact' if good else 'DIVERGES (out of range)'}]")
    print(f"  out-of-range exact: {oor_ok}/{oor_tot} (expected to diverge)")
    return ok, tot, oor_ok, oor_tot, r_ok, r_tot


def main():
    E._rss_watchdog(4000)
    print("=" * 90)
    print("#916 END-TO-END: assembled sparse full-ISA inline+FF model (base + inline")
    print("fp64 log-sink divmod), run real programs BYTE-EXACT vs isa.interpret")
    print("=" * 90)
    vm, info = E.build_sparse("full")
    print(f"\nASSEMBLED MODEL: {info['n_layers']} physical layers "
          f"(of which {info['n_divmod_blocks']} inline divmod), "
          f"hidden {info['hidden']}, inter {info['intermediate']}, "
          f"dtype {info['dtype']}")
    print(f"  attention CAM layers: {info['attn_layers']}  "
          f"(register / code-fetch / mem / recip-sink)")
    print(f"  D_used {info['D_used']}  build peak RSS {info['peak_rss_mb']} MB")
    print(f"  NO subroutine, NO loop (divmod UNROLLED, identity apply order)")

    all_prog_ok = 0
    all_prog_tot = 0
    all_step_ok = 0
    all_step_tot = 0
    divmod_ok = 0
    divmod_tot = 0

    for title, progs, is_divmod in (("DIVMOD boundary cases (b=1, b=2^k, b~256, "
                                     "large q, b=0 guard)", divmod_programs(), True),
                                    ("Multi-op branching / memory / cmp programs",
                                     multiop_programs(), False)):
        print("\n" + "-" * 90)
        print(title)
        print("-" * 90)
        for name, prog in progs.items():
            code = isa.assemble(prog)
            r = E.run_program(vm, code, max_steps=120)
            ok = r["exact"]
            all_prog_tot += 1
            all_prog_ok += int(ok)
            all_step_tot += r["n_cmp"]
            all_step_ok += (r["n_cmp"] if ok else
                            sum(1 for i in range(r["n_cmp"])
                                if r["ax_trace"][i] == r["ref_trace"][i]))
            if is_divmod:
                divmod_tot += 1
                divmod_ok += int(ok)
            tag = "PASS" if ok else "FAIL"
            extra = "" if ok else f"  ax={r['ax_trace']} ref={r['ref_trace']}"
            print(f"  {name:<22} steps={r['n_cmp']:>3}  [{tag}]{extra}")

    w_ok, w_tot, oor_ok, oor_tot, r_ok, r_tot = run_wide_divmod(vm)

    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    print(f"  8-bit programs byte-exact : {all_prog_ok}/{all_prog_tot}")
    print(f"  8-bit steps byte-exact    : {all_step_ok}/{all_step_tot}")
    print(f"  8-bit divmod cases        : {divmod_ok}/{divmod_tot}")
    print(f"  wide (>8-bit) in-range    : {w_ok}/{w_tot} + random {r_ok}/{r_tot}")
    print(f"  out-of-range 32-bit       : {oor_ok}/{oor_tot} (documented limit: a>~2^16)")
    print(f"  peak RSS                  : {E._rss_mb()} MB")
    print(f"  assembled layers          : {info['n_layers']} (hidden {info['hidden']}, "
          f"inter {info['intermediate']}, {info['n_divmod_blocks']} inline divmod)")
    return (all_prog_ok == all_prog_tot) and (w_ok == w_tot) and (r_ok == r_tot)


if __name__ == "__main__":
    ok = main()
    print(f"\nALL BYTE-EXACT: {ok}")
