import torch
from c4_min import isa
from c4_min.nibble_unified import build_unified_model
from c4_min.nibble_vm import load_program, _emit_and_reembed, _snap_lane


def make_runner(model, L):
    def run(prog, max_steps=64):
        code = isa.assemble(prog)
        state = load_program(model, L, code)
        last = 0
        for _ in range(max_steps):
            x = state.view(1, 1, -1)
            for blk in model.blocks:
                x = blk(x)
            out = x[0, 0]
            halted = float(out[L.HALTED]) > 0.5
            last = _snap_lane(out[L.AX_VAL]) & 0xFF
            state = _emit_and_reembed(out, L)
            if halted:
                break
        return last
    return run


if __name__ == "__main__":
    import sys
    with_mdm = "--mdm" in sys.argv
    with_bw = "--bw" in sys.argv
    model, L, meta = build_unified_model(code_size=16, include_mdm_table=with_mdm,
                                         include_bitwise=with_bw)
    run = make_runner(model, L)
    print("dim", meta["dim"], "blocks", meta["n_blocks"], "mdm", with_mdm, "bw", with_bw)

    cases = [("EQ", 5, 5, 1), ("EQ", 5, 6, 0), ("NE", 5, 6, 1), ("NE", 5, 5, 0),
             ("LT", 3, 5, 1), ("LT", 5, 3, 0), ("GT", 5, 3, 1), ("GT", 3, 5, 0),
             ("LE", 5, 5, 1), ("LE", 6, 5, 0), ("GE", 5, 5, 1), ("GE", 3, 5, 0)]
    allok = True
    for name, a, b, exp in cases:
        got = run([("IMM", a), ("PSH", 0), ("IMM", b), (name, 0), ("HALT", 0)])
        ok = (got == exp)
        allok = allok and ok
        print(f"  {name} {a},{b} -> {got} (exp {exp}) {'OK' if ok else 'FAIL'}")
    print("CMP all ok:", allok)

    if with_mdm:
        mdm = [("MUL", 6, 7, 42), ("MUL", 12, 12, 144), ("DIV", 20, 3, 6),
               ("DIV", 20, 0, 0), ("MOD", 20, 3, 2), ("MOD", 17, 5, 2)]
        mok = True
        for name, a, b, exp in mdm:
            got = run([("IMM", a), ("PSH", 0), ("IMM", b), (name, 0), ("HALT", 0)])
            ok = (got == (exp & 0xFF))
            mok = mok and ok
            print(f"  {name} {a},{b} -> {got} (exp {exp & 0xFF}) {'OK' if ok else 'FAIL'}")
        print("MDM all ok:", mok)

    if with_bw:
        bw = [("OR", 0x0C, 0x03, 0x0F), ("XOR", 0xFF, 0x0F, 0xF0),
              ("AND", 0xF0, 0x3C, 0x30), ("SHL", 0x03, 2, 0x0C),
              ("SHR", 0xF0, 4, 0x0F)]
        bok = True
        for name, a, b, exp in bw:
            got = run([("IMM", a), ("PSH", 0), ("IMM", b), (name, 0), ("HALT", 0)])
            ok = (got == (exp & 0xFF))
            bok = bok and ok
            print(f"  {name} {a:#x},{b} -> {got:#x} (exp {exp & 0xFF:#x}) {'OK' if ok else 'FAIL'}")
        print("BITWISE all ok:", bok)
