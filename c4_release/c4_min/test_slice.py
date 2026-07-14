"""Vertical-slice tests: the c4_min substrate compiles + runs + decodes 8-bit
programs whose emitted AX trace matches the reference interpreter, argmax-correct
through the LM head. Run: PYTHONPATH=<repo> python -m pytest c4_min/test_slice.py
(or execute directly: python c4_min/test_slice.py)."""
from __future__ import annotations

from c4_min import compiler, isa


def _check(prog):
    """Compile + run ``prog``; assert model AX-trace == reference interpreter."""
    code = isa.assemble(prog)
    ref = isa.interpret(code)
    model, L, code = compiler.compile_program(prog)
    got = compiler.run(model, L, code)
    assert got == ref, f"{prog}: model {got} != ref {ref}"
    return got, ref


def test_imm_add_halt():
    """The mission slice: IMM 5; PSH; IMM 3; ADD; HALT -> AX == 8."""
    got, ref = _check([("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)])
    assert got[-1] == 8
    assert ref[-1] == 8


def test_sub():
    """10 - 4 == 6."""
    got, _ = _check([("IMM", 10), ("PSH", 0), ("IMM", 4), ("SUB", 0), ("HALT", 0)])
    assert got[-1] == 6


def test_sub_underflow_wraps_mod_256():
    """7 - 9 == 254 (8-bit two's-complement underflow via the mod-256 fold)."""
    got, _ = _check([("IMM", 7), ("PSH", 0), ("IMM", 9), ("SUB", 0), ("HALT", 0)])
    assert got[-1] == (7 - 9) & 0xFF == 254


def test_add_wraps_mod_256():
    """200 + 100 == 44 (8-bit wrap)."""
    got, _ = _check([("IMM", 200), ("PSH", 0), ("IMM", 100), ("ADD", 0), ("HALT", 0)])
    assert got[-1] == (200 + 100) & 0xFF


def test_imm_only():
    """A lone immediate load decodes to itself."""
    got, _ = _check([("IMM", 42), ("HALT", 0)])
    assert got[0] == 42


def test_halt_terminator_fires():
    """The HALT terminator token is emitted by the LM head."""
    prog = [("IMM", 7), ("HALT", 0)]
    model, L, code = compiler.compile_program(prog)
    assert compiler.halted(model, L, code) is True


def test_jmp_forward_skips():
    """JMP 2 skips IMM 99 and lands on IMM 5 -> exit 5, 3 executed steps."""
    got, ref = _check([("JMP", 2), ("IMM", 99), ("IMM", 5), ("HALT", 0)])
    assert got[-1] == 5
    assert len(got) == 3  # JMP; IMM 5; HALT (IMM 99 skipped)


def test_bz_taken_and_nottaken():
    """BZ branches iff AX==0; both paths decode correctly + at the right length."""
    # AX==0 -> branch to idx 3 (skip IMM 99), exit 7.
    taken, _ = _check([("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)])
    assert taken[-1] == 7
    # AX!=0 -> fall through to IMM 42, exit 42.
    nottaken, _ = _check([("IMM", 1), ("BZ", 3), ("IMM", 42), ("HALT", 0)])
    assert nottaken[-1] == 42


def test_bnz_taken_and_nottaken():
    """BNZ branches iff AX!=0 (mirror of BZ)."""
    taken, _ = _check([("IMM", 1), ("BNZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)])
    assert taken[-1] == 7
    nottaken, _ = _check([("IMM", 0), ("BNZ", 3), ("IMM", 42), ("HALT", 0)])
    assert nottaken[-1] == 42


def test_if_then_else_computed_predicate():
    """if a==b then X else Y, with the branch predicate COMPUTED at runtime (SUB).

    Proves depth=time control flow: the instruction executed after BZ depends on
    the runtime a-b, and the not-taken branch's instructions are correctly
    skipped. THEN-path and ELSE-path decode byte-exact vs the reference.
    """
    def ite(a, b, X, Y):
        return [("IMM", a), ("PSH", 0), ("IMM", b), ("SUB", 0),
                ("BZ", 7), ("IMM", Y), ("JMP", 8), ("IMM", X), ("HALT", 0)]

    for a, b, X, Y in [(5, 5, 111, 222), (5, 7, 111, 222),
                       (9, 9, 42, 99), (3, 8, 42, 99)]:
        got, ref = _check(ite(a, b, X, Y))
        assert got[-1] == (X if a == b else Y), (a, b, got)


def test_carry_forward_attention():
    """carry_forward_head copies a band from position t-1 to t (t=0 keeps self)."""
    import torch
    from c4_min.compile_attn import carry_forward_head
    from c4_min.model import Attn

    dim, n_heads, S = 8, 4, 5
    one, ax = 0, 1
    w = carry_forward_head(dim, n_heads, S, one, [ax])
    attn = Attn(dim, n_heads)
    with torch.no_grad():
        attn.W_q.copy_(w["W_q"]); attn.W_k.copy_(w["W_k"])
        attn.W_v.copy_(w["W_v"]); attn.W_o.copy_(w["W_o"]); attn.mask = w["mask"]
    x = torch.zeros(1, S, dim); x[0, :, one] = 1.0
    x[0, :, ax] = torch.tensor([10., 20., 30., 40., 50.])
    y = attn(x)
    delta = (y[0, :, ax] - x[0, :, ax]).round().tolist()
    assert delta == [10.0, 10.0, 20.0, 30.0, 40.0]  # t gets t-1's AX; t=0 keeps self


def test_content_match_attention():
    """content_match_head selects the position whose key == q_offset and copies v."""
    import torch
    from c4_min.compile_attn import content_match_head
    from c4_min.dsl import AttentionSpec
    from c4_min.model import Attn

    dim, n_heads, S = 8, 4, 6
    one, keyb, ksqb, valb, dst, zero = 0, 1, 2, 3, 4, 5
    spec = AttentionSpec(q_band=zero, k_band=keyb, v_band=valb, dst_band=dst,
                         gain=5.0, q_offset=3.0)
    w = content_match_head(spec, ksqb, dim, n_heads, S, one)
    attn = Attn(dim, n_heads)
    with torch.no_grad():
        attn.W_q.copy_(w["W_q"]); attn.W_k.copy_(w["W_k"])
        attn.W_v.copy_(w["W_v"]); attn.W_o.copy_(w["W_o"]); attn.mask = w["mask"]
    x = torch.zeros(1, S, dim); x[0, :, one] = 1.0
    x[0, :, keyb] = torch.arange(S).float()
    x[0, :, ksqb] = torch.arange(S).float() ** 2
    x[0, :, valb] = torch.tensor([100., 101., 102., 103., 104., 105.])
    y = attn(x)
    copied = (y[0, :, dst] - x[0, :, dst]).round().tolist()
    assert copied == [103.0] * S  # every query selected key==3 -> value 103


if __name__ == "__main__":
    import traceback

    tests = [
        test_imm_add_halt, test_sub, test_sub_underflow_wraps_mod_256,
        test_add_wraps_mod_256,
        test_imm_only, test_halt_terminator_fires,
        test_jmp_forward_skips, test_bz_taken_and_nottaken,
        test_bnz_taken_and_nottaken, test_if_then_else_computed_predicate,
        test_carry_forward_attention, test_content_match_attention,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} slice tests passed")
