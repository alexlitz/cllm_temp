"""Depth-N operand-stack byte-exactness for the lean CFM drivers (doom wall #3).

The stock lean drivers mirrored the operand stack with a SINGLE ``STACK0``
register (the #702 1-slot wall), so a depth-2+ stack expression (``a*b + c*d``,
``mem[a] = x + y``, nested call args) diverged (``10 + (3+4)`` -> 10 not 17).
The fix keeps a REAL depth-N Python data stack driver-side (``qwen_lean_stack``),
wired into every lean driver behind a ``stack_depth=True`` default that is
byte-IDENTICAL to the legacy 1-slot path for depth<=1.

CPU tests cover the pure-Python draft (the perfect-draft transition the spec
drivers verify); GPU tests confirm byte-exactness through the actual neural
forward on a fast SUBSET_MEM_CMP CFM model (ADD/SUB-only depth expressions —
byte-exact WITHOUT the divmod build; the real MUL ``a*b+c*d`` form is covered by
the recurrent-model agent scripts, whose MUL op needs SUBSET_MULDIV recurrent).
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# CPU: the pure-Python perfect-draft transition (no model, no GPU).
# ---------------------------------------------------------------------------
def _asm(prog):
    from c4_min import isa
    return isa.assemble(prog)


# (name, program, expected last AX) — depth-2/3/4 stack expressions.
_DEPTH_PROGS = [
    ("d2_10+(3+4)=17", [
        ("IMM", 10), ("PSH", 0),
        ("IMM", 3), ("PSH", 0), ("IMM", 4), ("ADD", 0), ("ADD", 0), ("HALT", 0)], 17),
    ("d2_doomform_20+(6+8)=34", [
        ("IMM", 20), ("PSH", 0),
        ("IMM", 6), ("PSH", 0), ("IMM", 8), ("ADD", 0), ("ADD", 0), ("HALT", 0)], 34),
    ("d3_triple_park=26", [
        ("IMM", 5), ("PSH", 0), ("IMM", 6), ("PSH", 0), ("IMM", 7), ("PSH", 0),
        ("IMM", 8), ("ADD", 0), ("ADD", 0), ("ADD", 0), ("HALT", 0)], 26),
    ("d4_1+(2+(3+4))=10", [
        ("IMM", 1), ("PSH", 0), ("IMM", 2), ("PSH", 0),
        ("IMM", 3), ("PSH", 0), ("IMM", 4), ("ADD", 0), ("ADD", 0), ("ADD", 0),
        ("HALT", 0)], 10),
    ("d2_store_computed_mem[40]=(3+4)=7", [
        ("IMM", 0x40), ("PSH", 0),
        ("IMM", 3), ("PSH", 0), ("IMM", 4), ("ADD", 0), ("SI", 0),
        ("IMM", 0x40), ("LI", 0), ("HALT", 0)], 7),
]


@pytest.mark.parametrize("name,prog,want", _DEPTH_PROGS)
def test_stackfn_draft_matches_interpret(name, prog, want):
    """The unified real-stack + function draft reproduces isa.interpret byte-for-byte
    on a depth-2/3/4 expression (the transition the spec drivers verify)."""
    from c4_min import isa
    from c4_min import qwen_full_vm as Q
    from c4_min.qwen_lean_stack import draft_program_lean_stackfn
    code = _asm(prog)
    ref = isa.interpret(code, max_steps=200)
    d = draft_program_lean_stackfn(code, Q.SUBSET_MEM_CMP, max_steps=200)
    # the draft executes the SAME transition the spec drivers verify; its reference
    # trace matches isa.interpret, and it ran to completion (one draft step per op).
    assert d.ref_trace == ref
    assert d.ref_trace[-1] == want, (name, d.ref_trace[-1])
    assert d.halted and len(d.steps) == len(ref)


def test_stackfn_draft_functions_and_deep_stack():
    """The draft handles JSR/ENT/LEV AND a deep stack in the same program (doom is
    both function-heavy and deep-stacked) — the union the two prior drafts lacked."""
    from c4_min import isa
    from c4_min import qwen_full_vm as Q
    from c4_min.qwen_lean_forward import interpret_with_functions
    from c4_min.qwen_lean_stack import draft_program_lean_stackfn
    prog = [
        ("JSR", 6), ("PSH", 0), ("JSR", 6), ("ADD", 0), ("HALT", 0), ("NOP", 0),
        ("ENT", 0), ("IMM", 3), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("LEV", 0),
    ]
    code = _asm(prog)
    ref = interpret_with_functions(code, max_steps=200)
    d = draft_program_lean_stackfn(code, Q.SUBSET_MEM_CMP, max_steps=200,
                                   ref_trace=ref)
    assert d.steps, "draft must NOT fall back on a function program"
    assert d.ref_trace == ref


# ---------------------------------------------------------------------------
# GPU: byte-exact through the ACTUAL lean neural forward (fast SUBSET_MEM_CMP).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lean_memcmp():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs cuda")
    os.environ.setdefault("C4_VM_CACHE_DIR", "/tmp/c4cache_agent")
    from c4_min import qwen_full_vm as Q
    from c4_min import qwen_lean_forward as LF
    dev = torch.device("cuda:0")
    vm = Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)
    vm.embed = vm.embed.to(dev)
    return LF.LeanQwenVM.from_full_vm(vm, device=dev)


@pytest.mark.parametrize("name,prog,want", _DEPTH_PROGS)
def test_depth_stack_naive_byte_exact(lean_memcmp, name, prog, want):
    """The FIXED (stack_depth=True default) naive lean driver is byte-exact on a
    depth-2/3/4 expression, and the legacy 1-slot path is NOT (proves the wall)."""
    from c4_min import qwen_lean_forward as LF
    code = _asm(prog)
    r = LF.run_program_lean(lean_memcmp, code, max_steps=200)
    assert r["exact"], (name, r["ax_trace"], r["ref_trace"])
    assert r["ax_trace"][-1] == want
    legacy = LF.run_program_lean(lean_memcmp, code, max_steps=200, stack_depth=False)
    assert not legacy["exact"], f"{name}: 1-slot wall should still be present legacy-off"


@pytest.mark.parametrize("name,prog,want", _DEPTH_PROGS)
def test_depth_stack_spec_byte_exact(lean_memcmp, name, prog, want):
    """The FIXED big-K speculative lean driver is byte-exact on the depth expressions
    AND still batches (forwards << steps)."""
    from c4_min import qwen_lean_forward as LF
    code = _asm(prog)
    r = LF.speculative_run_lean(lean_memcmp, code, block_steps=32, max_steps=200)
    assert r.exact and r.status == "PASS", (name, r.ax_trace, r.ref_trace)
    assert r.ax_trace[-1] == want
    assert r.forwards <= r.naive_forwards


@pytest.mark.parametrize("name,prog,want", _DEPTH_PROGS)
def test_depth_stack_evict_byte_exact(lean_memcmp, name, prog, want):
    """The bounded-KV eviction driver is byte-exact on the depth expressions."""
    from c4_min import qwen_lean_evict as EV
    code = _asm(prog)
    ev = EV.run_program_lean_evict(lean_memcmp, code, max_steps=200, evict="off")
    assert ev.exact, (name, ev.ax_trace, ev.ref_trace)
    assert ev.ax_trace[-1] == want


def test_depth1_regression_byte_identical(lean_memcmp):
    """stack_depth=True must give the SAME trace as the legacy 1-slot path for a
    depth-1 program (byte-identity preserved)."""
    from c4_min import qwen_lean_forward as LF
    prog = [("IMM", 100), ("PSH", 0), ("IMM", 27), ("ADD", 0), ("HALT", 0)]
    code = _asm(prog)
    fixed = LF.run_program_lean(lean_memcmp, code, max_steps=200)
    legacy = LF.run_program_lean(lean_memcmp, code, max_steps=200, stack_depth=False)
    assert fixed["ax_trace"] == legacy["ax_trace"]
    assert fixed["exact"]
