"""The LEAN native forward of the compacted fused C4 VM (``qwen_lean_forward``).

Every test proves the lean RoPE + RMSNorm + softmax + SwiGLU forward (no
``transformers`` machinery) decodes BYTE-IDENTICAL to the HF ``Qwen2Model`` it
copied its weights from — and both match ``isa.interpret`` where the model is
correct.  These run on CPU (memory-light; the compacted model is 7-14 layers).
"""
from __future__ import annotations

import pytest

from c4_min import isa
from c4_min import qwen_full_vm as Q
from c4_min import qwen_lean_forward as LF


@pytest.fixture(scope="module")
def vm_base():
    return Q.build(code_size=24, subset=Q.SUBSET_BASE)


@pytest.fixture(scope="module")
def lean_base(vm_base):
    return LF.LeanQwenVM.from_full_vm(vm_base, device="cpu")


@pytest.fixture(scope="module")
def vm_memcmp():
    return Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)


@pytest.fixture(scope="module")
def lean_memcmp(vm_memcmp):
    return LF.LeanQwenVM.from_full_vm(vm_memcmp, device="cpu")


# ---------------------------------------------------------------------------
# The lean forward is a 7-14 layer, 6-CAM-head compacted model.
# ---------------------------------------------------------------------------
def test_lean_is_compacted_7_to_14_layers(lean_base, lean_memcmp):
    assert 7 <= lean_base.n_layers <= 14
    assert 7 <= lean_memcmp.n_layers <= 14
    # 5 register CAM heads + 1 memory head fit within the Qwen query-head budget.
    assert lean_base.n_heads <= 14
    assert lean_base.n_kv_heads == 2                       # GQA
    assert lean_base.head_dim == 64


# ---------------------------------------------------------------------------
# Byte-identity: lean forward == HF Qwen2Model, on the SAME weights.
# ---------------------------------------------------------------------------
def _same_bytes(vm, lean, prog):
    code = isa.assemble(prog)
    rh = Q.run_program(vm, code, max_steps=64)
    rl = LF.run_program_lean(lean, code, max_steps=64)
    return rl["ax_trace"] == rh["ax_trace"], rl, rh


@pytest.mark.parametrize("a,b", [(3, 4), (100, 27), (255, 1)])
def test_add_lean_eq_hf_and_exact(vm_base, lean_base, a, b):
    ok, rl, rh = _same_bytes(vm_base, lean_base,
                             [("IMM", a), ("PSH", 0), ("IMM", b), ("ADD", 0), ("HALT", 0)])
    assert ok, (rl["ax_trace"], rh["ax_trace"])
    assert rl["exact"], rl


@pytest.mark.parametrize("a,b", [(9, 4), (200, 55), (0, 1)])
def test_sub_lean_eq_hf_and_exact(vm_base, lean_base, a, b):
    ok, rl, rh = _same_bytes(vm_base, lean_base,
                             [("IMM", a), ("PSH", 0), ("IMM", b), ("SUB", 0), ("HALT", 0)])
    assert ok and rl["exact"], (rl, rh)


def test_branch_lean_eq_hf(vm_base, lean_base):
    for prog in (
        [("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)],
        [("IMM", 1), ("BNZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)],
        [("JMP", 2), ("IMM", 99), ("IMM", 5), ("HALT", 0)],
    ):
        ok, rl, rh = _same_bytes(vm_base, lean_base, prog)
        assert ok and rl["exact"], (prog, rl["ax_trace"], rh["ax_trace"])


def test_loop_lean_eq_hf(vm_base, lean_base):
    ok, rl, rh = _same_bytes(
        vm_base, lean_base,
        [("IMM", 3), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)])
    assert ok and rl["exact"], (rl["ax_trace"], rh["ax_trace"])


def test_memory_lean_eq_hf(vm_memcmp, lean_memcmp):
    for prog in (
        [("IMM", 5), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
         ("IMM", 5), ("LI", 0), ("HALT", 0)],                       # si/li
        [("IMM", 50), ("LI", 0), ("HALT", 0)],                      # zfod
        [("IMM", 30), ("PSH", 0), ("IMM", 1), ("SI", 0),
         ("IMM", 30), ("PSH", 0), ("IMM", 9), ("SI", 0),
         ("IMM", 30), ("LI", 0), ("HALT", 0)],                      # latest-write-wins
        [("IMM", 10), ("PSH", 0), ("IMM", 7), ("SI", 0),
         ("IMM", 10), ("LI", 0), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)],  # var
    ):
        ok, rl, rh = _same_bytes(vm_memcmp, lean_memcmp, prog)
        assert ok, (prog, rl["ax_trace"], rh["ax_trace"])


# #691 BUG 1: the ordering compares (LT/GT/LE/GE) degenerated to a constant before
# the cmp-finalize block was added to the compacted bake.  BOTH orderings of every
# op across the byte range must now be byte-exact vs isa.interpret AND lean == HF.
@pytest.mark.parametrize("op", ["EQ", "NE", "LT", "GT", "LE", "GE"])
@pytest.mark.parametrize("a,b", [
    (5, 5), (7, 9), (9, 7), (0, 255), (255, 0), (200, 100), (128, 127)])
def test_cmp_lean_eq_hf(vm_memcmp, lean_memcmp, op, a, b):
    ok, rl, rh = _same_bytes(vm_memcmp, lean_memcmp,
                             [("IMM", a), ("PSH", 0), ("IMM", b), (op, 0), ("HALT", 0)])
    assert ok, (op, a, b, rl["ax_trace"], rh["ax_trace"])   # lean == HF
    assert rl["exact"], (op, a, b, rl)                       # both == isa.interpret


# #691 BUG 2: a countdown from >= 64 exceeds the default 256-step oracle cap; the
# lean naive driver now runs the reference to its own step budget so the correct
# run-to-completion trace is not compared against a truncated golden.
@pytest.mark.parametrize("n", [63, 64, 100, 200, 255])
def test_countdown_over_256_steps_lean(vm_base, lean_base, n):
    code = isa.assemble([("IMM", n), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                         ("BNZ", 1), ("HALT", 0)])
    rl = LF.run_program_lean(lean_base, code, max_steps=n * 4 + 20)
    rh = Q.run_program(vm_base, code, max_steps=n * 4 + 20)
    assert rl["ax_trace"] == rh["ax_trace"]                  # lean == HF
    assert rl["exact"], (n, len(rl["ax_trace"]), len(rl["ref_trace"]))
    assert len(rl["ref_trace"]) == 4 * n + 2                 # full trace, not capped


# ---------------------------------------------------------------------------
# Perfect-draft speculation on the lean forward: byte-identical to the naive
# lean driver, with real forwards saved (deterministic VM = perfect draft).
# ---------------------------------------------------------------------------
def test_speculation_matches_naive_and_saves_forwards(lean_base):
    prog = isa.assemble(
        [("IMM", 20), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)])
    rn = LF.run_program_lean(lean_base, prog, max_steps=4096)
    rs = LF.speculative_run_lean(lean_base, prog, block_steps=32)
    assert rs.ax_trace == rn["ax_trace"]                 # byte-identical to naive
    assert rs.exact
    assert rs.forwards < rs.naive_forwards               # real forwards saved
    assert rs.speedup > 1.0


def test_speculation_memory_matches_naive(lean_memcmp):
    prog = isa.assemble(
        [("IMM", 10), ("PSH", 0), ("IMM", 7), ("SI", 0),
         ("IMM", 10), ("LI", 0), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)])
    rn = LF.run_program_lean(lean_memcmp, prog, max_steps=4096)
    rs = LF.speculative_run_lean(lean_memcmp, prog, block_steps=32)
    assert rs.ax_trace == rn["ax_trace"]
    assert rs.forwards < rs.naive_forwards


# ---------------------------------------------------------------------------
# The compute really runs in the lean SwiGLU MLPs (not a python copy): zeroing a
# lean layer's MLP annihilates the result.
# ---------------------------------------------------------------------------
def test_compute_is_in_the_lean_mlps(lean_base):
    import torch
    prog = isa.assemble([("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)])
    assert LF.run_program_lean(lean_base, prog)["exact"]
    saved = []
    with torch.no_grad():
        for layer in lean_base.layers:
            saved.append((layer.gate_w.clone(), layer.up_w.clone(), layer.down_w.clone()))
            layer.gate_w.zero_(); layer.up_w.zero_(); layer.down_w.zero_()
    got = LF.run_program_lean(lean_base, prog)
    with torch.no_grad():
        for layer, (g, u, d) in zip(lean_base.layers, saved):
            layer.gate_w.copy_(g); layer.up_w.copy_(u); layer.down_w.copy_(d)
    assert not got["exact"]                              # no MLP -> no compute
    assert LF.run_program_lean(lean_base, prog)["exact"]  # restored
