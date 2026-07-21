"""CUDA-graph capture of the lean forward (``qwen_lean_cuda_graph``).

Every test proves the CUDA-graph REPLAY of ``LeanQwenVM.forward`` decodes
BYTE-IDENTICAL to the eager lean forward (and thus to the HF ``Qwen2Model`` and to
``isa.interpret`` where the model is correct), for BOTH the naive one-forward-per-step
driver and the batched speculative driver.  Graph capture needs CUDA, so these are
GPU-guarded (skip cleanly on CPU-only hosts).  The compacted model is 7-14 layers,
~120-200 MB — memory-light on the tiny bench GPU.
"""
from __future__ import annotations

import pytest
import torch

from c4_min import isa
from c4_min import qwen_full_vm as Q
from c4_min import qwen_lean_forward as LF
from c4_min import qwen_lean_cuda_graph as CG


DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 \
    else ("cuda:0" if torch.cuda.is_available() else None)

requires_cuda = pytest.mark.skipif(DEVICE is None,
                                   reason="CUDA graph capture requires a GPU")


# ---------------------------------------------------------------------------
# Module-scoped builds (one bake per subset, shared across tests).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lean_base():
    if DEVICE is None:
        pytest.skip("no GPU")
    vm = Q.build(code_size=24, subset=Q.SUBSET_BASE)
    vm.embed = vm.embed.to(DEVICE)
    return LF.LeanQwenVM.from_full_vm(vm, device=DEVICE)


@pytest.fixture(scope="module")
def lean_memcmp():
    if DEVICE is None:
        pytest.skip("no GPU")
    vm = Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)
    vm.embed = vm.embed.to(DEVICE)
    return LF.LeanQwenVM.from_full_vm(vm, device=DEVICE)


# ---------------------------------------------------------------------------
# The single-forward graph replay is byte-identical to the eager forward.
# ---------------------------------------------------------------------------
@requires_cuda
def test_graph_replay_bit_identical_to_eager(lean_memcmp):
    lean = lean_memcmp
    g = CG.GraphedLeanForward(lean)
    reg_state = {"PC": 1, "AX": 3, "SP": LF.SP_INIT, "BP": LF.SP_INIT, "STACK0": 0}
    store_log = [{"addr": a, "val": a & 0xFF} for a in range(4)]
    code = isa.assemble([("IMM", 3), ("PSH", 0), ("IMM", 4), ("ADD", 0), ("HALT", 0)])
    x, pos = LF._build_stream_and_overlay(lean, code, reg_state, store_log, None)
    with torch.no_grad():
        he, _ = lean.forward(x, past=None, q_positions=pos)
    hg = g(x, pos)
    assert torch.equal(he, hg)                       # BYTE-identical
    # 2nd call of the same shape REPLAYS (does not recapture).
    hg2 = g(x, pos)
    assert torch.equal(he, hg2)
    assert g.n_capture == 1 and g.n_replay == 2


@requires_cuda
def test_graph_buckets_by_shape(lean_memcmp):
    """A new (B, S) shape captures a fresh graph; a seen shape replays."""
    lean = lean_memcmp
    g = CG.GraphedLeanForward(lean)
    code = isa.assemble([("IMM", 3), ("PSH", 0), ("IMM", 4), ("ADD", 0), ("HALT", 0)])
    for n_store in (0, 2, 2, 5):
        reg_state = {"PC": 1, "AX": 3, "SP": LF.SP_INIT, "BP": LF.SP_INIT, "STACK0": 0}
        store_log = [{"addr": a, "val": a & 0xFF} for a in range(n_store)]
        x, pos = LF._build_stream_and_overlay(lean, code, reg_state, store_log, None)
        with torch.no_grad():
            he, _ = lean.forward(x, past=None, q_positions=pos)
        assert torch.equal(he, g(x, pos))
    # 3 distinct S (n_store 0/2/5) -> 3 captures, 4 replays.
    assert g.n_capture == 3
    assert g.n_replay == 4


# ---------------------------------------------------------------------------
# The NAIVE graphed driver == the eager naive driver, byte-for-byte.
# ---------------------------------------------------------------------------
def _naive_matches(lean, prog):
    code = isa.assemble(prog)
    eager = LF.run_program_lean(lean, code, max_steps=4096)
    graphed = CG.run_program_lean_graphed(lean, code, max_steps=4096)
    return eager["ax_trace"] == graphed["ax_trace"], eager, graphed


@requires_cuda
@pytest.mark.parametrize("a,b", [(3, 4), (100, 27), (255, 1)])
def test_naive_graphed_add(lean_base, a, b):
    ok, e, g = _naive_matches(
        lean_base, [("IMM", a), ("PSH", 0), ("IMM", b), ("ADD", 0), ("HALT", 0)])
    assert ok, (e["ax_trace"], g["ax_trace"])


@requires_cuda
def test_naive_graphed_branch_and_loop(lean_base):
    for prog in (
        [("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)],
        [("IMM", 1), ("BNZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)],
        [("JMP", 2), ("IMM", 99), ("IMM", 5), ("HALT", 0)],
        [("IMM", 20), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)],
    ):
        ok, e, g = _naive_matches(lean_base, prog)
        assert ok, (prog, e["ax_trace"], g["ax_trace"])


@requires_cuda
def test_naive_graphed_memory(lean_memcmp):
    for prog in (
        [("IMM", 5), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
         ("IMM", 5), ("LI", 0), ("HALT", 0)],
        [("IMM", 50), ("LI", 0), ("HALT", 0)],
        [("IMM", 10), ("PSH", 0), ("IMM", 7), ("SI", 0),
         ("IMM", 10), ("LI", 0), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)],
    ):
        ok, e, g = _naive_matches(lean_memcmp, prog)
        assert ok, (prog, e["ax_trace"], g["ax_trace"])


# ---------------------------------------------------------------------------
# The SPECULATIVE graphed driver == the eager speculative driver, byte-for-byte.
# (The batched [B,Smax,H] forward is the ideal fixed-shape replay target.)
# ---------------------------------------------------------------------------
def _spec_matches(lean, prog, block_steps=32):
    code = isa.assemble(prog)
    eager = LF.speculative_run_lean(lean, code, block_steps=block_steps)
    graphed = CG.speculative_run_lean_graphed(lean, code, block_steps=block_steps)
    return eager.ax_trace == graphed.ax_trace, eager, graphed


@requires_cuda
def test_speculative_graphed_matches_eager_loops(lean_base):
    for prog in (
        [("IMM", 5), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)],
        [("IMM", 20), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)],
        [("IMM", 60), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)],
    ):
        ok, e, g = _spec_matches(lean_base, prog)
        assert ok, (prog, e.ax_trace, g.ax_trace)
        assert g.exact
        assert g.forwards < g.naive_forwards              # real forwards saved


@requires_cuda
def test_speculative_graphed_matches_eager_memory(lean_memcmp):
    prog = [("IMM", 10), ("PSH", 0), ("IMM", 7), ("SI", 0),
            ("IMM", 10), ("LI", 0), ("PSH", 0), ("IMM", 3), ("ADD", 0), ("HALT", 0)]
    ok, e, g = _spec_matches(lean_memcmp, prog)
    assert ok, (e.ax_trace, g.ax_trace)
    assert g.exact


# ---------------------------------------------------------------------------
# The pad-to-max-window bucketing scheme: ONE graph covers all naive windows and
# is still byte-identical (pad rows are causally invisible).
# ---------------------------------------------------------------------------
@requires_cuda
def test_pad_window_bucket_is_byte_identical(lean_memcmp):
    lean = lean_memcmp
    g = CG.GraphedLeanForward(lean, pad_window=16)
    prog = [("IMM", 10), ("PSH", 0), ("IMM", 7), ("SI", 0),
            ("IMM", 30), ("PSH", 0), ("IMM", 9), ("SI", 0),
            ("IMM", 10), ("LI", 0), ("HALT", 0)]         # window grows across steps
    code = isa.assemble(prog)
    eager = LF.run_program_lean(lean, code, max_steps=4096)
    graphed = CG.run_program_lean_graphed(lean, code, max_steps=4096, graphed=g)
    assert eager["ax_trace"] == graphed["ax_trace"]
    # every naive window (S = 7..) padded to 16 -> a SINGLE captured graph.
    assert g.n_capture == 1
    assert g.shapes == [(1, 16)]


# ---------------------------------------------------------------------------
# CPU fallback: on a non-CUDA lean the graphed wrapper transparently runs eager.
# ---------------------------------------------------------------------------
def test_cpu_fallback_runs_eager_and_matches():
    vm = Q.build(code_size=24, subset=Q.SUBSET_BASE)
    lean = LF.LeanQwenVM.from_full_vm(vm, device="cpu")
    g = CG.GraphedLeanForward(lean)
    assert not g.enabled
    code = isa.assemble([("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)])
    eager = LF.run_program_lean(lean, code)
    graphed = CG.run_program_lean_graphed(lean, code, graphed=g)
    assert eager["ax_trace"] == graphed["ax_trace"]
    assert g.n_eager > 0 and g.n_replay == 0             # never captured on CPU
