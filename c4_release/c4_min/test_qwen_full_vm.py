"""The FUSED FULL VM runs through a genuine ``Qwen2Model.forward`` — compute AND
control in one forward per step. Every test drives programs through an actual
``transformers`` ``Qwen2Model`` (RoPE + RMSNorm + plain-softmax GQA + SwiGLU) with
the whole VM step baked in, and asserts argmax-exactness vs the ``isa.interpret``
reference. See ``c4_min/qwen_full_vm.py``.
"""
from __future__ import annotations

import pytest

from c4_min import isa
from c4_min import qwen_full_vm as Q
from c4_min import qwen_full_vm_corpus as C


@pytest.fixture(scope="module")
def vm_base():
    return Q.build(code_size=24, subset=Q.SUBSET_BASE)


@pytest.fixture(scope="module")
def vm_memcmp():
    return Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)


def _exact(vm, prog):
    return Q.run_program(vm, isa.assemble(prog), max_steps=48)


# -- the model is a genuine Qwen2 (not a look-alike), fits stock 0.5B for base --
def test_it_is_a_real_qwen2_model(vm_base):
    from transformers.models.qwen2 import Qwen2Model
    assert isinstance(vm_base.qmodel, Qwen2Model)
    cfg = vm_base.qmodel.config
    assert cfg.num_attention_heads == 14 and cfg.num_key_value_heads == 2   # GQA
    assert cfg.hidden_act == "silu"                                         # SwiGLU
    assert cfg.rms_norm_eps == 1e-6                                         # RMSNorm
    assert cfg.head_dim == 64 and cfg.rope_theta == 1_000_000.0            # RoPE ladder


def test_base_subset_fits_stock_0_5b(vm_base):
    """The base op-family model fits the STOCK Qwen2.5-0.5B budget."""
    assert vm_base.hidden_size <= 896        # stock hidden_size
    assert vm_base.intermediate_size <= 4864  # stock intermediate_size
    assert vm_base.n_layers <= 24            # stock num_hidden_layers
    assert vm_base.fits_stock


# -- arith / branch / control through the fused forward ----------------------
@pytest.mark.parametrize("a,b", [(3, 4), (200, 55), (255, 1)])
def test_add_through_qwen(vm_base, a, b):
    r = _exact(vm_base, [("IMM", a), ("PSH", 0), ("IMM", b), ("ADD", 0), ("HALT", 0)])
    assert r["exact"], r


@pytest.mark.parametrize("a,b", [(9, 4), (7, 9), (0, 1)])
def test_sub_through_qwen(vm_base, a, b):
    r = _exact(vm_base, [("IMM", a), ("PSH", 0), ("IMM", b), ("SUB", 0), ("HALT", 0)])
    assert r["exact"], r


def test_branch_bz_bnz_jmp(vm_base):
    assert _exact(vm_base, [("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7),
                            ("HALT", 0)])["exact"]
    assert _exact(vm_base, [("IMM", 1), ("BNZ", 3), ("IMM", 99), ("IMM", 7),
                            ("HALT", 0)])["exact"]
    assert _exact(vm_base, [("JMP", 2), ("IMM", 99), ("IMM", 5), ("HALT", 0)])["exact"]


def test_loop_backbranch(vm_base):
    r = _exact(vm_base, [("IMM", 3), ("PSH", 0), ("IMM", 1), ("SUB", 0),
                         ("BNZ", 1), ("HALT", 0)])
    assert r["exact"], r


# -- FUNCTIONS (JSR/ENT/ADJ/LEV) through the fused forward -------------------
def test_function_call_leaf(vm_base):
    """main JSRs a leaf that ENTers a frame, returns 42, LEVs back."""
    r = _exact(vm_base, [("IMM", 0), ("JSR", 4), ("PSH", 0), ("HALT", 0),
                         ("ENT", 0), ("IMM", 42), ("LEV", 0)])
    assert r["exact"], r


def test_function_nested_calls(vm_base):
    """main -> f -> g; g returns 0x37 propagated back through two frames."""
    r = _exact(vm_base, [("IMM", 1), ("JSR", 4), ("HALT", 0), ("NOP", 0),
                         ("ENT", 0), ("JSR", 8), ("LEV", 0), ("NOP", 0),
                         ("ENT", 0), ("IMM", 0x37), ("LEV", 0)])
    assert r["exact"], r


def test_function_adj_ent(vm_base):
    assert _exact(vm_base, [("IMM", 0x99), ("PSH", 0), ("ADJ", 1),
                            ("IMM", 0x11), ("HALT", 0)])["exact"]
    assert _exact(vm_base, [("ENT", 8), ("IMM", 5), ("HALT", 0)])["exact"]


# -- comparisons through the fused forward -----------------------------------
@pytest.mark.parametrize("op,a,b,want", [
    ("EQ", 5, 5, 1), ("EQ", 7, 9, 0), ("NE", 7, 9, 1), ("LT", 7, 9, 1),
    ("GT", 9, 7, 1), ("LE", 9, 7, 0), ("GE", 9, 7, 1),
])
def test_cmp_through_qwen(vm_memcmp, op, a, b, want):
    r = _exact(vm_memcmp, [("IMM", a), ("PSH", 0), ("IMM", b), (op, 0), ("HALT", 0)])
    assert r["exact"], r
    assert r["ax_trace"][-1] == want


# -- memory (SI/LI) + variables + ZFOD through the fused forward -------------
def test_memory_store_load(vm_memcmp):
    r = _exact(vm_memcmp, [("IMM", 5), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
                           ("IMM", 5), ("LI", 0), ("HALT", 0)])
    assert r["exact"], r


def test_memory_zfod_unwritten_reads_zero(vm_memcmp):
    """An unwritten address loads 0 (softmax1 ZFOD reproduced by the BOS sink)."""
    r = _exact(vm_memcmp, [("IMM", 50), ("LI", 0), ("HALT", 0)])
    assert r["exact"] and r["ax_trace"][-1] == 0, r


def test_memory_latest_write_wins(vm_memcmp):
    """Two stores to one address; the load returns the LATEST (RoPE recency)."""
    r = _exact(vm_memcmp, [("IMM", 30), ("PSH", 0), ("IMM", 1), ("SI", 0),
                           ("IMM", 30), ("PSH", 0), ("IMM", 9), ("SI", 0),
                           ("IMM", 30), ("LI", 0), ("HALT", 0)])
    assert r["exact"] and r["ax_trace"][-1] == 9, r


def test_variable_via_memory(vm_memcmp):
    """A local variable: store x, load it, add — the c4 var idiom."""
    r = _exact(vm_memcmp, [("IMM", 10), ("PSH", 0), ("IMM", 7), ("SI", 0),
                           ("IMM", 10), ("LI", 0), ("PSH", 0), ("IMM", 3),
                           ("ADD", 0), ("HALT", 0)])
    assert r["exact"], r


# -- MUL/DIV/MOD through the fused forward (pruned FFN table) -----------------
@pytest.mark.slow
@pytest.mark.parametrize("op,a,b", [("MUL", 6, 7), ("DIV", 84, 7), ("MOD", 84, 5)])
def test_muldiv_through_qwen(op, a, b):
    keys = [(Q.isa.MUL if op == "MUL" else Q.isa.DIV if op == "DIV" else Q.isa.MOD,
             a, b)]
    vm = Q.build(code_size=16, subset=Q.SUBSET_MULDIV, mdm_keys=keys)
    r = _exact(vm, [("IMM", a), ("PSH", 0), ("IMM", b), (op, 0), ("HALT", 0)])
    assert r["exact"], r


# -- the compute really runs in the Qwen forward, not a python gadget --------
def test_compute_is_in_the_qwen_forward(vm_base):
    """Zeroing the Qwen MLPs annihilates the result -> the arithmetic is Qwen's
    own SwiGLU, not a python copy."""
    import torch
    prog = isa.assemble([("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)])
    assert Q.run_program(vm_base, prog)["exact"]
    saved = {}
    with torch.no_grad():
        for li, layer in enumerate(vm_base.qmodel.layers):
            for nm, lin in (("g", layer.mlp.gate_proj), ("u", layer.mlp.up_proj),
                            ("d", layer.mlp.down_proj)):
                saved[(li, nm)] = lin.weight.clone()
                lin.weight.zero_()
    got = Q.run_program(vm_base, prog)
    with torch.no_grad():
        for li, layer in enumerate(vm_base.qmodel.layers):
            for nm, lin in (("g", layer.mlp.gate_proj), ("u", layer.mlp.up_proj),
                            ("d", layer.mlp.down_proj)):
                lin.weight.copy_(saved[(li, nm)])
    assert not got["exact"]                 # no MLP -> no compute
    assert Q.run_program(vm_base, prog)["exact"]  # restored


# -- the corpus deliverable: argmax-exact fraction through Qwen2Model.forward -
def test_corpus_base_families_100pct():
    """arith / if / loop / func — the families that fit the STOCK 0.5B budget —
    are 100% argmax-exact through the real Qwen2 forward."""
    rep = C.run(subsets=["base"])
    assert rep["n_pass"] == rep["n_total"], [r for r in rep["results"] if not r["exact"]]
    assert rep["n_total"] >= 20


def test_corpus_memcmp_families_100pct():
    """cmp / memory / var — 100% argmax-exact through the real Qwen2 forward."""
    rep = C.run(subsets=["mem+cmp"])
    assert rep["n_pass"] == rep["n_total"], [r for r in rep["results"] if not r["exact"]]
