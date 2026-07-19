"""Per-op-family byte-exactness through the REAL Qwen2 MLP with RMSNorm active.

Each test builds a real ``Qwen2Model`` whose MLP layers carry an op family's
gadget SwiGLU weights (RMSNorm-compensator gamma active), drives operands through
``qmodel.forward``, and asserts the LM byte-head decodes the correct result with
a comfortable margin. These are the mission's deliverable proofs: the op COMPUTE
runs through Qwen's own SwiGLU under RMSNorm, not a python gadget.
"""
import pytest

from c4_min import isa
from c4_min import qwen_mlp_ops as Q


def _check(report, min_margin: float):
    assert report["exact"], (report["op"], report["first_fail"])
    assert report["worst_margin"] >= min_margin, (report["op"],
                                                   report["worst_margin"])


def test_add_byte_exact():
    _check(Q.verify_add(), 0.5)


def test_sub_byte_exact():
    _check(Q.verify_sub(), 0.5)


def test_add16_carry_byte_exact():
    _check(Q.verify_add16(n=128), 0.5)


@pytest.mark.parametrize("op", [isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE])
def test_cmp_byte_exact(op):
    _check(Q.verify_cmp(op), 0.4)


@pytest.mark.parametrize("kind", ["and", "or", "xor"])
def test_bitwise_byte_exact(kind):
    _check(Q.verify_bitwise(kind), 0.4)


def test_branch_predicate_byte_exact():
    _check(Q.verify_branch(), 0.4)


def test_compute_is_in_the_mlp_not_the_embedding():
    """Zeroing the compute MLP annihilates the result -> the arithmetic is done
    by Qwen's SwiGLU, not by the embedding or a python gadget."""
    import torch
    from c4_min import blogspec_vocab as V
    L = Q.OpLayout(Q.QWEN_TINY.hidden_size)
    vm = Q.build_qwen_op_vm(Q.add_specs(L), L, Q.QWEN_TINY, Q.NORM_K)
    an = V.nibbles_of_byte(200); bn = V.nibbles_of_byte(100)
    fields = {L.STACK0 + 0: an[0], L.STACK0 + 1: an[1],
              L.AX + 0: bn[0], L.AX + 1: bn[1]}
    x = Q._residual_from_fields(L, fields, Q.NORM_K)
    got, _ = Q.decode_byte_margin(Q._forward_last(vm, x), L, L.RES, 0)
    assert got == (200 + 100) & 0xFF                    # 44, computed in the MLP
    with torch.no_grad():
        for lyr in vm.qmodel.layers:
            for lin in (lyr.mlp.gate_proj, lyr.mlp.up_proj, lyr.mlp.down_proj):
                lin.weight.zero_()
    got2, _ = Q.decode_byte_margin(Q._forward_last(vm, x), L, L.RES, 0)
    assert got2 == 0                                    # no MLP -> no compute


def test_rmsnorm_k_threshold_is_honest():
    """The RMSNorm 1/K rescale DOES collapse the ADD margin below a threshold K
    (RELU_S=200 amplifies the rescale error on ~15-30-valued operands). Document
    it: K=400 fails, K>=700 is byte-exact, K=4000 (default) has margin ~0.98."""
    assert not Q.verify_add(Q.QWEN_TINY, K=400.0)["exact"]
    assert Q.verify_add(Q.QWEN_TINY, K=700.0)["exact"]
    r = Q.verify_add(Q.QWEN_TINY, K=4000.0)
    assert r["exact"] and r["worst_margin"] > 0.9


def test_memory_li_si_through_real_qwen():
    """LI/SI + ZFOD through the REAL Qwen forward (RoPE address-match + BOS-sink),
    via the qwen_embed VM the compute path shares its RMSNorm-compensator with."""
    from c4_min import qwen_embed as qe
    vm = qe.build_qwen_vm(qe.QWEN_TINY)
    assert qe.ingest_ax_through_qwen(vm, 123) == 123          # LI: load stored byte
    r = qe.ingest_ax_recency_through_qwen(vm, old_byte=11, new_byte=99)
    assert r["got"] == 99                                     # SI: latest-write-wins
    assert qe.zfod_no_ax_marker_through_qwen(vm) == 0         # ZFOD read-0-on-miss
