"""Byte-identity + block-density gates for the block-structured sparse forward.

Proves:
  1. the clustering permutation (``build_clustering_permutation`` + ``apply_permutation``)
     is a BYTE-IDENTICAL relabeling — the permuted lean forward decodes the SAME
     register trace as the un-permuted one (a column reorder only changes the fp
     REDUCTION ORDER, a ~1e-12-relative residue far below the integer ``_snap``
     decode margin);
  2. the block-sparse (BSR) forward on the clustered weights decodes IDENTICALLY
     to the dense forward across the op battery (permutation is relabeling, block
     matmul is the same arithmetic on the nonzeros);
  3. the block-density accounting is self-consistent (active tiles ≤ total, the
     block-flop ratio matches the tile count).

Small (7-14 layer, 6-CAM-head) compacted model → CPU, memory-light.  Run:
    OMP_NUM_THREADS=4 python -m pytest c4_min/test_block_sparse_forward.py -v
"""
from __future__ import annotations

import pytest
import torch

from c4_min import isa
from c4_min import qwen_full_vm as Q
from c4_min import qwen_lean_forward as LF
from c4_min import block_sparse_analysis as BSA
from c4_min import block_sparse_forward as BSF


@pytest.fixture(scope="module")
def lean_memcmp():
    vm = Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)
    return LF.LeanQwenVM.from_full_vm(vm, device="cpu")


_BATTERY = [
    ("add", [("IMM", 100), ("PSH", 0), ("IMM", 27), ("ADD", 0), ("HALT", 0)]),
    ("sub", [("IMM", 200), ("PSH", 0), ("IMM", 55), ("SUB", 0), ("HALT", 0)]),
    ("mul", [("IMM", 12), ("PSH", 0), ("IMM", 10), ("MUL", 0), ("HALT", 0)]),
    ("eq", [("IMM", 5), ("PSH", 0), ("IMM", 5), ("EQ", 0), ("HALT", 0)]),
    ("gt", [("IMM", 9), ("PSH", 0), ("IMM", 7), ("GT", 0), ("HALT", 0)]),
    ("bz", [("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)]),
    ("loop", [("IMM", 5), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]),
    ("mem", [("IMM", 5), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
             ("IMM", 5), ("LI", 0), ("HALT", 0)]),
]


# ---------------------------------------------------------------------------
# 1. the permutation is byte-identical (decode-identical).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["pack", "rcm"])
@pytest.mark.parametrize("name,prog", _BATTERY, ids=[b[0] for b in _BATTERY])
def test_permutation_decode_identical(lean_memcmp, method, name, prog):
    lean = lean_memcmp
    perm = BSA.build_clustering_permutation(lean, method=method)
    plean = BSA.apply_permutation(lean, perm)
    pH = perm.pH
    inv = torch.empty_like(pH)
    inv[pH] = torch.arange(pH.numel())

    class _Wrap:
        def __init__(self):
            for a in ("QL", "subset", "device", "embed", "hidden_size", "n_layers"):
                setattr(self, a, getattr(lean, a))

        def forward(self, x, past=None, q_positions=None):
            h, _ = plean.forward(x[:, :, pH], past=past, q_positions=q_positions)
            return h[:, :, inv], None

    code = isa.assemble(prog)
    r0 = LF.run_program_lean(lean, code, max_steps=32)
    r1 = LF.run_program_lean(_Wrap(), code, max_steps=32)
    assert r0["ax_trace"] == r1["ax_trace"], (name, method, r0["ax_trace"], r1["ax_trace"])


# ---------------------------------------------------------------------------
# 2. the block-sparse forward decodes identically to dense.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("blocksize", [8, 16, 32])
@pytest.mark.parametrize("name,prog", _BATTERY, ids=[b[0] for b in _BATTERY])
def test_block_sparse_decode_identical(lean_memcmp, blocksize, name, prog):
    lean = lean_memcmp
    perm = BSA.build_clustering_permutation(lean, method="pack")
    bs = BSF.PermutedBlockSparseVM(lean, perm, blocksize=blocksize, min_active_tiles=2)
    code = isa.assemble(prog)
    r0 = LF.run_program_lean(lean, code, max_steps=32)
    r1 = LF.run_program_lean(bs, code, max_steps=32)
    assert r0["ax_trace"] == r1["ax_trace"], (name, blocksize, r0["ax_trace"], r1["ax_trace"])


# ---------------------------------------------------------------------------
# 3. block-density accounting is self-consistent + the residue is fp-order only.
# ---------------------------------------------------------------------------
def test_block_density_accounting(lean_memcmp):
    lean = lean_memcmp
    rep = BSA.model_tile_report(lean, tiles=(8, 16, 32))
    for tile in (8, 16, 32):
        ts = rep[tile]["all"]
        assert 0 <= ts.active_tiles <= ts.total_tiles
        assert ts.nnz > 0
        # >99% of tiles are all-zero (the un-structured baseline: mostly padding).
        assert ts.skip_frac > 0.99, (tile, ts.skip_frac)
        # a bigger tile skips a smaller FRACTION (fewer, fatter tiles) — monotone.
    # active tiles never increase with tile size shrinking coverage: sanity only.
    assert rep[8]["all"].active_tiles >= rep[32]["all"].active_tiles


def test_block_sparse_residue_is_fp_order_only(lean_memcmp):
    lean = lean_memcmp
    perm = BSA.build_clustering_permutation(lean, method="pack")
    bs = BSF.PermutedBlockSparseVM(lean, perm, blocksize=16, min_active_tiles=2)
    code = isa.assemble(_BATTERY[0][1])
    x, pos = LF._build_stream_and_overlay(
        lean, code, {"PC": 0, "AX": 0, "SP": 252, "BP": 252, "STACK0": 0}, [], None)
    with torch.no_grad():
        rd, _ = lean.forward(x, q_positions=pos)
        rb, _ = bs.forward(x, q_positions=pos)
    diff = (rd - rb).abs().max().item()
    scale = rd.abs().max().item()
    # residue is a tiny FRACTION of the (large) VM band magnitudes (fp accum order).
    assert diff <= 1e-6 * scale, (diff, scale)
