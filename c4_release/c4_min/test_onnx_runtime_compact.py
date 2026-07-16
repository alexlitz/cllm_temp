"""Prove the C-in-C4 ONNX runtime runs the SMALL COMPACTED c4_min VM model.

The compaction (``compact_alloc.build_compact_pure_forward_model``) shrinks the
full pure-forward VM (dim=2024, ~300 blocks, ~30 GB dense) to a dim=1564, 42-block
dense model whose weights are 99.9% zero — a ~2 MB COO-sparse ``.nblbin`` the C
runtime loads.  Two graphs are validated:

  * **token-embedding graph** (``tokens -> logits``): the standard autoregressive
    interface.  The C runtime is BIT-EXACT (max|Δ|=0) to torch and to the numpy
    ``.nblbin`` reference on a battery of frames — argmax + raw fp32 both.

  * **block-stack graph** (``residual -> hidden``): the residual-in/out compute the
    corpus driver actually runs (``x=embed[toks]; overlay(x); for blk: x=blk(x)``,
    reading VM state straight out of the returned residual).  Here the C runtime is
    DECODE-EXACT: every register lane / nibble the VM decodes on the query row is
    bit-identical to torch; the only divergences are fp32 accumulation-order noise
    on INTERIOR (non-decoded) positions, which never reaches a register decode.

These tests are heavy (they build the compact model, ~40 s) — marked ``slow``.
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")

from c4_min import compact_alloc as CA
from c4_min import export_onnx as E
from c4_min import blogspec_vocab as V
from c4_min import onnx_c_driver as CD
from c4_min.onnx_to_c4bin import lower_onnx_to_bin
from c4_min.nbl_bin_interp import Graph

pytestmark = pytest.mark.slow

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "onnx_runtime_nibble.c")


def _cc(exe):
    for flags in (["-O2", "-static-libgcc"], ["-O2"]):
        r = subprocess.run(["gcc"] + flags + ["-o", exe, CSRC, "-lm"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
    raise RuntimeError("gcc failed:\n" + r.stderr)


def _run_c_tokens(exe, binp, tok):
    B, S = tok.shape
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{B} {S}\n")
        f.write(" ".join(str(int(x)) for x in tok.ravel()))
        tf = f.name
    out = subprocess.run([exe, binp, tf, "--dump-logits"],
                         capture_output=True, text=True, check=True).stdout
    os.unlink(tf)
    lines = [l for l in out.splitlines() if l and (l[0].isdigit() or l[0] in "+-.")]
    return np.array([float(x) for x in lines], dtype=np.float32).reshape(B, S, -1)


@pytest.fixture(scope="module")
def compact():
    model, L, _ = CA.build_compact_pure_forward_model(
        code_size=48, include_bitwise=True, include_divmod=False)
    model.eval()
    return model, L


@pytest.fixture(scope="module")
def token_rig(compact, tmp_path_factory):
    model, L = compact
    d = tmp_path_factory.mktemp("nblc_tok")
    dense = os.path.join(d, "m.onnx")
    sp = os.path.join(d, "m_sparse.onnx")
    binp = os.path.join(d, "m.nblbin")
    exe = os.path.join(d, "rt")
    E.export_onnx(model, dense)
    E.to_sparse_onnx(dense, sp)
    lower_onnx_to_bin(sp, binp)
    _cc(exe)
    return model, L, binp, exe


def test_compact_token_graph_bit_exact(token_rig):
    """C runtime == torch == numpy-ref, BIT-exact, on the compact token graph."""
    model, L, binp, exe = token_rig
    g = Graph(binp)
    rng = np.random.default_rng(0)
    frames = [[V.BOS, V.REG_AX, 0x2A, V.STEP_END]]
    for _ in range(4):
        S = int(rng.integers(4, 32))
        frames.append([V.BOS] + [int(rng.integers(0, V.VOCAB)) for _ in range(S - 1)])
    for toks in frames:
        tok = np.array([toks], dtype=np.int64)
        with torch.no_grad():
            tl = model(torch.tensor(tok, dtype=torch.long)).numpy()[0]
        nl = g.run(tok)[0]
        cl = _run_c_tokens(exe, binp, tok)[0]
        assert (cl.argmax(-1) == tl.argmax(-1)).all()
        assert (cl.argmax(-1) == nl.argmax(-1)).all()
        assert np.abs(cl - tl).max() == 0.0          # bit-exact
        assert np.abs(cl - nl).max() == 0.0


@pytest.fixture(scope="module")
def blockstack_rig(compact, tmp_path_factory):
    model, L = compact
    d = str(tmp_path_factory.mktemp("nblc_bs"))
    rt = CD.build_and_lower(model, L, d, sparse=True)
    return model, L, rt


def test_compact_blockstack_decode_exact(blockstack_rig):
    """The residual-graph C runtime is DECODE-exact to torch: every register lane /
    nibble on the query row matches (interior-position fp noise excepted)."""
    from c4_min import nibble_pure_forward_complete as PFC
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from src.compiler import compile_c
    model, L, rt = blockstack_rig
    embed = model.embed.detach().numpy()
    reg_mismatch = 0
    checked = 0
    for src in ("int main(){return 42;}", "int main(){return 6+7;}",
                "int main(){return 100-58;}"):
        code = bytecode_to_isa(compile_c(src)[0])
        stream = [V.BOS] + PFC._build_frame(0, 0, PFC.SP_INIT, PFC.SP_INIT, 0)
        for _ in range(3):
            overlay = PFC.make_overlay_complete(code, L, store_log={})
            x = torch.from_numpy(embed[np.asarray(stream)]).unsqueeze(0).clone()
            overlay(x)
            with torch.no_grad():
                xt = x.clone()
                for blk in model.blocks:
                    xt = blk(xt)
            hidden_c = rt.forward_residual(x.numpy())
            st_t, st_c = xt[0, -1], torch.from_numpy(hidden_c[0, -1])
            for base in (L.PC_VAL, L.SP_VAL, L.BP_VAL, L.STK_VAL):
                reg_mismatch += int(PFC._snap_lane(st_t[base]) != PFC._snap_lane(st_c[base]))
            reg_mismatch += int(PFC._decode_reg_from_nibbles(st_t, L, L.AX) !=
                                PFC._decode_reg_from_nibbles(st_c, L, L.AX))
            checked += 1
            pc = PFC._snap_lane(st_t[L.PC_VAL]); sp = PFC._snap_lane(st_t[L.SP_VAL])
            bp = PFC._snap_lane(st_t[L.BP_VAL]); stk = PFC._snap_lane(st_t[L.STK_VAL])
            ax = PFC._decode_reg_from_nibbles(st_t, L, L.AX)
            stream += PFC._build_frame(pc, ax, sp, bp, stk)
            if float(st_t[L.HALTED]) > 0.5 or pc < 0 or pc >= len(code):
                break
    assert checked > 0
    assert reg_mismatch == 0, f"{reg_mismatch} register-decode mismatches"


def test_compact_program_end_to_end(blockstack_rig):
    """A whole program run autoregressively through the C runtime decodes the same
    exit as torch (the corpus verdict, one step = one C-runtime block-stack call)."""
    from c4_min import nibble_pure_forward_complete as PFC
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from c4_min.nibble_pure_forward_complete import ref_interpret, run_pure_forward_complete
    from src.compiler import compile_c
    model, L, rt = blockstack_rig
    embed = model.embed.detach().numpy()
    for src, exp in (("int main(){return 42;}", 42),):
        code = bytecode_to_isa(compile_c(src)[0])
        rs = len(ref_interpret(code, max_steps=2000, mask=0xFFFFFFFF))
        tr_t = run_pure_forward_complete(model, L, code, max_steps=rs + 3, mask=0xFFFFFFFF)
        tr_c = CD.run_pure_forward_c(rt, L, code, max_steps=rs + 3, mask=0xFFFFFFFF, embed=embed)
        got_t = int(tr_t[-1]) & 0xFFFFFFFF
        got_c = int(tr_c[-1]) & 0xFFFFFFFF
        assert got_t == exp
        assert got_c == got_t             # C runtime verdict == torch verdict
