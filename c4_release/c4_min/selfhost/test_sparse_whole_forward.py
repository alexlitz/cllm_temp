"""The SPARSE whole-forward C runtime (onnx_runtime_nibble_sparse.c) is BYTE-EXACT
to the DENSE runtime (onnx_runtime_nibble.c) — same 23-op vanilla forward, but the
MatMul iterates ONLY the nonzero weight entries (O(nnz), incl. through Identity).

The load-bearing property proven here: at the VM's ~1e29 residual magnitudes, fp
accumulation ORDER decides low bits (and can flip a decode argmax).  The COO nonzeros
are stored row-major (ascending flat = r*N+q), so a single scatter pass over them in
stored order reduces each output column in ascending-r order — EXACTLY the dense
sum_r order — giving max|delta| == 0 vs the dense runtime.

Uses the tiny PROOF_PROG step model (fast to build) so the whole test runs in a few
seconds.  A heavier full-ISA byte-exact + speedup measurement lives in the
``_agent_*`` drivers in this dir (the compact 242-block model is ~140 s to build).
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

from c4_min import export_onnx as E
from c4_min import blogspec_compiler as C
from c4_min import blogspec_run as R
from c4_min import blogspec_vocab as V
from c4_min.onnx_to_c4bin import lower_onnx_to_bin

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DENSE_SRC = os.path.join(HERE, "onnx_runtime_nibble.c")
SPARSE_SRC = os.path.join(HERE, "onnx_runtime_nibble_sparse.c")


def _cc(src, exe):
    err = ""
    for flags in (["-O2", "-static", "-static-libgcc"], ["-O2", "-static"], ["-O2"]):
        r = subprocess.run(["gcc"] + flags + ["-o", exe, src, "-lm"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
        err = r.stderr
    raise RuntimeError("gcc failed:\n" + err)


def _run_tokens(exe, binp, tok):
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
def rig(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("nblrt_sparse"))
    dense_onnx = os.path.join(d, "m.onnx")
    sp_onnx = os.path.join(d, "m_sparse.onnx")
    binp = os.path.join(d, "m.nblbin")
    dense_exe = os.path.join(d, "rt_dense")
    sparse_exe = os.path.join(d, "rt_sparse")
    model, L, code = C.build_step_model(E.PROOF_PROG)
    model.eval()
    E.export_onnx(model, dense_onnx)
    E.to_sparse_onnx(dense_onnx, sp_onnx)     # COO-encode the big weights
    lower_onnx_to_bin(sp_onnx, binp)          # .nblbin carries is_init==2 COO tensors
    _cc(DENSE_SRC, dense_exe)
    _cc(SPARSE_SRC, sparse_exe)
    return model, L, code, binp, dense_exe, sparse_exe


def _frames(model, L, code):
    yield "proof_full", R.run_program(model, L, code, max_steps=20)[0]
    yield "single_step", R.run_program(model, L, code, max_steps=1)[0]
    yield "tiny", [V.BOS, V.REG_AX, 0x2A, V.STEP_END]


def test_sparse_token_graph_byte_exact_vs_dense(rig):
    """Sparse runtime == dense runtime, BIT-exact (max|delta|==0) on the token graph."""
    model, L, code, binp, dense_exe, sparse_exe = rig
    n = 0
    for tag, toks in _frames(model, L, code):
        tok = np.array([toks], dtype=np.int64)
        dl = _run_tokens(dense_exe, binp, tok)
        sl = _run_tokens(sparse_exe, binp, tok)
        assert dl.shape == sl.shape, (tag, dl.shape, sl.shape)
        assert np.abs(dl - sl).max() == 0.0, f"{tag}: sparse != dense (max|delta|>0)"
        assert (dl.argmax(-1) == sl.argmax(-1)).all()
        n += tok.shape[1]
    assert n > 20


def test_sparse_token_graph_argmax_matches_torch(rig):
    """Sparse runtime argmax == torch on every position (the byte-identity the VM needs)."""
    model, L, code, binp, dense_exe, sparse_exe = rig
    for tag, toks in _frames(model, L, code):
        tok = np.array([toks], dtype=np.int64)
        with torch.no_grad():
            tl = model(torch.tensor(tok, dtype=torch.long)).numpy()[0]
        sl = _run_tokens(sparse_exe, binp, tok)[0]
        assert (sl.argmax(-1) == tl.argmax(-1)).all(), f"{tag}: sparse vs torch argmax"


def test_sparse_program_result(rig):
    """IMM 6; PSH; IMM 7; ADD decodes AX=13 through the SPARSE runtime (== torch)."""
    model, L, code, binp, dense_exe, sparse_exe = rig
    tokens_full, frames = R.run_program(model, L, code, max_steps=20)
    tok = np.array([tokens_full], dtype=np.int64)
    sl = _run_tokens(sparse_exe, binp, tok)[0]
    with torch.no_grad():
        tl = model(torch.tensor(tok, dtype=torch.long)).numpy()[0]
    assert (sl.argmax(-1) == tl.argmax(-1)).all()
    assert frames[-1]["ax"] == 13
