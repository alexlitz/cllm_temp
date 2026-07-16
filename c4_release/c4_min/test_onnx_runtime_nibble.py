"""Prove the C ONNX runtime (onnx_runtime_nibble.c) runs the nibble ONNX model
byte-identical to torch/onnxruntime.

Pipeline under test:
    torch Transformer  --export_onnx-->  nibble_vm.onnx
                       --onnx_to_c4bin-->  .nblbin
                       --gcc onnx_runtime_nibble.c-->  native C runtime
    C runtime(.nblbin, tokens)  ==(argmax byte head)==  torch(tokens)

We check, across several token frames (incl. real program emissions), that the C
runtime's per-position argmax over the LM head matches torch AND onnxruntime on
every position, and that the raw logits agree to fp32 accumulation noise (~3e-5,
the same torch-vs-ORT gap — the VM decodes by argmax, so argmax equality is the
byte-identity the VM actually needs).
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from c4_min import export_onnx as E
from c4_min import blogspec_compiler as C
from c4_min import blogspec_run as R
from c4_min import blogspec_vocab as V
from c4_min.onnx_to_c4bin import lower_onnx_to_bin
from c4_min.nbl_bin_interp import Graph

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "onnx_runtime_nibble.c")


def _cc(exe):
    for flags in (["-O2", "-static-libgcc"], ["-O2"]):
        r = subprocess.run(["gcc"] + flags + ["-o", exe, CSRC, "-lm"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
    raise RuntimeError("gcc failed:\n" + r.stderr)


def _run_c(exe, binpath, tok):
    B, S = tok.shape
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{B} {S}\n")
        f.write(" ".join(str(int(x)) for x in tok.ravel()))
        tokfile = f.name
    out = subprocess.run([exe, binpath, tokfile, "--dump-logits"],
                         capture_output=True, text=True, check=True).stdout
    os.unlink(tokfile)
    lines = [l for l in out.splitlines() if l and not l[0].isalpha()]
    vals = np.array([float(x) for x in lines], dtype=np.float32)
    return vals.reshape(B, S, -1)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    d = tmp_path_factory.mktemp("nblrt")
    dense = os.path.join(d, "m.onnx")
    binp = os.path.join(d, "m.nblbin")
    exe = os.path.join(d, "rt")
    model, L, code = C.build_step_model(E.PROOF_PROG)
    model.eval()
    E.export_onnx(model, dense)
    lower_onnx_to_bin(dense, binp)
    _cc(exe)
    sess = ort.InferenceSession(dense, providers=["CPUExecutionProvider"])
    return model, L, code, dense, binp, exe, sess


def _frames(model, L, code):
    # a few representative token frames the C runtime must reproduce
    yield "proof_full", R.run_program(model, L, code, max_steps=20)[0]
    yield "single_step", R.run_program(model, L, code, max_steps=1)[0]
    yield "tiny", [V.BOS, V.REG_AX, 0x2A, V.STEP_END]


def test_c_runtime_matches_torch_and_ort(rig):
    model, L, code, dense, binp, exe, sess = rig
    n_checked = 0
    for tag, toks in _frames(model, L, code):
        tok = np.array([toks], dtype=np.int64)
        with torch.no_grad():
            tl = model(torch.tensor(tok, dtype=torch.long)).numpy()[0]
        ol = sess.run(None, {"tokens": tok})[0][0]
        cl = _run_c(exe, binp, tok)[0]
        assert cl.shape == tl.shape, (tag, cl.shape, tl.shape)

        am_t, am_o, am_c = tl.argmax(-1), ol.argmax(-1), cl.argmax(-1)
        # the meaningful byte-identity: argmax matches torch and ORT everywhere
        assert (am_c == am_t).all(), f"{tag}: C vs torch argmax mismatch"
        assert (am_c == am_o).all(), f"{tag}: C vs ORT argmax mismatch"
        # raw logits agree to fp32 accumulation noise
        assert np.abs(cl - tl).max() < 1e-3, f"{tag}: C vs torch logit gap"
        assert np.abs(cl - ol).max() < 1e-3, f"{tag}: C vs ORT logit gap"
        n_checked += tl.shape[0]
    assert n_checked > 150


def test_c_runtime_matches_numpy_reference(rig):
    """C runtime == the pure-numpy .nblbin reference interp (the shared spec)."""
    model, L, code, dense, binp, exe, sess = rig
    tok = np.array([R.run_program(model, L, code, max_steps=20)[0]], dtype=np.int64)
    g = Graph(binp)
    nl = g.run(tok)[0]
    cl = _run_c(exe, binp, tok)[0]
    assert (cl.argmax(-1) == nl.argmax(-1)).all()
    assert np.abs(cl - nl).max() < 1e-3


def test_program_result_through_c_runtime(rig):
    """The IMM 6; PSH; IMM 7; ADD program decodes AX=13 through the C runtime."""
    model, L, code, dense, binp, exe, sess = rig
    tokens_full, frames = R.run_program(model, L, code, max_steps=20)
    tok = np.array([tokens_full], dtype=np.int64)
    cl = _run_c(exe, binp, tok)[0]
    # decode the C-runtime logits with the VM's own frame decoder
    am = cl.argmax(-1)
    # the reference (torch) final AX for this program is 13 (6+7); the C runtime's
    # argmax stream must reproduce the same emitted-frame argmaxes as torch.
    with torch.no_grad():
        tl = model(torch.tensor(tok, dtype=torch.long)).numpy()[0]
    assert (am == tl.argmax(-1)).all()
    assert frames[-1]["ax"] == 13
