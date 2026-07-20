"""CHK-4 tests: the single-file small-model bundle + the C4-C bundler.

Fast tier (default): the torch-only container mechanics — weight COO round-trip
byte-identity, the flat-container section framing, and that the C4-C bundler
(``bundler/c4_bundler_small.c``) fuses the prepared parts into a byte-identical
``.c4bundle``.  These do NOT run the (slow) model forward.

Slow tier (``-m slow``): assemble a real bundle and run it end-to-end through the
bundled model, asserting the decoded result matches.

Skips gracefully without torch / a C compiler.
"""
import os
import shutil
import struct
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4_min.bundle_small import (  # noqa: E402
    assemble_bundle, prepare_bundle, run_bundle, read_header, _section,
    _build_model, _serialize_weights, _deserialize_weights,
    _compile_source_to_bytecode, _phys_blocks, _sw_to_dense, MAGIC,
)

CC = shutil.which("gcc") or shutil.which("cc")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_C4C = os.path.join(_ROOT, "bundler", "c4_bundler_small.c")

_CFG = dict(code_size=32)
_SRC = "int main(){ return 500 + 700; }"


@pytest.fixture(scope="module")
def model_L():
    return _build_model(_CFG)


def test_c4c_bundler_exists_and_gcc_syntax():
    """The C4-C bundler source exists and passes the repo's gcc syntax gate."""
    assert os.path.exists(_C4C), f"missing {_C4C}"
    if CC is None:
        pytest.skip("no C compiler")
    r = subprocess.run([CC, "-fsyntax-only", "-w", _C4C],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def _snapshot(model):
    """Every model weight as a dense tensor keyed by name (streaming-sparse form).

    The bundle's model is the memory-safe streaming-sparse ``SparseTransformer``,
    whose per-block Q/K/V/O + W_up/W_gate/W_down are ``SparseWeight`` objects (no
    ``state_dict``); ``_sw_to_dense`` densifies each so the round-trip can be
    checked bit-for-bit over the DISTINCT physical blocks.
    """
    snap = {"embed": model.embed.detach().clone(),
            "lm_head": model.lm_head.detach().clone(),
            "lm_bias": model.lm_bias.detach().clone()}
    for bi, blk in enumerate(_phys_blocks(model)):
        a, f = blk.attn, blk.ffn
        for nm, sw in [("Wq", a.W_q), ("Wk", a.W_k), ("Wv", a.W_v), ("Wo", a.W_o),
                       ("Wup", f.W_up), ("Wgate", f.W_gate), ("Wdown", f.W_down)]:
            snap[f"b{bi}.{nm}"] = _sw_to_dense(sw).clone()
        snap[f"b{bi}.slopes"] = a.alibi_slopes.detach().clone()
        snap[f"b{bi}.b_up"] = f.b_up.detach().clone()
        snap[f"b{bi}.b_gate"] = f.b_gate.detach().clone()
        snap[f"b{bi}.b_down"] = f.b_down.detach().clone()
    return snap


def test_weights_coo_roundtrip_byte_identical(model_L):
    """serialize -> deserialize reconstructs every weight bit-for-bit (L-inf=0)."""
    model, L = model_L
    blob, stats = _serialize_weights(model, L)
    sd1 = _snapshot(model)
    model2, L2 = _build_model(_CFG)
    _deserialize_weights(blob, model2, L2)
    sd2 = _snapshot(model2)
    assert set(sd1) == set(sd2)
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), f"tensor {k} differs after round-trip"
    # the COO blob is much smaller than the dense equivalent (the size lever).
    assert stats["weights_bytes"] < stats["dense_equiv_bytes"] // 100


def test_assemble_container_framing(tmp_path, model_L):
    """The .c4bundle is a well-formed flat container with 3 slotted sections."""
    model, L = model_L
    out = str(tmp_path / "add.c4bundle")
    info = assemble_bundle(_SRC, out, expected=1200, reuse_model=(model, L),
                           **_CFG)
    assert os.path.getsize(out) == info["bundle_bytes"]
    with open(out, "rb") as fh:
        data = fh.read()
    assert data[:len(MAGIC)] == MAGIC
    hdr, _ = read_header(out)
    # sections are contiguous, in order, and cover the whole file after the header.
    r_off, r_len = hdr.sections["runtime"]
    w_off, w_len = hdr.sections["weights"]
    b_off, b_len = hdr.sections["bytecode"]
    assert r_off < w_off < b_off
    assert w_off == r_off + r_len
    assert b_off == w_off + w_len
    assert b_off + b_len == len(data)
    # bytecode section decodes to the same instruction count the compiler emits.
    code, _data_seg = _compile_source_to_bytecode(_SRC)
    bc = _section(data, hdr, "bytecode")
    (n,) = struct.unpack_from("<I", bc, 0)
    assert n == len(code)


def test_c4c_bundler_produces_byte_identical_bundle(tmp_path, model_L):
    """The C4-C bundler fuses prepared parts into the SAME bytes as Python."""
    if CC is None:
        pytest.skip("no C compiler")
    model, L = model_L
    ref = str(tmp_path / "ref.c4bundle")
    assemble_bundle(_SRC, ref, expected=1200, reuse_model=(model, L), **_CFG)
    parts_dir = str(tmp_path / "parts")
    prepare_bundle(_SRC, parts_dir, expected=1200, reuse_model=(model, L), **_CFG)
    # compile the C4-C bundler and fuse.
    binr = str(tmp_path / "c4bnd")
    c = subprocess.run([CC, "-w", "-o", binr, _C4C], capture_output=True,
                       text=True, timeout=120)
    if c.returncode != 0:                 # some sandboxes lack -lgcc_s etc.
        c = subprocess.run([CC, "-w", "-static", "-o", binr, _C4C],
                           capture_output=True, text=True, timeout=120)
    assert c.returncode == 0, c.stderr
    fused = str(tmp_path / "c4c.c4bundle")
    with open(fused, "wb") as fh:
        r = subprocess.run([binr, os.path.join(parts_dir, "manifest.txt")],
                           stdout=fh, stderr=subprocess.PIPE, timeout=120)
    assert r.returncode == 0, r.stderr
    with open(ref, "rb") as a, open(fused, "rb") as b:
        assert a.read() == b.read(), "C4-C bundle differs from python bundle"


@pytest.mark.slow
def test_bundle_runs_end_to_end(tmp_path, model_L):
    """Assemble + run an ARITH bundle through the bundled model; result matches."""
    model, L = model_L
    out = str(tmp_path / "add.c4bundle")
    assemble_bundle(_SRC, out, expected=1200, reuse_model=(model, L), **_CFG)
    res = run_bundle(out, verbose=False)
    assert res["status"] == "PASS", res
    assert res["got"] == 1200


@pytest.mark.slow
def test_bundle_printf_runs_end_to_end_byte_exact(tmp_path):
    """Assemble + run a PRINTF bundle; stdout is captured byte-exact.

    Exercises the I/O path: the data segment (the ``"hi\\n"`` literal) rides in
    the header, ``run_bundle`` wires an fio sink at ``mask=0xFF``, and the
    verdict is byte-exact stdout, not the register value.
    """
    src = 'int main(){ printf("hi\\n"); return 0; }'
    out = str(tmp_path / "printf.c4bundle")
    # printf needs the low-256 pointer window -> code_size 64 (its own build).
    info = assemble_bundle(src, out, expected_stdout=b"hi\n",
                           description="printf", code_size=64, step_cap=400)
    assert info["config"]["code_size"] == 64
    res = run_bundle(out, verbose=False)
    assert res["status"] == "PASS", res
    assert res["stdout"] == "hi\n", res
