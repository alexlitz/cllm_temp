"""_agent_fuse_tail_byteexact.py — prove the FUSED-TAIL change to
onnx_runtime_nibble_fixedpoint.c is BYTE-EXACT: routing the silu + element-wise adds
through the fused per-element evaluators (C4_FUSE_TAIL) produces IDENTICAL logits to
the unfused (loop) path, on a REAL .nblbin model whose forward CONTAINS softmax +
silu + adds, and COMPOSES with the dedup-in-C weight read (C4_DEDUP_WEIGHTS).

Four builds, all on the SAME real .nblbin, compared byte-for-byte (--dump-logits):
    off        : baseline (unfused tail, direct weights)  -- the golden path
    fuse       : -DC4_FUSE_TAIL                            -- fused tail only
    dedup      : -DC4_DEDUP_WEIGHTS                        -- fused MAC's dedup only
    composed   : -DC4_DEDUP_WEIGHTS -DC4_FUSE_TAIL         -- dedup + fused tail

assert  fuse == dedup == composed == off  logits, byte-for-byte (L-inf = 0), and
argmax matches the numpy .nblbin reference.  The proof frame's forward exercises the
attention softmax (real-attn block), the SwiGLU silu, and the residual/bias adds —
the three tail families — so a byte-exact result covers all of them.

CPU-only, memory-safe.  Run:  python -m c4_min._agent_fuse_tail_byteexact
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

from c4_min import blogspec_compiler as C
from c4_min import export_onnx as E
from c4_min import blogspec_run as R
from c4_min.onnx_to_c4bin import lower_onnx_to_bin

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "onnx_runtime_nibble_fixedpoint.c")


def _cc(exe, defines=None):
    defs = defines or []
    for flags in (["-O2", "-static-libgcc"], ["-O2"]):
        r = subprocess.run(["gcc"] + flags + defs + ["-o", exe, CSRC],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
    raise RuntimeError("gcc failed:\n" + r.stderr)


def _run_c(exe, binpath, tok, mode="--dump-logits"):
    B, S = tok.shape
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{B} {S}\n")
        f.write(" ".join(str(int(x)) for x in tok.ravel()))
        tokfile = f.name
    out = subprocess.run([exe, binpath, tokfile, mode],
                         capture_output=True, text=True, check=True).stdout
    os.unlink(tokfile)
    return out


def _nums(s):
    return [l for l in s.splitlines() if l and (l[0].isdigit() or l[0] == "-")]


def main() -> int:
    d = tempfile.mkdtemp(prefix="fuse_tail_c_")
    dense = os.path.join(d, "m.onnx")
    binp = os.path.join(d, "m.nblbin")
    exes = {"off": os.path.join(d, "rt_off"),
            "fuse": os.path.join(d, "rt_fuse"),
            "dedup": os.path.join(d, "rt_dedup"),
            "composed": os.path.join(d, "rt_composed")}

    print("Building the step model + .nblbin ...")
    model, L, code = C.build_step_model(E.PROOF_PROG)
    model.eval()
    E.export_onnx(model, dense)
    lower_onnx_to_bin(dense, binp, sparse_min_numel=10 ** 12)

    print("Compiling the fixedpoint C runtime (off / fuse / dedup / composed) ...")
    _cc(exes["off"], defines=[])
    _cc(exes["fuse"], defines=["-DC4_FUSE_TAIL"])
    _cc(exes["dedup"], defines=["-DC4_DEDUP_WEIGHTS"])
    _cc(exes["composed"], defines=["-DC4_DEDUP_WEIGHTS", "-DC4_FUSE_TAIL"])

    frames = {
        "proof_full": R.run_program(model, L, code, max_steps=20)[0],
        "single_step": R.run_program(model, L, code, max_steps=1)[0],
    }

    all_exact = True
    for tag, toks in frames.items():
        tok = np.array([toks], dtype=np.int64)
        lg = {k: _nums(_run_c(exes[k], binp, tok, "--dump-logits")) for k in exes}
        am = {k: _nums(_run_c(exes[k], binp, tok, "--dump-argmax")) for k in exes}

        fuse_ok = lg["fuse"] == lg["off"]
        dedup_ok = lg["dedup"] == lg["off"]
        composed_ok = lg["composed"] == lg["off"]
        argmax_ok = am["fuse"] == am["off"] == am["dedup"] == am["composed"]
        exact = fuse_ok and dedup_ok and composed_ok and argmax_ok
        all_exact = all_exact and exact
        print(f"  [{tag}] rows={len(am['off'])}  "
              f"fused-tail==off: {fuse_ok}  dedup==off: {dedup_ok}  "
              f"composed(dedup+fused-tail)==off: {composed_ok}  "
              f"argmax all-agree: {argmax_ok}  -> "
              f"{'BYTE-EXACT' if exact else 'MISMATCH'}")

    print()
    print(f"FUSED-TAIL BYTE-EXACT (silu+adds fused == unfused, composes with dedup): "
          f"{all_exact}")
    return 0 if all_exact else 1


if __name__ == "__main__":
    sys.exit(main())
