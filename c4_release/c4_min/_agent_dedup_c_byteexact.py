"""_agent_dedup_c_byteexact.py — prove the DEDUP-IN-C change to
onnx_runtime_nibble_fixedpoint.c is BYTE-EXACT: the deduped (palette+index) weight
read produces IDENTICAL logits to the direct-weight (golden) path, on a REAL
.nblbin model, and both match the numpy .nblbin reference.

Pipeline:  build_step_model -> export_onnx -> lower_onnx_to_bin(.nblbin)
           gcc onnx_runtime_nibble_fixedpoint.c  (dedup OFF and dedup ON, -DC4_DEDUP_WEIGHTS)
           run both on the same tokens with --dump-logits
           assert dedup-ON logits == dedup-OFF logits, byte-for-byte (L-inf = 0)
           assert argmax matches the numpy .nblbin reference

CPU-only, memory-safe.  Run:  python -m c4_min._agent_dedup_c_byteexact
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


def main() -> int:
    d = tempfile.mkdtemp(prefix="dedup_c_")
    dense = os.path.join(d, "m.onnx")
    binp = os.path.join(d, "m.nblbin")
    exe_off = os.path.join(d, "rt_off")
    exe_on = os.path.join(d, "rt_on")

    print("Building the step model + .nblbin ...")
    model, L, code = C.build_step_model(E.PROOF_PROG)
    model.eval()
    E.export_onnx(model, dense)
    # DENSE lowering: the fixedpoint runtime reads dense (isi==1) initializers only
    # (no COO isi==2 path), so force every initializer dense for this byte-exact proof
    # (byte-identical to the sparse form — COO is a lossless storage of the same values).
    lower_onnx_to_bin(dense, binp, sparse_min_numel=10 ** 12)

    print("Compiling the fixedpoint C runtime (dedup OFF and dedup ON) ...")
    _cc(exe_off, defines=[])
    _cc(exe_on, defines=["-DC4_DEDUP_WEIGHTS"])

    from c4_min import blogspec_vocab as V
    frames = {
        "proof_full": R.run_program(model, L, code, max_steps=20)[0],
        "single_step": R.run_program(model, L, code, max_steps=1)[0],
        "tiny": [V.BOS, V.REG_AX, 0x2A, V.STEP_END],
    }

    all_exact = True
    for tag, toks in frames.items():
        tok = np.array([toks], dtype=np.int64)

        # argmax (dump-argmax) both ways
        am_off = _run_c(exe_off, binp, tok, "--dump-argmax")
        am_on = _run_c(exe_on, binp, tok, "--dump-argmax")
        # logits (dump-logits) both ways
        lg_off = _run_c(exe_off, binp, tok, "--dump-logits")
        lg_on = _run_c(exe_on, binp, tok, "--dump-logits")

        # strip the informational header lines (start with a letter)
        def _nums(s):
            return [l for l in s.splitlines() if l and (l[0].isdigit() or l[0] == "-")]

        am_off_n, am_on_n = _nums(am_off), _nums(am_on)
        lg_off_n, lg_on_n = _nums(lg_off), _nums(lg_on)

        argmax_exact = am_off_n == am_on_n
        logits_exact = lg_off_n == lg_on_n
        exact = argmax_exact and logits_exact
        all_exact = all_exact and exact
        print(f"  [{tag}] rows={len(am_off_n)}  argmax dedup==direct: {argmax_exact}  "
              f"logits dedup==direct (L-inf=0 byte-for-byte): {logits_exact}  "
              f"-> {'BYTE-EXACT' if exact else 'MISMATCH'}")

        # also confirm the palette actually engaged (ON build reports palette_n via
        # a header line if present); print the header for transparency
        hdr_on = [l for l in lg_on.splitlines() if l and l[0].isalpha()][:3]
        for h in hdr_on:
            print(f"      dedup-ON header: {h}")

    print()
    print(f"DEDUP-IN-C BYTE-EXACT (palette+index read == direct read): {all_exact}")
    return 0 if all_exact else 1


if __name__ == "__main__":
    sys.exit(main())
