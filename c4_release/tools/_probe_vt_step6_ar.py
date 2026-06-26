#!/usr/bin/env python3
"""CPU AR block-level probe for var_three step-6 LEA &b byte-0 (0xE8 stale vs
0xE0 correct). Replays the model's OWN autoregressive decode (no oracle TF) so
the probed context is exactly what the production AR decode sees, then dumps the
ALU/OUTPUT/FETCH bands at the step-6 LEA byte-0 predictor row block-by-block.

Run: CUDA_VISIBLE_DEVICES="" python tools/_probe_vt_step6_ar.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.unified_compiler.faithful_autoregressive import build_cpu_model  # noqa

SRC = "int main() { int a; int b; int c; a = 29; b = 6; c = 20; return a + b + c; }"
# step -> (label, want_ax)
TARGETS = {2: ("LEA &a (ok)", 0xffe8), 6: ("LEA &b WRONG", 0xffe0),
           10: ("LEA &c WRONG", 0xffd8)}


def main():
    model, layout = build_cpu_model(disk_cache=True)
    dp = dict(layout.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    print("STEP_TOKENS =", STEP)

    def nib(vec, name):
        x = dp[name]
        i = int(torch.argmax(vec[x:x + 16]))
        return i, float(vec[x + i])

    def byteval(vec, lon, hin):
        l, lv = nib(vec, lon)
        h, hv = nib(vec, hin)
        return h * 16 + l, lv, hv

    # ---- AR-decode to collect emitted windows (the production tape) ----
    from neural_vm.unified_compiler.faithful_autoregressive import (
        FaithfulAutoregressiveRunner,
    )
    runner = FaithfulAutoregressiveRunner(model=model, layout=layout)
    bc, data = compile_c(SRC)

    # Build prompt prefix exactly as the runner does.
    inner = runner._inner
    prompt = inner._serial._build_context(list(bc), b"", [], "")
    pl = len(prompt)

    # Manually AR-decode using the faithful forward to capture the tape.
    ctx = list(prompt)
    max_emit = 7 * STEP + 16  # through step 6 window + a bit
    emitted = []
    while len(emitted) < max_emit:
        padded = torch.tensor([ctx + emitted], dtype=torch.long)
        with torch.no_grad():
            logits = model.forward(padded)
            if logits.is_sparse:
                logits = logits.to_dense()
        nxt = int(torch.argmax(logits[0, -1]))
        emitted.append(nxt)
    full_ctx = ctx + emitted
    padded = torch.tensor([full_ctx], dtype=torch.long)
    print(f"prompt_len={pl} emitted={len(emitted)} total={len(full_ctx)}")

    n_blocks = len(model.blocks)
    print(f"n_blocks={n_blocks}")
    probe_blocks = [8, 11, 13, 19, 27, 32, 36, 38, 40, n_blocks - 1]
    probe_blocks = [b for b in probe_blocks if b < n_blocks]

    for ost, (label, want) in TARGETS.items():
        print(f"\n===== step {ost} {label} want_ax=0x{want:04x} =====")
        for offf in (5, 6):  # byte0 / byte1 predictor rows
            row = pl + ost * STEP + offf
            if row >= len(full_ctx):
                print(f"  off+{offf}: row {row} out of range (len {len(full_ctx)})")
                continue
            print(f"  -- addr byte{offf-5} predictor row {row} --")
            for blk in probe_blocks:
                with torch.no_grad():
                    full = model.forward(padded, stop_after_block=blk)
                    if full.is_sparse:
                        full = full.to_dense()
                r = full[0, row]
                alu, _, _ = byteval(r, "ALU_LO", "ALU_HI")
                out, olv, ohv = byteval(r, "OUTPUT_LO", "OUTPUT_HI")
                fe, _, _ = byteval(r, "FETCH_LO", "FETCH_HI")
                # OUTPUT_LO nibble winner + the 0 vs 8 contest
                xlo = dp["OUTPUT_LO"]
                v0 = float(r[xlo + 0]); v8 = float(r[xlo + 8])
                print(f"    b{blk:2d}: ALU{alu:02x} OUT{out:02x} FE{fe:02x} "
                      f"| OUTLO[0]={v0:+.3f} OUTLO[8]={v8:+.3f} "
                      f"(olv={olv:+.2f} ohv={ohv:+.2f})")


if __name__ == "__main__":
    main()
