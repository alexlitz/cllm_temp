#!/usr/bin/env python3
"""Pinpoint the late-block (41..61) writer that flips var_three step-6 LEA &b
byte-0 from the correct 0xE0 to 0xE8. Reuses the AR tape; scans every block
40..61 at the step-6 byte0 row and dumps ALU_HI+15, FETCH bands, OUTPUT_LO[0/8].
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
from neural_vm.unified_compiler.faithful_autoregressive import (  # noqa: E402
    build_cpu_model, FaithfulAutoregressiveRunner,
)

SRC = "int main() { int a; int b; int c; a = 29; b = 6; c = 20; return a + b + c; }"


def main():
    model, layout = build_cpu_model(disk_cache=True)
    dp = dict(layout.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    runner = FaithfulAutoregressiveRunner(model=model, layout=layout)
    bc, _ = compile_c(SRC)
    inner = runner._inner
    prompt = inner._serial._build_context(list(bc), b"", [], "")
    pl = len(prompt)
    ctx = list(prompt)
    emitted = []
    target_len = 7 * STEP + 6 + 2
    while len(emitted) < target_len:
        padded = torch.tensor([ctx + emitted], dtype=torch.long)
        with torch.no_grad():
            logits = model.forward(padded)
            if logits.is_sparse:
                logits = logits.to_dense()
        emitted.append(int(torch.argmax(logits[0, -1])))
    full = ctx + emitted
    padded = torch.tensor([full], dtype=torch.long)
    nb = len(model.blocks)

    def nibmax(vec, name):
        x = dp[name]; i = int(torch.argmax(vec[x:x + 16])); return i

    # step 6 (&b, want 0xE0): scan ONLY the L25 tail/post-op range
    for ost, want in ((6, 0xE0),):
        row = pl + ost * STEP + 5  # byte0 predictor
        print(f"\n=== step {ost} byte0 row {row} want low byte 0x{want:02x} ===", flush=True)
        prev = None
        for blk in range(40, nb):
            with torch.no_grad():
                fr = model.forward(padded, stop_after_block=blk)
                if fr.is_sparse:
                    fr = fr.to_dense()
            r = fr[0, row]
            olo = nibmax(r, "OUTPUT_LO"); ohi = nibmax(r, "OUTPUT_HI")
            ah15 = float(r[dp["ALU_HI"] + 15])
            felo = nibmax(r, "FETCH_LO"); fehi = nibmax(r, "FETCH_HI")
            v0 = float(r[dp["OUTPUT_LO"] + 0]); v8 = float(r[dp["OUTPUT_LO"] + 8])
            byte = ohi * 16 + olo
            tag = ""
            if prev is not None and prev != byte:
                tag = f"  <<< CHANGED from 0x{prev:02x}"
            print(f"  b{blk:2d}: OUT=0x{byte:02x} OUTLO[0]={v0:+11.1f} "
                  f"OUTLO[8]={v8:+11.1f} | ALU_HI+15={ah15:+8.2f} "
                  f"FE_lo={felo:x} FE_hi={fehi:x}{tag}", flush=True)
            prev = byte


if __name__ == "__main__":
    main()
