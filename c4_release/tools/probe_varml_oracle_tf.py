#!/usr/bin/env python3
"""DECISIVE per-step isolation: teacher-force the TRUE ORACLE 35-token context
(from the DraftVM, NOT the model's free-run) and read the model's argmax AX
emission at each LEA/LI step. With perfectly-clean input, a per-step bug shows;
a cross-step BP-noise bug disappears (the oracle BP frame is clean).

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_varml_oracle_tf.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
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
from neural_vm.speculative import DraftVM  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

CASES = [
    ("var_mul", "int main() { int a; int b; a = 23; b = 47; return a * b; }"),
    ("var_update", "int main() { int x; x = 50; x = x + 7; return x; }"),
    ("var_three", "int main() { int a; int b; int c; a = 29; b = 6; c = 20; return a + b + c; }"),
]


def oracle_windows(bc):
    vm = DraftVM(list(bc))
    steps, toks = [], []
    for _ in range(40):
        if vm.halted:
            break
        if not vm.step():
            break
        steps.append((int(vm.pc) & 0xFFFFFFFF, int(vm.ax) & 0xFFFF, "?"))
        toks.append([int(t) for t in vm.draft_tokens()])
        if vm.halted:
            break
    return steps, toks


def main():
    p = build_groundtruth_probe()
    STEP = int(Token.STEP_TOKENS)
    for name, src in CASES:
        bc, _ = compile_c(src)
        prompt = p._build_context(bc)
        pl = len(prompt)
        steps, toks = oracle_windows(bc)
        # Build the FULL teacher-forced oracle context: prompt + every window.
        ctx = list(prompt)
        for w in toks:
            ctx.extend(w)
        padded = torch.tensor([ctx], device=p._device)
        with torch.no_grad():
            logits = p.model(padded)
            if logits.is_sparse:
                logits = logits.to_dense()
        preds = torch.argmax(logits[0], dim=-1)
        print(f"\n===== {name} (oracle teacher-forced) =====")
        for si, (pc, ax, opn) in enumerate(steps):
            base = pl + si * STEP
            ax_bytes = [int(preds[base + 5 + j]) & 0xFF for j in range(4)]
            got = sum(b << (8 * j) for j, b in enumerate(ax_bytes)) & 0xFFFF
            ctx_ax = [ctx[base + 6 + j] & 0xFF for j in range(4)]
            ctx_axv = sum(b << (8 * j) for j, b in enumerate(ctx_ax)) & 0xFFFF
            tag = "" if got == ax else "  **STEP-BUG** (clean ctx, wrong emit)"
            print(f"  ostep{si:2d} want_ax=0x{ax:04x} "
                  f"model_emit=0x{got:04x} oracle_in_ctx=0x{ctx_axv:04x}{tag}")


if __name__ == "__main__":
    main()
