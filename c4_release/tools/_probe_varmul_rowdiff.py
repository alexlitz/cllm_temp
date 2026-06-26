#!/usr/bin/env python3
"""var_mul id278: diff EVERY residual dim between a's value-byte row (331, val=8)
and b's value-byte row (451, val=25) at L15-in (block 34). Any dim that DIFFERS
(beyond CLEAN_EMBED of the value itself) is a candidate discriminator the #313
CAM could key. If the only differences are the value bytes + position, the
address provenance is genuinely absent from the frame (deep wall).

Run: CUDA_VISIBLE_DEVICES=0 C4_VM_CACHE_DIR=/tmp/c4cache_vmcam \
     python tools/_probe_varmul_rowdiff.py
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.unified_compiler.faithful_autoregressive import build_cpu_model  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402

SE = int(Token.STEP_END); STEP = int(Token.STEP_TOKENS)


class _CtxStub:
    def __init__(self, model): self.model = model


def bic(model, bc):
    from neural_vm.run_vm import AutoregressiveVMRunner
    return list(AutoregressiveVMRunner._build_context(_CtxStub(model), list(bc), b"", [], ""))


@torch.no_grad()
def replay(model, ctx, max_steps):
    dev = next(model.parameters()).device
    ctx = list(ctx)
    for _ in range(max_steps * STEP):
        t = torch.tensor([ctx], dtype=torch.long, device=dev)
        ctx.append(int(model.forward(t)[0][len(ctx) - 1].argmax().item()))
        if ctx.count(SE) >= max_steps:
            break
    return ctx


@torch.no_grad()
def main():
    with contextlib.redirect_stderr(io.StringIO()):
        model, layout = build_cpu_model(disk_cache=True)
    dev = ("cuda" if (torch.cuda.is_available()
                      and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu")
    model = model.to(dev).eval()
    dp = layout.dim_positions
    inv = {v: k for k, v in dp.items()}
    progs = generate_test_programs()
    bc = compile_c(progs[278][0])[0]
    ctx = replay(model, bic(model, bc), 15)
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    x = model.forward(toks, stop_after_block=34)[0]

    # Compare a's value-byte-VB1 row (331) vs b's (451), and a's value-byte-VB0
    # row (330) vs b's (450), at L15-in.
    for ra, rb, tag in ((331, 451, "VB1 (a=8 vs b=25)"), (330, 450, "VB0 (both 0)")):
        print(f"\n=== diff row{ra} vs row{rb}  [{tag}] (|delta|>0.05, named dims) ===")
        diffs = []
        for d in range(x.shape[1]):
            va = float(x[ra, d]); vb = float(x[rb, d])
            if abs(va - vb) > 0.05:
                # find dim name (closest base)
                name = inv.get(d, "")
                if not name:
                    for base, n in dp.items():
                        if 0 <= d - n < 16:
                            name = f"{base}+{d-n}"; break
                diffs.append((abs(va - vb), d, name, round(va, 2), round(vb, 2)))
        diffs.sort(reverse=True)
        for delta, d, name, va, vb in diffs[:40]:
            print(f"  dim{d:4d} {name:24s} a={va:8.2f} b={vb:8.2f} |d|={delta:.2f}")
        if not diffs:
            print("  (NO dim differs -> rows are identical except position)")


if __name__ == "__main__":
    main()
