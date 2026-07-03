#!/usr/bin/env python3
"""Localize the loop_sum id450 step-9 SP-value-byte framing over-run.

Step 9 (the `SI sum=0` store) emits 41 tokens not 30: after the SP marker(259) +
byte0=0xE8(232), the model predicts a PC marker(257) instead of SP byte1. That
11-token over-run desyncs the fixed-30 slicer -> step 10 reads pc=114 not 106.

This probe replays to the divergence, finds the exact abs position where token
257 was (wrongly) emitted, and ranks the LM-head logits there + does a per-block
residual trace of the winning logit's driver. We compare to the analogous
SP-value position at a CLEAN step (step 7) to see what's different.

Usage: CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       C4_VM_CACHE_DIR=/tmp/c4cache_loopdesync python tools/_probe_loopsum_spdrift.py
"""
from __future__ import annotations
import os, sys
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch  # noqa
from neural_vm.verification.faithful_autoregressive import build_cpu_model  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END); STEP = int(Token.STEP_TOKENS)
PC = int(Token.REG_PC); AX = int(Token.REG_AX); SP = int(Token.REG_SP); BP = int(Token.REG_BP)


class _St:
    def __init__(self, m): self.model = m


def bic(model, bc):
    from neural_vm.run_vm import AutoregressiveVMRunner
    return list(AutoregressiveVMRunner._build_context(_St(model), list(bc), b"", [], ""))


@torch.no_grad()
def replay(model, ctx, ms):
    dev = next(model.parameters()).device
    ctx = list(ctx)
    for _ in range(ms * STEP):
        p = torch.tensor([ctx], dtype=torch.long, device=dev)
        l = model.forward(p)[0]
        ctx.append(int(l[len(ctx) - 1].argmax().item()))
        if ctx.count(SE) >= ms: break
    return ctx


@torch.no_grad()
def main():
    with contextlib.redirect_stderr(io.StringIO()):
        model, layout = build_cpu_model(disk_cache=True)
    dev = "cuda" if (torch.cuda.is_available()
                     and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu"
    model = model.to(dev); model.eval()
    dp = layout.dim_positions
    src, exp, desc = generate_test_programs()[450]
    bc = compile_c(src)[0]
    plc = bic(model, bc); pl = len(plc)
    ctx = replay(model, plc, 14)
    se_pos = [i for i, t in enumerate(ctx) if t == SE and i >= pl]
    bounds = [pl] + [p + 1 for p in se_pos]

    def step_span(st):
        return bounds[st], (bounds[st + 1] if st + 1 < len(bounds) else len(ctx))

    # In step 9, find the abs position whose PREDICTED token is the spurious 257
    a9, b9 = step_span(9)
    print(f"step9 span abs[{a9}:{b9}] len={b9 - a9}")
    # The spurious 257 is the token AFTER the SP marker+1byte. Find first 259 (SP)
    sp_marker = None
    for i in range(a9, b9):
        if ctx[i] == SP:
            sp_marker = i; break
    print(f"step9 SP marker at abs {sp_marker}; tokens after: {ctx[sp_marker:sp_marker+8]}")
    # The position that PREDICTED the spurious 257: it's the position whose
    # argmax produced ctx[sp_marker+2] (=257). Predicting token at index k means
    # the logits row is at index k-1.
    drift_tok_idx = sp_marker + 2  # the spurious 257
    pred_row = drift_tok_idx - 1   # logits row that emitted it
    print(f"spurious token ctx[{drift_tok_idx}]={ctx[drift_tok_idx]} (expect SP byte1)")
    print(f"  predicted FROM logits row {pred_row} (token there = ctx[{pred_row}]={ctx[pred_row]})")

    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    logits = model.forward(toks)[0]
    row = logits[pred_row]
    top = torch.topk(row, 8)
    print("  LM-head top-8 logits at the drift row:")
    for v, idx in zip(top.values.tolist(), top.indices.tolist()):
        tag = ""
        if idx == PC: tag = " <PC marker>"
        elif idx == SP: tag = " <SP marker>"
        elif idx == AX: tag = " <AX marker>"
        elif idx < 256: tag = f" <byte 0x{idx:02x}>"
        print(f"    tok {idx:4d} logit {v:9.3f}{tag}")

    # Compare to the analogous CLEAN SP-value-byte1 row at step 7
    a7, b7 = step_span(7)
    sp7 = None
    for i in range(a7, b7):
        if ctx[i] == SP:
            sp7 = i; break
    clean_row = sp7 + 1  # row that predicts SP byte1 (the next token after byte0)
    print(f"\n[CLEAN step7] SP marker abs {sp7}, tokens {ctx[sp7:sp7+6]}")
    rowc = logits[clean_row]
    topc = torch.topk(rowc, 6)
    print("  step7 SP byte1-predict row top-6:")
    for v, idx in zip(topc.values.tolist(), topc.indices.tolist()):
        tag = " <PC>" if idx == PC else (" <SP>" if idx == SP else (f" <0x{idx:02x}>" if idx < 256 else ""))
        print(f"    tok {idx:4d} logit {v:9.3f}{tag}")
    print(f"\n  step9 drift-row PC-vs-bytemargin: PC={float(row[PC]):.2f} "
          f"best_byte={float(row[:256].max()):.2f}(tok{int(row[:256].argmax())})")


if __name__ == "__main__":
    main()
