#!/usr/bin/env python3
"""Localize the no-STACK0 (30-token) register-frame emission bug.

Teacher-forces the oracle's correct token sequence (built by DraftVM with the
SAME Token.STEP_TOKENS the model bakes for) and at every position reads the
model's argmax. Reports each (step, offset) where the model's argmax disagrees
with the teacher-forced (oracle) token. spec_k=0, hook-free.

Run with C4_NO_STACK0_EMIT=1 to see the frame break; without it to confirm the
35-token frame is clean.

Usage:
    CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 python tools/probe_nostack0_frame.py
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import torch  # noqa: E402

from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402
from neural_vm.vm_step import Token  # noqa: E402
from neural_vm.speculative import DraftVM  # noqa: E402


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src)
    return bc, exp, desc


def _tok_name(t):
    names = {
        256: "SEP", 257: "REG_PC", 258: "REG_AX", 259: "REG_SP",
        260: "REG_BP", 261: "MEM", 262: "STEP_END", 263: "HALT",
        268: "STACK0",
    }
    if t in names:
        return names[t]
    if t < 256:
        return f"byte:0x{t:02x}({t})"
    return f"tok{t}"


def _offset_name(off, step_tokens):
    if step_tokens == 30:
        starts = [(0, "PC"), (5, "AX"), (10, "SP"), (15, "BP"), (20, "MEM")]
        if off == 29:
            return "STEP_END/HALT"
        if off >= 21:
            if off <= 24:
                return f"MEM_ADDR[{off-21}]"
            return f"MEM_VAL[{off-25}]"
    else:
        starts = [(0, "PC"), (5, "AX"), (10, "SP"), (15, "BP"),
                  (20, "STACK0"), (25, "MEM")]
        if off == 34:
            return "STEP_END/HALT"
        if off >= 26:
            if off <= 29:
                return f"MEM_ADDR[{off-26}]"
            return f"MEM_VAL[{off-30}]"
    for s, nm in reversed(starts):
        if off >= s:
            rel = off - s
            return f"{nm}_marker" if rel == 0 else f"{nm}[{rel-1}]"
    return f"off{off}"


def _oracle_tokens(bc, n_steps):
    """Teacher-forced reference token stream from DraftVM."""
    vm = DraftVM(list(bc))
    toks = []
    for _ in range(n_steps):
        if not vm.step():
            break
        toks.append(vm.draft_tokens())
        if vm.halted:
            break
    return toks


@torch.no_grad()
def main(ids, n_steps):
    probe = build_groundtruth_probe()
    model = probe.model
    dev = next(model.parameters()).device
    ST = Token.STEP_TOKENS
    print(f"### Token.STEP_TOKENS = {ST}")

    for idx in ids:
        bc, exp, desc = _corpus(idx)
        prompt = probe._build_context(bc)
        step_toks = _oracle_tokens(bc, n_steps)
        flat = [t for s in step_toks for t in s]
        # full teacher-forced context = prompt + all oracle step tokens
        context = list(prompt) + flat
        padded = torch.tensor([context], dtype=torch.long, device=dev)
        logits = model.forward(padded)[0]  # [S, V]
        argmax = logits.argmax(dim=-1).tolist()

        print(f"\n===== id={idx} {desc[:50]} exp={exp} | "
              f"{len(step_toks)} steps, prompt_len={len(prompt)} =====")
        prompt_len = len(prompt)
        first_div = None
        for si, st in enumerate(step_toks):
            line = []
            step_bad = False
            for off, expected in enumerate(st):
                pos = prompt_len + si * ST + off
                # model PREDICTS the token at pos given context[:pos]
                # i.e. logits at index pos-1 predict token at pos.
                pred = argmax[pos - 1]
                ok = (pred == expected)
                if not ok:
                    step_bad = True
                    if first_div is None:
                        first_div = (si, off, expected, pred)
                    line.append(
                        f"  [{_offset_name(off, ST)}] exp={_tok_name(expected)}"
                        f" got={_tok_name(pred)}")
            tag = "BAD " if step_bad else "ok  "
            print(f"  step {si} {tag}" + ("".join(line) if step_bad else ""))
        if first_div:
            si, off, e, p = first_div
            print(f"  >>> FIRST DIVERGENCE: step {si} offset {off} "
                  f"({_offset_name(off, ST)}) exp={_tok_name(e)} got={_tok_name(p)}")
        else:
            print("  >>> NO DIVERGENCE (frame clean)")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:] if x.isdigit()]
    n_steps = 4
    ids = args or [0]  # default: corpus program 0
    main(ids, n_steps)
