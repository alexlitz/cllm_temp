#!/usr/bin/env python3
"""Pin WHICH tail_bit32_result_correction rule writes OUTPUT cell 8 / cell 14
(0xE8) on a plain ``IMM v; EXIT`` AX decode row at block 36.

spec_k=0, hook-free.  Reads the residual at block 35 (tail op input) and
scores every tail rule via the exact symbolic firing math.
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch

from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.unified_compiler.ops.l10_ops import _tail_bit32_result_correction_rules


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            bc.append(op[0] | (op[1] << 8))
        else:
            bc.append(op)
    return bc


def find_ax_row(model, dp, dev, ctx):
    S = len(ctx)
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        emb = model.embed(toks)[0]
    axb, seb = dp["MARK_AX"], dp["MARK_SE_ONLY"]
    axr = [r for r in range(S) if emb[r, axb].abs().item() > 0.5]
    ser = [r for r in range(S) if emb[r, seb].abs().item() > 0.5]
    se = ser[-1] if ser else None
    if not axr:
        return S - 1
    return max((r for r in axr if (se is None or r < se)), default=axr[-1])


def build_state(row, dp, D):
    state = {}
    for name, base in dp.items():
        b = int(base)
        for k in range(16):
            if 0 <= b + k < D:
                state[f"{name}+{k}"] = float(row[b + k].item())
        if 0 <= b < D:
            state[name] = float(row[b].item())
    return state


def score_rules(rules, state, targets):
    out = []
    for rule in rules:
        score = 0.0
        for term in rule.conditions:
            score += state.get(term.dim.key(), 0.0) * term.weight
        fired = score >= rule.threshold
        gate_value = rule.gate_bias
        if rule.gate is not None:
            gate_value += state.get(rule.gate.key(), 0.0) * rule.gate_weight
        for term in rule.gate_terms:
            gate_value += state.get(term.dim.key(), 0.0) * term.weight
        contribs = {}
        for write in rule.writes:
            key = write.dim.key()
            if key in targets:
                contribs[key] = gate_value * write.weight
        if fired and contribs and any(abs(v) > 1e-6 for v in contribs.values()):
            out.append((rule.name, round(score, 2), round(rule.threshold, 2),
                        {k: round(v, 3) for k, v in contribs.items()}))
    return out


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    D = len(model.embed(torch.tensor([[0]], device=dev))[0, 0])

    rules = _tail_bit32_result_correction_rules()
    print(f"tail_bit32 rules: {len(rules)}")
    targets = {"OUTPUT_LO+8", "OUTPUT_HI_THIS_STEP+14"}

    for v in (0xFF, 0x18, 0x80):
        bc = _mk([(Opcode.IMM, v), Opcode.EXIT])
        ctx = probe._final_context(bc, max_steps=10)
        ax = find_ax_row(model, dp, dev, ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            r = model.forward(toks, stop_after_block=35)[0]
        row = r[ax]
        state = build_state(row, dp, D)
        fired = score_rules(rules, state, targets)
        fired.sort(key=lambda t: -max(abs(x) for x in t[3].values()))
        print(f"\n##### IMM {hex(v)}; EXIT  ax_row={ax} "
              f"(tail rules writing OLO+8 / OHI+14 @block35 input) #####")
        if not fired:
            print("    (no tail rule writes cell8-LO / cell14-HI)")
        for nm, sc, th, c in fired[:25]:
            print(f"    {nm:<52} score={sc:>12} thr={th:>12} {c}")


if __name__ == "__main__":
    main()
