#!/usr/bin/env python3
"""Attribute the residual 3 IMM mis-decodes (0xD8 / 0xE0 / 0xE8 -> 0xFFxx):
their byte-1 emits 0xFF instead of 0x00.  Find which l16 / l10-tail rule
writes the byte-1 0xFF on the IMM-EXIT byte-1 row.

spec_k=0, hook-free.
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
from neural_vm.unified_compiler.ops.l16_ops import _layer16_lev_routing_rules
from neural_vm.unified_compiler.ops.l10_ops import _tail_bit32_result_correction_rules


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            bc.append(op[0] | (op[1] << 8))
        else:
            bc.append(op)
    return bc


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
        score = sum(state.get(t.dim.key(), 0.0) * t.weight for t in rule.conditions)
        if score < rule.threshold:
            continue
        gate = rule.gate_bias
        if rule.gate is not None:
            gate += state.get(rule.gate.key(), 0.0) * rule.gate_weight
        for t in rule.gate_terms:
            gate += state.get(t.dim.key(), 0.0) * t.weight
        c = {}
        for w in rule.writes:
            if w.dim.key() in targets:
                c[w.dim.key()] = gate * w.weight
        if c and any(abs(v) > 1e-6 for v in c.values()):
            out.append((rule.name, round(score, 1), round(rule.threshold, 1),
                        {k: round(v, 2) for k, v in c.items()}))
    return out


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    D = len(model.embed(torch.tensor([[0]], device=dev))[0, 0])

    lev = _layer16_lev_routing_rules(100.0)
    tail = _tail_bit32_result_correction_rules()
    # byte-1 0xFF means OUTPUT_LO+15 and OUTPUT_HI+15 (0xF nibbles)
    targets = {"OUTPUT_LO+15", "OUTPUT_HI_THIS_STEP+15"}

    for v in (0xE0, 0xE8, 0xD8, 0x70):  # 0x70 = control (decodes fine)
        bc = _mk([(Opcode.IMM, v), Opcode.EXIT])
        ctx = probe._final_context(bc, max_steps=10)
        # find the AX byte-1 row: AX marker row + 2 (marker, byte0, byte1)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        emb = model.embed(toks)[0]
        axb = dp["MARK_AX"]
        axrows = [r for r in range(len(ctx)) if emb[r, axb].abs().item() > 0.5]
        # last AX marker (the decode); byte rows follow it
        print(f"\n##### IMM {hex(v)}; EXIT  ctx_len={len(ctx)} "
              f"ax_marker_rows={axrows} #####")
        # probe a few rows after the last AX marker (byte-0, byte-1 rows)
        if not axrows:
            continue
        axm = axrows[-1]
        for off in (1, 2, 3):
            pos = axm + off
            if pos >= len(ctx):
                continue
            with torch.no_grad():
                r30 = model.forward(toks, stop_after_block=30)[0]
                r36 = model.forward(toks, stop_after_block=36)[0]
            for blk_name, blk in (("blk30-lev", 30), ("blk36-tail", 36)):
                with torch.no_grad():
                    r = model.forward(toks, stop_after_block=blk)[0]
                row = r[pos]
                state = build_state(row, dp, D)
                rules = lev if blk == 30 else tail
                fired = score_rules(rules, state, targets)
                fired.sort(key=lambda t: -max(abs(x) for x in t[3].values()))
                if fired:
                    olo15 = round(float(row[dp["OUTPUT_LO"]+15].item()), 1)
                    ohi15 = round(float(row[dp["OUTPUT_HI_THIS_STEP"]+15].item()), 1)
                    print(f"  pos={pos}(ax+{off}) {blk_name} OLO15={olo15} OHI15={ohi15}")
                    for nm, sc, th, c in fired[:6]:
                        print(f"      {nm:<50} sc={sc} thr={th} {c}")


if __name__ == "__main__":
    main()
