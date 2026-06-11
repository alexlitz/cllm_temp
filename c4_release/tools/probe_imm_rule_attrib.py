#!/usr/bin/env python3
"""Pin WHICH l16 lev_routing rule writes OUTPUT cell 8 / cell 14 (0xE8) on
a plain ``IMM v; EXIT`` AX decode row.

Reads the residual at the lev_routing op's INPUT (we sweep candidate input
blocks), builds the name->value state map, runs ``_layer16_lev_routing_rules``
through the EXACT symbolic firing math (score >= threshold; contribution =
gate_value * write.weight), and reports every rule whose contribution to
OUTPUT_LO+8 or OUTPUT_HI_THIS_STEP+14 is large.

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
    """name -> value for every (name, base) in dim_positions.

    The rule condition dims use ``BASE+k`` names; _state_value resolves
    ``BASE`` + offset k via dim_positions. We expose a state map keyed by
    BASE name returning the base-slot value, plus BASE+k via a wrapper.
    """
    # We need a callable that maps any "NAME+k" to residual[base+k]. Easiest:
    # precompute a flat dict for every base and every k 0..15.
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
    """Return list of (rule_name, score, threshold, fired, {target: contrib})."""
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
                        fired, {k: round(v, 3) for k, v in contribs.items()}))
    return out


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    D = next(model.parameters()).shape[-1] if False else len(model.embed(
        torch.tensor([[0]], device=dev))[0, 0])

    rules = _layer16_lev_routing_rules(100.0)
    print(f"lev_routing rules: {len(rules)}")

    targets = {f"OUTPUT_LO+{k}" for k in range(16)} | {
        f"OUTPUT_HI_THIS_STEP+{k}" for k in range(16)}

    for v in (0xFF, 0x18):
        bc = _mk([(Opcode.IMM, v), Opcode.EXIT])
        ctx = probe._final_context(bc, max_steps=10)
        ax = find_ax_row(model, dp, dev, ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        print(f"\n##### IMM {hex(v)}; EXIT  ax_row={ax} ctx_len={len(ctx)} #####")
        # lev_routing runs late; sweep candidate input residuals (blocks
        # 29..35 = after L19..L25) to find where the rules see firing state.
        for in_blk in (29, 34, 35):
            with torch.no_grad():
                r = model.forward(toks, stop_after_block=in_blk)[0]
            row = r[ax]
            state = build_state(row, dp, D)
            fired = score_rules(rules, state, targets)
            # only rules touching cell 8 LO or cell 14 HI (the 0xE8 cells)
            key_cells = ("OUTPUT_LO+8", "OUTPUT_HI_THIS_STEP+14")
            fired = [t for t in fired
                     if any(k in t[4] and abs(t[4][k]) > 1e-6 for k in key_cells)]
            fired.sort(key=lambda t: -max(abs(x) for x in t[4].values()))
            print(f"  --- input residual after block {in_blk} "
                  f"(rules writing OLO+8 / OHI+14) ---")
            if not fired:
                print("    (no rule writes cell8-LO / cell14-HI)")
            for nm, sc, th, fr, c in fired[:20]:
                show = {k: v for k, v in c.items() if k in key_cells}
                print(f"    {nm:<48} score={sc:>10} thr={th:>10} {show}")
        # also dump key marker/opcode dims the suspect rules read
        with torch.no_grad():
            r = model.forward(toks, stop_after_block=35)[0]
        row = r[ax]
        keys = ["MARK_STACK0", "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP",
                "MARK_MEM", "HAS_SE", "OP_IMM", "OP_LEV", "OP_ENT", "OP_EXIT",
                "OP_LEA", "OP_JSR", "IS_BYTE", "MEM_STORE", "PSH_AT_SP",
                "CMP+7", "FETCH_LO+8", "FETCH_HI+15", "ADDR_B0_LO+0",
                "ADDR_B0_HI+14"]
        print("  marker/opcode dims @block35 ax row:")
        for k in keys:
            if "+" in k:
                base, off = k.split("+")
                b = dp.get(base)
                idx = int(b) + int(off) if b is not None else None
            else:
                b = dp.get(k)
                idx = int(b) if b is not None else None
            if idx is not None:
                print(f"    {k:<14} = {round(float(row[idx].item()),3)}")
        for nm in ("ALU_LO", "ALU_HI", "ADDR_B0_LO", "ADDR_B0_HI",
                   "AX_CARRY_LO", "AX_CARRY_HI", "AX_FULL_LO", "AX_FULL_HI"):
            b = dp.get(nm)
            if b is not None:
                vals = [(i, round(float(row[int(b)+i].item()), 2))
                        for i in range(16) if abs(row[int(b)+i].item()) > 0.3]
                print(f"    {nm:<14} = {vals}")


if __name__ == "__main__":
    main()
