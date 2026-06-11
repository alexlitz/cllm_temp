#!/usr/bin/env python3
"""Block-35 W_down ablation sweep for the AX bytes 1-3 = 0xFF leak.

spec_k=0, NO hooks. Uses tools/probe_groundtruth.py's GroundTruthProbe.

Stage 1: baseline emitted_result on the 5 target tests + sub_borrow_cascade.
Stage 2: 32-unit W_down ablation chunks on block 35 -> which chunks flip the
         or_basic exit code's bytes 1-3 from 0xFF to 0x00.
Stage 3: per-unit OUTPUT_LO[15]/OUTPUT_HI[15] W_down weight scan to map
         contributing units to lanes.

Run:  C4_SMOKE_SPEC_K=0 C4_TEST_SPEC_K=0 python tools/probe_ax_ff_ablation.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.embedding import Opcode as Op


def bc(ops):
    out = []
    for o in ops:
        if isinstance(o, tuple):
            op, imm = o
            out.append(op | (imm << 8))
        else:
            out.append(o)
    return out


TARGETS = {
    "or_basic":   (bc([(Op.IMM, 0x0F), Op.PSH, (Op.IMM, 0x30), Op.OR, Op.EXIT]), 0x3F),
    "xor_basic":  (bc([(Op.IMM, 0xFF), Op.PSH, (Op.IMM, 0xD5), Op.XOR, Op.EXIT]), 0x2A),
    "or_16bit":   (bc([(Op.IMM, 0x0F00), Op.PSH, (Op.IMM, 0x00FF), Op.OR, Op.EXIT]), 0x0FFF),
    "xor_16bit":  (bc([(Op.IMM, 0x0F0F), Op.PSH, (Op.IMM, 0x00FF), Op.XOR, Op.EXIT]), 0x0FF0),
    "sub_16bit":  (bc([(Op.IMM, 0x100), Op.PSH, (Op.IMM, 1), Op.SUB, Op.EXIT]), 0xFF),
    # MUST stay 0xFFFFFFFF (genuine all-FF):
    "sub_borrow_cascade": (bc([(Op.IMM, 0), Op.PSH, (Op.IMM, 1), Op.SUB, Op.EXIT]), 0xFFFFFFFF),
    "and_basic":  (bc([(Op.IMM, 0xFF), Op.PSH, (Op.IMM, 0x2A), Op.AND, Op.EXIT]), 0x2A),
    "and_16bit":  (bc([(Op.IMM, 0x0FFF), Op.PSH, (Op.IMM, 0x00FF), Op.AND, Op.EXIT]), 0x00FF),
}


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    print(f"# n_blocks={len(model.blocks)}")
    bmap = probe.block_layer_map()
    print("# block 35:", bmap[35] if len(bmap) > 35 else "N/A")

    def run(name):
        prog, want = TARGETS[name]
        _, code = probe.emitted_result(prog, max_steps=20)
        bts = [(code >> (8*j)) & 0xFF for j in range(4)]
        ok = (code == want)
        return code, bts, ok

    print("\n=== Stage 1: clean baseline ===")
    base = {}
    for name in TARGETS:
        code, bts, ok = run(name)
        base[name] = (code, bts, ok)
        print(f"  {name:20s} got=0x{code:08X} bytes={bts} want=0x{TARGETS[name][1]:08X} {'PASS' if ok else 'FAIL'}")

    blk = model.blocks[35]
    ffn = blk.ffn
    Wd = ffn.W_down.data
    n_units = Wd.shape[1]
    print(f"\n# block35 W_down shape={tuple(Wd.shape)}  n_units={n_units}")

    # Rule-name map (unit -> rule name), fast (no bake).
    from neural_vm.unified_compiler.ops.l10_ops import _tail_bit32_result_correction_rules
    rule_names = [r.name or f"rule_{i}" for i, r in enumerate(_tail_bit32_result_correction_rules())]

    SWEEP_TARGETS = os.environ.get("ABLATE_TARGETS", "xor_basic,or_16bit,xor_16bit,sub_16bit").split(",")
    chunk = 8
    for tgt in SWEEP_TARGETS:
        print(f"\n=== 8-unit W_down ablation on block 35 (target={tgt}, base=0x{base[tgt][0]:08X}) ===")
        any_change = False
        for a in range(0, n_units, chunk):
            b = min(a + chunk, n_units)
            saved = Wd[:, a:b].clone()
            Wd[:, a:b] = 0.0
            code, bts, _ = run(tgt)
            Wd[:, a:b] = saved
            if code != base[tgt][0]:
                any_change = True
                base_name = rule_names[a].split(".")[0]
                print(f"  units[{a:4d}:{b:4d}] {tgt} -> 0x{code:08X} bytes={bts}  rule~{base_name}")
        if not any_change:
            print(f"  (no chunk changed {tgt})")


if __name__ == "__main__":
    main()
