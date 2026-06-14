#!/usr/bin/env python3
"""Single-build deep probe of var_simple_0 step-1 STACK0 corruption.

The full_trace divergence is step 2 (PC), but its ROOT is the step-1 STACK0
emission: ref STACK0=[0,0,0,0] but neural emits [240,255,15,15] (=BP 0xFFF0
leaking in). That extra STACK0 content desyncs the MEM section and emits 2
spurious 255 tokens before step 2 (the '37-token desync'). This probe dumps:
  1. logits at every step-1 STACK0 byte predictor row (what wins, by how much)
  2. the STACK0_BYTE_VAL / BP / ALU residual bands feeding those rows
  3. the same for the step-0 frame for contrast
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import torch
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.compiler import compile_c
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import _step_offset_field
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic

reg = build_default_registry_dynamic()


def D(name):
    base, off = name, 0
    if "+" in name:
        base, o = name.split("+")
        off = int(o)
    if base not in reg.slots:
        return None
    return reg.slots[base].start + off


SRC = "int main() { int x; x = 990; return x; }"


def main():
    bytecode, data = compile_c(SRC)
    p = build_groundtruth_probe()
    prompt_len = len(p._build_context(bytecode))
    nblocks = len(p.model.blocks)

    # 1. logits at step-1 STACK0 + MEM byte predictor rows.
    trace = p.probe(bytecode, top_k=6, max_steps=12)
    print("=== STEP 1 frame logits (STACK0 + MEM) ===")
    base1 = prompt_len + 35  # step 1 starts here (step0 is 35 tokens)
    for off in range(20, 35):
        pos = base1 + off
        if pos in trace:
            r = trace[pos]
            top = [(t, round(v, 1)) for t, v in r["top_k_logits"]]
            print(f"  off{off:2d} {_step_offset_field(off):12s} emit={r['token']:4d} top6={top}")

    # 2. residual bands at the STACK0 byte-0 predictor row (step1 off20).
    bands = []
    for nm in ["STACK0_BYTE0", "STACK0_BYTE_VAL_0_LO", "STACK0_BYTE_VAL_0_HI",
               "STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI",
               "ALU_LO", "ALU_HI", "OUTPUT_LO", "OUTPUT_HI",
               "BP_CARRY_LO", "BP_CARRY_HI", "MARK_STACK0", "IS_BYTE"]:
        for k in range(16):
            bands.append(f"{nm}+{k}")
        bands.append(nm)
    dimn = {b: D(b) for b in bands if D(b) is not None}

    # STACK0[0] token is at step1 off21; predictor = off20 (STACK0_marker pos).
    for label, predoff in [("STACK0[0]_pred(s1.off20)", 35 + 20),
                           ("STACK0[1]_pred(s1.off21)", 35 + 21)]:
        res = p.residual_at(bytecode, block_idx=nblocks - 1,
                            position=prompt_len + predoff, dim_names=dimn,
                            max_steps=12)
        print(f"\n=== {label} nonzero residual bands ===")
        for nm in ["STACK0_BYTE_VAL_0_LO", "STACK0_BYTE_VAL_0_HI",
                   "STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI",
                   "ALU_LO", "ALU_HI", "OUTPUT_LO", "OUTPUT_HI"]:
            act = {k: round(res.get(f"{nm}+{k}", 0.0), 2)
                   for k in range(16) if abs(res.get(f"{nm}+{k}", 0.0)) > 0.1}
            base = res.get(nm, None)
            if act or (base is not None and abs(base) > 0.1):
                print(f"   {nm}: base={round(base,2) if base is not None else None} "
                      f"onehot={act}")


if __name__ == "__main__":
    main()
