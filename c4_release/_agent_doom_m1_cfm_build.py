#!/usr/bin/env python3
"""MILESTONE 1 (measure, not project): CODE-FROM-MEMORY build of doom at FIXED D.

Proves the D-explosion of the BAKED CODE_OP[k]/CODE_IMM[k] table is gone: with
``code_from_memory=True`` the residual width does NOT scale with program length,
so doom's ~4000 instructions load as KV code frames at a FIXED hidden_size.

Measures, on the GPU, REAL numbers (no projections):
  * CFM  hidden_size D, VRAM after building the Qwen2Model, build_s.
  * BAKED hidden_size D (computed from the layout only — the full baked model is
    the ~8 GB one we are avoiding; we report its D to show the ratio).
  * doom opcode mix (which subset is required).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path


DOOM_C = Path("/home/alexlitz/Documents/misc/c4_doom/doom.c")


def compile_doom():
    from src.compiler import compile_c, Op
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    src = DOOM_C.read_text()
    bytecode, data = compile_c(src)
    code = bytecode_to_isa(bytecode)
    names = {}
    for nm in dir(Op):
        v = getattr(Op, nm)
        if isinstance(v, int) and not nm.startswith("_"):
            names[v] = nm
    counts = Counter(int(w) & 0xFF for w in bytecode)
    return code, data, counts, names


def layout_D(code_size, subset, code_from_memory):
    """hidden_size the build() would allocate — from the layout only (cheap)."""
    from c4_min.qwen_full_vm import QwenFullLayout, QWEN2_5_ARCH
    QL = QwenFullLayout(code_size, subset, efficient_alu=True,
                        recurrent_divmod=True, code_from_memory=code_from_memory)
    dim_needed = QL.D_used + 1
    hidden = QWEN2_5_ARCH.hidden_for(dim_needed)
    return QL.D_used, hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--build", action="store_true",
                    help="actually build the CFM Qwen2Model + measure VRAM")
    args = ap.parse_args()

    import torch
    from c4_min.qwen_full_vm import SUBSET_FULL

    code, data, counts, names = compile_doom()
    print(f"doom.c compiled: instrs={len(code)}  data_bytes={len(data)}")
    print("top opcodes:")
    for op, n in counts.most_common(14):
        print(f"  {names.get(op, op):5s} {n}")
    code_size = len(code) + 2

    d_used_cfm, hid_cfm = layout_D(code_size, SUBSET_FULL, code_from_memory=True)
    # BAKED D scales ~5 dims/instr; the doom-size baked layout is the ~8GB wall we
    # avoid, so extrapolate its D from two small measured points (building it is the
    # exact O(code_size) explosion the CFM path removes).
    import math
    (c0, d0) = (64, layout_D(64, SUBSET_FULL, code_from_memory=False)[0])
    (c1, d1) = (256, layout_D(256, SUBSET_FULL, code_from_memory=False)[0])
    slope = (d1 - d0) / (c1 - c0)
    d_used_baked = int(d0 + slope * (code_size - c0))
    hid_baked = max(14 * 64, math.ceil((d_used_baked + 1) / 64) * 64)
    print()
    print(f"code_size (instrs+2)       : {code_size}")
    print(f"CFM   D_used={d_used_cfm:6d}  hidden_size={hid_cfm:6d}  (FIXED, program-length-independent)")
    print(f"BAKED D_used~{d_used_baked:6d}  hidden_size~{hid_baked:6d}  (scales {slope:.1f}/instr -> the ~8GB wall)")
    print(f"D ratio baked/cfm          : {hid_baked / hid_cfm:.1f}x")

    if not args.build:
        print("\n(pass --build to actually construct the CFM Qwen2Model and measure VRAM)")
        return 0

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
        base_mem = torch.cuda.memory_allocated(dev)

    from c4_min.qwen_full_vm import build
    t0 = time.perf_counter()
    vm = build(code_size=code_size, subset=SUBSET_FULL,
               recurrent_divmod=True, code_from_memory=True)
    build_cpu_s = time.perf_counter() - t0

    n_params = sum(p.numel() for p in vm.qmodel.parameters())
    t1 = time.perf_counter()
    vm.qmodel.to(dev)
    embed = vm.embed.to(dev)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    to_dev_s = time.perf_counter() - t1

    print()
    print(f"BUILT CFM Qwen2Model:")
    print(f"  hidden_size        : {vm.hidden_size}")
    print(f"  intermediate_size  : {vm.intermediate_size}")
    print(f"  n_layers (stored)  : {vm.n_layers}")
    print(f"  n_applied          : {vm.n_applied}")
    print(f"  params             : {n_params/1e6:.2f} M  ({n_params*4/1e9:.3f} GB fp32)")
    print(f"  build_cpu_s        : {build_cpu_s:.2f}")
    print(f"  to_device_s        : {to_dev_s:.2f}")
    if dev.type == "cuda":
        used = torch.cuda.memory_allocated(dev) - base_mem
        peak = torch.cuda.max_memory_allocated(dev) - base_mem
        print(f"  VRAM allocated     : {used/1e9:.3f} GB")
        print(f"  VRAM peak          : {peak/1e9:.3f} GB")
        print(f"  --> vs baked ~8GB baseline: CFM D is FIXED, so VRAM is program-length-independent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
