"""Cached-context probe of the L11 MUL combine pipeline at the MUL result-emit
row for var_mul vs lit_mul (campaign config). Dumps ALU/MUL_RESULT/OUTPUT bands
at blocks around L11 (16) to localize WHY the var-frame MUL writes a weak/wrong
OUTPUT while the literal-frame MUL writes the correct 41.56 product band.

Builds the spec_k=0 context ONCE per program (cached), then runs one truncated
forward per block (fast vs the 22-replay version).

Usage:
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
      C4_VM_CACHE_DIR=/tmp/c4cache_varmul2 python tools/_probe_varmul_l11.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.vm_step import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def band_sum_argmax(row, base, width=16):
    vals = [float(row[base + k].item()) for k in range(width)]
    am = max(range(width), key=lambda k: vals[k])
    s = sum(vals)
    absmx = max(abs(v) for v in vals)
    hot = [(k, round(v, 1)) for k, v in enumerate(vals) if abs(v) > 0.5]
    return f"argmax={am} sum={s:.1f} absmax={absmx:.1f} hot={hot[:8]}"


def main():
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    _model, layout = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(layout.dim_positions)
    p = build_groundtruth_probe()
    model = p.model
    device = p._device

    bands = ["ALU_LO", "ALU_HI", "OUTPUT_LO", "OUTPUT_HI",
             "MUL_RESULT_HI", "AX_FULL", "AX_CARRY_LO", "AX_CARRY_HI"]
    blocks = [int(x) for x in os.environ.get(
        "PROBE_BLOCKS", "14,15,16,17,28,30").split(",")]

    def run(prog_bytes, label, mul_step):
        ctx = p._final_context(list(prog_bytes))
        ax = [i for i, t in enumerate(ctx) if t == Token.REG_AX]
        if mul_step >= len(ax):
            print(f"\n### {label}: step {mul_step} >= n_ax {len(ax)}")
            return
        pos = ax[mul_step]
        print(f"\n### {label}  ctx_len={len(ctx)} MUL-AX pos={pos} (step {mul_step}) ###")
        padded = torch.tensor([ctx], dtype=torch.long, device=device)
        for blk in blocks:
            with torch.no_grad():
                resid = model.forward(padded, stop_after_block=blk)
            row = resid[0, pos]
            print(f"  -- block {blk} --")
            for nm in bands:
                base = dp.get(nm)
                if base is None:
                    print(f"    {nm}: <no dim>")
                    continue
                print(f"    {nm}: {band_sum_argmax(row, base)}")

    def find_mul_step(text, data):
        from neural_vm.unified_compiler.symbolic_program import (
            SymbolicDeclarativeProgramRunner,
        )
        r = SymbolicDeclarativeProgramRunner()
        st = r.run(text, data, max_steps=40)
        for i, tr in enumerate(st.trace):
            if tr.name == "MUL":
                return i
        return None

    vt, vd = compile_c(
        "int main() { int a; int b; a = 23; b = 47; return a * b; }")
    run(vt, "var_mul_275 (23*47=1081, 0x439)", find_mul_step(vt, vd))

    lt, ld = compile_c("int main() { return 23 * 47; }")
    run(lt, "lit_mul (23*47=1081, 0x439)", find_mul_step(lt, ld))


if __name__ == "__main__":
    main()
