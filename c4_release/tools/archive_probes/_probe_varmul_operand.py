"""Dump the MUL operand bands (ALU_LO / AX_CARRY_LO / AX_CARRY_HI) at the MUL
step's binop row for a var_mul id vs a literal-mul control, in the campaign
config. Reveals whether the memory-sourced (LI) operands reach the MUL operand
bands the same way the literal (IMM/PSH) operands do.

Usage (campaign env required):
  CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
      C4_VM_CACHE_DIR=/tmp/c4cache_varmul python tools/_probe_varmul_operand.py
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

from src.compiler import compile_c  # noqa: E402
from neural_vm.embedding import Opcode  # noqa: E402
from neural_vm.vm_step import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


def band(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return f"{name}: <no dim>"
    vals = [round(float(row.get(f"{name}+{k}", 0.0)), 2) for k in range(width)]
    # report the argmax nibble (the "decoded" value) and any other hot cells
    hot = [(k, v) for k, v in enumerate(vals) if abs(v) > 0.3]
    am = max(range(width), key=lambda k: vals[k]) if vals else None
    return f"{name}: argmax_nib={am} hot={hot}"


def dump_operands(p, dp, prog_bytes, label, step_idx, blocks):
    STEP = Token.STEP_TOKENS
    # the binop AX row: each step occupies STEP tokens; the AX bytes are at
    # offsets after REG_PC. The operand bands are read at the AX-marker row of
    # the binop step. Position = prefix + step_idx*STEP + AX_offset. We probe a
    # few positions around the step's AX row.
    ctx = p._final_context(list(prog_bytes))
    # find prefix length via the runner's context builder is internal; instead
    # probe positions step_idx*STEP .. +STEP from the END is fragile. Use the
    # known layout: AX marker is offset 5 (PC marker+4 bytes). We sweep offsets.
    print(f"\n### {label} step {step_idx} ###  ctx_len={len(ctx)}")
    # Probe each requested block at the AX-row of the step. We locate the AX
    # row by scanning for REG_AX marker tokens.
    ax_positions = [i for i, t in enumerate(ctx) if t == Token.REG_AX]
    if step_idx < len(ax_positions):
        pos = ax_positions[step_idx]
    else:
        pos = -1
    print(f"  AX-row position (step {step_idx}) = {pos}")
    names = {}
    for nm in ["ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
               "STACK0_BYTE_VAL_0", "STACK0_BYTE_VAL_1", "OP_MUL"]:
        base = dp.get(nm)
        if base is not None:
            for k in range(16):
                names[f"{nm}+{k}"] = base + k
    for blk in blocks:
        row = p.residual_at(list(prog_bytes), block_idx=blk, position=pos,
                            dim_names=names)
        print(f"  -- block {blk} --")
        for nm in ["ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI",
                   "STACK0_BYTE_VAL_0", "STACK0_BYTE_VAL_1"]:
            print("    " + band(row, dp, nm))


def main():
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    _model, layout = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(layout.dim_positions)
    p = build_groundtruth_probe()
    # blocks around the MUL pipeline: L11 (mul_partial) lives near physical
    # block ~13-16; sweep a range.
    blocks = [int(x) for x in os.environ.get("PROBE_BLOCKS", "11,12,13,14,15,16").split(",")]
    # var_mul 275: a=23, b=47. MUL is step 15.
    src = "int main() { int a; int b; a = 23; b = 47; return a * b; }"
    text, data = compile_c(src)
    dump_operands(p, dp, text, "var_mul_275 (23*47)", 15, blocks)
    # literal mul control via the SAME main() prologue so step framing matches
    # (corpus mul_104 = 23*65 passes). Use 23*47 as a literal to compare operand
    # delivery byte-for-byte with the var version.
    litsrc = "int main() { return 23 * 47; }"
    lt, ld = compile_c(litsrc)
    # find the MUL step by running the oracle.
    from neural_vm.unified_compiler.symbolic_program import (
        SymbolicDeclarativeProgramRunner,
    )
    r = SymbolicDeclarativeProgramRunner()
    st = r.run(lt, ld, max_steps=40)
    mul_step = None
    for i, tr in enumerate(st.trace):
        if tr.name == "MUL":
            mul_step = i
            break
    print(f"\n[lit_mul MUL at step {mul_step}]")
    dump_operands(p, dp, lt, "lit_mul (23*47)", mul_step, blocks)


if __name__ == "__main__":
    main()
