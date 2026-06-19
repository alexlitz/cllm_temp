"""Inc-3 claw-back: trace the add/sub byte-1 pipeline across blocks.

Shows, per step row (in-step index), for a chosen block:
  - the carrier STACK0_BYTE_VAL_1_LO/HI (operand-A byte 1)
  - the TEMP+8 (ADD) / TEMP+9 (SUB) byte-row discriminators
  - the BYTE_INDEX_0/1 emit-row flags
  - MARK_AX / OP_ADD / OP_SUB
so we can see WHERE the byte-1 carrier reaches and where the relay/adder
should fire. Campaign config: C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"; os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings; warnings.filterwarnings("ignore")
import sys, io, contextlib
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch
from src.compiler import compile_c
from neural_vm.batched_pure_neural import Token
from tools.probe_groundtruth import build_groundtruth_probe

SRC = os.environ.get("PROBE_SRC", "int main() { return 827 - 26; }")
RS = int(os.environ.get("PROBE_STEP", "3"))
BLKS = [int(b) for b in os.environ.get("PROBE_BLKS", "16,17,18,34,36,38").split(",")]


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def top(row, base, n=16, k=3):
    v = [(i, float(row[base + i])) for i in range(n)]
    v = [(i, x) for i, x in v if abs(x) > 0.3]
    v.sort(key=lambda z: -abs(z[1]))
    return " ".join(f"[{i}]={x:.1f}" for i, x in v[:k]) or "."


with contextlib.redirect_stdout(io.StringIO()):
    bc, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
dp = dict(_l.dim_positions)
STEP = int(Token.STEP_TOKENS)
pl = len(p._build_context(bc))
ctx = p._final_context(bc, max_steps=RS + 2)
padded = torch.tensor([ctx], device=p._device)

sb_lo, sb_hi = dp["STACK0_BYTE_VAL_1_LO"], dp["STACK0_BYTE_VAL_1_HI"]
ax_lo, ax_hi = dp["AX_FULL_LO"], dp["AX_FULL_HI"]
out_lo, out_hi = dp.get("OUTPUT_BYTE_LO"), dp.get("OUTPUT_BYTE_HI")
flagn = ["MARK_AX", "TEMP", "BYTE_INDEX_0", "BYTE_INDEX_1", "OP_SUB", "OP_ADD",
         "STACK0_BYTE1", "MEM_VAL_B2", "CARRY"]

print(f"=== add/sub byte-1 trace {SRC!r} step={RS} STEP={STEP} ===")
for BLK in BLKS:
    with contextlib.redirect_stdout(io.StringIO()):
        x = td(p.model.forward(padded, stop_after_block=BLK)[0])
    print(f"--- block {BLK} ---")
    for stp in (RS,):
        for j in range(STEP):
            pos = pl + stp * STEP + j
            if pos >= len(ctx):
                break
            row = x[pos]
            sl = top(row, sb_lo); sh = top(row, sb_hi)
            al = top(row, ax_lo); ah = top(row, ax_hi)
            ol = top(row, out_lo) if out_lo else "."
            oh = top(row, out_hi) if out_hi else "."
            fl = []
            for f in flagn:
                d = dp.get(f)
                if d is None:
                    continue
                for off in (0, 1, 3, 8, 9):
                    if off >= (32 if f == "TEMP" else 1) and f != "TEMP" and f != "CARRY":
                        continue
                    v = float(row[d + off])
                    if abs(v) > 0.4:
                        fl.append(f"{f}+{off}={v:.0f}")
            interesting = (sl != "." or sh != "." or al != "." or ah != "." or
                           any("BYTE_INDEX" in s or "MARK_AX" in s for s in fl))
            if interesting:
                print(f"  in={j:2d} tok={ctx[pos]:3d}: SBV1[{sl}|{sh}] AX[{al}|{ah}] OUT[{ol}|{oh}] {' '.join(fl)}")
