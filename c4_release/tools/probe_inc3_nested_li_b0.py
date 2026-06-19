#!/usr/bin/env python3
"""Inc-3 Inc-3 MAP — nested_quad LI byte-0 starvation (got_ax=0, PC-correct).

nested_quad id950 (`quad(double_it(double_it(x)))`, x=10) diverges full_trace at
step 3 with expected (pc=186, ax=10) got (pc=186, ax=0). PC is PERFECT (control
flow tracked), only the loaded operand value (the function arg `x`=10) is starved
to 0. This is a BYTE-0 LI-from-frame delivery failure (the rootA fix 6e2959f5
delivered BYTE-1 for x>=256 var_simple; this is the byte-0 facet for small args).

This probe block-scans the step-3 AX byte-0 predictor row (off = base + 0) and
decodes every candidate operand-delivery band to confirm WHERE byte-0 (=10=0x0A)
is present in GOLDEN and WHERE it collapses to 0 in CAMPAIGN.

Bands probed (byte-0 / LO):
  STACK0_BYTE_VAL_0_LO/_HI  -- mem[SP] CAM target (Inc-2 operand path)
  ALU_LO/HI                 -- the LI's ALU
  AX_FULL_LO/HI             -- carried/emitted AX
  OUTPUT_LO/HI              -- the emitted byte
  MEM_VAL_B0 (raw)          -- raw mem value byte-0

Run TWICE (clear ~/.cache/c4_release/compiled_vm/ between):
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=0  python tools/probe_inc3_nested_li_b0.py
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_nested_li_b0.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = (
    "int double_it(int x) { return x * 2; }\n"
    "            int quad(int x) { return double_it(double_it(x)); }\n"
    "            int main() { return quad(10); }"
)
DIV_STEP = 3
WANT_B0 = 10  # x = 10 = 0x0A


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def nib(row, lo, hi):
    li = int(torch.argmax(row[lo:lo + 16]).item()); lv = float(row[lo + li])
    hi_i = int(torch.argmax(row[hi:hi + 16]).item()); hv = float(row[hi + hi_i])
    return hi_i * 16 + li, min(lv, hv)


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30)" if nostk else "GOLDEN(35)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    ctx = p._final_context(bytecode, max_steps=DIV_STEP + 2)
    s = pl + DIV_STEP * STEP
    print(f"=== {cfg} STEP={STEP} === step{DIV_STEP} emitted AX bytes = "
          f"{ctx[s+6:s+10]} (want byte0={WANT_B0})")
    off = s + 6  # AX byte-0 predictor row (off = base + 0; AX field at +6)
    padded = torch.tensor([ctx], device=p._device)
    nblk = len(p.model.blocks)
    blk_map = p.block_layer_map()
    pairs = [
        ("STK0_B0", dp.get("STACK0_BYTE_VAL_0_LO"), dp.get("STACK0_BYTE_VAL_0_HI")),
        ("ALU", dp["ALU_LO"], dp["ALU_HI"]),
        ("AXF", dp["AX_FULL_LO"], dp["AX_FULL_HI"]),
        ("OUT", dp["OUTPUT_LO"], dp["OUTPUT_HI"]),
    ]
    pairs = [(n, lo, hi) for (n, lo, hi) in pairs if lo is not None and hi is not None]
    memb0 = dp.get("MEM_VAL_B0")
    print(f"  probing off={off} (step{DIV_STEP} byte-0 predictor, AX[0] row)")
    prev = None
    for blk in range(nblk):
        x = td(p.model.forward(padded, stop_after_block=blk)[0])
        row = x[off]
        parts = []
        hit = ""
        for name, lo, hi in pairs:
            v, c = nib(row, lo, hi)
            parts.append(f"{name}={v}(c{c:.1f})")
            if v == WANT_B0 and c > 0.3:
                hit += f" <<{name}={WANT_B0}!"
        if memb0 is not None:
            parts.append(f"MEMb0={float(row[memb0]):.1f}")
        key = tuple(parts)
        if key != prev:
            lg = blk_map[blk]
            lg = lg.get("logical") if isinstance(lg, dict) else lg
            print(f"  blk{blk:2d}(L{lg}): " + " ".join(parts) + hit)
            prev = key


if __name__ == "__main__":
    main()
