#!/usr/bin/env python3
"""Inc-3 ROOT A — Step 1 of the mem[SP-local] byte-1 brief: IS 0x03 PRESENT?

The 20 multi-byte var_simple (x>=256) all fail step-5 `return x` with AX byte-1
dropped (x=990=0x03DE -> AX=0x00DE). The LI delivers byte-0 (0xDE) CORRECTLY from
mem[SP-local], so the LI's mem-read WORKS. The decisive question (the build hinges
on it): is mem[SP-local] byte-1 (=0x03=3) PRESENT, block-by-block, at the LI
step's byte-1 predictor row, in ANY stride-stable source band?

This scans EVERY block at the step-5 LI byte-1 predictor row (off=6) and decodes
each candidate one-hot/nibble band to its VALUE, looking for 3. Bands probed:
  - STACK0_BYTE_VAL_1_LO/_HI  (the Inc-2 mem[SP] byte-1 CAM target)
  - MEM_VAL_B1                 (the raw mem value byte-1 slot, magnitude)
  - ALU_HI                     (the LI's high-byte ALU)
  - AX_FULL_HI                 (the carried prior-AX high byte)
  - OUTPUT_HI_THIS_STEP        (the emitted high byte)
  - H1_DUMP_OUT / H1_PREV_STEP (the documented carry path)
We ALSO decode the byte-0 bands (STACK0_BYTE_VAL_0 equiv / OUTPUT_LO) to confirm
0xDE IS present (the positive control — proves the mem-read reaches this row).

Run TWICE (clear ~/.cache/c4_release/compiled_vm/ between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_memsp_byte1.py             # golden
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_memsp_byte1.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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

SRC = "int main() { int x; x = 990; return x; }"
# x=990 = 0x03DE; byte0=0xDE=222, byte1=0x03=3.
WANT_B1 = 3
WANT_B0 = 222


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def nibble_pair_val(row, lo, hi):
    """Decode a LO(16)+HI(16) one-hot nibble pair into a 0..255 byte value."""
    lo_idx = int(torch.argmax(row[lo:lo + 16]).item())
    lo_v = float(row[lo + lo_idx])
    hi_idx = int(torch.argmax(row[hi:hi + 16]).item())
    hi_v = float(row[hi + hi_idx])
    val = hi_idx * 16 + lo_idx
    conf = min(lo_v, hi_v)
    return val, conf, lo_v, hi_v


def onehot16(row, base):
    idx = int(torch.argmax(row[base:base + 16]).item())
    return idx, float(row[base + idx])


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
    ctx = p._final_context(bytecode, max_steps=9)
    # step5 emitted AX bytes (sanity): AX field offset 5; bytes at 6,7,8,9.
    s5 = pl + 5 * STEP
    print(f"=== {cfg} STEP={STEP} ===  step5 AX bytes emitted = "
          f"{ctx[s5+6:s5+10]} (want [222,3,0,0])")
    # The byte-1 token is emitted at the AX[0] predictor row (off=6).
    off = s5 + 6
    padded = torch.tensor([ctx], device=p._device)
    nblk = len(p.model.blocks)
    blk_map = p.block_layer_map()

    # nibble-pair bands (decode to byte VALUE)
    pairs = [
        ("STACK0_BV1", dp["STACK0_BYTE_VAL_1_LO"], dp["STACK0_BYTE_VAL_1_HI"]),
        ("ALU", dp["ALU_LO"], dp["ALU_HI"]),
        ("AX_FULL", dp["AX_FULL_LO"], dp["AX_FULL_HI"]),
        ("OUTPUT", dp["OUTPUT_LO"], dp["OUTPUT_HI"]),
    ]
    # single one-hot bands (decode index)
    singles = [
        ("MEM_VAL_B1", dp["MEM_VAL_B1"]),  # magnitude slot (not nibble)
        ("H1_PREV", dp["H1_PREV_STEP"]),
        ("H1_DUMP_OUT", dp["H1_DUMP_OUT"]),
    ]
    print(f"  probing off={off} (step5 byte-1 predictor, AX[0] row)")
    prev = None
    for blk in range(nblk):
        x = td(p.model.forward(padded, stop_after_block=blk)[0])
        row = x[off]
        parts = []
        for name, lo, hi in pairs:
            v, c, lv, hv = nibble_pair_val(row, lo, hi)
            parts.append(f"{name}={v}(c{c:.1f})")
        for name, base in singles:
            idx, mag = onehot16(row, base)
            parts.append(f"{name}[{idx}]={mag:.1f}")
        # MEM_VAL_B1 is also worth reading as raw magnitude (the value itself)
        memb1 = float(row[dp["MEM_VAL_B1"]])
        parts.append(f"MEM_VAL_B1_raw={memb1:.1f}")
        key = tuple(parts)
        if key != prev:
            lg = blk_map[blk]
            lg = lg.get("logical") if isinstance(lg, dict) else lg
            # flag any band whose decoded value == 3 (the wanted byte1)
            hit = ""
            for name, lo, hi in pairs:
                v, c, _, _ = nibble_pair_val(row, lo, hi)
                if v == WANT_B1 and c > 0.3:
                    hit += f" <<{name}=3!"
            print(f"  blk{blk:2d}(L{lg}): " + " ".join(parts) + hit)
            prev = key


if __name__ == "__main__":
    main()
