#!/usr/bin/env python3
"""AR-LI byte-0 root probe: trace the LI value-load byte-0 block-by-block on the
TRUE autoregressive tape (campaign config), and decode the L15 head-0 attention
to find WHICH candidate row the LI read-CAM lands on and whether that row carries
the stored value.

var_simple x=28: ENT,LEA,PSH,IMM28,SI,LEA,LI,EXIT. The LI step (step index from
env) emits AX byte-0 at off=6. In AR campaign the LI returns 0x00 (AX=0x10000).

Run (campaign):
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 PROBE_LI_STEP=7 \
    python tools/probe_arli_byte0.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
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

SRC = os.environ.get("PROBE_SRC", "int main() { int x; x = 28; return x; }")
LI_STEP = int(os.environ.get("PROBE_LI_STEP", "7"))


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def onehot16(row, base):
    idx = int(torch.argmax(row[base:base + 16]).item())
    return idx, float(row[base + idx])


def pairval(row, lo, hi):
    li, lv = onehot16(row, lo)
    hi_i, hv = onehot16(row, hi)
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
    prompt_len = len(p._build_context(bytecode))
    ctx = p._final_context(bytecode, max_steps=LI_STEP + 1)
    padded = torch.tensor([ctx], device=p._device)
    nblk = len(p.model.blocks)
    blk_map = p.block_layer_map()

    s = prompt_len + LI_STEP * STEP
    off6 = s + int(os.environ.get("PROBE_OFF", "5"))  # AX marker row predicts byte-0
    print(f"=== {cfg} STEP={STEP} LI_STEP={LI_STEP} off6={off6} len(ctx)={len(ctx)} ===")
    print(f"  LI step raw tokens: {ctx[s:s+STEP]}")

    OUT_LO = dp["OUTPUT_LO"]
    OUT_HI = dp["OUTPUT_HI"]
    ALU_LO = dp["ALU_LO"]

    # Block-by-block OUTPUT byte-0 at the LI emit row.
    print("  --- OUTPUT_LO byte-0 value at LI emit row, per block (changes only) ---")
    prev = None
    for b in range(nblk):
        x = td(p.model.forward(padded, stop_after_block=b)[0])
        row = x[off6]
        lo_v, lo_c = onehot16(row, OUT_LO)
        alu_v, alu_c = onehot16(row, ALU_LO)
        key = (lo_v, round(lo_c, 1), alu_v, round(alu_c, 1))
        if key != prev:
            lg = blk_map[b]
            lg = lg.get("logical") if isinstance(lg, dict) else lg
            at = blk_map[b].get("attn", "") if isinstance(blk_map[b], dict) else ""
            print(f"   blk{b:2d}(L{lg} {at[:16]:16s}): OUT_LO_nib={lo_v}(c{lo_c:.1f}) "
                  f"ALU_LO_nib={alu_v}(c{alu_c:.1f})")
            prev = key

    # Now decode the L15 head-0 attention at the LI emit row: which K row wins?
    # L15 is a logical layer; find its physical block(s).
    print("  --- L15 head-0 attention: top-attended rows at LI emit row ---")
    l15_blocks = [b for b in range(nblk)
                  if (blk_map[b].get("logical") if isinstance(blk_map[b], dict)
                      else blk_map[b]) == 15]
    print(f"   L15 physical blocks: {l15_blocks}")

    # Decode candidate MEM value rows in the tape: any row that is a MEM value
    # byte-0 slot (L2H0[MEM]=1, H1[MEM]=0) -> read its CLEAN_EMBED byte value.
    MEM_I = 4
    L2H0 = dp["L2H0"]
    H1 = dp["H1"]
    CE_LO = dp["CLEAN_EMBED_LO"]
    CE_HI = dp["CLEAN_EMBED_HI"]
    MEM_STORE = dp["MEM_STORE"]
    # Use the residual AFTER the block before L15 to read candidate rows as L15 sees them.
    pre_l15 = (min(l15_blocks) - 1) if l15_blocks else nblk - 1
    xpre = td(p.model.forward(padded, stop_after_block=pre_l15)[0])
    print(f"   candidate MEM value-byte-0 rows (residual after blk{pre_l15}):")
    for pos in range(len(ctx)):
        row = xpre[pos]
        l2 = float(row[L2H0 + MEM_I])
        h1 = float(row[H1 + MEM_I])
        if l2 > 0.5 and h1 < 0.5:
            val, conf = pairval(row, CE_LO, CE_HI)
            ms = float(row[MEM_STORE])
            stepi = (pos - prompt_len) // STEP if pos >= prompt_len else -1
            offi = (pos - prompt_len) % STEP if pos >= prompt_len else pos
            print(f"     pos{pos} (step{stepi} off{offi}): CLEAN_EMBED_val={val} "
                  f"(c{conf:.1f}) MEM_STORE={ms:.1f} "
                  f"ADDR_B0_LO_nib={onehot16(row, dp['ADDR_B0_LO'])} "
                  f"ADDR_B0_HI_nib={onehot16(row, dp['ADDR_B0_HI'])}")

    # Also decode the LI query row's ADDR_B0 nibbles (what address it asks for).
    qrow = xpre[off6]
    print(f"   LI query row off6 ADDR_B0_LO_nib={onehot16(qrow, dp['ADDR_B0_LO'])} "
          f"ADDR_B0_HI_nib={onehot16(qrow, dp['ADDR_B0_HI'])} "
          f"MARK_AX={float(qrow[dp['MARK_AX']]):.1f} "
          f"OP_LI_RELAY={float(qrow[dp['OP_LI_RELAY']]):.1f}")


if __name__ == "__main__":
    main()
