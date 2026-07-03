#!/usr/bin/env python3
"""AR-LI: compare the STORE value-byte-0 row's address + store signals between
golden and campaign, on the TRUE AR tape. The store row is where the SI wrote
x=28; the L15 LI read-CAM must content-address it. Dumps, for every MEM
value-byte-0 row, the CLEAN_EMBED value + ADDR_B0/ADDR_B1 nibbles + MEM_STORE +
MEM_STORE_AT_VAL, as the residual entering L15.
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


def nib(row, base):
    idx = int(torch.argmax(row[base:base + 16]).item())
    return idx, round(float(row[base + idx]), 2)


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
    blk_map = p.block_layer_map()
    nblk = len(p.model.blocks)
    l15 = min(b for b in range(nblk) if blk_map[b].get("logical") == 15)
    x = td(p.model.forward(padded, stop_after_block=l15 - 1)[0])
    print(f"=== {cfg} STEP={STEP} pre-L15(blk{l15-1}) value-byte-0 rows ===")
    MEM_I = 4
    msv = dp.get("MEM_STORE_AT_VAL")
    for pos in range(len(ctx)):
        row = x[pos]
        l2 = float(row[dp["L2H0"] + MEM_I]); h1 = float(row[dp["H1"] + MEM_I])
        if l2 > 0.5 and h1 < 0.5:
            val_lo = int(torch.argmax(row[dp["CLEAN_EMBED_LO"]:dp["CLEAN_EMBED_LO"]+16]))
            val_hi = int(torch.argmax(row[dp["CLEAN_EMBED_HI"]:dp["CLEAN_EMBED_HI"]+16]))
            val = val_hi * 16 + val_lo
            stepi = (pos - prompt_len) // STEP if pos >= prompt_len else -1
            offi = (pos - prompt_len) % STEP if pos >= prompt_len else pos
            msval = round(float(row[msv]), 2) if msv is not None else None
            print(f"  pos{pos} (step{stepi} off{offi}): VAL={val} "
                  f"B0_LO={nib(row, dp['ADDR_B0_LO'])} B0_HI={nib(row, dp['ADDR_B0_HI'])} "
                  f"B1_LO={nib(row, dp['ADDR_B1_LO'])} B1_HI={nib(row, dp['ADDR_B1_HI'])} "
                  f"MEM_STORE={round(float(row[dp['MEM_STORE']]),2)} "
                  f"MEM_STORE_AT_VAL={msval} CMP3={round(float(row[dp['CMP']+3]),2)}")
    # The LI query row (AX marker = off5).
    qrow = x[prompt_len + LI_STEP * STEP + 5]
    print(f"  LI-query off5: B0_LO={nib(qrow, dp['ADDR_B0_LO'])} B0_HI={nib(qrow, dp['ADDR_B0_HI'])} "
          f"MARK_AX={round(float(qrow[dp['MARK_AX']]),2)} OP_LI={round(float(qrow[dp['OP_LI']]),2)} "
          f"OP_LI_RELAY={round(float(qrow[dp['OP_LI_RELAY']]),2)} "
          f"OP_LC_RELAY={round(float(qrow[dp['OP_LC_RELAY']]),2)} CMP3={round(float(qrow[dp['CMP']+3]),2)} "
          f"MARK_STACK0={round(float(qrow[dp['MARK_STACK0']]),2)} HAS_SE={round(float(qrow[dp['HAS_SE']]),2)} "
          f"IS_BYTE={round(float(qrow[dp['IS_BYTE']]),2)}")


if __name__ == "__main__":
    main()
