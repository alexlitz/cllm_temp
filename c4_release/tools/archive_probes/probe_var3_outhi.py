#!/usr/bin/env python3
"""var_three step-0 OUTPUT_HI leak probe (campaign 30-tok frame).

Autoregressive (spec_k=0) replay via GroundTruthProbe._final_context. Finds the
AX byte-1 EMITTING row in the var_three step-0 frame and reads
OUTPUT_HI_THIS_STEP / OUTPUT_LO band + the high-zero rule's gating conditions
across blocks to localize the over-write that leaks 0x10 into AX byte 1.

NOTE: dim_positions maps a BASE NAME to a base index; offsets are base+i.
``dp['OUTPUT_HI_THIS_STEP'] = 85`` -> band is 85..100.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_outhi python tools/probe_var3_outhi.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
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
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic,
)
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = os.environ.get(
    "PROBE_SRC",
    "int main() { int a; int b; int c; a = 29; b = 6; c = 20; return a + b + c; }",
)

_MARKERS = ["MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM", "MARK_SE",
            "MARK_STACK0"]
_COND_DIMS = ["IS_BYTE", "HAS_SE", "BYTE_INDEX_0", "BYTE_INDEX_1",
              "BYTE_INDEX_2", "BYTE_INDEX_3", "TEMP+8", "TEMP+9", "TEMP+10",
              "ALU_HI+0", "AX_CARRY_HI+0", "CARRY+1", "CARRY+2"]
_H1 = [f"H1+{i}" for i in range(6)]


def _band(dp, base, n):
    b = dp.get(base)
    if b is None:
        return {}
    return {f"{base}+{i}": b + i for i in range(n)}


def _get(dp, name):
    """Resolve NAME or BASE+OFF to a residual dim index."""
    if name in dp:
        return dp[name]
    if "+" in name:
        base, off = name.rsplit("+", 1)
        if base in dp and off.lstrip("-").isdigit():
            return dp[base] + int(off)
    return None


def main():
    STEP = int(Token.STEP_TOKENS)
    print(f"STEP_TOKENS={STEP}")
    _, layout = compile_full_vm_dynamic()
    dp = layout.dim_positions

    hi_band = _band(dp, "OUTPUT_HI_THIS_STEP", 16)
    lo_band = _band(dp, "OUTPUT_LO", 16)
    marker_dims = {m: dp[m] for m in _MARKERS if m in dp}
    cond_dims = {n: _get(dp, n) for n in (_COND_DIMS + _H1)}
    cond_dims = {n: d for n, d in cond_dims.items() if d is not None}

    bytecode, data = compile_c(SRC)
    probe = build_groundtruth_probe()

    ctx = probe._final_context(bytecode, max_steps=2)
    prompt_len = len(probe._build_context(bytecode))
    print(f"prompt_len={prompt_len}  total_ctx={len(ctx)}")
    step0 = ctx[prompt_len:prompt_len + STEP]
    print("step0 tokens:", step0)

    nblocks = len(probe.model.blocks)
    print(f"n_physical_blocks={nblocks}")

    last_block = nblocks - 1
    mark_block = 12
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    with torch.no_grad():
        resid_mark = probe.model.forward(padded, stop_after_block=mark_block)[0]
    print("\n== step-0 frame marker map (post block %d) ==" % mark_block)
    ax_marker_pos = None
    for j in range(STEP):
        pos = prompt_len + j
        row = resid_mark[pos]
        tags = [m for m, d in marker_dims.items() if float(row[d]) > 0.5]
        if "MARK_AX" in tags:
            ax_marker_pos = pos
        print(f"  j={j:2d} pos={pos} tok={ctx[pos]:3d} : {' '.join(tags)}")

    if ax_marker_pos is None:
        print("!! no MARK_AX found"); return
    # AX byte-1 row = marker + 2 (marker, byte0, byte1)
    ax_b1_pos = ax_marker_pos + 2
    # The position whose logits EMIT byte-1 is ax_b1_pos - 1 (= byte0 row)
    emit_pos = ax_b1_pos - 1
    print(f"\nAX marker pos={ax_marker_pos}, AX byte-1 row pos={ax_b1_pos} "
          f"tok={ctx[ax_b1_pos]} (0x{ctx[ax_b1_pos]:02x}); "
          f"emitting (logit) row pos={emit_pos}")

    # The token at pos ax_b1_pos is EMITTED by the logits at emit_pos. The
    # OUTPUT_HI/LO state that drives it lives at emit_pos.
    print("\n== OUTPUT_HI_THIS_STEP + cond band at EMIT row (pos %d) across blocks ==" % emit_pos)
    for blk in list(range(28, nblocks)):
        with torch.no_grad():
            resid = probe.model.forward(padded, stop_after_block=blk)[0]
        row = resid[emit_pos]
        hi = {k: float(row[d]) for k, d in hi_band.items()}
        hi_top = sorted(hi.items(), key=lambda kv: -kv[1])[:3]
        lo = {k: float(row[d]) for k, d in lo_band.items()}
        lo_top = sorted(lo.items(), key=lambda kv: -kv[1])[:2]
        print(f"  blk={blk:2d}: HI {[(k.split('+')[-1],round(v,2)) for k,v in hi_top]}"
              f"  LO {[(k.split('+')[-1],round(v,2)) for k,v in lo_top]}")

    # Gating conditions at EMIT row, mid-late block (post-marking, pre-tail)
    for probe_blk in (30, 33, last_block):
        with torch.no_grad():
            resid = probe.model.forward(padded, stop_after_block=probe_blk)[0]
        row = resid[emit_pos]
        print(f"\n== high-zero rule conditions at EMIT row pos={emit_pos} (blk {probe_blk}) ==")
        for n, d in cond_dims.items():
            print(f"    {n:16s} = {float(row[d]):+10.3f}")

    # LM-head logits at the emitting row for byte tokens 0..16
    with torch.no_grad():
        logits = probe.model.forward(padded)[0]  # full forward -> logits
    emit_logits = logits[emit_pos]
    print(f"\n== LM-head logits at emit row pos={emit_pos} (byte tokens) ==")
    top = torch.topk(emit_logits, 6)
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"    tok {i:3d}: {v:+.3f}")


if __name__ == "__main__":
    main()
