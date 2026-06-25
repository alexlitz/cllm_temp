#!/usr/bin/env python3
"""Faithful si_li_16bit LI byte-1 probe: run the REAL batched runner, capture
the exact emitted context (the one producing 564=0x234), then re-forward THAT
context with stop_after_block to trace the OUTPUT byte-1 at the LI predictor row.

This decouples the diagnosis from autoregressive replay drift: we read the
residual stream of the same tokens the smoke gate scored.

Run (campaign):
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_sili_real.py
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
from neural_vm.embedding import Opcode  # noqa: E402
from neural_vm.batched_pure_neural import Token, BatchedPureNeuralRunner  # noqa: E402
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic)


PROGRAMS = {
    "si_li_16bit": [(Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 0x1234),
                    Opcode.SI, (Opcode.IMM, 0x200), Opcode.LI, Opcode.EXIT],
    "si_li_roundtrip": [(Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 42),
                        Opcode.SI, (Opcode.IMM, 0x200), Opcode.LI, Opcode.EXIT],
}


def _make_bytecode(ops):
    out = []
    for op in ops:
        if isinstance(op, tuple):
            out.append(int(op[0]) | (int(op[1]) << 8))
        else:
            out.append(int(op))
    return out


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def nib(row, lo, hi):
    lo_i = int(torch.argmax(row[lo:lo + 16]).item())
    hi_i = int(torch.argmax(row[hi:hi + 16]).item())
    return hi_i * 16 + lo_i, min(float(row[lo + lo_i]), float(row[hi + hi_i]))


def _build_runner():
    from neural_vm.run_vm import AutoregressiveVMRunner
    serial = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
    serial.spec_k = 0
    return BatchedPureNeuralRunner(serial)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "si_li_16bit"
    ops = PROGRAMS[which]
    bc = _make_bytecode(ops)
    runner = _build_runner()
    model = runner.model
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)

    # Capture the emitted context via _ElementState. Patch _step_one to snapshot.
    captured = {}
    orig = runner._step_one

    def spy(state, next_tok, tok_i, *a, **k):
        r = orig(state, next_tok, tok_i, *a, **k)
        captured["ctx"] = list(state.context)
        captured["prefix_len"] = state.prefix_len
        return r
    runner._step_one = spy
    results = runner.run_batch([bc], data_list=[b""], expected_steps_list=[8],
                               max_steps=30)
    runner._step_one = orig
    print(f"=== {which} result={results} ===")
    ctx = captured["ctx"]
    pl = captured["prefix_len"]
    print(f"STEP={STEP} prefix_len={pl} ctxlen={len(ctx)}")
    # locate REG_AX rows
    axrows = []
    for i, t in enumerate(ctx):
        if t == int(Token.REG_AX) and i + 4 < len(ctx):
            step = (i - pl) // STEP if i >= pl else -1
            val = sum((ctx[i + 1 + j] & 0xFF) << (8 * j) for j in range(4))
            axrows.append((i, step, ctx[i + 1:i + 5], val))
            print(f"  REG_AX pos{i} step{step} bytes={ctx[i+1:i+5]} val={val}=0x{val:x}")
    # find LI step: the step whose AX should be the loaded value
    # For si_li programs LI is step 5 (0 IMM,1 PSH,2 IMM,3 SI,4 IMM,5 LI,6 EXIT)
    li_step = 5
    target = next((r for r in axrows if r[1] == li_step), None)
    if target is None:
        print("no LI-step REG_AX row found"); return
    axpos = target[0]
    print(f"\n  LI-step REG_AX at pos{axpos}; AX byte rows at {axpos+1}..{axpos+4}")
    padded = torch.tensor([ctx], device=next(model.parameters()).device)
    nblk = len(model.blocks)

    # --- Dump the MEM section: find rows carrying MEM_VAL_B* at the final
    #     residual (post all blocks) to see what value got stored. ---
    xfull = td(model.forward(padded)[0])
    print("\n  MEM value rows (post-all-blocks, scan for MEM_VAL_B* != 0):")
    for nm in ("MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3"):
        if nm not in dp:
            continue
        col = dp[nm]
        hits = [(i, float(xfull[i, col])) for i in range(len(ctx))
                if abs(float(xfull[i, col])) > 0.3]
        for i, mv in hits[:8]:
            clo, chi = dp.get("CLEAN_EMBED_LO"), dp.get("CLEAN_EMBED_HI")
            cval = ""
            if clo is not None and chi is not None:
                v, c = nib(xfull[i], clo, chi)
                cval = f" CLEAN_EMBED={v}(c{c:.1f})"
            print(f"    {nm} pos{i}={mv:.2f}{cval}")

    # byte-1 predictor row = the row that PREDICTS token at axpos+2 (byte1),
    # i.e. position axpos+1 (byte0 row predicts byte1).
    for off in (axpos + 1, axpos + 2):
        print(f"\n  --- predictor pos={off} (rel {off-axpos} from REG_AX) ---")
        prev = None
        for blk in range(nblk):
            x = td(model.forward(padded, stop_after_block=blk)[0])
            row = x[off]
            ov, oc = nib(row, dp["OUTPUT_LO"], dp["OUTPUT_HI"])
            parts = [f"OUT={ov}(c{oc:.1f})"]
            for nm in ("STACK0_BYTE_VAL_1_LO", "AX_FULL_LO", "ALU_LO"):
                hi = nm.replace("_LO", "_HI")
                if nm in dp and hi in dp:
                    v, c = nib(row, dp[nm], dp[hi])
                    parts.append(f"{nm[:-3]}={v}(c{c:.1f})")
            key = tuple(parts)
            if key != prev:
                print(f"    blk{blk:2d}: " + " ".join(parts))
                prev = key


if __name__ == "__main__":
    main()
