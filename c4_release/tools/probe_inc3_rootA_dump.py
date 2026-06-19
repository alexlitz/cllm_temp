#!/usr/bin/env python3
"""Inc-3 ROOT A — CORRECTED characterization (2026-06-19, GPU-grounded).

THE BRIEF'S ROOT A PREMISE IS REFUTED. The brief said the multi-byte
var_simple step-5 byte-1 (0x03 for x=990) is delivered in GOLDEN by the L10
``layer10_byte_passthrough_bake.head_1`` carry head, dropped in CAMPAIGN by a
30-tok ``MEM_STORE`` broadcast pinning that head on a PC-marker row, and that
the fix is to route the head's store-select slots 44-47 onto
``MEM_STORE_AT_VAL`` (approach b) / fix the relay (a) / new head (c).

GPU-grounded probing OVERTURNS this:
  * The L10 passthrough head delivers byte 0x00 at the step5 AX[1] row in BOTH
    GOLDEN and CAMPAIGN (it attends the step3 MEM marker whose CLEAN_EMBED is
    0). It is NOT the byte-1 deliverer. Routing its slots 44-47 onto
    ``MEM_STORE_AT_VAL`` (built + byte-identical-OFF-verified + GPU-probed) DID
    correctly re-align that head's row selection to golden, but had ZERO effect
    on the emitted byte-1 (cpu_full_trace id250 still failed step5 ax=222) —
    because that head was never the deliverer. REVERTED.

  * The REAL byte-1 cross-step carry for the LI reload (``return x``) is the
    H1-ONEHOT REGISTER-DUMP machinery (the documented "AX byte-1 dump is an
    H1-one-hot architectural wall — THE #1 full-trace root"):
      carry head ``layer13_ax_byte1_dump_carry.head_7`` (blk18)
        -> ``H1_PREV_STEP`` band
        -> ``ax_byte1_dump_repopulate`` FFN
        -> ``H1_DUMP_OUT`` band
        -> LM head ``head.weight[tok, H1_DUMP_OUT+(v+2)] = +5.0``.
    At the step5 byte-1 predictor row (x=990), blk18:
      GOLDEN   : H1_PREV_STEP slot 2 (=3.92), H1_DUMP_OUT LIT  -> byte-1 = 0x03 ✓
      CAMPAIGN : H1_PREV_STEP slot 5 (=12.0), H1_DUMP_OUT DARK -> byte-1 = 0x00 ✗
    The gate signals are IDENTICAL across configs (ΣAX_CARRY≈8.2, OVERFLOW=0,
    LEV-kill not the cause — verified C4_LEV_AX_BYTE1_KILL=0 doesn't help). The
    carry head attends nearly the SAME rows in both configs (step1 AX_marker
    primary), but the 30-tok stride shifts the SOFT-MIXTURE of H1 one-hots it
    V-copies -> the wrong prev-row one-hot lands in H1_PREV_STEP -> the
    repopulate FFN's per-slot copy produces a dark/wrong H1_DUMP_OUT.

  * The failure is UNIFORM: GPU full_trace ids 250-274 campaign config =
    var_simple 5/25 (only the 5 one-byte x<256 cases pass); every multi-byte
    case fails step5 with got_ax = expected & 0xFF (byte-1 dropped). Same
    mechanism for all 20.

CONCLUSION: ROOT A is the H1-onehot register-dump wall in the 30-tok frame, NOT
the MEM_STORE broadcast. Per memory (``project_ax_byte1_dump_is_h1_onehot_wall``)
this needs a DELIBERATE TWO-PART build (carry band + re-pointed dump that tracks
the correct prior row at the 30-tok stride), flag-gated; single-head re-keys are
zero-sum. The next session should target the ``layer13_ax_byte1_dump_carry``
carry head's source-row selection (re-key its K so it copies the PREVIOUS step's
byte-1 H1 one-hot at the 30-tok stride, not the 35-tok one) + verify the
repopulate FFN re-lights H1_DUMP_OUT — NOT the L10 passthrough / MEM_STORE path.

Probe the H1_PREV_STEP + H1_DUMP_OUT one-hot at the step5 byte-1 predictor row,
GOLDEN vs CAMPAIGN, block-by-block (companion probes:
``probe_inc3_rootA_emitted.py`` for the per-step emitted bytes,
``probe_inc3_rootA_gate.py`` for the repopulate-FFN gate signals).

Run TWICE (clear ~/.cache/c4_release/compiled_vm/ between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_rootA_dump.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_rootA_dump.py
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


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def onehot_val(vec, base, n=16):
    """Decode an N-wide one-hot to its index (or -1)."""
    best = -1
    bestv = 0.4
    for i in range(n):
        v = float(vec[base + i])
        if v > bestv:
            bestv = v
            best = i
    return best, bestv


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN" if nostk else "GOLDEN"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    # The byte-1 token is emitted at the AX[0] predictor row (off=6): the
    # decode reads H1_DUMP_OUT there to produce the byte-1 token.
    off = pl + 5 * STEP + 6
    ctx = p._final_context(bytecode, max_steps=9)
    padded = torch.tensor([ctx], device=p._device)
    nblk = len(p.model.blocks)
    blk_map = p.block_layer_map()
    h1p = dp.get("H1_PREV_STEP")
    h1d = dp.get("H1_DUMP_OUT")
    h1 = dp.get("H1")
    axf_hi = dp.get("AX_FULL_HI")
    print(f"=== {cfg} STEP={STEP} off(step5 AX[0] pred)={off} "
          f"H1_PREV_STEP={h1p} H1_DUMP_OUT={h1d} ===")
    prev = None
    for blk in range(nblk):
        x = td(p.model.forward(padded, stop_after_block=blk)[0])
        row = x[off]
        pv = onehot_val(row, h1p) if h1p is not None else (-1, 0)
        dv = onehot_val(row, h1d) if h1d is not None else (-1, 0)
        hv = onehot_val(row, h1) if h1 is not None else (-1, 0)
        av = onehot_val(row, axf_hi) if axf_hi is not None else (-1, 0)
        key = (pv[0], dv[0], hv[0], av[0])
        if key != prev:
            log = blk_map[blk].get("logical") if isinstance(blk_map[blk], dict) else blk_map[blk]
            print(f"  blk{blk:2d}(L{log}): H1_PREV={pv[0]}({pv[1]:.2f}) "
                  f"H1_DUMP_OUT={dv[0]}({dv[1]:.2f}) H1={hv[0]}({hv[1]:.2f}) "
                  f"AX_FULL_HI={av[0]}({av[1]:.2f})")
            prev = key


if __name__ == "__main__":
    main()
