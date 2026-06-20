#!/usr/bin/env python3
"""OUTBAND-2 campaign: attribute the blk41 (L25 tail) + blk26 (L14) FFN units
that drive the OUTPUT-band #2 verdict flip on the if_gt FALSE GT-step AX-marker
predictor row.

Builds the CAMPAIGN-config model (C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1),
runs a clean oracle tape for an if_gt FALSE program, finds the GT-step
AX-marker predictor row, and:
  (a) per-block residual scan of OUTPUT_LO/HI to localize the seed/explosion/
      sign-inversion blocks,
  (b) at a chosen block (default 41) attributes the FFN units by |contribution|
      to the dominant OUTPUT byte cell, printing the owning RULE NAME so the
      sign-inverting unit is named (not guessed).

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
      python tools/probe_outband2_blk41_attrib.py [SRC_OR_ID] [BLK]
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import warnings
warnings.filterwarnings("ignore")
import sys
import contextlib
import io

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def dense(w):
    return w.to_dense() if w.layout != torch.strided else w


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "int main(){ if (41 > 70) return 1; return 0; }"
    BLK = int(sys.argv[2]) if len(sys.argv) > 2 else 41
    STEP = int(Token.STEP_TOKENS)
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN" if nostk else "GOLDEN"

    src = arg
    bc, data = compile_c(src)
    with contextlib.redirect_stdout(io.StringIO()):
        p = build_groundtruth_probe()
        from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
            compile_full_vm_dynamic)
        _m, layout = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(layout.dim_positions)
    OL = dp["OUTPUT_LO"]
    OH = dp["OUTPUT_HI"]
    dev = p._device

    # Build a clean teacher-forced (oracle) tape.
    ctx = p._final_context(bc, max_steps=10)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    plen = len(p._build_context(bc))
    nsteps = (len(ctx) - plen) // STEP
    bl_map = p.block_layer_map()
    nblk = len(p.model.blocks)

    # Find the GT/cmp step: scan each step's AX-marker predictor row for the
    # one whose final argmax disagrees with byte 0 (the verdict-flip step).
    head_w = p.model.head.weight
    with torch.no_grad():
        full = p.model.forward(padded)[0]
    print(f"=== {cfg} STEP={STEP} src={src!r} nsteps={nsteps} OUTPUT_LO={OL} ===")
    # AX marker predictor row offset within a step: the AX value byte is
    # emitted right after the AX marker. Print argmax at every step's AX row.
    cand_rows = []
    for s in range(nsteps):
        base = plen + s * STEP
        for off in range(STEP):
            row = base + off
            if row + 1 >= len(ctx):
                continue
            am = int(full[row].argmax().item())
            tgt = int(ctx[row + 1])
            if am != tgt:
                cand_rows.append((s, off, row, am, tgt))
    print(f"  divergent rows (step,off,row,argmax,oracle): {cand_rows[:12]}")

    # Choose the row to attribute: env override else first divergent row else
    # the AX-marker predictor row of PROBE_STEP (default last full step).
    row_env = os.environ.get("PROBE_ROW")
    pstep = os.environ.get("PROBE_STEP")
    if row_env:
        row = int(row_env)
    elif pstep is not None:
        # AX value byte is predicted by the row right before it. We locate the
        # AX marker token in the step and use that row (+ offset via PROBE_OFF).
        off = int(os.environ.get("PROBE_OFF", "5"))
        row = plen + int(pstep) * STEP + off
    elif cand_rows:
        row = cand_rows[0][2]
    else:
        row = plen + (nsteps - 1) * STEP + 5
    row = min(row, len(ctx) - 2)
    print(f"  attributing predicting-row={row} (predicts pos {row+1}, "
          f"oracle byte={int(ctx[row+1]) if row+1 < len(ctx) else '?'})")

    # Per-block residual scan at that row.
    prev_lo = prev_hi = 0.0
    print("  --- per-block OUTPUT-band scan ---")
    with torch.no_grad():
        for blk in range(nblk):
            resid = p.model.forward(padded, stop_after_block=blk)[0, row]
            lo_vec = resid[OL:OL + 16]
            hi_vec = resid[OH:OH + 16]
            lo = float(lo_vec.abs().max().item())
            hi = float(hi_vec.abs().max().item())
            klo = int(lo_vec.abs().argmax().item())
            if abs(lo - prev_lo) > 1e3 or abs(hi - prev_hi) > 1e3:
                lm = bl_map[blk]
                print(f"   blk{blk:2d} (L{lm['logical']:2d} {lm['ffn'][:24]:24s}) "
                      f"|LO|max {prev_lo:+.2e}->{lo:+.2e} (LO+{klo}={lo_vec[klo].item():+.2e}) "
                      f"|HI|max->{hi:+.2e}")
            prev_lo, prev_hi = lo, hi

    # Attribute the chosen block's FFN units.
    xin = p.model.forward(padded, stop_after_block=BLK - 1)[0]
    x = xin[row].float()
    blk_obj = p.model.blocks[BLK]
    ffn = blk_obj.ffn
    post = list(getattr(blk_obj, "post_ops", []) or [])
    # The tail bank is a post_op PureFFN appended to its block. Pick the FFN
    # that actually has W_up (the tail bank); for blk41 the bank is in post_ops.
    targets = []
    if hasattr(ffn, "W_up"):
        targets.append(("ffn", ffn))
    for j, po in enumerate(post):
        if hasattr(po, "W_up"):
            targets.append((f"post{j}", po))
    if not targets:
        print(f"  block {BLK} has no W_up FFN (type {type(ffn).__name__})")
        return

    # Determine which OUTPUT cell to attribute (the dominant LO/HI byte at row).
    resid_at = p.model.forward(padded, stop_after_block=BLK)[0, row]
    lo_vec = resid_at[OL:OL + 16]
    hi_vec = resid_at[OH:OH + 16]
    use_hi = float(hi_vec.abs().max()) > float(lo_vec.abs().max())
    band = OH if use_hi else OL
    bk = int((hi_vec if use_hi else lo_vec).abs().argmax().item())
    d = band + bk
    bandname = "OUTPUT_HI" if use_hi else "OUTPUT_LO"

    for tag, fobj in targets:
        Wup = dense(fobj.W_up).float()
        Wgate = dense(fobj.W_gate).float()
        Wdown = dense(fobj.W_down).float()
        bup = fobj.b_up.float()
        bgate = fobj.b_gate.float()
        up = Wup @ x + bup
        gate = Wgate @ x + bgate
        hidden = F.silu(up) * gate
        contrib = Wdown[d] * hidden
        order = torch.argsort(contrib.abs(), descending=True)
        tot = float(contrib.sum())
        print(f"\n=== block {BLK} [{tag}] top units by |contrib to {bandname}[{bk}]| "
              f"row={row}  Σ={tot:+.3e} ===")
        # rule name map for the tail bank
        rule_names = None
        if BLK == 41 or tag.startswith("post"):
            try:
                from neural_vm.unified_compiler.ops.l10_ops import (
                    _tail_bit32_result_correction_rules)
                rs = _tail_bit32_result_correction_rules()
                rule_names = [getattr(r, "name", "?") for r in rs]
            except Exception:
                rule_names = None
        for h in order[:16].tolist():
            c = float(contrib[h])
            if abs(c) < 1e2:
                break
            nm = ""
            if rule_names is not None and h < len(rule_names):
                nm = rule_names[h]
            print(f"  unit{h:5d}: contrib={c:+.3e} hidden={float(hidden[h]):+.3e} "
                  f"up={float(up[h]):+.2e} gate={float(gate[h]):+.2e} "
                  f"Wdown={float(Wdown[d,h]):+.3e}  {nm}")


if __name__ == "__main__":
    main()
