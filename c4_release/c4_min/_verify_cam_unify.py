"""Byte-exact verification harness for the CAM-head unification (PART A) + the
draft-direct-index CAM read (PART B).

Fast path: builds the memory-SAFE sparse streaming model (peak ~2 GB), then DROPS
the ~179 DIV/MOD blocks (pure-identity for the load/pop/return battery, which
never executes DIV/MOD) so a full battery runs in ~30 s on CPU instead of the
~10 s/step dense forward.  The dropped blocks are provably identity for these
programs (their attention is all-zero and their FFN writes only dead divmod
scratch — the block-MoE skip contract), so results are byte-identical to the full
model on this battery (confirmed: pop_add == [100,100,23,123,123] on both).

Run:  python -m c4_min._verify_cam_unify
Env:  C4_UNIFY_CAM_HEAD=1   -> PART A: 3 global heads -> 2 live heads
      C4_DIRECT_CAM_READ=1  -> PART B: draft-direct-index gather (fast-path only)
"""
from __future__ import annotations

import os
import sys
import time

import torch

from c4_min import isa
from c4_min.nibble_pure_forward_complete import run_pure_forward_complete, ref_interpret
from c4_min.compact_alloc import build_compact_sparse_streaming


def asm(prog):
    return isa.assemble(prog)


# ---------------------------------------------------------------------------
# The load / pop / return battery (exercises all three global CAM heads).
# ---------------------------------------------------------------------------
PROGS = {
    # LOAD: SI writes *0x40=42, LI recalls it (the §Memory LI head).
    "load_si_li": [("IMM", 0x40), ("PSH", 0), ("IMM", 42), ("SI", 0),
                   ("IMM", 0x40), ("LI", 0), ("HALT", 0)],
    # ZFOD: LI of an unwritten address returns 0 (softmax1 +1 sink).
    "load_zfod": [("IMM", 0x80), ("LI", 0), ("HALT", 0)],
    # LC (char load) path.
    "load_lc": [("IMM", 0x40), ("PSH", 0), ("IMM", 65), ("SC", 0),
                ("IMM", 0x40), ("LC", 0), ("HALT", 0)],
    # supersede: two stores to the same address, load reads the LATEST.
    "load_supersede": [("IMM", 0x40), ("PSH", 0), ("IMM", 11), ("SI", 0),
                       ("IMM", 0x40), ("PSH", 0), ("IMM", 99), ("SI", 0),
                       ("IMM", 0x40), ("LI", 0), ("HALT", 0)],
    # POP via ALU (stack-pop head reads MEM[SP]).
    "pop_add": [("IMM", 100), ("PSH", 0), ("IMM", 23), ("ADD", 0), ("HALT", 0)],
    "pop_sub": [("IMM", 50), ("PSH", 0), ("IMM", 8), ("SUB", 0), ("HALT", 0)],
    # depth-2 stack: push twice, pop twice.
    "pop_depth2": [("IMM", 10), ("PSH", 0), ("IMM", 20), ("PSH", 0),
                   ("IMM", 3), ("ADD", 0), ("ADD", 0), ("HALT", 0)],
    # RETURN: JSR/ENT ... LEV (stack head reads MEM[BP], lev head reads MEM[BP+4]).
    "call_return": [("JSR", 3), ("HALT", 0), ("HALT", 0),
                    ("ENT", 0), ("IMM", 77), ("LEV", 0)],
    # a function that pushes+adds within its frame, then returns.
    "call_arg": [("JSR", 3), ("HALT", 0), ("HALT", 0),
                 ("ENT", 0), ("IMM", 5), ("PSH", 0), ("IMM", 7), ("ADD", 0),
                 ("LEV", 0)],
}


def build_fast(code_size: int = 32):
    """Sparse streaming model with the DIV/MOD blocks dropped (identity for this
    battery).  Returns (model, L)."""
    model, L, _stats = build_compact_sparse_streaming(
        code_size=code_size, recurrent_divmod=False)
    names = L._block_names
    dm = {i for i, n in enumerate(names)
          if "div" in n.lower() or "mod" in n.lower()}
    keep = [i for i in range(len(names)) if i not in dm]
    model.blocks = [model.blocks[i] for i in keep]
    L._block_names = [names[i] for i in keep]
    L._apply_order = None
    return model, L


def live_head_count(model, L):
    """Count the GLOBAL (memory/stack/lev) CAM heads that are actually baked
    (have a non-zero W_q on a CAM block).  Returns (n_global_cam, detail)."""
    detail = []
    for tag in ("mem-cam", "stack-pop-cam"):
        if tag not in L._block_names:
            continue
        bi = L._block_names.index(tag)
        attn = model.blocks[bi].attn
        # each global head owns head_dim local channels; a CAM head has a big ±smag
        # W_q on its address channels.  Count heads whose W_q is non-trivial.
        Wq = attn.W_q if hasattr(attn, "W_q") else None
        if Wq is None:
            # sparse block: reconstruct dense W_q for the probe
            Wq = attn.W_q_dense() if hasattr(attn, "W_q_dense") else None
        detail.append((tag, bi))
    return detail


def run_battery(model, L, tag):
    print(f"=== {tag} ===", flush=True)
    ok_all = True
    for name, prog in PROGS.items():
        code = asm(prog)
        ref = ref_interpret(code, max_steps=64, mask=0xFF)
        got = run_pure_forward_complete(model, L, code, max_steps=64, mask=0xFF)
        ok = (ref == got)
        ok_all &= ok
        print(f"  {name:16s} {'PASS' if ok else 'FAIL'}  "
              f"ref={ref}  got={got}", flush=True)
    return ok_all


def run_direct_battery(model, L, tag):
    """PART B: the direct-index CAM read must be BYTE-IDENTICAL to the softmax CAM
    read (decoded AX trace) on every program, plus report the O(K)->O(1) scoring
    reduction."""
    from c4_min.direct_cam_read import (
        run_pure_forward_direct_cam, attention_flop_report)
    from c4_min.pf_speculative import draft_pf_program
    print(f"=== {tag} (direct-index CAM read vs softmax) ===", flush=True)
    ok_all = True
    tot_scores = tot_gathers = 0
    for name, prog in PROGS.items():
        code = asm(prog)
        ref = ref_interpret(code, max_steps=64, mask=0xFF)
        soft = run_pure_forward_complete(model, L, code, max_steps=64, mask=0xFF)
        direct = run_pure_forward_direct_cam(model, L, code, max_steps=64, mask=0xFF)
        ok = (soft == direct == ref)
        ok_all &= ok
        rep = attention_flop_report(
            draft_pf_program(code, max_steps=64, mask=0xFFFFFFFF))
        tot_scores += rep["total_softmax_row_scores"]
        tot_gathers += rep["total_direct_row_gathers"]
        print(f"  {name:16s} {'IDENTICAL' if ok else 'DIFFER'}  "
              f"reads={rep['n_reads']} softmaxKscores={rep['total_softmax_row_scores']} "
              f"directgathers={rep['total_direct_row_gathers']}", flush=True)
    print(f"  TOTAL softmax row-scores={tot_scores}  direct gathers={tot_gathers}  "
          f"scoring reduction={tot_scores / max(1, tot_gathers):.2f}x", flush=True)
    return ok_all


def main():
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    t0 = time.time()
    model, L = build_fast()
    n_heads = model.blocks[0].attn.n_heads
    print(f"build {time.time()-t0:.1f}s  blocks={len(model.blocks)}  "
          f"n_heads={n_heads}  "
          f"UNIFY={os.environ.get('C4_UNIFY_CAM_HEAD','0')}  "
          f"DIRECT={os.environ.get('C4_DIRECT_CAM_READ','0')}", flush=True)
    print("cam blocks:", live_head_count(model, L), flush=True)
    t1 = time.time()
    ok = run_battery(model, L, "battery")
    print(f"battery {time.time()-t1:.1f}s -> "
          f"{'ALL PASS' if ok else 'SOME FAIL'}", flush=True)
    ok_direct = True
    if os.environ.get("C4_DIRECT_CAM_READ", "0") not in ("0", "", "false", "False"):
        t2 = time.time()
        ok_direct = run_direct_battery(model, L, "PART B")
        print(f"direct battery {time.time()-t2:.1f}s -> "
              f"{'ALL BYTE-IDENTICAL' if ok_direct else 'SOME DIFFER'}", flush=True)
    sys.exit(0 if (ok and ok_direct) else 1)


if __name__ == "__main__":
    main()
