"""PART-1 verify: a >255-address store/load is byte-exact through the FAST path.

Store 11@0x40 and 99@0x10040 (SAME low byte 0x40), then load 0x40 -> must be 11
(no aliasing), via build_lib_model_streaming(addr32=True) run through the SPECULATIVE
verify_blocks (the fast big-K path).  Also stores/loads a value >255 to prove the
32-bit KV memory value band survives.  Compared to the model's own draft transition.
"""
from __future__ import annotations
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
from c4_min import isa
from c4_min.pf_speculative import draft_pf_program, verify_blocks


def build():
    from c4_min.lib_neural import build_lib_model_streaming
    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(code_size=64, recurrent_divmod=True,
                                             addr32=True, compute_mode="dense_kernel")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dev != "cpu":
        sparse = sparse.to(dev)
        sparse.materialize_dense(device=dev)
    print(f"[addr32] built (addr32) blocks={len(sparse.blocks)} in {time.time()-t0:.1f}s "
          f"dev={dev}", flush=True)
    return sparse, L, dev


def run_prog(sparse, L, dev, code, label, want_ax, max_steps=32):
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF)
    draft_ax = draft.frames[-1]["ax"] if draft.frames else None
    stats = {}
    vr = verify_blocks(sparse, L, code, draft, block_steps=max_steps, device=dev,
                       evict=False, mask=0xFFFFFFFF, stats=stats, fast=True)
    spec_ax = vr.decoded_final_ax
    ok = (draft_ax == want_ax and spec_ax == want_ax and vr.all_matched)
    print(f"  {label:36s} draft_ax={draft_ax}  spec_ax={spec_ax}  "
          f"want={want_ax}  matched={vr.all_matched}  {'OK' if ok else 'DIFF'}",
          flush=True)
    return ok


def main():
    sparse, L, dev = build()
    print(flush=True)
    all_ok = True

    # (A) >255 ADDRESS no-aliasing: store 11@0x40 and 99@0x10040 (same low byte),
    #     then load 0x40 -> 11.  If the address were truncated to a byte the load
    #     would alias to the 99 stored at 0x10040.
    code = isa.assemble([
        ("IMM", 0x40), ("PSH", 0), ("IMM", 11), ("SI", 0),      # mem[0x40]=11
        ("IMM", 0x10040), ("PSH", 0), ("IMM", 99), ("SI", 0),   # mem[0x10040]=99
        ("IMM", 0x40), ("LI", 0),                               # AX = mem[0x40]
        ("HALT", 0),
    ])
    all_ok &= run_prog(sparse, L, dev, code, "store 11@0x40 + 99@0x10040, load 0x40", 11)

    # (B) the MIRROR: load 0x10040 -> 99 (the high address is not aliased down).
    code = isa.assemble([
        ("IMM", 0x40), ("PSH", 0), ("IMM", 11), ("SI", 0),
        ("IMM", 0x10040), ("PSH", 0), ("IMM", 99), ("SI", 0),
        ("IMM", 0x10040), ("LI", 0),
        ("HALT", 0),
    ])
    all_ok &= run_prog(sparse, L, dev, code, "same stores, load 0x10040", 99)

    # (C) a VALUE > 255 store/load round trip (32-bit KV value band).  Store 1000
    #     at 0x40, load it back -> 1000 (proves the value band is not byte-folded).
    code = isa.assemble([
        ("IMM", 0x40), ("PSH", 0), ("IMM", 1000), ("SI", 0),    # mem[0x40]=1000
        ("IMM", 0x40), ("LI", 0),                               # AX = 1000
        ("HALT", 0),
    ])
    all_ok &= run_prog(sparse, L, dev, code, "store 1000@0x40, load -> 1000", 1000)

    print(f"\n[addr32] all fast-path >255-addr / >255-val store-load byte-exact: {all_ok}",
          flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
