#!/usr/bin/env python3
"""CHK-1 deep-loop memory-fade diagnostic + first-divergence map.

Builds the sparse pure-forward model ONCE, then for each deep-loop program:
  * drafts the stream (reference VM),
  * verifies block-wise on the model (FAIL-ON-FIRST-DIVERGENCE),
  * records the EXACT first-divergence step + which register diverged.

With --instrument it also dumps, at the divergence step's LOAD query row, the raw
memory-head attention: how many stores to the queried address exist, each store's
raw address-match score, ALiBi recency penalty, combined score, and the softmax1
weight — to show whether latest-write-wins is failing to old-value fade.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import torch  # noqa: E402

import c4_min.nibble_pure_forward as _PF          # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min import isa  # noqa: E402
from c4_min.compact_alloc import build_compact_pure_forward_model  # noqa: E402
from c4_min.sparse_forward import SparseTransformer  # noqa: E402
from c4_min.pf_speculative import draft_pf_program, verify_blocks, speculative_run  # noqa: E402

_WORD = 8
_SLOT_SCALED_OPS = frozenset({isa.LEA, isa.ENT, isa.ADJ})


def _sign32(imm):
    return imm if imm < (1 << 31) else imm - (1 << 32)


def bytecode_to_isa(bytecode):
    out = []
    for word in bytecode:
        op = int(word) & 0xFF
        imm = int(word) >> 8
        if op in _SLOT_SCALED_OPS:
            simm = _sign32(imm)
            out.append(isa.Instr(op, simm // _WORD))
        else:
            out.append(isa.Instr(op, imm & 0xFFFFFFFF))
    return out


def cluster_of(description):
    base = description.split(":", 1)[0].strip()
    base = re.sub(r"_\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    return base.rstrip("_") or "misc"


DEEP_CLUSTERS = {
    "loop_sum", "gcd", "rec_factorial", "loop_countdown", "loop_mul", "loop_pow",
    "nested_quad", "nested_sumsq", "rec_fib", "rec_power", "rec_sum",
}


def _find_mem_head(model, L):
    """Return (block_idx, head_idx, slope) of the LI/LC §Memory CAM head — the one
    whose ALiBi slope == MEM_ALIBI_SLOPE and whose W_q reads QRY_BIN."""
    from c4_min.blogspec_memory import MEM_ALIBI_SLOPE
    N_ROLES = _PFC.N_ROLES
    for bi, blk in enumerate(model.blocks):
        slopes = blk.attn.alibi_slopes
        HD = blk.attn.head_dim
        for h in range(blk.attn.n_heads):
            if abs(float(slopes[h]) - MEM_ALIBI_SLOPE) < 1e-6:
                # is this the LI head (reads QRY_BIN) at head N_ROLES?
                if h == N_ROLES:
                    return bi, h, float(slopes[h])
    return None, None, None


def instrument_load_at(model, L, code, draft, step, device):
    """At the divergence step's LOAD query row, dump the raw §Memory head attention:
    every candidate store row's address-match, ALiBi recency, combined score, w."""
    from c4_min.nibble_pure_forward_cached import apply_overlay_window
    from c4_min.blogspec_model import softmax1
    from c4_min.nibble_pure_forward import N_ROLES

    bi, h, slope = _find_mem_head(model, L)
    if bi is None:
        return {"error": "no mem head found"}

    # Build the full-stream residual up to and including the divergence step's
    # query row, then run every block up to the mem-cam block, and read the head's
    # raw scores at the query row.
    q_pos = draft.win_starts[step]
    toks = draft.tokens[: q_pos + 1]
    S = len(toks)
    dev = torch.device(device)
    win_toks = torch.tensor([toks], device=dev)
    x = model.embed[win_toks].clone()
    apply_overlay_window(x, 0, code, L, draft.store_log, is_last_row_query=False)
    # tag the query row with all-ROLE one-hots (the driver's per-step query tag)
    for role in range(N_ROLES):
        x[0, q_pos, L.ROLE + role] = 1.0

    # run blocks 0..bi-1 to get the residual entering the mem-cam block
    with torch.no_grad():
        for b in range(bi):
            x = model.blocks[b](x)
        attn = model.blocks[bi].attn
        HD = attn.head_dim
        base = h * HD
        Wq = attn.W_q[base:base + HD]
        Wk = attn.W_k[base:base + HD]
        q = torch.nn.functional.linear(x[0], Wq)      # [S, HD]
        k = torch.nn.functional.linear(x[0], Wk)      # [S, HD]
        scale = attn.scale
        qrow = q[q_pos]                                # [HD]
        raw = (k @ qrow) * scale                       # [S] content score (pre-ALiBi)
        pos = torch.arange(S, device=dev)
        dist = (q_pos - pos).clamp(min=0).float()
        alibi = slope * dist
        combined = raw - alibi
        combined[pos > q_pos] = float("-inf")
        w = softmax1(combined.unsqueeze(0), dim=-1)[0]  # [S]

    # which store rows exist and their queried address? the load queries
    # QRY_BIN == AX_VAL bits. Identify store rows from the draft store_log.
    fr = draft.frames[step]
    # figure out the load address: it's the AX going INTO the LI (the previous ax
    # before this step's LI overwrote it). Reconstruct from store_log: the address
    # this load matched is whatever store the model recalled. Report top rows.
    store_rows = {}   # abs_pos -> (addr, val)
    for f_idx, (addr, val) in draft.store_log.items():
        # frame f_idx's MEM marker abs pos:
        from c4_min import blogspec_vocab as V
        # MEM marker local slot:
        mem_local = _PF._MEM_MARKER_LOCAL
        abs_pos = 1 + f_idx * V.FRAME_LEN + mem_local
        if abs_pos <= q_pos:
            store_rows[abs_pos] = (addr, val)

    # top-8 rows by softmax weight
    topw, topi = torch.topk(w, min(8, S))
    rows = []
    for wi, pi in zip(topw.tolist(), topi.tolist()):
        info = store_rows.get(pi)
        rows.append({
            "pos": pi, "w": round(wi, 6),
            "raw_score": round(float(raw[pi]), 3),
            "alibi_pen": round(float(alibi[pi]), 3),
            "combined": round(float(combined[pi]), 3),
            "is_store": info is not None,
            "store_addr": info[0] if info else None,
            "store_val": info[1] if info else None,
            "dist": int(q_pos - pi),
        })
    sink_w = 1.0 - float(w.sum())
    # the queried address: find the store row with the max raw_score (exact match)
    exact = max(store_rows.items(), key=lambda kv: float(raw[kv[0]]),
                default=(None, (None, None)))
    n_stores_matching = 0
    match_addr = exact[1][0] if exact[0] is not None else None
    if match_addr is not None:
        n_stores_matching = sum(1 for _, (a, _v) in store_rows.items() if a == match_addr)
    return {
        "mem_block": bi, "mem_head": h, "slope": slope,
        "query_pos": q_pos, "seq_len": S, "n_store_rows": len(store_rows),
        "matched_addr": match_addr,
        "n_stores_to_matched_addr": n_stores_matching,
        "sink_weight": round(sink_w, 6),
        "top_rows": rows,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--per-cluster", type=int, default=2,
                    help="max programs per cluster")
    ap.add_argument("--block-steps", type=int, default=48)
    ap.add_argument("--max-steps", type=int, default=300000)
    ap.add_argument("--no-evict", action="store_true")
    ap.add_argument("--prune-interval", type=int, default=120)
    ap.add_argument("--instrument", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    all_tests = generate_test_programs()

    keep = DEEP_CLUSTERS
    if args.clusters:
        keep = {c.strip() for c in args.clusters.split(",") if c.strip()}

    by_cluster = defaultdict(list)
    for i, (s, e, d) in enumerate(all_tests):
        cl = cluster_of(d)
        if cl in keep:
            by_cluster[cl].append(i)
    ids = []
    for cl in sorted(by_cluster):
        ids.extend(by_cluster[cl][: args.per_cluster])

    t0 = time.monotonic()
    print(f"[diag:{device}] building model ...", file=sys.stderr, flush=True)
    compact, L, cstats = build_compact_pure_forward_model(
        code_size=64, include_bitwise=False, include_divmod=True)
    sparse = SparseTransformer(compact, compute_mode="dense_kernel")
    del compact
    sparse = sparse.to(device)
    print(f"[diag:{device}] built in {time.monotonic()-t0:.0f}s | "
          f"dim={sparse.dim} blocks={len(sparse.blocks)} | {len(ids)} programs",
          file=sys.stderr, flush=True)

    evict = not args.no_evict
    results = []
    for idx in ids:
        source, expected, description = all_tests[idx]
        cl = cluster_of(description)
        try:
            code = bytecode_to_isa(compile_c(source)[0])
        except Exception as exc:
            results.append(dict(idx=idx, cluster=cl, status="COMPILE_ERR",
                                detail=repr(exc)))
            continue
        draft = draft_pf_program(code, max_steps=args.max_steps)
        if not draft.halted:
            results.append(dict(idx=idx, cluster=cl, status="DRAFT_TIMEOUT",
                                steps=draft.step_count))
            continue
        vr = verify_blocks(sparse, L, code, draft, block_steps=args.block_steps,
                           device=device, evict=evict,
                           prune_interval=args.prune_interval)
        rec = dict(idx=idx, cluster=cl, steps=draft.step_count,
                   expected=expected & 0xFFFFFFFF)
        if vr.all_matched:
            rec["status"] = "PASS"
            rec["final_ax"] = vr.decoded_final_ax
            rec["first_div_step"] = None
        else:
            fm = vr.first_mismatch
            rec["status"] = "DIVERGE"
            rec["first_div_step"] = fm["step"]
            rec["first_div_pos"] = fm["query_pos"]
            rec["got"] = fm["got"]
            rec["want"] = fm["want"]
            # which register(s) diverged
            diverged = [k for k in ("pc", "ax", "sp", "bp")
                        if fm["got"][k] != fm["want"][k]]
            rec["diverged_regs"] = diverged
            fr = draft.frames[fm["step"]]
            rec["op_at_step"] = fr["op"]
            rec["max_cache"] = vr.max_cache_size
            if args.instrument and "ax" in diverged:
                try:
                    rec["mem_probe"] = instrument_load_at(
                        sparse, L, code, draft, fm["step"], device)
                except Exception as exc:
                    rec["mem_probe"] = {"error": repr(exc)}
        results.append(rec)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        st = rec["status"]
        extra = ""
        if st == "DIVERGE":
            extra = (f" step={rec['first_div_step']} op={rec['op_at_step']} "
                     f"regs={rec['diverged_regs']} got={rec['got']} want={rec['want']}")
        print(f"[diag] id={idx:4d} [{cl:14s}] {st} steps={rec.get('steps')}{extra}",
              file=sys.stderr, flush=True)

    # summary map
    print("\n" + "=" * 78)
    print(f"FIRST-DIVERGENCE MAP  (evict={evict})")
    print("=" * 78)
    per_cl = defaultdict(list)
    for r in results:
        per_cl[r["cluster"]].append(r)
    for cl in sorted(per_cl):
        rs = per_cl[cl]
        npass = sum(1 for r in rs if r["status"] == "PASS")
        print(f"\n  {cl}  ({npass}/{len(rs)} pass):")
        for r in rs:
            if r["status"] == "PASS":
                print(f"    id={r['idx']:4d}  PASS  steps={r.get('steps')} "
                      f"final_ax={r.get('final_ax')}")
            elif r["status"] == "DIVERGE":
                print(f"    id={r['idx']:4d}  DIVERGE@step {r['first_div_step']} "
                      f"op={r['op_at_step']} regs={r['diverged_regs']} "
                      f"got={r['got']} want={r['want']}")
                if r.get("mem_probe") and "top_rows" in r["mem_probe"]:
                    mp = r["mem_probe"]
                    print(f"        mem: matched_addr={mp['matched_addr']} "
                          f"n_stores_to_addr={mp['n_stores_to_matched_addr']} "
                          f"sink_w={mp['sink_weight']} seq={mp['seq_len']}")
                    for row in mp["top_rows"][:5]:
                        print(f"          pos={row['pos']:6d} w={row['w']:.5f} "
                              f"raw={row['raw_score']:.1f} alibi={row['alibi_pen']:.1f} "
                              f"comb={row['combined']:.1f} dist={row['dist']:5d} "
                              f"store={row['is_store']} addr={row['store_addr']} "
                              f"val={row['store_val']}")
            else:
                print(f"    id={r['idx']:4d}  {r['status']} {r.get('detail','')}")

    if args.output:
        json.dump({"evict": evict, "results": results}, open(args.output, "w"),
                  indent=2)
        print(f"\n[diag] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
