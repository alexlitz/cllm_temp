#!/usr/bin/env python3
"""Instrument ONE deep-loop program's divergence load on the REAL model.

Builds the model once, drafts the target program, runs the full-stream forward up
to the divergence step's LOAD query row, and dumps the §Memory head's raw
attention: candidate store rows, address-match score, ALiBi recency, softmax1 w.
Also compares to the KVMemory reference recall. Runs on a chosen GPU.
"""
from __future__ import annotations
import argparse, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)
import torch
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0
from c4_min import isa
from c4_min import blogspec_vocab as V
from c4_min.compact_alloc import build_compact_pure_forward_model
from c4_min.sparse_forward import SparseTransformer
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from c4_min.nibble_pure_forward_cached import apply_overlay_window
from c4_min.nibble_pure_forward import N_ROLES
from c4_min.blogspec_model import softmax1
from c4_min.blogspec_memory import MEM_ALIBI_SLOPE

_WORD = 8
_SLOT = frozenset({isa.LEA, isa.ENT, isa.ADJ})
def _s32(x): return x if x < (1 << 31) else x - (1 << 32)
def b2i(bc):
    out = []
    for w in bc:
        op = int(w) & 0xFF; imm = int(w) >> 8
        if op in _SLOT: out.append(isa.Instr(op, _s32(imm) // _WORD))
        else: out.append(isa.Instr(op, imm & 0xFFFFFFFF))
    return out


def find_mem_head(model):
    for bi, blk in enumerate(model.blocks):
        for h in range(blk.attn.n_heads):
            if abs(float(blk.attn.alibi_slopes[h]) - MEM_ALIBI_SLOPE) < 1e-6 and h == N_ROLES:
                return bi, h
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--idx", type=int, default=450)
    ap.add_argument("--head", type=str, default="LI", choices=["LI", "STK", "LEV"])
    ap.add_argument("--step", type=int, default=None,
                    help="instrument this step directly (skip the memory-heavy "
                         "verify_blocks); use the known divergence step.")
    args = ap.parse_args()
    dev = torch.device(args.device)

    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    tests = generate_test_programs()
    s, e, d = tests[args.idx]
    print(f"idx={args.idx} desc={d} expected={e}")
    code = b2i(compile_c(s)[0])

    print("building model ...", file=sys.stderr, flush=True)
    compact, L, cstats = build_compact_pure_forward_model(
        code_size=64, include_bitwise=False, include_divmod=True)
    sparse = SparseTransformer(compact, compute_mode="dense_kernel")
    del compact
    sparse = sparse.to(dev)
    print("built.", file=sys.stderr, flush=True)

    draft = draft_pf_program(code, max_steps=300000)
    print(f"drafted: steps={draft.step_count} halted={draft.halted}")
    if args.step is not None:
        step = args.step
        fr = draft.frames[step]
        print(f"INSTRUMENTING step={step} (direct) op={fr['op']} "
              f"want ax={fr['ax']} pc={fr['pc']}")
    else:
        vr = verify_blocks(sparse, L, code, draft, block_steps=48, device=args.device,
                           evict=True, prune_interval=120)
        if vr.all_matched:
            print("NO DIVERGENCE (all matched)"); return 0
        fm = vr.first_mismatch
        step = fm["step"]
        print(f"FIRST DIVERGENCE: step={step} pos={fm['query_pos']} "
              f"op={draft.frames[step]['op']} got={fm['got']} want={fm['want']}")

    # dump the previous few frames context
    for s2 in range(max(0, step - 4), step + 1):
        fr = draft.frames[s2]
        print(f"  step {s2:3d}: op={fr['op']:5s} pc={fr['pc']:3d} ax={fr['ax']:5d} "
              f"sp={fr['sp']} bp={fr['bp']} store={fr['is_store']} "
              f"addr={fr['s_addr']} val={fr['s_val']}")

    # which head to probe
    bi, h = find_mem_head(sparse)
    if args.head == "LI":
        pass  # already the LI head
    print(f"\nprobing head block={bi} head={h} slope={MEM_ALIBI_SLOPE}")

    # build full-stream residual up to the divergence query row
    q_pos = draft.win_starts[step]
    toks = draft.tokens[: q_pos + 1]
    S = len(toks)
    print(f"query_pos={q_pos} seq_len={S}")
    win = torch.tensor([toks], device=dev)
    x = sparse.embed[win].clone()
    apply_overlay_window(x, 0, code, L, draft.store_log, is_last_row_query=False)
    for role in range(N_ROLES):
        x[0, q_pos, L.ROLE + role] = 1.0

    # -- FULL forward: decode the actual loaded AX at the query row (the answer) ----
    from c4_min.nibble_pure_forward_complete import _decode_reg_from_nibbles
    with torch.no_grad():
        xf = x.clone()
        for b in range(len(sparse.blocks)):
            xf = sparse.blocks[b](xf)
        got_ax = _decode_reg_from_nibbles(xf[0, q_pos], L, L.AX)
    print(f">>> DECODED AX at step {step} = {got_ax}  (want {draft.frames[step]['ax']}) "
          f"{'MATCH' if got_ax == draft.frames[step]['ax'] else 'MISMATCH'}")
    del xf

    with torch.no_grad():
        for b in range(bi):
            x = sparse.blocks[b](x)
        attn = sparse.blocks[bi].attn
        HD = attn.head_dim
        base = h * HD
        # project the whole residual then slice this head's channels
        Qall = attn.W_q.linear(x[0])          # [S, D]
        Kall = attn.W_k.linear(x[0])          # [S, D]
        q = Qall[:, base:base + HD]           # [S, HD]
        k = Kall[:, base:base + HD]           # [S, HD]
        qrow = q[q_pos]
        raw = (k @ qrow) * attn.scale
        pos = torch.arange(S, device=dev)
        dist = (q_pos - pos).clamp(min=0).float()
        alibi = MEM_ALIBI_SLOPE * dist
        comb = raw - alibi
        comb[pos > q_pos] = float("-inf")
        w = softmax1(comb.unsqueeze(0), dim=-1)[0]

    # map store rows
    mem_local = _PF._MEM_MARKER_LOCAL
    store_rows = {}
    for f_idx, (addr, val) in draft.store_log.items():
        ap2 = 1 + f_idx * V.FRAME_LEN + mem_local
        if ap2 <= q_pos:
            store_rows[ap2] = (addr, val)
    print(f"n_store_rows<=q_pos: {len(store_rows)}")

    # QRY_BIN of the query row = the load address (residual ENTERING the mem block)
    qbin = x[0, q_pos, L.QRY_BIN: L.QRY_BIN + 32]
    load_addr = 0
    setbits = []
    for b in range(32):
        if float(qbin[b]) > 0.5:
            load_addr |= (1 << b)
            setbits.append((b, round(float(qbin[b]), 3)))
    print(f"LOAD ADDRESS (QRY_BIN decoded) = {load_addr} (0x{load_addr:x})")
    print(f"  QRY_BIN set bits (bit,val): {setbits}")
    axv = float(x[0, q_pos, L.AX_VAL]) if hasattr(L, "AX_VAL") else None
    print(f"  AX_VAL (entering mem block) = {axv}")
    # raw QRY_BIN values for bits 0..10
    print(f"  QRY_BIN[0..10] = {[round(float(qbin[b]),3) for b in range(11)]}")
    # stores to this exact address:
    matching = [(p, v) for p, (a, v) in store_rows.items() if a == load_addr]
    matching.sort()
    print(f"stores to load_addr {load_addr}: {len(matching)} -> "
          f"{[(p, v, int(q_pos - p)) for p, v in matching[-6:]]}  (pos,val,dist)")
    exact_pos = int(raw.argmax().item())
    exact_addr = store_rows.get(exact_pos, (None, None))[0]
    print(f"top raw-score store pos={exact_pos} addr={exact_addr}")

    sink = 1.0 - float(w.sum())
    print(f"sink_weight (softmax1 +1)={sink:.6f}")
    topw, topi = torch.topk(w, min(12, S))
    print("TOP rows by softmax1 weight:")
    for wi, pi in zip(topw.tolist(), topi.tolist()):
        info = store_rows.get(pi)
        print(f"  pos={pi:6d} w={wi:.6f} raw={float(raw[pi]):8.2f} "
              f"alibi={float(alibi[pi]):8.2f} comb={float(comb[pi]):8.2f} "
              f"dist={int(q_pos-pi):5d} store={info is not None} "
              f"addr={info[0] if info else None} val={info[1] if info else None}")

    # ALSO: show the highest raw-score store rows (address matches) regardless of w
    print("\nTOP rows by RAW address-match score (the exact-address candidates):")
    rvals, ridx = torch.topk(raw, min(10, S))
    for rv, pi in zip(rvals.tolist(), ridx.tolist()):
        info = store_rows.get(pi)
        print(f"  pos={pi:6d} raw={rv:8.2f} alibi={float(alibi[pi]):8.2f} "
              f"comb={float(comb[pi]):8.2f} w={float(w[pi]):.6f} dist={int(q_pos-pi):5d} "
              f"store={info is not None} addr={info[0] if info else None} "
              f"val={info[1] if info else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
