#!/usr/bin/env python3
"""CAPSTONE PHASE B — ATTENTION SCALING at the 552K CODE-frame scale.

The CFM CODE-frame stream is ~552K tokens.  This measures:
  (A) peak VRAM to hold the 552K-row code KV cache resident (the code-select
      block's K/V), + the residual x for the frames.
  (B) softmax-CAM per-step cost: score the WHOLE 552K code KV per fetch query
      (O(S) per step) -- and its score-matrix VRAM.
  (C) direct-CAM per-step cost: O(1) code fetch (no softmax over 552K keys).
  (D) whether the full 552K prefix is single-GPU or needs span-split.

Reports real GPU numbers (no projection).
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def gb(dev):
    return torch.cuda.max_memory_allocated(dev) / 1e9 if dev != "cpu" else 0.0


def cur_gb(dev):
    return torch.cuda.memory_allocated(dev) / 1e9 if dev != "cpu" else 0.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=0,
                    help="0 = all 552K; else cap for a quick check")
    ap.add_argument("--n-query", type=int, default=64,
                    help="fetch queries per step-batch (the block-K)")
    args = ap.parse_args()

    from c4_min import blogspec_vocab as V
    from c4_min.blogspec_model import softmax1
    from c4_min.compact_alloc import build_compact_sparse_streaming
    from c4_min.nibble_pure_forward_complete import (
        CODE_ADDR_BITS, IMM_NIBS, _pf_cfm_enabled)

    assert _pf_cfm_enabled()
    snap = np.load(os.path.join(_HERE, "_doom_bytecode_snapshot.npz"))
    ops, imms = snap["ops"], snap["imms"]
    n_instr = len(ops)
    n_frames = n_instr if args.n_frames == 0 else min(args.n_frames, n_instr)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"n_instr={n_instr} n_frames(resident)={n_frames} dev={dev} "
          f"CODE_ADDR_BITS={CODE_ADDR_BITS}")

    model, L, _ = build_compact_sparse_streaming(
        code_size=64, recurrent_divmod=True, compute_mode="dense_kernel")
    model = model.to(dev)
    names = list(getattr(L, "_block_names", []))
    csel = names.index("code-select")
    attn = model.blocks[csel].attn
    D = model.embed.shape[1]
    H, HD = attn.n_heads, attn.head_dim
    if dev != "cpu":
        torch.cuda.reset_peak_memory_stats(dev)

    # ------------------------------------------------------------------
    # (A) Build the code-frame residual band CHUNK-WISE and project K/V so we
    # never hold a [1, 552K, D] fp32 residual (that alone is 552K*1416*4 = 3.1GB;
    # the FULL x is only needed transiently -- project per chunk, keep only K/V).
    # ------------------------------------------------------------------
    CH = 32768
    K_parts, V_parts = [], []
    t0 = time.time()
    for c0 in range(0, n_frames, CH):
        c1 = min(c0 + CH, n_frames)
        xc = torch.zeros(1, c1 - c0, D, dtype=torch.float32, device=dev)
        xc[0, :, L.ONE] = 1.0
        idx = np.arange(c0, c1)
        xc[0, :, L.IS_CODE] = 1.0
        for b in range(CODE_ADDR_BITS):
            xc[0, :, L.CODE_KEY_BIN + b] = torch.tensor(
                ((idx >> b) & 1).astype(np.float32), device=dev)
        xc[0, :, L.CODE_OPV] = torch.tensor(ops[c0:c1].astype(np.float32), device=dev)
        immc = imms[c0:c1].astype(np.int64)
        for j in range(IMM_NIBS):
            xc[0, :, L.CODE_IMM_NIB_MEM + j] = torch.tensor(
                ((immc >> (4 * j)) & 0xF).astype(np.float32), device=dev)
        with torch.no_grad():
            Kc = attn.W_k.linear(xc).view(1, c1 - c0, H, HD).transpose(1, 2)
            Vc = attn.W_v.linear(xc).view(1, c1 - c0, H, HD).transpose(1, 2)
        # keep ONLY the code-CAM head (last head) K/V -- that's the only head the
        # 552K frames feed; every other head windows them out (local) or ignores.
        chead = H - 1
        K_parts.append(Kc[:, chead:chead + 1].contiguous())
        V_parts.append(Vc[:, chead:chead + 1].contiguous())
        del xc, Kc, Vc
    K = torch.cat(K_parts, dim=2)   # [1,1,n_frames,HD]
    V = torch.cat(V_parts, dim=2)
    del K_parts, V_parts
    build_kv_s = time.time() - t0
    kv_bytes = (K.numel() + V.numel()) * K.element_size()
    print(f"(A) code KV built: K/V shape={tuple(K.shape)} "
          f"resident_KV={kv_bytes/1e9:.3f}GB build_kv={build_kv_s:.1f}s "
          f"peak_so_far={gb(dev):.2f}GB cur={cur_gb(dev):.2f}GB")

    # code-CAM head Q/K slopes (slope 0 -> position-invariant); take the ONE head.
    chead = H - 1
    slope = attn.alibi_slopes[chead:chead + 1].to(dev)
    scale = attn.scale

    # ------------------------------------------------------------------
    # (B) SOFTMAX-CAM per-step: score n_query fetch queries vs ALL n_frames keys.
    # This is the O(S) global-CAM cost the direct-CAM removes.  Measure both the
    # per-step latency and whether the [n_query, n_frames] score matrix fits.
    # ------------------------------------------------------------------
    nq = args.n_query
    # build query rows (random sampled PCs) -> Q of the code head.
    rng = np.random.default_rng(0)
    qpcs = rng.integers(0, n_instr, size=nq)
    xq = torch.zeros(1, nq, D, dtype=torch.float32, device=dev)
    xq[0, :, L.ONE] = 1.0
    xq[0, :, L.IS_FETCH] = 1.0
    for b in range(CODE_ADDR_BITS):
        xq[0, :, L.CODE_QRY_BIN + b] = torch.tensor(
            ((qpcs >> b) & 1).astype(np.float32), device=dev)
    with torch.no_grad():
        Qc = attn.W_k.linear(xq)  # placeholder; use W_q
        Qc = attn.W_q.linear(xq).view(1, nq, H, HD).transpose(1, 2)[:, chead:chead+1]

    def softmax_cam_step():
        with torch.no_grad():
            sc = torch.matmul(Qc, K.transpose(-2, -1)) * scale   # [1,1,nq,n_frames]
            # slope 0 -> no ALiBi dist term; causal not needed (code frames precede)
            a = softmax1(sc, dim=-1)
            out = torch.matmul(a, V)   # [1,1,nq,HD]
        return out

    # DECODE-CORRECTNESS of the softmax-CAM against ALL n_frames resident keys:
    # apply the head's W_o (isolated to the code head's HD slice) and decode
    # OP_VAL/IMM_NIB, compare to snapshot code[qpc].  This is the airtight full-
    # program aliasing test (each query scored vs every resident frame).
    def softmax_cam_decode_check():
        o = softmax_cam_step()  # [1,1,nq,HD]
        # W_o maps head-slot v0 -> OP_VAL, v0+1+j -> IMM_NIB[j].
        v0 = CODE_ADDR_BITS + 4
        n_ok = 0
        for i in range(nq):
            pc = int(qpcs[i])
            if pc >= n_frames:
                continue  # frame not resident in a capped run
            op_dec = int(round(float(o[0, 0, i, v0].item())))
            imm_dec = 0
            for j in range(IMM_NIBS):
                imm_dec |= (int(round(float(o[0, 0, i, v0 + 1 + j].item()))) & 0xF) << (4*j)
            exp_op = int(ops[pc])
            exp_imm = int(imms[pc]) & ((1 << (4*IMM_NIBS)) - 1)
            n_ok += (op_dec == exp_op and imm_dec == exp_imm)
        return n_ok

    torch.cuda.reset_peak_memory_stats(dev) if dev != "cpu" else None
    try:
        # warmup + time
        o = softmax_cam_step()
        if dev != "cpu":
            torch.cuda.synchronize(dev)
        t1 = time.time()
        REP = 5
        for _ in range(REP):
            o = softmax_cam_step()
        if dev != "cpu":
            torch.cuda.synchronize(dev)
        sm_ms = (time.time() - t1) / REP * 1000.0
        sm_peak = gb(dev)
        sm_ok = True
        sm_decode_ok = softmax_cam_decode_check()
        n_resident_q = sum(1 for pc in qpcs if pc < n_frames)
        print(f"(B) SOFTMAX-CAM DECODE vs ALL {n_frames} resident frames: "
              f"{sm_decode_ok}/{n_resident_q} queries byte-exact")
    except torch.cuda.OutOfMemoryError as e:
        sm_ms, sm_peak, sm_ok = float("nan"), float("nan"), False
        print(f"(B) SOFTMAX-CAM OOM at n_frames={n_frames} nq={nq}: {str(e)[:120]}")
    if sm_ok:
        score_bytes = 1 * 1 * nq * n_frames * 4
        print(f"(B) SOFTMAX-CAM: {sm_ms:.2f} ms / {nq}-query batch "
              f"({sm_ms/nq:.3f} ms/step)  score_matrix={score_bytes/1e9:.3f}GB "
              f"peak={sm_peak:.2f}GB")

    # ------------------------------------------------------------------
    # (C) DIRECT-CAM per-step: O(1) gather of code[pc] (the draft-resolved fetch),
    # no score over n_frames.  Reconstruct the head V vector from code[pc].
    # ------------------------------------------------------------------
    v0 = CODE_ADDR_BITS + 4
    def direct_cam_step():
        with torch.no_grad():
            out = torch.zeros(1, 1, nq, HD, device=dev)
            # gather is O(nq): read op + imm nibbles for each query pc directly.
            out[0, 0, :, v0] = torch.tensor(ops[qpcs].astype(np.float32), device=dev)
            for j in range(IMM_NIBS):
                out[0, 0, :, v0 + 1 + j] = torch.tensor(
                    ((imms[qpcs] >> (4*j)) & 0xF).astype(np.float32), device=dev)
        return out
    torch.cuda.reset_peak_memory_stats(dev) if dev != "cpu" else None
    o2 = direct_cam_step()
    if dev != "cpu":
        torch.cuda.synchronize(dev)
    t2 = time.time()
    for _ in range(20):
        o2 = direct_cam_step()
    if dev != "cpu":
        torch.cuda.synchronize(dev)
    dc_ms = (time.time() - t2) / 20 * 1000.0
    dc_peak = gb(dev)
    print(f"(C) DIRECT-CAM: {dc_ms:.3f} ms / {nq}-query batch "
          f"({dc_ms/nq:.4f} ms/step)  peak(step only)={dc_peak:.2f}GB  "
          f"(O(1), no score over {n_frames})")

    # ------------------------------------------------------------------
    # (D) verdict
    # ------------------------------------------------------------------
    total_kv_all = 2 * n_instr * HD * 4 / 1e9  # both K,V one head, all frames
    print("\n" + json.dumps({
        "n_instr": n_instr, "n_frames_resident": n_frames,
        "resident_code_KV_GB": round(kv_bytes/1e9, 3),
        "softmax_cam_ms_per_step": None if not sm_ok else round(sm_ms/nq, 4),
        "softmax_cam_peak_GB": None if not sm_ok else round(sm_peak, 2),
        "direct_cam_ms_per_step": round(dc_ms/nq, 5),
        "gpu_total_GB": round(torch.cuda.get_device_properties(0).total_memory/1e9, 1)
                        if dev != "cpu" else 0,
        "single_gpu_fits_full_code_KV": kv_bytes/1e9 < 22.0,
    }, indent=2))


if __name__ == "__main__":
    main()
