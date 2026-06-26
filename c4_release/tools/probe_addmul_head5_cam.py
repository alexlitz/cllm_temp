#!/usr/bin/env python3
"""Trace L8 head-5 (mem[SP] operand-A CAM) attention at the depth-2 ADD step.

============================================================================
ROOT + VALIDATED FIX SPEC (expr_add_mul depth-2 operand-A read), 2026-06-24
============================================================================
expr_add_mul (a+b*c) is the ONLY expr cluster with a genuine depth-2 stack
(PSH a, PSH b, IMM c, MUL pops b, ADD pops a -- TWO live stores at once). The
ADD must read operand-A = mem[SP] = a (the FIRST push; top-of-stack after the
MUL popped b). The recency-only L8 head-5 CAM picks the most-recent matching
STORE row (the b push) instead -> AX = b + b*c (id816 4+5*2: 5+10=15 vs 14).
The single-store clusters (mul_div/mod/paren) are UNAFFECTED (recency is
correct when only one store is live).

WHY recency fails / no existing discriminator (GPU+CPU, BUILT dims):
  * Both store value rows are byte-identical except the value nibble:
    MEM_STORE_AT_VAL=1, MEM_VAL_B1=0.97 on BOTH; ADDR_B0/1/2 = 0 (the L4
    ADDR_KEY staging is disabled in the campaign build); the EMITTED mem
    address byte is 0xE0 for BOTH stores. So there is NO per-row address.
  * The ONLY real discriminator is the SP at push time: store a was pushed at
    SP=0xF8, store b at SP=0xF0; after the MUL pop SP returns to 0xF8, so the
    ADD's operand address is 0xF8 == a's frame.
  * The SP low-byte IS a clean one-hot in OUTPUT_LO at the MARK_SP marker row
    (id816: row108->8 (0xF8), row168->0 (0xF0), row228 (MUL step)->8 (0xF8)).

FIX (validated end-to-end on CPU; demotion-by-40 proven via attn hook sim):
  (1) New campaign band SP_ADDR_LO (16-cell one-hot, flag-gated, never_share).
  (2) Relay head (L7, block 9 -- BEFORE head-5 reads at block 11): Q fires at
      MEM store value rows (MEM_VAL_B1) + binary-op AX markers (the head-5
      query rows); K matches MARK_SP; steep ALiBi recency picks the NEAREST
      PRIOR MARK_SP; V-copies its OUTPUT_LO (16 cells) -> SP_ADDR_LO. Mirrors
      make_layer7_mem_store_relay_op exactly. Validated:
         ADD query @253 SP_ADDR_LO=8;  store@123(a=4) SP_ADDR_LO=8 (MATCH);
         store@183(b=5) SP_ADDR_LO=0 (MISMATCH -> demote).
  (3) head-5 SP-mismatch penalty (free slots 30-46): 16 cell dims
      K[base+30+i]=SP_ADDR_LO+i (store side), Q[base+30+i]=G*SP_ADDR_LO+i
      (query side) + one const dim K=1/Q=-G, so score += G*(match-1) = 0 on
      match, -G on mismatch. G~=40 (proven: demoting the popped row by 40 flips
      head-5 ALU_LO 5->4 i.e. AX 15->14). Match==0 leaves single-store ops
      byte-identical; the popped store is demoted so recency picks the live one.
  Campaign + own kill-switch (e.g. C4_L8_OPERAND_SP_DISC) gated; golden 35-tok
  flag-OFF byte-identical (band + relay + head-5 dims only baked under the flag).
  COORDINATE: shared L7/L8 surface -- run lint_cross_op_attention + flag_
  regression_gate (the held clusters' live store must SP-match its query so the
  penalty stays 0 there).

This probe dumps, at the ADD MARK_AX query row, head-5's per-candidate score and
the dims that distinguish the live store from the popped store.

Usage: CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       C4_VM_CACHE_DIR=/tmp/c4cache_addmul python tools/probe_addmul_head5_cam.py 816 6
       (CUDA_VISIBLE_DEVICES="" forces CPU when the GPUs are contended.)
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)
import math
import torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src)
    return bc, exp, desc


def _ax_rows(ctx):
    return [i for i, t in enumerate(ctx) if t == Token.REG_AX]


def _l8_attn_block(model):
    """Physical block hosting L8 attn head 5 (layer8_sp_gather anchor)."""
    # head-5 writes ALU before the L8 ALU FFN; it lives at the L8 attn block.
    # Find the block whose attn has the most heads near L8 (post-_expand: ~block 11).
    # Simpler: scan for the block whose attn.num_heads matches the L8 layout and
    # whose W_o writes ALU_LO. We hard-probe candidate blocks.
    return None


def main(idx, step):
    p = build_groundtruth_probe()
    dp = p.model.dim_positions
    BD = dp
    bc, exp, desc = _corpus(idx)
    ctx = p._final_context(bc, max_steps=12)
    ax = _ax_rows(ctx)
    print(f"id={idx} {desc} exp={exp}  ax_rows={ax}", flush=True)
    qpos = ax[step]
    print(f"ADD step={step} query(MARK_AX) pos={qpos}", flush=True)

    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    seq = padded.shape[1]

    # Find the L8 attn block + head 5 base by replicating the head-allocator.
    from neural_vm.unified_compiler.ops.l8_ops import _L8_HEAD_LAYOUT_BY_NAME
    h5 = _L8_HEAD_LAYOUT_BY_NAME["layer8_mem_to_alu.head_5"]

    # Identify the L8 attn block: the one whose attn.W_o writes ALU_LO from head5 slots.
    cand = []
    for bi, blk in enumerate(p.model.blocks):
        attn = getattr(blk, "attn", None)
        if attn is None or not hasattr(attn, "W_q"):
            continue
        HD = attn.W_q.shape[0] // attn.num_heads
        if h5 >= attn.num_heads:
            continue
        base = h5 * HD
        # head-5 signature: W_q[base, MARK_AX]==2000, W_k[base+4, MEM_STORE_AT_VAL] set
        if abs(float(attn.W_q[base, BD["MARK_AX"]]) - 2000.0) < 1.0:
            cand.append((bi, base, HD, attn))
    print(f"L8 head-5 candidate blocks: {[c[0] for c in cand]}", flush=True)
    if not cand:
        print("NO head-5 block found (flag off?)"); return
    bi, base, HD, attn = cand[0]

    # Run forward up to the INPUT of that block (resid after block bi-1).
    resid = p.model.forward(padded, stop_after_block=bi - 1)[0]  # [S, D]

    # Recompute head-5 attention scores manually.
    def _dense(W):
        return W.to_dense() if W.is_sparse or getattr(W, "is_sparse_csr", False) else W
    Wq_full = _dense(attn.W_q)
    Wk_full = _dense(attn.W_k)
    Wq = Wq_full[base:base + HD]   # [HD, D]
    Wk = Wk_full[base:base + HD]
    q = resid[qpos] @ Wq.T          # [HD]
    K = resid @ Wk.T                # [S, HD]
    scores = (K @ q) / math.sqrt(HD)  # [S]
    # ALiBi
    slope = 0.0
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        slope = float(attn.alibi_slopes[h5])
    dist = (qpos - torch.arange(seq, device=resid.device)).clamp(min=0).float()
    alibi = -slope * dist
    causal = torch.arange(seq, device=resid.device) <= qpos
    total = scores + alibi
    total = total.masked_fill(~causal, float("-inf"))
    # softmax1 (implicit zero logit): include a virtual 0-logit
    print(f"  block={bi} base={base} HD={HD} alibi_slope={slope}", flush=True)

    def D(name):
        return BD[name] if name in BD else None

    def cell(row, name, n=16):
        b = D(name)
        if b is None:
            return None
        v = resid[row, b:b + n]
        m = int(v.argmax()); return m, float(v[m])

    # Rank candidate rows by total score.
    order = torch.argsort(total, descending=True)
    print("  top-12 candidate rows for head-5 (score, alibi-incl):", flush=True)
    hdr = "row tok  total   raw   alibi | MEM_STORE_AT_VAL MEM_VAL_B1 ADDR_B0 ADDR_B1 ADDR_B2 CLEAN_LO"
    print("  " + hdr, flush=True)
    for r in order[:12].tolist():
        tok = ctx[r]
        msav = float(resid[r, D("MEM_STORE_AT_VAL")]) if D("MEM_STORE_AT_VAL") else float("nan")
        mvb1 = float(resid[r, D("MEM_VAL_B1")]) if D("MEM_VAL_B1") else float("nan")
        a0 = cell(r, "ADDR_B0_LO"); a1 = cell(r, "ADDR_B1_LO"); a2 = cell(r, "ADDR_B2_LO")
        clo = cell(r, "CLEAN_EMBED_LO")
        def fmt(c):
            return f"{c[0]:>2}({c[1]:+.0f})" if c else "  -  "
        print(f"   {r:>3} {tok:>3} {float(total[r]):+7.1f} {float(scores[r]):+6.1f} {float(alibi[r]):+6.1f} | "
              f"{msav:+6.2f} {mvb1:+6.2f} {fmt(a0)} {fmt(a1)} {fmt(a2)} {fmt(clo)}", flush=True)

    # Decode operand-A (ALU_LO) at the query AFTER the L8 attn block.
    out = p.model.forward(padded, stop_after_block=bi)[0]
    al = out[qpos, D("ALU_LO"):D("ALU_LO") + 16]
    print(f"  ALU_LO at query after blk{bi}: cell={int(al.argmax())} ({float(al.max()):+.1f})", flush=True)


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 816
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    main(idx, step)
