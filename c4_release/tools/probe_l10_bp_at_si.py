"""Probe L10 byte_passthrough head 1 attention at SI step's AX byte 0 row.

For program: IMM 0x200; PSH; IMM 0x1234; SI; IMM 0x200; LI; EXIT

Captures L10 head 1 attention weights with Q at SI-step AX byte 0 row,
and reports the top-5 K rows + scores + V contributions to OUTPUT.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

PROG = [
    (Opcode.IMM, 0x200), Opcode.PSH,
    (Opcode.IMM, 0x1234), Opcode.SI,
    (Opcode.IMM, 0x200), Opcode.LI,
    Opcode.EXIT,
]


def make_bc(prog):
    out = []
    for it in prog:
        if isinstance(it, tuple):
            op, im = it; out.append(op | (im << 8))
        else:
            out.append(it)
    return out


def main():
    import contextlib, io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True,
                                        use_kv_cache=False)
    runner._func_call_handlers = {}; runner._syscall_handlers = {}
    runner._memory = {}; runner._mem_history = {}; runner._mem_access_order = []

    model = runner.model
    dim_positions = model.embed._dim_positions

    # L10 block index — find dynamically by signature
    is_byte_dim = dim_positions["IS_BYTE"]
    L10_IDX = None
    for bi, blk in enumerate(model.blocks):
        attn = blk.attn
        W_q = attn.W_q.to_dense() if attn.W_q.is_sparse_csr or attn.W_q.is_sparse else attn.W_q
        HD = W_q.shape[0] // attn.num_heads
        if HD < 34:
            continue
        base = 1 * HD
        if base < W_q.shape[0]:
            v = float(W_q[base, is_byte_dim].item())
            if abs(v) > 50:
                L10_IDX = bi
                break
    if L10_IDX is None:
        L10_IDX = 10
    print(f"Using L10_IDX = {L10_IDX}, blocks={len(model.blocks)}")

    all_caps = []
    current = {}

    def pre_hook(idx):
        def h(m, inp):
            nonlocal current
            if idx == 0 and current:
                all_caps.append(current); current = {}
            current[("pre", idx)] = inp[0].detach().clone()
        return h
    def post_hook(idx):
        def h(m, inp, out):
            current[("post", idx)] = out.detach().clone()
        return h

    handles = []
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_pre_hook(pre_hook(i)))
        handles.append(blk.register_forward_hook(post_hook(i)))

    bc = make_bc(PROG)
    try:
        try:
            result = runner.run(bc, b"", max_steps=30)
        except Exception as e:
            print(f"runner raised: {e}"); result = None
    finally:
        for h in handles: h.remove()
        if current: all_caps.append(current)

    print(f"Result={result} caps={len(all_caps)}")

    # Pick longest
    best = max(all_caps, key=lambda c: c.get(("pre", 0), torch.zeros(1, 1, 1)).shape[1])
    pre_l10 = best[("pre", L10_IDX)]
    seq_len = pre_l10.shape[1]
    print(f"seq_len = {seq_len}")

    def find(marker, thresh=0.5):
        col = pre_l10[0, :, dim_positions[marker]]
        return torch.nonzero(col > thresh, as_tuple=False).flatten().tolist()

    ax_rows = find("MARK_AX")
    mem_rows = find("MARK_MEM")
    print(f"AX={ax_rows} MEM={mem_rows}")
    si_ax = ax_rows[-3]  # IMM 0x200, PSH, IMM 0x1234, SI, IMM 0x200_b, LI
    li_ax = ax_rows[-1]
    si_byte0 = si_ax + 1
    li_byte0 = li_ax + 1
    print(f"SI ax_marker={si_ax}, SI byte 0 row = {si_byte0}")
    print(f"LI ax_marker={li_ax}, LI byte 0 row = {li_byte0}")
    # Override to look at LI byte 0
    import os
    if os.environ.get("PROBE_LI") == "1":
        si_byte0 = li_byte0

    # L10 head 1 attention
    blk = model.blocks[L10_IDX]
    attn = blk.attn
    HD = attn.W_q.shape[0] // attn.num_heads
    print(f"L10: num_heads={attn.num_heads}, HD={HD}")
    h = 1
    W_q = attn.W_q
    W_k = attn.W_k
    W_v = attn.W_v
    if W_q.is_sparse_csr or W_q.is_sparse:
        W_q = W_q.to_dense(); W_k = W_k.to_dense(); W_v = W_v.to_dense()

    x = pre_l10[0]
    if getattr(attn, "_is_compact", False):
        x_in = x[:, attn._compact_in_idx]
        Q = (x_in @ W_q.T).view(seq_len, attn.num_heads, -1)
        K = (x_in @ W_k.T).view(seq_len, attn.num_heads, -1)
        V = (x_in @ W_v.T).view(seq_len, attn.num_heads, -1)
        head_dim = Q.shape[-1]
    else:
        Q = (x @ W_q.T).view(seq_len, attn.num_heads, HD)
        K = (x @ W_k.T).view(seq_len, attn.num_heads, HD)
        V = (x @ W_v.T).view(seq_len, attn.num_heads, HD)
        head_dim = HD

    q_vec = Q[si_byte0, h]
    # Dump Q at key slots and the K at the IMM_0x1234 byte 1 row
    print(f"\nQ@{si_byte0} head {h}: ")
    for s in [0, 1, 2, 3, 4, 5, 33]:
        print(f"  Q[{s}] = {float(q_vec[s].item()):.2f}")
    # Dump dim activations at Q row that contribute to Q[0]
    print(f"\n  pre_l10[Q={si_byte0}] dims active near Q[0] sources:")
    for nm in ("IS_BYTE", "HAS_SE", "OP_IMM", "OP_LI_RELAY", "OP_LC_RELAY",
               "OP_SI", "OP_PSH", "CONST", "TEMP+3", "CMP+3", "H1+1"):
        if "+" in nm:
            base, off = nm.split("+")
            v = float(pre_l10[0, si_byte0, dim_positions[base] + int(off)].item())
        else:
            v = float(pre_l10[0, si_byte0, dim_positions[nm]].item())
        print(f"    {nm} = {v:.3f}")
    imm_byte1 = 135
    print(f"K@{imm_byte1} head {h}: ")
    for s in [0, 1, 2, 3, 4, 5, 33]:
        print(f"  K[{s}] = {float(K[imm_byte1, h, s].item()):.2f}")
    # Per-slot Q·K contribution
    print(f"Per-slot Q*K for K@{imm_byte1}:")
    for s in [0, 1, 2, 3, 4, 5, 33]:
        print(f"  s={s}: Q={float(q_vec[s].item()):.2f} K={float(K[imm_byte1,h,s].item()):.2f} prod={float(q_vec[s].item()) * float(K[imm_byte1, h, s].item()):.2f}")
    full_score_135 = float((q_vec * K[imm_byte1, h]).sum().item()) / (head_dim ** 0.5)
    print(f"  full Q*K score @ {imm_byte1} = {full_score_135:.2f}")

    # K@67 (IMM 0x200 byte 1)
    K67 = 67
    print(f"K@{K67} head {h}: ")
    for s in [0, 1, 2, 3, 4, 5, 33]:
        print(f"  K[{s}] = {float(K[K67, h, s].item()):.2f}")
    print(f"Per-slot Q*K for K@{K67}:")
    for s in [0, 1, 2, 3, 4, 5, 33]:
        print(f"  s={s}: Q={float(q_vec[s].item()):.2f} K={float(K[K67,h,s].item()):.2f} prod={float(q_vec[s].item()) * float(K[K67, h, s].item()):.2f}")
    full_score_67 = float((q_vec * K[K67, h]).sum().item()) / (head_dim ** 0.5)
    print(f"  full Q*K score @ {K67} = {full_score_67:.2f}")
    # Diff
    diff = full_score_67 - full_score_135
    print(f"  K@{K67} - K@{imm_byte1} = {diff:.2f}")
    # Per-slot diff for full head_dim
    print(f"\nFull per-slot Q*K diff (K@{K67} - K@{imm_byte1}), top 10 by abs:")
    diffs = []
    for s in range(head_dim):
        q = float(q_vec[s].item())
        k67 = float(K[K67, h, s].item())
        k135 = float(K[imm_byte1, h, s].item())
        d = q * (k67 - k135)
        if abs(d) > 1:
            diffs.append((s, q, k67, k135, d))
    diffs.sort(key=lambda x: -abs(x[4]))
    total = sum(d[4] for d in diffs)
    print(f"  TOTAL diff sum (all slots) = {total:.2f}")
    for s, q, k67, k135, d in diffs[:20]:
        print(f"  slot {s}: Q={q:.2f} K67={k67:.2f} K135={k135:.2f} prod_diff={d:.2f}")

    scores = (K[:, h, :] @ q_vec) / (head_dim ** 0.5)
    scores_m = scores.clone()
    scores_m[si_byte0+1:] = float("-inf")

    # ALiBi
    slope = float(attn.alibi_slopes[h].item()) if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None else 0.0
    print(f"head {h} alibi slope: {slope}")
    positions = torch.arange(seq_len, device=scores.device, dtype=scores.dtype)
    dist = (positions - si_byte0).abs().to(scores.dtype)
    scores_m = scores_m - slope * dist
    scores_m[si_byte0+1:] = float("-inf")

    topk = torch.topk(scores_m, k=10)
    print(f"\nL10 head 1 Q@{si_byte0} (SI byte 0) top-10 K rows:")
    for r in range(10):
        kp = int(topk.indices[r].item()); sc = float(topk.values[r].item())
        flags = []
        for nm in ("MARK_AX","MARK_PC","MARK_SP","MARK_BP","MARK_MEM","MARK_STACK0",
                   "BYTE_INDEX_0","BYTE_INDEX_1","BYTE_INDEX_2","BYTE_INDEX_3",
                   "H1+1","IS_BYTE","OP_IMM","OP_SI","HAS_SE"):
            if "+" in nm:
                base, off = nm.split("+")
                v = float(pre_l10[0, kp, dim_positions[base] + int(off)].item())
            else:
                v = float(pre_l10[0, kp, dim_positions[nm]].item())
            if v > 0.4:
                flags.append(f"{nm}={v:.1f}")
        cl_lo_band = pre_l10[0, kp, dim_positions["CLEAN_EMBED_LO"]:dim_positions["CLEAN_EMBED_LO"]+16]
        cl_hi_band = pre_l10[0, kp, dim_positions["CLEAN_EMBED_HI"]:dim_positions["CLEAN_EMBED_HI"]+16]
        cl_lo = int(torch.argmax(cl_lo_band).item())
        cl_hi = int(torch.argmax(cl_hi_band).item())
        print(f"  rank {r}: K@{kp} score={sc:.1f} CLEAN=0x{(cl_hi << 4) | cl_lo:02x} flags=[{', '.join(flags[:8])}]")

    # softmax probs and V contribution
    probs = torch.softmax(scores_m, dim=0)
    print(f"\nTop-5 probs:")
    topp = torch.topk(probs, k=5)
    for r in range(5):
        kp = int(topp.indices[r].item()); p = float(topp.values[r].item())
        print(f"  rank {r}: K@{kp} prob={p:.4f}")

    # V at top K rows (CLEAN_EMBED values)
    print(f"\nVattn (= sum_k prob[k] * V[k]) — projecting through W_o for head 1, slots 0..15, 16..31")
    v_combined = (probs.unsqueeze(-1) * V[:, h, :]).sum(dim=0)  # (head_dim,)
    # The byte passthrough head's V slots 0..15 = CLEAN_EMBED_LO[k], 16..31 = CLEAN_EMBED_HI[k]
    print(f"  V_attn[0..15] (lo nibbles): {[f'{float(v_combined[k].item()):.2f}' for k in range(16)]}")
    print(f"  V_attn[16..31] (hi nibbles): {[f'{float(v_combined[16+k].item()):.2f}' for k in range(16)]}")

    # Show OUTPUT_LO/HI contribution: o-side writes OUTPUT_LO[idx] += V[idx] * 2.0
    print(f"\nO-side contribution to OUTPUT_LO/HI (assuming weight 2.0):")
    for k in range(16):
        v_lo = float(v_combined[k].item())
        if abs(v_lo) > 0.05:
            print(f"  OUTPUT_LO+{k} += {v_lo * 2.0:.2f}")
    for k in range(16):
        v_hi = float(v_combined[16+k].item())
        if abs(v_hi) > 0.05:
            print(f"  OUTPUT_HI+{k} += {v_hi * 2.0:.2f}")


if __name__ == "__main__":
    main()
