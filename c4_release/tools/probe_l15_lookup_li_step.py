"""L15 mem_lookup head 0 attention pattern at LI step.

For test_si_li_roundtrip:
  IMM 0x200; PSH; IMM 42; SI; IMM 0x200; LI; EXIT

Result is 512 (0x200 = SP marker), meaning the L15 mem_lookup head 0 at
the LI step's AX position is selecting the MARK_SP row instead of the
MARK_MEM row's val byte 0.

Capture: per-head Q*K score at every K row at the LI step's AX marker
position (Q row), separated into Q*K breakdown by K-row marker (SP/BP/
MEM/STACK0/AX/PC).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

PROG = [
    (Opcode.IMM, 0x200),
    Opcode.PSH,
    (Opcode.IMM, 42),
    Opcode.SI,
    (Opcode.IMM, 0x200),
    Opcode.LI,
    Opcode.EXIT,
]


def make_bc(prog):
    out = []
    for item in prog:
        if isinstance(item, tuple):
            opcode, imm = item
            out.append(opcode | (imm << 8))
        else:
            out.append(item)
    return out


def main():
    import contextlib
    import io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
    _do_probe(runner)


def _do_probe(runner):
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []

    model = runner.model
    dim_positions = model.embed._dim_positions

    # Find L15 block index
    L15_IDX = None
    for i, blk in enumerate(model.blocks):
        # L15 has num_heads=4 or 12, but we'll find by name pattern by hooking each
        pass
    # Just probe at block 15 (or wherever it is)
    # Inspect num_blocks
    print(f"Model has {len(model.blocks)} blocks")
    # Find the L15 memory_lookup block dynamically: it's the one whose
    # attn has W_q[0, BD.OP_LI_RELAY] = 2000 — signature of heads_0_3 setter.
    # Also report all blocks with nonzero W_q for context.
    op_li_dim = dim_positions.get("OP_LI_RELAY")
    L15_IDX = None
    print(f"Scanning blocks for op_li_dim={op_li_dim} W_q signature...")
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        if wq.is_sparse_csr or wq.is_sparse:
            wq = wq.to_dense()
        nz = (wq.abs() > 1e-6).sum().item()
        v = float(wq[0, op_li_dim].item()) if op_li_dim is not None else 0.0
        if nz > 0:
            print(f"  block {bi}: W_q nonzero={nz} W_q[0,OP_LI_RELAY]={v}")
        # Look for any nonzero OP_LI_RELAY signature
        if abs(v) > 1000.0:
            L15_IDX = bi
    if L15_IDX is None:
        L15_IDX = 15 if len(model.blocks) > 15 else 14
    print(f"L15 memory_lookup at block {L15_IDX}")

    # Capture inputs to block 15
    captures = []

    def block_hook_pre(module, inputs):
        if captures:
            captures[-1]["pre_L15"] = inputs[0].detach().clone()
    def after_emb_hook(module, inputs, output):
        captures.append({"token_ids": inputs[0].detach().clone()})
    def after_L14_hook(module, inputs, output):
        if captures:
            captures[-1]["after_L14"] = output.detach().clone()

    handles = []
    handles.append(model.embed.register_forward_hook(after_emb_hook))
    if L15_IDX > 0:
        handles.append(model.blocks[L15_IDX - 1].register_forward_hook(after_L14_hook))
    handles.append(model.blocks[L15_IDX].register_forward_pre_hook(block_hook_pre))

    bc = make_bc(PROG)
    try:
        try:
            result = runner.run(bc, b"", max_steps=30)
        except Exception as e:
            print(f"runner raised: {e}")
            result = None
    finally:
        for h in handles:
            h.remove()

    print(f"\nResult: {result}")
    print(f"Forward captures: {len(captures)}")

    # Find capture with the LI step (the largest sequence with after_L14)
    best = None
    for c in captures:
        if "after_L14" not in c:
            continue
        if best is None or c["after_L14"].shape[1] > best["after_L14"].shape[1]:
            best = c

    if best is None:
        print("No capture with after_L14")
        return

    pre_L15 = best["pre_L15"]  # input to L15 (residual after L14)
    seq_len = pre_L15.shape[1]
    print(f"\npre_L15 seq_len={seq_len}")

    L15_attn = model.blocks[L15_IDX].attn
    n_heads = L15_attn.num_heads
    head_dim = L15_attn.head_dim
    dim = L15_attn.dim
    print(f"L15 attn: num_heads={n_heads} head_dim={head_dim} dim={dim}")

    # W_q is (dim, dim) tensor; attn.W_q[base, dim_idx] = ... means rows
    # indexed by head*head_dim+slot, columns by feature dim.
    # Q = x @ W_q.T  (so Q[t, base] = sum_d x[t, d] * W_q[base, d])
    is_compact = getattr(L15_attn, "_is_compact", False)
    print(f"L15 is_compact={is_compact}")
    W_q = L15_attn.W_q
    W_k = L15_attn.W_k
    W_v = L15_attn.W_v
    if W_q.is_sparse_csr or W_q.is_sparse:
        W_q = W_q.to_dense()
        W_k = W_k.to_dense()
        W_v = W_v.to_dense()
    rev_dim = {v: k for k, v in dim_positions.items()}
    print(f"W_q shape: {tuple(W_q.shape)}")
    # Dump head 0 rows 36 and 61 (dominating dims found by probe)
    print("\nHead 0 W_q row 36 (Q[36] makes K@81 score +1.9G):")
    for c in torch.nonzero(W_q[36].abs() > 1.0).flatten().tolist():
        print(f"  W_q[36, {c} ({rev_dim.get(c, '?')})]={float(W_q[36, c].item())}")
    print("\nHead 0 W_q row 61 (Q[61] makes K@193 score -14.5G):")
    for c in torch.nonzero(W_q[61].abs() > 1.0).flatten().tolist():
        print(f"  W_q[61, {c} ({rev_dim.get(c, '?')})]={float(W_q[61, c].item())}")
    print("\nHead 0 W_k row 36 (all nonzero):")
    for c in torch.nonzero(W_k[36].abs() > 0.0001).flatten().tolist():
        print(f"  W_k[36, {c} ({rev_dim.get(c, '?')})]={float(W_k[36, c].item())}")
    print("\nHead 0 W_k row 61:")
    for c in torch.nonzero(W_k[61].abs() > 1.0).flatten().tolist():
        print(f"  W_k[61, {c} ({rev_dim.get(c, '?')})]={float(W_k[61, c].item())}")
    print("\nHead 0 W_k row 0:")
    for c in torch.nonzero(W_k[0].abs() > 1.0).flatten().tolist():
        print(f"  W_k[0, {c} ({rev_dim.get(c, '?')})]={float(W_k[0, c].item())}")
    print("\nHead 0 W_q row 0:")
    for c in torch.nonzero(W_q[0].abs() > 1.0).flatten().tolist():
        print(f"  W_q[0, {c} ({rev_dim.get(c, '?')})]={float(W_q[0, c].item())}")
    print()
    x = pre_L15[0]  # (seq, dim)
    if is_compact:
        # Gather x cols by _compact_in_idx, compute, then output is in the
        # compact head space (n_out indices)
        x_in = x[:, L15_attn._compact_in_idx]
        n_out = len(L15_attn._compact_out_idx)
        Q = x_in @ W_q.T  # (seq, n_out)
        K = x_in @ W_k.T
        V = x_in @ W_v.T
        Q = Q.view(seq_len, n_heads, n_out // n_heads)
        K = K.view(seq_len, n_heads, n_out // n_heads)
        V = V.view(seq_len, n_heads, n_out // n_heads)
        head_dim = n_out // n_heads
        print(f"compact: n_out={n_out} -> head_dim={head_dim}")
    else:
        Q = (x @ W_q.T).view(seq_len, n_heads, head_dim)
        K = (x @ W_k.T).view(seq_len, n_heads, head_dim)
        V = (x @ W_v.T).view(seq_len, n_heads, head_dim)

    # Find markers in pre_L15
    def find_markers(after, dim_name, thresh=0.5):
        col = after[0, :, dim_positions[dim_name]]
        return torch.nonzero(col > thresh, as_tuple=False).flatten().tolist()

    ax_positions = find_markers(pre_L15, "MARK_AX")
    sp_positions = find_markers(pre_L15, "MARK_SP")
    bp_positions = find_markers(pre_L15, "MARK_BP")
    mem_positions = find_markers(pre_L15, "MARK_MEM")
    stack0_positions = find_markers(pre_L15, "MARK_STACK0")
    pc_positions = find_markers(pre_L15, "MARK_PC")
    print(f"\nMARK positions:")
    print(f"  PC:     {pc_positions}")
    print(f"  AX:     {ax_positions}")
    print(f"  SP:     {sp_positions}")
    print(f"  BP:     {bp_positions}")
    print(f"  MEM:    {mem_positions}")
    print(f"  STACK0: {stack0_positions}")

    # LI step: the last step. AX position in last step.
    # Find LI step by OP_LI_RELAY
    op_li_dim = dim_positions.get("OP_LI_RELAY")
    if op_li_dim is None:
        print("No OP_LI_RELAY dim")
        return
    op_li_col = pre_L15[0, :, op_li_dim]
    li_rows = torch.nonzero(op_li_col > 0.5, as_tuple=False).flatten().tolist()
    print(f"\nOP_LI_RELAY rows: {li_rows[:20]}... total {len(li_rows)}")

    # The Q position is the AX marker in the LI step. Last AX position
    # should be in the LI step.
    if not ax_positions:
        print("No AX positions")
        return
    Q_pos = ax_positions[-1]  # last AX = LI step's AX
    print(f"\nLI step Q position (last MARK_AX): {Q_pos}")

    # Compute Q[Q_pos] · K[k_pos] for all k_pos, for head 0
    for h in range(min(4, n_heads)):
        q_vec = Q[Q_pos, h]  # (head_dim,)
        scores = (K[:, h, :] @ q_vec) / (head_dim ** 0.5)
        # Apply causal mask: only positions <= Q_pos
        scores_masked = scores.clone()
        scores_masked[Q_pos+1:] = float("-inf")

        # Get top-5
        topk = torch.topk(scores_masked, k=5)
        print(f"\n--- L15 head {h} Q@{Q_pos} (LI-step AX marker) top-5 K rows ---")
        for rank in range(5):
            k_pos = int(topk.indices[rank].item())
            score = float(topk.values[rank].item())
            # Marker on this row
            markers = []
            for name in ("MARK_PC","MARK_AX","MARK_SP","MARK_BP","MARK_MEM","MARK_STACK0"):
                if pre_L15[0, k_pos, dim_positions[name]].item() > 0.5:
                    markers.append(name.replace("MARK_",""))
            byte_flags = []
            for name in ("BYTE_INDEX_0","BYTE_INDEX_1","BYTE_INDEX_2","BYTE_INDEX_3",
                         "MEM_VAL_B0","MEM_VAL_B1","MEM_VAL_B2","MEM_VAL_B3"):
                if name in dim_positions and pre_L15[0, k_pos, dim_positions[name]].item() > 0.5:
                    byte_flags.append(name)
            mem_store = float(pre_L15[0, k_pos, dim_positions["MEM_STORE"]].item())
            print(f"  rank {rank}: K@{k_pos} score={score:8.2f} markers={markers} "
                  f"flags={byte_flags} MEM_STORE={mem_store:.2f}")

    # Per-dim score breakdown for head 0: which Q-K dim drives K@81 vs K@193?
    h = 0
    print("\n--- L15 head 0 per-dim Q[d]*K[d] breakdown ---")
    q_vec = Q[Q_pos, h]  # (head_dim,)
    # MEM val byte 0 row should be at MEM_pos + 5: positions 90, 123, 158, 193
    mem_byte0_pos = [m + 5 for m in mem_positions]
    print(f"Expected target rows (MEM val byte 0): {mem_byte0_pos}")
    print(f"Wrong row attended: K@81 = STACK0[80]+1 (stack0 byte 0)")

    # Dump dims at K@193 to see what's at val byte 0 row
    mem_val_b0 = dim_positions.get("MEM_VAL_B0")
    mem_val_b1 = dim_positions.get("MEM_VAL_B1")
    mem_store = dim_positions.get("MEM_STORE")
    addr_key = dim_positions.get("ADDR_KEY")
    # Scan MEM marker frame and dump CLEAN_EMBED at val byte rows
    print(f"\nCLEAN_EMBED at MEM val byte rows of all MEM frames:")
    for m_pos in mem_positions:
        for d in range(0, 10):
            p = m_pos + d
            if p >= seq_len:
                continue
            cl_lo = pre_L15[0, p, dim_positions["CLEAN_EMBED_LO"]:dim_positions["CLEAN_EMBED_LO"]+16]
            cl_hi = pre_L15[0, p, dim_positions["CLEAN_EMBED_HI"]:dim_positions["CLEAN_EMBED_HI"]+16]
            mvb0 = pre_L15[0, p, dim_positions["MEM_VAL_B0"]].item()
            mvb1 = pre_L15[0, p, dim_positions["MEM_VAL_B1"]].item()
            mvb2 = pre_L15[0, p, dim_positions["MEM_VAL_B2"]].item()
            mvb3 = pre_L15[0, p, dim_positions["MEM_VAL_B3"]].item()
            lo_idx = int(torch.argmax(cl_lo).item())
            hi_idx = int(torch.argmax(cl_hi).item())
            print(f"  MEM@{m_pos} d={d} p={p}: VAL_B0/1/2/3=[{mvb0:.2f},{mvb1:.2f},{mvb2:.2f},{mvb3:.2f}] CLEAN_LO={lo_idx}/{cl_lo[lo_idx]:.2f} CLEAN_HI={hi_idx}/{cl_hi[hi_idx]:.2f}")
    print(f"\nMEM marker scan @ {mem_positions[-1]}: positions {mem_positions[-1]}..{mem_positions[-1]+9}")
    for p in range(mem_positions[-1], min(mem_positions[-1] + 10, seq_len)):
        row = []
        for n in ("MEM_VAL_B0","MEM_VAL_B1","MEM_VAL_B2","MEM_VAL_B3"):
            if n in dim_positions:
                row.append(f"{n}={pre_L15[0, p, dim_positions[n]].item():.2f}")
        print(f"  pos {p}: " + " ".join(row))
    print(f"\npre_L15 at K@193 (expected MEM val byte 0):")
    for n in ("MARK_MEM","MARK_STACK0","MEM_VAL_B0","MEM_VAL_B1","MEM_VAL_B2","MEM_VAL_B3","MEM_STORE","ADDR_B0_LO","CLEAN_EMBED_LO"):
        if n in dim_positions:
            d = dim_positions[n]
            v = float(pre_L15[0, 193, d].item())
            print(f"  {n} ({d}) = {v:.3f}")
    print(f"\npre_L15 at K@81 (winning STACK0 frame 1 byte 0):")
    for n in ("MARK_STACK0","STACK0_BYTE0","STACK0_BYTE1","BYTE_INDEX_0","HAS_SE","ADDR_B0_LO","CLEAN_EMBED_LO","MEM_STORE","MEM_VAL_B1","H1","H0","H2","L1H1","L1H4","L2H0"):
        if n in dim_positions:
            d = dim_positions[n]
            v = float(pre_L15[0, 81, d].item())
            print(f"  {n} ({d}) = {v:.3f}")
    # Dump nonzero dims at K@81 broadly
    print(f"\nK@81 nonzero dims (|val|>0.5):")
    nz = torch.nonzero(pre_L15[0, 81].abs() > 0.5).flatten().tolist()
    for d in nz:
        v = float(pre_L15[0, 81, d].item())
        print(f"  dim {d} ({rev_dim.get(d, '?')}) = {v:.3f}")
    print(f"\nK@192 nonzero dims (|val|>0.5):")
    nz = torch.nonzero(pre_L15[0, 192].abs() > 0.5).flatten().tolist()
    for d in nz:
        v = float(pre_L15[0, 192, d].item())
        print(f"  dim {d} ({rev_dim.get(d, '?')}) = {v:.3f}")
    # Print addr_b0_lo block
    print(f"\nADDR_B0_LO at K@81: {pre_L15[0, 81, addr_key_pos:addr_key_pos+16].tolist() if (addr_key_pos := dim_positions.get('ADDR_B0_LO')) else 'NA'}")
    print(f"ADDR_B0_LO at K@193: {pre_L15[0, 193, addr_key_pos:addr_key_pos+16].tolist() if (addr_key_pos := dim_positions.get('ADDR_B0_LO')) else 'NA'}")

    cmp_rows = [80, 81, 85, 89, 192, 193]
    for k_pos in cmp_rows:
        if k_pos >= seq_len:
            continue
        k_vec = K[k_pos, h]  # (head_dim,)
        prod = q_vec * k_vec
        score = float(prod.sum().item()) / (head_dim ** 0.5)
        markers = []
        for name in ("MARK_PC","MARK_AX","MARK_SP","MARK_BP","MARK_MEM","MARK_STACK0"):
            if pre_L15[0, k_pos, dim_positions[name]].item() > 0.5:
                markers.append(name.replace("MARK_",""))
        # Top 8 contributing dims
        topdims = torch.topk(prod.abs(), k=8)
        print(f"  K@{k_pos} ({'+'.join(markers) or 'none'}) total_score={score:.1f}")
        for j in range(8):
            d = int(topdims.indices[j].item())
            v = float(prod[d].item())
            print(f"    head_dim {d}: Q={float(q_vec[d].item()):.1f} K={float(k_vec[d].item()):.1f} prod={v:.1f}")

    print("\n--- L15 head 0 attention probabilities top-5 ---")
    scores = (K[:, h, :] @ q_vec) / (head_dim ** 0.5)
    scores_masked = scores.clone()
    scores_masked[Q_pos+1:] = float("-inf")
    # softmax1 (with +1 baseline) is what some models use; standard softmax
    # is fine for a top-row signal probe.
    probs = torch.softmax(scores_masked, dim=0)
    topk = torch.topk(probs, k=5)
    for rank in range(5):
        k_pos = int(topk.indices[rank].item())
        p = float(topk.values[rank].item())
        markers = []
        for name in ("MARK_PC","MARK_AX","MARK_SP","MARK_BP","MARK_MEM","MARK_STACK0"):
            if pre_L15[0, k_pos, dim_positions[name]].item() > 0.5:
                markers.append(name.replace("MARK_",""))
        cl_lo_band = pre_L15[0, k_pos, dim_positions["CLEAN_EMBED_LO"] : dim_positions["CLEAN_EMBED_LO"]+16]
        cl_lo_idx = int(torch.argmax(cl_lo_band).item())
        cl_lo_v = float(cl_lo_band[cl_lo_idx].item())
        print(f"  rank {rank}: K@{k_pos} prob={p:.4f} markers={markers} "
              f"CLEAN_LO={cl_lo_idx}/{cl_lo_v:.2f}")


if __name__ == "__main__":
    main()
