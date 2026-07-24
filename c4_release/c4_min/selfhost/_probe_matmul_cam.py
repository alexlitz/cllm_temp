"""Probe: reproduce the 2x2 matmul self-emulation divergence at ~296 steps and
instrument the §Memory address-CAM at the FAILING load.

Confirms the aliasing mechanism (task 1): dumps, for the load step whose AX/output
first diverges from the reference interpreter, the memory-head softmax1 attention
weights over the candidate store frames — identifying which store the CAM favours,
whether it is (a) address-key resolution, (b) ALiBi recency, or (c) fp32 softmax1
precision at large max_seq.

Run:
    OMP_NUM_THREADS=4 python -m c4_min.selfhost._probe_matmul_cam [--device=cuda:0]
"""
from __future__ import annotations
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch

from c4_min import isa
from c4_min.selfhost._matmul_src import matmul_c, matmul_reference, SCALE


def _compile_matmul(A, B):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    src = matmul_c(A, B, SCALE)
    bc, data = compile_c(src)
    code = bytecode_to_isa(bc)
    return code, data


def run_ref(code):
    from c4_min.nibble_pure_forward_complete import ref_interpret
    out = []
    tr = ref_interpret(code, max_steps=200000, mask=0xFFFFFFFF, out=out)
    return tr, out


def run_ref_regs(code, mask=0xFFFFFFFF, max_steps=200000):
    """Reference interpreter that records (pc_before, op, imm, ax, sp, bp) per step
    — a transparent copy of ref_interpret's transitions with 32-bit AX/registers so
    we can compare the neural registers step-for-step."""
    mem = {}
    sp = bp = 0
    from c4_min.nibble_pure_forward_cached import SP_INIT
    sp = bp = SP_INIT
    ax = pc = 0
    regs = []
    steps = 0
    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        i = pc
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.LEA:
            ax = (bp + 4 * imm) & 0xFF
        elif op == isa.PSH:
            sp -= 4; mem[sp] = ax & mask
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
            v = mem.get(sp, 0) & mask; sp += 4
            if op == isa.ADD: ax = (v + ax) & mask
            elif op == isa.SUB: ax = (v - ax) & mask
            elif op == isa.MUL: ax = (v * ax) & mask
            elif op == isa.DIV: ax = ((v // ax) if ax else 0) & mask
            else: ax = ((v % ax) if ax else 0) & mask
        elif op in (isa.LI, isa.LC):
            ax = mem.get(ax, 0) & 0xFF
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp, 0); sp += 4; mem[addr] = ax & 0xFF
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.JSR:
            sp -= 4; mem[sp] = (i + 1) & 0xFF; pc = imm
        elif op == isa.ENT:
            mem[sp - 4] = bp & 0xFFFFFFFF; sp -= 4; bp = sp; sp -= 4 * imm
        elif op == isa.ADJ:
            sp += 4 * imm
        elif op == isa.LEV:
            sp = bp; bp = mem.get(sp, 0); pc = mem.get(sp + 4, 0); sp += 8
        elif op == isa.PRTF:
            pass
        elif op in (isa.NOP,):
            pass
        elif op == isa.HALT:
            regs.append((i, op, imm, ax & mask, sp, bp)); break
        regs.append((i, op, imm, ax & mask, sp, bp))
    return regs


def build(device):
    from c4_min.lib_neural import build_lib_model_streaming
    sparse, L, _ = build_lib_model_streaming(
        code_size=256, recurrent_divmod=True, addr32=True)
    if device != "cpu":
        sparse = sparse.to(device)
    return sparse, L


def instrumented_run(model, L, code, ref_trace, device, max_steps,
                     data_seg=None, evict=True, prune_interval=60,
                     dump_all_loads=False, dump_from=None, dump_ops=None,
                     ref_regs=None, trace_regs_from=None):
    """A copy of run_pure_forward_cached's loop that, at EACH step, also computes
    the memory/stack/lev head attention over the FULL (uncached) window so we can
    read the softmax1 weights the CAM produced — and flags the first step where the
    neural AX diverges from the reference trace."""
    from c4_min.nibble_pure_forward_cached import (
        BlockKVCacheBatched, apply_overlay_window)
    from c4_min.nibble_pure_forward import (
        _MEM_MARKER_LOCAL, SP_INIT, _snap_lane)
    from c4_min.nibble_pure_forward_complete import (
        _build_frame, _decode_reg_from_nibbles, _mem_top)
    from c4_min import blogspec_vocab as V
    blocks = model.blocks
    n_blocks = len(blocks)
    H = blocks[0].attn.n_heads
    HD = blocks[0].attn.head_dim
    caches = [BlockKVCacheBatched(H, HD, blocks[b].attn.alibi_slopes)
              for b in range(n_blocks)]
    tokens_since_prune = 0

    store_log = {}
    init_frame = _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    stream = [V.BOS] + init_frame
    trace = []
    cur_pc = 0
    cur_sp = cur_bp = SP_INIT
    cur_ax = 0
    frame_idx = 0
    win_start = 0
    win_len = len(stream)

    first_div = None

    for step in range(max_steps):
        win_toks = torch.tensor([stream[win_start:win_start + win_len]], device=device)
        q_positions = torch.arange(win_start, win_start + win_len, device=device)
        with torch.no_grad():
            x = model.embed[win_toks].clone()
            apply_overlay_window(x, win_start, code, L, store_log,
                                 is_last_row_query=True)
            past = [caches[b].as_past_kv() for b in range(n_blocks)]
            hidden, new_kv = model.forward_hidden_cached(
                x, past_key_values=past, q_positions=q_positions, use_cache=True)
        state = hidden[0, -1]

        pc = _snap_lane(state[L.PC_VAL])
        sp = _snap_lane(state[L.SP_VAL])
        bp = _snap_lane(state[L.BP_VAL])
        stk = _snap_lane(state[L.STK_VAL])
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        ax = _decode_reg_from_nibbles(state, L, L.AX)

        ref_ax = ref_trace[step] if step < len(ref_trace) else None
        # reference register state for THIS step (pc_before, op, imm, ax, sp, bp).
        rr = ref_regs[step] if (ref_regs is not None and step < len(ref_regs)) else None
        reg_div = False
        if rr is not None:
            r_pc, r_op, r_imm, r_ax, r_sp, r_bp = rr
            reg_div = (sp != r_sp) or (bp != r_bp) or (cur_pc != r_pc)
        is_div = (ref_ax is not None and (ax & 0xFFFFFFFF) != (ref_ax & 0xFFFFFFFF))
        if rr is not None and (reg_div or (trace_regs_from is not None
                                          and step >= trace_regs_from)):
            flag = " <== REG DIVERGE" if reg_div else ""
            print(f"[reg] step {step} op={isa.NAMES.get(op, op):4s} pc={cur_pc}(ref {r_pc}) "
                  f"ax={ax & 0xFFFFFFFF}(ref {r_ax}) sp={sp}(ref {r_sp}) "
                  f"bp={bp}(ref {r_bp}) stk={stk} S={len(stream)}{flag}", flush=True)

        want_dump = False
        if op in (isa.LI, isa.LC, isa.LEV, isa.PSH, isa.SI, isa.ENT, isa.JSR):
            if dump_all_loads or is_div or reg_div:
                want_dump = True
            if dump_from is not None and step >= dump_from:
                if dump_ops is None or op in dump_ops:
                    want_dump = True
        if want_dump:
            tag = "REGDIV" if reg_div else ("DIVERGE" if is_div else "op")
            print(f"\n--- step {step} {tag} op={isa.NAMES.get(op, op)} pc={cur_pc} "
                  f"neural_ax={ax & 0xFFFFFFFF} ref_ax={ref_ax} "
                  f"sp={sp} bp={bp} stk={stk} S={len(stream)} cache={caches[0].size()} ---",
                  flush=True)
            _dump_cam_weights(model, L, code, store_log, stream, op, device,
                              force_all=(op not in (isa.LI, isa.LC)))

        if is_div and first_div is None:
            first_div = step
            print(f"\n=== FIRST DIVERGENCE at step {step} ===")
            print(f"  op={isa.NAMES.get(op, op)} pc={cur_pc} "
                  f"neural_ax={ax & 0xFFFFFFFF} ref_ax={ref_ax & 0xFFFFFFFF} "
                  f"sp={sp} bp={bp} stk={stk} S={len(stream)} cache={caches[0].size()}")
            if op not in (isa.LI, isa.LC):
                _dump_cam_weights(model, L, code, store_log, stream, op, device)

        # replicate driver bookkeeping (stores)
        s_addr = s_val = 0
        is_store = False
        if op in (isa.SI, isa.SC):
            is_store = True; s_addr = _mem_top(store_log, cur_sp); s_val = ax & 0xFFFFFFFF
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & 0xFFFFFFFF
        elif op == isa.JSR:
            is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
        elif op == isa.ENT:
            is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
        frame = _build_frame(pc, ax, sp, bp, stk,
                             mem_addr=(s_addr if is_store else 0),
                             mem_val=(s_val if is_store else 0))
        trace.append(ax & 0xFFFFFFFF)
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val & 0xFFFFFFFF)

        n_commit = win_len - 1
        if n_commit > 0:
            for b in range(n_blocks):
                K_all, V_all, pos_all = new_kv[b]
                K_win = K_all[:, :, -win_len:, :]
                V_win = V_all[:, :, -win_len:, :]
                pos_win = pos_all[-win_len:]
                caches[b].commit(K_win[:, :, :n_commit, :],
                                 V_win[:, :, :n_commit, :], pos_win[:n_commit])
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        if pc < 0 or pc >= len(code):
            break
        prev_query_pos = win_start + win_len - 1
        stream += frame
        win_start = prev_query_pos
        win_len = 1 + V.FRAME_LEN
        tokens_since_prune += V.FRAME_LEN
        if evict and tokens_since_prune >= prune_interval:
            for b in range(n_blocks):
                caches[b].evict(0.99, prune_interval, 1e-9, 1e-6)
            tokens_since_prune = 0

    return trace, first_div


def _dense(W):
    """Return a dense [out,in] tensor for a SparseWeight or a plain tensor."""
    if hasattr(W, "linear"):   # SparseWeight
        if getattr(W, "dense", None) is not None:
            return W.dense
        if getattr(W, "dense_resident", None) is not None:
            return W.dense_resident
        return W.csr.to_dense()
    return W


def _lin(W, x):
    if hasattr(W, "linear"):
        return W.linear(x)
    import torch.nn.functional as F
    return F.linear(x, W)


def _dump_cam_weights(model, L, code, store_log, stream, op, device,
                      force_all=False):
    """Recompute the FULL forward over the entire stream (no cache, no eviction)
    and print the memory/stack/lev head softmax1 attention weights over the
    candidate store frames at the query (last) row."""
    import torch.nn.functional as F
    from c4_min.nibble_pure_forward_complete import (
        make_overlay_complete, N_ROLES, _MEM_MARKER_LOCAL)
    from c4_min import blogspec_vocab as V
    from c4_min.blogspec_model import softmax1
    from c4_min.blogspec_memory import ADDR_BITS

    full = list(stream)
    toks = torch.tensor([full], device=device)
    with torch.no_grad():
        x = model.embed[toks].clone()
        overlay = make_overlay_complete(code, L, store_log)
        overlay(x)
    S = x.shape[1]

    HD = model.blocks[0].attn.head_dim
    heads = {
        "LI(mem)": (N_ROLES, L.AX),
        "POP(stack)": (N_ROLES + 1, L.STACK0),
        "LEV": (N_ROLES + 2, L.LEV_RET),
    }
    store_positions = {}
    for f, (addr, val) in store_log.items():
        p = 1 + f * V.FRAME_LEN + _MEM_MARKER_LOCAL
        if p < S:
            store_positions[p] = (f, addr, val)

    # Only dump the LI head for a load; others for context.
    if force_all:
        which = list(heads.keys())
    else:
        which = ["LI(mem)"] if op in (isa.LI, isa.LC) else list(heads.keys())
    for name in which:
        head, dest = heads[name]
        blk = None
        for bi, b in enumerate(model.blocks):
            attn = b.attn
            base = head * HD
            Wo = _dense(attn.W_o)
            wo = Wo[dest:dest + 1, base + ADDR_BITS + 3:base + ADDR_BITS + 3 + 16]
            if float(wo.abs().sum()) > 0:
                blk = bi
                break
        if blk is None:
            continue
        attn = model.blocks[blk].attn
        with torch.no_grad():
            h = x
            for bi in range(blk):
                h = model.blocks[bi](h)
            B, Sn, D = h.shape
            Hn, HDn = attn.n_heads, attn.head_dim
            # Only the LAST-row query: q [HDn], K [Sn, HDn] -> scores [Sn].
            Q = _lin(attn.W_q, h[:, -1:, :]).view(B, 1, Hn, HDn)[0, 0, head]  # [HDn]
            K = _lin(attn.W_k, h).view(B, Sn, Hn, HDn)[0, :, head, :]         # [Sn,HDn]
            base_scores = (K @ Q) * attn.scale                                # [Sn]
            pos = torch.arange(Sn, device=h.device).float()
            dist = (Sn - 1) - pos                                             # query at last row
            alibi = float(attn.alibi_slopes[head]) * dist
            addr_only = base_scores - alibi
            raw = addr_only.clone()   # causal: last row attends to all <= last (all)
            w = softmax1(raw.view(1, 1, 1, Sn), dim=-1)[0, 0, 0]
        cand = [(p, store_positions[p][0], store_positions[p][1],
                 store_positions[p][2], float(w[p]), float(raw[p]),
                 float(addr_only[p])) for p in store_positions]
        cand.sort(key=lambda t: -t[4])
        sink = 1.0 - float(w.sum())
        print(f"  [{name}] block={blk} head={head} alibi_slope="
              f"{float(attn.alibi_slopes[head]):.4g} sink_weight~{sink:.6e} "
              f"query_row_pos={Sn - 1}", flush=True)
        for (p, f, addr, val, ww, rw, ao) in cand[:10]:
            dist_qp = (Sn - 1) - p
            print(f"      store f={f:>4} pos={p:>5} addr=0x{addr:X} val={val:>10} "
                  f"weight={ww:.6e} raw={rw:.4e} (score={ao:.4e} alibi=-{dist_qp:.0f})",
                  flush=True)


def main():
    device = "cpu"
    dump_all = "--all-loads" in sys.argv
    dump_from = None
    for a in sys.argv:
        if a.startswith("--device="):
            device = a.split("=", 1)[1]
        if a.startswith("--dump-from="):
            dump_from = int(a.split("=", 1)[1])
    A = [[1, 1], [2, 1]]
    B = [[1, 2], [3, 1]]
    code, data = _compile_matmul(A, B)
    ref_trace, ref_out = run_ref(code)
    ref_regs = run_ref_regs(code)
    print(f"ref steps={len(ref_trace)} out={ref_out} numpy={matmul_reference(A, B, SCALE)}",
          flush=True)
    t0 = time.time()
    model, L = build(device)
    print(f"build {time.time()-t0:.1f}s device={device}", flush=True)
    trace, first_div = instrumented_run(
        model, L, code, ref_trace, device, max_steps=len(ref_trace) + 6,
        data_seg=data, evict=True, prune_interval=60, dump_all_loads=dump_all,
        dump_from=dump_from, ref_regs=ref_regs,
        trace_regs_from=(dump_from if dump_from is not None else None),
        dump_ops={isa.LI, isa.LC, isa.LEV, isa.PSH})
    print(f"\nfirst_div={first_div}  neural_steps={len(trace)}")
    print(f"ref_out={ref_out}")


if __name__ == "__main__":
    main()
