"""Hook the §Memory CAM head and dump its softmax weights at each LI query, to see
WHICH key the read-back query matches.  Runs the minimal frame-local read-back
sequence (store 228=131072 ; SC 131072=72 ; load 228) and prints, per LI, the top
attended store rows (their address-key match score + ALiBi dist + softmax weight +
relayed value nibbles).  This pins the root: (a) evicted, (b) recency/aliasing
wins the heap store, (c) corrupted query, (d) frame-local dropped.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torch
from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
from c4_min import libprog_corpus as LC
from c4_min.nibble_pure_forward import N_ROLES

P0 = 0x20000
FILL = 72
MEM_HEAD = N_ROLES     # the §Memory CAM head index


def build():
    # store *228 = 0x20000 ; SC 72 -> heap[0x20000] ; load *228 (twice)
    prog = [
        ("IMM", 228), ("PSH", 0), ("IMM", P0), ("SI", 0),      # *228 = 0x20000
        ("IMM", 228), ("LI", 0),                                # read #1 -> want 0x20000
        ("PSH", 0), ("IMM", FILL), ("SC", 0),                   # *0x20000 = 72
        ("IMM", 228), ("LI", 0),                                # read #2 -> want 0x20000
        ("HALT", 0),
    ]
    return isa.assemble(prog)


def build_frame():
    # SAME semantics but the frame-local address comes from LEA inside an ENT frame
    # (the malloc_printf/memset structure).  BP-4 local holds the pointer.
    prog = [
        ("JSR", None), ("HALT", 0),
    ]
    run_pc = len(prog)
    prog += [("ENT", 1)]
    prog += [("LEA", -1), ("PSH", 0), ("IMM", P0), ("SI", 0)]   # *(BP-4) = 0x20000
    prog += [("LEA", -1), ("LI", 0)]                             # read #1
    prog += [("PSH", 0), ("IMM", FILL), ("SC", 0)]              # *0x20000 = 72
    prog += [("LEA", -1), ("LI", 0)]                             # read #2
    if os.environ.get("C4_WRITEBACK", "0") == "1":
        # exact memset write-back tail (present in the FAILING repro).
        prog += [("LEA", -1), ("PSH", 0),
                 ("LEA", -1), ("LI", 0), ("IMM", 1), ("ADD", 0), ("SI", 0)]
    prog += [("LEV", 0)]
    prog[0] = ("JSR", run_pc)
    return isa.assemble(prog)


def build_args():
    # Reproduce the memset ARGUMENT read that actually fails: main pushes 3 args
    # (the middle/first holds the pointer value V), JSRs a callee, which ENTs and
    # reads an arg via LEA k ; LI.  Matches malloc_printf pc 110-111.
    V = int(os.environ.get("C4_ARGVAL", str(P0)), 0)
    prog = [("IMM", V), ("PSH", 0),       # arg s = V   (the pointer)
            ("IMM", 72), ("PSH", 0),      # arg c
            ("IMM", 3), ("PSH", 0),       # arg n
            ("JSR", None), ("ADJ", 3), ("HALT", 0)]
    callee_pc = int(os.environ.get("C4_CALLEE_PC", str(len(prog))))
    while len(prog) < callee_pc:
        prog += [("NOP", 0)]
    run_pc = len(prog)
    prog += [("ENT", 2)]                  # 2 locals (like memset)
    # read arg s (LEA 4 in the real memset accesses the arg holding V):
    prog += [("LEA", 4), ("LI", 0)]       # AX = *(&arg) -> want V
    prog += [("LEV", 0)]
    prog[6] = ("JSR", run_pc)
    return isa.assemble(prog)


def main():
    os.system("free -g | head -2")
    mode = os.environ.get("C4_MODE", "min")
    code = {"min": build, "frame": build_frame, "args": build_args}[mode]()
    model, L, _ = build_lib_model_streaming(
        code_size=max(int(os.environ.get("C4_CODESIZE", "64")), len(code) + 2),
        recurrent_divmod=True, addr32=True)

    # locate the memory-cam block
    mem_blk_idx = None
    for i, blk in enumerate(model.blocks):
        nm = getattr(L, "_block_names", [None] * len(model.blocks))
        # fall back: the block whose attn has the memory head baked (alibi slope small
        # AND W_o writes AX from a value slot).  Detect by nonzero W_q on QRY_BIN.
    names = getattr(L, "_block_names", None)
    if names:
        for i, nm in enumerate(names):
            if nm == "mem-cam":
                mem_blk_idx = i
    print("mem-cam block idx:", mem_blk_idx, "num blocks:", len(model.blocks),
          "MEM_HEAD:", MEM_HEAD, flush=True)

    target = model.blocks[mem_blk_idx].attn
    orig_forward = target.forward
    li_counter = {"n": 0}

    def hooked(x, past_kv=None, q_positions=None, use_cache=False):
        # replicate the score computation for the memory head only, then call orig.
        B, S, D = x.shape
        H, HD = target.n_heads, target.head_dim
        Q = target.W_q.linear(x).view(B, S, H, HD).transpose(1, 2)
        Knew = target.W_k.linear(x).view(B, S, H, HD).transpose(1, 2)
        Vnew = target.W_v.linear(x).view(B, S, H, HD).transpose(1, 2)
        if q_positions is None:
            q_pos = torch.arange(S, device=x.device)
        else:
            q_pos = q_positions.to(device=x.device, dtype=torch.long)
        if past_kv is not None:
            K_cache, V_cache, pos_cache = past_kv
            K = torch.cat([K_cache, Knew], dim=2)
            V = torch.cat([V_cache, Vnew], dim=2)
            k_pos = torch.cat([pos_cache.to(x.device), q_pos], dim=0)
        else:
            K, V, k_pos = Knew, Vnew, q_pos
        scores = torch.matmul(Q, K.transpose(-2, -1)) * target.scale
        dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs().float()
        scores2 = scores - target.alibi_slopes.view(1, H, 1, 1) * dist.unsqueeze(0)
        mask = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))
        scores2 = scores2.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        from c4_min.blogspec_model import softmax1
        attn = softmax1(scores2, dim=-1)
        # last query row, memory head
        qrow = -1
        w = attn[0, MEM_HEAD, qrow]            # [Sk] softmax weights
        sc = scores2[0, MEM_HEAD, qrow]        # [Sk] scores (post-alibi)
        # value nibbles each key would relay (the head's V on its local value slots).
        base = MEM_HEAD * HD
        vslots = list(range(base + 32 + 3, base + 32 + 3 + 8))  # value slots
        # only report rows with meaningful weight
        sink = 1.0 - float(w.sum())
        top = torch.topk(w, min(6, w.numel()))
        # decode the query row's QRY_BIN address + IS_LOAD from the INPUT residual.
        xrow = x[0, qrow]
        qaddr = 0
        for b in range(32):
            bit = 1 if float(xrow[L.QRY_BIN + b]) > 0.5 else 0
            qaddr |= bit << b
        is_load = float(xrow[L.IS_LOAD])
        # also the raw AX band value on the query row (what the query SHOULD be).
        axval = 0
        for j in range(8):
            nb = int(round(float(xrow[L.AX + j])))
            nb = max(0, min(15, nb))
            axval |= nb << (4 * j)
        print(f"\n  [mem-head] query row abs_pos={int(k_pos[qrow])}  "
              f"softmax1 sink={sink:.4e}  QRY_BIN_addr={qaddr} (0x{qaddr:X})  "
              f"IS_LOAD={is_load:.2f}  AX_band={axval}", flush=True)
        for rank in range(top.indices.numel()):
            ki = int(top.indices[rank])
            valnib = V[0, MEM_HEAD, ki, 32 + 3:32 + 3 + 8]
            val = 0
            for j in range(8):
                nb = int(round(float(valnib[j])))
                nb = max(0, min(15, nb))
                val |= nb << (4 * j)
            print(f"     key[{ki}] abs_pos={int(k_pos[ki]):4d} score={float(sc[ki]):+.3e} "
                  f"w={float(w[ki]):.4e} relay_val={val}", flush=True)
        # ALSO: find every candidate whose relay value == the target 0x20000, and
        # print its score (to see if the frame-local store exists but scores wrong).
        target_val = int(os.environ.get("C4_TARGET_VAL", str(P0)))
        if is_load > 0.5:
            allV = V[0, MEM_HEAD, :, 32 + 3:32 + 3 + 8]
            for ki in range(allV.shape[0]):
                val = 0
                for j in range(8):
                    nb = max(0, min(15, int(round(float(allV[ki, j])))))
                    val |= nb << (4 * j)
                if val == target_val:
                    print(f"     >>> STORE relay={target_val} at key[{ki}] "
                          f"abs_pos={int(k_pos[ki])} score={float(sc[ki]):+.3e} "
                          f"w={float(w[ki]):.4e}", flush=True)
                    # raw Q and K for the address channels (0..31) of the mem head,
                    # to see fractional query/key residue.
                    qch = Q[0, MEM_HEAD, qrow, :32]
                    kch = K[0, MEM_HEAD, ki, :32]
                    qkbit = (qch * kch)
                    print(f"         per-bit Q*K sum(addr chans 0..31) = "
                          f"{float(qkbit.sum()):+.4e}  (exact match would be "
                          f"+32*|smag|^2)", flush=True)
                    # raw query bits (fractional) — which lanes are non-{0, big}?
                    raw = [round(float(qch[b]), 1) for b in range(12)]
                    print(f"         Q addr chans[0..11] = {raw}", flush=True)
                    rawk = [round(float(kch[b]), 1) for b in range(12)]
                    print(f"         K addr chans[0..11] = {rawk}", flush=True)
                    # gate channels 32=BIAS(IS_STORE/IS_LOAD), 33=store-role, 34=load-en
                    for name, ch in (("BIAS", 32), ("STORE_ROLE", 33), ("LOAD_EN", 34)):
                        qv = float(Q[0, MEM_HEAD, qrow, ch])
                        kv = float(K[0, MEM_HEAD, ki, ch])
                        print(f"         chan {ch}({name}): Q={qv:+.4e} K={kv:+.4e} "
                              f"Q*K={qv * kv:+.4e}", flush=True)
                    # the store row's IS_STORE input flag (from the cached K we can't
                    # read x; report via the K decomposition above).
                    addr_qk = float((Q[0, MEM_HEAD, qrow, :32] *
                                     K[0, MEM_HEAD, ki, :32]).sum())
                    gate_qk = float((Q[0, MEM_HEAD, qrow, 32:35] *
                                     K[0, MEM_HEAD, ki, 32:35]).sum())
                    print(f"         raw score = (addr {addr_qk:+.3e} + gate "
                          f"{gate_qk:+.3e}) * hs; hs={target.scale:.4e}", flush=True)
            # also dump the raw QRY_BIN float on the query input row (0..11).
            rawq = [round(float(x[0, qrow, L.QRY_BIN + b]), 2) for b in range(12)]
            print(f"     QRY_BIN raw floats[0..11] = {rawq}", flush=True)
        return orig_forward(x, past_kv=past_kv, q_positions=q_positions,
                            use_cache=use_cache)

    target.forward = hooked
    # SparseBlock.__call__ dispatches to forward_hidden_cached -> attn.forward via
    # the cached path.  Patch the block __call__ to use our hooked attn.
    with LC._low_stack_sp():
        trace = run_pure_forward_cached(
            model, L, code, max_steps=len(code) + 2, mask=0xFFFFFFFF,
            verbose=True, evict=False, prune_interval=999999)
    print("\ntrace:", trace, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
