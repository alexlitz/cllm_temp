"""Call the REAL memset bytecode DIRECTLY (skip the ~45 slow malloc steps) with
args set up so its running byte-pointer local starts at 0x20000 — the exact
malloc_printf structure that fails.  memset is copied VERBATIM at its original PC
offsets (109-166) so its internal absolute branch targets are unchanged; a tiny
prologue at pc 0 pushes (s=0x20000, c=72, n=3) and JSRs to it.

Per-step model-AX vs word-truth (execution-order aligned, flushed) pins the FIRST
divergence — expected in the memset loop (the frame-local pointer re-read).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from c4_min import isa
from c4_min import libprog_corpus as LC

S_PTR = 0x20000
FILL = 72
COUNT = int(os.environ.get("C4_COUNT", "3"))
MEMSET_PC = 109


def build():
    entry = [e for e in LC.CORPUS if e.name == "malloc_printf"][0]
    raw, _ = LC.compile_entry(entry)
    full = LC.retarget_to_neural_abi(raw)          # 256 instrs; memset at 109-166
    # new code table = a copy of the full table so memset stays at its offsets.
    code = [isa.Instr(isa.NOP, 0) for _ in range(len(full))]
    for pc in range(MEMSET_PC, 167):
        code[pc] = full[pc]
    # prologue at pc 0: push args left-to-right (s, c, n) then JSR memset ; HALT.
    prologue = [
        ("IMM", S_PTR), ("PSH", 0),        # s
        ("IMM", FILL), ("PSH", 0),         # c
        ("IMM", COUNT), ("PSH", 0),        # n
        ("JSR", MEMSET_PC),                # call memset(s,c,n)
        ("ADJ", 3), ("HALT", 0),           # pop args, halt
    ]
    pro = isa.assemble(prologue)
    for i, ins in enumerate(pro):
        code[i] = ins
    return code


def word_ref_trace(code, max_steps, sp_init):
    mem = {}; sp = bp = sp_init; ax = pc = 0; out = []; steps = 0
    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]; op, imm = ins.op, ins.imm; i = pc; pc += 1
        li_addr = li_val = None
        if op == isa.IMM: ax = imm & 0xFFFFFFFF
        elif op == isa.LEA: ax = (bp + 4 * imm) & 0xFFFFFFFF
        elif op == isa.PSH: sp -= 4; mem[sp] = ax
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
            v = mem.get(sp, 0); sp += 4
            ax = {isa.ADD: v + ax, isa.SUB: v - ax, isa.MUL: v * ax,
                  isa.DIV: (v // ax if ax else 0),
                  isa.MOD: (v % ax if ax else 0)}[op] & 0xFFFFFFFF
        elif op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            v = mem.get(sp, 0) & 0xFF; sp += 4
            ax = 1 if {isa.EQ: v == ax, isa.NE: v != ax, isa.LT: v < ax,
                       isa.GT: v > ax, isa.LE: v <= ax, isa.GE: v >= ax}[op] else 0
        elif op in (isa.LI, isa.LC):
            li_addr = ax
            ax = mem.get(ax, 0) & (0xFFFFFFFF if op == isa.LI else 0xFF); li_val = ax
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp, 0); sp += 4
            mem[addr] = ax & (0xFFFFFFFF if op == isa.SI else 0xFF)
        elif op == isa.JMP: pc = imm
        elif op == isa.BZ: pc = imm if ax == 0 else pc
        elif op == isa.BNZ: pc = imm if ax != 0 else pc
        elif op == isa.JSR: sp -= 4; mem[sp] = i + 1; pc = imm
        elif op == isa.ENT: mem[sp - 4] = bp; sp -= 4; bp = sp; sp -= 4 * imm
        elif op == isa.ADJ: sp += 4 * imm
        elif op == isa.LEV: sp = bp; bp = mem.get(sp, 0); pc = mem.get(sp + 4, 0); sp += 8
        elif op == isa.NOP: pass
        elif op == isa.HALT: out.append((i, op, ax, sp, bp, li_addr, li_val)); break
        out.append((i, op, ax, sp, bp, li_addr, li_val))
    return out


def main():
    os.system("free -g | head -2")
    code = build()
    STEPS = int(os.environ.get("C4_STEPS", "80"))
    from c4_min.lib_neural import build_lib_model_streaming
    print("building model ...", flush=True)
    sparse, L, _ = build_lib_model_streaming(
        code_size=len(code) + 2, recurrent_divmod=True, addr32=True)
    print("built dim", L.D, flush=True)
    os.system("free -g | head -2")

    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    import c4_min.nibble_pure_forward_cached as _pfc

    evict = os.environ.get("C4_EVICT", "1") not in ("0", "")
    pi = int(os.environ.get("C4_PRUNE_INTERVAL", "60"))
    print(f"evict={evict} prune_interval={pi} steps={STEPS} count={COUNT}", flush=True)

    # -- optional CAM hook: dump the memory head's query + top keys at each LI --
    if os.environ.get("C4_CAM", "0") == "1":
        import torch
        from c4_min.nibble_pure_forward import N_ROLES
        from c4_min.blogspec_model import softmax1
        MEM_HEAD = N_ROLES
        names = getattr(L, "_block_names", None)
        mem_idx = names.index("mem-cam") if names and "mem-cam" in names else None
        tgt = sparse.blocks[mem_idx].attn
        _of = tgt.forward

        def hooked(x, past_kv=None, q_positions=None, use_cache=False):
            B, S, D = x.shape
            H, HD = tgt.n_heads, tgt.head_dim
            Q = tgt.W_q.linear(x).view(B, S, H, HD).transpose(1, 2)
            Kn = tgt.W_k.linear(x).view(B, S, H, HD).transpose(1, 2)
            Vn = tgt.W_v.linear(x).view(B, S, H, HD).transpose(1, 2)
            qp = torch.arange(S) if q_positions is None else q_positions.long()
            if past_kv is not None:
                Kc, Vc, pc_ = past_kv
                K = torch.cat([Kc, Kn], dim=2); Vv = torch.cat([Vc, Vn], dim=2)
                kp = torch.cat([pc_.long(), qp], dim=0)
            else:
                K, Vv, kp = Kn, Vn, qp
            sc = torch.matmul(Q, K.transpose(-2, -1)) * tgt.scale
            dist = (qp.unsqueeze(1) - kp.unsqueeze(0)).abs().float()
            sc = sc - tgt.alibi_slopes.view(1, H, 1, 1) * dist.unsqueeze(0)
            sc = sc.masked_fill((kp.unsqueeze(0) > qp.unsqueeze(1)).unsqueeze(0).unsqueeze(0),
                                float("-inf"))
            w = softmax1(sc, dim=-1)[0, MEM_HEAD, -1]
            scr = sc[0, MEM_HEAD, -1]
            xr = x[0, -1]
            isload = float(xr[L.IS_LOAD])
            if isload > 0.5:
                qaddr = 0
                for b in range(32):
                    if float(xr[L.QRY_BIN + b]) > 0.5:
                        qaddr |= 1 << b
                sink = 1.0 - float(w.sum())
                top = torch.topk(w, min(5, w.numel()))
                print(f"    [CAM] q_addr={qaddr}(0x{qaddr:X}) IS_LOAD={isload:.1f} "
                      f"sink={sink:.3e}", flush=True)
                for r in range(top.indices.numel()):
                    ki = int(top.indices[r])
                    vn = Vv[0, MEM_HEAD, ki, 32 + 3:32 + 3 + 8]
                    val = 0
                    for j in range(8):
                        nb = max(0, min(15, int(round(float(vn[j])))))
                        val |= nb << (4 * j)
                    print(f"        key[{ki}] pos={int(kp[ki])} score={float(scr[ki]):+.3e} "
                          f"w={float(w[ki]):.3e} relay={val}", flush=True)
            return _of(x, past_kv=past_kv, q_positions=q_positions, use_cache=use_cache)
        tgt.forward = hooked

    with LC._low_stack_sp():
        sp_init = _pfc.SP_INIT
        ref = word_ref_trace(code, STEPS, sp_init)
        _orig = _pfc._decode_reg_from_nibbles
        step = {"i": 0}

        def _traced(state, L_, reg_base):
            v = _orig(state, L_, reg_base)
            if reg_base == L_.AX:
                i = step["i"]
                if i < len(ref):
                    pcv, op, rax, rsp, rbp, la, lv = ref[i]
                    tag = ""
                    if op in (isa.LI, isa.LC):
                        tag = (f"  {isa.NAMES[op]} @0x{la:X} truth={lv} model={v}"
                               + ("  <<< WRONG" if v != lv else ""))
                    print(f"step {i:3d} pc={pcv:3d} {isa.NAMES.get(op, op):4s} "
                          f"truth_ax={rax} model_ax={v}{tag}", flush=True)
                step["i"] += 1
            return v
        _pfc._decode_reg_from_nibbles = _traced
        try:
            got = run_pure_forward_cached(
                sparse, L, code, max_steps=STEPS, mask=0xFFFFFFFF, verbose=False,
                evict=evict, prune_interval=pi)
        finally:
            _pfc._decode_reg_from_nibbles = _orig
    print("\ndone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
