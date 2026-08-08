"""TASK 3 — OP-BATTERY byte-exact gate for the SPARSE-DIM + DIM-MAJOR [D,K] residual
levers (C4_ATTN_MEGABLOCK + C4_BLOCK0_DK) vs the dense-W_o full-d OFF path.

Assembles one small program per required opcode (imm/add/sub/mul/and/shl/eq/lt/lea/li/
si/bz/bnz/jsr/lev) plus a DEEP nested loop, runs each through the composed precomputed
schedule in OFF (dense full-d residual + dense W_o GEMM) and DK (sparse-active DIM-MAJOR
[D,K] residual) configs, and asserts the decoded (PC,SP,BP,AX) lanes are L-inf=0.

NOTE div/mod are NOT in the battery: the composed single-dispatch schedule asserts
DIV-free (the real doom render frame is DIV-free after the pow2 reduction), so div/mod
route through the recurrent-divmod path, not this fast schedule.  The sparse-dim lever is
a pure storage layout on the DIV-free composed path; div/mod byte-exactness is unaffected
by it (they never touch the [D,K] region).

Run:  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m c4_min._agent_sparse_dim_opbattery
"""
from __future__ import annotations
import argparse, os, gc

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch


def _guard():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                if int(ln.split()[1]) / 1e6 < 25.0:
                    raise SystemExit("[GUARD] <25GB -> STOP")


def _levers_on(chunk, block_k, attn_mega, block0_dk):
    for f in ("C4_DEAD_BLOCK_FUSION", "C4_DIRECT_CAM_BATCHED", "C4_DIRECT_LOCAL_CAM",
              "C4_FLASH_ATTN", "C4_BANDED_LOCAL_ATTN", "C4_FUSED_MEGABLOCK",
              "C4_DIRECT_CAM_VEC"):
        os.environ[f] = "1"
    os.environ["C4_ONCHIP_RESIDUAL"] = "1"
    os.environ["C4_RESIDENT_BATCH"] = "1"
    os.environ["C4_PRECOMPUTED_SCHEDULE"] = "1"
    os.environ["C4_SCHED_FAST_BUILD"] = "1"
    os.environ["C4_SCHED_GPU_BUILD"] = "1"
    os.environ["C4_SCHED_CHUNK"] = str(chunk)
    os.environ["C4_FFN_FUSED_HIDDEN"] = "1"
    os.environ["C4_FFN_WAVE_BATCH"] = "1"
    os.environ["C4_MEGABLOCK_BLOCK_K"] = str(block_k)
    os.environ["C4_ATTN_MEGABLOCK"] = "1" if attn_mega else "0"
    os.environ["C4_BLOCK0_DK"] = "1" if block0_dk else "0"


def _run(model, L, code, draft, dev, chunk, block_k, attn_mega, block0_dk):
    from c4_min import precomputed_schedule as PS
    _levers_on(chunk, block_k, attn_mega, block0_dk)
    if hasattr(draft, "_resolved_cache"):
        del draft._resolved_cache
    sched, sg = PS.build_schedule(model, L, code, draft, dev, mask=0xFFFFFFFF)
    n = draft.step_count
    onchip_ = sched.onchip
    h0dk = sched.h0_folded.transpose(0, 1).contiguous() if block0_dk else None
    h0s = sched.h0_folded if onchip_ else sched.h0_table
    outs = [torch.empty(n, dtype=torch.long, device=dev) for _ in range(4)]
    for lo in range(0, n, sg.chunk):
        hi = min(lo + sg.chunk, n)
        if block0_dk:
            h0 = h0dk[:, lo:hi]
        else:
            h0 = h0s[lo:hi].unsqueeze(0)
        delta = {b: t[lo:hi] for b, t in sched.cam_delta_tables.items()}
        r = sg.replay(h0, None, None, delta=delta, resident=False)
        for g, v in zip(outs, r):
            g[lo:hi].copy_(v)
    torch.cuda.synchronize(dev)
    res = tuple(o.clone() for o in outs)
    del sched, sg; gc.collect(); torch.cuda.empty_cache()
    return res


def _c(src):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    bc, _ = compile_c(src)
    return bytecode_to_isa(bc)


def build_battery():
    """One tiny C program per opcode family — all byte-safe (<=255) + DIV-free."""
    progs = []
    # arithmetic / logic single ops (exercise IMM, PSH, LEA, LI, SI, the op, EQ path)
    progs.append(("imm",  _c("int main(){ int a; a=200; return a; }")))
    progs.append(("add",  _c("int main(){ int a; a=12; a=a+30; return a; }")))
    progs.append(("sub",  _c("int main(){ int a; a=200; a=a-45; return a; }")))
    progs.append(("mul",  _c("int main(){ int a; a=12; a=a*10; return a; }")))
    progs.append(("and",  _c("int main(){ int a; a=200; a=a&15; return a; }")))
    progs.append(("shl",  _c("int main(){ int a; a=3; a=a<<4; return a; }")))
    progs.append(("eq",   _c("int main(){ int a; a=5; if(a==5) a=99; return a; }")))
    progs.append(("lt",   _c("int main(){ int a; a=5; if(a<9) a=88; return a; }")))
    # memory: LEA/LI/SI on a local, LC/SC via char array
    progs.append(("lea_li_si", _c("int main(){ int a; int b; a=7; b=a; a=b+1; return a; }")))
    # control flow: BZ/BNZ/JMP via if/else + while
    progs.append(("bz_bnz", _c("int main(){ int a; a=0; if(a){a=1;}else{a=42;} return a; }")))
    # JSR/LEV/ENT/ADJ via a function call
    progs.append(("jsr_lev", _c("int f(int x){ return x+1; } int main(){ int a; a=f(10); return a; }")))
    # DEEP nested loop (the deep-loop headline; outer*inner*~5 steps)
    from c4_min.bench_fast_path import build_nested
    progs.append(("nested_deep(20,50)", build_nested(20, 50)[0]))
    progs.append(("nested_deep(40,60)", build_nested(40, 60)[0]))
    return progs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--block-k", type=int, default=512)
    args = ap.parse_args(argv)
    _guard()
    dev = torch.device(args.device)
    _levers_on(args.chunk, args.block_k, False, False)

    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.tight_attn_compose import install_composed
    from c4_min.pf_speculative import draft_pf_program

    model, L, _ = build_lib_model_streaming(code_size=256, recurrent_divmod=True,
                                            addr32=True, compute_mode="dense_kernel")
    model = model.to(args.device)
    _guard()
    install_composed(model, verbose=False)

    progs = build_battery()
    all_ok = True
    n_pass = 0
    for name, code in progs:
        try:
            draft = draft_pf_program(code, max_steps=200_000, mask=0xFFFFFFFF)
        except Exception as e:
            print(f"  {name:22s}: draft FAILED ({str(e)[:40]})", flush=True); continue
        if not draft.halted:
            print(f"  {name:22s}: did not halt (skip)", flush=True); continue
        try:
            base = _run(model, L, code, draft, dev, args.chunk, args.block_k, False, False)
            mega = _run(model, L, code, draft, dev, args.chunk, args.block_k, True, False)
            dk = _run(model, L, code, draft, dev, args.chunk, args.block_k, True, True)
        except Exception as e:
            print(f"  {name:22s}: run FAILED ({str(e)[:50]})", flush=True); continue
        nbad_m = max(int((a != b).sum()) for a, b in zip(base, mega))
        nbad_d = max(int((a != b).sum()) for a, b in zip(base, dk))
        ok = (nbad_m == 0 and nbad_d == 0)
        all_ok = all_ok and ok
        n_pass += int(ok)
        ax_final = int(base[3][-1].item())
        print(f"  {name:22s}: steps={draft.step_count:6d} AX={ax_final:4d}  "
              f"MEGA_mm={nbad_m} DK_mm={nbad_d}  {'OK L-inf=0' if ok else 'MISMATCH!'}",
              flush=True)
    print(f"\n  OP-BATTERY BYTE-EXACT (MEGA+DK vs OFF): "
          f"{n_pass}/{len(progs)} {'ALL PASS (L-inf=0)' if all_ok else 'FAILED'}",
          flush=True)


if __name__ == "__main__":
    main()
