"""_agent_mandelbrot_doom_fast_verify.py — does the HARDENED doom pure-forward
transformer (the byte-exact doom build) run the REAL (signed) Mandelbrot byte-exact?

The prior Mandelbrot divergence (agent a0a8e302: AX 0xFF-leak + KV-CAM aliasing) was
measured on the VANILLA ``qwen_vanilla_vm`` 1096-corpus model / the plain
``run_pure_forward_cached`` driver — NOT the hardened composed doom build.  The doom
build runs the REAL id linuxdoom byte-exact after the #824/#826/#829 hardening
(LEA_WIDE, CMP32-order, pow2-DIV, ...).  Mandelbrot (``mandelbrot_native``, the REAL
signed fixed-point set: toward-zero MUL, DIV rescale, signed LT/GT, JSR/ENT/LEV loop)
exercises the SAME opcodes doom does — so the hardened build SHOULD run it byte-exact.

This harness runs the ``mandelbrot_native.pixel_program_native`` (the REAL signed
Mandelbrot per-pixel escape loop) through the EXACT byte-exact doom verify path:

  * ``build_lib_model_streaming(recurrent_divmod=True, addr32=True,
    compute_mode='dense_kernel')`` — the SAME model the doom verify builds.
  * ``install_local_attention`` + ``install_dead_block_fusion`` — the composed
    doom attention stack.
  * ``verify_blocks(block_moe=False, fast=True)`` — the byte-exact full-242-block
    composed verify (the SAME entry the doom byte-exact verify uses; it carries the
    recurrent-divmod span, so DIV/MOD steps are handled — the DIV-free precomputed
    schedule collapse is a *speed* lever that falls THROUGH to this path on a DIV
    program).
  * Env matches the doom byte-exact verify: ``C4_DRAFT_CMP32=1`` (32-bit signed cmp,
    matching the model — REQUIRED for Mandelbrot's signed LT/GT), ``C4_MEM_ADDR_BITS``,
    ``C4_EXACT_EVICT``, ``C4_PF_CFM=1`` (LEAN build).
  * ORACLE = ``nibble_pure_forward_complete.ref_interpret`` at ``mask=0xFFFFFFFF`` (the
    c4vm32 word oracle) — the SAME reference the draft mirrors and the verify compares
    every query-row register (PC/AX/SP/BP) against, argmax byte-exact (L-inf=0).

The verify's ``all_matched`` + ``accepted_steps`` + ``first_mismatch`` report the
EXACT diverging step + opcode + kind (``register`` = a real model-arithmetic argmax
divergence; ``cam_addr`` = the KV address-CAM aliasing the vanilla build hit).

LEAN + memory-safe: streaming build (~1-6 GB), single process, STOP < 25 GB host RAM.
Idle GPU only.  Golden 069cc32f is never touched (this authors NO weights).

FINDINGS (2026-08-06, HEAD 9566fc45, golden 069cc32f, RTX-class idle GPU)
------------------------------------------------------------------------
The doom pure-forward build runs Mandelbrot BYTE-EXACT — the vanilla-build
divergence (agent a0a8e302: AX 0xFF-leak + KV-CAM aliasing at ~step 136) is NOT
reproduced here; the hardened build gets 5x further and the wall is a DIFFERENT,
DATA-DEPENDENT one.  Measured (real signed ``mandelbrot_native``, mask=0xFFFFFFFF
oracle):

  * pixel (-33,0) mi12, escapes at i=1  : 508/508  MATCHED, final_ax=1  == oracle. BYTE-EXACT.
  * pixel (0,0)   mi12, interior (no esc): 4201/4201 MATCHED, final_ax=12 == oracle. BYTE-EXACT
                                           (the FULL 12-iteration escape loop, all MUL/DIV/SHR/
                                           XOR/signed-LT-GT/JSR/ENT/LEV, completes).
  * pixel (-10,5) mi12, boundary        : accepted 688/4237, DIVERGES at step 688.
  * ``_mandel_src`` unsigned-safe 2x1 mi3: 730/730 MATCHED (render b'* \n'). BYTE-EXACT.

The (0,0) interior pixel — the SAME signed program, full loop — completes byte-exact,
so LEV/BP restore is structurally correct.  The (-10,5) divergence is DATA-DEPENDENT:

  step 688, opcode LEV (the 10th subroutine return; LEVs 1-9 all matched):
    PC ok (101), AX ok (0xFFFFFFFD = -3), SP ok (65548);
    BP: model decoded 65263 (0xFEEF) vs oracle 65536 (0x10000 = SP_INIT).
  kind = register (NOT ``cam_addr``): the CAM resolves a row but the 32-bit
  memory-VALUE decode of the restored BP drops the high byte (0x01_0000 -> 0x00_FEEF).
  eviction ON vs OFF is IDENTICAL (step 688 both) -> NOT an eviction bug; a hard
  multi-byte memory-value decode wall that trips only once the loop has pushed enough
  WRAPPED-NEGATIVE signed values (AX reaches -3/-50/...) into the stack/mem KV.  The
  interior pixel keeps every value small/non-negative and never trips it.

FIX TARGET: the BP-restore memory-value decode on LEV (the ``L.BP_VAL`` / "pop"-head
MEM[BP] read) — the high-byte carry of a 3-byte saved-BP value under accumulated
signed KV traffic.  This is the AX/BP high-byte 0xFF-leak family, surfacing on the
frame-pointer restore rather than an AX arithmetic op — a REAL model decode wall, not
the vanilla build's address-CAM aliasing.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import argparse
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["C4_PF_CFM"] = "1"                      # LEAN streaming build
os.environ.setdefault("C4_DRAFT_CMP32", "1")       # 32-bit signed cmp (matches model)
os.environ.setdefault("C4_MEM_ADDR_BITS", "18")
os.environ.setdefault("C4_EXACT_EVICT", "1")
os.environ.setdefault("C4_MEM_EFF", "2000000")
os.environ.setdefault("C4_KV_STACK", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def _guard():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                if int(ln.split()[1]) / 1e6 < 25.0:
                    raise SystemExit("[GUARD] host MemAvailable < 25 GB -> STOP")


def _op_hist(code, isa):
    h = {}
    for i in code:
        nm = isa.NAMES.get(i.op, str(i.op))
        h[nm] = h.get(nm, 0) + 1
    return h


def build(dev, code, cap, window):
    """Build the byte-exact doom model + draft for the given Mandelbrot ``code``."""
    from c4_min.pf_speculative import draft_pf_program
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion

    draft = draft_pf_program(code, max_steps=cap, mask=0xFFFFFFFF)
    sparse, L, _ = build_lib_model_streaming(
        code_size=max(len(code) + 2, 64), recurrent_divmod=True,
        addr32=True, compute_mode="dense_kernel")
    _guard()
    sparse = sparse.to(dev)
    install_local_attention(sparse, window=window, drop_local_kv=True,
                            content_bound_global=False, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    return sparse, L, draft


def run(cx, cy, mi, cap, K, window, device, evict=True):
    import torch
    from c4_min import isa
    from c4_min import mandelbrot_native as MN
    from c4_min.nibble_pure_forward_complete import ref_interpret
    from c4_min.pf_speculative import verify_blocks

    dev = device
    if dev != "cpu":
        torch.cuda.init()

    code = MN.pixel_program_native(cx, cy, mi)
    hist = _op_hist(code, isa)
    # ORACLE: the c4vm32 word reference (mask=0xFFFFFFFF) — steps + final AX.
    tr = ref_interpret(code, max_steps=2_000_000, mask=0xFFFFFFFF)
    ref_steps = len(tr)
    ref_final_ax = tr[-1] & 0xFFFFFFFF
    ref_esc = MN.escape_count_native(cx, cy, mi)
    print(f"[mandel] REAL signed pixel cx={cx} cy={cy} mi={mi}: instrs={len(code)} "
          f"ref_steps={ref_steps} ref_escape={ref_esc} ref_final_ax={ref_final_ax}",
          flush=True)
    print(f"[mandel] op-hist: {sorted(hist.items())}", flush=True)
    if ref_steps > cap:
        print(f"[mandel] NOTE: ref_steps {ref_steps} > cap {cap}; draft is capped "
              f"(verify runs the first {cap} steps).", flush=True)

    _guard()
    t0 = time.time()
    sparse, L, draft = build(dev, code, cap, window)
    print(f"[mandel] built: dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} "
          f"draft_steps={draft.step_count} build_wall={time.time() - t0:.1f}s", flush=True)

    stats = {}
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev, evict=evict,
                       mask=0xFFFFFFFF, stats=stats, fast=True, block_moe=False,
                       oom_backoff=True, min_block_steps=4, prime_chunk=2048)
    wall = time.time() - t0

    peak_vram = stats.get("peak_vram_gb")
    if dev != "cpu":
        try:
            peak_vram = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        except Exception:
            pass
    ms = wall / max(draft.step_count, 1) * 1e3
    print(f"\n[mandel] VERIFY (byte-exact doom path): matched={vr.all_matched} "
          f"accepted={vr.accepted_steps}/{vr.total_steps} forwards={vr.forwards} "
          f"final_ax={vr.decoded_final_ax} ms/step={ms:.2f} wall={wall:.1f}s "
          f"peak_vram_gb={peak_vram}", flush=True)

    # BYTE-EXACT gate: every checked query-row register matched (all_matched) AND the
    # model's decoded final AX == the c4vm32 oracle's final AX (when the run completed).
    completed = (draft.step_count >= ref_steps)
    ax_ok = (vr.decoded_final_ax is None) or (vr.decoded_final_ax == ref_final_ax)
    byte_exact = vr.all_matched and ax_ok
    print(f"[mandel] ref_final_ax={ref_final_ax} model_final_ax={vr.decoded_final_ax} "
          f"ax_match={ax_ok} completed_full_pixel={completed}", flush=True)

    if not vr.all_matched and vr.first_mismatch:
        fm = vr.first_mismatch
        s_bad = fm.get("step")
        op_bad = None
        if s_bad is not None and s_bad < draft.step_count:
            op_bad = draft.frames[s_bad].get("op")
        print(f"\n[mandel] *** DIVERGENCE ***", flush=True)
        print(f"[mandel]   step={s_bad} opcode={op_bad} kind={fm.get('kind')}", flush=True)
        print(f"[mandel]   got={fm.get('got')} want={fm.get('want')}", flush=True)
        print(f"[mandel]   detail={fm.get('detail')}", flush=True)
        # classify: register (real arithmetic argmax bug) vs cam_addr (the vanilla
        # build's KV address-CAM aliasing).
        kind = fm.get("kind")
        if kind == "cam_addr":
            print(f"[mandel]   -> KV address-CAM aliasing (the vanilla-build failure mode).",
                  flush=True)
        else:
            print(f"[mandel]   -> REGISTER argmax divergence (model arithmetic/decode at "
                  f"opcode {op_bad}).", flush=True)

    print(f"\n[mandel] >>> BYTE-EXACT (doom build runs REAL Mandelbrot): "
          f"{'YES' if byte_exact and completed else ('YES-so-far(capped)' if byte_exact else 'NO')}"
          f" <<<", flush=True)
    return 0 if byte_exact else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cx", type=int, default=-10)
    ap.add_argument("--cy", type=int, default=5)
    ap.add_argument("--mi", type=int, default=12)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-evict", dest="evict", action="store_false")
    args = ap.parse_args(argv)
    _guard()
    return run(args.cx, args.cy, args.mi, args.cap, args.K, args.window, args.device,
               evict=args.evict)


if __name__ == "__main__":
    raise SystemExit(main())
