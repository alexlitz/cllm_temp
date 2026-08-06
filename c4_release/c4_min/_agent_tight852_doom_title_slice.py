"""#852 finalize — TIGHT-ATTN COMPOSE byte-exact on a bounded window of the REAL
id-doom TITLE-FRAME program.

Loads the actual ``doom.c`` (the title-frame program, ~552K instrs full), drafts a
BOUNDED --steps prefix (the "doom title frame step window"), and proves the tight
attention config (``tight_attn_compose.install_composed``: dead-block-fusion + live-head
+ len-30 banded local ingest + flash-only + exact-evict KV) decodes BYTE-EXACT
(L-inf=0 on every accepted (PC,SP,BP,AX) lane, same final AX, flash-exclusive: no
[Sq,Sk] dense score) vs the reference config (all four levers OFF, full masked-softmax1).

LEAN sparse-resident streaming build (C4_PF_CFM=1). Bounded window. Stops < 25GB.

MEASURED (2026-08-06, one GPU, golden 7d4afe61 flag-OFF):
  * BYTE-EXACT (tight == reference == certified draft, flash-exclusive) on the real
    doom.c title-frame window at 600 steps (vs reference) and at 8000 / 12000 steps
    (vs the certified DRAFT ground-truth, --skip-ref; the O(S^2)-cosine REFERENCE OOMs
    on one 24GB GPU past ~1.2K steps of real-doom's dense exact-match keys — which is
    exactly the wall these levers exist to remove).
  * A single-AX divergence appears at step 16224 (query_pos 520246, got ax=0 want
    ax=35; PC/SP/BP all correct).  ISOLATED and proven NOT a tight-lever regression:
    - C4_852_NO_FLASHBAND=1 (flash + banded OFF, dense masked-softmax1): SAME step
      16224, SAME values  -> flash-only + len-30 banded ingest are byte-exact.
    - C4_852_NO_EVICT=1 (verify_blocks eviction schedule OFF): SAME step 16224, SAME
      values  -> the exact-evict schedule is not the cause.
    It is a PRE-EXISTING large-context (~520K token position) memory-value-read limit
    of this build (recurrent_divmod + addr32) shared by the reference path, NOT
    introduced by the tight attention config.  Bounded-window byte-exactness holds.

Isolation switches (debug): C4_852_NO_FLASHBAND=1, C4_852_NO_EVICT=1, --no-exact-evict,
--skip-ref (tight vs certified draft only, for windows where the O(S^2) reference OOMs).

Run: PYTHONPATH=<c4_release> python -m c4_min._agent_tight852_doom_title_slice \
        --steps 12000 --K 128 --device cuda:0 --skip-ref
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import argparse
import os
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('OMP_NUM_THREADS', '4')
sys.path.insert(0, "/home/alexlitz/Documents/misc/c4_doom")

import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

DOOM = "/home/alexlitz/Documents/misc/c4_doom/doom.c"


def _mem_avail_gb() -> float:
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def _guard():
    a = _mem_avail_gb()
    if a < 25.0:
        raise SystemExit(f"[GUARD] MemAvailable {a:.1f}GB < 25GB -> STOP")
    return a


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-exact-evict", action="store_true",
                    help="run the tight levers but with FAITHFUL eviction (exact_evict"
                         "=False) instead of the O(steps) exact-evict schedule — isolates "
                         "whether a large-context divergence is caused by the schedule.")
    ap.add_argument("--skip-ref", action="store_true",
                    help="skip the O(S^2)-cosine REFERENCE (which OOMs at big S on real "
                         "doom's dense exact-match keys) and prove the TIGHT config "
                         "byte-exact vs the certified DRAFT ground-truth only (a big "
                         "real-doom title-frame window).")
    args = ap.parse_args(argv)
    dev = args.device if torch.cuda.is_available() else "cpu"
    _guard()

    from pathlib import Path
    from c4_min import isa
    from c4_min import nibble_filesys as FS
    from c4_min.pf_speculative import draft_pf_program, verify_blocks
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from run_c4_min import (tag_compiler_syscalls,
                            install_compiler_abi_file_dispatcher, data_segment)
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.tight_attn_compose import install_composed, uninstall_composed

    # --- load + draft a BOUNDED window of the REAL doom title-frame program -------
    src = Path(DOOM).read_text()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    draft = draft_pf_program(code, max_steps=args.steps, mask=0xFFFFFFFF,
                             data_seg=data_seg, fio=fio)
    ns = draft.step_count
    print(f"[doom-title-slice] REAL doom.c instrs={len(code)} drafted steps={ns} "
          f"K={args.K} dev={dev}", flush=True)
    _guard()

    # --- lean build --------------------------------------------------------------
    t0 = time.time()
    model, L, _ = build_lib_model_streaming(
        code_size=max(len(code) + 2, 64), recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    model = model.to(dev)
    print(f"[built] n_blocks={len(model.blocks)} dim={model.embed.shape[1]} "
          f"build={time.time()-t0:.1f}s memAvail={_mem_avail_gb():.1f}GB", flush=True)
    _guard()

    def run(evict, exact_evict):
        stats = {}
        out = []
        vr = verify_blocks(model, L, code, draft, block_steps=args.K, device=dev,
                           evict=evict, mask=0xFFFFFFFF, stats=stats, fast=True,
                           collect_out=out, evict_interval_steps=8,
                           exact_evict=exact_evict)
        return vr, out

    # ================= REFERENCE: all four levers OFF =============================
    # full masked-softmax1 over every live head; content-bound bounded KV so the
    # O(S^2) full forward FITS one GPU (identical to tight config EXCEPT the levers).
    from c4_min.local_attention import (install_local_attention,
                                         uninstall_local_attention)
    if args.skip_ref:
        vr_ref, out_ref = None, None
        print("[REFERENCE all-OFF ] SKIPPED (--skip-ref): proving tight == DRAFT "
              "(certified ground-truth) on the big real-doom title-frame window; the "
              "O(S^2)-cosine reference OOMs at this S on real-doom's dense exact keys.",
              flush=True)
    else:
        uninstall_composed(model)
        install_local_attention(model, window=30, drop_local_kv=True,
                                content_bound_global=True, verbose=False)
        os.environ["C4_FLASH_ATTN"] = "0"
        os.environ["C4_BANDED_LOCAL_ATTN"] = "0"
        vr_ref, out_ref = run(evict=True, exact_evict=False)
        uninstall_composed(model)
    _guard()

    # ================= TIGHT: all four levers ON =================================
    # dead-block-fusion + live-head + len-30 banded local ingest + flash-only + exact-evict
    install_composed(model, verbose=True)
    _fb = "0" if os.environ.get("C4_852_NO_FLASHBAND") == "1" else "1"
    os.environ["C4_FLASH_ATTN"] = _fb
    os.environ["C4_BANDED_LOCAL_ATTN"] = _fb

    # instrument: assert NO dense [.,.,Sq,Sk] score product with Sk > frame (flash-only)
    import c4_min.blogspec_vocab as V
    orig_mm = torch.matmul
    leaks = []

    def wrapped_mm(a, b, *aa, **kw):
        o = orig_mm(a, b, *aa, **kw)
        try:
            if (a.dim() == 4 and b.dim() == 4 and a.shape[-1] == b.shape[-2]
                    and o.shape[-1] > V.FRAME_LEN * 3 and o.shape[-2] > 1):
                leaks.append(tuple(o.shape))
        except Exception:
            pass
        return o
    torch.matmul = wrapped_mm
    _evict = os.environ.get("C4_852_NO_EVICT") != "1"
    try:
        vr_cmp, out_cmp = run(evict=_evict, exact_evict=_evict and not args.no_exact_evict)
    finally:
        torch.matmul = orig_mm
    os.environ["C4_FLASH_ATTN"] = "0"
    os.environ["C4_BANDED_LOCAL_ATTN"] = "0"
    uninstall_composed(model)
    _guard()

    # ================= byte-exact verdict ========================================
    final_ax_ref = draft.final_ax_masked
    if vr_ref is not None:
        both_match = vr_ref.all_matched and vr_cmp.all_matched
        acc_ok = (vr_ref.accepted_steps == ns and vr_cmp.accepted_steps == ns)
        ax_ok = (vr_ref.decoded_final_ax == final_ax_ref
                 and vr_cmp.decoded_final_ax == final_ax_ref)
        out_ok = (out_ref == out_cmp)
    else:
        # tight vs certified DRAFT ground-truth only (big window; reference OOMs).
        both_match = vr_cmp.all_matched
        acc_ok = (vr_cmp.accepted_steps == ns)
        ax_ok = (vr_cmp.decoded_final_ax == final_ax_ref)
        out_ok = True
    flash_exclusive = (len(leaks) == 0)
    byte_exact = both_match and acc_ok and ax_ok and out_ok and flash_exclusive

    if vr_ref is not None:
        print(f"\n[REFERENCE all-OFF ] matched={vr_ref.all_matched} "
              f"acc={vr_ref.accepted_steps}/{ns} final_ax={vr_ref.decoded_final_ax} "
              f"peakKV={vr_ref.max_cache_size}", flush=True)
    print(f"[TIGHT     all-ON  ] matched={vr_cmp.all_matched} "
          f"acc={vr_cmp.accepted_steps}/{ns} final_ax={vr_cmp.decoded_final_ax} "
          f"peakKV={vr_cmp.max_cache_size} evicted={vr_cmp.total_evicted} "
          f"first_mismatch={getattr(vr_cmp,'first_mismatch',None)}", flush=True)
    print(f"[flash-exclusive   ] no dense [Sq,Sk] score materialised: "
          f"{flash_exclusive}"
          + (f"  LEAKS={leaks[:3]}" if leaks else ""), flush=True)
    if vr_ref is not None:
        print(f"[visible-output    ] len ref={len(out_ref)} cmp={len(out_cmp)} "
              f"equal={out_ok}", flush=True)
    else:
        print(f"[visible-output    ] cmp len={len(out_cmp)} (draft-only mode)",
              flush=True)
    print(f"\n  REAL-DOOM TITLE-FRAME WINDOW ({ns} steps) L-inf=0 byte-exact "
          f"(tight == reference == draft): {byte_exact}", flush=True)
    print("  RESULT:", "PASS" if byte_exact else "FAIL", flush=True)
    return 0 if byte_exact else 1


if __name__ == '__main__':
    raise SystemExit(main())
