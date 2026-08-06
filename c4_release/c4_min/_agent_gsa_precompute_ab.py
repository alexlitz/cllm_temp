"""A/B the ACTUAL build_faithful_precompute (the pipelined build-thread value re-resolution the
faithful SD calls) with C4_GENUINE_STRUCTURED_ATTN ON vs OFF, on a REAL deep-loop draft.

This is the ONE function GSA changes (the value re-resolution algorithm); everything else in the
faithful path — the in-graph W_q address decode dispatch, the vectorized critical-path compare
(verify_faithful_fast) — is byte-identical.  So the throughput delta of GSA == the delta of THIS
precompute, measured here on the exact call the pipeline makes.  Memory-safe (CPU numpy + a small
model build; no doom snapshot, no 120K eviction).
"""
import warnings; warnings.filterwarnings('ignore')
import os, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_MEM_EFF', '500000.0')
import numpy as np
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC
from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program
from c4_min.faithful_single_dispatch import (build_faithful_plan, build_faithful_precompute,
                                             verify_faithful, verify_faithful_fast,
                                             _read_frame_of_step)
from c4_min._agent_faithful_sd_byteexact import _programs


def _time_precompute(draft, plan, ws, n, reps=30):
    build_faithful_precompute(draft, plan, ws, n)  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        for a in ("_faithful_precompute_cache",):
            if hasattr(draft, a): delattr(draft, a)
        build_faithful_precompute(draft, plan, ws, n)
    return (time.perf_counter() - t0) / reps


def main():
    model, L, _ = build_lib_model_streaming(code_size=64)
    print("[build] done", flush=True)
    P = _programs()
    prog, want = P["loop_sum300"]              # the 6315-step deep loop (heavy mem/pop/lev)
    code_isa = prog if isinstance(prog[0], isa.Instr) else isa.assemble(prog)
    draft = draft_pf_program(code_isa, max_steps=20000, mask=0xFFFFFFFF)
    n = draft.step_count
    plan = build_faithful_plan(model, L, code_isa, draft)
    ws = np.asarray(draft.win_starts[:n], dtype=np.int64)
    print(f"[draft] loop_sum300 steps={n} committed_stores={plan.store_frames.shape[0]}", flush=True)

    # A/B the precompute (the pipelined build-thread value re-resolution).
    os.environ.pop("C4_GENUINE_STRUCTURED_ATTN", None); os.environ.pop("C4_HASH_CAM", None)
    t_sd = _time_precompute(draft, plan, ws, n)
    os.environ["C4_HASH_CAM"] = "1"; os.environ.pop("C4_GENUINE_STRUCTURED_ATTN", None)
    if hasattr(draft, "_faithful_precompute_cache"): delattr(draft, "_faithful_precompute_cache")
    t_hash = _time_precompute(draft, plan, ws, n)
    os.environ.pop("C4_HASH_CAM", None); os.environ["C4_GENUINE_STRUCTURED_ATTN"] = "1"
    if hasattr(draft, "_faithful_precompute_cache"): delattr(draft, "_faithful_precompute_cache")
    t_gsa = _time_precompute(draft, plan, ws, n)
    os.environ.pop("C4_GENUINE_STRUCTURED_ATTN", None)

    print(f"\n  precompute (build-thread value re-resolution) us/step:", flush=True)
    print(f"    single-dispatch (searchsorted) : {t_sd/n*1e6:8.4f} us/step  (whole={t_sd*1e3:.3f} ms)", flush=True)
    print(f"    C4_HASH_CAM (O(1) atom)         : {t_hash/n*1e6:8.4f} us/step  (whole={t_hash*1e3:.3f} ms)", flush=True)
    print(f"    C4_GENUINE_STRUCTURED_ATTN      : {t_gsa/n*1e6:8.4f} us/step  (whole={t_gsa*1e3:.3f} ms)", flush=True)
    print(f"\n  The faithful DISPATCH floor is ~2.6 us/step (in-graph W_q addr decode; UNCHANGED "
          f"by GSA).  GSA build {t_gsa/n*1e6:.3f} us/step {'<' if t_gsa/n*1e6 < 2.6 else '>='} 2.6 "
          f"-> {'DISPATCH-BOUND (meets the fast ~2.6 us throughput)' if t_gsa/n*1e6 < 2.6 else 'BUILD-bound'}",
          flush=True)

    # byte-exact verdict with GSA on: precompute must produce the SAME accept verdict as the SD.
    os.environ["C4_GENUINE_STRUCTURED_ATTN"] = "1"
    if hasattr(draft, "_faithful_precompute_cache"): delattr(draft, "_faithful_precompute_cache")
    pre = build_faithful_precompute(draft, plan, ws, n)
    # model addresses == the draft addresses on a correct execution (the address check passes),
    # so feed the plan's own draft addr as the model addr (correct-execution stand-in for the
    # in-graph decode, which the byte-exact battery already validated end-to-end).
    from c4_min.direct_cam_batched import ADDR_BITS
    model_addrs = {}
    for (bi, kind), _ in [((7, "mem"), None)]:
        pass
    # build model_addrs from the plan's draft addr per kind (the correct decode).
    pos_to_step = {int(ws[s]): s for s in range(n)}
    for kind in ("mem", "pop", "lev", "uni"):
        ad = plan.draft_addr.get(kind, {})
        if not ad: continue
        arr = np.zeros(n, dtype=np.int64)
        for p, a in ad.items():
            s = pos_to_step.get(int(p))
            if s is not None: arr[s] = int(a)
        # block idx: use the head map — but for the verdict we only need per-kind arrays keyed
        # by (block, kind); reuse the plan's heads_by_block to find the block for this kind.
        blk = next((b for b, heads in plan.heads_by_block.items()
                    for (h, k, _c) in heads if k == kind), 7)
        model_addrs[(blk, kind)] = arr
    vd = verify_faithful_fast(pre, model_addrs)
    print(f"\n  [byte-exact verdict, GSA on, correct-execution model addrs] ok={vd.ok} "
          f"addr_checked={vd.n_addr_checked} value_checked={vd.n_value_checked} "
          f"-> {'ACCEPTS all (no false positive)' if vd.ok else 'FALSE POSITIVE at '+str(vd.first_bad_step)}",
          flush=True)
    os.environ.pop("C4_GENUINE_STRUCTURED_ATTN", None)
    import sys; sys.exit(0 if vd.ok and t_gsa/n*1e6 < 2.6 else 1)


if __name__ == "__main__":
    main()
