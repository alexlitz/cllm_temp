"""PROFILE the KV eviction prune (issue: prune is the fast-path dominant cost).

Two probes:
  (A) synthetic O(S^2) confirmation + exact-dup vs cosine-near-dup fraction over a
      realistic live-heap cache (the shape make_live_heap_cache builds), sweeping S;
  (B) real deep nested-loop run with the ACTUAL per-block/per-head prune instrumented
      — measures, per prune, the total wall, and the fraction of dropped rows that are
      EXACT-DUPLICATE key frames (evictable by a hash) vs genuine cosine-near-dups.

Usage:
    python -m c4_min._prof_prune synth  [--device cuda:1]
    python -m c4_min._prof_prune real   [--device cuda:1] [--outer 6 --inner 100]
"""
from __future__ import annotations
import argparse, time
import torch


def synth(args):
    from c4_min.nibble_pure_forward_cached import prune_keep_mask_head
    from c4_min.bench_kv_evict import make_live_heap_cache
    dev = args.device
    slope, scale = 0.25, 8 ** -0.5
    cos_thr, zeps, reps = 0.99, 1e-9, 1e-6
    sizes = [500, 1000, 2000, 4000, 8000, 16000]
    metric = args.metric
    ca = (metric == "exact")
    print(f"SYNTH prune profile  device={dev}  metric={metric}")
    print(f"{'S':>7} {'kept':>7} {'dropped':>8} {'prune ms':>9} {'dupmat ms':>10} "
          f"{'fixpt ms':>9} {'passes':>7} {'exact%':>7} {'nearcos%':>9}")
    for S in sizes:
        keys, vals, positions = make_live_heap_cache(S, device=dev, metric=metric)

        def _p():
            return prune_keep_mask_head(keys, vals, positions, slope, scale,
                                        cos_thr, zeps, reps, dup_metric=metric,
                                        content_addressed=ca)
        _p()
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); _p()
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        t_prune = min(ts)
        surv = _p()
        knorm = keys.norm(dim=-1)
        if metric == "exact":
            tol = 1.0 - cos_thr

            def _dupmat():
                diff = torch.cdist(keys.unsqueeze(0), keys.unsqueeze(0),
                                   compute_mode="donot_use_mm_for_euclid_dist").squeeze(0)
                denom = torch.maximum(knorm.unsqueeze(1), knorm.unsqueeze(0)).clamp(min=1e-30)
                D = diff <= tol * denom
                D.fill_diagonal_(False)
                return D
        else:
            safe = knorm.clamp(min=1e-30)
            unit = keys / safe.unsqueeze(-1)
            unit[knorm == 0] = 0.0

            def _dupmat():
                D = (unit @ unit.t()) > cos_thr
                D.fill_diagonal_(False)
                return D
        _dupmat()
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        tm = []
        for _ in range(3):
            t0 = time.perf_counter(); D = _dupmat()
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            tm.append(time.perf_counter() - t0)
        t_dupmat = min(tm)
        idx = torch.arange(S, device=dev)
        order_key = positions.to(torch.int64) * S - idx
        rank = torch.argsort(torch.argsort(order_key, descending=True))
        newer = D & (rank.unsqueeze(1) > rank.unsqueeze(0))

        def _fix():
            keep = torch.ones(S, dtype=torch.bool, device=dev)
            npass = 0
            for _ in range(S + 1):
                npass += 1
                drop = (newer & keep.unsqueeze(0)).any(dim=1)
                new_keep = ~drop
                if bool(torch.equal(new_keep, keep)):
                    return new_keep, npass
                keep = new_keep
            return keep, npass
        _fix()
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        tf = []
        passes = 0
        for _ in range(3):
            t0 = time.perf_counter(); _, passes = _fix()
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            tf.append(time.perf_counter() - t0)
        t_fix = min(tf)
        # exact-dup vs near-cosine breakdown of the DROPPED rows
        dropped_mask = ~surv
        keep_keys = keys[surv]
        exact = near = 0
        if keep_keys.shape[0] > 0:
            for e in torch.nonzero(dropped_mask, as_tuple=False).flatten().tolist():
                d = (keep_keys - keys[e]).norm(dim=-1)
                if bool((d == 0.0).any()):
                    exact += 1
                else:
                    near += 1
        nd = exact + near
        print(f"{S:>7} {int(surv.sum()):>7} {nd:>8} {t_prune*1e3:>9.2f} "
              f"{t_dupmat*1e3:>10.2f} {t_fix*1e3:>9.2f} {passes:>7} "
              f"{(100*exact/max(nd,1)):>6.1f}% {(100*near/max(nd,1)):>8.1f}%")


def real(args):
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    _PF.SP_INIT = 0xFC
    _PFC.SP_INIT = 0xFC
    import c4_min.nibble_pure_forward_cached as PFCa
    PFCa.SP_INIT = 0xFC
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.pf_speculative import draft_pf_program, verify_blocks
    from c4_min.bench_fast_path import build_nested

    code, expected, data, label = build_nested(args.outer, args.inner)
    device = args.device
    draft = draft_pf_program(code, max_steps=5_000_000, mask=0xFFFFFFFF)
    print(f"=== {label} ===  steps={draft.step_count} halted={draft.halted}", flush=True)
    sparse, L, _ = build_lib_model_streaming(
        code_size=max(len(code) + 2, 64), recurrent_divmod=True, addr32=True,
        compute_mode="dense_kernel")
    if device != "cpu":
        sparse = sparse.to(device)

    # Instrument evict_all_blocks_fused (the fast-path prune) for WALL, and
    # prune_keep_mask_batched for the exact-dup vs near-cosine breakdown of the
    # DROPPED near-dup rows (mechanism-1) per group.
    orig_fused = PFCa.evict_all_blocks_fused
    orig_batched = PFCa.prune_keep_mask_batched
    st = {"n": 0, "wall": 0.0, "maxS": 0, "dropped": 0,
          "exact": 0, "near": 0, "batched_calls": 0, "cos_groups": 0,
          "exact_groups": 0, "total_groups": 0}

    def wrap_fused(caches, cos_threshold, zero_eps, recency_eps,
                   protect_positions=None):
        dev = None
        for c in caches:
            if c.K is not None:
                dev = c.K.device
                st["maxS"] = max(st["maxS"], int(c.pos.shape[0]))
        if dev is not None and dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        res = orig_fused(caches, cos_threshold, zero_eps, recency_eps,
                         protect_positions=protect_positions)
        if dev is not None and dev.type == "cuda":
            torch.cuda.synchronize(dev)
        st["wall"] += time.perf_counter() - t0
        st["n"] += 1
        return res

    st["rows_total"] = 0
    st["rows_unique"] = 0

    def wrap_batched(keys, vals, positions, slope, scale, cos_threshold,
                     zero_eps, recency_eps, exact, content_addressed, valid=None):
        st["batched_calls"] += 1
        N = keys.shape[0]
        st["total_groups"] += N
        st["exact_groups"] += int(exact.sum())
        st["cos_groups"] += N - int(exact.sum())
        surv = orig_batched(keys, vals, positions, slope, scale, cos_threshold,
                            zero_eps, recency_eps, exact, content_addressed, valid=valid)
        # EXACT-DUPLICATE COLLAPSE measurement: for the COSINE (register-marker)
        # groups, how many of the valid rows are exact-duplicate keys (evictable by
        # the O(S) hash tier) vs distinct keys that the O(R^2) cosine must still see?
        # rows_total = valid cosine rows; rows_unique = distinct NON-zero keys among
        # them (zero-key rows kept singleton).  1 - unique/total = the exact-dup
        # fraction the two-tier hash collapses for free.  Sample a few cos groups.
        v = valid if valid is not None else torch.ones_like(surv)
        cos_idx = torch.nonzero(~exact, as_tuple=False).flatten().tolist()
        for n in cos_idx[:8]:
            k = keys[n]
            real = v[n]
            kn = k.norm(dim=-1)
            realnz = real & (kn > 0)
            rows = k[realnz]
            if rows.shape[0] == 0:
                continue
            st["rows_total"] += rows.shape[0]
            st["rows_unique"] += int(torch.unique(rows, dim=0).shape[0])
        return surv

    PFCa.evict_all_blocks_fused = wrap_fused
    PFCa.prune_keep_mask_batched = wrap_batched
    # pf_speculative imported evict_all_blocks_fused by NAME at module load, so patch
    # its binding too.
    import c4_min.pf_speculative as PFS
    PFS.evict_all_blocks_fused = wrap_fused

    fast_stats = {}
    t = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=args.block_steps,
                       device=device, evict=True, prune_interval=args.prune_interval,
                       mask=0xFFFFFFFF, stats=fast_stats, fast=True,
                       collect_out=None, block_moe=args.block_moe)
    t_fast = time.time() - t
    PFCa.evict_all_blocks_fused = orig_fused
    PFCa.prune_keep_mask_batched = orig_batched
    PFS.evict_all_blocks_fused = orig_fused

    fps = t_fast / max(draft.step_count, 1)
    print(f"\nFAST: {vr.forwards} forwards in {t_fast:.1f}s -> {fps*1000:.1f} ms/step "
          f"matched={vr.all_matched} evicted={fast_stats.get('total_evicted')} "
          f"max_cache={fast_stats.get('max_cache_size')}", flush=True)
    rt = st.get("rows_total", 0)
    ru = st.get("rows_unique", 0)
    print(f"\nPRUNE PROFILE:")
    print(f"  evict_all_blocks_fused rounds : {st['n']}")
    print(f"  prune_keep_mask_batched calls : {st['batched_calls']}")
    print(f"  groups (block*head)           : total={st['total_groups']}  "
          f"cosine={st['cos_groups']}  exact={st['exact_groups']}")
    print(f"  total prune wall              : {st['wall']:.2f}s "
          f"= {100*st['wall']/t_fast:.1f}% of fast wall")
    print(f"  max cache S at prune          : {st['maxS']}")
    print(f"  cosine rows (sampled)         : total={rt}  distinct-nonzero-keys={ru}  "
          f"exact-dup fraction={100*(1-ru/max(rt,1)):.1f}%")
    print(f"  ms/step if prune==0           : "
          f"{(t_fast-st['wall'])/max(draft.step_count,1)*1000:.1f} ms/step")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("synth")
    ps.add_argument("--device", default="cpu")
    ps.add_argument("--metric", default="cosine", choices=["cosine", "exact"])
    pr = sub.add_parser("real")
    pr.add_argument("--device", default="cpu")
    pr.add_argument("--outer", type=int, default=6)
    pr.add_argument("--inner", type=int, default=100)
    pr.add_argument("--block-steps", type=int, default=64)
    pr.add_argument("--prune-interval", type=int, default=60)
    pr.add_argument("--block-moe", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "synth":
        synth(args)
    else:
        real(args)


if __name__ == "__main__":
    raise SystemExit(main())
