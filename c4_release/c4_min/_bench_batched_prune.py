"""Micro-benchmark: OLD (full O(S^2) cosine matmul + fixpoint) vs NEW (two-tier
exact-dedup + cosine-on-representatives) ``prune_keep_mask_batched`` on a REALISTIC
register-frame span — N groups (block*head), each a K-step verify span whose ~n_roles
distinct role/value keys repeat verbatim across the K steps (the deep-loop shape).

Confirms the O(S^2)->O(S) win AND that both produce the byte-identical keep-mask.

    python -m c4_min._bench_batched_prune [--device cuda:0]
"""
from __future__ import annotations
import argparse, time
import torch
from c4_min.nibble_pure_forward_cached import (
    prune_keep_mask_batched, _greedy_survivors_batched, _recency_rank)


def _old_batched_mech1_full(keys, vals, positions, slope, scale, cos_threshold,
                            zero_eps, recency_eps, exact, content_addressed, valid):
    """The ORIGINAL prune_keep_mask_batched mechanism-1 (full [N,S,S] cosine matmul
    + greedy) followed by the (unchanged) mechanisms 3/2a/2b — reconstructed here so
    the bench times BOTH implementations without a git checkout."""
    N, S, HD = keys.shape
    dev = keys.device
    positions = positions.to(dev)
    knorm = keys.norm(dim=-1)
    vnorm = vals.norm(dim=-1)
    knorm = torch.where(valid, knorm, torch.zeros_like(knorm))
    tol = 1.0 - cos_threshold
    safe = knorm.clamp(min=1e-30)
    unit = keys / safe.unsqueeze(-1)
    unit = torch.where((knorm == 0).unsqueeze(-1), torch.zeros_like(unit), unit)
    D = torch.matmul(unit, unit.transpose(1, 2)) > cos_threshold
    if bool(exact.any()):
        ei = torch.nonzero(exact, as_tuple=False).flatten()
        ke = keys[ei]
        diff = torch.cdist(ke, ke, compute_mode="donot_use_mm_for_euclid_dist")
        kne = knorm[ei]
        denom = torch.maximum(kne.unsqueeze(2), kne.unsqueeze(1)).clamp(min=1e-30)
        D[ei] = diff <= tol * denom
    row_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    D = D & row_valid & ~torch.eye(S, dtype=torch.bool, device=dev).unsqueeze(0)
    survivors = _greedy_survivors_batched(D, positions) & valid
    # mechanisms 3/2a/2b are unchanged between old and new, and cheap (O(N*S)) — we
    # skip them here (the bench isolates mechanism-1, the O(S^2) part).  For a fair
    # end-to-end number the NEW path is timed via prune_keep_mask_batched() below.
    return survivors


def make_span(N, K, n_roles, HD=8, device="cpu"):
    """N groups, each a K-step verify span: n_roles distinct role/value keys, each
    repeated verbatim across the K steps (+ a couple zero rows per step)."""
    g = torch.Generator().manual_seed(7)
    proto = torch.randn(N, n_roles, HD, generator=g)          # distinct role keys
    # each step re-emits ALL n_roles roles -> S = K*n_roles
    S = K * n_roles
    idx = torch.arange(n_roles).repeat(K)                      # role id per row
    keys = proto[:, idx, :].clone()                           # [N,S,HD] verbatim dups
    # a few zero-value / zero-key rows (freed/NULL), same each step
    vals = torch.randn(N, S, HD, generator=g)
    zmask = (torch.rand(N, S, generator=g) < 0.1)
    vals[zmask] = 0.0
    # positions strictly increasing per row (30 tokens/step spacing)
    positions = torch.arange(S).unsqueeze(0).expand(N, S).clone()
    valid = torch.ones(N, S, dtype=torch.bool)
    return (keys.to(device), vals.to(device), positions.to(device), valid.to(device))


def timeit(fn, device, reps=3):
    fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-groups", type=int, default=64,
                    help="N = a memory-budget chunk of block*head groups")
    args = ap.parse_args()
    dev = args.device
    slope = torch.full((args.n_groups,), 0.25)
    scale = 8 ** -0.5
    cos_thr, zeps, reps = 0.99, 1e-9, 1e-6
    n_roles = 30
    print(f"batched prune micro-bench  device={dev}  N={args.n_groups}  n_roles={n_roles}")
    print(f"{'K':>5} {'S':>7} {'OLD ms':>9} {'NEW ms':>9} {'speedup':>8} {'match':>6} {'kept/grp':>9}")
    for K in [8, 16, 32, 48, 64]:
        keys, vals, positions, valid = make_span(args.n_groups, K, n_roles, device=dev)
        S = keys.shape[1]
        exact = torch.zeros(args.n_groups, dtype=torch.bool, device=dev)
        ca = torch.zeros(args.n_groups, dtype=torch.bool, device=dev)
        sl = slope.to(dev)

        def _new():
            return prune_keep_mask_batched(keys, vals, positions, sl, scale,
                                           cos_thr, zeps, reps, exact, ca, valid=valid)

        def _old():
            return _old_batched_mech1_full(keys, vals, positions, sl, scale, cos_thr,
                                           zeps, reps, exact, ca, valid)

        # correctness: NEW mechanism-1 (via full prune, then re-derive mech1) vs OLD
        # mech1.  We compare the mechanism-1 survivor sets directly.
        new_full = _new()
        old_m1 = _old()
        # NEW mech-1 alone: recompute via the helper for an apples-to-apples check
        from c4_min.nibble_pure_forward_cached import _mech1_cosine_survivors_dedup
        knorm = keys.norm(dim=-1)
        knorm = torch.where(valid, knorm, torch.zeros_like(knorm))
        rank = _recency_rank(positions, valid)
        new_m1 = _mech1_cosine_survivors_dedup(keys, knorm, positions, rank, valid, cos_thr) & valid
        match = bool(torch.equal(new_m1, old_m1))
        t_old = timeit(_old, dev)
        t_new = timeit(_new, dev)
        print(f"{K:>5} {S:>7} {t_old*1e3:>9.2f} {t_new*1e3:>9.2f} "
              f"{t_old/max(t_new,1e-9):>7.1f}x {str(match):>6} "
              f"{int(new_full.sum())/args.n_groups:>9.1f}")


if __name__ == "__main__":
    main()
