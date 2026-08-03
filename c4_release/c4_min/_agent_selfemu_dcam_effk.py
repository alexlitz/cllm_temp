"""eff_K ceiling: at a fixed large working-set (n_store), how large a K-batch can the
OFF (softmax-over-stores) path sustain before the O(K*n_store) score matrix OOMs,
vs the ON (O(1) gather) path?  The eff_K lift = fewer forwards = the self-emu wall cut.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["C4_POS_SPARSE"] = "1"
import torch
from . import isa
from .compact_alloc import build_compact_sparse_streaming
from .pf_kbatch import KBatchBoundedRunner
from .selfemu_direct_cam import build_direct_cam_table
from .nibble_pure_forward import N_ROLES


class _RR:
    __slots__ = ("head", "value")
    def __init__(self, h, v): self.head, self.value = h, v


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n-store", type=int, default=16384)
    a = ap.parse_args(argv)
    dev = a.device
    model, L, _ = build_compact_sparse_streaming(code_size=64, compute_mode="dense_kernel")
    model.to(dev); model.materialize_dense(dev)
    D = model.embed.shape[1]
    print("[built] dim=%d n_store=%d" % (D, a.n_store), flush=True)
    runner = KBatchBoundedRunner(model, L, window=64)
    cuda = dev.startswith("cuda")

    print("\n  %-6s %14s %14s %10s" % ("K", "OFF ms/fwd", "ON ms/fwd", "OFF->ON"),
          flush=True)
    for K in (8, 32, 128, 512, 1024):
        pack = 2
        S = a.n_store * pack + K * 30 + 200
        try:
            x = torch.zeros(1, S, D, device=dev, dtype=model.embed.dtype)
        except torch.cuda.OutOfMemoryError:
            print("  K=%-4d input alloc OOM (S=%d)" % (K, S), flush=True)
            torch.cuda.empty_cache(); continue
        x[0, :, int(L.ONE)] = 1.0
        for r in range(1, a.n_store * pack, pack):
            x[0, r, int(L.IS_STORE)] = 1.0
        q_idxs = [S - 1 - 30 * (K - 1 - j) for j in range(K)]
        q_idxs = [q for q in q_idxs if q >= 0]
        for q in q_idxs:
            for role in range(N_ROLES):
                x[0, q, L.ROLE + role] = 1.0
        ops = [isa.ADD] * len(q_idxs)

        def _t(fn, n=10, w=3):
            for _ in range(w): fn()
            if cuda: torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n): fn()
            if cuda: torch.cuda.synchronize()
            return (time.time() - t0) / n * 1e3

        # OFF
        runner.arm_direct_cam(None)
        try:
            ms_off = _t(lambda: runner.forward_span(x, ops, q_idxs))
            off_s = "%.2f" % ms_off
        except torch.cuda.OutOfMemoryError:
            off_s = "OOM"; torch.cuda.empty_cache()
        # ON
        tbl = build_direct_cam_table({q: [_RR("pop", (q * 7) & 0xFF)] for q in q_idxs})
        runner.arm_direct_cam(tbl)
        try:
            ms_on = _t(lambda: runner.forward_span(x, ops, q_idxs))
            on_s = "%.2f" % ms_on
        except torch.cuda.OutOfMemoryError:
            on_s = "OOM"; torch.cuda.empty_cache()
        runner.arm_direct_cam(None)
        spd = ("%.1fx" % (float(off_s) / float(on_s))) if (off_s != "OOM" and on_s != "OOM") \
              else ("OFF-OOM" if off_s == "OOM" and on_s != "OOM" else "-")
        print("  %-6d %14s %14s %10s" % (len(q_idxs), off_s, on_s, spd), flush=True)
        del x
        if cuda: torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
