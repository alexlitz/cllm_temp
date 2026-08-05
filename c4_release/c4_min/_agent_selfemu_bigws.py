"""LARGE working-set self-emu measurement: the INVERTED doom case.

Doom = tiny program, huge frame.  Self-emu = ~262k persistent-heap KV/step (the
emulated transformer's weights+activations live in the store log).  This sweeps the
store working set n_store from small up toward ~262k with the SELF-EMU fast path
(C4_SELFEMU_DIRECT_CAM) ON vs OFF, tracks VRAM peak, and answers directly:

  * does the O(1) direct-CAM gather still scale at 262k stores (or does the
    softmax-over-stores OFF path OOM long before)?
  * does the single-dispatch / megakernel schedule BUILD over 262k KV fit VRAM,
    or blow up?  (Report the build tensor size + peak.)

Byte-exactness of the fast path is proven separately by the end-to-end runner;
here we measure THROUGHPUT + VRAM at the realistic working-set scale.
"""
from __future__ import annotations
import os, time
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


def _time(fn, n=8, w=3, cuda=True):
    for _ in range(w): fn()
    if cuda: torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): fn()
    if cuda: torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--stores", default="1024,4096,16384,65536,131072,262144")
    a = ap.parse_args(argv)
    dev = a.device
    cuda = dev.startswith("cuda")
    model, L, _ = build_compact_sparse_streaming(code_size=64, compute_mode="dense_kernel")
    model.to(dev); model.materialize_dense(dev)
    D = model.embed.shape[1]
    print("[built] dim=%d K=%d dev=%s" % (D, a.K, dev), flush=True)
    runner = KBatchBoundedRunner(model, L, window=64)
    K = a.K

    print("\n  %-9s %13s %13s %13s %10s %12s %12s" %
          ("n_store", "OFF ms/fwd", "ON ms/fwd", "ON+graph", "OFF->ON", "peakVRAM_GB", "in_GB"),
          flush=True)
    for n_store in (int(x) for x in a.stores.split(",") if x):
        pack = 2
        S = n_store * pack + K * 30 + 200
        in_gb = S * D * model.embed.element_size() / 1e9
        try:
            x = torch.zeros(1, S, D, device=dev, dtype=model.embed.dtype)
        except torch.cuda.OutOfMemoryError:
            print("  %-9d input alloc OOM (S=%d, %.1f GB)" % (n_store, S, in_gb), flush=True)
            torch.cuda.empty_cache(); continue
        x[0, :, int(L.ONE)] = 1.0
        for r in range(1, n_store * pack, pack):
            x[0, r, int(L.IS_STORE)] = 1.0
        q_idxs = [S - 1 - 30 * (K - 1 - j) for j in range(K)]
        q_idxs = [q for q in q_idxs if q >= 0]
        for q in q_idxs:
            for role in range(N_ROLES):
                x[0, q, L.ROLE + role] = 1.0
        ops = [isa.ADD] * len(q_idxs)

        if cuda: torch.cuda.reset_peak_memory_stats(dev)
        # OFF (softmax over stores, O(n_store))
        runner.arm_direct_cam(None)
        try:
            ms_off = _time(lambda: runner.forward_span(x, ops, q_idxs), cuda=cuda)
            off_s = "%.2f" % ms_off
        except torch.cuda.OutOfMemoryError:
            off_s = "OOM"; torch.cuda.empty_cache()
        # ON (O(1) gather)
        tbl = build_direct_cam_table({q: [_RR("pop", (q * 7) & 0xFF)] for q in q_idxs})
        runner.arm_direct_cam(tbl)
        try:
            ms_on = _time(lambda: runner.forward_span(x, ops, q_idxs), cuda=cuda)
            on_s = "%.2f" % ms_on
        except torch.cuda.OutOfMemoryError:
            on_s = "OOM"; torch.cuda.empty_cache()
        # ON + GRAPHED FFN megakernel tail (direct-CAM + graphed passthrough-FFN chain).
        # The graph is over the [1,K,D] query rows only (S-independent); measure that
        # composing direct-CAM (reads) + graphed FFN (per-step math) holds at scale.
        runner.arm_direct_cam(tbl)
        try:
            ms_g = _time(lambda: runner.forward_span_graphed(x, ops, q_idxs), cuda=cuda)
            g_s = "%.2f" % ms_g
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            g_s = "OOM/err"; torch.cuda.empty_cache()
        runner.arm_direct_cam(None)
        peak = torch.cuda.max_memory_allocated(dev) / 1e9 if cuda else 0.0
        spd = ("%.1fx" % (float(off_s) / float(on_s))) if (off_s != "OOM" and on_s != "OOM") \
              else ("OFF-OOM" if off_s == "OOM" and on_s != "OOM" else "-")
        print("  %-9d %13s %13s %13s %10s %12.2f %12.2f"
              % (n_store, off_s, on_s, g_s, spd, peak, in_gb), flush=True)
        del x
        if cuda: torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
