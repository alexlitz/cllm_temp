"""SHAPE-accurate FULL-megablock (L9, I=160465) FFN GEMM cost: block-MoE vs cond.

The FULL dense VM is 51.6 GB and unsafe to build under RAM contention, but the
block-MoE-vs-conditional question on the megablock is a PURE KERNEL-COST question
that depends only on the SHAPE (I, H, M), not the weight VALUES: a
``down(silu(gate·x)·up·x)`` GEMM of ``[M,H]·[H,I]->[M,I]`` costs the same whether
the weights are the real base-16 long-division table or random of the same shape.

So this bench builds ONE megablock-shaped FFN (I=160465, H, [I,H]/[H,I]) with
random weights on the GPU (~1.9 GB for one layer -- fits, no 54 GB CPU build) and
times the FFN GEMM under the three configs at spec batch sizes:

    dense      = 160465 units (== block-MoE on a DIVMOD step: it keeps the whole
                 layer, block granularity)
    block-MoE  = 160465 on a divmod step / 0 (skipped) on a non-divmod step
    cond(MOD)  = 508 firing units / cond(DIV) = 247 / cond(non-divmod) = 14

The ACTIVE-UNIT COUNTS (508 MOD / 247 DIV / 14 non-divmod / 160465 dense) are the
MEASURED per-op fractions from ``perlayer_conditional_sparse`` (FULL, I=160465); the
byte-identity of the pruning is proven separately by ``verify_conditional_decode``
on a subset that fits.  This bench answers ONLY: given those active counts, how much
faster is the fine-grained conditional GEMM than block-MoE's whole-megablock GEMM?

Run:
    python -m c4_min.bench_megablock_kernel --device cuda:0
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F


def _bench(fn, n, warmup, cuda):
    if cuda:
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


def _vram_ok(device, need_gb, headroom_gb=2.0):
    if not device.startswith("cuda"):
        return True
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return need_gb + headroom_gb < free / (1024 ** 3)


@torch.no_grad()
def _ffn_ms(H, k, device, M, n, warmup, cuda):
    """Time a k-unit FFN GEMM at M rows (random weights of the megablock shape)."""
    if k == 0:
        return 0.0
    # weights (3*k*H) + input (M*H) + THREE [M,k] intermediates (silu(gate·x),
    # up·x, their product) live simultaneously -> ~3*M*k peak, plus cuBLAS scratch.
    need_gb = (3 * k * H + M * H + 3 * M * k) * 4 / (1024 ** 3)
    if not _vram_ok(device, need_gb, headroom_gb=3.0):
        return float("nan")
    xn = torch.randn(M, H, device=device)
    gate = torch.randn(k, H, device=device)
    up = torch.randn(k, H, device=device)
    down = torch.randn(H, k, device=device)

    def fwd():
        return F.linear(F.silu(F.linear(xn, gate)) * F.linear(xn, up), down)
    ms = _bench(fwd, n, warmup, cuda)
    del xn, gate, up, down
    if cuda:
        torch.cuda.empty_cache()
    return ms


def run(device="cuda:0", H=1152, I=160465, S=7,
        batches=(512, 2048, 8192), n=20, warmup=5):
    cuda = device.startswith("cuda")
    print(f"# FULL megablock (L9) FFN GEMM cost: block-MoE vs conditional")
    print(f"# device={device} H={H} I(dense)={I} S={S} n={n} warmup={warmup}", flush=True)
    # measured per-op firing counts of the L9 megablock (perlayer_conditional_sparse).
    cases = {
        "MOD  (fires megablock)": 508,
        "DIV  (fires megablock)": 247,
        "MUL  (fires megablock)": 517,
        "non-divmod (megablock DEAD)": 14,
    }
    print("\n# on a DIVMOD step block-MoE runs the WHOLE megablock (I units); on a")
    print("# non-divmod step BOTH block-MoE and conditional SKIP it (0 vs 14 units).")
    print(f"\n  {'B':>6} {'M':>9} {'dense/blkMoE':>13} | "
          + " ".join(f"{name.split()[0]:>9}" for name in cases) + " |  best c/moe")
    for B in batches:
        M = B * S
        dense_ms = _ffn_ms(H, I, device, M, n, warmup, cuda)
        cond_ms = {name: _ffn_ms(H, k, device, M, n, warmup, cuda)
                   for name, k in cases.items()}
        dstr = f"{dense_ms:11.3f}m" if dense_ms == dense_ms else "    OOM    "
        cells = []
        for name, k in cases.items():
            cm = cond_ms[name]
            cells.append(f"{cm:8.4f}m" if cm == cm else "   OOM   ")
        # net win on a DIVMOD step = dense(blockMoE) / cond(MOD) (worst divmod case).
        mod_ms = cond_ms["MOD  (fires megablock)"]
        best = (dense_ms / mod_ms) if (dense_ms == dense_ms and mod_ms) else float("nan")
        print(f"  {B:>6} {M:>9} {dstr} | " + " ".join(cells) +
              f" | {best:8.1f}x", flush=True)
    print("\n# INTERPRETATION")
    print("#  * DIVMOD step: block-MoE runs the full 160465-unit megablock; conditional")
    print("#    runs ~508 (MOD)/247 (DIV) -> the 'best c/moe' column is the NET-NEW win")
    print("#    of fine-grained conditional OVER block-MoE (block-MoE gives 0 help")
    print("#    WITHIN an active layer -- it is block-granular).")
    print("#  * NON-DIVMOD step: block-MoE already skips the whole megablock (0 units),")
    print("#    so conditional's 14-unit block is a WASH there -- the net win on the")
    print("#    megablock is ZERO; the only conditional gain is on the small layers.")
    print("#  * dense OOMs at large M (the [M,160465] intermediate is 26-206 GB) -- the")
    print("#    conditional/MoE-skipped model is the only runnable form on 24 GB.")


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hidden", type=int, default=1152)
    ap.add_argument("--intermediate", type=int, default=160465)
    ap.add_argument("--seq", type=int, default=7)
    ap.add_argument("--batch", default="512,2048,8192")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(device=a.device, H=a.hidden, I=a.intermediate, S=a.seq,
        batches=tuple(int(b) for b in a.batch.split(",")),
        n=a.n, warmup=a.warmup)
