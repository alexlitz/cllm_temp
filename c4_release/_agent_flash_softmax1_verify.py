"""Bit-exactness + O(S)-memory microbench for the flash-softmax1 attention kernels.

Compares the byte-exact flash backends (``sdpa_flash_softmax1`` SDPA+LSE-rescale,
``triton_flash_softmax1`` custom online-softmax1) against the reference masked-full
softmax1 + ALiBi + causal path that ``SparseAttn.forward`` /
``local_attention._attend_group`` use today, on:
  (1) un-cached FULL (Sq==Sk, pos 0..S-1) — the SDPA top-left-causal regime,
  (2) CACHED (Sq<Sk, q_pos = the newest positions, bottom-right causal) — the
      self-emulation KV-cache regime SDPA is_causal can't express, Triton path,
  (3) random K/V worst case (no ZFOD tail) + a windowed (sliding) case.
Asserts max_abs_diff < 1e-4 (fp32 tiled-reduction noise), then times masked-full
vs flash at S=3000/6000/12000 to show the O(S) collapse + that 12000 no longer OOMs.

Golden 069cc32f unchanged (imports only; touches no stored weight).
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c4_min.blogspec_model import softmax1
from c4_min.flash_softmax1 import (sdpa_flash_softmax1, triton_flash_softmax1,
                                   flash_softmax1_context)


def ref_masked_full(Q, K, V, q_pos, k_pos, slopes, scale, window=None):
    """The EXACT masked-full softmax1 + ALiBi + causal reference (SparseAttn.forward /
    local_attention._attend_group)."""
    B, H, Sq, HD = Q.shape
    sc = torch.matmul(Q, K.transpose(-2, -1)) * scale
    dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).float()
    sc = sc - slopes.view(1, H, 1, 1) * dist.abs().unsqueeze(0)
    m = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))
    if window is not None:
        m = m | (dist >= window)
    sc = sc.masked_fill(m.unsqueeze(0).unsqueeze(0), float('-inf'))
    return torch.matmul(softmax1(sc, dim=-1), V)


def case(H, Sq, Skpast, HD, dev, seed=0, contiguous=True):
    torch.manual_seed(seed)
    B = 1
    span_start = Skpast
    q_pos = torch.arange(span_start, span_start + Sq, device=dev, dtype=torch.long)
    if contiguous:
        past_pos = torch.arange(span_start - Skpast, span_start, device=dev, dtype=torch.long)
    else:
        past_pos = torch.arange(0, Skpast, device=dev, dtype=torch.long)
    k_pos = torch.cat([past_pos, q_pos], dim=0)
    Sk = k_pos.numel()
    Q = torch.randn(B, H, Sq, HD, device=dev)
    K = torch.randn(B, H, Sk, HD, device=dev)
    V = torch.randn(B, H, Sk, HD, device=dev)
    slopes = torch.rand(H, device=dev) * 0.5 + 0.01
    return Q, K, V, q_pos, k_pos, slopes, HD ** -0.5


def check(name, got, ref):
    d = (ref - got).abs().max().item()
    ok = d < 1e-4
    print(f"  [{name}] max_abs_diff={d:.3e}  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    HD, H = 64, 4
    print(f"device={dev} HD={HD} H={H}", flush=True)
    allok = True

    print("BIT-EXACTNESS (flash vs masked-full softmax1):", flush=True)
    # --- un-cached FULL (Sq==Sk, pos 0..S-1): SDPA path ---
    for seed in range(3):
        Q, K, V, qp, kp, sl, sc = case(H, 128, 0, HD, dev, seed=seed)
        ref = ref_masked_full(Q, K, V, qp, kp, sl, sc)
        got = sdpa_flash_softmax1(Q, K, V, qp, kp, sl, sc)
        allok &= check(f"SDPA full Sq=Sk=128 seed={seed}", got, ref)
        got2 = flash_softmax1_context(Q, K, V, qp, kp, sl, sc, uncached_full=True)
        allok &= check(f"dispatch(uncached_full) seed={seed}", got2, ref)
    # --- CACHED (Sq<Sk, bottom-right causal): Triton path ---
    if dev == 'cuda:0':
        for seed in range(3):
            Q, K, V, qp, kp, sl, sc = case(H, 30, 200, HD, dev, seed=seed)
            ref = ref_masked_full(Q, K, V, qp, kp, sl, sc)
            got = triton_flash_softmax1(Q, K, V, qp, kp, sl, sc)
            allok &= check(f"Triton cached Sq=30 past=200 seed={seed}", got, ref)
        # bigger cache (self-emu long KV)
        Q, K, V, qp, kp, sl, sc = case(H, 64, 4000, HD, dev, seed=7)
        ref = ref_masked_full(Q, K, V, qp, kp, sl, sc)
        allok &= check("Triton cached Sq=64 past=4000", triton_flash_softmax1(Q, K, V, qp, kp, sl, sc), ref)
        # full via Triton (Sq==Sk) — the general path must also match
        Q, K, V, qp, kp, sl, sc = case(H, 200, 0, HD, dev, seed=9)
        ref = ref_masked_full(Q, K, V, qp, kp, sl, sc)
        allok &= check("Triton full Sq=Sk=200", triton_flash_softmax1(Q, K, V, qp, kp, sl, sc), ref)
        # windowed (sliding local) via Triton
        Q, K, V, qp, kp, sl, sc = case(H, 300, 96, HD, dev, seed=11)
        for W in (96, 28):
            ref = ref_masked_full(Q, K, V, qp, kp, sl, sc, window=W)
            allok &= check(f"Triton windowed W={W}", triton_flash_softmax1(Q, K, V, qp, kp, sl, sc, window=W), ref)
        # non-contiguous positions (gap) robustness
        Q, K, V, qp, kp, sl, sc = case(H, 50, 50, HD, dev, seed=13, contiguous=False)
        ref = ref_masked_full(Q, K, V, qp, kp, sl, sc)
        allok &= check("Triton GAP Sq=50 past=50", triton_flash_softmax1(Q, K, V, qp, kp, sl, sc), ref)
    print(f"BIT-EXACT ALL: {'PASS' if allok else 'FAIL'}", flush=True)

    if dev == 'cpu':
        print("(cpu — skipping GPU O(S) curve)"); return 0 if allok else 1

    # --- O(S) MEMORY + SPEED: masked-full (OOMs) vs flash (flat) ---
    print("\nO(S) MEMORY + SPEED (masked-full softmax1 vs flash), single fwd of the GLOBAL group:", flush=True)
    for S in (3000, 6000, 12000):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        # un-cached full case at position 0..S-1 (the priming / self-emu global regime)
        Q, K, V, qp, kp, sl, sc = case(H, S, 0, HD, dev, seed=1)
        full_ms = full_gb = float('nan')
        try:
            torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(dev)
            t = time.perf_counter()
            _ = ref_masked_full(Q, K, V, qp, kp, sl, sc)
            torch.cuda.synchronize(dev)
            full_ms = (time.perf_counter() - t) * 1e3
            full_gb = torch.cuda.max_memory_allocated(dev) / 1024**3
        except RuntimeError as e:
            oom = 'OOM' if 'out of memory' in str(e).lower() else str(e)[:30]
            print(f"  S={S:6d}  masked-full: {oom}", flush=True)
            torch.cuda.empty_cache()
        # SDPA flash
        torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(dev)
        t = time.perf_counter()
        _ = sdpa_flash_softmax1(Q, K, V, qp, kp, sl, sc)
        torch.cuda.synchronize(dev)
        sdpa_ms = (time.perf_counter() - t) * 1e3
        sdpa_gb = torch.cuda.max_memory_allocated(dev) / 1024**3
        # Triton flash (general)
        torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(dev)
        t = time.perf_counter()
        _ = triton_flash_softmax1(Q, K, V, qp, kp, sl, sc)
        torch.cuda.synchronize(dev)
        tri_ms = (time.perf_counter() - t) * 1e3
        tri_gb = torch.cuda.max_memory_allocated(dev) / 1024**3
        fm = f"{full_ms:7.1f}ms/{full_gb:5.2f}GB" if full_ms == full_ms else "   OOM       "
        print(f"  S={S:6d}  masked-full={fm}   SDPA-flash={sdpa_ms:6.1f}ms/{sdpa_gb:5.3f}GB"
              f"   Triton-flash={tri_ms:6.1f}ms/{tri_gb:5.3f}GB", flush=True)
    return 0 if allok else 1


if __name__ == '__main__':
    raise SystemExit(main())
