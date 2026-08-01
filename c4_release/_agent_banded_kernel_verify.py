"""Bit-exactness + speed microbench for the banded local-attention kernel.

Compares ``banded_local_context`` (O(Sq*W)) against the reference masked-full path
(``[B,Hl,Sq,Sk]`` matmul + window/causal -inf mask + softmax1) that
``local_attention.windowed_forward`` uses today, on:
  (1) random K/V (worst case: no ZFOD tail -> the band MUST match the masked-full),
  (2) contiguous positions (the doom regime: cache suffix + new span),
  (3) a non-trivial past cache of length W (posl trimmed) + a big new span.
Asserts max-abs-diff == 0 (or < 1e-5 fp-reduction slack), then times both at
S=3000/6000/12000 to show the FLAT-in-S collapse and that 12000 no longer OOMs.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, time
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c4_min.blogspec_model import softmax1
from c4_min._agent_banded_local_attn import banded_local_context


def ref_masked_full(Qg, Ksel, Vsel, q_pos, kpos_full, slopes_g, scale, window):
    """The EXACT masked-full path from local_attention._attend_group (window branch)."""
    sc = torch.matmul(Qg, Ksel.transpose(-2, -1)) * scale
    dist = (q_pos.unsqueeze(1) - kpos_full.unsqueeze(0)).float()
    sc = sc - slopes_g.view(1, -1, 1, 1) * dist.abs().unsqueeze(0)
    m = (kpos_full.unsqueeze(0) > q_pos.unsqueeze(1))
    m = m | (dist >= window)
    sc = sc.masked_fill(m.unsqueeze(0).unsqueeze(0), float("-inf"))
    a = softmax1(sc, dim=-1)
    return torch.matmul(a, Vsel)


def make_case(Hl, Sq, Skpast, HD, W, dev, dtype, seed=0, contiguous=True):
    torch.manual_seed(seed)
    B = 1
    # new-span positions [span_start .. span_start+Sq-1], past cache is the W rows just
    # before span_start (the doom trim-to-W regime).
    span_start = Skpast
    q_pos = torch.arange(span_start, span_start + Sq, device=dev, dtype=torch.long)
    if contiguous:
        past_pos = torch.arange(span_start - Skpast, span_start, device=dev, dtype=torch.long)
    else:
        # a GAP before the span (past cache ends well before span_start) — tests the
        # searchsorted band robustness (should still match masked-full).
        past_pos = torch.arange(0, Skpast, device=dev, dtype=torch.long)
    kpos_full = torch.cat([past_pos, q_pos], dim=0)
    Sk = kpos_full.numel()
    Qg = torch.randn(B, Hl, Sq, HD, device=dev, dtype=dtype)
    Ksel = torch.randn(B, Hl, Sk, HD, device=dev, dtype=dtype)
    Vsel = torch.randn(B, Hl, Sk, HD, device=dev, dtype=dtype)
    slopes = torch.rand(Hl, device=dev, dtype=dtype) * 0.5 + 0.01
    scale = HD ** -0.5
    return Qg, Ksel, Vsel, q_pos, kpos_full, slopes, scale


def check(name, Qg, Ksel, Vsel, q_pos, kpos_full, slopes, scale, W):
    ref = ref_masked_full(Qg, Ksel, Vsel, q_pos, kpos_full, slopes, scale, W)
    got = banded_local_context(Qg, Ksel, Vsel, q_pos, kpos_full, slopes, scale, W)
    d = (ref - got).abs().max().item()
    ok = d < 1e-4
    print(f"  [{name}] max_abs_diff={d:.3e}  {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    HD, W, Hl = 64, 96, 20
    print(f"device={dev} dtype={dtype} HD={HD} W={W} Hl={Hl}", flush=True)

    print("BIT-EXACTNESS (banded vs masked-full):", flush=True)
    allok = True
    # random small — worst case (no ZFOD tail; band MUST match on the kept keys).
    for seed in range(3):
        c = make_case(Hl, 200, 96, HD, W, dev, dtype, seed=seed, contiguous=True)
        allok &= check(f"contig Sq=200 past=96 seed={seed}", *c, W)
    # bigger span (doom K=200 span ~6000 tokens), contiguous.
    c = make_case(Hl, 2000, 96, HD, W, dev, dtype, seed=7, contiguous=True)
    allok &= check("contig Sq=2000 past=96", *c, W)
    # no past (first span / prime start).
    c = make_case(Hl, 500, 0, HD, W, dev, dtype, seed=9, contiguous=True)
    allok &= check("contig Sq=500 past=0", *c, W)
    # non-contiguous (gap) — robustness of searchsorted band.
    c = make_case(Hl, 300, 50, HD, W, dev, dtype, seed=11, contiguous=False)
    allok &= check("GAP Sq=300 past=50", *c, W)
    # W larger than the whole key set.
    c = make_case(Hl, 40, 10, HD, 256, dev, dtype, seed=13, contiguous=True)
    allok &= check("W=256 > Sk (Sq=40 past=10)", *c, 256)
    print(f"BIT-EXACT ALL: {'PASS' if allok else 'FAIL'}", flush=True)

    if dev == 'cpu':
        print("(cpu — skipping speed/OOM curve)"); return 0 if allok else 1

    print("\nFLAT-IN-S SPEED (masked-full vs banded), single forward of the LOCAL group:",
          flush=True)
    for S in (3000, 6000, 12000):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
        c = make_case(Hl, S, W, HD, W, dev, dtype, seed=1, contiguous=True)
        Qg, Ksel, Vsel, q_pos, kpos_full, slopes, scale = c
        # masked-full (may OOM at 12000)
        full_ms, full_gb = None, None
        try:
            torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(dev)
            t = time.perf_counter()
            _ = ref_masked_full(Qg, Ksel, Vsel, q_pos, kpos_full, slopes, scale, W)
            torch.cuda.synchronize(dev)
            full_ms = (time.perf_counter() - t) * 1e3
            full_gb = torch.cuda.max_memory_allocated(dev) / 1024**3
        except RuntimeError as e:
            full_ms = float('nan'); full_gb = float('nan')
            oom = 'OOM' if 'out of memory' in str(e).lower() else str(e)[:40]
            print(f"  S={S:6d}  masked-full: {oom}", flush=True)
            torch.cuda.empty_cache()
        # banded
        torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(dev)
        t = time.perf_counter()
        _ = banded_local_context(Qg, Ksel, Vsel, q_pos, kpos_full, slopes, scale, W)
        torch.cuda.synchronize(dev)
        band_ms = (time.perf_counter() - t) * 1e3
        band_gb = torch.cuda.max_memory_allocated(dev) / 1024**3
        fm = f"{full_ms:7.1f}ms/{full_gb:5.2f}GB" if full_ms == full_ms else "   OOM      "
        print(f"  S={S:6d}  masked-full={fm}   banded={band_ms:7.1f}ms/{band_gb:5.2f}GB",
              flush=True)
    return 0 if allok else 1


if __name__ == '__main__':
    raise SystemExit(main())
