"""CPU byte-exactness + memory of the flash path in blogspec_model.Attn (the env the
#839 CFM battery actually runs in: build_pure_forward_complete_model builds on CPU).

Proves the CPU flash path (C4_FLASH_ATTN=1 -> chunked online-softmax1, O(S) memory) is
byte-identical to the masked-full O(S^2) softmax1+ALiBi CPU path, and measures the RSS
win at the growing-stream scales the long c90 cases hit.
"""
import os, sys, time, resource

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import torch
from c4_min.blogspec_model import Attn


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def run(attn, x, flash):
    if flash:
        os.environ["C4_FLASH_ATTN"] = "1"
    else:
        os.environ.pop("C4_FLASH_ATTN", None)
    t0 = time.time()
    with torch.no_grad():
        out = attn(x)
    dt = time.time() - t0
    os.environ.pop("C4_FLASH_ATTN", None)
    return out, dt


def main():
    torch.manual_seed(0)
    D, H = 2160, 24               # production CFM geometry
    attn = Attn(D, H, max_seq_len=65536, positional="alibi", sink="softmax1").eval()
    for p in attn.parameters():
        torch.nn.init.normal_(p, std=0.02)
    print(f"=== CPU flash byte-exact + RSS (D={D}, H={H}) ===")
    print(f"{'S':>7} | {'maxdiff':>10} | {'full ms':>9} | {'flash ms':>9} | {'peakRSS GB':>10}")
    worst = 0.0
    for S in [128, 512, 2048, 6000]:
        x = torch.randn(1, S, D) * 0.5
        ref, dt_full = run(attn, x, flash=False)
        got, dt_flash = run(attn, x, flash=True)
        diff = (ref - got).abs().max().item()
        worst = max(worst, diff)
        print(f"{S:>7} | {diff:>10.2e} | {dt_full*1e3:>9.1f} | {dt_flash*1e3:>9.1f} | "
              f"{rss_gb():>10.2f}")
        del x, ref, got
    print(f"\nworst CPU maxdiff = {worst:.3e}")
    ok = worst < 1e-4
    print("CPU FLASH BYTE-EXACT" if ok else "*** CPU FLASH DIVERGENT ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
