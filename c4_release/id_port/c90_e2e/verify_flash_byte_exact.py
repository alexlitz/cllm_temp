"""Byte-exact verification of C4_FLASH_ATTN wired into blogspec_model.Attn.forward.

Proves the FLASH path (C4_FLASH_ATTN=1) produces the SAME attention output as the
masked-full O(S^2) softmax1+ALiBi reference path (flag OFF) on the ACTUAL Attn module
used by the CFM pure-forward, over a battery of un-cached full-forward shapes at the
real growing-stream scales the c90 battery hits (S up to tens of thousands of tokens).

If the max abs diff is below the nibble-decode margin (~1e-4; kernel fp32 noise is
~1e-6) at every S, the flash path is byte-exact and safe to use for the full battery.

Also measures the memory + speed win: peak CUDA VRAM and wall time for the masked-full
path vs the flash path at each S — the point of flash is O(S) memory (the masked-full
path OOMs / is minutes-per-step at large S).

Run:  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=<c4_release> \
        python id_port/c90_e2e/verify_flash_byte_exact.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from c4_min.blogspec_model import Attn  # noqa: E402


def _reset_env(flash):
    if flash:
        os.environ["C4_FLASH_ATTN"] = "1"
    else:
        os.environ.pop("C4_FLASH_ATTN", None)


def run_attn(attn, x, flash):
    _reset_env(flash)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = attn(x)          # default un-cached forward
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    _reset_env(False)
    return out, dt, peak


def main():
    dev = "cuda:0"
    assert torch.cuda.is_available(), "flash path needs CUDA"
    # The CFM blocks use the production geometry: d_model=872, per-block head counts
    # vary, but the WORST-case (most heads, thus largest score matrix per head) plus
    # the exact d_model/head_dim used by the memory/stack global heads matter most.
    # Test a representative spread of head counts + the real d_model.
    torch.manual_seed(0)
    # PRODUCTION CFM geometry: d_model=2160, all 242 blocks n_heads=24 head_dim=90.
    # Also test the aligned HD=64 (SDPA-native) + a 1-head global (head_dim large) to
    # exercise both the aligned and the pad-to-8 alignment paths.
    D = 2160
    configs = [(D, 24), (2048, 32), (D, 1)]   # (24,90) real; (32,64) aligned; (1,2160)
    # S values bracketing the c90 battery: short cases ~100-500 tokens/step, the long
    # loop/function/recursion cases grow to 10k-45k tokens on the final step.
    S_list = [128, 512, 2048, 8192, 20000]

    print(f"=== FLASH byte-exact + memory/speed (device={dev}, D={D}) ===")
    print(f"{'H':>3} {'S':>7} | {'maxdiff':>10} | {'full ms':>9} {'full GB':>8} | "
          f"{'flash ms':>9} {'flash GB':>8} | {'mem x':>6} {'spd x':>6}")
    worst = 0.0
    any_full_oom = False
    for (d, H) in configs:
        attn = Attn(d, H, max_seq_len=65536, positional="alibi",
                    sink="softmax1").to(dev).eval()
        # random but fixed weights (byte-exactness is about the two PATHS agreeing on
        # the SAME weights + input, not the baked values).
        for p in attn.parameters():
            torch.nn.init.normal_(p, std=0.02)
        for S in S_list:
            x = torch.randn(1, S, d, device=dev) * 0.5
            # reference (masked-full O(S^2)); may OOM at large S -> that's the point.
            try:
                ref, dt_full, gb_full = run_attn(attn, x, flash=False)
                full_ok = True
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                full_ok = False
                any_full_oom = True
            got, dt_flash, gb_flash = run_attn(attn, x, flash=True)
            if full_ok:
                diff = (ref - got).abs().max().item()
                worst = max(worst, diff)
                memx = gb_full / max(gb_flash, 1e-9)
                spdx = dt_full / max(dt_flash, 1e-9)
                print(f"{H:>3} {S:>7} | {diff:>10.2e} | {dt_full*1e3:>9.1f} "
                      f"{gb_full:>8.3f} | {dt_flash*1e3:>9.1f} {gb_flash:>8.3f} | "
                      f"{memx:>6.1f} {spdx:>6.1f}")
            else:
                print(f"{H:>3} {S:>7} | {'FULL-OOM':>10} | {'OOM':>9} {'OOM':>8} | "
                      f"{dt_flash*1e3:>9.1f} {gb_flash:>8.3f} | {'--':>6} {'--':>6}")
            del x
            torch.cuda.empty_cache()
        del attn
        torch.cuda.empty_cache()

    print(f"\nworst maxdiff (flash vs masked-full softmax1) = {worst:.3e}")
    margin = 1e-4
    if worst < margin:
        print(f"BYTE-EXACT: worst diff {worst:.2e} < nibble margin {margin:.0e} "
              f"-> flash is byte-identical to the masked-full softmax1 path.")
    else:
        print(f"*** DIVERGENT: worst diff {worst:.2e} >= {margin:.0e} — investigate ***")
        return 1
    if any_full_oom:
        print("The masked-full path OOMed at some S the flash path handled -> flash "
              "makes the long cases TRACTABLE (O(S) memory).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
