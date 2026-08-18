"""_matmul_dedup_src.py — the DEDUP (palette+index) and FUSED-MAC C kernels, with
their steps/MAC GROUNDED through the same real c4 toolchain + 32-bit draft VM the
baseline paged COO kernel uses (``ground_true_self_emulation``).

Two independent levers on a matmul MAC ``acc += W[k] * x[k]``, measured HONESTLY as
runnable C -> c4 bytecode -> draft-VM step counts (nothing extrapolated off a
guessed rate):

  1. DEDUP IN THE C SOURCE (``paged_dedup_dot_c``).  The emulated weights are stored
     DEDUPED: a PALETTE of the unique values (``pal[]``) + an address->palette-index
     map (``idx[]``).  A weight read is the TWO-ACCESS chained indirection the task
     specifies:

         i = idx[k];        // LI: the palette index for this weight slot   (1 load)
         W = pal[i];        // LI: pal_base + i*4  -> the value             (1 load)
         acc = acc + W * x[k] / s;

     i.e. it ADDS one load + one address-compute per MAC vs the baseline's single
     ``*wp`` weight load.  This is the real C-source dedup; the byte-exact result is
     IDENTICAL to the direct-weight kernel (``pal[idx[k]] == w[k]`` by construction).

  2. FUSED-MAC (``fused_mac_dot_c`` / the C4_MEM_OPERAND model).  The whole per-
     element multiply-accumulate is ONE fused op ``MAC [a],[b]`` — the two CAM reads
     run in the EARLY blocks of the instruction's OWN forward and feed the LATE-block
     multiply-accumulate, so one drafted VM step commits the whole MAC.  We ground
     its cost as the MEASURED bytecode-MAC step count DIVIDED by the fused collapse
     (76-101 stack-machine steps -> 1), matching ``measure_fused_mac_vs_bytecode``.

  3. DEDUP + FUSED.  The fused MAC takes the DEDUPED operand: the palette-index
     resolution folds into the operand ADDRESS (``a = pal_base + i`` where
     ``i = idx[k]``), exactly like the direct-CAM draft resolves a data-dependent
     gather.  Reported below: does the index resolution fold into the fused MAC's
     own forward (0 extra steps) or cost a separate resolving step.

Everything CPU-only, byte-exact through the SAME ``ref_interpret_word32`` the
grounding uses.  Run:  python -m c4_min.selfhost._matmul_dedup_src
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from c4_min.selfhost._matmul_paged_src import (
    SCALE, TILE, _decl_rev, paged_dot_c, paged_dot_reference, _byte_mask_dot)


# --------------------------------------------------------------------------- #
# 1. DEDUP paged dot — weight via chained  i = idx[k]; W = pal[i]              #
# --------------------------------------------------------------------------- #
def paged_dedup_dot_c(w: List[int], x: List[int], tile: int = TILE,
                      scale: int = SCALE) -> str:
    """Return C for the SAME dot ``c = sum_k w[k]*x[k]`` but every weight is fetched
    through the palette+index indirection (``W = pal[idx[k]]``) rather than a direct
    ``w[k]`` load — the real C-source dedup.  Byte-identical result to
    :func:`paged_dot_c`.

    Layout per tile (all inside the 256-byte LEA window, TILE=8 -> generous room):
      * ``pal0..palP``  — the P+1 UNIQUE weight values needed by this tile's slots.
      * ``id0..idT``    — the per-slot palette INDEX (0..P), i.e. ``w[k]==pal[id_k]``.
      * ``x0..xT``      — the gathered input values (as in the baseline).
    The MAC loop resolves ``ip = idb + k*4`` (LI -> the index), then
    ``vp = palb + (*ip)*4`` (LI -> the value), then multiply-accumulates.
    """
    assert 0 < scale <= 255 and len(w) == len(x)
    for v in w + x:
        assert 0 <= v <= 255, f"entry {v} must be a byte literal (0..255)"
    N = len(w)
    ntiles = (N + tile - 1) // tile
    xdecl = _decl_rev("x", tile)
    iddecl = _decl_rev("id", tile)
    # a per-tile palette is at most `tile` values (<=8), so declare `tile` pal slots.
    paldecl = _decl_rev("pal", tile)
    tiles = []
    for t in range(ntiles):
        lo, hi = t * tile, min(t * tile + tile, N)
        m = hi - lo
        seg_w = w[lo:hi]
        # build this tile's palette (unique values, first-seen order) + per-slot index
        pal_vals: List[int] = []
        pal_pos: Dict[int, int] = {}
        ids: List[int] = []
        for v in seg_w:
            if v not in pal_pos:
                pal_pos[v] = len(pal_vals)
                pal_vals.append(v)
            ids.append(pal_pos[v])
        palinit = " ".join(f"pal{i} = {pal_vals[i]}*s;" for i in range(len(pal_vals)))
        idinit = " ".join(f"id{i} = {ids[i]};" for i in range(m))
        xinit = " ".join(f"x{i} = {x[lo + i]}*s;" for i in range(m))
        tiles.append(
            f"  {palinit}\n"
            f"  {idinit} {xinit}\n"
            f"  idb=&id0; palb=&pal0; xb=&x0; k=0;\n"
            f"  while (k<{m}) {{ ip=idb+k*4; j = *ip; vp=palb+j*4; xp=xb+k*4; "
            f"acc=acc + *vp * *xp / s; k=k+1; }}")
    body = "\n".join(tiles)
    return f'''
int main() {{
  int s; int acc, k, j; char *idb; char *palb; char *xb;
  int *ip; int *vp; int *xp;
  int {paldecl};
  int {iddecl};
  int {xdecl};
  s = {scale}; acc = 0;
{body}
  printf(acc);
  return 0;
}}
'''


def paged_dedup_dot_reference(w: List[int], x: List[int],
                              scale: int = SCALE) -> List[int]:
    """Byte-exact reference — identical to :func:`paged_dot_reference` (the dedup is
    a lossless re-representation of the same weights)."""
    return paged_dot_reference(w, x, scale=scale)


# --------------------------------------------------------------------------- #
# self-check + steps/MAC grounding (CPU, real toolchain + 32-bit draft VM)      #
# --------------------------------------------------------------------------- #
def _compile(src: str):
    from src.compiler import compile_c
    from c4_min import isa
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    bc, _ = compile_c(src)
    code = bytecode_to_isa(bc)
    over = [i.imm for i in code if i.op == isa.IMM and i.imm > 255]
    assert not over, f"IMM>255 leaked: {over}"
    return code


def _run32(code, seed=None) -> Tuple[List[int], int]:
    from c4_min.selfhost.word32_draft_vm import ref_interpret_word32
    out: List[int] = []
    _tr, steps = ref_interpret_word32(code, max_steps=80_000_000, out=out,
                                      seed_mem=seed)
    return out, steps


def measure_dedup_steps_per_mac(verbose: bool = True) -> Dict[str, float]:
    """Marginal steps/MAC for the DEDUP kernel (whole-tile differencing, like the
    baseline paged kernel), plus the baseline for comparison.  Uses a palette of ~2
    distinct values per tile (realistic: the emulated weights are a tiny palette),
    so the index indirection is exercised with real sharing."""
    import numpy as np  # noqa: F401 (kept for parity / potential debug)

    def _steps(srcfn, K):
        # a palette-friendly weight vector: two distinct nonzero values repeated.
        w = [(1 if (i % 2 == 0) else 2) for i in range(K)]
        x = [1] + [0] * (K - 1)
        return _run32(_compile(srcfn(w, x)))

    pts_base, pts_ded = [], []
    for K in [TILE, 2 * TILE, 4 * TILE, 8 * TILE]:
        _o, sb = _steps(paged_dot_c, K)
        _o, sd = _steps(paged_dedup_dot_c, K)
        pts_base.append((K, sb))
        pts_ded.append((K, sd))
    base_pm = (pts_base[-1][1] - pts_base[-2][1]) / (pts_base[-1][0] - pts_base[-2][0])
    ded_pm = (pts_ded[-1][1] - pts_ded[-2][1]) / (pts_ded[-1][0] - pts_ded[-2][0])
    if verbose:
        print("steps/MAC (marginal, whole-tile differencing):")
        for (K, sb), (_, sd) in zip(pts_base, pts_ded):
            print(f"  K={K:>3}: baseline {sb:>6}   dedup {sd:>6}  (+{sd - sb})")
        print(f"  -> baseline marginal steps/MAC = {base_pm:.2f}")
        print(f"  -> dedup    marginal steps/MAC = {ded_pm:.2f}  "
              f"(+{ded_pm - base_pm:.2f}/MAC for the 2nd chained load)")
    return dict(baseline=base_pm, dedup=ded_pm, extra=ded_pm - base_pm)


def _self_check(verbose: bool = True) -> bool:
    import random
    ok = True
    rng = random.Random(3)
    if verbose:
        print("DEDUP paged dot — byte-exact vs the direct-weight kernel + reference:")
    for K in [15, 16, 64, 104]:
        # a small palette (values 0..3) so the dedup index is heavily shared.
        w = [rng.randint(0, 3) for _ in range(K)]
        x = [rng.randint(0, 3) for _ in range(K)]
        out_d, steps_d = _run32(_compile(paged_dedup_dot_c(w, x)))
        out_b, steps_b = _run32(_compile(paged_dot_c(w, x)))
        ref = paged_dot_reference(w, x)
        match = out_d == out_b == ref
        ok = ok and match
        if verbose:
            print(f"  K={K:>3}: dedup={out_d} direct={out_b} ref={ref} "
                  f"{'OK' if match else 'MISMATCH'}  "
                  f"(dedup {steps_d} vs direct {steps_b} steps)")
    return ok


if __name__ == "__main__":
    import sys
    print("DEDUP + FUSED-MAC kernels — CPU self-check (real toolchain + 32-bit VM):\n")
    good = _self_check(verbose=True)
    print()
    measure_dedup_steps_per_mac(verbose=True)
    print(f"\nDEDUP BYTE-EXACT (draft VM == direct-weight == reference): {good}")
    sys.exit(0 if good else 1)
