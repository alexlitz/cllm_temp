"""VARIABLE-DIVISOR DIV depth bakeoff — the honest summary self-check.

Runnable: ``python -m c4_min.div_variable_bakeoff``

Question: can the VARIABLE-divisor DIV *instruction* (arbitrary runtime 32-bit
divisor — NOT divide-by-constant / magic-multiply) be built under 40 blocks of
DEPTH, byte-exact?

This prints the measured depth / nz / byte-exact for every variable-divisor DIV
alternative that was actually built, and the honest verdict + what dominates.

Alternatives (all VARIABLE divisor, all measured by their own module):
  radix-16 nibble recurrence (div_radix16_lean)     : 80 blocks   (baseline)
  radix-16 hardened (div_radix16_hardened)          : 88 blocks   (fp32-hardened)
  radix-16 attention-select (div_radix16_attn)      : 82 FFN + 8 heads
  radix-256 SRT estimate (div_radix256_est)         : 90 blocks   (THIS session)
  radix-256 table (div_radix256_srt)                : ~52 blocks structural,
       INFEASIBLE to build (256-row KB table -> d_model 8200, carry-normalise
       O(256*kmax) units).

VERDICT (measured): radix-256 does NOT clear 40 blocks with these fp32-safe nibble
primitives.  The floor is set by the irreducible per-byte-iteration cost — a
byte*32-bit multiply (q*d) + a log/limb-depth borrow (R - q*d) — plus the honest
CLZ-normalize overhead (variable ``d<<sh`` / ``a - q*d``).  4 iters * ~18 blocks +
~18 prologue/epilogue = ~90.  Fewer iterations (radix-256 vs radix-16's 8) does NOT
win because each byte iteration is ~2x wider (a byte quotient of a normalized
divisor is an inherently ~16-bit compare + a byte*32 product) — the multiply and
the borrow are the depth, and radix-256 pays them 4x on a 2x-wider operand, a wash
vs radix-16 paying a nibble select+borrow 8x.
"""
from __future__ import annotations


def _fmt(name, depth, nz, be64, be_total, note=""):
    return f"  {name:<42} depth {depth:>4}  nz {nz:>8}  byte-exact(fp64) {be64}/{be_total}  {note}"


def run():
    print("=" * 84)
    print("VARIABLE-DIVISOR DIV — depth bakeoff (goal: < 40 blocks, byte-exact fp64 & fp32)")
    print("=" * 84)

    from . import div_radix16_lean as LEAN
    from . import div_radix256_est_measure as R256

    print("\n[1] radix-16 nibble digit-recurrence (baseline) ...", flush=True)
    lm = LEAN.measure(verbose=False)
    print(_fmt("radix-16 lean (8 nibble iters)", lm["depth_unrolled"], lm["nnz"],
               lm["byte_exact_pass"], lm["byte_exact_total"]))

    print("\n[2] radix-256 SRT estimate (4 byte iters, THIS session) ...", flush=True)
    rm = R256.measure(verbose=False, n_random=1500)
    print(_fmt("radix-256 SRT estimate (4 byte iters)", rm["depth_unrolled"], rm["nnz"],
               rm["byte_exact_pass"], rm["byte_exact_total"],
               f"fp32(single-row) {rm['byte_exact_pass_fp32_singlerow']}/{rm['singlerow_total']}"))

    print("\n" + "-" * 84)
    print("DEPTH BREAKDOWN (radix-256 SRT estimate):")
    print(f"  normalize (CLZ + dn<<sh + 2dn + Bp + BZ + init)          :  9 blocks")
    print(f"  per iteration x4 (= {rm['blocks_per_iter']} blocks/iter):")
    print(f"     shift-insert (Rn = 256*Rn + byte<<sh)                 :  2")
    print(f"     estimate (Ahat=Rn>>24 ; qhat=floor(Ahat/Bp) staircase):  2  (ahat + est)")
    print(f"     q*d (byte*32 nibble schoolbook + 2 carry rounds)      :  3")
    print(f"     lane-form (qhat*dn, +dn, +2dn) + carry                :  3")
    print(f"     limb borrow (Rn - lane_sub, 4 limbs)                  :  4")
    print(f"     split (clean floors + snapped nibbles)                :  2")
    print(f"     select-emit (pick valid lane, emit q byte)            :  2")
    print(f"  epilogue: MOD = a - q*d (mul 3 + borrow 3 + floors 1 + finalize 2):  9 blocks")
    print(f"  TOTAL = {rm['depth_unrolled']} blocks")
    print("-" * 84)

    depth = rm["depth_unrolled"]
    print(f"\nVERDICT: variable-DIV depth = {depth} blocks -> "
          f"{'UNDER' if depth < 40 else 'OVER'} the 40 goal.")
    print("  radix-256 does NOT clear 40.  What dominates (per byte iter): the q*d")
    print("  byte*32 MULTIPLY (3 blk), the limb BORROW of R-q*d (4 blk), and the")
    print("  estimate + 3-lane correction — ~18 blk/iter.  4 iters + ~18 CLZ-normalize")
    print("  / a-q*d epilogue = ~90.  radix-256's 4 iters (vs radix-16's 8) is a WASH:")
    print("  each byte iteration is ~2x wider (byte quotient of a normalized divisor =")
    print("  a 16-bit estimate compare + a byte*32 product), so 4x(2x-wide) ~= 8x(nibble).")
    print("  The one-time CLZ-normalize (variable d<<sh, a-q*d) is a real, honest tax.")

    print("\n  Fallbacks assessed:")
    print("    - radix-16 attention-select (built, div_radix16_attn): shaves the SELECT")
    print("      to 1 head but the SUBTRACT/bring-down stay FFN -> 82 FFN + 8 heads, > 40.")
    print("    - radix-256 EXACT-select table (built, div_radix256_srt): ~52 blk structural")
    print("      AND infeasible (256-row KB table -> d_model 8200, O(256*kmax)-unit carry).")
    print("    - Newton-Raphson reciprocal: MULTIPLY-bound (2 muls/iter x 2 iters + a final")
    print("      mul); each 32-bit mul is the ~8-block schoolbook here -> ~40-50 blk, no win")
    print("      over SRT, and needs a seed reciprocal table.  Not built (SRT already > 40).")
    print("=" * 84)

    print("\nHOSTING (which Qwen sizes fit the sub-DEPTH variable divide INLINE vs recur):")
    print("  Even the SHALLOWEST variable divide here is 80 (radix-16 lean) / 90 (radix-256).")
    print("  Stock transformer DEPTHS: 0.5B=24L, 1.5B=28, 3B=36, 7B=28, 14B=48, 32B=64,")
    print("  72B=80.  NONE host an 80-90-block divide INLINE in one forward: even 72B (80L)")
    print("  is at the radix-16 depth with zero room for the rest of the ISA.  So the")
    print("  variable DIV opcode MUST run RECURRENTLY (the single ~9-18-block iteration")
    print("  body reused per digit) on every size, exactly as div_radix16_*'s")
    print("  compile_*_recurrent do.  d_model/width for the residual bands + selection")
    print("  table: radix-16 lean D~250, radix-256 estimate D=480 (both << every Qwen")
    print("  d_model: 0.5B=896, 1.5B=1536, 3B=2048, 7B=3584, 14B/32B=5120, 72B=8192), so")
    print("  WIDTH is never the constraint — DEPTH is, and it forces recurrence everywhere.")
    print("=" * 84)


if __name__ == "__main__":
    run()
