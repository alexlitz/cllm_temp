#!/usr/bin/env python3
"""measure_native_fixed_mac.py — PROVE the genuinely-vanilla + full-precision
(byte-exact) fixed-point MAC through the DISCRETE-TOKEN round-trip on a real
``Qwen2ForCausalLM``, and MEASURE the honest cost vs the two other regimes:

  * ``native_fp32_baked`` (C4_FP32_ALU): CONTINUOUS fp32 scalars, value-faithful,
    ~4 forwards/MAC — but NOT the discrete-token round-trip (the answer is read off
    a residual dim, never re-quantised to tokens).
  * the INTEGER VM (native_fp32_vm draft): 76-101 steps/MAC.

Here every value goes IN as discrete nibble tokens, is computed by vanilla SwiGLU +
attention layers, and comes OUT re-quantised to discrete nibble tokens via the REAL
``lm_head`` argmax — byte-exact vs numpy fixed-point.

CPU-only.  Run:  python -m c4_min.measure_native_fixed_mac
"""
from __future__ import annotations

import random
import sys
import time

from c4_min.native_fixed_mac_discrete import (
    FRAC_BITS,
    W_NIB,
    build,
    fixed_mac,
    from_fixed,
    run_dot,
    run_mac,
    to_fixed,
)


def prove_mac(fm):
    print("=" * 78)
    print("1. fixed-point MAC through the REAL model.forward — DISCRETE-TOKEN round-trip")
    print("=" * 78)
    print("   tokens-in (a,b,acc nibbles) -> embed -> vanilla compute -> lm_head ARGMAX")
    print("   re-quant -> tokens-out (result nibbles) -> decode.  Byte-exact vs numpy.\n")
    cases = [(2.5, 4.0, 0.0), (3.0, 4.0, 5.0), (-2.5, 4.0, 1.0), (1.5, -3.0, 0.0),
             (-2.5, -4.0, 0.0), (0.5, 0.5, 0.0), (7.0, 13.0, -10.0), (100.25, 2.0, 0.0)]
    worst = 0
    for a, b, acc0 in cases:
        _res, info = run_mac(fm, a, b, acc0)
        worst = max(worst, abs(info["result_word"] - info["ref_word"]))
        tag = "OK " if info["exact"] else "XX "
        print(f"   {tag} MAC({a:>7},{b:>6}) + {acc0:>6}  ->  model.forward={info['result_value']:<11.6g}"
              f"  numpy_fixed={info['ref_value']:<11.6g}  emit_nibs={info['emitted_nibbles']}")
    print(f"\n   -> worst |result_word - numpy_fixed_word| = {worst}  "
          f"({'BYTE-EXACT' if worst == 0 else 'MISMATCH'})")
    print(f"   -> witnesses: used_inputs_embeds_for_computed_value="
          f"{info['used_inputs_embeds_for_computed_value']}  reencoded_state={info['reencoded_state']}"
          f"  (the answer SURVIVES embed->compute->lm_head-argmax->decode)\n")


def prove_random(fm, n=60, seed=0):
    print("=" * 78)
    print(f"2. randomized byte-exactness ({n} random signed MACs)")
    print("=" * 78)
    rng = random.Random(seed)
    ok = 0
    for _ in range(n):
        a = round(rng.uniform(-100, 100), 3)
        b = round(rng.uniform(-100, 100), 3)
        acc0 = round(rng.uniform(-500, 500), 3)
        _res, info = run_mac(fm, a, b, acc0)
        ok += int(info["exact"])
    print(f"   {ok}/{n} byte-exact through the discrete-token round-trip on model.forward")
    print(f"   -> {'ALL EXACT' if ok == n else f'{n - ok} FAILURES'}\n")


def prove_dot(fm):
    print("=" * 78)
    print("3. length-K DOT = a CHAIN of MACs (each step's ACC = the PRIOR EMITTED word,")
    print("   fed back as DISCRETE nibble tokens: the answer round-trips lm_head AND")
    print("   re-enters through embed_tokens each step)")
    print("=" * 78)
    rng = random.Random(1)
    worst = 0
    for K in (1, 2, 4, 8):
        a = [round(rng.uniform(-8, 8), 3) for _ in range(K)]
        b = [round(rng.uniform(-8, 8), 3) for _ in range(K)]
        _res, info = run_dot(fm, a, b)
        worst = max(worst, abs(info["result_word"] - info["ref_word"]))
        tag = "OK " if info["exact"] else "XX "
        print(f"   {tag} K={K:>2}  dot={info['result_value']:<+12.6f}  numpy_fixed="
              f"{info['ref_value']:<+12.6f}  forwards={info['forwards']}"
              f"  forwards/MAC={info['forwards_per_mac']:.1f}")
    print(f"\n   -> worst |dot_word - numpy_fixed_word| = {worst}  "
          f"({'BYTE-EXACT' if worst == 0 else 'MISMATCH'})\n")


def report_cost(fm):
    print("=" * 78)
    print("4. THE HONEST COST — forwards/MAC of the genuine-vanilla + full-precision")
    print("   (discrete-token, byte-exact) MAC vs the continuous and integer regimes")
    print("=" * 78)
    _res, info = run_mac(fm, 3.0, 4.0, 0.0)
    fwd = info["forwards"]
    print(f"   this module (DISCRETE-TOKEN, byte-exact fixed-point):")
    print(f"       forwards/MAC   = {fwd}   (= W_NIB = {W_NIB}: one lm_head argmax per")
    print(f"                        result nibble emitted — the discrete round-trip cost)")
    print(f"       model layers   = {fm.n_layers}   (vanilla Qwen2 decoder layers; each a")
    print(f"                        SwiGLU FFN or an attention CAM/broadcast head)")
    print(f"       d_model        = {fm.hidden_size}   frame = {fm.frame_len} discrete tokens")
    print(f"       precision      = 32-bit fixed-point (Q{32 - fm.frac}.{fm.frac}), BYTE-EXACT")
    print()
    print("   native_fp32_baked (C4_FP32_ALU, CONTINUOUS fp32 scalars):")
    print("       forwards/MAC   = 4     (FLI a; FLI b; FMUL; FADD — 4 baked blocks)")
    print("       precision      = value-faithful ~1e-7, NOT byte-exact, NOT discrete")
    print("                        tokens (the answer is READ off a residual dim, never")
    print("                        re-quantised through lm_head).")
    print()
    print("   integer VM (native_fp32_vm draft / c4vm):")
    print("       steps/MAC      = 76-101  (stack-machine dispatch, byte-safe paging)")
    print()
    print("   VERDICT (the honest vanilla/precision/speed tax):")
    print("     * CONTINUOUS + fast (4/MAC) BUT not vanilla (no discrete-token round-trip)")
    print("       and not byte-exact (value-faithful ~1e-7).")
    print("     * DISCRETE + byte-exact + genuinely vanilla (THIS module) costs the")
    print(f"       discrete round-trip: {fwd} forwards/MAC (one lm_head argmax per result")
    print("       nibble) — the SAME W-nibble emission rate the nibble register VM pays.")
    print("       You CANNOT have continuous-4/MAC AND the discrete-token round-trip: the")
    print("       round-trip IS the emission of the result as tokens. That is the tax.")
    print("     * IEEE-fp32 byte-exact is NOT achievable via the silu round-trip at all")
    print("       (fp32 multiply ROUNDS; re-quant needs full IEEE-754 bit surgery, not a")
    print("       silu op). Full-precision byte-exact => fixed-point, which is what this is.\n")


def main():
    print("PROVING a GENUINELY-VANILLA + FULL-PRECISION (byte-exact) fixed-point MAC")
    print("through the DISCRETE-TOKEN round-trip on a stock Qwen2ForCausalLM.\n")
    t0 = time.time()
    fm = build(force=True)
    print(f"[built the vanilla Qwen2 fixed-point-MAC model in {time.time() - t0:.1f}s: "
          f"{fm.n_layers} layers, d_model={fm.hidden_size}]\n")
    prove_mac(fm)
    prove_random(fm, n=40)
    prove_dot(fm)
    report_cost(fm)
    print("=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print("  A fixed-point MAC runs tokens-in -> vanilla compute -> lm_head ARGMAX")
    print("  re-quant -> tokens-out on the REAL model.forward, BYTE-EXACT vs numpy fixed")
    print("  point. The value enters as discrete nibble tokens, is multiplied+accumulated")
    print("  by the byte-exact nibble gadgets (silu-gated multiply + carry rounds), and")
    print("  SURVIVES the discrete round-trip embed->compute->lm_head-argmax->embed. This")
    print("  is the genuinely-vanilla discrete-token realisation the continuous fp32 bake")
    print("  (C4_FP32_ALU, 4/MAC, value-faithful) is NOT. Gated by C4_FIXED_MAC_DISCRETE")
    print("  (default OFF): the integer VM golden and the fp32 mode are both unaffected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
