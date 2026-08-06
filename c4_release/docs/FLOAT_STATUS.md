# FP / fixed-point status on `consolidate-0.5b` (§2 "soft-float lib + native FP" claim)

CPU-only verification (no transformer model build). Every check below runs on the
CPU c4 VM / bit-exact integer gadgets, cross-checked against gcc / numpy.float32 /
Berkeley SoftFloat. Golden fingerprint `069cc32f` untouched (all FP work is behind
`C4_FLOAT_OPS`, default OFF; the E2E soft-float path is a pure-integer C library, no
weight change).

## Verdict (one line)

The §2 claim holds SUBSTANTIALLY and is REAL on the consolidated branch: fixed-point
is fully proven, and a genuine `float` C program compiles through the c4_release
toolchain and runs BYTE-EXACT (full 32-bit IEEE-754) on the CPU native c4 VM via a
soft-float lib — with one honesty caveat: the soft-float C library source
(`softfloat.c`) lives in the sibling `c4_doom` repo, not inside `c4_release`, and the
native FP ISA opcodes (`C4_FLOAT_OPS` F_ADD/F_SUB/F_MUL/F_DIV) are a real but
*optional/gated* accelerator that is NOT wired into the plain CPU integer interpreter.

## FIXED-POINT — WORKS (byte-exact, CPU, in-repo)

The Mandelbrot capstone is genuine fixed-point (Q_FRAC=4, SCALE=16; signed
`_sfp_mul = (a*b)>>Q_FRAC` via native `MUL` + `DIV` rescale; NO float). Byte-exact on
the CPU word reference (`nibble_pure_forward_complete.ref_interpret`, mask 0xFFFFFFFF)
vs the software render.

    python -m pytest c4_min/test_mandelbrot_native.py -q
    # 8 passed  (escape counts byte-exact vs lean_mandelbrot for every pixel;
    #  signed cross-term truncate-toward-zero verified; PPM round-trips)

## SOFT-FLOAT — WORKS (byte-exact, CPU) — lib source is in sibling `c4_doom` repo

A real `float` C program, lowered to the gcc soft-float ABI (`__mulsf3` / `__addsf3`
/ `__subsf3` / `__divsf3` / `__fixsfsi`, a `float` carried as one `int` holding its
32 IEEE-754 bits), compiled by `src.compiler.compile_c` and run on the faithful CPU
`id_port/c90_e2e/native_c4.py` VM, matches gcc / numpy.float32 BIT-EXACT.

Verified this session (ad-hoc CPU harness, softfloat.c from
`c4_doom/id_port/softfloat/softfloat.c`):

    (int)(1.5f*2.25f*100.0f) = 337        c4-native 337  == gcc 337 (lowbyte 81)
    (int)((10.0f/4.0f+0.5f)*1000.0f)=3000 c4-native 3000 == gcc 3000
    (int)((5.5f-2.25f)*100.0f) = 325      c4-native 325  == gcc 325
    3.0f*7.0f raw bits                    == gcc bits
    randomized full-32-bit battery: __mulsf3 37 + __addsf3 40 + __divsf3 38 checks,
      0 mismatches vs numpy.float32 (subnormal-result cases excluded — documented gap)

The soft-float lib itself (`softfloat.c` f32+f64 ops, all 6 comparisons, int<->fp
conversions; `softfloat_str.c` for `%f/%e/%g`/strtod) is documented in
`c4_doom/id_port/softfloat/softfloat_lowering.md` and reports on the native c4 VM:
verify_c4.c 16,251 checks 0 fail; verify_f32.c 2,239,291 checks 0 fail; verify_f64.c
533,589 checks 0 fail; string %f exact, 11 last-ULP `%e/%g/strtod` ties.

Caveat: `softfloat.c`/`verify_c4.c` are NOT in `c4_release` (they live in `c4_doom`,
branch `render-superinstruction-step-collapse`). The transpiler that would insert
these `__*sf3` calls automatically is referenced by the contract doc but not run here;
the lowering was applied by hand in the harness above. So the soft-float lib is REAL
and byte-exact on the c4 VM, but the *in-repo* c90 E2E conformance battery
(`id_port/c90_e2e/`) is INTEGER-subset ONLY (int/char/pointer; no `float`/`double`).

## NATIVE FP ISA (`C4_FLOAT_OPS`, F_ADD/F_SUB/F_MUL/F_DIV) — WORKS, gated, transformer-side

The native single-opcode FP path (opcodes 40..43, above golden NUM_OPS=40; band widens
only when `C4_FLOAT_OPS=1`, golden byte-identical off) is real and CPU-verifiable:

* `isa.f32_op_bits` (host-float oracle) vs gcc: 43600/43600 each for
  F_ADD/F_SUB/F_MUL/F_DIV, 0/174400 mismatches — `python -m c4_min._fp_battery`.
* `native_ieee_fp32.py` — the ACTUAL IEEE-754 algorithm reduced to exact-integer
  nibble gadgets (mul/add/shift/cmp/select), bit-identical to numpy.float32 over
  normal×normal, both signs, RNE ties, overflow/underflow, ±0/inf/NaN specials:

      python -m pytest c4_min/test_native_ieee_fp32.py \
        c4_min/test_native_ieee_fp32_gadgets.py -q       # 233 passed, 44 skipped

* The transformer-weight F_DIV MEGABLOCK (`nibble_fp32.compile_fp_div_blocks`, the
  `C4_FLOAT_OPS` ISA extension realized as blocks) is byte-exact vs the oracle on CPU:

      CUDA_VISIBLE_DEVICES="" C4_FLOAT_OPS=1 \
        python -m pytest c4_min/test_fp_div_megablock.py -q   # 23 passed

  Cross-checked vs Berkeley SoftFloat refvectors: F_DIV normal×normal-with-normal-result
  238/238 bit-exact; the misses are exactly the subnormal-result / subnormal-operand
  cases (documented shared GAP for all four ops). See `c4_min/_fp_battery_results.md`.

## HONEST GAPS

* `double`/f64 native opcode: not in the `C4_FLOAT_OPS` ISA (only f32 F_ADD/SUB/MUL/DIV).
  f64 exists only in the `c4_doom` softfloat.c lib (`__adddf3` etc.), not as a c4 opcode.
* Subnormal operands and gradual-underflow (subnormal RESULT): flushed to signed zero
  in both the native ISA path and the gadget path — documented deferral, shared by all
  four ops. gcc handles these; the native ISA path does not.
* `%f/%e/%g` string formatting: 11 last-ULP ties fail (not the full Gay shortest-round-trip).
* The in-`c4_release` c90 E2E battery is integer-subset only; the soft-float lib and its
  native-c4 verification harness are in `c4_doom`.

## Reproduce

    cd c4_release   # the package root (c4_min/, src/, id_port/ live here)
    python -m pytest c4_min/test_mandelbrot_native.py -q
    python -m pytest c4_min/test_native_ieee_fp32.py c4_min/test_native_ieee_fp32_gadgets.py -q
    python -m c4_min._fp_battery
    CUDA_VISIBLE_DEVICES="" C4_FLOAT_OPS=1 python -m pytest c4_min/test_fp_div_megablock.py -q
    # E2E float-C-through-toolchain: compile_c(softfloat.c + program) -> native_c4.run vs gcc
    # (softfloat.c from c4_doom/id_port/softfloat/; hand-lowered __*sf3 calls)
