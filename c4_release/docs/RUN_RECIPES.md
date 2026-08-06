# RUN_RECIPES — reproduce every capstone claim

A copy-pasteable quickstart for re-running each headline result. Every command is
run from the repo `c4_release/` directory (the one that contains `c4_min/`, `src/`,
`id_port/`, `docs/`). Set `REPO` once:

```bash
REPO=/path/to/c4_release          # the dir containing c4_min/ src/ id_port/
cd "$REPO"
```

CPU recipes (1, 2, 4, 5, 6) are memory-safe and were **VERIFIED by running on
2026-08-06** — the observed output is quoted inline. GPU recipes (3, and the
transformer half of 4/5/6) are marked **[GPU]**: they build the transformer and
must run on a card; the expected result is the one reported in the capstone /
perf docs. The golden build fingerprint is untouched by anything here.

> Memory safety: never `load_sparse_transformer` / full-densify the transformer on
> a box with < 25 GB free — it OOMs the whole process. The CPU recipes below use
> either the lean sparse-streaming build (peak ≈ one dense block, ~1 GB) or the
> small in-weights compiler; none full-densify.

---

## 1. Build fingerprint / golden [CPU — VERIFIED]

Deterministic SHA-256 over the dense-reconstructed per-block weights of the single
full-op interpreter (streaming sparse build, peak = one dense block). This is the
byte-identity gate: the hash must stay constant across merges except for
intended-by-construction moves.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 PYTHONPATH="$REPO" \
  python -m c4_min._fingerprint_build
# → FINGERPRINT 7d4afe61a12fc7aecfb9bed97adc4a50bccfae88e83dc6076c881a63b4874e44

# Escape hatch: roll back the default-ON BP-restore fix to the pre-fix golden
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 C4_BP_RESTORE_HIBYTE=0 PYTHONPATH="$REPO" \
  python -m c4_min._fingerprint_build
# → FINGERPRINT 069cc32fa7cecfbceae448a7dbf6e2140b3db6cf6857c8accec5639b9c55c0ca
```

**Observed:** default → `7d4afe61…`, `C4_BP_RESTORE_HIBYTE=0` → `069cc32f…`. Both
match exactly. (`7d4afe61` is the current default golden; `069cc32f` is the
pre-BP-restore golden the escape hatch reproduces.)

---

## 2. C90 conformance — CPU oracle (native ./c4 == gcc) [CPU — VERIFIED]

Two conformance batteries under `id_port/c90_e2e/`. Each C case is compiled and run
three ways and compared byte-exact: `gcc-15 -std=c90` exit code (independent ground
truth), the faithful full-word `native_c4` VM (the authoritative c4-semantics
reference), and — on a GPU pass — the transformer. These CPU validators run only
the gcc + native stages (no transformer), so they are fast and memory-safe.

The compiler is `src.compiler.compile_c` (the Python c4 front-end); `native_c4.py`
is the faithful c4 interpreter. `gcc-15` must be on PATH (override with
`C90_GCC=<gcc>`). The standalone `/tmp/c4c` binary (built from
`bundler/c4_compile.c`) is the C-source compiler used by recipe 4, not by these two.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
  python id_port/c90_e2e/validate_ext.py
# → native==gcc : 112/112   ALL GREEN

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
  python id_port/c90_e2e/validate_ext2.py
# → native==gcc : 81/81     ALL GREEN
```

**Observed:** `validate_ext.py` → gcc-compiled 112/112, c4-compiled 112/112, native
ran 112/112, **native==gcc 112/112 ALL GREEN**. `validate_ext2.py` → **native==gcc
81/81 ALL GREEN** (plus the static transformer-divergence-class tally and the
by-design out-of-subset rejections; it rewrites `CONFORMANCE_MATRIX_EXT2.json`).
Combined the two batteries are the 193-case oracle (`112 + 81`) that the hardened
transformer in recipe 3 is scored against.

---

## 3. C90 on the hardened transformer [GPU]

Runs the same ext (112) + ext2 (81) in-subset cases through the **hardened doom
pure-forward transformer** (lean sparse-streaming CFM build) and compares the
decoded final `AX & 0xFF` byte-exact vs the `native_c4` oracle. Requires a GPU and
≥ 25 GB free host RAM (it aborts otherwise).

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  C4_PF_CFM=1 C4_CMP32=1 C4_CMP32_ORDER=1 C4_MEM_ADDR_BITS=18 \
  C4_EXACT_EVICT=1 C4_MEM_EFF=500000 C4_GLOBAL_ADDR32=1 C4_FLASH_ATTN=1 \
  C4_BP_RESTORE_HIBYTE=1 \
  PYTHONPATH="$REPO" python id_port/c90_e2e/run_hardened_passrate.py
```

**Expected (from the capstone):** **166/170 byte-exact = 97.6%** of cases run
(L-inf=0 vs `native_c4`). Every conformance class is 100% *except* signed-negative
`DIV`/`MOD` (the 4 residual cases): the nibble ALU does unsigned base-16 long
division where gcc does signed trunc-toward-zero. Adding **`C4_DIVMOD_SIGNED=1`**
(the negate-by-dividend-sign wrapper) closes those 4 → **100%**. All flags are
default-OFF / correctness-only; the bare-env golden `7d4afe61` is unchanged.
(`C4_BP_RESTORE_HIBYTE` is now default-ON, listed for explicitness.)

---

## 4. Compiler-in-weights byte-exact vs real c4 [CPU — VERIFIED]

Proves the compiler-in-WEIGHTS transformer emits bytecode **byte-identical to the
real c4 compiler**. The model reads a raw C expression, EMITs bytecode (one
recurrent forward per VM step, argmax decode), runs it; the arithmetic core is
compared against `bundler/c4_compile.c` (built with gcc), and the decoded result
against `eval(expr)`. This builds a *small* in-weights compiler (not the full
transformer), so it is CPU-tractable.

```bash
# build the real c4 compiler once
gcc -w -fpermissive -static-libgcc -o /tmp/c4c bundler/c4_compile.c

CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
  python c4_min/validate_vs_real_c4.py --c4 /tmp/c4c
```

**Observed (exit 0):** all **8/8** straight-line expr cores (`2+3*4`, `2*3+4`,
`1+2+3`, `2*3*4`, `4+5*6`, `3*4+5`, `9+8+7`, `5*6*7`) print **MATCH / ok** →
`model-produced arith cores byte-identical to real c4 (all): True`; and all **6/6**
fetch-dedup variable-length LOOP-compiler chains (`2+3+4+5`, `2*3*4*5`, `7+8+9`,
`1+2+3+4+5+6`, `3+3+3+3+3+3+3`, `1+1+1+1+1+1+1+1+1+1`) print **MATCH** →
`dedup loop-compiler cores byte-identical to real c4 (all): True`. (Runs ~15 min on
CPU — many recurrent transformer forwards per expression; small in-weights compiler
`D=676, 19 blocks`. Comparisons + shifts are covered in the companion
`c4_min/nibble_compiler_ext.py`.)

---

## 5. Mandelbrot byte-exact [CPU — VERIFIED; transformer half GPU]

The native-MUL Mandelbrot render (`mandelbrot_native`). The CPU test gates the
native-MUL escape counts byte-for-byte against the software-mul reference on the
CPU word oracle (`ref_interpret`, mask `0xFFFFFFFF`) — the neural model is never
built — plus that the pixel program is a real LOOP (constant size in `max_iter`)
and the PPM stream round-trips.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
  python -m pytest c4_min/test_mandelbrot_native.py -q
# → 8 passed
```

**Observed:** **8 passed** (grid mapping, native-MUL loop, escape counts byte-exact
for MI∈{8,12,20}, signed cross-term truncate-toward-zero, PPM well-formed,
pixel-program-is-small).

**[GPU]** On the hardened doom transformer build the render is byte-exact end to end
(capstone §5: interior pixel 4201/4201 steps, escape 508/508, boundary 3/3) — the
correctness fix that made this ship by default is `C4_BP_RESTORE_HIBYTE` (default-ON,
golden `7d4afe61`).

---

## 6. Doom byte-exact — CPU proxies [CPU — VERIFIED; full frame GPU]

The two in-repo CPU proxies exercise the doom fixed-point and name-compare
intrinsics (the two hot doom primitives) against C references + the c4 word VM,
byte-exact, with the native-op-is-one-step and megablock-schedule invariants. No
transformer / GPU.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
  python -m pytest c4_min/test_doom_fixedpoint.py c4_min/test_doom_nameeq.py -q
# → 31 passed  (15 fixedpoint + 16 nameeq)
```

**Observed:** **31 passed** — `test_doom_fixedpoint.py` 15/15 (FixedMul==int64-shift,
FixedDiv guard/normal, reference==gadget full battery, byte-exact on the c4vm32
word VM, one-step-per-native-op) and `test_doom_nameeq.py` 16/16 (toupper==C,
case-insensitive name-eq, byte-exact vs real compiled name-eq on c4vm32).

**[GPU / out-of-git]** The full Doom **title frame is byte-exact vs gcc, hash
`2e883404`**, rendered start-to-finish by the transformer (capstone §3). That
end-to-end artifact lives in the out-of-git `c4_doom` tree, not in this repo; the
two proxies above are the in-repo byte-exact evidence for the doom primitives.

---

## 7. Performance ladder

The measured, byte-exact perf story is documented (not re-run here — GPU + long
frames). Two authoritative docs:

- [`docs/PERF_LADDER_FINAL.md`](PERF_LADDER_FINAL.md) — the honest #841/#849 ladder:
  each rung (draft speculation, fused KV eviction, FFN megablock, attention
  megakernel, block-0 fold, WAD-hash + direct-CAM O(1) reads, render
  superinstruction, 2-GPU frame parallelism), its **measured** factor, and whether
  it is byte-exact. Headline: the composed full VM step is **0.788 µs/step**
  (idle A5000, byte-exact, `C4_BLOCK0_DK`); the render-reduced 1 s/frame target is
  **met**, and reaching the raw 6.89 M-steps/s frame on one card is shown to be
  HBM-bandwidth-bound (impossible byte-exactly on one A5000) — closing it is
  algorithmic (fold traversals) or hardware (~8–14 A5000s frame-parallel).
- [`c4_min/COMPOSED_FULL_STEP_841.md`](../c4_min/COMPOSED_FULL_STEP_841.md) — #841:
  all levers composed into ONE measured full step, byte-exact vs the certified c4
  draft (K∈{512,2048,8192}, scalar==GPU decode, L-inf=0), with the
  attention/FFN/decode breakdown (FFN ≈ 72%, attention ≈ 28%, decode ≈ 1%).

Every perf lever is a **DEFAULT-OFF runtime flag** (see
[`docs/DOOM_FLAG_REGISTRY.md`](DOOM_FLAG_REGISTRY.md)); none touches weights, so
golden `7d4afe61` is unchanged.
