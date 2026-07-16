# CHK-4 — single-file bundle (runtime + small model + bytecode)

**Branch:** `chk1-bundle-small` (off `chk1-corpus-fast` @ `c0614d90`)
**Authority:** `docs/BLOG_SPEC.md` §"Bundling Programs" (line 910): *"a bundler
which takes the [runtime], the [model] weights and the target c program,
compiles them into a single binary."*
**Status:** DELIVERED — the bundle assembles + runs end-to-end; the C4-C bundler
produces a byte-identical container. ONNX runtime = interim torch/KV-cached
driver (CHK-2/CHK-3 ONNX not yet committed; integration point marked).

---

## 1. Why this is feasible now

The dense bitwise config is ~30 GB (7.5 B params, 99.998 % zero) and CANNOT be
bundled. The **small** model — the sparse COO re-encode from `compact_alloc.py`
/ `sparse_forward.py` (nnz-only: `index_i32 + value_f32` per nonzero) — is what
makes a single self-contained file practical. Measured on the LEAN config
(`code_size=64, include_bitwise=True, include_divmod=False`, 42 blocks,
dim 2576): the weights blob is **14.2 MB** (dense-equivalent 32.4 GB, ~2286×
compression, 129 815 nonzeros across 465 tensors).

## 2. The bundle (`c4_min/bundle_small.py`)

A `.c4bundle` is a flat binary container fusing THREE sections + a header:

```
MAGIC("C4BUNDLE") VERSION(u32) header_json_len(u32) header_json
[ RUNTIME  ]   # the interim driver's self-describing entry-point note
[ WEIGHTS  ]   # sparse COO, nnz-only  (the 14–32 MB size lever)
[ BYTECODE ]   # the target C program's ISA (op i32, imm i32)*n
```

The header JSON carries the model config (dim/blocks/heads/vocab/code_size),
the program metadata (expected / description / step_cap / n_instrs / source) and
the byte offset+length of each section. Nothing external is needed at run time.

CLI:

```
python -m c4_min.bundle_small assemble --source 'int main(){return 500+700;}' \
    --expected 1200 --out add.c4bundle
python -m c4_min.bundle_small run  add.c4bundle      # decodes 1200
python -m c4_min.bundle_small info add.c4bundle      # header/section dump
python -m c4_min.bundle_small prepare --source ... --out-dir DIR   # for the C4-C bundler
```

### Assembled + run end-to-end (measured)

| program                       | expected | bundle size | run verdict     |
|-------------------------------|----------|-------------|-----------------|
| `return 500 + 700;`           | 1200     | 14.19 MB    | **PASS** got 1200 |
| `return 9 * 8;`               | 72       | 14.19 MB    | **PASS** got 72   |
| `return 1000 - 337;`          | 663      | 14.19 MB    | **PASS** got 663  |

Each runs with NO external files: `run_bundle` rebuilds the zeroed model at the
header config, scatters the embedded COO weights (byte-identical reconstruction,
verified across all 465 tensors), decodes the embedded bytecode, and drives it
through `run_pure_forward_cached` (the KV-cached torch driver), then prints the
argmax-decoded 32-bit result. Load ~10–37 s + run ~24–38 s on CPU (small model).

## 3. Runtime = interim torch/KV-cached driver; ONNX integration point

CHK-2 (`chk1-onnx-export`) and CHK-3 (`chk1-c-onnx-runtime`) are still at the
base commit `c0614d90` with **no committed ONNX artifacts**, so per the task
brief the bundle runs via the **interim** runtime: `run_pure_forward_cached`
(softmax1 + ALiBi + SwiGLU, KV-cached, evicting). The single seam the ONNX C
runtime replaces is the driver call in `run_bundle` (marked *ONNX INTEGRATION
POINT*). The header (config + program metadata), the weights COO blob and the
bytecode section are **runtime-agnostic**: CHK-3's C-in-C4 onnxruntime consumes
the same three sections (it needs the weights as a `.nblbin` — the same COO
re-encode with ONNX tensor names). No format change is required to swap runtimes.

## 4. The C4-C bundler (`bundler/c4_bundler_small.c`)

A bundler **written in the C4 subset of C** (same dialect + gcc `-fsyntax-only`
validation gate as the sibling `bundler/c4_bundler.c` / `neural_c4_bundler.c`).
Division of labour keeps it torch-free:

* the **weight serialisation** (torch tensors → sparse COO) needs torch → stays
  in Python (`bundle_small prepare` writes the raw parts: `header.bin`,
  `runtime.bin`, `weights.bin`, `bytecode.bin` + a `manifest.txt`). This is the
  bytecode-compile-of-the-target step (§912) + the model export.
* the **container fusion** (concatenate parts → single self-contained file) is
  pure byte I/O (`open`/`read`/`close`/`putchar`) → done by the C4-C program.

**VERIFIED byte-identical:** `./c4_bundler_small DIR/manifest.txt > out.c4bundle`
produces a file `cmp`-identical to `bundle_small assemble --out out.c4bundle`,
and that C4-C-produced bundle then **runs end-to-end (PASS, got 1200)**.

**Implemented:** the full container-fusion bundler (manifest walk, per-part
streaming, byte-exact output), gcc-compilable, byte-identity proven in a pytest.
**Scoped (not re-implemented in C4-C):** the torch-only weight COO serialisation
and the C-front-end bytecode compile — these live in Python's `prepare` step
(the C4-C bundler consumes their output), exactly as the sibling `c4_bundler.c`
delegates the ONNX export to Python. A fully-C4-C weight serialiser would need a
torch-free tensor reader; out of scope for the container-fusion deliverable.

## 5. Corpus (§ "passes the 1000+ tests")

The bundle uses the **same** model + `run_pure_forward_cached` driver as the
corpus runner (`run_1096_pure_forward_cached`), so a program's bundle verdict is
its corpus verdict. Validated on a stratified non-deep sample (one per cluster):

```
add_0      PASS(768)   sub_0  PASS(801)   mul_0  PASS(1239)
var_simple PASS(990)   if_gt_0 PASS(0)                    5/5, section round-trip 5/5
```

**Dependency:** the full-1096 number is the corpus result on the small model —
non-deep ~725 pass now; the deep-loop tail is the separate `chk1-memfix-deeploop`
track. This deliverable validates the bundle *mechanism* + a stratified sample;
the absolute full-1096 pass count moves with memfix, not the bundler.

## 6. Tests (`c4_min/test_bundle_small.py`)

Fast tier (default, ~5 s): C4-C bundler gcc-syntax gate, weight COO round-trip
byte-identity (465 tensors), container framing (contiguous ordered sections),
and C4-C-vs-Python byte-identical fusion. Slow tier (`-m slow`): assemble + run
end-to-end. All fast tests green.

## Files

- `c4_min/bundle_small.py`      — the bundler + `run_bundle` interim runtime.
- `bundler/c4_bundler_small.c`  — the C4-C container-fusion bundler.
- `c4_min/test_bundle_small.py` — the CHK-4 tests.
