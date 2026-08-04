# Extended C90 conformance battery (`id_port/c90_e2e/`)

Broadens the C90 end-to-end conformance testing for the neural c4 VM from the
original 25-case matrix (#813/#817) to **112 cases** systematically covering the
c4 C-subset. Every case is a small self-contained C program with a deterministic
integer return value, run through three stages and compared byte-exact.

## The three-stage flow (per case)

```
C source ──► gcc-15 -std=c90          ─► exit code               (independent ground truth)
         ──► native_c4.run            ─► final AX & 0xFF          (faithful full-word c4 VM)
         ──► run_pure_forward_complete─► final AX & 0xFF          (the transformer, lean CFM)
```

- **gcc reference** (`gcc-15 -std=c90`): a real compiler on x86; the ultimate
  ground truth. Where c4 semantics legitimately differ (`sizeof(int)==8` in c4 vs
  4 on x86) the case is tagged `c4-vs-x86`, not a failure.
- **native ./c4** (`native_c4.py`): a faithful full-word c4 interpreter with the
  FULL control set (JSR/ENT/ADJ/LEV), a data segment seeded from the compiler's
  string/global literals, and a bump heap for `malloc`. This is the authoritative
  c4-semantics reference the transformer is compared to. It uses the **compiler's
  ABI**: 8-byte stack cells, `LEA/ENT/ADJ` immediates are byte offsets (NO `*4`).
  (The three shipped interpreters were each partial — `ref_interpret` is 8-bit
  masked, `ref_interpret_words` has no call frames, `isa.interpret` neither — so
  this module is the missing faithful reference.)
- **transformer** (`run_pure_forward_complete`): the LEAN STREAMING CFM build
  (`C4_PF_CFM=1`, code-from-memory KV; ~1.5 GB, code-size-independent residual).

## Files

| file | role |
|------|------|
| `cases_ext.py` | the 112 conformance cases: `(name, category, source, expect)` |
| `native_c4.py` | faithful full-word c4 VM (the native ./c4 reference) |
| `validate_ext.py` | Stage-1/2 CPU pre-flight: gcc + compiler + native (no transformer) |
| `run_battery_ext.py` | the full three-stage durable battery |
| `CONFORMANCE_MATRIX_EXT.json` | per-case results + divergence-class tallies |

## Running

Pre-flight (fast, CPU-only — must be green before the transformer run):
```
PYTHONPATH=<c4_release> python id_port/c90_e2e/validate_ext.py
```

**Flash battery (`run_battery_flash.py`, the GPU sparse-streaming + FLASH build — this
is the one that reaches 112/112).**  The original `run_battery_ext.py` ran the CFM on
CPU with masked-full O(S²) attention and stopped at 61/112 (OOM / minutes-per-step on
the long control/loop/function cases).  `run_battery_flash.py` fixes all three of the
#839 divergence classes and makes the long cases tractable:
```
CUDA_VISIBLE_DEVICES=0,1 C4_PF_CFM=1 C4_CMP32=1 C4_CMP32_ORDER=1 C4_MEM_ADDR_BITS=18 \
  C4_EXACT_EVICT=1 C4_MEM_EFF=500000 C4_GLOBAL_ADDR32=1 C4_FLASH_ATTN=1 C90_ORDER=fast \
  PYTHONPATH=<c4_release> python id_port/c90_e2e/run_battery_flash.py
```
The three fixes over #839:
1. **FLASH ATTENTION** (`C4_FLASH_ATTN=1`): byte-exact O(S)-memory softmax1+ALiBi (SDPA
   mem-efficient un-cached full case; Triton online-softmax1 cached/windowed case).
   softmax1 == plain-softmax over a BOS-sink column (§40-44), so a standard tiled flash
   kernel + the sink recovers it EXACTLY (verified worst 3.2e-5 vs the masked-full path,
   `verify_flash_byte_exact.py`).  Wired into `blogspec_model.Attn.forward` +
   `SparseAttn.forward`; the compact SPARSE-STREAMING model runs CSR-resident on GPU
   (~0.02-0.1 GB VRAM) so the whole battery is well under the 25 GB floor.
2. **`bytecode_to_isa` /8 SLOT re-encode** for the transformer code (the doom aligned-
   oracle isomorphism): the compiler emits LEA/ENT/ADJ immediates as 8-byte-cell BYTE
   offsets; the transformer's ABI is 4-byte cells with a *4 LEA/ENT/ADJ scale, so the
   immediate must be re-encoded byte-offset→slot (imm//8), exactly as the 1096 corpus +
   doom do.  This dissolves the #839 "fn_call ABI wall" — it was a REFERENCE/feeding bug
   (the raw byte-offset code was fed to the *4 ABI), NOT a transformer gap.  `native_c4`
   keeps its own consistent 8-byte-cell / byte-offset ABI (already 112/112 == gcc).
3. **`C4_GLOBAL_ADDR32=1`** (#838): widen the LI/LC load-address CAM query to 32 bits so
   a global / string-literal pointer at data-segment 0x10000+ recalls its own store
   instead of aliasing addr&0xFF → the ZFOD sink.  Fixes global_rw/global_two +
   str_first_char/str_literal_index.

- `C90_ORDER=fast` runs untested cases fastest-first (by native step count).
- `C90_RESUME_FROM=<snapshot.json>` reuses prior clean results across restarts.
- `C90_CAP_MAX=N` bounds the per-case transformer step cap (default 3000); the cap is
  `min(C90_CAP_MAX, native_steps*3 + 80)`.  fn_recursion_fib (3185 native steps) needs
  `C90_CAP_MAX>=4000`.
- Memory guard: aborts if `MemAvailable < 25 GB` (never `load_sparse_transformer`).

`sizeof_int` is the ONE documented c4-vs-x86 gap (c4 `sizeof(int)==8` vs x86 4): the
transformer + native agree at 8, gcc-on-x86 is 4 — EXPECTED c4 behaviour, not a fail.

## Compiler subset (what the cases can and cannot use)

Verified against `src/compiler.py`. SUPPORTED: `int`, `char`, `enum`, pointers
(`* &`), `if/else`, `while`, `return`, `sizeof`, functions (recursion + mutual via
forward decl), globals, locals, all binary ops (`+ - * / % & | ^ << >> == != <
> <= >= && ||`), `?:`, `=`, unary `-/*/&`, array indexing `p[k]`, `++/--`, string
literals, `printf`, `malloc/free/memset` (stdlib .c4). NOT SUPPORTED: `for`,
`do-while`, `switch`, `goto`, `break`, `continue`, `struct`, `union`, bitfields,
`static`, `extern`, array declarations `T a[N]`, compound assignment (`+=`),
function pointers, varargs. Out-of-subset features are exercised via the subset
equivalent (for→while, switch→if/else chain, arrays→malloc) and flagged here.

## Divergence classes (transformer fix backlog)

A transformer miss is classified so it feeds a fix backlog (this battery CLASSIFIES,
it does not FIX — that is #838/follow-on scope):

- `data-segment-recall` — a global/string literal at data address 0x10000+ is not
  recalled by the memory CAM under `C4_MEM_ADDR_BITS=18` (the original 15-case
  battery read globals via `C4_GLOBAL_ADDR32`, which is NOT in this flag set).
- `ref-abi-divergence` — a string-literal pointer read where even the shipped 8-bit
  `ref_interpret` (the transformer's own 4-byte-cell oracle) diverges from faithful c4.
- `abi-wall-4byte-cell` — the transformer's 4-byte-cell / `*4`-LEA frame ABI cannot
  pass function ARGUMENTS (the compiler emits 8-byte-cell / byte-offset frames);
  locals-only frames coincide, arg-passing calls do not.
- (reserved) `width/ALU-32bit`, `cmp/EQ-decode`, `shift-width`, `CAM-recall`,
  `decode-fidelity/other`, `tf-timeout/non-halt`, `tf-exception`.

## Golden gate

Every file here is a pure ADDITION under `id_port/c90_e2e/`; no production /
weight-authoring file is touched, so the golden build fingerprint `069cc32f`
(flags-off) is byte-identical by construction.
