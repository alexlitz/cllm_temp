# Self-hosting-compiler baseline for the #848/#853 transformer route

*2026-08-06. CPU-only investigation (no transformer build, no GPU, no code
changes). Branch `consolidate-0.5b-2026-07-22` @ `deca9567`. Golden untouched.
Native `./c4` oracle built at `/tmp/selfhost_probe/c4` via
`gcc -O2 -static-libgcc -o c4 old/c4_original.c`.*

This doc answers one question precisely: **what is the real, correct
self-hosting C compiler that the a61d336a transformer route (#848/#853) should
aim at, and does a form of it that runs on the CPU c4 VM exist?** It extends
`docs/CAPSTONE_TOOLCHAIN_STATUS.md`, which audited `bundler/c4_compile.c` (a
limited transpiler) and `src/compiler.py` (Python) but **never tested
`old/c4_original.c`** — the classic rswier c4.c, the only genuine
self-hosting full C compiler in the tree.

## TL;DR verdict

| Question | Answer |
|---|---|
| Is there a *genuine* self-hosting C compiler (functions + recursion)? | **YES — `old/c4_original.c`** (rswier c4.c). It self-hosts *natively* via gcc: `./c4 c4.c c4.c hello.c` (triple) prints `hello, world`; recursion (`fib`) works. Handles functions/recursion, unlike `bundler/c4_compile.c` (which core-dumps on any user function). |
| Can it be compiled to c4 bytecode by the repo's C→c4 compiler? | **YES.** `src/compiler.py` (`compile_c`) compiles the whole of c4.c to **6890 bytecode words** after a mechanical, shallow preprocess (3 grammar gaps below — none deep). |
| Can that bytecode **run and self-host on the CPU c4 VM**? | **NO — hard wall at VM I/O, not at compilation.** The compiled c4.c uses `OPEN×2 READ×2 CLOS×2 PRTF×51`; neither CPU VM supports the combination c4.c needs (call frames **and** file I/O **and** output). It runs 47 steps on `native_c4.run`, hits the unimplemented `open()`, takes c4.c's "could not open" error path, returns −1. It never reaches the compiler loop. |
| Is #848/#853 reachable at the source level? | **Reachable in principle; blocked today at the CPU-VM I/O layer, not at the compiler.** The compiler + C→c4→VM pipeline is byte-exact and real (112/112 C90, incl. recursion); the missing piece is a CPU c4 VM with **JSR/ENT/LEV *and* OPEN/READ/CLOS *and* a visible PRTF channel** all in one interpreter. No such VM exists in the tree today. |

**One-line:** The correct self-hosting target is `old/c4_original.c` compiled by
`src/compiler.py`; it compiles clean to c4 bytecode, but no single CPU c4 VM in
the repo can run it end-to-end (the frame-capable VM has no working file-I/O /
output; the I/O-capable VM has no call frames). The capstone loop is reachable
in principle — the wall is a ~one-VM-feature gap (wire OPEN/READ/CLOS + a real
PRTF into `native_c4.run`), **not** a compiler or source-level impossibility.

---

## 1. The genuine self-hosting compiler: `old/c4_original.c`

`old/c4_original.c` is Robert Swierczek's c4.c verbatim (555-ish lines,
`#define int long long`, 4 functions `next/expr/stmt/main`, its own
LEA…EXIT opcode VM). It is a **real** self-hosting C compiler.

Native evidence (gcc oracle at `/tmp/selfhost_probe/c4`):
```
./c4 hello.c            -> hello, world           exit(0) cycle=9
./c4 fib.c              -> fib(10)=55             (recursion works)
./c4 c4.c hello.c       -> hello, world           (self-host: c4 runs c4 runs hello)
./c4 c4.c c4.c hello.c  -> hello, world           (TRIPLE self-host)
./c4 c4.c fib.c         -> fib(10)=55             (self-host + recursion)
```
This is the classic self-host chain. It handles user-defined functions and
recursion — **the exact things `bundler/c4_compile.c` core-dumps on** (confirmed
in `CAPSTONE_TOOLCHAIN_STATUS.md` §1). So `c4_compile.c` is NOT the self-hosting
target; `old/c4_original.c` is.

## 2. It compiles clean under `src/compiler.py` (the only C→c4→VM path)

`src/compiler.py` (`compile_c`) is the repo's real function/recursion-handling
C→c4 compiler (proven: **112/112 C90 battery** byte-exact vs gcc-15 on
`native_c4.run`, incl. `fn_recursion_fib/fac/sum`, pointers, arrays — re-verified
green this session). `bundler/c4_compile.c` native can't self-compile; the Python
compiler is the path whose output can then run on the CPU VM.

`compile_c` compiles the *whole* of c4.c to **6890 words** after this mechanical
preprocess (all shallow, all a real C preprocessor / a small grammar patch would
subsume):

1. **Preprocessor** — drop `#include` / `#define int long long` lines
   (`src/compiler.py` has no preprocessor). c4.c's own `next()` skips `#`
   lines too, so this is cosmetic.
2. **`void`** — `src/compiler.py`'s keyword table lacks `void`; rewrite
   `void`→`int` (c4.c's functions are effectively `int`-returning).
3. **Comma-separated pointer *locals*** — `int t, *d;` fails
   ("Expected ID, got MUL"). The **local**-decl parser doesn't re-read a
   per-declarator `while(tk==Mul)` star; the **global**-decl parser
   (`parse_global_decl`) already does. ~5 lines in c4.c; split them
   (`int t; int *d;`). *This is the one genuine, tiny compiler bug — a
   ~5-line fix to mirror the global path into the local path.*
4. **Adjacent string-literal concatenation** — `"a" "b"` (the keyword table)
   isn't concatenated by the lexer; join them.

After that, `compile_c(..., link_stdlib=True)` succeeds: **6890 words, 2064 data
bytes**; `memset/memcmp` link as stdlib JSR functions (works). No deep semantic
wall — the compiler swallows the entire self-hosting compiler.

## 3. The wall: no CPU c4 VM can run it end-to-end

The compiled c4.c's opcode histogram (measured):
```
OPEN=2  READ=2  CLOS=2  PRTF=51   JSR=150  ENT=8  LEV=31  ADJ=129  MALC=23
```
To self-host it needs, in ONE interpreter: (a) call frames JSR/ENT/ADJ/LEV
(recursive `expr()`), (b) file I/O OPEN/READ/CLOS (read the input `.c`),
(c) a visible PRTF output channel. The two CPU c4 VMs split these:

| CPU VM | JSR/ENT/LEV/ADJ | OPEN/READ/CLOS | PRTF (visible) | MALC/MSET/MCMP |
|---|---|---|---|---|
| `id_port/c90_e2e/native_c4.py` `run` (full-word) | **YES** | **NO** | **no-op** (line 159) | YES |
| `c4_min/isa.py` `interpret` (8-bit ref) | **NO** (no JSR/ENT/LEV/ADJ dispatch) | READ yes / OPEN,CLOS **NO** | YES | **NO** |

- `native_c4.run` is the frame-capable VM the C90 battery uses, but `PRTF` is an
  explicit no-op and there is **no OPEN/READ/CLOS and no argv/argc setup**.
- `isa.interpret` has READ + a real PRTF, but its dispatch has **no
  JSR/ENT/ADJ/LEV** at all — it cannot execute a single function call, so it
  cannot run c4.c (or any multi-function program).

**Direct run evidence** — compiled c4.c on `native_c4.run`:
```
RAN on native_c4: AX=4294967295 (=-1)  steps=47
```
c4.c's `main` does `if ((fd = open(*argv,0)) < 0) { printf("could not open..."); return -1; }`.
`open` is unimplemented (falls through), so `fd` is not a valid descriptor, the
guard fires, and main returns −1 in 47 steps — **before the compiler loop ever
starts**. The wall is 100% the VM's missing file-I/O / output, *not* the
compiler and *not* the source.

## 4. Reachability verdict for #848/#853 (source level)

- **Reachable in principle.** The full self-hosting compiler exists
  (`old/c4_original.c`), compiles to clean c4 bytecode via the proven
  `src/compiler.py` path (6890 words), and its control-flow (JSR/ENT/LEV/ADJ)
  is already executable on `native_c4.run`. Nothing at the *source* or
  *compiler* level blocks the loop.
- **Blocked today at the CPU-VM I/O layer.** No single interpreter in the tree
  has {call frames} ∪ {OPEN/READ/CLOS} ∪ {visible PRTF}. Closing the loop needs
  a CPU c4 VM that wires OPEN/READ/CLOS + a real PRTF (and argv/argc startup)
  into the *frame-capable* `native_c4.run` (or adds JSR/ENT/LEV to the
  *I/O-capable* `isa.interpret`). This is a bounded VM-feature addition, not a
  research wall.
- **Perf is a *second*, separate wall for the neural route** (the transformer
  running the compiled c4.c step-for-step): even the tiny-instance
  self-host extrapolation in `docs/SELFHOST_3LAYER_FEASIBILITY.md` exceeds a
  year of neural-VM wall-clock. So even once the CPU VM can self-host c4.c, the
  #853 "transpiler on the transformer" capstone inherits that measured perf
  wall on top of the I/O-VM gap.

**Net:** The #848/#853 target should be `old/c4_original.c` → `src/compiler.py` →
a (to-be-unified) full-capability CPU c4 VM. At the *source* level the capstone
is reachable; the concrete blocker is the missing single-VM combination of call
frames + file I/O + output, followed by the known neural perf wall.

## Where each artifact lives (absolute paths)

- Genuine self-hosting compiler: `/home/alexlitz/Documents/misc/c4_release/old/c4_original.c`
  (built to `/tmp/selfhost_probe/c4`).
- C→c4 compiler (the runnable path): `c4_release/src/compiler.py` (`compile_c`).
- Frame-capable CPU VM (no I/O): `c4_release/id_port/c90_e2e/native_c4.py` (`run`).
- I/O-capable CPU VM (no frames): `c4_release/c4_min/isa.py` (`interpret`).
- Limited transpiler (NOT self-hosting — core-dumps on functions):
  `c4_release/bundler/c4_compile.c`.
- Prior capstone audit (this doc extends it): `c4_release/docs/CAPSTONE_TOOLCHAIN_STATUS.md`.
- Perf wall: `c4_release/docs/SELFHOST_3LAYER_FEASIBILITY.md`;
  `c4_release/c4_min/selfhost/run_selfhost_feasibility.py`.

## Reproduce

```
# native self-host (the real target proves itself)
gcc -O2 -static-libgcc -o /tmp/c4 old/c4_original.c
/tmp/c4 old/c4_original.c old/c4_original.c hello.c    # triple self-host

# compile c4.c under the repo compiler + run on the CPU VM (hits the I/O wall)
#   (preprocess: drop #-lines, void->int, split `int t,*d;` locals, join "a""b")
PYTHONPATH=c4_release python3 -c "..."   # see §2/§3; 6890 words, AX=-1 @ 47 steps
```
