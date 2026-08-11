# Broad-corpus transpiler bug classes (beyond B1..B9)

The broad C90 corpus (117 hand + 220 c-testsuite = 337 cases, see
`BROAD_CORPUS.md`) exposes transpiler conformance gaps the hand-tuned 117-case
set (95.7%) never hit.  This is the case-by-case debugging foundation (task #880):
each class below has a minimal reproducer and a priority so a follow-up agent can
take one class.

Oracle = `gcc -m32 -std=c90` (exit%256 + stdout).  Port = `transpile.py ->
compile_c -> c4vm`.  The c4 front-end supports ONLY `if/else/while/return/sizeof/
enum` + `int`/`char`; EVERYTHING else (`for`, `do`, `switch`, `goto`, `break`,
`continue`, `long`/`short`/`unsigned`, structs, unions, ...) must be LOWERED by
`transpile.py`.  That is why a broad corpus stresses the transpiler so hard.

## Headline matrix (measured)

| bucket | count |
|--------|------:|
| corpus attempted | 337 |
| gcc-rejected (c99/c11-only, needs-64-bit — no valid oracle) | 23 |
| out-of-subset (B3 fnptr / B9 varargs / float / long-long, by design) | 34 |
| **IN-SUBSET judged** | **280** |
| IN-SUBSET PASS (pre-fix) | 207/286 (72.4%) |
| IN-SUBSET PASS (after C10 do-while fix) | 208/280 (74.3%) |
| IN-SUBSET PASS (measured baseline, task #880 STORAGE) | 210/282 (74.5%) |
| **IN-SUBSET PASS (after C11 preproc-cond + C16 designated-init fixes)** | **220/282 (78.0%)** |

The 117 hand cases stay at 113/113 in-subset = 100% (0 regression).  The ~78%
aggregate is the c-testsuite portion pulling the transpiler onto un-lowered
constructs.  (The out-of-subset bucket grew 28->34 once the runner correctly
classified the hand corpus's own fnptr/varargs cases as by-design boundary.)

STORAGE cluster (task #880): 8 of the 9 in-subset storage failures fixed
(+10 overall — the C11 preprocessor evaluator also fixed cts_00071/cts_00188 in
other categories).  Doom linked transpiled output BYTE-IDENTICAL (0 regression).
Remaining storage fail = cts_00201 (function-like macro token-paste, C12/OOB).

## NEW classes (C10..C21) — minimal reproducers + priority

Priority key: **P0** clean in-subset transpiler bug (fix next) · **P1** in-subset
but multi-part (struct-engine / goto state-machine) · **P2** needs a real
preprocessor/libc (arguably out-of-subset for a Doom source-flattener) · **OOB**
genuinely out-of-subset (document, don't fix).

### C10 — do-while (unbraced)  ✅ FIXED
`do stmt; while(c);` (no braces) left the `do` keyword verbatim.  `lower_do_while`
only handled the braced form.  **FIXED** (c4_doom `transpile.py`): extended the
body extractor to the unbraced single-statement form via `_find_stmt_end`.
Reproducer `cts_00008`:
```c
int main(){ int x; x=50; do x=x-1; while(x); return x; }   /* gcc 0, port now 0 */
```
Affects: `cts_00008 cts_00101`.  Both now PASS.

### C15 — global struct-VALUE initialization   ✅ FIXED
`struct S s = {1,2};` at file scope was left completely unlowered: the tagged
def survived, the brace-init was not deferred into `c4_static_init`, and
`s.a`/`s.b` never lowered.  B6 handled scalar/array global init but not
struct-VALUE globals.
```c
struct S { int a; int b; }; struct S s = {1, 2};
int main(){ return (s.a==1 && s.b==2) ? 0 : 1; }           /* gcc 0, port now 0 */
```
**FIXED** (c4_doom `transpile.py` #880): `lower_struct_array_initializers` now
matches tagged `struct TAG`/bare-alias types (was `_t`-only) AND treats a SCALAR
struct value (`struct S s = {..}`) as ONE element placed at field offsets, not
a mis-strided array (the old path landed field 2 at offset SIZE — also a latent
bug for doom's cheatseq_t/event_t/menu_t globals, now corrected).
Affects (now PASS): `cts_00047 cts_00091 cts_00118 cts_00146` (+ the nested
tagged case `cts_00106`).  Still OOB: `cts_00048 cts_00049 cts_00050 cts_00148
cts_00149 cts_00150 cts_00153` — C99 designated-init / compound-literals / C11
anon-members / preproc member-rename (see C16 below + the OOB roster).

### C17 — anonymous-inline LOCAL/GLOBAL struct / union   ✅ FIXED
`struct { int x; int y; } s;` (tagless, no typedef) as a local: the transpiler
half-lowered it (hoisted `int x; int y;` but left `struct { } s;` and unlowered
`s.x`) because the per-file struct contract did not capture the tagless inline
aggregate.  Same for `union { int a; int b; } u;` and the global form.
```c
int main(){ struct { int x; int y; } s; s.x=3; s.y=5; return s.y-s.x-2; }  /* gcc 0, port now 0 */
```
**FIXED** (c4_doom `transpile.py` #880): new `lower_anonymous_aggregates` pass
rewrites each tagless inline aggregate variable decl to a synthetic TAGGED form
(`struct __anonvar_N { .. }; struct __anonvar_N s;`) + a contract layout (union
= all members @ offset 0), so the ordinary tagged-struct machinery lowers it.
A UNION with ARRAY members (doom's w_wad/i_video byte-overlay) is left to
`lower_local_unions` (byte-identity of the doom build preserved).  A block-scoped
tagged struct SHADOW (`struct T{int y;}s2;` redefining a differently-shaped `T`)
also gets its own synthetic layout.
Affects (now PASS): `cts_00017 cts_00042 cts_00043 cts_00044 cts_00053`.
Still OOB: `cts_00046 cts_00050` — C11 ANONYMOUS (nameless) members, not C90.

Also FIXED alongside the struct cluster: `fold_float_literals` no longer eats
the member dots of a struct chain (`s2.s1.x` tokenised as `2.`/`1.` and lost its
dots -> `s2s1x`); forward struct declarations `struct TAG;` are erased.

### C19 — goto-label lowering edge forms   **P1**
The goto->state-machine rewrite leaves some labels in the output (`start:`,
`foo:`) that the c4 front-end (no labels) rejects, and one case (`cts_00199`)
takes a wrong fallthrough after a `goto`.  The label-lift misses labels that are
immediately targets / at function tail / consecutive.
```c
int main(){ start: goto next; return 1; success: return 0;
            next: foo: goto success; return 1; }             /* gcc 0, port CERR 'Undefined: start' */
```
Fix path: harden `_lift_one_label` / `rewrite_gotos` for tail + consecutive
labels; audit the fallthrough guard.  HIGH regression risk (goto ×4 hand cases
pass) — test carefully.  Affects: `cts_00010 cts_00199 cts_00207`.

### C18 — missing libc functions   **P2**
`calloc`, `strcpy`, `strncpy`, `strcmp`, `sprintf` are not in the c4 stdlib, so
`compile_c` rejects them ("Undefined function").  `malloc`/`free`/`printf` exist.
```c
#include <string.h>
int main(){ char a[10]; strcpy(a,"hi"); return a[0]; }       /* port CERR 'Undefined function: strcpy' */
```
Fix path: add C4 stdlib implementations (they are trivial byte loops) OR a
transpiler shim.  Affects: `cts_00040 cts_00179 cts_00180 cts_00186`.

### C11 — preprocessor conditionals (`#if/#elif/#ifdef/#ifndef`)  ✅ FIXED
`#if 0/#elif 1` branches and `#ifdef FOO` gating were not evaluated: guarded
decls either LEAKED their raw body (`XXX` -> c4 SyntaxError) or the false branch
was kept unconditionally (`strip_if_zero_blocks` only special-cased literal
`#if 0`).  **FIXED** (c4_doom `transpile.py` `eval_pp_conditionals`): a
top-to-bottom preprocessor-conditional evaluator tracks the live `#define`/
`#undef` table AS IT SCANS (a `#define` can gate a LATER `#ifdef`), evaluates
`#if`/`#elif` constant-expressions (`defined()` + macro substitution + the C
arithmetic/boolean operator subset; undefined identifier -> 0), and drops
dead-branch bodies.  Runs BEFORE object-like macro inlining so `#if X` sees the
raw macro.  Unevaluable `#if`/`#elif` default TRUE (keep body) -> a real-Doom
platform guard is never silently dropped; the Doom linked output is BYTE-IDENTICAL.
```c
#if 0
X
#elif 1
int x = 0;
#endif
int main(){ return x; }                                       /* gcc 0, port now 0 */
```
Affects: `cts_00062 cts_00063 cts_00068 cts_00069 cts_00070 cts_00074` (+ cross-
category `cts_00071 cts_00188`).  All now PASS.

### C12 — function-like macros (`#define ADD(X,Y) ...`)   **P2/OOB**
Object-like `#define` is handled; MULTI-arg function-like macros (and `##`
token-paste) are not expanded, so `ADD(1,2)` becomes a call to an undefined
function.  (`header_macros` DOES expand Doom's SINGLE-arg function-like macros —
SHORT/LONG/MTOF — so those work.)  `cts_00201` needs 2-arg macros + `##`
token-paste (`#define CAT2(a,b) a##b`): this is the macro-EXPANSION path (NOT
storage/init), and extending `header_macros._expand_fnlike` risks the byte-exact
Doom build, so it is OUT-OF-SUBSET for the STORAGE cluster (task #880) — deferred
to the preprocessor owner.
```c
#define ADD(X,Y) (X+Y)
int main(){ return ADD(1,2)-3; }                              /* port CERR 'Undefined function: ADD' */
```
Affects: `cts_00065 cts_00066 cts_00122 cts_00138 cts_00141 cts_00201`.

### C13 — stdint typedefs (`int32_t`/`int64_t`/...)   **P2/OOB**
`<stdint.h>` types are not mapped to `int`, and `int64_t` + `0xffffffffffffffff`
imply 64-bit width (the c4 word IS 64-bit, so it would diverge from gcc -m32's
32-bit int anyway).  Borderline out-of-subset.
Affects: `cts_00104 cts_00214`.

### C16 — C99 designated initializers (`.field=` / `[i]=`)
The 1-D ARRAY designated form `int a[]={5,[2]=2,3}` / `int arr[3]={[2]=2,[0]=0,
[1]=1}` is now LOWERED ✅ (c4_doom `transpile.py` `_resolve_designated_1d`, #880
storage cluster): `[idx]=val` designators resolve to explicit (index,value)
pairs (a designator sets the running position to idx; a later plain element
lands at idx+1; holes default 0; `[]` size = max-index+1).  gcc -std=c90 accepts
these as an extension and the c-testsuite ships them, so the harness judges them
in-subset.  Affects (fixed): `cts_00092 cts_00147`.

The STRUCT designated form `struct S s={.b=2,.a=1}`, NESTED array-of-struct
`arr[2]={[1]={3,4},[0]={1,2}}`, the compound literal `&(struct S){1,2}`, and C11
ANONYMOUS (nameless) struct/union members (`struct S2 { int a; union { int c;
int d; }; };`) remain **OOB** — all C99/C11, NOT C90; gcc -std=c90 accepts them
as extensions (with -w) but they are out of the stated C90 subset.  Document,
don't fix.  Affects (struct cluster): `cts_00048 cts_00049 cts_00148` (struct
designated init) · `cts_00149 cts_00150` (compound literal + designated init) ·
`cts_00046 cts_00050` (C11 anonymous members).

### C21 — bit-fields (`int f : N;`) + preproc member-rename   **OOB**
`enum tree_code code : 8;` / `unsigned flag : 1;` bit-fields have no representation
in the byte-word c4 VM (no sub-word bit packing).  `cts_00153`'s `#define x f` /
`#define y() f` renames a struct MEMBER through a function-like macro — a real
preprocessor pass (P2), out of scope for a Doom source-flattener.
Affects (struct cluster): `cts_00218` (bit-field) · `cts_00153` (preproc member
macro).  Note `cts_00205` (J-interpreter `PT cases[]`) now COMPILES + RUNS (was
CERR) but MISMATCHes on `%ld`-of-`long` (the c4 word=8 vs x86 `long`=4 sizeof
gap) plus a flat brace-less nested-array `c[4]` initializer — a sizeof-gap +
implicit-flattening residual, not a clean struct-lowering bug.

### C14 — wide char / string literals (`L'x'` / `L"..."`)   **OOB**
Wide literals are out-of-subset (the c4 VM is byte-oriented).
Affects: `cts_00098`.

### C20 — semantic MISMATCHes (not CERR)
* `cts_00184` — `printf("%d", sizeof(char))` prints `8` (c4 word) vs `1` (x86).
  This is the documented **sizeof-gap**, a false-fail; the loader should tag it
  (it returns the sizeof via stdout, not exit, so the exit-based gap tag misses
  it).  **Not a transpiler bug.**
* `cts_00171` — `NULL` in a printed string context renders `0`; a `#define NULL`
  / string-literal detail (P2, preprocessor-adjacent).
* `cts_00206` — uses `signal`/`abort` (SIGABRT exit 12); libc signal, **OOB**.
* `cts_00199 cts_00215` — genuine goto/switch fallthrough lowering divergences
  (see C19 / switch); **P1**.

## Prioritized remaining list (for follow-up agents, one class each)

1. **C15 global struct-value init** — ✅ FIXED (#880).
2. **C17 anonymous-inline local/global struct/union** — ✅ FIXED (#880).
3. **C19 goto-label edge forms** (P1, ~3 cases) — harden label-lift for tail/
   consecutive labels + the fallthrough guard; HIGH regression risk.
4. **C18 missing libc** (P2, ~4 cases) — add `str*`/`calloc`/`sprintf` to the c4
   stdlib.
5. **C11/C12 preprocessor** (P2/OOB, ~13 cases) — a real `#if`/`#elif`/function-
   macro pass, OR reclassify as out-of-subset (Doom feeds preprocessed source).
6. **C13/C14/C16/C20-signal** — out-of-subset; document as the honest boundary.

The clean P0 (do-while) is fixed; everything else in-subset is P1 (multi-part
struct/goto lowering) or P2 (preprocessor/libc).  The honest out-of-subset
boundary is: function pointers (B3), user varargs (B9), float/double, long-long,
wide chars, C99 designated inits, signal/setjmp libc, and a full C preprocessor.
