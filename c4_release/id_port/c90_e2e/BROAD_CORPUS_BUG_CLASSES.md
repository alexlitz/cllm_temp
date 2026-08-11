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
| IN-SUBSET PASS (baseline, this task's env) | 210/282 (74.5%) |
| **IN-SUBSET PASS (after C19 goto + MISC decl fixes, task #880)** | **215/282 (76.2%)** |

The 117 hand cases stay at 100% in-subset (117 hand -> now 122 with 5 added
regression cases, 118/118 in-subset PASS, 0 regression).  The ~76% aggregate is
the c-testsuite portion pulling the transpiler onto un-lowered constructs.
(The out-of-subset bucket grew 28->34 once the runner correctly classified the
hand corpus's own fnptr/varargs cases as by-design boundary.)

### task #880 delta (MISC cluster + C19 goto edge forms) — measured

+5 in-subset (210 -> 215), 0 regression (case-by-case JSON diff of the full 337
corpus: every previously-PASS case still PASSes; only the 5 target cases flipped
compile_error/MISMATCH -> PASS).  The `goto` category is now 7/7 (0 fail).  Fixed
in `c4_doom/id_port/transpile.py`:

* **C19 goto edge forms (3)** — `cts_00010` (dead + consecutive labels
  `start:`/`next: foo:` that are not goto targets), `cts_00199` (a `goto` inside
  a bare `{ }` block did not skip the rest of THAT block), `cts_00207` f1 (a
  label that is the immediate body of an unbraced `if(0) label: stmt` was not
  recognised and would orphan the header / wrongly fall through).
* **MISC declaration edge forms (2)** — `cts_00096` (`int x, x=3, x;` tentative
  redeclarations collapse to one decl + deferred init), `cts_00121`
  (`int f(int a), g(int a), a;` mixed prototype/variable decl drops the
  prototypes, keeps `int a;`).

Doom no-regression: the goto byte-exact suite (`verify_goto.py`) is 6/6 with
IDENTICAL shas; the whole-Doom transpile is byte-identical except a benign
goto-guard split in the 4 goto modules (f_finale/p_enemy/p_map/r_bsp), where a
rewritten-goto block `{ st=K; jmp=1; }` now guards each assignment
(`{ if(jmp==0){st=K;} if(jmp==0){jmp=1;} }`) — semantically identical since jmp==0
on block entry, and the 6/6 runtime byte-exact suite confirms it.

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

### C15 — global struct-VALUE initialization   **P1**
`struct S s = {1,2};` at file scope is left completely unlowered: the tagged def
survives, the brace-init is not deferred into `c4_static_init`, and `s.a`/`s.b`
never lower.  B6 handled scalar/array global init but not struct-VALUE globals.
```c
struct S { int a; int b; }; struct S s = {1, 2};
int main(){ return (s.a==1 && s.b==2) ? 0 : 1; }           /* gcc 0, port CERR */
```
Fix path: extend the B2 struct-value machinery + B6 deferred-init to global
struct-value decls (erase def, allocate/emit field words, lower access).
Affects: `cts_00047 cts_00049 cts_00050 cts_00118 cts_00146 cts_00153` (+ nested).

### C17 — anonymous-inline LOCAL struct / union   **P1**
`struct { int x; int y; } s;` (tagless, no typedef) as a local: the transpiler
half-lowers it (hoists `int x; int y;` but leaves `struct { } s;` and unlowered
`s.x`) because the per-file struct contract does not capture the tagless inline
aggregate.  Same for `union { int a; int b; } u;`.
```c
int main(){ struct { int x; int y; } s; s.x=3; s.y=5; return s.y-s.x-2; }  /* gcc 0, port CERR */
```
Fix path: give `lower_local_struct_values` / `lower_local_unions` an
anonymous-aggregate offset table synthesized from the inline `{...}` body.
Affects: `cts_00017 cts_00042 cts_00043 cts_00046`.

### C19 — goto-label lowering edge forms  ✅ FIXED (task #880)
The goto->state-machine rewrite left some labels in the output (`start:`,
`foo:`) that the c4 front-end (no labels) rejects, one case (`cts_00199`) took a
wrong fallthrough after a `goto`, and one (`cts_00207` f1) had a label as an
unbraced-`if` branch body.
```c
int main(){ start: goto next; return 1; success: return 0;
            next: foo: goto success; return 1; }             /* gcc 0, was CERR 'Undefined: start' */
```
**FIXED** (`c4_doom/id_port/transpile.py`), three independent root causes:
1. `_strip_dead_labels` — erase any `LABEL:` that no `goto` targets (dead /
   consecutive labels `start:`, `next: foo:`); a shared lookbehind
   `_LABEL_DEF_RE` recognises labels after `)` (unbraced branch) and after `:`
   (consecutive).  Fixes `cts_00010`.
2. `_guard_stmt_recursive` bare-`{}`-block case — a `goto` inside a bare block
   now skips the rest of THAT block (was: only the segment's top level guarded).
   Fixes `cts_00199`.
3. `_brace_labelled_branches` — a label that is the immediate body of an unbraced
   `if/while/for` (`if(0) label: stmt`) is braced so the existing nested-label
   lift produces the correct conditional-body semantics.  Fixes `cts_00207` f1.

HIGH regression risk was real (goto is byte-exact in Doom).  Verified: 4 hand
goto + 3 new C19 regression cases (`gt_dead_and_consecutive_labels`,
`gt_into_block_skip_rest`, `gt_label_in_if_branch`) PASS, and the Doom goto
byte-exact suite (`verify_goto.py`) is 6/6 with IDENTICAL shas.  Was:
`cts_00010 cts_00199 cts_00207`.

### C18 — missing libc functions   **P2**
`calloc`, `strcpy`, `strncpy`, `strcmp`, `sprintf` are not in the c4 stdlib, so
`compile_c` rejects them ("Undefined function").  `malloc`/`free`/`printf` exist.
```c
#include <string.h>
int main(){ char a[10]; strcpy(a,"hi"); return a[0]; }       /* port CERR 'Undefined function: strcpy' */
```
Fix path: add C4 stdlib implementations (they are trivial byte loops) OR a
transpiler shim.  Affects: `cts_00040 cts_00179 cts_00180 cts_00186`.

### C11 — preprocessor conditionals (`#if/#elif/#ifdef/#ifndef`)   **P2/OOB**
The transpiler does NOT run a full C preprocessor; `#if 0/#elif 1` branches and
`#ifdef FOO` gating are not evaluated, so guarded decls vanish -> `Undefined: x`.
The Doom port feeds ALREADY-PREPROCESSED source, so this is out-of-subset for the
source-flattener, but the c-testsuite ships raw sources.
```c
#if 0
X
#elif 1
int x = 0;
#endif
int main(){ return x; }                                       /* port CERR 'Undefined: x' */
```
Affects: `cts_00062 cts_00063 cts_00068 cts_00069 cts_00070 cts_00071 cts_00074`.

### C12 — function-like macros (`#define ADD(X,Y) ...`)   **P2/OOB**
Object-like `#define` is handled; function-like macros are not expanded, so
`ADD(1,2)` becomes a call to an undefined function.  Same preprocessor boundary
as C11.
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

### C16 — C99 designated initializers (`.field=` / `[i]=`)   **OOB**
`struct S s = {.b=2,.a=1};` and `arr[2]={[1]={3,4},[0]={1,2}}` are **C99**, not
C90; gcc -std=c90 accepts them as an extension but they are out of the stated
C90 subset.  Document, don't fix.
Affects: `cts_00048 cts_00148`.

### C14 — wide char / string literals (`L'x'` / `L"..."`)   **OOB**
Wide literals are out-of-subset (the c4 VM is byte-oriented).
Affects: `cts_00098`.

### C21 — MISC cluster (assorted declaration / preprocessor / literal)  (task #880)
The `misc` category's in-subset failures resolve into TWO genuine transpiler bugs
(now FIXED) and SIX out-of-subset (preprocessor / wide-char) cases (documented,
NOT fixed — they need a real C preprocessor or a wide-char VM, both out of the
stated C90-subset / Doom-source-flattener boundary).

* ✅ **`cts_00096` — repeated-name tentative global** `int x, x = 3, x;`.  Legal
  C90 tentative definitions; the c4 front-end rejects the redeclaration.  FIXED:
  `split_multi_declarator_globals` now collapses repeated declarator names to one
  `int x;` + a deferred `x = 3;` init.  Narrow: the distinct-name-with-init form
  (`int a=1, b=2;`) keeps the original verbatim behaviour, so whole-Doom transpile
  stays byte-identical (am_map static-local hoist unchanged).
* ✅ **`cts_00121` — mixed prototype + variable global decl**
  `int f(int a), g(int a), a;`.  FIXED: `split_mixed_global_prototype_decls` drops
  the prototype declarators (c4 needs no forward decls; `reorder_functions` orders
  the definitions) and keeps the variable as `int a;`.
* **`cts_00065 cts_00066 cts_00122` — function-like macros** (`#define ADD(X,Y)`,
  empty-arg `F(,1)`).  **OOB (C12)** — needs a real preprocessor.
* **`cts_00141` — `##` token-paste macro**.  **OOB (C12)**.
* **`cts_00071` — `#undef` + `#ifdef` gating**.  **OOB (C11)**.
* **`cts_00098` — `L'\0'` wide-char literal**.  **OOB (C14)** — the c4 VM is
  byte-oriented.

### C20 — semantic MISMATCHes (not CERR)
* `cts_00184` — `printf("%d", sizeof(char))` prints `8` (c4 word) vs `1` (x86).
  This is the documented **sizeof-gap**, a false-fail; the loader should tag it
  (it returns the sizeof via stdout, not exit, so the exit-based gap tag misses
  it).  **Not a transpiler bug.**
* `cts_00171` — `NULL` in a printed string context renders `0`; a `#define NULL`
  / string-literal detail (P2, preprocessor-adjacent).
* `cts_00206` — uses `signal`/`abort` (SIGABRT exit 12); libc signal, **OOB**.
* `cts_00199` — ✅ FIXED (task #880, see C19: goto inside a bare block now skips
  the rest of that block).  `cts_00215` — genuine switch fallthrough lowering
  divergence (see switch); **P1**.

## Prioritized remaining list (for follow-up agents, one class each)

1. **C15 global struct-value init** (P1, ~6 cases) — extend B2/B6 to file-scope
   struct-value decls.
2. **C17 anonymous-inline local struct/union** (P1, ~4 cases) — synthesize an
   offset table for the tagless inline aggregate.
3. ~~**C19 goto-label edge forms**~~ ✅ FIXED (task #880) — dead/consecutive
   labels, goto-in-bare-block, label-as-unbraced-if-branch; Doom goto byte-exact
   6/6 unchanged.
4. **C18 missing libc** (P2, ~4 cases) — add `str*`/`calloc`/`sprintf` to the c4
   stdlib.
5. **C11/C12 preprocessor** (P2/OOB, ~13 cases) — a real `#if`/`#elif`/function-
   macro pass, OR reclassify as out-of-subset (Doom feeds preprocessed source).
6. **C13/C14/C16/C20-signal** — out-of-subset; document as the honest boundary.

The clean P0 (do-while) and P1-goto (C19) are fixed, plus the two clean MISC
declaration bugs (C21: `cts_00096` tentative-redecl, `cts_00121` mixed
prototype/var).  Everything else in-subset is P1 (multi-part struct lowering) or
P2 (preprocessor/libc).  The honest out-of-subset boundary is unchanged: function
pointers (B3), user varargs (B9), float/double, long-long, wide chars, C99
designated inits, signal/setjmp libc, and a full C preprocessor.
