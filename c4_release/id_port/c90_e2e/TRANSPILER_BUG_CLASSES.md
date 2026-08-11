# Transpiler C90 conformance — bug classes at the SOURCE

Broad-corpus test of the `transpile.py -> compile_c -> c4vm` port pipeline
(the path the Doom port uses) against **117 vanilla-C90 cases** with `gcc -m32`
as the oracle.  Goal: find the transpiler's systematic bug classes at the
SOURCE so they can be fixed once (a clean *vanilla linuxdoom -> generic
transpiler -> byte-exact Doom, no hand-patches*) rather than hand-patched
per-bug in `doom_run.c`.

Run it (CPU-only, no model load):

```
cd <c4_doom>/id_port                       # for transpile.py + c4vm.py + struct_engine
python3 <c4_release>/id_port/c90_e2e/run_transpiler_conformance.py \
        --json TRANSPILER_CONFORMANCE.json
```

## Conformance matrix (measured, 112/117 = 95.7% clean — UPDATED)

After fixing B1/B5/B7/B2/B4/B6/B8 (see below), only the 5 documented
OUT-OF-SUBSET cases remain (B3 function pointers ×4, B9 user varargs ×1).

| category   | pass | mismatch | compileErr | note |
|------------|-----:|---------:|-----------:|------|
| multidim   |  8   |  0       | 0          | true `T a[N][M]` flattened (B1 **FIXED**) |
| ptrarith   |  8   |  0       | 0          | clean |
| structarr  | 11   |  0       | 0          | `_t` + tagged + local-value + non-`_t` (B2 **FIXED**); `st_sizeof` = sizeof-gap PASS |
| compound   | 14   |  0       | 0          | `+= |= &= ...` all lower; `ca_on_field` now PASS (B2) |
| loops      | 12   |  0       | 0          | for/while/do + continue/break clean |
| goto       |  4   |  0       | 0          | clean |
| switch     |  7   |  0       | 0          | mid-list `default:` fallthrough (B8 **FIXED**) |
| bitwise    | 12   |  0       | 0          | signed+unsigned shifts clean |
| fnptr      |  0   |  0       | 4          | function pointers unsupported (B3, OUT OF SCOPE) |
| union      |  5   |  0       | 0          | anon + tagged/`_t` (B4 **FIXED**) |
| enum       |  4   |  0       | 0          | explicit-value inlining + enum-typed decls (B5 **FIXED**) |
| varargs    |  1   |  0       | 1          | `printf` ok; user `va_list`/`va_arg` unsupported (B9, OUT OF SCOPE) |
| recursion  |  4   |  0       | 0          | clean |
| strings    |  6   |  0       | 0          | clean |
| storage    |  9   |  0       | 0          | static-local (B7) + 2D (B1) + string-ptr / multi-init (B6) all **FIXED** |
| programs   |  7   |  0       | 0          | matrix_mul (B1) + linked_list (B2) now PASS |
| **TOTAL**  | **112** | **0** | **5**    | 5 bug-revealing failures, all OUT-OF-SUBSET (B3/B9) |

`transpile_error`/`vm_error` = 0 (the transpiler never crashed and every
compiled program halted).  `mismatch` = 0.

### History
Baseline was 91/117 = 77.8% (below).  The 7 remaining Doom-relevant classes were
fixed incrementally: **B1** true `T a[N][M]` (declaration-shape flatten + row-major
linearize), **B5** enum explicit-value inlining + `enum State s;` decls, **B7**
`static` locals -> mangled file-scope globals (init-once), **B2** tagged/local-value
structs + non-`_t` global aliases, **B4** tagged/`_t` unions (byte-buffer overlay +
`__rd`/`__wr` views incl WRITES), **B6** `char *p="s"` / multi-scalar global init,
**B8** mid-list `default:` fallthrough.  Every fix is byte-exact against the
`gcc -m32 -std=c90` oracle with NO regression in the previously-clean classes.

### Original baseline matrix (91/117 = 77.8%, pre-fix)

| category   | pass | mismatch | compileErr | note |
|------------|-----:|---------:|-----------:|------|
| multidim   |  5   |  3       | 0          | true `T a[N][M]` un-flattened |
| structarr  |  7   |  0       | 4          | `_t` global/ptr works; tagged/local-value fail |
| switch     |  6   |  1       | 0          | mid-list `default:` fallthrough mis-ordered |
| union      |  1   |  0       | 4          | only LOCAL-ANON union; tagged/`_t` fail |
| enum       |  2   |  0       | 2          | explicit-value inlining + enum-typed decls broken |
| storage    |  5   |  2       | 2          | static-local + 2D + string-ptr + multi-init fail |
| programs   |  5   |  1       | 1          | matrix_mul (2D); linked_list (tagged struct) |
| (others: ptrarith 8, compound 13/1cerr, loops 12, goto 4, bitwise 12, fnptr 0/4cerr, varargs 1/1cerr, recursion 4, strings 6) |
| **TOTAL**  | **91** | **7** | **19**    | 26 bug-revealing failures |

## The 7 known (hand-patched) classes — status against the broad corpus

The Doom port hand-patched 7 classes per-bug.  Measured on the current
transpiler, **6 of the 7 are now absorbed into general lowering** (they PASS the
broad corpus in isolation).  Only Class-5 (function pointers) remains, and it is
OUT OF SUBSET by design (Doom uses the FN_ID `__actions__` dispatch contract):

| # | known class | status | evidence |
|---|-------------|--------|----------|
| 1 | compound-index stride `arr[i*k]` / `ptr[i][j]` | **FIXED** | flat `a[k*2+1]` + manual `m[i*3+j]` PASS; true `int a[N][M]` now flattened (`lower_multidim_arrays`): `md_2d_sum md_3d md_row_major sg_2d_global_init pg_matrix_mul` PASS |
| 2 | compound-assign `+= |= &=` dropped to `=` | **FIXED** | `ca_*` 14/14 pass; `lower_compound_assign` works |
| 3 | struct-array stride `&arr[i]->arr[i*N]` | **FIXED** | `st_t_addr_of_elem` / `st_t_ptr` pass; tagged + non-`_t` now general (B2) |
| 4 | for-continue increment inside guard -> spin | **FIXED** | `lp_for_continue*` all pass |
| 5 | function stubbing (return-0 for unhandled fns) | **OUT OF SCOPE** | `fp_*` all CERR — generic function pointers reject; Doom uses FN_ID dispatch |
| 6 | signedness (arith vs logical `>>` on unsigned) | **FIXED** | `bw_shr_signed`, `bw_shr_unsigned` both pass |
| 7 | struct-field i16 width/offset | **FIXED** | `st_t_short_field` / `st_t_char_field` pass |

## Bug classes with minimal reproducers

### B1 — true multi-dimensional array declaration (the live stride bug)
`int a[N][M]` with `a[i][j]` access is **not flattened**.  The malloc sizes only
the OUTER dim, and `a[i][j]` is left as a double-subscript on the flat `int*`
(c4 compiles `a[i][j]` = `*(*(a+i)+j)`, so `a[i]` is read as an *address* -> garbage).

```c
int main(){ int a[2][2]; a[0][0]=1;a[0][1]=2;a[1][0]=3;a[1][1]=4;
            return a[0][0]+a[0][1]+a[1][0]+a[1][1]; }   /* gcc 10, port 14 */
```
Transpiled: `a = malloc((2)*sizeof(int));` (should be `2*2`), and `a[0][0]` kept
verbatim (should be `a[0*2+0]`).  DOOM AVOIDS this by writing manual strides
(`m[i*W+j]` on a malloc'd `int*`, which PASSES) — that's why the port works.
Fix: lower `T a[D0][D1]...` -> `int* a; a=malloc(D0*D1*..*8)` and rewrite every
`a[i][j][...]` -> `a[((i*D1)+j)*D2+...]`.  Affects: `md_2d_sum md_3d md_row_major
sg_2d_global_init pg_matrix_mul`.

### B2 — tagged (`struct X{...}`) + local struct-VALUE variables
The transpiler lowers only the DOOM idiom: `_t`-suffixed typedef alias +
GLOBAL struct array / struct-pointer (`sector_t sectors[]`, `mobj_t *mo`).
It does NOT handle:
  * tagged non-typedef `struct P{...}; struct P p;` (definition not erased, local
    not allocated — the `struct P{...};` def survives to the c4 front-end -> CERR),
  * LOCAL struct-VALUE variables (`P p;` — `infer_rich_var_types` returns `{}` for
    a local struct-value decl, so `p.x` never lowers),
  * non-`_t` typedef aliases as file-scope globals (`collect_global_struct_vars`
    hard-requires the `[A-Za-z_]\w*_t` regex; `P arr[4]` is invisible).

```c
struct P{int x; int y;}; int main(){ struct P p; p.x=3; p.y=4; return p.x+p.y; }
```
Field-OFFSET lowering itself is correct (`p.x`->`p[0]` when a contract exists);
the gap is (a) erasing the tagged def, (b) allocating the local struct value, and
(c) the `_t`-only global detector.  Affects: `st_tag_local st_tag_ptr
st_typedef_local st_sizeof ca_on_field pg_linked_list` (+ all `union` tagged).

### B3 — function pointers (Class-5)
`int (*f)(int); f=dbl; f(x)` rejects at the c4 front-end: the fn-ptr decl lowers
to `int f;`, but `f=dbl` ("Bad identifier: dbl") and `f(x)` ("Not a function")
have no lowering.  DOOM sidesteps this with the `__actions__` FN_ID dispatch
contract (store an integer id, lower `(*p->think)()` to `call_think(id)` if-chains)
— generic function pointers are unsupported.  Affects: `fp_call fp_apply fp_table
fp_dispatch`.

### B4 — tagged / `_t` unions
Only the LOCAL ANONYMOUS union idiom (`union {char s[8]; int x[2];} u;` — DOOM's
w_wad) is lowered (`lower_local_unions`, PASSES).  A tagged `union U{...}` or a
`_t` typedef union with member access is left verbatim -> CERR.  Affects:
`un_int_char un_reuse un_in_struct un_two_view`.

### B5 — enum: explicit values + enum-typed declarations
Two sub-bugs:
  * **explicit-value members not inlined**: `enum E{A=1,B=10,C,D=100}` inlines only
    the AUTO-increment member `C`->11; `A`,`B`,`D` survive as bare identifiers ->
    "Undefined: A".  (Pure auto-increment enums `enum{X,Y,Z}` DO inline — `en_basic`
    and `en_as_index` pass.)
  * **enum-typed variable decls not lowered**: `enum State s;` is kept verbatim ->
    "Unexpected token: enum".  Should become `int s;`.
Affects: `en_explicit en_in_switch`.

### B6 — global-init: string-literal pointer + multiple initialized scalars
  * `char *msg = "XYZ";` — the pointer-to-string-literal global initializer is not
    deferred into `c4_static_init` (left verbatim -> CERR).
  * `int a=1; int b=2; int c=3;` — only the FIRST is lifted; `b`,`c` get swallowed
    as illegal mid-body decls inside `c4_static_init` -> CERR.
(Single `int g=77;` and array `int t[5]={...}` init DO work.)  Affects:
`sg_global_string sg_multi_global`.

### B7 — `static` local variables lose persistence
`static int n = 0;` inside a function is lowered to a plain auto local (`static`
is stripped, init runs every call), so a call counter resets each call.
```c
int counter(){ static int n=0; n++; return n; }   /* returns 1 every call, want 1,2,3 */
```
Fix: hoist `static` locals to file-scope globals with a deferred init.  Affects:
`sg_static_local`.

### B8 — switch: mid-list `default:` that falls through
The switch->if-chain reorders `default` to the END, so a `default:` placed BEFORE
a later `case` and falling through into it loses the fallthrough.
```c
switch(x){ case 1: r=1; break; default: r+=10; case 6: r+=20; }  /* x=5: gcc 30, port 10 */
```
Ordinary switch/`default`/fallthrough/`break` all pass (`sw_*` 6/7); only this
mid-list-default-fallthrough edge mis-lowers.  Affects: `sw_no_break_default`.

### B9 — user varargs (`va_list`/`va_arg`)
`int f(int n,...){ va_list ap; va_start(ap,n); va_arg(ap,int); }` rejects at the
front-end (no `...`/`va_list` grammar).  `printf(...)` works (builtin syscall,
runtime argc).  Affects: `va_sum`.

## Link-step note (not a transpiler bug)

The transpiler DEFERS global/array/static-value initializers into a generated
`c4_static_init()` that the STANDALONE per-file output never calls — the doom
LINK step (`link_doom_transpiled.py:260`) injects `c4_static_init();` at startup.
The harness emulates that (`_inject_static_init_call`, placed after main's leading
local decls) so it tests the transpiler's LOWERING, not the link wiring.  A
generic single-file transpiler should emit the init call itself.

## Clean-transpiler path — STATUS (every Doom-relevant class now general)

The Doom port runs byte-exact through general lowering for ALL of Classes
1/2/3/4/6/7, and every Doom-relevant idiom class B1/B2/B4/B5/B6/B7/B8 is now
absorbed into the generic transpiler (MEASURED on the corpus, no hand-patches):

1. **B1 true `T a[N][M]`** — HIGH — **FIXED**.  `lower_multidim_arrays`:
   declaration-shape flatten (dims -> product malloc + row-major index
   linearization).  2D/3D, transpose, local+global, initialized+uninitialized.
2. **B2 tagged/local-value structs + non-`_t` globals** — MED — **FIXED**.
   Tagged-def erase, tagged/non-`_t` local + global struct-value allocation, and
   contract-driven type inference for bare aliases.
3. **B5 enum explicit values + enum-typed decls** — MED — **FIXED**.  Enum-def
   body excluded from the collision scan (explicit members inline); `enum State s;`
   lowered to `int s;`.
4. **B7 static locals** — MED — **FIXED**.  `static` locals hoisted to mangled
   file-scope globals with init-once via `c4_static_init`.
5. **B8 mid-list-default switch** — **FIXED** (default emitted in position);
   **B4 tagged/`_t` union** — **FIXED** (byte-buffer overlay + `__rd`/`__wr` incl
   writes); **B6 string-ptr / multi-scalar global init** — **FIXED**.
6. **B3 function pointers / B9 varargs** — OUT OF SUBSET by design; DOOM uses the
   FN_ID dispatch contract + builtin printf, so these stay unsupported (documented
   subset boundary, the only 5 residual corpus fails).

Remaining gap to a fully hand-patch-free vanilla re-transpile: **only** B3
(function pointers) and B9 (user varargs), which the Doom port already avoids via
the `__actions__` FN_ID dispatch + builtin `printf`.  No general C90 idiom that
linuxdoom actually uses is left un-lowered.

Measured vs projected: the 112/117 = 95.7% matrix and every B1..B9 reproducer are
MEASURED (gcc-m32 oracle vs port, byte-exact).  The claim that these fixes REMOVE
specific Doom hand-patches (rather than merely handling the idiom generically) is a
PROJECTION — NOT re-measured end-to-end on a freshly re-transpiled linuxdoom (a
whole-Doom re-transpile is a separate, deliberately-deferred decision).
