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

## Conformance matrix (measured, 91/117 = 77.8% clean)

| category   | pass | mismatch | compileErr | note |
|------------|-----:|---------:|-----------:|------|
| multidim   |  5   |  3       | 0          | true `T a[N][M]` un-flattened (**NEW**) |
| ptrarith   |  8   |  0       | 0          | clean |
| structarr  |  7   |  0       | 4          | `_t` global/ptr works; tagged/local-value fail |
| compound   | 13   |  0       | 1          | `+= |= &= ...` all lower; the 1 CERR is a struct-form limit |
| loops      | 12   |  0       | 0          | for/while/do + continue/break clean (Class-4 fixed) |
| goto       |  4   |  0       | 0          | clean |
| switch     |  6   |  1       | 0          | mid-list `default:` fallthrough mis-ordered |
| bitwise    | 12   |  0       | 0          | signed+unsigned shifts clean (Class-6 fixed) |
| fnptr      |  0   |  0       | 4          | function pointers unsupported (Class-5) |
| union      |  1   |  0       | 4          | only LOCAL-ANON union; tagged/`_t` fail |
| enum       |  2   |  0       | 2          | explicit-value inlining + enum-typed decls broken |
| varargs    |  1   |  0       | 1          | `printf` ok; user `va_list`/`va_arg` unsupported |
| recursion  |  4   |  0       | 0          | clean |
| strings    |  6   |  0       | 0          | clean |
| storage    |  5   |  2       | 2          | global-scalar/array init ok; static-local + 2D + string-ptr + multi-init fail |
| programs   |  5   |  1       | 1          | mismatch=matrix_mul (2D); cerr=linked_list (tagged struct) |
| **TOTAL**  | **91** | **7** | **19**    | 26 bug-revealing failures |

`transpile_error`/`vm_error` = 0 (the transpiler never crashed and every
compiled program halted).

## The 7 known (hand-patched) classes — status against the broad corpus

The Doom port hand-patched 7 classes per-bug.  Measured on the current
transpiler, **5 of the 7 are already absorbed into general lowering** (they PASS
the broad corpus in isolation); only Class-1 and Class-5 remain, and Class-1 is
narrower than stated:

| # | known class | status | evidence |
|---|-------------|--------|----------|
| 1 | compound-index stride `arr[i*k]` / `ptr[i][j]` | **PARTIAL** | flat `a[k*2+1]` + manual `m[i*3+j]` PASS (a318d77c src fix); **true `int a[N][M]` with `a[i][j]` still MISMATCH** (the remaining bug) |
| 2 | compound-assign `+= |= &=` dropped to `=` | **FIXED** | `ca_*` 13/14 pass; `lower_compound_assign` works |
| 3 | struct-array stride `&arr[i]->arr[i*N]` | **FIXED** | `st_t_addr_of_elem` / `st_t_ptr` pass (`_t` global) |
| 4 | for-continue increment inside guard -> spin | **FIXED** | `lp_for_continue*` all pass |
| 5 | function stubbing (return-0 for unhandled fns) | **OPEN** | `fp_*` all CERR — function pointers reject at compile |
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

## Clean-transpiler path (fix these to drop the Doom hand-patches)

The Doom port already runs byte-exact through general lowering for Classes 2/3/4/6/7.
The remaining classes that a *vanilla-linuxdoom -> generic transpiler -> byte-exact
Doom, no hand-patches* would need — ranked by Doom relevance:

1. **B1 true `T a[N][M]`** — HIGH.  Any 2D local/global array in engine code.  The
   port survives only because DOOM's hot paths use manual strides; residual 2D
   decls are hand-patched.  Fix = declaration-shape-driven flatten (dims -> malloc
   size + index linearization).
2. **B2 tagged/local-value structs + non-`_t` globals** — MED.  linuxdoom is
   overwhelmingly `_t` typedef + global/pointer (works), so this is mostly a
   generality gap, but any tagged struct or local struct value is hand-patched.
3. **B5 enum explicit values + enum-typed decls** — MED.  Doom enums with explicit
   values (`ML_*`, `MF_*` are `#define`, but real `enum{...=N}` exist) need this.
4. **B7 static locals** — MED.  Doom has `static` function-local state.
5. **B8 mid-list-default switch**, **B4 tagged/`_t` union**, **B6 string-ptr /
   multi-scalar global init** — LOW (edge idioms; DOOM's specific forms already work).
6. **B3 function pointers / B9 varargs** — by design out-of-subset; DOOM uses the
   FN_ID dispatch contract + builtin printf, so these stay unsupported (documented
   subset boundary, not a fix target for the Doom path).

Measured vs projected: the matrix, the 5-of-7-fixed status, and every B1..B9
reproducer are MEASURED (gcc-m32 oracle vs port).  The clean-transpiler *ranking*
(which fixes most reduce Doom hand-patches) is a PROJECTION from the class ->
Doom-idiom mapping, not re-measured end-to-end on a re-transpiled linuxdoom.
