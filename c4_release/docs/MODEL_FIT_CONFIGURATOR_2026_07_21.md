# Model-Fit Configurator (`c4_min/qwen_fit_solver.py`)

Given a constraint box (max depth / width / precision, or a named stock Qwen
target) and an op-set to COVER, pick the lever combo that FITS, account it, rank
by an objective, and (for the buildable ones) bake it. Generalises
`qwen_full_vm.fit_report` / `fit_report_efficient` from fixed printed rows into a
solver. All accounting runs on CPU with NO model materialised.

## Where each op-cost goes (the four axes)

Every op-family cost flows onto ONE axis; precision scales all of them:

| lever (muldiv strategy)      | cost axis  | what it buys |
|------------------------------|------------|--------------|
| `lookup-table`               | WIDTH      | 256x256x3 MUL/DIV/MOD table = `intermediate 160465` (the ~45 GB wall), ZERO extra depth |
| `efficient-ALU-unrolled`     | DEPTH      | `nibble_alu32` gadgets: small width (`intermediate 11272`), MUL +10 layers, DIV/MOD +262 |
| `efficient-ALU-recurrent`    | DEPTH      | same applied depth, ~half STORED (one reused long-division iteration body, applied 8x) |
| `subroutine`                 | STEPS      | JSR/LEV into a baked bytecode routine: ~0 extra width/depth, many PROGRAM STEPS per op [EST] |
| `precision` 8/16/32          | scales all | nibble granularity scales ALU div-iteration depth + subroutine steps; table is inherently 8-bit |

## Verified vs Estimated levers

- **VERIFIED** (computed live from the REAL `qwen_full_vm._block_specs` builders,
  reproducing #698 `fit_report_efficient` exactly, and bakeable via `bake()`):
  8-bit `lookup-table`; 32-bit `efficient-ALU-unrolled` / `efficient-ALU-recurrent`.
- **ESTIMATED** (`verified=False`, `bake()` raises `NotImplementedError`):
  - `precision != 32` on the efficient ALU -- a linear-in-nibble PROJECTION of the
    div-iteration depth (the baked ALU is 32-bit-exact only).
  - `subroutine` steps_per_op -- from #699 (branch `verify-lean-muldiv-mandelbrot`),
    LABELLED estimated-until-#699-reverified. Precision-scaled (O(bytes^2) MUL,
    O(bytes) DIV/MOD).
  - `precision != 8` on the `lookup-table` -- 16/32-bit domain is unbuildable.

## Solver API

```python
from c4_min import qwen_fit_solver as S

# one-call front door -> SolveResult
res = S.fit(target=None, *, ops=S.FULL, max_depth=None, max_width=None,
            max_hidden=None, max_intermediate=None, precision=None,
            max_steps_per_op=None, minimize="depth", require_buildable=False,
            code_size=24)

res.ok            # bool: a feasible config was found
res.best          # Accounting: the winning config (None if infeasible)
res.feasible      # List[Accounting] ranked by the objective
res.infeasible    # List[(Accounting, [violations])]
res.binding       # the axis blocking the MOST configs (when infeasible)
res.relaxation    # the closest single-cap bump to feasibility

S.best_summary(res)          # one-line best (or the binding+relaxation)
S.tradeoff_table(res)        # the ranked config table
S.bake(config)               # -> QwenFullVM (VERIFIED configs only; else raises)
S.measure_built(vm)          # read back the ACTUAL geometry of a baked VM
```

`minimize` in `{"depth", "width", "steps", "params", "fits-named"}`
(`fits-named` needs a `target=`). `max_width` caps BOTH hidden and intermediate.
A named `target` (`stock-0.5b` / `stock-1.5b` / `stock-7b`) folds its
depth/width/head budget into the box.

### Config schema

```python
@dataclass(frozen=True)
class FitConfig:
    ops: frozenset                 # the opcodes to COVER (lights Subset flags)
    muldiv_strategy: str = "efficient-ALU-recurrent"
    precision: int = 32            # 8 / 16 / 32
    granularity: str = "nibble"   # nibble | byte
    code_size: int = 24
    arch: QwenArch = QWEN2_5_ARCH  # GQA head geometry

@dataclass
class Accounting:
    config, hidden, intermediate, stored_layers, applied_depth,
    steps_per_op, query_heads, verified, params_estimate, notes
```

## Accounting model

Per candidate config the solver reports
`(hidden, intermediate, stored_layers, applied_depth, steps_per_op)`:

- **hidden / intermediate / stored_layers / applied_depth** come from the REAL
  `qwen_full_vm._block_specs` builders (the same ones `build` bakes), memoized on
  the spec-determining levers `(code_size, subset-flags, efficient_alu,
  recurrent_divmod)`. Precision is a post-hoc PROJECTION and does not change the
  specs, so the P8/P16/P32 variants share ONE build.
- **steps_per_op** = 1 for the persistent-stack strategies (one `Qwen2Model.forward`
  per op); the `subroutine` strategy costs many program steps per op (each step = one
  forward), precision-scaled [EST].
- **applied_depth** >= stored_layers only for `efficient-ALU-recurrent` (the reused
  long-division body is applied more times than it is stored).

### Memory safety

The `lookup-table` MUL/DIV/MOD table (`intermediate 160465`) is the ~45 GB / ~55 GB
peak-RSS wall -- materialising `compile_mdm_select` alone peaks ~55 GB. For the
*accounting* the solver counts the nonzero `op(a,b)` entries over all 256x256 pairs
with the builder's own `_MDM_FN` (a tensor-free 3x256x256 int loop) and adds the two
lookup blocks to the memory-LIGHT muldiv-OFF build (~4.5 GB). This reproduces the
real build (width `160465`, +2 stored layers, identical `D_used`) EXACTLY -- verified
by `test_analytic_lookup_table_matches_the_real_table_width`. Set
`C4_FIT_BUILD_TABLE=1` to force the real (45 GB) build as a byte-identity self-check.
Whole solver over the FULL enumeration peaks ~4.7 GB.

## Worked-example fits

All four brief examples come out INFEASIBLE against the tight boxes -- the C4 full
VM is intrinsically larger than these budgets (hidden 1600, base FFN width 11272,
min stored 15) -- so they exercise the binding-constraint + closest-relaxation path.
Feasible fits appear once the box is relaxed (see the test suite `_BIG` box).
`vfd` column: V=VERIFIED-buildable, E=ESTIMATED.

### `fit("stock-0.5b", ops=FULL, minimize="depth")`

```
INFEASIBLE for objective=depth. Binding: hidden. closest config [mem+cmp+bit+muldiv] muldiv=lookup-table P8 nibble: relax hidden: need <= 1600 (cap 896); intermediate: need <= 160465 (cap 4864)

config                                               hidden   inter stored applied stp/op   params  ok vfd
----------------------------------------------------------------------------------------------------------
[mem+cmp+bit+muldiv] muldiv=lookup-table P8 nibble     1600  160465     17      17      1    13.2B  no   V
[mem+cmp+bit+muldiv] muldiv=lookup-table P16 nibble    1600  160465     17      17      1    13.2B  no   E
[mem+cmp+bit+muldiv] muldiv=lookup-table P32 nibble    1600  160465     17      17      1    13.2B  no   E
[mem+cmp+bit+muldiv] muldiv=subroutine P8 nibble       1600   11272     15      15    200   861.6M  no   E
[mem+cmp+bit+muldiv] muldiv=subroutine P16 nibble      1600   11272     15      15    400   861.6M  no   E
[mem+cmp+bit+muldiv] muldiv=subroutine P32 nibble      1600   11272     15      15    800   861.6M  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-unrolled P8 nibble   2176   11272    110     110      1     8.6B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-unrolled P16 nibble   2176   11272    170     170      1    13.3B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-unrolled P32 nibble   2176   11272    290     290      1    22.6B  no   V
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-recurrent P8 nibble   2176   11272    143     110      1    11.2B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-recurrent P16 nibble   2176   11272    143     170      1    11.2B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-recurrent P32 nibble   2176   11272    143     290      1    11.2B  no   V

INFEASIBLE — binding constraint: hidden
  closest config [mem+cmp+bit+muldiv] muldiv=lookup-table P8 nibble: relax hidden: need <= 1600 (cap 896); intermediate: need <= 160465 (cap 4864)
```

### `fit(ops=FULL, max_depth=10, minimize="depth")`

```
INFEASIBLE for objective=depth. Binding: depth (stored layers). closest config [mem+cmp+bit+muldiv] muldiv=subroutine P8 nibble: relax depth (stored layers): need <= 15 (cap 10)

config                                               hidden   inter stored applied stp/op   params  ok vfd
----------------------------------------------------------------------------------------------------------
[mem+cmp+bit+muldiv] muldiv=lookup-table P8 nibble     1600  160465     17      17      1    13.2B  no   V
[mem+cmp+bit+muldiv] muldiv=lookup-table P16 nibble    1600  160465     17      17      1    13.2B  no   E
[mem+cmp+bit+muldiv] muldiv=lookup-table P32 nibble    1600  160465     17      17      1    13.2B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-unrolled P8 nibble   2176   11272    110     110      1     8.6B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-unrolled P16 nibble   2176   11272    170     170      1    13.3B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-unrolled P32 nibble   2176   11272    290     290      1    22.6B  no   V
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-recurrent P8 nibble   2176   11272    143     110      1    11.2B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-recurrent P16 nibble   2176   11272    143     170      1    11.2B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-recurrent P32 nibble   2176   11272    143     290      1    11.2B  no   V
[mem+cmp+bit+muldiv] muldiv=subroutine P8 nibble       1600   11272     15      15    200   861.6M  no   E
[mem+cmp+bit+muldiv] muldiv=subroutine P16 nibble      1600   11272     15      15    400   861.6M  no   E
[mem+cmp+bit+muldiv] muldiv=subroutine P32 nibble      1600   11272     15      15    800   861.6M  no   E

INFEASIBLE — binding constraint: depth (stored layers)
  closest config [mem+cmp+bit+muldiv] muldiv=subroutine P8 nibble: relax depth (stored layers): need <= 15 (cap 10)
```

### `fit(ops=FULL, max_width=4864, minimize="depth")`

```
INFEASIBLE for objective=depth. Binding: intermediate. closest config [mem+cmp+bit+muldiv] muldiv=efficient-ALU-unrolled P8 nibble: relax intermediate: need <= 11272 (cap 4864)

config                                               hidden   inter stored applied stp/op   params  ok vfd
----------------------------------------------------------------------------------------------------------
[mem+cmp+bit+muldiv] muldiv=lookup-table P8 nibble     1600  160465     17      17      1    13.2B  no   V
[mem+cmp+bit+muldiv] muldiv=lookup-table P16 nibble    1600  160465     17      17      1    13.2B  no   E
[mem+cmp+bit+muldiv] muldiv=lookup-table P32 nibble    1600  160465     17      17      1    13.2B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-unrolled P8 nibble   2176   11272    110     110      1     8.6B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-unrolled P16 nibble   2176   11272    170     170      1    13.3B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-unrolled P32 nibble   2176   11272    290     290      1    22.6B  no   V
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-recurrent P8 nibble   2176   11272    143     110      1    11.2B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-recurrent P16 nibble   2176   11272    143     170      1    11.2B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-recurrent P32 nibble   2176   11272    143     290      1    11.2B  no   V
[mem+cmp+bit+muldiv] muldiv=subroutine P8 nibble       1600   11272     15      15    200   861.6M  no   E
[mem+cmp+bit+muldiv] muldiv=subroutine P16 nibble      1600   11272     15      15    400   861.6M  no   E
[mem+cmp+bit+muldiv] muldiv=subroutine P32 nibble      1600   11272     15      15    800   861.6M  no   E

INFEASIBLE — binding constraint: intermediate
  closest config [mem+cmp+bit+muldiv] muldiv=efficient-ALU-unrolled P8 nibble: relax intermediate: need <= 11272 (cap 4864)
```

### `fit(ops=FULL, max_depth=14, precision=16, minimize="depth")`

```
INFEASIBLE for objective=depth. Binding: depth (stored layers). closest config [mem+cmp+bit+muldiv] muldiv=subroutine P8 nibble: relax depth (stored layers): need <= 15 (cap 14)

config                                               hidden   inter stored applied stp/op   params  ok vfd
----------------------------------------------------------------------------------------------------------
[mem+cmp+bit+muldiv] muldiv=lookup-table P8 nibble     1600  160465     17      17      1    13.2B  no   V
[mem+cmp+bit+muldiv] muldiv=lookup-table P16 nibble    1600  160465     17      17      1    13.2B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-unrolled P8 nibble   2176   11272    110     110      1     8.6B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-unrolled P16 nibble   2176   11272    170     170      1    13.3B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-recurrent P8 nibble   2176   11272    143     110      1    11.2B  no   E
[mem+cmp+bit+muldiv] muldiv=efficient-ALU-recurrent P16 nibble   2176   11272    143     170      1    11.2B  no   E
[mem+cmp+bit+muldiv] muldiv=subroutine P8 nibble       1600   11272     15      15    200   861.6M  no   E
[mem+cmp+bit+muldiv] muldiv=subroutine P16 nibble      1600   11272     15      15    400   861.6M  no   E

INFEASIBLE — binding constraint: depth (stored layers)
  closest config [mem+cmp+bit+muldiv] muldiv=subroutine P8 nibble: relax depth (stored layers): need <= 15 (cap 14)
```

## Bake + verify

`bake(config)` -> a genuine `Qwen2Model` via `qwen_full_vm.build` (VERIFIED configs
only; the subroutine / non-32-bit-ALU accountings are ESTIMATES and raise).
`measure_built(vm)` reads back the ACTUAL geometry.

Verified in `test_qwen_fit_solver.py`:

- `test_bake_base_geometry_matches_accounting` (slow): bakes the base op-set VM
  and confirms `measure_built` == the accounted `(hidden, intermediate, stored,
  applied)` -- hidden 960, intermediate 896, 7 stored layers.
- Byte-exactness of that same `build(code_size=24, subset=base)` VM is covered by
  the existing `test_qwen_full_vm.py` battery (ADD/SUB/BZ/BNZ/JMP/loop through the
  genuine Qwen2 forward, all PASS).
- `test_bake_rejects_estimate_only_configs` (slow): `bake()` raises on the
  subroutine + non-32-bit-ALU ESTIMATE levers.
- `test_accounting_matches_fit_report_efficient`: the efficient-ALU rows reproduce
  `qwen_full_vm.fit_report_efficient` byte-for-byte.

> NOTE: the pre-existing `test_qwen_full_vm.py::test_base_subset_fits_stock_0_5b`
> FAILS on this branch (base hidden 960 > stock 896) -- independent of this work,
> and CONFIRMED by the solver, which honestly reports base needs hidden 960 and FULL
> needs 1600 (neither fits the stock 0.5B hidden budget).

## Tests

```
# accounting only (CPU, memory-safe, ~40s):
python -m pytest c4_min/test_qwen_fit_solver.py -m "not slow"
# + the tiny base bake+verify:
python -m pytest c4_min/test_qwen_fit_solver.py --runslow
```

23 tests: op-set->Subset mapping, accounting == ground truth, the analytic
lookup-table width == the real table, the four cost-axis invariants, precision
scaling, VERIFIED-vs-ESTIMATED labelling, solver feasibility / objectives / binding
+ relaxation, and the bake+verify.

## Files

- `c4_min/qwen_fit_solver.py` -- the configurator (solver + accounting + bake).
- `c4_min/test_qwen_fit_solver.py` -- the test suite.
- Depends on `c4_min/qwen_full_vm.py` #698 efficient-ALU API (`fit_report_efficient`,
  `build(..., efficient_alu=, recurrent_divmod=)`) -- UNMODIFIED by this work.
