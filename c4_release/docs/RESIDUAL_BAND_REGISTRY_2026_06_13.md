# Op-local residual-band registry (2026-06-13)

Removes a centralization anti-pattern: residual-band registration used to be a
single hand-edited dict (`_PRODUCTION_EXTRA_RESIDUAL_DIMS` in
`full_vm_compiler_dynamic.py`) that EVERY band-adding op had to edit — a
cross-lane merge-conflict magnet. Bands are now declared **op-locally** and
**auto-collected** by the compiler.

## What an over-width residual band is

The widen keystone (commit c3e73ed1) made d_model GROWTH automatic:
`compile_full_vm_dynamic(extra_residual_dims={name: size})` appends a fresh
residual band at the tail and runs the head-dim-preserving widen (rounds
d_model up to a multiple of the BASE `head_dim`=109, ADDS heads instead of
repartitioning existing ones — so attention content is byte-behaviour-identical;
naively re-deriving head_dim from the widened width regresses `test_bnz_branch`).
The collected set also flows into the disk + in-proc cache keys (so widened and
baseline builds never share a serialised entry) and into `_LIVENESS_NEVER_SHARE`
(so carry/dump bands keep private dim-liveness slots).

What was still manual: the band REGISTRATION. This registry fixes that.

## The API

`neural_vm/unified_compiler/ops/residual_band_registry.py`:

```python
from .residual_band_registry import register_residual_band

register_residual_band(
    name,            # residual band dim name (str); op rules ref it via
                     # layout.dim_positions[name]
    size,            # number of residual dims (positive int)
    owner="make_my_op",   # owning op / op-family name (diagnostics + collision msgs)
    flag=None,       # optional zero-arg predicate; band is collected only when
                     # flag() is truthy. Evaluated FRESH at each compile, so a
                     # flag-off build omits the band (smaller d_model, byte-identical
                     # to the pre-band geometry). None => always present.
    never_share=False,    # True => band keeps a PRIVATE dim-liveness slot (required
                     # for carry/dump bands holding cross-step / cross-op state).
)
```

### How to add a band-adding op (the whole point)

1. At the **top of your op's `lN_ops.py` module** (next to the op factory that
   reads/writes the band), call `register_residual_band(...)`. Registration runs
   at IMPORT time — `all_core_ops` wildcard-imports every `lN_ops` module, so it
   always fires.
2. That's it. The compiler auto-collects your band, grows d_model
   head-dim-preservingly, threads it into the cache keys, and (if
   `never_share=True`) into the liveness never-share set. **You never touch
   `full_vm_compiler_dynamic.py` or `layer_compiler.py`.**

### Flag-gated bands

A band whose presence depends on a runtime env flag passes a `flag=` predicate.
Registration is always unconditional (import time); the predicate is evaluated
at COLLECT time (per compile). A flag-off build omits the band and stays
byte-identical to the pre-band geometry. Example (the width=2 MUL band):

```python
register_residual_band(
    "MUL_RESULT_HI_LO", 16, owner="make_efficient_l11_alumul_wrap_op",
    flag=mul_width2_enabled,   # C4_MUL_WIDTH2; default ON
)
```

### `never_share`

Carry/dump bands hold cross-step or cross-op state. A dim-liveness merge onto a
same-width donor whose lifetime "ended" would leave stale residue in the shared
slot and clobber the carried one-hot (observed: `H1_DUMP_OUT` slot 323 returned
a stale 1.0). Such bands pass `never_share=True`; their names are threaded into
`LayerCompiler._LIVENESS_NEVER_SHARE` automatically (per-compile, via
`compile_full_vm_dynamic` → `LayerCompiler.add_never_share_names`). Bands safe to
liveness-merge (e.g. the single-step MUL result band) pass `never_share=False`
(the default).

## Auto-collection wiring

`compile_full_vm_dynamic`:

```python
from .ops.residual_band_registry import (
    collect_registered_residual_bands, collect_never_share_band_names,
)
_merged_extra = collect_registered_residual_bands()          # {name: size}
if os.environ.get("C4_DISABLE_AX_CARRY_BANDS") == "1":       # diag escape hatch
    _merged_extra = {}
if extra_residual_dims:                                       # caller / env override
    _merged_extra.update(extra_residual_dims)                #   (C4_EXTRA_RESIDUAL_DIMS
extra_residual_dims = _merged_extra or None                  #    parsed in run_vm.py)
_never_share_band_names = collect_never_share_band_names()
```

The collected `extra_residual_dims` then flows through the EXISTING widen path
(`_bake_from_scheduled_ops`): `base_head_dim` captured BEFORE the bands are
declared, bands forward-declared as `pending_extra_dims` names (so op rules
validate at `add_op`), `add_never_share_names(_never_share_band_names)` threads
the private-slot names in, then the bands are bump-pointer declared post-loop and
the head-dim-preserving widen runs.

**Load-bearing ordering** preserved: registration order in the op modules fixes
the tail `dim_positions`. The op-module import order (l11 before alu) places the
AX + Root 2 bands before the MUL band, matching the legacy central-dict ordering
→ byte-identical layout.

## Current registry (default config)

| Band | Size | Owner module | Flag | never_share |
|------|------|--------------|------|-------------|
| `H1_PREV_STEP` | 7 | l11_ops (AX carry) | always | yes |
| `H1_DUMP_OUT` | 7 | l11_ops (AX carry) | always | yes |
| `AX_CARRY_OVERFLOW` | 1 | l11_ops (AX carry) | always | yes |
| `STACK0_B0_H1_PREV` | 7 | l11_ops (Root 2) | always | yes |
| `STACK0_B0_H3_PREV` | 7 | l11_ops (Root 2) | always | yes |
| `STACK0_B0_DUMP_H1` | 7 | l11_ops (Root 2) | always | yes |
| `STACK0_B0_DUMP_H3` | 7 | l11_ops (Root 2) | always | yes |
| `STACK0_B0_CARRIED` | 1 | l11_ops (Root 2) | always | yes |
| `STACK0_B0_SHARP` | 1 | l11_ops (Root 2) | always | yes |
| `STACK0_B0_PREV_DOM` | 1 | l11_ops (Root 2) | always | yes |
| `STACK0_B0_NOT_CMP` | 1 | l11_ops (Root 2) | always | yes |
| `MUL_RESULT_HI_LO` | 16 | alu_ops (wide_mul) | `C4_MUL_WIDTH2` | no |
| `MUL_RESULT_HI_HI` | 16 | alu_ops (wide_mul) | `C4_MUL_WIDTH2` | no |

Default config (MUL on): d_model 981, n_heads 9. `C4_MUL_WIDTH2=0` drops the two
MUL bands. NOTE: `C4_AX_BYTE1_DUMP` / `C4_STACK0_B0_DUMP` (both default-ON) gate
only the LM-HEAD emission columns (model-level head bakes), NOT the residual
bands — the bands are always present (production-default geometry), exactly as
in the legacy central dict.

## Verification (this was a pure refactor)

- Auto-collected band set == the legacy `_PRODUCTION_EXTRA_RESIDUAL_DIMS` for the
  default config (same set, same order). Model is BYTE-IDENTICAL: final layout
  `dim_positions` / `dim_sizes` hash unchanged (`606ce9d9…`, d_model 981, 218
  dims). `pytest tests/test_smoke.py` = 51/0 (spec_k=0).
- Flag-off (`C4_MUL_WIDTH2=0`) drops the MUL bands → smaller d_model, same as
  before.
- Building-blocks DSL + allocator tests pass (135/135).
