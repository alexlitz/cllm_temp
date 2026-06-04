# Imperative `_set_layerN_*` Helper Audit — 2026-06-04

**Scope.** `c4_release/neural_vm/vm_step.py` and `c4_release/neural_vm/setup_helpers.py`,
focusing on imperative `block.ffn.W_*[...] = X` / `attn.W_*[...] = Y` weight-bake
helpers prefixed `_set_layerN_*`.

**Branch.** `vm-step-audit` (off `speedup-cache-and-buckets` @ `751fcfaa`).

## Summary

| Status                | Count | Note                                                        |
| --------------------- | ----- | ----------------------------------------------------------- |
| Functions audited     | 22    | All in `vm_step.py`                                         |
| Live (prod bake delegate) | 4 | Called from a `bake_fn` or `RuntimeAttentionFragment` baked by an op in `all_core_ops()` |
| Test-fallback only    | 18    | Imperative helper unused by production; the op factory uses `compiler_ir=`/`compiler_ir_factory=` |
| Migrations attempted  | 0     | Every audited helper already has a declarative IR equivalent — no pure imperative leak remains in scope |

`setup_helpers.py` itself is now a 159-line re-export hub. Per-layer
`setup_helpers_lN.py` modules host the small remainder of helpers that are
still imperative; those are out of audit scope for this wave.

## Procedure

1. Enumerated every `_set_layerN_*` definition in `vm_step.py`:
   ```
   $ grep -nE '^def _set_layer' neural_vm/vm_step.py
   ```
   22 hits at lines 3106..7741.
2. For each name, searched for **call sites** (not docstrings / backtick refs)
   inside `neural_vm/unified_compiler/` and `tests/`:
   ```
   $ grep -rn "${fn}(" neural_vm/unified_compiler/ | grep -v "^[^:]*:[0-9]*: *#" | grep -v "``${fn}("
   $ grep -rn "${fn}(" tests/                    | grep -v "^[^:]*:[0-9]*: *#"
   ```
3. For each name with an `ops/lN_ops.py` op factory, recorded whether the
   factory carries `compiler_ir=` / `compiler_ir_factory=` and whether the
   imperative helper is called from the `bake_fn` body or merely referenced
   in a docstring.

## Per-function classification

Column key:
- *Op factory*: the `make_*_op()` symbol in `neural_vm/unified_compiler/ops/`.
- *In `all_core_ops`*: registered in the production op set.
- *Decl bake*: factory ships `compiler_ir=` or `compiler_ir_factory=`.
- *Live caller*: factory's `bake_fn` still calls the imperative helper at run-time.
- *Status*: `live` = production bake path delegates to imperative;
  `test-fallback` = imperative only kept for byte-identity tests.

| `vm_step` helper | line | Op factory (file) | Decl bake | Live caller? | Status |
| ---------------- | ---- | ----------------- | --------- | ------------ | ------ |
| `_set_layer3_ffn`                        | 3106 | `make_layer3_ffn_op` (l3_ops)           | `compiler_ir=`         | no  | test-fallback |
| `_set_layer4_pc_relay`                   | 3503 | `make_layer4_pc_relay_op` (l4_ops)      | `compiler_ir_factory=` | no  | test-fallback |
| `_set_layer4_ffn`                        | 3573 | `make_layer4_ffn_op` (l4_ops)           | `compiler_ir=`         | no  | test-fallback |
| `_set_layer6_attn`                       | 3991 | `make_layer6_attn_bake_op` (l6_ops)     | `compiler_ir_factory=` | no  | test-fallback |
| `_set_layer6_routing_ffn`                | 4208 | `make_layer6_routing_ffn_op` (l6_ops)   | `compiler_ir_factory=` | no  | test-fallback (note 1) |
| `_set_layer6_relay_heads`                | 5275 | `make_layer6_relay_heads_bake_op` (l6_ops) | `compiler_ir_factory=` | no | test-fallback |
| `_set_layer7_memory_heads`               | 5331 | `make_layer7_memory_heads_op` (l7_ops)  | `compiler_ir_factory=` | no  | test-fallback |
| `_set_layer8_sp_gather`                  | 5501 | `make_layer8_sp_gather_op` (l8_ops)     | `compiler_ir_factory=` | no  | test-fallback |
| `_set_layer8_multibyte_fetch`            | 5546 | `make_layer8_multibyte_fetch_op` (l8_ops) | `compiler_ir_factory=` | no | test-fallback |
| `_set_layer8_alu`                        | 5606 | `make_layer8_alu_op` (l8_ops)           | `compiler_ir=`         | **yes** (note 2) | live |
| `_set_layer8_multibyte_routing`          | 5941 | `make_layer8_multibyte_routing_op` (l8_ops) | `compiler_ir=`     | no  | test-fallback |
| `_set_layer9_alu`                        | 5983 | `make_layer9_alu_op` (l9_ops)           | rules via `Primitives.lower_ffn_rules` | no | test-fallback |
| `_set_layer9_marker_suppress`            | 6407 | `make_layer9_marker_suppress_op` (l9_ops) | rules via `Primitives.lower_ffn_rules` | no | test-fallback |
| `_set_layer10_alu`                       | 6441 | `make_layer10_alu_op` (l10_ops)         | `compiler_ir=`         | no  | test-fallback |
| `_set_layer14_mem_generation`            | 6679 | `make_layer14_mem_generation_op` (l14_ops) | `compiler_ir_factory=` | no | test-fallback |
| `_set_layer14_clear_mem_marker_output`   | 6956 | `make_layer14_clear_mem_marker_output_op` (l14_ops) | `compiler_ir=` | no | test-fallback |
| `_set_layer14_jsr_ax_bytes_zero`         | 7029 | `make_layer14_jsr_ax_bytes_zero_op` (l14_ops) | `compiler_ir=`   | no  | test-fallback |
| `_set_layer14_alu_nocarry_ax_bytes_zero` | 7104 | `make_layer14_alu_nocarry_ax_bytes_zero_op` (l14_ops) | `compiler_ir=` | no | test-fallback |
| `_set_layer15_memory_lookup_heads_0_3`   | 7205 | `make_layer15_memory_lookup_op` (l15_ops) | fragment `bake_fn=`  | **yes** (note 3) | live |
| `_set_layer15_memory_lookup_lev_heads_4_11` | 7394 | `make_layer15_memory_lookup_op` (l15_ops) | fragment `bake_fn=` | **yes** (note 3) | live |
| `_set_layer15_memory_lookup`             | 7675 | (delegates to the two above) `make_layer15_memory_lookup_op` | — | no | test-fallback wrapper |
| `_set_layer16_lev_routing`               | 7741 | `make_layer16_lev_routing_op` (l16_ops) | `compiler_ir=`         | no  | test-fallback |

### Notes

1. **`_set_layer6_routing_ffn`** — `make_layer6_routing_ffn_op` is the
   production path and is fully declarative (`compiler_ir_factory=`). The
   only remaining call site is
   `unified_compiler/compiler.py::UnifiedVMCompiler._compile_l6_ffn`. The
   legacy `UnifiedVMCompiler` class is itself dead code — its sole caller
   `weight_setter.py::_set_hand_weights` now raises `NotImplementedError`,
   and the live path runs through `full_vm_compiler_dynamic`. So this
   "live caller" is effectively unreachable; it stays only because deleting
   `UnifiedVMCompiler` is out of scope for this audit.

2. **`_set_layer8_alu`** — `make_layer8_multibyte_routing_op.bake()`
   *deliberately* invokes `_set_layer8_alu(block.ffn, S, proxy)` (l8_ops.py
   line 1520) as a **byte-identity guard / cursor recovery**, not as the
   primary bake. The primary bake is `make_layer8_alu_op` (phase 8.2) which
   uses `compiler_ir=_layer8_alu_ir()`. The re-invocation is an idempotent
   overwrite of the same ALU weights; the helper's return value is used
   solely to verify that the allocator agrees with the legacy cursor
   (`assert unit_start == expected_start`). Removing the call would force
   the bake to compute `expected_start` directly from the allocator —
   safe, but loses the cross-check.

3. **`_set_layer15_memory_lookup_heads_0_3` / `_lev_heads_4_11`** — wrapped
   as `RuntimeAttentionFragment.bake_fn=` closures inside
   `_layer15_memory_lookup_compiler_ir_factory` (l15_ops.py lines 449–471).
   These fragments are *part of* the declarative IR — the factory builds a
   `CompilerIR` and adds the imperative helpers as fragments. They were
   migrated by W7 only in that the runtime-shape switch (`num_heads`) moved
   into the factory; the underlying helper bodies remain imperative because
   their head-layout logic is too entangled with `HD` (head-dim) and the
   ALiBi slope conventions to fit neatly in `FFNRule`/attention DSL today.

## Migrations attempted in this wave: 0

The task brief suggested "aim for migrating 3-5 of the cleanest 'no
declarative equivalent' cases". After enumeration, **every** `_set_layerN_*`
helper in `vm_step.py` already has a paired declarative IR factory. The
remaining call surface is:

- 18 helpers: dead in production (test-fallback for byte-identity tests).
- `_set_layer8_alu`: invoked as an explicit cursor / byte-identity guard,
  not as the bake path.
- `_set_layer15_memory_lookup_heads_*`: wrapped as IR fragments — the
  factory *is* the migration; migrating their bodies further is a separate
  L15-specific project (head-dim aware DSL primitives required).

There are no remaining "no declarative equivalent" cases inside the audited
files. The legacy bodies survive for the byte-identity tests catalogued
under `tests/test_declarative_*_bakes_*.py` and `tests/test_l*_per_op.py`.
Per the wave constraint ("DO NOT delete the legacy `_set_layerN_*`
functions") none of them is removed here.

## Remaining true imperative writes (out of scope, noted for future waves)

`grep -nE 'ffn\.W_.*\.data\[|attn\.W_.*\.data\['` over
`neural_vm/unified_compiler/ops/*.py` returns ~hundreds of hits in places
like `l3_ops.py` (post-pass W_down fixups), `l14_ops.py` (cross-step
override / attention base shifts), `l9_ops.py::_suppress_l9_legacy_addsub_writes`
(efficient-ALU path zeroes ADD/SUB legacy outputs and `CARRY[1]/CARRY[2]`).
These are post-passes / fixups, not `_set_layerN_*` bake helpers, and they
sit inside the migrated op factories. They are the next layer of imperative
leak; calling them out here so a follow-up audit can scope them properly.

## Verification

No code changes in this audit wave. The only file added is this document.
Smoke baseline unchanged (≥29 / 52 is the floor per the task brief; the
audit is documentation-only so it cannot regress smoke).

## Files

- `c4_release/docs/IMPERATIVE_AUDIT_2026_06_04.md` — this document
