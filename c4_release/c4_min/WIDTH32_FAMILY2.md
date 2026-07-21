# Family-2 fix: full 32-bit compare / branch / loop counters (`C4_VM_WIDTH32`)

Issue #667: `while (i < n)` with `n > 255` diverges from ideal C, and a MOD with a
`> 255` operand diverges (the "matmul step-9 MOD"). The comparison / branch path
was **effectively 8-bit**: values above 255 aliased to `v & 0xFF` and negatives
aliased to large positives.

## Diagnosis — where the truncation actually is

The compare gadgets are NOT the narrow part. `nibble_unified.compile_cmp_expert`
(the SwiGLU zero-detector + sign step baked into the model) is already
**32-bit-correct in isolation** — probed at operands 300 / 1000 / 65535 it returns
the right `LT/GT/EQ` boolean (see `_diag_family2_width.py`, PROBE 1/2). The
recompose is exact to 100000 (PROBE 3).

The truncation is the **8-bit value substrate the compare inherits**, in
`nibble_vm.py`:

| piece | 8-bit behaviour | why it truncates |
|-------|-----------------|------------------|
| `compile_fold` | `AX_VAL mod 256` **every step** | `IMM 300 -> 44`, every ADD/SUB result folded to a byte |
| `base_dispatch_rules` ADD/SUB | `+256` byte wrap, relies on the fold | 8-bit two's-complement |
| `compile_nibble_to_scalar` | reads 5 nibbles | fine for 8-bit, but a symptom |
| LEA | adds BP's LOW byte only | 8-bit `(BP+imm)&0xFF` |
| `_snap_lane` requant | one flat argmax over `VALVOCAB ≈ 0x10100` | cannot represent a value > ~65 K |

So a `while (i < n)` operand is truncated at the fold **before** the (already
32-bit-capable) compare sees it. End-to-end, `IMM 300 -> 44` and a countdown from
300 finished in 44 iterations, not 300 (`_diag_family2_width` + the removed
end-to-end probe).

Note the reference `isa.interpret` is itself 8-bit (`MASK = 0xFF`) — the folded
corpus. The full-32-bit target is the word-width oracle
(`nibble_runtime.ref_interpret_words`, `mask = 0xFFFFFFFF`) and the shipped 32-bit
`nibble_muldivmod` gadgets.

## Fix — `C4_VM_WIDTH32=1` widens the value substrate to full 32-bit

Default OFF ⇒ byte-identical to the 8-bit folded corpus (golden weight hashes
unchanged: `build_step_model` `2b5923de…`, `build_unified_model` `e2386f34…`).
ON ⇒ full 32-bit, **fp64-exact**:

- **recompose** reads all 8 nibbles (`_recompose_hi_nibbles()`).
- **ADD/SUB** drop the `+256` byte hack (the SUB shift is 0 — routing a `2^32`
  constant through the SwiGLU gate is NOT fp-exact even in fp64: the silu-identity's
  ~1e-9 relative error times `2^32` is ~4 units). The per-byte / cascade requant
  wraps a negative SUB result mod `2^32` (two's complement) directly.
- **`compile_fold`** becomes a NO-OP (a mod-`2^32` ramp at `2^32-0.5` is not fp32
  materialisable); the requant carries the wrap.
- **LEA** uses the full BP value.
- **requant** (`_snap_lane` → `_snap_lane_bytes`): snap to the nearest integer
  (`floor(x+½)`, the round-free argmax equivalent — the AST guard forbids the
  `round` builtin in this module), reduce to the unsigned 32-bit word `v mod 2^32`
  (two's complement, so a borrow / overflow / large loop counter never aliases to
  zero), then split into 4 little-endian bytes. This is the §720 high-to-low
  cascade at the token round-trip — exact to the full `2^32` with **no `2^32`-wide
  vocab and no fp64 nibble-pack**.
- **exec dtype**: the width-32 model runs in **fp64** (`maybe_cast_model_for_width`
  → `model.double()`) so the `16^7` recompose and the requant resolve every value
  up to `2^32` (fp32's 24-bit mantissa would lose the low bits past `2^24`).

### Precision / layer cost

| approach | cost | exactness |
|----------|------|-----------|
| §590 fp64 nibble-pack | fp64 exec | full 2^32 |
| §720 nibble cascade | more layers, fp32-safe | full 2^32 |
| **this fix** | fp64 exec + per-byte requant cascade | **full 2^32** |

The cascade lives at the requant boundary (one Python round-trip step, already in
the driver), so it adds **no new model layers**. The fp64 exec is the honest
tradeoff: full `2^32` needs > 24 bits of mantissa, and `fp32` is exact only to
`2^24 ≈ 16.7 M`. A pure-fp32 build stays exact for loop counters / compare operands
up to `2^24` (covers n=1000, large MOD, matmul step-9); beyond `2^24` the value
lanes need fp64 (this fix uses it unconditionally under width-32 for a clean full
`2^32`).

### Signedness

Unsigned 32-bit ordering is exact (the sign cascade / zero detector are 32-bit).
Negative loop counters do not alias: the requant reduces `v mod 2^32` (two's
complement) so `-1 -> 0xFFFFFFFF`, and the `AX==0` branch predicate
(`relu(1 - AX_VAL)`) is 1 only at exactly 0 for any non-negative AX.

## Verified (fp64, `C4_VM_WIDTH32=1`)

`c4_min/test_width32_family2.py` (18/18):
- `_snap_lane_bytes` exact to `2^32` (incl. near-integer residues, borrows,
  overflows).
- `loop_countdown` reaches 0 and runs the **correct iteration count** at
  n = 3, 200, 255, 256, 300, **1000** (300 → 1202 steps = 300 iters, not 178 = 44
  iters in the 8-bit path).
- compare `EQ/NE/LT/GT/LE/GE` full 32-bit: **84/84** at operands crossing 255
  (300 vs 100, 1000 vs 5, 65535 vs 1, …). The 8-bit path fails 8/84 (the ordering
  ops on a 300-vs-100 operand — the aliasing this fixes).

No regression (default 8-bit): `test_nibble_vm.py` 9/9, `test_nibble_runtime.py`
10/10, `verify_unified` base+cmp / bitwise / muldiv all PASS, weight hashes
unchanged. The 2 `kv-memory` `verify_unified` FAILs are **pre-existing** on the
base (`fix-memcam-frame-local-wt @ 2bf94d45`, identical outputs) — a separate open
memory-CAM bug, untouched here.

## Honest boundary — MUL / DIV / MOD at 32-bit

MOD/DIV/MUL do **NOT** share the compare narrow path. They are the
`nibble_unified.UNFOLDABLE` 8-bit lookup-table experts (`mdm-select`, a
`256×256 -> byte` table). Under width-32 the operands reach them un-truncated, but
the operand one-hot band has only 256 cells, so a `> 255` operand indexes nothing
→ result 0 (MOD ≤ 255 still exact; this is the SAME 8-bit-table limit, now honestly
exposed rather than pre-truncated by the fold). Full 32-bit MUL/DIV/MOD is the
module's documented non-folding boundary: the base-16 long-division / carry-round
gadgets in `nibble_muldivmod.py` (fp64, `mul32`/`divmod32`/`mod32`) are iterative
and lower to O(width) unrolled FFN blocks (or the recurrent gadget), not a fixed
table. A cheap operand-range widening is infeasible (a 4096×256 MOD table ≈ 3 M FFN
units). This is a distinct, larger follow-up; the compare/branch/loop narrow path
(the primary Family-2 divergence) is fixed here.
