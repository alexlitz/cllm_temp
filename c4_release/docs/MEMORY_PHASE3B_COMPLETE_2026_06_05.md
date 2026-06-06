# Memory Phase 3b — L13 anchor split (2026-06-05)

Per `docs/MEMORY_PHASE4_BLOCKER_2026_06_05.md` (commit `66d5e9ba`), the
prerequisite Phase 4 needs is an actual split of the joint L13 attn
anchor — Phase 3 (commit `3423aff1`) split `layer10_carry_relay`
instead, leaving `_layer13_attn_dep_anchor` (the anchor
`layer13_mem_addr_gather` actually binds to) intact. This commit
delivers that split as a smoke-neutral, byte-identical refactor:
`_layer13_attn_dep_anchor` splits into two sibling topology anchors
that co-resolve to the same physical layer today, but expose
independent retarget surface for Phase 4.

## Change summary

`_layer13_attn_dep_anchor` previously co-served:

| Consumer (`target_op_name`) | Role |
|---|---|
| `layer13_mem_addr_gather` | L13 mem-addr block bake (heads 0/1/2 -> ADDR_B*_LO/HI) |
| `layer13_shifts` | L13 FFN SHL/SHR lookup-mode bake |
| `l13_alu_shift_install` | Efficient-mode composite install (`block.ffn = ALUShiftComposite`) |
| `l13_alu_postop_attach` | Lookup-mode post-op attach |

Phase 3b adds a sibling `_layer13_mem_addr_anchor` (`kind="attn"`,
`declarative_authority="topology_anchor"`, empty IR, `phase=13.0`,
`requires={"after": "_layer12_ffn_dep_anchor", "same_layer_as":
"_layer13_attn_dep_anchor"}`). Only the first row above retargets to
the new anchor; the three FFN-side consumers stay on
`_layer13_attn_dep_anchor`. The `same_layer_as` constraint keeps both
anchors co-located at the same physical layer (L16 today) so the
refactor is byte-identical at this stage; Phase 4 will drop the
constraint and pin the mem-addr anchor to `layer_idx=13` (paired with
the L13 head-0 decoupling and composite-late-placement work described
in the BLOCKER doc §"What needs to change before Phase 4 can land").

## Why not pin to `layer_idx=13` in this commit?

The brief draft contemplated landing `layer_idx=13` on the new anchor
inside Phase 3b. Empirically this regresses smoke from 45/51 to 12/51
(SHL/SHR + 32-bit ALU + comparison + memory tests all break). The
regressions stem from `layer13_mem_addr_gather` baking into
`model.blocks[13].attn` while `layer10_carry_relay_bake` already owns
head 0 there (the "head_0 contest" documented in
`MEMORY_CLUSTER_FIX_PLAN_2026_06_05.md` §Phase 2). That contest is
Phase 2's scope, not Phase 3b's; landing the pin here would replicate
the V4 zero-sum regression pattern.

The metadata-only split is the conservative landing: it cleanly exposes
the retarget surface Phase 4 needs (independent
`layer13_mem_addr_gather` binding) without smoke risk. Phase 4 will
land the pin and the head-0 decoupling together.

## Verification

### Compile (efficient + lookup, strict mode)

```
$ python -c "from neural_vm.unified_compiler import compile_full_vm_dynamic; \
   compile_full_vm_dynamic(alu_mode='efficient', strict=True, disk_cache=False)"
   ... OK
```

Layer placement post-split (efficient mode):

```
L13: layer10_carry_relay      (attn) -- Phase 3 anchor
L13: _layer10_attn_anchor     (attn) -- Phase 3 sibling
L16: _layer13_attn_dep_anchor (attn) -- legacy joint anchor (now FFN-side only)
L16: _layer13_mem_addr_anchor (attn) -- Phase 3b sibling (mem-addr only)
L17: _layer14_attn_dep_anchor (attn) -- still pinned past L13 anchor

Block-op placements (unchanged from baseline):
  layer13_mem_addr_gather  -> L16 (now via _layer13_mem_addr_anchor)
  l13_alu_shift_install    -> L16 (still via _layer13_attn_dep_anchor)
  layer13_shifts           -> L16 (still via _layer13_attn_dep_anchor)
```

### Byte-identity (block.attn weight hashes)

```
$ diff /tmp/baseline_block_hashes.txt /tmp/phase3b_block_hashes.txt
$ echo $?
0
```

All 36 block attn weight hashes identical before/after the split. The
mem-addr gather bakes into the same `model.blocks[16].attn` as before
(its `target_op_name` chain still resolves to L16 via the
`same_layer_as` constraint).

### Smoke (CUDA_VISIBLE_DEVICES=1)

```
$ pytest c4_release/tests/test_smoke.py --no-header -q --tb=no
45 passed, 6 failed, 1 deselected in 106.36s
```

Failures (matches BLOCKER doc baseline exactly):
* `TestSmokeAddress::test_lea_basic`
* `TestSmokeMemory::test_si_li_roundtrip`
* `TestSmokeMemory::test_sc_lc_roundtrip`
* `TestSmokeMemory::test_si_li_multiple_stores`
* `TestSmokeMemory::test_si_li_overwrite`
* `TestSmokeMemory::test_si_li_16bit_value`

Phase 3b is byte-identity-preserving and smoke-neutral; the 5 memory
tests are still failing and remain Phase 4's responsibility.

### Slot registry

```
$ pytest c4_release/tests/test_slot_registry.py -q --tb=no
25 passed, 2 deselected in 0.18s
```

The new anchor is a topology anchor; `derive_slot_ids_for_op` returns
`[]` for it, so no slot claims are registered. The existing claims on
`layer13_mem_addr_gather` (3 attn heads at L16) and the FFN-side ops
are unchanged.

## Files

* `c4_release/neural_vm/unified_compiler/ops/l13_ops.py` -- adds
  `make_layer13_mem_addr_anchor_op`; retargets `layer13_mem_addr_gather`
  to the new anchor.
* `c4_release/neural_vm/unified_compiler/ops/all_core_ops.py` -- registers
  the new anchor next to `make_layer13_attn_dep_anchor_op`.

## Follow-up: Phase 4 retry

With the split landed, Phase 4 can amend its design doc per the BLOCKER
recommendation:

1. Drop `same_layer_as` from `_layer13_mem_addr_anchor` and add
   `layer_idx=13` -- pulls `layer13_mem_addr_gather` to block[13] for
   the L14 same-step mem-value chain.
2. Decouple the L13 head-0 contest (Phase 2 scope): either retarget
   `layer10_carry_relay_bake` to a different head index, or move the
   L10 attn family to a non-L13 layer using the
   `_layer10_attn_anchor` (Phase 3) infrastructure.
3. Re-pin the shift composite stages so they stay in the L13-L16 range
   (per the BLOCKER doc §"What needs to change" item B).
