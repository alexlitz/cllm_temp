# Phase 7 (heap + DIV/MOD) sample run — 2026-05-11

## Scope

Sampled 7 of 25 tests from `c4_release/tests/test_pure_neural_heap_div.py`
in `pure_neural=True, trust_neural_alu=True` mode to see whether recent
work (Wave 2 Unit 10/11/12 L8/L9/L10 ALU FFN migrations + vanilla-flattened
divmod composite + long-division replacing fp64 MAGIC-floor) has shifted
any Phase 7 outcomes.

GPU: CUDA_VISIBLE_DEVICES=0 (auto-picked least-utilized).
Total wall time: ~11m 49s for 7 tests.

## Results

| Test                                  | xfail decorator? | Outcome | Notes |
|---------------------------------------|------------------|---------|-------|
| `test_mul_small[3-4-12]`              | yes              | XFAIL   | unchanged |
| `test_mul_small[5-5-25]`              | yes              | XFAIL   | unchanged |
| `test_mul_small[16-16-256]`           | yes              | XFAIL   | unchanged |
| `test_div_small[20-5-4]`              | **no**           | FAIL    | got `320017173` (`0x13131315`) |
| `test_mod_small[7-3-1]`               | **no**           | FAIL    | got `101058054`  (`0x06060606`) |
| `test_si_then_li`                     | yes              | XFAIL   | unchanged |
| `test_sc_then_lc`                     | yes              | XFAIL   | unchanged |

Summary: **5 XFAIL, 2 FAIL, 0 XPASS, 0 PASS.**

No xfail markers needed to be removed. No regressions in the xfail group.

## Notable patterns

### DIV / MOD have no xfail markers but still fail

`test_div_small` and `test_mod_small` are not decorated with `@pytest.mark.xfail`
in the current source. Whoever wrote them apparently expected them to pass
once divmod was migrated (FlattenedDivMod / long-division work). They still
don't pass in pure-neural.

The failure mode is striking: the returned values are byte-broadcast garbage.

- `0x06060606` = byte `0x06` replicated across 4 byte lanes
- `0x13131315` = `0x13` repeated with a `0x15` low byte — likely some carry
  propagation slop on top of the broadcast

This is the classic "we didn't reduce a per-byte tensor back to a single
8-bit scalar before EXIT" pattern. The neural FlattenedDivMod composite
must be writing a per-byte signal into AX (or into the output decode path)
and pure-neural's EXIT is reading all four byte lanes rather than the low
byte.

This is **not** the same shape as the MUL / SHL / SHR XFAIL reasons (which
say "_set_layer11_mul_partial / _set_layer12_mul_combine not wired"). DIV
and MOD must be reaching SOME handler now, otherwise we'd see a
deterministic wrong number (e.g. zero or the unmodified `a`), not a
byte-broadcast pattern.

### MUL still completely unwired

All three MUL samples (including the byte-crossing `16*16=256` case) XFAIL.
The bake spec referenced in the xfail reason
(`_set_layer11_mul_partial` / `_set_layer12_mul_combine`) hasn't been
migrated yet. None of today's Wave 2 ALU FFN migrations touched MUL.

### Heap tests still completely unwired

Both `test_si_then_li` and `test_sc_then_lc` XFAIL. The xfail reasons
reference `_set_layer14_mem_generation` / `_set_layer15_memory_lookup` /
`_set_layer7_memory_heads` — none of those have been migrated yet either.

## Recommendation for follow-up

1. **Add a focused investigation for DIV/MOD output reduction.** The fact
   that values are byte-replicated suggests the divmod neural path
   produces a 4-byte tensor and the AX output collapse step is missing
   (or is being bypassed in pure-neural). A 30-min trace of the output
   layer with `a=20, b=5` should pinpoint it. The fact that the markers
   were already removed (and presumably someone tried to land them as
   passing) makes this look like the *last* mile of divmod, not the
   first.

2. **Consider re-adding xfail markers to `test_div_small` and
   `test_mod_small`** until that fix lands, so the suite stays green
   on CI. They are currently red.

3. **MUL and heap remain blocked on their original bake-spec migrations.**
   No useful signal from this sample about them; defer until someone
   tackles `_set_layer11_mul_partial` / `_set_layer14_mem_generation`.

4. **Larger MUL sweep not needed yet** — all three samples (`3*4`,
   `5*5`, and the byte-crossing `16*16`) hit the same XFAIL, so MUL is
   uniformly unwired rather than partially working.

## Reproducer

```bash
PICK_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
CUDA_VISIBLE_DEVICES=$PICK_GPU timeout 2400 python -m pytest \
    c4_release/tests/test_pure_neural_heap_div.py -v --tb=line --timeout=300 \
    -k "mul_small[3-4-12] or mul_small[5-5-25] or mul_small[16-16-256] or \
        div_small[20-5-4] or mod_small[7-3-1] or test_si_then_li or test_sc_then_lc"
```

Wall clock ~12 min on a single GPU.
