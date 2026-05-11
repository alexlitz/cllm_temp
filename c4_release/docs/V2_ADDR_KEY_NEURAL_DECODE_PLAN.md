# V2 ADDR_KEY Neural Decode Plan

Goal: Replace the Python ``NeuralVMEmbedding._inject_mem_metadata`` write
of ``ADDR_KEY[lo, 16+hi, 32+top]`` one-hots at MEM val-byte positions with
a baked neural decode that matches BLOG_SPEC.md:830:

> First the simple case key value retrieval, we want an exact match for
> the key so we take the binary key break it up into bytes or nibbles,
> perform an equality check for each byte or nibble, then in a subsequent
> layer we perform a logical AND over those results. [...] We can then
> use the outputs of the second layer as a one hot mask for the next
> layer, which can be implemented as a MoE router for efficiency.

Status: scoping + Phase 0 bake (gated, ``enable=False``). The ADDR_KEY
metadata stays produced by the Python ``_inject_mem_metadata`` until the
bake is verified bit-identical and flipped on.

Source files referenced:

- ``c4_release/docs/BLOG_SPEC.md`` (lines 814-870, canonical for memory
  + lookup).
- ``c4_release/neural_vm/neural_embedding.py`` (``_inject_mem_metadata``).
- ``c4_release/neural_vm/setup_helpers.py``
  (``_set_layer13_mem_addr_gather`` — already gathers addr bytes from
  MEM positions to val byte positions, but writes them as 8-bit
  ``CLEAN_EMBED_LO/HI`` decompositions into ``ADDR_B*_LO/HI``, not as
  the 12-bit ``ADDR_KEY[lo, 16+hi, 32+top]`` encoding ``_inject_mem_metadata``
  produces).
- ``c4_release/neural_vm/unified_compiler/ops/l4_ops.py``
  (``make_layer4_sp_to_addr_key_op``) — the closest existing pattern: a
  gated bake that stages SP-byte CLEAN_EMBED nibbles into the ADDR_KEY
  band at the AX marker.
- ``c4_release/neural_vm/vm_step.py``
  (``_set_layer15_memory_lookup``) — current ADDR_KEY consumer.
- ``c4_release/neural_vm/unified_compiler/ops/all_core_ops.py`` — op
  registration ordering.

---

## 1. What ``_inject_mem_metadata`` does today

Walks ``token_ids`` for each batch row, finds each 9-token MEM section
``[MEM, addr_b0..3, val_b0..3]``, and for each ``byte_off ∈ {0,1,2,3}``
writes at position ``i + 5 + byte_off``:

| Dim band                         | Value | Notes                              |
| -------------------------------- | ----- | ---------------------------------- |
| ``MEM_STORE`` (459)              | 2.0   | L15 K-side discrimination          |
| ``MEM_VAL_B{byte_off}``          | 1.0   | L15 byte select dim                |
| ``ADDR_KEY + lo``  (206..221)    | 1.0   | byte address lo nibble (4 bits)    |
| ``ADDR_KEY + 16 + hi`` (222..237) | 1.0   | byte address hi nibble (4 bits)    |
| ``ADDR_KEY + 32 + top`` (238..253) | 1.0   | byte address top nibble (4 bits) |

where ``byte_addr = addr + byte_off`` and ``addr`` is the 32-bit
little-endian decode of ``token_ids[i+1..i+5]`` (treated as raw 8-bit
byte values).

The ``MEM_STORE`` write on the MEM marker (``x[b, i, MEM_STORE] = 2.0``)
and the ``MEM_VAL_B0..B3`` writes on val byte positions are already
produced neurally by ``_set_layer2_mem_byte_flags`` (L2 FFN, see
``setup_helpers.py:100``) and by ``L6 head 6`` for the marker. The only
piece that is purely Python is the per-val-byte **ADDR_KEY nibble
decomposition**.

---

## 2. Why the existing bakes don't already cover this

``_set_layer13_mem_addr_gather`` (L13 attn, ``setup_helpers.py:1207``)
already attends from MEM val byte positions to MEM addr byte positions
``d=1..3`` and copies the ``CLEAN_EMBED_LO/HI`` (16-dim one-hot lo/hi
nibble of the addr byte's *raw 8-bit token value*) into:

- addr byte 0 → ``ADDR_B0_LO`` (12..27) + ``ADDR_B0_HI`` (206..221)
- addr byte 1 → ``ADDR_B1_LO`` (28..43) + ``ADDR_B1_HI`` (222..237)
- addr byte 2 → ``ADDR_B2_LO`` (44..59) + ``ADDR_B2_HI`` (238..253)

This *almost* produces the same encoding as ``_inject_mem_metadata``
because ``ADDR_B0_HI / ADDR_B1_HI / ADDR_B2_HI`` (the 16-dim hi-nibble
bands) **alias** the ADDR_KEY band ``206-253``. So L13 already places
addr byte 0's hi nibble into ADDR_KEY[0..15], byte 1's hi nibble into
ADDR_KEY[16..31], byte 2's hi nibble into ADDR_KEY[32..47].

But ``_inject_mem_metadata`` writes the *lo nibble of byte_addr* into
``ADDR_KEY[0..15]``, the *hi nibble of byte_addr* into
``ADDR_KEY[16..31]``, and the *low nibble of (byte_addr >> 8)* into
``ADDR_KEY[32..47]``. The semantic is:

- ADDR_KEY[lo]  = ``byte_addr & 0xF``        — bits 0..3 of full addr
- ADDR_KEY[16+hi] = ``(byte_addr >> 4) & 0xF`` — bits 4..7
- ADDR_KEY[32+top] = ``(byte_addr >> 8) & 0xF`` — bits 8..11

i.e., ADDR_KEY is the **lower 12 bits of the byte address**, decomposed
into three 4-bit nibbles.

L13's writes happen to land in the right *bands*, but they encode the
**nibbles of three separate addr byte token values**, not the nibbles of
``byte_addr``. The two encodings coincide when:

1. ``byte_off == 0`` (so ``byte_addr == addr``), AND
2. ``addr_b0_hi == addr_b0_lo + carry_from_byte_off`` (trivially true at
   byte_off=0).

For ``byte_off ∈ {1, 2, 3}``, the encodings diverge because
``_inject_mem_metadata`` advances the byte address by ``byte_off`` (with
carry propagation across nibble/byte boundaries) before nibbling, while
L13 just copies the unchanged addr byte values.

In other words: **today's L13 gather + today's
``_inject_mem_metadata`` write to overlapping dims with different
content per val byte position.** This is mediated by:

- ``_set_layer14_clear_addr_key_pollution`` (an L14 FFN) clears parts of
  the band post-L13 to give ``_inject_mem_metadata``'s injection
  authority. The "pollution" comment refers exactly to L13's
  CLEAN_EMBED copies overwriting val-byte ADDR_KEY content.
- The L15 K-side reads ADDR_KEY via 48 nibble dims with high scale
  (``L_addr = 10.0``), so any per-val-byte mismatch from L13's
  raw-byte copies vs the inject-time ``byte_addr`` decode would break
  the match.

The upshot: there is **no existing bake that produces ``byte_addr``
(addr + byte_off) and decomposes it into the three nibbles.** The
``byte_addr`` increment-with-carry is the genuinely new computation that
the migration must replicate.

---

## 3. Two design options

### Option A — Phase-0 staging (gated, this PR)

Add a new attention head + FFN that:

1. **Attention head**: at MEM val byte positions (gated by
   ``MEM_VAL_B0|B1|B2|B3``), attend to the addr byte at the *same MEM
   section* (gated by ``L1H1[MEM]`` for d=1, etc., the same
   per-byte-offset masks ``_set_layer13_mem_addr_gather`` uses).
2. **FFN**: take the 8-bit addr byte value from CLEAN_EMBED_LO/HI and
   compute the 12-bit ``byte_addr = addr + byte_off`` nibble
   decomposition, emitting one-hots into ADDR_KEY.

This is the most direct port of the Python code: same input
(``addr``, ``byte_off``), same output (the three nibble one-hots).

**Pros:**
- Bit-identical to ``_inject_mem_metadata`` by construction.
- Plays nicely with the L13/L14 pollution-clear pipeline (we can
  schedule the new op after L14's clear, so it owns the final write,
  same as ``_inject_mem_metadata`` does today).
- ``MEM_VAL_B{0..3}`` flags are already neural (L2 FFN), and
  ``byte_off`` is implicit from which flag is set.
- The FFN nibble decomposition is a 256-input × 12-bit-output lookup
  table — fully baked, no Python.

**Cons:**
- Doesn't yet implement the BLOG_SPEC.md:830 "binary key + per-byte AND
  + MoE one-hot" pattern. It just replicates ``_inject_mem_metadata``
  neurally.
- Adds another op that must be reasoned about alongside L13/L14 in the
  pollution dance. Eventually L13's CLEAN_EMBED copies into ADDR_KEY
  bands and the L14 clear could be deleted once this op fully owns
  ADDR_KEY at val byte positions.

**Scope:** Bake the op gated by ``enable=False``. Add a unit test that
asserts the bake is byte-identical to ``_inject_mem_metadata`` over a
range of address+byte_off combinations. Don't flip the gate.

### Option B — BLOG_SPEC.md:830 binary key + MoE one-hot

Rewire L15 K-side to no longer read the 48-dim ``ADDR_KEY`` one-hot
encoding. Instead:

1. K-side reads the raw 8-bit address bytes (via L13's CLEAN_EMBED
   gathers, no advance-by-byte_off needed at all).
2. Q-side advances PC + byte_off and decomposes its own query address
   into 4 byte values (already done by L4 FFN's ``FETCH_LO/HI`` and
   L8 multibyte head).
3. Per-byte equality check is the standard dot product on
   ``CLEAN_EMBED_LO[query] ↔ CLEAN_EMBED_LO[key]`` (16-dim one-hot
   match per byte).
4. N-way AND across the 4 byte equality scores: realized implicitly by
   summing the 4 byte-wise scores under a single softmax. (A score-
   budget design with high scale produces effective AND behaviour
   because all-4-bytes-match collects 4× the score-margin of 3-of-4
   matches.)
5. The selected attention output (value) is the MEM val byte
   ``CLEAN_EMBED_LO/HI`` — the value associated with the matched key.
   Spec's "expert that returns the value" reduces to "attention head
   that copies the embedding."

**Pros:**
- Spec-aligned final state. Drops ADDR_KEY entirely as a residual band;
  drops ``_inject_mem_metadata`` entirely; drops the L13 ADDR_B*_HI
  pollution + L14 clear dance.
- Naturally handles arbitrary 32-bit addresses (today's ADDR_KEY is
  truncated to 12 bits, which is why L15 has separate ``ADDR_B*_LO/HI``
  K-side projections to cover bits 16..23).

**Cons:**
- Touches L15 + L8 + L13 + L14 + the dim registry. Multi-day.
- L15 score budgets are tight (``_set_layer15_memory_lookup`` has very
  carefully tuned constants — see the comment block around
  ``vm_step.py:6359``). A change in K-side encoding from 48-dim one-hot
  to 4×16-dim per-byte one-hot must preserve the score margins.
- The MoE-router formulation requires actually adding a kind="block"
  op that wraps the L15 attention output in a routed expert FFN. The
  current architecture doesn't have MoE infrastructure.

**Scope:** explicitly out-of-scope for this PR; deferred to Phase 2
(separate work item).

### Recommendation

Phase 1 (this PR): Option A, gated. Adds the bake without enabling it,
adds the regression test, and updates the documentation.

Phase 2 (separate): once Option A is enabled in CI and stable, attempt
Option B by rewiring L15 K-side. Validate against the L8/L15 attention
score budgets. Eventually delete the residual ``ADDR_KEY`` band.

---

## 4. Implementation sketch — Option A

Add ``make_layer14_addr_key_decode_op(enable=False)`` to
``c4_release/neural_vm/unified_compiler/ops/l14_ops.py``. Layer 14 is
chosen because:

1. ``_set_layer13_mem_addr_gather`` (L13) has already written addr byte
   values into the ``ADDR_B*_LO/HI`` bands at val byte positions, so
   the input data is available there.
2. ``_set_layer14_clear_addr_key_pollution`` (L14 FFN) runs at L14 and
   clears the bands so the Python ``_inject_mem_metadata`` write
   becomes the authority. Once the new bake takes over, the L14 clear
   becomes unnecessary, but for Phase 0 (parallel mode) it must
   continue to run because ``_inject_mem_metadata`` is still firing.
3. The new op runs at L14 after the clear, so it owns the final
   ADDR_KEY band at val byte positions.

Sketch:

```python
def make_layer14_addr_key_decode_op(enable: bool = False) -> Operation:
    """L14 FFN: decode (addr_b0, byte_off) → ADDR_KEY[lo, 16+hi, 32+top].

    BLOG_SPEC.md:830 binary-key decode. Produces the same nibble one-hot
    encoding as the Python ``_inject_mem_metadata``, baked into FFN
    weights.

    Per-val-byte position (gated by MEM_VAL_B{0..3}):
      byte_addr = addr_b0 + byte_off  (with carry into addr_b1 if needed)
      ADDR_KEY[byte_addr & 0xF]        = 1.0   # lo nibble
      ADDR_KEY[16 + (byte_addr>>4) & 0xF] = 1.0  # hi nibble
      ADDR_KEY[32 + (addr_b1 + carry) & 0xF] = 1.0  # top nibble

    Input residual reads:
      - ADDR_B0_LO[k]  for k in 0..15  : addr byte 0 lo nibble (from L13)
      - ADDR_B0_HI[k]  for k in 0..15  : addr byte 0 hi nibble (from L13)
      - ADDR_B1_LO[k]  for k in 0..15  : addr byte 1 lo nibble (from L13)
      - MEM_VAL_B{0..3}                : byte_off + position gate (from L2)

    Output residual writes:
      - ADDR_KEY + {lo, 16+hi, 32+top}: one-hot nibble of byte_addr
    """
```

The bake encodes a 16 × 4 lookup table per output band (each combination
of ``addr_b0_lo`` × ``byte_off`` yields a deterministic
``byte_addr_lo`` plus a carry-out for the hi nibble; similarly for the
hi nibble). 256 (lo×hi) × 4 (byte_off) = 1024 FFN units max — fits
comfortably.

The matching FFN structure:

```
for byte_off in range(4):
    gate_dim = [MEM_VAL_B0, MEM_VAL_B1, MEM_VAL_B2, MEM_VAL_B3][byte_off]
    for lo in range(16):
        for hi in range(16):
            byte_addr = ((hi << 4) | lo) + byte_off
            new_lo = byte_addr & 0xF
            new_hi = (byte_addr >> 4) & 0xF
            carry  = byte_addr >> 8  # 0 or 1
            # 3-way AND: MEM_VAL_B{byte_off} + ADDR_B0_LO[lo] + ADDR_B0_HI[hi]
            ffn.W_up[unit, gate_dim] = S
            ffn.W_up[unit, ADDR_B0_LO + lo] = S
            ffn.W_up[unit, ADDR_B0_HI + hi] = S
            ffn.b_up[unit] = -S * 2.5
            ffn.W_down[ADDR_KEY + new_lo, unit] = 2.0 / S
            ffn.W_down[ADDR_KEY + 16 + new_hi, unit] = 2.0 / S
            # Carry-out also bumps ADDR_B1_LO virtually; the top nibble
            # then needs separate units that take addr_b1_lo + carry.
            unit += 1
```

A second set of units takes ``ADDR_B1_LO`` (from L13) plus carry-out
(implicit in whether ``lo + byte_off >= 16``) to produce ``ADDR_KEY +
32 + top``. Total bake unit count: ~1024 + ~256 ≈ 1300 FFN units.

The output band write **collides** with whatever was already there
(L13's pollution, then ``_inject_mem_metadata``'s injection). For Phase
0 with ``enable=False`` no writes happen; for Phase 1 with
``enable=True`` the order is:

1. L13 writes CLEAN_EMBED copies into ADDR_KEY bands (pollution).
2. L14 ``clear_addr_key_pollution`` zeros those.
3. L14 ``addr_key_decode`` (new) writes the correct nibble one-hots.
4. ``_inject_mem_metadata`` (still active in parallel) ALSO writes the
   same nibble one-hots into the same positions, causing
   double-writes.

To make parallel mode safe, the new op writes with weight ``0.5`` so the
combined parallel-mode output stays in ``[0, 1]`` for each one-hot bit.
Once ``_inject_mem_metadata`` is deleted, the weight goes back to
``1.0``.

Actually, the cleaner design is for the new op to be a true no-op when
``enable=False`` and run alone when ``enable=True`` (with the Python
injector deleted). The intermediate "both fire" mode is a debugging /
validation mode triggered by setting both ``enable=True`` AND patching
the FFN's output scale to 0.5 — not the production path.

---

## 5. Verification plan

### Phase 0 — gated bake registered (this PR)

1. Add ``make_layer14_addr_key_decode_op(enable=False)`` to
   ``l14_ops.py`` and ``all_core_ops()``.
2. Add ``test_addr_key_neural_decode.py`` with:
   - Registration test (op is in ``all_core_ops``, lands at L14, is a
     no-op when disabled).
   - Manual bake test: construct an FFN with ``enable=True``, feed it
     a synthetic residual with known ``addr_b0_lo``, ``addr_b0_hi``,
     ``addr_b1_lo``, ``MEM_VAL_B{0..3}`` values, and verify the output
     ADDR_KEY band matches the Python ``_inject_mem_metadata`` decode
     for the same inputs.
3. Constraint gates: smoke + runtime_vanilla + layer_idx_consistency +
   compile_determinism all pass (because the bake is gated off).

### Phase 1 — flip the gate

Once Phase 0's parity test passes, in a separate PR:

1. Set ``enable=True``.
2. Move the L13 ``ADDR_B*_HI`` writes back to ``ADDR_B*_LO`` only (the
   new op owns ADDR_KEY).
3. Verify L14 ``clear_addr_key_pollution`` still does the right thing
   (clear ``ADDR_B*_HI`` so the new op's writes are authoritative).
4. Run the full test gate.

### Phase 2 — delete the Python injector

1. Delete ``_inject_mem_metadata`` from ``neural_embedding.py``.
2. Delete the ``self._inject_mem_metadata(...)`` call from
   ``NeuralVMEmbedding.forward``.
3. Run the full test gate. Any failures indicate a bake bug — track
   down via per-MEM-section ADDR_KEY tensor diff.

---

## 6. Risks

| Concern                                            | Phase 0 | Phase 1 | Phase 2 |
| -------------------------------------------------- | ------- | ------- | ------- |
| Breaks compile determinism                         | None    | None    | None    |
| Breaks L15 memory lookup score budget              | None    | Medium  | High if bake mis-encodes carry |
| Breaks L13/L14 ADDR_KEY pollution pipeline         | None    | Medium  | Low (L14 clear becomes vestigial) |
| Behavioral diff in addr_b0 + byte_off carry case   | None    | Medium  | Medium  |
| Performance                                        | Neutral | Slight gain | Solid gain (no Python loop) |

The carry case is the main correctness concern. ``addr_b0 + byte_off``
overflows when ``addr_b0_lo + byte_off >= 16`` (carry into hi nibble),
or when the full byte ``addr_b0 + byte_off >= 256`` (carry into
``addr_b1``). The FFN bake must enumerate all 16×16×4 = 1024 (lo, hi,
byte_off) combinations and emit the correct nibble decomposition for
each.

---

## 7. What this PR ships

- This design doc.
- The gated bake op ``make_layer14_addr_key_decode_op(enable=False)``
  registered in ``all_core_ops()``.
- A unit test that exercises the bake with ``enable=True`` against
  ``_inject_mem_metadata``'s output on synthetic inputs.
- No production behaviour changes — the gate stays off; existing tests
  remain byte-identical.

Future work (separate PRs): flip the gate (Phase 1), then delete
``_inject_mem_metadata`` (Phase 2), then optionally rewire L15 to use
the BLOG_SPEC.md:830 binary-key + MoE one-hot pattern (Phase 3).
