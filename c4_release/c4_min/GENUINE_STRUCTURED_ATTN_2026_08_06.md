# GENUINE STRUCTURED ATTENTION (C4_GENUINE_STRUCTURED_ATTN) — a fuller-genuine memory read at fast throughput

2026-08-06.  Answers: is there a GENUINE attention (real recompute, NOT draft-trusted) that
meets the fast throughput (~2.6 µs/step), and HOW genuine — pushing the single-dispatch's
partly-trusted VALUE read to a genuinely-recomputed softmax1+ALiBi read at O(structured) cost.
Gate `C4_GENUINE_STRUCTURED_ATTN` DEFAULT-OFF, runtime-only, golden `069cc32f` unchanged.

## 1. The genuineness spectrum characterized (what each path genuinely computes vs shortcuts)

| path (flag) | QUERY (address the read looks up) | KEY resolution (which store row wins) | VALUE (the retrieved bytes) | catches scenario E? |
|---|---|---|---|---|
| **draft-CAM** (`C4_DIRECT_CAM_BATCHED`, 2.16 µs) | model's `W_q` computed then **DISCARDED**; the address is INJECTED from `resolve_load_rows(draft)` | **INJECTED** (draft resolves addr→row) | **INJECTED** `store_log[row][1]` atom | **NO** (rubber-stamp) |
| **single-dispatch faithful** (`C4_FAITHFUL_SINGLE_DISPATCH`, 2.6 µs) | **GENUINE** — model's own `sign(W_q(x))` address decoded in-graph, compared to draft | **GENUINE INDEX** — `latest_write_wins(model_addr, draft.store_log)` (searchsorted / `C4_HASH_CAM` O(1)) | **draft-trusted ATOMS** — the winning row's value is the Python `store_log[frame][1]` dict value; NO softmax1, NO `W_k`/`W_v`, NO physical K/V | **YES** (addr layer + value-vs-store_log) |
| **GENUINE STRUCTURED** (`C4_GENUINE_STRUCTURED_ATTN`, this) | **GENUINE** (same in-graph `W_q` decode) | **GENUINE O(1)** — the hash-CAM winner row | **GENUINE RECOMPUTE** — runs the model's `softmax1([score_winner, sink])·[W_v·V_row, 0]`; `score = (W_q·W_k)·scale − slope·\|dist_tok\|` over the winner's **PHYSICAL** ADDR_BIN bits, value decoded from its **PHYSICAL** VAL_NIB nibbles | **YES** + more (below) |
| **full O(S) softmax** (`C4_FAITHFUL_ATTN_EVICT`, ~8 ms) | GENUINE | GENUINE (scores ALL S rows) | GENUINE (real softmax1+ALiBi over physical K/V) | YES |

### The precise genuineness gap the single-dispatch has (and what the full softmax verifies that it doesn't)

The single-dispatch VALUE = `store_log[winner_frame][1]` — a **Python dict read**.  It is
"genuine" in the INDEX (which frame wins, via the model's own address) but it TRUSTS the
store_log's `(addr, val)` atoms and the **latest-write-wins == softmax1-CAM-winner IDENTITY
ARGUMENT**.  It never runs the softmax1, never applies `W_k`/`W_v`, never touches the store
row's physical residual encoding.  The full O(S) softmax verifies exactly those four things
the single-dispatch assumes:

1. the winning row's **physical KEY** (`ADDR_BIN` bits) actually encodes the queried address
   (a mis-encoded K scores low → the softmax1 sink wins → ZFOD);
2. the **softmax1 normalization + the +1 sink** genuinely clear the match (an unwritten /
   evicted address genuinely resolves to 0, not a claimed value);
3. the **ALiBi recency** term the latest-write-wins identity *assumes* (the real recall
   horizon — see §4);
4. the winning row's **physical VALUE** (`VAL_NIB` nibbles) actually decodes to the claimed
   value (a mis-encoded V decodes wrong).

## 2. The fuller-genuine structured attention (real query + genuine key + genuine value, O(1))

The structural fact: for a §Memory read exactly ONE stored address matches, so the softmax1 is
a hard argmax over the same-address stores with an ALiBi recency tiebreak — the winner is the
LATEST store to the queried address.  So the GENUINE softmax is O(1) if we (a) genuinely
compute the query (the in-graph `W_q` sign-decode the faithful path already does), (b) genuinely
resolve the winning KEY row (the O(1) hash-CAM), and (c) genuinely READ the value by RUNNING
the model's softmax1+ALiBi over that ONE winner row + the +1 sink — reconstructing the winner's
PHYSICAL K and V from its store-residual encoding and applying the model's baked
`W_q`/`W_k`/`W_v`/`W_o` numerics (`EFF`/`BIAS`/`MEM_ALIBI_SLOPE`/`FRAME_LEN` read live).

`value = decode_nibbles( softmax1([score, sink]) · [ W_v·V_winner , 0 ] )`, per read, at
O(1) hash probe + a constant-width (HD) score/value reconstruct — O(reads), NOT O(reads·S).

Files: `genuine_structured_attn.py` (new); `faithful_single_dispatch.py`
(`build_faithful_precompute._resolve` swaps in `genuine_structured_value` when the flag is on —
the SAME build-thread precompute hook `C4_HASH_CAM` uses).

### µs/step — MEETS the fast throughput

The value re-resolution is a **build-thread precompute** (pipelined off the critical path — the
critical-path compare `verify_faithful_fast` is UNCHANGED at ~0.005 µs/step; the in-graph `W_q`
address dispatch is UNCHANGED).  So GSA's throughput delta == the delta of the numpy value
precompute, measured on the REAL 6315-step deep-loop draft (`_agent_gsa_precompute_ab.py`):

| precompute (build-thread value re-resolution) | µs/step |
|---|---|
| single-dispatch (searchsorted) | 0.951 |
| `C4_HASH_CAM` (O(1) atom) | 1.034 |
| **`C4_GENUINE_STRUCTURED_ATTN`** | **1.076** |

At doom scale (`_agent_gsa_throughput.py`, S=113K committed stores, 42K reads/batch): GSA =
**0.34 µs/read**.  The faithful **dispatch floor is ~2.6 µs/step** (in-graph `W_q` address
decode, unchanged by GSA).  GSA build **1.076 µs/step < 2.6 → DISPATCH-BOUND**: the genuine
structured path meets the ~2.6 µs/step fast throughput (identical dispatch to the single-dispatch
faithful path; +0.13 µs/step build over single-dispatch, hidden under the dispatch).

**One-line: YES — there is a genuine attention (real recompute, not draft-trusted) that meets
the fast ~2.6 µs/step throughput, and it is MORE genuine than the single-dispatch (it runs the
model's softmax1+ALiBi over the physical winner K/V instead of trusting the store_log atoms).**

## 3. Genuineness gate — catches scenario E AND does MORE genuinely than single-dispatch

Scenario E (planted-wrong-read): store 66@200 then 999@200; LI@200; draft injects the STALE 66
at the CORRECT address.  `_agent_gsa_genuineness.py` (through the real model):

* **Caught (same as single-dispatch):** the genuine structured read recomputes `999`
  (softmax1+ALiBi over the winner store row) ≠ the draft's stale `66` → `cam_value` FAIL at the
  LI step (`model_value=999 draft_value=66 model_addr=200`).
* **MORE genuine than single-dispatch (quantified — what it recomputes that SD does not):**
  1. **VALUE from physical V:** corrupting the winner store row's physical `VAL_NIB` nibbles →
     the structured read decodes the corrupted value (it runs `W_v`/`W_o` + softmax1, NOT a
     `store_log[frame][1]` atom the single-dispatch trusts);
  2. **KEY-binding via the softmax1 sink physics:** querying an unwritten address → the
     structured read genuinely SCORES the (absent) key below the softmax1 `+1` sink → ZFOD 0
     (single-dispatch's atom read never runs the sink physics — it relies on the address→slot
     map identity);
  3. **SCORE of the physical key:** a full physical-address match nets `+EFF` (softmax1_w=1.0);
     any 1-bit-off physical key nets `−EFF` (softmax1_w=0, below sink) — proving the read scores
     the winner's PHYSICAL address bits, not a claimed address.

So GSA recomputes, per read, the 4 quantities §1 lists as the single-dispatch's trusted
assumptions — the softmax1 score of the physical key, the +1 sink normalization, the ALiBi
recency, and the value from the physical V nibbles — none of which the single-dispatch computes.

## 4. Byte-exact (L-inf=0) + golden

* **Pure-numpy** (`_agent_gsa_pure_numpy.py`): 8008 reads / 200 random store-logs —
  `genuine_structured_value == _genuine_value_at == resolve_value_hashed`, L-inf=0.
* **Real model path** (`_agent_gsa_byteexact.py`): the DIV-free battery
  (add/mul/mem/func_lev/bigimm/jmp/bnz) + the **6315-step deep loop** — genuine-structured
  faithful SD == fast single-dispatch == the wanted answer, L-inf=0, verify ACCEPTS every step
  (2404 addr + 2404 value + 6315 routing checks, 0 false positive).
* **Golden `069cc32f` UNCHANGED** flag-OFF **and** flag-ON (`CUDA_VISIBLE_DEVICES=""
  python -m c4_min._fingerprint_build`, and with `C4_GENUINE_STRUCTURED_ATTN=1`) — the flag is
  a runtime resolver swap in `build_faithful_precompute`, no weight authoring, weight-neutral.
* **VRAM:** the resolver is pure CPU/numpy (no VRAM); the faithful graph is unchanged from the
  single-dispatch faithful path (~14–16 GB @ chunk 131072, fits the 24 GB card).

### The recall horizon (a genuineness FEATURE, honest limit)

Because the structured read runs the model's REAL softmax1+ALiBi, it honors the model's REAL
recall horizon (`EFF/(slope·FRAME_LEN)` = **16 667 frames** = 500 000 tokens, wall #6): a store
further back than that scores below the +1 sink → ZFOD.  The distance-blind `_genuine_value_at`
/ hash-atom resolvers return the raw latest value at ANY distance, so BEYOND the horizon they
OVER-claim a value the model cannot recall.  Real programs stay WITHIN the horizon (the deepest
1096-corpus store→load gap is ~8 361 frames << 16 667; the byte-exact battery + the 120K doom
slice all pass), so the structured read is byte-exact there.  Beyond the horizon it genuinely
returns ZFOD — matching the FULL O(S) softmax (the most-genuine reference), i.e. MORE correct
than the distance-blind resolvers (`_agent_gsa_horizon.py`: within-horizon 6/6 agree; beyond,
GSA→0 == the model's genuine fade).

## Bottom line

`C4_GENUINE_STRUCTURED_ATTN` closes the single-dispatch's value-genuineness gap: it replaces the
draft-trusted `store_log` value ATOM with a genuine softmax1+ALiBi READ over the ONE hash-CAM
winner row (physical K/V), recomputing the score / sink / recency / value-decode the
single-dispatch assumes — at 1.076 µs/step build-thread (< the ~2.6 µs dispatch floor →
dispatch-bound, meets the fast throughput).  Byte-exact within the model's real recall horizon
(where all real programs live), golden `069cc32f` unchanged.  It does NOT reach the full-softmax's
verification that the KEY/VALUE bytes are PHYSICALLY RESIDENT IN THE GPU KV TENSORS (it
reconstructs them from the store-residual encoding, an identity argument on the encoding — a
narrower trust than the store_log atom, but not zero); closing that last gap is the O(S) softmax
gathering the actual cache rows, which is the ~3000× path this replaces.
