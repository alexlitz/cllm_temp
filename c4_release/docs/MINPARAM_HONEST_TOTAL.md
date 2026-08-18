# MINPARAM_HONEST_TOTAL — the honest byte-exact real-attention nonzero total of the fully-unrolled min-param clever VM, and a blunt blog audit

*Every number below is code-traced: produced by **instantiating the real cell /
head modules and calling `count_nonzero()` / the census functions**, or by running
the byte-exact verify path. Golden
`174ece66edff1bb5ab8e9213e484bb1b9560f44ab6c23077a366491543d05637`
(short `174ece66`, `python -m c4_min._fingerprint_build`) re-verified **UNCHANGED**
after this work (docs-only, off every build path). Branch
`minparam-fp32-limb-audit-2026-08-09` off `consolidate-0.5b-2026-07-22`.*

This corrects the **idealized 4,887** (whole-value + direct-CAM) 39-op figure from
`docs/MINPARAM_CENSUS.md` (commit `0580b505`) for the two shortcuts it takes:

| shortcut | idealized | honest | why |
|----------|-----------|--------|-----|
| MEMORY = 10 (direct-CAM) | O(1) host-side pointer-walk + gather, **not** attention | **166** = real blog-spec softmax1-KV attention head | the 10 is a resolver, the model never runs `softmax1(Q·Kᵀ)·V` |
| ARITH = whole-value | one fp scalar, **NON-byte-exact above 2²⁴** in fp32 | **same 51-cell, 0 new params** (byte-exact via LIMBS) | limbs restore byte-exactness by spending DEPTH, not width |

**Bottom line up front:**

| convention | idealized (whole-value + direct-CAM) | **HONEST (byte-exact limb + real softmax-KV attn)** | Δ |
|------------|------|------|---|
| 39-op **unrolled** | 4,887 | **5,043** | **+156** |
| 39-op **looped**   | 2,801 | **2,957** | +156 |

The entire +156 delta is the real-attention memory head (166 − 10). The byte-exact
limb requirement adds **+0** stored params (it reuses the identical shared decode
cell). So **4,887 is *nearly* right on parameter count** — the whole-value form was
never wrong about *how many* weights arith stores (it reuses one 51-cell either
way); it was wrong about **byte-exactness** (non-exact >2²⁴) and the memory head
was a **shortcut** (a resolver, not attention). Fixing both honestly lands at
**5,043 unrolled** — and the *dense realizable* model that carries those 5,043
nonzeros is **~217 M** tensor params (mostly structural zeros).

---

## 1. The real softmax-KV attention memory head — the honest replacement for the 10-param direct-CAM

The census's memory `10` (`MemoryCAMCell` = 8 nibble-match lanes + temp +
value_read) is an **O(1) direct-CAM shortcut**, not attention. Its resolver is a
**host-side pointer/dict walk over the write log** that produces a store ROW INDEX
per read; the GPU then direct-gathers that one row. The GPU **never runs
`softmax1(Q·Kᵀ)·V` over the store** — its own docstring
(`DirectCAMReadHead`, `examples/clever_honest_attn_realtime.py:102-104`) says so
verbatim: *"the resolver is a host-side pointer/dict walk over the write log, NOT a
GPU softmax over the store."*

**The honest count = the model's OWN baked §Memory head** (`bake_memory_head`,
`c4_min/blogspec_memory.py:263-348`) — a genuine `softmax1(Q·Kᵀ·scale + alibi)·V`
over the whole store, keyed on the **32-bit binary address** (±smag CAM), with a
ZFOD store-bias channel, a store-role penalty gate, an ALiBi recency slope, and a
value relay. **Instantiated and counted** (baked into a 1-head `Attn`, `MemoryLayout`
`D=188`, `ADDR_BITS=32`, `NIB_PER_REG=16`):

| tensor | nonzero | what |
|--------|--------:|------|
| `W_q` | 66 | 32 addr-bits × 2 (±smag QRY_BIN + ONE) + ZFOD IS_LOAD (1) + PEN IS_LOAD (1) |
| `W_k` | 67 | 32 addr-bits × 2 (±smag ADDR_BIN + ONE) + ZFOD IS_STORE (1) + PEN ONE+IS_STORE (2) |
| `W_v` | 16 | VAL_NIB store-value nibble relay |
| `W_o` | 16 | AX-band write of the value nibbles |
| `alibi_slopes` | 1 | recency (latest-write-wins tie-break) |
| **TOTAL** | **166** | the honest real softmax1-KV memory head |

**Byte-exact** (`python -m examples.clever_real_softmax_kv_memory --verify`): the
genuine O(S) softmax1+ALiBi read == the O(1) direct-CAM gather, **L-inf=0** on the
full Doom battery (hit / miss-ZFOD / latest-write-wins / STACK / FRAMEBUFFER) at
S ∈ {8K, 65K, 262K} — same latest-write-wins semantics, now *actually computed by
softmax over the whole store*.

**Caveat on the `RealSoftmaxKVMemoryHead` module's OWN param count.** That module
(`clever_real_softmax_kv_memory.py`) stores only `W_v`/`W_o` as
`torch.nn.Parameter`s (dense `d_model×d_model`, `torch.randn`) and bakes the
score-side numerics (EFF/BIAS/slope/±smag) as **constants inside
`scores_over_store`**, not tensors. So counting *its* Parameters gives **8,192**
(2·64² at d=64) — a dense "shape-budget" proxy, not the real head. The genuine
sparse head is the baked `bake_memory_head` = **166**, which is what a real model
carries. We use **166** as the honest memory count.

> Memory-cost note (not params): the honest read is **O(S)** per step — at S=262K it
> is ~thousands× the O(1) direct-CAM gather (the direct-CAM shortcut is
> **load-bearing for realtime**, not just for params). See the module's `--bench`
> verdict; that is a *time* cost, orthogonal to this param audit.

---

## 2. Byte-exact LIMB arith — adds **0** stored params (depth, not width)

Shortcut #2 is precision. The whole-value construction holds a value in **one fp
scalar**; fp32's 24-bit mantissa can't hold a 32×32→64-bit product, so the
whole-value machine is **NON-byte-exact above 2²⁴** in fp32 (measured:
**100,000/100,000** random large-operand MULs mismatch the exact product), and even
fp64 loses the product above 2⁵³ (→ needs fp128 GPUs don't have).

The byte-exact fix is a **LIMB decomposition** (`examples/clever_fp32_fullops.py`):
MUL in 8-bit limbs (radix 256) — every partial ≤ 255² = 65,025, worst column
accumulator **260,864 < 2²⁴** (a 64× fp32 margin); DIV on 16-bit halves. Verified
`python -m examples.clever_fp32_fullops --verify`: **full 64-bit MUL 20,000/20,000
byte-exact, FixedMul/FixedDiv 1,618/1,618 byte-exact, ADD/DIV radix-4096 exact —
NO fp64/fp128 anywhere**.

**The param consequence: none.** The limb carry-resolve is the **same
difference-min floor decode** — the identical shared `ArithCell`:

```
ArithCell.count_nonzero() = 51  (fp32 AND fp64, identical)
  embed 12 + W_q/W_k/W_v/W_o identity 16 + cand_center 10 + cand_value 9 + scalars 4
```

The limb-ness is a **DATAPATH** change (more forward-passes / carry columns per
step: MUL depth 20, DIV 15, ADD 15 at radix 4096) — it spends **DEPTH** to keep
every value < 2²⁴, storing **no new weights**. So the honest byte-exact-limb arith
count is **exactly the whole-value census arith**: **2,091 unrolled / 153 looped /
51 distinct** — the whole-value 2,091 was *right on params*, *wrong on
byte-exactness*, and the limb fix restores exactness at **+0 params**.

---

## 3. THE HONEST TOTAL — byte-exact, real-attention, fully-unrolled 39-op

Assembled from the code-traced components (`assembled_machine_census` +
`syscall_param_account`, memory swapped 10→166, arith unchanged):

| bucket | nonzero (unrolled) | note |
|--------|-------------------:|------|
| byte-exact-limb arith (51-cell × 41 places) | 2,091 | **same as whole-value** — limbs add depth, not params |
| bit-serial bitwise (OR/AND/XOR, one cell) | 8 | radix-2 peel (LUT-form would be 23,792) |
| **REAL softmax1-KV memory head** | **166** | ← replaces the 10-param direct-CAM shortcut |
| sequencer + one-hot dispatch | 256 | fetch-decode-execute fabric |
| embed + LM-head framing | 194 | 22 + 4×43 per-layer framing |
| neural stdin I/O head (8 heads) | 2,328 | W_v 2048 + W_o 256 + W_q 8 + W_k 8 + alibi 8 |
| **HONEST 39-op UNROLLED TOTAL** | **5,043** | vs idealized **4,887** |
| **HONEST 39-op LOOPED TOTAL** | **2,957** | vs idealized **2,801** |

**Delta attribution (vs the 4,887 idealization):**

| source | Δ nonzero |
|--------|----------:|
| real softmax-KV memory head (166 − 10 direct-CAM) | **+156** |
| byte-exact limbs (same 51-cell decode reused) | **+0** |
| **total honest delta** | **+156** |

**Every bit of the correction is the memory head.** The 4,887 was honest about the
*number* of arith weights (limbs don't change it) — its two sins were (a) the
memory `10` was a resolver, not attention, and (b) the whole-value arith was
non-byte-exact >2²⁴ (fixed by limbs at 0 param cost). The honest byte-exact
real-attention fully-unrolled 39-op machine is **5,043 nonzero**.

---

## 4. NONZERO vs DENSE — what "fully-unrolled language model params" means

Two legitimate senses of "params of the fully-unrolled language model," and they
differ by **~43,000×**:

| sense | 39-op unrolled | what it counts |
|-------|---------------:|----------------|
| **HONEST NONZERO (signal)** | **5,043** | every weight-tensor position holding a non-zero, byte-exact, real-attention |
| **DENSE realizable (tensor)** | **~216.9 M** | total tensor params incl zeros, at the realizable Qwen2-0.5B shape (d=896, 51 layers) — what a checkpoint actually stores |

**Dense count, code-traced** (`CleverShapeModel`, d=896, real Q/K/V/O + SwiGLU
FFN + ALiBi per layer, `examples/clever_realtime_model.py`):

```
per layer (d=896, inter=896, 14 heads, GQA 2 KV) = 4,243,470 params
51-layer stack                                    = 216,416,970  (~217 M)
+ embed (276×896) + LM-head (896×276)             = +494,592
FULL realizable UNROLLED dense                    = 216,911,562  (~217 M)
looped (6-cell) dense                             = 25,460,820   (~26 M)
```

So the **same realizable model** carries **5,043 nonzero signal** inside **~217 M
dense tensor** — **0.0023 % nonzero**. This is the "sparse-but-wide" gap: the
*information* is 5,043 numbers, but the *realized matmul* multiplies zeros across
full 896-wide blocks. "Fully unrolled language model params" means **5,043** if you
mean the information the weights encode; it means **~217 M** if you mean the
checkpoint tensor the stock-shape transformer stores and the GEMM actually runs.

---

## 5. BLOG AUDIT — `docs/BLOG_NOTE_CLEVER_MINPARAM_VM.md`, blunt, per item

### (a) Does it claim "whole value in one fp scalar" WITHOUT the fp32/limb caveat?
**COVERED-honestly.** The claim appears (§1 *"A d-digit number lives in one fp
scalar (exact because integers are exact in floats up to 2^53 for fp64)"* line 15;
§2 *"Hold the whole value in one float"* line 61) — but the **fp32/limb caveat is
explicitly there**, in a dedicated section §7 *"The fp32 tricks — a min-param VM
byte-exact with no fp64/fp128"* (line 310): *"The whole-value construction wants
fp64 … and, for MUL's 64-bit product, fp128 — which GPUs don't have"* (line 311),
and *"1. Limb MUL — the fp128-killer. Never form the 64-bit product as one scalar.
Do MUL in 8-bit limbs (radix-256) … 260,864 — a 64× margin under fp32's 2²⁴
exact-integer ceiling"* (line 320). The "one scalar exact to 2⁵³" statement is
*true for fp64* and the note flags the fp32 break + the limb fix. **Not
over-stated.**

### (b) Does it describe the memory as real softmax1-KV attention while the impl uses direct-CAM host gather?
**OVER-STATED in §2, corrected in §6/§7.** §2 (line 83-84) says *"Memory (LI / LC /
SI / SC). One shared content-addressed attention (CAM), ~10 weights, reused by all
four (per-op marginal ≈ 0)."* — calling the 10-weight cell an **"attention (CAM)"**.
That is the shortcut: the 10-param form is a **host-side resolver + O(1) gather, not
`softmax1(Q·Kᵀ)·V`**. The note *does* come clean later: §6 (line 298-306) labels it
*"the VM's real **O(1) direct-CAM memory-read** attention"* and §7 trick 4 (line
337) is explicit — *"**O(1) direct-CAM memory.** Reads resolve by **address decode**
(one gather), not an O(S) softmax scan over the heap"*. **Verdict:** the *word*
"attention" for the 10-weight cell in §2 is misleading (it's a resolver); §6/§7 are
honest that it's O(1) direct-CAM, not softmax-KV. The note **never states the honest
real-softmax-KV param count** — the real baked head is **166**, not 10. **The `10`
quoted as the memory cost is the shortcut's cost, not the attention's.**

### (c) Does it state the fp32 24-bit precision limit / the limb requirement?
**COVERED-honestly.** §7 states both precisely: the 2²⁴ ceiling (line 320 *"under
fp32's 2²⁴ exact-integer ceiling"*, line 323 *"the whole-value form at radix 4096
has a column peak ~50M > 2²⁴"*), the fp128-forcing whole-value shape, and the
8-bit-limb requirement as the fix (measured L∞=0). §3 also notes the *"precision
ceiling"* binds radix (line 108). **Fully covered.**

### (d) Does it state the param count, and is it the idealized or the honest one?
**OMITTED (honest total) / states a DIFFERENT census.** The note's headline counts
are **3,183 looped / 26,119 unrolled** (§6 line 220-222, §9 line 403) — the **Family
A LUT-bitwise op-cell census** (nibble-LUT bitwise 2,974/layer, **no
sequencer/dispatch, no I/O head, direct-CAM memory 10**). It does **NOT** state the
**4,887 idealized** *assembled* 39-op figure, and does **NOT** state the **5,043
honest** byte-exact real-attention figure. So on params the note is:
- **Neither the idealized-assembled (4,887) nor the honest (5,043)** — it quotes the
  op-cell LUT census (26,119) which *omits* dispatch (256), the I/O head (2,328),
  uses the LUT bitwise (23,792 vs bit-serial 8), and uses the **direct-CAM 10** for
  memory. It is internally consistent for *what it counts* (§6 line 231-235 breaks
  it down and `clever_nonzero_table.py --check` asserts it) but it is **not** the
  honest real-attention byte-exact total. **The honest 5,043 is OMITTED.**

**Audit summary:**

| item | verdict | blog line |
|------|---------|-----------|
| (a) whole-value-in-one-fp caveat | **COVERED-honestly** | §1 L15, §7 L310-323 (limb fix explicit) |
| (b) memory = real softmax-KV vs direct-CAM | **OVER-STATED** (§2 calls the 10 an "attention CAM"); **corrected** §6/§7 as O(1) direct-CAM; honest 166-param head **OMITTED** | §2 L83-84 vs §6 L298-306 / §7 L337 |
| (c) fp32 24-bit limit / limb requirement | **COVERED-honestly** | §7 L310-323 |
| (d) param count stated — idealized or honest? | honest 5,043 **OMITTED**; states the Family-A LUT op-cell census (3,183 / 26,119), not the assembled 4,887 nor the honest 5,043 | §6 L220-222, §9 L403 |

---

## 6. VERDICT

- **Honest byte-exact real-attention nonzero total (fully-unrolled, 39-op): 5,043**
  (looped **2,957**), vs the idealized **4,887** (looped 2,801) — **+156**, *all*
  from the real softmax1-KV memory head (166) replacing the 10-param direct-CAM
  shortcut. **The byte-exact limb requirement costs +0 params** (same 51-cell decode,
  paid in depth). So **4,887 is nearly the honest total — off by only the memory
  head** — but it was *idealized* on two counts: a resolver called "memory" and a
  non-byte-exact whole-value arith.
- **Dense realizable count: ~216.9 M** tensor params (d=896 Qwen2-0.5B shape, 51
  layers) — the same model that carries the 5,043 nonzero signal, **0.0023 %
  nonzero**. "Fully-unrolled language model params" = **5,043** (information/nonzero)
  or **~217 M** (dense checkpoint tensor); both are real, they differ by ~43,000×.
- **Does the blog honestly cover the mechanism?** **Mostly — with one over-statement
  and one omission.** The fp32/limb precision story (§7) is honest and complete. The
  memory story **over-states in §2** (calls the 10-weight direct-CAM resolver an
  "attention CAM") and **omits the honest 166-param real softmax-KV head**, though
  §6/§7 do disclose it is an O(1) direct-CAM shortcut, not a softmax scan. The
  **param headline (3,183 / 26,119) is a *different, cleaner census*** (Family-A LUT
  op-cell) — it is *neither* the assembled 4,887 idealization *nor* the honest 5,043,
  and the honest byte-exact real-attention total is **not stated anywhere in the
  note**. **Net: the blog describes the clean idealization, not the honest
  byte-exact real-attention machine — accurate about precision, soft about the
  memory shortcut, and silent on the honest assembled total.**

---

## 7. Provenance — every number → the code it came from

| number | source (all instantiated / run, not read off a docstring) |
|--------|-----------|
| real softmax-KV memory head **166** (W_q 66 / W_k 67 / W_v 16 / W_o 16 / alibi 1) | `bake_memory_head` (`c4_min/blogspec_memory.py`) baked into a 1-head `Attn`, `count_nonzero` |
| real head byte-exact (L-inf=0, S∈{8K,65K,262K}) | `python -m examples.clever_real_softmax_kv_memory --verify` |
| `RealSoftmaxKVMemoryHead` module Parameters = 8,192 (dense proxy) | `clever_real_softmax_kv_memory.py` `named_parameters` (W_v/W_o randn d=64) |
| direct-CAM shortcut **10** | `MemoryCAMCell.count_nonzero` = 8+1+1 |
| ArithCell **51** (fp32==fp64) | `ArithCell(dtype).count_nonzero()` |
| limb MUL/DIV byte-exact, NO fp128; whole-value fp32 MUL 100k/100k non-exact >2²⁴ | `python -m examples.clever_fp32_fullops --verify`; direct fp32-product mismatch check |
| assembled census 473/2559 (31-op), 2801/4887 (39-op), memory 10, I/O 2328, dispatch 256, framing 194 | `assembled_machine_census` + `syscall_param_account(8)` |
| honest 39-op **5,043 unrolled / 2,957 looped** | assembled census with memory 10→166, arith unchanged |
| dense **216.4 M** (51-layer stack), 216.9 M (+embed/head), 25.5 M looped, 4.24 M/layer | `CleverShapeModel(n_layers=…, d=896).parameters()` numel |
| golden `174ece66` UNCHANGED | `python -m c4_min._fingerprint_build` |
