# OUTPUT-band self-reinforcement decouple — audit + design + prototype

_Status: PHASE-1 AUDIT/DESIGN/PROTOTYPE. mega-root #2. Base `aa1d1b75`._
_Author: mega-root #2 Phase-1 agent, 2026-06-18._

Companion to memory `project_output_band_self_reinforcement_megaroot`,
`docs/B9_OUTPUT_HI_SPLIT_SPEC.md`, `docs/PHASE_9_SSA_PROTOTYPE.md`,
`docs/CROSS_STEP_SSA_ANTIPATTERN_AUDIT.md`.

---

## 0. TL;DR (the make-or-break result)

The "split `OUTPUT` into a READ band (prev step) + WRITE band (this step) so the
gate reads the stable prior value" decoupling **cannot break the feedback as an
SSA / `*_PREV_STEP` alias rename**, because the existing split aliases back onto
the **same physical column** (`OUTPUT_HI` and `OUTPUT_HI_THIS_STEP` both resolve
to slot **85**; `OUTPUT_HI_PREV_STEP` resolves to **`None`** — it is not even an
allocated dim, only an SSA scheduler label). A rule's `W_gate` therefore reads
the **identical live, bootstrapping column** regardless of the SSA suffix. This
is the mechanistic root cause of the GPU-refuted coordinated sweep (memory:
"426 pass = EXACTLY baseline, ZERO programs flipped"): the rename is a
**scheduler back-edge fiction**, not a value decoupling.

**Proven this session (prototype, l16 `l16_lev_stack0_byte0_preserve` family):**

| gate | golden 35-tok | meaning |
|---|---|---|
| `OUTPUT_HI_THIS_STEP` (HEAD) | `4958b35b18108745` | baseline |
| `C4_OUTBAND_DECOUPLE_PROTO=alias` (`→ OUTPUT_HI`, distinct name, same slot 85) | `4958b35b18108745` **(identical)** | the rename is byte-identical **because** it writes the same weights to the same column = **structurally inert** |

A TRUE decouple requires a **physically distinct** prev-step band populated by a
**cross-step copy** (an attention KV-relay, like `H1_PREV_STEP`), not an alias.
That is a multi-block build, and per the prior coordinated-sweep GPU result the
OUTPUT value is a converged *symptom* of the token-COUNT desync, not the
verdict-flipping root — so even a physically-real decouple is unlikely to flip
the framing clusters. **Recommendation: do NOT greenlight the OUTPUT-band
decouple sweep as a count lever.** Re-target the campaign at the token-COUNT
source (why a framing step emits ≠35 tokens). See §5.

---

## 1. AUDIT — the self-reinforcing emitter catalog

Tool: `tools/lint_output_selfreinforce.py` (read-only, CPU, IR-level; the
mega-root #2 analog of `lint_positional_invariants.py`). It walks every FFN rule
in `all_core_ops()` + ALU post-ops and flags a rule as **self-reinforcing** when
it READS an `OUTPUT_*` dim (in the SwiGLU **GATE** branch `gate`/`gate_terms`,
or the **UP** branch `conditions`) AND WRITES an `OUTPUT_*` dim. The GATE-read
variant is the dangerous multiplicative one (`silu(up) * gate` with `gate` = the
live OUTPUT value → bootstraps).

`--use-built-layout` confirms the catalog is stable and complete (the 7
over-width-band ops the static registry skips add NO further OUTPUT
self-reinforcement).

### Ranked catalog (GATE-read = bootstrap-dangerous)

| op (cluster proxy) | GATE | UP | writes-OUTPUT | memory-named family | gates clusters |
|---|---:|---:|---:|---|---|
| `tail_bit32_result_correction` | **417** | 1742 | 2059 | `tail_stack0_store_loaded_byte` (L25 tail bank, width-sensitive) | arith byte / if-bool / var (step-3/5/10) |
| `layer6_routing_ffn` | **288** | 0 | 1184 | (not in memory list) L6 routing | per-step OUTPUT decode (all) |
| `layer16_lev_routing` | **113** | 274 | 826 | `l16_stack0_e8_output_authoritative` + the e0/preserve LEV family | func/var/if-bool LEV frame (step-6/11) |
| `post_l9_bz_bnz_pc_override` | **96** | 0 | 192 | (not in memory list) BZ/BNZ branch | if/bool branch PC |
| `function_call_weights` | **32** | 0 | 290 | (not in memory list) func entry | func step-6/11 |
| `layer14_clear_output_corruption` | 0 | 2 | 3 | (tiny) | — |

Total: **6** self-reinforcing ops, **946** GATE-read rules, **2018** UP-read
rules. This is BROADER than the memory's hand-listed 4 families — it adds
`layer6_routing_ffn` (288 GATE), `post_l9_bz_bnz_pc_override` (96), and
`function_call_weights` (32). NONE of the GATE reads is `prev_step_only` today
(every one reads the live base band).

The two top memory-named families map exactly:
`l16_stack0_e8_output_authoritative` → `layer16_lev_routing`;
`tail_stack0_store_loaded_byte` → `tail_bit32_result_correction`.

---

## 2. The mechanism (why it self-reinforces, and why the split doesn't help)

Per-unit SwiGLU FFN: `out += W_down[:,u] * silu(W_up[u,:]·x) * (W_gate[u,:]·x)`.
A `l16_lev_stack0_byte0_preserve_lo_k` rule has `conditions` (the structural AND
→ `up`), `gate = OUTPUT_LO+k`, `writes = OUTPUT_LO+k`. So the unit's contribution
to column `OUTPUT_LO+k` is `≈ silu(structural)·x[OUTPUT_LO+k]·strength` — a
**term proportional to the value it writes back**: a positive-feedback loop. Any
residue on `OUTPUT_LO+k` (e.g. a leaked ALU operand projected onto an empty-stack
STACK0 marker row) is read by the gate, amplified, re-written, re-read by the
next family, … → saturates ~1e27–5e29 → wins the byte dump → STACK0 byte-0
decodes a leaked operand → fixed-35-token slicer re-anchors wrong (the framing
drift).

**Why the `OUTPUT_HI`/`_THIS_STEP`/`_PREV_STEP` split does NOT break this:**
the split (`docs/B9_OUTPUT_HI_SPLIT_SPEC.md`, Phase 9 SSA `docs/PHASE_9_SSA_
PROTOTYPE.md`) was built for **scheduler legality** — it drops the SCC
back-edges the dynamic scheduler's cycle detector chokes on (81 of ~110
back-edges were `OUTPUT_HI`). By design (`ssa_dim.py` docstring; `LayerCompiler.
add_op` line ~1290: _"the SSA form [is declared] as an ALIAS of the base. Same
numeric slot, same size"_) **every SSA / PREV_STEP spelling resolves to the base
dim's column.** Confirmed empirically: `dim_positions` gives `OUTPUT_HI = 85`,
`OUTPUT_HI_THIS_STEP = 85`, `OUTPUT_HI_PREV_STEP = None`. A `W_gate` write at
`resolve("OUTPUT_HI_THIS_STEP.…-1")` lands at column 85 — the SAME live cell.
Renaming the gate is byte-identical AND inert. **Proven** (§0 table).

---

## 3. The TRUE decouple (what would actually break the feedback)

To make the gate read a STABLE prior value, the prev-step band must be a
**physically distinct column**, populated by a **cross-step copy**:

1. `register_residual_band("OUTPUT_LO_PREVSTEP", 16, ...)` +
   `OUTPUT_HI_PREVSTEP` (flag-gated, `never_share=True`) — fresh columns past
   d_model (auto-widen).
2. A **cross-step copy op** (attention KV-relay, the `H1_PREV_STEP` /
   `make_layer11_ax_byte1_dump_carry_op` template at `l11_ops.py:1316`): a head
   whose V copies the PRIOR step's `OUTPUT_LO/HI` one-hot into the fresh
   `OUTPUT_*_PREVSTEP` band, landed at the step's first block so it is ready
   before any L6/L16 gate reads it.
3. Re-point the self-reinforcing gates: `gate = OUTPUT_LO_PREVSTEP+k` (reads the
   stable prior value, breaks the read→write loop). The WRITE stays on the live
   `OUTPUT_LO+k`.

The prototype ships this as `C4_OUTBAND_DECOUPLE_PROTO=decouple` for the l16
preserve family **minus the copy op** (band registered, gate re-pointed). Without
the copy op the prevstep band is 0 → the preserve rule never fires → it is a
"disable the rule", NOT a real decouple. Building the copy relay is the
multi-block work this design hands off; but see §4 for why it is NOT greenlit.

---

## 4. Why this is NOT a count lever (the prior GPU verdict + this audit)

The full coordinated decouple sweep `C4_OUTPUT_SELFREINFORCE_DECOUPLE` (all 4
memory-named families) was **GPU-verified on the full 1096 full_trace: 426 pass =
EXACTLY baseline, ZERO flipped, ZERO broke** (memory, commit `57a93bce`,
flag-ON hash `01da0c78` ≠ golden so the flag IS active — not a cache artifact).
That sweep used the SSA/alias rename (§2) which this audit now proves is
physically inert — so a no-op is expected. But the memory's deeper finding holds
regardless of mechanism: the framing programs emit the **wrong TOKEN COUNT** (the
fixed-35-slicer 34/36/37-token desync); making one OUTPUT leak row inert just
lets a DIFFERENT token win → still ≠35 tokens. The OUTPUT-band crush is a
**converged symptom**, not the verdict-flipping root.

A physically-real decouple (§3) MIGHT change which token wins on the leak row,
but there is no evidence it changes the COUNT, and it is a multi-block build at
real regression risk (a new always-live attn relay touching the OUTPUT bus +
d_model widen → the kind of cross-op surface that historically regresses
`test_bnz_branch` / memory smoke). **Do not greenlight the decouple sweep as the
campaign count lever.**

---

## 5. Sweep plan / recommendation

1. **Keep the audit tool** (`tools/lint_output_selfreinforce.py`) as the
   standing mega-root #2 catalog (analog of `lint_positional_invariants.py`).
2. **Keep the prototype flag** `C4_OUTBAND_DECOUPLE_PROTO` DEFAULT-OFF
   (byte-identical golden) as the banked building block + the inertness proof.
3. **Do NOT run the OUTPUT-band decouple sweep** as a count lever — it is a
   verdict no-op (GPU-proven for the alias form; structurally inert here).
4. **Re-target the campaign** at the token-COUNT source: instrument WHY a
   framing step emits ≠35 tokens (a value byte failing → a spurious marker
   token). The cumulative fixed-35-slice desync is the verdict-flipping root;
   only GPU `full_trace` / `tools/cpu_full_trace.py` can see it (the CPU
   oracle re-anchors per step and cannot). Candidate next probe: the
   step-end token-emission completeness (`tools/probe_step_end_completeness.py`)
   on a confirmed framing-drift program, comparing the emitted token count
   per step OFF vs a single-emitter OUTPUT fix.
