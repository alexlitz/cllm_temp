# Blog draft — revision notes (what holds / revise / expand)

**As of 2026-08-04.** Companion to the "C compiler that compiles to a transformer" (c4llm) draft.
The draft was written ~March 2026, early in the project. Since then the project grew a full
C90 toolchain, a byte-exact **Doom** port that runs on the transformer, extensive honest perf
work, and — critically — the **compile-and-run-in-one-transformer-run** capability the draft
only sketched is now *built, live, and tested*. These notes annotate the draft section-by-section:

- ✅ **HOLDS** — still accurate, keep as-is (maybe tighten prose).
- ⚠️ **REVISE** — a concrete fact/number/claim has changed or is now wrong.
- ➕ **EXPAND** — now demonstrated / has real data / deserves its own treatment.

Evidence is cited as `file`, task `#NNN`, or golden/SHA where it helps you verify.

---

## Part 0 — The two biggest structural updates (read first)

### 0.1 There are now TWO models; the draft only describes one
The draft describes the **from-scratch minimal hand-built VM** (`PureFFN` / `PureAttention` /
`SoftMoEFFN`, ~800 unique nonzero weights, ~96% sparse). That still exists and is still the
clearest teaching artifact. **But the mainline work is now the same VM baked into *stock
Qwen2.5-0.5B*** — the "c4_min" model (`c4_min/`), whose authoritative weight fingerprint is
**golden `069cc32f`** (the `C4_PF_CFM=1` code-from-memory sibling is `7d19cdc3`; both lean ~1 GB).
This is a genuine strengthening of the "vanillaness" thesis: it's not just a hand-built transformer
*shaped* like a standard one, it's an *actual production LLM's architecture* with the VM baked in.

**Action:** decide the framing. Recommend keeping the minimal VM as the pedagogical spine and
adding a "…and then we baked it into a real 0.5B model" arc. Every weight-count / opcode / width
number in the draft should say *which* model it describes.

### 0.2 "A subset of C" undersells it — it's now ~full C90, and it runs Doom
The draft's "C4 supports a subset of C (int, char, pointers, if/while/return…)" is the *original*
c4. The toolchain now has: a real recursive-descent **C90 front-end** (`#811`/`#818`), a full
**preprocessor** (`##`, `#`, nested macros, `#if` eval — `#812`), a c4-subset **soft-float IEEE-754
library** (`#799`) plus **native float ops**, and a **self-hosting transpiler** (`transpile.c`
compiles itself under the c4 VM — `#835`). The proof-of-completeness is that **id Software's Doom
(62/62 modules, ~555K instructions) transpiles and runs byte-exact** — title frame SHA
`2e883404` == gcc reference, E1M1 fully byte-exact vs gcc. This is the flagship demonstration and
the draft predates all of it.

**Action:** the whole "C4" section needs a "how far C-support went" expansion + a Doom capstone.

---

## Part 1 — Section-by-section

### Introduction / Why / Elephant in the Room
✅ **HOLDS** entirely — philosophy, motivation, the "got scooped by Can-LLMs-be-Computers"
honesty, the selling points. This is the voice of the piece; don't touch it.
➕ One selling point to add: *"a full 1993 video game (Doom) runs on it byte-exact, and it can
compile C to itself in a single run."* Both are stronger than anything in the current bullet list.

### C4 (the compiler description)
⚠️ **REVISE** — "~500 lines… subset of C" is the *original* c4. See §0.2. Keep the 4-function
(`next`/`expr`/`stmt`/`main`) explanation as the *starting point*, then note the front-end grew to
C90. The line **"C4 only supports int and chars"** (repeated later in the Sparse-Tensors and
Self-Hosting sections) is now false in two ways: (a) the front-end handles structs/unions/fn-ptrs/
float; (b) there are native float opcodes.

### C4 Registers table
✅ **HOLDS** conceptually (PC/AX/SP/BP + immediate).
⚠️ **REVISE** width: registers are emitted as **4 bytes / 8 nibbles (32-bit)** — see §Tokenization
below. The table's "32-bit" is right; just make sure it's consistent with the "16 nibbles" claim
elsewhere (which is wrong — should be 8).

### The two opcode tables (the big one)
⚠️ **REVISE — this is the most out-of-date artifact in the draft.** The shipped ISA (`c4_min/isa.py`)
is:
- **40 base opcodes**, `NUM_OPS = 40`, and **`HALT = 38` is an alias for EXIT** (not a separate 38+39+40).
- The base `NAMES` set: `LEA IMM JMP JSR BZ BNZ ENT ADJ LEV LI LC SI SC PSH OR XOR AND EQ NE LT GT
  LE GE SHL SHR ADD SUB MUL DIV MOD OPEN READ CLOS PRTF MALC FREE MSET MCMP NOP HALT`.
- **`F_ADD F_SUB F_MUL F_DIV` occupy 40–43**, gated behind `C4_FLOAT_OPS`; when on, the opcode
  one-hot band widens to `NUM_OPS_FLOAT = 44` (off → stays 40, golden byte-identical). This
  "widen the OP_IS band only when the flag is on" is the general mechanism for adding ops without
  moving the golden.
- **There is no separate `GETCHAR`(64)/`PUTCHAR`(65)/`BLT`(41)/`BGE`(42) in the base ISA.** The
  draft's 64/65/41/42 numbering is from an earlier design. I/O is `PRTF`/`READ` + the neural
  think-tag path (see §Neural I/O); signed branches fold into `BZ/BNZ` + the CMP32 comparator.
- **Doom accel superinstructions are separate gated modules, not base opcodes:** `BLIT`
  (`doom_blit.py`), `NAMEEQ` (`doom_nameeq.py`, WAD name lookup), `DRAWSPAN`=47 (`doom_drawspan.py`,
  V_DrawPatch column-copy — *built this session*), plus fixed-point (`doom_fixedpoint.py`). Each is
  default-OFF; flag-ON widens the OP_IS band and produces a deliberate new fingerprint (e.g.
  DRAWSPAN-ON = `2f69350f`).

**Action:** replace both tables with the current 40-op set + a small "extension opcodes" table
(float, Doom accel) clearly marked as gated. Regenerate L/W (layers/weights) columns against the
current build rather than hand-maintaining them.

### Summary / Redundancy / weight-count tables (5,487 → 1,397 → ~800)
✅ **HOLDS** as a description of the **minimal hand-built VM** — the sharing story (8 nibbles ∥,
8 cascade layers, 6 iterations, KV projections, 96.4% sparse, ~800 unique nonzero) is the good
insight and is real.
⚠️ **REVISE** the framing: say explicitly these counts are for the minimal VM, *not* the
Qwen-0.5B-baked model. Mark "TODO format tables" done by regenerating from the actual state_dict
sparsity rather than the hand-tallied numbers.

### Vanillaness (incl. the PureFFN / PureAttention / SoftMoEFFN code)
✅ **HOLDS** — softmax1 + ALiBi as the only real deviations (both defensible as taste / simulable),
standard SwiGLU MoE FFNs, the forward functions. This is the heart of the honesty pitch and it
survived contact with a 0.5B model and Doom. Keep it.
➕ **EXPAND**: strengthen the claim — "we baked this into unmodified `Qwen2ForCausalLM` and it stays
byte-exact through the *standard* autoregressive loop, real embed + lm_head, no driver overlay"
(the vanilla-discrete-token-registers result). One honest caveat to keep: in that fully-vanilla
variant the per-step register *frame skeleton* was still a template-add (one assist) — mention it
rather than imply zero assists.
➕ "TODO generate function" is answerable: it's the **standard** HF generation loop; point at
`blogspec_run.py`. No custom loop — that's the whole point.

### Tokenization
⚠️ **REVISE** the width inconsistency. The draft says both "32 bit integers" **and** "16 4-bit
nibbles" — those disagree (16 nibbles = 64 bits). The shipped model is **32-bit: 4 bytes /
8 nibbles per value** (`BYTES_PER_REG = 4`, `ADDR_BITS = 32`). Historically c4's `int` is
`long long` (8 bytes, word machine), but the transformer **standardized on 4-byte int**
(`#855`, weights-neutral). Fix to "8 nibbles / 32-bit," and if you want to mention 64-bit,
frame it as "the same construction extends to 16 nibbles / 64-bit at higher cost."
✅ The rest (bytes→nibbles for easy bitwise, writing registers each step, many passes are just
register I/O) HOLDS.

### Memory
✅ **HOLDS** — binary-address keys (+scale/−scale), query=key retrieval, softmax1 ⇒ ZFOD /
zero-default, ALiBi recency ⇒ latest-write-wins without rewriting memory or special masking. The
"kludge: info lives higher than where it's needed so it must round-trip through tokens → looped
transformers could shortcut this" reflection is a genuinely good point; keep it.
➕ **EXPAND** with **direct-CAM O(1) reads** (`#834`, `#871`, `C4_DIRECT_CAM_VEC`): the recency/
ALiBi scheme works but its "recency horizon" becomes a *sizing* wall for very long runs (Doom's
~262k persistent-heap KV/step). Direct-CAM resolves LI/LC/pop/LEV/code reads by the exact address
match in O(1), which both fixes the horizon and is faster. Worth a paragraph — it's the mature form
of "memory = attention."

### How Bytecode is Passed / System-Prompt Format / Registers (30 tokens)
✅ **HOLDS** — the `[op:1][imm:4 LE]` bytecode, SEP=256, DATA, ARGV format; the 30-token/step
register frame (PC/AX/SP/BP each 4 bytes + MEM 8 + markers). Nice and concrete, keep it.
⚠️ Minor: reconcile "30 tokens" with the earlier "31 tok/step" note elsewhere — the frame is 30;
some variants add 1 (e.g. a halt/step token). State the canonical number once.

### The Operations — FFN / Attention / Building Blocks
✅ **HOLDS, and this is the best part of the piece.** All of it survives: SiLU≈ReLU-when-scaled,
step functions `silu(S(x+ε))−silu(S(x−ε))`, point indicators (second difference), range checks,
residual cancellation (`b≈1.27846 ⇒ silu(b)≈1`), magic floor (`(x+MAGIC)−MAGIC`, 2^23), bit-range
extraction, efficient exp via BOS-sink log attention. These are the reusable primitives and they're
all still exactly how it works. Just finish the "TODO cleanup and add graph" items.

### Comparisons / Arithmetic / Shifts / Addition / Multiplication / Division / Mod / Bitwise
✅ **HOLDS** — zero-detector (11 params, +1/−2/+1 second difference), add/sub in 4 weights, gated
multiply in 6 weights, schoolbook multiply (byte-level: 10 partials, 3 carry rounds, skip
i+j≥threshold overflow), base-16 long division by threshold counting, mod = div-subroutine +
mul + sub, bitwise = per-nibble lookup tables. All correct and still shipped.
⚠️ **REVISE the Division section's honesty.** The draft presents three ideas (log-attention,
long division, log-sink-via-attention). The *shipped* production path is **long division /
recurrent divmod**; the elegant **log-sink attention division is designed but does not fit** —
its depth (the recurrent divmod variant is ~291 layers) blows the stock 24-layer budget, so it's
**not the wired path**. Keep the beautiful log-sink derivation, but label it "the construction I'd
use with more depth / native float; the shipped model uses long division." (This matches the
project note: "log-sink div not yet wired.")
⚠️ Also note the Qwen-baked path had a real multi-byte MUL/DIV/MOD arch-block history (L15/L20
result corruptors, d_model pressure) that the *clean minimal VM* doesn't have — the minimal-VM
constructions in the draft are the honest, working ones; don't let a reader assume the 0.5B path
was as clean.

### Chars / Memset-Memcmp-Memcpy / Reading Arguments / Exiting
✅ **HOLDS** — chars as top-8-bits (except store/load/cast/shr), mem* as baked-in bytecode loops
(not syscalls), argv via the `__argv_setup` bake read like user input, HALT ⇒ EOS/halt-token.
⚠️ "TODO is this actually how argv is implemented?" — verify `__argv_setup` against the current
build before publishing; the *mechanism* (read like stdin, store to memory) is right, confirm the
exact routine.

### Neural I/O (Printing / Reading / Position Offset / RoPE binary matching)
✅ **HOLDS** — think-tag protocol (compute inside `<think>`, exit to emit a visible byte, re-enter),
input between USER_INPUT markers, BOS-sink multi-slope-ALiBi position signature, nibble-cascade
offset extraction (8 layers base-16 vs 32 layers bitwise), the RoPE-binary-distance alternative.
The position-offset and RoPE sections are strong; just finish "TODO elaborate/edit."
➕ **EXPAND** with a forward pointer: this same "framebuffer lives in memory, host reads it"
insight is what makes Doom's present cheap — see §New/Doom below (host-offloaded present kills the
~128K putchar/frame emit floor). It ties the I/O section to the capstone.

### MoE Routing / Internal Representation / Efficient Floor / Efficient Exp
✅ **HOLDS** — opcode-routed experts (avoid interference + reuse residual dims), 8-nibble values
(⚠ not 16), efficient floor via per-nibble range checks / rescale-subtract, exp via log-sink.
All still exact.

### Memory Allocation and Freeing
✅ **HOLDS** and it's a highlight — bump allocator (+4), free = overwrite-with-zero ⇒ ZFOD ⇒
"eviction of the zero entry is a no-op under softmax1." The "hooking the transformer output to the
eviction policy would be hacky; instead free-is-zero and eviction-of-zero-is-free" argument is
genuinely elegant. Keep verbatim.

### Baking Prompts/Programs into Weights
✅ **HOLDS** and is now **load-bearing / demonstrated**, not hypothetical. The key→value retrieval
as (per-nibble EQ) → (AND) → (MoE-router one-hot → value expert) = **read-only code segment** is
exactly the shipped **CFM (Code-From-Memory)** code-CAM (`_bake_code_cam`). The full-attention-
simulation-via-SwiGLU discussion (needs per-layer real division ⇒ expensive) HOLDS as the honest
limitation.
➕ **EXPAND**: this is where to connect to the compile-in-one-run proof (below) — baking the
*compiler's* bytecode as the read-only code segment is what "compiler in the system prompt" means.

### Model that Directly Runs C Code
➕➕ **BIGGEST UPGRADE — was aspirational, is now DONE, TESTED, and byte-exact.** The draft says
"we simply bake in a c4 bytecode compiler and hand off." That is now real:
`c4_min/nibble_compiler.py` + `demo_model_runs_c.py` — a compiler **baked into the transformer's
weights** takes C source `2+3*4`, **compiles it, EMITs the bytecode into a writable code-memory
band, jumps in, and the universal fetch runs its own freshly-produced bytecode → 14**, correct
precedence, produced bytecode **byte-identical to real c4**. 16/16 tests pass
(`test_nibble_compiler.py`, `test_handoff.py`). The load-bearing unknown — *can a transformer
execute code it generated at runtime?* — is answered **yes**, with the exact mechanism (hybrid
weight/memory instruction fetch + an EMIT store into a writable `CODE_WORD` band).
⚠️ Honest gap to state: that live JIT is a **bespoke small machine** (fixed-width code band capped
by d_model); wiring the same EMIT into the **golden CFM address-CAM** (so runtime-appended code
frames are fetchable with *no* d_model growth = program-length-independent) is in progress this
session, and scaling the baked compiler from the toy expr grammar to full c4 is the remaining lift.
So: *the capability is proven; Doom-scale source→compile→run in one run is feasible and being wired,
not yet demonstrated end-to-end.* Say exactly that — it's the honest, still-impressive claim.

### Adapting to Different Precisions
⚠️/➕ "TODO write" is answerable: the knob is the **ALU chunk size** (`BYTES_PER_REG`, nibble vs
byte vs larger) traded against runtime float precision (fp32 vs fp64) — bigger chunks need more
mantissa; bitwise excepted. Note the real result: fp32 forces small chunks (nibble/byte) for the
ALU; fp64 would let the network be dramatically smaller/simpler but "less vanilla," which is the
deliberate choice. The self-hosting path especially wants native float (see below).

### Tool Use Mode
✅ **HOLDS** — two I/O modes (TOOL_CALL token vs neural think-tag I/O); mem*/malloc/free are
neural bytecode, only true syscalls cross the boundary. Accurate.

### Table of OPcode Neural Implementations
➕ "TODO write" — build this from `c4_min/isa.py` + the per-op modules. It's the natural home for
the corrected opcode/L/W tables.

### Self-Hosting (3 interlocking relationships)
✅ **HOLDS** — the C-runtime-hosts-itself / ONNX-runtime-hosts-itself / transformer-runs-itself
framing via the C ONNX runtime is still exactly right and is a highlight.
➕ **EXPAND** with what actually got built: an **all-C ONNX runtime** (`onnx_runtime_nibble_allc.c`,
static ELF, byte-exact echo/yes/cat/quine), the self-hosting **transpile.c** (compiles itself under
c4, `#835`), and the honest performance ("TODO performance analysis"): emulating fp is expensive on
c4 (int/char only historically), which is exactly why sparse tensors + native float matter. The
self-emulation cost is now measured (~18 min composed, byte-exact — see §Perf).
⚠️ The "c4 only supports int and chars" caveat here is now softened by native float ops.

### Example programs / Bundling / Quine
✅ **HOLDS** — Mandelbrot + Eliza framing, the bundler (yes/cat/echo, ~150–200 KB, C ONNX runtime +
weights + bytecode), the fixed-point bundle variant, the multi-interpretation Quine (C quine ==
ONNX quine == transformer quine). All real.
➕ The **quine is done and byte-exact** (self-outputs), per the c4-interpreter capstone set; mark
"TODO elaborate" resolvable. "TODO cli examples" — the bundles exist; just paste real invocations.

### Speculation
✅ **HOLDS** and is now quantified. The draft's "~1000× faster, perfect speculation, huge blocks"
is right; the honest numbers: the **Rust c4 draft runs ~665M steps/s** raw (byte-identical), and
because the draft is byte-exact **every speculated token is accepted** — so verification is a
*per-query-row independent map with zero cross-row attention* (giant-K single dispatch). Add these;
they make the section concrete.

### KV Cache Pruning
✅ **HOLDS** — the two mechanisms (cosine>0.99 + ALiBi recency ⇒ latest-write-wins; zero-write ⇒
evict-to-nothing), eviction every ~120 tokens, 99.999%+ pruning, log-growth. Accurate and is
what makes million-step runs (Doom!) feasible.

### Sparse Tensors / Computational Efficiency
✅ Sparse-tensor rationale HOLDS (>99% sparse; COO beats dense; smaller ONNX; matters doubly for
self-hosting since c4 has no dense-matmul accel).
➕➕ **"Computational Efficiency — TODO comparison…" is now a whole answerable section.** Real data
to fill it:
- **FLOP floor ≈ 0.0071 µs/step** (≈142M steps/s) at ~196K FLOP/step.
- **Measured single-dispatch dispatch ≈ 3.09 µs/step**; the honest whole-frame wall turned out to
  be the **one-time schedule build** (routing + gather resolution), not the dispatch.
- **Doom frame ≈ 6,889,905 steps** raw → **358,058** with the render superinstruction. Measured
  residual = **357,417 game-sim steps + 640 render (DRAWSPAN)** — i.e. after the render macros,
  **~99.8% of the per-frame verifier load is the game simulation itself** (P_*/thinkers), which is
  irreducible c4 execution; render is essentially free.
- **Host-offloaded present** takes the **128,000 framebuffer tokens/frame OFF the critical path**
  (host reads the framebuffer from VM memory / the store-log; the transformer emits nothing) — a
  *token-path* win (no per-frame emit floor), byte-exact (title `2e883404`, 3-frame `a333a608`
  unchanged). It does **not** reduce the ~358K step count (loopfuse already collapsed the emit to
  one macro-step); don't conflate the two.
- **Self-emulation** (transformer emulating itself) ≈ **18 min** via composed levers, byte-exact.
- The natural "vs old videogames" comparison writes itself: this literally runs **Doom (1993)**,
  byte-exact, on a 0.5B transformer.
- Honest real-time status: **not** real-time yet (~2–4 s/frame best case on one A5000), but the
  FLOP floor makes 35 fps *FLOP-feasible* on one GPU with the render step-count cut — the gap is
  the schedule-build + dispatch overhead, not raw compute.

### Reflections
✅ **HOLDS** — every reflection (division/log/discretization are hard for nets; convolutions/weight-
tying within a token; ALiBi+RoPE together; complex weights; softmax-temperature control; precision-
as-a-tool; thinking-as-kludge; pre-populated KV) is still apt and several were *reinforced* by the
later work. Keep essentially all of it.
➕ Add the ones the last five months taught: (a) direct-CAM O(1) beats the recency-horizon for data
reads; (b) baking a compiler and having the model run its own output *works*; (c) the honest wall
for "fast" is host dispatch / schedule build, not FLOPs; (d) a real, large, unforgiving program
(Doom) is the strongest possible completeness test and it passed.

---

## Part 2 — New sections to ADD

1. **The Doom capstone.** id's Doom → c4 (62/62 modules, ~555K instrs), byte-exact on the
   transformer: title frame SHA `2e883404` == gcc; E1M1 fully byte-exact vs gcc. Framebuffer =
   320×200 hex. This is the headline demonstration — a whole 1993 game, not a toy.
2. **Full C90 toolchain.** Recursive-descent C90 front-end, full preprocessor, soft-float IEEE-754
   library, native float opcodes (F_ADD/F_SUB/F_MUL/F_DIV, bit-exact vs soft-float), fixed-point
   accel (FIXEDMUL/FIXEDDIV), self-hosting transpile.c. Retire "subset of C."
3. **Doom render superinstructions (keeping the vanilla core).** DRAWSPAN (V_DrawPatch collapse,
   built this session, byte-exact vs the compiled loop), BLIT, NAMEEQ (WAD lookup, the 38% init
   step-cut), + host-offloaded present (framebuffer read from memory ⇒ no emit floor). All gated
   default-OFF; ON widens the OP_IS band to a deliberate new fingerprint. Good illustration of
   "stay vanilla by default, opt into accel."
4. **Compile-and-run in one transformer run.** The realization of "Model that Directly Runs C
   Code": proof (nibble_compiler / demo_model_runs_c, byte-exact vs c4), the mechanism (baked
   compiler + EMIT-into-writable-code-band + universal fetch = runtime code execution), and the
   honest roadmap to Doom-scale on the golden CFM VM.
5. **Honest performance chapter.** Fill the "Computational Efficiency" TODO with the numbers in
   §Sparse/Efficiency above, plus the speculation + KV-pruning quantification and the real-time
   ladder (what's achieved, what's feasible, what the true wall is).

---

## Part 3 — The draft's TODOs, resolved

| TODO in draft | Status / where |
|---|---|
| format opcode + weight tables | Regenerate from `c4_min/isa.py` + state_dict sparsity (see opcode REVISE) |
| clean up PureFFN/Attention/MoE code, "generate function" | Forwards hold; generate = standard HF loop (`blogspec_run.py`) |
| range-check / step-function graphs | Just add the plots; math is correct |
| self-hosting performance analysis | ~18 min self-emulation, byte-exact; all-C runtime static ELF |
| "comparison with other models / vs old videogames" | Doom byte-exact + FLOP floor + 3.09 µs/step dispatch |
| is `__argv_setup` current? | Verify routine against build; mechanism is right |
| cli examples | yes/cat/echo bundles exist — paste real invocations |
| adapting to precisions | chunk-size (BYTES_PER_REG) ↔ fp32/fp64 tradeoff |
| table of opcode neural implementations | Build from isa + per-op modules |
| mandelbrot/eliza writeups | Programs exist |
| quine elaboration | Quine done, byte-exact self-output |

---

## Part 4 — Honest-claim audit (keep the piece trustworthy)

**Under-claims to upgrade (now demonstrated):**
- "we *could* bake a compiler and hand off" → **we did; it runs its own compiled output byte-exact.**
- "highly amenable to speculation" → **665M-steps/s byte-identical draft, all-accept map.**
- "arbitrarily long programs feasible" → **Doom: millions of steps/frame, byte-exact.**

**Over-claims / stale to fix:**
- Opcode table numbering (64/65/41/42; separate 38/39/40) → current 40-op set, HALT=38 alias.
- "16 4-bit nibbles" ↔ "32-bit" → **8 nibbles / 32-bit**, BYTES_PER_REG=4.
- "C4 only supports int and chars" → native float + soft-float + full C90 front-end.
- Log-sink attention division reads as shipped → it's **designed but depth-infeasible**; shipped
  path is long division. Label it.
- Weight counts (5,487/1,397/800) are the **minimal VM**, not the Qwen-0.5B model — say so.
- Fully-vanilla register frame had a **template-add assist** — don't imply zero assists.

**Honest limits worth stating plainly (they make it more credible):**
- Full *general* attention baking needs per-layer real division ⇒ expensive (draft already says
  this — keep it).
- Real-time Doom **not** achieved; FLOP-feasible, overhead-bound (~2–4 s/frame best), not compute-bound.
- The Qwen-baked multi-byte ALU had genuine arch-block history; the clean constructions are the
  minimal-VM ones.

---

## Part 5 — Evidence pointers (for fact-checking while you revise)

- ISA / opcodes / bands: `c4_min/isa.py` (`NUM_OPS=40`, `NUM_OPS_FLOAT=44`, `HALT=38`, `F_ADD..F_DIV`).
- Widths: `c4_min/nibble_pure_forward.py` (`BYTES_PER_REG=4`), `blogspec_memory.py` (`ADDR_BITS=32`).
- Vanilla forwards / registers / bytecode format / neural I/O: `c4_min/blogspec_*.py`, `blogspec_run.py`.
- Compile-and-run-in-one-run: `c4_min/nibble_compiler.py`, `demo_model_runs_c.py`,
  `test_nibble_compiler.py`, `test_handoff.py`. CFM code-CAM: `c4_min/qwen_full_vm.py`
  (`_bake_code_cam`, `_overlay_code_frames`).
- Float / fixed-point: `c4_min/doom_fixedpoint.py`, the F_* ops in `isa.py`, soft-float `#799`/`#800`.
- Doom render ops: `c4_min/doom_blit.py`, `doom_nameeq.py`, `doom_drawspan.py` (DRAWSPAN=47).
- Golden: weight `069cc32f`; CFM sibling `7d19cdc3`; DRAWSPAN-ON `2f69350f`. Doom title `2e883404`.
- Self-hosting / bundler / all-C runtime: `onnx_runtime_nibble_allc.c`, `transpile.c`, `bundler/`.
- Perf reality: `REALTIME_SETUP.md`, the single-dispatch harness `c4_min/_agent_wholeframe_giantk.py`,
  `precomputed_schedule.py`.

> Bottom line: the draft's *architecture and primitives chapters are still gold and mostly need only
> number fixes.* The revisions cluster in (a) the opcode/width tables, (b) upgrading the
> "we could run a compiler" and "subset of C" sections to the now-demonstrated Doom + compile-in-one-run
> reality, and (c) filling the performance chapter with the honest measured numbers.
