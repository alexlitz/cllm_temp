# BLOG_SPEC ↔ c4_min implementation correspondence audit

**Date:** 2026-07-20
**Audit branch:** `blog-impl-correspondence-audit` off `unify-full-vm-no-lean-splits @ 49907f07`
**Scope:** every concrete claim in `docs/BLOG_SPEC.md` (943 lines) checked against the
`c4_release/c4_min/` implementation — reading the code AND running cheap tests, NOT
taking commit messages / docstrings at face value.

**Classification legend**
- ✅ **VERIFIED-IN-UNIFIED** — claim holds in the code on this branch (read + where feasible run).
- 🟡 **BRANCH-ONLY** — implemented only on a side branch, not on this unified trunk.
- 🗄 **ARCHIVE-ONLY** — exists only in the old `neural_vm` / `src/` archive, not c4_min.
- ❌ **MISSING** — claimed in the blog, not implemented anywhere in c4_min.
- ⚠ **OVERCLAIM-OR-MISMATCH** — code contradicts the claim (file:line + evidence + diff).

> **Two-models caveat (applies throughout).** c4_min ships TWO transformers: (1) the
> **headline nibble VM** (`nibble_pure_forward_complete.py`, softmax1 + ALiBi, the one
> the 1096 corpus runs on), and (2) a **Qwen 8-bit slice** (`qwen_full_vm.py` /
> `qwen_embed.py`, a real Qwen2Model.forward, RoPE + RMSNorm, 8-bit only, corpus
> sample 58/58). The blog is written as one VM; claims land on different models. Where
> it matters the audit says which.

---

## RANKED — most important overclaims / mismatches / gaps

Ranked by how misleading the blog is versus what the code actually does. **NEW**
findings (beyond the already-logged revisions in `BLOG_SPEC_REVISIONS.md`) are tagged.

1. **⚠ NEW — GETCHAR (64) / PUTCHAR (65) opcodes are in the blog table but REMOVED
   from the implementation.** The blog opcode table (BLOG_SPEC.md:160-161) lists
   `GETCHAR` and `PUTCHAR` as first-class neural opcodes with `L=9, W=220` each (and
   the Summary counts them: "I/O | 2 | 9 | 440", line 194). But the code explicitly
   removed them: `cli_tools.py:6` — *"the neural PUTCHAR / GETCHAR opcodes REMOVED"*;
   `chat_eliza.py:10` — *"neural PUTCHAR/GETCHAR were removed from the small model"*;
   `quine_prtf.py:7` — the quine uses PRTF *"NOT PUTCHAR"*. There is no `GETCHAR` /
   `PUTCHAR` opcode constant anywhere in `c4_min/`. All character I/O is funnelled
   through **PRTF (33)** + the think-tag / tool-call message protocol. So the blog's
   "I/O category, 2 ops, 440 weights" row and the 64/65 table rows are **stale**.
   *Structural confirmation:* `isa.py:14` sets `NUM_OPS = 40`, so the opcode one-hot
   routing band only spans values 0-39 — opcodes 64/65 are structurally out-of-band and
   cannot be dispatched even in principle (grep for a 64/65 neural handler = empty).

2. **⚠ NEW — MALC (34) / FREE (35) / MSET (36) / MCMP (37) are NOT neural opcodes in
   c4_min.** The blog table lists them as opcodes 34-37 with L/W budgets
   (BLOG_SPEC.md:149-152). `grep` for any of them as an opcode constant in `c4_min/`
   returns **nothing** (`nibble_pure_forward_complete.py` references opcodes 0-33 only;
   0 hits for MALC/FREE/MSET/MCMP). This is actually **consistent with the blog's own
   later prose** (§Tool Use Mode, line 853: *"malloc, free, memset, and memcmp are not
   tool calls. They're compiled from C into the VM's bytecode"*, i.e. they lower to
   LEA/LI/ADD/SI/SC loops, not dedicated opcodes). So the **opcode table (34-37) is the
   overclaim** — they should not be in the opcode-number table at all. malloc-on-the-
   transformer is real but as a *compiled bytecode subroutine*, not an opcode. (Extends
   BLOG_SPEC_REVISIONS #21 which caught the memcpy stray-header; the 34-37 rows are the
   same class of "syscalls presented as opcodes" mismatch.)

3. **⚠ NEW — BLT (41) / BGE (42) / POP (40) opcodes are in the table but absent from
   c4_min.** The blog table adds `POP(40)`, `BLT(41)`, `BGE(42)` (BLOG_SPEC.md:156-158).
   No opcode constant for BLT / BGE / POP exists in `c4_min/`. `isa.py` defines up to
   MOD(29) + OPEN/READ/CLOS/PRTF(30-33) + NOP(39) + HALT/EXIT(38); it stops there. BLT/
   BGE (signed branches) have no neural handler. (POP-as-`SP+=8` is subsumed — the
   POP_OPS mechanism in `nibble_pure_forward_complete.py` is a different thing: the
   implicit stack-pop that ADD/SUB/etc. do, not a standalone opcode 40.)

4. **⚠ (extends REV #17) — "45 opcodes" headline is not the c4_min op count.** The
   blog Summary claims **45 ops / 5,487 weights** (line 196). The c4_min pure-forward VM
   actually implements roughly **30 neural opcodes** (LEA…MOD = 0-29, plus PRTF, plus
   the SI/LI/SC/LC memory ops, plus JMP/BZ/BNZ/JSR/ENT/LEV/ADJ control) and delegates
   the rest (malloc-family to compiled bytecode, GETCHAR/PUTCHAR removed, BLT/BGE/POP
   absent, OPEN/READ/CLOS to tool-call). The "45" is the aspirational full-C4 table,
   not the built model's op set.

5. **⚠ (confirms REV #22) — KV-pruning "99.999%" / "logarithmic" / "hundreds of
   millions of tokens" are extrapolations, not measured.** Verified the revision log's
   correction is grounded: `nibble_kv_prune.py` implements the eviction, and the
   measured ceiling is ~98.7% at ~4.9K tokens; growth is FLAT (small-heap) or
   O(live-heap) (memory-heavy), never logarithmic. (Details in the KV section below;
   this is REV #22 confirmed by reading the code, not just trusting the log.)

6. **⚠ (confirms REV #21) — "Memset, Memcmp and Memcpy" header — memcpy is not a C4
   opcode.** Confirmed: no MCPY / memcpy anywhere in `c4_min/`; the classic-C4 syscall
   set is MSET/MCMP only. Stray blog header.

7. **⚠ NEW — the dense `build_pure_forward_complete_model` is a 54-108 GB memory
   hazard and is what TESTING_CHECKLIST_STATUS.md fingerprints — AND its own vanilla-
   exec guard test is effectively unrunnable.** The status doc's "Default-build
   fingerprint" table (TESTING_CHECKLIST_STATUS.md:39-42) is computed over
   `build_pure_forward_complete_model` — the DENSE builder the task warns never to run.
   **Measured during this audit:** running `test_exec_path_vanilla.py`'s
   `test_pure_forward_complete_exec_is_vanilla_argmax` (which calls
   `build_pure_forward_complete_model(code_size=16)`) ballooned the pytest process to
   **~105 GB RSS** before I killed it — so even the "lean" `code_size=16` guard test for
   the complete model is a dense-build hazard, not runnable on a normal machine. The
   blog says nothing about this build being impractical dense; the honest story is only
   the lean (`include_muldiv=False` ~0.7 GB) / streaming (~3.6 GB) builds are usable, and
   the `nibble_pure_forward_complete` vanilla-exec claim can only be *statically*
   (AST + settrace) verified for the complete model, not run.

8. **⚠ NEW — `torch.round` on 4 non-headline exec paths.** The vanilla-argmax
   requantiser is used on the headline paths (recurrent / pure-forward / blogspec_run,
   guarded by `test_exec_path_vanilla.py`), but `nibble_bake.run_baked` (:558),
   `universal.py` (:582), `nibble_handoff.py` (:347), and `nibble_compiler.py` (:817)
   all call `torch.round` — and those four are exactly the capstone paths (baked-program/
   malloc, universal bytecode, model-runs-C, compiler). So "no rounding anywhere on the
   exec path" (REV #4 / status:76) is true only for the headline 3.

9. **⚠ NEW — MAGIC-floor fp trick (§555) is described-only; efficient-exp (§561-564) and
   division-via-log-sink (§653-679) are not implemented.** No `2**23`/`8388608` constant
   exists; floor is bit-plane selection. Division is long-division only (the blog does
   flag log-sink division as non-default, so that one is honest; MAGIC-floor is presented
   as used but isn't).
   > **[2026-07-23 UPDATE]** Partly superseded — see §2026-07-23 REFRESH R2. The log-sink
   > divide WAS later built (`226d38bb`/`c5641182`, full ISA 291→118 layers) then REMOVED
   > with fp64 (`fa6dadea` deleted `nibble_logsink_*.py`), so "not implemented" → "built,
   > then removed for fp32-vanillaness." A radix-16 hardened divide (`div_radix16_hardened.py`,
   > 88 blocks, fp32) now exists as a standalone module (not the default on consolidate). A
   > Granlund–Montgomery MAGIC-number CONSTANT divide was written (`60b27f6f`) but is on an
   > UNMERGED side branch — NOT on this tree — so the §555 MAGIC-floor gap is still open here.

*(Sections below add per-claim evidence; each section committed incrementally.)*

---

## 2026-07-23 REFRESH — deltas since the 2026-07-20 audit

**Refresh scope.** Four changes landed after the 2026-07-20 audit was written; this
section re-checks each against the code on the CURRENT tree
(`consolidate-0.5b-2026-07-22`, worktree HEAD contains commit `046efd8e`). Findings
#9, the §Vanillaness / §Adapting-to-Precision discussion, the §Division table rows, and
the §KV-cache / §Memory / §How-Bytecode-is-Passed sections are affected. The original
findings below are LEFT INTACT as the historical 2026-07-20 record; where one is now
stale I add an inline `> **[2026-07-23 UPDATE]**` pointer back here.

> **Branch-state discipline for this refresh.** Several of the divide changes the task
> brief cited live ONLY on side worktree branches and are **NOT on consolidate / this
> tree**. I verified merge state with `git merge-base --is-ancestor <c> consolidate-0.5b-2026-07-22`
> for every commit and say explicitly, per item, whether the code is present here. Two
> of the brief's claims were **wrong for the current tree** and are corrected below
> (const_div_truncated.py absent; radix-16 not the wired default here).

### R1 — fp64 REMOVED from the `qwen_full_vm` production path (genuinely fp32-vanilla). ✅ VERIFIED

Commit `fa6dadea` ("remove ALL fp64 from the qwen_full_vm production path") is on this
tree. The Qwen VM config now HARD-CODES fp32: `qwen_full_vm.py:287`
`model_dtype: object = torch.float32   # the model is always fp32 (0 fp64 params)`, and
the build docstring asserts "The model is ALWAYS fp32 (`model_dtype == torch.float32`,
zero fp64 params)" (`qwen_full_vm.py:298`, echoed :338). The float64 branch and the
`K_DIV` raise were dropped; `nibble_vm.maybe_cast_model_for_width` is now a no-op
(always fp32) and the two-limb path is forced under width-32 so the single-scalar fp64
fallback is retired (per the `fa6dadea` message, confirmed by grep — `nibble_vm.py` has
no float64 model param).

**Caveat I confirmed (not a hole, but worth stating precisely):** fp64 is NOT
zero-in-the-repo — it survives as RUNTIME-PRECISION-ONLY, never a model param:
- `qwen_full_vm.py:981` (`_snap`, the value-argmax DECODE helper) uses
  `torch.arange(VALVOCAB, dtype=torch.float64)` because `v²ᐧ≈4.3e9` at the SP/BP value
  vocab (`~0x10100`) exceeds fp32's 2^24 integer precision — a Python-side snap of the
  model's fp32 lane, not a weight. Its own docstring says so (:978-979: "The MODEL is
  still fully fp32 (zero fp64 params); only this local Python argmax uses doubles").
- `nibble_cmp.py`/`nibble_muldivmod.py`/`nibble_rope.py`/`nibble_io_position.py` still
  compute weight-VALUE coefficients in float64 and cast the output to fp32 (these feed
  the HEADLINE nibble VM's baked weights, not the Qwen params).
So "0 fp64 params in the built qwen model" holds; "0 fp64 anywhere in c4_min" does not,
and the `fa6dadea` message is explicit that the residual fp64 is runtime-precision only.
This UPDATES finding #9's implicit "fp64 divide" concern and the §Vanillaness discussion.

### R2 — Divide is no longer long-division-ONLY (but the DEFAULT here is still long division). ✅ VERIFIED with corrections

The 2026-07-20 §Division rows say "long division only; log-sink & MAGIC-floor
described-only." Status on THIS tree:

- **Radix-16 hardened divide EXISTS as a module.** `div_radix16_hardened.py` is present:
  a base-16 digit-recurrence divide, **NO reciprocal / NO per-digit multiply**, with a
  Kogge–Stone parallel-prefix borrow keeping every quantity in nibble-lane `[-16,15]`.
  Depth is **88 blocks** (`:51` "depth 80 -> 88"; `:53` "88 < 127"), fp32, "0 fp64
  params" (`:60`). ⚠ **Correction to the brief:** the module's own header line 3 states
  it is a **"Standalone MEASURE-ONLY bakeoff module"** — its byte-exact self-check
  numbers are COMPUTED AT RUNTIME by `measure()` (`:699`, "edge grid + adversarial +
  >=6000 random (a,b<2^32) pairs" through a CPU SwiGLU forward sim), NOT baked as a
  fixed count in the docstring. So "88 blocks, fp32, byte-exact" is right, but there is
  no hardcoded self-check total in this file; the **3594/3594** figure the brief cited
  comes from the (unmerged) `bc6c7684` commit message, not from this tree's code.

- **Radix-16 as the DEFAULT divide is NOT on this tree.** Commit `bc6c7684` (#735) wires
  radix-16 as default + width-narrow + `C4_CONST_OPERAND` hooks, but
  `git merge-base --is-ancestor bc6c7684 consolidate-0.5b-2026-07-22` = **NO** (it lives
  only on `worktree-agent-a4262510a4c274337`, pending a GPU gate). On THIS tree
  `qwen_full_vm.py` contains **zero** references to `radix16` (grep empty) — the default
  divide is the fp32 base-16 **recurrent/unrolled long division** (`recurrent_divmod`,
  the `nibble_alu32` gadget; `qwen_full_vm.py:193-202, 463-466`). So the audit's
  "long-division default" is STILL CORRECT for consolidate; radix-16 is an available
  alternative module, not the wired default here.

- **fp64 log-sink divide was BUILT then REMOVED.** `git log --all` shows `226d38bb`
  ("§653 log-sink softmax division reference — ~11 blocks vs 262-block long division")
  and `c5641182` ("log-sink division as the DEFAULT divide — full ISA 291→118 layers")
  DID build it. `fa6dadea` then **DELETED** `nibble_logsink_blocks.py` (856 lines) and
  `nibble_logsink_div.py` (confirmed in the `fa6dadea --stat`), unwired it from
  `qwen_full_vm` + `full_native_fast`, and replaced `logsink_integration_state()` with a
  fp32 `divide_integration_state()`. `ls nibble_logsink*.py` and
  `git log --all | grep logsink` confirm the files are absent from the current tree. So
  finding #9's "log-sink NOT IMPLEMENTED" is now "**built (291→118 layers), then removed
  with fp64 for vanillaness**." (The blog was always honest that log-sink is non-default;
  it was briefly the default on c5641182, now reverted.)

- **Granlund–Montgomery CONSTANT divide — ⚠ NOT ON THIS TREE (brief was wrong).** The
  brief says `const_div_truncated.py` (commit `60b27f6f`) "now exists." It does NOT
  exist on consolidate/HEAD: `ls const_div*.py` = no matches;
  `grep -rln "Granlund\|Montgomery\|const_div_truncated"` = empty. `60b27f6f` lives ONLY
  on side branch `worktree-agent-ac37a51826c7b1460`
  (`git merge-base --is-ancestor 60b27f6f consolidate-0.5b-2026-07-22` = **NO**). The
  const-divisor work that IS on consolidate is a DIFFERENT, exploratory module —
  `div_automaton_attn.py` (commit `2b4fe0d1`, "const-divisor divmod as an ATTENTION-CAM
  automaton") plus the `div_radix16_attn.py` bakeoff (`5b1dae8e`) — both standalone
  measure/bakeoff modules, neither wired into the qwen production build, and neither is
  the Granlund–Montgomery magic-number truncated-multiply construction. The
  `C4_CONST_OPERAND` flag likewise does NOT exist on this tree (grep empty) — it is
  introduced by the unmerged `bc6c7684`. So the "MAGIC-floor described-only" gap (finding
  #9 / §555) is **still open on consolidate**: the magic-number constant divide is
  written, but on a branch that is not merged here.

**Net for R2:** on consolidate, the default divide is still long division (audit correct);
radix-16-hardened is a present-but-standalone module (88 blocks, fp32, measure-only);
log-sink was built-then-removed (audit's "not implemented" is superseded by "removed for
fp32-vanillaness"); and the Granlund–Montgomery constant divide + `C4_CONST_OPERAND` are
NOT on this tree at all (only on unmerged side branches) — a correction to the brief.

### R3 — KV eviction is CONTENT-BOUNDED for the fast-path GLOBAL heads (runtime-independent). ✅ VERIFIED, with a model-attribution correction

Commit `046efd8e` (#723, MERGED to consolidate —
`git merge-base --is-ancestor 046efd8e consolidate-0.5b-2026-07-22` = **YES**)
content-bounds the 3 GLOBAL address-CAM heads (memory / stack-pop / LEV; block 7 head 20,
block 11 heads 21-22). The mechanism, verified in code:
- `local_attention.py:63-71` documents the store-role GATE channel `cR = ADDR_BITS+1`:
  every global head bakes `W_k[base+ADDR_BITS+1, ONE] = -p` + `W_k[..., IS_STORE] = p`,
  so a NON-STORE row scores `-PEN` under a load/pop/lev query → softmax1 weight EXACTLY 0.
- `nibble_pure_forward_cached.py:1155-1187` `_global_content_keep` applies exactly this:
  a row is kept iff its `cR` gate is above `-p/2` (store rows ~0, non-store ~`-p`), union
  across the 3 global heads; the split commit (`:1129-1147`) routes only surviving store
  rows into the global cache. `test_local_attention.py:165-180` asserts
  `c.size() == #store rows, not span length`.
- `local_attention.py:277` — "This is what makes the TOTAL KV runtime-independent."
- Quantitatively (from the `046efd8e` message, a probe/bench not re-run in this CPU-only
  refresh): on `nested 5x20` (1684 steps, 457 stores) the content-bound global cache
  rises 7→~11 as the working set is covered then stays **FLAT at 11-12** (peak **13-16**)
  for the whole run, vs drop-KV-only holding ~472. Across K=8/32/64/128/256 the content
  gcache is 7/7/18/34/66 vs drop-KV 34/268/1197/2795/6530 — so it is bounded by the LIVE
  ADDRESS SET, not step count (the "K-invariant" claim; note it still grows MODESTLY
  7→66 with the working set — it is working-set-bounded, not literally constant).

⚠ **Correction to the brief's model attribution.** The brief calls this "the Qwen
fast-path." It is NOT the Qwen slice. `qwen_full_vm.py` imports neither
`nibble_pure_forward_cached` nor `local_attention` nor `install_local_attention` (grep
empty). All three files are the **KV-cached fast-path DRIVER of the HEADLINE sparse
pure-forward nibble VM** — `nibble_pure_forward_cached.py:2,5,1301` is explicitly "KV-cached
form of `run_pure_forward_complete`", and `pf_speculative.py:1` is "speculation for the
SPARSE pure-forward whole-VM model." So this lands on **model (1) the headline nibble VM's
fast-path cache**, a DISTINCT thing from the reference `nibble_kv_prune.py` the §KV section
below audits (that module is the ground-truth eviction POLICY; this is the incremental
per-block cache the deep-loop driver actually runs). This UPDATES the §KV-Cache section's
"memory eviction is UNBOUNDED heap-mirroring" — that verdict was about `nibble_kv_prune.py`'s
content-addressed recency horizon (still a no-op there), but the fast-path driver now DROPS
provably-inert non-store frames so its GLOBAL cache is content-bounded/runtime-independent.
Both statements are true of DIFFERENT modules; the audit's "unbounded" was never about the
fast-path driver.

### R4 — code-from-memory is now the DEFAULT. ✅ VERIFIED

`qwen_full_vm.py` flips `code_from_memory=True` by default at every entry point:
config field `qwen_full_vm.py:284` (`code_from_memory: bool = True   # program in KV
§Memory (fetch@PC), not a baked table`), the layout builder kwarg `:167`, and `build()`
`:294`. With it on (`:203-217, 480-489`) the baked `CODE_OP[k]`/`PC_IS[k]` table +
`pc-fetch`/`code-select` blocks are replaced by a single `code-cam` attention block that
FETCHES the instruction at PC out of KV §Memory — one code frame per instruction — and
the build docstring names it "the 'Universal = bytecode fetched by PC' mechanism …
Fetch is then program-length-independent (no `CODE_OP[k]` table)" (`:468-473`). This
STRENGTHENS the blog's §How-Bytecode-is-Passed / §Universal correspondence: the program
genuinely lives in §Memory as CODE frames fetched at PC, not a baked lookup table. (The
`code_from_memory=False` path — the old baked table — is retained as an explicit opt-out.)

### Refresh summary of corrections to the task brief

1. **fp64 is NOT gone from the whole repo** — only from qwen model PARAMS. Runtime-precision
   fp64 (value-argmax snap, weight-coefficient computation, reference oracles) remains, by
   design; the model params are 0-fp64. (Brief's "genuinely fp32-vanilla" holds for the
   built model, as stated.)
2. **radix-16 is present but NOT the wired default on consolidate** — it's a standalone
   MEASURE-ONLY module; the default divide here is still recurrent long division. The
   "88 blocks, fp32, byte-exact" is right; the 3594/3594 self-check total is from the
   unmerged `bc6c7684`, not this tree.
3. **`const_div_truncated.py` (Granlund–Montgomery) does NOT exist on this tree** — it and
   `C4_CONST_OPERAND` are on unmerged side branches only. The MAGIC-floor constant-divide
   gap (finding #9) is still open on consolidate. The const-div work that IS here is the
   different `div_automaton_attn.py` CAM-automaton (also not wired into the build).
4. **The KV bound lands on the HEADLINE nibble VM's fast-path driver, not the Qwen slice.**
   Otherwise the R3 mechanism and the flat ~11-16-row / working-set-bounded (not strictly
   constant) behavior are exactly as the brief describes.

---

## §C4 Opcode table (BLOG_SPEC.md:53-196) → per-op classification

The blog has TWO opcode tables: the "How implemented in C4" table (lines 54-104,
0-40) and the "L/W" table (lines 106-161, adds 41/42/64/65). Reference impl:
`c4_min/isa.py` (8-bit slice interpreter + opcode constants) and
`c4_min/nibble_pure_forward_complete.py` (the neural VM that actually dispatches ops).

| # | Op | Blog L/W | c4_min status | Evidence |
|--:|----|----------|---------------|----------|
| 0 | LEA | 1/20 | ✅ | `isa.LEA`; dispatched in complete forward (LEA handler) |
| 1 | IMM | 0/0 | ✅ | `isa.IMM`; `ax = imm` |
| 2 | JMP | 1/12 | ✅ | `isa.JMP`; PC←imm |
| 3 | JSR | 2/30 | ✅ | PUSH_OPS incl JSR; `MEM[SP-4]=pc+1;SP-=4;PC=t` (complete:19) |
| 4 | BZ | 1/10 | ✅ | `isa.BZ` |
| 5 | BNZ | 1/10 | ✅ | `isa.BNZ` |
| 6 | ENT | 2/40 | ✅ | PUSH_OPS incl ENT (complete:20,92) |
| 7 | ADJ | 1/16 | ✅ | `ADJ` const (complete:71-72); `SP+=4n` |
| 8 | LEV | 2/40 | ✅ | IS_LEV head, `SP=BP;BP=MEM[SP];PC=MEM[SP+4];SP+=8` (complete:22) |
| 9 | LI | 1/500 | ✅ | `isa.LI` memory KV load |
| 10 | LC | 1/500 | ✅ | `isa.LC` (referenced complete) |
| 11 | SI | 1/500 | ✅ | `isa.SI` memory KV write |
| 12 | SC | 1/500 | ✅ | `isa.SC` (referenced complete) |
| 13 | PSH | 2/80 | ✅ | PUSH_OPS incl PSH |
| 14 | OR | 1/80 | ✅ | POP_OPS bitwise; per-nibble table |
| 15 | XOR | 1/80 | ✅ | POP_OPS bitwise |
| 16 | AND | 1/80 | ✅ | POP_OPS bitwise |
| 17 | EQ | 1/80 | ✅ | zero-detector (nibble_cmp) |
| 18 | NE | 1/80 | ✅ | zero-detector |
| 19 | LT | 2/100 | ✅ | sign+EQ |
| 20 | GT | 2/100 | ✅ | sign+EQ |
| 21 | LE | 2/110 | ✅ | LT∨EQ |
| 22 | GE | 2/110 | ✅ | GT∨EQ |
| 23 | SHL | 2/160 | ✅ | POP_OPS shift |
| 24 | SHR | 2/160 | ✅ | POP_OPS shift |
| 25 | ADD | 2/120 | ✅ | 6-weight nibble add + carry |
| 26 | SUB | 2/120 | ✅ | nibble sub + borrow |
| 27 | MUL | 4/320 | ✅ | schoolbook (nibble_muldivmod) |
| 28 | DIV | 6/250 | ✅ | base-16 long division (default; radix-16 hardened alt exists off-tree — 2026-07-23 REFRESH R2) |
| 29 | MOD | 7/370 | ✅ | div-then-subtract |
| 30 | OPEN | 1/20 | 🟡 tool-call | `nibble_filesys` TOOL_CALL protocol, not neural |
| 31 | READ | 9/250 | 🟡 tool-call / neural-msg | conversational-IO + tool path |
| 32 | CLOS | 1/10 | 🟡 tool-call | tool path |
| 33 | PRTF | 1/20 | ✅ | `isa.PRTF` (isa.py:148); visible-byte channel |
| 34 | MALC | 1/51 | ⚠ not-an-opcode | compiled to bytecode (blog line 853), 0 opcode const |
| 35 | FREE | 1/27 | ⚠ not-an-opcode | zero-overwrite via SI; not an opcode |
| 36 | MSET | 1/16 | ⚠ not-an-opcode | compiled loop of SC |
| 37 | MCMP | 1/16 | ⚠ not-an-opcode | compiled loop |
| 38 | EXIT | 1/9 | ✅ | `isa.HALT = 38`; emits HALT/EOS |
| 39 | NOP | 0/0 | ✅ | `isa.NOP = 39` |
| 40 | POP | 1/16 | ❌ | no opcode-40 constant; POP_OPS is a different mechanism |
| 41 | BLT | 1/17 | ❌ | no signed-branch handler |
| 42 | BGE | 1/17 | ❌ | no signed-branch handler |
| 64 | GETCHAR | 9/220 | ⚠ REMOVED | "neural GETCHAR removed" (chat_eliza:10, cli_tools:6) |
| 65 | PUTCHAR | 9/220 | ⚠ REMOVED | "neural PUTCHAR removed"; PRTF only (quine_prtf:7) |

**Note on isa.py:** `c4_min/isa.py` is an **8-bit-only slice** reference interpreter
(header line 1) — its `interpret()` only handles the arithmetic/compare/mem subset and
`raise NotImplementedError` for anything else (isa.py:161). It is NOT the opcode table
authority; the neural dispatch lives in `nibble_pure_forward_complete.py`. But even
that dispatches only 0-33 as neural ops.

**Verdict (opcode table):** The core computational ISA (0-33, 38, 39) is faithfully
implemented. The table **overclaims** by listing malloc-family (34-37) and POP/BLT/BGE
(40-42) and GETCHAR/PUTCHAR (64/65) as opcodes when they are either compiled-to-bytecode
subroutines, absent, or explicitly removed. The blog's own §Tool-Use prose (line 853)
already contradicts its own opcode table on 34-37.

---

## §Vanillaness (BLOG_SPEC.md:233-403) → model architecture

Reference: `c4_min/blogspec_model.py` (the headline VM's transformer),
`c4_min/nibble_moe.py` (MoE).

| Claim | Class | Evidence |
|-------|-------|----------|
| softmax1 = `exp(x)/(1+Σexp(x))`, only real deviation | ✅ | `blogspec_model.py:66-78` `softmax1`; +1 sink, ZFOD |
| softmax1 realizable via BOS sink under plain softmax | ✅ | `softmax1_via_bos_sink` (:81-101); a 0-logit column reproduces `+1` exactly; toggle `sink="bos_sink"` |
| ALiBi slopes = geometric `2^(-8/n·(i+1))` | ✅ | `blogspec_model.py:150-153` — matches blog §307-311 char-for-char |
| ALiBi replaceable by RoPE | ✅ | `positional="rope"` toggle (:181-215); RoPE binary-distance recency, content Q/K untouched (matches REV #11) |
| SwiGLU FFN `down(silu(up·x)·(gate·x))` + additive residual | ✅ | `FFN.forward` (:314-318) `silu(up)*gate` |
| MoE = soft blend, all experts run, opcode-one-hot weighted | ✅ | `NibbleStandardMoEFFN.forward` (:179-198) — verbatim port of blog `StandardMoEFFN`; `x + Σ w_i·(E_i(x)−x)` |
| MoE ONNX-clean (static unroll, no `.item()`) | ✅ | opcode list is Python ints; loop unrolls at trace (nibble_moe:169-172) |
| top-1 routing (compute only active expert) — REV #13 | ✅ | `NibbleTop1MoEFFN` (:204-293) ArgMax + `index_select` (Gather), argmax-identical to soft |
| No RMSNorm / norm-free residual (spec's ref is norm-free) | ✅ | default `norm="none"`; RMSNorm-compensator is an *optional* toggle (:104-120, matches REV #11) |
| No exotic ONNX ops / no special masking | ✅ | forward uses only linear/matmul/softmax/triu-causal-mask; 1 `scatter`-mention is a comment saying it is NOT used (nibble_moe:66) |
| No `torch.round` on the **exec path** — REV #4 | ⚠ **split** | see below |

**⚠ NEW — `torch.round` IS on FOUR non-headline exec paths.** The headline paths
(`recurrent.py`, `nibble_pure_forward_complete.py`, `blogspec_run.py`) use the vanilla
LM-head argmax requantiser `argmax_n(2·n·x − n²)` (`recurrent.py:138-156`,
`nibble_pure_forward_complete.py:1189`), and `test_exec_path_vanilla.py` guards them
(AST + settrace tripwire). **BUT** four other recurrent exec loops call `torch.round`
for their per-step requant:
- `nibble_bake.py:526` (`_requantize`, called by `run_baked` :558 — the *baked-program-
  in-weights* / "malloc-on-transformer" demo loop),
- `universal.py:582` (universal bytecode-as-system-prompt recurrent run),
- `nibble_handoff.py:347` (model-runs-C handoff),
- `nibble_compiler.py:817` (compiler recurrent run).

`test_exec_path_vanilla.py` does **not** cover these four modules, so the "zero exec-
time torch.round" claim (TESTING_CHECKLIST_STATUS.md:76-77) holds only for the headline
3 paths, not for the baked/universal/handoff/compiler paths. Since these produce the
capstones (model-runs-C, universal, quine, malloc), the vanilla-requant story is
**inconsistent across paths** — the blog/REV #4 claim of "no rounding anywhere on the
exec path" is an overclaim for those four. (Note the round versions are byte-equivalent
to argmax on clean integer state, so this is a *vanillaness* purity issue, not a
correctness bug.)

## §The Building Blocks + ALU/math primitives (BLOG_SPEC.md:504-682)

Reference: `nibble_cmp.py` (comparisons), `nibble_muldivmod.py` (mul/div/mod),
`nibble_bitwise.py` (shifts/bitwise/floor), the baked versions in
`nibble_pure_forward_complete.py`.

| Primitive | Class | Evidence |
|-----------|-------|----------|
| Step fn `silu(S(x+ε))−silu(S(x−ε))` | ✅ | `nibble_cmp._step_ge1` / `_step_ge` (muldivmod) |
| Point indicator (2nd-difference bump) | ✅ | `zero_detector` (nibble_cmp:96-106) — exact blog §584-586 transcription |
| Range-check via two step fns | ✅ | band-range checks in nibble_bitwise dispatch |
| ReLU-from-SiLU (scale away from 0) | ✅ | `_relu(z)=silu(RELU_S·z)/RELU_S` (muldivmod) |
| Zero-detector for EQ/NE/BZ/BNZ | ✅ | `zero_detector`; 11 params (9w+2b); 4-nibble packing BASE=31 (§590) |
| 6-weight ADD/SUB (`silu(S(a+b))/S`) | ✅ | nibble add+carry; matches §593 |
| 6-weight MUL (`(silu(Sa)+silu(−Sa))·b/S`) | ✅ | `_mul` (muldivmod); exact blog §593 |
| Schoolbook MUL, 10 partial products, i+j≥4 skip, 3 carry rounds | ✅ | muldivmod docstring:6-14 + impl; matches §620-640 |
| Base-16 long division, `q=Σ_{k=1}^{15} step(rem−k·div)` | ✅ | muldivmod:16-19; matches §648 |
| MOD = div-then-mul-then-subtract | ✅ | muldivmod:22 (§682) |
| Per-nibble bitwise tables (256-entry AND/OR/XOR ×16) | ✅ | nibble_bitwise dispatch rules; §685 |
| Comparisons are UNSIGNED 32-bit — REV #9 | ✅ | nibble_cmp:66 "unsigned 32-bit ordering, matching isa.interpret's masked comparisons" |
| **MAGIC floor `(x+2^23)−2^23`** (§555) | ⚠ **DESCRIBED-ONLY** | no `8388608`/`2**23` constant anywhere in c4_min; floor is done by bit-plane selection (`nibble_bitwise._floor_div_pow2` :235-243), NOT the fp-mantissa MAGIC trick. §555 is prose, not code. |
| **Efficient exp (BOS-sink, key √d, value e^B)** (§561-564) | ❌ **NOT IMPLEMENTED** | grep for the exp/log-sink construction finds nothing; division uses long division only (§Division default) — REV #19 already notes the shipped divider is long-division. §561-564 is described-only. |
| **Division via attention-with-log-sink** (§653-679) | ❌ **NOT IMPLEMENTED** | the blog itself says it "set[s] the default behavior to be using long division" (§679); the log-sink divider is not built. Described-only (correctly flagged as non-default by the blog). |

> **[2026-07-23 UPDATE — log-sink divide row + MAGIC-floor row]** See §2026-07-23 REFRESH R2.
> The log-sink divider WAS subsequently built (`226d38bb` §653 reference ~11 blocks;
> `c5641182` made it the default, full ISA 291→118 layers) and then REMOVED with fp64
> (`fa6dadea` deleted `nibble_logsink_blocks.py` + `nibble_logsink_div.py`; grep for
> `logsink` in the current tree = empty). So "NOT IMPLEMENTED" → "built, then removed for
> fp32-vanillaness"; long division is again the default (audit still correct on the default).
> Separately, a radix-16 hardened digit-recurrence divide (`div_radix16_hardened.py`, 88
> blocks, fp32, reciprocal-free, measure-only) now exists as an ALTERNATIVE divide. The
> MAGIC-floor §555 fp trick is now realized for the CONSTANT-divisor case (Granlund–Montgomery,
> `60b27f6f`) but ONLY on an UNMERGED side branch, so §555 remains described-only on consolidate.

**Verdict (ALU/building blocks):** the SHIPPED constructions — zero-detector, step,
6-weight add/sub/mul, schoolbook mul, base-16 long division, mod, per-nibble bitwise —
all match the blog's math faithfully (`nibble_cmp` / `nibble_muldivmod` are near-verbatim
transcriptions). Two of the blog's *alternative/optional* gadgets are **prose-only**: the
MAGIC-floor fp trick (§555) and efficient-exp/division-via-log-sink (§561-564, §653-679).
The blog does flag the log-sink divider as non-default, so that one is honestly labeled;
MAGIC-floor is presented as if used but isn't.

## §Memory (BLOG_SPEC.md:408-412, 687-691) + §Registers (442-461)

Reference: `blogspec_memory.py`, `blogspec_vocab.py`.

| Claim | Class | Evidence |
|-------|-------|----------|
| Store attends to registers for addr/val, binary-address ±scale keys | ✅ | blogspec_memory:39-50; `W_k` maps bit→`2·smag·bit − smag` (±smag) |
| Query identical to key retrieves the address | ✅ | QRY_BIN with identical ±smag map (:47-50) |
| Exact match dominates; 1 differing bit loses | ✅ | +n·EFF vs 2k·EFF cost; ZFOD BIAS `(n-1)·EFF` |
| ALiBi recency = latest-write-wins | ✅ | MEM_ALIBI_SLOPE; among equal addresses newest wins (:65-69) |
| softmax1 ZFOD — unwritten reads 0 | ✅ | +1 sink dominates when no match (:71-75) |
| free = zero-overwrite → newest match returns 0 | ✅ | :73-74 (§691) |
| 4-byte aligned, 32-bit addresses | ✅ | ADDR_BITS=32, "4-byte aligned" (:122) |
| NULL memory write every step (keeps it simple) | ✅ | blogspec_memory:29-33; vocab MEM slot every frame (§461) |
| 30-token register frame (PC/AX/SP/BP/MEM/STEP_END) | ✅ | `blogspec_vocab.build_step_frame` (:build_step_frame); exactly 30, asserted |
| re-quant = emit→re-embed byte token (argmax) not round | ✅ | blogspec_vocab:17 header; the byte tokens ARE the quantiser |
| deep-recursion scale EFF=500000 (REV #6) live | ✅ | `blogspec_memory.py:123` `EFF = 500000.0`; horizon EFF/slope=500000 tokens |
| Eviction is UNBOUNDED / heap-mirroring (REV #3) | ✅ | see KV section below |

**Verdict (memory):** an exceptionally faithful transcription — the memory subsystem is
the strongest blog↔code correspondence in the whole project. Binary-CAM keys, query=key,
ALiBi recency, softmax1 ZFOD, zero-overwrite free, 4-byte alignment, NULL-per-step, and
the 30-token frame all match. EFF=500000 (the deep-recursion fix) is live.

## §Weight / sparsity numbers (BLOG_SPEC.md:170-229) — MEASURED

The blog's Summary claims **45 ops / 5,487 reported → 1,397 unique → ~800 with full
op-class sharing**, and **96.4% sparse** ("With full op-class sharing | 88,548 | ~800";
"tensors are 96.4% sparse"). I MEASURED the actual built c4_min nibble models (memory-
safe, RSS reported):

| Build | dim | blocks | total dense entries | **nonzero** | **sparsity** | unique nonzero values | RSS |
|-------|----:|-------:|--------------------:|------------:|-------------:|----------------------:|----:|
| lean (`include_muldiv=False`) | 1533 | 15 | 919,784,304 | **86,602** | **99.991%** | 314 | 4.1 GB |
| muldiv (`include_muldiv=True`) | 1533 | 17 | 12,711,741,907 | **1,052,510** | **99.9917%** | 550 | 50.8 GB |

**⚠ NEW — the blog's weight numbers do not describe the built model:**
- **Sparsity is understated.** Blog says "96.4% sparse"; the measured lean model is
  **99.991% sparse** and the muldiv model **99.9917%**. The 96.4% figure is far too low
  — the real tensors are >99.99% zeros. (The blog's own §Sparse Tensors prose says
  "over 99%+ sparse", which is right; the 96.4% in the Summary table is inconsistent
  with the blog's own later text AND with measurement.)
- **Nonzero count is ~15×–190× higher than claimed.** Blog "5,487 reported nonzero";
  measured **86,602** (lean) / **1,052,510** (muldiv). The c4_min *nibble* VM with real
  fp32-exact 32-bit divmod needs ~1M nonzero weights, not ~5.5K. The 5,487 is the
  blog's *idealized per-op minimal* count for a 45-op model, NOT the built nibble model.
- **"~800 unique weights" vs measured 314 / 550 unique VALUES.** These aren't directly
  comparable (blog's "unique" = unique per-op-class *positions* after sharing; measured
  = unique nonzero *scalar values*), but the built model genuinely uses only **314-550
  distinct scalar values** across ~1M nonzeros — heavy value replication, consistent with
  the blog's redundancy thesis (and with REV #12's weight-tying dedup). So the *spirit*
  of "few unique weights" holds; the *specific* 5,487/1,397/800 table does not map to the
  built model.

**Verdict (weight numbers):** the blog's per-op L/W table and the 5,487/1,397/800 Summary
are an **idealized accounting for a minimal 45-op model**, not a measurement of the
shipped nibble VM (which has ~86K–1.05M nonzeros, is 99.99% sparse, and uses 314-550
unique values). The qualitative claims (extreme sparsity, few unique values, heavy
sharing) are TRUE and if anything understated; the specific numbers are aspirational and
should be labeled as such (extends REV #17/#18).

## §File Operations + §Printing and Reading Input + §Tool Use Mode (693-717, 849-853)

Reference: `nibble_filesys.py`, `blogspec_vocab.py` (think-tag), `vm_causal_lm.py`.

| Claim | Class | Evidence |
|-------|-------|----------|
| OPEN/READ/CLOS via TOOL_CALL token protocol | ✅ | nibble_filesys `ToolCall`/`FileRunner.handle` (:285-299); wire format `TOOL_CALL:<type>:<id>` |
| runtime lib (malloc/free/memset) are neural bytecode, not tool calls (§853) | ✅ | consistent with opcode audit — no MALC/MSET opcode; they lower to bytecode |
| Two I/O modes (tool-call + native conversational) — REV #15 | 🟡 partial | tool-call fully present; "native" READ is a Python buffer (below) |
| think-tag protocol: THINK_END, visible byte, THINK_START | ✅ (token-level) | `blogspec_vocab.visible_output` (:visible_output) parses visible bytes outside think tags; THINK_START/END tokens 265/266 |
| **Native stdin via position-signature attention** (§702-739): multiple ALiBi-slope heads at BOS producing exponential tuples that uniquely map to position, nibble-cascade offset extraction | ⚠ **DESCRIBED-ONLY / Python buffer** | `InputKVStream` (nibble_filesys:242-255) is a plain Python byte buffer — `self.data[self.pos:self.pos+n]` — NOT the neural position-signature mechanism. Docstring claims "read out by attention, 100%-native pathway" but no IO/position-signature attention heads are baked anywhere (grep for IO_ALIBI/position-signature heads in `nibble_pure_forward*` = empty). |
| **PRTF visible output is neural** | ⚠ **Python driver** | the visible byte is `out.append(ax & 0xFF)` read by the Python driver from decoded AX (nibble_pure_forward_complete:913), not emitted through a neural think-tag mechanism on the model |
| position-offset via nibble cascade (§718-739) | ❌ not built | the O(log N) comparison-cascade offset extractor is described, not implemented |
| System-prompt format (BYTECODE/SEP/DATA/SEP/ARGV) | ✅ | `blogspec_vocab` SEP=264; `universal.py` loads code table |

> **[2026-07-23 UPDATE — "Universal = bytecode fetched by PC"]** Strengthened; see
> §2026-07-23 REFRESH R4. `qwen_full_vm.py` now DEFAULTS `code_from_memory=True`
> (`:284, 167, 294`): the program lives in KV §Memory as CODE frames fetched at PC by a
> single `code-cam` attention block (`:480-489`), replacing the baked `CODE_OP[k]`/`PC_IS[k]`
> table (fetch is now program-length-independent — the build docstring `:468-473` names it
> "the 'Universal = bytecode fetched by PC' mechanism"). This is a closer match to the
> blog's §How-Bytecode-is-Passed than the baked-table path the audit saw on 2026-07-20.
| argv (`__argv_setup` read-as-input, §751-793) | 🟡 BRANCH-ONLY | not plumbed on this unified branch (grep for argv_setup/bake_argv = empty); the task notes argv was re-plumbed on branch `argv-read-plumb` |

**⚠ NEW — the "100% native conversational I/O" is largely a Python harness, not neural.**
The blog's most elaborate I/O section (§702-739, the multi-head ALiBi position-signature
+ nibble-cascade offset machinery for reading user input by attention) is **not
implemented as a neural mechanism**. Native stdin (`InputKVStream`) is a Python buffer;
PRTF's visible byte is read out by the Python driver. The think-tag *token stream*
contract (visible bytes outside THINK tags) exists at the vocab level, and
`vm_causal_lm.py` drives read/write turns via HF `model.generate()`, but the
position-signature attention that the blog spends ~40 lines describing is not there. So
"basic IO possible with a 100% native version of the transformer" (§695) is an
**overclaim** for the position-addressed input path — it works, but through a Python
runner, not the described neural position signatures.

## §KV Cache Pruning (800-816) — confirms REV #22 by reading `nibble_kv_prune.py`

| Blog claim | Class | Evidence |
|-----------|-------|----------|
| ">99.999% pruning" | ⚠ extrapolation | REV #22: measured max 98.744%; 99.999% is a steps→∞ ratio, not measured |
| "cache at 1-10K tokens", "hundreds of millions of tokens" | ⚠ | measured ceiling ~4.9K tokens; 10⁸ is plausible-but-unmeasured |
| "grows logarithmically" | ⚠ **wrong** | actually FLAT (small-heap) or O(live-heap) UNBOUNDED (memory-heavy); no program is logarithmic |
| single "cosine > 0.99 for memory AND registers" | ⚠ **wrong→fixed in code** | `nibble_kv_prune.py:237-264` SPLITS it: registers = cosine dup-merge (`dup_metric="cosine"`); memory (`content_addressed`) = exact relative-L2 (:241-256). Cosine would wrongly merge distinct stores (shared ADDR_BIN common-mode ⇒ raw cosine 0.999). Exactly REV #22. |
| latest-write-wins register eviction | ✅ | mechanism 1 (:235-264) |
| same-address supersession + ZFOD/free zero-overwrite eviction | ✅ | zero-value branch (:310-311); free = value→0 |
| memory eviction is UNBOUNDED heap-mirroring (REV #3) | ✅ | content_addressed recency-horizon is a NO-OP (:312-324): "tracks the UNBOUNDED live heap ... NO fixed recency/size cap"; a live store evicted only by supersede or free |

> **[2026-07-23 UPDATE]** Still true FOR `nibble_kv_prune.py` (the reference eviction
> POLICY this section audits). But see §2026-07-23 REFRESH R3: a SEPARATE module — the
> headline VM's fast-path KV-cached DRIVER (`nibble_pure_forward_cached.py` +
> `local_attention.py`, commit `046efd8e`, MERGED) — now CONTENT-bounds the 3 global
> address-CAM heads by dropping provably-inert non-store frames (softmax1 weight exactly 0),
> so its GLOBAL cache stays flat at ~11-16 rows (working-set-bounded, runtime-independent),
> NOT growing with history. The "unbounded" verdict was never about this driver; both are
> true of their respective modules. (This is the fast-path cache, NOT the Qwen slice.)

**⚠ NEW nuance — the module's OWN docstring header still repeats the wrong claims.** The
implementation (lines 235+) correctly splits cosine/exact and makes memory unbounded, but
`nibble_kv_prune.py:1-15` still says "keeps the cache **bounded (logarithmic)**" and
"**cos-sim > 0.99 + ALiBi**" for registers-and-memory unified — the same errors REV #22
corrects. So the code is right but its top-of-file doc lags the fix (a self-inconsistency
worth cleaning). Correctness (byte-exact decode under eviction) is asserted by
`test_kv_cache_equivalence.py` / `test_kv_free_driven.py` (present on branch).

## §Sparse Tensors (817-819) + §Exiting (794-796)

| Claim | Class | Evidence |
|-------|-------|----------|
| network >99% sparse, sparse-tensor (COO) storage | ✅ | measured 99.99% (above); `export_onnx.py:20` sparse_initializer; `onnx_to_c4bin.py:127` COO |
| sparse makes ONNX significantly smaller | ✅ | export_onnx:265 "shrinks because 99% of entries are zeros" |
| HALT/EXIT triggers EOS-token generation | ✅ | `isa.HALT=38`; vocab `HALT=262` "ends generation (§Exiting)"; complete forward emits HALT |
| special halt token distinct from EOS in message mode | ✅ | vocab HALT=262 separate from message-end; matches §796 |

## Capstones (baking, model-runs-C, speculation, self-hosting, quine, bundling)

Tests RUN this session are marked (ran); memory-safe small models only.

| Capstone | Class | Evidence |
|----------|-------|----------|
| **Baking bytecode into weights** (KV-retrieval, "read-only code segment") §828-832 | ✅ (ran) | `nibble_bake.py` — per-nibble equality → one-hot mask → MoE-value select; `test_nibble_bake.py` **12/12 passed** |
| baked program runs with NO bytecode in input ("malloc-on-transformer" foundation) | ✅ (ran) | `run_baked`/`BakedProgram`; `initial_state_no_program` asserts bare embedding; 12/12 |
| **Full-attention baking via SwiGLU** §834-838 | 🟡 BRANCH-ONLY / approximate-by-design | not on this unified branch (grep empty); task notes branch `full-attn-baking-completeness`. Blog itself frames it "for completeness"; only KV-retrieval baking (the needed one) is on trunk |
| **Model-runs-C** (C in → compiler-in-weights → result) §840-842 | ✅ present | `nibble_compiler.py`, `demo_model_runs_c.py`, `nibble_handoff.py`; ⚠ its recurrent run uses `torch.round` (handoff:347, compiler:817) |
| **Speculation** (perfect draft, 100% accept, ~1000× / huge blocks) §797-799 | ✅ present (not re-run) | `nibble_speculative.py` (perfect-draft logical VM, accepts in one pass), `batched_speculative.py` (cross-program). Status doc's 901-vs-80,524 forwards (89.4×) is documented, not independently re-run here. REV #10. |
| **Self-hosting, 3 relationships** §859-899 | ✅ present (artifacts) | `onnx_runtime_nibble.c`, `onnx_runtime_nibble_fixedpoint.c` (C4-C ONNX runtime); the 3 relationships are realized by running the runtime under c4vm and vice-versa. No dedicated verification re-run on this branch (status doc: "not re-run full-1096 in this CPU session") |
| **Quine** (outputs own source via transformer) §922-930 | ✅ (ran) | `quine_prtf.py` (PRTF string quine, byte-exact self-output through the small pure-forward transformer), `quine_bundle.py`; `test_quine_prtf.py` **6 passed, 1 skipped**. Uses PRTF not PUTCHAR (quine_prtf:7) |
| **Bundling** (runtime+weights+bytecode → single ~150-200KB binary; cat/echo/yes CLIs) §910-920 | ✅ present | `bundle_small.py`, `bundler/neural_bundler.c`, `bundler/bundle_c4.c`, `cli_tools.py`; not size-measured this session |
| fixed-point bundle variant (no float, 2^12 scale, SiLU LUT) §920 | ✅ present | `onnx_runtime_nibble_fixedpoint.c`; `nbl_bin_interp.py` |

**Verdict (capstones):** all capstones are PRESENT on this unified branch, and the two I
ran cheaply (baking/malloc-foundation 12/12, quine 6/7) PASS byte-exact. Full-attention
baking is branch-only + approximate-by-design (as the blog anticipates). The
model-runs-C / handoff / compiler / universal recurrent paths carry the non-vanilla
`torch.round` (finding #8). Speculation and self-hosting are present but their headline
performance numbers (89.4× forward reduction; full self-hosting execution) are documented
in the status doc, not independently re-measured in this CPU-only audit.

## §Registers / §Tokenization / §RoPE Binary Distance / two-models nuance

| Claim | Class | Evidence |
|-------|-------|----------|
| Registers PC/AX/SP/BP 32-bit; SP=BP init 0x10000 (or 0x8000) | ✅ / 🟡 | `nibble_pure_forward.py:746 SP_INIT=0x10000` (matches blog default) BUT corpus runners override to **0xFC** (`run_corpus_resumable.py:74`, REV #7 small addressable stack). So the blog's 0x10000 is the *module default*, not the *corpus config*. |
| Immediate = 32-bit operand read from code at PC | ✅ | `isa.assemble` keeps imm at 32-bit width (isa.py:60-69) |
| Nibbles: 32-bit value = 16×4-bit nibbles, little-endian | ✅ | `blogspec_vocab.nibbles_of_value`, `nibble_cmp` NIB_PER_REG=16 |
| Registers written every step → 30 tokens/step | ✅ | `blogspec_vocab.FRAME_LEN=30`; matches §444-459 exactly |
| Many passes just output registers (no real compute) | ✅ | true — the 30-token frame is mostly bookkeeping (REV #16) |
| RoPE binary distance `theta_k=2^k`, `alpha=sqrt(2·SCALE/d)`, aligned score≈SCALE | ✅ | `nibble_rope.py:14-94` — `binary_thetas` = `[2^k]`, `enc_k(p)=alpha·cos(2^k·p)` — matches §743-746 char-for-char |
| **"pure transformer, no exotic architecture"** (§3 intro) | ✅ (headline) / 🟡 (nuance) | headline nibble VM is clean vanilla; the Qwen slice is a genuine `Qwen2Model.forward` (`qwen_full_vm.py`) but 8-bit-only (pruned MUL/DIV/MOD lookup table, `& 0xFF`, 8-bit addresses). So "the VM as a genuine Qwen2" holds only for the 8-bit slice; the fp32-exact 32-bit ALU runs on the softmax1+ALiBi headline model. |

> **[2026-07-23 UPDATE]** See §2026-07-23 REFRESH R1: as of `fa6dadea` the Qwen slice is
> now genuinely fp32-vanilla — `model_dtype` is hard `torch.float32` with ZERO fp64 params
> (`qwen_full_vm.py:287`), and the 32-bit ALU MUL/DIV/MOD now run as fp32 `nibble_alu32`
> gadgets INSIDE the real `Qwen2Model.forward` (the 256×256×3 lookup table was removed).
> The "8-bit-only" caveat above described the earlier lookup-table build; the efficient-ALU
> path is full 32-bit-exact in fp32. The only remaining fp64 is a Python-side value-argmax
> decode snap (`:981`), never a model param.
| headline "1096/1096" | 🟡 documented | measured 1085/1096 (`934ea608`), +11 deep-recursion fix live (EFF=500000); the full re-sweep with the fix is a documented GPU follow-up (status:56-73), not re-run here |

**⚠ two-models nuance (REV #11 territory).** The blog reads as ONE transformer. In fact
there are two: (a) the **headline nibble VM** (softmax1 + ALiBi, fp32-exact 32-bit ALU,
the 1096 corpus) and (b) the **Qwen 8-bit slice** (RoPE + RMSNorm + GQA, genuine
`Qwen2Model.forward`, 8-bit only, 58/58 sample). The blog's "you could look at the ONNX
and say yup that is a standard transformer" is best evidenced by (b); the full-width math
lives in (a). Both are vanilla, but the "embed the VM into a real Qwen2" claim is the
8-bit slice, not the full 32-bit VM.

---

## Honest verdict

The c4_min implementation is a **strong, largely faithful** realization of the blog's
core: the vanilla transformer (softmax1 + ALiBi/RoPE + SwiGLU-MoE, no exotic ops), the
softmax1-KV binary-address memory with ALiBi recency and ZFOD/free, the 30-token register
frame, the argmax re-quantization, and the ALU math gadgets (zero-detector, 6-weight
add/sub/mul, schoolbook mul, base-16 long division, per-nibble bitwise) are all present
and, where I ran them (foundation 10/10, bake 12/12, quine 6/7), byte-exact — the memory
subsystem and comparison/arithmetic constructions are near-verbatim transcriptions of the
spec's math. Every capstone (baking, model-runs-C, speculation, self-hosting, quine,
bundling) is present on the unified branch. **Where the blog and code DIVERGE, it is
almost always the blog overclaiming a clean/uniform story the code refines:** the opcode
table lists ops that were removed (GETCHAR/PUTCHAR), are compiled-to-bytecode not opcodes
(malloc-family 34-37), or are absent (BLT/BGE/POP 40-42); the weight numbers (5,487 /
96.4%) describe an idealized 45-op model, not the measured nibble VM (99.99% sparse,
~86K-1M nonzeros); the "no torch.round on the exec path" holds only for the 3 headline
paths and not the 4 baked/universal/handoff/compiler capstone loops; the elaborate native
position-signature I/O (§702-739) is a Python buffer, not neural heads; the MAGIC-floor
and efficient-exp/log-sink-division gadgets are described but unbuilt; and the KV-pruning
"99.999% / logarithmic / cosine-for-both" claims are corrected in the code (and
`BLOG_SPEC_REVISIONS.md`) but the pruning module's own header still repeats the old wrong
text. The `BLOG_SPEC_REVISIONS.md` log already catches most of these directionally
(#3/#4/#9/#11/#17/#18/#21/#22); this audit adds concrete file:line evidence and surfaces
NEW ones the log missed — the removed GETCHAR/PUTCHAR, the malloc-family/BLT/BGE not being
opcodes, the four `torch.round` capstone paths, the native-IO-is-a-Python-buffer, the
described-only MAGIC-floor/efficient-exp, and the measured 99.99%-sparse / 105 GB
dense-build hazard. **Net:** the implementation is more honest than the blog; the blog
should be trimmed of the removed/idealized/described-only items so a reader isn't told the
model does things it doesn't (or does them differently than described).
