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
   hazard and is what TESTING_CHECKLIST_STATUS.md fingerprints.** The status doc's
   "Default-build fingerprint" table (TESTING_CHECKLIST_STATUS.md:39-42) is computed over
   `build_pure_forward_complete_model` — the DENSE builder the task warns never to run.
   The blog says nothing about this build being impractical dense; the honest story is
   only the lean (`include_muldiv=False` ~0.7 GB) / streaming (~3.6 GB) builds are
   usable. (Verified below by NOT building it and by reading the builder.)

*(Sections below add per-claim evidence; each section committed incrementally.)*

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
| 28 | DIV | 6/250 | ✅ | base-16 long division |
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
