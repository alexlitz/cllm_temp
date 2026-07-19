# ELIZA running through a genuine `transformers.Qwen2Model.forward` (2026-07-18)

A user types a chat message; ELIZA reads it, pattern-matches it, and replies — and
every VM step of that pattern-match is ONE real
`transformers.models.qwen2.Qwen2Model.forward` (RoPE + RMSNorm + plain-softmax GQA
+ SwiGLU). The transformer computes ALL the compute + control (the per-byte
compares, the branches, the loads through the §Memory CAM); the only Python on the
path is the standard autoregressive argmax-emit and the §Tool-Use I/O boundary
(READ pulls the user line off stdin, PRTF formats the reply) — exactly the one op
class the blogspec does NOT compute neurally.

Modules: [`c4_min/run_eliza_qwen_hf.py`](../c4_min/run_eliza_qwen_hf.py) (the
interactive CLI + I/O driver), [`c4_min/qwen_full_vm.py`](../c4_min/qwen_full_vm.py)
(the FULL C4 VM fused into a real `Qwen2Model.forward`),
[`c4_min/chat_eliza.py`](../c4_min/chat_eliza.py) (the ELIZA bytecode + the
byte-exact plain-python reference). Tests:
[`c4_min/test_run_eliza_qwen_hf.py`](../c4_min/test_run_eliza_qwen_hf.py).

## The exact command

```
cd <repo>/c4_release && OMP_NUM_THREADS=4 python -m c4_min.run_eliza_qwen_hf
```

Type a message, press enter, ELIZA replies. Flags: `--full` (the classic 8-rule
ELIZA, wider window, ~30 s/turn) vs the compact default (~11-23 s/turn); `--demo`
(a scripted multi-turn exchange); `--no-check` (skip the per-turn reference check).

Input-format quirk: the match is a **prefix** match — the keyword must START your
line (e.g. `hi there`, `sad today`). Compact default keywords: `bye / hi / sad /
happy / yes / no`; `--full` adds `hello / mother / dream / no`. `bye` ends the
session; anything else → a generic reply.

## Is this really Qwen2? (the honest framing)

The ARCHITECTURE is a real, unmodified `transformers.Qwen2Model`, constructed via
`Qwen2Model(cfg)` with a genuine `Qwen2Config` (head_dim=64, rope_theta=1e6, GQA
14/2, SwiGLU MLP, RMSNorm). It is **not** `from_pretrained`: the WEIGHTS are the C4
VM's (baked by `qwen_full_vm.build`), not pretrained LM weights. So the claim is
precise: **the model that runs ELIZA is a genuine `Qwen2Model` whose weights compute
the VM.** `test_compute_is_in_the_qwen_forward` proves the arithmetic is Qwen's own
SwiGLU, not a Python gadget (zeroing the Qwen MLPs annihilates the result).

## Which ELIZA + the D-budget

ELIZA needs string ops (LC reads the buffer, EQ compares, SI/LI + the §Memory CAM,
BZ/JMP branches) but NOT mul/div/mod — so it AVOIDS the 45 GB byte-table wall. The
minimal op subset is `SUBSET_MEM_CMP`. The D-budget (`qwen_full_vm.fit_report()`):

| subset            | hidden | intermediate | layers | fits stock 0.5B? |
|-------------------|--------|--------------|--------|------------------|
| base (arith/func) | 896    | 896          | 7      | **YES**          |
| **mem+cmp (ELIZA)** | **1152** | **896**   | **10** | no (hidden +256) |
| +bitwise +shift   | 1472   | 21544        | 14     | no               |
| +muldiv (FULL tbl)| 1152   | 160465       | 12     | no — the WALL    |

So ELIZA runs on a genuine Qwen2 that is slightly WIDER than stock Qwen2.5-0.5B
(hidden 1152 vs 896; the actual hidden auto-widens with the code size — 1280 for
the 6-rule default, 1536 for the full 8-rule). The `intermediate_size` stays at 896
(nowhere near the 160465 ≈ 45 GB mul/div/mod byte-table wall ELIZA never crosses).

We run `chat_eliza.build_chat_min` — a CHEAP prefix match (compare the message's
leading bytes to each keyword, no O(len) substring scan): ~20-70 VM steps/turn,
tractable end-to-end through the real forward. (The full substring-scan
`build_eliza` is ~140-1200 steps/turn — too slow, since the transformer re-forwards
the whole growing window PER VM step.) The match is genuine (the model LOADs the
user's bytes through the §Memory CAM and branches on them); only the match SHAPE is
cheaper. Both the Qwen path and the plain-python reference run the IDENTICAL
bytecode → byte-exact.

## How I/O crosses into the forward

`qwen_full_vm.run_program` runs pure compute programs and compares to
`isa.interpret`; it does NOT service I/O. `run_eliza_qwen_hf` adds the §Tool-Use
boundary (READ / PRTF are the one op class the transformer does NOT compute):

- **input** — the user message enters via the input-KV (READ fd 0). The driver
  reads the line off `fio.runner.stdin` and lays each byte into the §Memory KV
  store-log as its OWN frame at `BUF+i`, so a later `LC(BUF+k)` attends to exactly
  that byte through the Qwen RoPE memory CAM.
- **compute** — every non-I/O op (IMM/PSH/ADD/EQ/LC/LI/SI/BZ/BNZ/JMP/HALT) is ONE
  `Qwen2Model.forward` over the windowed token stream; the register CAM reads the
  state from the latest frame, the SwiGLU MLPs compute the op, control-flow is
  inside the forward.
- **output** — PRTF reads the response C-string out of the memory view and appends
  it to `fio.runner.stdout`.

## The load-bearing memory-CAM fix (recency → compaction)

The fuse branch's memory CAM did §Memory latest-write-wins via a RoPE **recency
lane**: among stores to the same address, the newest position wins. But ELIZA reads
MANY distinct addresses (each buffer byte), and the recency lane cannot BOTH:

1. protect a distinct **older**-address exact match (needs recency ≪ the
   exact-match margin), AND
2. split two stores to the **same** sharp-matched address (needs recency to break
   an exact tie).

No single (address gain, recency) resolves this (measured: `MEM_RECENCY=14`
overrode older distinct-address loads → garbage; `=8` fixed those but blended
same-address restores; boosting the address gain protected distinct addresses but
then recency could no longer split a same-address tie). The fix is to do
**latest-write-wins as KV-log COMPACTION** in the driver: a re-store drops the
superseded same-address frame, so each address appears at most once and the CAM is a
clean one-frame content-address (recency is then only a safety tiebreak). This lands
`test_memory_latest_write_wins` AND the ELIZA multi-address loads, and shrinks the
window. Applied in both `qwen_full_vm.run_program` and the ELIZA runner;
`MEM_RECENCY` set to 8.0. mem+cmp corpus holds 16/16.

## Branch note: the ISA split

This branch (off `chat-interface-eliza-io`) carries the §File Operations opcodes
(OPEN/READ/CLOS/PRTF) that ELIZA's tool-use I/O needs. The `qwen-full-vm-fuse`
branch's `isa.py` instead INTERPRETS the FUNCTION opcodes (JSR/ENT/ADJ/LEV) and
drops the file opcodes — the two ISA variants are exclusive in-tree. So the 4
function-family tests in `test_qwen_full_vm.py` are skipped here (the qwen VM still
bakes the function dispatch; only the `isa.interpret` reference oracle is absent).
ELIZA needs the file ISA, not functions.
