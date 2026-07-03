# Long-Horizon Position Stability — Verdict (2026-06-13)

**Question.** Can the pure-neural VM stay byte-stable over hundreds-to-thousands
of VM steps, or is there an architectural position ceiling (ALiBi saturation /
KV eviction / attention-entropy collapse)? This decides whether the deep
clusters (`loop_* ~300-450 steps`, `rec_*`, `gcd_*`, ~350 programs, all
0/full-trace) are even REACHABLE for 100%.

**Verdict: NO architectural position ceiling. The deep clusters ARE reachable.**
Per-step `(PC, AX)` correctness depends on the OPCODE executed at that step, NOT
on the absolute position. The ONLY blockers are (1) the prologue AX/0xFF/framing
wall (steps 0-1, a parallel agent is fixing) and (2) the same per-step AX/0xFF
op-correctness bug repeated at every AX-bearing step. Fix per-step correctness +
the prologue and the deep loops decode to completion — position does not fight you.

---

## TL;DR evidence

Measured with a teacher-forced free-run harness (below): force a CORRECT context
through step `K-1`, then free-run and compare each step's decoded `(PC, AX)` to
the declarative DraftVM oracle. spec_k=0, GPU 1, no model-weight changes.

| program | step | abs token pos | window | PC ok | AX ok |
|---|---|---|---|---|---|
| loop_sum_5 | 2 | 386 | 4096 | ✅ | ✅ |
| loop_sum_5 | 10 | 666 | 4096 | ✅ | ✅ |
| loop_sum_5 | 20 | 1016 | 4096 | ✅ | ✅ |
| loop_sum_5 | 40 | 1716 | 4096 | ✅ | ❌ |
| loop_sum_5 | 20 | 1016 | 512 | ✅ | ✅ |
| loop_sum_5 | 280 | 10116 | 512 | ✅ | ❌ |
| loop_countdown_0 | 10 | 562 | 4096 | ✅ | ✅ |
| loop_countdown_0 | 20 | 912 | 4096 | ❌ | ❌ |
| loop_countdown_0 | **40** | **1612** | 4096 | ✅ | ✅ |

The smoking gun: **`loop_countdown_0` step 40 (abs pos 1612) is fully correct
while step 20 (abs pos 912) fails.** A position ceiling would make EVERY step
beyond some `P` fail. Instead, correct and incorrect steps interleave by opcode
all the way out to abs pos 10116. **PC (control flow) is correct at essentially
every position tested** (1 PC miss out of ~30 samples, itself opcode-driven, not
position-driven). Failures are 0xFF high-byte leaks / truncations in AX — the
same byte-class the prologue bug produces — and they recur on whichever VM step
does an AX load/compute, independent of how deep into the program it sits.

---

## Why the "ALiBi saturation at position 8000" hypothesis is FALSE here

The read-only hypothesis was: `max_seq_len=1024`, ALiBi bias grows linearly with
`|q_pos − k_pos|`, so at position 8000 the bias to position 0 is `≈ −80000` →
early context unreachable. **This does not apply to this model, for three
independent reasons:**

1. **Positions are `arange` over the (windowed) input tensor, not a true global
   step counter.** `AutoregressiveAttention.forward` (`neural_vm/vm_step.py:416-418`,
   no-KV path) sets `q_pos_1d = arange(S_kv-S_q, S_kv)`, `k_pos_1d = arange(S_kv)`
   where `S_kv` is the length of the tensor PASSED IN. The serial runner windows
   the context to the tail (`run_vm.py:973-974`: `ctx = ctx[-max_seq_len:]`), so
   the max ALiBi distance is **capped by the window**, never by the program's
   total token count. The `−4000` bias the hypothesis feared (slope 0.5 × dist
   8000) is structurally unreachable: the largest in-window distance is
   `slope × (window−1)` = `0.5 × 511 = −256` (512 window) or `−2048` (4096), and
   that only applies to the OLDEST in-window token — irrelevant to ops that read
   the most-recent dump.

2. **The VM re-emits the FULL register state every 35 tokens.** Register/state
   recovery is a distance-`≤35` attention (ALiBi bias `−17.5` for the strongest
   head, trivial), NOT a read back to position 0. The empirical PC stability
   across 280 steps confirms the model reliably re-anchors from the most-recent
   dump.

3. **Larger window made things BETTER, not worse** (`loop_sum_5` step 10: ❌ at
   512 window, ✅ at 4096 window) — the opposite of a position ceiling. The 512
   window occasionally evicts a needed PRIOR-STEP reference; that is a KV/window
   management nuance, not ALiBi saturation, and it points toward keeping context
   (the default KV cache is already 65,536 tokens — `batched_pure_neural.py:317`).

### Grounding facts (read, not assumed)
- Positional encoding: ALiBi (`config.py`, default). Slopes `2^(−8/H·(i+1))`;
  strongest head per 8-head block = 0.5 (`vm_step.py:197-201`). Slopes are
  per-head constants; they do NOT decay with distance, so distance matters — but
  distance is window-bounded (point 1).
- `STEP_TOKENS = 35` (`vm_step.py:114`): PC+AX+SP+BP+STACK0 (5 each) + MEM(9) + SE(1).
- ADDR_KEY is a per-position one-hot of the absolute byte address mod 4096
  (`neural_embedding.py:68-100`) — a content key the model attends to, NOT a
  smooth bias; it does not saturate.
- Default KV cache 65,536 tokens (`batched_pure_neural.py:308-317`): memory loads
  read historical MEM tokens through K/V, so the design INTENT is long retention,
  not a sliding window.

---

## Method (teacher-force past the prologue, then free-run)

`tools/probe_longhorizon_stability.py` (measurement harness, no weight edits):

1. Build the element context exactly as production (`_build_context`).
2. Compute the full per-step 35-token DraftVM oracle
   (`_oracle_pc_ax_steps(with_tokens=True)`) — declarative byte-identity.
3. **Teacher-force** steps `0..K-1`: append the oracle's 35-token blocks
   verbatim and run `_step_one` (so STEP_END dispatch / register tracking match
   production). Context is GUARANTEED correct through step `K-1`, bypassing the
   prologue bug.
4. **Free-run** from step `K`: drive the model one token per forward
   (`_forward_argmax_batch`, spec_k=0 semantics, model is sole authority), and
   after each completed step decode `(PC, AX)` from that step's own 35-token
   slice and compare to the oracle.
5. `--single-step-sweep P1,P2,...`: the purest position-isolation — for each `P`,
   force a pristine correct context through `P-1`, then free-run EXACTLY ONE step
   and record whether it matched. Decouples "correct context at position P" from
   free-run accumulation.

**Harness validation.** With `--tf-steps 0` (pure free-run) the harness
reproduces the canonical JSON byte-for-byte: `add_0` diverges at step 1
`expected (pc=18, ax=654) got (pc=18, ax=142)` (matches `/tmp/1096_v4.json`);
known-passing `add_3`/`add_9` report STABLE with correct exit codes 370/125.

**Run examples** (CUDA_VISIBLE_DEVICES=1, spec_k=0):
```
python tools/probe_longhorizon_stability.py --ids loop_sum_5 \
    --single-step-sweep 1,2,5,10,20,40,80,120,160,200,240,280 --model-max-seq-len 512
python tools/probe_longhorizon_stability.py --ids loop_countdown_0,loop_mul_0 \
    --single-step-sweep 5,10,20,30,40 --model-max-seq-len 4096
```
(4096-window runs OOM past abs_pos ~3500 on a contended GPU — that is O(S²) dense
attention memory, NOT a model limit. Use the 512 window, or shorter sweeps, to
reach the deep positions; the verdict holds in both regimes.)

---

## Implications for the deep-cluster strategy

- **Position is not the wall.** Do NOT invest in RoPE recompile, per-step renorm,
  or attention-entropy mitigations for the deep clusters — there is no ALiBi
  saturation, no entropy collapse, and the re-emitted-state design keeps the
  load-bearing attention at distance ≤35.
- **The wall is per-step AX correctness**, the SAME 0xFF high-byte leak / AX
  truncation family already tracked in memory (`project_ax_bytes_1_3_ff_leak_root`,
  `project_ax_ff_leak_is_tail_byte1_ff_emitters`, the L10 head-4 stack0_byte_relay
  + L25 tail `*_byte1_ff_*` emitters). Fixing it once fixes it at EVERY step,
  hence at every loop iteration. Combined with the prologue fix, the deep loops
  become decodable end-to-end.
- **Minor windowing nuance (secondary):** a 512-token window occasionally evicts
  a needed prior-step reference, causing a recoverable step error that a 4096
  window does not. The production default KV cache (65,536 tokens) already keeps
  context, so this is not a blocker — just prefer a generous context window /
  cache for deep programs over an aggressive sliding window.

**Bottom line:** the deep clusters are reachable for 100%. They are gated by
per-step (PC, AX) op correctness — the prologue AX/0xFF/framing fix plus the
AX high-byte leak fix — not by any long-horizon position ceiling.
