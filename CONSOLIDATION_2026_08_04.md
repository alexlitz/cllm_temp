# Session consolidation manifest — 2026-08-04

Secured all of this session's verified work to durable commits (was scattered as
uncommitted worktree edits — nearly lost the wide-LEA to a fresh isolation checkout).
Golden `069cc32f` (authoritative c4_min weight fingerprint) unchanged: every lever is
gated default-OFF, and no fix touches the weight-authoring path (`unified_compiler/`).

## consolidate-0.5b (main checkout)
- `2e838335` — c4 toolchain fixes: char-literal-escape lexer, pointer-to-non-char stride
  = sizeof(int)=8, unary `+` (src/compiler.py); signed trunc DIV/MOD (native_c4.py). Byte-safe.
- `1e505cc3` — c4_original.c poolsz 256KB→512KB (self-host transpile.c). *Superseded by #855 (4-byte).*
- `22ea987a` — all-C CPU runtime: dead-block fusion (#879) + windowed/key-subset live attention
  (af32b58) + tail-only overlay (a1de45). Byte-exact echo/yes/cat/quine; static ELF.

## doom-e1m1-playable-transpiler-fixes (c4_doom)
- `2936cab` — transpiler wide-struct-pointer stride fixes → E1M1 renders hang-free & fully-lit
  (cliprange_t copy, crunch-loop next++, binary/cast-deref ptr arith). Title 2e883404 byte-exact.
  NOT yet byte-exact vs gcc (residual floor-plane light-precision divergence).
- `a2a0783` — menu link (#833) + interactive assembly + capstone notes.

## worktree branches (secured in-place; merge into consolidate-0.5b is the remaining step)
- `worktree-agent-ae788ce91cf3f6fa4` @ `4809beff` — wide-LEA CFM model + draft (a42384/ae788ce)
- `worktree-agent-a8adceb0ab701dbc7` @ `0fd76307` — wide-LEA re-gated to fine-grained set (#856, GO for #854 Stage 2)
- `worktree-agent-ac00906ee0d3bbdea` @ `648c9bf0` — #873 fused-delta sparse FFN
- `worktree-agent-a08c10af93999c6d2` @ `79f9b253` — #880 per-row block-skip
- `worktree-agent-ada75424369f546a5` @ `98f3a178` — #842 O(K) flash band (eff_K→1M)
- `worktree-agent-doom-divfree-bench` @ `e83ae914` — a19aa2f DIV-free re-measure (0.034 ms/step = 234× from 1s/frame)

## Deliberately NOT banked
- Megakernel / whole-step launch-collapse (a1ee254) — measured net loss (2.6–3×), dead end.
- ~25 prior-session worktrees (based on old commits) — out of scope.

## Remaining (post-secure)
1. Merge the worktree levers into consolidate-0.5b as one composed base (delicate: pf_kbatch/pf_speculative cross-lever conflicts).
2. #854 Stage 1 re-run under the fine-grained wide gate (0fd76307), then Stage 2 (flip + delete fold + new golden).
3. #855 4-byte-int standardization (weights-neutral; reverts 1e505cc3).
4. Perf: host-side O(steps) decode loop (the last 1s/frame lever, per ada75424).
