# ARCHIVE INDEX — unmerged branch work (negatives / prior-session levers / findings)

Read-only audit index of the work that lives OFF the default line
(`consolidate-0.5b-2026-07-22` @ `f5f31bd8`) — the negative-result probes, the
prior-session perf levers, and the findings/measurement branches. It answers the
question "is old / superseded work preserved and indexed, or is it about to be
GC'd?".

Generated 2026-08-06. Golden `069cc32f` (re-confirmed intact via
`c4_min._fingerprint_build`). **Docs-only; no weight/build change.**

Companion docs:
- [`DOC_ARCHIVE_MANIFEST_2026_07_23.md`](DOC_ARCHIVE_MANIFEST_2026_07_23.md) — archives
  stale *doc FILES* (moved into `docs/archive/`). This index catalogs unmerged *BRANCH
  work*. The two are complementary and do not overlap.
- [`DOOM_FLAG_REGISTRY.md`](DOOM_FLAG_REGISTRY.md) — the flag inventory many of these
  lever branches land as `C4_*` toggles.

---

## TL;DR

- **The named negative-result / lever branches are ALL preserved on real branch refs**
  (not at risk of GC). Verified: each of the 9 negative-result commits the audit named
  is reachable from `git branch --contains` (a live ref keeps it alive past the 2-week
  `gc.pruneExpire`).
- **⚠ RISK FLAG: there are ~306 genuinely-orphaned dangling commits** (non-stash,
  `git branch --contains` == 0) in the main repo — real lever + findings work
  (multi-byte ADD/SUB/DIV relays, mul byte-1 delivery, JSR PC-byte1, deepframe LEA,
  AX-hibyte clears, wall-4 SE-CMP session notes, etc.). These are **at risk under the
  default `gc.pruneExpire=2 weeks`** since nothing references them. Most predate the
  current c4_min/doom focus and their *conclusions* have landed elsewhere (in memory
  notes / merged commits), but the *artifacts* are not on any ref. See "At-risk
  dangling work" below for the recommended `git branch`/`git tag` rescue if any is
  wanted.
- The separate `c4_doom` repo (`/home/alexlitz/Documents/misc/c4_doom`) is self-hosted
  (own `.git`, no remote), 9 branches, only 8 dangling commits — safe.

---

## Reachability model (why "on a branch" == "preserved")

Git only GCs commits that are **unreachable from any ref** (branch, tag, remote-tracking,
reflog) once they age past `gc.pruneExpire` (this repo: DEFAULT = 2 weeks;
`gc.auto=6700`). A commit on a named branch — even a stale `worktree-agent-*` one — is
reachable and **cannot be pruned**. So the audit's real question splits in two:

1. Is each named lever on a live ref?  → YES for all 9 (table below).
2. Is there lever/findings work ONLY in a dangling commit?  → YES, ~306 commits (the
   risk flag).

---

## Category A — negative-result branches (the named set)

All 9 are on a live `worktree-agent-*` (or named) ref. `contains` = number of branches
that contain the commit (≥1 ⇒ preserved).

| branch (ref) | commit | contains | what it is |
|---|---|---|---|
| `worktree-agent-a9ffa4aeacfea630f` | `f769fb87` | 1 | DENSE bf16 tensor-core FFN probe (`C4_FFN_DENSE_BF16`, default-OFF) — negative perf lever |
| `worktree-agent-af598490243f8eb6f` | `df4e4a23` | 1 | GPU-vectorize the faithful-path eviction (`C4_EVICT_GPU_VEC`, default-OFF) + #861 breakdown |
| `worktree-agent-a7f75bb0373f5b4ad` | `0e47d814` | 2 | bf16 KV-scoring on the GENUINE faithful path (`C4_FAITHFUL_BF16_SCORE`, default-OFF) |
| `worktree-agent-a5509a09bf6d7f18d` | `4b688546` | 1 | INCREMENTAL-PEEL larger-radix MUL prototype — byte-exact, default-OFF analysis lever |
| `worktree-agent-ad72025f587e87f44` | `d088bb26` | 1 | byte-radix (radix-256) MUL prototype — byte-exact, default-OFF analysis lever |
| `agent-lowprec-doom-2026-08-05` | `c88d31ee` | ≥1 | doom low-precision (bf16/fp16/int8) byte-exactness + fps probe |
| `worktree-agent-aa245004869a69088` | `4870c455` | 1 | three FFN-compute-structure refinements on the doom-active dead-FFN chain |
| `worktree-agent-a35eefe79b186d4bd` | `df226322` | 1 | de-entangle the dead-FFN residual chain (`C4_DEENTANGLE_RESIDUAL`, default-OFF) |
| `worktree-agent-a1796e1613335c707` | `6ea58f17` | 1 | PERSIST whole-chain dead-FFN megakernel (`C4_MEGABLOCK_PERSIST`, default-OFF) |

Verdict: **negative-result set fully preserved.** These are the "we tried X, it did not
net-win / it was byte-exact-but-no-gain" probes; each is a default-OFF flag so none can
regress the golden.

## Category B — prior-session perf levers (named)

| branch (ref) | commit | what it is |
|---|---|---|
| `megakernel-fusion-741` | `9d8a05db` | #741 megakernel-fusion merge line (composed FFN-span fusion) |
| `doom-wall4-imm32` | `2d68dad9` | WIP: WALL #4 32-bit immediate materialization (`C4_WIDE_IMM`, default-OFF) |
| `doom-nameeq-827b` | `dc33c44c` | #824 LEA_WIDE + #826 CMP32 order — doom 40K byte-exact merge |
| `doom-nameeq-intrinsic` | `f6145c92` | #828 doom NAMEEQ intrinsic — fused 8-char WAD lump-name compare (default-OFF) |
| `logsink-division-653` | `226d38bb` | §653 log-sink softmax division reference — ~11 blocks vs 262-block long division |
| `big-k-speculation-2026-07-20` | `4eae7770` | big-K speculation lever + measured deep-loop numbers |
| `block-moe-divmod-skip` | `49907f07` | block-MoE divmod-skip + config-toggle-matrix collapse |

## Category C — findings / measurement branches (named)

| branch (ref) | commit | what it is |
|---|---|---|
| `measure-self-emulation-wall` / `selfhost-qwen-efficient-perf` | `9489da75` | self-emulation cost measurement + efficient-ALU (nibble_alu32) replacing the 45 GB table |
| `one-layer-self-emulation` | `b58f862e` | one-layer self-emulation: A@x over OWN weights byte-exact through model.forward |
| `chk1-longrange-validate` | `ad71eceb` | long-range KV / 1096 memory validation |
| `chk1-measure-full1096-run2` | `a46a52d3` | full-1096 measurement run |
| `mandelbrot-significant-render` | `89fc171e` | Mandelbrot general-program render (the doom-agnostic showcase) |

> Note: the main repo carries **2660 local branches** (1685 `worktree-agent-*`). The
> tables above are the AUDIT-NAMED subset. The full branch list is not reproduced here
> (it is enumerable via `git for-each-ref refs/heads/`); every one of them is a live ref
> and therefore preserved. The overwhelming majority are per-op / per-fix worktrees from
> the superseded neural-VM 1096 campaign whose conclusions live in the user memory notes.

---

## ⚠ At-risk dangling work (the real gap)

`git fsck --no-reflogs --unreachable` reports **463 unreachable commits**; **306** of
them are genuine standalone commits (not `WIP on` / `index on` / `untracked files on`
stash-like entries — those are recoverable via their named parent branch). Spot-checked
samples (`254f54dc`, `085d2957`, `4220db04`) confirm `git branch --contains` == 0.

These 306 carry real work from the 2026-06/07 neural-VM campaign, e.g.:

| dangling commit | what it is |
|---|---|
| `254f54dc` / `f9b1c376` | multi-byte dividend relay + 2-byte long divide → 1096 div/mod 45→76 |
| `472d2da2` / `8d86652a` | multi-byte ADD byte-1 adder + addend relay head |
| `62b3d0ff` / `6d4cb23e` / `14ce656a` | multi-byte SUB minuend relay + no-borrow passthrough |
| `4a679e64` | mul byte-1 reconstruct onto EMIT-G5 fold (`C4_MUL_B1_DELIVERY`) |
| `bb18269c` / `e2d35166` | JSR PC-byte1 initial-AX-leak fix |
| `085d2957` | deepframe LEA (#330) scaffold |
| `d74c5339` | SP deep-frame depth-track (`C4_SP_DEEP_FRAME_DEPTH_TRACK`) |
| `f7b6205c` / `5119f45c` | AX byte-1 0xFF-leak clear — REFUTED net-negative |
| `41488809` / `3d8092c3` / `5819599f` | wall-4 SE-CMP session root-cause notes |
| `512eed03` / `0398e463` | OPCODE-DISPATCH muldiv to bytecode subroutines + barrel-shifter |

**Risk:** under default `gc.pruneExpire=2 weeks`, a `git gc` (auto-triggered at
`gc.auto=6700` loose objects) will prune any of these that age out. Their *findings*
are largely captured in the user memory notes (multi-byte relay walls, the AX-hibyte
refutation, the wall-4 SE-CMP-is-dead conclusion) and superseded by the current
declarative ALU + c4_min efficient-ALU path, so the *loss* is artifact-only — but the
audit's remit is to FLAG it, not to silently accept it.

**Recommended rescue (if any is wanted — NOT done here, this is a read-only audit):**

```sh
# Pin every non-stash orphan under a namespace so gc can never prune them:
git for-each-ref --format='%(objectname)' | sort -u > /tmp/reachable.txt
git fsck --no-reflogs --unreachable 2>/dev/null | awk '/^unreachable commit/{print $3}' \
  | while read sha; do
      subj=$(git log -1 --format='%s' "$sha")
      case "$subj" in "WIP on"*|"index on"*|"untracked files on"*|"On (no branch)"*) ;; \
        *) git tag -f "archive/orphan/$sha" "$sha" ;; esac
    done
# (or disable pruning entirely for this repo:)  git config gc.pruneExpire never
```

---

## Golden safety

- Golden fingerprint `069cc32fa7cecfbceae448a7dbf6e2140b3db6cf6857c8accec5639b9c55c0ca`
  (short `069cc32f`) re-confirmed via `python -m c4_min._fingerprint_build`.
- Every negative-result / lever branch above lands its work behind a **DEFAULT-OFF**
  `C4_*` flag, so even if merged the golden flag-OFF gate is unchanged by construction.
- This doc adds no build/weight change.
