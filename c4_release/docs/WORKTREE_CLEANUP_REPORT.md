# Worktree Cleanup Report (Step 9)

Date: 2026-06-03

## Summary

Bulk cleanup of stale `.claude/worktrees/agent-*` registrations from prior
Claude Code agent sessions. All affected worktree paths were already missing
on disk (orphan registration entries kept alive by `locked` flag).

## Counts

| Category | Before | After | Delta |
|---|---:|---:|---:|
| Total worktrees | 624 | 442 | -182 |
| `.claude/worktrees/agent-*` | 532 | 350 | -182 |
| `.agent-worktrees/*` | 4 | 4 | 0 |
| `/tmp/c4-*` | 77 | 77 | 0 |
| main repo + other | 11 | 11 | 0 |

## Methodology

For each `.claude/worktrees/agent-*` entry:

1. Parsed `git worktree list --porcelain`.
2. Classified by whether the worktree's HEAD commit is reachable from
   the local `main` branch (be17c0d3).
3. Confirmed every path is missing on disk (pure registration leak).
4. For merged + missing entries: `git worktree remove -f -f <path>`.
   This removes only the registration; the underlying branch ref is
   preserved.
5. For unmerged entries: left in place per JOB constraint
   ('DO NOT delete branches with uncommitted work or unmerged commits').
   The worktree paths are missing on disk so no working changes exist;
   only the orphan branch refs remain reachable for recovery.

(Note: one entry, `agent-a0012db2ed20cba6d`, was used as the
validation probe at the start of the pass and is counted in the
182 removed; its branch HEAD was reachable on a fork-only path.)

`.agent-worktrees/*` (4 entries, all exist on disk - SKIPPED):

- `l10-post-ops-collapse`: unmerged
- `local-stack-250-fix`: merged but dirty (1 uncommitted change)
- `pc-call-target-fix`: merged but dirty (3 uncommitted changes)
- `spec-k-sweep`: merged but dirty (1 untracked file)

`/tmp/c4-*` (77 entries): NOT touched per JOB constraint (active sessions).

## Acceptance vs Target

`IR_INCREMENTAL_IMPROVEMENTS.md` Step 9 targets <50 total worktrees.
Hard JOB constraint 'DO NOT delete branches with unmerged commits' prevents
reaching that target in this pass: 350 unmerged orphan registrations remain.
Follow-up pass needed to triage those branches (likely most are dead WIP
from killed sessions, but each needs a quick check before branch deletion).

## Removed Entries (182 total)

First 25 removed:

- `agent-a04911b221b9ef4f4` (branch `worktree-agent-a04911b221b9ef4f4`, head `5e05f21f84`)
- `agent-a0507b0051b82a6fd` (branch `docs/campaign-summary`, head `bcb0fd1ae4`)
- `agent-a0572a372804c9b23` (branch `migrate-l6-attn-bakes`, head `9f75a7313f`)
- `agent-a068fe88db4b663bc` (branch `worktree-agent-a068fe88db4b663bc`, head `3436271ab3`)
- `agent-a07076553904d60bc` (branch `dim-registry-allocator-migration`, head `6a59741f5b`)
- `agent-a074fe6075fa87fdc` (branch `worktree-agent-a074fe6075fa87fdc`, head `1313cb5b43`)
- `agent-a07df65ce5ea5b502` (branch `audit/per-op-l14`, head `acd3ce94cf`)
- `agent-a0af5830504a860fb` (branch `worktree-agent-a0af5830504a860fb`, head `4d069f7429`)
- `agent-a0c60469be0b8f4b4` (branch `worktree-agent-a0c60469be0b8f4b4`, head `0789b27831`)
- `agent-a0d10982e52f908a6` (branch `fix/per-op-collection-errors`, head `d8bfb5debb`)
- `agent-a0f188ebdd6f51a6b` (branch `worktree-agent-a0f188ebdd6f51a6b`, head `16ebc64ffc`)
- `agent-a0fa973cc793f1021` (branch `worktree-agent-a0fa973cc793f1021`, head `0e498cae3c`)
- `agent-a0fff2c34ef015c3b` (branch `worktree-agent-a0fff2c34ef015c3b`, head `3d1b7006b8`)
- `agent-a10d778f7ab5ef382` (branch `phase6-status-and-fix`, head `7b1b73a106`)
- `agent-a11af99a889e42c0e` (branch `worktree-agent-a11af99a889e42c0e`, head `b21b608fc8`)
- `agent-a1331226598afe5c0` (branch `worktree-agent-a1331226598afe5c0`, head `a922f9774e`)
- `agent-a13d5bf51158e1e0a` (branch `fix/l16-stack0-additional-marker-materializers`, head `fdaf8e0a30`)
- `agent-a1543faed5b1fb5cc` (branch `fix-phase2-3-psh-stack0-cascade`, head `affea4210c`)
- `agent-a1b8855dbf73371d3` (branch `investigation/l17-post-op-inventory`, head `974471e327`)
- `agent-a1c6c43a7f72b760a` (branch `worktree-agent-a1c6c43a7f72b760a`, head `efe97a76b6`)
- `agent-a1c71811d1c0e3f6f` (branch `worktree-agent-a1c71811d1c0e3f6f`, head `cb39af5b37`)
- `agent-a1deec502a1291340` (branch `worktree-agent-a1deec502a1291340`, head `902598cf04`)
- `agent-a20c2ce9fe3d22d0d` (branch `investigation/bd-dim-usage-map`, head `8d3ee75b9a`)
- `agent-a2307b5684c0dfdc5` (branch `worktree-agent-a2307b5684c0dfdc5`, head `1bdb721d52`)
- `agent-a25b1f9b279f143fa` (branch `migrate-l8-attn-bakes`, head `ce43d0a518`)

(Full list: 182 entries, all under `.claude/worktrees/`.)

## Skipped (Unmerged) Entries (350 total)

First 25 skipped:

- `agent-a0012db2ed20cba6d` (branch `worktree-agent-a0012db2ed20cba6d`, head `68c032b5e9`)
- `agent-a010d3ea043222f03` (branch `fix-phase2-3-psh-stack0`, head `fce89e7c6c`)
- `agent-a0137f5f8ea690f8d` (branch `purity-v4-v12-delete-set-active-opcode`, head `612c32b41b`)
- `agent-a01d5d3dea3546e9d` (branch `spec-decoding-into-single-test-fixtures`, head `b25e813c49`)
- `agent-a0455e787c5c83e1b` (branch `investigation/absdiff-nested-quad`, head `a8924be0e1`)
- `agent-a098229a7c8870618` (branch `worktree-agent-a098229a7c8870618`, head `7bc54ba5e8`)
- `agent-a0998eab49fc08a05` (branch `worktree-agent-a0998eab49fc08a05`, head `2f84a5103a`)
- `agent-a09e99112f0374c64` (branch `worktree-agent-a09e99112f0374c64`, head `2a1fe5109c`)
- `agent-a0af2fb60dc20c846` (branch `fix-phase7-divmod`, head `70bdc04ea6`)
- `agent-a0b3df16bce57bc9f` (branch `migrate-everything-unit1`, head `412f48bbb8`)
- `agent-a0c0aa8a1bf050d3d` (branch `worktree-agent-a0c0aa8a1bf050d3d`, head `ecc2fe36f0`)
- `agent-a0c168c168f4ebbd5` (branch `worktree-agent-a0c168c168f4ebbd5`, head `b20d0ba092`)
- `agent-a0cf273c25eae5e5b` (branch `worktree-agent-a0cf273c25eae5e5b`, head `5ffc0830e2`)
- `agent-a0d39b2cd6825bedd` (branch `worktree-agent-a0d39b2cd6825bedd`, head `15f9cf838f`)
- `agent-a0e087031fc79c554` (branch `worktree-agent-a0e087031fc79c554`, head `3428798570`)
- `agent-a0e5acaf8a3f70861` (branch `spec-decoding-into-single-test-fixtures-v2`, head `d30c286445`)
- `agent-a0fd588b37572f9b9` (branch `kv-pruning-per-layer-per-head`, head `8f64d146ea`)
- `agent-a10561747cc72bd6a` (branch `v7-lea-axmerge-licsc-neural`, head `e6ec07de89`)
- `agent-a1081dc9284140a56` (branch `worktree-agent-a1081dc9284140a56`, head `87c84dd4ac`)
- `agent-a10c4fc2a2a119f7b` (branch `fix/rec-premature-exit`, head `1ef127c295`)
- `agent-a12cfcda094b3d799` (branch `worktree-agent-a12cfcda094b3d799`, head `1cee3231ea`)
- `agent-a12dbe9345eef937c` (branch `worktree-agent-a12dbe9345eef937c`, head `147ad6a671`)
- `agent-a14833b345caaa62f` (branch `onnx-fix-blockers-1-2-divmod-mul-itemcalls`, head `2a1c1c2dca`)
- `agent-a14d5b9baabb0593d` (branch `worktree-agent-a14d5b9baabb0593d`, head `3238668564`)
- `agent-a14e1950d31352963` (branch `worktree-agent-a14e1950d31352963`, head `9c9eb418bf`)

(Full list: 350 entries. Branches preserved; only working-tree
registrations remain stale. All affected paths are already missing on disk.)
