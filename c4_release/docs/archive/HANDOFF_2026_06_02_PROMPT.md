# Prompt for next Claude Code session

Copy/paste this to start the next session:

---

I'm continuing work on the C4-release neural-VM compiler. The prior session completed
the structural compiler vision (V1-V5 + G6 + G7=SCC=0 all green per verifier round 6)
but hit two runtime correctness gaps that need fixing. Full context is in
`c4_release/docs/HANDOFF_2026_06_02.md` — read it first.

The two open items:

1. **1096 pass rate = 0/137 at HEAD** (was 120/411 = 29.2% in Phase 6 baseline commit
   c7da668). The model compiles + bakes + runs forward but emits `neural=None` on
   every prediction. Find the regression via `git bisect`.

2. **39 smoke ASSERTs** at HEAD. Bisect identified `ae64239b` as the root cause
   (L13 anchor regression). A "proper fix" was attempted at `ff454052`. Per
   verifier round 6, smoke is still 12 pass / 39 fail — verify whether the fix
   actually placed `l13_alu_shift_install` at L13, and either deepen the fix or
   investigate elsewhere.

Suggested workflow:

1. Read the handoff doc completely.
2. Start with the 1096 bisect (higher impact). Use an isolated git worktree per the
   doc's instructions. Single canary test, `git bisect run`, find the first bad
   commit.
3. Apply a targeted fix (likely a revert + reapply with the missing piece, NOT a
   single-rule corrective op — per the zero-sum memory note).
4. Verify 1096 recovers, then verify smoke 39 fixes itself (likely same root
   cause).
5. If time, the declarative architectural-toggle IR types (`PositionalEncodingSpec`,
   `AttentionActivationSpec`, `NormSpec`, `FFNActivationSpec`) are designed but not
   implemented — see Section "Architectural toggles" in the handoff.

Constraints (from memory):
- Don't use `git stash` (worktrees should hold their own state)
- Don't use `gh` CLI (sandbox-blocked)
- Don't stack corrective ops on broken upstream rules — find root causes (
  `feedback_single_rule_fixes_are_zero_sum.md`)
- Cap parallel agents at 2-3 to avoid the harness-instability pattern the prior
  session hit

First steps:
1. `cat c4_release/docs/HANDOFF_2026_06_02.md` (full context)
2. `git log --oneline -10` (latest commits)
3. Decide whether to start with 1096 bisect or smoke verification

Memory notes that matter:
- `feedback_single_rule_fixes_are_zero_sum.md`
- `project_var_failure_mode_shifted.md`
- `project_1096_sentinel_baseline.md`
- `feedback_agent_briefs.md`

Begin.
