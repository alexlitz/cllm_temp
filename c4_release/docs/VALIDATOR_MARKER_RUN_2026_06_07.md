# Validator marker test run — 2026-06-07

Baseline measurement of `@pytest.mark.validator` tests, which are skipped by
default in smoke (per `pyproject.toml:151`, "~60s wall"). These tests are
THE static correctness gate for the declarative IR — they have been
silently passing/failing all session without being tracked.

- **Base ref**: `main` HEAD `77ed77f0` ("cache(sparse): COO sidecar storage")
- **Command**: `pytest tests/test_declarative_verification.py tests/test_teacher_forced_lowering_support.py -m validator --tb=line -q --timeout=300`
- **Wall**: 41.8 s (clean GPU); 60.0 s on contended GPU
- **GPU**: `CUDA_VISIBLE_DEVICES=1` (GPU 0 was OOM-contended by other processes; OOM is an environmental artifact, not a real failure)

## Pass / fail counts

| Bucket | Count |
|---|---|
| Collected (total) | 59 |
| Deselected (non-validator) | 34 |
| Selected | 25 |
| **Passed** | **19** |
| **Failed** | **5** |
| Skipped | 1 (pure-neural runner unavailable) |
| xfailed | 0 |
| xpassed (strict → failure) | counted under Failed |

GPU-contended re-run (GPU 0) reported 7 failed because 2 teacher-forced
probes hit `torch.OutOfMemoryError`; both passed on a clean GPU. Treat
**5 failed** as the real baseline.

## Failure breakdown

### Category 1 — Static claims drift (declaration audit fix)

1. **`TestSpecCoverage::test_spec_file_parses_headings`**
   - Assertion: `len(report.all_sections) >= 1`; actual `0`.
   - Cause: `SpecCoverageReport.all_sections` is empty even though
     `coverage` references `BLOG_SPEC.md#registers`. The spec-file
     parser is no longer extracting any heading anchors.
   - Fix surface: `tests/declarative_oracle.py` (or wherever the
     `SpecCoverageReport.all_sections` is populated) — verify the
     BLOG_SPEC.md heading parser still walks `#`/`##` lines after a
     recent spec-file restructure.

2. **`TestDeclarativeAuthorityAudit::test_no_production_ops_remain_wrappers`**
   - Assertion: `report.legacy_wrapper_ops == []`; actual
     `['layer14_jsr_mem_default_suppress',
       'layer14_mem_addr_src_default_suppress']`.
   - Cause: Two L14 ops are still classified as legacy wrappers under
     the Phase 7 audit. Likely they hold an imperative shim around a
     declarative payload, or their declaration metadata is missing the
     "production" marker.
   - Fix surface: `neural_vm/unified_compiler/ops/l14_ops.py`
     factories for the two named ops, plus the audit classifier in
     the declarative-authority audit module. Confirm they pass through
     `lower_ffn` and not a legacy wrapper.

### Category 2 — Real bake bug (teacher-forced support gap)

3. **`test_teacher_forced_symbolic_byte_survives_final_tail[local_frame_push_mem_addr0_e0]`**
   - Marked `xfail(strict=True)` but now **XPASS** — the bake was
     fixed at some point and the xfail marker was never removed. Easy
     win: delete the `marks=pytest.mark.xfail(...)` wrapper around
     this parametrize entry (lines 56–63 of
     `tests/test_teacher_forced_lowering_support.py`).

4. **`test_teacher_forced_symbolic_byte_survives_final_tail[local_frame_jsr_mem_addr0_e0]`**
   - `IDENTITY_SRC`, step 4, `MEM_addr0` expected `0xE0`, got `0x21`.
   - Final block 35 layer 25, width 192. `OUT_HI[14]` and `OUT_LO[0]`
     are both highly negative; the argmax flipped to LO=1/HI=2 (0x21).
   - Aligns with the known **L10 PSH addr0_e0 missing OP_ENT guard**
     bug in memory (`l10_ops.py:3888-3927` pins `MEM_addr0=0xE0` on
     ENT-main). Mirror failure: this is the JSR variant of the same
     bug.

5. **`test_teacher_forced_symbolic_byte_survives_final_tail[positive_saved_frame_stack0_byte0_e8]`**
   - `LOCAL_FRAME_SRC`, step 4, `STACK0_byte0` expected `0xE8`, got
     `0xE8` (output_byte) but argmax remains `REG_PC` (logit `-10`
     beats expected `-2,908,375`). Bake **DOES** emit 0xE8 in the
     residual lanes but the head argmax is wrong because the
     REG_PC-suppression rule isn't firing on positive-saved-frame
     contexts.
   - Fix surface: REG_PC default-suppression rule for the
     positive-saved-frame condition (likely L14/L10 area, near the
     `layer14_*_default_suppress` ops flagged by test #2).

## Recommended next-session work

1. **Delete the now-spurious xfail** for `local_frame_push_mem_addr0_e0`
   (1-line cleanup, will turn one failure into a pass).
2. **Audit/restore the L14 default-suppress op classification** —
   the same `layer14_*_default_suppress` op names appear in test #2's
   legacy-wrapper list AND test #5's bake-bug surface. Likely a single
   root cause: those ops were partially migrated and are now both
   misclassified and undershooting their REG_PC suppression duty.
3. **Investigate `SpecCoverageReport.all_sections` parser** — empty
   `all_sections` despite non-empty `coverage` is structurally
   suspicious; check whether a BLOG_SPEC.md restructure changed
   heading levels.
4. **JSR variant of L10 PSH `MEM_addr0=0xE0` bug** (test #4) — the
   existing memory note covers the PSH/ENT-main variant; the same
   missing-guard pattern likely needs to be added on the JSR path of
   `l10_ops.py:3888-3927`.

## Constraints honored

- No `git stash`.
- Doc-only commit; no test or source edits.
- Measurement only; no fix attempts.

## Update — 2026-06-07 follow-up fix

- **Test #1 (`TestSpecCoverage::test_spec_file_parses_headings`) FIXED**.
  Root cause: ``audit_spec_coverage`` was using the relative path
  ``c4_release/docs/BLOG_SPEC.md``, which only resolves when pytest is
  invoked from the repo root. When invoked from inside the
  ``c4_release/`` directory, the path becomes
  ``c4_release/c4_release/docs/BLOG_SPEC.md`` (doesn't exist) and the
  ``open()`` silently falls through ``except OSError`` to leave
  ``spec_lines = []``, producing an empty ``all_sections`` even though
  ``coverage`` is non-empty (the op-side scan succeeds regardless).
  Fix: added candidate-path resolution that also tries stripping the
  leading ``c4_release/`` prefix and a fallback derived from the
  module's own location. See ``decl_verifier.py`` ``audit_spec_coverage``.
- **Tests #4 (`local_frame_jsr_mem_addr0_e0`) and #5
  (`positive_saved_frame_stack0_byte0_e8`) REMAIN OPEN** as
  multi-session bake bugs. Diagnostic from this session:
  - Both fail at the L34 ``tail_bit32_result_correction`` post-op.
    Block 33 (layer 24) has the correct argmax for both probes;
    block 34 (the L34 tail) overrides into wrong bytes.
  - For #5 (e8), the byte assembly is correct (``output_byte=0xE8``,
    ``OUT_LO[8]=-292747 ; OUT_HI[14]=-288927`` argmax LO=8, HI=14)
    but the absolute logit (~−581k) loses to ``REG_PC``'s
    head_bake default bias of ``-10``. A clear-output style rule is
    firing on the STACK0_byte0 row driving all 32 OUT_LO/HI lanes
    by ~−707M; a paired byte-emitter rule lifts the correct lane
    back to ~−297k but the byte logit still loses to the
    marker-token default.
  - For #4 (JSR e0), the byte assembly itself flips (block 33 had
    LO[0]/HI[14] dominant; block 34 drives LO[1] / HI[2] to +42M
    and LO[0] / HI[14] to −49M, emitting ``0x21`` instead of ``0xE0``).
    This is the JSR variant of the L10 PSH ``MEM_addr0=0xE0`` bug;
    the L34 tail has rules with ~5e7 strength writing wrong lanes.
  - Both align with the memory note about
    ``tail_bit32_result_correction`` being the dominant 1096 carrier;
    a single-rule whack-a-mole has historically been zero-sum here.
  - Validator pass count after the SpecCoverage fix:
    **22 passed, 2 failed, 1 skipped** (was 19/25 → now 22/25).
