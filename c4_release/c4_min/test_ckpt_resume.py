#!/usr/bin/env python3
"""test_ckpt_resume.py — byte-exactness of the logical-VM checkpoint/resume (Task #814).

CPU-only, fast (no model build).  Proves that a bounded ``draft_pf_program`` run
checkpointed at step N and RESUMED reproduces the tail of an uninterrupted run
byte-for-byte, and that the checkpoint survives a disk round-trip.  These are the
invariants the model-driving runner (``run_doom_checkpoint``) relies on: the model
decode is a deterministic function of the draft, so byte-identical drafts across the
checkpoint boundary => byte-identical model decodes (proven separately L-inf=0 on GPU).
"""
from __future__ import annotations

import os
import tempfile

import c4_min.nibble_pure_forward_complete as _PFC
_PFC.SP_INIT = 0xFC

from c4_min.pf_speculative import draft_pf_program  # noqa: E402
from c4_min.bench_fast_path import build_nested      # noqa: E402
from c4_min.ckpt_resume import save_checkpoint, load_checkpoint  # noqa: E402

MASK = 0xFFFFFFFF


def _drafts_equal(a, b) -> bool:
    """Full byte-equality of every field a schedule/verify reads off a PFDraft."""
    return (a.step_count == b.step_count
            and a.tokens == b.tokens
            and a.frames == b.frames
            and a.store_log == b.store_log
            and a.win_starts == b.win_starts
            and (a.load_log or {}) == (b.load_log or {})
            and (a.read_log or {}) == (b.read_log or {})
            and (a.out or []) == (b.out or [])
            and (a.prtf_steps or []) == (b.prtf_steps or [])
            and a.final_ax_masked == b.final_ax_masked
            and a.code_off == b.code_off)


def test_resume_tail_byte_exact_in_memory():
    """A checkpoint captured at each split -> a resumed draft == the uninterrupted one."""
    code = build_nested(30, 40)[0]
    full = draft_pf_program(code, max_steps=4000, mask=MASK)
    assert full.step_count == 4000 and not full.halted
    for split in (1, 10, 137, 500, 999, 2500):
        part = draft_pf_program(code, max_steps=split, mask=MASK, capture_state=True)
        assert part.step_count == split
        assert part.resume_state.steps == split
        resumed = draft_pf_program(code, max_steps=4000, mask=MASK,
                                   resume=part.resume_state)
        assert _drafts_equal(resumed, full), f"resume mismatch at split={split}"


def test_checkpoint_disk_roundtrip_byte_exact():
    """save_checkpoint -> load_checkpoint -> resume == uninterrupted (survives a kill)."""
    code = build_nested(40, 60)[0]
    full = draft_pf_program(code, max_steps=3000, mask=MASK)
    split = 1234
    part = draft_pf_program(code, max_steps=split, mask=MASK, capture_state=True)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck.npz")
        sz = save_checkpoint(path, part.resume_state, meta={"prog": "nested_40_60"})
        assert sz > 0 and os.path.exists(path)
        st, meta = load_checkpoint(path)
        assert meta["prog"] == "nested_40_60"
        assert st.steps == split
        resumed = draft_pf_program(code, max_steps=3000, mask=MASK, resume=st)
    assert _drafts_equal(resumed, full)


def test_multi_window_resume_chain():
    """Chained windows (draft -> checkpoint -> resume -> ...) reach the same end as one shot."""
    code = build_nested(50, 90)[0]
    total = 5000
    one_shot = draft_pf_program(code, max_steps=total, mask=MASK)
    rs = None
    done = 0
    last = None
    while done < total:
        target = min(done + 700, total)
        last = draft_pf_program(code, max_steps=target, mask=MASK, resume=rs,
                                capture_state=True)
        done = last.step_count
        rs = last.resume_state
    assert _drafts_equal(last, one_shot)


def test_resume_config_guard():
    """A resume with a mismatched draft-width config MUST assert (never silently diverge)."""
    import pytest
    code = build_nested(20, 30)[0]
    part = draft_pf_program(code, max_steps=200, mask=MASK, capture_state=True)
    st = part.resume_state
    # tamper the recorded IMM_NIBS -> the resume guard must reject it.
    st.imm_nibs = st.imm_nibs + 1
    with pytest.raises(AssertionError):
        draft_pf_program(code, max_steps=400, mask=MASK, resume=st)


def test_default_path_unaffected():
    """capture_state unset -> no resume_state attribute (existing consumers unchanged)."""
    code = build_nested(10, 20)[0]
    d = draft_pf_program(code, max_steps=500, mask=MASK)
    assert not hasattr(d, "resume_state")


if __name__ == "__main__":
    test_resume_tail_byte_exact_in_memory()
    test_checkpoint_disk_roundtrip_byte_exact()
    test_multi_window_resume_chain()
    test_resume_config_guard()
    test_default_path_unaffected()
    print("ALL ckpt_resume tests PASSED")
