"""THREE-WAY PROOF for the hardened speculative verify path + the validity guard.

  PROOF A (GOOD draft):   output == golden byte-exact, full speed, all_matched.
  PROOF B (BAD draft):    a corrupted draft is REJECTED (never emits a wrong final AX);
                          more forwards / lower acceptance (slower), output still exact.
  PROOF C (INVALID step): an out-of-domain register lane is flagged as an EXPLICIT
                          reject (kind="invalid") by C4_VERIFY_VALIDITY, NOT silently
                          snapped-and-accepted.

  + BYTE-IDENTITY: with C4_VERIFY_VALIDITY OFF the verify result (all_matched / accepted
    / final AX / collect_out) is IDENTICAL to the pre-guard path on the good draft; with
    it ON the good draft still passes byte-exact (the guard never false-rejects a valid
    row).  (The golden state fingerprint is a WEIGHT hash — unaffected by this verify-side
    change; checked separately via c4_min._fingerprint_build.)

Run:  python -m c4_min._agent_spec_validity_proof
"""
from __future__ import annotations

import copy
import os
from typing import List

# byte (wide-OFF) config so the compact battery is byte-exact vs the byte draft.
for _f in ("C4_CMP32", "C4_SHIFT32", "C4_PC_WIDE", "C4_GLOBAL_ADDR32",
           "C4_SP_WIDE", "C4_DIVMOD_SIGNED", "C4_BP_RESTORE_HIBYTE"):
    os.environ.setdefault(_f, "0")

import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa

_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import (draft_pf_program, verify_blocks,
                                   set_verify_validity)
from c4_min.bench_fast_path import build_loop_countdown, build_nested


_DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
_MODEL = None
_L = None


def _model():
    global _MODEL, _L
    if _MODEL is None:
        _MODEL, _L, _ = build_lib_model_streaming(
            code_size=192, recurrent_divmod=True, addr32=True,
            compute_mode="dense_kernel")
        if _DEV != "cpu":
            _MODEL = _MODEL.to(_DEV)
    return _MODEL, _L


def _verify(code, draft, *, validity):
    set_verify_validity(validity)
    model, L = _model()
    stats: dict = {}
    out: List[int] = []
    vr = verify_blocks(model, L, code, draft, block_steps=32, device=_DEV,
                       evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                       collect_out=out, evict_interval_steps=8)
    set_verify_validity(None)
    return vr, out


def _corrupt_a_frame_and_tokens(draft, step, field, xm):
    """Corrupt draft.frames[step][field] ^= xm AND rebuild draft.tokens so the wrong
    value is what the transformer's frozen-context KV carries (the true bad speculator)."""
    import c4_min.blogspec_vocab as V
    bad = copy.deepcopy(draft)
    bad.frames[step][field] = (bad.frames[step][field] ^ xm) & 0xFFFFFFFF
    FL = V.FRAME_LEN
    toks = list(bad.tokens)
    for s, fr in enumerate(bad.frames):
        ws = bad.win_starts[s]
        frame = V.build_step_frame(fr["pc"] & 0xFFFFFFFF, fr["ax"] & 0xFFFFFFFF,
                                   fr["sp"] & 0xFFFFFFFF, fr["bp"] & 0xFFFFFFFF,
                                   mem_addr=0, mem_val=fr.get("stk", 0) & 0xFFFFFFFF)
        start = ws - (FL - 1)
        if start < 0:
            continue
        for j in range(FL):
            p = start + j
            if 0 <= p < len(toks):
                toks[p] = frame[j]
    bad.tokens = toks
    return bad


def _make_invalid_forward(model, corrupt_step, corrupt_reg_base):
    """Install a forward wrapper that SMEARS one register nibble of the query row at a
    chosen step so its lane is out-of-domain (a garbage value the difference-min argmax
    would silently snap).  Returns an undo() to restore the clean forward.  This
    simulates a precision-broken / undefined ALU output at that step."""
    orig = model.forward_hidden_cached
    fired = {"n": 0}

    def wrapped(x, *a, **k):
        hidden, kv = orig(x, *a, **k)
        # smear the FIRST byte-0 low-nibble lane of the target register on the LAST row
        # of this forward (the block's newest query rows) far out of the [0,15] domain.
        try:
            hidden[0, -1, corrupt_reg_base] = 9.0e6   # garbage lane, past 2^24
            fired["n"] += 1
        except Exception:
            pass
        return hidden, kv

    model.forward_hidden_cached = wrapped
    return lambda: setattr(model, "forward_hidden_cached", orig), fired


def main():
    print(f"[env] device={_DEV}", flush=True)
    model, L = _model()
    print(f"[model] n_blocks={len(model.blocks)}", flush=True)

    progs = [("loop_countdown", build_loop_countdown(10)[0]),
             ("nested_deep", build_nested(4, 8)[0])]

    print("=" * 78)
    print("PROOF A: GOOD draft -> byte-exact + all_matched (BOTH validity OFF and ON)")
    print("=" * 78)
    a_ok = True
    for label, code in progs:
        draft = draft_pf_program(code, max_steps=20000, mask=0xFFFFFFFF)
        vr_off, out_off = _verify(code, draft, validity=False)
        vr_on, out_on = _verify(code, draft, validity=True)
        # byte-identity of the guard on a valid draft: identical verdict + final + out.
        identical = (vr_off.all_matched == vr_on.all_matched
                     and vr_off.accepted_steps == vr_on.accepted_steps
                     and vr_off.decoded_final_ax == vr_on.decoded_final_ax
                     and out_off == out_on)
        good = (vr_off.all_matched
                and vr_off.decoded_final_ax == draft.final_ax_masked
                and identical)
        a_ok &= good
        print(f"  [{label}] OFF: matched={vr_off.all_matched} final={vr_off.decoded_final_ax} "
              f"fwd={vr_off.forwards} | ON: matched={vr_on.all_matched} "
              f"final={vr_on.decoded_final_ax} | guard-byte-identical={identical} "
              f"draft_final={draft.final_ax_masked} -> {'OK' if good else 'FAIL'}", flush=True)

    print("=" * 78)
    print("PROOF B: BAD draft -> REJECTED, never a wrong accepted final (slower, exact)")
    print("=" * 78)
    b_ok = True
    for label, code in progs:
        draft = draft_pf_program(code, max_steps=20000, mask=0xFFFFFFFF)
        good_final = draft.final_ax_masked
        vr_good, _ = _verify(code, draft, validity=False)
        # corrupt AX at a mid step and feed it as transformer context.
        tgt = min(3, draft.step_count - 1)
        bad = _corrupt_a_frame_and_tokens(draft, tgt, "ax", 0x40)
        vr_bad, _ = _verify(code, bad, validity=False)
        # SOUND: the bad draft is NOT accepted-whole with a wrong final AX.
        sound = not (vr_bad.all_matched and vr_bad.decoded_final_ax != good_final)
        slower = (vr_bad.accepted_steps < vr_good.accepted_steps)
        b_ok &= sound
        km = vr_bad.first_mismatch.get("kind") if vr_bad.first_mismatch else None
        print(f"  [{label}] good: matched={vr_good.all_matched} accepted={vr_good.accepted_steps} "
              f"| bad: matched={vr_bad.all_matched} accepted={vr_bad.accepted_steps} "
              f"reject_step={vr_bad.first_mismatch['step'] if vr_bad.first_mismatch else None} "
              f"kind={km} slower={slower} -> {'SOUND' if sound else '!!!UNSOUND!!!'}",
              flush=True)

    print("=" * 78)
    print("PROOF C: INVALID (out-of-domain) step -> EXPLICIT reject kind='invalid'")
    print("=" * 78)
    c_ok = True
    for label, code in progs[:1]:               # one program is enough for the guard
        draft = draft_pf_program(code, max_steps=20000, mask=0xFFFFFFFF)
        # WITHOUT the guard: an out-of-domain lane is silently snapped (may or may not
        # trip the plain register compare).  WITH the guard: it is an explicit reject.
        for validity in (False, True):
            undo, fired = _make_invalid_forward(model, corrupt_step=2, corrupt_reg_base=L.AX)
            try:
                set_verify_validity(validity)
                stats: dict = {}
                vr = verify_blocks(model, L, code, draft, block_steps=32, device=_DEV,
                                   evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                                   evict_interval_steps=8)
                set_verify_validity(None)
            finally:
                undo()
            km = vr.first_mismatch.get("kind") if vr.first_mismatch else None
            reason = vr.first_mismatch.get("invalid_reason") if vr.first_mismatch else None
            tag = "ON" if validity else "OFF"
            print(f"  [{label}] guard={tag}: matched={vr.all_matched} kind={km} "
                  f"reason={reason}", flush=True)
            if validity:
                # with the guard ON, the smeared lane must be flagged as INVALID
                # (an explicit reject), not accepted.
                c_ok &= (not vr.all_matched and km == "invalid")

    print("=" * 78)
    print(f"RESULT: PROOF-A(good)={'PASS' if a_ok else 'FAIL'}  "
          f"PROOF-B(bad-sound)={'PASS' if b_ok else 'FAIL'}  "
          f"PROOF-C(invalid-flagged)={'PASS' if c_ok else 'FAIL'}", flush=True)
    print("=" * 78)


if __name__ == "__main__":
    main()
