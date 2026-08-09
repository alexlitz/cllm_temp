"""AUDIT + SOUNDNESS PROBE for the speculative byte-exact VM verify path.

Question 1 (AUDIT): is the transformer's COMPUTED step authoritative (accept/reject
on model-vs-draft mismatch), or does the verify TRUST the draft?

Question 2 (BAD SPECULATOR): inject a deliberately WRONG draft (corrupt drafted
register frames / store values / tokens) and check the FINAL decoded output.  Does it
stay BYTE-EXACT (transformer catches the mismatch, rejects) or does it emit the
corrupted draft's wrong result?

Run:  python -m c4_min._agent_spec_soundness_audit
"""
from __future__ import annotations

import copy
import os
from typing import Dict, List, Tuple

# BYTE-BATTERY CONFIG: the compact SP_INIT=0xFC byte battery needs the byte (wide-OFF)
# config — the 069cc32f-family rollback (C4_*=0).  The wide-address DEFAULT (174ece66)
# is the doom full-32-bit-address regime; on the compact byte battery its folded ALU
# diverges from the byte draft at the frame-prologue step (a known config sensitivity,
# NOT a soundness bug).  Set the byte flags BEFORE importing the compiler modules so the
# draft + model both build in the byte config.  This is purely a TEST-config choice; the
# soundness property under test is config-independent.
for _f in ("C4_CMP32", "C4_SHIFT32", "C4_PC_WIDE", "C4_GLOBAL_ADDR32",
           "C4_SP_WIDE", "C4_DIVMOD_SIGNED", "C4_BP_RESTORE_HIBYTE"):
    os.environ.setdefault(_f, "0")

import torch

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa

_SP_INIT = 0xFC
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = _SP_INIT

from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from c4_min.bench_fast_path import (  # noqa
    build_loop_countdown, build_malloc, build_malloc_free, build_nested)
import c4_min.blogspec_vocab as V


_MODEL = None
_L = None
_DEVICE = None


def _device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def _model():
    global _MODEL, _L
    if _MODEL is None:
        _MODEL, _L, _ = build_lib_model_streaming(
            code_size=192, recurrent_divmod=True, addr32=True,
            compute_mode="dense_kernel")
        if _device() != "cpu":
            _MODEL = _MODEL.to(_device())
    return _MODEL, _L


def _programs() -> List[Tuple[str, object]]:
    # a REAL op battery + a DEEP nested loop.
    return [
        # The byte (wide-OFF) config is byte-exact on the pure-arithmetic battery; the
        # malloc heap programs need the addr32 wide heap (0x30008) and are covered by
        # the wide-config doom path, not this compact soundness harness.
        ("op_add", _c("int main(){ int a; a=3; a=a+4; return a; }")),
        ("op_sub", _c("int main(){ int a; a=90; a=a-7; return a; }")),
        ("op_mul", _c("int main(){ int a; a=6; a=a*7; return a; }")),
        ("op_cmp", _c("int main(){ int a; a=5; if(a>3){a=9;} return a; }")),
        ("loop_countdown", build_loop_countdown(10)[0]),
        ("nested_deep", build_nested(4, 8)[0]),          # deep nested loop
    ]


def _c(src: str):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    bc, _data = compile_c(src)
    return bytecode_to_isa(bc)


def _run_verify(model, L, code, draft):
    stats: dict = {}
    out: List[int] = []
    vr = verify_blocks(model, L, code, draft, block_steps=32, device=_device(),
                       evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                       collect_out=out, evict_interval_steps=8)
    return vr, stats, out


def _rebuild_tokens_from_frames(draft):
    """Re-emit draft.tokens[BOS + code + seed + init + per-frame] from the (possibly
    corrupted) draft.frames + store_log, so a corruption of the CONTEXT is consistent
    with what the overlay/token stream would carry.  Mirrors draft_pf_program's own
    token layout: [BOS] + code_frame_toks + seed_frames + init_frame + per-step frames.

    We only need to REBUILD the per-step frame token region.  The init frame and seed
    frames are before win_starts[0]; each step s's frame tokens occupy
    [win_starts[s], win_starts[s] + FRAME_LEN).  A store step carries mem_addr/mem_val
    in the frame; the store_log holds (addr,val) at the *next* frame_idx, but for the
    IN-STREAM token frame we use the frame dict's own stk/mem.
    """
    toks = list(draft.tokens)
    FL = V.FRAME_LEN
    for s, fr in enumerate(draft.frames):
        ws = draft.win_starts[s]
        # build_step_frame emits the register frame; STACK0 rides mem_val slot.
        frame = V.build_step_frame(fr["pc"] & 0xFFFFFFFF, fr["ax"] & 0xFFFFFFFF,
                                   fr["sp"] & 0xFFFFFFFF, fr["bp"] & 0xFFFFFFFF,
                                   mem_addr=0, mem_val=fr.get("stk", 0) & 0xFFFFFFFF)
        # the query row is the LAST token of the frame (STEP_END); the driver's window
        # ends at win_starts[s]; but tokens are laid contiguously.  Overwrite the frame
        # tokens at [ws - (FL-1), ws + 1) — the frame ENDS at the query row ws.
        start = ws - (FL - 1)
        if start < 0:
            continue
        for j in range(FL):
            p = start + j
            if 0 <= p < len(toks):
                toks[p] = frame[j]
    return toks


def audit_perfect(label, code):
    """Baseline: perfect draft -> output byte-exact + all_matched."""
    model, L = _model()
    draft = draft_pf_program(code, max_steps=20000, mask=0xFFFFFFFF)
    assert draft.halted, f"{label}: draft did not halt"
    vr, stats, out = _run_verify(model, L, code, draft)
    return dict(
        label=label, steps=draft.step_count,
        all_matched=vr.all_matched, accepted=vr.accepted_steps,
        final_ax=vr.decoded_final_ax, draft_final=draft.final_ax_masked,
        forwards=vr.forwards, out=list(out))


def _corrupt_frame(draft, step, field, xor_mask):
    """Corrupt draft.frames[step][field] ^= xor_mask (a WRONG drafted register)."""
    draft.frames[step][field] = (draft.frames[step][field] ^ xor_mask) & 0xFFFFFFFF


def audit_bad_speculator(label, code, corruption):
    """Inject a WRONG draft and observe the FINAL decoded output + accept behavior.

    ``corruption`` = dict describing what to corrupt.  Two REGIMES:
      - 'frame_only': corrupt draft.frames (the accept TARGET) only.  Tests the
        accept/reject: the transformer's context tokens are still correct, so the
        model recomputes the CORRECT step and the decode-vs-frames compare must REJECT.
      - 'context': corrupt draft.frames AND rebuild draft.tokens so the transformer is
        fed the WRONG prior-step register value as CONTEXT (the true bad-speculator).
    """
    model, L = _model()
    draft = draft_pf_program(code, max_steps=20000, mask=0xFFFFFFFF)
    assert draft.halted
    good_final = draft.final_ax_masked

    bad = copy.deepcopy(draft)
    tgt_step = corruption["step"] if corruption["step"] >= 0 else \
        max(0, bad.step_count + corruption["step"])
    for field, xm in corruption["fields"].items():
        _corrupt_frame(bad, tgt_step, field, xm)
    regime = corruption.get("regime", "frame_only")
    if regime == "context":
        # rebuild the token stream so the corrupted frame is what the transformer's
        # frozen-context KV actually carries (the real bad-speculator: wrong INPUT).
        bad.tokens = _rebuild_tokens_from_frames(bad)

    vr, stats, out = _run_verify(model, L, code, bad)
    # The DECODED FINAL AX the verify would report: only trustworthy if all_matched.
    # If it rejects (all_matched False), decoded_final_ax is None; the SPEC driver
    # would then re-draft (NOT emit the bad value).  The soundness property is:
    #   the verify NEVER returns all_matched=True with a wrong final AX.
    reported_final = vr.decoded_final_ax
    sound = True
    verdict = ""
    if vr.all_matched:
        # verify accepted the WHOLE bad stream -> it TRUSTED the draft IF the reported
        # final != good_final.  (If it happens to equal good_final the corruption was
        # a no-op / self-correcting; still sound.)
        if reported_final != good_final:
            sound = False
            verdict = "UNSOUND: accepted a bad draft, emitted WRONG final AX"
        else:
            verdict = "accepted (corruption was output-neutral)"
    else:
        verdict = (f"REJECTED at step {vr.first_mismatch['step'] if vr.first_mismatch else '?'}"
                   f" (accepted_prefix={vr.accepted_steps})")
    return dict(
        label=label, regime=regime, corrupt_step=tgt_step,
        fields=list(corruption["fields"].keys()),
        good_final=good_final, all_matched=vr.all_matched,
        reported_final=reported_final, accepted=vr.accepted_steps,
        steps=bad.step_count, forwards=vr.forwards,
        first_mismatch=vr.first_mismatch, sound=sound, verdict=verdict)


def main():
    dev = _device()
    print(f"[env] device={dev}", flush=True)
    model, L = _model()
    print(f"[model] built streaming lib model, n_blocks={len(model.blocks)}", flush=True)
    print("=" * 78, flush=True)
    print("PART 1: PERFECT-DRAFT BASELINE (good draft -> byte-exact + all_matched)")
    print("=" * 78, flush=True)
    base = {}
    n_perfect_ok = 0
    for label, code in _programs():
        r = audit_perfect(label, code)
        base[label] = r
        ok = (r["all_matched"] and r["final_ax"] == r["draft_final"])
        n_perfect_ok += int(ok)
        print(f"  [{label}] steps={r['steps']} all_matched={r['all_matched']} "
              f"final={r['final_ax']} draft={r['draft_final']} forwards={r['forwards']} "
              f"-> {'OK' if ok else 'FAIL'}", flush=True)

    print("=" * 78, flush=True)
    print("PART 2: BAD SPECULATOR — inject WRONG draft, check FINAL output byte-exact")
    print("=" * 78)
    # Corruption battery: flip AX / PC / SP / BP / stk at various steps, in BOTH the
    # frame-only regime (accept target) and the context regime (transformer INPUT).
    corruptions = [
        dict(step=1, fields={"ax": 0x40}, regime="frame_only",
             desc="flip AX bit at step 1 (accept target only)"),
        dict(step=1, fields={"ax": 0x40}, regime="context",
             desc="flip AX bit at step 1 (fed as transformer context)"),
        dict(step=2, fields={"pc": 0x04}, regime="context",
             desc="flip PC at step 2 (wrong control-flow context)"),
        dict(step=3, fields={"sp": 0x08}, regime="context",
             desc="flip SP at step 3 (wrong stack pointer context)"),
        dict(step=3, fields={"bp": 0x10}, regime="context",
             desc="flip BP at step 3 (wrong base pointer context)"),
        dict(step=2, fields={"stk": 0xFF}, regime="context",
             desc="flip STACK0/operand byte at step 2 (wrong operand context)"),
        dict(step=-2, fields={"ax": 0x55}, regime="context",
             desc="flip AX near the LAST step (late corruption)"),
    ]
    n_sound = 0
    n_tests = 0
    # Run the corruption battery on the programs that pass the perfect draft (so a
    # rejection is unambiguously the corruption, not a pre-existing config divergence).
    bad_progs = [(lbl, code) for lbl, code in _programs()
                 if base[lbl]["all_matched"]]
    for label, code in bad_progs:
        for c in corruptions:
            r = audit_bad_speculator(label, code, c)
            n_tests += 1
            n_sound += int(r["sound"])
            print(f"  [{label}] {c['desc']}", flush=True)
            print(f"      all_matched={r['all_matched']} reported_final={r['reported_final']} "
                  f"good_final={r['good_final']} accepted={r['accepted']}/{r['steps']} "
                  f"-> {r['verdict']} {'[SOUND]' if r['sound'] else '[!!! UNSOUND !!!]'}",
                  flush=True)
    print("=" * 78, flush=True)
    print(f"SUMMARY: perfect-draft OK={n_perfect_ok}/{len(_programs())}; "
          f"bad-speculator {n_sound}/{n_tests} cases SOUND "
          f"(never accepted-a-bad-draft-and-emitted-wrong-final).", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
