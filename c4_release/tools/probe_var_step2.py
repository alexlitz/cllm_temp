#!/usr/bin/env python3
"""Probe var_simple step-2 divergence (spec_k=0, ground truth).

Compiles var_simple_0, builds the DraftVM 35-token reference per step, runs the
spec_k=0 neural probe, and aligns the two streams token-by-token to pinpoint the
exact step-2 divergence (offset, expected vs got token), surfacing the
'37-token desync' the framing-drift doc reports.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.compiler import compile_c
from neural_vm.batched_pure_neural import Token, _step_offset_field, _UNSAFE_OFFSETS
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

# full_trace checks only PC bytes (offsets 0-4) and AX bytes (5-9).
_FULL_TRACE_OFFSETS = frozenset(range(0, 10))
# strict_trace checks safe offsets 0..25 and 34.
_STRICT_SAFE = frozenset(o for o in range(35) if o not in _UNSAFE_OFFSETS)

OPN = {int(getattr(Opcode, n)): n for n in dir(Opcode)
       if not n.startswith("_") and isinstance(getattr(Opcode, n), int)}

SRC = "int main() { int x; x = 990; return x; }"


def reference_steps_tokens(bytecode):
    """DraftVM per-step (pc, ax, sp, bp) + 35-token frame, the strict_trace ref."""
    vm = DraftVM(list(bytecode))
    steps = []
    toks = []
    for _ in range(64):
        if vm.halted:
            break
        if not vm.step():
            break
        steps.append((int(vm.pc) & 0xFFFFFFFF, int(vm.ax) & 0xFFFFFFFF,
                      int(vm.sp) & 0xFFFFFFFF, int(vm.bp) & 0xFFFFFFFF))
        toks.append([int(t) for t in vm.draft_tokens()])
        if vm.halted:
            break
    return steps, toks


def main():
    bytecode, data = compile_c(SRC)
    print("=== var_simple_0:", SRC)
    print("disasm:")
    for i, w in enumerate(bytecode):
        print(f"  idx={i:2d} pc={i*8:3d}: {OPN.get(w & 0xFF, '??'):6s} operand={w >> 8}")

    ref_steps, ref_toks = reference_steps_tokens(bytecode)
    print(f"\nreference: {len(ref_steps)} steps")
    for i, (pc, ax, sp, bp) in enumerate(ref_steps):
        idx = pc // 8
        opn = OPN.get(bytecode[idx] & 0xFF, "?") if 0 <= idx < len(bytecode) else "?"
        print(f"  step{i}: pc={pc}(idx={idx} {opn}) ax={ax} sp={sp:#x} bp={bp:#x}")

    # Build neural probe + run spec_k=0 emission.
    from tools.probe_groundtruth import build_groundtruth_probe
    p = build_groundtruth_probe()

    # Get the FULL emitted context (prompt + emissions), then slice into 35-token steps.
    ctx = p._final_context(bytecode, max_steps=len(ref_steps) + 3)
    prompt_len = len(p._build_context(bytecode))
    emitted = ctx[prompt_len:]
    STEP = int(Token.STEP_TOKENS)
    print(f"\nprompt_len={prompt_len} emitted_tokens={len(emitted)} "
          f"(={len(emitted)/STEP:.2f} x 35-token steps)")

    # Align: walk emitted as fixed 35-token slices vs the reference frames.
    n_neural_steps = len(emitted) // STEP
    print(f"\n--- per-step token alignment (fixed 35-slice vs DraftVM ref) ---")
    first_div = None
    for s in range(max(len(ref_toks), n_neural_steps + 1)):
        nslice = emitted[s * STEP:(s + 1) * STEP]
        rslice = ref_toks[s] if s < len(ref_toks) else None
        if rslice is None:
            print(f"step{s}: NEURAL-ONLY slice {nslice}")
            continue
        if not nslice:
            print(f"step{s}: ref present, NO neural slice (neural ran short)")
            continue
        diffs = []
        safe_diffs = []
        ft_diffs = []
        for off in range(STEP):
            g = nslice[off] if off < len(nslice) else None
            e = rslice[off] if off < len(rslice) else None
            if g != e:
                diffs.append((off, _step_offset_field(off), e, g))
                if off in _STRICT_SAFE:
                    safe_diffs.append((off, _step_offset_field(off), e, g))
                if off in _FULL_TRACE_OFFSETS:
                    ft_diffs.append((off, _step_offset_field(off), e, g))
        tag = "OK" if not diffs else (
            f"DIVERGE all={len(diffs)} SAFE={len(safe_diffs)} FULL_TRACE={len(ft_diffs)}")
        print(f"step{s}: {tag}")
        if diffs:
            if first_div is None and safe_diffs:
                first_div = s
            # show the FULL_TRACE (PC/AX) and other SAFE diffs first
            shown = ft_diffs + [d for d in safe_diffs if d not in ft_diffs]
            for off, name, e, g in shown[:12]:
                kind = "FULL_TRACE" if off in _FULL_TRACE_OFFSETS else "safe"
                print(f"    off={off:2d} {name:12s} expected={e} got={g}  [{kind}]")
            if not safe_diffs:
                print(f"    (only UNSAFE/MEM offsets diverge -> gate IGNORES this step)")

    # Raw token streams around the first divergence so we can see the
    # 37-token (2 spurious) shape.
    if first_div is not None:
        print(f"\n--- RAW token streams around step {first_div} ---")
        lo = first_div * STEP
        win = emitted[max(0, lo - 5):lo + STEP + 10]
        print("neural emitted (window):", win)
        if first_div < len(ref_toks):
            print("ref frame step%d:        " % first_div, ref_toks[first_div])
        # Re-emit a flat compare from the divergence point, NOT pre-sliced,
        # to detect a length desync (a STEP_END landing early/late).
        print("\n--- flat scan from first diverging step (detect desync) ---")
        flat_n = emitted[lo:lo + 2 * STEP + 6]
        # reference flattened from that step onward
        flat_r = []
        for st in range(first_div, len(ref_toks)):
            flat_r.extend(ref_toks[st])
        align = 0
        for k in range(min(len(flat_n), len(flat_r))):
            mark = " " if flat_n[k] == flat_r[k] else " <--DIFF"
            nm = _step_offset_field(k % STEP)
            print(f"  +{k:3d} ({nm:12s}) neural={flat_n[k]:4d} ref={flat_r[k]:4d}{mark}")
            if k > 50:
                break


if __name__ == "__main__":
    main()
