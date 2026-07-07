#!/usr/bin/env python3
"""Row-signature probe for the 0xE8/0xE0 frame-address slam (PROJECT #430, scope+design).

For a set of corpus program IDs, this probe:

  1. Builds the production efficient model (CPU, isolated cache).
  2. Runs the DraftVM teacher-forced tape + faithful forward (the SAME path as
     ``tools/interp_oracle_gate.py``).
  3. At the register byte-0 PREDICTING position of a given step (where the wrong
     0xE8/0xE0 is emitted), it dumps the ROW SIGNATURE: the value of every dim
     the ``l16_psh_mem_addr0_restore_*`` rules condition on (PSH_AT_SP,
     MARK_MEM, MARK_AX, MARK_PC, MEM_STORE, HAS_SE, IS_BYTE, OP_ENT/OP_JSR
     flags, OUTPUT_LO/HI nibble cells) — so we can see EXACTLY which conditions
     are/aren't satisfied and how the -1e6 marker blockers evaluate on this row.
  4. Computes each candidate rule's actual SwiGLU firing (up, silu(up), gate,
     hidden, per-write contribution) at that position — the runtime firing,
     not the static max-contribution.
  5. Prints the winning writers of the OUTPUT_LO nibble cells that decode byte-0.

This is diagnostics only: no weight changes, model byte-identical to golden.

Usage:
    CUDA_VISIBLE_DEVICES="" C4_VM_CACHE_DIR=/tmp/e8scope_$$ \
        python tools/probe_e8_slam_rowsig.py --ids 360,402,650 --step 0

    # A GENUINE PSH/frame-restore row (0xE0/0xE8 restore is CORRECT):
    CUDA_VISIBLE_DEVICES="" python tools/probe_e8_slam_rowsig.py \
        --ids 550 --step 3 --marker MEM
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from tools.interp_oracle_gate import (  # noqa: E402
    build_gate_context, build_code_prompt, oracle_tape_and_steps,
    _REG_OFFSETS,
)
from neural_vm.verification.faithful_interpreter import (  # noqa: E402
    extract_op_ir, STEP_TOKENS,
)

# Register marker offset for the MEM address byte (35-tok: 26; 30-tok: 21).
_MEM_MARKER_OFF = 21 if STEP_TOKENS == 30 else 26

# The dims the restore rules + byte decode care about. Resolved against the
# BUILT layout (NOT the static registry).
_SIG_DIMS = [
    "PSH_AT_SP", "MEM_STORE", "MEM_ADDR_SRC", "HAS_SE", "IS_BYTE",
    "MARK_MEM", "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0",
    "MARK_SE_ONLY", "OP_ENT", "OP_JSR", "OP_LEV", "OP_IMM", "OP_PSH",
    "OP_GT", "OP_EQ", "OP_LT",
]
_ALIASES = {
    "MARK_MEM": ["MARK_MEM", "MARK_MEMORY"],
    "MARK_SE_ONLY": ["MARK_SE_ONLY", "MARK_SE"],
}


def _resolve(dp, name):
    for cand in _ALIASES.get(name, [name]):
        if cand in dp:
            return dp[cand], cand
    return None, None


def _fire(rule, x, dp, S, decompose=False):
    """Return (up, silu, gate, hidden) for a rule at residual x.

    If ``decompose`` is set, also returns a list of
    ``(term_repr, weight, val, S*weight*val)`` per-condition contributions to
    ``up`` (sorted by |contribution| descending) so we can see EXACTLY which
    condition term dominates the pre-activation."""
    def rd(ref):
        try:
            return ref.resolve(dp)
        except KeyError:
            return -1

    def val(c):
        return float(x[c]) if 0 <= c < x.shape[0] else 0.0

    up = -S * float(rule.threshold)
    terms = []
    if decompose:
        terms.append(("(-S*threshold)", -float(rule.threshold), 1.0,
                      -S * float(rule.threshold)))
    for term in rule.conditions:
        c = S * float(term.weight) * val(rd(term.dim))
        up += c
        if decompose:
            terms.append((repr(getattr(term.dim, "name", term.dim)),
                          float(term.weight), val(rd(term.dim)), c))
    gate = float(rule.gate_bias)
    if rule.gate is not None:
        gate += float(rule.gate_weight) * val(rd(rule.gate))
    for term in rule.gate_terms:
        gate += float(term.weight) * val(rd(term.dim))
    silu = float(torch.nn.functional.silu(torch.tensor(up)).item())
    if decompose:
        terms.sort(key=lambda t: abs(t[3]), reverse=True)
        return up, silu, gate, silu * gate, terms
    return up, silu, gate, silu * gate


def _scan_all(ctx, dp, S, cid, desc, prefix, resid, rule_prefix):
    """Scan EVERY tape row; report each position where a rule matching
    ``rule_prefix`` fires (up>0). Prints the OUTPUT nibble argmaxes + the
    condition-dim signature so GENUINE vs LEAK firings can be compared."""
    olo = dp.get("OUTPUT_LO")
    ohi = dp.get("OUTPUT_HI_THIS_STEP") or dp.get("OUTPUT_HI")
    rules = []
    for op in ctx.flat_ffn_ops:
        ir = extract_op_ir(op, dp, ctx.interp.HD)
        if ir is None:
            continue
        for rule in ir.layer(0).ffn.rules:
            if (rule.name or "").startswith(rule_prefix):
                rules.append(rule)
    print("\n" + "=" * 90)
    print(f"id{cid}: {desc[:60]}  SCAN-ALL  ({len(rules)} rules match "
          f"'{rule_prefix}', {resid.shape[0]} rows)")
    print("-" * 90)
    n_fire = 0
    for pos in range(prefix, resid.shape[0]):
        x = resid[pos]
        best = None
        for rule in rules:
            up, silu, gate, hidden = _fire(rule, x, dp, S)
            if up > 0 and abs(hidden) > 1e-3:
                if best is None or up > best[1]:
                    best = (rule.name, up, silu, gate, hidden)
        if best is None:
            continue
        n_fire += 1
        step = (pos - prefix) // STEP_TOKENS
        off = (pos - prefix) % STEP_TOKENS
        lo_a = (max(range(16), key=lambda k: float(x[olo + k]))
                if olo is not None else -1)
        hi_a = (max(range(16), key=lambda k: float(x[ohi + k]))
                if ohi is not None else -1)
        byte = (hi_a << 4) | lo_a
        lo_mag = float(x[olo + lo_a]) if olo is not None else 0.0
        sig = {nm: float(x[dp[c]]) for nm in ("MARK_AX", "MARK_MEM",
               "MARK_PC", "PSH_AT_SP", "MEM_STORE", "HAS_SE", "IS_BYTE")
               for c in [_resolve(dp, nm)[1]] if c}
        sigs = " ".join(f"{k}={v:+.2f}" for k, v in sig.items()
                        if abs(v) > 0.05)
        print(f"  pos={pos} step={step} off={off:2d}  byte=0x{byte:02x} "
              f"(lo={lo_a} hi={hi_a} lo_mag={lo_mag:+.1f})  "
              f"up={best[1]:+.3e}  {best[0]}")
        print(f"        sig: {sigs}")
    if n_fire == 0:
        print("  (no restore-rule firing anywhere in this program's tape)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="360,402", help="comma corpus ids")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--marker", default="AX",
                    help="AX/PC/SP/BP/STACK0 or MEM (the store-address marker)")
    ap.add_argument("--byte", type=int, default=0)
    ap.add_argument("--rule-prefix", default="l16_psh_mem_addr0",
                    help="only dump rules whose name starts with this")
    ap.add_argument("--decompose", default="",
                    help="rule name to decompose term-by-term (up = sum of "
                         "S*weight*val); shows which condition dominates")
    ap.add_argument("--smoke-mem", type=int, default=-1,
                    help="use _MEM_TESTS[N] smoke bytecode instead of corpus "
                         "ids (0=si_li_roundtrip PSH-store; the GENUINE restore)")
    ap.add_argument("--src", default="",
                    help="raw C source (overrides --ids/--smoke-mem)")
    ap.add_argument("--scan-all", action="store_true",
                    help="scan EVERY tape row; report each position where a "
                         "'--rule-prefix' rule fires (up>0), its OUTPUT nibble "
                         "argmaxes, and the row signature. Finds GENUINE restore "
                         "firings without knowing the store-token offset a priori.")
    args = ap.parse_args()

    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    from src.compiler import compile_c

    if args.src:
        tests = None
        smoke_prog = (compile_c(args.src)[0], b"", f"src:{args.src[:40]}")
        ids = [0]
    elif args.smoke_mem >= 0:
        from tests.test_smoke import _MEM_TESTS
        T = _MEM_TESTS[args.smoke_mem]
        tests = None
        smoke_prog = (T["bytecode"], b"", T["name"])
        ids = [args.smoke_mem]
    else:
        from tests.test_suite_1000 import generate_test_programs
        tests = generate_test_programs()
        smoke_prog = None

    print("[probe] building production model (CPU)...", file=sys.stderr, flush=True)
    ctx = build_gate_context(verbose=False)
    dp = ctx.dim_positions
    S = ctx.interp.S

    print(f"\n[probe] STEP_TOKENS={STEP_TOKENS} d_model={ctx.model.d_model} "
          f"blocks={len(ctx.model.blocks)} MEM_marker_off={_MEM_MARKER_OFF}")

    if args.marker == "MEM":
        moff = _MEM_MARKER_OFF
    else:
        moff = _REG_OFFSETS[args.marker]

    for cid in ids:
        if smoke_prog is not None:
            bc, data, desc = smoke_prog
        else:
            src, expected, desc = tests[cid]
            bc, data = compile_c(src)
        ot = oracle_tape_and_steps(bc, data, max_steps=48)
        prompt = build_code_prompt(bc, data)
        prefix = len(prompt)
        full_ctx = prompt + ot.draft_tokens
        resid = ctx.fwd._residual_pre_head(full_ctx)

        if args.scan_all:
            _scan_all(ctx, dp, S, cid, desc, prefix, resid,
                      args.rule_prefix)
            continue

        pred_pos = prefix + args.step * STEP_TOKENS + moff + args.byte
        x = resid[pred_pos]

        o_pc, o_ax = ot.steps[args.step] if args.step < len(ot.steps) else (-1, -1)
        opc = ot.opcodes[args.step] if args.step < len(ot.opcodes) else -1

        print("\n" + "=" * 90)
        print(f"id{cid}: {desc[:60]}")
        print(f"  step={args.step} opcode={opc} marker={args.marker}[{args.byte}] "
              f"moff={moff} pred_pos={pred_pos} (prefix={prefix})")
        print(f"  oracle step: pc={o_pc} ax={o_ax}")
        print("-" * 90)
        print("  ROW SIGNATURE (dim value at the predicting position):")
        for nm in _SIG_DIMS:
            col, resolved = _resolve(dp, nm)
            if col is None:
                print(f"    {nm:16s} = <not in layout>")
                continue
            print(f"    {nm:16s} = {float(x[col]):+12.4f}   (col {col} as {resolved})")

        print("-" * 90)
        olo = dp.get("OUTPUT_LO")
        ohi = dp.get("OUTPUT_HI_THIS_STEP") or dp.get("OUTPUT_HI")
        if olo is not None:
            cells = [float(x[olo + k]) for k in range(16)]
            amax = max(range(16), key=lambda k: cells[k])
            print(f"  OUTPUT_LO nibble cells (base {olo}): argmax={amax} "
                  f"vals={[round(c, 1) for c in cells]}")
        if ohi is not None:
            cells = [float(x[ohi + k]) for k in range(16)]
            amax = max(range(16), key=lambda k: cells[k])
            print(f"  OUTPUT_HI  nibble cells (base {ohi}): argmax={amax} "
                  f"vals={[round(c, 1) for c in cells]}")

        print("-" * 90)
        print(f"  RUNTIME FIRING of rules matching '{args.rule_prefix}':")
        fired = []
        for op in ctx.flat_ffn_ops:
            ir = extract_op_ir(op, dp, ctx.interp.HD)
            if ir is None:
                continue
            op_name = getattr(op, "name", "<anon>")
            for rule in ir.layer(0).ffn.rules:
                rn = rule.name or ""
                if not rn.startswith(args.rule_prefix):
                    continue
                up, silu, gate, hidden = _fire(rule, x, dp, S)
                if abs(hidden) < 1e-6:
                    continue
                for w in rule.writes:
                    try:
                        wc = w.dim.resolve(dp)
                    except KeyError:
                        continue
                    contrib = hidden * float(w.weight)
                    if abs(contrib) > 1e-3:
                        fired.append((abs(contrib), op_name, rn, up, silu, gate,
                                      hidden, wc, float(w.weight), contrib))
        fired.sort(reverse=True)
        if not fired:
            print("    (no rule with this prefix has nonzero runtime firing here)")
        for (_, op_name, rn, up, silu, gate, hidden, wc, ww, contrib) in fired[:14]:
            if olo is not None and olo <= wc < olo + 16:
                cellinfo = f"OUTPUT_LO+{wc - olo}"
            elif ohi is not None and ohi <= wc < ohi + 16:
                cellinfo = f"OUTPUT_HI+{wc - ohi}"
            else:
                cellinfo = f"col{wc}"
            print(f"    {rn:42s} up={up:+9.2f} silu={silu:+8.2f} "
                  f"gate={gate:+7.3f} hidden={hidden:+10.2f} "
                  f"-> {cellinfo} w={ww:+.4f} contrib={contrib:+.3e}")

        if args.decompose:
            for op in ctx.flat_ffn_ops:
                ir = extract_op_ir(op, dp, ctx.interp.HD)
                if ir is None:
                    continue
                for rule in ir.layer(0).ffn.rules:
                    if (rule.name or "") != args.decompose:
                        continue
                    up, silu, gate, hidden, terms = _fire(
                        rule, x, dp, S, decompose=True)
                    print("-" * 90)
                    print(f"  DECOMPOSE '{args.decompose}': "
                          f"up={up:+.3e} silu={silu:+.3e} gate={gate:+.3f} "
                          f"(fires iff up>0). Per-term S*w*val (|desc|):")
                    for (nm, w, v, c) in terms:
                        print(f"      {nm:34s} w={w:+12.1f} val={v:+10.4f} "
                              f"S*w*val={c:+.4e}")


if __name__ == "__main__":
    main()
