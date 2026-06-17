#!/usr/bin/env python3
"""Re-tune wide_mul width=2 weights against the captured 32-case operand vectors.

Goal: fire the true quad for the CLEAN-operand cases the current thr=19.5 just
barely blocks (mul_11 100*68 cond=19.45, mul_31 52*86 cond=19.44) WITHOUT
letting spurious quads fire (which would corrupt the result bands) and WITHOUT
regressing the smoke cases (6*7, 100*5). Cases whose operand-gather is itself
defective (9*98, 89*26: ALU_HI reads wrong nibble) are NOT fixable here and are
reported separately.
"""
import sys, json
sys.path.insert(0, "tools")
import numpy as np
import tune_mul_width2 as T

data = json.load(open("/tmp/mulvec_full.json"))
cases = data["cases"]

# operand-gather-defective cases (ALU_HI reads wrong nibble) — not fixable by weights
GATHER_DEFECT = {(9, 98), (89, 26)}
SMOKE = {(6, 7), (100, 5)}


def eval_weights(w_alu, w_axc, w_mark, w_blk, thr):
    ok = 0
    smoke_ok = True
    fixable_ok = 0
    n_fixable = 0
    fails = []
    for c in cases:
        ol, oh, hl, hh = T.simulate(c, w_alu, w_axc, w_mark, w_blk, thr)
        got = T.decode(ol, oh, hl, hh)
        good = got == c["product"]
        ok += good
        ab = (c["a"], c["b"])
        if ab in SMOKE and not good:
            smoke_ok = False
        if ab not in GATHER_DEFECT:
            n_fixable += 1
            fixable_ok += good
            if not good:
                fails.append((c["a"], c["b"], c["product"], got))
    return ok, fixable_ok, n_fixable, smoke_ok, fails


# current baseline
base = eval_weights(0.6, 6.0, 4.0, 3.0, 19.5)
print(f"BASELINE alu=0.6 axc=6.0 mark=4.0 blk=3.0 thr=19.5: "
      f"all={base[0]}/32 fixable={base[1]}/{base[2]} smoke_ok={base[3]}")
print(f"  fixable fails: {base[4]}")
print()

# Search: keep alu/axc/mark, vary blk and thr finely around current.
best = None
for w_blk in [3.0, 2.5, 2.0, 1.5]:
    for thr in [x * 0.1 for x in range(175, 200)]:  # 17.5..19.9
        ok, fok, nf, smoke_ok, fails = eval_weights(0.6, 6.0, 4.0, w_blk, thr)
        if not smoke_ok:
            continue
        key = (fok, -len(fails))
        if best is None or key > best[0]:
            best = (key, (0.6, 6.0, 4.0, w_blk, thr), ok, fok, nf, fails)

if best:
    _, w, ok, fok, nf, fails = best
    print(f"BEST: alu={w[0]} axc={w[1]} mark={w[2]} blk={w[3]} thr={w[4]:.2f}")
    print(f"  all={ok}/32 fixable={fok}/{nf} fails={fails}")

# Also report: just lowering thr at current blk=3.0
print("\nThreshold sweep at blk=3.0 (smoke-safe only):")
for thr in [19.5, 19.4, 19.3, 19.2, 19.1, 19.0, 18.8, 18.5, 18.0]:
    ok, fok, nf, smoke_ok, fails = eval_weights(0.6, 6.0, 4.0, 3.0, thr)
    print(f"  thr={thr}: all={ok}/32 fixable={fok}/{nf} smoke_ok={smoke_ok} nfails={len(fails)} {[(f[0],f[1]) for f in fails]}")
