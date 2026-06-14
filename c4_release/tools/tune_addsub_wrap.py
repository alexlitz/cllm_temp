#!/usr/bin/env python3
"""Offline weight tuner for the declarative ADD/SUB byte-0 wrap (numpy).

Simulates the SwiGLU AND-rule forward of ``wide_add_rules`` /
``wide_sub_rules`` (width_bytes=2 = two nibble lanes of byte 0, with the
inter-nibble carry chain) over the REAL probed MARK_AX operand vectors
(/tmp/addsub_vectors.json from tools/probe_addsub_operand_vectors.py).

The wrap output OUTPUT_LO/HI is summed with the downstream block-11 L9 leak
floor (a ~uniform ~83 with a +12.6 spike at cell 0 and +2 at cell 8) so the
search optimizes the FINAL argmax that the model decodes, not just the wrap's
local band. Searches (w_alu, w_axc, w_mark, w_blk, threshold, write_amp) so
EVERY probed byte-0 decodes correctly with a clear margin.

Forward (swiglu_and_gate, gate=OP_*, bias 0):
  pre = S*(cond - thr); hid = silu(pre)/S * gate; out[d] += (write_amp/S)*hid
"""
import json
import sys
import itertools

import numpy as np

S = 100.0

# Downstream L9 (block-11) OUTPUT_LO leak floor, measured spec_k=0
# (tools/probe_addsub_grid.py mag): uniform ~83.57 with +12.59 at cell 0 and
# +2.04 at cell 8. OUTPUT_HI sees NO leak (stays at the wrap's write).
LEAK_LO = np.full(16, 83.57)
LEAK_LO[0] += 12.59
LEAK_LO[8] += 2.04
LEAK_HI = np.zeros(16)


def silu_over_S(z):
    x = S * z
    return (x / (1.0 + np.exp(-x))) / S


def _lane_out(a_band, b_band, carry_in_band, w_alu, w_axc, w_mark,
              w_blk, thr, amp, gate, mark, *, sub):
    """Simulate one nibble lane's 256 (a,b) AND-rules -> (result16, carry).

    carry_in_band: None for lo lane (byte 0 nibble 0 has no carry-in), else
    a scalar carry-in indicator (the lo lane's carry-out magnitude).
    Returns (result_write[16], carry_out_scalar).
    """
    out = np.zeros(16)
    carry_out = 0.0
    a = np.asarray(a_band)
    b = np.asarray(b_band)
    a_nonzero_sum = a[1:].sum()
    for a_nib in range(16):
        # blocker: -w_blk * (sum of OTHER nonzero a cells)
        blk_term = -w_blk * (a_nonzero_sum - (a[a_nib] if a_nib >= 1 else 0.0))
        for b_nib in range(16):
            if sub:
                raw = a_nib - b_nib
            else:
                raw = a_nib + b_nib
            cond = (w_mark * mark + w_alu * a[a_nib]
                    + w_axc * b[b_nib] + blk_term)
            # carry-in handling for the hi lane
            if carry_in_band is not None:
                # cin=0 rule: suppressed when carry present; cin=1 requires it.
                # We approximate by running BOTH and letting the carry magnitude
                # pick. Simpler: fold carry into raw if carry-in present > 0.5.
                cin = 1 if carry_in_band > 0.5 else 0
                if sub:
                    raw = a_nib - b_nib - cin
                else:
                    raw = a_nib + b_nib + cin
            res_nib = raw & 0xF if sub else raw % 16
            cout = 1.0 if (raw < 0 if sub else raw >= 16) else 0.0
            z = cond - thr
            if z > 0:
                hid = silu_over_S(z) * gate
                w = (amp / S) * hid
                out[res_nib] += w
                if cout:
                    carry_out += w
    return out, carry_out


def simulate(case, w_alu, w_axc, w_mark, w_blk, thr, amp):
    sub = case["op"] == "SUB"
    mark = case["MARK_AX"]
    gate = case["OP_SUB"] if sub else case["OP_ADD"]
    # lo lane: nibble 0 (ALU_LO / AX_CARRY_LO)
    lo_out, lo_carry = _lane_out(
        case["ALU_LO"], case["AX_CARRY_LO"], None,
        w_alu, w_axc, w_mark, w_blk, thr, amp, gate, mark, sub=sub)
    # hi lane: nibble 1 (ALU_HI / AX_CARRY_HI) with carry-in from lo lane
    hi_out, _ = _lane_out(
        case["ALU_HI"], case["AX_CARRY_HI"], lo_carry,
        w_alu, w_axc, w_mark, w_blk, thr, amp, gate, mark, sub=sub)
    final_lo = lo_out + LEAK_LO
    final_hi = hi_out + LEAK_HI
    lo_nib = int(np.argmax(final_lo))
    hi_nib = int(np.argmax(final_hi))
    byte0 = lo_nib | (hi_nib << 4)
    # margin = min (top - 2nd) across the two leaked bands
    def m(v):
        s = np.sort(v)[::-1]
        return s[0] - s[1]
    return byte0, min(m(final_lo), m(final_hi))


def main():
    data = json.load(open("/tmp/addsub_vectors.json"))
    cases = data["cases"]
    best = None
    # Grid search. A is dirty (~6 + index0 ~5.4), B clean (~1.0-1.3).
    for w_alu in [3.0, 4.0, 5.0, 6.0]:
        for w_axc in [20.0, 30.0, 40.0]:
            for w_mark in [40.0]:
                for w_blk in [2.0, 3.0, 4.0, 5.0, 6.0]:
                    for thr in [70.0, 80.0, 90.0]:
                        for amp in [20.0]:
                            ok = 0
                            min_margin = 1e9
                            for c in cases:
                                want_b0 = c["want"] & 0xFF
                                got_b0, marg = simulate(
                                    c, w_alu, w_axc, w_mark, w_blk, thr, amp)
                                if got_b0 == want_b0:
                                    ok += 1
                                    min_margin = min(min_margin, marg)
                            key = (ok, min_margin if ok == len(cases) else -1)
                            if best is None or key > best[0]:
                                best = (key, (w_alu, w_axc, w_mark, w_blk,
                                              thr, amp))
                                print(f"ok={ok}/{len(cases)} "
                                      f"margin={min_margin:.2f} "
                                      f"alu={w_alu} axc={w_axc} blk={w_blk} "
                                      f"thr={thr} amp={amp}", flush=True)
    print("BEST", best)
    # Per-case detail for the best.
    _, (wa, wx, wm, wb, th, am) = best
    for c in cases:
        gb, mg = simulate(c, wa, wx, wm, wb, th, am)
        wb0 = c["want"] & 0xFF
        print(f"  id{c['id']:3d} {c['op']} want0x{wb0:02x} got0x{gb:02x} "
              f"{'OK' if gb == wb0 else 'X'} margin={mg:.2f}")


if __name__ == "__main__":
    main()
