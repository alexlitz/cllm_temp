#!/usr/bin/env python3
"""Offline weight tuner for the width=2 MUL 5-way AND (numpy, FULL lookup).

Simulates the SwiGLU FFN forward of ``wide_mul_rules(width_bytes=2, ...)``
over the REAL probed MARK_AX operand vectors
(/tmp/mul_w2_vectors_clean.json from tools/probe_mul_operand_vectors.py)
and searches (W_alu, W_axc, W_mark, W_blocker, threshold) so EVERY probed
case decodes the correct 16-bit product.

Forward (primitives.swiglu_and_gate, gate = x[OP_MUL], bias 0):
  pre  = S*(cond - thr) ; hid = silu(pre)/S * gate ; out[d] += (2/S)*hid
NO pruning -- iterates the full 16^4 rule grid per case (vectorized).
"""
import json
import sys

import numpy as np

S = 100.0


def silu_over_S(z):
    x = S * z
    return (x / (1.0 + np.exp(-x))) / S


# Precompute the (a_lo,a_hi,b_lo,b_hi) -> product nibble lane index tables.
_AL, _AH, _BL, _BH = np.meshgrid(
    np.arange(16), np.arange(16), np.arange(16), np.arange(16), indexing="ij")
_ABYTE = (_AH << 4) | _AL
_BBYTE = (_BH << 4) | _BL
_PROD = (_ABYTE * _BBYTE) & 0xFFFF
_NIB0 = (_PROD & 0xF).ravel()
_NIB1 = ((_PROD >> 4) & 0xF).ravel()
_NIB2 = ((_PROD >> 8) & 0xF).ravel()
_NIB3 = ((_PROD >> 12) & 0xF).ravel()
_ALf = _AL.ravel()
_AHf = _AH.ravel()
_BLf = _BL.ravel()
_BHf = _BH.ravel()


def simulate(case, w_alu, w_axc, w_mark, w_blk, thr):
    alu_lo = np.array(case["ALU_LO"])
    alu_hi = np.array(case["ALU_HI"])
    axc_lo = np.array(case["AX_CARRY_LO"])
    axc_hi = np.array(case["AX_CARRY_HI"])
    mark = case["MARK_AX"]
    gate = case["OP_MUL"]

    slp = alu_lo[1:].sum()
    shp = alu_hi[1:].sum()
    # blocker per a_lo = slp - alu_lo[a_lo](if a_lo>=1)
    blk_lo_by = slp - np.where(np.arange(16) >= 1, alu_lo, 0.0)
    blk_hi_by = shp - np.where(np.arange(16) >= 1, alu_hi, 0.0)

    cond = (w_mark * mark
            + w_alu * alu_lo[_ALf] + w_alu * alu_hi[_AHf]
            + w_axc * axc_lo[_BLf] + w_axc * axc_hi[_BHf]
            - w_blk * blk_lo_by[_ALf] - w_blk * blk_hi_by[_AHf])
    z = cond - thr
    hid = np.where(z > 0, silu_over_S(z) * gate, 0.0)
    amp = (2.0 / S) * hid

    out_lo = np.bincount(_NIB0, weights=amp, minlength=16)
    out_hi = np.bincount(_NIB1, weights=amp, minlength=16)
    hi_lo = np.bincount(_NIB2, weights=amp, minlength=16)
    hi_hi = np.bincount(_NIB3, weights=amp, minlength=16)
    return out_lo, out_hi, hi_lo, hi_hi


def decode(ol, oh, hl, hh):
    def amx(v):
        m = int(np.argmax(v))
        return m if v[m] > 0 else 0
    return (amx(ol) | (amx(oh) << 4)) | ((amx(hl) | (amx(hh) << 4)) << 8)


def margin(ol, oh, hl, hh):
    """Min decode margin (top - 2nd) across the four nibble bands."""
    def m(v):
        s = np.sort(v)[::-1]
        return s[0] - s[1]
    return min(m(ol), m(oh), m(hl), m(hh))


def main():
    data = json.load(open("/tmp/mul_w2_vectors_clean.json"))
    cases = data["cases"]

    alu_grid = [0.3, 0.4, 0.5, 0.6]
    axc_grid = [6.0, 8.0, 10.0]
    mark_grid = [4.0, 6.0]
    blk_grid = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    thr_grid = [x * 0.5 for x in range(20, 60)]  # 10.0 .. 29.5

    results = []
    for w_alu in alu_grid:
        for w_axc in axc_grid:
            for w_mark in mark_grid:
                for w_blk in blk_grid:
                    for thr in thr_grid:
                        ok = 0
                        min_margin = 1e9
                        smoke_ok = True
                        for c in cases:
                            ol, oh, hl, hh = simulate(
                                c, w_alu, w_axc, w_mark, w_blk, thr)
                            got = decode(ol, oh, hl, hh)
                            good = (got == c["product"])
                            ok += good
                            if (c["a"], c["b"]) in ((6, 7), (100, 5)):
                                if not good:
                                    smoke_ok = False
                                else:
                                    min_margin = min(min_margin,
                                                     margin(ol, oh, hl, hh))
                        results.append((ok, smoke_ok, round(min_margin, 3),
                                        (w_alu, w_axc, w_mark, w_blk,
                                         round(thr, 2))))

    # rank: smoke must pass; then most cases; then widest smoke margin.
    results.sort(key=lambda r: (-int(r[1]), -r[0], -r[2]))
    print(f"=== ranked (smoke_ok, total_ok, smoke_margin, params) ===")
    for ok, sk, mg, params in results[:15]:
        print(f"  smoke={sk} total={ok}/{len(cases)} margin={mg} params={params}")

    # detail for the top smoke-passing config
    top = next((r for r in results if r[1]), None)
    if top:
        print("\n--- detail for top smoke-passing params ---")
        params = top[3]
        print("params", params)
        for c in cases:
            ol, oh, hl, hh = simulate(c, *params)
            got = decode(ol, oh, hl, hh)
            print(f"  {c['a']}x{c['b']}={c['product']} got={got} "
                  f"{'OK' if got == c['product'] else 'XX'}")


if __name__ == "__main__":
    main()
