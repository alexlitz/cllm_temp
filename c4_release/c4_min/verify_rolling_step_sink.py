"""Byte-exact verification of the ROLLING PER-STEP SINK (BLOG_SPEC §Memory).

Runs the full evidence battery, all MEASURED against the real blogspec_model
§Memory CAM head (plain softmax + ALiBi, NO softmax1 / NO exemption for the
rolling variant), and asserts every N/N verdict:

  1. Shallow correctness (all 3 sink modes): store/load/ZFOD byte-exact.
  2. FULL == MINIMAL stream equivalence (the memory-safe accelerator is faithful).
  3. LONG-RANGE HEAD-TO-HEAD across program depth (the decisive test):
       global-BOS BREAKS at depth (negative control), rolling + softmax1 HOLD.
       Verified at accelerated EFF=2000 (real full forward) AND default
       EFF=500000 (minimal stream, memory-safe) — same break at pos_q > EFF/slope.
  4. BOTH roles: intra-step gather (<=W_step) + cross-step long-range recall,
     one rolling sink, C_sink=0, byte-exact.
  5. READ HORIZON: rolling preserves (marginally extends) softmax1's horizon.
  6. CALIBRATION window: C_sink in (-EFF+slope*W, EFF-slope*(gap-W)); 0 is valid.

Run:  PYTHONPATH=. python c4_min/verify_rolling_step_sink.py
"""
from __future__ import annotations

import resource

from c4_min.rolling_step_sink_mem import RollingSinkMemory

W = 30
MODES = ("softmax1", "global_bos", "rolling")


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def full_probe(mode, depth, eff, slope, known=0x200, kv=0xABCD):
    m = RollingSinkMemory(sink_mode=mode, eff=eff, slope=slope)
    m.store(known, kv)
    a = 0x10000
    for _ in range(max(0, depth - 2)):
        m.store(a, 0x11); a += 0x40
    if depth >= 2:
        m.store(known, kv)
    return m.load(known), m.load(known ^ (1 << 8))


def minimal_probe(mode, depth, eff, slope, known=0x200, kv=0xABCD):
    m = RollingSinkMemory(sink_mode=mode, eff=eff, slope=slope)
    m._rows = []
    if mode == "global_bos":
        m._rows.append((0, "sink", m._sink_overlay()))
    m._rows.append((20, "store", m._store_overlay(known, kv, False)))
    m._rows.append(((depth - 1) * W + 20, "store", m._store_overlay(known, kv, False)))
    if mode == "rolling":
        m._rows.append(((depth - 1) * W, "sink", m._sink_overlay()))
    m._step = depth
    rk = m.load(known)
    m._step = depth
    rz = m.load(known ^ (1 << 8))
    return rk, rz


def deep_intra(mode, depth, eff=500000.0, slope=1.0):
    m = RollingSinkMemory(sink_mode=mode, eff=eff, slope=slope)
    m._rows = []
    if mode == "global_bos":
        m._rows.append((0, "sink", m._sink_overlay()))
    step_start = depth * W
    if mode == "rolling":
        m._rows.append((step_start, "sink", m._sink_overlay()))
    m._rows.append((step_start + 3, "store", m._store_overlay(0x300, 0x7B, False)))
    m._step = depth + 1
    got = m.load_at(0x300, step_start + 8)
    zfod = m.load_at(0x304, step_start + 9)
    return got, zfod


def read_at_gap(mode, gap_tokens, eff, slope, known=0x200, kv=0xABCD):
    m = RollingSinkMemory(sink_mode=mode, eff=eff, slope=slope)
    m._rows = []
    store_pos = 20
    if mode == "global_bos":
        m._rows.append((0, "sink", m._sink_overlay()))
    m._rows.append((store_pos, "store", m._store_overlay(known, kv, False)))
    q_pos = store_pos + gap_tokens
    step_start = (q_pos // W) * W
    if mode == "rolling":
        m._rows.append((step_start, "sink", m._sink_overlay()))
    m._step = q_pos // W + 1
    return m.load_at(known, q_pos)


def find_horizon(mode, eff, slope):
    kv = 0xABCD
    lo, hi = 1, int(eff / slope) + 200
    while read_at_gap(mode, hi, eff, slope) == kv:
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if read_at_gap(mode, mid, eff, slope) == kv:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# battery
# ---------------------------------------------------------------------------
def run():
    npass = ntot = 0

    def check(cond, label):
        nonlocal npass, ntot
        ntot += 1
        npass += 1 if cond else 0
        print(f"   [{'PASS' if cond else 'FAIL'}] {label}")

    print("=== 1. SHALLOW correctness (store/load/ZFOD), all 3 modes ===")
    for mode in MODES:
        m = RollingSinkMemory(sink_mode=mode)
        m.store(0x200, 42); m.store(0x400, 99)
        check(m.load(0x200) == 42 and m.load(0x400) == 99 and m.load(0x800) == 0,
              f"{mode}: 0x200=42, 0x400=99, 0x800=ZFOD 0")

    print("\n=== 2. FULL == MINIMAL stream equivalence (memory-safe accelerator) ===")
    eq = 0; eqn = 0
    for eff in (2000.0, 500000.0):
        for depth in (10, 50, 100, 500, 1000):
            for mode in MODES:
                fk, fz = full_probe(mode, depth, eff, 1.0)
                mk, mz = minimal_probe(mode, depth, eff, 1.0)
                eqn += 1; eq += 1 if (fk == mk and fz == mz) else 0
    check(eq == eqn, f"full==minimal byte-exact {eq}/{eqn}")

    print("\n=== 3. LONG-RANGE HEAD-TO-HEAD (the decisive test) ===")
    print("   EFF=2000 (real full forward): break threshold pos_q>2000 (~step 67)")
    # global-BOS breaks at 67; rolling+softmax1 hold.
    for depth, exp_gbos in ((50, "OK"), (66, "OK"), (67, "BREAK"), (120, "BREAK")):
        rk, rz = full_probe("global_bos", depth, 2000.0, 1.0)
        gbos = "OK" if rz == 0 else "BREAK"
        _, rzr = full_probe("rolling", depth, 2000.0, 1.0)
        _, rzs = full_probe("softmax1", depth, 2000.0, 1.0)
        check(gbos == exp_gbos and rzr == 0 and rzs == 0,
              f"depth={depth}: global_bos zfod={gbos}(exp {exp_gbos}), rolling+softmax1 hold")
    print("   EFF=500000 (minimal stream): break threshold pos_q>500000 (~step 16667)")
    for depth, exp_gbos in ((16666, "OK"), (16667, "BREAK"), (50000, "BREAK"),
                            (100000, "BREAK")):
        _, rz = minimal_probe("global_bos", depth, 500000.0, 1.0)
        gbos = "OK" if rz == 0 else "BREAK"
        _, rzr = minimal_probe("rolling", depth, 500000.0, 1.0)
        _, rzs = minimal_probe("softmax1", depth, 500000.0, 1.0)
        check(gbos == exp_gbos and rzr == 0 and rzs == 0,
              f"depth={depth}: global_bos zfod={gbos}(exp {exp_gbos}), rolling+softmax1 hold")

    print("\n=== 4. BOTH roles (intra-step gather + cross-step), one rolling sink ===")
    for depth in (100, 16667, 50000):
        for mode, exp_z in (("rolling", "OK"), ("global_bos",
                             "OK" if depth < 16667 else "BREAK")):
            g, z = deep_intra(mode, depth)
            zf = "OK" if z == 0 else "BREAK"
            check(g == 0x7B and zf == exp_z,
                  f"intra-step depth={depth} {mode}: read fires, zfod={zf}(exp {exp_z})")
    # cross-step already covered by head-to-head; assert rolling holds deep.
    _, rz = minimal_probe("rolling", 50000, 500000.0, 1.0)
    rk, _ = minimal_probe("rolling", 50000, 500000.0, 1.0)
    check(rk == 0xABCD and rz == 0, "cross-step depth=50000 rolling: read OK, zfod OK")

    print("\n=== 5. READ HORIZON (rolling vs softmax1) ===")
    hs = find_horizon("softmax1", 500000.0, 1.0)
    hr = find_horizon("rolling", 500000.0, 1.0)
    print(f"   softmax1 horizon = {hs} tokens; rolling horizon = {hr} tokens")
    check(hr >= hs, f"rolling horizon ({hr}) >= softmax1 horizon ({hs}) — NOT shortened")

    print("\n=== 6. CALIBRATION window (C_sink), depth past global-BOS break ===")
    def cal_probe(C_sink, depth=17000):
        def build(known_present):
            m = RollingSinkMemory(sink_mode="rolling", C_sink=C_sink,
                                  eff=500000.0, slope=1.0)
            m._rows = []
            ss = depth * W
            m._rows.append((ss, "sink", m._sink_overlay()))
            m._rows.append((ss - 10, "store",
                            m._store_overlay(0x200 ^ (1 << 8), 0x55, False)))
            if known_present:
                m._rows.append((ss - 10, "store", m._store_overlay(0x200, 0xABCD, False)))
            m._step = depth + 1
            return m
        z = build(False).load_at(0x200, depth * W + 8)
        r = build(True).load_at(0x200, depth * W + 8)
        return z, r
    z0, r0 = cal_probe(0)
    check(z0 == 0 and r0 == 0xABCD, "C_sink=0: ZFOD holds AND read fires (valid)")
    zlo, _ = cal_probe(-600000)
    check(zlo != 0, "C_sink=-600000 (below bound): ZFOD BREAKs (window edge)")
    _, rhi = cal_probe(600000)
    check(rhi != 0xABCD, "C_sink=+600000 (above bound): read fails (window edge)")

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"\n==== {npass}/{ntot} PASS   peak RSS = {rss:.1f} MB ====")
    return npass == ntot


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
