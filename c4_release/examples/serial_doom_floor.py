#!/usr/bin/env python3
r"""serial_doom_floor.py — the Doom FLOP floor under an ALL-OPS-DEEP-SERIAL c4 VM.

Grounded in the MEASURED per-op FLOPs from tiny_serial_alu.py (ADD 320, MUL 5120,
DIV/MOD 3008 FLOP/op, all fp32-exact) and the MEASURED current build numbers
(5.68 MFLOP/step; _flop_gauge.py at S=91: IMM 334M, ADD 669M, MUL 622M, DIV 8.99G;
264 blocks; dim=1440).

It estimates the FLOP floor of a minimal NARROW-SERIAL c4 VM step and multiplies by
the Doom step counts (raw 6,889,264 / render-reduced 358,058) to get the Doom-frame
FLOP floor + the reduction factor vs the current ~39 TFLOP raw / ~2 TFLOP render.

This is an analytic floor, not a byte-exact build (Part 1 proves the ALU cells are
exact; Part 2 composes their measured costs into a VM-step estimate).  Every input
number is labelled MEASURED or ESTIMATE.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# MEASURED per-op FLOP from tiny_serial_alu.py (fp32-exact, verified >=100k each)
# --------------------------------------------------------------------------- #
SERIAL_ALU = {           # FLOP/op = depth * MAC/layer * 2
    "ADD": 320,          # depth 32, 5 MAC/layer   (also covers SUB/AND/OR/XOR/SHL/SHR ~<=ADD)
    "MUL": 5120,         # depth 32, 80 MAC/layer
    "DIV": 3008,         # depth 32, 47 MAC/layer   (MOD shares the divider)
}

# --------------------------------------------------------------------------- #
# The NARROW-SERIAL VM-STEP model
# --------------------------------------------------------------------------- #
# A c4 VM step = fetch opcode + decode + execute-one-op + writeback(PC/SP/BP).
# In the narrow-serial style every register is 32 bits processed bit/nibble-serial.
#
#   fetch:     read the opcode byte at PC (a nibble-serial byte read + PC compare).
#   decode:    select the op's cell (a tiny one-hot over ~40 opcodes).
#   execute:   run the op's serial ALU cell (ADD/MUL/DIV) OR a pointer op (PSH/LEA/LI
#              = an address add + a memory-cell read/write; ~an ADD's cost).
#   writeback: PC += insn_len (a small ADD); SP/BP +/- 1 for stack ops (a small ADD).
#
# FRAMING = fetch+decode+writeback.  Each is bounded by ~1 ADD-class serial op over a
# 32-bit register (a byte/address compare, a PC increment, an SP bump).  We charge
# framing generously as ~4 ADD-class ops (= 4*320 FLOP) plus a small opcode-select
# one-hot (~40 opcodes * a few MACs ~= a few hundred FLOP).
ADD = SERIAL_ALU["ADD"]
FRAMING_FLOP = 4 * ADD + 300      # ~4 ADD-class framing ops + ~40-way decode select
print(f"[model] framing (fetch+decode+PC/SP writeback) ~= 4 ADD-class + decode "
      f"= {FRAMING_FLOP} FLOP/step")

# Per-opcode EXECUTE cost in the serial style.
# Pointer/stack/branch/compare ops reduce to an address-ADD + a byte move (~1 ADD).
# ALU ops use their measured serial cell.
EXEC_FLOP = {
    "PSH":  ADD,           # push: SP-- (ADD) + write stack cell (byte move ~0)
    "POP":  ADD,
    "LEA":  ADD,           # effective address = BP + offset (one ADD)
    "LI":   ADD,           # load indirect: address in reg -> memory byte move (~1 ADD)
    "SI":   ADD,           # store indirect
    "IMM":  ADD // 2,      # load immediate: a byte move into a register (< ADD)
    "JMP":  ADD,           # PC <- target (a move + compare)
    "BZ":   ADD, "BNZ": ADD,
    "JSR":  2 * ADD,       # push return addr + set PC
    "ENT":  2 * ADD, "LEV": 2 * ADD,   # frame enter/leave: SP,BP bumps
    "CMP":  ADD,           # subtract-and-test = one ADD-class op
    "ADD":  SERIAL_ALU["ADD"], "SUB": SERIAL_ALU["ADD"],
    "AND":  SERIAL_ALU["ADD"], "OR": SERIAL_ALU["ADD"], "XOR": SERIAL_ALU["ADD"],
    "SHL":  SERIAL_ALU["ADD"], "SHR": SERIAL_ALU["ADD"],
    "MUL":  SERIAL_ALU["MUL"],
    "DIV":  SERIAL_ALU["DIV"], "MOD": SERIAL_ALU["DIV"],
}


def step_flop(op: str) -> int:
    return FRAMING_FLOP + EXEC_FLOP.get(op, ADD)


# --------------------------------------------------------------------------- #
# Doom opcode MIX.  MEASURED fact (capstone §6 / PERF_LADDER §5): 63% of the render
# is PSH/LEA/LI pointer-walking.  The rest is branches/compares/ALU, with MUL/DIV
# rare (Doom's inner render loops are fixed-point but the heavy divides are folded
# into the render superinstruction; per-step DIV is a small minority).
# We use a grounded ESTIMATE mix (labelled), then also give a PSH-only lower bound
# and an ADD-only upper-ish bound to bracket it.
# --------------------------------------------------------------------------- #
DOOM_MIX = {            # fractions sum to 1.0  (ESTIMATE grounded in the 63% fact)
    "PSH": 0.355, "LEA": 0.18, "LI": 0.12,         # ~63% pointer-walk (measured)
    "IMM": 0.08, "JMP": 0.03, "BZ": 0.03, "BNZ": 0.02,
    "CMP": 0.05, "ADD": 0.06, "SUB": 0.03,
    "AND": 0.01, "SHL": 0.01, "JSR": 0.005, "LEV": 0.005,
    "MUL": 0.01, "DIV": 0.005,                     # heavy ALU is rare per-step
}
assert abs(sum(DOOM_MIX.values()) - 1.0) < 1e-6, sum(DOOM_MIX.values())

mix_flop = sum(f * step_flop(op) for op, f in DOOM_MIX.items())
psh_only = step_flop("PSH")           # if EVERY step were the lightest pointer op
add_only = step_flop("ADD")           # if EVERY step were an ADD-class op

print(f"[model] serial VM-step FLOP:  PSH-only floor = {psh_only}, "
      f"ADD-class = {add_only}, DIV-step = {step_flop('DIV')}")
print(f"[model] Doom-mix weighted serial VM-step = {mix_flop:.0f} FLOP/step "
      f"(ESTIMATE, 63% pointer-walk measured)")

# --------------------------------------------------------------------------- #
# CURRENT (WIDE) build — MEASURED
# --------------------------------------------------------------------------- #
CUR_MFLOP_STEP = 5.68e6          # MEASURED, capstone §6
RAW_STEPS = 6_889_264            # MEASURED, raw title-redraw frame
RENDER_STEPS = 358_058          # MEASURED, render-reduced steady frame

cur_raw = CUR_MFLOP_STEP * RAW_STEPS
cur_render = CUR_MFLOP_STEP * RENDER_STEPS

# --------------------------------------------------------------------------- #
# DEEP-SERIAL floor
# --------------------------------------------------------------------------- #
ser_raw = mix_flop * RAW_STEPS
ser_render = mix_flop * RENDER_STEPS

print("\n" + "=" * 84)
print("DOOM FRAME FLOP — current WIDE build vs all-ops-deep-serial floor")
print("=" * 84)
hdr = f"{'frame':<20s}{'steps':>12s}{'FLOP/step':>14s}{'total FLOP':>16s}{'':>6s}"
print(hdr)
print("-" * 84)


def fmt(x):
    for u, s in [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")]:
        if x >= u:
            return f"{x/u:.2f}{s}FLOP"
    return f"{x:.0f}FLOP"


print(f"{'RAW current':<20s}{RAW_STEPS:>12,d}{CUR_MFLOP_STEP:>14,.0f}{fmt(cur_raw):>16s}")
print(f"{'RAW serial-floor':<20s}{RAW_STEPS:>12,d}{mix_flop:>14,.0f}{fmt(ser_raw):>16s}")
print(f"{'RENDER current':<20s}{RENDER_STEPS:>12,d}{CUR_MFLOP_STEP:>14,.0f}{fmt(cur_render):>16s}")
print(f"{'RENDER serial-floor':<20s}{RENDER_STEPS:>12,d}{mix_flop:>14,.0f}{fmt(ser_render):>16s}")
print("-" * 84)
print(f"  per-step reduction   : {CUR_MFLOP_STEP/mix_flop:,.0f}x  "
      f"({CUR_MFLOP_STEP/1e6:.2f} MFLOP -> {mix_flop:.0f} FLOP)")
print(f"  RAW frame reduction  : {cur_raw/ser_raw:,.0f}x  "
      f"({fmt(cur_raw)} -> {fmt(ser_raw)})")
print(f"  RENDER frame reduction: {cur_render/ser_render:,.0f}x  "
      f"({fmt(cur_render)} -> {fmt(ser_render)})")

# --------------------------------------------------------------------------- #
# DEPTH (the cost) — layer-applications per step and per frame
# --------------------------------------------------------------------------- #
# current build: 264 blocks, ~1 pass/step (per-step re-embed) => 264 layer-apps/step.
CUR_DEPTH = 264
# serial floor: framing (~4 ADD-class * 32 bits) + exec (op's depth).  A pointer/ADD
# step ~= (4+1)*32 = ~160 serial layer-applications; a DIV step ~= (4+1)*32 = ~160 too
# (the divider is 32 layers, framing 4*32); a MUL step ~= 5*32 = 160.  So ~160
# layer-apps/step for the common case (vs 264 dense blocks) — but each layer is now
# ~tens of MACs not ~1440-wide, and the SERIAL ones cannot be parallelized away.
SER_DEPTH_STEP = 5 * 32   # ~160 serial layer-apps for the ADD/pointer common case
print("\n" + "=" * 84)
print("DEPTH / OCCUPANCY caveat (the cost of the FLOP win)")
print("=" * 84)
print(f"  current WIDE build : ~{CUR_DEPTH} layer-apps/step, each ~1440-wide "
      f"(mostly parallel within a block)")
print(f"  deep-serial floor  : ~{SER_DEPTH_STEP} layer-apps/step, each ~3-18 wide, "
      f"STRICTLY SEQUENTIAL (bit i needs bit i-1's carry)")
print(f"  raw-frame serial depth : ~{SER_DEPTH_STEP*RAW_STEPS/1e9:.2f}G "
      f"layer-applications (was ~{CUR_DEPTH*RAW_STEPS/1e9:.2f}G, but those parallel-wide)")
print(f"  => FLOP collapses ~{CUR_MFLOP_STEP/mix_flop:,.0f}x, but the serial carry chain forbids "
      f"width-parallelism;")
print("     wall-clock is a SEPARATE occupancy question (a GPU MMA unit is starved by a")
print("     3-18-wide, 160-deep dependency chain).  The triple is: params DOWN, FLOP DOWN,")
print("     DEPTH UP.")

# headline
print("\n" + "=" * 84)
print("HEADLINE")
print("=" * 84)
print(f"  With ALL c4 ops deep-serial + minimal-nonzero-param, a byte-exact Doom frame's")
print(f"  arithmetic FLOP floor is ~{fmt(ser_raw)} raw / ~{fmt(ser_render)} render-reduced,")
print(f"  i.e. ~{cur_raw/ser_raw:,.0f}x below today's ~{fmt(cur_raw)} raw / ~{fmt(cur_render)} render.")
print(f"  The width overhead (~1440x per gate) is what vanishes; the price is depth")
print(f"  (~{SER_DEPTH_STEP} serial layer-apps/step) that a GPU cannot parallelize.")
