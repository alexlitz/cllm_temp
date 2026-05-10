#!/usr/bin/env python3
"""Diagnostic: inspect MEM tokens generated during a JSR step.

Phase 5 test `test_jsr_callee_writes_ax` claims the JSR return-address push
is incorrect. The L14 attention block (`_set_layer14_mem_generation`) is
responsible for routing values into the MEM section's addr/val byte slots:
  * addr heads (0-3): copy SP bytes (PSH/JSR/ENT) or STACK0 bytes (SI/SC)
  * val heads (4-7): copy AX bytes (default) or STACK0 bytes (JSR/ENT)

For JSR, MEM[SP] should hold the return address (PC + INSTR_WIDTH = PC + 8).
The chain that produces this value:
  L6 head 7    : reads OUTPUT_LO/HI at PC marker (= NEW PC byte 0 from L3 incr)
                 writes to AX_CARRY_LO/HI at STACK0 marker.
  L6 FFN units : at STACK0 marker AND at STACK0 byte 0..3 positions, writes
                 AX_CARRY into OUTPUT (gated on CMP[4] = OP_JSR relay).
  Sampler      : emits STACK0 byte 0..3 tokens from those OUTPUT logits.
  L14 val heads: read CLEAN_EMBED at STACK0 byte positions (gated on OP_JSR)
                 and route into MEM val bytes.

Findings (run on this branch with `_func_call_handlers={}`):
  Expected: MEM section [MEM, F8 FF FF FF, 0A 00 00 00]
                        addr=SP-8=0xFFFFFFF8, value=return_addr=0x0A
  Actual:   MEM section [261(MEM), 257(REG_PC), 02, 00, 00, 00, 258(REG_AX), 04, 00]

So the model produces *register marker tokens* (REG_PC=257 at addr_b0,
REG_AX=258 at val_b0) instead of byte values. The byte values for the other
slots are also wrong (02 / 0 / 4 / 0 — none of them are the expected 0xF8 /
0xFF / 0x0A / 0x00).

ROOT CAUSE ANALYSIS — the bug is NOT contained in
`_set_layer14_mem_generation`. That function is internally consistent:
  * Q/K dims correctly select target byte slots (dim 0 + dim 33 position gate).
  * Dim 1 (AX source) is disabled when OP_JSR is set (-2L on OP_JSR).
  * Dim 2 (STACK0 source) is enabled when OP_JSR is set (+L on OP_JSR).
  * V copies CLEAN_EMBED_LO/HI at STACK0 byte positions.
  * MEM_STORE gate (dim 34) and OP_JSR relay are both broadcast to MEM byte
    positions by L7 head 7 (lines 5485-5495), so the gating works.

The actual bug is upstream, in the JSR-specific L6 FFN units that write the
return_addr value into STACK0 byte positions:

  vm_step.py:9436-9468 (--- JSR STACK0 = return_addr (128 units) ---)

  These units read AX_CARRY_LO/HI[k] at every STACK0 byte position J (gated
  on BYTE_INDEX_J + IS_BYTE + L1H4[BP] AND NOT H1[BP]) and write it into
  OUTPUT_LO/HI[k]. But:

    1. AX_CARRY only contains return_addr BYTE 0 — L6 head 7 reads only
       the OUTPUT at PC marker (a single position holding PC byte 0). For
       a proper 32-bit return_addr push, AX_CARRY would need bytes 0-3.

    2. The unit at byte_idx=J fires at STACK0 byte J position (because
       BYTE_INDEX_J fires there). OUTPUT at STACK0 byte J predicts STACK0
       byte J+1 (autoregressive shift). So the value AX_CARRY (which is
       return_addr byte 0) gets emitted at STACK0 bytes 1, 2, 3 — clobbering
       the high-order zero bytes.

  Combined effect: STACK0 byte 0 = return_addr byte 0 (correct), but STACK0
  bytes 1-3 also = return_addr byte 0 (should be 0).

  Then L14 val heads read these CLEAN_EMBED values and emit them into MEM
  val bytes — producing a corrupted return_addr.

Why are *marker tokens* (REG_PC, REG_AX) emitted at addr_b0/val_b0? The
prediction at MEM marker (predicting addr_b0) reads from V at the SP marker
position. SP after JSR push should be 0xFFFFFFF8, but the L6 head 7 chain
above wrote AX_CARRY into STACK0 bytes (and the L6 FFN at STACK0 marker
also writes AX_CARRY into OUTPUT, see line 9419-9434). The end result is
that OUTPUT/CLEAN_EMBED at SP byte 0 contains noise that pushes byte logits
below the marker logits' bias floor (-10) → REG_PC token wins.

CONCLUSION
==========
`_set_layer14_mem_generation` is correct. The fix lies elsewhere:

  PRIMARY FIX (out-of-scope per task constraint):
    vm_step.py:9436-9468 — the JSR STACK0 byte-write units must emit
    return_addr BYTE J at STACK0 byte J-1 position (not byte 0 at every
    position). This requires either:
      (a) routing return_addr byte J via a head that gathers from PC byte J,
          or
      (b) generating high-order zero bytes via a separate L6 FFN unit set,
          since for a 32-bit return_addr that fits in one byte (most cases)
          bytes 1-3 are simply 0.

  SECONDARY:
    Verify the SP-bytes-after-JSR-decrement chain (L6/L7 SP-=8 path) actually
    writes 0xFFFFFFF8 into SP byte positions; if it does not, L14 addr heads
    have nothing valid to read.

NOTE ON TESTABILITY
===================
This branch (worktree-agent-a2a80b952e04550e2 ≈ migrate-l7-ops-to-compiler)
does NOT support `pure_neural=True` mode in AutoregressiveVMRunner. The
runner's `_dispatch_step` always runs hardcoded JSR/ENT/LEV fix-up
(_inject_mem_section, override REG_SP/STACK0/PC), masking the L14/L6
output. Running this diagnostic shows the raw model tokens via
[MEM EXTRACT] log lines but the runner still corrects state, so subsequent
steps' inputs differ from a true pure-neural execution.

The Phase 5 test fixture `pure_neural_runner` (defined in
phase5-pure-neural-jsr-ent-lev branch's tests/conftest.py) requires the
`pure_neural=True` constructor flag added in commit b0037c1 on `main`. Phase 5
verification therefore must happen on a branch containing both b0037c1 and
this fix.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET


def make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def main():
    runner = AutoregressiveVMRunner(debug_memory=True)
    runner._func_call_handlers = {}

    bc = make_bc([
        (Opcode.JSR, 2),
        Opcode.EXIT,
        (Opcode.ENT, 0),
        (Opcode.IMM, 42),
        Opcode.LEV,
    ])

    expected_return_addr = PC_OFFSET + 0 * INSTR_WIDTH + INSTR_WIDTH
    print(f"Expected return_addr after JSR: 0x{expected_return_addr:08x}")
    print(f"Expected SP after JSR push: 0xFFFFFFF8 (= 0 - 8 mod 2^32)")
    print()

    out, ec = runner.run(bc, b"", max_steps=3)
    print(f"\nFinal exit code: {ec!r}, output: {out!r}")
    print(f"Final last_sp: 0x{runner._last_sp:08x}, "
          f"last_pc: 0x{(runner._last_pc or 0):08x}, "
          f"last_ax: {runner._last_ax}")
    print()

    print("Mem history (runner-injected, AFTER JSR fix-up):")
    for addr, section in runner._mem_history.items():
        addr_val = sum((section[1 + j] & 0xFF) << (j * 8) for j in range(4))
        val_val = sum((section[5 + j] & 0xFF) << (j * 8) for j in range(4))
        print(f"  addr=0x{addr:08x}: stored addr=0x{addr_val:08x} value=0x{val_val:08x}")
    print()
    print("(See [MEM EXTRACT] log lines above for raw model tokens that L14 emitted.)")


if __name__ == "__main__":
    main()
