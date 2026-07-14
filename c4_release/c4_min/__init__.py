"""c4_min — the green-field minimal-ISA (8-bit C4) compiler + model + oracle.

Clean-room: does **not** import from ``neural_vm/`` on the model-under-test path
(the oracle's *expected* side reuses the reference ISA semantics only).

Package pieces:

  * ``c4_min/compiler.py`` — ``compile_program(prog) -> (model, layout, code)``
    lowers a program into a depth-unrolled transformer, plus ``run(...)`` which
    decodes the per-step AX trace via the LM head. This is the REAL substrate.
  * ``c4_min/model.py``    — the runtime ``Transformer`` (exact reference
    architecture: ALiBi/softmax attention + SwiGLU FFN, additive residual).
  * ``c4_min/isa.py``      — opcode subset, encoding, and the clean-room 8-bit
    reference interpreter.
  * ``c4_min/DESIGN.md``   — the design contract (compiler + the Model interface
    the oracle depends on).

  * ``c4_min/oracle.py`` + ``c4_min/run_oracle.py`` — the ORACLE HARNESS: it
    computes the EXPECTED ISA behaviour (reference semantics via
    ``neural_vm.verification.symbolic_program`` — *expected* side ONLY), then
    drives the REAL ``compiler.compile_program`` / ``compiler.run`` pair
    (through a thin adapter) to check the model reproduces it, printing a per-op
    PASS/FAIL table. A ``--self-test`` exercises a hand-mocked correct+buggy
    model (no compiler/GPU needed), proving the harness detects both PASS and
    FAIL.
"""
