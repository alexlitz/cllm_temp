"""c4_min — the green-field minimal-ISA compiler + model package.

This package is being built in parallel:

  * ``c4_min/compile.py``  — ``compile_program(prog) -> state_dict`` : lowers a
    program (bytecode + data) into a small transformer's ``state_dict``.
  * ``c4_min/model.py``    — ``run(state_dict, prog) -> decoded_bytes`` : runs
    the transformer autoregressively over the compiled program and returns the
    decoded ISA output (the per-step (PC, AX) trace, terminating with the EXIT
    exit code).
  * ``c4_min/DESIGN.md``   — the contract between the two.

This file (``oracle.py`` + ``run_oracle.py``) is the ORACLE HARNESS: it does not
depend on the green-field compiler being finished. It computes the EXPECTED ISA
behaviour with the SAME reference semantics the reference model's per-op oracle
trusts (``neural_vm.verification.symbolic_program`` — used for the *expected*
side ONLY), then drives the green-field ``compile_program`` / ``run`` pair to
check the model-under-test reproduces it. Until the real compiler lands, the
harness is exercised against a hand-mocked correct+incorrect model (the
self-test in ``run_oracle.py``), proving it detects both PASS and FAIL.
"""
