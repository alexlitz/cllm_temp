"""L5 FFN conversational_io weight inspection (debug script).

Originally a standalone debug script that pytest pulled into collection
because of the ``test_`` filename. The script body inspects a single
FFN unit (410, "PRTF detection") by reading W_up against the
``ACTIVE_OPCODE_PRTF`` dimension -- a layout-snapshot check, not a
pytest-style assertion. Two structural problems make it useless to
collect under the current main branch:

  * ``BD.ACTIVE_OPCODE_PRTF`` is now 504, well past the legacy W_up
    column count (90), so a bare module-level read raises ``IndexError``
    at collection time and bricks every other test in the same session.
  * ``runner.model.set_active_opcode`` was retired in favor of an
    embedding-level peek, so even if the index ran, the assertion path
    would AttributeError.

Wrapping the original body in ``if __name__ == "__main__":`` keeps the
debug entry point usable (``python tests/test_l5_weights.py``) while
making module import a no-op so pytest collection succeeds without
attempting to run a script that has no ``def test_*`` functions.
"""

from __future__ import annotations


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from neural_vm.run_vm import AutoregressiveVMRunner
    from neural_vm.vm_step import _SetDim as BD

    runner = AutoregressiveVMRunner(conversational_io=True)

    ffn5 = runner.model.blocks[5].ffn

    # Check unit 410 (PRTF detection)
    unit = 410

    print("L5 FFN Unit 410 (PRTF detection):")
    if unit < ffn5.W_up.shape[0] and BD.ACTIVE_OPCODE_PRTF < ffn5.W_up.shape[1]:
        print(
            f"  W_up[{unit}, ACTIVE_OPCODE_PRTF={BD.ACTIVE_OPCODE_PRTF}]: "
            f"{ffn5.W_up[unit, BD.ACTIVE_OPCODE_PRTF].item():.2f}"
        )
        print(f"  b_up[{unit}]: {ffn5.b_up[unit].item():.2f}")
        print(
            f"  W_gate[{unit}, :] non-zero: "
            f"{(ffn5.W_gate[unit] != 0).sum().item()} dims"
        )
        print(f"  b_gate[{unit}]: {ffn5.b_gate[unit].item():.2f}")
        print(
            f"  W_down[IO_IS_PRTF={BD.IO_IS_PRTF}, {unit}]: "
            f"{ffn5.W_down[BD.IO_IS_PRTF, unit].item():.2f}"
        )

        if ffn5.W_up[unit, BD.ACTIVE_OPCODE_PRTF].item() == 0:
            print(
                "\nERROR: W_up is not set! Conversational I/O weights "
                "may not be applied."
            )
        else:
            print("\nWeights are set correctly.")
    else:
        print(
            f"  Index out of range: unit={unit} W_up.shape={tuple(ffn5.W_up.shape)} "
            f"ACTIVE_OPCODE_PRTF={BD.ACTIVE_OPCODE_PRTF}; the layout has changed "
            f"since this debug script was last useful."
        )
