"""L6 relay debug script (IO_IS_PRTF -> CMP[3] tracing).

Originally a standalone debug script that pytest pulled into collection
because of its ``test_`` filename. It has no ``def test_*`` functions
and the model API it pokes (``runner.model.set_active_opcode``) was
retired in favor of an embedding-level peek, so importing the module
under pytest crashed during collection.

Wrapping the original body in ``if __name__ == "__main__":`` preserves
the manual debug entry point (``python tests/test_l6_relay.py``) while
making collection a no-op. If/when the L5/L6 PRTF relay chain needs a
real test, this file is the right home for it -- a fresh top-level
``def test_*`` function added below will be picked up automatically.
"""

from __future__ import annotations


if __name__ == "__main__":
    import os
    import sys

    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from neural_vm.run_vm import AutoregressiveVMRunner
    from neural_vm.vm_step import Token, _SetDim as BD

    runner = AutoregressiveVMRunner(conversational_io=True)
    if torch.cuda.is_available():
        runner.model = runner.model.cuda()

    # Set active opcode to PRTF. The legacy ``set_active_opcode`` method
    # was retired; AutoregressiveVM now reads the opcode straight from the
    # embedding peek. Pass it through to ``embed(..., active_opcode=33)``
    # instead.
    set_active_opcode = getattr(runner.model, "set_active_opcode", None)
    if callable(set_active_opcode):
        set_active_opcode(33)

    # Create a context with STEP_END at the last position
    test_tokens = [
        Token.REG_PC, 0, 0, 0, 0,
        Token.REG_AX, 0, 0, 0, 0,
        Token.REG_SP, 0, 0, 0, 0,
        Token.REG_BP, 0, 0, 0, 0,
        Token.STACK0, 0, 0, 0, 0,
        Token.MEM, 0, 0, 0, 0, 0, 0, 0, 0,
        Token.STEP_END,
    ]  # Position 34

    test_tokens = torch.tensor(
        [test_tokens], device=runner.model.embed.embed.weight.device,
    )

    with torch.no_grad():
        x = runner.model.embed(test_tokens, active_opcode=33)

        print("After embedding:")
        print(
            f"  ACTIVE_OPCODE_PRTF at pos 34 (SE): "
            f"{x[0, 34, BD.ACTIVE_OPCODE_PRTF].item():.2f}"
        )
        print(f"  NEXT_SE at pos 34: {x[0, 34, BD.NEXT_SE].item():.2f}")

        # Run through L5
        for i in range(6):
            x = runner.model.blocks[i](x)
            if i == 4:  # After L5 FFN
                print(f"\nAfter L5 FFN:")
                print(f"  IO_IS_PRTF at pos 34 (SE): {x[0, 34, BD.IO_IS_PRTF].item():.2f}")
            elif i == 5:  # After L6
                print(f"\nAfter L6 (attention + relay):")
                print(f"  IO_IS_PRTF at pos 34: {x[0, 34, BD.IO_IS_PRTF].item():.2f}")
                print(f"  CMP[3] at pos 34: {x[0, 34, BD.CMP + 3].item():.2f}")
                print(f"  NEXT_SE at pos 34: {x[0, 34, BD.NEXT_SE].item():.2f}")
                print(
                    f"  NEXT_THINKING_END at pos 34: "
                    f"{x[0, 34, BD.NEXT_THINKING_END].item():.2f}"
                )

                if x[0, 34, BD.CMP + 3].item() > 0.5:
                    print("\nCMP[3] is set by L6 relay")
                else:
                    print("\nCMP[3] NOT set by L6 relay")

                if x[0, 34, BD.NEXT_THINKING_END].item() > 0.5:
                    print("NEXT_THINKING_END is set by L6 state machine")
                else:
                    print("NEXT_THINKING_END NOT set by L6 state machine")
