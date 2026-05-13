"""V9 user-input neural read scaffolding ops (GETCHAR via attention).

These bakes implement the spec contract at ``BLOG_SPEC.md:851``:

    Input bytes are injected into the token stream between
    USER_INPUT_START/END markers, and the VM reads them via attention.

Phase 1 (this commit): both ops registered, both gated ``enable=False`` by
default. The runner-side ``_inject_getchar`` shim (see ``run_vm.py``) still
owns the byte transfer when these are disabled. When ``enable=True``, the L5
attention head pair locates the first byte after USER_INPUT_START and routes
its low/high nibbles into an L6 FFN unit pair that drives ``AX_CARRY_LO/HI``
at the GETCHAR step's AX marker.

See ``docs/V9_GETCHAR_READ_NEURAL_PLAN.md`` for the multi-phase rollout
plan: cursor advancement, EOF, READ-into-memory.
"""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer5_user_input_gather_op(enable: bool = False) -> Operation:
    """L5 attention head pair: locate next-unread USER_INPUT byte at GETCHAR.

    Phase 1 (``enable=False``): no-op. Registered so the dep-graph layout is
    stable when phase 2 flips the flag.

    Phase 1 (``enable=True``, when wiring lands):
      - Head A fires at ``OP_GETCHAR & MARK_AX``, attends to
        ``Token.USER_INPUT_START`` via the existing ``IS_MARK`` projection
        plus a dedicated ``MARK_USER_INPUT_START`` dim (allocated in the
        same migration).
      - Head B uses Head A's output position + cursor=0 (phase 1) to gather
        the byte at offset ``start + 1`` into ``STDIN_BYTE_LO/HI`` nibble
        dims.

    The two heads slot into L5 alongside the existing instruction-fetch
    heads (``_set_layer5_fetch``, 8 heads). Heads 0-7 are taken by the
    fetch bake; this op would extend to heads 8-9 in phase 2 (requires the
    L5 attention to be widened from 8 to 10 heads at compile time).

    Phase 5.7 places this AFTER ``layer5_fetch`` (phase 5) so the head
    allocation order is stable.
    """
    def bake(block, dim_positions, S):
        if not enable:
            return
        # TODO(v9-phase2): allocate heads 8/9 on block.attn, write Q/K/V/O
        # weights for the USER_INPUT_START locator and byte gather. Requires:
        #   1. Dim registry alloc for MARK_USER_INPUT_START, MARK_USER_INPUT_END,
        #      STDIN_BYTE_LO (16 dims), STDIN_BYTE_HI (16 dims).
        #   2. L5 attention widened from 8 to 10 heads in compile_full_vm.
        #   3. Embedding bake adds MARK_USER_INPUT_START=1.0 on token 269,
        #      MARK_USER_INPUT_END=1.0 on token 270.
        #   4. ALiBi position-bias setup for Head B's offset arithmetic
        #      (USER_INPUT_START_POS + 1 + cursor).
        # See docs/V9_GETCHAR_READ_NEURAL_PLAN.md §3.2 for the full design.
        raise NotImplementedError(
            "layer5_user_input_gather bake body is staged for V9 phase 2. "
            "Set enable=False until the dim allocations + L5 head widening "
            "are in place."
        )

    return Operation(
        name="layer5_user_input_gather",
        phase=5.7,
        # Phase 1 declares empty reads/writes since the bake body is a no-op.
        # Phase 2 will populate with MARK_USER_INPUT_START/END, OP_GETCHAR,
        # MARK_AX, IS_BYTE, EMBED_LO/HI -> STDIN_BYTE_LO/HI.
        reads=set(),
        writes=set(),
        kind="block",
        layer_idx=5,
        bake_fn=bake,
        migrated=True,
        smoke_tests=set(),
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer6_getchar_routing_op(enable: bool = False) -> Operation:
    """L6 FFN units: route STDIN_BYTE_LO/HI -> AX_CARRY_LO/HI at GETCHAR AX.

    Mirrors ``_set_io_putchar_routing`` (vm_step.py:6987) structurally, but
    going the *other* direction: PUTCHAR copies ``AX_CARRY -> OUTPUT`` so
    the byte logits select an output byte token; GETCHAR copies the
    USER_INPUT-derived ``STDIN_BYTE -> AX_CARRY`` so the AX byte 0 emit
    produces the input byte.

    Phase 1 (``enable=False``): no-op. The runner-side ``_inject_getchar``
    shim still overrides REG_AX after the step completes.

    Phase 1 (``enable=True``, when wiring lands):
      - Adds 33 FFN units starting at unit 1600 (past the PUTCHAR routing
        block at 1500-1531):
          unit 1600:          OP_GETCHAR & MARK_AX -> IO_IS_GETCHAR flag
          units 1601-1616:    STDIN_BYTE_LO[k] -> AX_CARRY_LO[k] (k=0..15)
          units 1617-1632:    STDIN_BYTE_HI[k] -> AX_CARRY_HI[k] (k=0..15)
      - bytes 1..3 of AX stay zero (handled by the existing AX-carry zero-fill).

    Phase 998.9: between ``io_putchar_routing`` (998) and ``legacy_bake``
    (999), so the FFN units we program survive ``_right_size_ffns`` (which
    runs at the end of legacy_bake).
    """
    def bake(model, dim_positions, S):
        if not enable:
            return
        # TODO(v9-phase2): bake the 33 FFN units described above. Requires
        # STDIN_BYTE_LO/HI dim allocations (see make_layer5_user_input_gather_op).
        # See docs/V9_GETCHAR_READ_NEURAL_PLAN.md §3.3 for the wire pattern.
        raise NotImplementedError(
            "layer6_getchar_routing bake body is staged for V9 phase 2. "
            "Set enable=False until STDIN_BYTE_LO/HI dims are allocated and "
            "the L5 user_input gather head is verified."
        )

    return Operation(
        name="layer6_getchar_routing",
        reads=set(),
        writes=set(),
        kind="model",
        bake_fn=bake,
        phase=998.9,
        migrated=True,
        smoke_tests=set(),
        spec_section="BLOG_SPEC.md#printing-and-reading-input",
    )
