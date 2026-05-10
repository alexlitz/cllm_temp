"""Phase 6 gate: I/O syscalls (PRTF, READ, OPEN, CLOS, GETCHAR, PUTCHAR).

These tests run AutoregressiveVMRunner(pure_neural=True) with NO Python
overrides at all. The neural network must drive the I/O opcodes:
emitting the right TOOL_CALL marker (where applicable) and resuming
correctly so subsequent steps see consistent PC/SP/AX state.

Tool boundary I/O (the actual byte transfer) is intentionally external
in non-pure modes; in pure_neural mode the syscall handlers are skipped
entirely, so output bytes can only appear if the model autoregressively
emits them. Most tests here are expected to xfail until that path lands.
"""

import pytest

from neural_vm.embedding import Opcode


def _make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def _run(runner, prog, data=b"", stdin="", max_steps=30):
    bc = _make_bc(prog)
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    out, result = runner.run(bc, data, stdin=stdin, max_steps=max_steps)
    return out, result


class TestPureNeuralPutchar:
    """PUTCHAR — model must route AX byte to OUTPUT autoregressively."""

    @pytest.mark.xfail(
        reason="PUTCHAR has no neural OUTPUT path in pure_neural mode; "
               "_syscall_handlers are skipped and there is no autoregressive "
               "byte-emit weight in the current model.",
        strict=False,
    )
    def test_putchar_one_char(self, pure_neural_runner):
        out, _ = _run(pure_neural_runner, [
            (Opcode.IMM, ord('A')),
            Opcode.PUTCHAR,
            Opcode.EXIT,
        ])
        assert out == "A"

    @pytest.mark.xfail(
        reason="PUTCHAR neural emit path missing — same as test_putchar_one_char.",
        strict=False,
    )
    def test_putchar_two_chars(self, pure_neural_runner):
        out, _ = _run(pure_neural_runner, [
            (Opcode.IMM, ord('A')),
            Opcode.PUTCHAR,
            (Opcode.IMM, ord('B')),
            Opcode.PUTCHAR,
            Opcode.EXIT,
        ], max_steps=40)
        assert out == "AB"


class TestPureNeuralPrtf:
    """PRTF — model must emit TOOL_CALL and externalize the format string."""

    @pytest.mark.xfail(
        reason="PRTF in pure_neural skips _syscall_prtf; no neural TOOL_CALL "
               "emit + DATA-section walk to externalize the format string. "
               "Suspected layer: _set_layer7_memory_heads (string addressing) "
               "and the TOOL_CALL output weight.",
        strict=False,
    )
    def test_prtf_simple(self, pure_neural_runner):
        # Format string "Hi" + NUL placed at start of DATA (0x10000).
        data = b"Hi\x00"
        out, _ = _run(pure_neural_runner, [
            (Opcode.IMM, 0x10000),
            Opcode.PSH,
            Opcode.PRTF,
            (Opcode.ADJ, 8),
            Opcode.EXIT,
        ], data=data, max_steps=40)
        assert "Hi" in out

    @pytest.mark.xfail(
        reason="PRTF with %d arg requires neural format-string interpreter "
               "and integer-to-decimal emit; not implemented in pure_neural.",
        strict=False,
    )
    def test_prtf_with_arg(self, pure_neural_runner):
        data = b"%d\x00"
        out, _ = _run(pure_neural_runner, [
            (Opcode.IMM, 42),
            Opcode.PSH,
            (Opcode.IMM, 0x10000),
            Opcode.PSH,
            Opcode.PRTF,
            (Opcode.ADJ, 16),
            Opcode.EXIT,
        ], data=data, max_steps=40)
        assert "42" in out


class TestPureNeuralGetchar:
    """GETCHAR — model must inject next stdin byte into AX."""

    @pytest.mark.xfail(
        reason="GETCHAR neural USER_INPUT attention head not baked; "
               "_inject_getchar is bypassed in pure_neural mode.",
        strict=False,
    )
    def test_getchar_returns_byte(self, pure_neural_runner):
        _, ax = _run(pure_neural_runner, [
            Opcode.GETCHAR,
            Opcode.EXIT,
        ], stdin="x", max_steps=20)
        assert (ax & 0xFF) == ord('x')


class TestPureNeuralFileOps:
    """OPEN/CLOS lifecycle — pure tool-boundary, no neural fallback."""

    @pytest.mark.xfail(
        reason="OPEN/CLOS are pure tool-boundary calls; with handlers "
               "disabled in pure_neural and no neural _set_layer7_memory_heads "
               "path for filename addressing, no fd is produced.",
        strict=False,
    )
    def test_open_close_cycle_xfail(self, pure_neural_runner):
        # Filename "x\0" at DATA[0]
        data = b"x\x00"
        _, exit_code = _run(pure_neural_runner, [
            (Opcode.IMM, 0),
            Opcode.PSH,
            (Opcode.IMM, 0x10000),
            Opcode.PSH,
            Opcode.OPEN,
            (Opcode.ADJ, 16),
            Opcode.PSH,
            Opcode.CLOS,
            (Opcode.ADJ, 8),
            Opcode.EXIT,
        ], data=data, max_steps=60)
        assert exit_code == 0


class TestPureNeuralRead:
    """READ — buffer addressing requires neural memory head."""

    @pytest.mark.xfail(
        reason="READ writes to a buffer in DATA/heap; needs "
               "_set_layer7_memory_heads to address the buffer, plus "
               "neural USER_INPUT consumption. Both missing in pure_neural.",
        strict=False,
    )
    def test_read_from_stdin(self, pure_neural_runner):
        # READ(fd=0, buf=0x10000, count=1) — push count, buf, fd left-to-right.
        _, exit_code = _run(pure_neural_runner, [
            (Opcode.IMM, 1),
            Opcode.PSH,
            (Opcode.IMM, 0x10000),
            Opcode.PSH,
            (Opcode.IMM, 0),
            Opcode.PSH,
            Opcode.READ,
            (Opcode.ADJ, 24),
            Opcode.EXIT,
        ], stdin="z", max_steps=50)
        # MEM check: byte at 0x10000 should be 'z' if neural buffer write worked.
        assert pure_neural_runner._memory.get(0x10000, 0) == ord('z')
        assert exit_code == 1
