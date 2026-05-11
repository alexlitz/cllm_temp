"""Batched pure-neural runner.

Wraps the same compiled neural model as ``AutoregressiveVMRunner(pure_neural=True)``
and runs N programs in lockstep using a single batched forward pass per token.

The serial pure-neural runner does:

    for token_pos in range(max_steps * STEP_TOKENS):
        next_token = model.generate_next(context)
        context.append(next_token)
        if next_token == STEP_END: maybe_dispatch(...)
        if next_token == HALT: break

The batched version replaces ``model.generate_next`` with a single forward over
the padded tensor ``[B, max_len]``. Each program keeps its own context, runner
state (last_pc/ax/sp/bp, mem_history, etc.) and output buffer. Programs
terminate independently: once a program emits HALT, its slot is no longer
read out of the next forward pass, but it stays in the batch tensor (filled
with HALT tokens) until all programs finish.

Scope (Phase 1):
    * Designed for tests that run with ``pure_neural=True, trust_neural_alu=True``
      with NO Python overrides on syscalls or func-call handlers.
    * Programs that use MEM-store ops (SI/SC/PSH/JSR/ENT) update each batch
      element's ``_mem_history`` and rebuild contexts the same way the serial
      runner does. The shared ``model.embed._mem_history_end`` is set to the
      maximum across the active batch so historical MEM markers in any program
      get MEM_STORE injection. (This is conservative; a batch with mixed
      mem-store lengths may inject MEM_STORE flags onto a few padding-region
      MEM tokens. For Phase 1/2/7 tests this is harmless because the active
      programs are functionally homogeneous.)

Output equivalence:
    For each batch element, the produced ``(output_string, exit_code)`` must
    match the serial pure-neural runner running that program alone. See
    ``test_batched_pure_neural.py`` for element-by-element verification.
"""

from __future__ import annotations

import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .vm_step import Token
from .embedding import Opcode
from .constants import INSTR_WIDTH, PC_OFFSET
from .run_vm import (
    AutoregressiveVMRunner,
    _MEM_STORE_OPS,
)


@dataclass
class _ElementState:
    """Per-batch-element runtime state. Mirrors the relevant fields on
    ``AutoregressiveVMRunner`` (the serial runner) that the pure_neural
    dispatch branch reads/writes."""

    bytecode: List[int]
    context: List[int]
    prefix_len: int
    output: List[str] = field(default_factory=list)
    halted: bool = False
    exit_code: int = 0

    last_pc: Optional[int] = None
    last_ax: int = 0
    last_sp: int = 0x10000
    last_bp: int = 0x10000

    memory: dict = field(default_factory=dict)        # addr -> byte
    mem_history: dict = field(default_factory=dict)   # addr -> 9-token MEM section
    mem_access_order: list = field(default_factory=list)
    mem_history_end: int = 0  # boundary in current context for MEM_STORE injection

    stdin_buffer: list = field(default_factory=list)
    stdin_pos: int = 0

    token_pos: int = 0  # number of tokens generated since context start

    def exec_pc(self) -> int:
        return PC_OFFSET if self.last_pc is None else self.last_pc


class BatchedPureNeuralRunner:
    """Run N pure-neural programs through one shared model in lockstep.

    Usage:
        runner = BatchedPureNeuralRunner()
        results = runner.run_batch(list_of_bytecodes)
        for output, exit_code in results: ...

    The model build is expensive (~3-7s), so callers should hold the runner
    across many ``run_batch`` invocations (e.g. session-scoped pytest fixture).
    """

    def __init__(
        self,
        model_runner: Optional[AutoregressiveVMRunner] = None,
        *,
        d_model=None,
        n_layers=None,
        n_heads=8,
        ffn_hidden=4096,
        max_seq_len=4096,
    ):
        if model_runner is None:
            model_runner = AutoregressiveVMRunner(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                ffn_hidden=ffn_hidden,
                max_seq_len=max_seq_len,
                pure_neural=True,
                trust_neural_alu=True,
            )
            model_runner._func_call_handlers = {}
            model_runner._syscall_handlers = {}
        self._serial = model_runner
        self.model = model_runner.model
        self._device = next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def _build_element(
        self,
        bytecode: List[int],
        data: bytes,
        argv: List[str],
        stdin: str,
    ) -> _ElementState:
        ctx = self._serial._build_context(bytecode, data or b"", argv or [], stdin or "")
        st = _ElementState(
            bytecode=list(bytecode),
            context=list(ctx),
            prefix_len=len(ctx),
            stdin_buffer=list(stdin) if stdin else [],
        )
        if isinstance(data, (bytes, bytearray)):
            for i, b in enumerate(data):
                st.memory[0x10000 + i] = b
        elif isinstance(data, list):
            for i, b in enumerate(data):
                st.memory[0x10000 + i] = b
        return st

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_batch(
        self,
        bytecodes: List[List[int]],
        *,
        data_list: Optional[List[bytes]] = None,
        argv_list: Optional[List[List[str]]] = None,
        stdin_list: Optional[List[str]] = None,
        max_steps: int = 100,
        max_context_window: int = 512,
    ) -> List[Tuple[str, int]]:
        """Run N programs in lockstep.

        Args:
            bytecodes: list of bytecode programs (each a list of packed ints).
            data_list: optional per-program data section bytes.
            argv_list: optional per-program argv list.
            stdin_list: optional per-program stdin string.
            max_steps: maximum VM steps (35 tokens each) per program.
            max_context_window: tail context window passed to the model.

        Returns:
            list of (output_string, exit_code) per program, in the same order
            as ``bytecodes``.
        """
        B = len(bytecodes)
        if B == 0:
            return []
        data_list = data_list or [b""] * B
        argv_list = argv_list or [[]] * B
        stdin_list = stdin_list or [""] * B

        states = [
            self._build_element(bc, data_list[i], argv_list[i], stdin_list[i])
            for i, bc in enumerate(bytecodes)
        ]

        # Total tokens to generate per program (matches serial loop).
        total_tokens = max_steps * Token.STEP_TOKENS

        for tok_i in range(total_tokens):
            active_idx = [i for i, s in enumerate(states) if not s.halted]
            if not active_idx:
                break

            # Build padded input tensor over active programs.
            # We slice the tail (windowed) context for each program to match
            # the serial runner's behavior of feeding only the last
            # max_context_window tokens to the model.
            windowed = []
            for i in active_idx:
                s = states[i]
                if len(s.context) > s.prefix_len + max_context_window:
                    win = s.context[:s.prefix_len] + s.context[-max_context_window:]
                else:
                    win = s.context
                windowed.append(win)

            # Pad to max length with HALT (276 is the safest sentinel — HALT=263
            # is not a token the model is meant to attend to, but it lives in
            # vocab. Padding is on the right, after the real tokens, so causal
            # attention masks it from the real positions we read out).
            max_len = max(len(w) for w in windowed)
            B_active = len(active_idx)
            padded = torch.full(
                (B_active, max_len),
                Token.HALT,
                dtype=torch.long,
                device=self._device,
            )
            real_lens = [len(w) for w in windowed]
            for b, w in enumerate(windowed):
                padded[b, : len(w)] = torch.tensor(
                    w, dtype=torch.long, device=self._device
                )

            # MEM_STORE injection boundary: use max across active programs.
            # If a program has mem_history_end=K it expects MEM tokens up to
            # position K-1 in its context to be flagged. Setting the model's
            # boundary to max(mem_history_end) over-injects on some elements
            # but the affected positions are inside that element's own MEM
            # history, which is the desired behavior.
            mh_end = max(s.mem_history_end for s in states)
            self.model.embed.set_mem_history_end(mh_end)

            # Forward pass and pick last-real-position logits per element.
            logits = self.model.forward(padded)  # [B_active, max_len, V]
            for b, i in enumerate(active_idx):
                last_pos = real_lens[b] - 1
                next_tok = int(logits[b, last_pos, :].argmax(-1).item())
                self._step_one(states[i], next_tok, tok_i)

        # Build results, decoding exit codes for any program that never
        # emitted HALT (matches serial: _decode_exit_code reads the last AX).
        results = []
        for s in states:
            if not s.halted:
                s.exit_code = self._decode_exit_code(s.context)
            results.append(("".join(s.output), s.exit_code))
        return results

    # ------------------------------------------------------------------
    # Per-element step (pure_neural dispatch, simplified)
    # ------------------------------------------------------------------

    def _step_one(self, s: _ElementState, next_token: int, tok_i: int) -> None:
        """Append ``next_token`` to ``s.context`` and run pure_neural dispatch
        if it is a STEP_END / TOOL_CALL / HALT boundary."""
        s.context.append(next_token)
        s.token_pos += 1

        if next_token == Token.HALT:
            s.exit_code = self._decode_exit_code(s.context)
            s.halted = True
            return

        if next_token == Token.STEP_END or next_token == Token.TOOL_CALL:
            self._dispatch_pure_neural(s)
            return

    # ------------------------------------------------------------------
    # Pure-neural dispatch: mirrors the pure_neural branch in
    # AutoregressiveVMRunner._dispatch_step.
    # ------------------------------------------------------------------

    def _dispatch_pure_neural(self, s: _ElementState) -> None:
        exec_pc = s.exec_pc()
        exec_idx = exec_pc // INSTR_WIDTH
        if not (0 <= exec_idx < len(s.bytecode)):
            return
        exec_op = s.bytecode[exec_idx] & 0xFF

        # Update tracking registers from neural outputs.
        neural_pc = self._extract_register(s.context, Token.REG_PC)
        neural_ax = self._extract_register(s.context, Token.REG_AX)
        neural_sp = self._extract_register(s.context, Token.REG_SP)
        neural_bp = self._extract_register(s.context, Token.REG_BP)
        if neural_pc is not None:
            s.last_pc = neural_pc
        if neural_ax is not None:
            s.last_ax = neural_ax
        if neural_sp is not None:
            s.last_sp = neural_sp
        if neural_bp is not None:
            s.last_bp = neural_bp

        # PUTCHAR: append AX byte 0 to output.
        if exec_op == Opcode.PUTCHAR and neural_ax is not None:
            s.output.append(chr(neural_ax & 0xFF))

        # GETCHAR: runner-side stdin injection. We have to overwrite REG_AX
        # bytes in the just-completed step.
        if exec_op == Opcode.GETCHAR:
            byte_val = -1
            if s.stdin_pos < len(s.stdin_buffer):
                byte_val = ord(s.stdin_buffer[s.stdin_pos]) & 0xFF
                s.stdin_pos += 1
            else:
                byte_val = 0xFFFFFFFF
            self._override_register_in_last_step(s.context, Token.REG_AX, byte_val)
            s.last_ax = byte_val

        # PRTF / OPEN / CLOS / READ: defer to serial runner for shim parity.
        if exec_op in (Opcode.PRTF, Opcode.OPEN, Opcode.CLOS, Opcode.READ):
            # These shims touch serial-runner state (e.g. _stdin_pos, _heap_*,
            # _open_fds, _memory). For batched mode in Phase 1/2/7 these
            # opcodes aren't exercised in the tested phases. If they appear
            # we mirror the serial state onto the element and call through;
            # this keeps the runner usable while explicitly not equivalent in
            # multi-program batches that mix I/O. Callers should use the
            # serial runner for such phases.
            self._borrow_serial_state(s)
            if exec_op == Opcode.PRTF:
                self._serial._neural_prtf_emit(
                    s.context, s.output, exec_idx, s.bytecode
                )
            elif exec_op == Opcode.OPEN:
                self._serial._neural_open_emit(s.context)
            elif exec_op == Opcode.CLOS:
                self._serial._neural_clos_emit(s.context)
            elif exec_op == Opcode.READ:
                self._serial._neural_read_emit(s.context)
            self._unborrow_serial_state(s)

        # MEM persistence shim for store ops.
        if exec_op in _MEM_STORE_OPS:
            mem_section = self._extract_mem_section(s.context)
            if mem_section is not None:
                addr = sum((mem_section[1 + j] & 0xFF) << (j * 8) for j in range(4))
                value = sum((mem_section[5 + j] & 0xFF) << (j * 8) for j in range(4))
                for j in range(4):
                    s.memory[(addr + j) & 0xFFFFFFFF] = (value >> (j * 8)) & 0xFF
                self._track_mem_access(s, addr, mem_section)
                # Rebuild context: prefix + mem_history + last_step.
                last_step = s.context[-Token.STEP_TOKENS :]
                mem_flat = []
                for tokens in s.mem_history.values():
                    mem_flat.extend(tokens)
                s.context[s.prefix_len :] = mem_flat + list(last_step)
                s.mem_history_end = s.prefix_len + len(mem_flat)

        if exec_op == Opcode.EXIT:
            # In serial, EXIT is detected and the loop breaks; HALT is the
            # final token. Match that: mark halted but keep the model run
            # until it emits HALT naturally. Here we mark halted now to stop
            # taking further forwards on this element (avoids divergence from
            # a serial run where the loop also exits).
            s.exit_code = self._decode_exit_code(s.context)
            s.halted = True

    # ------------------------------------------------------------------
    # Helpers reused from serial runner (small enough to inline)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_register(context, marker_token) -> Optional[int]:
        scan_back = Token.STEP_TOKENS + 5
        for i in range(len(context) - 1, max(0, len(context) - scan_back), -1):
            if context[i] == marker_token and i + 4 < len(context):
                val = 0
                for j in range(4):
                    val |= (context[i + 1 + j] & 0xFF) << (j * 8)
                return val
        return None

    @staticmethod
    def _override_register_in_last_step(context, marker_token, value) -> None:
        scan_back = Token.STEP_TOKENS + 5
        for i in range(len(context) - 1, max(0, len(context) - scan_back), -1):
            if context[i] == marker_token and i + 4 < len(context):
                for j in range(4):
                    context[i + 1 + j] = (value >> (j * 8)) & 0xFF
                return

    @staticmethod
    def _extract_mem_section(context) -> Optional[list]:
        # MEM section is 9 tokens: MEM + 4 addr + 4 value, near end of step.
        scan_back = Token.STEP_TOKENS + 5
        for i in range(len(context) - 1, max(0, len(context) - scan_back), -1):
            if context[i] == Token.MEM and i + 8 < len(context):
                return list(context[i : i + 9])
        return None

    @staticmethod
    def _decode_exit_code(context) -> int:
        for i in range(len(context) - 1, -1, -1):
            if context[i] == Token.REG_AX and i + 4 < len(context):
                val = 0
                for j in range(4):
                    val |= (context[i + 1 + j] & 0xFF) << (j * 8)
                return val
        return 0

    @staticmethod
    def _track_mem_access(s: _ElementState, addr: int, mem_section: list) -> None:
        max_history = 64
        if addr in s.mem_history:
            s.mem_access_order.remove(addr)
        s.mem_history[addr] = mem_section
        s.mem_access_order.append(addr)
        while len(s.mem_access_order) > max_history:
            evict_addr = s.mem_access_order.pop(0)
            del s.mem_history[evict_addr]

    # ------------------------------------------------------------------
    # Serial-state borrow helpers (used by PRTF/OPEN/CLOS/READ shims)
    # ------------------------------------------------------------------

    def _borrow_serial_state(self, s: _ElementState) -> None:
        ser = self._serial
        self._saved_serial = {
            "_bytecode": getattr(ser, "_bytecode", None),
            "_memory": getattr(ser, "_memory", None),
            "_stdin_buffer": getattr(ser, "_stdin_buffer", []),
            "_stdin_pos": getattr(ser, "_stdin_pos", 0),
            "_last_pc": getattr(ser, "_last_pc", None),
            "_last_ax": getattr(ser, "_last_ax", 0),
            "_last_sp": getattr(ser, "_last_sp", 0x10000),
            "_last_bp": getattr(ser, "_last_bp", 0x10000),
        }
        ser._bytecode = s.bytecode
        ser._memory = s.memory
        ser._stdin_buffer = s.stdin_buffer
        ser._stdin_pos = s.stdin_pos
        ser._last_pc = s.last_pc
        ser._last_ax = s.last_ax
        ser._last_sp = s.last_sp
        ser._last_bp = s.last_bp

    def _unborrow_serial_state(self, s: _ElementState) -> None:
        ser = self._serial
        # Read back any state the serial shim may have mutated on the element.
        s.stdin_pos = ser._stdin_pos
        s.last_pc = ser._last_pc
        s.last_ax = ser._last_ax
        s.last_sp = ser._last_sp
        s.last_bp = ser._last_bp
        for k, v in self._saved_serial.items():
            setattr(ser, k, v)
        self._saved_serial = None
