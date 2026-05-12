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
from collections import deque
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .vm_step import Token
from .embedding import Opcode
from .constants import INSTR_WIDTH, PC_OFFSET
from .run_vm import (
    AutoregressiveVMRunner,
    _MEM_STORE_OPS,
)
from .speculative import DraftVM


# Adaptive speculative-K tuning knobs. The runner accepts ``spec_k <= 0`` as a
# sentinel that enables adaptive mode: each element tracks its own
# ``adaptive_k`` starting at ``_ADAPTIVE_START_K`` and a rolling rejection
# rate over the last ``_ADAPTIVE_WINDOW`` batched-forward iterations. On high
# rejection (>= ``_ADAPTIVE_BACKOFF_THRESHOLD``) the per-element K halves
# (floor ``_ADAPTIVE_MIN_K``); on consistent acceptance
# (<= ``_ADAPTIVE_RAMP_THRESHOLD``) it doubles (cap ``_ADAPTIVE_MAX_K``).
# Per-element bookkeeping lets a batch that mixes programs (some perfectly
# predicted, some flaky) auto-tune each slot independently.
_ADAPTIVE_START_K = 32
_ADAPTIVE_MIN_K = 1
_ADAPTIVE_MAX_K = 64
_ADAPTIVE_WINDOW = 10
_ADAPTIVE_BACKOFF_THRESHOLD = 0.5
_ADAPTIVE_RAMP_THRESHOLD = 0.1


# Step-relative offsets where the speculative-mode logit disagrees with the
# unspeculative-mode logit. See the speculation comment in
# ``BatchedPureNeuralRunner._run_speculative`` for the derivation. These are
# the 4 MEM-addr bytes (offsets 26..29) and 4 MEM-val bytes (offsets 30..33)
# inside each 35-token VM step: the embedding's MEM_STORE/ADDR_KEY injection
# only kicks in when the *full* 9-token MEM section is present, which makes
# spec-mode (full section visible) and unspec-mode (val bytes not yet emitted)
# disagree at these positions. DraftVM is trusted for these offsets — its
# values are derived from synchronized register state and match the model's
# would-be emission for any correct C4 program.
_UNSAFE_OFFSETS = frozenset(range(26, 34))


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

    # Speculative-decoding helper. `draft_vm` is a per-element DraftVM (Python
    # C4 interpreter) that emits the deterministic next-N tokens. Set when
    # `spec_k > 0`; left None when speculation is disabled.
    draft_vm: Optional[DraftVM] = None
    # `spec_disabled` is flipped True for an element if its DraftVM has drifted
    # from the model's emission far enough that we stop speculating on it
    # (avoids wasted forwards over guaranteed-rejected drafts).
    spec_disabled: bool = False
    # Count of consecutive iterations where the model rejected the very first
    # draft token. Used to flip ``spec_disabled`` after 4 in a row.
    spec_zero_streak: int = 0

    # Adaptive spec-K state. ``adaptive_k`` is the current per-element draft
    # horizon in VM steps. ``recent_rejections`` is a rolling window of
    # rejection-rate samples (one per batched forward where this element was
    # speculated): each sample is in [0.0, 1.0] = (rejected_tokens /
    # drafted_tokens). The window is bounded by ``_ADAPTIVE_WINDOW`` (oldest
    # samples drop off automatically). When ``adaptive_k`` is 0 the element
    # is in non-adaptive mode (legacy fixed-K path).
    adaptive_k: int = 0
    recent_rejections: deque = field(
        default_factory=lambda: deque(maxlen=_ADAPTIVE_WINDOW)
    )

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
        spec_k: int = 0,
        adaptive_start_k: int = 0,
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
        if adaptive_start_k > 0:
            st.adaptive_k = adaptive_start_k
        if spec_k > 0 or adaptive_start_k > 0:
            st.draft_vm = DraftVM(list(bytecode))
            # Load data section into DraftVM memory so reads from the data
            # segment behave correctly. DraftVM stores 32-bit values per addr;
            # we use byte-addressed entries to match `s.memory`'s convention.
            if isinstance(data, (bytes, bytearray, list)):
                for i, b in enumerate(data):
                    st.draft_vm.memory[0x10000 + i] = int(b)
            if stdin:
                st.draft_vm.set_stdin(stdin)
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
        spec_k: int = 0,
    ) -> List[Tuple[str, int]]:
        """Run N programs in lockstep.

        Args:
            bytecodes: list of bytecode programs (each a list of packed ints).
            data_list: optional per-program data section bytes.
            argv_list: optional per-program argv list.
            stdin_list: optional per-program stdin string.
            max_steps: maximum VM steps (35 tokens each) per program.
            max_context_window: tail context window passed to the model.
            spec_k: number of full VM steps to speculate per batched forward.
                ``0`` disables speculation entirely and runs one token per
                forward (the legacy non-speculative batched behavior). A
                **negative** value (e.g. ``-1``) is a sentinel that enables
                **adaptive** per-element speculation: each batch slot starts
                at ``_ADAPTIVE_START_K`` and dynamically halves / doubles its
                horizon based on its rolling rejection rate over the last
                ``_ADAPTIVE_WINDOW`` batched forwards. A **positive** value
                runs every element at that fixed K (no adaptation). When
                speculation is enabled, a per-element :class:`DraftVM` emits
                the deterministic next ``K * 35`` tokens; the model verifies
                them in one forward pass and accepts the matching prefix.
                Byte-identity with ``spec_k=0`` is preserved: a model
                emission overrides any mismatching draft, so the model is
                the final arbiter. Recommended: ``spec_k=-1`` (adaptive)
                for mixed workloads; ``spec_k=32`` for known-clean batches.

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

        # Adaptive mode: ``spec_k < 0`` is the sentinel. ``spec_k == 0`` keeps
        # the legacy non-speculative path; ``spec_k > 0`` keeps the legacy
        # fixed-K path. Adaptive mode reuses the same speculative loop but each
        # element carries its own ``adaptive_k`` that is updated post-forward.
        adaptive = spec_k < 0
        adaptive_start_k = _ADAPTIVE_START_K if adaptive else 0

        states = [
            self._build_element(
                bc,
                data_list[i],
                argv_list[i],
                stdin_list[i],
                spec_k=spec_k if not adaptive else 0,
                adaptive_start_k=adaptive_start_k,
            )
            for i, bc in enumerate(bytecodes)
        ]

        if adaptive:
            self._run_speculative(
                states,
                max_steps=max_steps,
                max_context_window=max_context_window,
                spec_k=0,
                adaptive=True,
            )
        elif spec_k > 0:
            self._run_speculative(
                states,
                max_steps=max_steps,
                max_context_window=max_context_window,
                spec_k=spec_k,
                adaptive=False,
            )
        else:
            self._run_unspeculative(
                states,
                max_steps=max_steps,
                max_context_window=max_context_window,
            )

        # Build results, decoding exit codes for any program that never
        # emitted HALT (matches serial: _decode_exit_code reads the last AX).
        results = []
        for s in states:
            if not s.halted:
                s.exit_code = self._decode_exit_code(s.context)
            results.append(("".join(s.output), s.exit_code))
        return results

    # ------------------------------------------------------------------
    # Non-speculative inner loop (legacy: one token per batched forward)
    # ------------------------------------------------------------------

    def _run_unspeculative(
        self,
        states: List[_ElementState],
        *,
        max_steps: int,
        max_context_window: int,
    ) -> None:
        total_tokens = max_steps * Token.STEP_TOKENS

        for tok_i in range(total_tokens):
            active_idx = [i for i, s in enumerate(states) if not s.halted]
            if not active_idx:
                break

            windowed = [
                self._windowed_context(states[i], max_context_window)
                for i in active_idx
            ]
            padded, real_lens = self._pad_to_tensor(windowed)

            mh_end = max(s.mem_history_end for s in states)
            self.model.embed.set_mem_history_end(mh_end)

            logits = self.model.forward(padded)  # [B_active, max_len, V]
            # Batch argmax+CPU transfer once per outer iter rather than per
            # element to avoid per-element GPU→CPU sync stalls.
            preds_cpu = logits.argmax(dim=-1).cpu().tolist()
            for b, i in enumerate(active_idx):
                last_pos = real_lens[b] - 1
                next_tok = int(preds_cpu[b][last_pos])
                self._step_one(states[i], next_tok, tok_i)

    # ------------------------------------------------------------------
    # Speculative inner loop: DraftVM proposes K*35 tokens per element,
    # model verifies them in one batched forward pass, accepted prefix is
    # replayed through ``_step_one`` (so STEP_END dispatch, mem_history
    # rebuilds, EXIT halting all behave identically to the unspeculative
    # path).
    # ------------------------------------------------------------------

    def _run_speculative(
        self,
        states: List[_ElementState],
        *,
        max_steps: int,
        max_context_window: int,
        spec_k: int,
        adaptive: bool = False,
    ) -> None:
        total_tokens = max_steps * Token.STEP_TOKENS
        STEP = Token.STEP_TOKENS

        # Maximum sequence length the model accepts. We cap per-element K so
        # the windowed context + drafts never overflows. The cap is also
        # bounded by ``max_context_window`` (the tail kept after windowing).
        # If the model exposes ``max_seq_len`` we use it; otherwise we fall
        # back to ``max_context_window`` (which is always a hard ceiling on
        # the forward-input length anyway).
        model_max_seq = int(getattr(self.model, "max_seq_len", max_context_window))

        # Global token counter used as the `tok_i` argument to `_step_one`.
        # It must monotonically advance across replayed tokens so the legacy
        # contract (loop iteration count) is preserved.
        tok_i = 0
        while tok_i < total_tokens:
            active_idx = [i for i, s in enumerate(states) if not s.halted]
            if not active_idx:
                break

            # 1) Build per-element draft sequences. Elements with speculation
            #    disabled (or mid-step from a previous iter's rejection) get
            #    an empty draft -> single-token decode for them this
            #    iteration. Elements whose DraftVM halts mid-speculation stop
            #    adding more drafts but keep the prefix.
            drafts = []  # list[list[int]] aligned with active_idx
            per_elem_k = []  # K actually used this iter, aligned with active_idx
            for i in active_idx:
                s = states[i]
                d: List[int] = []
                # Only speculate when the element is at a clean VM-step
                # boundary. If a previous iter's verification rejected
                # mid-step, the context now contains a partial step; we'd
                # need to splice DraftVM output around it, which is fragile.
                # Falling back to single-token decode until the next STEP_END
                # is simpler and only costs at most 35 forwards per
                # rejection event.
                at_step_boundary = (s.token_pos % STEP == 0)
                # Pick K for this element this iteration:
                #   * Adaptive mode reads ``s.adaptive_k`` (updated below).
                #   * Fixed mode uses the global ``spec_k``.
                # Then cap by available context room: drafts append to the
                # windowed context, and the forward must fit under
                # ``model_max_seq``. The windowed context length is bounded
                # above by ``prefix_len + max_context_window`` (see
                # ``_windowed_context``).
                base_k = s.adaptive_k if adaptive else spec_k
                # Estimate the windowed-context length to compute room.
                est_ctx_len = min(
                    len(s.context),
                    s.prefix_len + max_context_window,
                )
                room_tokens = max(0, model_max_seq - est_ctx_len)
                room_k = room_tokens // STEP
                effective_k = max(0, min(base_k, room_k))
                if (
                    not s.spec_disabled
                    and s.draft_vm is not None
                    and at_step_boundary
                    and effective_k > 0
                ):
                    # Sync DraftVM register state from the element. Registers
                    # are the dominant correctness signal; sparse memory
                    # drift only causes extra rejections (not divergence —
                    # the model remains the source of truth).
                    self._sync_draft_vm(s)
                    for _ in range(effective_k):
                        if s.draft_vm.halted:
                            break
                        ok = s.draft_vm.step()
                        if not ok:
                            break
                        d.extend(s.draft_vm.draft_tokens())
                        if s.draft_vm.halted:
                            break
                drafts.append(d)
                per_elem_k.append(effective_k)

            # 2) Build padded tensor: each active element gets windowed context
            #    + its draft tokens appended. Padded to the max combined length.
            windowed_with_drafts = []
            real_prefix_lens = []  # len of context prefix per element (no draft)
            for k, i in enumerate(active_idx):
                s = states[i]
                ctx_win = self._windowed_context(s, max_context_window)
                real_prefix_lens.append(len(ctx_win))
                windowed_with_drafts.append(ctx_win + drafts[k])

            padded, _ = self._pad_to_tensor(windowed_with_drafts)

            mh_end = max(s.mem_history_end for s in states)
            self.model.embed.set_mem_history_end(mh_end)

            logits = self.model.forward(padded)  # [B_active, max_len, V]

            # Pre-compute argmax for the entire tensor once (GPU op) and move
            # to CPU in a single transfer. Per-position .item() calls inside
            # the verification loop force one GPU→CPU sync each, which made
            # the speculative path slower than non-speculative on small
            # batches. Batching the argmax+move avoids that.
            preds_cpu = logits.argmax(dim=-1).cpu().tolist()  # [B_active][max_len]

            # 3) For each element: verify drafts and replay accepted prefix + correction.
            for k, i in enumerate(active_idx):
                s = states[i]
                if s.halted:
                    continue
                draft = drafts[k]
                prefix_len = real_prefix_lens[k]

                if len(draft) == 0:
                    # Pure single-token decode for this element (no spec budget).
                    last_pos = prefix_len - 1
                    next_tok = int(preds_cpu[k][last_pos])
                    self._step_one(s, next_tok, tok_i)
                    continue

                # Verify each draft slot. Offsets in `_UNSAFE_OFFSETS` are
                # trusted from DraftVM (the embedding's MEM metadata injection
                # makes them disagree with the unspec-mode model — see the
                # module-level constant docstring); other offsets are verified
                # against the model's argmax. First rejected safe offset stops
                # the scan and the model's prediction becomes the correction.
                accepted = 0
                correction: Optional[int] = None
                row_preds = preds_cpu[k]
                for j in range(len(draft)):
                    if (j % STEP) in _UNSAFE_OFFSETS:
                        accepted = j + 1
                        continue
                    pred = row_preds[prefix_len - 1 + j]
                    if pred == draft[j]:
                        accepted = j + 1
                    else:
                        correction = pred
                        break

                # Replay accepted tokens through _step_one, then the
                # correction (if any). _step_one mutates s.context so the
                # dispatch path (STEP_END mem_history rebuild, EXIT halting)
                # stays identical to the unspeculative loop.
                #
                # We deliberately do NOT take the "free bonus" emission at
                # position prefix-1+len(draft) when all drafts accept: that
                # would advance the element by exactly one extra token,
                # landing it mid-step (token_pos % STEP_TOKENS != 0), which
                # disables speculation on the next iter (see the
                # ``at_step_boundary`` gate above). The follow-up
                # single-token-decode tax — STEP_TOKENS - 1 forwards to
                # re-align — wipes out the bonus's gain on any non-halting
                # program.
                emitted: List[int] = list(draft[:accepted])
                if correction is not None:
                    emitted.append(correction)

                for tok in emitted:
                    if s.halted:
                        break
                    self._step_one(s, tok, tok_i)

                # If the model rejected the very first draft token for several
                # iterations in a row, the DraftVM is clearly out of sync (or
                # the model never matches its drafts for this program). Disable
                # speculation for this element to avoid wasted forwards.
                if correction is not None and accepted == 0:
                    s.spec_zero_streak += 1
                    if s.spec_zero_streak >= 4:
                        s.spec_disabled = True
                else:
                    s.spec_zero_streak = 0

                # Adaptive K bookkeeping. Record rejection rate for this
                # iteration (rejected_tokens / drafted_tokens) and adjust K
                # only when the window has at least a few samples (avoids
                # over-reacting to a single iter). High rejection rate halves
                # K (with a floor), low rate doubles it (with a ceiling).
                if adaptive and len(draft) > 0:
                    rej = (len(draft) - accepted) / len(draft)
                    s.recent_rejections.append(rej)
                    if len(s.recent_rejections) >= 3:
                        avg_rej = sum(s.recent_rejections) / len(s.recent_rejections)
                        if avg_rej >= _ADAPTIVE_BACKOFF_THRESHOLD:
                            new_k = max(_ADAPTIVE_MIN_K, s.adaptive_k // 2)
                            if new_k != s.adaptive_k:
                                s.adaptive_k = new_k
                                s.recent_rejections.clear()
                        elif avg_rej <= _ADAPTIVE_RAMP_THRESHOLD:
                            new_k = min(_ADAPTIVE_MAX_K, s.adaptive_k * 2)
                            if new_k != s.adaptive_k:
                                s.adaptive_k = new_k
                                s.recent_rejections.clear()

            tok_i += 1  # outer-iteration counter, not strict token count

    # ------------------------------------------------------------------
    # Speculation helpers
    # ------------------------------------------------------------------

    def _sync_draft_vm(self, s: _ElementState) -> None:
        """Pull register state from the element into its DraftVM.

        Also clear `halted` so a DraftVM that previously hit EXIT can be
        replayed from a fresh point (the element may still be running
        because the model has not yet emitted HALT). Memory is not
        transplanted: ``s.memory`` is byte-addressed while
        ``DraftVM.memory`` is word-addressed. Register drift is the dominant
        rejection driver; memory drift only causes extra rejections, which
        gracefully fall back to single-token decode for affected elements.
        """
        vm = s.draft_vm
        if vm is None:
            return
        if s.last_pc is not None:
            vm.pc = int(s.last_pc) & 0xFFFFFFFF
            # idx is computed by step() from pc when fetching from static
            # code; keep it in sync for direct-code-array fetches.
            vm.idx = (vm.pc - PC_OFFSET) // INSTR_WIDTH
        vm.ax = int(s.last_ax) & 0xFFFFFFFF
        vm.sp = int(s.last_sp) & 0xFFFFFFFF
        vm.bp = int(s.last_bp) & 0xFFFFFFFF
        vm.halted = False
        vm._last_mem_addr = 0
        vm._last_mem_val = 0

    def _windowed_context(
        self, s: _ElementState, max_context_window: int
    ) -> List[int]:
        if len(s.context) > s.prefix_len + max_context_window:
            return s.context[: s.prefix_len] + s.context[-max_context_window:]
        return s.context

    def _pad_to_tensor(
        self, sequences: List[List[int]]
    ) -> Tuple[torch.Tensor, List[int]]:
        """Right-pad a list of token sequences to a single ``[B, max_len]``
        tensor on ``self._device``. Padding token is ``Token.HALT`` (matches
        legacy behavior). Returns ``(tensor, real_lens)``."""
        real_lens = [len(w) for w in sequences]
        max_len = max(real_lens)
        B = len(sequences)
        # Build the padded matrix on CPU (one allocation, one host→device
        # copy) — issuing a per-row ``padded[b, :n] = torch.tensor(...)``
        # would trigger B separate H2D transfers per outer iter, which
        # showed up in profiles as the dominant cost on small batches.
        padded_host = torch.full((B, max_len), Token.HALT, dtype=torch.long)
        for b, (w, n) in enumerate(zip(sequences, real_lens)):
            padded_host[b, :n] = torch.tensor(w, dtype=torch.long)
        return padded_host.to(self._device, non_blocking=True), real_lens

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
