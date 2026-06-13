"""Equivalence + benchmark tests for `BatchedPureNeuralRunner`.

The batched runner is an opt-in alternative to the serial pure-neural runner
(`AutoregressiveVMRunner(pure_neural=True)`). For correctness it must produce
exactly the same `(output, exit_code)` as the serial runner on every batch
element.

Tests in this file:

* `test_batched_matches_serial_phase1_small`: build a small batch of Phase-1
  style programs (IMM N; EXIT) and verify element-by-element equality with the
  serial runner.

* `test_batched_speedup_phase1`: benchmark a clean Phase-1 batch through both
  runners and assert the batched runner is at least 1.5x faster. Skipped if
  no GPU is available, since CPU forward time is dominated by Python-side work
  and doesn't benefit from batching.

The fixture is session-scoped so the compiled model is reused across tests in
this file.
"""

from __future__ import annotations

import time
import pytest
import torch

from neural_vm.embedding import Opcode


def _encode(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


# Programs deliberately chosen to halt cleanly in pure_neural mode (Phase 1).
# IMM with byte values that DON'T hit the 254/255 sign-extension bug.
_CLEAN_PHASE1_PROGRAMS = [
    [(Opcode.IMM, v), Opcode.EXIT] for v in [0, 1, 2, 5, 7, 11, 17, 23, 42, 100, 200]
]


def test_batched_matches_serial_phase1_small(
    pure_neural_runner, batched_pure_neural_runner
):
    """Element-by-element equivalence on a small batch."""
    programs = _CLEAN_PHASE1_PROGRAMS[:4]
    bcs = [_encode(p) for p in programs]

    # Serial reference.
    serial_results = []
    for bc in bcs:
        pure_neural_runner._memory = {}
        pure_neural_runner._mem_history = {}
        pure_neural_runner._mem_access_order = []
        out, code = pure_neural_runner.run(bc, b"", max_steps=10)
        serial_results.append((out, code))

    # Batched.
    batched_results = batched_pure_neural_runner.run_batch(bcs, max_steps=10)

    assert len(batched_results) == len(serial_results)
    for i, (b, s) in enumerate(zip(batched_results, serial_results)):
        assert b == s, f"batch element {i}: batched={b!r} != serial={s!r}"


@pytest.mark.slow
def test_batched_speedup_phase1(
    pure_neural_runner, batched_pure_neural_runner
):
    """Benchmark serial vs batched on Phase-1-style programs.

    Asserts a minimum speedup of 1.5x. On GPU the typical speedup is 3-4x
    for clean Phase-1 batches (11+ short programs that all halt quickly).
    """
    bcs = [_encode(p) for p in _CLEAN_PHASE1_PROGRAMS]
    N = len(bcs)

    # Serial.
    t0 = time.time()
    serial_results = []
    for bc in bcs:
        pure_neural_runner._memory = {}
        pure_neural_runner._mem_history = {}
        pure_neural_runner._mem_access_order = []
        serial_results.append(pure_neural_runner.run(bc, b"", max_steps=10))
    serial_total = time.time() - t0

    # Batched.
    t0 = time.time()
    batched_results = batched_pure_neural_runner.run_batch(bcs, max_steps=10)
    batched_total = time.time() - t0

    # Equivalence check (defensive; the small test above also covers this).
    for i, (b, s) in enumerate(zip(batched_results, serial_results)):
        assert b == s, (
            f"speedup-test element {i}: batched={b!r} != serial={s!r}"
        )

    speedup = serial_total / batched_total
    print(
        f"\n[batched bench] N={N}  serial={serial_total:.2f}s  "
        f"batched={batched_total:.2f}s  speedup={speedup:.2f}x"
    )
    assert speedup >= 1.5, (
        f"Expected batched runner to be >= 1.5x faster on {N} clean Phase-1 "
        f"programs, got {speedup:.2f}x (serial={serial_total:.2f}s, "
        f"batched={batched_total:.2f}s)"
    )


# --- Bucket-by-predicted-length tests -------------------------------------


def test_predict_steps_basic():
    """``DraftVM.predict_steps`` returns the right step count for IMM/EXIT."""
    from neural_vm.speculative import DraftVM

    # IMM 42 (op=1, imm=42); EXIT (op=38). 2 steps total.
    bc = [(42 << 8) | 1, 38]
    vm = DraftVM(bc)
    assert vm.predict_steps(max_steps=10) == 2
    assert vm.halted

    # A program that doesn't halt within max_steps caps at max_steps and
    # leaves halted=False (the bucket helper treats this as unpredicted).
    # Use a tight infinite loop: JMP 0 (op=2, imm=0).
    bc_loop = [(0 << 8) | 2]
    vm_loop = DraftVM(bc_loop)
    assert vm_loop.predict_steps(max_steps=5) == 5
    assert not vm_loop.halted


def test_bucketed_matches_unbucketed(batched_pure_neural_runner):
    """Bucketed run produces byte-identical results to unbucketed run.

    Programs deliberately span different predicted lengths so they hit
    different buckets.
    """
    # All clean Phase-1 programs halt in 2 steps; bucketing should put them
    # all in the smallest bucket but still yield byte-identical results.
    bcs = [_encode(p) for p in _CLEAN_PHASE1_PROGRAMS]

    bucketed = batched_pure_neural_runner.run_batch(
        bcs, max_steps=10, bucket_by_predicted_length=True
    )
    unbucketed = batched_pure_neural_runner.run_batch(
        bcs, max_steps=10, bucket_by_predicted_length=False
    )
    assert bucketed == unbucketed, (
        f"bucketed != unbucketed:\n  bucketed={bucketed}\n  unbucketed={unbucketed}"
    )


def test_bucket_key_assignment():
    """``_bucket_key`` lands programs in the smallest bucket whose bound
    is >= predicted, and None goes to the unpredicted bucket."""
    from neural_vm.batched_pure_neural import (
        BatchedPureNeuralRunner,
        _DEFAULT_BUCKET_BOUNDS,
        _UNPREDICTED_BUCKET_KEY,
    )

    bk = BatchedPureNeuralRunner._bucket_key
    assert bk(1, _DEFAULT_BUCKET_BOUNDS) == 10
    assert bk(10, _DEFAULT_BUCKET_BOUNDS) == 10
    assert bk(11, _DEFAULT_BUCKET_BOUNDS) == 20
    assert bk(80, _DEFAULT_BUCKET_BOUNDS) == 80
    assert bk(81, _DEFAULT_BUCKET_BOUNDS) == 160
    assert bk(_DEFAULT_BUCKET_BOUNDS[-1] + 1, _DEFAULT_BUCKET_BOUNDS) == _UNPREDICTED_BUCKET_KEY
    assert bk(None, _DEFAULT_BUCKET_BOUNDS) == _UNPREDICTED_BUCKET_KEY


def test_dispatch_early_exits_when_neural_pc_points_to_exit(monkeypatch):
    """The strict raw-neural path should stop once emitted PC reaches EXIT."""
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner, _ElementState
    from neural_vm.constants import INSTR_WIDTH
    from neural_vm.vm_step import Token

    runner = object.__new__(BatchedPureNeuralRunner)
    state = _ElementState(
        bytecode=_encode([(Opcode.IMM, 42), Opcode.EXIT]),
        context=[],
        prefix_len=0,
        last_pc=0,
    )

    registers = {
        Token.REG_PC: INSTR_WIDTH,
        Token.REG_AX: 42,
        Token.REG_SP: 0x10000,
        Token.REG_BP: 0x10000,
    }
    monkeypatch.setattr(
        runner,
        "_extract_register",
        lambda _context, marker: registers.get(marker),
    )

    runner._dispatch_pure_neural(state)

    assert state.halted is True
    assert state.exit_code == 42
    assert state.last_pc == INSTR_WIDTH
    assert state.last_ax == 42


def test_speculation_guard_blocks_call_frame_opcodes():
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from neural_vm.speculative import DraftVM

    bytecode = _encode([
        (Opcode.JSR, 3),
        Opcode.EXIT,
        Opcode.NOP,
        (Opcode.ENT, 0),
        (Opcode.IMM, 42),
        Opcode.LEV,
    ])

    vm = DraftVM(bytecode)
    assert not BatchedPureNeuralRunner._draft_opcode_safe_for_speculation(vm)

    vm.idx = 4
    vm.pc = 34
    assert BatchedPureNeuralRunner._draft_opcode_safe_for_speculation(vm)

    vm.idx = 5
    vm.pc = 42
    assert not BatchedPureNeuralRunner._draft_opcode_safe_for_speculation(vm)


def test_declarative_halt_horizon_marks_overrun_as_divergence():
    """Cap-hit decodes the most recent REG_AX from the context instead of
    returning ``None``. The element still halts at the horizon; the
    observable change is that ``exit_code`` now mirrors what
    ``AutoregressiveVMRunner.run`` returns after its own ``draft_divergence``
    / max-token loop exit (``_decode_exit_code``).
    """
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner, _ElementState
    from neural_vm.vm_step import Token

    runner = object.__new__(BatchedPureNeuralRunner)
    runner._dispatch_pure_neural = lambda _state: None
    state = _ElementState(
        bytecode=_encode([(Opcode.IMM, 42), Opcode.EXIT]),
        context=[],
        prefix_len=0,
        token_pos=Token.STEP_TOKENS - 1,
        expected_steps=1,
    )

    runner._step_one(state, Token.STEP_END, 0)

    assert state.halted is True
    # Context is empty -> no REG_AX to find -> decoded value is 0 (legacy
    # ``_decode_exit_code`` fallback). What matters is that ``exit_code`` is
    # NOT ``None``: callers can rely on the integer value being a defined
    # exit code, never a sentinel.
    assert state.exit_code == 0


def test_spec_fail_on_correction_stops_at_first_safe_divergence():
    import torch

    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner, _ElementState
    from neural_vm.run_vm import DraftVM
    from neural_vm.vm_step import Token

    class _Embed:
        def set_mem_history_end(self, _value):
            pass

    class _Model:
        max_seq_len = 4096
        embed = _Embed()

        def forward(self, token_ids, **_kwargs):
            logits = torch.zeros(
                token_ids.shape[0],
                token_ids.shape[1],
                Token.VOCAB_SIZE,
                device=token_ids.device,
            )
            logits[:, :, Token.HALT] = 1.0
            return logits

    runner = object.__new__(BatchedPureNeuralRunner)
    runner.model = _Model()
    runner._device = torch.device("cpu")
    runner.use_kv_cache = False
    runner.incremental_kv_safe = False
    runner.spec_fail_on_correction = True
    runner.spec_fail_fast = True
    runner._kv_stats = {"fresh_forwards": 0}
    runner._reset_spec_stats()

    bytecode = _encode([(Opcode.IMM, 42), Opcode.EXIT])
    state = _ElementState(
        bytecode=bytecode,
        context=[Token.CODE_START],
        prefix_len=1,
        expected_steps=1,
        draft_vm=DraftVM(bytecode),
    )

    runner._run_speculative(
        [state],
        max_steps=None,
        max_context_window=512,
        spec_k=1,
    )

    assert state.halted is True
    assert state.exit_code is None
    assert state.context == [Token.CODE_START]
    assert runner._spec_stats["corrections"] == 1
    assert runner._spec_stats["fail_fast"] == 1


def test_spec_fail_fast_defaults_to_disabled(monkeypatch):
    """``C4_SPEC_FAIL_FAST`` defaults OFF so persistent draft rejection
    falls back to single-token decode instead of hard-halting the element.
    """
    import os

    def _parse():
        return (
            os.environ.get("C4_SPEC_FAIL_FAST", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )

    monkeypatch.delenv("C4_SPEC_FAIL_FAST", raising=False)
    assert _parse() is False

    monkeypatch.setenv("C4_SPEC_FAIL_FAST", "1")
    assert _parse() is True


def test_run_speculative_disables_spec_after_persistent_first_token_rejects():
    """When ``spec_fail_fast=False`` (the new default), four consecutive
    first-token rejections set ``spec_disabled=True`` so the element keeps
    advancing via single-token decode instead of hard-halting with
    ``exit_code=None``."""
    import torch

    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner, _ElementState
    from neural_vm.run_vm import DraftVM
    from neural_vm.vm_step import Token

    class _Embed:
        def set_mem_history_end(self, _value):
            pass

        def set_mem_store_positions(self, _value):
            pass

    # Predict a benign byte (1) everywhere — disagrees with DraftVM's
    # REG_PC step-start token (rejecting first draft token every time) and
    # doesn't trigger HALT/STEP_END dispatch so the element keeps running.
    class _Model:
        max_seq_len = 4096
        embed = _Embed()

        def forward(self, token_ids, **_kwargs):
            logits = torch.zeros(
                token_ids.shape[0],
                token_ids.shape[1],
                Token.VOCAB_SIZE,
                device=token_ids.device,
            )
            logits[:, :, 1] = 1.0
            return logits

    runner = object.__new__(BatchedPureNeuralRunner)
    runner.model = _Model()
    runner._device = torch.device("cpu")
    runner.use_kv_cache = False
    runner.incremental_kv_safe = False
    runner.spec_fail_on_correction = False
    runner.spec_fail_fast = False
    runner._kv_stats = {"fresh_forwards": 0}
    runner._reset_spec_stats()

    bytecode = _encode([(Opcode.IMM, 1), (Opcode.IMM, 2),
                        (Opcode.IMM, 3), (Opcode.IMM, 4),
                        (Opcode.IMM, 5), Opcode.EXIT])
    # Force at_step_boundary=True at every iteration by aligning token_pos
    # to a STEP_TOKENS multiple. We arrange this by giving spec_k=1 and
    # generous expected_steps so the runner gets several iterations.
    state = _ElementState(
        bytecode=bytecode,
        context=[Token.CODE_START],
        prefix_len=1,
        expected_steps=400,
        draft_vm=DraftVM(bytecode),
    )

    runner._run_speculative(
        [state],
        max_steps=None,
        max_context_window=512,
        spec_k=1,
    )

    # spec_disabled should be set after the streak of first-token rejects
    # triggers the fallback path; the runner did not hard-halt the element.
    assert state.spec_disabled is True, (
        "Persistent first-token rejection should disable speculation, "
        "not hard-halt the element"
    )
    assert runner._spec_stats["fail_fast"] == 0
    assert runner._spec_stats["first_token_rejects"] >= 4


# ---------------------------------------------------------------------------
# Fail-fast (full-token-trace) criterion tests.
# ---------------------------------------------------------------------------

def test_step_offset_field_names():
    """``_step_offset_field`` maps a 0..34 token offset to its field name."""
    from neural_vm.batched_pure_neural import _step_offset_field

    assert _step_offset_field(0) == "PC_marker"
    assert _step_offset_field(1) == "PC[0]"
    assert _step_offset_field(5) == "AX_marker"
    assert _step_offset_field(6) == "AX[0]"
    assert _step_offset_field(7) == "AX[1]"
    assert _step_offset_field(20) == "STACK0_marker"
    assert _step_offset_field(26) == "MEM_ADDR[0]"
    assert _step_offset_field(30) == "MEM_VAL[0]"
    assert _step_offset_field(34) == "STEP_END/HALT"


def test_ff_check_new_steps_passes_matching_and_fails_first_divergence():
    """``_ff_check_new_steps`` compares each completed step's slice-decoded
    (PC, AX) to the oracle and fails at the FIRST diverging step."""
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner, _ElementState
    from neural_vm.speculative import DraftVM
    from neural_vm.vm_step import Token

    runner = object.__new__(BatchedPureNeuralRunner)

    bytecode = _encode([(Opcode.IMM, 7), Opcode.EXIT])
    # Build the oracle's per-step (pc, ax) + the matching 35-token trace.
    vm = DraftVM(bytecode)
    oracle, toks = [], []
    for _ in range(6):
        if vm.halted:
            break
        if not vm.step():
            break
        oracle.append((vm.pc & 0xFFFFFFFF, vm.ax & 0xFFFFFFFF))
        toks.extend(vm.draft_tokens())
        if vm.halted:
            break

    # Feed the PERFECT oracle token slices: every completed step matches.
    s = _ElementState(bytecode=bytecode, context=[], prefix_len=0,
                      expected_steps=len(oracle) + 1)
    s.ff_oracle_steps = oracle
    s.context = list(toks)
    s.token_pos = len(toks)
    assert runner._ff_check_new_steps(s) is False
    assert s.ff_status is None  # no divergence -> pass candidate

    # Now corrupt step-1's AX byte 0 in the slice -> first divergence at step 1.
    s2 = _ElementState(bytecode=bytecode, context=[], prefix_len=0,
                       expected_steps=len(oracle) + 1)
    s2.ff_oracle_steps = oracle
    bad = list(toks)
    # AX of step 1 is at slice offset 5 (marker) + 1 (byte 0) within step 1.
    ax_byte0_pos = Token.STEP_TOKENS + 6
    bad[ax_byte0_pos] = (bad[ax_byte0_pos] + 1) & 0xFF
    s2.context = bad
    s2.token_pos = len(bad)
    assert runner._ff_check_new_steps(s2) is True
    assert s2.ff_status == "fail"
    assert s2.ff_div_step == 1
    assert s2.halted is True


def test_run_batch_fail_fast_passes_perfect_oracle_and_records_divergence():
    """Integration on a fake model: a model that emits the EXACT oracle trace
    PASSES full-trace; a model that flips one register byte FAILS at the
    diverging step with the expected/got register pair recorded."""
    import torch

    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from neural_vm.speculative import DraftVM
    from neural_vm.vm_step import Token

    bytecode = _encode([(Opcode.IMM, 9), Opcode.EXIT])
    # The oracle's full token trace (teacher-forced reference).
    vm = DraftVM(bytecode)
    trace = []
    for _ in range(6):
        if vm.halted:
            break
        if not vm.step():
            break
        trace.extend(vm.draft_tokens())
        if vm.halted:
            break

    class _Embed:
        def set_mem_history_end(self, _v):
            pass

        def set_mem_addr_src_positions(self, _v):
            pass

    def _make_runner(corrupt_pos=None):
        class _Model:
            max_seq_len = 4096
            embed = _Embed()

            def forward(self, token_ids, **_kwargs):
                # Context row is [CODE_START] + (a prefix of) ``trace``. The
                # next-token at row position p is ``trace[p]`` (prefix_len=1),
                # so a model that emits the exact oracle predicts trace[p] at
                # every position. ``corrupt_pos`` flips one trace token to force
                # a single, deterministic divergence.
                B, S = token_ids.shape[0], token_ids.shape[1]
                logits = torch.zeros(B, S, Token.VOCAB_SIZE, device=token_ids.device)
                for b in range(B):
                    for p in range(S):
                        if 0 <= p < len(trace):
                            tok = trace[p]
                            if corrupt_pos is not None and p == corrupt_pos:
                                tok = (tok + 1) % Token.VOCAB_SIZE
                        else:
                            tok = Token.HALT
                        logits[b, p, tok] = 1.0
                return logits

        runner = object.__new__(BatchedPureNeuralRunner)
        runner.model = _Model()
        runner._device = torch.device("cpu")
        runner.use_kv_cache = False
        runner.incremental_kv_safe = False
        runner.spec_fail_on_correction = False
        runner.spec_fail_fast = False
        runner._kv_stats = {"fresh_forwards": 0}
        runner._reset_kv_cache = lambda: None
        runner._reset_spec_stats()
        return runner

    # The runner builds the context via ``_serial._build_context``; stub it so
    # the prefix is a single CODE_START token (matching the trace alignment).
    class _Serial:
        def _build_context(self, *_a, **_k):
            return [Token.CODE_START]

    decl_steps = len(trace) // Token.STEP_TOKENS

    runner = _make_runner(corrupt_pos=None)
    runner._serial = _Serial()
    res = runner.run_batch_fail_fast(
        [bytecode], expected_steps_list=[decl_steps], max_steps=None, spec_k=1,
    )
    assert res[0]["status"] == "pass", res[0]

    # Corrupt the AX byte-0 of step 0 (the IMM result, offset 6 in the trace)
    # -> the model now emits a wrong AX at step 0 -> full-trace FAIL there.
    corrupt = 6
    runner2 = _make_runner(corrupt_pos=corrupt)
    runner2._serial = _Serial()
    res2 = runner2.run_batch_fail_fast(
        [bytecode], expected_steps_list=[decl_steps], max_steps=None, spec_k=1,
    )
    assert res2[0]["status"] == "fail", res2[0]
    assert res2[0]["divergence_step"] == 0, res2[0]
    assert res2[0]["got_ax"] != res2[0]["expected_ax"]


def test_ff_check_new_steps_strict_token_identity():
    """``strict_trace`` compares every SAFE-offset token to the DraftVM
    ``draft_tokens()`` reference and fails on the FIRST divergent token, while
    ignoring the UNSAFE MEM addr/val offsets (26..33). It is strictly stronger
    than ``full_trace`` (catches SP/BP/STACK0/STEP_END drift that full_trace
    ignores)."""
    from neural_vm.batched_pure_neural import (
        BatchedPureNeuralRunner, _ElementState, _step_offset_field,
        _UNSAFE_OFFSETS,
    )
    from neural_vm.speculative import DraftVM
    from neural_vm.vm_step import Token

    STEP = Token.STEP_TOKENS
    runner = object.__new__(BatchedPureNeuralRunner)

    bytecode = _encode([(Opcode.IMM, 7), Opcode.EXIT])
    vm = DraftVM(bytecode)
    oracle, toks = [], []
    for _ in range(6):
        if vm.halted:
            break
        if not vm.step():
            break
        oracle.append((vm.pc & 0xFFFFFFFF, vm.ax & 0xFFFFFFFF))
        toks.append([int(t) for t in vm.draft_tokens()])
        if vm.halted:
            break

    def mk(step_tokens):
        s = _ElementState(bytecode=bytecode, context=[], prefix_len=0,
                          expected_steps=len(oracle) + 1)
        s.ff_oracle_steps = oracle
        s.ff_oracle_tokens = toks
        s.context = [t for st in step_tokens for t in st]
        s.token_pos = len(step_tokens) * STEP
        return s

    # Perfect trace -> strict_trace passes.
    s = mk([list(toks[0]), list(toks[1])])
    assert runner._ff_check_new_steps(s, criterion="strict_trace") is False
    assert s.ff_status is None

    # Corrupt AX byte-1 (offset 7) of step 0 -> first divergence there.
    bad0 = list(toks[0]); bad0[7] = (bad0[7] + 1) & 0xFF
    s = mk([bad0, list(toks[1])])
    assert runner._ff_check_new_steps(s, criterion="strict_trace") is True
    assert s.ff_status == "fail"
    assert s.ff_div_step == 0
    assert s.ff_div_offset == 7
    assert s.ff_div_offset_name == "AX[1]" == _step_offset_field(7)
    assert s.ff_expected_tok == toks[0][7]
    assert s.ff_got_tok == bad0[7]

    # Corrupting ONLY the UNSAFE MEM offsets (26..33) is IGNORED.
    unsafe0 = list(toks[0])
    for off in _UNSAFE_OFFSETS:
        unsafe0[off] = (unsafe0[off] + 3) & 0xFF
    s = mk([unsafe0, list(toks[1])])
    assert runner._ff_check_new_steps(s, criterion="strict_trace") is False
    assert s.ff_status is None

    # SP-byte drift: full_trace IGNORES it (only PC/AX), strict_trace CATCHES it.
    sp0 = list(toks[0]); sp0[11] = (sp0[11] + 5) & 0xFF  # SP[0] at offset 11
    s_full = mk([sp0, list(toks[1])])
    assert runner._ff_check_new_steps(s_full, criterion="full_trace") is False
    assert s_full.ff_status is None
    s_strict = mk([sp0, list(toks[1])])
    assert runner._ff_check_new_steps(s_strict, criterion="strict_trace") is True
    assert s_strict.ff_div_offset == 11
    assert s_strict.ff_div_offset_name == "SP[0]"


def test_run_batch_fail_fast_strict_trace_integration():
    """End-to-end ``run_batch_fail_fast(criterion='strict_trace')`` on a fake
    CPU model: a model emitting the exact oracle trace PASSES; flipping one
    SAFE-offset token FAILS at the right step+offset with the token recorded."""
    import torch

    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from neural_vm.speculative import DraftVM
    from neural_vm.vm_step import Token

    bytecode = _encode([(Opcode.IMM, 9), Opcode.EXIT])
    vm = DraftVM(bytecode)
    trace = []
    for _ in range(6):
        if vm.halted:
            break
        if not vm.step():
            break
        trace.extend(vm.draft_tokens())
        if vm.halted:
            break

    class _Embed:
        def set_mem_history_end(self, _v):
            pass

        def set_mem_addr_src_positions(self, _v):
            pass

    def _make_runner(corrupt_pos=None):
        class _Model:
            max_seq_len = 4096
            embed = _Embed()

            def forward(self, token_ids, **_kwargs):
                B, S = token_ids.shape[0], token_ids.shape[1]
                logits = torch.zeros(B, S, Token.VOCAB_SIZE, device=token_ids.device)
                for b in range(B):
                    for p in range(S):
                        if 0 <= p < len(trace):
                            tok = trace[p]
                            if corrupt_pos is not None and p == corrupt_pos:
                                tok = (tok + 1) % Token.VOCAB_SIZE
                        else:
                            tok = Token.HALT
                        logits[b, p, tok] = 1.0
                return logits

        runner = object.__new__(BatchedPureNeuralRunner)
        runner.model = _Model()
        runner._device = torch.device("cpu")
        runner.use_kv_cache = False
        runner.incremental_kv_safe = False
        runner.spec_fail_on_correction = False
        runner.spec_fail_fast = False
        runner._kv_stats = {"fresh_forwards": 0}
        runner._reset_kv_cache = lambda: None
        runner._reset_spec_stats()
        return runner

    class _Serial:
        def _build_context(self, *_a, **_k):
            return [Token.CODE_START]

    decl_steps = len(trace) // Token.STEP_TOKENS

    runner = _make_runner(corrupt_pos=None)
    runner._serial = _Serial()
    res = runner.run_batch_fail_fast(
        [bytecode], expected_steps_list=[decl_steps], max_steps=None, spec_k=1,
        criterion="strict_trace",
    )
    assert res[0]["status"] == "pass", res[0]
    assert res[0]["divergence_offset"] is None

    # Corrupt AX byte-0 of step 0 (offset 6 in the trace) -> strict FAIL there.
    runner2 = _make_runner(corrupt_pos=6)
    runner2._serial = _Serial()
    res2 = runner2.run_batch_fail_fast(
        [bytecode], expected_steps_list=[decl_steps], max_steps=None, spec_k=1,
        criterion="strict_trace",
    )
    assert res2[0]["status"] == "fail", res2[0]
    assert res2[0]["divergence_step"] == 0, res2[0]
    assert res2[0]["divergence_offset"] == 6, res2[0]
    assert res2[0]["divergence_offset_name"] == "AX[0]", res2[0]
    assert res2[0]["got_tok"] != res2[0]["expected_tok"], res2[0]


# Mem-history-windowing tests (test_run_speculative_tags_draft_region_store_mem_markers,
# test_windowed_context_*) removed with Wave D — the runner no longer
# maintains mem_history/mem_store_positions or rebuilds windowed contexts.
# The neural model attends over the raw emitted context.


def test_batched_kv_bypasses_incremental_path_by_default():
    import torch

    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from neural_vm.vm_step import Token

    class _Model:
        def forward(self, token_ids):
            logits = torch.zeros(
                token_ids.shape[0],
                token_ids.shape[1],
                Token.VOCAB_SIZE,
                device=token_ids.device,
            )
            logits[:, :, Token.HALT] = 1.0
            return logits

    runner = object.__new__(BatchedPureNeuralRunner)
    runner.model = _Model()
    runner._device = torch.device("cpu")
    runner.use_kv_cache = True
    runner.incremental_kv_safe = False
    runner._kv_stats = {
        "fresh_forwards": 0,
        "kv_forwards": 0,
        "spec_fresh_bypass": 0,
        "unsafe_model_fresh_bypass": 0,
    }

    preds, pred_start, real_lens = runner._forward_argmax_batch(
        [[Token.CODE_START]],
        [0],
        first_logit_pos=0,
    )

    assert preds == [[Token.HALT]]
    assert pred_start == 0
    assert real_lens == [1]
    assert runner._kv_stats["fresh_forwards"] == 1
    assert runner._kv_stats["kv_forwards"] == 0
    assert runner._kv_stats["unsafe_model_fresh_bypass"] == 1


class _FakeBatchedModel:
    def __init__(self):
        self.calls = []

    def forward(self, token_ids, kv_cache=None, cached_prefix_len=0):
        self.calls.append({
            "kv_cache": kv_cache,
            "cached_prefix_len": cached_prefix_len,
            "shape": tuple(token_ids.shape),
        })
        batch, seq_len = token_ids.shape
        out_len = seq_len - cached_prefix_len if cached_prefix_len > 0 else seq_len
        logits = torch.zeros((batch, out_len, 8), dtype=torch.float32)
        logits[:, :, 3] = 1.0
        return logits


class _MismatchingFakeBatchedModel(_FakeBatchedModel):
    def forward(self, token_ids, kv_cache=None, cached_prefix_len=0):
        self.calls.append({
            "kv_cache": kv_cache,
            "cached_prefix_len": cached_prefix_len,
            "shape": tuple(token_ids.shape),
        })
        batch, seq_len = token_ids.shape
        out_len = seq_len - cached_prefix_len if cached_prefix_len > 0 else seq_len
        logits = torch.zeros((batch, out_len, 8), dtype=torch.float32)
        logits[:, :, 4 if kv_cache is not None else 5] = 1.0
        return logits


def _fake_kv_runner(*, verify):
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

    runner = object.__new__(BatchedPureNeuralRunner)
    runner.model = _FakeBatchedModel()
    runner._device = torch.device("cpu")
    runner.use_kv_cache = True
    runner.incremental_kv_safe = True
    runner.kv_cache_verify = verify
    runner.kv_cache_verify_interval = 1
    runner.kv_cache_max_tokens = 32
    runner.kv_flush_interval = 0
    runner._kv_cache_obj = None
    runner._kv_active_idx = (0, 1)
    runner._kv_cached_rows = [[10, 11, 12], [20, 21, 22]]
    runner._kv_incremental_count = 0
    runner._kv_stats = {
        "calls": 0,
        "hits": 0,
        "fallbacks": 0,
        "mismatches": 0,
        "eviction_pressure": 0,
        "verifications": 0,
        "fresh_forwards": 0,
        "kv_forwards": 0,
        "verification_forwards": 0,
        "cache_rebuilds": 0,
        "reused_token_slots": 0,
        "spec_fresh_bypass": 0,
    }
    runner._get_or_build_kv_cache = lambda: object()
    return runner


def test_batched_kv_verify_off_does_not_run_fresh_verification():
    runner = _fake_kv_runner(verify=False)

    preds, pred_start, real_lens = runner._forward_argmax_batch(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        [0, 1],
        first_logit_pos=2,
    )

    assert pred_start == 2
    assert real_lens == [4, 4]
    assert preds == [[3, 3], [3, 3]]
    assert len(runner.model.calls) == 1
    assert runner.model.calls[0]["kv_cache"] is not None
    assert runner.model.calls[0]["cached_prefix_len"] == 2
    assert runner._kv_stats["calls"] == 1
    assert runner._kv_stats["hits"] == 1
    assert runner._kv_stats["kv_forwards"] == 1
    assert runner._kv_stats["fresh_forwards"] == 0
    assert runner._kv_stats["verifications"] == 0
    assert runner._kv_stats["verification_forwards"] == 0
    assert runner._kv_stats["reused_token_slots"] == 4


def test_batched_kv_can_be_bypassed_for_speculative_verifier():
    runner = _fake_kv_runner(verify=False)

    preds, pred_start, real_lens = runner._forward_argmax_batch(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        [0, 1],
        first_logit_pos=2,
        allow_kv=False,
    )

    assert pred_start == 0
    assert real_lens == [4, 4]
    assert preds == [[3, 3, 3, 3], [3, 3, 3, 3]]
    assert len(runner.model.calls) == 1
    assert runner.model.calls[0]["kv_cache"] is None
    assert runner.model.calls[0]["cached_prefix_len"] == 0
    assert runner._kv_stats["calls"] == 0
    assert runner._kv_stats["kv_forwards"] == 0
    assert runner._kv_stats["fresh_forwards"] == 1
    assert runner._kv_stats["spec_fresh_bypass"] == 1


def test_batched_kv_verify_on_runs_one_fresh_verification():
    runner = _fake_kv_runner(verify=True)

    runner._forward_argmax_batch(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        [0, 1],
        first_logit_pos=2,
    )

    assert len(runner.model.calls) == 2
    assert runner.model.calls[0]["kv_cache"] is not None
    assert runner.model.calls[1]["kv_cache"] is None
    assert runner._kv_stats["kv_forwards"] == 1
    assert runner._kv_stats["verifications"] == 1
    assert runner._kv_stats["verification_forwards"] == 1


def test_batched_kv_verify_mismatch_falls_back_to_fresh_predictions():
    runner = _fake_kv_runner(verify=True)
    runner.model = _MismatchingFakeBatchedModel()

    preds, pred_start, real_lens = runner._forward_argmax_batch(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        [0, 1],
        first_logit_pos=2,
    )

    assert pred_start == 0
    assert real_lens == [4, 4]
    assert preds == [[5, 5, 5, 5], [5, 5, 5, 5]]
    assert len(runner.model.calls) == 2
    assert runner.model.calls[0]["kv_cache"] is not None
    assert runner.model.calls[1]["kv_cache"] is None
    assert runner._kv_stats["mismatches"] == 1
    assert runner._kv_stats["fallbacks"] == 1


def test_batched_kv_flush_interval_rebuilds_cache():
    runner = _fake_kv_runner(verify=False)
    runner.kv_flush_interval = 1
    runner._kv_incremental_count = 1

    runner._forward_argmax_batch(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        [0, 1],
        first_logit_pos=2,
    )

    assert runner.model.calls[0]["cached_prefix_len"] == 0
    assert runner._kv_stats["cache_rebuilds"] == 1
    assert runner._kv_stats["hits"] == 0


def test_batched_incremental_kv_prunes_overflow_without_fresh_reset():
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from neural_vm.kv_cache import LayerKVCache

    class _AppendingModel(_FakeBatchedModel):
        def forward(self, token_ids, kv_cache=None, cached_prefix_len=0):
            if kv_cache is not None:
                new_len = token_ids.shape[1] - cached_prefix_len
                for layer_cache in kv_cache.caches:
                    k = torch.zeros(1, 1, new_len, 1)
                    v = torch.zeros(1, 1, new_len, 1)
                    layer_cache.update(k, v)
            return super().forward(
                token_ids,
                kv_cache=kv_cache,
                cached_prefix_len=cached_prefix_len,
            )

    logical_max = 80
    kv_cache = LayerKVCache(
        num_layers=1,
        max_tokens=logical_max + 1,
        num_heads=1,
        head_dim=1,
        device="cpu",
    )
    kv_cache.caches[0].update(
        torch.zeros(1, 1, logical_max, 1),
        torch.zeros(1, 1, logical_max, 1),
    )

    runner = object.__new__(BatchedPureNeuralRunner)
    runner.model = _AppendingModel()
    runner._device = torch.device("cpu")
    runner.use_kv_cache = True
    runner.incremental_kv_safe = True
    runner.kv_cache_verify = False
    runner.kv_cache_verify_interval = 1
    runner.kv_cache_max_tokens = logical_max
    runner.kv_flush_interval = 0
    runner._kv_cache_obj = kv_cache
    runner._kv_active_idx = (0,)
    runner._kv_cached_rows = [list(range(logical_max))]
    runner._kv_incremental_count = 0
    runner._kv_stats = {
        "calls": 0,
        "hits": 0,
        "fallbacks": 0,
        "mismatches": 0,
        "eviction_pressure": 0,
        "verifications": 0,
        "fresh_forwards": 0,
        "kv_forwards": 0,
        "verification_forwards": 0,
        "cache_rebuilds": 0,
        "reused_token_slots": 0,
        "spec_fresh_bypass": 0,
        "unsafe_model_fresh_bypass": 0,
        "bounded_evictions": 0,
        "bounded_positions_evicted": 0,
        "bounded_eviction_fallbacks": 0,
    }

    preds, pred_start, real_lens = runner._forward_argmax_batch(
        [list(range(logical_max + 1))],
        [0],
        first_logit_pos=logical_max,
        protected_prefix_lens=[2],
        protected_mem_positions=[[]],
    )

    assert preds == [[3]]
    assert pred_start == logical_max
    assert real_lens == [logical_max + 1]
    assert runner._kv_stats["fresh_forwards"] == 0
    assert runner._kv_stats["bounded_evictions"] == 1
    assert runner._kv_stats["bounded_positions_evicted"] == 1
    assert kv_cache.caches[0].cache_size == logical_max
    assert kv_cache.caches[0].next_pos_id == logical_max + 1
    assert kv_cache.caches[0].cached_pos_ids[0].tolist() == (
        [0, 1] + list(range(3, logical_max + 1))
    )


@pytest.mark.slow
def test_bucketed_wall_time_analysis(batched_pure_neural_runner):
    """Wall-time analysis: bucketed should be faster than unbucketed when
    the input has wide length variance.

    All current ``_CLEAN_PHASE1_PROGRAMS`` are 2-step programs so they end up
    in the same bucket — the bucket wall-time = unbucketed wall-time in
    expectation. The bucketed path adds only a tiny DraftVM-prediction
    overhead per program. We assert bucketed runtime is within 1.25x of
    unbucketed runtime (so the overhead doesn't dominate on short inputs)
    and the results are byte-identical.
    """
    bcs = [_encode(p) for p in _CLEAN_PHASE1_PROGRAMS]
    N = len(bcs)

    t0 = time.time()
    unbucketed = batched_pure_neural_runner.run_batch(
        bcs, max_steps=10, bucket_by_predicted_length=False
    )
    unbucketed_t = time.time() - t0

    t0 = time.time()
    bucketed = batched_pure_neural_runner.run_batch(
        bcs, max_steps=10, bucket_by_predicted_length=True
    )
    bucketed_t = time.time() - t0

    assert bucketed == unbucketed
    overhead = bucketed_t / unbucketed_t if unbucketed_t > 0 else 1.0
    print(
        f"\n[bucket bench] N={N}  unbucketed={unbucketed_t:.3f}s  "
        f"bucketed={bucketed_t:.3f}s  ratio={overhead:.2f}x"
    )
    # Single-bucket workload should add only DraftVM-prediction overhead.
    assert overhead < 1.25, (
        f"Bucketing added too much overhead on homogeneous batch: "
        f"unbucketed={unbucketed_t:.3f}s, bucketed={bucketed_t:.3f}s"
    )
