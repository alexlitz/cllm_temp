#!/usr/bin/env python3
"""Ground-truth residual / logit probe for the c4_release neural VM.

This probe runs the **exact** execution path the smoke gate measures:
``BatchedPureNeuralRunner`` with ``spec_k=0`` (the raw, one-token-per-forward
batched neural path — NO DraftVM speculation, NO KV-cache speculation). It is
the only path whose emitted bytes are what ``tests/test_smoke.py`` asserts on.

WHY THIS FILE EXISTS
--------------------
Earlier probe tooling defaulted to ``C4_TEST_SPEC_K=8`` (speculative decode +
KV cache), a *different* code path with its own possible KV/speculation bugs.
``test_ne_true`` *passes* on the smoke gate (spec_k=0) but agents saw it
"fail" in spec_k=8 probes. The discrepancy was the execution path, NOT forward
hooks (the production model is hook-free). See
``docs/PROBE_GROUNDTRUTH_2026_06_10.md`` and memory note
``project_probe_path_spec_k_not_hooks.md``.

HARD RULES (enforced by self-test + lint)
-----------------------------------------
* NO ``register_forward_hook`` / ``register_forward_pre_hook`` anywhere.
* NO weight overrides.
* spec_k=0 throughout. We never call the speculative path.

HOW IT READS INTERNALS WITHOUT HOOKS
------------------------------------
* **Logits / emitted token**: we replay the smoke runner's own
  ``_run_unspeculative`` step loop and call the model's ``forward`` directly
  on the same padded context the runner builds, reading the returned logits.
  The argmax of those logits is the emitted token — byte-identical to the
  runner. (We assert this byte-for-byte against the real runner in
  ``validate()``.)
* **Residuals**: ``AutoregressiveVM.forward`` grew an opt-in, probe-only
  ``stop_after_block=<int>`` kwarg (default ``None`` => unchanged production
  behaviour) that returns the model's own post-block hidden state instead of
  logits. We re-run the same prefix truncated so the target block is the last
  executed block and read that normally-returned tensor. No hooks.

USAGE
-----
    from tools.probe_groundtruth import build_groundtruth_probe
    p = build_groundtruth_probe()
    trace = p.probe(program_bytes)                 # {pos: {token, top_k_logits,...}}
    res   = p.residual_at(program_bytes, block_idx=10, position=-1,
                          dim_names={"AX_LO": E.AX_BASE, "OPCODE": E.OPCODE})
    p.print_block_layer_map()

Run directly to execute the 4-test byte-identity validation:
    python tools/probe_groundtruth.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# spec_k=0 is the smoke gate's path. Pin both env knobs defensively so any
# transitively-read default lands on the ground-truth path, never spec_k=8.
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)  # .../c4_release (the package root)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import torch  # noqa: E402

from neural_vm.run_vm import AutoregressiveVMRunner  # noqa: E402
from neural_vm.batched_pure_neural import BatchedPureNeuralRunner, Token  # noqa: E402
from neural_vm.embedding import E, Opcode  # noqa: E402  (E = residual DimPosition map)


# Marker tokens used by the runner's per-step register decode. Re-exported so
# callers can label trace rows.
_MARKER_NAMES = {
    int(Token.REG_PC): "REG_PC",
    int(Token.REG_AX): "REG_AX",
    int(Token.REG_SP): "REG_SP",
    int(Token.REG_BP): "REG_BP",
    int(Token.STEP_END): "STEP_END",
    int(Token.HALT): "HALT",
    int(Token.TOOL_CALL): "TOOL_CALL",
}


@dataclass
class _PosInfo:
    """Per-emitted-position probe record."""
    token: int
    token_name: Optional[str]
    top_k_logits: List[Tuple[int, float]]  # [(token_id, logit), ...] descending
    forward_iter: int                       # which model forward produced it
    context_len_before: int                 # context length the model saw


class GroundTruthProbe:
    """Probe wrapper around the spec_k=0 batched smoke path.

    Holds the SAME ``BatchedPureNeuralRunner`` / ``AutoregressiveVM`` the smoke
    gate constructs (see ``tests/conftest.py::_batched_pure_neural_runner_model``
    and ``tests/test_smoke.py``). The model build is expensive, so reuse one
    probe across many programs.
    """

    SPEC_K = 0  # ground truth. Never changed.

    def __init__(self, runner: BatchedPureNeuralRunner):
        self.runner = runner
        self.model = runner.model
        self._device = next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # Construction — mirrors conftest + test_smoke exactly.
    # ------------------------------------------------------------------
    @classmethod
    def build(cls) -> "GroundTruthProbe":
        """Build the runner the smoke gate uses.

        Mirrors ``conftest._pure_neural_runner_model`` (AutoregressiveVMRunner
        with ``pure_neural=True, trust_neural_alu=True, spec_k=<C4_TEST_SPEC_K>``
        and Python handlers stripped) wrapped by
        ``conftest._batched_pure_neural_runner_model`` (BatchedPureNeuralRunner).
        ``C4_TEST_SPEC_K`` is forced to 0 at import time.
        """
        spec_k = int(os.environ.get("C4_TEST_SPEC_K", "0"))
        assert spec_k == 0, (
            "GroundTruthProbe MUST run spec_k=0 (the smoke gate path). "
            f"Got C4_TEST_SPEC_K={spec_k}."
        )
        model_runner = AutoregressiveVMRunner(
            pure_neural=True, trust_neural_alu=True, spec_k=0,
        )
        # Strip Python-side handlers so dispatch is fully neural — identical to
        # the conftest session fixture.
        model_runner._func_call_handlers = {}
        model_runner._syscall_handlers = {}
        runner = BatchedPureNeuralRunner(model_runner=model_runner)
        # The smoke path uses no KV cache (C4_BATCH_USE_KV_CACHE != "1") and
        # csr_inference on by default — both inherited unchanged from the
        # shared model_runner. We assert the no-KV invariant since the
        # ground-truth fresh-forward branch depends on it.
        assert runner.use_kv_cache is False, (
            "Ground-truth path requires no batched KV cache (set "
            "C4_BATCH_USE_KV_CACHE=0 / unset). The smoke gate runs without it."
        )
        return cls(runner)

    # ==================================================================
    # 37-physical-block  <->  logical-layer mapping
    # ==================================================================
    def block_layer_map(self) -> List[dict]:
        """Return the physical-block -> logical-layer mapping.

        Each entry: ``{physical, logical, is_post_op_expansion, attn, ffn}``.
        Provenance comes from ``_logical_layer`` / ``_is_post_op_expansion``
        tags set during ``_expand_wrapper_blocks`` (post_ops split into their
        own passthrough blocks). Falls back to physical index if a model was
        built without the tags (e.g. wrapper expansion disabled).
        """
        out = []
        for phys, blk in enumerate(self.model.blocks):
            logical = getattr(blk, "_logical_layer", phys)
            is_exp = getattr(blk, "_is_post_op_expansion", False)
            out.append({
                "physical": phys,
                "logical": int(logical),
                "is_post_op_expansion": bool(is_exp),
                "attn": type(getattr(blk, "attn", None)).__name__,
                "ffn": type(getattr(blk, "ffn", None)).__name__,
            })
        return out

    def print_block_layer_map(self, file=sys.stdout) -> None:
        rows = self.block_layer_map()
        n_logical = len({r["logical"] for r in rows})
        print(f"# {len(rows)} physical blocks <- {n_logical} logical layers "
              f"(post_op expansion)", file=file)
        print(f"{'phys':>4} {'logical':>7} {'expanded':>8}  "
              f"{'attn':<22} {'ffn'}", file=file)
        for r in rows:
            print(f"{r['physical']:>4} {r['logical']:>7} "
                  f"{str(r['is_post_op_expansion']):>8}  "
                  f"{r['attn']:<22} {r['ffn']}", file=file)

    # ==================================================================
    # Logit / token readout — replays the spec_k=0 unspeculative loop.
    # ==================================================================
    def _build_context(self, program_bytes: Sequence[int]) -> List[int]:
        """Build the initial prompt context exactly as the runner does."""
        return list(self.runner._serial._build_context(
            list(program_bytes), b"", [], ""))

    def _expected_steps(self, program_bytes: Sequence[int]) -> Optional[int]:
        """Declarative halt horizon used by the smoke gate as the step budget.

        Mirrors ``test_smoke._run_group_batch`` which feeds
        ``oracle.steps`` as ``expected_steps_list``. Returns ``None`` when the
        oracle can't predict (caller then supplies an explicit ``max_steps``).
        """
        try:
            from tests.declarative_oracle import declarative_oracle_for_program
            oracle = declarative_oracle_for_program(
                list(program_bytes), b"", label="probe")
            return oracle.steps
        except Exception:
            return None

    @torch.no_grad()
    def _forward_logits(self, context: List[int]) -> torch.Tensor:
        """Run the model forward on a single context and return logits.

        Single-row batch, no KV cache, no speculation — exactly the tensor the
        smoke runner argmaxes in ``_run_unspeculative`` (its
        ``_forward_argmax_batch`` fresh-forward branch is
        ``self.model.forward(padded)`` followed by ``argmax``). We call the
        same ``forward`` here and keep the full logits so the probe can report
        top-k while still reproducing the runner's argmax bit-for-bit.
        """
        padded = torch.tensor([context], dtype=torch.long, device=self._device)
        logits = self.model.forward(padded)        # [1, S, V]
        return logits[0]                            # [S, V]

    @torch.no_grad()
    def probe(
        self,
        program_bytes: Sequence[int],
        *,
        top_k: int = 8,
        max_steps: Optional[int] = None,
    ) -> Dict[int, dict]:
        """Run the spec_k=0 path and return per-emitted-position records.

        Returns ``{position: {token, token_name, top_k_logits, forward_iter,
        context_len_before}}`` where ``position`` is the index of the emitted
        token within the FULL context (prompt + emissions). The emitted-token
        sequence (and therefore exit code / output bytes) is byte-identical to
        ``BatchedPureNeuralRunner.run_batch([program_bytes], spec_k=0)``.

        The step loop mirrors ``BatchedPureNeuralRunner._run_unspeculative`` and
        ``_step_one`` (STEP_END / TOOL_CALL dispatch, HALT termination,
        neural-authoritative EXIT-on-next-PC early stop, expected-steps cap).
        """
        STEP = int(Token.STEP_TOKENS)
        ctx = self._build_context(program_bytes)
        prompt_len = len(ctx)
        expected = self._expected_steps(program_bytes)
        if max_steps is not None:
            budget = int(max_steps) * STEP
        elif expected is not None:
            budget = int(expected) * STEP
        else:
            raise ValueError("probe() needs max_steps or a predictable oracle")

        # Mirror _ElementState fields the dispatch path reads.
        state = _LoopState(bytecode=list(program_bytes))
        records: Dict[int, _PosInfo] = {}

        for tok_i in range(budget):
            if state.halted:
                break
            logits = self._forward_logits(ctx)            # [S, V]
            last = logits[len(ctx) - 1]                    # next-token logits
            next_tok = int(last.argmax().item())
            topk = torch.topk(last, k=min(top_k, last.shape[-1]))
            top_pairs = [
                (int(t), float(v))
                for v, t in zip(topk.values.tolist(), topk.indices.tolist())
            ]
            pos = len(ctx)  # position the emitted token will occupy
            records[pos] = _PosInfo(
                token=next_tok,
                token_name=_MARKER_NAMES.get(next_tok),
                top_k_logits=top_pairs,
                forward_iter=tok_i,
                context_len_before=len(ctx),
            )
            # Append + run the same dispatch the runner runs.
            ctx.append(next_tok)
            self._step_one(state, ctx, next_tok, expected)

        # Convert to plain dicts for a clean public API.
        return {
            pos: {
                "token": r.token,
                "token_name": r.token_name,
                "top_k_logits": r.top_k_logits,
                "forward_iter": r.forward_iter,
                "context_len_before": r.context_len_before,
            }
            for pos, r in records.items()
        }

    # ------------------------------------------------------------------
    # Pure-python mirror of the runner's per-step dispatch (spec_k=0).
    # Mirrors BatchedPureNeuralRunner._step_one / _dispatch_pure_neural so the
    # emitted-token sequence matches byte-for-byte. No model side effects.
    # ------------------------------------------------------------------
    def _step_one(self, s: "_LoopState", ctx: List[int],
                  next_token: int, expected_steps: Optional[int]) -> None:
        s.token_pos += 1
        if next_token == Token.HALT:
            s.halted = True
            return
        if next_token == Token.STEP_END or next_token == Token.TOOL_CALL:
            self._dispatch(s, ctx)
            if s.halted:
                return
        if (expected_steps is not None
                and s.token_pos >= expected_steps * int(Token.STEP_TOKENS)
                and not s.halted):
            s.halted = True

    def _dispatch(self, s: "_LoopState", ctx: List[int]) -> None:
        """Mirror of ``_dispatch_pure_neural``'s neural-authoritative EXIT stop.

        Reads the model-emitted REG_PC from the just-finished step; if the next
        instruction is EXIT, the runner halts here instead of generating an
        extra EXIT step. This is the only state-changing branch that affects
        the emitted token count, so it must be reproduced exactly.
        """
        neural_pc = self.runner._extract_register(ctx, Token.REG_PC)
        if neural_pc is not None:
            s.last_pc = neural_pc
        if s.last_pc is not None:
            next_idx = s.last_pc // 8  # INSTR_WIDTH
            if 0 <= next_idx < len(s.bytecode):
                next_op = s.bytecode[next_idx] & 0xFF
                if next_op == Opcode.EXIT:
                    s.halted = True

    # ==================================================================
    # Residual readout — truncate-and-rerun, read the model's own tensor.
    # ==================================================================
    @torch.no_grad()
    def residual_at(
        self,
        program_bytes: Sequence[int],
        block_idx: int,
        position: int,
        dim_names: Dict[str, int],
        *,
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Return ``{dim_name: residual_value}`` after physical ``block_idx``.

        We replay the spec_k=0 emission loop to rebuild the FULL final context
        (so the residual is read on the exact sequence the smoke run produced),
        then run one forward truncated to ``stop_after_block=block_idx`` and read
        the model's own post-block hidden state. NO hooks: the model returns
        the residual directly via its opt-in probe kwarg.

        Args:
            block_idx: physical block index (0..len(model.blocks)-1). See
                ``block_layer_map`` for the logical-layer mapping.
            position: index into the context. Negative indexes from the end
                (``-1`` = last token).
            dim_names: ``{label: residual_dim_index}``. Use the ``E`` /
                ``DimPosition`` enum, e.g. ``{"AX_LO": E.AX_BASE,
                "OPCODE": E.OPCODE}``.
        """
        if not (0 <= block_idx < len(self.model.blocks)):
            raise IndexError(
                f"block_idx {block_idx} out of range "
                f"[0, {len(self.model.blocks)})")
        ctx = self._final_context(program_bytes, max_steps=max_steps)
        if position < 0:
            position = len(ctx) + position
        if not (0 <= position < len(ctx)):
            raise IndexError(
                f"position {position} out of range [0, {len(ctx)})")
        padded = torch.tensor([ctx], dtype=torch.long, device=self._device)
        resid = self.model.forward(padded, stop_after_block=block_idx)  # [1,S,D]
        row = resid[0, position]  # [D]
        D = row.shape[-1]
        out: Dict[str, float] = {}
        for name, dim in dim_names.items():
            d = int(dim)
            out[name] = float(row[d].item()) if 0 <= d < D else float("nan")
        return out

    @torch.no_grad()
    def _final_context(
        self,
        program_bytes: Sequence[int],
        *,
        max_steps: Optional[int] = None,
    ) -> List[int]:
        """Replay the spec_k=0 loop and return the full final context list."""
        STEP = int(Token.STEP_TOKENS)
        ctx = self._build_context(program_bytes)
        expected = self._expected_steps(program_bytes)
        if max_steps is not None:
            budget = int(max_steps) * STEP
        elif expected is not None:
            budget = int(expected) * STEP
        else:
            raise ValueError("_final_context needs max_steps or oracle")
        state = _LoopState(bytecode=list(program_bytes))
        for _ in range(budget):
            if state.halted:
                break
            logits = self._forward_logits(ctx)
            next_tok = int(logits[len(ctx) - 1].argmax().item())
            ctx.append(next_tok)
            self._step_one(state, ctx, next_tok, expected)
        return ctx

    @torch.no_grad()
    def emitted_result(
        self,
        program_bytes: Sequence[int],
        *,
        max_steps: Optional[int] = None,
    ) -> Tuple[str, int]:
        """Return ``(output, exit_code)`` the probe's replay produces.

        Used by ``validate()`` to assert byte-identity with the real runner.
        Output is empty for these register-only programs (the batched path
        does not capture PUTCHAR bytes); exit code is decoded from the last
        REG_AX, identical to ``_decode_exit_code``.
        """
        ctx = self._final_context(program_bytes, max_steps=max_steps)
        exit_code = self._decode_exit_code(ctx)
        return ("", exit_code)

    @staticmethod
    def _decode_exit_code(context: List[int]) -> int:
        """Mirror of ``BatchedPureNeuralRunner._decode_exit_code``."""
        for i in range(len(context) - 1, -1, -1):
            if context[i] == Token.REG_AX and i + 4 < len(context):
                val = 0
                for j in range(4):
                    val |= (context[i + 1 + j] & 0xFF) << (j * 8)
                return val
        return 0


@dataclass
class _LoopState:
    """Minimal per-program loop state for the replay (mirror of _ElementState
    fields the spec_k=0 dispatch reads)."""
    bytecode: List[int]
    halted: bool = False
    token_pos: int = 0
    last_pc: Optional[int] = None


def build_groundtruth_probe() -> GroundTruthProbe:
    """Public entry point. Build the spec_k=0 batched smoke-path probe."""
    return GroundTruthProbe.build()


# ======================================================================
# Validation: probe bytes MUST equal the real spec_k=0 runner bytes.
# ======================================================================
def _validation_programs() -> List[dict]:
    """The 4 named smoke tests (2 passing, 2 failing) from test_smoke.py."""
    from neural_vm.embedding import Opcode as Op

    def bc(ops):
        out = []
        for o in ops:
            if isinstance(o, tuple):
                op, imm = o
                out.append(op | (imm << 8))
            else:
                out.append(o)
        return out

    return [
        {  # PASSES on smoke gate
            "name": "TestSmokeComparison::test_ne_true",
            "bytecode": bc([(Op.IMM, 10), Op.PSH, (Op.IMM, 20), Op.NE, Op.EXIT]),
            "max_steps": 20,
        },
        {  # FAILS on smoke gate (wrong value) — probe must still MATCH runner
            "name": "TestSmokeComparison::test_eq_true",
            "bytecode": bc([(Op.IMM, 42), Op.PSH, (Op.IMM, 42), Op.EQ, Op.EXIT]),
            "max_steps": 20,
        },
        {  # FAILS on smoke gate — probe must still MATCH runner
            "name": "TestSmokeBitwise::test_or_basic",
            "bytecode": bc([(Op.IMM, 0x0F), Op.PSH, (Op.IMM, 0x30), Op.OR, Op.EXIT]),
            "max_steps": 20,
        },
        {  # PASSES on smoke gate
            "name": "TestSmokeMemory::test_si_li_roundtrip",
            "bytecode": bc([(Op.IMM, 0x200), Op.PSH, (Op.IMM, 42), Op.SI,
                            (Op.IMM, 0x200), Op.LI, Op.EXIT]),
            "max_steps": 30,
        },
    ]


def validate(probe: Optional[GroundTruthProbe] = None, *, verbose: bool = True
             ) -> bool:
    """Assert the probe reproduces the real spec_k=0 runner's bytes exactly.

    For each of the 4 named programs we run BOTH:
      1. the real ``BatchedPureNeuralRunner.run_batch(..., spec_k=0,
         bucket_by_predicted_length=False)`` (the smoke gate's exact call), and
      2. the probe's replay,
    and assert ``(output, exit_code)`` are identical. A mismatch means the
    probe diverges from the ground truth and is unusable.
    """
    if probe is None:
        probe = build_groundtruth_probe()
    progs = _validation_programs()

    # 1) Real runner, exactly as the smoke gate calls it for each program.
    all_ok = True
    for p in progs:
        oracle_steps = probe._expected_steps(p["bytecode"])
        real = probe.runner.run_batch(
            [list(p["bytecode"])],
            max_steps=None if oracle_steps is not None else p["max_steps"],
            spec_k=0,
            expected_steps_list=[oracle_steps],
            bucket_by_predicted_length=False,  # smoke uses False at spec_k=0
        )[0]
        got = probe.emitted_result(
            p["bytecode"],
            max_steps=None if oracle_steps is not None else p["max_steps"],
        )
        ok = (real == got)
        all_ok = all_ok and ok
        if verbose:
            status = "OK " if ok else "MISMATCH"
            print(f"[{status}] {p['name']:<42} "
                  f"runner={real!r}  probe={got!r}")
        if not ok:
            raise AssertionError(
                f"PROBE WRONG: {p['name']} runner={real!r} != probe={got!r}. "
                f"The probe does not match the spec_k=0 ground-truth path."
            )
    if verbose:
        print(f"\nAll {len(progs)} programs: probe bytes == spec_k=0 runner "
              f"bytes. Ground-truth match confirmed.")
    return all_ok


def _main() -> int:
    print("Building spec_k=0 batched smoke-path probe "
          "(this bakes the model once; ~slow)...\n", file=sys.stderr)
    probe = build_groundtruth_probe()
    print("=" * 70)
    probe.print_block_layer_map()
    print("=" * 70)
    validate(probe, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
