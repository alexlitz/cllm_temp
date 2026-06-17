"""DSL-interpreter verdict authority — non-re-anchored CPU faithful decode.

The VEHICLE is the DSL interpreter: every per-token argmax that drives the
production decode comes from :class:`~neural_vm.unified_compiler.faithful_interpreter.IRBlockForward`,
which runs the :class:`FaithfulInterpreter` engine over the per-physical-block
**IR** (attention executed via ``_apply_attention_op`` over
``DeclarativeAttentionHeadSpec`` IR objects; FFN via the engine's SwiGLU). It is
NOT a re-anchored single forward (that is ``tools/interp_oracle_gate.py``, which
re-anchors per step to attribute rules and so HIDES the autoregressive framing
desync) and it is NOT the recovered-weight autoregressive token-runner
(``faithful_autoregressive.py``). It is the DSL interpreter executing the spec,
fed back autoregressively, so the framing-drift verdict (a step emitting 34/37
tokens instead of 35, which makes the fixed-35-token slice misread the next
step's PC) is reproduced at the SPEC level.

Why this reproduces framing-drift
---------------------------------
The production decode is autoregressive: the moment the interpreter emits a
register byte that differs from the DraftVM oracle tape, that WRONG byte is
appended to the context and the DraftVM is re-synced from the now-wrong register
state, so every later step reads a poisoned context. A re-anchored single
forward never sees this (it re-anchors each step's marker). By feeding the
interpreter's own emissions back through the UNMODIFIED production decode
machinery (``_run_fail_fast`` speculation, ``_step_one`` /
``_dispatch_pure_neural`` STEP_END/HALT/early-EXIT, ``_ff_check_new_steps``
per-step ``(PC, AX)`` fixed-35-slice compare + ``_UNSAFE_OFFSETS``,
``_oracle_pc_ax_steps``), the verdict is byte-identical to the neural decode by
construction — the only substitution is the per-token argmax source, which is
argmax-equivalent (validated by ``tools/faithful_interpreter_validate.py``).

Design: subclass, override one method
-------------------------------------
:class:`DSLInterpreterVerdictRunner` subclasses ``BatchedPureNeuralRunner`` and
overrides only ``_forward_argmax_batch`` to source the per-token argmax from the
DSL-interpreter IR forward. Everything else is the unmodified production verdict
logic.

This module is TOOLING ONLY — never imported by ``compile_full_vm_dynamic`` or
any build path, so the model stays byte-identical (golden ``88b52dfa8c7c521b``).
"""

from __future__ import annotations

import contextlib
import io
import os
from typing import List, Optional

import torch

from .faithful_interpreter import IRBlockForward


def build_cpu_model(*, alu_mode: str = "efficient", disk_cache: bool = True):
    """Build the real baked ``AutoregressiveVM`` on CPU (cached compile).

    ``alu_mode='efficient'`` matches what ``BatchedPureNeuralRunner`` builds
    (``trust_neural_alu=True``), so the block layout + decode are production-
    faithful. Returns ``(model, layout)``; use ``layout.dim_positions`` (the
    BUILT layout) to resolve any residual dim.
    """
    os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
    os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
    os.environ.setdefault("C4_TEST_SPEC_K", "0")
    os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
    from .full_vm_compiler_dynamic import compile_full_vm_dynamic

    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = compile_full_vm_dynamic(
            alu_mode=alu_mode, disk_cache=disk_cache,
        )
    model = model.to("cpu")
    model.eval()
    return model, layout


class DSLInterpreterVerdictRunner:
    """Production fail-fast / exit-code decode driven by the DSL interpreter.

    Construct with a CPU-baked model (or let it build one). The public
    :meth:`run_batch_fail_fast` and :meth:`run_batch` are the production
    ``BatchedPureNeuralRunner`` methods, so the verdict logic is identical; the
    only difference is the per-token argmax comes from the DSL-interpreter IR
    forward (:class:`IRBlockForward`).

    Usage::

        runner = DSLInterpreterVerdictRunner()
        verdicts = runner.run_batch_fail_fast(
            [bytecode], data_list=[data], expected_steps_list=[oracle_steps],
            max_steps=None, criterion="full_trace",
        )
        # verdicts[0]["status"] in {"pass","fail","error"}, byte-identical to
        # the CPU-neural full_trace verdict (the GPU-only floor is the model's
        # own saturated-tie fp instability, not a decoder bug).
    """

    def __init__(self, model=None, layout=None, *, alu_mode: str = "efficient",
                 disk_cache: bool = True):
        if model is None:
            model, layout = build_cpu_model(alu_mode=alu_mode, disk_cache=disk_cache)
        self.model = model
        self.layout = layout
        self.dim_positions = dict(getattr(layout, "dim_positions", {}) or {})
        self._inner = _build_inner(model, self)
        self.n_forwards = 0

    def run_batch_fail_fast(self, *args, **kwargs):
        """Production per-step (PC, AX) / token-identity verdict, DSL-interp CPU."""
        return self._inner.run_batch_fail_fast(*args, **kwargs)

    def run_batch(self, *args, **kwargs):
        """Production exit-code verdict, DSL-interp CPU forward."""
        return self._inner.run_batch(*args, **kwargs)

    @property
    def forward_count(self) -> int:
        return self._inner.dsl_forward_count


_VERDICT_SUBCLASS = None


def _build_inner(model, owner):
    """Lazily instantiate the DSL-interpreter subclass of the production runner."""
    global _VERDICT_SUBCLASS
    if _VERDICT_SUBCLASS is None:
        from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

        _VERDICT_SUBCLASS = _make_subclass(BatchedPureNeuralRunner)
    return _VERDICT_SUBCLASS(model, owner)


def _make_subclass(base):
    """Return a ``BatchedPureNeuralRunner`` subclass whose per-token argmax comes
    from the DSL-interpreter IR forward instead of the batched GPU model forward.

    Only ``_forward_argmax_batch`` is overridden — every decode loop
    (``_run_unspeculative`` / ``_run_fail_fast`` / ``_step_one`` /
    ``_dispatch_pure_neural`` / ``_ff_check_new_steps`` / ``_oracle_pc_ax_steps``)
    is the UNMODIFIED production code, so the verdict is byte-identical to neural.
    """

    class _Inner(base):  # type: ignore[valid-type, misc]
        def __init__(self, model, owner):
            # Bypass the heavy production ``__init__`` (it wraps a serial runner
            # + KV cache + CSR shim). Install only what the decode loops read.
            self.model = model
            self._owner = owner
            self._device = torch.device("cpu")
            self.use_kv_cache = False
            self.incremental_kv_safe = False
            self.kv_cache_verify = False
            self.spec_fail_fast = False
            self.spec_fail_on_correction = False
            self.enable_moe_routing = False
            self.kv_cache_max_tokens = int(getattr(model, "max_seq_len", 4096))
            self._kv_cache_storage_max_tokens = self.kv_cache_max_tokens
            self._kv_cache_obj = None
            self._kv_active_idx = None
            self._kv_cached_rows = []
            self._kv_incremental_count = 0
            self._kv_stats = {}
            self._spec_stats = {}
            self.dsl_forward_count = 0
            self._reset_spec_stats()
            # The DSL-interpreter IR forward (per-block recovery cached once).
            self._dsl = IRBlockForward(model)
            self._serial = _ContextShim(model)

        # ---- the ONE override: per-token argmax from the DSL interpreter -----
        #
        # Speculation is NOT suppressed: the production fail-fast path
        # teacher-forces the DraftVM tokens at the UNSAFE MEM offsets (those
        # bytes are always accepted from the draft because the embedding's
        # MEM-metadata injection makes them unreadable from a flat forward). If
        # we decoded the MEM bytes from the interpreter's own argmax, the wrong
        # MEM bytes would feed back and corrupt the next step's decode, making
        # the CPU verdict disagree with the spec_k=32 neural canonical run. So
        # we keep the production speculative loop verbatim and only swap the
        # per-token argmax source — which is exactly what makes the verdict
        # byte-identical to neural.
        @torch.no_grad()
        def _forward_argmax_batch(
            self,
            sequences,
            active_idx,
            *,
            first_logit_pos=None,
            allow_kv=True,
            protected_prefix_lens=None,
            protected_mem_positions=None,
            gather_positions=None,
        ):
            real_lens = [len(s) for s in sequences]
            preds_cpu: List[List[int]] = []
            for b, seq in enumerate(sequences):
                if gather_positions is not None:
                    gp = list(gather_positions[b])
                else:
                    gp = [real_lens[b] - 1]
                logits = self._dsl.forward(list(seq), return_logits=True)
                self.dsl_forward_count += 1
                row_argmax = logits.argmax(dim=-1)  # [S]
                preds_cpu.append([int(row_argmax[p]) for p in gp])
            pred_start = (
                min(real_lens) - 1 if first_logit_pos is None else first_logit_pos
            )
            return preds_cpu, pred_start, real_lens

    _Inner.__name__ = "DSLInterpreterVerdictInner"
    return _Inner


class _ContextShim:
    """Minimal stand-in for the serial runner: supplies ``_build_context``."""

    def __init__(self, model):
        self.model = model

    def _build_context(self, bytecode, data, argv, stdin=""):
        from neural_vm.run_vm import AutoregressiveVMRunner

        return AutoregressiveVMRunner._build_context(
            self, bytecode, data, argv, stdin
        )


__all__ = [
    "DSLInterpreterVerdictRunner",
    "build_cpu_model",
]
