"""CPU autoregressive decoder that mirrors ``batched_pure_neural`` EXACTLY.

Motivation
----------
``tools/interp_oracle_gate.py`` (the CPU rule-attribution gate) runs a
RE-ANCHORED single forward: it re-anchors each step's register markers so it
can attribute a wrong byte to its owning rule. That deliberately HIDES the
autoregressive framing desync — the ~13% of the corpus (the framing-drift
clusters: ``var_*`` / ``func_identity`` / ``nested_*`` / ``if_var``) where a
step emits 34 or 37 tokens instead of 35 and the fixed-35-token slice misreads
the NEXT step's PC. Because those verdicts can only be reproduced by the REAL
autoregressive decode, the gate currently DEFERS them to a GPU run.

This module closes that gap on CPU. It runs the byte-identical
:func:`~neural_vm.unified_compiler.faithful_interpreter.run_faithful_blocks`
forward (validated max-argmax-diff = 0 vs the neural model) inside an
autoregressive loop that reuses the PRODUCTION decode machinery of
:class:`~neural_vm.batched_pure_neural.BatchedPureNeuralRunner` verbatim:
the SAME argmax (``logits.argmax(-1)``, torch's first-max tie-break), the SAME
fixed-35-token slice + ``_UNSAFE_OFFSETS``, the SAME ``_step_one`` /
``_dispatch_pure_neural`` STEP_END / HALT / early-EXIT handling, the SAME
``_ff_check_new_steps`` per-step ``(PC, AX)`` compare, and the SAME DraftVM
oracle (``_oracle_pc_ax_steps``). The ONLY thing swapped is the source of the
next-token argmax: the faithful CPU forward instead of the batched GPU model
forward. NO re-anchoring — emitted tokens are fed back as the next step's input,
so the 34/37-token miscount is reproduced exactly when the neural model would
miscount.

Design: subclass, override one method
--------------------------------------
:class:`FaithfulAutoregressiveRunner` subclasses ``BatchedPureNeuralRunner`` and
overrides exactly two things:

  * the model build is skipped — the faithful path reads the SAME baked model
    the runner would build (passed in or built CPU-only via
    ``compile_full_vm_dynamic``);
  * :meth:`_faithful_next_token` replaces the batched GPU forward with the
    per-token faithful CPU forward at the row's last real position.

Everything else (``run_batch_fail_fast`` -> ``_build_element`` ->
``_oracle_pc_ax_steps`` -> the per-step verdict, and ``run_batch`` -> the
exit-code verdict) is the UNMODIFIED production code path. That is what makes
the CPU verdict byte-identical to the neural verdict by construction: the only
substitution is mathematically argmax-equivalent (proven by
``faithful_interpreter_validate``).

This module is TOOLING ONLY. It is never imported by ``compile_full_vm_dynamic``
or any build path, so the model stays byte-identical (golden ``ce9bf9616f3379c4``
unchanged).
"""

from __future__ import annotations

import contextlib
import io
import os
from typing import List, Optional, Sequence, Tuple

import torch

from .faithful_interpreter import CachedFaithfulForward


def build_cpu_model(*, disk_cache: bool = True):
    """Build the real baked ``AutoregressiveVM`` on CPU (cached compile).

    Identical to ``tools/faithful_interpreter_validate.build_model`` — the SAME
    baked weights the smoke-gate / canonical runner uses, just pinned to CPU and
    with the dim-integrity / gate checks skipped (they do not affect weights).
    Returns ``(model, layout)``. ``layout.dim_positions`` is the BUILT layout —
    use it (not the static registry) to resolve any residual dim.
    """
    os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
    os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
    os.environ.setdefault("C4_TEST_SPEC_K", "0")
    os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
    from .full_vm_compiler_dynamic import compile_full_vm_dynamic

    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = compile_full_vm_dynamic(disk_cache=disk_cache)
    model = model.to("cpu")
    model.eval()
    return model, layout


class FaithfulAutoregressiveRunner:
    """Drive the production fail-fast / exit-code decode with a faithful CPU forward.

    Construct with a CPU-baked model (or let it build one). The public
    :meth:`run_batch_fail_fast` and :meth:`run_batch` methods are the production
    ``BatchedPureNeuralRunner`` ones, so the verdict logic is identical; the only
    difference is the per-token argmax comes from the faithful CPU forward.

    Usage::

        runner = FaithfulAutoregressiveRunner()
        verdicts = runner.run_batch_fail_fast(
            [bytecode], data_list=[data], expected_steps_list=[oracle_steps],
            max_steps=None, criterion="full_trace",
        )
        # verdicts[0]["status"] in {"pass", "fail", "error"}, byte-identical to
        # tools/run_1096_canonical.py --criterion full_trace (spec_k=0).
    """

    def __init__(self, model=None, layout=None, *, disk_cache: bool = True):
        if model is None:
            model, layout = build_cpu_model(disk_cache=disk_cache)
        self.model = model
        self.layout = layout
        self.dim_positions = dict(getattr(layout, "dim_positions", {}) or {})
        # The production runner's decode machinery, with the GPU forward
        # overridden. We compose (not deep-inherit) to keep the override surface
        # one method; the production class supplies the (unmodified) loop logic.
        self._inner = _build_faithful_inner(model, self)
        # Stats for the speed envelope report.
        self.n_forwards = 0

    # -- public production-equivalent entry points -------------------------

    def run_batch_fail_fast(self, *args, **kwargs):
        """Production per-step (PC, AX) / token-identity verdict, faithful CPU.

        Same signature + semantics as
        ``BatchedPureNeuralRunner.run_batch_fail_fast``. The verdict for each
        program is byte-identical to the neural run because the decode tape is
        byte-identical (faithful argmax == neural argmax) and the verdict logic
        is the unmodified production code.
        """
        return self._inner.run_batch_fail_fast(*args, **kwargs)

    def run_batch(self, *args, **kwargs):
        """Production exit-code verdict, faithful CPU forward."""
        return self._inner.run_batch(*args, **kwargs)

    @property
    def forward_count(self) -> int:
        return self._inner.faithful_forward_count


_FAITHFUL_SUBCLASS = None


def _build_faithful_inner(model, owner):
    """Instantiate the faithful-forward subclass of ``BatchedPureNeuralRunner``.

    Subclassing is done lazily (here, not at import) so importing this module
    never imports the heavy runner unless a faithful run is actually requested.
    The single overridden method is ``_forward_argmax_batch`` — the production
    decode loops (``_run_unspeculative`` / ``_run_fail_fast``) call it for the
    per-token argmax, and we serve those argmaxes from the faithful CPU forward.
    """
    global _FAITHFUL_SUBCLASS
    if _FAITHFUL_SUBCLASS is None:
        from neural_vm.batched_pure_neural import BatchedPureNeuralRunner

        _FAITHFUL_SUBCLASS = _make_faithful_subclass(BatchedPureNeuralRunner)
    return _FAITHFUL_SUBCLASS(model, owner)


def _make_faithful_subclass(base):
    """Return a subclass of ``BatchedPureNeuralRunner`` whose per-token argmax
    comes from the faithful CPU forward instead of the batched GPU forward.

    Only ``_forward_argmax_batch`` is overridden — the smallest possible surface
    that intercepts every per-token decode while leaving ``_run_unspeculative``,
    ``_run_fail_fast``, ``_step_one``, ``_dispatch_pure_neural``,
    ``_ff_check_new_steps`` and ``_oracle_pc_ax_steps`` as the UNMODIFIED
    production code (so the verdict logic is byte-identical).
    """

    class _FaithfulSubclass(base):  # type: ignore[valid-type, misc]
        def __init__(self, model, owner):
            # Bypass the heavy ``BatchedPureNeuralRunner.__init__`` (it builds /
            # wraps a serial runner + KV cache); install only what the decode
            # loops read. The faithful path needs no KV cache (it never reuses a
            # prefix) and no CSR shim (it runs the recovered dense weights).
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
            self.faithful_forward_count = 0
            self._reset_spec_stats()
            # The faithful CPU forward with the per-block recovery cached ONCE
            # (recovery is tape-independent; caching it makes the per-token
            # forward ~15x faster than re-recovering every call).
            self._faithful = CachedFaithfulForward(model)
            # A serial-runner shim for ``_build_context`` (the prompt prefix
            # builder). Build one CPU runner lazily — it shares the SAME model.
            self._serial = _ContextShim(model)

        # ---- the ONE override: per-token argmax from the faithful forward ----
        #
        # IMPORTANT — speculation is NOT suppressed. The production fail-fast
        # path teacher-forces the DraftVM tokens at the UNSAFE MEM offsets (26..33
        # / 21..28) — those bytes are ALWAYS accepted from the draft because the
        # embedding's MEM-metadata injection makes them unreadable from a flat
        # forward argmax (see ``_run_fail_fast``). If we decoded the MEM bytes
        # from the model's own argmax instead (i.e. suppressed speculation), the
        # WRONG MEM bytes would feed back and corrupt the NEXT step's decode,
        # making the CPU verdict DISAGREE with the neural canonical run (which
        # runs spec_k=32). So the faithful CPU decode keeps the production
        # speculative loop verbatim and only swaps the per-token argmax source —
        # which is exactly what makes the verdict byte-identical to neural.

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
            """Faithful per-row argmax at the requested gather positions.

            Mirrors the production return contract: ``(preds_cpu, pred_start,
            real_lens)`` where ``preds_cpu[b]`` is the list of argmax token ids
            at ``gather_positions[b]`` for row ``b`` (the production code passes
            ``[real_len - 1]`` in the unspeculative path and a per-draft list in
            the speculative path). Each row runs an INDEPENDENT faithful forward
            over its own (untrimmed) context — exactly what the neural model does
            on the padded batch, minus the padding (faithful forward is per-row
            so there is no cross-row padding interaction to reproduce; the neural
            batch forward is causal + per-row positional, so a row's argmax is
            independent of its batch-mates and padding, which makes the per-row
            faithful forward byte-identical).
            """
            real_lens = [len(s) for s in sequences]
            preds_cpu: List[List[int]] = []
            for b, seq in enumerate(sequences):
                if gather_positions is not None:
                    gp = list(gather_positions[b])
                else:
                    gp = [real_lens[b] - 1]
                logits = self._faithful.forward(list(seq), return_logits=True)
                self.faithful_forward_count += 1
                row_argmax = logits.argmax(dim=-1)  # [S]
                preds_cpu.append([int(row_argmax[p]) for p in gp])
            pred_start = (
                min(real_lens) - 1 if first_logit_pos is None else first_logit_pos
            )
            return preds_cpu, pred_start, real_lens

    _FaithfulSubclass.__name__ = "FaithfulAutoregressiveInner"
    return _FaithfulSubclass


class _ContextShim:
    """Minimal stand-in for the serial runner: supplies ``_build_context``.

    ``BatchedPureNeuralRunner._build_element`` calls ``self._serial._build_context``
    to build the prompt prefix (the seed tokens before the first generated
    token). We reuse the REAL ``AutoregressiveVMRunner._build_context`` bound to
    the shared CPU model so the prefix is byte-identical, without building a
    second model copy.
    """

    def __init__(self, model):
        self.model = model

    def _build_context(self, bytecode, data, argv, stdin=""):
        # Lazily import + bind the real method (it is a pure tokeniser: it reads
        # the model's embedding vocab constants, not any GPU state).
        from neural_vm.run_vm import AutoregressiveVMRunner

        return AutoregressiveVMRunner._build_context(
            self, bytecode, data, argv, stdin
        )


__all__ = [
    "FaithfulAutoregressiveRunner",
    "build_cpu_model",
]
