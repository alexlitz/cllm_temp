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
the padded tensor ``[B, max_len]``. Each program keeps its own context, observed
register tracking (last_pc/ax/sp/bp) and output buffer. Programs terminate
independently: once a program emits HALT, its slot is no longer read out of the
next forward pass, but it stays in the batch tensor (filled with HALT tokens)
until all programs finish.

Pure forward-pass contract: the runner appends model-emitted tokens verbatim
to each element's context and never re-writes register bytes from Python. The
only Python-side semantics are: PUTCHAR byte append (tool-boundary side
effect), EXIT halt detection, and a neural-authoritative "next instruction is
EXIT" early stop. The neural model is the sole authority for AX/PC/SP/BP/STACK0
and for every MEM section.

Output equivalence:
    For each batch element, the produced ``(output_string, exit_code)`` must
    match the serial pure-neural runner running that program alone. See
    ``test_batched_pure_neural.py`` for element-by-element verification.
"""

from __future__ import annotations

import os
import torch
from collections import deque
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .vm_step import Token, DEFAULT_N_HEADS, DEFAULT_FFN_HIDDEN
from .embedding import Opcode
from .constants import INSTR_WIDTH, PC_OFFSET
from .run_vm import (
    AutoregressiveVMRunner,
)
from .speculative import DraftVM


# Adaptive speculative-K tuning knobs. The runner treats ``spec_k < 0`` as
# the adaptive sentinel. ``spec_k == 0`` is the raw neural path: one model
# token per forward with no speculative DraftVM. In adaptive mode each element
# tracks its own ``adaptive_k`` starting at ``_ADAPTIVE_START_K`` and a rolling
# rejection rate over the last ``_ADAPTIVE_WINDOW`` batched-forward iterations.
# On high rejection (>= ``_ADAPTIVE_BACKOFF_THRESHOLD``) the per-element K
# halves (floor ``_ADAPTIVE_MIN_K``); on consistent acceptance
# (<= ``_ADAPTIVE_RAMP_THRESHOLD``) it doubles (cap ``_ADAPTIVE_MAX_K``).
# Per-element bookkeeping lets a batch that mixes programs (some perfectly
# predicted, some flaky) auto-tune each slot independently.
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


_ADAPTIVE_START_K = max(1, _env_int("C4_ADAPTIVE_START_K", 32))
_ADAPTIVE_MIN_K = max(1, _env_int("C4_ADAPTIVE_MIN_K", 1))
_ADAPTIVE_MAX_K = max(_ADAPTIVE_MIN_K, _env_int("C4_ADAPTIVE_MAX_K", 64))
_ADAPTIVE_WINDOW = 10
_ADAPTIVE_BACKOFF_THRESHOLD = 0.5
_ADAPTIVE_RAMP_THRESHOLD = 0.1


# Bucket boundaries (in predicted VM steps) for length-aware batching. A
# program with predicted_steps P lands in the smallest bucket whose upper
# bound is >= P. The "+inf" tail bucket holds anything DraftVM couldn't
# bound (e.g. programs that hit the prediction-cap without halting, or
# programs whose DraftVM raised an error). Tuned with a 2x ratio so each
# bucket's slowest program is at most 2x the bucket's fastest, which caps
# wasted padding at ~50%.
#
# Within a bucket programs still pad to the longest member's length. The
# wall-time win is: instead of one batch padded to max(P_i), we get B
# batches each padded to its own max — usually a big win when the input
# has wide length variance (1098-suite typical).
_DEFAULT_BUCKET_BOUNDS: tuple = (10, 20, 40, 80, 160, 320, 640, 1280, 2560)
# Sentinel for unbounded / unpredicted programs.
_UNPREDICTED_BUCKET_KEY = "unpredicted"


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

# Some opcodes are neural-authoritative in raw one-token decoding but are not
# safe speculation boundaries yet. In particular, call-frame control changes
# rewrite PC/SP/BP in ways where a long DraftVM continuation can poison the
# verifier tensor and produce persistent first-token rejection. We still test
# these ops through the neural path; we just do not append DraftVM futures for
# them.
_SPEC_UNSAFE_OPS = frozenset({
    int(Opcode.JSR),
    int(Opcode.ENT),
    int(Opcode.LEV),
    int(Opcode.OPEN),
    int(Opcode.READ),
    int(Opcode.CLOS),
    int(Opcode.PRTF),
    int(Opcode.GETCHAR),
    int(Opcode.PUTCHAR),
})


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
    exit_code: Optional[int] = 0

    last_pc: Optional[int] = None
    last_ax: int = 0
    last_sp: int = 0x10000
    last_bp: int = 0x10000

    memory: dict = field(default_factory=dict)        # addr -> byte

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
    # Declarative/DraftVM-predicted halt horizon. When set, this is not an
    # arbitrary test cap: it is the exact number of VM steps the declarative
    # program semantics need to halt. If the neural path is still running after
    # this many generated steps, it has diverged.
    expected_steps: Optional[int] = None

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
        n_heads=DEFAULT_N_HEADS,
        ffn_hidden=DEFAULT_FFN_HIDDEN,
        max_seq_len=4096,
        use_kv_cache: Optional[bool] = None,
        kv_cache_max_tokens: Optional[int] = None,
        kv_cache_verify: Optional[bool] = None,
        kv_cache_verify_interval: Optional[int] = None,
        enable_moe_routing: Optional[bool] = None,
        csr_inference: Optional[bool] = None,
        compact_gather: Optional[bool] = None,
    ):
        """``csr_inference`` enables the explicit CSR inference path
        (commit cb0f396f bench: 2.11x CUDA speedup at 99.9% argmax-match).
        ``None`` reads ``C4_CSR_INFERENCE`` and otherwise keeps dense
        weights, preserving byte-identity-oriented tests by default.
        Only consulted when ``model_runner`` is None; if an externally-built
        runner is supplied its CSR mode is taken from that runner.

        ``compact_gather`` enables the byte-identical column-shrink path
        (2026-06-06 bench: 1.17x CUDA speedup at 100% argmax match, max
        diff 0.0). ``None`` reads ``C4_COMPACT_GATHER`` and otherwise
        keeps dense weights. Only consulted when ``model_runner`` is
        None. Mutually exclusive with ``csr_inference``.
        """
        if use_kv_cache is None:
            use_kv_cache = os.environ.get("C4_BATCH_USE_KV_CACHE") == "1"
        if kv_cache_verify is None:
            raw_verify = os.environ.get("C4_BATCH_KV_VERIFY", "0").strip().lower()
            kv_cache_verify = raw_verify not in {"", "0", "false", "no", "off"}
            if kv_cache_verify_interval is None:
                if raw_verify in {"sample", "sampled", "periodic"}:
                    kv_cache_verify_interval = _env_int(
                        "C4_BATCH_KV_VERIFY_INTERVAL", 32
                    )
                else:
                    try:
                        numeric_verify = int(raw_verify)
                    except ValueError:
                        numeric_verify = 1
                    kv_cache_verify_interval = max(1, numeric_verify)
        if kv_cache_verify_interval is None:
            kv_cache_verify_interval = _env_int("C4_BATCH_KV_VERIFY_INTERVAL", 1)
        if kv_cache_max_tokens is None:
            raw_max_tokens = os.environ.get("C4_BATCH_KV_MAX_TOKENS")
            if raw_max_tokens:
                kv_cache_max_tokens = int(raw_max_tokens)
            elif use_kv_cache:
                # Neural memory loads read historical MEM tokens through K/V.
                # A small sliding window silently turns older stores into ZFOD;
                # keep a long cache by default and let explicit env settings
                # opt back into bounded eviction experiments.
                kv_cache_max_tokens = 65_536
            else:
                kv_cache_max_tokens = None
        if enable_moe_routing is None:
            enable_moe_routing = (
                os.environ.get("C4_BATCH_ENABLE_MOE_ROUTING") == "1"
                or os.environ.get("C4_ENABLE_MOE_ROUTING") == "1"
            )
        if csr_inference is None:
            # Default ON. CSR matmul gives 2.11× speedup at B=8 and up to
            # 3.51× at B=128 with 99.9% argmax match (fp32 sum-order noise
            # — see SPARSE_INFERENCE_BENCHMARK_2026_06_06.md). Opt out with
            # ``C4_CSR_INFERENCE=0`` for byte-identity-required test paths.
            _csr_env = os.environ.get("C4_CSR_INFERENCE", "1").strip().lower()
            csr_inference = _csr_env not in {"0", "false", "no", "off"}
        if compact_gather is None:
            compact_gather = (
                os.environ.get("C4_COMPACT_GATHER", "").strip().lower()
                in {"1", "true", "yes", "on"}
            )
        if model_runner is None:
            model_runner = AutoregressiveVMRunner(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                ffn_hidden=ffn_hidden,
                max_seq_len=max_seq_len,
                pure_neural=True,
                trust_neural_alu=True,
                enable_moe_routing=enable_moe_routing,
                csr_inference=csr_inference,
                compact_gather=compact_gather,
            )
            model_runner._func_call_handlers = {}
            model_runner._syscall_handlers = {}
        elif csr_inference:
            # Externally-built model_runner: defensively re-install the CSR
            # F.linear shim so that if a non-CSR runner was constructed
            # earlier in this process and restored F.linear, our CSR
            # weights still dispatch correctly. Idempotent.
            from .base_layers import install_csr_linear_shim
            install_csr_linear_shim()
        self._serial = model_runner
        self.model = model_runner.model
        self._device = next(self.model.parameters()).device
        self.enable_moe_routing = bool(
            getattr(model_runner, "enable_moe_routing", enable_moe_routing)
        )
        self.use_kv_cache = bool(use_kv_cache)
        self.incremental_kv_safe = (
            os.environ.get("C4_BATCH_FORCE_INCREMENTAL_KV", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.kv_cache_verify = bool(kv_cache_verify)
        self.kv_cache_verify_interval = max(1, int(kv_cache_verify_interval))
        # Default OFF: persistent first-token rejection is not terminal when
        # the K=0 path can still complete via single-token decode. The legacy
        # "1" default was tuned before the L10/L16 PSH-addr guards (commit
        # 0ce6030) shifted some model boundary predictions by one position;
        # since then hard-halting hides recoverable exits. Set
        # ``C4_SPEC_FAIL_FAST=1`` to opt back in for benchmarking.
        self.spec_fail_fast = (
            os.environ.get("C4_SPEC_FAIL_FAST", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.spec_fail_on_correction = (
            os.environ.get("C4_SPEC_FAIL_ON_CORRECTION", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.kv_cache_max_tokens = int(
            kv_cache_max_tokens or getattr(self.model, "max_seq_len", max_seq_len)
        )
        self._kv_cache_eviction_overshoot = max(
            1,
            _env_int("C4_BATCH_KV_EVICTION_OVERSHOOT", Token.STEP_TOKENS),
        )
        self._kv_cache_storage_max_tokens = self.kv_cache_max_tokens
        if self.use_kv_cache and self.incremental_kv_safe:
            self._kv_cache_storage_max_tokens += self._kv_cache_eviction_overshoot
        self.kv_flush_interval = max(
            0,
            _env_int("C4_BATCH_KV_FLUSH_INTERVAL", 0),
        )
        self._kv_cache_obj = None
        self._kv_active_idx: Optional[Tuple[int, ...]] = None
        self._kv_cached_rows: List[List[int]] = []
        self._kv_incremental_count = 0
        self._kv_stats = {
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
        self._spec_stats = {}
        self._reset_spec_stats()

    # ------------------------------------------------------------------
    # Batched KV cache
    # ------------------------------------------------------------------

    def _reset_kv_cache(self) -> None:
        self._kv_cache_obj = None
        self._kv_active_idx = None
        self._kv_cached_rows = []
        self._kv_incremental_count = 0

    def _reset_spec_stats(self) -> None:
        self._spec_stats = {
            "iterations": 0,
            "drafted": 0,
            "accepted": 0,
            "full_accepts": 0,
            "corrections": 0,
            "first_token_rejects": 0,
            "fail_fast": 0,
            "no_draft_slots": 0,
            "unsafe_opcode_stops": 0,
            "max_effective_k": 0,
        }

    def _get_or_build_kv_cache(self):
        if not self.use_kv_cache:
            return None
        if self._kv_cache_obj is not None:
            return self._kv_cache_obj
        from .kv_cache import LayerKVCache

        first_attn = self.model.blocks[0].attn
        self._kv_cache_obj = LayerKVCache(
            num_layers=len(self.model.blocks),
            max_tokens=getattr(
                self,
                "_kv_cache_storage_max_tokens",
                self.kv_cache_max_tokens,
            ),
            num_heads=first_attn.num_heads,
            head_dim=first_attn.head_dim,
            device=self._device,
        )
        return self._kv_cache_obj

    @staticmethod
    def _common_prefix_len(a: List[int], b: List[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _trim_kv_cache(self, keep_len: int) -> None:
        if self._kv_cache_obj is None:
            return
        keep_len = int(max(0, keep_len))
        for layer_cache in self._kv_cache_obj.caches:
            if layer_cache.cached_k is None:
                continue
            cur = layer_cache.cache_size
            pos_ids = getattr(layer_cache, "cached_pos_ids", None)
            if pos_ids is not None and pos_ids.shape[-1] == cur:
                keep_idx = (pos_ids[0] < keep_len).nonzero(
                    as_tuple=False
                ).flatten()
                if int(keep_idx.numel()) == cur:
                    layer_cache.next_pos_id = keep_len
                    continue
                if int(keep_idx.numel()) <= 0:
                    self._clear_layer_kv_storage(
                        layer_cache,
                        next_pos_id=keep_len,
                    )
                else:
                    self._index_select_layer_kv(
                        layer_cache,
                        keep_idx,
                        next_pos_id=keep_len,
                    )
                continue

            keep_count = min(keep_len, cur)
            if keep_count >= cur:
                continue
            if keep_count <= 0:
                self._clear_layer_kv_storage(layer_cache, next_pos_id=0)
            else:
                keep_idx = torch.arange(
                    keep_count,
                    dtype=torch.long,
                    device=layer_cache.cached_k.device,
                )
                self._index_select_layer_kv(
                    layer_cache,
                    keep_idx,
                    next_pos_id=keep_count,
                )
        self._kv_cached_rows = [row[:keep_len] for row in self._kv_cached_rows]

    @staticmethod
    def _clear_layer_kv_storage(layer_cache, *, next_pos_id: int = 0) -> None:
        layer_cache.cached_k = None
        layer_cache.cached_v = None
        layer_cache.cached_pos_ids = None
        layer_cache.per_head_keep_mask = None
        layer_cache.cache_size = 0
        layer_cache.next_pos_id = int(max(0, next_pos_id))
        layer_cache.stats.current_size = 0

    @staticmethod
    def _index_select_layer_kv(layer_cache, keep_idx, *, next_pos_id: int) -> None:
        layer_cache.cached_k = layer_cache.cached_k.index_select(
            2, keep_idx
        ).contiguous()
        layer_cache.cached_v = layer_cache.cached_v.index_select(
            2, keep_idx
        ).contiguous()
        if layer_cache.cached_pos_ids is not None:
            layer_cache.cached_pos_ids = layer_cache.cached_pos_ids.index_select(
                1, keep_idx
            ).contiguous()
        if layer_cache.per_head_keep_mask is not None:
            layer_cache.per_head_keep_mask = (
                layer_cache.per_head_keep_mask.index_select(2, keep_idx)
                .contiguous()
            )
        layer_cache.cache_size = int(keep_idx.numel())
        layer_cache.next_pos_id = int(max(0, next_pos_id))
        layer_cache.stats.current_size = layer_cache.cache_size

    @staticmethod
    def _kv_position_is_protected(
        pos: int,
        *,
        max_prefix_len: int,
        protected_mem_positions: Optional[List[List[int]]],
    ) -> bool:
        if pos < max_prefix_len:
            return True
        if not protected_mem_positions:
            return False
        for starts in protected_mem_positions:
            for start in starts:
                start = int(start)
                if start <= pos < start + 9:
                    return True
        return False

    def _kv_cache_append_is_aligned(
        self,
        cached_prefix_len: int,
        new_tokens: int,
    ) -> bool:
        """Return True when bounded overflow can append without stale K/V.

        This intentionally rejects cold/rebuild overflow. A full forward with
        ``S > max_tokens`` would let ``TransformerKVCache.update`` discard K/V
        before attention consumes it. The bounded path is only valid when the
        existing cache is an ordered suffix ending at ``cached_prefix_len - 1``
        and has enough physical overshoot room for the next incremental suffix.
        """
        if self._kv_cache_obj is None or cached_prefix_len <= 0 or new_tokens <= 0:
            return False
        for layer_cache in self._kv_cache_obj.caches:
            if layer_cache.cached_k is None:
                return False
            pos_ids = getattr(layer_cache, "cached_pos_ids", None)
            if pos_ids is None or pos_ids.shape[-1] != layer_cache.cache_size:
                return False
            if layer_cache.cache_size <= 0:
                return False
            if layer_cache.cache_size + new_tokens > layer_cache.max_tokens:
                return False
            pos_1d = pos_ids[0]
            if int(pos_1d[-1].item()) != cached_prefix_len - 1:
                return False
            if int(layer_cache.next_pos_id) != cached_prefix_len:
                return False
            if bool((pos_1d >= cached_prefix_len).any().item()):
                return False
            if pos_1d.numel() > 1 and bool((pos_1d[1:] <= pos_1d[:-1]).any().item()):
                return False
        return True

    def _prune_kv_cache_to_budget(
        self,
        *,
        max_tokens: int,
        protected_prefix_lens: Optional[List[int]],
        protected_mem_positions: Optional[List[List[int]]],
    ) -> Tuple[bool, int]:
        """Prune old unprotected cached positions back to ``max_tokens``.

        Protected positions are the immutable bytecode/data prefix and tracked
        MEM-store sections. Those are known long-range dependencies for fetch
        and neural memory loads, so if the requested budget cannot retain them
        plus a small recent dynamic tail, the caller must discard the cache
        instead of producing a silently stale bounded-cache result.
        """
        if self._kv_cache_obj is None:
            return True, 0

        ref = None
        for layer_cache in self._kv_cache_obj.caches:
            if layer_cache.cached_k is not None:
                ref = layer_cache
                break
        if ref is None or ref.cache_size <= max_tokens:
            return True, 0

        ref_pos_ids = getattr(ref, "cached_pos_ids", None)
        if ref_pos_ids is None or ref_pos_ids.shape[-1] != ref.cache_size:
            return False, 0
        positions = ref_pos_ids[0]
        old_size = int(ref.cache_size)

        for layer_cache in self._kv_cache_obj.caches:
            if layer_cache.cached_k is None:
                continue
            pos_ids = getattr(layer_cache, "cached_pos_ids", None)
            if (
                layer_cache.cache_size != old_size
                or pos_ids is None
                or pos_ids.shape[-1] != old_size
                or not torch.equal(pos_ids[0].to(positions.device), positions)
            ):
                return False, 0

        max_prefix_len = max(protected_prefix_lens or [0])
        protected_flags = [
            self._kv_position_is_protected(
                int(pos),
                max_prefix_len=max_prefix_len,
                protected_mem_positions=protected_mem_positions,
            )
            for pos in positions.tolist()
        ]
        protected_mask = torch.tensor(
            protected_flags,
            dtype=torch.bool,
            device=positions.device,
        )
        protected_count = int(protected_mask.sum().item())
        target = int(max(0, max_tokens))
        if protected_count > target:
            return False, 0

        keep_unprotected_budget = target - protected_count
        nonprotected_idx = (~protected_mask).nonzero(as_tuple=False).flatten()
        nonprotected_count = int(nonprotected_idx.numel())
        min_recent_dynamic = min(Token.STEP_TOKENS * 2, nonprotected_count)
        if (
            nonprotected_count > keep_unprotected_budget
            and keep_unprotected_budget < min_recent_dynamic
        ):
            return False, 0

        keep_mask = protected_mask.clone()
        if keep_unprotected_budget > 0 and nonprotected_count > 0:
            keep_mask[nonprotected_idx[-keep_unprotected_budget:]] = True
        keep_idx = keep_mask.nonzero(as_tuple=False).flatten()
        if int(keep_idx.numel()) == old_size:
            return True, 0
        if int(keep_idx.numel()) <= 0 or int(keep_idx[-1].item()) != old_size - 1:
            return False, 0

        self._kv_cache_obj.prune(keep_idx)
        return True, old_size - int(keep_idx.numel())

    def _batched_kv_common_prefix(
        self, active_idx: List[int], sequences: List[List[int]]
    ) -> int:
        active_tuple = tuple(active_idx)
        if self._kv_active_idx != active_tuple:
            self._reset_kv_cache()
            self._kv_active_idx = active_tuple
            return 0
        if len(self._kv_cached_rows) != len(sequences):
            return 0
        if not self._kv_cached_rows:
            return 0
        return min(
            self._common_prefix_len(cached, current)
            for cached, current in zip(self._kv_cached_rows, sequences)
        )

    def _forward_argmax_batch(
        self,
        sequences: List[List[int]],
        active_idx: List[int],
        *,
        first_logit_pos: int,
        allow_kv: bool = True,
        protected_prefix_lens: Optional[List[int]] = None,
        protected_mem_positions: Optional[List[List[int]]] = None,
        gather_positions: Optional[List[List[int]]] = None,
    ) -> Tuple[List[List[int]], int, List[int]]:
        """Forward a padded active batch and return argmax rows.

        When batched KV is enabled, all active rows share one
        ``cached_prefix_len``. That prefix is the longest unchanged prefix
        common to every active row, capped to the earliest logit any caller
        will inspect. If a cached pass disagrees with a fresh pass, the fresh
        logits are returned and the cache is discarded.

        Perf 2026-06-05: when ``gather_positions`` is provided (per-row
        absolute positions in the *padded* tensor that the caller actually
        needs), only those positions are argmaxed and transferred to CPU.
        This shrinks the per-step host transfer from ``[B, max_len]`` longs
        to ``[B, P]`` longs (where P is typically 1 for the unspec path and
        ``K*35`` for the speculative path). On the 32-case profile the
        ``.cpu()`` dominated wall time (37%); the gather path takes it to
        a small fraction of a step. ``preds_cpu`` shape is ``[B, P]`` and
        ``pred_start`` is the implicit ``0`` (callers index relative to
        their own ``gather_positions``). When ``gather_positions`` is
        ``None`` the legacy ``[B, max_len-pred_start]`` return is preserved.
        """
        padded, real_lens = self._pad_to_tensor(sequences)
        gather_idx = self._build_gather_idx(gather_positions, padded.device) \
            if gather_positions is not None else None
        if (
            not self.use_kv_cache
            or not allow_kv
            or not self.incremental_kv_safe
        ):
            if self.use_kv_cache and not allow_kv:
                self._kv_stats["spec_fresh_bypass"] += 1
            elif self.use_kv_cache and not self.incremental_kv_safe:
                # Expanded Neural VM blocks include ALU/composite modules that
                # read fixed sequence slots (for example x[:, 0]) as global
                # control lanes. Slicing to only the new suffix changes those
                # semantics, so incremental KV is not parity-safe for this
                # model yet. Keep C4_BATCH_USE_KV_CACHE as an opt-in knob, but
                # prefer correctness unless C4_BATCH_FORCE_INCREMENTAL_KV=1 is
                # explicitly set for experiments.
                self._kv_stats["unsafe_model_fresh_bypass"] += 1
            self._kv_stats["fresh_forwards"] += 1
            logits = self.model.forward(padded)
            return self._argmax_to_cpu(logits, gather_idx), 0, real_lens

        max_len = padded.shape[1]
        first_logit_pos = int(max(0, min(first_logit_pos, max_len - 1)))

        if (
            self.kv_flush_interval > 0
            and self._kv_incremental_count >= self.kv_flush_interval
        ):
            self._reset_kv_cache()

        prefix_match = self._batched_kv_common_prefix(active_idx, sequences)
        cached_prefix_len = min(prefix_match, first_logit_pos)
        self._trim_kv_cache(cached_prefix_len)

        bounded_overflow = max_len > self.kv_cache_max_tokens
        if bounded_overflow:
            self._kv_stats["eviction_pressure"] += 1
            new_tokens = max_len - cached_prefix_len
            if not self._kv_cache_append_is_aligned(
                cached_prefix_len,
                new_tokens,
            ):
                self._kv_stats["bounded_eviction_fallbacks"] += 1
                self._reset_kv_cache()
                self._kv_stats["fresh_forwards"] += 1
                logits = self.model.forward(padded)
                return self._argmax_to_cpu(logits, gather_idx), 0, real_lens

        kv_cache = self._get_or_build_kv_cache()
        self._kv_stats["calls"] += 1
        self._kv_stats["kv_forwards"] += 1
        self._kv_stats["cache_rebuilds"] += int(cached_prefix_len == 0)
        self._kv_stats["reused_token_slots"] += cached_prefix_len * len(sequences)
        logits = self.model.forward(
            padded,
            kv_cache=kv_cache,
            cached_prefix_len=cached_prefix_len,
        )
        # KV path returns logits sliced from ``cached_prefix_len``; offset the
        # gather indices accordingly so the gather still picks the absolute
        # positions the caller asked for.
        kv_gather_idx = None
        if gather_idx is not None:
            kv_gather_idx = (gather_idx - cached_prefix_len).clamp_min(0)
        cached_preds = self._argmax_to_cpu(logits, kv_gather_idx)

        if (
            self.kv_cache_verify
            and cached_prefix_len > 0
            and (
                self.kv_cache_verify_interval <= 1
                or self._kv_stats["calls"] % self.kv_cache_verify_interval == 0
            )
        ):
            self._kv_stats["verifications"] += 1
            self._kv_stats["verification_forwards"] += 1
            fresh_logits = self.model.forward(padded)
            fresh_preds = self._argmax_to_cpu(fresh_logits, gather_idx)
            if gather_idx is None:
                fresh_tail = [row[cached_prefix_len:] for row in fresh_preds]
            else:
                fresh_tail = fresh_preds
            if cached_preds != fresh_tail:
                self._kv_stats["mismatches"] += 1
                self._kv_stats["fallbacks"] += 1
                self._reset_kv_cache()
                return fresh_preds, 0, real_lens

        cache_still_valid = True
        if bounded_overflow:
            cache_still_valid, evicted = self._prune_kv_cache_to_budget(
                max_tokens=self.kv_cache_max_tokens,
                protected_prefix_lens=protected_prefix_lens,
                protected_mem_positions=protected_mem_positions,
            )
            if not cache_still_valid:
                self._kv_stats["bounded_eviction_fallbacks"] += 1
                self._reset_kv_cache()
            elif evicted > 0:
                self._kv_stats["bounded_evictions"] += 1
                self._kv_stats["bounded_positions_evicted"] += evicted

        self._kv_stats["hits"] += int(cached_prefix_len > 0)
        if cache_still_valid:
            self._kv_active_idx = tuple(active_idx)
            self._kv_cached_rows = [list(seq) for seq in sequences]
            self._kv_incremental_count += 1
        # When ``gather_positions`` is set the caller treats positions as
        # absolute (independent of ``cached_prefix_len``); preserve the
        # ``pred_start=0`` convention for that mode by returning 0 here.
        if gather_idx is not None:
            return cached_preds, 0, real_lens
        return cached_preds, cached_prefix_len, real_lens

    @staticmethod
    def _build_gather_idx(
        gather_positions: List[List[int]],
        device,
    ) -> torch.Tensor:
        """Right-pad per-row position lists into a single ``[B, P]`` LongTensor.

        Empty rows fall back to position 0 (their argmax result is unused
        because the caller knows that row had no requested positions). Ragged
        rows are tolerated; callers use ``len(gather_positions[b])`` to slice
        their per-row preds back to the right length.
        """
        if not gather_positions:
            return torch.zeros((0, 0), dtype=torch.long, device=device)
        B = len(gather_positions)
        # Pre-compute the per-row length and the global max so we know the
        # padded width. Pad to ``max_p`` with 0; callers consume only the
        # leading ``len(row)`` entries.
        max_p = 0
        for row in gather_positions:
            if len(row) > max_p:
                max_p = len(row)
        if max_p == 0:
            return torch.zeros((B, 0), dtype=torch.long, device=device)
        flat = []
        for row in gather_positions:
            if len(row) == max_p:
                flat.extend(row)
            else:
                flat.extend(row)
                flat.extend([0] * (max_p - len(row)))
        return torch.tensor(flat, dtype=torch.long, device=device).view(B, max_p)

    @staticmethod
    def _argmax_to_cpu(
        logits: torch.Tensor,
        gather_idx: Optional[torch.Tensor],
    ) -> List[List[int]]:
        """Argmax over vocab and transfer to CPU.

        When ``gather_idx`` is a ``[B, P]`` LongTensor we gather just those
        positions before the host transfer (cutting ``cpu()`` time from
        ``O(B*S)`` to ``O(B*P)``). When ``gather_idx`` is None we fall back to
        the legacy full ``[B, S]`` argmax-then-transfer.
        """
        if gather_idx is None:
            return logits.argmax(dim=-1).cpu().tolist()
        # Logits: [B, S, V]. Compute argmax along V first (B*S int64 result),
        # then gather only the requested positions along S. This avoids
        # materialising the V dim during the transfer.
        amax = logits.argmax(dim=-1)  # [B, S]
        if gather_idx.numel() == 0:
            return [[] for _ in range(amax.shape[0])]
        # Clamp gather indices into the valid range of ``amax``'s S dim. Padded
        # rows (those whose caller passed an empty list) get 0 here; the caller
        # ignores those slots because it slices by its own row lengths.
        s_dim = amax.shape[1]
        clamped = gather_idx.clamp_max(s_dim - 1) if s_dim > 0 else gather_idx
        gathered = amax.gather(1, clamped)  # [B, P]
        return gathered.cpu().tolist()

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
        expected_steps: Optional[int] = None,
    ) -> _ElementState:
        ctx = self._serial._build_context(bytecode, data or b"", argv or [], stdin or "")
        st = _ElementState(
            bytecode=list(bytecode),
            context=list(ctx),
            prefix_len=len(ctx),
            stdin_buffer=list(stdin) if stdin else [],
            expected_steps=expected_steps,
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
        max_steps: Optional[int] = 100,
        max_context_window: int = 512,
        spec_k: int = 0,
        expected_steps_list: Optional[List[Optional[int]]] = None,
        bucket_by_predicted_length: bool = True,
        bucket_bounds: Optional[tuple] = None,
        batch_chunk: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """Run N programs in lockstep.

        Args:
            bytecodes: list of bytecode programs (each a list of packed ints).
            data_list: optional per-program data section bytes.
            argv_list: optional per-program argv list.
            stdin_list: optional per-program stdin string.
            max_steps: maximum VM steps (35 tokens each) per program. Pass
                ``None`` only when ``expected_steps_list`` is provided; in that
                mode the runner uses the declarative halt horizons instead of
                a fixed cap.
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
            bucket_by_predicted_length: when True (default), run DraftVM
                upfront on each program to predict its step count, then group
                programs into buckets of similar predicted length (2x ratio
                per bucket) and batch each bucket independently. Pure speed
                optimization: every program in a single mega-batch pads to
                ``max(P_i)``, while bucketed batches pad to each bucket's max.
                Set ``False`` to use the legacy single-batch behavior.
                Per-program ``(output, exit_code)`` results are byte-identical
                regardless of this flag — bucketing only changes the order
                programs are run through the model; results are reassembled
                into input order before return.
            bucket_bounds: optional tuple of bucket upper bounds (in
                predicted VM steps). Default is ``_DEFAULT_BUCKET_BOUNDS``
                = ``(10, 20, 40, 80, 160, 320, 640, 1280, 2560)``. A program
                with predicted P steps lands in the smallest bucket whose
                bound is >= P; anything larger goes in the "unpredicted"
                long-tail bucket. Ignored when
                ``bucket_by_predicted_length`` is False.
            batch_chunk: optional cap on programs per single batched
                forward. When set, each bucket is split into chunks of at
                most ``batch_chunk`` programs (so a very large bucket
                doesn't bloat one tensor). Default ``None`` runs each bucket
                as one batch. Ignored when
                ``bucket_by_predicted_length`` is False.

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
        expected_steps_list = expected_steps_list or [None] * B

        # Length-aware bucketing dispatch. When enabled, we predict each
        # program's step count via DraftVM, sort descending, bucket by 2x
        # length ratio, and run each bucket as its own ``_run_batch_core``
        # invocation. Results are reassembled into the original input order.
        if bucket_by_predicted_length and B > 1:
            return self._run_bucketed(
                bytecodes,
                data_list=data_list,
                argv_list=argv_list,
                stdin_list=stdin_list,
                max_steps=max_steps,
                max_context_window=max_context_window,
                spec_k=spec_k,
                expected_steps_list=expected_steps_list,
                bucket_bounds=bucket_bounds or _DEFAULT_BUCKET_BOUNDS,
                batch_chunk=batch_chunk,
            )

        return self._run_batch_core(
            bytecodes,
            data_list=data_list,
            argv_list=argv_list,
            stdin_list=stdin_list,
            max_steps=max_steps,
            max_context_window=max_context_window,
            spec_k=spec_k,
            expected_steps_list=expected_steps_list,
        )

    # ------------------------------------------------------------------
    # Core run path (no bucketing). Used directly by ``run_batch`` when
    # bucketing is disabled, and called per-bucket from ``_run_bucketed``.
    # ------------------------------------------------------------------

    def _run_batch_core(
        self,
        bytecodes: List[List[int]],
        *,
        data_list: List,
        argv_list: List,
        stdin_list: List,
        max_steps: Optional[int],
        max_context_window: int,
        spec_k: int,
        expected_steps_list: Optional[List[Optional[int]]] = None,
    ) -> List[Tuple[str, int]]:
        B = len(bytecodes)
        if B == 0:
            return []
        self._reset_kv_cache()
        self._reset_spec_stats()

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
                expected_steps=(
                    expected_steps_list[i]
                    if expected_steps_list is not None
                    else None
                ),
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

        # Build results. For programs that never emitted HALT, decode the
        # last REG_AX from the context.
        results = []
        for s in states:
            if not s.halted:
                s.exit_code = self._decode_exit_code(s.context)
            results.append(("".join(s.output), s.exit_code))
        return results

    # ------------------------------------------------------------------
    # Length-aware bucketing: predict each program's step count via
    # DraftVM, group by 2x length ratio, run each bucket as its own batch.
    # ------------------------------------------------------------------

    def _predict_steps(self, bytecode, data, stdin, max_steps: int) -> Optional[int]:
        """Run DraftVM upfront to predict step count. Returns None if the
        DraftVM raised or hit max_steps without halting (treat as
        unpredicted)."""
        try:
            vm = DraftVM(list(bytecode))
            if isinstance(data, (bytes, bytearray, list)):
                for i, b in enumerate(data):
                    vm.memory[0x10000 + i] = int(b)
            if stdin:
                vm.set_stdin(stdin)
            steps = vm.predict_steps(max_steps=max_steps)
        except Exception:
            return None
        # If DraftVM hit the cap without halting, treat as "unpredicted"
        # (long-tail bucket) so we don't underestimate it.
        if not vm.halted:
            return None
        return steps

    @staticmethod
    def _bucket_key(predicted_steps: Optional[int], bucket_bounds: tuple):
        """Map a predicted step count to a bucket key (the bucket's upper
        bound), or ``_UNPREDICTED_BUCKET_KEY`` if unbounded.

        Bucket keys are the upper bound integers from ``bucket_bounds``;
        sorting buckets by key descending (with the unpredicted bucket last
        in the iteration order) lets us run long programs first — DraftVM's
        prediction is most reliable on those, and getting them off the
        critical path early benefits any continuous-batching layer below.
        """
        if predicted_steps is None:
            return _UNPREDICTED_BUCKET_KEY
        for bound in bucket_bounds:
            if predicted_steps <= bound:
                return bound
        return _UNPREDICTED_BUCKET_KEY

    def _run_bucketed(
        self,
        bytecodes: List[List[int]],
        *,
        data_list: List,
        argv_list: List,
        stdin_list: List,
        max_steps: Optional[int],
        max_context_window: int,
        spec_k: int,
        expected_steps_list: List[Optional[int]],
        bucket_bounds: tuple,
        batch_chunk: Optional[int],
    ) -> List[Tuple[str, int]]:
        """Predict per-program lengths, bucket by 2x ratio, run each bucket
        separately, then re-assemble results in input order.

        Byte-identity invariant: per-program results must NOT depend on which
        bucket a program lands in or the order programs were run. We achieve
        this by running each bucket via ``_run_batch_core`` (which builds an
        independent ``_ElementState`` per program — no cross-program shared
        mutable state, modulo the shared model — and the model itself is
        stateless under ``torch.no_grad`` forwards).
        """
        B = len(bytecodes)

        # 1) Predict per-program step counts for length-aware bucketing. When
        #    declarative halt horizons are supplied, use them directly instead
        #    of applying a fixed prediction cap.
        if any(p is not None for p in expected_steps_list):
            predicted = list(expected_steps_list)
        else:
            if max_steps is None:
                raise ValueError(
                    "max_steps=None requires expected_steps_list for bucketing"
                )
            predicted = [
                self._predict_steps(bytecodes[i], data_list[i], stdin_list[i], max_steps)
                for i in range(B)
            ]

        # 2) Assign each program to a bucket. Bucket key is the upper bound.
        bucket_members: dict = {}  # key -> list[orig_idx]
        for i, p in enumerate(predicted):
            key = self._bucket_key(p, bucket_bounds)
            bucket_members.setdefault(key, []).append(i)

        # 3) Iterate buckets in descending-size order (longest first), within
        #    each bucket sort members descending by predicted steps (so the
        #    rare program at the top of the bucket pads itself out, not the
        #    many smaller programs below). The unpredicted bucket runs last.
        ordered_keys = sorted(
            (k for k in bucket_members if k != _UNPREDICTED_BUCKET_KEY),
            key=lambda k: -int(k),
        )
        if _UNPREDICTED_BUCKET_KEY in bucket_members:
            ordered_keys.append(_UNPREDICTED_BUCKET_KEY)

        # 4) Run each bucket; map results back to original positions.
        out: List[Optional[Tuple[str, int]]] = [None] * B
        for key in ordered_keys:
            members = bucket_members[key]
            # Sort within bucket: largest-predicted first (so spec/adaptive K
            # warmup happens on the slowest member; smaller members halt early
            # and stop padding the tensor).
            members.sort(
                key=lambda i: -(
                    predicted[i]
                    if predicted[i] is not None
                    else (max_steps if max_steps is not None else 0)
                )
            )
            # Optionally sub-chunk a very large bucket to avoid bloating one
            # tensor. ``batch_chunk=None`` means one chunk per bucket.
            chunk_size = batch_chunk if batch_chunk and batch_chunk > 0 else len(members)
            for chunk_start in range(0, len(members), chunk_size):
                chunk_idx = members[chunk_start:chunk_start + chunk_size]
                chunk_bcs = [bytecodes[i] for i in chunk_idx]
                chunk_data = [data_list[i] for i in chunk_idx]
                chunk_argv = [argv_list[i] for i in chunk_idx]
                chunk_stdin = [stdin_list[i] for i in chunk_idx]
                chunk_expected = [expected_steps_list[i] for i in chunk_idx]
                # ``max_steps`` is per-program and not affected by bucketing.
                chunk_results = self._run_batch_core(
                    chunk_bcs,
                    data_list=chunk_data,
                    argv_list=chunk_argv,
                    stdin_list=chunk_stdin,
                    max_steps=max_steps,
                    max_context_window=max_context_window,
                    spec_k=spec_k,
                    expected_steps_list=chunk_expected,
                )
                for local_j, orig_i in enumerate(chunk_idx):
                    out[orig_i] = chunk_results[local_j]

        # Defensive: every slot must be filled.
        for i, v in enumerate(out):
            if v is None:
                out[i] = ("", 0)
        return out  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Non-speculative inner loop (legacy: one token per batched forward)
    # ------------------------------------------------------------------

    def _run_unspeculative(
        self,
        states: List[_ElementState],
        *,
        max_steps: Optional[int],
        max_context_window: int,
    ) -> None:
        total_tokens = self._total_token_budget(states, max_steps)

        for tok_i in range(total_tokens):
            active_idx = [i for i, s in enumerate(states) if not s.halted]
            if not active_idx:
                break

            # Pure forward: raw, untrimmed context per element. No
            # mem-history windowing, no MEM_STORE-position re-tagging. The
            # neural model attends over the actual emitted sequence.
            #
            # Wave D (2026-06-10): explicit ``set_mem_store_positions(None)``
            # disable-call removed (per lint VANILLA_RESTORE_INVENTORY).
            # The embedding-side position injection is already inert when
            # the runner does not call its setter; the default state is no
            # MEM_STORE-position re-tagging.
            windowed = [list(states[i].context) for i in active_idx]
            if hasattr(self.model.embed, "set_mem_history_end"):
                self.model.embed.set_mem_history_end(0)
            if hasattr(self.model.embed, "set_mem_addr_src_positions"):
                self.model.embed.set_mem_addr_src_positions(None)

            # Each row only needs the argmax at ``real_len - 1`` (last real
            # token's logit predicts the next token). Pass per-row positions
            # so ``_forward_argmax_batch`` gathers just those and the
            # ``cpu()`` transfer is ``[B, 1]`` instead of ``[B, max_len]``.
            row_lens = [len(seq) for seq in windowed]
            gather_positions = [[n - 1] for n in row_lens]
            preds_cpu, _pred_start, real_lens = self._forward_argmax_batch(
                windowed,
                active_idx,
                first_logit_pos=min(row_lens) - 1,
                protected_prefix_lens=[states[i].prefix_len for i in active_idx],
                protected_mem_positions=None,
                gather_positions=gather_positions,
            )
            for b, i in enumerate(active_idx):
                next_tok = int(preds_cpu[b][0])
                self._step_one(states[i], next_tok, tok_i)

    # ------------------------------------------------------------------
    # Speculative inner loop: DraftVM proposes K*35 tokens per element,
    # model verifies them in one batched forward pass, accepted prefix is
    # replayed through ``_step_one`` (so STEP_END dispatch + EXIT halting
    # behave identically to the unspeculative path).
    # ------------------------------------------------------------------

    def _run_speculative(
        self,
        states: List[_ElementState],
        *,
        max_steps: Optional[int],
        max_context_window: int,
        spec_k: int,
        adaptive: bool = False,
    ) -> None:
        total_tokens = self._total_token_budget(states, max_steps)
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
            self._spec_stats["iterations"] += 1

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
                # context, and the forward must fit under ``model_max_seq``.
                base_k = s.adaptive_k if adaptive else spec_k
                est_ctx_len = len(s.context)
                room_tokens = max(0, model_max_seq - est_ctx_len)
                room_k = room_tokens // STEP
                effective_k = max(0, min(base_k, room_k))
                self._spec_stats["max_effective_k"] = max(
                    self._spec_stats["max_effective_k"],
                    int(effective_k),
                )
                if (
                    not s.spec_disabled
                    and s.draft_vm is not None
                    and at_step_boundary
                    and effective_k > 0
                ):
                    self._sync_draft_vm(s)
                    unsafe_stop = False
                    for step_j in range(effective_k):
                        if not self._draft_opcode_safe_for_speculation(s.draft_vm):
                            unsafe_stop = True
                            break
                        if s.draft_vm.halted:
                            break
                        ok = s.draft_vm.step()
                        if not ok:
                            break
                        d.extend(s.draft_vm.draft_tokens())
                        if s.draft_vm.halted:
                            break
                    if unsafe_stop:
                        self._spec_stats["unsafe_opcode_stops"] += 1
                if len(d) == 0:
                    self._spec_stats["no_draft_slots"] += 1
                else:
                    self._spec_stats["drafted"] += len(d)
                drafts.append(d)
                per_elem_k.append(effective_k)

            # 2) Build padded tensor: each active element gets the raw context
            #    + its draft tokens appended. No mem-history windowing.
            windowed_with_drafts = []
            real_prefix_lens = []  # len of context prefix per element (no draft)
            for k, i in enumerate(active_idx):
                s = states[i]
                ctx_win = list(s.context)
                real_prefix_lens.append(len(ctx_win))
                windowed_with_drafts.append(ctx_win + drafts[k])

            # Wave D (2026-06-10): explicit ``set_mem_store_positions(None)``
            # disable-call removed (per lint VANILLA_RESTORE_INVENTORY). The
            # embedding-side position injection is inert when its setter is
            # not called by the runner.
            if hasattr(self.model.embed, "set_mem_history_end"):
                self.model.embed.set_mem_history_end(0)
            if hasattr(self.model.embed, "set_mem_addr_src_positions"):
                self.model.embed.set_mem_addr_src_positions(None)

            # Pre-compute argmax for the verifier tensor once (GPU op) and
            # move to CPU in a single transfer. Per-row we only need positions
            # ``[prefix_len-1, prefix_len-1+len(draft))`` (the verifier scans
            # those slots) — pass them to ``_forward_argmax_batch`` so the
            # ``cpu()`` transfer is ``[B, max(P_per_row)]`` instead of
            # ``[B, max_len]``. The K=0 (no-draft) row just needs the single
            # position ``prefix_len-1``.
            gather_positions_per_row: List[List[int]] = []
            for k, i in enumerate(active_idx):
                prefix_len_k = real_prefix_lens[k]
                draft_len = len(drafts[k])
                # No-draft rows need only the single next-token position
                # ``prefix_len_k - 1``. Drafted rows need one position per
                # drafted token (the verifier reads ``row_preds[j]`` for
                # j in [0, draft_len)). The position 0 entry is always the
                # next-token prediction for ``draft[0]``.
                count = draft_len if draft_len > 0 else 1
                positions = [prefix_len_k - 1 + j for j in range(count)]
                gather_positions_per_row.append(positions)
            preds_cpu, _pred_start, _ = self._forward_argmax_batch(
                windowed_with_drafts,
                active_idx,
                first_logit_pos=min(real_prefix_lens) - 1,
                allow_kv=not any(drafts),
                protected_prefix_lens=[states[i].prefix_len for i in active_idx],
                protected_mem_positions=None,
                gather_positions=gather_positions_per_row,
            )

            # 3) For each element: verify drafts and replay accepted prefix + correction.
            for k, i in enumerate(active_idx):
                s = states[i]
                if s.halted:
                    continue
                draft = drafts[k]
                prefix_len = real_prefix_lens[k]

                if len(draft) == 0:
                    # Pure single-token decode for this element (no spec budget).
                    # Position ``prefix_len-1`` is the first entry of this row's
                    # ``gather_positions`` -> ``preds_cpu[k][0]``.
                    next_tok = int(preds_cpu[k][0])
                    self._step_one(s, next_tok, tok_i)
                    continue

                # Verify each draft slot. Offsets in `_UNSAFE_OFFSETS` are
                # trusted from DraftVM (the embedding's MEM metadata injection
                # makes them disagree with the unspec-mode model — see the
                # module-level constant docstring); other offsets are verified
                # against the model's argmax. First rejected safe offset stops
                # the scan and the model's prediction becomes the correction.
                # With gather, ``row_preds[j]`` is the model's argmax at
                # absolute position ``prefix_len - 1 + j``.
                accepted = 0
                correction: Optional[int] = None
                row_preds = preds_cpu[k]
                for j in range(len(draft)):
                    if (j % STEP) in _UNSAFE_OFFSETS:
                        accepted = j + 1
                        continue
                    pred = row_preds[j]
                    if pred == draft[j]:
                        accepted = j + 1
                    else:
                        correction = pred
                        break
                self._spec_stats["accepted"] += accepted
                if correction is None:
                    self._spec_stats["full_accepts"] += 1
                else:
                    self._spec_stats["corrections"] += 1
                    if self.spec_fail_on_correction:
                        s.exit_code = None
                        s.halted = True
                        self._spec_stats["fail_fast"] += 1
                        continue

                # Replay accepted tokens through _step_one, then the
                # correction (if any). _step_one mutates s.context so the
                # dispatch path (STEP_END dispatch + EXIT halting) stays
                # identical to the unspeculative loop.
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
                    self._spec_stats["first_token_rejects"] += 1
                    s.spec_zero_streak += 1
                    if s.spec_zero_streak >= 4:
                        if self.spec_fail_fast:
                            # Neural-authoritative gate: persistent first-token
                            # disagreement means this program has already
                            # diverged. Stop spending GPU time trying to
                            # recover via one-token fallback; surface a hard
                            # per-test failure instead.
                            s.exit_code = None
                            s.halted = True
                            self._spec_stats["fail_fast"] += 1
                        else:
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

    @staticmethod
    def _total_token_budget(
        states: List[_ElementState],
        max_steps: Optional[int],
    ) -> int:
        """Return the outer decode budget for this batch.

        With ``max_steps=None`` every active element must have a declarative
        halt horizon. This is a semantic budget, not a fixed test cap: running
        past it means the neural execution failed to halt where the declarations
        say it must.
        """
        if max_steps is not None:
            return int(max_steps) * Token.STEP_TOKENS
        expected = [s.expected_steps for s in states if s.expected_steps is not None]
        if not expected:
            raise ValueError("max_steps=None requires declarative expected_steps")
        return max(expected) * Token.STEP_TOKENS

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

    @staticmethod
    def _draft_current_opcode(vm: DraftVM) -> Optional[int]:
        """Return the opcode DraftVM will execute on its next ``step()``.

        Handles both static-code fetch (``vm.code[vm.idx]``) and
        unified-memory fetch (``_fetch_instr_from_memory``). Returns
        ``None`` if the VM is halted or no instruction is fetchable.
        """
        if vm.halted:
            return None
        if 0 <= vm.idx < len(vm.code):
            return int(vm.code[vm.idx] & 0xFF)
        fetched = vm._fetch_instr_from_memory(vm.pc)
        if fetched is None:
            return None
        op, _imm = fetched
        return int(op) & 0xFF

    @classmethod
    def _draft_opcode_safe_for_speculation(cls, vm: DraftVM) -> bool:
        """Return whether DraftVM's current opcode may be appended as draft.

        Unsafe opcodes still execute neurally through the normal one-token path.
        This guard only prevents long speculative continuations from being used
        as verifier context across call-frame or external-I/O boundaries.
        """
        if vm.halted:
            return True
        op = cls._draft_current_opcode(vm)
        if op is None:
            return False
        return op not in _SPEC_UNSAFE_OPS

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
        del tok_i
        s.context.append(next_token)
        s.token_pos += 1

        if next_token == Token.HALT:
            s.exit_code = self._decode_exit_code(s.context)
            s.halted = True
            return

        if next_token == Token.STEP_END or next_token == Token.TOOL_CALL:
            self._dispatch_pure_neural(s)
            if s.halted:
                return

        if (
            s.expected_steps is not None
            and s.token_pos >= s.expected_steps * Token.STEP_TOKENS
            and not s.halted
        ):
            # Cap reached without an explicit HALT. Take the most recent
            # well-formed REG_AX as the exit code (pure neural emit, no
            # forward-scan recovery).
            s.exit_code = self._decode_exit_code(s.context)
            s.halted = True
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

        # Update tracking registers from neural outputs. These are
        # observation-only — no Python override re-writes the context. The
        # neural model is the sole authority for register/AX/PC values.
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

        # Wave D removal (2026-06-10): ``exec_op == Opcode.PUTCHAR`` /
        # ``exec_op == Opcode.EXIT`` per-opcode branches deleted (lint
        # VANILLA_RESTORE_INVENTORY rule). PUTCHAR output capture must
        # flow through the neural THINK-protocol bake (see
        # ``docs/IO_NEURAL_BAKE_QUEUE_2026_06_09.md``); EXIT termination
        # is detected below by reading the next bytecode opcode against
        # the model-emitted PC, which uses ``next_op`` (not ``exec_op``)
        # and therefore does not trip the per-op-branch rule.

        # Neural-authoritative early exit: after a completed step, the model's
        # emitted PC is the next instruction address and the emitted AX is the
        # value EXIT would return. If that next instruction is EXIT, stop here
        # instead of asking the model to generate an extra EXIT step whose
        # register/default bytes are not semantically needed for the result.
        if s.last_pc is not None:
            next_idx = s.last_pc // INSTR_WIDTH
            if 0 <= next_idx < len(s.bytecode):
                next_op = s.bytecode[next_idx] & 0xFF
                if next_op == Opcode.EXIT:
                    s.exit_code = int(s.last_ax) & 0xFFFFFFFF
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
    def _decode_exit_code(context) -> int:
        for i in range(len(context) - 1, -1, -1):
            if context[i] == Token.REG_AX and i + 4 < len(context):
                val = 0
                for j in range(4):
                    val |= (context[i + 1 + j] & 0xFF) << (j * 8)
                return val
        return 0
