"""
NeuralVMEmbedding: Pure embedding layer with integrated augmentations.

This module provides an embedding layer that encapsulates position-dependent
metadata augmentations (ADDR_KEY and MEM_STORE) inside the embedding itself,
achieving autoregressive purity in the forward pass.
"""

import torch
import torch.nn as nn


class NeuralVMEmbedding(nn.Module):
    """Embedding layer with integrated ADDR_KEY and MEM_STORE augmentations.

    Wraps nn.Embedding and adds position-dependent metadata:
    - ADDR_KEY: Sequential code byte addresses (dims 206-253, 48 dims total)
    - MEM_STORE: Historical memory marker flags (dim 455, 1 dim)

    These augmentations are deterministic transformations based on token IDs
    and positions, similar to positional encodings (RoPE/ALiBi).

    This encapsulation achieves autoregressive purity: the forward() method in
    AutoregressiveVM can be purely: embed → blocks → head with no modifications.
    """

    def __init__(self, vocab_size, d_model, dim_positions=None, max_seq_len=1024):
        """Initialize neural VM embedding.

        Args:
            vocab_size: Number of tokens in vocabulary (typically 272)
            d_model: Embedding dimension (typically 512)
            dim_positions: Optional dict mapping dim name -> start position.
                When provided, injection methods use these compiler-allocated
                positions. When None, falls back to `_SetDim` constants
                (backward-compat).
            max_seq_len: Maximum sequence length this embedding will see.
                Used to pre-allocate the ADDR_KEY positional-encoding buffer
                at construction time (rather than lazily growing it) so the
                traced ONNX graph never embeds a frozen capacity.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # Phase 0 M4: dim positions from compiler; None means fall back to _SetDim.
        self._dim_positions = dim_positions

        # Standard PyTorch embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Track memory history end for MEM_STORE injection
        # (set by KV cache eviction logic)
        self._mem_history_end = 0

        # === Prefix embedding cache (perf optimization, 2026-05-11) ===
        # Many of the augmentations below loop over the entire context, but
        # the CODE/DATA prefix tokens (positions 0..CODE_END/DATA_END) never
        # change within a single ``run()``. We cache the augmentation outputs
        # for that prefix region and apply them as a single in-place add.
        #
        # Cache invariants:
        #   - ``_prefix_cache_token_ids``: 1D LongTensor of prefix tokens (CPU).
        #   - ``_prefix_cache_delta``: [1, prefix_len, d_model] tensor of the
        #     augmentation modifications applied to the prefix region. Added
        #     to the un-augmented embedding output on cache hit.
        #   - ``_prefix_cache_len``: int, number of prefix positions cached.
        #
        # The cache is invalidated whenever the leading tokens of the input
        # differ from the cached prefix (e.g. new ``run()`` with different
        # bytecode) via ``reset_prefix_cache()`` or automatic mismatch check.
        self._prefix_cache_token_ids = None
        self._prefix_cache_delta = None
        self._prefix_cache_len = 0

        # === Compiler-baked ADDR_KEY positional encoding (2026-05-11) ===
        # ADDR_KEY is morally a positional encoding: each code byte position
        # gets a deterministic one-hot encoding of its byte address (PC), with
        # the address computed from the position alone (CODE_START is always
        # at sequence position 0 — see run_vm._build_context).
        #
        # We precompute the per-position one-hot ADDR_KEY values as a buffer
        # of shape ``[max_seq_len, 48]`` at construction time. The 48 dims
        # correspond to 3 nibbles × 16 one-hot positions. At forward time we
        # add a bounded slice (positions [1, code_end)) into the ADDR_KEY band
        # of the residual stream, replacing the per-position Python loop.
        #
        # The exact write pattern matches the legacy ``_add_code_addr_keys``:
        #   - position i (relative to CODE_START at index 0) corresponds to
        #     code byte offset seq_pos = i - 1
        #   - byte_offset = seq_pos % 8, instr_idx = seq_pos // 8
        #   - skip padding bytes where byte_offset >= 5
        #   - addr = instr_idx * INSTR_WIDTH + PC_OFFSET + byte_offset
        #   - write one-hots: ADDR_KEY[addr & 0xF], ADDR_KEY[16 + (addr>>4)&0xF],
        #     ADDR_KEY[32 + (addr>>8)&0xF]
        #
        # CODE_END detection still happens dynamically (the buffer is bounded
        # at apply time so it does not pollute positions past CODE_END). The
        # buffer is registered eagerly at ``max_seq_len`` capacity so the
        # traced ONNX graph never embeds a dynamic capacity check; it lives on
        # CPU initially and moves with ``.to(device)`` via the registered-buffer
        # machinery.
        self._addr_key_pos_encoding_size = int(max_seq_len)
        self.register_buffer(
            "_addr_key_pos_encoding_buf",
            self._build_addr_key_pos_encoding(int(max_seq_len), torch.device("cpu"), torch.float32),
            persistent=False,
        )

    def _dim(self, name: str) -> int:
        """Resolve a dim name to its start position.

        Phase 0 M4: when self._dim_positions is set (compiler mode), use it.
        Otherwise fall back to _SetDim (hand-set mode).
        """
        if self._dim_positions is not None and name in self._dim_positions:
            return self._dim_positions[name]
        from .vm_step import _SetDim
        return getattr(_SetDim, name)

    def reset_prefix_cache(self):
        """Invalidate the prefix-embedding cache.

        Called at the start of each ``run()`` to ensure the cache does not
        survive across distinct bytecode programs. Also called by callers
        that mutate the embedding inputs in ways the cache doesn't track.
        """
        self._prefix_cache_token_ids = None
        self._prefix_cache_delta = None
        self._prefix_cache_len = 0

    def _compute_prefix_len(self, token_ids):
        """Find the length of the immutable CODE/DATA prefix in token_ids.

        Returns the index just after CODE_END (and DATA_END if present),
        i.e. the boundary up to which augmentations are deterministic and
        cacheable. Returns 0 if no CODE_END is found in the first batch row.
        """
        from .vm_step import Token

        # Use batch row 0 as the authority; the prefix is identical across
        # batch elements in practice (build_context produces a single sequence
        # that is broadcast). If token_ids has fewer rows or shorter seq we
        # bail out and don't cache.
        if token_ids.dim() != 2 or token_ids.shape[0] < 1:
            return 0
        row = token_ids[0]
        S = row.shape[0]
        # Find CODE_END (and, optionally, DATA_END to extend the cache).
        # Use tensor.tolist() over a bounded prefix (we don't need to scan
        # past the end of the typical prefix; bytecode prefixes are short).
        # In practice the entire context is short for the prefix scan.
        row_list = row.tolist()
        code_end_idx = -1
        for i, tok in enumerate(row_list):
            if tok == Token.CODE_END:
                code_end_idx = i
                break
        if code_end_idx < 0:
            return 0
        # Try to extend to DATA_END if it appears shortly after CODE_END.
        # The standard layout is CODE_END, DATA_START, ..., DATA_END.
        end_idx = code_end_idx
        for j in range(code_end_idx + 1, min(S, code_end_idx + 1 + 4096)):
            if row_list[j] == Token.DATA_END:
                end_idx = j
                break
            # If we hit something other than DATA_START / data bytes (which
            # are always < 256 ints), bail out at DATA_END heuristic check.
        # Cache length is end_idx + 1 (inclusive of the END marker).
        return end_idx + 1

    def _prefix_cache_matches(self, token_ids):
        """Return True if cached prefix matches the leading tokens of ``token_ids``."""
        if self._prefix_cache_token_ids is None:
            return False
        cache_len = self._prefix_cache_len
        if cache_len == 0 or token_ids.shape[1] < cache_len:
            return False
        # Compare on the same device (cache lives on CPU; bring slice to CPU).
        cached = self._prefix_cache_token_ids
        # token_ids[0, :cache_len] vs cached; both are 1D long tensors.
        prefix_slice = token_ids[0, :cache_len].detach().to(cached.device)
        return torch.equal(prefix_slice, cached)

    def forward(self, token_ids):
        """Apply embedding + augmentations.

        Args:
            token_ids: [batch, seq] tensor of token IDs

        Returns:
            x: [batch, seq, d_model] embeddings with augmentations applied
        """
        # Standard embedding lookup
        x = self.embed(token_ids)

        # --- ONNX export / torch.compile fast path -----------------------
        # The prefix cache is purely a runtime perf optimization (it caches
        # the augmentation delta for the immutable CODE/DATA prefix region
        # across forward calls within a single ``run()``). It involves Python
        # state mutation and a ``.tolist()`` scan that the ONNX tracer and
        # ``torch.compile``'s Dynamo tracer can't capture. When exporting
        # or compiling, bypass the cache entirely and run the un-cached
        # augmentations every call.
        #
        # ONNX export additionally uses the pure-tensor ADDR_KEY computation
        # (``_compute_addr_keys``) instead of the pre-allocated buffer so the
        # exported model accepts *any* sequence length at inference. Under
        # ``torch.compile`` the buffer path is fine (single shape, traceable).
        if torch.onnx.is_in_onnx_export():
            self._compute_addr_keys(token_ids, x)
            self._inject_mem_store(token_ids, x, start_pos=0)
            return x
        if torch.compiler.is_compiling():
            # torch.compile path: avoid the prefix cache (Python state) and
            # use the buffer-based addr_keys (single static shape — fine for
            # Dynamo trace, unlike ONNX which needs dynamic seq_len).
            self._add_code_addr_keys(token_ids, x)
            self._inject_mem_store(token_ids, x, start_pos=0)
            return x

        # --- Prefix cache fast path ---------------------------------------
        # If we have a cached prefix delta that matches the leading tokens,
        # add it in one shot and skip the per-position Python loops on that
        # region. The augmentations that scan the full sequence accept a
        # ``start_pos`` parameter so they only process positions beyond the
        # cached prefix.
        cache_hit = self._prefix_cache_matches(token_ids)
        if cache_hit:
            cache_len = self._prefix_cache_len
            delta = self._prefix_cache_delta
            # Broadcast delta (shape [1, cache_len, d_model]) over batch.
            x[:, :cache_len, :].add_(delta.to(x.device, dtype=x.dtype))
            start_pos = cache_len
        else:
            start_pos = 0

        # Apply augmentations. ``start_pos`` lets us skip the cached region
        # for methods that scan token-by-token. Methods whose work is fully
        # local to the prefix (only ``_add_code_addr_keys`` today) are
        # entirely skipped on cache hit.
        if not cache_hit:
            self._add_code_addr_keys(token_ids, x)

        # THINKING_START/END markers (MARK_THINKING_START/_END) are now baked
        # directly into the embedding table (see
        # ``unified_compiler.ops.shared.setup_token_embeddings``), so no
        # runtime injection is needed.

        # Initial PC value for step 0 (no previous step to carry from) is now
        # baked into the REG_PC token-embedding row at compile time. See
        # `make_initial_pc_bake_op` in unified_compiler/ops/model_ops.py and
        # the matching cancel units in `_set_layer3_ffn` (vm_step.py). The
        # runtime `_inject_initial_pc` injection is no longer needed.

        # ADDR_KEY nibble decode at MEM val byte positions is now baked
        # into the L14 FFN (``layer14_addr_key_neural_decode`` in
        # unified_compiler/ops/l14_ops.py); MEM_VAL_B and MEM_STORE
        # flags are produced by L2 FFN + L6/L7 attention.  See
        # docs/V2_ADDR_KEY_NEURAL_DECODE_PLAN.md.  The retained-history
        # MEM_STORE injection below remains a no-op unless
        # ``_mem_history_end > 0`` (set by KV-cache eviction).
        self._inject_mem_store(token_ids, x, start_pos=start_pos)

        # --- Populate prefix cache on first call within a run -------------
        if not cache_hit:
            self._populate_prefix_cache(token_ids, x)

        return x

    def _populate_prefix_cache(self, token_ids, x):
        """Cache the augmentation delta for the immutable prefix region.

        The delta is computed by subtracting the un-augmented embedding from
        the fully-augmented embedding for positions ``[0, prefix_len)``. On
        subsequent forward passes with matching leading tokens, the delta is
        added directly, bypassing the per-position Python loops.
        """
        prefix_len = self._compute_prefix_len(token_ids)
        if prefix_len <= 0:
            # Nothing to cache (e.g. context shorter than CODE_END).
            return
        # Re-run the unaugmented embedding lookup on the prefix tokens only
        # so we can isolate the additive contribution of the augmentations.
        with torch.no_grad():
            row = token_ids[0:1, :prefix_len]
            unaug = self.embed(row)
            delta = (x[0:1, :prefix_len, :] - unaug).detach()
        # Store the cache. Keep the token IDs on CPU so device transfers
        # (e.g. for runs that share an embedding across devices) stay cheap.
        self._prefix_cache_token_ids = token_ids[0, :prefix_len].detach().cpu().clone()
        self._prefix_cache_delta = delta
        self._prefix_cache_len = prefix_len

    def _build_addr_key_pos_encoding(self, capacity, device, dtype):
        """Compute the ADDR_KEY positional encoding table.

        The encoding is a deterministic, position-only one-hot table that
        replaces the per-position Python loop in the legacy
        ``_add_code_addr_keys``. It has shape ``[capacity, 48]``; entry
        ``[pos, :]`` holds the three one-hot nibble flags for the byte address
        at that position (or zeros if the position is a non-code-byte slot —
        padding bytes, the CODE_START token itself, or anything past the code
        region).
        """
        from .constants import INSTR_WIDTH, PC_OFFSET

        BYTES_PER_INSTR = 8  # Matches legacy: opcode + 4 imm + 3 pad
        DATA_BYTES = 5       # Bytes that carry ADDR_KEY (opcode + 4 imm)

        enc = torch.zeros(capacity, 48, device=device, dtype=dtype)
        # CODE_START is at position 0. The first code byte is at position 1.
        # For each position i in [1, capacity), compute seq_pos = i - 1.
        for i in range(1, capacity):
            seq_pos = i - 1
            byte_offset = seq_pos % BYTES_PER_INSTR
            if byte_offset >= DATA_BYTES:
                # Padding byte: legacy injection skips these.
                continue
            instr_idx = seq_pos // BYTES_PER_INSTR
            addr = instr_idx * INSTR_WIDTH + PC_OFFSET + byte_offset
            lo = addr & 0xF
            hi = (addr >> 4) & 0xF
            top = (addr >> 8) & 0xF
            enc[i, lo] = 1.0
            enc[i, 16 + hi] = 1.0
            enc[i, 32 + top] = 1.0
        return enc

    def _ensure_addr_key_pos_encoding(self, S, device, dtype):
        """Return the ADDR_KEY positional encoding on the requested device/dtype.

        The buffer is pre-allocated at construction time with capacity
        ``max_seq_len`` so this method never grows it at forward time. If S
        exceeds ``max_seq_len`` (rare backward-compat path) we rebuild eagerly,
        but this branch never fires on the ONNX trace path.

        Args:
            S: Required minimum number of positions.
            device: Target device for the buffer.
            dtype: Target dtype for the buffer.

        Returns:
            A tensor of shape ``[>=S, 48]`` on ``device`` with ``dtype``.
            Caller should slice ``[:S]`` before use.
        """
        buf = self._addr_key_pos_encoding_buf

        if S > self._addr_key_pos_encoding_size:
            # Backward-compat path: a caller passed a sequence longer than the
            # max_seq_len declared at construction. Rebuild at the requested
            # capacity. Not exercised in ONNX export (max_seq_len is sized to
            # cover all expected inputs).
            new_capacity = max(S, self._addr_key_pos_encoding_size * 2)
            new_buf = self._build_addr_key_pos_encoding(new_capacity, device, dtype)
            if "_addr_key_pos_encoding_buf" in self._buffers:
                del self._buffers["_addr_key_pos_encoding_buf"]
            self.register_buffer("_addr_key_pos_encoding_buf", new_buf, persistent=False)
            self._addr_key_pos_encoding_size = new_capacity
            return new_buf

        if buf.device != device or buf.dtype != dtype:
            return buf.to(device=device, dtype=dtype)
        return buf

    def _compute_addr_keys(self, token_ids, x):
        """Pure-tensor ADDR_KEY injection for ONNX export at arbitrary seq_len.

        This is the dynamic-shape counterpart to :meth:`_add_code_addr_keys`.
        The buffer-based path pre-allocates ``[max_seq_len, 48]`` at
        construction time, which freezes the traced ONNX graph at that
        capacity. To make the exported model usable at any sequence length,
        we recompute the ADDR_KEY one-hot table from scratch with
        ``torch.arange(S)`` so the graph has no constant capacity edge.

        The encoding produced here is bit-identical to the buffer path:
        - position 0 is CODE_START; first code byte at position 1
        - seq_pos = i - 1; byte_offset = seq_pos % 8; instr_idx = seq_pos // 8
        - byte_offset >= 5 are padding bytes (skipped)
        - addr = instr_idx * INSTR_WIDTH + PC_OFFSET + byte_offset
        - write one-hot at ADDR_KEY[addr & 0xF], 16+(addr>>4)&0xF, 32+(addr>>8)&0xF
        - bounded by code_end_idx (position of first CODE_END token)

        Args:
            token_ids: [batch, seq] LongTensor of token IDs.
            x: [batch, seq, d_model] residual stream; modified in-place.
        """
        import torch.nn.functional as F
        from .constants import INSTR_WIDTH, PC_OFFSET
        from .vm_step import Token

        BYTES_PER_INSTR = 8  # opcode + 4 imm + 3 pad
        DATA_BYTES = 5       # bytes that carry ADDR_KEY

        B, S = token_ids.shape
        addr_key = self._dim("ADDR_KEY")
        device = x.device
        dtype = x.dtype

        # --- position-only address computation -----------------------------
        pos = torch.arange(S, device=device, dtype=torch.long)  # [S]
        seq_pos = pos - 1
        byte_offset = seq_pos % BYTES_PER_INSTR
        instr_idx = seq_pos // BYTES_PER_INSTR
        addr = instr_idx * INSTR_WIDTH + PC_OFFSET + byte_offset  # [S]

        # Per-position validity: seq_pos >= 0 AND non-padding byte.
        pos_valid = (seq_pos >= 0) & (byte_offset < DATA_BYTES)  # [S] bool

        # --- code_end_idx as a tensor (no .item()) -------------------------
        # A position is "before the first CODE_END" iff cumsum of CODE_END hits
        # up to and including that position is zero. We compute this per batch
        # row. ``cumsum`` works in ONNX and avoids any Python int conversion.
        is_code_end = (token_ids == Token.CODE_END).to(torch.long)  # [B, S]
        before_code_end = (is_code_end.cumsum(dim=1) == 0)          # [B, S]

        # Combined per-(b,i) mask: position is a valid code byte slot.
        mask = before_code_end & pos_valid.unsqueeze(0)             # [B, S]
        mask_f = mask.to(dtype).unsqueeze(-1)                       # [B, S, 1]

        # --- one-hot scatter for the three nibbles -------------------------
        # Use arithmetic (mod / floor-div) instead of bitwise AND/SHIFT: ONNX
        # does not support bitwise ops on non-bool ints in opset 17. The
        # results are identical to ``addr & 0xF`` / ``(addr >> k) & 0xF`` for
        # non-negative addresses, and we mask out the only seq_pos<0 case
        # (i==0, CODE_START itself) via ``pos_valid``.
        # ``.clamp(0, 15)`` is a defensive guard so F.one_hot never sees an
        # out-of-range index on traced runs that may temporarily produce a
        # negative value before the mask gates it to zero.
        lo = (addr % 16).clamp(min=0, max=15)
        hi = ((addr // 16) % 16).clamp(min=0, max=15)
        top = ((addr // 256) % 16).clamp(min=0, max=15)
        one_hot_lo = F.one_hot(lo, num_classes=16).to(dtype)         # [S, 16]
        one_hot_hi = F.one_hot(hi, num_classes=16).to(dtype)         # [S, 16]
        one_hot_top = F.one_hot(top, num_classes=16).to(dtype)       # [S, 16]
        # Concatenate to [S, 48] in the order (lo, hi, top) to match the
        # legacy buffer layout (lo at [0..16), hi at [16..32), top at [32..48)).
        enc = torch.cat([one_hot_lo, one_hot_hi, one_hot_top], dim=-1)  # [S, 48]

        # Broadcast over batch and apply mask.
        addr_slice = enc.unsqueeze(0)                                # [1, S, 48]
        gated = addr_slice * mask_f                                  # [B, S, 48]

        # In-place add into the ADDR_KEY band of the residual stream.
        x[:, :, addr_key:addr_key + 48].add_(gated)

    def _add_code_addr_keys(self, token_ids, x):
        """Add ADDR_KEY to code byte embeddings (compiler-baked positional encoding).

        ADDR_KEY is morally a positional encoding: each code byte at sequence
        position i has a deterministic byte address (PC) that depends ONLY on
        i (since CODE_START is always at position 0 — see _build_context).
        L5/L7 attention heads dot-product their query addresses against
        ``ADDR_KEY`` keys to do content-addressed bytecode fetch.

        Previously this was a per-position Python loop run on every forward
        call. We now precompute a ``[max_seq_len, 48]`` one-hot table at
        construction time (lazily, sized by actual S) and add a bounded slice
        into the residual stream's ADDR_KEY band. CODE_END is detected
        dynamically and bounds the add so we don't pollute non-code positions.

        Uses PC-aligned addressing to match L5 fetch queries:
        - L5 head 1 fetches opcode using EMBED_LO/HI (which holds PC value)
        - L5 head 0 fetches immediate using TEMP (which holds PC+1, PC+2, etc.)
        - ADDR_KEY must equal PC for opcode, PC+1 for imm[0], etc.

        With PC_OFFSET=2, INSTR_WIDTH=8:
        - Instruction i opcode: PC = i * 8 + 2, ADDR_KEY = PC
        - Instruction i imm[j]: ADDR_KEY = PC + j + 1
        """
        from .vm_step import Token

        B, S = token_ids.shape
        addr_key = self._dim("ADDR_KEY")

        # Find CODE_END per batch row (a small loop, runs once per forward,
        # not per code byte). The legacy implementation also bounded at
        # CODE_END and used CODE_START to anchor; here we use the structural
        # invariant that CODE_START is at position 0 and just find CODE_END.
        # Different batch rows can in principle have different CODE_END
        # positions, so we handle the per-row case.
        # In practice all rows of a single run() share the same context.

        # Identify positions that are code bytes per row (tok < 256, between
        # positions [1, code_end_row)). We compute this via a tensor mask.
        # token_ids[b, 0] is expected to be CODE_START; CODE_END terminates.
        code_end_tok = Token.CODE_END
        # For each batch row, find the first index where token == CODE_END.
        # We use argmax over a boolean mask of the equality (returns first
        # True position for tied values).
        is_code_end = (token_ids == code_end_tok)
        # If a row has no CODE_END, fall back to scanning whole sequence.
        has_code_end = is_code_end.any(dim=1)
        # First-True index per row (or S if absent).
        # argmax on bool returns the first True if any is True; combine with
        # has_code_end to default to S.
        code_end_idx = torch.where(
            has_code_end,
            is_code_end.float().argmax(dim=1),
            torch.full_like(has_code_end, S, dtype=torch.long),
        )

        # If the table doesn't cover S yet, build/extend it.
        enc = self._ensure_addr_key_pos_encoding(S, x.device, x.dtype)

        # The encoding rows for positions [1, code_end_idx[b]) are added to
        # x[b, :, ADDR_KEY:ADDR_KEY+48]. We build a per-row mask of valid
        # positions (1 <= pos < code_end_idx[b]) and use it to gate the add.
        pos_range = torch.arange(S, device=x.device).unsqueeze(0)  # [1, S]
        # mask[b, i] is True iff position i is in [1, code_end_idx[b])
        # (and code_end_idx[b] > 0 — i.e. CODE_END was found at index >=1).
        ce = code_end_idx.unsqueeze(1)  # [B, 1]
        mask = (pos_range >= 1) & (pos_range < ce)  # [B, S]

        # Build the slice to add. Broadcast enc[:S] over batch and mask out
        # positions outside [1, code_end_idx[b]).
        addr_slice = enc[:S].unsqueeze(0)  # [1, S, 48]
        # Apply mask: zero out positions we shouldn't touch.
        # Use float mask so we keep dtype.
        gated = addr_slice * mask.unsqueeze(-1).to(addr_slice.dtype)  # [B, S, 48]

        # In-place add into the ADDR_KEY band of the residual stream.
        x[:, :, addr_key:addr_key + 48].add_(gated)

    def _inject_mem_store(self, token_ids, x, start_pos=0):
        """Inject MEM_STORE=1.0 on historical MEM markers for L15 K-side.

        Historical MEM sections (from prior store ops retained in context)
        lack the MEM_STORE flag that L6 head 6 sets for the current step.
        Without this flag, L15 memory lookup won't match these positions.

        Only injects on MEM markers in the retained history region
        (0 .. _mem_history_end), not the current step's MEM section.

        Args:
            start_pos: Skip positions ``< start_pos`` (prefix-cache hint).
                The CODE/DATA prefix contains no MEM markers, so this is safe.
        """
        from .vm_step import Token

        end = self._mem_history_end
        if end == 0:
            return

        mem_store = self._dim("MEM_STORE")
        B, S = token_ids.shape
        for b in range(B):
            for i in range(start_pos, min(end, S)):
                if token_ids[b, i].item() == Token.MEM:
                    x[b, i, mem_store] = 1.0

    def set_mem_history_end(self, end):
        """Set the memory history boundary for MEM_STORE injection.

        Called by KV cache eviction logic when retaining historical tokens.

        Args:
            end: Position marking end of historical memory region
        """
        self._mem_history_end = end
