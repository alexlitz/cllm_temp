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

    def __init__(self, vocab_size, d_model, dim_positions=None):
        """Initialize neural VM embedding.

        Args:
            vocab_size: Number of tokens in vocabulary (typically 272)
            d_model: Embedding dimension (typically 512)
            dim_positions: Optional dict mapping dim name -> start position.
                When provided, injection methods use these compiler-allocated
                positions. When None, falls back to `_SetDim` constants
                (backward-compat).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
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
        # buffer is lazily realized — it is filled on first forward() once we
        # know the actual sequence length to size against.
        self._addr_key_pos_encoding = None  # lazy: [max_seq_len, 48] buffer
        self._addr_key_pos_encoding_size = 0  # how many positions are filled

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

    def forward(self, token_ids, active_opcode=None):
        """Apply embedding + augmentations.

        Args:
            token_ids: [batch, seq] tensor of token IDs
            active_opcode: Current active opcode (0-255) or None

        Returns:
            x: [batch, seq, d_model] embeddings with augmentations applied
        """
        # Standard embedding lookup
        x = self.embed(token_ids)

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
        self._inject_mem_store(token_ids, x, start_pos=start_pos)

        # Inject THINKING_START/END markers for lookback detection.
        # Prefix never contains THINKING tokens, so skip those positions.
        self._inject_thinking_markers(token_ids, x, start_pos=start_pos)

        # Initial PC value for step 0 (no previous step to carry from) is now
        # baked into the REG_PC token-embedding row at compile time. See
        # `make_initial_pc_bake_op` in unified_compiler/ops/model_ops.py and
        # the matching cancel units in `_set_layer3_ffn` (vm_step.py). The
        # runtime `_inject_initial_pc` injection is no longer needed.

        # Unified memory: write ADDR_KEY/MEM_STORE/MEM_VAL_Bn metadata on
        # every MEM section. Prefix has no MEM markers; skip prefix positions
        # on cache hit.
        self._inject_mem_metadata(token_ids, x, start_pos=start_pos)

        # --- Populate prefix cache on first call within a run -------------
        # Populate the cache BEFORE applying ``_inject_active_opcode`` so that
        # opcode-dependent writes are NOT baked into the cached delta. The
        # active opcode may change between forward calls, and its writes are
        # absolute ``x[:, :, dim] = 5.0`` rather than additive — caching them
        # in the delta would leave stale opcode dims set on opcode changes.
        if not cache_hit:
            self._populate_prefix_cache(token_ids, x)

        # Inject active opcode flags for conversational I/O detection.
        # This writes globally across all positions; not cached because the
        # opcode value can change between forward calls within a single run.
        if active_opcode is not None:
            self._inject_active_opcode(token_ids, x, active_opcode)

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

    def _ensure_addr_key_pos_encoding(self, S, device, dtype):
        """Build (or extend) the precomputed ADDR_KEY positional encoding.

        The encoding is a deterministic, position-only one-hot table that
        replaces the per-position Python loop in the legacy
        ``_add_code_addr_keys``. It has shape ``[N, 48]`` where N is the
        number of sequence positions we have filled; entry [pos, :] holds the
        three one-hot nibble flags for the byte address at that position
        (or zeros if the position is a non-code-byte slot — padding bytes,
        the CODE_START token itself, or anything past the code region).

        The buffer is bounded by S and grown lazily: on first call we fill
        ``max(S, 256)`` positions; subsequent calls grow it if S exceeds the
        cached capacity. The buffer is registered as a module buffer so it
        moves with ``.to(device)``.

        Args:
            S: Required minimum number of positions.
            device: Target device for the buffer.
            dtype: Target dtype for the buffer.

        Returns:
            A tensor of shape ``[capacity, 48]`` on ``device`` with ``dtype``.
            Caller should slice ``[:S]`` before use.
        """
        from .constants import INSTR_WIDTH, PC_OFFSET

        # If we already have a buffer that covers S positions and matches
        # device/dtype, return it directly.
        if (self._addr_key_pos_encoding is not None
                and self._addr_key_pos_encoding_size >= S
                and self._addr_key_pos_encoding.device == device
                and self._addr_key_pos_encoding.dtype == dtype):
            return self._addr_key_pos_encoding

        # Otherwise (re)build at a capacity >= S. Round up so we don't rebuild
        # on every small S increment.
        capacity = max(S, max(256, self._addr_key_pos_encoding_size * 2))

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

        # Register (or replace) as a non-persistent buffer so .to() moves it.
        # We avoid ``register_buffer`` for persistent state because the table
        # is fully deterministic from constants and should not be saved.
        if hasattr(self, "_buffers") and "_addr_key_pos_encoding_buf" in self._buffers:
            # Replace existing buffer in place
            del self._buffers["_addr_key_pos_encoding_buf"]
        self.register_buffer("_addr_key_pos_encoding_buf", enc, persistent=False)
        self._addr_key_pos_encoding = self._addr_key_pos_encoding_buf
        self._addr_key_pos_encoding_size = capacity
        return self._addr_key_pos_encoding

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

    def _inject_active_opcode(self, token_ids, x, active_opcode):
        """Inject active opcode flags into step output positions only.

        For conversational I/O: exposes the MoE routing signal globally so
        L5 FFN can detect PRTF/READ opcodes reliably.

        For LEV: sets OP_LEV at all positions so L15 heads 8-11 can do
        return_addr lookup at PC marker (which comes before AX marker).

        Args:
            x: [batch, seq, d_model] embedding tensor (modified in-place)
            active_opcode: Current opcode value (0-255)
        """
        _OPCODE_INJECTION_MAP = {
            33: "ACTIVE_OPCODE_PRTF",
            31: "ACTIVE_OPCODE_READ",
            8:  "OP_LEV",
            4:  "OP_BZ",
            5:  "OP_BNZ",
        }
        name = _OPCODE_INJECTION_MAP.get(active_opcode)
        if name is not None:
            x[:, :, self._dim(name)] = 5.0

    def _inject_thinking_markers(self, token_ids, x, start_pos=0):
        """Inject THINKING_START and THINKING_END markers for lookback detection.

        Sets dedicated marker dimensions (not overlapping with OUTPUT_BYTE)
        so L2 lookback head can reliably detect these special tokens.

        Args:
            token_ids: [batch, seq] tensor of token IDs
            x: [batch, seq, d_model] embedding tensor (modified in-place)
            start_pos: Skip positions ``< start_pos`` (prefix-cache hint).
        """
        from .vm_step import Token

        thinking_start = self._dim("MARK_THINKING_START")
        thinking_end = self._dim("MARK_THINKING_END")
        B, S = token_ids.shape
        for b in range(B):
            for i in range(start_pos, S):
                tok = token_ids[b, i].item()
                if tok == Token.THINKING_START:
                    x[b, i, thinking_start] = 1.0
                elif tok == Token.THINKING_END:
                    x[b, i, thinking_end] = 1.0

    def _inject_mem_metadata(self, token_ids, x, start_pos=0):
        """Write ADDR_KEY/MEM_STORE/MEM_VAL_Bn metadata on every MEM section.

        Walks the token stream and, for each 9-token MEM section
        ``[MEM, addr_b0..3, val_b0..3]``, writes the residual dims required
        by L5 fetch / L8 multibyte / L15 memory lookup:

        - ``MEM_STORE = 2.0`` on the MEM marker and on each val-byte position
          (matches L6 head 6's scale and prevents L15 K-side dim-37 suppression
          from favoring the marker over the bytes).
        - ``MEM_VAL_B{0..3} = 1.0`` on val-byte positions for L15 byte selection.
        - ``ADDR_KEY + {lo, 16+hi, 32+top}`` on val-byte positions, computed
          from ``addr + byte_off``, so L5/L8/L15 attention can match by address.

        Args:
            start_pos: Skip positions ``< start_pos``. Safe when the prefix
                contains no MEM markers (true for the CODE/DATA system prompt).
        """
        from .vm_step import Token

        mem_store = self._dim("MEM_STORE")
        addr_key = self._dim("ADDR_KEY")
        mem_val_dims = [
            self._dim("MEM_VAL_B0"),
            self._dim("MEM_VAL_B1"),
            self._dim("MEM_VAL_B2"),
            self._dim("MEM_VAL_B3"),
        ]

        B, S = token_ids.shape

        for b in range(B):
            i = start_pos
            while i < S:
                tok = token_ids[b, i].item()
                if tok == Token.MEM and i + 8 < S:
                    # Extract address from MEM section (bytes 1-4, little-endian)
                    addr_bytes = [token_ids[b, i + 1 + j].item() for j in range(4)]
                    addr = (addr_bytes[0] | (addr_bytes[1] << 8) |
                            (addr_bytes[2] << 16) | (addr_bytes[3] << 24))

                    # Set MEM_STORE on MEM marker for L15 K-side matching.
                    # L15 attention uses MEM_STORE in K-side to identify store
                    # entries. Without this, attention gets -312.5 penalty.
                    x[b, i, mem_store] = 2.0  # Same scale as L6 head 6 output

                    # Set MEM_VAL_B0-B3 flags + MEM_STORE on val-byte positions
                    # for L15 byte selection. MEM_STORE on val bytes prevents
                    # dim-37 suppression from favoring MEM markers over val
                    # bytes in L15 attention.
                    for byte_off in range(4):
                        val_pos = i + 5 + byte_off
                        if val_pos < S:
                            x[b, val_pos, mem_val_dims[byte_off]] = 1.0
                            x[b, val_pos, mem_store] = 2.0  # Match MEM marker

                    # Write ADDR_KEY nibble decomposition on val-byte positions
                    # so L5/L8/L15 attention can match by address.
                    for byte_off in range(4):
                        val_pos = i + 5 + byte_off
                        if val_pos < S:
                            byte_addr = addr + byte_off
                            lo = byte_addr & 0xF
                            hi = (byte_addr >> 4) & 0xF
                            top = (byte_addr >> 8) & 0xF
                            x[b, val_pos, addr_key + lo] = 1.0
                            x[b, val_pos, addr_key + 16 + hi] = 1.0
                            x[b, val_pos, addr_key + 32 + top] = 1.0
                    i += 9  # Skip past MEM section
                else:
                    i += 1
