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

        # Track executable memory addresses for unified memory
        # (set by runner when code is written to memory)
        self._exec_addrs = set()

        # Autoregressive mode: infer executable regions from token stream
        # When True, no external add_exec_addr() calls needed
        self._autoregressive_exec = True

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

        # Unified memory: either autoregressive inference or external hints.
        # Prefix has no MEM markers; skip prefix positions on cache hit.
        if self._autoregressive_exec:
            self._inject_mem_exec_autoregressive(token_ids, x, start_pos=start_pos)
        elif self._exec_addrs:
            self._inject_mem_exec(token_ids, x, start_pos=start_pos)

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

    def _add_code_addr_keys(self, token_ids, x):
        """Add ADDR_KEY to code byte embeddings (pure autoregressive).

        Code bytes (between CODE_START and CODE_END) are system prompt input.
        ADDR_KEY is position-dependent metadata added to embeddings,
        similar to how RoPE adds positional info. This is weight-based:
        the computation is deterministic from position, no learned parameters.

        Uses PC-aligned addressing to match L5 fetch queries:
        - L5 head 1 fetches opcode using EMBED_LO/HI (which holds PC value)
        - L5 head 0 fetches immediate using TEMP (which holds PC+1, PC+2, etc.)
        - ADDR_KEY must equal PC for opcode, PC+1 for imm[0], etc.

        With PC_OFFSET=2, INSTR_WIDTH=8:
        - Instruction i opcode: PC = i * 8 + 2, ADDR_KEY = PC
        - Instruction i imm[j]: ADDR_KEY = PC + j + 1

        Examples:
        - Instruction 0 opcode: ADDR_KEY = 0 * 8 + 2 = 2
        - Instruction 0 imm[0]: ADDR_KEY = 2 + 1 = 3
        - Instruction 1 opcode: ADDR_KEY = 1 * 8 + 2 = 10
        - Instruction 1 imm[0]: ADDR_KEY = 10 + 1 = 11
        """
        # Import here to avoid circular dependency
        from .vm_step import Token
        from .constants import INSTR_WIDTH, PC_OFFSET

        # Bytes per instruction in token stream: opcode + 4 immediate + 3 padding = 8 total
        # ADDR_KEY is only set for the first 5 bytes (opcode + immediate), not padding
        BYTES_PER_INSTR = 8  # Total bytes in token stream
        DATA_BYTES = 5       # Bytes with ADDR_KEY (opcode + 4 immediate)

        B, S = token_ids.shape
        addr_key = self._dim("ADDR_KEY")

        for b in range(B):
            cs_pos = None
            for i in range(S):
                tok = token_ids[b, i].item()
                if tok == Token.CODE_START:
                    cs_pos = i
                elif tok == Token.CODE_END:
                    break
                elif cs_pos is not None and tok < 256:
                    seq_pos = i - cs_pos - 1  # Sequential position in code
                    if seq_pos < 0:
                        continue
                    # Convert to PC-aligned address
                    instr_idx = seq_pos // BYTES_PER_INSTR
                    byte_offset = seq_pos % BYTES_PER_INSTR

                    # Skip padding bytes (beyond first 5 bytes of instruction)
                    if byte_offset >= DATA_BYTES:
                        continue

                    addr = instr_idx * INSTR_WIDTH + PC_OFFSET + byte_offset
                    # Write address as nibbles to ADDR_KEY (3 nibbles × 16 one-hot)
                    lo = addr & 0xF
                    hi = (addr >> 4) & 0xF
                    top = (addr >> 8) & 0xF
                    x[b, i, addr_key + lo] = 1.0
                    x[b, i, addr_key + 16 + hi] = 1.0
                    x[b, i, addr_key + 32 + top] = 1.0

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

    def _inject_mem_exec(self, token_ids, x, start_pos=0):
        """Inject MEM_EXEC and ADDR_KEY on MEM sections containing executable code.

        For unified memory execution: when code is written to memory via SI/SC,
        the resulting MEM sections need ADDR_KEY so L5 can fetch from them.

        MEM section format: [MEM, addr_b0, addr_b1, addr_b2, addr_b3,
                             val_b0, val_b1, val_b2, val_b3]
        - addr bytes are at positions 1-4 from MEM marker
        - val bytes are at positions 5-8 from MEM marker (these get ADDR_KEY)

        The ADDR_KEY on val bytes equals addr + byte_offset (0-3).
        """
        if not self._exec_addrs:
            return

        from .vm_step import Token

        mem_exec = self._dim("MEM_EXEC")
        addr_key = self._dim("ADDR_KEY")

        B, S = token_ids.shape

        for b in range(B):
            i = start_pos
            while i < S:
                tok = token_ids[b, i].item()
                if tok == Token.MEM and i + 8 < S:
                    # Extract address from MEM section (bytes 1-4, little-endian)
                    addr_bytes = [token_ids[b, i + 1 + j].item() for j in range(4)]
                    addr = addr_bytes[0] | (addr_bytes[1] << 8) | (addr_bytes[2] << 16) | (addr_bytes[3] << 24)

                    # Check if this address is in executable region (word-aligned)
                    if (addr & ~3) in self._exec_addrs:
                        # Set MEM_EXEC flag on MEM marker
                        x[b, i, mem_exec] = 1.0

                        # Add ADDR_KEY to value bytes (positions 5-8)
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

    def set_exec_addrs(self, addrs):
        """Set the set of addresses containing executable code.

        Called by runner when tracking which memory regions are executable.

        Args:
            addrs: Set or iterable of word-aligned addresses
        """
        self._exec_addrs = set(a & ~3 for a in addrs)

    def add_exec_addr(self, addr):
        """Add an address to the executable set.

        Args:
            addr: Address to mark as executable (will be word-aligned)
        """
        self._exec_addrs.add(addr & ~3)

    def clear_exec_addrs(self):
        """Clear all executable address markers."""
        self._exec_addrs.clear()

    def _inject_mem_exec_autoregressive(self, token_ids, x, start_pos=0):
        """Infer executable regions from token stream (fully autoregressive).

        This method analyzes the token context to determine which MEM sections
        contain executable code, without requiring external hints.

        Strategies:
        1. Von Neumann: All MEM sections get ADDR_KEY (any memory can be code)
        2. Jump target inference: Detect PC values from jump instructions
        3. Code region heuristic: Low addresses (< 0x10000) are typically code

        Currently implements strategy 1+3: Von Neumann for code region,
        which marks MEM sections in the code address range as executable.

        MEM section format: [MEM, addr_b0-3, val_b0-3] (9 tokens)

        Args:
            start_pos: Skip positions ``< start_pos``. Safe when the prefix
                contains no MEM markers (true for the CODE/DATA system prompt).
        """
        from .vm_step import Token

        mem_exec = self._dim("MEM_EXEC")
        mem_store = self._dim("MEM_STORE")
        addr_key = self._dim("ADDR_KEY")
        mem_val_dims = [
            self._dim("MEM_VAL_B0"),
            self._dim("MEM_VAL_B1"),
            self._dim("MEM_VAL_B2"),
            self._dim("MEM_VAL_B3"),
        ]

        # Code region boundary (addresses below this are code, above are data)
        # This matches C4 convention: code at low addresses, data at 0x10000+
        CODE_REGION_END = 0x10000

        B, S = token_ids.shape

        # Also collect jump targets from REG_PC sections for precise detection
        jump_targets = self._extract_jump_targets_autoregressive(
            token_ids, start_pos=start_pos
        )

        for b in range(B):
            i = start_pos
            while i < S:
                tok = token_ids[b, i].item()
                if tok == Token.MEM and i + 8 < S:
                    # Extract address from MEM section (bytes 1-4, little-endian)
                    addr_bytes = [token_ids[b, i + 1 + j].item() for j in range(4)]
                    addr = (addr_bytes[0] | (addr_bytes[1] << 8) |
                            (addr_bytes[2] << 16) | (addr_bytes[3] << 24))

                    # Determine if this address should be executable:
                    # 1. Address is in code region (< CODE_REGION_END)
                    # 2. Address appears as a jump target in context
                    is_code_region = addr < CODE_REGION_END
                    is_jump_target = (addr & ~3) in jump_targets

                    if is_code_region or is_jump_target:
                        # Set MEM_EXEC flag on MEM marker (for code only)
                        x[b, i, mem_exec] = 1.0

                    # FIX 2026-04-16: Always add ADDR_KEY to ALL MEM sections, not just code.
                    # This enables L15 memory lookup for LEV (stack), LI/LC/SI/SC (data).
                    # ADDR_KEY is needed for address matching in attention-based memory reads.

                    # FIX 2026-04-17: Set MEM_STORE on ALL MEM markers for L15 K-side matching.
                    # L15 attention uses MEM_STORE in K-side to identify store entries.
                    # Without this, the attention score gets -312.5 penalty and doesn't match.
                    x[b, i, mem_store] = 2.0  # Same scale as L6 head 6 output

                    # FIX 2026-04-17: Set MEM_VAL_B0-B3 flags on val byte positions for L15 byte selection.
                    # These flags indicate which value byte each position represents.
                    # FIX 2026-04-17: Also inject MEM_STORE on val byte positions so that after L7
                    # broadcast, MEM marker and val bytes have equal MEM_STORE. This prevents dim 37
                    # suppression from favoring MEM markers over val bytes in L15 attention.
                    for byte_off in range(4):
                        val_pos = i + 5 + byte_off
                        if val_pos < S:
                            x[b, val_pos, mem_val_dims[byte_off]] = 1.0
                            x[b, val_pos, mem_store] = 2.0  # Match MEM marker injection

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

    def _extract_jump_targets_autoregressive(self, token_ids, start_pos=0):
        """Extract jump target addresses from PC outputs in context.

        When a jump occurs (JMP/JSR/BZ/BNZ), the next PC value is the target.
        By detecting non-sequential PC changes, we can identify jump targets.

        Args:
            start_pos: Skip positions ``< start_pos``. Safe when the prefix
                contains no REG_PC markers (true for the CODE/DATA prompt).

        Returns:
            Set of word-aligned addresses that appear as jump targets
        """
        from .vm_step import Token
        from .constants import INSTR_WIDTH

        jump_targets = set()
        B, S = token_ids.shape

        for b in range(B):
            prev_pc = None
            i = start_pos
            while i < S:
                tok = token_ids[b, i].item()
                if tok == Token.REG_PC and i + 4 < S:
                    # Extract PC value (bytes 1-4, little-endian)
                    pc_bytes = [token_ids[b, i + 1 + j].item() for j in range(4)]
                    pc = (pc_bytes[0] | (pc_bytes[1] << 8) |
                          (pc_bytes[2] << 16) | (pc_bytes[3] << 24))

                    if prev_pc is not None:
                        # Check if this is a non-sequential PC change (jump)
                        expected_next = prev_pc + INSTR_WIDTH
                        if pc != expected_next:
                            # This PC came from a jump - it's a potential code location
                            jump_targets.add(pc & ~3)

                    prev_pc = pc
                    i += 5  # Skip past REG_PC section
                else:
                    # Don't reset prev_pc - we want to track jumps across steps
                    i += 1

        return jump_targets

    def set_autoregressive_exec(self, enabled):
        """Enable or disable autoregressive executable region detection.

        Args:
            enabled: If True, infer executable regions from token stream.
                     If False, use external add_exec_addr() hints.
        """
        self._autoregressive_exec = enabled
