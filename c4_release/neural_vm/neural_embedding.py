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

    def __init__(self, vocab_size, d_model):
        """Initialize neural VM embedding.

        Args:
            vocab_size: Number of tokens in vocabulary (typically 272)
            d_model: Embedding dimension (typically 512)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

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

        # Apply augmentations in-place (deterministic transformations)
        self._add_code_addr_keys(token_ids, x)
        self._inject_mem_store(token_ids, x)

        # Inject active opcode flags for conversational I/O detection
        if active_opcode is not None:
            self._inject_active_opcode(x, active_opcode)

        # Inject THINKING_START/END markers for lookback detection
        self._inject_thinking_markers(token_ids, x)

        # Unified memory: either autoregressive inference or external hints
        if self._autoregressive_exec:
            self._inject_mem_exec_autoregressive(token_ids, x)
        elif self._exec_addrs:
            self._inject_mem_exec(token_ids, x)

        return x

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
        from .vm_step import _SetDim, Token
        from .constants import INSTR_WIDTH, PC_OFFSET
        BD = _SetDim

        # Bytes per instruction in token stream: opcode + 4 immediate + 3 padding = 8 total
        # ADDR_KEY is only set for the first 5 bytes (opcode + immediate), not padding
        BYTES_PER_INSTR = 8  # Total bytes in token stream
        DATA_BYTES = 5       # Bytes with ADDR_KEY (opcode + 4 immediate)

        B, S = token_ids.shape

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

                    # PC = instr_idx * INSTR_WIDTH + PC_OFFSET
                    # Opcode (byte_offset=0): addr = PC
                    # Imm[j] (byte_offset=j+1): addr = PC + j + 1
                    addr = instr_idx * INSTR_WIDTH + PC_OFFSET + byte_offset
                    # Write address as nibbles to ADDR_KEY (3 nibbles × 16 one-hot)
                    lo = addr & 0xF
                    hi = (addr >> 4) & 0xF
                    top = (addr >> 8) & 0xF
                    x[b, i, BD.ADDR_KEY + lo] = 1.0
                    x[b, i, BD.ADDR_KEY + 16 + hi] = 1.0
                    x[b, i, BD.ADDR_KEY + 32 + top] = 1.0

    def _inject_mem_store(self, token_ids, x):
        """Inject MEM_STORE=1.0 on historical MEM markers for L15 K-side.

        Historical MEM sections (from prior store ops retained in context)
        lack the MEM_STORE flag that L6 head 6 sets for the current step.
        Without this flag, L15 memory lookup won't match these positions.

        Only injects on MEM markers in the retained history region
        (0 .. _mem_history_end), not the current step's MEM section.
        """
        # Import here to avoid circular dependency
        from .vm_step import _SetDim, Token
        BD = _SetDim

        end = self._mem_history_end
        if end == 0:
            return

        B, S = token_ids.shape
        for b in range(B):
            for i in range(min(end, S)):
                if token_ids[b, i].item() == Token.MEM:
                    x[b, i, BD.MEM_STORE] = 1.0

    def set_mem_history_end(self, end):
        """Set the memory history boundary for MEM_STORE injection.

        Called by KV cache eviction logic when retaining historical tokens.

        Args:
            end: Position marking end of historical memory region
        """
        self._mem_history_end = end

    def _inject_active_opcode(self, x, active_opcode):
        """Inject active opcode flags into all positions.

        For conversational I/O: exposes the MoE routing signal globally so
        L5 FFN can detect PRTF/READ opcodes reliably.

        Args:
            x: [batch, seq, d_model] embedding tensor (modified in-place)
            active_opcode: Current opcode value (0-255)
        """
        from .vm_step import _SetDim as BD

        if active_opcode == 33:  # PRTF = 0x21
            x[:, :, BD.ACTIVE_OPCODE_PRTF] = 1.0
        elif active_opcode == 31:  # READ = 0x1F
            x[:, :, BD.ACTIVE_OPCODE_READ] = 1.0

    def _inject_thinking_markers(self, token_ids, x):
        """Inject THINKING_START and THINKING_END markers for lookback detection.

        Sets dedicated marker dimensions (not overlapping with OUTPUT_BYTE)
        so L2 lookback head can reliably detect these special tokens.

        Args:
            token_ids: [batch, seq] tensor of token IDs
            x: [batch, seq, d_model] embedding tensor (modified in-place)
        """
        from .vm_step import _SetDim as BD, Token

        B, S = token_ids.shape
        for b in range(B):
            for i in range(S):
                tok = token_ids[b, i].item()
                if tok == Token.THINKING_START:
                    x[b, i, BD.MARK_THINKING_START] = 1.0
                elif tok == Token.THINKING_END:
                    x[b, i, BD.MARK_THINKING_END] = 1.0

    def _inject_mem_exec(self, token_ids, x):
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

        from .vm_step import _SetDim, Token
        BD = _SetDim

        B, S = token_ids.shape

        for b in range(B):
            i = 0
            while i < S:
                tok = token_ids[b, i].item()
                if tok == Token.MEM and i + 8 < S:
                    # Extract address from MEM section (bytes 1-4, little-endian)
                    addr_bytes = [token_ids[b, i + 1 + j].item() for j in range(4)]
                    addr = addr_bytes[0] | (addr_bytes[1] << 8) | (addr_bytes[2] << 16) | (addr_bytes[3] << 24)

                    # Check if this address is in executable region (word-aligned)
                    if (addr & ~3) in self._exec_addrs:
                        # Set MEM_EXEC flag on MEM marker
                        x[b, i, BD.MEM_EXEC] = 1.0

                        # Add ADDR_KEY to value bytes (positions 5-8)
                        for byte_off in range(4):
                            val_pos = i + 5 + byte_off
                            if val_pos < S:
                                byte_addr = addr + byte_off
                                lo = byte_addr & 0xF
                                hi = (byte_addr >> 4) & 0xF
                                top = (byte_addr >> 8) & 0xF
                                x[b, val_pos, BD.ADDR_KEY + lo] = 1.0
                                x[b, val_pos, BD.ADDR_KEY + 16 + hi] = 1.0
                                x[b, val_pos, BD.ADDR_KEY + 32 + top] = 1.0
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

    def _inject_mem_exec_autoregressive(self, token_ids, x):
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
        """
        from .vm_step import _SetDim, Token
        BD = _SetDim

        # Code region boundary (addresses below this are code, above are data)
        # This matches C4 convention: code at low addresses, data at 0x10000+
        CODE_REGION_END = 0x10000

        B, S = token_ids.shape

        # Also collect jump targets from REG_PC sections for precise detection
        jump_targets = self._extract_jump_targets_autoregressive(token_ids)

        for b in range(B):
            i = 0
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
                        # Set MEM_EXEC flag on MEM marker
                        x[b, i, BD.MEM_EXEC] = 1.0

                        # Add ADDR_KEY to value bytes (positions 5-8)
                        for byte_off in range(4):
                            val_pos = i + 5 + byte_off
                            if val_pos < S:
                                byte_addr = addr + byte_off
                                lo = byte_addr & 0xF
                                hi = (byte_addr >> 4) & 0xF
                                top = (byte_addr >> 8) & 0xF
                                x[b, val_pos, BD.ADDR_KEY + lo] = 1.0
                                x[b, val_pos, BD.ADDR_KEY + 16 + hi] = 1.0
                                x[b, val_pos, BD.ADDR_KEY + 32 + top] = 1.0
                    i += 9  # Skip past MEM section
                else:
                    i += 1

    def _extract_jump_targets_autoregressive(self, token_ids):
        """Extract jump target addresses from PC outputs in context.

        When a jump occurs (JMP/JSR/BZ/BNZ), the next PC value is the target.
        By detecting non-sequential PC changes, we can identify jump targets.

        Returns:
            Set of word-aligned addresses that appear as jump targets
        """
        from .vm_step import Token
        from .constants import INSTR_WIDTH

        jump_targets = set()
        B, S = token_ids.shape

        for b in range(B):
            prev_pc = None
            i = 0
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
