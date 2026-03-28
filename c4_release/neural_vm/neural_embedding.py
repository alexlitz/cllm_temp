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

    def forward(self, token_ids):
        """Apply embedding + augmentations.

        Args:
            token_ids: [batch, seq] tensor of token IDs

        Returns:
            x: [batch, seq, d_model] embeddings with augmentations applied
        """
        # Standard embedding lookup
        x = self.embed(token_ids)

        # Apply augmentations in-place (deterministic transformations)
        self._add_code_addr_keys(token_ids, x)
        self._inject_mem_store(token_ids, x)

        return x

    def _add_code_addr_keys(self, token_ids, x):
        """Add ADDR_KEY to code byte embeddings (pure autoregressive).

        Code bytes (between CODE_START and CODE_END) are system prompt input.
        ADDR_KEY is position-dependent metadata added to embeddings,
        similar to how RoPE adds positional info. This is weight-based:
        the computation is deterministic from position, no learned parameters.

        Uses INSTR_WIDTH-aligned addressing to match DraftVM PC values:
        - Instructions are 5 bytes in the token stream (1 opcode + 4 immediate)
        - But PC addresses use INSTR_WIDTH (8) byte alignment
        - So: addr = (seq_pos // 5) * 8 + (seq_pos % 5)

        Examples with INSTR_WIDTH=8:
        - Code byte 0 (opcode 0): addr=0
        - Code byte 1 (imm[0]): addr=1
        - Code byte 4 (imm[3]): addr=4
        - Code byte 5 (opcode 1): addr=8
        - Code byte 6 (imm[0]): addr=9
        """
        # Import here to avoid circular dependency
        from .vm_step import _SetDim, Token
        from .constants import INSTR_WIDTH
        BD = _SetDim

        # Bytes per instruction in token stream (opcode + 4 immediate bytes)
        BYTES_PER_INSTR = 5

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
                    # Convert to INSTR_WIDTH-aligned address
                    instr_idx = seq_pos // BYTES_PER_INSTR
                    byte_offset = seq_pos % BYTES_PER_INSTR
                    addr = instr_idx * INSTR_WIDTH + byte_offset
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
