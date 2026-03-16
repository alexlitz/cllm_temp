"""
KV Cache Memory System for Neural VM V7.

Architecture:
- Memory stored in KV cache with ±scale encoding for bits
- AliBi positional bias for address matching and recency
- 30-token VM step structure: PC(5) + AX(5) + SP(5) + BP(5) + MEM(9) + STEP_END(1)
- LOAD via attention over KV cache (address matching)
- STORE via V-overwriting (new value replaces old)
- 4-byte aligned memory addresses

Token structure per VM step:
  Tokens 0-4:   PC nibbles (20-bit program counter)
  Tokens 5-9:   AX nibbles (20-bit accumulator)
  Tokens 10-14: SP nibbles (20-bit stack pointer)
  Tokens 15-19: BP nibbles (20-bit base pointer)
  Tokens 20-28: MEM: addr_hi, addr_lo, value (3 tokens × 3 = 9 tokens for parallel access)
  Token 29:     STEP_END marker

Memory encoding:
  - Each nibble stored as float in [-scale, +scale] range
  - Binary: +scale = 1, -scale = 0
  - Decimal: direct float value

Address matching via AliBi:
  - Query encodes target address
  - Keys encode stored addresses
  - AliBi bias adds recency preference (newer writes preferred)
  - Softmax1 (softmax with +1 in denominator) for graceful missing values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .embedding import E


class VMStepLayout:
    """Layout of tokens within a single VM step."""

    # Token positions within a step
    PC_START = 0
    PC_END = 5      # 5 nibbles = 20 bits

    AX_START = 5
    AX_END = 10     # 5 nibbles = 20 bits

    SP_START = 10
    SP_END = 15     # 5 nibbles = 20 bits

    BP_START = 15
    BP_END = 20     # 5 nibbles = 20 bits

    # Memory access tokens (3 parallel accesses × 3 tokens each)
    MEM_START = 20
    MEM_END = 29    # 9 tokens for memory

    STEP_END = 29   # End marker

    TOKENS_PER_STEP = 30

    # Memory token structure (per access)
    MEM_ADDR_HI = 0   # Upper nibbles of address
    MEM_ADDR_LO = 1   # Lower nibbles of address
    MEM_VALUE = 2     # Value nibble

    # Number of nibbles per register
    NIBBLES_PER_REG = 5


class KVMemoryConfig:
    """Configuration for KV cache memory."""

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 4,
        max_steps: int = 1024,
        memory_size: int = 65536,  # 64KB addressable
        scale: float = 100.0,
        alibi_slope: float = 0.1,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_steps = max_steps
        self.memory_size = memory_size
        self.scale = scale
        self.alibi_slope = alibi_slope

        # Derived
        self.max_tokens = max_steps * VMStepLayout.TOKENS_PER_STEP


class AddressEncoder(nn.Module):
    """
    Encodes memory addresses for KV cache lookup.

    Address is split into nibbles and encoded into query/key vectors
    that will match via dot product attention.
    """

    def __init__(self, config: KVMemoryConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim

        # Each address nibble gets encoded into a portion of the vector
        # 4 nibbles for 16-bit address = 4 × (dim/4) dimensions
        self.nibble_dim = config.dim // 4

        # Learnable embeddings for each nibble value (0-15)
        # Using baked weights instead
        self.register_buffer(
            'nibble_basis',
            self._create_nibble_basis()
        )

    def _create_nibble_basis(self) -> torch.Tensor:
        """Create orthogonal basis vectors for nibble values."""
        # Each nibble (0-15) gets a unique vector
        # Using binary encoding for exact matching
        basis = torch.zeros(16, self.nibble_dim)

        for val in range(16):
            for bit in range(4):
                if val & (1 << bit):
                    basis[val, bit * (self.nibble_dim // 4):(bit + 1) * (self.nibble_dim // 4)] = 1.0
                else:
                    basis[val, bit * (self.nibble_dim // 4):(bit + 1) * (self.nibble_dim // 4)] = -1.0

        return basis

    def encode_address(self, addr_nibbles: torch.Tensor) -> torch.Tensor:
        """
        Encode address nibbles into a query/key vector.

        Args:
            addr_nibbles: [batch, 4] tensor of nibble values (0-15)

        Returns:
            [batch, dim] encoded address vector
        """
        batch = addr_nibbles.shape[0]
        encoded = torch.zeros(batch, self.dim, device=addr_nibbles.device)

        for i in range(4):
            nibble_vals = addr_nibbles[:, i].long()
            start = i * self.nibble_dim
            end = (i + 1) * self.nibble_dim
            encoded[:, start:end] = self.nibble_basis[nibble_vals]

        return encoded


class AliBiPositionalBias(nn.Module):
    """
    ALiBi (Attention with Linear Biases) for memory access.

    Provides:
    1. Recency bias: newer writes are preferred
    2. Address matching: exact address matches get highest scores

    The bias is added to attention logits before softmax.
    """

    def __init__(self, config: KVMemoryConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads

        # Different slopes per head for varied temporal preferences
        slopes = self._get_slopes(config.num_heads)
        self.register_buffer('slopes', slopes)

    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """Generate geometric sequence of slopes."""
        # Following ALiBi paper: slopes = 2^(-8/n * [1,2,...,n])
        ratio = 2 ** (-8 / num_heads)
        slopes = torch.tensor([ratio ** i for i in range(1, num_heads + 1)])
        return slopes.view(num_heads, 1, 1)

    def forward(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ALiBi bias for attention.

        Args:
            query_pos: [batch, q_len] position indices
            key_pos: [batch, k_len] position indices

        Returns:
            [batch, num_heads, q_len, k_len] bias tensor
        """
        # Distance matrix: query_pos - key_pos
        # Negative distance means key is in the past (should be accessible)
        # Positive distance means key is in the future (should be masked)

        q_pos = query_pos.unsqueeze(-1)  # [batch, q_len, 1]
        k_pos = key_pos.unsqueeze(-2)    # [batch, 1, k_len]

        distance = q_pos - k_pos  # [batch, q_len, k_len]

        # Apply slopes per head
        # Negative slope * negative distance = positive bias (good for past)
        # Negative slope * positive distance = negative bias (masks future)
        bias = -self.slopes * distance.unsqueeze(1)  # [batch, heads, q_len, k_len]

        return bias


class Softmax1(nn.Module):
    """
    Softmax with +1 in denominator for graceful handling of missing values.

    softmax1(x) = exp(x) / (1 + sum(exp(x)))

    When no keys match (all low scores), output approaches 0.
    When keys match well, output approaches standard softmax.
    """

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        exp_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])  # Numerical stability
        return exp_x / (1.0 + exp_x.sum(dim=dim, keepdim=True))


class KVMemoryAttention(nn.Module):
    """
    Memory attention layer using KV cache.

    For LOAD: Query with address, retrieve value from matching key
    For STORE: Overwrite V at matching position (handled externally)
    """

    def __init__(self, config: KVMemoryConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections (can be baked)
        self.W_q = nn.Linear(config.dim, config.dim, bias=False)
        self.W_k = nn.Linear(config.dim, config.dim, bias=False)
        self.W_v = nn.Linear(config.dim, config.dim, bias=False)
        self.W_o = nn.Linear(config.dim, config.dim, bias=False)

        self.alibi = AliBiPositionalBias(config)
        self.softmax1 = Softmax1()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for identity-like behavior."""
        with torch.no_grad():
            # Start with small random weights
            nn.init.xavier_uniform_(self.W_q.weight)
            nn.init.xavier_uniform_(self.W_k.weight)
            nn.init.xavier_uniform_(self.W_v.weight)
            nn.init.xavier_uniform_(self.W_o.weight)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Attention over KV cache for memory lookup.

        Args:
            query: [batch, q_len, dim] query embeddings (address)
            keys: [batch, k_len, dim] key embeddings (stored addresses)
            values: [batch, k_len, dim] value embeddings (stored values)
            query_pos: [batch, q_len] query positions
            key_pos: [batch, k_len] key positions
            mask: [batch, q_len, k_len] optional attention mask

        Returns:
            [batch, q_len, dim] retrieved values
        """
        batch, q_len, _ = query.shape
        k_len = keys.shape[1]

        # Project to Q, K, V
        Q = self.W_q(query).view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(keys).view(batch, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(values).view(batch, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Add ALiBi bias
        alibi_bias = self.alibi(query_pos, key_pos)
        scores = scores + alibi_bias

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        # Softmax1 for graceful missing value handling
        attn_weights = self.softmax1(scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attn_weights, V)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch, q_len, self.dim)
        output = self.W_o(output)

        return output


class KVMemoryCache(nn.Module):
    """
    KV cache for VM memory storage.

    Manages the growing cache of memory writes during VM execution.
    Supports efficient lookup and overwriting.
    """

    def __init__(self, config: KVMemoryConfig):
        super().__init__()
        self.config = config

        self.attention = KVMemoryAttention(config)
        self.address_encoder = AddressEncoder(config)

        # Cache state (managed externally during generation)
        self.register_buffer('cache_keys', None)
        self.register_buffer('cache_values', None)
        self.register_buffer('cache_positions', None)
        self.cache_len = 0

    def reset_cache(self):
        """Clear the KV cache."""
        self.cache_keys = None
        self.cache_values = None
        self.cache_positions = None
        self.cache_len = 0

    def append_to_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
    ):
        """
        Append new entries to the cache.

        Args:
            keys: [batch, new_len, dim] new key embeddings
            values: [batch, new_len, dim] new value embeddings
            positions: [batch, new_len] position indices
        """
        if self.cache_keys is None:
            self.cache_keys = keys
            self.cache_values = values
            self.cache_positions = positions
        else:
            self.cache_keys = torch.cat([self.cache_keys, keys], dim=1)
            self.cache_values = torch.cat([self.cache_values, values], dim=1)
            self.cache_positions = torch.cat([self.cache_positions, positions], dim=1)

        self.cache_len = self.cache_keys.shape[1]

    def load(
        self,
        address_nibbles: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Load value from memory at given address.

        Args:
            address_nibbles: [batch, 4] address as nibbles
            query_pos: [batch] current position for ALiBi

        Returns:
            [batch, dim] retrieved value (or zero if not found)
        """
        if self.cache_keys is None or self.cache_len == 0:
            return torch.zeros(address_nibbles.shape[0], self.config.dim,
                             device=address_nibbles.device)

        # Encode address as query
        query = self.address_encoder.encode_address(address_nibbles)
        query = query.unsqueeze(1)  # [batch, 1, dim]
        query_pos = query_pos.unsqueeze(1)  # [batch, 1]

        # Attend over cache
        result = self.attention(
            query, self.cache_keys, self.cache_values,
            query_pos, self.cache_positions
        )

        return result.squeeze(1)  # [batch, dim]

    def store(
        self,
        address_nibbles: torch.Tensor,
        value: torch.Tensor,
        position: torch.Tensor,
    ):
        """
        Store value to memory at given address.

        Appends to cache - ALiBi recency bias ensures latest write is used.

        Args:
            address_nibbles: [batch, 4] address as nibbles
            value: [batch, dim] value to store
            position: [batch] current position
        """
        # Encode address as key
        key = self.address_encoder.encode_address(address_nibbles)
        key = key.unsqueeze(1)  # [batch, 1, dim]
        value = value.unsqueeze(1)  # [batch, 1, dim]
        position = position.unsqueeze(1)  # [batch, 1]

        self.append_to_cache(key, value, position)


class VMStateEncoder(nn.Module):
    """
    Encodes VM state (registers) into token embeddings.

    Used for autoregressive generation of the 30-token VM step.
    """

    def __init__(self, config: KVMemoryConfig):
        super().__init__()
        self.config = config
        self.layout = VMStepLayout

        # Nibble embedding (0-15 values)
        self.nibble_embed = nn.Embedding(16, config.dim)

        # Position embedding within step
        self.pos_embed = nn.Embedding(VMStepLayout.TOKENS_PER_STEP, config.dim)

        # Token type embedding (PC, AX, SP, BP, MEM, END)
        self.type_embed = nn.Embedding(6, config.dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize with orthogonal embeddings."""
        with torch.no_grad():
            nn.init.orthogonal_(self.nibble_embed.weight)
            nn.init.orthogonal_(self.pos_embed.weight)
            nn.init.orthogonal_(self.type_embed.weight)

    def get_token_type(self, pos: int) -> int:
        """Get token type index for position within step."""
        if pos < self.layout.PC_END:
            return 0  # PC
        elif pos < self.layout.AX_END:
            return 1  # AX
        elif pos < self.layout.SP_END:
            return 2  # SP
        elif pos < self.layout.BP_END:
            return 3  # BP
        elif pos < self.layout.MEM_END:
            return 4  # MEM
        else:
            return 5  # END

    def encode_register(
        self,
        value: int,
        reg_start: int,
        step_offset: int = 0,
    ) -> torch.Tensor:
        """
        Encode a register value as token embeddings.

        Args:
            value: 20-bit register value
            reg_start: starting token position within step
            step_offset: global step offset for position

        Returns:
            [5, dim] embeddings for 5 nibbles
        """
        nibbles = []
        for i in range(VMStepLayout.NIBBLES_PER_REG):
            nibbles.append((value >> (i * 4)) & 0xF)

        embeddings = []
        for i, nib in enumerate(nibbles):
            pos = reg_start + i
            token_type = self.get_token_type(pos)

            emb = (self.nibble_embed.weight[nib] +
                   self.pos_embed.weight[pos] +
                   self.type_embed.weight[token_type])
            embeddings.append(emb)

        return torch.stack(embeddings)

    def encode_step(
        self,
        pc: int,
        ax: int,
        sp: int,
        bp: int,
        mem_ops: list = None,
    ) -> torch.Tensor:
        """
        Encode a complete VM step as token embeddings.

        Args:
            pc: program counter value
            ax: accumulator value
            sp: stack pointer value
            bp: base pointer value
            mem_ops: list of (addr, value) tuples for memory operations

        Returns:
            [30, dim] embeddings for full step
        """
        embeddings = []

        # Registers
        embeddings.append(self.encode_register(pc, self.layout.PC_START))
        embeddings.append(self.encode_register(ax, self.layout.AX_START))
        embeddings.append(self.encode_register(sp, self.layout.SP_START))
        embeddings.append(self.encode_register(bp, self.layout.BP_START))

        # Memory operations (9 tokens = 3 ops × 3 tokens)
        mem_embs = []
        for i in range(3):
            if mem_ops and i < len(mem_ops):
                addr, val = mem_ops[i]
                addr_hi = (addr >> 8) & 0xFF
                addr_lo = addr & 0xFF
            else:
                addr_hi, addr_lo, val = 0, 0, 0

            pos_base = self.layout.MEM_START + i * 3
            for j, nib in enumerate([addr_hi, addr_lo, val]):
                pos = pos_base + j
                emb = (self.nibble_embed.weight[nib & 0xF] +
                       self.pos_embed.weight[pos] +
                       self.type_embed.weight[4])  # MEM type
                mem_embs.append(emb)

        embeddings.append(torch.stack(mem_embs))

        # Step end token
        end_emb = (self.pos_embed.weight[self.layout.STEP_END] +
                   self.type_embed.weight[5])  # END type
        embeddings.append(end_emb.unsqueeze(0))

        return torch.cat([e.view(-1, self.config.dim) for e in embeddings], dim=0)


# Export classes
__all__ = [
    'VMStepLayout',
    'KVMemoryConfig',
    'AddressEncoder',
    'AliBiPositionalBias',
    'Softmax1',
    'KVMemoryAttention',
    'KVMemoryCache',
    'VMStateEncoder',
]
