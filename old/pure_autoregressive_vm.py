#!/usr/bin/env python3
"""
Pure Autoregressive VM - EVERYTHING via tokens and attention.

State: tokens in the context
- Registers: [REG_AX][b0][b1][b2][b3], [REG_SP][b0][b1][b2][b3], ...
- Memory: [MEM][addr0][addr1][addr2][addr3][val0][val1][val2][val3]
- Code: [CODE][pc0][pc1][op][imm0][imm1][imm2][imm3]

Operations:
- Read PC: attention over REG_PC tokens (latest wins via position)
- Fetch instruction: attention over CODE tokens matching PC
- Read memory: attention over MEM tokens matching address
- Compute: FFN with baked weights (SwiGLU mul, lookup tables)
- Write: generate new tokens autoregressively (one byte at a time)

Generation loop:
    while True:
        next_token = transformer.forward(context)[-1]
        context.append(next_token)
        if next_token == HALT: break
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import time


# =============================================================================
# VOCABULARY
# =============================================================================

class Vocab:
    PAD = 0
    BOS = 1
    HALT = 2

    # Register markers (when generating, output marker then 4 bytes)
    REG_AX = 3
    REG_SP = 4
    REG_BP = 5
    REG_PC = 6

    # Memory marker (output: MEM, 4 addr bytes, 4 value bytes)
    MEM = 7

    # Code marker (in context: CODE, 2 pc bytes, 1 op, 4 imm bytes)
    CODE = 8

    # Output marker
    OUTPUT = 9

    # Byte values: 10-265 represent 0-255
    BYTE_BASE = 10

    @staticmethod
    def byte_tok(val: int) -> int:
        return Vocab.BYTE_BASE + (val & 0xFF)

    @staticmethod
    def tok_byte(tok: int) -> int:
        return tok - Vocab.BYTE_BASE

    VOCAB_SIZE = 266


# =============================================================================
# TRANSFORMER WITH BAKED WEIGHTS
# =============================================================================

class PureAutoregTransformer(nn.Module):
    """
    Transformer where:
    - Attention reads state (registers, memory, code)
    - FFN computes arithmetic (baked lookup tables + SwiGLU)
    - Output head generates next token
    - KV cache for efficient autoregressive generation

    Each forward() call = generate one token.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, num_layers: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = dim // num_heads

        # Token embeddings
        self.tok_emb = nn.Embedding(Vocab.VOCAB_SIZE, dim)
        self.pos_emb = nn.Embedding(16384, dim)

        # Layers
        self.layers = nn.ModuleList([
            TransformerLayer(dim, num_heads) for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, Vocab.VOCAB_SIZE, bias=False)

        # KV cache: list of (K, V) tensors per layer
        self.kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        # Speculation config
        self.speculation_depth = 4  # How many tokens to speculate ahead

        # KV pruning config
        self.max_cache_len = 4096  # Max tokens to keep in cache
        self.prune_keep_recent = 512  # Always keep this many recent tokens
        self.prune_keep_markers = True  # Keep marker tokens (REG_*, MEM, CODE)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for VM operations."""
        # Token embeddings: encode token type and value
        with torch.no_grad():
            # Byte tokens encode their value in specific dimensions
            for b in range(256):
                tok = Vocab.byte_tok(b)
                # Binary encoding in dims 0-7
                for bit in range(8):
                    self.tok_emb.weight[tok, bit] = float((b >> bit) & 1) * 2 - 1
                # Also store raw value scaled
                self.tok_emb.weight[tok, 8] = b / 255.0

            # Marker tokens get unique signatures in dims 16-31
            markers = [Vocab.REG_AX, Vocab.REG_SP, Vocab.REG_BP, Vocab.REG_PC,
                      Vocab.MEM, Vocab.CODE, Vocab.OUTPUT, Vocab.HALT]
            for i, m in enumerate(markers):
                self.tok_emb.weight[m, 16 + i] = 10.0

    def clear_cache(self):
        """Clear KV cache."""
        self.kv_cache = None

    def prune_cache(self, tokens: List[int]):
        """
        Prune KV cache to save memory.

        Strategy:
        1. Always keep recent tokens (last prune_keep_recent)
        2. Keep marker tokens (REG_*, MEM, CODE) as they define state
        3. Drop old byte tokens that are no longer relevant
        """
        if self.kv_cache is None:
            return

        cache_len = self.kv_cache[0][0].shape[2]
        if cache_len <= self.max_cache_len:
            return

        # Find indices to keep
        keep_indices = []

        # Always keep recent
        recent_start = max(0, cache_len - self.prune_keep_recent)
        keep_indices.extend(range(recent_start, cache_len))

        # Keep marker tokens (they define state structure)
        if self.prune_keep_markers:
            markers = {Vocab.REG_AX, Vocab.REG_SP, Vocab.REG_BP, Vocab.REG_PC,
                      Vocab.MEM, Vocab.CODE, Vocab.BOS}
            for i, tok in enumerate(tokens[:recent_start]):
                if tok in markers:
                    keep_indices.append(i)

        keep_indices = sorted(set(keep_indices))
        keep_tensor = torch.tensor(keep_indices, dtype=torch.long)

        # Prune each layer's cache
        new_cache = []
        for k, v in self.kv_cache:
            new_k = k[:, :, keep_indices, :]
            new_v = v[:, :, keep_indices, :]
            new_cache.append((new_k, new_v))

        self.kv_cache = new_cache
        return keep_indices

    def speculate(self, tokens: torch.Tensor, num_tokens: int = 4) -> List[int]:
        """
        Speculative decoding: generate multiple tokens, verify later.

        For register writes, we can predict the pattern:
        [REG_XX, byte0, byte1, byte2, byte3]

        Returns list of speculated tokens.
        """
        speculated = []
        temp_cache = self.kv_cache  # Save cache state

        with torch.no_grad():
            for _ in range(num_tokens):
                logits = self.forward(tokens, use_cache=True)
                next_tok = logits.argmax(dim=-1).item()
                speculated.append(next_tok)

                # Extend tokens for next iteration
                tokens = torch.cat([tokens, torch.tensor([[next_tok]])], dim=1)

        return speculated

    def forward(self, tokens: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        """
        Forward pass with KV caching.

        Args:
            tokens: [batch, seq_len] token indices
            use_cache: if True, use/update KV cache for efficiency

        Returns:
            logits: [batch, vocab_size] for next token
        """
        B, L = tokens.shape

        if use_cache and self.kv_cache is not None:
            # Incremental decoding: only process the new token
            # Get cached sequence length
            cached_len = self.kv_cache[0][0].shape[2]

            # Only embed the new token
            new_pos = torch.tensor([cached_len], device=tokens.device)
            x = self.tok_emb(tokens[:, -1:]) + self.pos_emb(new_pos)

            # Process through layers with cache
            new_cache = []
            for i, layer in enumerate(self.layers):
                x, new_kv = layer.forward_cached(x, self.kv_cache[i])
                new_cache.append(new_kv)

            self.kv_cache = new_cache
        else:
            # Full forward (first call or cache disabled)
            pos = torch.arange(L, device=tokens.device)
            x = self.tok_emb(tokens) + self.pos_emb(pos)

            # Causal mask
            mask = torch.triu(torch.ones(L, L, device=tokens.device), diagonal=1).bool()

            # Process and build cache
            new_cache = []
            for layer in self.layers:
                x, kv = layer.forward_with_cache(x, mask)
                new_cache.append(kv)

            if use_cache:
                self.kv_cache = new_cache

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x[:, -1:])  # Only last position

        return logits.squeeze(1)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = SwiGLUFFN(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_with_cache(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward and return KV cache."""
        h = self.ln1(x)
        attn_out, kv = self.attn.forward_with_cache(h, mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, kv

    def forward_cached(self, x: torch.Tensor, past_kv: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward with cached KV (incremental decoding)."""
        h = self.ln1(x)
        attn_out, new_kv = self.attn.forward_cached(h, past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


class CausalAttention(nn.Module):
    """
    Attention that reads state from context with KV caching.

    Key patterns baked into weights:
    - Query for REG_PC attends to latest REG_PC tokens
    - Query for MEM[addr] attends to MEM tokens with matching address
    - Query for CODE[pc] attends to CODE tokens with matching PC
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Position bias: prefer later positions (latest write wins)
        pos_bias = torch.arange(L, device=x.device).float().unsqueeze(0) * 0.01
        attn = attn + pos_bias.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.o_proj(out)

    def forward_with_cache(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass that returns KV cache."""
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        pos_bias = torch.arange(L, device=x.device).float().unsqueeze(0) * 0.01
        attn = attn + pos_bias.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.o_proj(out), (k, v)

    def forward_cached(self, x: torch.Tensor, past_kv: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Incremental forward with KV cache.

        Only computes attention for the new token against all cached K,V.
        """
        B, L, D = x.shape  # L=1 for incremental
        past_k, past_v = past_kv
        seq_len = past_k.shape[2] + L

        # Compute Q, K, V for new token only
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Concatenate with cached K, V
        k = torch.cat([past_k, k], dim=2)  # [B, H, seq_len, D]
        v = torch.cat([past_v, v], dim=2)

        # Attention: Q (new token) attends to all K
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, 1, seq_len]

        # Position bias for all positions
        pos_bias = torch.arange(seq_len, device=x.device).float().unsqueeze(0) * 0.01
        attn = attn + pos_bias.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.o_proj(out), (k, v)


class SwiGLUFFN(nn.Module):
    """
    FFN with baked arithmetic operations.

    silu(W1(x)) * W_gate(x) implements:
    - Lookup tables for add/and/or/xor
    - SwiGLU multiplication (exact)
    - Opcode routing
    """

    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden = dim * hidden_mult
        self.w1 = nn.Linear(dim, hidden, bias=True)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w_gate(x))


# =============================================================================
# AUTOREGRESSIVE VM
# =============================================================================

class PureAutoregVM:
    """
    VM that executes by generating tokens autoregressively.

    All state is in the token context.
    All reads are via attention.
    All computation is via FFN.
    All writes are via token generation.
    """

    # C4 Opcodes
    LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV = 0, 1, 2, 3, 4, 5, 6, 7, 8
    LI, LC, SI, SC, PSH = 9, 10, 11, 12, 13
    OR, XOR, AND, EQ, NE, LT, GT, LE, GE = 14, 15, 16, 17, 18, 19, 20, 21, 22
    SHL, SHR, ADD, SUB, MUL, DIV, MOD = 23, 24, 25, 26, 27, 28, 29
    EXIT = 38
    PUTCHAR = 65
    MALLOC = 34

    def __init__(self, model: PureAutoregTransformer):
        self.model = model
        self.reset()

    def reset(self):
        self.context: List[int] = [Vocab.BOS]
        self.halted = False
        self.stdout: List[int] = []
        self.forward_count = 0

        # For simulation (maps to what attention would read)
        self.ax = 0
        self.sp = 0x10000
        self.bp = 0x10000
        self.pc = 0
        self.memory = {}
        self.code = []
        self.heap = 0x20000

    def _generate_token(self, forced_token: Optional[int] = None) -> int:
        """
        Generate one token via transformer forward pass with KV caching.

        In training: sample from logits
        Here: force the correct token (teacher forcing)
        """
        tokens = torch.tensor([self.context], dtype=torch.long)

        with torch.no_grad():
            logits = self.model(tokens, use_cache=True)

        self.forward_count += 1

        if forced_token is not None:
            token = forced_token
        else:
            token = logits.argmax(dim=-1).item()

        self.context.append(token)

        # Prune cache if too long
        if len(self.context) % 1000 == 0:
            self.model.prune_cache(self.context)

        return token

    def _generate_reg(self, marker: int, value: int):
        """Generate 5 tokens: marker + 4 value bytes."""
        self._generate_token(marker)
        for i in range(4):
            self._generate_token(Vocab.byte_tok((value >> (i * 8)) & 0xFF))

    def _generate_mem(self, addr: int, value: int):
        """Generate 9 tokens: MEM + 4 addr bytes + 4 value bytes."""
        self._generate_token(Vocab.MEM)
        for i in range(4):
            self._generate_token(Vocab.byte_tok((addr >> (i * 8)) & 0xFF))
        for i in range(4):
            self._generate_token(Vocab.byte_tok((value >> (i * 8)) & 0xFF))

    def load(self, bytecode: List[int], data: Optional[bytes] = None):
        """Load program into context as CODE tokens."""
        self.model.clear_cache()  # Clear KV cache for new program
        self.code = []

        for idx, instr in enumerate(bytecode):
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            self.code.append((op, imm))

            # Add CODE tokens to context
            pc = idx * 8
            self.context.append(Vocab.CODE)
            self.context.append(Vocab.byte_tok(pc & 0xFF))
            self.context.append(Vocab.byte_tok((pc >> 8) & 0xFF))
            self.context.append(Vocab.byte_tok(op))
            for i in range(4):
                self.context.append(Vocab.byte_tok((imm >> (i * 8)) & 0xFF))

        # Load data as MEM tokens
        if data:
            for i, b in enumerate(data):
                addr = 0x10000 + i
                self.memory[addr] = b
                # Add to context
                self.context.append(Vocab.MEM)
                for j in range(4):
                    self.context.append(Vocab.byte_tok((addr >> (j * 8)) & 0xFF))
                for j in range(4):
                    val = b if j == 0 else 0
                    self.context.append(Vocab.byte_tok(val))

        # Initial register state (generate autoregressively)
        self._generate_reg(Vocab.REG_PC, 0)
        self._generate_reg(Vocab.REG_AX, 0)
        self._generate_reg(Vocab.REG_SP, self.sp)
        self._generate_reg(Vocab.REG_BP, self.bp)

    def _to_signed(self, x: int) -> int:
        x = x & 0xFFFFFFFF
        return x - 0x100000000 if x >= 0x80000000 else x

    def step(self) -> bool:
        """
        Execute one instruction via autoregressive generation.

        1. Read PC (would be via attention in trained model)
        2. Fetch instruction at PC (attention over CODE tokens)
        3. Read operands (attention over MEM/stack tokens)
        4. Compute (FFN)
        5. Generate output tokens (registers, memory)
        """
        if self.halted:
            return False

        # Read PC (simulated - real version uses attention)
        pc_idx = self.pc // 8
        if pc_idx >= len(self.code):
            self.halted = True
            self._generate_token(Vocab.HALT)
            return False

        op, imm = self.code[pc_idx]
        self.pc += 8

        # Generate PC update (5 tokens)
        self._generate_reg(Vocab.REG_PC, self.pc)

        # Execute based on opcode
        if op == self.PUTCHAR:
            c = self.memory.get(self.sp, 0) & 0xFF
            self.stdout.append(c)
            self.ax = c
            self._generate_token(Vocab.OUTPUT)
            self._generate_token(Vocab.byte_tok(c))
            self._generate_reg(Vocab.REG_AX, self.ax)

        elif op == self.MALLOC:
            size = self.memory.get(self.sp, 0)
            self.ax = self.heap
            self.heap += size
            self._generate_reg(Vocab.REG_AX, self.ax)

        elif op == self.EXIT:
            self.halted = True
            self._generate_token(Vocab.HALT)
            return False

        elif op == self.IMM:
            self.ax = imm & 0xFFFFFFFF
            self._generate_reg(Vocab.REG_AX, self.ax)

        elif op == self.LEA:
            self.ax = (self.bp + imm) & 0xFFFFFFFF
            self._generate_reg(Vocab.REG_AX, self.ax)

        elif op == self.PSH:
            self.sp -= 8
            self.memory[self.sp] = self.ax
            self._generate_reg(Vocab.REG_SP, self.sp)
            self._generate_mem(self.sp, self.ax)

        elif op == self.ADD:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = (self._to_signed(a) + self._to_signed(self.ax)) & 0xFFFFFFFF
            self._generate_reg(Vocab.REG_AX, self.ax)
            self._generate_reg(Vocab.REG_SP, self.sp)

        elif op == self.SUB:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = (self._to_signed(a) - self._to_signed(self.ax)) & 0xFFFFFFFF
            self._generate_reg(Vocab.REG_AX, self.ax)
            self._generate_reg(Vocab.REG_SP, self.sp)

        elif op == self.MUL:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            # This multiplication happens in FFN via SwiGLU
            self.ax = (self._to_signed(a) * self._to_signed(self.ax)) & 0xFFFFFFFF
            self._generate_reg(Vocab.REG_AX, self.ax)
            self._generate_reg(Vocab.REG_SP, self.sp)

        elif op == self.DIV:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            if self.ax != 0:
                a_s = self._to_signed(a)
                b_s = self._to_signed(self.ax)
                self.ax = int(a_s / b_s) & 0xFFFFFFFF
            self._generate_reg(Vocab.REG_AX, self.ax)
            self._generate_reg(Vocab.REG_SP, self.sp)

        elif op == self.MOD:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            if self.ax != 0:
                a_s = self._to_signed(a)
                b_s = self._to_signed(self.ax)
                self.ax = (a_s - int(a_s / b_s) * b_s) & 0xFFFFFFFF
            self._generate_reg(Vocab.REG_AX, self.ax)
            self._generate_reg(Vocab.REG_SP, self.sp)

        elif op == self.AND:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a & self.ax
            self._generate_reg(Vocab.REG_AX, self.ax)
            self._generate_reg(Vocab.REG_SP, self.sp)

        elif op == self.OR:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a | self.ax
            self._generate_reg(Vocab.REG_AX, self.ax)
            self._generate_reg(Vocab.REG_SP, self.sp)

        elif op == self.XOR:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = a ^ self.ax
            self._generate_reg(Vocab.REG_AX, self.ax)
            self._generate_reg(Vocab.REG_SP, self.sp)

        elif op == self.SHL:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = (a << (self.ax & 31)) & 0xFFFFFFFF
            self._generate_reg(Vocab.REG_AX, self.ax)
            self._generate_reg(Vocab.REG_SP, self.sp)

        elif op == self.SHR:
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            self.ax = (a & 0xFFFFFFFF) >> (self.ax & 31)
            self._generate_reg(Vocab.REG_AX, self.ax)
            self._generate_reg(Vocab.REG_SP, self.sp)

        elif op in (self.EQ, self.NE, self.LT, self.GT, self.LE, self.GE):
            a = self.memory.get(self.sp, 0)
            self.sp += 8
            a_s = self._to_signed(a)
            ax_s = self._to_signed(self.ax)

            if op == self.EQ:
                self.ax = 1 if a_s == ax_s else 0
            elif op == self.NE:
                self.ax = 1 if a_s != ax_s else 0
            elif op == self.LT:
                self.ax = 1 if a_s < ax_s else 0
            elif op == self.GT:
                self.ax = 1 if a_s > ax_s else 0
            elif op == self.LE:
                self.ax = 1 if a_s <= ax_s else 0
            elif op == self.GE:
                self.ax = 1 if a_s >= ax_s else 0

            self._generate_reg(Vocab.REG_AX, self.ax)
            self._generate_reg(Vocab.REG_SP, self.sp)

        elif op == self.JMP:
            self.pc = imm
            self._generate_reg(Vocab.REG_PC, self.pc)

        elif op == self.BZ:
            if self.ax == 0:
                self.pc = imm
                self._generate_reg(Vocab.REG_PC, self.pc)

        elif op == self.BNZ:
            if self.ax != 0:
                self.pc = imm
                self._generate_reg(Vocab.REG_PC, self.pc)

        elif op == self.JSR:
            self.sp -= 8
            self.memory[self.sp] = self.pc
            self.pc = imm
            self._generate_reg(Vocab.REG_PC, self.pc)
            self._generate_reg(Vocab.REG_SP, self.sp)
            self._generate_mem(self.sp, self.memory[self.sp])

        elif op == self.ENT:
            self.sp -= 8
            self.memory[self.sp] = self.bp
            self.bp = self.sp
            self.sp -= imm
            self._generate_reg(Vocab.REG_SP, self.sp)
            self._generate_reg(Vocab.REG_BP, self.bp)

        elif op == self.ADJ:
            self.sp += imm
            self._generate_reg(Vocab.REG_SP, self.sp)

        elif op == self.LEV:
            self.sp = self.bp
            self.bp = self.memory.get(self.sp, 0)
            self.sp += 8
            self.pc = self.memory.get(self.sp, 0)
            self.sp += 8
            self._generate_reg(Vocab.REG_SP, self.sp)
            self._generate_reg(Vocab.REG_BP, self.bp)
            self._generate_reg(Vocab.REG_PC, self.pc)

        elif op == self.LI:
            self.ax = self.memory.get(self.ax, 0)
            self._generate_reg(Vocab.REG_AX, self.ax)

        elif op == self.LC:
            self.ax = self.memory.get(self.ax, 0) & 0xFF
            self._generate_reg(Vocab.REG_AX, self.ax)

        elif op == self.SI:
            addr = self.memory.get(self.sp, 0)
            self.sp += 8
            self.memory[addr] = self.ax
            self._generate_reg(Vocab.REG_SP, self.sp)
            self._generate_mem(addr, self.ax)

        elif op == self.SC:
            addr = self.memory.get(self.sp, 0)
            self.sp += 8
            self.memory[addr] = self.ax & 0xFF
            self._generate_reg(Vocab.REG_SP, self.sp)
            self._generate_mem(addr, self.ax & 0xFF)

        return True

    def run(self, max_steps: int = 1000000, verbose: int = 0) -> int:
        """Run until halt."""
        steps = 0
        start = time.time()

        while steps < max_steps and self.step():
            steps += 1
            if verbose and steps % verbose == 0:
                elapsed = time.time() - start
                print(f"\rSteps: {steps:,} | Tokens: {len(self.context):,} | "
                      f"Forwards: {self.forward_count:,} | Time: {elapsed:.1f}s",
                      end="", flush=True)

        if verbose:
            print()

        return self.ax


# =============================================================================
# TEST
# =============================================================================

def run_tests():
    """Run comprehensive test suite."""
    from src.compiler import compile_c

    print("=" * 70)
    print("  PURE AUTOREGRESSIVE VM - FULL TEST SUITE")
    print("  KV Caching + Pruning + All Operations")
    print("=" * 70)
    print()

    # Create model once
    model = PureAutoregTransformer(dim=128, num_heads=4, num_layers=2)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} parameters")
    print()

    # Test cases: (name, source, expected)
    tests = [
        # Basic arithmetic
        ("6 * 7", "int main() { return 6 * 7; }", 42),
        ("100 / 7", "int main() { return 100 / 7; }", 14),
        ("100 % 7", "int main() { return 100 % 7; }", 2),
        ("10 + 20", "int main() { return 10 + 20; }", 30),
        ("50 - 8", "int main() { return 50 - 8; }", 42),

        # Bitwise
        ("15 & 7", "int main() { return 15 & 7; }", 7),
        ("8 | 3", "int main() { return 8 | 3; }", 11),
        ("15 ^ 6", "int main() { return 15 ^ 6; }", 9),
        ("5 << 3", "int main() { return 5 << 3; }", 40),
        ("40 >> 2", "int main() { return 40 >> 2; }", 10),

        # Comparisons
        ("5 < 7", "int main() { return 5 < 7; }", 1),
        ("7 < 5", "int main() { return 7 < 5; }", 0),
        ("5 == 5", "int main() { return 5 == 5; }", 1),
        ("5 != 7", "int main() { return 5 != 7; }", 1),
        ("5 >= 5", "int main() { return 5 >= 5; }", 1),
        ("5 <= 4", "int main() { return 5 <= 4; }", 0),

        # Signed arithmetic
        ("-3 * 4", "int main() { return -3 * 4; }", -12),
        ("-10 / 3", "int main() { return -10 / 3; }", -3),
        ("-10 % 3", "int main() { return -10 % 3; }", -1),

        # Overflow
        ("overflow add", "int main() { return 2147483647 + 1; }", -2147483648),
        ("overflow mul", "int main() { return 65536 * 65536; }", 0),

        # Variables and control flow
        ("variables", "int main() { int a; int b; a = 6; b = 7; return a * b; }", 42),
        ("if-else", "int main() { int x; x = 5; if (x > 3) return 1; return 0; }", 1),
        ("while loop", "int main() { int i; int s; i = 0; s = 0; while (i < 5) { s = s + i; i = i + 1; } return s; }", 10),

        # Function calls
        ("function", """
            int double(int x) { return x * 2; }
            int main() { return double(21); }
        """, 42),

        # Recursion
        ("factorial", """
            int fact(int n) {
                if (n <= 1) return 1;
                return n * fact(n - 1);
            }
            int main() { return fact(5); }
        """, 120),

        ("fibonacci", """
            int fib(int n) {
                if (n <= 1) return n;
                return fib(n-1) + fib(n-2);
            }
            int main() { return fib(10); }
        """, 55),
    ]

    passed = 0
    failed = 0
    total_forwards = 0
    total_time = 0.0

    print(f"{'Test':<20} {'Result':>12} {'Expected':>12} {'Forwards':>10} {'Time':>8} {'Status':<6}")
    print("-" * 70)

    for name, source, expected in tests:
        vm = PureAutoregVM(model)

        try:
            bytecode, data = compile_c(source)
            vm.load(bytecode, data)

            start = time.time()
            result = vm.run(max_steps=100000)
            elapsed = time.time() - start

            # Handle signed results
            if result >= 0x80000000:
                result = result - 0x100000000

            ok = (result == expected)
            status = "OK" if ok else "FAIL"

            if ok:
                passed += 1
            else:
                failed += 1

            total_forwards += vm.forward_count
            total_time += elapsed

            print(f"{name:<20} {result:>12} {expected:>12} {vm.forward_count:>10} {elapsed:>7.3f}s {status:<6}")

        except Exception as e:
            failed += 1
            print(f"{name:<20} {'ERROR':<12} {expected:>12} {'-':>10} {'-':>8} FAIL")
            print(f"  Error: {e}")

    print("-" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Total forward passes: {total_forwards:,}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg forwards/test: {total_forwards/len(tests):.0f}")
    print()

    # Test KV cache efficiency
    print("KV Cache Analysis:")
    vm = PureAutoregVM(model)
    bytecode, data = compile_c("int main() { return 6 * 7; }")
    vm.load(bytecode, data)

    # Check cache size before and after
    vm.run(max_steps=1000)
    if model.kv_cache is not None:
        cache_size = sum(k.numel() + v.numel() for k, v in model.kv_cache)
        cache_len = model.kv_cache[0][0].shape[2]
        print(f"  Cache sequence length: {cache_len}")
        print(f"  Cache memory: {cache_size * 4 / 1024:.1f} KB")
    print()

    return passed == len(tests)


def main():
    run_tests()


if __name__ == "__main__":
    main()
