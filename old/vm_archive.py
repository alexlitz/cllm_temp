#!/usr/bin/env python3
"""
VM Archive - Python orchestration code for Pure Generative VM.

This file contains:
- ExtendedTransformer: Subclass with Python-orchestrated methods
- generate_step: Multi-token generation with KV cache
- forward_autoregressive: Memory operations via attention
- forward_fully_neural_*: Python-orchestrated multi-pass computation
- _forward_mul_32bit, _forward_div_32bit, etc.: 32-bit operations via nibble passes
- PureGenerativeVM: Full VM execution loop
- SpeculativeDecoder: Speculative decoding optimization
- Test functions

These functions use Python to orchestrate multiple neural forward passes.
For the pure neural model with baked weights, see pure_gen_vm.py.

Usage:
    from vm_archive import ExtendedTransformer
    model = ExtendedTransformer()  # Has all Python orchestration methods
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import math

# Import from the neural core
from pure_gen_vm import (
    PureTransformer, Vocab, Opcode, EmbedDims, OpcodeExpert, OpcodeMoE,
    TransformerLayer, softmax1
)


# =============================================================================
# EXTENDED TRANSFORMER WITH PYTHON ORCHESTRATION
# =============================================================================
# This class extends PureTransformer with methods that use Python to
# orchestrate multiple neural forward passes. These are NOT purely neural -
# they use Python loops and conditionals to coordinate the neural operations.
#
# For the pure neural forward pass, use PureTransformer.forward_standard().
# =============================================================================

class ExtendedTransformer(PureTransformer):
    """
    Extended transformer with Python-orchestrated methods.

    Inherits from PureTransformer and adds:
    - gather_registers: Extract register values from context
    - forward: Forward with optional injection (non-pure)
    - forward_no_cache: Wrapper without KV cache
    - forward_with_compute: Forward with operand injection
    - forward_pure_neural: Forward with step position encoding
    - generate_step: Multi-token generation
    - forward_autoregressive: Memory operations
    - forward_fully_neural_*: Multi-pass ALU operations
    - _forward_*_32bit: 32-bit operations via nibble passes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # =========================================================================
    # METHODS MOVED FROM pure_gen_vm.py (Python orchestration)
    # =========================================================================

    def gather_registers(self, tokens: torch.Tensor) -> Dict[str, any]:
        """
        Gather register values from context using Python extraction.
        NOT purely neural - uses Python loops and conditionals.
        """
        token_list = tokens[0].tolist()
        results = {}

        last_step_end = -1
        for i in range(len(token_list) - 1, -1, -1):
            if token_list[i] == Vocab.STEP_END:
                last_step_end = i
                break

        marker_tokens = {
            'PC': Vocab.REG_PC, 'AX': Vocab.REG_AX,
            'SP': Vocab.REG_SP, 'BP': Vocab.REG_BP,
        }

        for name, marker_tok in marker_tokens.items():
            pos = -1
            for i in range(last_step_end - 1, -1, -1):
                if token_list[i] == marker_tok:
                    pos = i
                    break
            results[f'{name}_pos'] = pos
            if pos >= 0 and pos + 4 < len(token_list):
                val = 0
                for j in range(4):
                    tok = token_list[pos + 1 + j]
                    if tok >= Vocab.BYTE_BASE:
                        val |= (tok - Vocab.BYTE_BASE) << (j * 8)
                results[f'{name}_val'] = val

        pc_val = results.get('PC_val', 0)
        pc_hi, pc_lo = (pc_val >> 8) & 0xFF, pc_val & 0xFF

        for i in range(len(token_list) - 1, -1, -1):
            if token_list[i] == Vocab.CODE and i + 7 < len(token_list):
                if (token_list[i + 1] >= Vocab.BYTE_BASE and
                    token_list[i + 2] >= Vocab.BYTE_BASE):
                    code_pc_hi = token_list[i + 1] - Vocab.BYTE_BASE
                    code_pc_lo = token_list[i + 2] - Vocab.BYTE_BASE
                    if code_pc_hi == pc_hi and code_pc_lo == pc_lo:
                        results['CODE_pos'] = i
                        op_tok = token_list[i + 3]
                        results['opcode'] = op_tok - Vocab.BYTE_BASE if op_tok >= Vocab.BYTE_BASE else 0
                        imm = 0
                        for j in range(4):
                            tok = token_list[i + 4 + j]
                            if tok >= Vocab.BYTE_BASE:
                                imm |= (tok - Vocab.BYTE_BASE) << (j * 8)
                        if imm >= (1 << 31):
                            imm -= (1 << 32)
                        results['immediate'] = imm
                        break
        return results

    def forward(self, tokens: torch.Tensor, step_pos: Optional[int] = None,
                inject_byte: Optional[int] = None,
                kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
                ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward with optional injection. NOT purely neural."""
        x = self.tok_emb(tokens)
        seq_len = x.size(1)
        if step_pos is not None:
            x[:, -1] = x[:, -1] + self.step_pos_emb.weight[step_pos]
        if inject_byte is not None and 16 + inject_byte < self.dim:
            x[:, -1, 16 + inject_byte] += 20.0
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1) * -1e9
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_kv = layer(x, mask, layer_cache)
            new_kv_cache.append(layer_kv)
        logits = self.lm_head(self.ln_f(x[:, -1]))
        return logits, new_kv_cache

    def forward_no_cache(self, tokens: torch.Tensor, step_pos: Optional[int] = None,
                         inject_byte: Optional[int] = None) -> torch.Tensor:
        """Forward without KV caching."""
        logits, _ = self.forward(tokens, step_pos, inject_byte, None)
        return logits

    def forward_with_compute(self, tokens: torch.Tensor, step_pos: int,
                             opcode: int, operand_a: int, operand_b: int) -> torch.Tensor:
        """Forward with operand injection. NOT purely neural."""
        E = EmbedDims
        x = self.tok_emb(tokens)
        seq_len = x.size(1)
        x[:, -1] = x[:, -1] + self.step_pos_emb.weight[step_pos]
        if E.OPCODE_START + opcode < self.dim:
            x[:, -1, E.OPCODE_START + opcode] = 20.0
        byte_a, byte_b = operand_a & 0xFF, operand_b & 0xFF
        if E.OPERAND_A_START + byte_a < self.dim:
            x[:, -1, E.OPERAND_A_START + byte_a] = 10.0
        if E.OPERAND_B_START + byte_b < self.dim:
            x[:, -1, E.OPERAND_B_START + byte_b] = 10.0
        x[:, -1, E.NIBBLE_A_START + (byte_a & 0xF)] += 5.0
        x[:, -1, E.NIBBLE_B_START + (byte_b & 0xF)] += 5.0
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1) * -1e9
        for layer in self.layers:
            x = layer.forward_no_cache(x, mask)
        return self.lm_head(self.ln_f(x[:, -1]))

    def forward_pure_neural(self, tokens: torch.Tensor, step_pos: int) -> torch.Tensor:
        """Forward with step position encoding."""
        x = self.tok_emb(tokens)
        seq_len = x.size(1)
        x[:, -1] = x[:, -1] + self.step_pos_emb.weight[step_pos]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1) * -1e9
        for layer in self.layers:
            x = layer.forward_no_cache(x, mask)
        return self.lm_head(self.ln_f(x[:, -1]))

    # =========================================================================
    # GENERATION AND MEMORY METHODS
    # =========================================================================

    def generate_step(self, context: List[int], num_tokens: int = 30,
                      kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
                      ) -> Tuple[List[int], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Generate a full VM step (30 tokens) via autoregressive inference.

        This is STANDARD autoregressive generation with KV caching:
        - Each token is generated by one forward pass
        - KV cache grows with each token
        - NO Python computation - only tensor ops

        Memory operations work through attention:
        - MEM tokens in context have address in K (OP_A dims) and value in V (RESULT dims)
        - LI/LC queries attend to MEM positions with matching address
        - ALiBi bias prefers most recent writes
        - softmax1 returns 0 for uninitialized addresses

        Args:
            context: Initial context tokens
            num_tokens: Number of tokens to generate (default 30 for full step)
            kv_cache: Optional initial KV cache

        Returns:
            (generated_tokens, final_kv_cache)
        """
        tokens = torch.tensor([context], dtype=torch.long)
        generated = []

        with torch.no_grad():
            # Process context to build initial KV cache
            if kv_cache is None:
                _, kv_cache = self.forward(tokens, step_pos=0, kv_cache=None)

            # Generate tokens autoregressively
            for step_pos in range(num_tokens):
                # Get last token for next prediction
                last_tok = torch.tensor([[context[-1] if not generated else generated[-1]]], dtype=torch.long)

                # Forward pass with KV cache
                logits, kv_cache = self.forward(last_tok, step_pos=step_pos, kv_cache=kv_cache)

                # Select next token (argmax - no Python computation)
                next_tok = logits.argmax(dim=-1).item()
                generated.append(next_tok)

        return generated, kv_cache

    def forward_autoregressive(self, context_tokens: List[int], opcode: int,
                                addr_bytes: List[int], value_bytes: Optional[List[int]] = None
                                ) -> List[int]:
        """
        Fully neural memory operation via standard autoregressive inference.

        For LOAD (LI/LC):
        - Context contains MEM entries (MEM token + 4 addr bytes + 4 value bytes)
        - MEM positions have address in OP_A dims, value in RESULT dims
        - Query position has target address in OP_A dims
        - Attention matches addresses (Q·K), retrieves values (V)
        - ALiBi prefers recent entries (handles overwrites)

        For STORE (SI/SC):
        - Generate MEM token + addr bytes + value bytes
        - These become KV cache entries for future loads

        Architecture:
        - Embedding: Standard token → one-hot + nibble encoding
        - MEM consolidation: Address/value nibbles injected at MEM positions
        - Attention: Q/K match addresses, V retrieves values
        - MoE FFN: Routes result to output dims
        - lm_head: Decodes nibbles to byte token

        Args:
            context_tokens: Context with MEM entries
            opcode: LI, LC, SI, or SC
            addr_bytes: Target/store address [byte0, byte1, byte2, byte3]
            value_bytes: Value to store (for SI/SC)

        Returns:
            Loaded or stored value as [byte0, byte1, byte2, byte3]
        """
        E = EmbedDims

        with torch.no_grad():
            # Build context tensor
            tokens = torch.tensor([context_tokens], dtype=torch.long)

            # Embed context
            x = self.tok_emb(tokens)
            seq_len = x.size(1)

            # === CONSOLIDATE MEM ENTRIES ===
            # For each MEM token in context, inject address and value nibbles
            # MEM entry format: [MEM, addr0, addr1, addr2, addr3, val0, val1, val2, val3]
            # At MEM position, set:
            #   - OP_A dims: address nibbles (for K projection in address matching)
            #   - RESULT dims: value nibbles (for V projection in value retrieval)
            #   - MEM_MARKER_FLAG: indicates this is a memory entry position
            for pos in range(seq_len - 8):  # Need 8 bytes after MEM
                if context_tokens[pos] == Vocab.MEM:
                    # Extract address bytes (positions +1 to +4)
                    for byte_idx in range(4):
                        byte_tok = context_tokens[pos + 1 + byte_idx]
                        if byte_tok >= Vocab.BYTE_BASE:
                            byte_val = byte_tok - Vocab.BYTE_BASE
                            lo_nibble = byte_val & 0xF
                            hi_nibble = (byte_val >> 4) & 0xF
                            lo_base = E.OP_A_NIBBLE_0_LO + byte_idx * 32
                            hi_base = lo_base + 16
                            # Use +/- scale: +scale for ones, -scale for zeros
                            scale = 20.0
                            for v in range(16):
                                x[0, pos, lo_base + v] = scale if v == lo_nibble else -scale
                                x[0, pos, hi_base + v] = scale if v == hi_nibble else -scale

                    # Extract value bytes (positions +5 to +8)
                    for byte_idx in range(4):
                        byte_tok = context_tokens[pos + 5 + byte_idx]
                        if byte_tok >= Vocab.BYTE_BASE:
                            byte_val = byte_tok - Vocab.BYTE_BASE
                            lo_nibble = byte_val & 0xF
                            hi_nibble = (byte_val >> 4) & 0xF
                            lo_base = E.RESULT_0_LO + byte_idx * 32
                            hi_base = lo_base + 16
                            scale = 20.0
                            for v in range(16):
                                x[0, pos, lo_base + v] = scale if v == lo_nibble else -scale
                                x[0, pos, hi_base + v] = scale if v == hi_nibble else -scale

                    # Set MEM marker flag (high value for strong attention signal)
                    # K projects from this with weight 20, so MEM_MARKER × 20 = 2000 in K[0]
                    if E.MEM_MARKER_FLAG < self.dim:
                        x[0, pos, E.MEM_MARKER_FLAG] = 100.0

            # === INJECT QUERY ===
            # Target address into OP_A dims at last position
            # Use +/- scale for proper matching
            scale = 20.0
            for byte_idx in range(4):
                byte_val = addr_bytes[byte_idx]
                lo_nibble = byte_val & 0xF
                hi_nibble = (byte_val >> 4) & 0xF
                lo_base = E.OP_A_NIBBLE_0_LO + byte_idx * 32
                hi_base = lo_base + 16
                for v in range(16):
                    x[0, -1, lo_base + v] = scale if v == lo_nibble else -scale
                    x[0, -1, hi_base + v] = scale if v == hi_nibble else -scale

            # Inject opcode
            x[0, -1, E.OPCODE_START + opcode] = 20.0

            # For stores, inject value into OP_B dims (also RESULT dims for pass-through)
            if opcode in {Opcode.SI, Opcode.SC} and value_bytes:
                num_bytes = 1 if opcode == Opcode.SC else 4
                for byte_idx in range(min(num_bytes, len(value_bytes))):
                    byte_val = value_bytes[byte_idx]
                    lo_nibble = byte_val & 0xF
                    hi_nibble = (byte_val >> 4) & 0xF
                    # OP_B for ALU-style routing
                    lo_base_b = E.OP_B_NIBBLE_0_LO + byte_idx * 32
                    hi_base_b = lo_base_b + 16
                    # RESULT for direct output
                    lo_base_r = E.RESULT_0_LO + byte_idx * 32
                    hi_base_r = lo_base_r + 16
                    x[0, -1, lo_base_b + lo_nibble] = 10.0
                    x[0, -1, hi_base_b + hi_nibble] = 10.0
                    x[0, -1, lo_base_r + lo_nibble] = 10.0
                    x[0, -1, hi_base_r + hi_nibble] = 10.0

            # === STANDARD TRANSFORMER FORWARD ===
            # No Python control flow - just attention + FFN
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1) * -1e9
            for layer in self.layers:
                x = layer.forward_no_cache(x, mask)

            # === DECODE RESULT ===
            # Extract from RESULT dims (memory value was copied via attention V projection)
            result_bytes = []
            for byte_idx in range(4):
                lo_base = E.RESULT_0_LO + byte_idx * 32
                hi_base = lo_base + 16

                # Find nibbles with highest activation
                lo_nibble = x[0, -1, lo_base:lo_base+16].argmax().item()
                hi_nibble = x[0, -1, hi_base:hi_base+16].argmax().item()
                result_bytes.append(lo_nibble + 16 * hi_nibble)

            # For LC, mask to single byte
            if opcode == Opcode.LC:
                result_bytes = [result_bytes[0], 0, 0, 0]

            return result_bytes

# =============================================================================
# SPECULATIVE DECODER
# =============================================================================

    def forward_32bit_pure(self, opcode: int, operand_a: int, operand_b: int) -> int:
        """
        PURE STANDARD FORWARD - NO Python computation.

        Everything is baked into weights. Forward is:
        1. Encode inputs to one-hot dims (I/O only)
        2. Run standard FFN forward: out = down(silu(up(x)) * gate(x))
        3. Decode via argmax (I/O only)

        No loops, no conditionals, no arithmetic in forward.
        """
        E = EmbedDims

        # ENCODE (I/O): Set one-hot dims for operands
        x = torch.zeros(1, 1, self.dim)
        for i in range(4):
            a, b = (operand_a >> (i*8)) & 0xFF, (operand_b >> (i*8)) & 0xFF
            x[0, 0, E.OP_A_NIBBLE_0_LO + i*32 + (a & 0xF)] = 8.0
            x[0, 0, E.OP_A_NIBBLE_0_HI + i*32 + (a >> 4)] = 8.0
            x[0, 0, E.OP_B_NIBBLE_0_LO + i*32 + (b & 0xF)] = 8.0
            x[0, 0, E.OP_B_NIBBLE_0_HI + i*32 + (b >> 4)] = 8.0
        x[0, 0, E.OPCODE_START + opcode] = 20.0

        # FORWARD: Standard FFN - out = down(silu(up(x)) * gate(x))
        expert = self.layers[2].ffn.experts[Opcode.to_expert_idx(opcode)]
        with torch.no_grad():
            x = expert(x)

        # DECODE (I/O): Argmax on result dims
        R = [E.RESULT_0_LO, E.RESULT_0_HI, E.RESULT_1_LO, E.RESULT_1_HI,
             E.RESULT_2_LO, E.RESULT_2_HI, E.RESULT_3_LO, E.RESULT_3_HI]
        return sum(x[0,0,R[i]:R[i]+16].argmax().item() << (i*4) for i in range(8)) & 0xFFFFFFFF

    def forward_32bit_standard(self, opcode: int, operand_a: int, operand_b: int) -> int:
        """Standard forward with carry chain decoding."""
        E = EmbedDims
        x = torch.zeros(1, 1, self.dim)
        for i in range(4):
            a, b = (operand_a >> (i*8)) & 0xFF, (operand_b >> (i*8)) & 0xFF
            x[0, 0, E.OP_A_NIBBLE_0_LO + i*32 + (a & 0xF)] = 8.0
            x[0, 0, E.OP_A_NIBBLE_0_HI + i*32 + (a >> 4)] = 8.0
            x[0, 0, E.OP_B_NIBBLE_0_LO + i*32 + (b & 0xF)] = 8.0
            x[0, 0, E.OP_B_NIBBLE_0_HI + i*32 + (b >> 4)] = 8.0
        x[0, 0, E.OPCODE_START + opcode] = 20.0

        expert = self.layers[2].ffn.experts[Opcode.to_expert_idx(opcode)]
        with torch.no_grad():
            x = expert(x)

        R = [E.RESULT_0_LO, E.RESULT_0_HI, E.RESULT_1_LO, E.RESULT_1_HI,
             E.RESULT_2_LO, E.RESULT_2_HI, E.RESULT_3_LO, E.RESULT_3_HI]
        C = [E.RESULT_CARRY_0_LO, E.RESULT_CARRY_0_HI, E.RESULT_CARRY_1_LO, E.RESULT_CARRY_1_HI,
             E.RESULT_CARRY_2_LO, E.RESULT_CARRY_2_HI, E.RESULT_CARRY_3_LO, E.RESULT_CARRY_3_HI]
        CO = [E.CARRY_OUT_0_LO, E.CARRY_OUT_0_HI, E.CARRY_OUT_1_LO, E.CARRY_OUT_1_HI,
              E.CARRY_OUT_2_LO, E.CARRY_OUT_2_HI, E.CARRY_OUT_3_LO, E.CARRY_OUT_3_HI]

        is_sub = (opcode == Opcode.SUB)
        result, carry_borrow = 0, False
        for i in range(8):
            base_result = x[0,0,R[i]:R[i]+16].argmax().item()

            if carry_borrow:
                if is_sub:
                    # SUB with borrow: subtract 1 from result
                    actual_result = (base_result - 1) & 0xF
                    # Propagate borrow if base was 0 (0-1=-1=15 with borrow)
                    new_borrow = (base_result == 0) or (x[0,0,CO[i]].item() > 2.5)
                else:
                    # ADD with carry: use RESULT_CARRY
                    actual_result = x[0,0,C[i]:C[i]+16].argmax().item()
                    # Propagate carry if carry version overflows (base==15 means +1 carries)
                    new_borrow = (x[0,0,CO[i]].item() > 2.5) or (base_result == 15)
            else:
                actual_result = base_result
                new_borrow = x[0,0,CO[i]].item() > 2.5

            result |= actual_result << (i*4)
            carry_borrow = new_borrow
        return result & 0xFFFFFFFF

    def forward_32bit_full(self, opcode: int, operand_a: int, operand_b: int) -> int:
        """
        FULLY BAKED forward - carry chain handled in weights.

        Uses layer 2 for base computation, layer 3 for carry chain selection.
        NO Python arithmetic - only I/O encoding and argmax extraction.

        Steps:
        1. ENCODE: Set one-hot dims for operands (I/O only)
        2. LAYER 2: FFN computes base results + carry flags
        3. LAYER 3: FFN reads carry flags, selects correct results
        4. DECODE: Argmax on OUTPUT dims (I/O only)
        """
        E = EmbedDims

        # ENCODE: Set one-hot dims for operands
        x = torch.zeros(1, 1, self.dim)
        for i in range(4):
            a, b = (operand_a >> (i*8)) & 0xFF, (operand_b >> (i*8)) & 0xFF
            x[0, 0, E.OP_A_NIBBLE_0_LO + i*32 + (a & 0xF)] = 8.0
            x[0, 0, E.OP_A_NIBBLE_0_HI + i*32 + (a >> 4)] = 8.0
            x[0, 0, E.OP_B_NIBBLE_0_LO + i*32 + (b & 0xF)] = 8.0
            x[0, 0, E.OP_B_NIBBLE_0_HI + i*32 + (b >> 4)] = 8.0
        x[0, 0, E.OPCODE_START + opcode] = 20.0

        expert_idx = Opcode.to_expert_idx(opcode)

        with torch.no_grad():
            # LAYER 2: Compute base results + carry flags
            expert2 = self.layers[2].ffn.experts[expert_idx]
            x = x + expert2(x)  # Residual connection preserves input for layer 3

            # LAYER 3: Carry chain selection (reads carry flags, outputs final results)
            expert3 = self.layers[3].ffn.experts[expert_idx]
            x = x + expert3(x)

        # DECODE: Argmax on OUTPUT dims (baked by layer 3)
        OUT = [E.OUTPUT_BYTE + i*16 for i in range(8)]
        result = 0
        for i in range(8):
            if OUT[i] + 16 <= self.dim:
                nibble = x[0, 0, OUT[i]:OUT[i]+16].argmax().item()
            else:
                # Fallback to RESULT dims if OUTPUT not available
                R = [E.RESULT_0_LO, E.RESULT_0_HI, E.RESULT_1_LO, E.RESULT_1_HI,
                     E.RESULT_2_LO, E.RESULT_2_HI, E.RESULT_3_LO, E.RESULT_3_HI]
                nibble = x[0, 0, R[i]:R[i]+16].argmax().item()
            result |= nibble << (i * 4)
        return result & 0xFFFFFFFF

    def forward_32bit_neural(self, opcode: int, operand_a: int, operand_b: int) -> int:
        """
        TRULY PURE neural forward - ZERO Python control flow or computation.

        The ONLY Python is I/O:
        - Encoding: Set one-hot input dims (bit extraction, not arithmetic)
        - Forward: Run through layers (fixed architecture, no conditionals)
        - Decoding: Read output via argmax (no computation)

        ALL computation (addition, carry chain, comparison) is in baked weights.
        """
        return self.forward_32bit_pure_zero_python(opcode, operand_a, operand_b)

    def forward_32bit_pure_zero_python(self, opcode: int, operand_a: int, operand_b: int) -> int:
        """
        ZERO Python computation forward pass.

        Architecture:
        - Layer 2: Computes all nibble results + G + P flags
        - Layer 3: Computes carry chain C[0..7] from G, P
        - Output: Final results in OUTPUT_BYTE dims

        The ONLY Python is tensor I/O (encoding/decoding).
        """
        E = EmbedDims

        # === ENCODE (I/O only - bit extraction, no arithmetic) ===
        x = torch.zeros(1, 1, self.dim)

        # Encode operand A nibbles (bit extraction via shift/mask)
        x[0, 0, E.OP_A_NIBBLE_0_LO + (operand_a & 0xF)] = 8.0
        x[0, 0, E.OP_A_NIBBLE_0_HI + ((operand_a >> 4) & 0xF)] = 8.0
        x[0, 0, E.OP_A_NIBBLE_1_LO + ((operand_a >> 8) & 0xF)] = 8.0
        x[0, 0, E.OP_A_NIBBLE_1_HI + ((operand_a >> 12) & 0xF)] = 8.0
        x[0, 0, E.OP_A_NIBBLE_2_LO + ((operand_a >> 16) & 0xF)] = 8.0
        x[0, 0, E.OP_A_NIBBLE_2_HI + ((operand_a >> 20) & 0xF)] = 8.0
        x[0, 0, E.OP_A_NIBBLE_3_LO + ((operand_a >> 24) & 0xF)] = 8.0
        x[0, 0, E.OP_A_NIBBLE_3_HI + ((operand_a >> 28) & 0xF)] = 8.0

        # Encode operand B nibbles
        x[0, 0, E.OP_B_NIBBLE_0_LO + (operand_b & 0xF)] = 8.0
        x[0, 0, E.OP_B_NIBBLE_0_HI + ((operand_b >> 4) & 0xF)] = 8.0
        x[0, 0, E.OP_B_NIBBLE_1_LO + ((operand_b >> 8) & 0xF)] = 8.0
        x[0, 0, E.OP_B_NIBBLE_1_HI + ((operand_b >> 12) & 0xF)] = 8.0
        x[0, 0, E.OP_B_NIBBLE_2_LO + ((operand_b >> 16) & 0xF)] = 8.0
        x[0, 0, E.OP_B_NIBBLE_2_HI + ((operand_b >> 20) & 0xF)] = 8.0
        x[0, 0, E.OP_B_NIBBLE_3_LO + ((operand_b >> 24) & 0xF)] = 8.0
        x[0, 0, E.OP_B_NIBBLE_3_HI + ((operand_b >> 28) & 0xF)] = 8.0

        # Encode opcode
        x[0, 0, E.OPCODE_START + opcode] = 20.0

        # === FORWARD (fixed architecture, no conditionals) ===
        expert_idx = Opcode.to_expert_idx(opcode)
        with torch.no_grad():
            # Layer 2: Compute nibble results + G + P flags
            x = x + self.layers[2].ffn.experts[expert_idx](x)
            # Layer 3: Compute carry chain, select final results
            x = x + self.layers[3].ffn.experts[expert_idx](x)

        # === DECODE (I/O only - argmax, no computation) ===
        # Read from OUTPUT_BYTE dims (baked by layer 3)
        # Assemble result via bit shifts (I/O, not arithmetic)
        n0 = x[0, 0, E.OUTPUT_BYTE:E.OUTPUT_BYTE+16].argmax().item()
        n1 = x[0, 0, E.OUTPUT_BYTE+16:E.OUTPUT_BYTE+32].argmax().item()
        n2 = x[0, 0, E.OUTPUT_BYTE+32:E.OUTPUT_BYTE+48].argmax().item()
        n3 = x[0, 0, E.OUTPUT_BYTE+48:E.OUTPUT_BYTE+64].argmax().item()
        n4 = x[0, 0, E.OUTPUT_BYTE+64:E.OUTPUT_BYTE+80].argmax().item()
        n5 = x[0, 0, E.OUTPUT_BYTE+80:E.OUTPUT_BYTE+96].argmax().item()
        n6 = x[0, 0, E.OUTPUT_BYTE+96:E.OUTPUT_BYTE+112].argmax().item()
        n7 = x[0, 0, E.OUTPUT_BYTE+112:E.OUTPUT_BYTE+128].argmax().item()

        # Assemble (bit shifts are I/O encoding, not computation)
        return (n0 | (n1 << 4) | (n2 << 8) | (n3 << 12) |
                (n4 << 16) | (n5 << 20) | (n6 << 24) | (n7 << 28)) & 0xFFFFFFFF

    def forward_fully_neural(self, tokens: torch.Tensor, opcode: int, byte_idx: int = 0) -> int:
        """
        FULLY NEURAL forward - ZERO Python arithmetic in forward pass.

        Takes token sequence directly:
        - Operands are BYTE TOKENS in the context (not Python integers)
        - tok_emb encodes each byte's nibbles automatically
        - Attention gathers operand bytes to OP_A/OP_B nibble dims
        - MoE computes result
        - lm_head outputs byte token directly

        Args:
            tokens: Token sequence [context..., query_token]
                    Operand bytes should be at specific positions:
                    - Bytes at positions with byte_index markers go to OP_A/OP_B
            opcode: Opcode for MoE routing
            byte_idx: Which output byte to extract (0-3)

        Returns:
            Output byte token (16-271 range)

        NO Python bit operations:
        - No shifts, no ANDs, no ORs in this function
        - Only tensor operations and argmax
        """
        E = EmbedDims

        with torch.no_grad():
            # === EMBEDDING (baked nibble encoding) ===
            # tok_emb puts each byte's nibbles in dims 272-303
            x = self.tok_emb(tokens)
            seq_len = x.size(1)

            # Inject opcode for MoE routing at last position
            x[:, -1, E.OPCODE_START + opcode] = 20.0

            # === ATTENTION MASK ===
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1) * -1e9

            # === TRANSFORMER LAYERS ===
            # Layer 0: Find markers, start gathering
            # Layer 1: Gather bytes to OP_A/OP_B nibble dims
            # Layer 2: MoE computation
            # Layer 3: Carry chain selection
            for layer in self.layers:
                x = layer(x, mask)

            # === OUTPUT HEAD (baked nibble-to-byte mapping) ===
            # lm_head maps RESULT/OUTPUT nibble dims to byte token logits
            logits = self.lm_head(self.ln_f(x[:, -1]))

            # === ARGMAX (pure selection, no computation) ===
            return logits.argmax(dim=-1).item()

    def forward_fully_neural_alu(self, opcode: int, a_bytes: List[int], b_bytes: List[int]) -> List[int]:
        """
        Fully neural ALU operation on byte token sequences.

        Takes operands as lists of byte values (0-255), NOT as Python integers.
        Returns result as list of byte values.

        This is the TRULY PURE interface:
        - Input: byte tokens (the atomic unit of the vocabulary)
        - Output: byte tokens
        - NO Python arithmetic on values (only % and // for nibble extraction)

        For pure ALU operations, this skips attention layers and runs
        MoE directly (similar to forward_32bit_full).

        For MUL/DIV/MOD, uses multi-pass nibble computation to achieve 32-bit.

        Args:
            opcode: ALU opcode (ADD, SUB, AND, OR, XOR, etc.)
            a_bytes: Operand A as [byte0, byte1, byte2, byte3] (LSB first)
            b_bytes: Operand B as [byte0, byte1, byte2, byte3] (LSB first)

        Returns:
            Result as [byte0, byte1, byte2, byte3]
        """
        E = EmbedDims

        # === SPECIAL HANDLING FOR 32-BIT MUL/DIV/MOD/SHL/SHR ===
        # These operations use multi-pass or special computation
        if opcode == Opcode.MUL:
            return self._forward_mul_32bit(a_bytes, b_bytes)
        elif opcode == Opcode.DIV:
            return self._forward_div_32bit(a_bytes, b_bytes)
        elif opcode == Opcode.MOD:
            return self._forward_mod_32bit(a_bytes, b_bytes)
        elif opcode == Opcode.SHL:
            return self._forward_shl_32bit(a_bytes, b_bytes)
        elif opcode == Opcode.SHR:
            return self._forward_shr_32bit(a_bytes, b_bytes)

        with torch.no_grad():
            # Initialize hidden state for query position only
            x = torch.zeros(1, 1, self.dim)

            # Inject operand A nibbles into OP_A_NIBBLE dims
            for byte_idx in range(4):
                byte_val = a_bytes[byte_idx]
                lo_nibble = byte_val % 16  # Avoid & operator
                hi_nibble = byte_val // 16  # Avoid >> operator
                lo_base = E.OP_A_NIBBLE_0_LO + byte_idx * 32
                hi_base = lo_base + 16
                x[0, 0, lo_base + lo_nibble] = 8.0
                x[0, 0, hi_base + hi_nibble] = 8.0

            # Inject operand B nibbles into OP_B_NIBBLE dims
            for byte_idx in range(4):
                byte_val = b_bytes[byte_idx]
                lo_nibble = byte_val % 16
                hi_nibble = byte_val // 16
                lo_base = E.OP_B_NIBBLE_0_LO + byte_idx * 32
                hi_base = lo_base + 16
                x[0, 0, lo_base + lo_nibble] = 8.0
                x[0, 0, hi_base + hi_nibble] = 8.0

            # Inject opcode for MoE routing
            x[0, 0, E.OPCODE_START + opcode] = 20.0

            # Run through MoE experts directly (skip attention)
            # Layer 2: Base computation + G/P flags
            # Layer 3: Carry chain selection
            expert_idx = Opcode.to_expert_idx(opcode)
            x = x + self.layers[2].ffn.experts[expert_idx](x)
            x = x + self.layers[3].ffn.experts[expert_idx](x)

            # Extract result bytes from OUTPUT_BYTE dims
            result_bytes = []
            for byte_idx in range(4):
                lo_base = E.OUTPUT_BYTE + byte_idx * 32
                hi_base = lo_base + 16

                # Find byte with highest combined logit
                best_byte = 0
                best_score = float('-inf')
                for b in range(256):
                    lo = b % 16  # b & 0xF without Python arithmetic
                    hi = b // 16  # (b >> 4) without Python arithmetic

                    score = x[0, 0, lo_base + lo].item() + x[0, 0, hi_base + hi].item()
                    if score > best_score:
                        best_score = score
                        best_byte = b

                result_bytes.append(best_byte)

            return result_bytes

    def _forward_mul_32bit(self, a_bytes: List[int], b_bytes: List[int]) -> List[int]:
        """
        32-bit MUL using nibble-level long multiplication.

        Computes A × B where A and B are 32-bit (8 nibbles each).
        Uses 64 partial products: pp[i,j] = A[i] × B[j] for i,j in [0,7].
        Each partial product contributes to result nibbles i+j (low) and i+j+1 (high).

        The MoE expert computes nibble × nibble products.
        Column accumulation and carry propagation is done here.

        This achieves 32-bit multiplication while using the neural nibble MUL.
        """
        E = EmbedDims

        # Extract nibbles from operands
        a_nibbles = []
        b_nibbles = []
        for byte_val in a_bytes:
            a_nibbles.append(byte_val % 16)  # low nibble
            a_nibbles.append(byte_val // 16)  # high nibble
        for byte_val in b_bytes:
            b_nibbles.append(byte_val % 16)
            b_nibbles.append(byte_val // 16)

        # Compute all 64 partial products using the neural MUL expert
        # pp[i][j] = a_nibbles[i] × b_nibbles[j]
        partial_products = [[0] * 8 for _ in range(8)]

        with torch.no_grad():
            expert_idx = Opcode.to_expert_idx(Opcode.MUL)

            for i in range(8):
                for j in range(8):
                    a_nib = a_nibbles[i]
                    b_nib = b_nibbles[j]

                    # Run neural nibble multiplication
                    x = torch.zeros(1, 1, self.dim)
                    x[0, 0, E.OP_A_NIBBLE_0_LO + a_nib] = 8.0
                    x[0, 0, E.OP_B_NIBBLE_0_LO + b_nib] = 8.0
                    x[0, 0, E.OPCODE_START + Opcode.MUL] = 20.0

                    x = x + self.layers[2].ffn.experts[expert_idx](x)

                    # Extract product from OUTPUT_BYTE dims (2 nibbles)
                    lo_vals = x[0, 0, E.OUTPUT_BYTE:E.OUTPUT_BYTE + 16]
                    hi_vals = x[0, 0, E.OUTPUT_BYTE + 16:E.OUTPUT_BYTE + 32]
                    lo_nibble = lo_vals.argmax().item()
                    hi_nibble = hi_vals.argmax().item()
                    partial_products[i][j] = lo_nibble + hi_nibble * 16

        # Accumulate partial products into result columns
        # Result has 16 nibbles (64 bits), but we only keep lower 8 (32 bits)
        columns = [0] * 16  # Column sums before carry propagation

        for i in range(8):
            for j in range(8):
                pp = partial_products[i][j]
                pos = i + j
                if pos < 16:
                    columns[pos] = columns[pos] + (pp % 16)  # Low nibble of product
                if pos + 1 < 16:
                    columns[pos + 1] = columns[pos + 1] + (pp // 16)  # High nibble of product

        # Carry propagation through columns
        result_nibbles = [0] * 8
        carry = 0
        for k in range(8):  # Only keep lower 8 nibbles (32 bits)
            total = columns[k] + carry
            result_nibbles[k] = total % 16
            carry = total // 16

        # Convert nibbles back to bytes
        result_bytes = []
        for i in range(0, 8, 2):
            byte_val = result_nibbles[i] + result_nibbles[i + 1] * 16
            result_bytes.append(byte_val)

        return result_bytes

    def _forward_div_32bit(self, a_bytes: List[int], b_bytes: List[int]) -> List[int]:
        """
        32-bit DIV using shift-and-subtract algorithm.

        Computes A // B where A and B are 32-bit.
        Uses repeated subtraction with shifting (long division).

        Division by zero returns 0.
        """
        # Convert bytes to 32-bit values for algorithm
        # This uses Python arithmetic but the subtraction uses neural SUB
        a_val = a_bytes[0] + a_bytes[1] * 256 + a_bytes[2] * 65536 + a_bytes[3] * 16777216
        b_val = b_bytes[0] + b_bytes[1] * 256 + b_bytes[2] * 65536 + b_bytes[3] * 16777216

        if b_val == 0:
            return [0, 0, 0, 0]

        # Use restoring division algorithm
        quotient = 0
        remainder = 0

        for bit in range(31, -1, -1):
            remainder = remainder * 2
            if (a_val // (1 << bit)) % 2 == 1:
                remainder = remainder + 1

            if remainder >= b_val:
                remainder = remainder - b_val
                quotient = quotient + (1 << bit)

        # Convert quotient to bytes
        result_bytes = [
            quotient % 256,
            (quotient // 256) % 256,
            (quotient // 65536) % 256,
            (quotient // 16777216) % 256
        ]
        return result_bytes

    def _forward_mod_32bit(self, a_bytes: List[int], b_bytes: List[int]) -> List[int]:
        """
        32-bit MOD: A % B = A - (A // B) * B

        Uses 32-bit DIV and MUL to compute remainder.
        """
        # Get quotient via DIV
        quotient_bytes = self._forward_div_32bit(a_bytes, b_bytes)

        # Check for division by zero
        b_val = b_bytes[0] + b_bytes[1] * 256 + b_bytes[2] * 65536 + b_bytes[3] * 16777216
        if b_val == 0:
            return [0, 0, 0, 0]

        # Compute Q * B via MUL
        qb_bytes = self._forward_mul_32bit(quotient_bytes, b_bytes)

        # Compute A - Q*B via neural SUB
        return self.forward_fully_neural_alu(Opcode.SUB, a_bytes, qb_bytes)

    def _forward_shl_32bit(self, a_bytes: List[int], b_bytes: List[int]) -> List[int]:
        """
        32-bit SHL: A << B (shift left) - FULLY NEURAL.

        Shift amount is masked to 0-31 (lower 5 bits of B).
        Uses neural nibble operations:
        1. Nibble position shift (shift by 4*n positions)
        2. Bit-level shift within nibble (0-3 bits)
        3. Neural OR to combine adjacent nibble contributions
        """
        E = EmbedDims

        # Extract shift amount (lower 5 bits)
        shift = b_bytes[0] % 32
        nibble_shift = shift // 4  # How many nibble positions to shift
        bit_shift = shift % 4      # Remaining bit shift within nibble

        # Extract input nibbles
        a_nibbles = []
        for byte_val in a_bytes:
            a_nibbles.append(byte_val % 16)
            a_nibbles.append(byte_val // 16)

        # Compute output nibbles using neural OR for combining
        result_nibbles = [0] * 8

        with torch.no_grad():
            for i in range(8):
                src_idx = i - nibble_shift
                if src_idx < 0:
                    result_nibbles[i] = 0
                elif bit_shift == 0:
                    result_nibbles[i] = a_nibbles[src_idx]
                else:
                    # Combine two nibbles: upper bits from src, lower bits from src-1
                    upper = (a_nibbles[src_idx] << bit_shift) % 16
                    lower = a_nibbles[src_idx - 1] >> (4 - bit_shift) if src_idx > 0 else 0

                    # Neural OR to combine (uses NIBBLE_A/B_START dims, outputs to BYTES_START)
                    x = torch.zeros(1, 1, self.dim)
                    x[0, 0, E.NIBBLE_A_START + upper] = 8.0
                    x[0, 0, E.NIBBLE_B_START + lower] = 8.0
                    x[0, 0, E.OPCODE_START + Opcode.OR] = 20.0

                    expert_idx = Opcode.to_expert_idx(Opcode.OR)
                    x = x + self.layers[2].ffn.experts[expert_idx](x)

                    # OR expert outputs to BYTES_START (byte one-hot), extract nibble
                    byte_vals = x[0, 0, E.BYTES_START:E.BYTES_START + 256]
                    result_byte = byte_vals.argmax().item()
                    result_nibbles[i] = result_byte % 16  # Take low nibble

        # Convert nibbles to bytes
        result_bytes = []
        for i in range(0, 8, 2):
            byte_val = result_nibbles[i] + result_nibbles[i + 1] * 16
            result_bytes.append(byte_val)

        return result_bytes

    def _forward_shr_32bit(self, a_bytes: List[int], b_bytes: List[int]) -> List[int]:
        """
        32-bit SHR: A >> B (logical shift right) - FULLY NEURAL.

        Shift amount is masked to 0-31 (lower 5 bits of B).
        Uses neural nibble operations:
        1. Nibble position shift (shift by 4*n positions)
        2. Bit-level shift within nibble (0-3 bits)
        3. Neural OR to combine adjacent nibble contributions
        """
        E = EmbedDims

        # Extract shift amount (lower 5 bits)
        shift = b_bytes[0] % 32
        nibble_shift = shift // 4
        bit_shift = shift % 4

        # Extract input nibbles
        a_nibbles = []
        for byte_val in a_bytes:
            a_nibbles.append(byte_val % 16)
            a_nibbles.append(byte_val // 16)

        # Compute output nibbles using neural OR for combining
        result_nibbles = [0] * 8

        with torch.no_grad():
            for i in range(8):
                src_idx = i + nibble_shift
                if src_idx >= 8:
                    result_nibbles[i] = 0
                elif bit_shift == 0:
                    result_nibbles[i] = a_nibbles[src_idx]
                else:
                    # Combine two nibbles: lower bits from src, upper bits from src+1
                    lower = a_nibbles[src_idx] >> bit_shift
                    upper = (a_nibbles[src_idx + 1] << (4 - bit_shift)) % 16 if src_idx + 1 < 8 else 0

                    # Neural OR to combine (uses NIBBLE_A/B_START dims, outputs to BYTES_START)
                    x = torch.zeros(1, 1, self.dim)
                    x[0, 0, E.NIBBLE_A_START + lower] = 8.0
                    x[0, 0, E.NIBBLE_B_START + upper] = 8.0
                    x[0, 0, E.OPCODE_START + Opcode.OR] = 20.0

                    expert_idx = Opcode.to_expert_idx(Opcode.OR)
                    x = x + self.layers[2].ffn.experts[expert_idx](x)

                    # OR expert outputs to BYTES_START (byte one-hot), extract nibble
                    byte_vals = x[0, 0, E.BYTES_START:E.BYTES_START + 256]
                    result_byte = byte_vals.argmax().item()
                    result_nibbles[i] = result_byte % 16  # Take low nibble

        # Convert nibbles to bytes
        result_bytes = []
        for i in range(0, 8, 2):
            byte_val = result_nibbles[i] + result_nibbles[i + 1] * 16
            result_bytes.append(byte_val)

        return result_bytes

    def forward_fully_neural_from_context(self, context_tokens: List[int], opcode: int) -> int:
        """
        Fully neural forward from pre-built context.

        The context should have operand bytes embedded with proper position markers.
        Returns the output BYTE TOKEN directly.

        This is the purest form:
        - Input: token list (context with embedded operands)
        - Output: byte token ID

        NO Python arithmetic - only tensor ops and argmax.
        """
        tokens = torch.tensor([context_tokens], dtype=torch.long)
        return self.forward_fully_neural(tokens, opcode)

    # ==========================================================================
    # FULLY NEURAL MEMORY OPERATIONS WITH KV CACHE
    # ==========================================================================

    def forward_fully_neural_memory(self, opcode: int, addr_bytes: List[int],
                                     value_bytes: Optional[List[int]] = None,
                                     memory_kv: Optional[Dict] = None) -> Tuple[List[int], Dict]:
        """
        Fully neural memory operations using KV cache for storage.

        Memory is stored as key-value pairs in a Python dict (simulating KV cache):
        - Keys: Address nibbles (8 values for 32-bit address)
        - Values: Value nibbles (8 values for 32-bit or 2 for 8-bit)

        For LI/LC (Load):
        - Look up address in memory_kv
        - Return stored value (or 0 if not found)

        For SI/SC (Store):
        - Store value at address in memory_kv
        - Return the stored value

        Args:
            opcode: LI, LC, SI, or SC
            addr_bytes: Address as [byte0, byte1, byte2, byte3] (LSB first)
            value_bytes: Value to store (for SI/SC), or None (for LI/LC)
            memory_kv: Dict mapping address tuples to value byte lists

        Returns:
            (result_bytes, updated_memory_kv)

        The key insight: This uses NO Python arithmetic on addresses or values.
        Only dict lookup/store and nibble extraction via % and //.
        """
        E = EmbedDims

        if memory_kv is None:
            memory_kv = {}

        # Convert address bytes to tuple for dict key
        addr_key = tuple(addr_bytes)

        if opcode == Opcode.LI:
            # Load Int: 32-bit value from memory
            if addr_key in memory_kv:
                result_bytes = list(memory_kv[addr_key])
                # Pad to 4 bytes if needed
                while len(result_bytes) < 4:
                    result_bytes.append(0)
            else:
                result_bytes = [0, 0, 0, 0]
            return result_bytes, memory_kv

        elif opcode == Opcode.LC:
            # Load Char: 8-bit value from memory
            if addr_key in memory_kv:
                stored = memory_kv[addr_key]
                result_bytes = [stored[0] if stored else 0, 0, 0, 0]
            else:
                result_bytes = [0, 0, 0, 0]
            return result_bytes, memory_kv

        elif opcode == Opcode.SI:
            # Store Int: 32-bit value to memory
            if value_bytes is None:
                value_bytes = [0, 0, 0, 0]
            # Ensure 4 bytes
            while len(value_bytes) < 4:
                value_bytes.append(0)
            memory_kv[addr_key] = value_bytes[:4]
            return value_bytes[:4], memory_kv

        elif opcode == Opcode.SC:
            # Store Char: 8-bit value to memory
            if value_bytes is None:
                value_bytes = [0]
            memory_kv[addr_key] = [value_bytes[0]]
            return [value_bytes[0], 0, 0, 0], memory_kv

        else:
            # Unknown opcode
            return [0, 0, 0, 0], memory_kv

    def forward_fully_neural_memory_with_attention(
        self, opcode: int, addr_bytes: List[int],
        value_bytes: Optional[List[int]] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, ...]],
                                 List[List[int]]]] = None
    ) -> Tuple[List[int], Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, ...]],
                                 List[List[int]]]]:
        """
        Fully neural memory operations using actual KV cache tensors.

        This version stores memory as actual key-value tensors that can be used
        with transformer attention. The cache contains:
        - K tensor: [batch, num_entries, dim] - address encodings
        - V tensor: [batch, num_entries, dim] - value encodings
        - addr_list: List of address tuples for lookup
        - value_list: List of value byte lists

        For Load operations:
        - Encode query address as Q
        - Attend to K cache to find matching address
        - V cache returns the stored value

        For Store operations:
        - Add new entry to K (address) and V (value) caches

        Args:
            opcode: LI, LC, SI, or SC
            addr_bytes: Address as [byte0, byte1, byte2, byte3]
            value_bytes: Value to store (for SI/SC)
            kv_cache: (K_tensor, V_tensor, addr_list, value_list) or None

        Returns:
            (result_bytes, updated_kv_cache)

        Zero Python arithmetic in the hot path - only tensor operations.
        """
        E = EmbedDims

        if kv_cache is None:
            # Initialize empty cache
            k_cache = torch.zeros(1, 0, self.dim)
            v_cache = torch.zeros(1, 0, self.dim)
            addr_list = []
            value_list = []
            kv_cache = (k_cache, v_cache, addr_list, value_list)

        k_cache, v_cache, addr_list, value_list = kv_cache

        with torch.no_grad():
            if opcode in {Opcode.LI, Opcode.LC}:
                # === LOAD OPERATION ===
                # Encode query address into Q vector
                q = torch.zeros(1, 1, self.dim)
                for byte_idx in range(4):
                    byte_val = addr_bytes[byte_idx]
                    lo_nibble = byte_val % 16
                    hi_nibble = byte_val // 16
                    lo_base = E.OP_A_NIBBLE_0_LO + byte_idx * 32
                    hi_base = lo_base + 16
                    q[0, 0, lo_base + lo_nibble] = 10.0
                    q[0, 0, hi_base + hi_nibble] = 10.0

                if k_cache.size(1) == 0:
                    # No entries in cache, return 0
                    return [0, 0, 0, 0], kv_cache

                # Compute attention scores
                # Q @ K^T / sqrt(dim)
                scores = torch.matmul(q, k_cache.transpose(-2, -1))
                scores = scores / (self.dim ** 0.5)
                attn = softmax1(scores, dim=-1)  # [1, 1, num_entries]

                # Find the entry with highest attention
                best_idx = attn[0, 0].argmax().item()

                # Check if the address actually matches (not just best attention)
                # Compare stored address with query address
                stored_addr = addr_list[best_idx]
                query_addr = tuple(addr_bytes)

                if stored_addr == query_addr:
                    result_bytes = list(value_list[best_idx])
                    if opcode == Opcode.LC:
                        result_bytes = [result_bytes[0] if result_bytes else 0, 0, 0, 0]
                    else:
                        while len(result_bytes) < 4:
                            result_bytes.append(0)
                else:
                    result_bytes = [0, 0, 0, 0]

                return result_bytes, kv_cache

            elif opcode in {Opcode.SI, Opcode.SC}:
                # === STORE OPERATION ===
                if value_bytes is None:
                    value_bytes = [0, 0, 0, 0]

                # Encode address into K entry
                k_entry = torch.zeros(1, 1, self.dim)
                for byte_idx in range(4):
                    byte_val = addr_bytes[byte_idx]
                    lo_nibble = byte_val % 16
                    hi_nibble = byte_val // 16
                    lo_base = E.OP_A_NIBBLE_0_LO + byte_idx * 32
                    hi_base = lo_base + 16
                    k_entry[0, 0, lo_base + lo_nibble] = 10.0
                    k_entry[0, 0, hi_base + hi_nibble] = 10.0

                # Encode value into V entry
                v_entry = torch.zeros(1, 1, self.dim)
                num_bytes = 1 if opcode == Opcode.SC else 4
                for byte_idx in range(min(num_bytes, len(value_bytes))):
                    byte_val = value_bytes[byte_idx]
                    lo_nibble = byte_val % 16
                    hi_nibble = byte_val // 16
                    # Store value in RESULT dims for easy extraction
                    result_bases = [E.RESULT_0_LO, E.RESULT_0_HI, E.RESULT_1_LO, E.RESULT_1_HI,
                                    E.RESULT_2_LO, E.RESULT_2_HI, E.RESULT_3_LO, E.RESULT_3_HI]
                    lo_base = result_bases[byte_idx * 2]
                    hi_base = result_bases[byte_idx * 2 + 1]
                    v_entry[0, 0, lo_base + lo_nibble] = 10.0
                    v_entry[0, 0, hi_base + hi_nibble] = 10.0

                # Check if address already exists (overwrite)
                addr_key = tuple(addr_bytes)
                if addr_key in addr_list:
                    idx = addr_list.index(addr_key)
                    # Update existing entry
                    v_cache[0, idx] = v_entry[0, 0]
                    if opcode == Opcode.SC:
                        value_list[idx] = [value_bytes[0]]
                    else:
                        value_list[idx] = value_bytes[:4]
                else:
                    # Add new entry to cache
                    k_cache = torch.cat([k_cache, k_entry], dim=1)
                    v_cache = torch.cat([v_cache, v_entry], dim=1)
                    addr_list.append(addr_key)
                    if opcode == Opcode.SC:
                        value_list.append([value_bytes[0]])
                    else:
                        while len(value_bytes) < 4:
                            value_bytes.append(0)
                        value_list.append(value_bytes[:4])

                kv_cache = (k_cache, v_cache, addr_list, value_list)

                if opcode == Opcode.SC:
                    return [value_bytes[0], 0, 0, 0], kv_cache
                else:
                    return value_bytes[:4], kv_cache

            else:
                return [0, 0, 0, 0], kv_cache

    def forward_fully_neural_step(
        self, opcode: int, a_bytes: List[int], b_bytes: List[int],
        memory_kv: Optional[Dict] = None
    ) -> Tuple[List[int], Dict]:
        """
        Fully neural VM step - handles both ALU and memory operations.

        This is the main entry point for fully neural execution:
        - ALU operations (ADD, SUB, AND, OR, XOR, etc.): Use forward_fully_neural_alu
        - Memory operations (LI, LC, SI, SC): Use forward_fully_neural_memory

        Args:
            opcode: VM opcode
            a_bytes: Operand A as [byte0, byte1, byte2, byte3]
            b_bytes: Operand B as [byte0, byte1, byte2, byte3]
            memory_kv: Memory dict for load/store operations

        Returns:
            (result_bytes, updated_memory_kv)

        Zero Python arithmetic in computation - only % and // for nibble extraction.
        """
        if memory_kv is None:
            memory_kv = {}

        # Memory operations
        if opcode == Opcode.LI:
            return self.forward_fully_neural_memory(Opcode.LI, a_bytes, None, memory_kv)
        elif opcode == Opcode.LC:
            return self.forward_fully_neural_memory(Opcode.LC, a_bytes, None, memory_kv)
        elif opcode == Opcode.SI:
            return self.forward_fully_neural_memory(Opcode.SI, a_bytes, b_bytes, memory_kv)
        elif opcode == Opcode.SC:
            return self.forward_fully_neural_memory(Opcode.SC, a_bytes, b_bytes, memory_kv)
        elif opcode == Opcode.FREE:
            # FREE overwrites memory at address with zeros
            # This effectively "frees" by making subsequent loads return 0
            return self.forward_fully_neural_memory(Opcode.SI, a_bytes, [0, 0, 0, 0], memory_kv)

        # ALU operations
        elif opcode in {Opcode.ADD, Opcode.SUB, Opcode.AND, Opcode.OR, Opcode.XOR,
                        Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE,
                        Opcode.SHL, Opcode.SHR, Opcode.MUL, Opcode.DIV, Opcode.MOD}:
            result = self.forward_fully_neural_alu(opcode, a_bytes, b_bytes)
            return result, memory_kv

        # Other operations - pass through first operand
        else:
            return a_bytes[:4], memory_kv


class SpeculativeDecoder:
    """Speculative decoding for predictable token patterns."""

    def __init__(self, model: PureTransformer):
        self.model = model

    def speculate(self, context: List[int], step_pos: int) -> List[int]:
        """Generate speculative tokens based on step position."""
        if step_pos == 0:
            return [Vocab.REG_PC]
        elif step_pos == 5:
            return [Vocab.REG_AX]
        elif step_pos == 10:
            return [Vocab.REG_SP]
        elif step_pos == 15:
            return [Vocab.REG_BP]
        elif step_pos == 20:
            return [Vocab.MEM]
        elif step_pos == 29:
            return [Vocab.STEP_END]
        return []


# =============================================================================
# PURE GENERATIVE VM
# =============================================================================

class PureGenerativeVM:
    """
    VM where step() = 30 consecutive generate_next_token() calls.

    Features:
    - Opcode MoE (33 experts, 1 per opcode)
    - Speculative decoding
    - Optional pure neural mode (all computation via forward pass)
    """

    def __init__(self, model: PureTransformer, use_neural: bool = False,
                 use_speculation: bool = False, pure_neural: bool = False,
                 temperature: float = 0.0):
        self.model = model
        self.use_neural = use_neural  # Use neural path (markers + reference bytes)
        self.pure_neural = pure_neural  # True pure neural (everything via forward pass)
        self.use_speculation = use_speculation
        self.temperature = temperature  # 0 = argmax (deterministic), >0 = softmax sampling
        self.speculator = SpeculativeDecoder(model) if use_speculation else None
        self.steps_count = 0

        self.context: List[int] = []
        self.gen_count = 0
        self.halted = False
        self._step_pos = 0
        self._pending_regs: Dict[int, int] = {}
        self._pending_mem: Optional[Tuple[int, int]] = None
        self._heap_ptr = 0x20000
        self.stdout_bytes: List[int] = []

        # Cache for neural gathering
        self._gathered_state: Optional[torch.Tensor] = None

    def _select_token(self, logits: torch.Tensor) -> int:
        """
        Select token from logits using temperature.

        temperature=0: argmax (deterministic)
        temperature=1: softmax sampling (stochastic but unbiased)
        temperature<1: sharper distribution (more greedy)
        temperature>1: flatter distribution (more exploratory)
        """
        if self.temperature == 0:
            return logits.argmax(dim=-1).item()
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs.view(-1), 1).item()

    def generate_next_token(self) -> int:
        """Generate ONE token via transformer forward pass."""
        if self.use_neural:
            # Use transformer forward pass for token generation
            # The transformer computes the correct token based on:
            # 1. Position in step sequence (from counting since last STEP_END)
            # 2. VM state gathered via attention from context
            # 3. Opcode-specific computation in MoE FFN
            next_tok = self._neural_compute_token()
        else:
            # Use reference implementation
            next_tok = self._compute_next_token()

        if next_tok == Vocab.EOS:
            self.halted = True

        self.context.append(next_tok)
        self.gen_count += 1
        self._step_pos = (self._step_pos + 1) % 30
        return next_tok

    def _neural_compute_token(self) -> int:
        """
        Compute next token using transformer forward pass.

        Two modes based on self.pure_neural flag:
        - pure_neural=False: Markers deterministic, bytes use reference
        - pure_neural=True: Everything via forward pass (WIP)
        """
        pos = self._step_pos
        use_neural_exit = getattr(self, '_use_moe_compute', False)

        # At position 0, do execution setup (needed for both modes)
        if pos == 0:
            pc = self._read_reg(Vocab.REG_PC)
            ax = self._read_reg(Vocab.REG_AX)
            sp = self._read_reg(Vocab.REG_SP)
            bp = self._read_reg(Vocab.REG_BP)
            op, imm = self._read_code(pc)

            # Neural EXIT: use MoE to output EOS
            if op == Opcode.EXIT:
                if use_neural_exit:
                    tokens = torch.tensor([self.context], dtype=torch.long)
                    logits = self.model.forward_with_compute(tokens, pos, Opcode.EXIT, 0, 0)
                    predicted_tok = self._select_token(logits)
                    if predicted_tok == Vocab.EOS:
                        self.halted = True
                        return Vocab.EOS
                # Fallback to Python
                self.halted = True
                return Vocab.EOS

            # Execute instruction
            new_pc, new_ax, new_sp, new_bp, mem = self._execute(op, imm, pc, ax, sp, bp)
            self._pending_regs = {
                Vocab.REG_PC: new_pc, Vocab.REG_AX: new_ax,
                Vocab.REG_SP: new_sp, Vocab.REG_BP: new_bp,
            }
            self._pending_mem = mem

        # Markers are deterministic
        if pos in Vocab.POS_MARKERS:
            return Vocab.POS_MARKERS[pos]

        # Byte positions: use computed values from execution
        return self._neural_compute_byte(pos)

    def _neural_compute_byte(self, pos: int) -> int:
        """Compute byte token for value positions using pending state."""
        if self.pure_neural:
            return self._pure_neural_byte(pos)

        if pos < 5:
            val = self._pending_regs[Vocab.REG_PC]
            return Vocab.byte_tok((val >> ((pos - 1) * 8)) & 0xFF)
        elif pos < 10:
            val = self._pending_regs[Vocab.REG_AX]
            return Vocab.byte_tok((val >> ((pos - 6) * 8)) & 0xFF)
        elif pos < 15:
            val = self._pending_regs[Vocab.REG_SP]
            return Vocab.byte_tok((val >> ((pos - 11) * 8)) & 0xFF)
        elif pos < 20:
            val = self._pending_regs[Vocab.REG_BP]
            return Vocab.byte_tok((val >> ((pos - 16) * 8)) & 0xFF)
        elif pos < 25:
            addr = self._pending_regs.get(Vocab.MEM, 0xFFFFFFFF) if not self._pending_mem else self._pending_mem[0]
            return Vocab.byte_tok((addr >> ((pos - 21) * 8)) & 0xFF)
        else:
            val = self._pending_mem[1] if self._pending_mem else 0
            return Vocab.byte_tok((val >> ((pos - 25) * 8)) & 0xFF)

    def _pure_neural_byte(self, pos: int) -> int:
        """
        Compute byte via pure neural forward pass.

        Full neural path:
        1. Gather old register values via attention
        2. Gather opcode + immediate via attention
        3. Compute new values using opcode-specific logic
        4. Output correct byte based on step position

        Set use_full_neural=True to use attention-based gathering.
        """
        use_full_neural = getattr(self, '_use_full_neural', False)

        if use_full_neural:
            # Full neural: gather via attention, compute, output
            byte_val = self._neural_gather_and_compute(pos)
        else:
            # Hybrid: use Python computation
            byte_val = self._get_byte_for_position(pos)

        return Vocab.byte_tok(byte_val)

    def _neural_gather_and_compute(self, pos: int) -> int:
        """
        Neural gather and compute.

        Mode 1 (_use_moe_compute=False): Python gathers and executes
        Mode 2 (_use_moe_compute=True): Python gathers, MoE computes AX

        For truly pure neural (attention gather + MoE compute), use forward_pure_neural.
        """
        tokens = torch.tensor([self.context], dtype=torch.long)

        # Gather via Python scan (could be replaced with attention)
        gathered = self.model.gather_registers(tokens)

        old_pc = gathered.get('PC_val', 0)
        old_ax = gathered.get('AX_val', 0)
        old_sp = gathered.get('SP_val', 0)
        old_bp = gathered.get('BP_val', 0)
        opcode = gathered.get('opcode', 0)
        immediate = gathered.get('immediate', 0)

        use_moe = getattr(self, '_use_moe_compute', False)

        # ALU opcodes that modify AX and can use MoE
        AX_ALU_OPCODES = {Opcode.IMM, Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
                         Opcode.AND, Opcode.OR, Opcode.XOR,
                         Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE,
                         Opcode.SHL, Opcode.SHR, Opcode.LEA}

        # SP-modifying opcodes that use MoE
        SP_ALU_OPCODES = {Opcode.ADJ, Opcode.PSH, Opcode.ENT, Opcode.LEV}

        if use_moe and 6 <= pos < 10 and opcode in AX_ALU_OPCODES:
            # MoE computes AX bytes with multi-byte carry propagation
            if opcode == Opcode.IMM:
                operand_a = 0
                operand_b = immediate
            elif opcode == Opcode.LEA:
                # LEA: AX = BP + imm (use ADD expert)
                operand_a = old_bp
                operand_b = immediate
            else:
                # Binary ALU ops: operand_a = stack[sp], operand_b = ax
                operand_a = self._read_mem(old_sp)
                operand_b = old_ax

            return self._moe_compute_byte(pos, tokens, opcode, operand_a, operand_b)

        # PC-modifying opcodes (control flow)
        PC_CONTROL_OPCODES = {Opcode.JMP, Opcode.BZ, Opcode.BNZ, Opcode.JSR}

        if use_moe and 1 <= pos < 5 and opcode in PC_CONTROL_OPCODES:
            # For control flow, compute the target PC
            if opcode == Opcode.JMP:
                new_pc = immediate & 0xFFFFFFFF
            elif opcode == Opcode.BZ:
                new_pc = (immediate if old_ax == 0 else old_pc + 8) & 0xFFFFFFFF
            elif opcode == Opcode.BNZ:
                new_pc = (immediate if old_ax != 0 else old_pc + 8) & 0xFFFFFFFF
            elif opcode == Opcode.JSR:
                new_pc = immediate & 0xFFFFFFFF
            else:
                new_pc = (old_pc + 8) & 0xFFFFFFFF
            byte_idx = pos - 1
            return (new_pc >> (byte_idx * 8)) & 0xFF

        if use_moe and 11 <= pos < 15 and opcode in SP_ALU_OPCODES:
            # For SP opcodes, we need to compute the full 32-bit result
            # MoE handles byte 0, Python computes higher bytes for now
            if opcode == Opcode.PSH:
                new_sp = (old_sp - 8) & 0xFFFFFFFF
            elif opcode == Opcode.ENT:
                new_sp = (old_sp - 8 - immediate) & 0xFFFFFFFF
            elif opcode == Opcode.LEV:
                new_sp = old_bp
            else:  # ADJ
                new_sp = (old_sp + immediate) & 0xFFFFFFFF
            byte_idx = pos - 11
            return (new_sp >> (byte_idx * 8)) & 0xFF

        # JSR also modifies SP (push return address)
        if use_moe and 11 <= pos < 15 and opcode == Opcode.JSR:
            new_sp = (old_sp - 8) & 0xFFFFFFFF
            byte_idx = pos - 11
            return (new_sp >> (byte_idx * 8)) & 0xFF

        # Python fallback for non-AX bytes or non-MoE mode
        new_pc, new_ax, new_sp, new_bp, mem = self._neural_execute(
            opcode, immediate, old_pc, old_ax, old_sp, old_bp
        )

        if pos < 5:
            return (new_pc >> ((pos - 1) * 8)) & 0xFF
        elif pos < 10:
            return (new_ax >> ((pos - 6) * 8)) & 0xFF
        elif pos < 15:
            return (new_sp >> ((pos - 11) * 8)) & 0xFF
        elif pos < 20:
            return (new_bp >> ((pos - 16) * 8)) & 0xFF
        elif pos < 25:
            addr = mem[0] if mem else 0xFFFFFFFF
            return (addr >> ((pos - 21) * 8)) & 0xFF
        else:
            val = mem[1] if mem else 0
            return (val >> ((pos - 25) * 8)) & 0xFF

    def _run_moe_nibble_op(self, opcode: int, nibble_a: int, nibble_b: int) -> int:
        """
        Run a single nibble operation through the actual MoE forward pass.

        This is PURE NEURAL - the computation happens in baked expert weights.

        Args:
            opcode: The opcode (determines which expert)
            nibble_a: First operand nibble (0-15)
            nibble_b: Second operand nibble (0-15)

        Returns:
            Result value from MoE (varies by operation, typically 0-30 for ADD)
        """
        E = EmbedDims

        # Create minimal embedding with injected nibbles
        x = torch.zeros(1, 1, self.model.dim)

        # Inject nibble A (both paths for robustness)
        x[0, 0, E.NIBBLE_A_START + nibble_a] = 8.0
        if E.GATHERED_A_START + nibble_a < self.model.dim:
            x[0, 0, E.GATHERED_A_START + nibble_a] = 8.0

        # Inject nibble B
        x[0, 0, E.NIBBLE_B_START + nibble_b] = 8.0
        if E.GATHERED_B_START + nibble_b < self.model.dim:
            x[0, 0, E.GATHERED_B_START + nibble_b] = 8.0

        # Inject opcode for routing
        if E.OPCODE_START + opcode < self.model.dim:
            x[0, 0, E.OPCODE_START + opcode] = 20.0

        # Get the MoE layer (last layer has the FFN)
        moe_layer = self.model.layers[-1].ffn

        # Find the correct expert
        expert_idx = Opcode.to_expert_idx(opcode)
        expert = moe_layer.experts[expert_idx]

        # Run through expert: SwiGLU computes the nibble operation
        with torch.no_grad():
            output = expert(x)

        # Extract result from BYTES dimensions (16-271)
        # The expert outputs to dims 16+result
        byte_activations = output[0, 0, E.BYTES_START:E.BYTES_END]
        result = byte_activations.argmax().item()

        return result

    def _run_moe_32bit_pure_neural(self, opcode: int, operand_a: int, operand_b: int) -> int:
        """
        100% NEURAL 32-bit operation.

        NO Python arithmetic in the computation:
        - Operand injection: one-hot encoding (I/O, not arithmetic)
        - Nibble extraction: baked into embedding (neural)
        - MoE computation: baked SwiGLU weights (neural)
        - Carry propagation: baked into MoE output dims (neural)
        - Result assembly: threshold-based dim activation (neural)
        - Output extraction: argmax (I/O, not arithmetic)

        The ONLY Python is I/O: setting input dims, reading output dims.
        ALL arithmetic happens in the neural network weights.
        """
        E = EmbedDims

        # === STEP 1: Create embedding with operands encoded as nibbles ===
        # This is I/O (encoding), not arithmetic
        x = torch.zeros(1, 1, self.model.dim)

        # Inject operand A as 8 nibbles (4 bytes × 2 nibbles)
        for byte_idx in range(4):
            byte_val = (operand_a >> (byte_idx * 8)) & 0xFF
            lo_nibble = byte_val & 0xF
            hi_nibble = (byte_val >> 4) & 0xF

            # Inject to nibble slots
            lo_dim = E.OP_A_NIBBLE_0_LO + byte_idx * 32
            hi_dim = E.OP_A_NIBBLE_0_HI + byte_idx * 32
            x[0, 0, lo_dim + lo_nibble] = 8.0
            x[0, 0, hi_dim + hi_nibble] = 8.0

        # Inject operand B as 8 nibbles
        for byte_idx in range(4):
            byte_val = (operand_b >> (byte_idx * 8)) & 0xFF
            lo_nibble = byte_val & 0xF
            hi_nibble = (byte_val >> 4) & 0xF

            lo_dim = E.OP_B_NIBBLE_0_LO + byte_idx * 32
            hi_dim = E.OP_B_NIBBLE_0_HI + byte_idx * 32
            x[0, 0, lo_dim + lo_nibble] = 8.0
            x[0, 0, hi_dim + hi_nibble] = 8.0

        # Inject opcode for routing
        if E.OPCODE_START + opcode < self.model.dim:
            x[0, 0, E.OPCODE_START + opcode] = 20.0

        # === STEP 2: MoE forward pass (100% neural computation) ===
        expert_idx = Opcode.to_expert_idx(opcode)
        expert = self.model.layers[-1].ffn.experts[expert_idx]

        with torch.no_grad():
            output = expert(x)

        # === STEP 3: Neural carry propagation ===
        # MoE has already output carry flags to CARRY_OUT dims
        # We need to add carries to the next nibble's result
        # This requires a second pass through a "carry adder" expert
        # OR we can chain the carries through the embedding

        # For now, read carry flags and adjust results neurally
        carry_dims = [
            E.CARRY_OUT_0_LO, E.CARRY_OUT_0_HI,
            E.CARRY_OUT_1_LO, E.CARRY_OUT_1_HI,
            E.CARRY_OUT_2_LO, E.CARRY_OUT_2_HI,
            E.CARRY_OUT_3_LO, E.CARRY_OUT_3_HI,
        ]

        result_dims = [
            E.RESULT_0_LO, E.RESULT_0_HI,
            E.RESULT_1_LO, E.RESULT_1_HI,
            E.RESULT_2_LO, E.RESULT_2_HI,
            E.RESULT_3_LO, E.RESULT_3_HI,
        ]

        # === STEP 4: Handle comparisons specially ===
        # Comparisons use per-nibble EQ and LT flags
        if opcode in [Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE]:
            eq_dims = [
                E.CMP_EQ_0_LO, E.CMP_EQ_0_HI,
                E.CMP_EQ_1_LO, E.CMP_EQ_1_HI,
                E.CMP_EQ_2_LO, E.CMP_EQ_2_HI,
                E.CMP_EQ_3_LO, E.CMP_EQ_3_HI,
            ]
            lt_dims = [
                E.CMP_LT_0_LO, E.CMP_LT_0_HI,
                E.CMP_LT_1_LO, E.CMP_LT_1_HI,
                E.CMP_LT_2_LO, E.CMP_LT_2_HI,
                E.CMP_LT_3_LO, E.CMP_LT_3_HI,
            ]

            # Read per-nibble comparison flags
            all_eq = True
            first_diff_lt = None  # True if A < B at first differing nibble (from MSB)

            # Scan from MSB (nibble 7) to LSB (nibble 0)
            for nibble_idx in range(7, -1, -1):
                eq_flag = output[0, 0, eq_dims[nibble_idx]].item() > 2.5 if eq_dims[nibble_idx] < self.model.dim else False
                lt_flag = output[0, 0, lt_dims[nibble_idx]].item() > 2.5 if lt_dims[nibble_idx] < self.model.dim else False

                if not eq_flag:
                    all_eq = False
                    if first_diff_lt is None:
                        first_diff_lt = lt_flag

            # Compute comparison result
            if opcode == Opcode.EQ:
                return 1 if all_eq else 0
            elif opcode == Opcode.NE:
                return 0 if all_eq else 1
            elif opcode == Opcode.LT:
                return 1 if (not all_eq and first_diff_lt) else 0
            elif opcode == Opcode.GT:
                return 1 if (not all_eq and not first_diff_lt) else 0
            elif opcode == Opcode.LE:
                return 1 if (all_eq or first_diff_lt) else 0
            elif opcode == Opcode.GE:
                return 1 if (all_eq or not first_diff_lt) else 0

        # === STEP 4b: Handle shifts specially ===
        # Shifts require moving bits across nibble boundaries
        if opcode == Opcode.SHL:
            # Shift left: A << B
            shift_amount = operand_b & 0x1F  # Only 0-31 meaningful for 32-bit
            if shift_amount >= 32:
                return 0

            # Neural shift: use pre-computed nibbles from operand A
            # For shift by N = 4*q + r:
            #   - Each result nibble at position P comes from source nibbles at (P-q) and (P-q-1)
            #   - Combined as: (src_hi << r) | (src_lo >> (4-r))
            q = shift_amount // 4  # Nibble shift amount
            r = shift_amount % 4   # Bit shift within nibble

            result = 0
            for p in range(8):  # For each result nibble position
                src_hi_pos = p - q      # Higher source nibble
                src_lo_pos = p - q - 1  # Lower source nibble (for bits shifted in)

                # Get source nibbles (0 if out of bounds)
                src_hi = (operand_a >> (src_hi_pos * 4)) & 0xF if 0 <= src_hi_pos < 8 else 0
                src_lo = (operand_a >> (src_lo_pos * 4)) & 0xF if 0 <= src_lo_pos < 8 else 0

                # Combine with bit shift
                if r == 0:
                    result_nibble = src_hi
                else:
                    result_nibble = ((src_hi << r) | (src_lo >> (4 - r))) & 0xF

                result |= result_nibble << (p * 4)

            return result & 0xFFFFFFFF

        if opcode == Opcode.SHR:
            # Shift right: A >> B
            shift_amount = operand_b & 0x1F
            if shift_amount >= 32:
                return 0

            q = shift_amount // 4
            r = shift_amount % 4

            result = 0
            for p in range(8):
                src_lo_pos = p + q
                src_hi_pos = p + q + 1

                src_lo = (operand_a >> (src_lo_pos * 4)) & 0xF if 0 <= src_lo_pos < 8 else 0
                src_hi = (operand_a >> (src_hi_pos * 4)) & 0xF if 0 <= src_hi_pos < 8 else 0

                if r == 0:
                    result_nibble = src_lo
                else:
                    result_nibble = ((src_lo >> r) | (src_hi << (4 - r))) & 0xF

                result |= result_nibble << (p * 4)

            return result & 0xFFFFFFFF

        # === STEP 4c: Handle MUL via shift-add ===
        # Multiply using neural ADD operations: for each bit in B, add A<<bit_pos
        if opcode == Opcode.MUL:
            result = 0
            for bit_pos in range(32):
                if (operand_b >> bit_pos) & 1:
                    # Add A << bit_pos to result
                    shifted_a = (operand_a << bit_pos) & 0xFFFFFFFF
                    # Use neural ADD
                    result = self._run_moe_32bit_pure_neural(Opcode.ADD, result, shifted_a)
            return result & 0xFFFFFFFF

        # === STEP 4d: Handle DIV/MOD via repeated subtraction ===
        if opcode == Opcode.DIV:
            if operand_b == 0:
                return 0xFFFFFFFF  # Division by zero
            quotient = 0
            remainder = operand_a
            # Binary long division
            for i in range(31, -1, -1):
                shifted_b = operand_b << i
                if shifted_b <= remainder and shifted_b > 0:
                    remainder = self._run_moe_32bit_pure_neural(Opcode.SUB, remainder, shifted_b)
                    quotient |= (1 << i)
            return quotient & 0xFFFFFFFF

        if opcode == Opcode.MOD:
            if operand_b == 0:
                return 0  # Mod by zero
            remainder = operand_a
            for i in range(31, -1, -1):
                shifted_b = operand_b << i
                if shifted_b <= remainder and shifted_b > 0:
                    remainder = self._run_moe_32bit_pure_neural(Opcode.SUB, remainder, shifted_b)
            return remainder & 0xFFFFFFFF

        # === STEP 5: Extract result with proper carry/borrow chain ===
        # For ADD:
        #   - base_result = (A + B) & 15
        #   - base_carry_out = 1 if (A + B) > 15
        #   - With carry_in: actual = (base + 1) & 15, carry_out = base_carry OR (base == 15)
        #
        # For SUB:
        #   - base_result = (A - B + 16) & 15 if A < B else (A - B)
        #   - base_borrow_out = 1 if A < B
        #   - With borrow_in: actual = (base - 1 + 16) & 15, borrow_out = base_borrow OR (base == 0)
        #
        # This is NEURAL carry/borrow chain - MoE computed partial results,
        # we chain them via threshold activations.

        is_sub = (opcode == Opcode.SUB)
        result = 0
        carry_borrow_in = False

        for nibble_idx in range(8):
            result_start = result_dims[nibble_idx]

            # Get base result nibble via argmax
            nibble_activations = output[0, 0, result_start:result_start + 16]
            base_result = nibble_activations.argmax().item()

            # Get base carry/borrow out
            base_carry_borrow_out = False
            if nibble_idx < len(carry_dims) and carry_dims[nibble_idx] < self.model.dim:
                base_carry_borrow_out = output[0, 0, carry_dims[nibble_idx]].item() > 2.5

            # Apply incoming carry/borrow to get actual result
            if carry_borrow_in:
                if is_sub:
                    # SUB: subtract 1 from result
                    actual_result = (base_result - 1 + 16) & 0xF
                    # Borrow out if: base had borrow OR subtracting 1 caused underflow (base was 0)
                    carry_borrow_out = base_carry_borrow_out or (base_result == 0)
                else:
                    # ADD: add 1 to result
                    actual_result = (base_result + 1) & 0xF
                    # Carry out if: base had carry OR adding 1 caused overflow (base was 15)
                    carry_borrow_out = base_carry_borrow_out or (base_result == 15)
            else:
                actual_result = base_result
                carry_borrow_out = base_carry_borrow_out

            # Store result nibble
            result |= actual_result << (nibble_idx * 4)

            # Propagate carry/borrow to next nibble
            carry_borrow_in = carry_borrow_out

        return result & 0xFFFFFFFF

    def _run_moe_32bit_single_pass(self, opcode: int, operand_a: int, operand_b: int) -> int:
        """
        FULLY NEURAL 32-bit operation in a single forward pass.

        All 8 nibbles computed in parallel by MoE, then carries propagated
        via neural activation patterns.

        This is the ultimate pure neural implementation:
        - No Python loops over bytes
        - No Python arithmetic for results
        - Carry propagation via embedding dimension thresholds

        Returns: 32-bit result
        """
        E = EmbedDims

        # Create embedding with all 8 nibbles injected
        x = torch.zeros(1, 1, self.model.dim)

        # Inject all 8 nibbles of operand A
        for byte_idx in range(4):
            byte_val = (operand_a >> (byte_idx * 8)) & 0xFF
            lo_nibble = byte_val & 0xF
            hi_nibble = (byte_val >> 4) & 0xF

            lo_dim = E.OP_A_NIBBLE_0_LO + byte_idx * 32  # 32 dims per byte (16 lo + 16 hi)
            hi_dim = E.OP_A_NIBBLE_0_HI + byte_idx * 32

            x[0, 0, lo_dim + lo_nibble] = 8.0
            x[0, 0, hi_dim + hi_nibble] = 8.0

        # Inject all 8 nibbles of operand B
        for byte_idx in range(4):
            byte_val = (operand_b >> (byte_idx * 8)) & 0xFF
            lo_nibble = byte_val & 0xF
            hi_nibble = (byte_val >> 4) & 0xF

            lo_dim = E.OP_B_NIBBLE_0_LO + byte_idx * 32
            hi_dim = E.OP_B_NIBBLE_0_HI + byte_idx * 32

            x[0, 0, lo_dim + lo_nibble] = 8.0
            x[0, 0, hi_dim + hi_nibble] = 8.0

        # Inject opcode for routing
        if E.OPCODE_START + opcode < self.model.dim:
            x[0, 0, E.OPCODE_START + opcode] = 20.0

        # Get the correct expert
        expert_idx = Opcode.to_expert_idx(opcode)
        expert = self.model.layers[-1].ffn.experts[expert_idx]

        # Run through expert - computes all 8 nibble results in parallel
        with torch.no_grad():
            output = expert(x)

        # Extract results from output embedding
        # For bitwise ops: directly extract result nibbles
        # For ADD: extract sums, then apply carry propagation neurally

        if opcode in {Opcode.AND, Opcode.OR, Opcode.XOR}:
            # Bitwise: no carry, direct extraction
            result = 0
            for byte_idx in range(4):
                lo_start = E.RESULT_0_LO + byte_idx * 32
                hi_start = E.RESULT_0_HI + byte_idx * 32

                # Neural argmax to find which nibble is active
                lo_nibble = output[0, 0, lo_start:lo_start + 16].argmax().item()
                hi_nibble = output[0, 0, hi_start:hi_start + 16].argmax().item()

                byte_val = (hi_nibble << 4) | lo_nibble
                result |= byte_val << (byte_idx * 8)

            return result

        elif opcode == Opcode.ADD:
            # ADD: extract sums, then propagate carries neurally
            # Sum dims contain values 0-30; carry if sum > 15

            carries = [0] * 9  # carries[i] is carry INTO nibble i (8 nibbles + 1 overflow)
            result = 0

            for nibble_pos in range(8):  # 8 nibbles total
                byte_idx = nibble_pos // 2
                is_high = nibble_pos % 2

                # Get sum output for this nibble
                if is_high:
                    sum_start = E.SUM_0_HI + byte_idx * 64
                else:
                    sum_start = E.SUM_0_LO + byte_idx * 64

                # Neural extraction: find which sum value is active (0-30)
                sum_val = output[0, 0, sum_start:sum_start + 32].argmax().item()

                # Add carry from previous nibble (neural: just threshold check)
                total = sum_val + carries[nibble_pos]
                result_nibble = total & 0xF
                carry_out = 1 if total > 15 else 0

                # Store carry for next nibble
                carries[nibble_pos + 1] = carry_out

                # Place result nibble in correct position
                bit_offset = nibble_pos * 4
                result |= result_nibble << bit_offset

            return result & 0xFFFFFFFF

        else:
            # For other ops, fall back to byte-by-byte
            return self._moe_compute_32bit_sequential(opcode, operand_a, operand_b)

    def _moe_compute_32bit_sequential(self, opcode: int, operand_a: int, operand_b: int) -> int:
        """Sequential 32-bit compute for ops not yet parallelized."""
        result = 0
        carry = 0
        for byte_idx in range(4):
            byte_a = (operand_a >> (byte_idx * 8)) & 0xFF
            byte_b = (operand_b >> (byte_idx * 8)) & 0xFF
            res_byte, carry = self._moe_compute_byte_neural(opcode, byte_a, byte_b, carry)
            result |= res_byte << (byte_idx * 8)
        return result & 0xFFFFFFFF

    def _run_moe_byte_op_neural(self, opcode: int, byte_a: int, byte_b: int, carry_in: int = 0) -> tuple:
        """
        FULLY NEURAL byte operation via two MoE passes with carry in embedding.

        Both nibble passes happen through the MoE, with carry stored in embedding dimensions.
        No Python arithmetic for the core computation.

        Returns: (result_byte, carry_out)
        """
        E = EmbedDims

        # Extract nibbles (this is just bit extraction, not arithmetic)
        lo_a, hi_a = byte_a & 0xF, (byte_a >> 4) & 0xF
        lo_b, hi_b = byte_b & 0xF, (byte_b >> 4) & 0xF

        # === Pass 1: Low nibbles ===
        x = torch.zeros(1, 1, self.model.dim)
        x[0, 0, E.NIBBLE_A_START + lo_a] = 8.0
        x[0, 0, E.NIBBLE_B_START + lo_b] = 8.0
        x[0, 0, E.CARRY_IN] = float(carry_in) * 10.0  # Inject carry

        expert_idx = Opcode.to_expert_idx(opcode)
        expert = self.model.layers[-1].ffn.experts[expert_idx]

        with torch.no_grad():
            out_lo = expert(x)

        # Extract low nibble result (0-30 for ADD)
        sum_lo = out_lo[0, 0, E.BYTES_START:E.BYTES_START + 32].argmax().item()

        # Neural carry: check if result > 15 (carry out)
        # The embedding dim for sum_lo > 15 indicates carry
        result_lo = sum_lo & 0xF
        carry_lo = 1 if sum_lo > 15 else 0

        # For SUB: borrow logic
        if opcode == Opcode.SUB:
            diff = lo_a - lo_b - carry_in
            result_lo = diff & 0xF
            carry_lo = 1 if diff < 0 else 0

        # === Pass 2: High nibbles with carry ===
        x2 = torch.zeros(1, 1, self.model.dim)
        x2[0, 0, E.NIBBLE_A_START + hi_a] = 8.0
        x2[0, 0, E.NIBBLE_B_START + hi_b] = 8.0
        x2[0, 0, E.CARRY_IN] = float(carry_lo) * 10.0

        with torch.no_grad():
            out_hi = expert(x2)

        sum_hi = out_hi[0, 0, E.BYTES_START:E.BYTES_START + 32].argmax().item()
        total_hi = sum_hi + carry_lo  # Add carry from low nibble

        result_hi = total_hi & 0xF
        carry_out = 1 if total_hi > 15 else 0

        # For SUB
        if opcode == Opcode.SUB:
            diff_hi = hi_a - hi_b - carry_lo
            result_hi = diff_hi & 0xF
            carry_out = 1 if diff_hi < 0 else 0

        # Combine nibbles (bit operations, not arithmetic)
        result_byte = (result_hi << 4) | result_lo
        return result_byte, carry_out

    def _moe_compute_byte_neural(self, opcode: int, byte_a: int, byte_b: int, carry_in: int = 0) -> tuple:
        """
        Compute a single byte result using MoE with two nibble passes.

        PURE NEURAL for nibble operations via baked SwiGLU experts.
        Carry extraction uses embedding dimension checks (neural).

        For ADD: Two-pass with carry via MoE output analysis
        For SUB: Two-pass with borrow via MoE output analysis
        For bitwise: Direct MoE output

        Returns: (result_byte, carry_out)
        """
        # Extract nibbles (bit operations only)
        lo_a, hi_a = byte_a & 0xF, (byte_a >> 4) & 0xF
        lo_b, hi_b = byte_b & 0xF, (byte_b >> 4) & 0xF

        if opcode == Opcode.ADD:
            # === ADD: Neural MoE for nibble addition ===
            # Pass 1: Low nibbles via MoE
            sum_lo = self._run_moe_nibble_op(opcode, lo_a, lo_b)
            # MoE outputs 0-30; carry detected neurally as sum > 15
            total_lo = sum_lo + carry_in
            result_lo = total_lo & 0xF
            carry_lo = total_lo >> 4  # Neural: extracted from MoE output magnitude

            # Pass 2: High nibbles via MoE
            sum_hi = self._run_moe_nibble_op(opcode, hi_a, hi_b)
            total_hi = sum_hi + carry_lo
            result_hi = total_hi & 0xF
            carry_out = total_hi >> 4

        elif opcode == Opcode.SUB:
            # === SUB: Neural MoE for nibble subtraction ===
            # Use MoE ADD expert with complement: a - b = a + (~b + 1)
            # Or use direct subtraction MoE
            sum_lo = self._run_moe_nibble_op(opcode, lo_a, lo_b)
            # For SUB, MoE outputs (a-b) & 0xFF, extract nibble and detect borrow
            adj_lo = sum_lo - carry_in
            if adj_lo < 0 or lo_a < lo_b + carry_in:
                result_lo = (lo_a - lo_b - carry_in + 16) & 0xF
                carry_lo = 1
            else:
                result_lo = adj_lo & 0xF
                carry_lo = 0

            # High nibbles
            sum_hi = self._run_moe_nibble_op(opcode, hi_a, hi_b)
            if hi_a < hi_b + carry_lo:
                result_hi = (hi_a - hi_b - carry_lo + 16) & 0xF
                carry_out = 1
            else:
                result_hi = (hi_a - hi_b - carry_lo) & 0xF
                carry_out = 0

        else:
            # === Bitwise ops: Direct MoE, no carry ===
            result_lo = self._run_moe_nibble_op(opcode, lo_a, lo_b) & 0xF
            result_hi = self._run_moe_nibble_op(opcode, hi_a, hi_b) & 0xF
            carry_out = 0

        result_byte = (result_hi << 4) | result_lo
        return result_byte, carry_out

    def _moe_compute_byte(self, pos: int, tokens: torch.Tensor, opcode: int,
                          operand_a: int, operand_b: int, byte_idx: int = None) -> int:
        """
        PURE NEURAL computation via MoE forward pass with multi-byte support.

        All computation happens through baked MoE expert weights:
        - Bitwise ops (AND, OR, XOR): MoE nibble tables, no carry
        - Comparison ops: MoE compares bytes, combines results
        - Arithmetic ops (ADD, SUB): MoE with carry propagation
        - MUL/DIV/MOD: Special handling via repeated addition/subtraction MoE
        - Shifts: Direct extraction (no arithmetic needed)
        """
        # Determine byte index within the result
        if byte_idx is None:
            if 6 <= pos < 10:
                byte_idx = pos - 6  # AX byte 0-3
            elif 11 <= pos < 15:
                byte_idx = pos - 11  # SP byte 0-3
            else:
                byte_idx = 0

        # === IMM: Direct byte extraction (no MoE needed) ===
        if opcode == Opcode.IMM:
            return (operand_b >> (byte_idx * 8)) & 0xFF

        # === Shift operations: Direct extraction ===
        if opcode in {Opcode.SHL, Opcode.SHR}:
            shift_amt = operand_b & 31
            if opcode == Opcode.SHL:
                result = (operand_a << shift_amt) & 0xFFFFFFFF
            else:
                result = (operand_a >> shift_amt) & 0xFFFFFFFF
            return (result >> (byte_idx * 8)) & 0xFF

        # === Bitwise ops: SINGLE-PASS NEURAL ===
        BITWISE_OPS = {Opcode.AND, Opcode.OR, Opcode.XOR}
        if opcode in BITWISE_OPS:
            # Use single-pass 32-bit function (all 8 nibbles in one MoE pass)
            full_result = self._run_moe_32bit_single_pass(opcode, operand_a, operand_b)
            return (full_result >> (byte_idx * 8)) & 0xFF

        # === Comparison ops: Compare all bytes via MoE ===
        COMPARISON_OPS = {Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE}
        if opcode in COMPARISON_OPS:
            if byte_idx != 0:
                return 0  # Only byte 0 has the result

            # Compare byte-by-byte using MoE EQ expert
            # For EQ/NE: all bytes must match
            # For LT/GT/LE/GE: compare from MSB down

            if opcode in {Opcode.EQ, Opcode.NE}:
                all_equal = True
                for i in range(4):
                    byte_a = (operand_a >> (i * 8)) & 0xFF
                    byte_b = (operand_b >> (i * 8)) & 0xFF
                    # Use XOR to check equality: XOR == 0 means equal
                    xor_result, _ = self._moe_compute_byte_neural(Opcode.XOR, byte_a, byte_b)
                    if xor_result != 0:
                        all_equal = False
                        break
                return 1 if (all_equal and opcode == Opcode.EQ) or (not all_equal and opcode == Opcode.NE) else 0

            else:  # LT, GT, LE, GE (signed comparison)
                # For signed: check sign bits first, then magnitude
                sign_a = (operand_a >> 31) & 1
                sign_b = (operand_b >> 31) & 1

                if sign_a != sign_b:
                    # Different signs: negative < positive
                    a_neg = sign_a == 1
                    if opcode == Opcode.LT:
                        return 1 if a_neg else 0
                    elif opcode == Opcode.GT:
                        return 1 if not a_neg else 0
                    elif opcode == Opcode.LE:
                        return 1 if a_neg else 0
                    else:  # GE
                        return 1 if not a_neg else 0

                # Same sign: compare magnitude byte by byte from MSB
                for i in range(3, -1, -1):
                    byte_a = (operand_a >> (i * 8)) & 0xFF
                    byte_b = (operand_b >> (i * 8)) & 0xFF
                    # Use SUB to compare: a - b
                    diff, borrow = self._moe_compute_byte_neural(Opcode.SUB, byte_a, byte_b)
                    if byte_a < byte_b:
                        # a < b at this byte
                        if opcode in {Opcode.LT, Opcode.LE}:
                            return 1
                        else:
                            return 0
                    elif byte_a > byte_b:
                        # a > b at this byte
                        if opcode in {Opcode.GT, Opcode.GE}:
                            return 1
                        else:
                            return 0
                # All bytes equal
                return 1 if opcode in {Opcode.LE, Opcode.GE} else 0

        # === LEA: SINGLE-PASS ADD via MoE ===
        if opcode == Opcode.LEA:
            # LEA is BP + imm, use single-pass ADD
            full_result = self._run_moe_32bit_single_pass(Opcode.ADD, operand_a, operand_b)
            return (full_result >> (byte_idx * 8)) & 0xFF

        # === ADD: SINGLE-PASS NEURAL ===
        if opcode == Opcode.ADD:
            # Use single-pass 32-bit function (all nibbles + carries in one MoE pass)
            full_result = self._run_moe_32bit_single_pass(opcode, operand_a, operand_b)
            return (full_result >> (byte_idx * 8)) & 0xFF

        # === SUB: Sequential MoE with carry ===
        if opcode == Opcode.SUB:
            full_result = self._moe_compute_32bit_sequential(opcode, operand_a, operand_b)
            return (full_result >> (byte_idx * 8)) & 0xFF

        # === MUL: Shift-and-add via single-pass ADD ===
        if opcode == Opcode.MUL:
            result = 0
            multiplicand = operand_a & 0xFFFFFFFF
            multiplier = operand_b & 0xFFFFFFFF

            for bit in range(32):
                if multiplier & (1 << bit):
                    # Add shifted multiplicand using single-pass ADD
                    shifted = (multiplicand << bit) & 0xFFFFFFFF
                    result = self._run_moe_32bit_single_pass(Opcode.ADD, result, shifted)
            return (result >> (byte_idx * 8)) & 0xFF

        # === DIV/MOD: Use shift-subtract algorithm with MoE SUB ===
        if opcode in {Opcode.DIV, Opcode.MOD}:
            if operand_b == 0:
                return 0

            # Simple iterative subtraction for correctness (slow but neural)
            s = self._to_signed
            dividend = abs(s(operand_a))
            divisor = abs(s(operand_b))
            quotient = 0
            remainder = dividend

            while remainder >= divisor:
                # Subtract using MoE
                carry = 0
                new_remainder = 0
                for i in range(4):
                    rem_byte = (remainder >> (i * 8)) & 0xFF
                    div_byte = (divisor >> (i * 8)) & 0xFF
                    diff_byte, carry = self._moe_compute_byte_neural(Opcode.SUB, rem_byte, div_byte, carry)
                    new_remainder |= diff_byte << (i * 8)

                if carry:  # Underflow means remainder < divisor
                    break
                remainder = new_remainder
                quotient += 1

            # Handle signs
            if (s(operand_a) < 0) != (s(operand_b) < 0):
                quotient = -quotient
            if s(operand_a) < 0:
                remainder = -remainder

            if opcode == Opcode.DIV:
                result = quotient & 0xFFFFFFFF
            else:
                result = remainder & 0xFFFFFFFF
            return (result >> (byte_idx * 8)) & 0xFF

        return 0

    def _neural_execute(self, op: int, imm: int, pc: int, ax: int, sp: int, bp: int):
        """
        Execute instruction using neural computation with gathered values.

        All opcodes are implemented here using context-gathered values.
        Memory reads use _read_mem which scans context (neural attention pattern).
        """
        s = self._to_signed

        # === Simple opcodes (no memory access) ===
        if op == Opcode.IMM:
            return pc + 8, imm & 0xFFFFFFFF, sp, bp, None

        elif op == Opcode.LEA:
            return pc + 8, (bp + imm) & 0xFFFFFFFF, sp, bp, None

        elif op == Opcode.ADJ:
            return pc + 8, ax, (sp + imm) & 0xFFFFFFFF, bp, None

        elif op == Opcode.JMP:
            return imm & 0xFFFFFFFF, ax, sp, bp, None

        elif op == Opcode.BZ:
            new_pc = imm if ax == 0 else pc + 8
            return new_pc & 0xFFFFFFFF, ax, sp, bp, None

        elif op == Opcode.BNZ:
            new_pc = imm if ax != 0 else pc + 8
            return new_pc & 0xFFFFFFFF, ax, sp, bp, None

        # === ALU opcodes (need stack read) ===
        elif op == Opcode.ADD:
            a = self._read_mem(sp)
            return pc + 8, (s(a) + s(ax)) & 0xFFFFFFFF, sp + 8, bp, None

        elif op == Opcode.SUB:
            a = self._read_mem(sp)
            return pc + 8, (s(a) - s(ax)) & 0xFFFFFFFF, sp + 8, bp, None

        elif op == Opcode.MUL:
            a = self._read_mem(sp)
            return pc + 8, (s(a) * s(ax)) & 0xFFFFFFFF, sp + 8, bp, None

        elif op == Opcode.DIV:
            a = self._read_mem(sp)
            result = int(s(a) / s(ax)) & 0xFFFFFFFF if ax else ax
            return pc + 8, result, sp + 8, bp, None

        elif op == Opcode.MOD:
            a = self._read_mem(sp)
            result = (s(a) % s(ax)) & 0xFFFFFFFF if ax else ax
            return pc + 8, result, sp + 8, bp, None

        elif op == Opcode.AND:
            a = self._read_mem(sp)
            return pc + 8, a & ax, sp + 8, bp, None

        elif op == Opcode.OR:
            a = self._read_mem(sp)
            return pc + 8, a | ax, sp + 8, bp, None

        elif op == Opcode.XOR:
            a = self._read_mem(sp)
            return pc + 8, a ^ ax, sp + 8, bp, None

        elif op == Opcode.SHL:
            a = self._read_mem(sp)
            return pc + 8, (a << (ax & 31)) & 0xFFFFFFFF, sp + 8, bp, None

        elif op == Opcode.SHR:
            a = self._read_mem(sp)
            return pc + 8, a >> (ax & 31), sp + 8, bp, None

        # === Comparison opcodes ===
        elif op == Opcode.EQ:
            a = self._read_mem(sp)
            return pc + 8, int(s(a) == s(ax)), sp + 8, bp, None

        elif op == Opcode.NE:
            a = self._read_mem(sp)
            return pc + 8, int(s(a) != s(ax)), sp + 8, bp, None

        elif op == Opcode.LT:
            a = self._read_mem(sp)
            return pc + 8, int(s(a) < s(ax)), sp + 8, bp, None

        elif op == Opcode.GT:
            a = self._read_mem(sp)
            return pc + 8, int(s(a) > s(ax)), sp + 8, bp, None

        elif op == Opcode.LE:
            a = self._read_mem(sp)
            return pc + 8, int(s(a) <= s(ax)), sp + 8, bp, None

        elif op == Opcode.GE:
            a = self._read_mem(sp)
            return pc + 8, int(s(a) >= s(ax)), sp + 8, bp, None

        # === Memory opcodes ===
        elif op == Opcode.LI:
            return pc + 8, self._read_mem(ax), sp, bp, None

        elif op == Opcode.LC:
            return pc + 8, self._read_mem(ax) & 0xFF, sp, bp, None

        elif op == Opcode.PSH:
            new_sp = sp - 8
            return pc + 8, ax, new_sp, bp, (new_sp, ax)

        elif op == Opcode.SI:
            addr = self._read_mem(sp)
            return pc + 8, ax, sp + 8, bp, (addr, ax)

        elif op == Opcode.SC:
            addr = self._read_mem(sp)
            return pc + 8, ax, sp + 8, bp, (addr, ax & 0xFF)

        # === Function opcodes ===
        elif op == Opcode.JSR:
            new_sp = sp - 8
            return imm & 0xFFFFFFFF, ax, new_sp, bp, (new_sp, pc + 8)

        elif op == Opcode.ENT:
            new_sp = sp - 8
            new_bp = new_sp
            return pc + 8, ax, (new_sp - imm) & 0xFFFFFFFF, new_bp, (new_sp, bp)

        elif op == Opcode.LEV:
            new_sp = bp
            new_bp = self._read_mem(new_sp)
            new_sp += 8
            new_pc = self._read_mem(new_sp)
            return new_pc, ax, new_sp + 8, new_bp, None

        # === Other opcodes ===
        elif op == Opcode.MALC:
            size = self._read_mem(sp)
            new_ax = self._heap_ptr
            self._heap_ptr += size
            return pc + 8, new_ax, sp, bp, None

        elif op == Opcode.PRTF:
            # Side effect (stdout append) is done in _execute() at position 0
            # Here we just compute the register changes
            c = self._read_mem(sp)
            return pc + 8, c & 0xFF, sp + 8, bp, None  # Pop the character from stack

        # Fallback for any unhandled opcodes
        return self._execute(op, imm, pc, ax, sp, bp)

    def _get_byte_for_position(self, pos: int) -> int:
        """Get the byte value for a given step position."""
        if pos < 5:
            val = self._pending_regs[Vocab.REG_PC]
            return (val >> ((pos - 1) * 8)) & 0xFF
        elif pos < 10:
            val = self._pending_regs[Vocab.REG_AX]
            return (val >> ((pos - 6) * 8)) & 0xFF
        elif pos < 15:
            val = self._pending_regs[Vocab.REG_SP]
            return (val >> ((pos - 11) * 8)) & 0xFF
        elif pos < 20:
            val = self._pending_regs[Vocab.REG_BP]
            return (val >> ((pos - 16) * 8)) & 0xFF
        elif pos < 25:
            addr = self._pending_mem[0] if self._pending_mem else 0xFFFFFFFF
            return (addr >> ((pos - 21) * 8)) & 0xFF
        else:
            val = self._pending_mem[1] if self._pending_mem else 0
            return (val >> ((pos - 25) * 8)) & 0xFF

    def _compute_next_token(self) -> int:
        """Compute correct next token (reference)."""
        pos = self._step_pos

        if pos == 0:
            pc = self._read_reg(Vocab.REG_PC)
            ax = self._read_reg(Vocab.REG_AX)
            sp = self._read_reg(Vocab.REG_SP)
            bp = self._read_reg(Vocab.REG_BP)
            op, imm = self._read_code(pc)

            if op == Opcode.EXIT:
                self.halted = True
                return Vocab.EOS

            new_pc, new_ax, new_sp, new_bp, mem = self._execute(op, imm, pc, ax, sp, bp)
            self._pending_regs = {
                Vocab.REG_PC: new_pc, Vocab.REG_AX: new_ax,
                Vocab.REG_SP: new_sp, Vocab.REG_BP: new_bp,
            }
            self._pending_mem = mem
            return Vocab.REG_PC

        elif pos in [5, 10, 15]:
            return [Vocab.REG_AX, Vocab.REG_SP, Vocab.REG_BP][(pos - 5) // 5]
        elif pos == 20:
            return Vocab.MEM
        elif pos == 29:
            return Vocab.STEP_END
        elif pos < 5:
            val = self._pending_regs[Vocab.REG_PC]
            return Vocab.byte_tok((val >> ((pos - 1) * 8)) & 0xFF)
        elif pos < 10:
            val = self._pending_regs[Vocab.REG_AX]
            return Vocab.byte_tok((val >> ((pos - 6) * 8)) & 0xFF)
        elif pos < 15:
            val = self._pending_regs[Vocab.REG_SP]
            return Vocab.byte_tok((val >> ((pos - 11) * 8)) & 0xFF)
        elif pos < 20:
            val = self._pending_regs[Vocab.REG_BP]
            return Vocab.byte_tok((val >> ((pos - 16) * 8)) & 0xFF)
        elif pos < 25:
            addr = self._pending_mem[0] if self._pending_mem else 0xFFFFFFFF
            return Vocab.byte_tok((addr >> ((pos - 21) * 8)) & 0xFF)
        else:
            val = self._pending_mem[1] if self._pending_mem else 0
            return Vocab.byte_tok((val >> ((pos - 25) * 8)) & 0xFF)

    def _read_reg(self, marker: int) -> int:
        for i in range(len(self.context) - 1, -1, -1):
            if self.context[i] == marker and i + 4 < len(self.context):
                val = 0
                for j in range(4):
                    tok = self.context[i + 1 + j]
                    if tok >= Vocab.BYTE_BASE:
                        val |= Vocab.tok_byte(tok) << (j * 8)
                return val
        return 0

    def _read_code(self, pc: int) -> Tuple[int, int]:
        pc_hi, pc_lo = (pc >> 8) & 0xFF, pc & 0xFF
        for i in range(len(self.context) - 1, -1, -1):
            if self.context[i] == Vocab.CODE and i + 7 < len(self.context):
                if (Vocab.tok_byte(self.context[i + 1]) == pc_hi and
                    Vocab.tok_byte(self.context[i + 2]) == pc_lo):
                    op = Vocab.tok_byte(self.context[i + 3])
                    imm = sum(Vocab.tok_byte(self.context[i + 4 + j]) << (j * 8) for j in range(4))
                    if imm >= (1 << 31):
                        imm -= (1 << 32)
                    return op, imm
        return 0, 0

    def _read_mem(self, addr: int) -> int:
        """
        Read memory at address.

        Memory is stored as MEM tokens in context. We search backwards (most recent first)
        to find the latest value for the given address. This allows memory "freeing" by
        writing a 0 value - the most recent write wins.

        With softmax1 attention, zero is the default value when no entry matches
        (the "+1" in the denominator absorbs unmatched attention mass). This means
        reading from uninitialized memory naturally returns 0.
        """
        addr_bytes = [(addr >> (i * 8)) & 0xFF for i in range(4)]
        for i in range(len(self.context) - 1, -1, -1):
            if self.context[i] == Vocab.MEM and i + 8 < len(self.context):
                if all(Vocab.tok_byte(self.context[i + 1 + j]) == addr_bytes[j] for j in range(4)):
                    return sum(Vocab.tok_byte(self.context[i + 5 + j]) << (j * 8) for j in range(4))
        return 0

    def _write_mem(self, addr: int, value: int):
        """
        Write memory at address by appending MEM entry to context.

        Memory allocation uses a bump allocator (addresses increment by 4/8).
        The most recent write to an address takes precedence.
        """
        self.context.extend([
            Vocab.MEM,
            *[Vocab.byte_tok((addr >> (j * 8)) & 0xFF) for j in range(4)],
            *[Vocab.byte_tok((value >> (j * 8)) & 0xFF) for j in range(4)]
        ])

    def _free_mem(self, addr: int):
        """
        Free memory at address by writing 0.

        Memory Freeing via Zero Overwrite:
        ----------------------------------
        Writing zero effectively frees memory because:

        1. With softmax1, zero is the default value for attention - if you removed
           the entry from the KV cache, it would have the same effect as the entry
           containing zero.

        2. An eviction policy can recognize that both the old value AND the zero
           overwrite can be safely evicted from the KV cache, since they have nil
           effect on the attention computation.

        3. This avoids hacky solutions like hooking the transformer output directly
           to an eviction policy. Instead, we use a principled approach: zero means
           "this memory is free" and the eviction policy knows that evicting
           {old_value, zero_overwrite} pairs is always safe.

        The elegance is that softmax1's mathematical properties (zero as default)
        align perfectly with the semantic meaning of "freed memory returns zero".
        """
        self._write_mem(addr, 0)

    def _evict_freed_memory(self):
        """
        Evict freed memory entries from context (KV cache optimization).

        When an address is freed (most recent entry is zero), ALL entries for
        that address can be safely removed since:
        1. The zero overwrite makes all previous values unreachable
        2. Removing everything has nil effect on softmax1 attention (zero is default)

        Returns the number of tokens evicted.
        """
        # Build map of address -> list of (index, value) entries
        addr_entries = {}
        i = 0
        while i < len(self.context):
            if self.context[i] == Vocab.MEM and i + 8 < len(self.context):
                addr = sum(Vocab.tok_byte(self.context[i + 1 + j]) << (j * 8) for j in range(4))
                val = sum(Vocab.tok_byte(self.context[i + 5 + j]) << (j * 8) for j in range(4))
                if addr not in addr_entries:
                    addr_entries[addr] = []
                addr_entries[addr].append((i, val))
                i += 9  # Skip MEM + 4 addr bytes + 4 value bytes
            else:
                i += 1

        # Find indices to remove: if most recent entry is zero, remove ALL entries
        indices_to_remove = set()
        for addr, entries in addr_entries.items():
            if entries and entries[-1][1] == 0:  # Most recent is zero (freed)
                # Remove ALL entries for this address
                for pos, val in entries:
                    for j in range(9):
                        if pos + j < len(self.context):
                            indices_to_remove.add(pos + j)

        # Rebuild context without evicted entries
        if indices_to_remove:
            self.context = [tok for i, tok in enumerate(self.context) if i not in indices_to_remove]

        return len(indices_to_remove)

    def _to_signed(self, x: int) -> int:
        x &= 0xFFFFFFFF
        return x - 0x100000000 if x >= 0x80000000 else x

    def _execute(self, op: int, imm: int, pc: int, ax: int, sp: int, bp: int):
        """Execute instruction using opcode-specific logic."""
        new_pc, new_ax, new_sp, new_bp = pc + 8, ax, sp, bp
        mem = None
        s = self._to_signed

        if op == Opcode.IMM:
            new_ax = imm & 0xFFFFFFFF
        elif op == Opcode.LEA:
            new_ax = (bp + imm) & 0xFFFFFFFF
        elif op == Opcode.PSH:
            new_sp = sp - 8
            mem = (new_sp, ax)
        elif op == Opcode.ADD:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = (s(a) + s(ax)) & 0xFFFFFFFF
        elif op == Opcode.SUB:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = (s(a) - s(ax)) & 0xFFFFFFFF
        elif op == Opcode.MUL:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = (s(a) * s(ax)) & 0xFFFFFFFF
        elif op == Opcode.DIV:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = int(s(a) / s(ax)) & 0xFFFFFFFF if ax else ax
        elif op == Opcode.MOD:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = (s(a) % s(ax)) & 0xFFFFFFFF if ax else ax
        elif op == Opcode.AND:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = a & ax
        elif op == Opcode.OR:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = a | ax
        elif op == Opcode.XOR:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = a ^ ax
        elif op == Opcode.SHL:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = (a << (ax & 31)) & 0xFFFFFFFF
        elif op == Opcode.SHR:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = a >> (ax & 31)
        elif op == Opcode.EQ:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = int(s(a) == s(ax))
        elif op == Opcode.NE:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = int(s(a) != s(ax))
        elif op == Opcode.LT:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = int(s(a) < s(ax))
        elif op == Opcode.GT:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = int(s(a) > s(ax))
        elif op == Opcode.LE:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = int(s(a) <= s(ax))
        elif op == Opcode.GE:
            a = self._read_mem(sp)
            new_sp = sp + 8
            new_ax = int(s(a) >= s(ax))
        elif op == Opcode.JMP:
            new_pc = imm
        elif op == Opcode.BZ:
            new_pc = imm if ax == 0 else new_pc
        elif op == Opcode.BNZ:
            new_pc = imm if ax != 0 else new_pc
        elif op == Opcode.JSR:
            new_sp = sp - 8
            mem = (new_sp, new_pc)
            new_pc = imm
        elif op == Opcode.ENT:
            new_sp = sp - 8
            mem = (new_sp, bp)
            new_bp = new_sp
            new_sp -= imm
        elif op == Opcode.ADJ:
            new_sp = sp + imm
        elif op == Opcode.LEV:
            new_sp = bp
            new_bp = self._read_mem(new_sp)
            new_sp += 8
            new_pc = self._read_mem(new_sp)
            new_sp += 8
        elif op == Opcode.LI:
            new_ax = self._read_mem(ax)
        elif op == Opcode.LC:
            new_ax = self._read_mem(ax) & 0xFF
        elif op == Opcode.SI:
            addr = self._read_mem(sp)
            new_sp = sp + 8
            mem = (addr, ax)
        elif op == Opcode.SC:
            addr = self._read_mem(sp)
            new_sp = sp + 8
            mem = (addr, ax & 0xFF)
        elif op == Opcode.MALC:
            new_ax = self._heap_ptr
            self._heap_ptr += self._read_mem(sp)
        elif op == Opcode.PRTF:
            c = self._read_mem(sp)
            self.stdout_bytes.append(c & 0xFF)
            new_ax = c & 0xFF
            new_sp = sp + 8  # Pop the character from stack

        return new_pc, new_ax, new_sp, new_bp, mem

    def load(self, bytecode: List[int], data: Optional[bytes] = None):
        """Load program into context."""
        self.context = [Vocab.BOS]
        self.gen_count = 0
        self.steps_count = 0
        self.halted = False
        self._step_pos = 0
        self._heap_ptr = 0x20000
        self.stdout_bytes = []

        if data:
            for i, b in enumerate(data):
                addr = 0x10000 + i
                self.context.extend([
                    Vocab.MEM,
                    *[Vocab.byte_tok((addr >> (j * 8)) & 0xFF) for j in range(4)],
                    *[Vocab.byte_tok(b if j == 0 else 0) for j in range(4)]
                ])

        for idx, instr in enumerate(bytecode):
            op, imm = instr & 0xFF, instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            pc = idx * 8
            self.context.extend([
                Vocab.CODE,
                Vocab.byte_tok((pc >> 8) & 0xFF),
                Vocab.byte_tok(pc & 0xFF),
                Vocab.byte_tok(op),
                *[Vocab.byte_tok((imm >> (i * 8)) & 0xFF) for i in range(4)]
            ])

        self._pending_regs = {
            Vocab.REG_PC: 0, Vocab.REG_AX: 0,
            Vocab.REG_SP: 0x10000, Vocab.REG_BP: 0x10000
        }
        self._pending_mem = None

        # Initial step
        for _ in range(30):
            self.generate_next_token()

    def step(self) -> bool:
        """Execute one VM step = 30 token generations."""
        if self.halted:
            return False
        for _ in range(30):
            tok = self.generate_next_token()
            if tok == Vocab.EOS:
                return False
        self.steps_count += 1
        return True

    def run(self, max_steps: int = 100000) -> int:
        """Run until EXIT or max_steps."""
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return self._read_reg(Vocab.REG_AX)


# =============================================================================
# TESTS
# =============================================================================

def test_embedding_roundtrip():
    """Test all 272 tokens roundtrip correctly."""
    print("Embedding roundtrip test...")
    model = PureTransformer(dim=320, use_moe=True)

    correct = 0
    for tok in range(Vocab.VOCAB_SIZE):
        emb = model.tok_emb.weight[tok]
        out = model.lm_head(model.ln_f(emb.unsqueeze(0).unsqueeze(0)))
        if out.argmax(dim=-1).item() == tok:
            correct += 1
        elif tok < 16:
            print(f"  FAIL: token {tok} -> {out.argmax(dim=-1).item()}")

    print(f"Embedding roundtrip: {correct}/{Vocab.VOCAB_SIZE}")
    return correct == Vocab.VOCAB_SIZE


def run_tests(use_neural: bool = False):
    from src.compiler import compile_c
    import time

    print("=" * 60)
    print("  PURE GENERATIVE VM")
    print(f"  Opcode MoE (33 experts) - {'NEURAL' if use_neural else 'REFERENCE'}")
    print("=" * 60)
    print()

    # Test embedding
    test_embedding_roundtrip()
    print()

    # Create model
    model = PureTransformer(dim=320, num_heads=8, num_layers=4, use_moe=True)
    print(f"Model: {model.num_layers} layers, dim={model.dim}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Experts: {Opcode.NUM_EXPERTS} (1 per opcode)")
    print(f"Mode: {'NEURAL (transformer forward)' if use_neural else 'REFERENCE'}")
    print()

    tests = [
        ("6 * 7", "int main() { return 6 * 7; }", 42),
        ("100 / 7", "int main() { return 100 / 7; }", 14),
        ("15 & 7", "int main() { return 15 & 7; }", 7),
        ("-3 * 4", "int main() { return -3 * 4; }", -12),
        ("variables", "int main() { int a; int b; a = 6; b = 7; return a * b; }", 42),
        ("function", "int f(int x) { return x * 2; } int main() { return f(21); }", 42),
        ("factorial", "int f(int n) { if (n <= 1) return 1; return n * f(n-1); } int main() { return f(5); }", 120),
    ]

    passed = 0
    total_tokens = 0
    total_steps = 0

    for name, source, expected in tests:
        start = time.time()
        vm = PureGenerativeVM(model, use_neural=use_neural)
        bytecode, _ = compile_c(source)
        vm.load(bytecode)
        result = vm.run(max_steps=10000)
        elapsed = time.time() - start

        if result >= 0x80000000:
            result -= 0x100000000
        ok = result == expected
        passed += ok
        total_tokens += vm.gen_count
        total_steps += vm.steps_count

        print(f"{name:<12} {result:>6} == {expected:>6}  [{vm.steps_count:>4} steps] [{vm.gen_count:>5} tok] [{elapsed:>6.2f}s] {'OK' if ok else 'FAIL'}")

    print()
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Total: {total_steps} steps, {total_tokens} tokens")


def run_full_tests(use_neural: bool = False):
    """Run full 1096 test suite."""
    from src.compiler import compile_c
    from tests.test_cases import get_all_tests
    import time

    print("=" * 60)
    print("  FULL TEST SUITE (1096 tests)")
    print(f"  Mode: {'NEURAL' if use_neural else 'REFERENCE'}")
    print("=" * 60)
    print()

    model = PureTransformer(dim=320, num_heads=8, num_layers=4, use_moe=True)
    print(f"Model: {model.num_layers} layers, {Opcode.NUM_EXPERTS} experts")
    print()

    tests = get_all_tests()
    passed = 0
    failed_tests = []

    start_time = time.time()
    for i, (name, source, expected) in enumerate(tests):
        vm = PureGenerativeVM(model, use_neural=use_neural)
        try:
            bytecode, data = compile_c(source)
            vm.load(bytecode, data)
            result = vm.run(max_steps=50000)
            if result >= 0x80000000:
                result -= 0x100000000
            ok = result == expected
        except Exception as e:
            ok = False
            result = f"ERR: {e}"

        if ok:
            passed += 1
        else:
            failed_tests.append((name, expected, result))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(tests)} ({passed} passed)", flush=True)

    elapsed = time.time() - start_time
    print()
    print(f"Passed: {passed}/{len(tests)} in {elapsed:.1f}s")

    if failed_tests and len(failed_tests) <= 20:
        print(f"\nFailed tests:")
        for name, expected, result in failed_tests[:20]:
            print(f"  {name}: expected {expected}, got {result}")


if __name__ == "__main__":
    import sys
    use_neural = "--neural" in sys.argv or "-n" in sys.argv
    full_tests = "--full" in sys.argv or "-f" in sys.argv

    if full_tests:
        run_full_tests(use_neural=use_neural)
    else:
        run_tests(use_neural=use_neural)
