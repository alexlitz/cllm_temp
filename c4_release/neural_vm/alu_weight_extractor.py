"""
ALU Weight Extractor

Extracts FFN weights from the real ALU implementations (alu/ops/*.py)
and formats them for loading into AutoregressiveVM layers.

The real ALU operations use 3-layer pipelines:
- Layer 1: Compute raw results + generate flags
- Layer 2: Carry/borrow lookahead (cross-position)
- Layer 3: Finalize with carries/borrows

This module bridges the chunk-generic ALU with the AutoregressiveVM architecture.
"""

import torch
from typing import Dict, Tuple
from dataclasses import dataclass

from .alu.chunk_config import NIBBLE
from .alu.ops.add import AddRawAndGenFFN, AddCarryLookaheadFFN
from .alu.ops.sub import SubRawAndGenFFN, SubBorrowLookaheadFFN
from .alu.ops.mul import build_mul_layers
from .alu.ops.shift import build_shl_layers, build_shr_layers
from .alu.ops.div import build_div_layers
from .alu.ops.mod import build_mod_layers
from .alu.ops.common import GenericE, GenericPureFFN, GenericFlattenedFFN
from .embedding import Opcode


@dataclass
class ThreeLayerWeights:
    """Weights for a 3-layer ALU operation."""
    layer1: Dict[str, torch.Tensor]  # Raw computation + generate
    layer2: Dict[str, torch.Tensor]  # Carry lookahead
    layer3: Dict[str, torch.Tensor]  # Finalize (if needed)


@dataclass
class MultiLayerWeights:
    """Weights for a multi-layer ALU operation (variable number of layers)."""
    layers: list[Dict[str, torch.Tensor]]  # List of layer weights


class ALUWeightExtractor:
    """Extract weights from chunk-generic ALU implementations."""

    def __init__(self):
        self.config = NIBBLE
        self.ge = GenericE(self.config)

        # Mapping from GenericE slots to our E slots
        # GenericE: NIB_A=0, NIB_B=1, RAW_SUM=2, CARRY_IN=3, CARRY_OUT=4, RESULT=5, TEMP=6, OP_START=7
        # Our E: NIB_A=0, NIB_B=1, (bits 2-9), RAW_SUM=10, CARRY_IN=11, CARRY_OUT=12, RESULT=13, TEMP=14, OPCODE=15, OP_START=16
        from .embedding import E
        self.slot_map = {
            self.ge.NIB_A: E.NIB_A,          # 0 -> 0
            self.ge.NIB_B: E.NIB_B,          # 1 -> 1
            self.ge.RAW_SUM: E.RAW_SUM,      # 2 -> 10
            self.ge.CARRY_IN: E.CARRY_IN,    # 3 -> 11
            self.ge.CARRY_OUT: E.CARRY_OUT,  # 4 -> 12
            self.ge.RESULT: E.RESULT,        # 5 -> 13
            self.ge.TEMP: E.TEMP,            # 6 -> 14
        }
        # OP_START mapping: GenericE.OP_START + i -> E.OP_START + i
        self.op_start_ge = self.ge.OP_START
        self.op_start_e = E.OP_START

        self.dim_per_pos_ge = self.ge.DIM      # 160
        self.dim_per_pos_e = E.DIM              # 169
        self.num_positions = self.ge.NUM_POSITIONS  # 8

    def extract_add_weights(self, opcode: int = Opcode.ADD) -> ThreeLayerWeights:
        """Extract 3-layer ADD weights.

        Args:
            opcode: Opcode for gating (default: Opcode.ADD)

        Returns:
            ThreeLayerWeights with layer1 and layer2 weights
        """
        # Create the ADD layers
        layer1 = AddRawAndGenFFN(self.ge, opcode=opcode)
        layer2 = AddCarryLookaheadFFN(self.ge, opcode=opcode)

        # Extract layer 1 weights (PureFFN per-position)
        layer1_ffn = layer1.ffn
        layer1_weights = self._extract_pure_ffn_weights(layer1_ffn)

        # Extract layer 2 weights (FlattenedFFN cross-position)
        layer2_ffn = layer2.flat_ffn
        layer2_weights = self._extract_flattened_ffn_weights(layer2_ffn)

        # Layer 3 is implicit: result = (RAW_SUM + CARRY_IN) mod 16
        # This can be done with a simple cancel pair for each position
        layer3_weights = self._create_finalize_weights(opcode)

        return ThreeLayerWeights(
            layer1=layer1_weights,
            layer2=layer2_weights,
            layer3=layer3_weights
        )

    def extract_sub_weights(self, opcode: int = Opcode.SUB) -> ThreeLayerWeights:
        """Extract 3-layer SUB weights.

        Args:
            opcode: Opcode for gating (default: Opcode.SUB)

        Returns:
            ThreeLayerWeights with layer1 and layer2 weights
        """
        # Create the SUB layers
        layer1 = SubRawAndGenFFN(self.ge, opcode=opcode)
        layer2 = SubBorrowLookaheadFFN(self.ge, opcode=opcode)

        # Extract layer 1 weights (PureFFN per-position)
        layer1_ffn = layer1.ffn
        layer1_weights = self._extract_pure_ffn_weights(layer1_ffn)

        # Extract layer 2 weights (FlattenedFFN cross-position)
        layer2_ffn = layer2.flat_ffn
        layer2_weights = self._extract_flattened_ffn_weights(layer2_ffn)

        # Layer 3: Finalize (RESULT = RAW_SUM - CARRY_IN + base when borrow)
        # For SUB, this is slightly different - need to add base when there's a borrow
        layer3_weights = self._create_sub_finalize_weights(opcode)

        return ThreeLayerWeights(
            layer1=layer1_weights,
            layer2=layer2_weights,
            layer3=layer3_weights
        )

    def _extract_pure_ffn_weights(self, ffn: GenericPureFFN) -> Dict[str, torch.Tensor]:
        """Extract weights from GenericPureFFN (per-position).

        GenericPureFFN shape: [batch, seq_pos, dim] where dim=160
        AutoregressiveVM PureFFN expects: [batch, seq_tokens, d_model] where d_model=1280

        We need to flatten the position dimension into the model dimension:
        - Input: [batch, seq=1, 1280] (8 positions × 160 dims each)
        - Hidden: [batch, seq=1, hidden_dim]

        Strategy: Expand GenericPureFFN weights from per-position to flattened.
        """
        # Get weights from GenericPureFFN
        # Shape: W_up [hidden, dim], W_gate [hidden, dim], W_down [dim, hidden]
        # where dim=160 (per position)

        W_up_pos = ffn.W_up.data  # [hidden_pos, 160]
        W_gate_pos = ffn.W_gate.data  # [hidden_pos, 160]
        W_down_pos = ffn.W_down.data  # [160, hidden_pos]
        b_up_pos = ffn.b_up.data  # [hidden_pos]

        num_positions = self.config.num_positions  # 8
        dim_per_pos = self.ge.DIM  # 160
        d_model = num_positions * dim_per_pos  # 1280
        hidden_per_pos = W_up_pos.shape[0]
        total_hidden = num_positions * hidden_per_pos

        # Create flattened weights for AutoregressiveVM
        W_up = torch.zeros(total_hidden, d_model, dtype=W_up_pos.dtype)
        b_up = torch.zeros(total_hidden, dtype=b_up_pos.dtype)
        W_gate = torch.zeros(total_hidden, d_model, dtype=W_gate_pos.dtype)
        W_down = torch.zeros(d_model, total_hidden, dtype=W_down_pos.dtype)

        # Copy per-position weights into flattened structure
        for pos in range(num_positions):
            # Hidden unit range for this position
            h_start = pos * hidden_per_pos
            h_end = (pos + 1) * hidden_per_pos

            # Input dimension range for this position
            d_start = pos * dim_per_pos
            d_end = (pos + 1) * dim_per_pos

            # Copy weights
            W_up[h_start:h_end, d_start:d_end] = W_up_pos
            b_up[h_start:h_end] = b_up_pos
            W_gate[h_start:h_end, d_start:d_end] = W_gate_pos
            W_down[d_start:d_end, h_start:h_end] = W_down_pos

        return {
            'W_up': W_up,
            'b_up': b_up,
            'W_gate': W_gate,
            'b_gate': torch.zeros_like(b_up),  # No gate bias in GenericPureFFN
            'W_down': W_down,
            'b_down': torch.zeros(d_model, dtype=W_down.dtype),  # No down bias
        }

    def _extract_flattened_ffn_weights(self, ffn: GenericFlattenedFFN) -> Dict[str, torch.Tensor]:
        """Extract weights from GenericFlattenedFFN (cross-position).

        GenericFlattenedFFN operates on flattened [batch, num_pos * dim] = [batch, 1280]
        This matches AutoregressiveVM format when seq_len=1.
        """
        # Flattened FFN already operates on full 1280-dim space
        # Just need to add seq dimension handling

        W_up = ffn.W_up.data  # [hidden, 1280]
        b_up = ffn.b_up.data  # [hidden]
        W_gate = ffn.W_gate.data  # [hidden, 1280]
        W_down = ffn.W_down.data  # [1280, hidden]

        return {
            'W_up': W_up,
            'b_up': b_up,
            'W_gate': W_gate,
            'b_gate': torch.zeros_like(b_up),
            'W_down': W_down,
            'b_down': torch.zeros(W_down.shape[0], dtype=W_down.dtype),
        }

    def _create_finalize_weights(self, opcode: int) -> Dict[str, torch.Tensor]:
        """Create weights for finalization layer.

        Computes: RESULT = (RAW_SUM + CARRY_IN) mod 16

        For each position:
        - Read RAW_SUM and CARRY_IN
        - Add them
        - Modulo 16 (automatic via nibble representation)
        - Write to RESULT
        """
        num_positions = self.config.num_positions
        dim_per_pos = self.ge.DIM
        d_model = num_positions * dim_per_pos

        # Need 4 units per position (2 cancel pairs for add)
        hidden = num_positions * 4
        S = self.config.scale

        W_up = torch.zeros(hidden, d_model)
        b_up = torch.zeros(hidden)
        W_gate = torch.zeros(hidden, d_model)
        W_down = torch.zeros(d_model, hidden)

        for pos in range(num_positions):
            h_base = pos * 4
            d_base = pos * dim_per_pos

            # Cancel pair 1: +RAW_SUM, +CARRY_IN
            W_up[h_base, d_base + self.ge.RAW_SUM] = S
            W_up[h_base, d_base + self.ge.CARRY_IN] = S
            W_gate[h_base, d_base + self.ge.OP_START + opcode] = S
            W_down[d_base + self.ge.RESULT, h_base] = 1.0 / (S * S)

            # Cancel pair 2: -RAW_SUM, -CARRY_IN
            W_up[h_base + 1, d_base + self.ge.RAW_SUM] = -S
            W_up[h_base + 1, d_base + self.ge.CARRY_IN] = -S
            W_gate[h_base + 1, d_base + self.ge.OP_START + opcode] = -S
            W_down[d_base + self.ge.RESULT, h_base + 1] = 1.0 / (S * S)

        return {
            'W_up': W_up,
            'b_up': b_up,
            'W_gate': W_gate,
            'b_gate': torch.zeros_like(b_up),
            'W_down': W_down,
            'b_down': torch.zeros(d_model),
        }

    def _remap_flattened_weights(self, weights_ge: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remap weights from GenericE layout (1280) to our E layout (1352).

        Args:
            weights_ge: Weights in GenericE format (d_model=1280, dim_per_pos=160)

        Returns:
            Weights in our E format (d_model=1352, dim_per_pos=169)
        """
        W_up_ge = weights_ge['W_up']      # [hidden, 1280]
        b_up_ge = weights_ge['b_up']      # [hidden]
        W_gate_ge = weights_ge['W_gate']  # [hidden, 1280]
        b_gate_ge = weights_ge['b_gate']  # [hidden]
        W_down_ge = weights_ge['W_down']  # [1280, hidden]
        b_down_ge = weights_ge['b_down']  # [1280]

        hidden = W_up_ge.shape[0]
        d_model_e = self.num_positions * self.dim_per_pos_e  # 1352

        # Create expanded weight matrices
        W_up_e = torch.zeros(hidden, d_model_e, dtype=W_up_ge.dtype)
        W_gate_e = torch.zeros(hidden, d_model_e, dtype=W_gate_ge.dtype)
        W_down_e = torch.zeros(d_model_e, hidden, dtype=W_down_ge.dtype)
        b_down_e = torch.zeros(d_model_e, dtype=b_down_ge.dtype)

        # Remap each position
        for pos in range(self.num_positions):
            ge_start = pos * self.dim_per_pos_ge
            e_start = pos * self.dim_per_pos_e

            # Map each slot
            for slot_ge, slot_e in self.slot_map.items():
                ge_idx = ge_start + slot_ge
                e_idx = e_start + slot_e

                # Copy weights for this slot
                W_up_e[:, e_idx] = W_up_ge[:, ge_idx]
                W_gate_e[:, e_idx] = W_gate_ge[:, ge_idx]
                W_down_e[e_idx, :] = W_down_ge[ge_idx, :]
                b_down_e[e_idx] = b_down_ge[ge_idx]

            # Map opcode slots (OP_START to OP_START+NUM_OPS)
            for op_offset in range(self.ge.NUM_OPS):
                ge_idx = ge_start + self.op_start_ge + op_offset
                e_idx = e_start + self.op_start_e + op_offset

                W_up_e[:, e_idx] = W_up_ge[:, ge_idx]
                W_gate_e[:, e_idx] = W_gate_ge[:, ge_idx]
                W_down_e[e_idx, :] = W_down_ge[ge_idx, :]
                b_down_e[e_idx] = b_down_ge[ge_idx]

            # Map any other slots (like POS=79, SLOT_REMAINDER=8, SLOT_QUOTIENT=9, etc.)
            # For now, map all remaining slots from GenericE to E
            for slot_ge in range(self.dim_per_pos_ge):
                # Skip slots we've already mapped
                if slot_ge in self.slot_map or (self.op_start_ge <= slot_ge < self.op_start_ge + self.ge.NUM_OPS):
                    continue

                ge_idx = ge_start + slot_ge
                e_idx = e_start + slot_ge  # Keep same relative position for unmapped slots

                # Only copy if e_idx is valid
                if slot_ge < self.dim_per_pos_e:
                    W_up_e[:, e_idx] = W_up_ge[:, ge_idx]
                    W_gate_e[:, e_idx] = W_gate_ge[:, ge_idx]
                    W_down_e[e_idx, :] = W_down_ge[ge_idx, :]
                    b_down_e[e_idx] = b_down_ge[ge_idx]

        return {
            'W_up': W_up_e,
            'b_up': b_up_ge,  # Bias doesn't need remapping
            'W_gate': W_gate_e,
            'b_gate': b_gate_ge,  # Bias doesn't need remapping
            'W_down': W_down_e,
            'b_down': b_down_e,
        }

    def _extract_layer_weights(self, layer_module) -> Dict[str, torch.Tensor]:
        """Extract weights from a layer module (either PureFFN or FlattenedFFN).

        Args:
            layer_module: Module with either .ffn (GenericPureFFN) or .flat_ffn (GenericFlattenedFFN)

        Returns:
            Dictionary of weights suitable for AutoregressiveVM (remapped to E layout), or None for non-FFN layers
        """
        # Check if it has flat_ffn (GenericFlattenedFFN)
        if hasattr(layer_module, 'flat_ffn'):
            weights_ge = self._extract_flattened_ffn_weights(layer_module.flat_ffn)
            return self._remap_flattened_weights(weights_ge)
        # Check if it has ffn (GenericPureFFN)
        elif hasattr(layer_module, 'ffn'):
            weights_ge = self._extract_pure_ffn_weights(layer_module.ffn)
            return self._remap_flattened_weights(weights_ge)
        # Some layers (like Softmax1ReciprocalModule, ModDivScalarModule) compute dynamically
        # These can't be represented as pure FFN weights
        # Return None to indicate this layer needs special handling
        else:
            return None

    def _create_sub_finalize_weights(self, opcode: int) -> Dict[str, torch.Tensor]:
        """Create weights for SUB finalization layer.

        Computes: RESULT = (RAW_SUM - CARRY_IN) mod 16

        For SUB, CARRY_IN represents borrow. The final result is RAW_SUM - borrow.

        For each position:
        - Read RAW_SUM and CARRY_IN (borrow)
        - Subtract borrow from RAW_SUM
        - Modulo 16 (automatic via nibble representation)
        - Write to RESULT
        """
        num_positions = self.config.num_positions
        dim_per_pos = self.ge.DIM
        d_model = num_positions * dim_per_pos

        # Need 4 units per position (2 cancel pairs for subtraction)
        hidden = num_positions * 4
        S = self.config.scale

        W_up = torch.zeros(hidden, d_model)
        b_up = torch.zeros(hidden)
        W_gate = torch.zeros(hidden, d_model)
        W_down = torch.zeros(d_model, hidden)

        for pos in range(num_positions):
            h_base = pos * 4
            d_base = pos * dim_per_pos

            # Cancel pair 1: +RAW_SUM, -CARRY_IN (borrow)
            W_up[h_base, d_base + self.ge.RAW_SUM] = S
            W_up[h_base, d_base + self.ge.CARRY_IN] = -S  # Subtract borrow
            W_gate[h_base, d_base + self.ge.OP_START + opcode] = S
            W_down[d_base + self.ge.RESULT, h_base] = 1.0 / (S * S)

            # Cancel pair 2: -RAW_SUM, +CARRY_IN (borrow)
            W_up[h_base + 1, d_base + self.ge.RAW_SUM] = -S
            W_up[h_base + 1, d_base + self.ge.CARRY_IN] = S  # Add borrow (for cancellation)
            W_gate[h_base + 1, d_base + self.ge.OP_START + opcode] = -S
            W_down[d_base + self.ge.RESULT, h_base + 1] = 1.0 / (S * S)

        return {
            'W_up': W_up,
            'b_up': b_up,
            'W_gate': W_gate,
            'b_gate': torch.zeros_like(b_up),
            'W_down': W_down,
            'b_down': torch.zeros(d_model),
        }

    def extract_mul_weights(self, opcode: int = Opcode.MUL) -> MultiLayerWeights:
        """Extract weights for multi-layer MUL operation.

        Args:
            opcode: Opcode for gating (default: Opcode.MUL)

        Returns:
            MultiLayerWeights with 5-7 layers depending on config
        """
        layers_list = build_mul_layers(self.config, opcode=opcode)
        layer_weights = []

        for layer_module in layers_list:
            weights = self._extract_layer_weights(layer_module)
            layer_weights.append(weights)

        return MultiLayerWeights(layers=layer_weights)

    def extract_shl_weights(self, opcode: int = Opcode.SHL) -> MultiLayerWeights:
        """Extract weights for multi-layer SHL operation.

        Args:
            opcode: Opcode for gating (default: Opcode.SHL)

        Returns:
            MultiLayerWeights with 1-2 layers (2 for chunk_bits > 1)
        """
        layers_list = build_shl_layers(self.config, opcode=opcode)
        layer_weights = []

        for layer_module in layers_list:
            weights = self._extract_layer_weights(layer_module)
            layer_weights.append(weights)

        return MultiLayerWeights(layers=layer_weights)

    def extract_shr_weights(self, opcode: int = Opcode.SHR) -> MultiLayerWeights:
        """Extract weights for multi-layer SHR operation.

        Args:
            opcode: Opcode for gating (default: Opcode.SHR)

        Returns:
            MultiLayerWeights with 1-2 layers (2 for chunk_bits > 1)
        """
        layers_list = build_shr_layers(self.config, opcode=opcode)
        layer_weights = []

        for layer_module in layers_list:
            weights = self._extract_layer_weights(layer_module)
            layer_weights.append(weights)

        return MultiLayerWeights(layers=layer_weights)

    def extract_div_weights(self, opcode: int = Opcode.DIV) -> MultiLayerWeights:
        """Extract weights for multi-layer DIV operation.

        Args:
            opcode: Opcode for gating (default: Opcode.DIV)

        Returns:
            MultiLayerWeights with 3-4 layers depending on config
        """
        layers_list = build_div_layers(self.config, opcode=opcode, fp32_floor=False)
        layer_weights = []

        for layer_module in layers_list:
            weights = self._extract_layer_weights(layer_module)
            layer_weights.append(weights)

        return MultiLayerWeights(layers=layer_weights)

    def extract_mod_weights(self, opcode: int = Opcode.MOD) -> MultiLayerWeights:
        """Extract weights for multi-layer MOD operation.

        Args:
            opcode: Opcode for gating (default: Opcode.MOD)

        Returns:
            MultiLayerWeights with 4-5 layers depending on config
        """
        layers_list = build_mod_layers(self.config, opcode=opcode)
        layer_weights = []

        for layer_module in layers_list:
            weights = self._extract_layer_weights(layer_module)
            layer_weights.append(weights)

        return MultiLayerWeights(layers=layer_weights)
