"""LEA Opcode Correction Utility

The LEA opcode (Load Effective Address: AX = BP + immediate) has neural prediction
issues due to dimension amplification across layers. This module provides arithmetic
correction for LEA predictions.

Usage:
    from neural_vm.lea_correction import correct_lea_prediction

    # In your prediction loop:
    predicted_ax_byte = model_prediction
    corrected_ax_byte = correct_lea_prediction(context, draft_tokens, predicted_ax_byte)
"""

from .vm_step import Token
from .constants import PC_OFFSET, INSTR_WIDTH


def correct_lea_prediction(context, draft_tokens, predicted_ax_byte0):
    """Correct LEA prediction using arithmetic.

    LEA computes AX = BP + immediate. When LEA is detected, this function
    computes the correct result arithmetically instead of using the neural prediction.

    NOTE: The draft_tokens contain the state AFTER executing the instruction,
    so PC has already advanced. We need to look at the PREVIOUS instruction
    (PC - INSTR_WIDTH) to check if it was LEA.

    Args:
        context: list of token IDs (context before current step)
        draft_tokens: list of 35 tokens for current step (REG outputs)
        predicted_ax_byte0: neural VM's prediction for AX byte 0

    Returns:
        corrected AX byte 0 value (same as predicted if not LEA or if parsing fails)
    """
    try:
        # 1. Parse PC from draft tokens (positions 1-4: little-endian)
        # This is the PC AFTER executing the instruction
        if len(draft_tokens) < 20:
            return predicted_ax_byte0

        pc_bytes = draft_tokens[1:5]
        pc_after = pc_bytes[0] | (pc_bytes[1] << 8) | (pc_bytes[2] << 16) | (pc_bytes[3] << 24)

        # 2. Calculate PC BEFORE executing (PC - INSTR_WIDTH)
        if pc_after < PC_OFFSET + INSTR_WIDTH:
            return predicted_ax_byte0
        pc_before = pc_after - INSTR_WIDTH

        # 3. Find CODE_START in context to locate bytecode
        code_start_idx = None
        for i in range(len(context) - 1, -1, -1):
            if context[i] == Token.CODE_START:
                code_start_idx = i
                break

        if code_start_idx is None:
            return predicted_ax_byte0

        # 4. Calculate instruction index from PC_BEFORE
        if pc_before < PC_OFFSET:
            return predicted_ax_byte0
        idx = (pc_before - PC_OFFSET) // INSTR_WIDTH

        # 5. Parse instruction at idx (5 bytes: opcode + 4-byte immediate)
        instr_offset = code_start_idx + 1 + idx * 5
        if instr_offset + 5 > len(context):
            return predicted_ax_byte0

        opcode = context[instr_offset]
        imm_bytes = context[instr_offset + 1:instr_offset + 5]

        # 6. Check if opcode is LEA (opcode 0)
        if opcode != 0:
            return predicted_ax_byte0  # Not LEA, use neural prediction

        # 7. Parse immediate (little-endian 32-bit)
        immediate = (imm_bytes[0] |
                     (imm_bytes[1] << 8) |
                     (imm_bytes[2] << 16) |
                     (imm_bytes[3] << 24))

        # 8. Get BP from draft tokens (positions 16-19)
        bp_bytes = draft_tokens[16:20]
        bp = (bp_bytes[0] |
              (bp_bytes[1] << 8) |
              (bp_bytes[2] << 16) |
              (bp_bytes[3] << 24))

        # 9. Compute LEA: AX = BP + immediate (mod 2^32)
        correct_ax = (bp + immediate) & 0xFFFFFFFF
        correct_ax_byte0 = correct_ax & 0xFF

        return correct_ax_byte0

    except (IndexError, KeyError, AttributeError):
        # If parsing fails for any reason, return original prediction
        return predicted_ax_byte0


def is_lea_instruction(context, draft_tokens):
    """Check if the just-executed instruction is LEA.

    NOTE: draft_tokens contain state AFTER execution, so we check PC - INSTR_WIDTH.

    Args:
        context: list of token IDs (context before current step)
        draft_tokens: list of 35 tokens for current step

    Returns:
        bool: True if the just-executed instruction is LEA, False otherwise
    """
    try:
        if len(draft_tokens) < 5:
            return False

        pc_bytes = draft_tokens[1:5]
        pc_after = pc_bytes[0] | (pc_bytes[1] << 8) | (pc_bytes[2] << 16) | (pc_bytes[3] << 24)

        if pc_after < PC_OFFSET + INSTR_WIDTH:
            return False
        pc_before = pc_after - INSTR_WIDTH

        code_start_idx = None
        for i in range(len(context) - 1, -1, -1):
            if context[i] == Token.CODE_START:
                code_start_idx = i
                break

        if code_start_idx is None:
            return False

        if pc_before < PC_OFFSET:
            return False
        idx = (pc_before - PC_OFFSET) // INSTR_WIDTH

        instr_offset = code_start_idx + 1 + idx * 5
        if instr_offset >= len(context):
            return False

        opcode = context[instr_offset]
        return opcode == 0  # LEA is opcode 0

    except (IndexError, KeyError, AttributeError):
        return False
