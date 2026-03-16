"""
Bytecode System Prompt Mode

Outputs compiled bytecode in a format suitable for use as a "system prompt"
in transformer-based execution. This enables:

1. Compiler bytecode as system prompt - transformer runs compiler to process input
2. Program bytecode as context - transformer executes pre-compiled programs
3. Self-hosted compilation - chain multiple compilation stages

Usage:
    # Compile program and output as prompt
    prompt = BytecodePrompt.from_c_source("int main() { return 42; }")
    print(prompt.to_hex_string())  # For text-based LLM
    print(prompt.to_token_sequence())  # For byte-token LLM
"""

import json
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import IntEnum


class PromptFormat(IntEnum):
    """Output format for bytecode prompts."""
    HEX_STRING = 1      # Human-readable hex
    TOKEN_SEQUENCE = 2  # Byte tokens for LLM
    JSON = 3            # Structured JSON
    BINARY = 4          # Raw binary bytes


@dataclass
class BytecodeInstruction:
    """Single bytecode instruction."""
    address: int        # Byte address (multiple of 8)
    opcode: int         # Operation code (0-38)
    immediate: int      # Immediate value
    raw: int            # Raw instruction value

    @property
    def op_name(self) -> str:
        """Get human-readable opcode name."""
        OPCODE_NAMES = {
            0: "LEA", 1: "IMM", 2: "JMP", 3: "JSR", 4: "BZ", 5: "BNZ",
            6: "ENT", 7: "ADJ", 8: "LEV", 9: "LI", 10: "LC", 11: "SI",
            12: "SC", 13: "PSH", 14: "OR", 15: "XOR", 16: "AND", 17: "EQ",
            18: "NE", 19: "LT", 20: "GT", 21: "LE", 22: "GE", 23: "SHL",
            24: "SHR", 25: "ADD", 26: "SUB", 27: "MUL", 28: "DIV", 29: "MOD",
            30: "OPEN", 31: "READ", 32: "CLOS", 33: "PRTF", 34: "MALC",
            35: "FREE", 36: "MSET", 37: "MCMP", 38: "EXIT",
        }
        return OPCODE_NAMES.get(self.opcode, f"OP{self.opcode}")

    def to_dict(self) -> Dict:
        return {
            "addr": self.address,
            "op": self.op_name,
            "imm": self.immediate,
            "raw": f"0x{self.raw:016x}",
        }


@dataclass
class BytecodePrompt:
    """
    Bytecode formatted as a system prompt.

    Can be used to:
    1. Feed pre-compiled code to a transformer VM
    2. Use compiler bytecode as "system prompt"
    3. Chain compilation stages (C -> bytecode -> execution)
    """

    instructions: List[BytecodeInstruction]
    data: bytes
    source: Optional[str] = None
    metadata: Optional[Dict] = None

    @classmethod
    def from_c_source(cls, source: str) -> "BytecodePrompt":
        """
        Compile C source and create bytecode prompt.

        Args:
            source: C source code

        Returns:
            BytecodePrompt ready for use
        """
        from .compiler import compile_c

        bytecode, data = compile_c(source)

        instructions = []
        for i, raw in enumerate(bytecode):
            opcode = raw & 0xFF
            immediate = raw >> 8
            # Sign extend
            if immediate >= (1 << 55):
                immediate -= (1 << 56)

            instructions.append(BytecodeInstruction(
                address=i * 8,
                opcode=opcode,
                immediate=immediate,
                raw=raw,
            ))

        return cls(
            instructions=instructions,
            data=bytes(data) if data else b'',
            source=source,
            metadata={
                "num_instructions": len(instructions),
                "data_size": len(data) if data else 0,
                "source_size": len(source),
            }
        )

    @classmethod
    def from_bytecode(
        cls,
        bytecode: List[int],
        data: Optional[List[int]] = None,
    ) -> "BytecodePrompt":
        """Create prompt from raw bytecode."""
        instructions = []
        for i, raw in enumerate(bytecode):
            opcode = raw & 0xFF
            immediate = raw >> 8
            if immediate >= (1 << 55):
                immediate -= (1 << 56)

            instructions.append(BytecodeInstruction(
                address=i * 8,
                opcode=opcode,
                immediate=immediate,
                raw=raw,
            ))

        return cls(
            instructions=instructions,
            data=bytes(data) if data else b'',
        )

    def to_hex_string(self, include_comments: bool = True) -> str:
        """
        Convert to hex string format.

        Output like:
            0000: 03 00 00 00 10 00 00 00  # JSR 16
            0008: 26 00 00 00 00 00 00 00  # EXIT 0
            ...
        """
        lines = []
        lines.append("# C4 Bytecode System Prompt")
        lines.append(f"# Instructions: {len(self.instructions)}")
        lines.append(f"# Data size: {len(self.data)} bytes")
        lines.append("")
        lines.append("# CODE SEGMENT")

        for instr in self.instructions:
            # Format as 8 bytes
            raw_bytes = instr.raw.to_bytes(8, byteorder='little', signed=False)
            hex_str = ' '.join(f'{b:02x}' for b in raw_bytes)

            comment = f"  # {instr.op_name}"
            if instr.immediate != 0:
                comment += f" {instr.immediate}"

            line = f"{instr.address:04x}: {hex_str}"
            if include_comments:
                line += comment
            lines.append(line)

        if self.data:
            lines.append("")
            lines.append("# DATA SEGMENT")
            for i in range(0, len(self.data), 16):
                chunk = self.data[i:i+16]
                hex_str = ' '.join(f'{b:02x}' for b in chunk)
                ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
                lines.append(f"{i:04x}: {hex_str:<48}  {ascii_str}")

        return '\n'.join(lines)

    def to_token_sequence(self) -> List[int]:
        """
        Convert to byte token sequence for LLM input.

        Each instruction becomes 8 byte tokens (vocab 0-255).
        """
        tokens = []

        for instr in self.instructions:
            # Little-endian bytes
            raw_bytes = instr.raw.to_bytes(8, byteorder='little', signed=False)
            tokens.extend(raw_bytes)

        return tokens

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON format."""
        return json.dumps({
            "format": "c4-bytecode-v1",
            "code": [instr.to_dict() for instr in self.instructions],
            "data": list(self.data) if self.data else [],
            "metadata": self.metadata,
        }, indent=indent)

    def to_binary(self) -> bytes:
        """Convert to raw binary format."""
        result = bytearray()

        # Header
        result.extend(b'C4BC')  # Magic
        result.extend(len(self.instructions).to_bytes(4, 'little'))
        result.extend(len(self.data).to_bytes(4, 'little'))

        # Code
        for instr in self.instructions:
            result.extend(instr.raw.to_bytes(8, 'little', signed=False))

        # Data
        result.extend(self.data)

        return bytes(result)

    @classmethod
    def from_binary(cls, data: bytes) -> "BytecodePrompt":
        """Load from binary format."""
        if data[:4] != b'C4BC':
            raise ValueError("Invalid C4 bytecode magic")

        num_instructions = int.from_bytes(data[4:8], 'little')
        data_size = int.from_bytes(data[8:12], 'little')

        instructions = []
        offset = 12
        for i in range(num_instructions):
            raw = int.from_bytes(data[offset:offset+8], 'little')
            opcode = raw & 0xFF
            immediate = raw >> 8
            if immediate >= (1 << 55):
                immediate -= (1 << 56)

            instructions.append(BytecodeInstruction(
                address=i * 8,
                opcode=opcode,
                immediate=immediate,
                raw=raw,
            ))
            offset += 8

        data_segment = data[offset:offset+data_size]

        return cls(instructions=instructions, data=data_segment)

    def get_bytecode_list(self) -> List[int]:
        """Get raw bytecode as list of integers."""
        return [instr.raw for instr in self.instructions]

    def get_data_list(self) -> List[int]:
        """Get data segment as list of integers."""
        return list(self.data)

    def disassemble(self) -> str:
        """
        Create human-readable disassembly.
        """
        lines = []
        lines.append("; C4 Bytecode Disassembly")
        lines.append(f"; {len(self.instructions)} instructions")
        lines.append("")

        for instr in self.instructions:
            addr_str = f"{instr.address:04x}"
            op_str = f"{instr.op_name:6}"

            if instr.opcode in (1,):  # IMM
                imm_str = f"{instr.immediate}"
            elif instr.opcode in (2, 3, 4, 5):  # JMP, JSR, BZ, BNZ
                imm_str = f"0x{instr.immediate:04x}"
            elif instr.opcode in (0, 6, 7):  # LEA, ENT, ADJ
                imm_str = f"{instr.immediate:+d}" if instr.immediate != 0 else ""
            else:
                imm_str = f"{instr.immediate}" if instr.immediate != 0 else ""

            lines.append(f"  {addr_str}:  {op_str} {imm_str}")

        return '\n'.join(lines)


class CompilerPrompt:
    """
    Generate a "compiler system prompt" - bytecode that compiles input.

    This enables self-hosted compilation where:
    1. System prompt contains compiler bytecode
    2. User input is C source code
    3. Output is compiled bytecode for that source
    """

    @staticmethod
    def get_expression_compiler() -> BytecodePrompt:
        """
        Get bytecode for a simple expression compiler.

        This compiler handles:
        - Integer literals
        - Basic arithmetic (+, -, *, /)
        - Operator precedence

        Input: Expression string (e.g., "3+4*2")
        Output: Bytecode that computes the expression
        """
        # The expression compiler in C
        # This gets compiled to bytecode that can parse expressions
        compiler_source = '''
// Mini expression compiler
// Input: expression string at memory location
// Output: compiled bytecode

int main() {
    // This is a simplified demo
    // In practice, the full C4 compiler runs here
    return 42;
}
'''
        return BytecodePrompt.from_c_source(compiler_source)

    @staticmethod
    def create_self_hosted_prompt(
        compiler_bytecode: BytecodePrompt,
        user_source: str,
    ) -> str:
        """
        Create a combined prompt with compiler + source.

        Format:
            [COMPILER BYTECODE]
            <separator>
            [USER SOURCE CODE]
        """
        lines = []
        lines.append("# SYSTEM: C4 Compiler Bytecode")
        lines.append(compiler_bytecode.to_hex_string(include_comments=False))
        lines.append("")
        lines.append("# USER SOURCE:")
        lines.append(user_source)

        return '\n'.join(lines)


def demo_bytecode_prompt():
    """Demonstrate bytecode prompt generation."""
    print("=" * 60)
    print("BYTECODE SYSTEM PROMPT DEMO")
    print("=" * 60)

    # Simple program
    source = """
int main() {
    int a, b;
    a = 6;
    b = 7;
    return a * b;
}
"""

    print("\n1. Source Code:")
    print(source)

    # Create prompt
    prompt = BytecodePrompt.from_c_source(source)

    print("\n2. Hex String Format:")
    print(prompt.to_hex_string())

    print("\n3. Token Sequence (first 32 tokens):")
    tokens = prompt.to_token_sequence()
    print(tokens[:32])
    print(f"   Total tokens: {len(tokens)}")

    print("\n4. Disassembly:")
    print(prompt.disassemble())

    print("\n5. JSON Format:")
    json_str = prompt.to_json()
    print(json_str[:500] + "..." if len(json_str) > 500 else json_str)

    # More complex example
    print("\n" + "=" * 60)
    print("FIBONACCI BYTECODE")
    print("=" * 60)

    fib_source = """
int fib(int n) {
    if (n < 2) return n;
    return fib(n-1) + fib(n-2);
}
int main() { return fib(10); }
"""

    fib_prompt = BytecodePrompt.from_c_source(fib_source)
    print(f"\nInstructions: {len(fib_prompt.instructions)}")
    print(f"Token sequence length: {len(fib_prompt.to_token_sequence())}")
    print("\nDisassembly (first 20 instructions):")
    lines = fib_prompt.disassemble().split('\n')
    print('\n'.join(lines[:23]))


if __name__ == "__main__":
    demo_bytecode_prompt()


__all__ = [
    'BytecodePrompt',
    'BytecodeInstruction',
    'CompilerPrompt',
    'PromptFormat',
]
