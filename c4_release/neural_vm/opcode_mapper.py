#!/usr/bin/env python3
"""
C4 VM Opcode Mapper

Maps C4 compiler opcodes to computation graphs built from primitive operations.
This allows the graph weight compiler to support all C4 opcodes by composing
primitives rather than implementing each opcode separately.

C4 compiler emits 42 opcodes total:
- Core opcodes 0-38 (39 opcodes)
- I/O opcodes 64-66 (3 opcodes)

Opcodes 39-42 (NOP, POP, BLT, BGE) are neural VM extensions, not emitted by C4 compiler.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enum import IntEnum
from typing import Dict, List, Tuple, Optional
from neural_vm.graph_weight_compiler import OpType, ComputationGraph


class C4Opcode(IntEnum):
    """C4 VM opcodes (matching docs/OPCODE_TABLE.md)"""
    # Stack/Address
    LEA = 0      # AX = BP + imm
    IMM = 1      # AX = imm
    JMP = 2      # PC = imm
    JSR = 3      # push PC, PC = imm
    BZ = 4       # if AX==0: PC = imm
    BNZ = 5      # if AX!=0: PC = imm
    ENT = 6      # push BP, BP=SP, SP-=imm
    ADJ = 7      # SP += imm
    LEV = 8      # SP=BP, pop BP, pop PC

    # Memory
    LI = 9       # AX = *AX (load int)
    LC = 10      # AX = *(char*)AX (load char)
    SI = 11      # *pop = AX (store int)
    SC = 12      # *(char*)pop = AX (store char)
    PSH = 13     # push AX

    # Bitwise
    OR = 14      # AX = pop | AX
    XOR = 15     # AX = pop ^ AX
    AND = 16     # AX = pop & AX

    # Comparison
    EQ = 17      # AX = (pop == AX)
    NE = 18      # AX = (pop != AX)
    LT = 19      # AX = (pop < AX)
    GT = 20      # AX = (pop > AX)
    LE = 21      # AX = (pop <= AX)
    GE = 22      # AX = (pop >= AX)

    # Shift
    SHL = 23     # AX = pop << AX
    SHR = 24     # AX = pop >> AX

    # Arithmetic
    ADD = 25     # AX = pop + AX
    SUB = 26     # AX = pop - AX
    MUL = 27     # AX = pop * AX
    DIV = 28     # AX = pop / AX
    MOD = 29     # AX = pop % AX

    # System
    OPEN = 30    # AX = open(file)
    READ = 31    # AX = read(fd, buf, n)
    CLOS = 32    # close(fd)
    PRTF = 33    # printf(fmt, ...)
    MALC = 34    # AX = malloc(n)
    FREE = 35    # free(ptr)
    MSET = 36    # memset(p, v, n)
    MCMP = 37    # memcmp(a, b, n)
    EXIT = 38    # exit(code)

    # Control (neural VM extensions)
    NOP = 39     # no operation
    POP = 40     # SP += 8
    BLT = 41     # branch if signed <
    BGE = 42     # branch if signed >=

    # I/O
    GETCHAR = 64 # AX = getchar()
    PUTCHAR = 65 # putchar(AX)
    PRINTF2 = 66 # printf variant (unused)


class OpcodeSupport(IntEnum):
    """Level of support for each opcode"""
    FFN_PRIMITIVE = 1      # Direct FFN primitive (ADD, CMP_EQ, etc.)
    FFN_COMPOSITE = 2      # Composition of FFN primitives (LEA = ADD(BP, imm))
    ATTENTION_NEEDED = 3   # Requires attention (LI, LC, SI, SC, GETCHAR, PUTCHAR)
    TOOL_CALL = 4          # Requires tool call (OPEN, READ, PRTF)
    NOT_IMPLEMENTED = 5    # Not yet implemented


class OpcodeMapper:
    """Maps C4 opcodes to computation graphs."""

    def __init__(self):
        self.support_map = self._build_support_map()

    def _build_support_map(self) -> Dict[C4Opcode, OpcodeSupport]:
        """Build map of opcode support levels."""
        return {
            # Arithmetic - Direct FFN primitives
            C4Opcode.ADD: OpcodeSupport.FFN_PRIMITIVE,
            C4Opcode.SUB: OpcodeSupport.FFN_PRIMITIVE,
            C4Opcode.MUL: OpcodeSupport.FFN_PRIMITIVE,
            C4Opcode.DIV: OpcodeSupport.FFN_PRIMITIVE,
            C4Opcode.MOD: OpcodeSupport.FFN_PRIMITIVE,

            # Comparison - Direct FFN primitives
            C4Opcode.EQ: OpcodeSupport.FFN_PRIMITIVE,
            C4Opcode.NE: OpcodeSupport.FFN_PRIMITIVE,
            C4Opcode.LT: OpcodeSupport.FFN_PRIMITIVE,
            C4Opcode.GT: OpcodeSupport.FFN_PRIMITIVE,
            C4Opcode.LE: OpcodeSupport.FFN_PRIMITIVE,
            C4Opcode.GE: OpcodeSupport.FFN_PRIMITIVE,

            # Bitwise - Direct FFN primitives
            C4Opcode.OR: OpcodeSupport.FFN_PRIMITIVE,
            C4Opcode.XOR: OpcodeSupport.FFN_PRIMITIVE,
            C4Opcode.AND: OpcodeSupport.FFN_PRIMITIVE,
            C4Opcode.SHL: OpcodeSupport.FFN_PRIMITIVE,
            C4Opcode.SHR: OpcodeSupport.FFN_PRIMITIVE,

            # Stack - Composite FFN operations
            C4Opcode.LEA: OpcodeSupport.FFN_COMPOSITE,   # ADD(BP, imm)
            C4Opcode.IMM: OpcodeSupport.FFN_COMPOSITE,   # MOVE(CONST(imm), AX)
            C4Opcode.ADJ: OpcodeSupport.FFN_COMPOSITE,   # ADD(SP, imm)
            C4Opcode.POP: OpcodeSupport.FFN_COMPOSITE,   # ADD(SP, 8)
            C4Opcode.NOP: OpcodeSupport.FFN_COMPOSITE,   # MOVE(in, out)

            # Control - Composite FFN operations
            C4Opcode.JMP: OpcodeSupport.FFN_COMPOSITE,   # MOVE(imm, PC)
            C4Opcode.BZ: OpcodeSupport.FFN_COMPOSITE,    # SELECT(EQ(AX,0), imm, PC+1)
            C4Opcode.BNZ: OpcodeSupport.FFN_COMPOSITE,   # SELECT(NE(AX,0), imm, PC+1)
            C4Opcode.BLT: OpcodeSupport.FFN_COMPOSITE,   # SELECT(LT(AX,0), imm, PC+1)
            C4Opcode.BGE: OpcodeSupport.FFN_COMPOSITE,   # SELECT(GE(AX,0), imm, PC+1)

            # Function calls - Require stack operations (attention for memory)
            C4Opcode.JSR: OpcodeSupport.ATTENTION_NEEDED,  # push PC, PC = imm
            C4Opcode.ENT: OpcodeSupport.ATTENTION_NEEDED,  # push BP, BP=SP, SP-=imm
            C4Opcode.LEV: OpcodeSupport.ATTENTION_NEEDED,  # SP=BP, pop BP, pop PC
            C4Opcode.PSH: OpcodeSupport.ATTENTION_NEEDED,  # push AX

            # Memory - Attention-based
            C4Opcode.LI: OpcodeSupport.ATTENTION_NEEDED,
            C4Opcode.LC: OpcodeSupport.ATTENTION_NEEDED,
            C4Opcode.SI: OpcodeSupport.ATTENTION_NEEDED,
            C4Opcode.SC: OpcodeSupport.ATTENTION_NEEDED,

            # I/O - Attention-based
            C4Opcode.GETCHAR: OpcodeSupport.ATTENTION_NEEDED,
            C4Opcode.PUTCHAR: OpcodeSupport.ATTENTION_NEEDED,

            # System - Mixed
            C4Opcode.MALC: OpcodeSupport.FFN_COMPOSITE,     # Bump allocator (ADD)
            C4Opcode.EXIT: OpcodeSupport.FFN_COMPOSITE,     # Set flag
            C4Opcode.FREE: OpcodeSupport.ATTENTION_NEEDED,  # Zero overwrite
            C4Opcode.MSET: OpcodeSupport.ATTENTION_NEEDED,  # Memory loop
            C4Opcode.MCMP: OpcodeSupport.ATTENTION_NEEDED,  # Memory loop
            C4Opcode.OPEN: OpcodeSupport.TOOL_CALL,
            C4Opcode.READ: OpcodeSupport.TOOL_CALL,
            C4Opcode.CLOS: OpcodeSupport.TOOL_CALL,
            C4Opcode.PRTF: OpcodeSupport.TOOL_CALL,
            C4Opcode.PRINTF2: OpcodeSupport.NOT_IMPLEMENTED,
        }

    def get_support_level(self, opcode: C4Opcode) -> OpcodeSupport:
        """Get support level for an opcode."""
        return self.support_map.get(opcode, OpcodeSupport.NOT_IMPLEMENTED)

    def can_compile_to_ffn(self, opcode: C4Opcode) -> bool:
        """Check if opcode can be compiled to pure FFN."""
        support = self.get_support_level(opcode)
        return support in [OpcodeSupport.FFN_PRIMITIVE, OpcodeSupport.FFN_COMPOSITE]

    def compile_opcode(self, opcode: C4Opcode, **params) -> ComputationGraph:
        """
        Compile an opcode to a computation graph.

        Args:
            opcode: C4 opcode to compile
            **params: Opcode-specific parameters (imm, registers, etc.)

        Returns:
            ComputationGraph representing the opcode's behavior

        Raises:
            NotImplementedError: If opcode requires attention or tool calls
        """
        support = self.get_support_level(opcode)

        if support == OpcodeSupport.FFN_PRIMITIVE:
            return self._compile_primitive(opcode, **params)
        elif support == OpcodeSupport.FFN_COMPOSITE:
            return self._compile_composite(opcode, **params)
        elif support == OpcodeSupport.ATTENTION_NEEDED:
            raise NotImplementedError(
                f"Opcode {opcode.name} requires attention mechanism "
                f"(handled by neural_vm, not graph compiler)"
            )
        elif support == OpcodeSupport.TOOL_CALL:
            raise NotImplementedError(
                f"Opcode {opcode.name} requires tool call "
                f"(handled externally)"
            )
        else:
            raise NotImplementedError(f"Opcode {opcode.name} not implemented")

    def _compile_primitive(self, opcode: C4Opcode, **params) -> ComputationGraph:
        """Compile a direct FFN primitive opcode."""
        graph = ComputationGraph()

        # Map C4 opcodes to graph compiler OpType
        opcode_map = {
            C4Opcode.ADD: OpType.ADD,
            C4Opcode.SUB: OpType.SUB,
            C4Opcode.MUL: OpType.MUL,
            C4Opcode.DIV: OpType.DIV,
            C4Opcode.MOD: OpType.MOD,
            C4Opcode.EQ: OpType.CMP_EQ,
            C4Opcode.NE: OpType.CMP_NE,
            C4Opcode.LT: OpType.CMP_LT,
            C4Opcode.GT: OpType.CMP_GT,
            C4Opcode.LE: OpType.CMP_LE,
            C4Opcode.GE: OpType.CMP_GE,
            C4Opcode.OR: OpType.BIT_OR,
            C4Opcode.XOR: OpType.BIT_XOR,
            C4Opcode.AND: OpType.BIT_AND,
            C4Opcode.SHL: OpType.SHL,
            C4Opcode.SHR: OpType.SHR,
        }

        prim_op = opcode_map[opcode]

        # Create input nodes
        a = graph.add_input("a")
        b = graph.add_input("b")

        # Create operation
        result = graph.add_op(prim_op, [a, b], "result", params=params)

        return graph

    def _compile_composite(self, opcode: C4Opcode, **params) -> ComputationGraph:
        """Compile a composite opcode from multiple primitives."""
        graph = ComputationGraph()

        if opcode == C4Opcode.LEA:
            # LEA: AX = BP + imm
            # result = ADD(BP, CONST(imm))
            bp = graph.add_input("BP")
            imm = graph.add_const(params.get('imm', 0))
            result = graph.add_op(OpType.ADD, [bp, imm], "AX")

        elif opcode == C4Opcode.IMM:
            # IMM: AX = imm
            # result = CONST(imm)
            result = graph.add_const(params.get('imm', 0), output_name="AX")

        elif opcode == C4Opcode.ADJ:
            # ADJ: SP += imm
            # result = ADD(SP, CONST(imm))
            sp = graph.add_input("SP")
            imm = graph.add_const(params.get('imm', 0))
            result = graph.add_op(OpType.ADD, [sp, imm], "SP")

        elif opcode == C4Opcode.POP:
            # POP: SP += 8
            # result = ADD(SP, CONST(8))
            sp = graph.add_input("SP")
            eight = graph.add_const(8)
            result = graph.add_op(OpType.ADD, [sp, eight], "SP")

        elif opcode == C4Opcode.NOP:
            # NOP: no operation (identity)
            # result = MOVE(input, output)
            inp = graph.add_input("input")
            result = graph.add_op(OpType.MOVE, [inp], "output")

        elif opcode == C4Opcode.JMP:
            # JMP: PC = imm
            # result = CONST(imm)
            result = graph.add_const(params.get('imm', 0), output_name="PC")

        elif opcode == C4Opcode.BZ:
            # BZ: if AX==0: PC = imm else PC = PC+1
            # cond = CMP_EQ(AX, 0)
            # result = SELECT(cond, imm, PC+1)
            ax = graph.add_input("AX")
            zero = graph.add_const(0)
            cond = graph.add_op(OpType.CMP_EQ, [ax, zero], "cond")

            target = graph.add_const(params.get('imm', 0))
            pc = graph.add_input("PC")
            one = graph.add_const(1)
            fallthrough = graph.add_op(OpType.ADD, [pc, one], "PC+1")

            result = graph.add_op(OpType.SELECT, [cond, target, fallthrough], "PC")

        elif opcode == C4Opcode.BNZ:
            # BNZ: if AX!=0: PC = imm else PC = PC+1
            ax = graph.add_input("AX")
            zero = graph.add_const(0)
            cond = graph.add_op(OpType.CMP_NE, [ax, zero], "cond")

            target = graph.add_const(params.get('imm', 0))
            pc = graph.add_input("PC")
            one = graph.add_const(1)
            fallthrough = graph.add_op(OpType.ADD, [pc, one], "PC+1")

            result = graph.add_op(OpType.SELECT, [cond, target, fallthrough], "PC")

        elif opcode == C4Opcode.BLT:
            # BLT: if AX<0: PC = imm else PC = PC+1
            ax = graph.add_input("AX")
            zero = graph.add_const(0)
            cond = graph.add_op(OpType.CMP_LT, [ax, zero], "cond")

            target = graph.add_const(params.get('imm', 0))
            pc = graph.add_input("PC")
            one = graph.add_const(1)
            fallthrough = graph.add_op(OpType.ADD, [pc, one], "PC+1")

            result = graph.add_op(OpType.SELECT, [cond, target, fallthrough], "PC")

        elif opcode == C4Opcode.BGE:
            # BGE: if AX>=0: PC = imm else PC = PC+1
            ax = graph.add_input("AX")
            zero = graph.add_const(0)
            cond = graph.add_op(OpType.CMP_GE, [ax, zero], "cond")

            target = graph.add_const(params.get('imm', 0))
            pc = graph.add_input("PC")
            one = graph.add_const(1)
            fallthrough = graph.add_op(OpType.ADD, [pc, one], "PC+1")

            result = graph.add_op(OpType.SELECT, [cond, target, fallthrough], "PC")

        elif opcode == C4Opcode.MALC:
            # MALC: result = HEAP_PTR; HEAP_PTR += n
            # This needs multiple outputs, so we'll return the new heap pointer
            heap_ptr = graph.add_input("HEAP_PTR")
            n = graph.add_input("n")
            result = graph.add_op(OpType.ADD, [heap_ptr, n], "HEAP_PTR_new")
            # Note: actual malloc returns OLD heap_ptr, this returns NEW
            # Full implementation needs dual outputs

        elif opcode == C4Opcode.EXIT:
            # EXIT: Set exit code and flag
            # result = MOVE(code, EXIT_CODE)
            code = graph.add_input("code")
            result = graph.add_op(OpType.MOVE, [code], "EXIT_CODE")
            # Also need to set IO_PROGRAM_END flag (separate output)

        else:
            raise NotImplementedError(f"Composite opcode {opcode.name} not implemented")

        return graph

    def print_support_summary(self):
        """Print summary of opcode support levels."""
        categories = {
            'FFN Primitive (Direct)': OpcodeSupport.FFN_PRIMITIVE,
            'FFN Composite': OpcodeSupport.FFN_COMPOSITE,
            'Attention Needed': OpcodeSupport.ATTENTION_NEEDED,
            'Tool Call': OpcodeSupport.TOOL_CALL,
            'Not Implemented': OpcodeSupport.NOT_IMPLEMENTED,
        }

        print("=" * 70)
        print("C4 VM Opcode Support Summary")
        print("=" * 70)

        for category, support_level in categories.items():
            opcodes = [op for op, sup in self.support_map.items() if sup == support_level]
            count = len(opcodes)
            print(f"\n{category} ({count} opcodes):")
            if opcodes:
                for op in sorted(opcodes):
                    print(f"  {op.value:2d} - {op.name}")

        total = len(self.support_map)
        ffn_count = sum(1 for s in self.support_map.values()
                       if s in [OpcodeSupport.FFN_PRIMITIVE, OpcodeSupport.FFN_COMPOSITE])

        print("\n" + "=" * 70)
        print(f"Total: {total} opcodes")
        print(f"FFN-compilable: {ffn_count} ({100 * ffn_count / total:.1f}%)")
        print("=" * 70)


def main():
    """Demo opcode mapper."""
    mapper = OpcodeMapper()
    mapper.print_support_summary()

    print("\n" + "=" * 70)
    print("Example: Compiling LEA opcode")
    print("=" * 70)

    try:
        graph = mapper.compile_opcode(C4Opcode.LEA, imm=16)
        print(f"\nLEA with imm=16:")
        print(f"  Nodes: {len(graph.nodes)}")
        for node_id, node in graph.nodes.items():
            print(f"    {node}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("Example: BZ (branch if zero)")
    print("=" * 70)

    try:
        graph = mapper.compile_opcode(C4Opcode.BZ, imm=0x1000)
        print(f"\nBZ with target=0x1000:")
        print(f"  Nodes: {len(graph.nodes)}")
        for node_id, node in graph.nodes.items():
            print(f"    {node}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
