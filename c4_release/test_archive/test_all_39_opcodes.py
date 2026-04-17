#!/usr/bin/env python3
"""
Test All 39 Core C4 Opcodes

Verifies that all core C4 opcodes (0-38) can be mapped to computation graphs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.full_opcode_mapper import FullOpcodeMapper
from neural_vm.embedding import Opcode


def test_all_39_opcodes():
    """Test all 39 core C4 opcodes (0-38)."""
    print("="*70)
    print("Testing All 39 Core C4 Opcodes (0-38)")
    print("="*70)

    mapper = FullOpcodeMapper()

    # All core C4 opcodes 0-38
    core_opcodes = [
        (0, "LEA", "single"),
        (1, "IMM", "single"),
        (2, "JMP", "single"),
        (3, "JSR", "multi"),
        (4, "BZ", "single"),
        (5, "BNZ", "single"),
        (6, "ENT", "multi"),
        (7, "ADJ", "single"),
        (8, "LEV", "multi"),
        (9, "LI", "multi"),
        (10, "LC", "multi"),
        (11, "SI", "multi"),
        (12, "SC", "multi"),
        (13, "PSH", "multi"),
        (14, "OR", "single"),
        (15, "XOR", "single"),
        (16, "AND", "single"),
        (17, "EQ", "single"),
        (18, "NE", "single"),
        (19, "LT", "single"),
        (20, "GT", "single"),
        (21, "LE", "single"),
        (22, "GE", "single"),
        (23, "SHL", "single"),
        (24, "SHR", "single"),
        (25, "ADD", "single"),
        (26, "SUB", "single"),
        (27, "MUL", "single"),
        (28, "DIV", "single"),
        (29, "MOD", "single"),
        (30, "OPEN", "single"),
        (31, "READ", "single"),
        (32, "CLOS", "single"),
        (33, "PRTF", "single"),
        (34, "MALC", "single"),
        (35, "FREE", "single"),
        (36, "MSET", "single"),
        (37, "MCMP", "single"),
        (38, "EXIT", "single"),
    ]

    results = []
    for opcode, name, layer_type in core_opcodes:
        try:
            if layer_type == "single":
                # Try single-layer mapping
                if opcode in [0, 1, 2, 3, 4, 5, 6, 7, 34, 38]:
                    graph = mapper.map_opcode(opcode, imm=0)
                else:
                    graph = mapper.map_opcode(opcode)

                node_count = len(graph.nodes)
                print(f"  ✅ {opcode:2d} {name:8s}: {node_count} nodes (single-layer)")
                results.append((name, True, "single", node_count))

            elif layer_type == "multi":
                # Try multi-layer mapping
                if opcode in [3, 6]:  # JSR, ENT need imm
                    multi = mapper.map_opcode_multilayer(opcode, imm=0)
                else:
                    multi = mapper.map_opcode_multilayer(opcode)

                layer_count = len(multi.layers)
                print(f"  ✅ {opcode:2d} {name:8s}: {layer_count} layers (multi-layer)")
                results.append((name, True, "multi", layer_count))

        except NotImplementedError as e:
            print(f"  ❌ {opcode:2d} {name:8s}: Not implemented - {e}")
            results.append((name, False, layer_type, 0))
        except Exception as e:
            print(f"  ❌ {opcode:2d} {name:8s}: Error - {e}")
            results.append((name, False, layer_type, 0))

    # Summary
    passed = sum(1 for _, p, _, _ in results if p)
    total = len(results)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Mapped: {passed}/{total} opcodes ({100*passed/total:.1f}%)")

    # Breakdown by type
    single_passed = sum(1 for _, p, t, _ in results if p and t == "single")
    single_total = sum(1 for _, _, t, _ in results if t == "single")
    multi_passed = sum(1 for _, p, t, _ in results if p and t == "multi")
    multi_total = sum(1 for _, _, t, _ in results if t == "multi")

    print(f"\n  Single-layer: {single_passed}/{single_total}")
    print(f"  Multi-layer: {multi_passed}/{multi_total}")

    if passed == total:
        print("\n  🎉 ALL 39 CORE OPCODES IMPLEMENTED!")
        return True
    else:
        failed = [name for name, p, _, _ in results if not p]
        print(f"\n  ⚠️  Failed: {', '.join(failed)}")
        return False


if __name__ == "__main__":
    success = test_all_39_opcodes()
    sys.exit(0 if success else 1)
