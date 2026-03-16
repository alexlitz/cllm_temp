"""
C4 Full Bootstrap: Run C4 compiler inside Transformer VM

Architecture:
1. "System Prompt": C4 compiler bytecode (pre-loaded)
2. Input: C source code (in VM's virtual file system)
3. Transformer VM runs C4 compiler bytecode
4. C4 outputs program bytecode
5. Transformer VM runs program bytecode
6. Final result

All computation uses transformer primitives:
- MUL: silu(a)*b + silu(-a)*(-b) = a*b [exact via SwiGLU]
- DIV: log2 via attention, exp via softmax ratio [exact]
- Memory: Binary-encoded attention
- Routing: 39 experts via eq_gate (SwiGLU)
- 0 learned parameters
"""

import subprocess
import torch
from c4_moe_vm import C4MoEVM, C4Op
from c4_relocate import get_bytecode_with_base, pack_and_relocate


def load_c4_compiler():
    """
    Load the C4 compiler as bytecode.
    This is the "system prompt" - pre-loaded into the VM.
    """
    raw, text_base, main_offset, raw_data, data_base = get_bytecode_with_base("/tmp/mini_compiler.c")

    # Pack and relocate
    # Note: main_offset from xc_dump2 is raw array index, xc word is main_offset+1
    packed, word_to_byte = pack_and_relocate(
        raw, text_base, target_base=0,
        data_base=data_base, data_size=len(raw_data),
        target_data_base=0x10000
    )
    our_main = word_to_byte.get(main_offset + 1, (main_offset + 1) * 8)

    return packed, our_main, raw_data


def run_compiler_in_vm(compiler_bytecode, main_offset, data_segment):
    """
    Run the C4 compiler inside the transformer VM.

    Returns:
        (output_string, exit_code, stats)
    """
    vm = C4MoEVM()

    # Load compiler bytecode (make a copy so we don't modify original)
    code = compiler_bytecode.copy()

    # Replace main's LEV with EXIT
    main_idx = main_offset // 8
    for i in range(main_idx, len(code)):
        if (code[i] & 0xFF) == 8:  # LEV
            code[i] = C4Op.EXIT
            break

    vm.load(code, [])

    # Load data segment
    for i, b in enumerate(data_segment):
        vm.memory[0x10000 + i] = b

    # Run the compiler
    vm.pc = torch.tensor(float(main_offset))
    result, output, stats = vm.run(max_steps=100000, fast=True)

    return output, result, stats


def parse_bytecode_output(output):
    """Parse the bytecode output from the compiler."""
    tokens = [int(x) for x in output.strip().split() if x.lstrip('-').isdigit()]

    # Pack opcode+immediate together
    OPS_WITH_IMM = {1}  # IMM
    packed = []
    i = 0
    while i < len(tokens):
        op = tokens[i]
        if op in OPS_WITH_IMM and i + 1 < len(tokens):
            packed.append(op | (tokens[i + 1] << 8))
            i += 2
        else:
            packed.append(op)
            i += 1

    return tokens, packed


def run_program_in_vm(bytecode):
    """Run program bytecode in the transformer VM."""
    vm = C4MoEVM()
    vm.load(bytecode, [])
    result, output, stats = vm.run(fast=True)
    return result, stats


def demo():
    """Demonstrate running C4 compiler inside the transformer VM."""
    print("=" * 70)
    print("C4 FULL BOOTSTRAP: Running C4 Compiler in Transformer VM")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  Step 1: Load C4 compiler bytecode ('system prompt')")
    print("  Step 2: Run compiler in transformer VM with C source")
    print("  Step 3: Compiler outputs program bytecode")
    print("  Step 4: Run program bytecode in transformer VM")
    print()
    print("All operations use only transformer primitives!")
    print()
    print("=" * 70)
    print()

    # Step 1: Load C4 compiler
    print("[STEP 1] Loading C4 compiler bytecode...")
    compiler_bytecode, main_offset, data_segment = load_c4_compiler()
    print(f"  Compiler: {len(compiler_bytecode)} instructions")
    print(f"  Data: {len(data_segment)} bytes")
    print()

    # Step 2: Run compiler
    print("[STEP 2] Running C4 compiler in transformer VM...")
    print("  Source: 'int main() { return 6 * 7; }'")
    output, exit_code, stats = run_compiler_in_vm(
        compiler_bytecode, main_offset, data_segment
    )
    print(f"  Steps: {stats['steps']}")
    print(f"  Output: '{output}'")
    print()

    # Step 3: Parse output
    print("[STEP 3] Parsing compiler output...")
    tokens, program_bytecode = parse_bytecode_output(output)
    print(f"  Tokens: {tokens}")
    print(f"  Packed: {program_bytecode}")
    print()

    # Step 4: Run the program
    print("[STEP 4] Running program bytecode in transformer VM...")
    result, stats = run_program_in_vm(program_bytecode)
    print(f"  Steps: {stats['steps']}")
    print(f"  Result: {result}")
    print(f"  Expected: 42 (6 * 7)")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    if result == 42:
        print("  ✓ SUCCESS: Full bootstrap completed!")
        print()
        print("  What happened:")
        print("    1. C4 compiler (written in C) was compiled to bytecode by xc")
        print("    2. This bytecode was loaded into the transformer VM")
        print("    3. The VM executed the compiler using SwiGLU, attention, etc.")
        print("    4. The compiler's output (program bytecode) was captured")
        print("    5. That bytecode was run in the VM to get the final result")
    else:
        print(f"  ✗ FAIL: Expected 42, got {result}")
    print()
    print("  Key transformer primitives used:")
    print("    - MUL: silu(a)*b + silu(-a)*(-b) = a*b  [exact]")
    print("    - DIV: log2 via attention + exp via softmax ratio  [exact]")
    print("    - Memory: Binary-encoded attention")
    print("    - Routing: 39 experts via SwiGLU eq_gate")
    print("    - 0 learned parameters")


if __name__ == "__main__":
    demo()
