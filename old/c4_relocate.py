"""
C4 Bytecode Relocator

Takes bytecode from xc_dump with absolute addresses and relocates
them for our VM's memory layout.

Key insight: xc stores one value per 8-byte word. We pack opcode+immediate
into one word. So we need to translate addresses between xc's layout and ours.
"""

import subprocess
import torch
from c4_moe_vm import C4MoEVM, C4Op


def get_bytecode_with_base(source_file):
    """Get bytecode, data segment, and bases from xc_dump2."""
    result = subprocess.run(
        ["/tmp/xc_dump2", source_file],
        capture_output=True, text=True, timeout=10
    )

    lines = result.stdout.strip().split('\n')

    text_base = None
    data_base = None
    data_size = 0
    raw_bytecode = []
    raw_data = []
    main_offset = 0
    in_bytecode = False
    in_data = False

    for line in lines:
        line = line.strip()
        if line == "=== BYTECODE ===":
            in_bytecode = True
            in_data = False
            continue
        if line == "=== END ===":
            in_bytecode = False
            continue
        if line == "=== DATA ===":
            in_data = True
            in_bytecode = False
            continue
        if line == "=== DATA_END ===":
            in_data = False
            continue
        if line.startswith("TEXT_BASE:"):
            text_base = int(line.replace("TEXT_BASE:", ""))
            continue
        if line.startswith("DATA_BASE:"):
            data_base = int(line.replace("DATA_BASE:", ""))
            continue
        if line.startswith("DATA_SIZE:"):
            data_size = int(line.replace("DATA_SIZE:", ""))
            continue
        if line.startswith("MAIN:"):
            main_offset = int(line.replace("MAIN:", ""))
            continue
        if in_bytecode and line:
            try:
                raw_bytecode.append(int(line))
            except:
                pass
        if in_data and line:
            try:
                raw_data.append(int(line))
            except:
                pass

    return raw_bytecode, text_base, main_offset, raw_data, data_base


def pack_and_relocate(raw_bytecode, text_base, target_base=0, data_base=None, data_size=0, target_data_base=0x10000):
    """
    Pack raw bytecode and relocate addresses.

    xc layout: one value per 8-byte word
    Our layout: opcode+immediate packed into one 8-byte word

    NOTE: xc_dump2 outputs bytecode starting from old_text+1, skipping word 0.
    So raw[0] = old_text[1], raw[i] = old_text[i+1].
    When xc generates a branch to absolute address A, the word index is (A - text_base) / 8.
    But our raw array is offset by 1, so we need to subtract 1 to get raw index.

    Returns: (packed_bytecode, raw_word_to_our_byte_map)
    """
    OPS_WITH_IMM = {0, 1, 2, 3, 4, 5, 6, 7}  # LEA, IMM, JMP, JSR, JZ, JNZ, ENT, ADJ

    # First pass: build mapping from xc word index to our byte offset
    # Note: xc word N corresponds to raw[N-1] because dump skips word 0
    xc_word_to_our_byte = {}
    packed = []
    i = 0

    text_size = len(raw_bytecode) * 8

    while i < len(raw_bytecode):
        # raw[i] corresponds to xc word (i+1) because dump starts at old_text+1
        xc_word = i + 1
        xc_word_to_our_byte[xc_word] = len(packed) * 8

        op = raw_bytecode[i]

        if op in OPS_WITH_IMM and i + 1 < len(raw_bytecode):
            imm = raw_bytecode[i + 1]
            xc_word_to_our_byte[xc_word + 1] = len(packed) * 8  # imm is part of same word

            # Check if this immediate is a text address that needs relocation
            if text_base <= imm < text_base + text_size + 0x10000:
                # Convert absolute address to xc word index
                xc_byte_offset = imm - text_base
                xc_word_index = xc_byte_offset // 8
                # Mark as text relocation (xc_word_index is the actual xc word)
                packed.append((op, xc_word_index, 'text'))
            # Check if this immediate is a data address that needs relocation
            elif data_base is not None and data_base <= imm < data_base + data_size + 0x10000:
                # Data address - will relocate to our data segment
                data_offset = imm - data_base
                packed.append((op, data_offset, 'data'))
            else:
                packed.append((op, imm, False))
            i += 2
        else:
            packed.append((op, None, False))
            i += 1

    # Second pass: resolve addresses
    final_packed = []
    for item in packed:
        if len(item) == 3:
            op, val, reloc_type = item
            if reloc_type == 'text':
                # Convert xc word index to our byte offset in text segment
                our_byte = xc_word_to_our_byte.get(val, val * 8)
                final_packed.append(op | ((target_base + our_byte) << 8))
            elif reloc_type == 'data':
                # Data address - relocate to our data segment
                final_packed.append(op | ((target_data_base + val) << 8))
            elif val is not None:
                # Pack immediate as signed 64-bit value
                # The immediate occupies bits 8-63 (56 bits)
                final_packed.append(op | (val << 8))
            else:
                final_packed.append(op)
        else:
            final_packed.append(item)

    return final_packed, xc_word_to_our_byte


def test_relocation():
    """Test relocation with programs."""
    print("=" * 60)
    print("TESTING C4 BYTECODE RELOCATION")
    print("=" * 60)

    # Simple test
    print("\n1. Simple program (no relocation needed):")

    with open("/tmp/simple.c", "w") as f:
        f.write("int main() { return 6 * 7; }")

    raw, text_base, main_offset, raw_data, data_base = get_bytecode_with_base("/tmp/simple.c")

    print(f"   TEXT_BASE: {text_base}")
    print(f"   Raw bytecode: {raw}")
    print(f"   Main offset (xc word): {main_offset}")

    packed, word_to_byte = pack_and_relocate(raw, text_base, target_base=0,
                                              data_base=data_base, data_size=len(raw_data))
    our_main_byte = word_to_byte.get(main_offset, main_offset * 8)

    print(f"   Packed: {packed}")
    print(f"   Main at our byte: {our_main_byte}")

    # Replace first LEV with EXIT
    for i in range(len(packed)):
        if (packed[i] & 0xFF) == 8:  # LEV
            packed[i] = 38  # EXIT
            break

    vm = C4MoEVM()
    vm.load(packed, [])
    vm.pc = torch.tensor(float(our_main_byte))
    result, _, _ = vm.run()

    print(f"   Result: {result} (expected 42)")
    print(f"   Status: {'✓ PASS' if result == 42 else '✗ FAIL'}")

    # Test with function call
    print("\n2. Program with function call (needs relocation):")

    with open("/tmp/func.c", "w") as f:
        f.write("""
int double_it(int x) {
    return x + x;
}
int main() {
    return double_it(21);
}
""")

    raw, text_base, main_offset, raw_data, data_base = get_bytecode_with_base("/tmp/func.c")

    print(f"   TEXT_BASE: {text_base}")
    print(f"   Raw (first 25): {raw[:25]}")
    print(f"   Main offset (xc word): {main_offset}")

    packed, word_to_byte = pack_and_relocate(raw, text_base, target_base=0,
                                              data_base=data_base, data_size=len(raw_data))
    our_main_byte = word_to_byte.get(main_offset, main_offset * 8)

    print(f"   Packed (first 15): {packed[:15]}")
    print(f"   Main at our byte: {our_main_byte}")

    # Decode a few packed instructions
    print(f"   Decoded:")
    op_names = ['LEA', 'IMM', 'JMP', 'JSR', 'JZ', 'JNZ', 'ENT', 'ADJ', 'LEV',
                'LI', 'LC', 'SI', 'SC', 'PSH']
    for i, p in enumerate(packed[:10]):
        op = p & 0xFF
        imm = p >> 8
        name = op_names[op] if op < len(op_names) else f"OP{op}"
        print(f"     [{i}] byte {i*8}: {name} {imm}")

    # Need to set up stack with return address
    # Push EXIT as return address before jumping to main
    vm = C4MoEVM()
    vm.load(packed, [])

    # Set up initial stack: push EXIT address
    # When main's LEV executes, it should return here
    exit_addr = len(packed) * 8  # Put EXIT after all code
    vm.push(exit_addr)  # Return address for main

    # Actually, let's just replace main's LEV with EXIT
    # Find LEV instructions and replace the one in main
    main_start_idx = our_main_byte // 8
    for i in range(main_start_idx, len(packed)):
        if (packed[i] & 0xFF) == 8:  # LEV
            packed[i] = 38  # EXIT
            print(f"   Replaced LEV at packed[{i}] with EXIT")
            break

    # Reload and run
    vm = C4MoEVM()
    vm.load(packed, [])
    vm.pc = torch.tensor(float(our_main_byte))

    result, _, stats = vm.run(max_steps=100, trace=True)

    print(f"\n   Result: {result} (expected 42)")
    print(f"   Steps: {stats['steps']}")
    print(f"   Status: {'✓ PASS' if result == 42 else '✗ FAIL'}")


if __name__ == "__main__":
    test_relocation()
