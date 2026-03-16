#!/usr/bin/env python3
"""
Demonstrate: C program → Compiler → Bytecode → Transformer VM → Mandelbrot
"""

from c4_byte_to_nibble import C4ByteNibbleVM
from c4_compiler_full import compile_c

# Step 1: Write a C program
C_PROGRAM = '''
int main() {
    int scale, cx, cy, zx, zy, zx2, zy2, tmp, iter, maxiter;
    int px, py, width, height;
    int bitmap;

    scale = 1024;
    width = 20;
    height = 10;
    maxiter = 30;

    // Compute row 5 (middle) of Mandelbrot
    cy = 0;  // y = 0 (middle)

    bitmap = 0;
    px = 0;
    while (px < width) {
        // x from -2.0 to 1.0
        cx = -2048 + px * 153;  // 3072/20 ≈ 153

        zx = 0;
        zy = 0;
        iter = 0;

        // z = z^2 + c iteration
        while (iter < maxiter) {
            zx2 = (zx * zx) / scale;
            zy2 = (zy * zy) / scale;

            if (zx2 + zy2 > 4 * scale) {
                iter = maxiter + 10;  // escaped
            }

            if (iter < maxiter) {
                tmp = zx2 - zy2 + cx;
                zy = 2 * zx * zy / scale + cy;
                zx = tmp;
                iter = iter + 1;
            }
        }

        // Encode result in bitmap
        bitmap = bitmap * 2;
        if (iter == maxiter) {
            bitmap = bitmap + 1;  // inside set
        }

        px = px + 1;
    }

    return bitmap;
}
'''

print("=" * 70)
print("C PROGRAM → TRANSFORMER VM → MANDELBROT")
print("=" * 70)

# Step 2: Show the C program
print("\n1. C PROGRAM:")
print("-" * 40)
for i, line in enumerate(C_PROGRAM.strip().split('\n')[:15]):
    print(f"   {line}")
print("   ...")

# Step 3: Compile to bytecode
print("\n2. COMPILE TO BYTECODE:")
print("-" * 40)
bytecode, data = compile_c(C_PROGRAM)
print(f"   Compiled to {len(bytecode)} instructions")
print(f"   First 5 instructions:")
for i, instr in enumerate(bytecode[:5]):
    op = instr & 0xFF
    imm = instr >> 8
    OP_NAMES = {0:'LEA', 1:'IMM', 2:'JMP', 3:'JSR', 4:'BZ', 5:'BNZ',
                6:'ENT', 7:'ADJ', 8:'LEV', 9:'LI', 13:'PSH', 25:'ADD',
                26:'SUB', 27:'MUL', 28:'DIV', 38:'EXIT'}
    print(f"      [{i}] {OP_NAMES.get(op, f'OP{op}')} {imm}")

# Step 4: Run on transformer VM
print("\n3. RUN ON TRANSFORMER VM:")
print("-" * 40)
print("   C4ByteNibbleVM components:")
print("   • ByteEncoder: int → [4 bytes] → [4×256 one-hot]")
print("   • SwiGLUMul: multiply via silu(a)*b + silu(-a)*(-b)")
print("   • DivisionFFN: 256-entry table + Newton iterations")
print("   • NibbleTableFFN: 16×16 lookup tables for bitwise ops")
print("   • CompareFFN: sharp gate for <, >, ==")

vm = C4ByteNibbleVM()
vm.reset()
vm.load_bytecode(bytecode, data)

print(f"\n   Running {len(bytecode)} instructions...")
result = vm.run(max_steps=50000)

# Step 5: Decode result
print("\n4. RESULT:")
print("-" * 40)
print(f"   Bitmap value: {result} = {bin(result)}")
print(f"   Decoded row: ", end="")
for i in range(20):
    bit = (result >> (19 - i)) & 1
    print("█" if bit else " ", end="")
print()

# Show what this represents
print(f"\n   This is row y=0 of Mandelbrot, x from -2.0 to 1.0")
print(f"   █ = point is INSIDE the Mandelbrot set")
print(f"   The leftmost points (x ≈ -2) escape quickly")
print(f"   The filled region (x ≈ -0.5 to 0.3) is inside the set")

print("\n" + "=" * 70)
print("ALL ARITHMETIC WAS DONE BY THE TRANSFORMER:")
print("  • Multiply (zx*zx, zy*zy, 2*zx*zy): SwiGLU activation")
print("  • Divide (/scale): FFN table lookup + Newton refinement")
print("  • Compare (> 4*scale, < maxiter): Sharp gate FFN")
print("  • NO Python arithmetic on data values!")
print("=" * 70)
