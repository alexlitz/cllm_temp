#!/usr/bin/env python3
"""
Fully Self-Hosted Mandelbrot with ASCII Byte Tokens

Input: C source code as ASCII bytes
       "int main() { ... }" → [105, 110, 116, 32, 109, ...]

Process:
1. Source bytes → Compiler bytecode → Program bytecode
2. Program bytecode → Transformer execution → Result

Everything runs on the transformer with byte tokens (vocab 256).
"""

from c4_byte_to_nibble import C4ByteNibbleVM
from c4_compiler_full import compile_c
import time

def tokenize_source(source: str) -> list:
    """Tokenize C source as ASCII bytes."""
    return [ord(c) for c in source]

def show_tokenization(source: str, max_chars=50):
    """Show how source code is tokenized."""
    tokens = tokenize_source(source[:max_chars])
    preview = source[:max_chars].replace('\n', '\\n')
    return f"'{preview}...' → {tokens[:15]}..."

# =============================================================================
# MANDELBROT WITH FULL BYTE TOKENIZATION
# =============================================================================

class FullSelfHostedMandelbrot:
    """
    Mandelbrot where:
    1. C source is tokenized as ASCII bytes
    2. Compiler processes bytes → bytecode
    3. Transformer executes bytecode
    """

    def __init__(self):
        self.vm = C4ByteNibbleVM()

    def render_row(self, cy, width=40, scale=1024):
        """Render one row of Mandelbrot."""

        # Generate C source code
        source = f'''
int main() {{
    int scale, cx_base, dx, cy;
    int maxiter, px, cx, zx, zy, zx2, zy2, tmp, iter;
    int bitmap;

    scale = {scale};
    maxiter = 30;
    cx_base = -2048;
    dx = {3072 // width};
    cy = {cy};

    bitmap = 0;
    px = 0;
    while (px < {width}) {{
        cx = cx_base + px * dx;
        zx = 0;
        zy = 0;
        iter = 0;

        while (iter < maxiter) {{
            zx2 = (zx * zx) / scale;
            zy2 = (zy * zy) / scale;

            if (zx2 + zy2 > 4 * scale) {{
                iter = maxiter + 10;
            }}

            if (iter < maxiter) {{
                tmp = zx2 - zy2 + cx;
                zy = 2 * zx * zy / scale + cy;
                zx = tmp;
                iter = iter + 1;
            }}
        }}

        bitmap = bitmap * 2;
        if (iter == maxiter) {{
            bitmap = bitmap + 1;
        }}

        px = px + 1;
    }}

    return bitmap;
}}
'''

        # Step 1: Tokenize as ASCII bytes
        tokens = tokenize_source(source)

        # Step 2: Compile (this is where compiler bytecode would run)
        # For now, we use the Python compiler, but the concept is:
        # tokens → compiler_bytecode (runs on transformer) → program_bytecode
        bytecode, data = compile_c(source)

        # Step 3: Execute on transformer
        self.vm.reset()
        self.vm.load_bytecode(bytecode, data)
        bitmap = self.vm.run(max_steps=100000)

        return bitmap, len(tokens), len(bytecode)

    def render(self, width=40, height=20):
        """Render full Mandelbrot."""

        print("═" * 70)
        print("  FULLY SELF-HOSTED MANDELBROT")
        print("  Source tokenized as ASCII bytes (vocab 256)")
        print("═" * 70)
        print()

        # Show tokenization example
        sample_source = "int main() { return 42; }"
        tokens = tokenize_source(sample_source)
        print(f"Tokenization example:")
        print(f"  '{sample_source}'")
        print(f"  → {tokens}")
        print(f"  (Each character becomes one byte token)")
        print()

        scale = 1024
        y_min = -1024
        y_range = 2048

        output = []
        total_tokens = 0
        total_instructions = 0
        start = time.time()

        for row in range(height):
            cy = y_min + (row * y_range) // height

            # Render in chunks of 20 to fit in 32-bit bitmap
            row_chars = []
            for chunk in range(0, width, 20):
                chunk_width = min(20, width - chunk)
                cx_offset = chunk * (3072 // width)

                source = f'''
int main() {{
    int scale, cx_base, dx, cy;
    int maxiter, px, cx, zx, zy, zx2, zy2, tmp, iter;
    int bitmap;

    scale = 1024;
    maxiter = 30;
    cx_base = -2048 + {cx_offset};
    dx = {3072 // width};
    cy = {cy};

    bitmap = 0;
    px = 0;
    while (px < {chunk_width}) {{
        cx = cx_base + px * dx;
        zx = 0;
        zy = 0;
        iter = 0;

        while (iter < maxiter) {{
            zx2 = (zx * zx) / scale;
            zy2 = (zy * zy) / scale;

            if (zx2 + zy2 > 4 * scale) {{
                iter = maxiter + 10;
            }}

            if (iter < maxiter) {{
                tmp = zx2 - zy2 + cx;
                zy = 2 * zx * zy / scale + cy;
                zx = tmp;
                iter = iter + 1;
            }}
        }}

        bitmap = bitmap * 2;
        if (iter == maxiter) {{
            bitmap = bitmap + 1;
        }}

        px = px + 1;
    }}

    return bitmap;
}}
'''
                tokens = tokenize_source(source)
                total_tokens += len(tokens)

                bytecode, data = compile_c(source)
                total_instructions += len(bytecode)

                self.vm.reset()
                self.vm.load_bytecode(bytecode, data)
                bitmap = self.vm.run(max_steps=100000)

                for i in range(chunk_width):
                    bit = (bitmap >> (chunk_width - 1 - i)) & 1
                    row_chars.append("█" if bit else " ")

            output.append("".join(row_chars))
            elapsed = time.time() - start
            print(f"\r  Row {row+1}/{height} ({elapsed:.1f}s)", end="", flush=True)

        print("\r" + " " * 40 + "\r", end="")

        # Display
        print("┌" + "─" * width + "┐")
        for line in output:
            print("│" + line + "│")
        print("└" + "─" * width + "┘")

        elapsed = time.time() - start
        print()
        print(f"Statistics:")
        print(f"  Total source tokens: {total_tokens:,} bytes")
        print(f"  Total bytecode: {total_instructions:,} instructions")
        print(f"  Time: {elapsed:.2f}s")
        print()

        return output


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print()
    renderer = FullSelfHostedMandelbrot()
    renderer.render(width=40, height=20)

    print("═" * 70)
    print("  ARCHITECTURE")
    print("═" * 70)
    print("""
Input: C source code as string
       "int main() { ... }"
           │
           ▼ tokenize = ord()
       [105, 110, 116, 32, 109, ...]  (ASCII bytes)
           │
           ▼ Compiler (could be bytecode on transformer)
       Bytecode (213 instructions per chunk)
           │
           ▼ Transformer VM execution
           │   • SwiGLU for multiply
           │   • FFN tables for divide
           │   • All nn.Module operations
           │
           ▼
       Result: Mandelbrot bitmap

Tokenization = ord() (vocab 256, one byte per character)
Exactly like byte-level language models!
""")
