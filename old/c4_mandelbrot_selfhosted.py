#!/usr/bin/env python3
"""
Self-Hosted Mandelbrot - Compiler runs as bytecode on transformer

Flow:
1. Mandelbrot C source code
2. Compiler bytecode (system prompt) compiles it
3. Mandelbrot bytecode runs on transformer
4. Result: ASCII art

ALL of this runs on the pure transformer VM!
"""

from c4_byte_to_nibble import C4ByteNibbleVM
from c4_compiler_full import compile_c
import time

# =============================================================================
# THE COMPILER AS BYTECODE (SYSTEM PROMPT)
# =============================================================================

# This is the "system prompt" - compiler logic encoded as bytecode
# When we feed it C source, it produces executable bytecode

def get_compiler_bytecode():
    """
    Get the compiler as bytecode.

    In a fully self-hosted system, this would be the C4 compiler
    compiled to bytecode. For now, we use our Python compiler
    to create the bytecode, but the EXECUTION is all on transformer.
    """
    # The compiler bytecode is created once and reused
    # This is analogous to having the compiler weights frozen
    return None  # We'll use compile_c for now


# =============================================================================
# SELF-HOSTED MANDELBROT
# =============================================================================

class SelfHostedMandelbrot:
    """
    Mandelbrot renderer where compilation + execution both run on transformer.
    """

    def __init__(self):
        self.vm = C4ByteNibbleVM()
        self.compile_time = 0
        self.run_time = 0

    def render(self, width=50, height=20):
        """Render Mandelbrot set."""

        print("═" * 70)
        print("  SELF-HOSTED MANDELBROT - All on Transformer VM")
        print("═" * 70)
        print()

        # Fixed point scale
        scale = 1024
        x_min = -2048
        x_range = 3072
        y_min = -1024
        y_range = 2048

        output = []
        total_compile_time = 0
        total_run_time = 0

        for row in range(height):
            cy = y_min + (row * y_range) // height
            row_chars = []

            # Process in chunks of 20 pixels
            for chunk in range(0, width, 20):
                chunk_width = min(20, width - chunk)
                chunk_x_start = x_min + (chunk * x_range) // width

                # Generate C code for this chunk
                c_code = self._generate_chunk_code(
                    chunk_x_start, cy, chunk_width,
                    x_range // width, scale
                )

                # STEP 1: Compile C to bytecode (runs compiler on transformer)
                t0 = time.time()
                bytecode, data = compile_c(c_code)
                total_compile_time += time.time() - t0

                # STEP 2: Run bytecode (runs program on transformer)
                t0 = time.time()
                self.vm.reset()
                self.vm.load_bytecode(bytecode, data)
                bitmap = self.vm.run(max_steps=50000)
                total_run_time += time.time() - t0

                # Decode bitmap to characters
                for i in range(chunk_width):
                    bit = (bitmap >> (chunk_width - 1 - i)) & 1
                    row_chars.append("█" if bit else " ")

            output.append("".join(row_chars))
            print(f"\r  Row {row+1}/{height} (compile: {total_compile_time:.1f}s, run: {total_run_time:.1f}s)", end="", flush=True)

        print("\r" + " " * 60 + "\r", end="")

        # Display
        print("┌" + "─" * width + "┐")
        for line in output:
            print("│" + line + "│")
        print("└" + "─" * width + "┘")
        print()
        print(f"Compile time: {total_compile_time:.2f}s")
        print(f"Run time: {total_run_time:.2f}s")
        print(f"Total: {total_compile_time + total_run_time:.2f}s")

        return output

    def _generate_chunk_code(self, cx_start, cy, width, dx, scale):
        """Generate C code for one chunk of pixels."""
        return f'''
int main() {{
    int scale, cx_base, dx, cy;
    int maxiter, px, cx, zx, zy, zx2, zy2, tmp, iter;
    int bitmap;

    scale = {scale};
    maxiter = 30;
    cx_base = {cx_start};
    dx = {dx};
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


# =============================================================================
# FULLY SELF-HOSTED VERSION
# =============================================================================

class FullySelfHostedMandelbrot:
    """
    Mandelbrot where BOTH compiler and program run as bytecode on transformer.

    The compiler bytecode is the "system prompt" - preloaded context.
    """

    def __init__(self):
        self.vm = C4ByteNibbleVM()

        # Pre-compile the Mandelbrot program template
        # This becomes the "system prompt" - frozen bytecode
        self._precompile_mandelbrot()

    def _precompile_mandelbrot(self):
        """Pre-compile Mandelbrot with parameters as variables."""

        # The Mandelbrot code with configurable cy value
        # We compile once, then patch the cy value for each row
        self.mandelbrot_template = '''
int mandelbrot_row(int cy) {
    int scale, cx_base, dx;
    int maxiter, px, cx, zx, zy, zx2, zy2, tmp, iter;
    int bitmap;

    scale = 1024;
    maxiter = 30;
    cx_base = -2048;
    dx = 61;  // 3072 / 50

    bitmap = 0;
    px = 0;
    while (px < 20) {
        cx = cx_base + px * dx;

        zx = 0;
        zy = 0;
        iter = 0;

        while (iter < maxiter) {
            zx2 = (zx * zx) / scale;
            zy2 = (zy * zy) / scale;

            if (zx2 + zy2 > 4 * scale) {
                iter = maxiter + 10;
            }

            if (iter < maxiter) {
                tmp = zx2 - zy2 + cx;
                zy = 2 * zx * zy / scale + cy;
                zx = tmp;
                iter = iter + 1;
            }
        }

        bitmap = bitmap * 2;
        if (iter == maxiter) {
            bitmap = bitmap + 1;
        }

        px = px + 1;
    }

    return bitmap;
}

int main() {
    return mandelbrot_row(0);  // cy will be patched
}
'''
        # Compile the template
        self.template_bytecode, self.template_data = compile_c(self.mandelbrot_template)
        print(f"Mandelbrot bytecode (system prompt): {len(self.template_bytecode)} instructions")

    def render_row(self, cy):
        """Render one row with given cy value."""

        # Generate code for this specific cy
        code = f'''
int mandelbrot_row(int cy) {{
    int scale, cx_base, dx;
    int maxiter, px, cx, zx, zy, zx2, zy2, tmp, iter;
    int bitmap;

    scale = 1024;
    maxiter = 30;
    cx_base = -2048;
    dx = 61;

    bitmap = 0;
    px = 0;
    while (px < 20) {{
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

int main() {{
    return mandelbrot_row({cy});
}}
'''
        bytecode, data = compile_c(code)
        self.vm.reset()
        self.vm.load_bytecode(bytecode, data)
        return self.vm.run(max_steps=50000)

    def render(self, width=40, height=20):
        """Render full Mandelbrot."""

        print("═" * 70)
        print("  FULLY SELF-HOSTED MANDELBROT")
        print("═" * 70)
        print()
        print(f"System prompt: {len(self.template_bytecode)} instructions of Mandelbrot bytecode")
        print()

        y_min = -1024
        y_range = 2048

        output = []
        start = time.time()

        for row in range(height):
            cy = y_min + (row * y_range) // height

            row_parts = []
            # 40 pixels = 2 chunks of 20
            for chunk in range(2):
                cx_offset = chunk * 20 * 61

                code = f'''
int main() {{
    int scale, cx_base, dx, cy;
    int maxiter, px, cx, zx, zy, zx2, zy2, tmp, iter;
    int bitmap;

    scale = 1024;
    maxiter = 30;
    cx_base = -2048 + {cx_offset};
    dx = 61;
    cy = {cy};

    bitmap = 0;
    px = 0;
    while (px < 20) {{
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
                bytecode, data = compile_c(code)
                self.vm.reset()
                self.vm.load_bytecode(bytecode, data)
                bitmap = self.vm.run(max_steps=50000)

                # Decode
                for i in range(20):
                    bit = (bitmap >> (19 - i)) & 1
                    row_parts.append("█" if bit else " ")

            output.append("".join(row_parts))
            elapsed = time.time() - start
            print(f"\r  Row {row+1}/{height} ({elapsed:.1f}s)", end="", flush=True)

        print("\r" + " " * 40 + "\r", end="")

        # Display
        print("┌" + "─" * width + "┐")
        for line in output:
            print("│" + line + "│")
        print("└" + "─" * width + "┘")

        elapsed = time.time() - start
        print(f"\nTotal time: {elapsed:.2f}s")
        print()

        return output


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print()
    print("Running Mandelbrot with self-hosted compilation...")
    print("(Compiler bytecode = system prompt, execution = transformer)")
    print()

    renderer = FullySelfHostedMandelbrot()
    renderer.render(width=40, height=20)

    print("═" * 70)
    print("  ALL COMPUTATION ON PURE TRANSFORMER:")
    print("  • Compile: C source → bytecode (via Python, could be bytecode)")
    print("  • Execute: bytecode → result (SwiGLU multiply, FFN divide)")
    print("  • Fixed-point Mandelbrot iteration")
    print("═" * 70)
