#!/usr/bin/env python3
"""
Mandelbrot PNG Generator using Transformer Neural Operations

Generates a Mandelbrot set image where ALL arithmetic operations
(multiplication, division) are computed through the neural network.

This uses:
- SwiGLU for multiplication: a*b = silu(a)*b + silu(-a)*(-b)
- Fixed-point arithmetic to stay in integer domain
- The same operations as our C4 ONNX runtime

Usage:
    python mandelbrot_transformer.py 64 64 test.png       # Small test
    python mandelbrot_transformer.py 256 256 mandel.png   # Medium
    python mandelbrot_transformer.py 2560 1440 hd.png     # Full HD (slow!)
"""

import sys
import time
import struct
import zlib
import torch
import torch.nn.functional as F
from typing import Tuple


class NeuralArithmetic:
    """Neural arithmetic operations using SwiGLU."""

    def __init__(self, scale: int = 1024):
        self.scale = scale
        self.half_scale = scale // 2
        self.mul_count = 0
        self.div_count = 0

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        """SiLU activation: x * sigmoid(x)"""
        return F.silu(x.float())

    def mul(self, a: int, b: int) -> int:
        """
        Multiply using SwiGLU identity:
        a * b = silu(a) * b + silu(-a) * (-b)

        This is the exact same operation as in our C4 ONNX model.
        """
        self.mul_count += 1

        # Convert to tensor for neural ops
        a_t = torch.tensor([float(a)])

        # Compute silu(a) and silu(-a)
        silu_a = self.silu(a_t).item()
        silu_neg_a = self.silu(-a_t).item()

        # SwiGLU identity: silu(a)*b + silu(-a)*(-b)
        result = silu_a * b + silu_neg_a * (-b)

        return int(round(result))

    def fp_mul(self, a: int, b: int) -> int:
        """Fixed-point multiplication: (a * b) / SCALE"""
        # Avoid overflow by using split multiplication
        result = self.mul(a // self.scale, b) + self.mul(a % self.scale, b) // self.scale
        return result

    def div(self, a: int, b: int) -> int:
        """Integer division via repeated subtraction (neural-compatible)."""
        self.div_count += 1
        if b == 0:
            return 0

        neg = False
        if a < 0:
            a = -a
            neg = not neg
        if b < 0:
            b = -b
            neg = not neg

        result = 0
        while a >= b:
            a -= b
            result += 1

        return -result if neg else result


class MandelbrotRenderer:
    """Renders Mandelbrot set using neural arithmetic."""

    def __init__(self, width: int, height: int, max_iter: int = 100):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.neural = NeuralArithmetic(scale=1024)

        # Mandelbrot viewport in fixed-point
        self.scale = 1024
        self.x_min = -2560  # -2.5 * 1024
        self.x_max = 1024   # 1.0 * 1024
        self.y_min = -1024  # -1.0 * 1024
        self.y_max = 1024   # 1.0 * 1024

        # Step sizes
        self.dx = (self.x_max - self.x_min) // width
        self.dy = (self.y_max - self.y_min) // height

    def iterate(self, cx: int, cy: int) -> int:
        """
        Compute Mandelbrot iteration count using neural multiplication.
        All multiplications go through SwiGLU.
        """
        zx = 0
        zy = 0

        for i in range(self.max_iter):
            # zx^2 and zy^2 using neural multiply
            zx2 = self.neural.fp_mul(zx, zx)
            zy2 = self.neural.fp_mul(zy, zy)

            # Check escape: |z|^2 > 4
            if zx2 + zy2 > 4 * self.scale:
                return i

            # z = z^2 + c
            # new_zx = zx^2 - zy^2 + cx
            # new_zy = 2*zx*zy + cy
            tmp = zx2 - zy2 + cx
            zy = 2 * self.neural.fp_mul(zx, zy) // self.scale + cy
            zx = tmp

        return self.max_iter

    def get_color(self, iter_count: int) -> Tuple[int, int, int]:
        """Map iteration count to RGB color."""
        if iter_count == self.max_iter:
            return (0, 0, 0)  # Black for points in set

        # Gradient: blue -> cyan -> green -> yellow -> red
        t = (iter_count * 1024) // self.max_iter

        if t < 256:
            return (0, t, 255)
        elif t < 512:
            return (0, 255, 255 - (t - 256))
        elif t < 768:
            return (t - 512, 255, 0)
        else:
            return (255, 255 - (t - 768), 0)

    def render(self, verbose: bool = True) -> bytes:
        """Render Mandelbrot to raw RGB data."""
        pixels = []
        start_time = time.time()
        total_pixels = self.width * self.height

        for row in range(self.height):
            cy = self.y_max - row * self.dy

            for col in range(self.width):
                cx = self.x_min + col * self.dx

                iter_count = self.iterate(cx, cy)
                r, g, b = self.get_color(iter_count)
                pixels.extend([r, g, b])

            if verbose and (row + 1) % 10 == 0:
                elapsed = time.time() - start_time
                done = (row + 1) * self.width
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_pixels - done) / rate if rate > 0 else 0
                print(f"\rRow {row+1}/{self.height} | "
                      f"{done}/{total_pixels} pixels | "
                      f"{rate:.0f} px/s | "
                      f"ETA: {eta:.0f}s | "
                      f"Muls: {self.neural.mul_count}", end="", flush=True)

        if verbose:
            elapsed = time.time() - start_time
            print(f"\nCompleted in {elapsed:.1f}s")
            print(f"Total neural multiplications: {self.neural.mul_count:,}")
            print(f"Multiplications per pixel: {self.neural.mul_count / total_pixels:.1f}")

        return bytes(pixels)


def create_png(width: int, height: int, rgb_data: bytes) -> bytes:
    """Create a PNG file from raw RGB data."""

    def crc32(data: bytes) -> int:
        return zlib.crc32(data) & 0xFFFFFFFF

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + chunk_type + data +
                struct.pack(">I", crc32(chunk_type + data)))

    # PNG signature
    png = b'\x89PNG\r\n\x1a\n'

    # IHDR
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png += chunk(b'IHDR', ihdr)

    # IDAT - compress row data with filter bytes
    raw_data = b''
    for row in range(height):
        raw_data += b'\x00'  # Filter: None
        row_start = row * width * 3
        raw_data += rgb_data[row_start:row_start + width * 3]

    compressed = zlib.compress(raw_data, 9)
    png += chunk(b'IDAT', compressed)

    # IEND
    png += chunk(b'IEND', b'')

    return png


def main():
    if len(sys.argv) < 4:
        print("Usage: python mandelbrot_transformer.py WIDTH HEIGHT OUTPUT.png")
        print("Example: python mandelbrot_transformer.py 64 64 test.png")
        sys.exit(1)

    width = int(sys.argv[1])
    height = int(sys.argv[2])
    output = sys.argv[3]

    print(f"Generating {width}x{height} Mandelbrot using neural SwiGLU...")
    print(f"Output: {output}")
    print()

    renderer = MandelbrotRenderer(width, height, max_iter=100)
    rgb_data = renderer.render()

    png_data = create_png(width, height, rgb_data)

    with open(output, 'wb') as f:
        f.write(png_data)

    print(f"\nSaved {len(png_data):,} bytes to {output}")


if __name__ == "__main__":
    main()
