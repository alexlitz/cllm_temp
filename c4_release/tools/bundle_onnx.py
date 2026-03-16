#!/usr/bin/env python3
"""
ONNX Model Bundler

Bundles a C4 ONNX model with the runtime into a standalone executable.

Pipeline:
1. ONNX model -> onnx_to_text.py -> text format
2. text format -> text_to_c4onnx -> binary format
3. binary + runtime -> onnx_bundler -> C code
4. C code -> gcc -> executable (optional)

Usage:
    python bundle_onnx.py model.onnx output_name [--compile]
    python bundle_onnx.py model.c4onnx output_name [--compile]
    python bundle_onnx.py model.txt output_name [--compile]
"""

import argparse
import subprocess
import sys
import os
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()


def run_cmd(cmd, description=None):
    """Run a command and return success status."""
    if description:
        print(f"  {description}...")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True


def ensure_tools():
    """Ensure bundler tool is compiled."""
    bundler = SCRIPT_DIR / "onnx_bundler"
    bundler_c = SCRIPT_DIR / "onnx_bundler.c"

    if not bundler.exists() or bundler_c.stat().st_mtime > bundler.stat().st_mtime:
        print("Compiling onnx_bundler...")
        if not run_cmd(f"gcc -O2 -o {bundler} {bundler_c}"):
            print("Failed to compile onnx_bundler")
            return False

    text_to_c4 = SCRIPT_DIR / "text_to_c4onnx"
    text_to_c4_c = SCRIPT_DIR / "text_to_c4onnx.c"

    if not text_to_c4.exists() or text_to_c4_c.stat().st_mtime > text_to_c4.stat().st_mtime:
        print("Compiling text_to_c4onnx...")
        if not run_cmd(f"gcc -O2 -o {text_to_c4} {text_to_c4_c}"):
            print("Failed to compile text_to_c4onnx")
            return False

    return True


def convert_onnx_to_text(onnx_path, text_path):
    """Convert ONNX to text format."""
    converter = SCRIPT_DIR / "onnx_to_text.py"
    return run_cmd(f"python3 {converter} {onnx_path} {text_path}", "Converting ONNX to text")


def convert_text_to_binary(text_path, binary_path):
    """Convert text to C4 ONNX binary."""
    converter = SCRIPT_DIR / "text_to_c4onnx"
    return run_cmd(f"{converter} {text_path} {binary_path}", "Converting text to binary")


def bundle_model(binary_path, output_c, program_path=None):
    """Bundle model with runtime into C code."""
    bundler = SCRIPT_DIR / "onnx_bundler"

    if program_path:
        cmd = f"{bundler} {binary_path} {program_path} > {output_c}"
    else:
        cmd = f"{bundler} {binary_path} > {output_c}"

    return run_cmd(cmd, "Bundling model with runtime")


def compile_to_executable(c_path, exe_path):
    """Compile C code to executable."""
    return run_cmd(f"gcc -O2 -o {exe_path} {c_path}", "Compiling to executable")


def main():
    parser = argparse.ArgumentParser(description="Bundle ONNX model into standalone executable")
    parser.add_argument("input", help="Input model (.onnx, .txt, or .c4onnx)")
    parser.add_argument("output", help="Output name (without extension)")
    parser.add_argument("--program", "-p", help="Optional user program C file")
    parser.add_argument("--compile", "-c", action="store_true", help="Compile to executable")
    parser.add_argument("--keep-intermediate", "-k", action="store_true",
                        help="Keep intermediate files")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_base = Path(args.output).resolve()

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Ensure tools are compiled
    if not ensure_tools():
        return 1

    print(f"Bundling {input_path.name}...")

    # Determine input format and convert as needed
    suffix = input_path.suffix.lower()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        if suffix == ".onnx":
            # ONNX -> text -> binary
            text_path = tmpdir / "model.txt"
            binary_path = tmpdir / "model.c4onnx"

            if not convert_onnx_to_text(input_path, text_path):
                return 1
            if not convert_text_to_binary(text_path, binary_path):
                return 1

        elif suffix == ".txt":
            # text -> binary
            binary_path = tmpdir / "model.c4onnx"
            if not convert_text_to_binary(input_path, binary_path):
                return 1

        elif suffix in [".c4onnx", ".bin"]:
            # Already binary
            binary_path = input_path

        else:
            print(f"Error: Unknown format: {suffix}")
            return 1

        # Bundle
        output_c = output_base.with_suffix(".c")
        if not bundle_model(binary_path, output_c, args.program):
            return 1

        # Optionally compile
        if args.compile:
            output_exe = output_base
            if not compile_to_executable(output_c, output_exe):
                return 1

            print(f"\nGenerated executable: {output_exe}")

            # Show size
            size = output_exe.stat().st_size
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            print(f"Size: {size_str}")

            if not args.keep_intermediate:
                output_c.unlink()
        else:
            print(f"\nGenerated C code: {output_c}")
            print(f"Compile with: gcc -O2 -o {output_base} {output_c}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
