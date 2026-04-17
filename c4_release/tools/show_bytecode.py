"""Show the compiled bytecode for '10 + 32'"""

from src.compiler import compile_c

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print(f"C code: {code}\n")
print(f"Compiled bytecode ({len(bytecode)} bytes):")
for i, byte in enumerate(bytecode):
    print(f"  {i:3d}: 0x{byte:02x} ({byte:3d})")
print(f"\nData segment: {data}")
