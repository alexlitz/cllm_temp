"""
Test Phase 2: Multi-Layer Opcodes

Tests the 8 opcodes that require multi-layer compilation (FFN + Attention + FFN):
- LI, LC, SI, SC (memory operations)
- PSH (stack push)
- JSR, ENT, LEV (function call operations)
"""

from neural_vm.opcode_nibble_integration import OpcodeNibbleCompiler
from neural_vm.embedding import Opcode

def test_phase2_opcodes():
    """Test all 8 multi-layer opcodes."""
    compiler = OpcodeNibbleCompiler()

    # Phase 2 opcodes that need multi-layer compilation
    opcodes = [
        (Opcode.LI, "LI", "Load Indirect: AX = *AX"),
        (Opcode.LC, "LC", "Load Char: AX = *(char*)AX"),
        (Opcode.SI, "SI", "Store Indirect: *AX = pop"),
        (Opcode.SC, "SC", "Store Char: *(char*)AX = pop"),
        (Opcode.PSH, "PSH", "Push: *(--SP) = AX"),
        (Opcode.JSR, "JSR", "Jump Subroutine: *(--SP) = PC; PC = AX"),
        (Opcode.ENT, "ENT", "Enter: *(--SP) = BP; BP = SP; SP -= imm"),
        (Opcode.LEV, "LEV", "Leave: SP = BP; BP = *SP; SP += 8"),
    ]

    print("="*70)
    print("Testing 8 Multi-Layer Opcodes (Phase 2)")
    print("="*70)
    print()

    results = []

    for opcode, name, description in opcodes:
        print(f"{name} ({opcode}): {description}")
        print(f"  Support level: {compiler.get_support_level(opcode)}")

        try:
            # Compile multi-layer opcode
            layer_weights = compiler.compile_multilayer_opcode(opcode)

            print(f"  ✅ Compiled successfully!")
            print(f"     Layers: {len(layer_weights)}")

            # Count non-zero params for each layer
            for layer_idx in sorted(layer_weights.keys()):
                weights = layer_weights[layer_idx]

                if weights is None:
                    print(f"     Layer {layer_idx}: Attention (no FFN)")
                    continue

                # Count non-zero params
                nonzero = 0
                total = 0
                for tensor in weights.values():
                    nonzero += (tensor.abs() > 1e-9).sum().item()
                    total += tensor.numel()

                sparsity = 100.0 * (1 - nonzero / total) if total > 0 else 100.0

                print(f"     Layer {layer_idx}: {nonzero:,} non-zero / {total:,} total ({sparsity:.2f}% sparse)")

            results.append((name, True, layer_weights))

        except NotImplementedError as e:
            print(f"  ⚠️  Not implemented: {e}")
            results.append((name, False, None))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, None))

        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    compiled = sum(1 for _, success, _ in results if success)
    print(f"Compiled: {compiled}/8")
    print()

    if compiled > 0:
        print("✅ Working opcodes:")
        for name, success, layer_weights in results:
            if success:
                total_params = 0
                for weights in layer_weights.values():
                    if weights:
                        for tensor in weights.values():
                            total_params += (tensor.abs() > 1e-9).sum().item()
                print(f"   {name}: {total_params} total non-zero params across layers")

    failed = [(name, success) for name, success, _ in results if not success]
    if failed:
        print()
        print("❌ Failed opcodes:")
        for name, _ in failed:
            print(f"   {name}")

if __name__ == "__main__":
    test_phase2_opcodes()
