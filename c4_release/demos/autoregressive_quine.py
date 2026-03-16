"""
Autoregressive Neural Quine

The quine source (cllm/quine_cllm.c) is compiled to C4 bytecode, encoded as
the context prefix for the autoregressive transformer, and the model generates
tokens step-by-step. PUTCHAR steps emit characters that reproduce the source.

This is a 100% autoregressive quine: every character of output flows through
the transformer's forward pass (embed → attention → FFN → head → argmax).

Required opcodes (16 total):
  Control flow: LEA, IMM, JMP, JSR, BZ, ENT, ADJ, LEV
  Memory/stack: LI, LC, SI, PSH
  Computation:  EQ, ADD
  I/O:          PUTCHAR, EXIT

Usage:
    python demos/autoregressive_quine.py           # status check
    python demos/autoregressive_quine.py --run     # run (needs all opcodes baked)
    python demos/autoregressive_quine.py --verify  # compile + verify quine property
    python demos/autoregressive_quine.py --bundle  # bundle into standalone C file
    python demos/autoregressive_quine.py --bundle --compile  # bundle + compile with gcc
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler import compile_c, Op
from neural_vm.vm_step import Token, AutoregressiveVM
from neural_vm.run_vm import AutoregressiveVMRunner

QUINE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'cllm', 'quine_cllm.c')

# The 16 opcodes the quine needs
REQUIRED_OPS = {
    Op.LEA, Op.IMM, Op.JMP, Op.JSR, Op.BZ, Op.ENT, Op.ADJ, Op.LEV,
    Op.LI, Op.LC, Op.SI, Op.PSH,
    Op.EQ, Op.ADD,
    38,  # EXIT
    65,  # PUTCHAR
}

# Opcodes with baked weights in the autoregressive VM
BAKED_OPS = {Op.IMM, 38}  # IMM and EXIT only


def compile_quine():
    """Compile the quine source to C4 bytecode."""
    with open(QUINE_PATH) as f:
        source = f.read()
    code, data = compile_c(source)
    return source, code, data


def status():
    """Print status of quine readiness."""
    source, code, data = compile_quine()

    print("Autoregressive Neural Quine")
    print("=" * 50)
    print(f"Source: {QUINE_PATH}")
    print(f"Source length: {len(source)} chars")
    print(f"Bytecode: {len(code)} instructions")
    print(f"Data section: {len(data)} bytes")

    # Context size
    context_tokens = 2 + len(code) * 5 + 2 + len(data)  # CS + code + CE/DS + data + DE
    print(f"Context prefix: {context_tokens} tokens")

    # Opcode readiness
    op_names = {v: k for k, v in Op.__members__.items()}
    op_names[65] = 'PUTCHAR'

    print(f"\nOpcodes needed: {len(REQUIRED_OPS)}")
    baked = REQUIRED_OPS & BAKED_OPS
    missing = REQUIRED_OPS - BAKED_OPS
    print(f"  Baked:   {len(baked)}/{len(REQUIRED_OPS)}", end="")
    if baked:
        print(f"  ({', '.join(op_names.get(o, str(o)) for o in sorted(baked))})")
    else:
        print()
    print(f"  Missing: {len(missing)}/{len(REQUIRED_OPS)}", end="")
    if missing:
        print(f"  ({', '.join(op_names.get(o, str(o)) for o in sorted(missing))})")
    else:
        print()

    # Estimate tokens for full execution
    # Each VM step = 30 tokens. Quine has ~550 chars output, each char needs
    # several VM steps (loop iteration + putchar call)
    est_steps = len(code) * 5  # rough estimate
    print(f"\nEstimated VM steps: ~{est_steps}")
    print(f"Estimated tokens: ~{est_steps * 30}")

    if missing:
        print(f"\nStatus: WAITING ({len(missing)} opcodes need baking)")
        print("\nOpcode categories needed:")
        categories = {
            'Control flow': [Op.LEA, Op.JMP, Op.JSR, Op.BZ, Op.ENT, Op.ADJ, Op.LEV],
            'Memory/stack': [Op.LI, Op.LC, Op.SI, Op.PSH],
            'Computation': [Op.EQ, Op.ADD],
            'I/O': [65],  # PUTCHAR
        }
        for cat, ops in categories.items():
            cat_missing = [o for o in ops if o in missing]
            cat_baked = [o for o in ops if o in baked]
            if cat_missing:
                print(f"  {cat}: {len(cat_baked)}/{len(ops)} baked"
                      f" (need: {', '.join(op_names.get(o, str(o)) for o in cat_missing)})")
            else:
                print(f"  {cat}: {len(ops)}/{len(ops)} done")
    else:
        print("\nStatus: READY")

    return missing


def run_quine():
    """Run the quine on the autoregressive VM."""
    source, code, data = compile_quine()

    missing = REQUIRED_OPS - BAKED_OPS
    if missing:
        op_names = {v: k for k, v in Op.__members__.items()}
        op_names[65] = 'PUTCHAR'
        print(f"Cannot run: {len(missing)} opcodes not yet baked:")
        for o in sorted(missing):
            print(f"  {op_names.get(o, str(o))}")
        print("\nBake these opcodes in vm_step.py, then re-run.")
        return False

    runner = AutoregressiveVMRunner()

    # Import and apply weight baking
    from test_bake_v2 import bake_v2
    bake_v2(runner.model)

    print("Running autoregressive quine...")
    output, exit_code = runner.run(code, bytes(data), max_steps=100000)

    print(f"Exit code: {exit_code}")
    print(f"Output length: {len(output)} chars")

    if output == source:
        print("QUINE VERIFIED: output == source")
        return True
    else:
        print("QUINE FAILED: output != source")
        # Show diff
        for i, (a, b) in enumerate(zip(output, source)):
            if a != b:
                print(f"  First diff at char {i}: got {repr(a)}, expected {repr(b)}")
                break
        return False


def verify_with_c4():
    """Verify the quine property using the standard (non-neural) C4 VM."""
    source, code, data = compile_quine()

    print("Verifying quine property with C4 interpreter...")
    print(f"Source: {len(source)} chars")
    print(f"Bytecode: {len(code)} instructions")

    # Try to run with the speculator/reference VM if available
    try:
        from tests.test_speculator import run_c4_reference
        output = run_c4_reference(code, data)
        if output == source:
            print("VERIFIED: C4 VM output == source (quine property holds)")
            return True
        else:
            print("FAILED: output != source")
            return False
    except ImportError:
        print("(C4 reference VM not available for verification)")
        print("Compile and run natively to verify:")
        print(f"  gcc -o /tmp/quine {QUINE_PATH}")
        print(f"  /tmp/quine > /tmp/quine_out.c")
        print(f"  diff {QUINE_PATH} /tmp/quine_out.c")
        return None


def bundle_quine():
    """Bundle the quine into a standalone C file.

    Bakes model weights, exports to .arvm, generates bundled C file.
    With --compile, also compiles with gcc.
    """
    import tempfile
    import subprocess

    from neural_vm.vm_step import AutoregressiveVM
    from test_bake_v2 import bake_v2
    from tools.export_autoregressive import export_autoregressive
    from tools.bundle_autoregressive_quine import bundle

    # Create and bake model
    print("Creating and baking model...")
    model = AutoregressiveVM()
    bake_v2(model)
    model.eval()

    # Export to .arvm
    arvm_path = os.path.join(tempfile.gettempdir(), 'quine_baked.arvm')
    export_autoregressive(model, arvm_path)

    # Bundle into standalone C file
    c4_mode = '--c4' in sys.argv
    fp32_mode = '--fp32' in sys.argv
    quine_mode = '--quine' in sys.argv
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'build', 'bundled')
    suffix_parts = []
    if c4_mode:
        suffix_parts.append('c4')
    if fp32_mode:
        suffix_parts.append('fp32')
    if quine_mode:
        suffix_parts.append('quine')
    suffix = '_'.join(suffix_parts)
    if suffix:
        output_path = os.path.join(output_dir, f'quine_autoregressive_{suffix}.c')
    else:
        output_path = os.path.join(output_dir, 'quine_autoregressive.c')

    bundle(arvm_path, QUINE_PATH, output_path, c4=c4_mode, fp32=fp32_mode,
           quine=quine_mode)

    # Optionally compile with gcc
    if '--compile' in sys.argv:
        exe_path = output_path.replace('.c', '')
        print(f"\nCompiling with gcc...")
        result = subprocess.run(
            ['gcc', '-O2', '-o', exe_path, output_path],
            capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Compiled: {exe_path}")
            print(f"\nRun with: {exe_path}")
        else:
            print(f"Compilation failed:")
            print(result.stderr)

    # Clean up temp file
    try:
        os.unlink(arvm_path)
    except OSError:
        pass


def main():
    if '--run' in sys.argv:
        run_quine()
    elif '--verify' in sys.argv:
        verify_with_c4()
    elif '--bundle' in sys.argv:
        bundle_quine()
    else:
        status()


if __name__ == '__main__':
    main()
