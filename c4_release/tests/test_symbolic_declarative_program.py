from c4_release.neural_vm.verification.symbolic_program import (
    SymbolicDeclarativeProgramRunner,
    build_default_opcode_declarations,
)
from c4_release.src.compiler import compile_c
from c4_release.tests.test_suite_1000 import generate_test_programs


def _run_c(source, *, max_steps=50000):
    bytecode, data = compile_c(source)
    return SymbolicDeclarativeProgramRunner().run(
        bytecode,
        data,
        max_steps=max_steps,
    )


def test_opcode_declarations_cover_core_mul_semantics():
    specs = build_default_opcode_declarations()

    assert specs[27].name == "MUL"
    assert specs[27].reads == frozenset({"AX", "SP", "MEM"})
    assert specs[27].writes == frozenset({"AX", "SP"})


def test_symbolic_program_runner_executes_larger_mul_cases():
    state = _run_c("int main() { return 65 * 98; }")
    assert state.halted
    assert state.ax == 6370

    state = _run_c("int main() { return 65 * 78; }")
    assert state.halted
    assert state.ax == 5070


def test_symbolic_program_runner_executes_compiler_control_flow():
    state = _run_c("""
        int gcd(int a, int b) {
            int temp;
            while (b != 0) {
                temp = b;
                b = a % b;
                a = temp;
            }
            return a;
        }
        int main() { return gcd(84, 30); }
    """)

    assert state.halted
    assert state.ax == 6
    assert any(step.name in {"BZ", "BNZ", "JMP"} for step in state.trace)


def test_symbolic_declarative_runner_passes_full_1096_suite():
    runner = SymbolicDeclarativeProgramRunner()
    failures = []

    for idx, (source, expected, description) in enumerate(generate_test_programs()):
        bytecode, data = compile_c(source)
        state = runner.run(bytecode, data, max_steps=50000)
        if not state.halted or state.ax != (expected & 0xFFFFFFFF):
            failures.append(
                (idx, description, expected, state.ax, state.halted, state.steps)
            )
            if len(failures) >= 10:
                break

    assert failures == []
