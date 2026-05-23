import pytest

from neural_vm.speculative import DraftVM
from src.compiler import Op, compile_c


def _decode(bytecode):
    return [(instr & 0xFF, instr >> 8) for instr in bytecode]


def _run_draft(source, max_steps=1000):
    bytecode, data = compile_c(source)
    vm = DraftVM(bytecode)
    vm.load_data(bytes(data))
    steps = vm.predict_steps(max_steps)
    return bytecode, vm.ax, vm.halted, steps


def test_leaf_main_strips_startup_and_duplicate_exit():
    bytecode, ax, halted, steps = _run_draft("int main() { return 42; }")

    assert _decode(bytecode) == [(Op.IMM, 42), (Op.EXIT, 0)]
    assert (ax, halted, steps) == (42, True, 2)


def test_main_with_locals_keeps_startup_and_runs_in_draftvm():
    source = "int main() { int a; int b; a = 10; b = 20; return a + b; }"
    bytecode, ax, halted, _ = _run_draft(source)
    ops = _decode(bytecode)

    assert ops[:4] == [(Op.JSR, 3), (Op.EXIT, 0), (Op.NOP, 0), (Op.ENT, 16)]
    assert ax == 30
    assert halted


def test_forward_call_patches_to_later_definition():
    source = """
    int main() { return add(10, 20); }
    int add(int a, int b) { return a + b; }
    """
    bytecode, ax, halted, _ = _run_draft(source)
    jsr_targets = [imm for op, imm in _decode(bytecode) if op == Op.JSR]

    assert jsr_targets[0] == 3
    assert all(0 <= target < len(bytecode) for target in jsr_targets)
    assert all(target != 0 for target in jsr_targets[1:])
    assert ax == 30
    assert halted


def test_function_prototypes_allow_mutual_recursion():
    source = """
    int is_odd(int n);
    int is_even(int n) {
        if (n == 0) return 1;
        return is_odd(n - 1);
    }
    int is_odd(int n) {
        if (n == 0) return 0;
        return is_even(n - 1);
    }
    int main() { return is_even(10); }
    """
    _, ax, halted, _ = _run_draft(source)

    assert ax == 1
    assert halted


def test_undefined_forward_call_is_rejected():
    with pytest.raises(SyntaxError, match="Undefined function: missing"):
        compile_c("int main() { return missing(); }")


def test_stdlib_is_not_linked_for_comments_or_user_definitions():
    comment_bytecode, comment_ax, comment_halted, _ = _run_draft(
        "/* malloc(16) */ int main() { return 7; }"
    )
    user_bytecode, user_ax, user_halted, _ = _run_draft(
        "int malloc(int n) { return n + 1; } int main() { return malloc(4); }"
    )

    assert _decode(comment_bytecode) == [(Op.IMM, 7), (Op.EXIT, 0)]
    assert (comment_ax, comment_halted) == (7, True)
    assert len(user_bytecode) < 40
    assert (user_ax, user_halted) == (5, True)


def test_stdlib_functions_remain_callable_when_unresolved():
    source = """
    int main() {
        int p;
        p = malloc(16);
        free(p);
        return p != 0;
    }
    """
    bytecode, ax, halted, steps = _run_draft(source, max_steps=2000)

    assert len(bytecode) > 100
    assert ax == 1
    assert halted
    assert steps < 2000
