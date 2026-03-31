#!/usr/bin/env python3
"""Test I/O support in speculative execution."""
import pytest
from src.compiler import compile_c
from neural_vm.batch_runner import BatchedSpeculativeRunner


@pytest.fixture
def runner():
    return BatchedSpeculativeRunner(batch_size=1)


def test_simple_printf(runner):
    """Test basic printf with string literal."""
    source = 'int main() { printf("Hello World\\n"); return 0; }'
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=200)
    output, exit_code = results[0]
    assert output == "Hello World\n"
    assert exit_code == 0


def test_printf_integer(runner):
    """Test printf with %d format."""
    source = '''
    int main() {
        int x;
        x = 42;
        printf("x=%d\\n", x);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=300)
    output, exit_code = results[0]
    assert output == "x=42\n"
    assert exit_code == 0


def test_printf_multiple_args(runner):
    """Test printf with multiple arguments."""
    source = '''
    int main() {
        int a;
        int b;
        a = 10;
        b = 32;
        printf("%d + %d = %d\\n", a, b, a + b);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=400)
    output, exit_code = results[0]
    assert output == "10 + 32 = 42\n"
    assert exit_code == 0


def test_printf_hex(runner):
    """Test printf with %x format."""
    source = '''
    int main() {
        int x;
        x = 255;
        printf("0x%x\\n", x);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=300)
    output, exit_code = results[0]
    assert output == "0xff\n"
    assert exit_code == 0


def test_printf_char(runner):
    """Test printf with %c format."""
    source = '''
    int main() {
        int x;
        x = 65;
        printf("Char: %c\\n", x);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=300)
    output, exit_code = results[0]
    assert output == "Char: A\n"
    assert exit_code == 0


def test_printf_negative(runner):
    """Test printf with negative numbers."""
    source = '''
    int main() {
        int x;
        x = -42;
        printf("x=%d\\n", x);
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=300)
    output, exit_code = results[0]
    assert output == "x=-42\n"
    assert exit_code == 0


def test_multiple_printfs(runner):
    """Test multiple printf calls."""
    source = '''
    int main() {
        printf("Line 1\\n");
        printf("Line 2\\n");
        printf("Line 3\\n");
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=400)
    output, exit_code = results[0]
    assert output == "Line 1\nLine 2\nLine 3\n"
    assert exit_code == 0


def test_printf_in_loop(runner):
    """Test printf in a loop."""
    source = '''
    int main() {
        int i;
        i = 0;
        while (i < 3) {
            printf("i=%d\\n", i);
            i = i + 1;
        }
        return 0;
    }
    '''
    bytecode, data = compile_c(source)
    results = runner.run_batch([bytecode], [data], max_steps=500)
    output, exit_code = results[0]
    assert output == "i=0\ni=1\ni=2\n"
    assert exit_code == 0
