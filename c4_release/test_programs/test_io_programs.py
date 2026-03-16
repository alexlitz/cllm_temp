#!/usr/bin/env python3
"""
Test I/O Programs (echo, yes, cat)

Tests the test programs with the streaming I/O system to verify
end-to-end correctness.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler import compile_c
from llm_io import LLMSession


def run_program(source: str, inputs: list = None, max_output: int = 10000) -> str:
    """
    Compile and run a C program, returning its output.

    Args:
        source: C source code
        inputs: List of input strings (for programs that read stdin)
        max_output: Maximum output length

    Returns:
        Program output as string
    """
    bytecode, data = compile_c(source)

    session = LLMSession(llm_mode=False)
    session.load(bytecode, data)

    output = session.run_until_input_or_done()

    if inputs:
        for inp in inputs:
            if session.waiting_for_input and not session.finished:
                output += session.provide_input(inp)

            if len(output) > max_output:
                break

    return output[:max_output]


def test_echo():
    """Test echo program."""
    print("=" * 60)
    print("TEST: echo")
    print("=" * 60)

    # Simple echo without arguments (just newline)
    source = '''
int main() {
    putchar(10);
    return 0;
}
'''
    output = run_program(source)
    expected = "\n"
    passed = output == expected
    print(f"  echo (no args): '{repr(output)}' == '{repr(expected)}' {'✓' if passed else '✗'}")

    # Echo "Hello"
    source = '''
int main() {
    char *s;
    s = "Hello";
    while (*s) {
        putchar(*s);
        s = s + 1;
    }
    putchar(10);
    return 0;
}
'''
    output = run_program(source)
    expected = "Hello\n"
    passed2 = output == expected
    print(f"  echo Hello: '{repr(output)}' == '{repr(expected)}' {'✓' if passed2 else '✗'}")

    # Echo "Hello World"
    source = '''
int main() {
    char *s;
    s = "Hello World";
    while (*s) {
        putchar(*s);
        s = s + 1;
    }
    putchar(10);
    return 0;
}
'''
    output = run_program(source)
    expected = "Hello World\n"
    passed3 = output == expected
    print(f"  echo Hello World: '{repr(output)}' == '{repr(expected)}' {'✓' if passed3 else '✗'}")

    all_passed = passed and passed2 and passed3
    print(f"\n  Result: {'✓ PASS' if all_passed else '✗ FAIL'}")
    return all_passed


def test_yes():
    """Test yes program (limited output)."""
    print("\n" + "=" * 60)
    print("TEST: yes")
    print("=" * 60)

    # Yes with default "y"
    source = '''
int main() {
    int i;
    i = 0;
    while (i < 5) {
        putchar(121);
        putchar(10);
        i = i + 1;
    }
    return 0;
}
'''
    output = run_program(source)
    expected = "y\n" * 5
    passed = output == expected
    print(f"  yes (5 lines): {len(output)} chars, {output.count('y')} y's {'✓' if passed else '✗'}")

    # Yes with custom string
    source = '''
int main() {
    int i;
    char *msg;
    char *s;

    msg = "hello";
    i = 0;
    while (i < 3) {
        s = msg;
        while (*s) {
            putchar(*s);
            s = s + 1;
        }
        putchar(10);
        i = i + 1;
    }
    return 0;
}
'''
    output = run_program(source)
    expected = "hello\n" * 3
    passed2 = output == expected
    print(f"  yes hello (3 lines): '{repr(output[:30])}...' {'✓' if passed2 else '✗'}")

    all_passed = passed and passed2
    print(f"\n  Result: {'✓ PASS' if all_passed else '✗ FAIL'}")
    return all_passed


def test_cat():
    """Test cat program with streaming input."""
    print("\n" + "=" * 60)
    print("TEST: cat (streaming I/O)")
    print("=" * 60)

    # Cat with small input - read until newline (10) - no break, use while condition
    source = '''
int main() {
    int c;
    int done;
    done = 0;
    while (done == 0) {
        c = getchar();
        if (c == 0) {
            done = 1;
        } else if (c == 10) {
            done = 1;
        } else {
            putchar(c);
        }
    }
    putchar(10);
    return 0;
}
'''

    bytecode, data = compile_c(source)
    session = LLMSession(llm_mode=False)
    session.load(bytecode, data)

    # Run until input needed
    output = session.run_until_input_or_done()

    # Provide input
    test_input = "Hello from cat!"
    if session.waiting_for_input:
        output += session.provide_input(test_input)

    expected = test_input + "\n"
    passed = output == expected
    print(f"  cat '{test_input}': '{repr(output)}' == '{repr(expected)}' {'✓' if passed else '✗'}")

    # Cat with limited count - no break
    source = '''
int main() {
    int c;
    int count;
    int done;
    count = 0;
    done = 0;
    while (done == 0) {
        c = getchar();
        if (c == 0) {
            done = 1;
        } else if (count >= 20) {
            done = 1;
        } else {
            putchar(c);
            count = count + 1;
        }
    }
    return 0;
}
'''

    bytecode, data = compile_c(source)
    session = LLMSession(llm_mode=False)
    session.load(bytecode, data)

    output = session.run_until_input_or_done()

    test_input = "Line1\nLine2\n"
    if session.waiting_for_input:
        output += session.provide_input(test_input)

    # Should echo back the input (up to 20 chars)
    passed2 = test_input[:20] in output or output.startswith(test_input[:10])
    print(f"  cat multiline: output contains input {'✓' if passed2 else '✗'}")

    all_passed = passed and passed2
    print(f"\n  Result: {'✓ PASS' if all_passed else '✗ FAIL'}")
    return all_passed


def test_cat_large():
    """Test cat with larger input to verify streaming works at scale."""
    print("\n" + "=" * 60)
    print("TEST: cat (large input)")
    print("=" * 60)

    # Cat that reads up to 500 characters - no break
    source = '''
int main() {
    int c;
    int count;
    int done;
    count = 0;
    done = 0;
    while (done == 0) {
        c = getchar();
        if (c == 0) {
            done = 1;
        } else if (count >= 500) {
            done = 1;
        } else {
            putchar(c);
            count = count + 1;
        }
    }
    return 0;
}
'''

    bytecode, data = compile_c(source)
    session = LLMSession(llm_mode=False)
    session.load(bytecode, data)

    output = session.run_until_input_or_done()

    # Generate large input
    large_input = "ABCDEFGHIJ" * 50  # 500 chars
    if session.waiting_for_input:
        output += session.provide_input(large_input)

    # Check that output matches input (up to 500 chars)
    match_count = 0
    for i, c in enumerate(output[:500]):
        if i < len(large_input) and c == large_input[i]:
            match_count += 1

    accuracy = match_count / min(500, len(output)) if output else 0
    passed = accuracy > 0.95

    print(f"  Input: {len(large_input)} chars")
    print(f"  Output: {len(output)} chars")
    print(f"  Match accuracy: {accuracy:.1%}")
    print(f"\n  Result: {'✓ PASS' if passed else '✗ FAIL'}")

    return passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("I/O PROGRAM TEST SUITE")
    print("=" * 60)
    print()

    results = []

    results.append(("echo", test_echo()))
    results.append(("yes", test_yes()))
    results.append(("cat", test_cat()))
    results.append(("cat (large)", test_cat_large()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
