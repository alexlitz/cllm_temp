# Conversational I/O - TODO List

## Status Overview

**✅ Complete**: Transformer detection and THINKING_END generation
**🔄 In Progress**: Full integration testing
**📋 Todo**: Runner-side features and edge cases

---

## High Priority: Integration & Core Features

### 1. ✅ Full Integration Test (IN PROGRESS)
- [x] Create test that runs through `runner.run()`
- [ ] Verify THINKING_END is generated during execution
- [ ] Verify runner's hybrid output injection works
- [ ] Confirm program continues after printf
- [ ] Test with simple literal string (no format specifiers)

**Status**: Test created (`test_full_integration.py`), running now

### 2. 📋 Format String Handling (Runner-side)

**Current**: Runner reads format string and emits raw bytes
**Needed**: Parse and handle format specifiers

```python
# Need to implement in run_vm.py:
def _parse_format_string(fmt_str, stack_args):
    """Parse format string with %d, %x, %s, %c specifiers.

    Args:
        fmt_str: Format string bytes
        stack_args: Arguments from stack

    Returns:
        List of output bytes
    """
    # Handle:
    # - %d (decimal integer)
    # - %x (hexadecimal)
    # - %s (string pointer)
    # - %c (character)
    # - %% (literal %)
    # - Literal text between specifiers
```

**Priority**: HIGH (needed for real printf functionality)

**Test Cases**:
```c
printf("Number: %d\n", 42);      // Format decimal
printf("Hex: 0x%x\n", 255);      // Format hex
printf("String: %s\n", "hello"); // Dereference string
printf("Char: %c\n", 'A');       // Single character
printf("Mixed: %d %s\n", 10, "items"); // Multiple args
```

### 3. 📋 Multiple Printf Calls

**Test**: Does the system handle sequential I/O?

```c
int main() {
    printf("First\n");
    printf("Second\n");
    printf("Third\n");
    return 0;
}
```

**Questions**:
- Does THINKING_START properly reset state?
- Does each printf trigger THINKING_END?
- Are output strings kept separate or concatenated?

**Priority**: MEDIUM

---

## Medium Priority: READ Opcode Support

### 4. 📋 READ Detection Pipeline

Similar to PRTF, but for input:

**Needed**:
- [ ] L5 FFN already detects READ (unit 411)
- [ ] L6 relay already copies IO_IS_READ
- [ ] State machine needs to handle READ → THINKING_END
- [ ] Runner needs to handle input injection

**Test Case**:
```c
int main() {
    int x = read();
    printf("You entered: %d\n", x);
    return 0;
}
```

**Priority**: MEDIUM (important for interactivity)

### 5. 📋 User Input Parsing

**Current**: Runner can append USER_INPUT tokens
**Needed**: Extract input from conversational context

```python
# When THINKING_END is emitted for READ:
def _extract_user_input(context):
    """Parse text after </thinking> as user input.

    Should support:
    - Text after </thinking> tag
    - Multiple lines
    - Special characters
    - EOF handling
    """
```

**Priority**: MEDIUM

---

## Low Priority: Edge Cases & Polish

### 6. 📋 Edge Case Testing

**Empty/Null Cases**:
```c
printf("");           // Empty string
printf(NULL);         // Null pointer (should handle gracefully)
printf("%d");         // Missing argument (undefined behavior)
```

**Long Output**:
```c
// 1000-character string
// Does token generation handle it?
```

**Format Edge Cases**:
```c
printf("%%d");        // Literal %d (not format specifier)
printf("%5d", 42);    // Width specifiers
printf("%.2f", 3.14); // Precision (if supporting float)
```

**Priority**: LOW (nice to have)

### 7. 📋 Performance & Efficiency

**Tests Needed**:
- [ ] Benchmark: 100 printf calls
- [ ] Memory usage with large outputs
- [ ] Token generation speed during I/O

**Optimizations**:
- [ ] Batch output bytes before emitting
- [ ] Cache format string parsing
- [ ] Reuse THINKING_START token

**Priority**: LOW (only if performance is an issue)

### 8. 📋 Error Handling

**Scenarios**:
- Invalid format specifiers
- Stack underflow (missing arguments)
- Memory read errors (invalid string pointers)
- Infinite loops in printf

**Current**: No explicit error handling
**Needed**: Graceful degradation

**Priority**: LOW (unless causing crashes)

---

## Documentation & Examples

### 9. 📋 User Documentation

**Needed**:
- [ ] README section on conversational I/O
- [ ] Example programs
- [ ] Limitations and known issues
- [ ] API documentation

**Example Programs**:
```c
// Echo program
int main() {
    printf("Enter name: ");
    char name[64];
    read_line(name, 64);
    printf("Hello, %s!\n", name);
    return 0;
}
```

**Priority**: MEDIUM (for usability)

### 10. 📋 Performance Documentation

Document:
- Token overhead (THINKING_END + bytes + THINKING_START)
- Memory requirements
- Limitations (max output size, etc.)

**Priority**: LOW

---

## Future Enhancements

### 11. 📋 Advanced Format Specifiers

- [ ] Width and precision: `%5d`, `%.2f`
- [ ] Flags: `%-5d`, `%+d`, `%#x`
- [ ] Length modifiers: `%ld`, `%lld`
- [ ] Float support: `%f`, `%e`, `%g` (requires float arithmetic)

**Priority**: VERY LOW (nice to have)

### 12. 📋 Other I/O Functions

- [ ] `puts()` - simpler than printf
- [ ] `putchar()` - already has Python handler
- [ ] `getchar()` - already has Python handler
- [ ] `fgets()` - read line
- [ ] `scanf()` - formatted input

**Priority**: VERY LOW (after basic printf/read work)

### 13. 📋 Streaming Output

Instead of buffering entire output:
- [ ] Stream bytes as they're generated
- [ ] Real-time output display
- [ ] Interrupt/cancel support

**Priority**: VERY LOW (future optimization)

---

## Testing Strategy

### Immediate Tests (This Session)
1. ✅ Unit tests (all passing)
2. 🔄 Full integration test (running)
3. 📋 Simple literal string printf
4. 📋 Program with return value after printf

### Next Session Tests
1. 📋 Format specifier tests (%d, %x, %s)
2. 📋 Multiple printf calls
3. 📋 READ opcode test
4. 📋 Mixed I/O (printf + read)

### Long-term Tests
1. 📋 Stress test (100+ I/O operations)
2. 📋 Edge cases (null, empty, malformed)
3. 📋 Performance benchmarks
4. 📋 Real programs (games, utilities)

---

## Summary: What's Actually Blocking?

### Blocking for Basic Functionality
1. **Full integration test** - Need to verify end-to-end
2. **Format string parsing** - Printf without %d/%x/%s is limited

### Nice to Have
- Multiple printf handling
- READ support
- Edge case handling
- Documentation

### Future Work
- Advanced format specifiers
- Other I/O functions
- Performance optimizations

---

## Next Steps

1. **Wait for integration test** to complete
2. **If it passes**: Move to format string parsing
3. **If it fails**: Debug runner-side issues
4. **Then**: Implement %d, %x, %s format specifiers
5. **Finally**: Test multiple printf calls

The transformer side is **done**. All remaining work is **Python runner** implementation and testing.
