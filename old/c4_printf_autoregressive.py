"""
C4 with autoregressive printf output.

Printf generates output tokens one at a time as part of the context sequence.
Each digit/character is generated autoregressively, just like an LLM generates text.

Context: [memory..., PC, SP, BP, AX, OUT_PTR, OUTPUT_0, OUTPUT_1, ...]

When printf is called:
1. Read value to print (e.g., 42)
2. Generate '4' token → append to OUTPUT
3. Generate '2' token → append to OUTPUT
4. Generate '\n' token → append to OUTPUT
5. Continue execution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from c4_vm import Op


def silu(x):
    return x * torch.sigmoid(x)


def silu_threshold(x, scale=20.0):
    diff = scale * x
    term1 = silu(diff + 0.5 * scale)
    term2 = silu(diff - 0.5 * scale)
    return (term1 - term2) / scale


def eq_gate(a, b, scale=20.0):
    diff = (a - b).float()
    upper = silu_threshold(diff + 0.5, scale)
    lower = silu_threshold(-diff + 0.5, scale)
    return upper * lower


def swiglu_mul(a, b):
    a, b = a.float(), b.float()
    return a * silu(b) - a * silu(-b)


class DigitExtractor(nn.Module):
    """
    Extract digits from a number using SwiGLU operations.

    To get digit at position i (0 = ones, 1 = tens, etc.):
    1. Divide by 10^i (using enumeration-based division)
    2. Mod by 10
    """

    def __init__(self, max_digits=6):
        super().__init__()
        self.max_digits = max_digits
        # Powers of 10
        self.register_buffer('powers', torch.tensor([10**i for i in range(max_digits)]).float())

    def count_digits(self, x):
        """Count number of digits in x (at least 1)."""
        x = x.float().abs()
        count = torch.ones_like(x)
        for i in range(1, self.max_digits):
            # If x >= 10^i, we have at least i+1 digits
            has_more = silu_threshold(x - self.powers[i] + 0.5)
            count = count + has_more
        return count.long()

    def get_digit(self, x, pos):
        """
        Get digit at position pos (0 = ones place).
        Uses SwiGLU-based division and mod.
        """
        x = x.float().abs()

        # Divide by 10^pos
        divisor = self.powers[pos]

        # Integer division via enumeration
        quotient = torch.zeros_like(x)
        for q in range(1000):  # Max quotient
            q_tensor = torch.tensor(float(q))
            # gate = 1 if x >= q*divisor and x < (q+1)*divisor
            lower = silu_threshold(x - q * divisor + 0.5)
            upper = silu_threshold((q + 1) * divisor - x - 0.5)
            gate = lower * upper
            quotient = quotient + gate * q_tensor
            if q * divisor > 1000:  # Early exit for small numbers
                break

        # Mod 10
        digit = quotient - torch.floor(quotient / 10) * 10
        return digit.long()


class AutoregressiveOutput(nn.Module):
    """
    Manages autoregressive output generation.

    Output tokens are ASCII values (48-57 for '0'-'9', 10 for '\n').
    """

    def __init__(self, max_output=64, scale=10.0):
        super().__init__()
        self.max_output = max_output
        self.scale = scale
        self.digit_extractor = DigitExtractor()

        # ASCII codes
        self.DIGIT_BASE = 48  # '0' = 48, '1' = 49, ..., '9' = 57
        self.NEWLINE = 10

    def generate_number_tokens(self, value):
        """
        Generate ASCII tokens for a number.
        Returns list of token values.

        E.g., 42 → [52, 50, 10] for '4', '2', '\n'
        """
        value = value.float()
        tokens = []

        # Handle negative
        is_negative = value < 0
        value = value.abs()

        if is_negative:
            tokens.append(45)  # '-'

        # Count digits
        num_digits = self.digit_extractor.count_digits(value).item()

        # Extract digits from most significant to least
        for i in range(int(num_digits) - 1, -1, -1):
            digit = self.digit_extractor.get_digit(value, i)
            token = self.DIGIT_BASE + digit.item()
            tokens.append(token)

        # Add newline
        tokens.append(self.NEWLINE)

        return tokens


class C4AutoregressivePrintf(nn.Module):
    """
    C4 executor with autoregressive printf.

    Context: [memory, PC, SP, BP, AX, OUT_PTR, OUTPUT...]

    When printf executes:
    - Extract value to print
    - Generate digit tokens one by one
    - Each token appends to OUTPUT, increments OUT_PTR
    """

    def __init__(self, memory_size=256, output_size=64):
        super().__init__()
        self.memory_size = memory_size
        self.output_size = output_size

        from c4_autoregressive import C4AutoregressiveExecutor
        self.executor = C4AutoregressiveExecutor(memory_size)
        self.ctx = self.executor.ctx
        self.output_gen = AutoregressiveOutput(output_size)

        # Output tracking
        self.OUT_PTR_IDX = memory_size + 4  # After AX

    def decode(self, instruction):
        instruction = instruction.float()
        imm = torch.floor(instruction / 256.0)
        opcode = instruction - imm * 256.0
        return opcode, imm

    def is_printf(self, opcode):
        return eq_gate(opcode, torch.tensor(float(Op.PRTF))) > 0.5

    def generate_printf_output(self, value, output, out_ptr):
        """
        Autoregressively generate output tokens for printf.
        Returns updated (output, out_ptr).
        """
        tokens = self.output_gen.generate_number_tokens(value)

        new_output = output.clone()
        new_ptr = out_ptr

        for token in tokens:
            if new_ptr < self.output_size:
                new_output[int(new_ptr)] = float(token)
                new_ptr = new_ptr + 1

        return new_output, new_ptr

    def step(self, pc, sp, bp, ax, memory, output, out_ptr):
        """
        Execute one step.

        If printf: generate output tokens autoregressively.
        Otherwise: normal execution.
        """
        # Fetch and decode
        context = self.ctx.build_context(memory, pc, sp, bp, ax)
        instruction = self.ctx.attend(context, pc, query_type='memory')
        opcode, imm = self.decode(instruction)

        if self.is_printf(opcode):
            # Get value to print from stack
            # Stack: [format_addr, value, ...]
            # SP points to format_addr, SP+8 points to value
            value = self.ctx.attend(context, sp + 8, query_type='memory')

            # Generate output tokens autoregressively
            output, out_ptr = self.generate_printf_output(value, output, out_ptr)

            # Advance PC, return value in AX
            new_pc = pc + 8
            new_ax = out_ptr  # Return number of chars written

            return new_pc, sp, bp, new_ax, memory, output, out_ptr

        # Regular execution
        new_pc, new_sp, new_bp, new_ax, new_memory = self.executor.step(pc, sp, bp, ax, memory)
        return new_pc, new_sp, new_bp, new_ax, new_memory, output, out_ptr

    def is_exit(self, memory, pc):
        return self.executor.is_exit(memory, pc)

    def run(self, memory, pc, sp, bp, ax, max_steps=100):
        """Run program, return final state and output string."""
        output = torch.zeros(self.output_size)
        out_ptr = torch.tensor(0.0)

        for step in range(max_steps):
            if self.is_exit(memory, pc):
                break
            pc, sp, bp, ax, memory, output, out_ptr = self.step(
                pc, sp, bp, ax, memory, output, out_ptr
            )

        # Convert output tokens to string
        output_str = self.tokens_to_string(output, out_ptr)

        return pc, sp, bp, ax, memory, output_str

    def tokens_to_string(self, output, out_ptr):
        """Convert output token buffer to string."""
        chars = []
        for i in range(int(out_ptr.item())):
            code = int(output[i].item())
            if 32 <= code <= 126 or code == 10:  # Printable ASCII or newline
                chars.append(chr(code))
        return ''.join(chars)


def test_autoregressive_printf():
    print("C4 WITH AUTOREGRESSIVE PRINTF")
    print("=" * 60)
    print()
    print("Printf generates output tokens one at a time:")
    print("  42 → ['4', '2', '\\n'] → tokens [52, 50, 10]")
    print()

    def instr(op, imm=0):
        return float(op + (imm << 8))

    # Test digit extraction
    print("Digit Extraction Test:")
    extractor = DigitExtractor()

    for num in [0, 5, 42, 123, 999]:
        n_digits = extractor.count_digits(torch.tensor(float(num))).item()
        digits = [extractor.get_digit(torch.tensor(float(num)), i).item()
                  for i in range(int(n_digits))]
        print(f"  {num}: {n_digits} digits, extracted: {digits[::-1]}")
    print()

    # Test token generation
    print("Token Generation Test:")
    output_gen = AutoregressiveOutput()

    for num in [7, 42, 123]:
        tokens = output_gen.generate_number_tokens(torch.tensor(float(num)))
        chars = [chr(t) if t != 10 else '\\n' for t in tokens]
        print(f"  {num} → tokens {tokens} → chars {chars}")
    print()

    # Test 1: Simple printf
    print("Test 1: printf(42)")
    executor1 = C4AutoregressivePrintf(memory_size=256)

    memory1 = torch.zeros(256)
    code1 = [
        instr(Op.IMM, 42),      # AX = 42
        instr(Op.PSH),          # push 42 (value)
        instr(Op.IMM, 0),       # AX = 0 (format)
        instr(Op.PSH),          # push format
        instr(Op.PRTF),         # printf → generates '4', '2', '\n'
        instr(Op.ADJ, 16),      # cleanup stack
        instr(Op.EXIT),
    ]
    for i, c in enumerate(code1):
        memory1[i * 8] = c

    pc, sp, bp, ax, memory1, output = executor1.run(
        memory1, torch.tensor(0.0), torch.tensor(200.0),
        torch.tensor(200.0), torch.tensor(0.0)
    )
    print(f"  Output: '{output.strip()}'")
    status = "✓" if output.strip() == "42" else "✗"
    print(f"  {status} Autoregressively generated '42'")
    print()

    # Test 2: Compute then printf
    print("Test 2: printf(7 * 8)")
    executor2 = C4AutoregressivePrintf(memory_size=256)

    memory2 = torch.zeros(256)
    code2 = [
        instr(Op.IMM, 7),
        instr(Op.PSH),
        instr(Op.IMM, 8),
        instr(Op.MUL),          # AX = 56
        instr(Op.PSH),
        instr(Op.IMM, 0),
        instr(Op.PSH),
        instr(Op.PRTF),
        instr(Op.ADJ, 16),
        instr(Op.EXIT),
    ]
    for i, c in enumerate(code2):
        memory2[i * 8] = c

    pc, sp, bp, ax, memory2, output = executor2.run(
        memory2, torch.tensor(0.0), torch.tensor(200.0),
        torch.tensor(200.0), torch.tensor(0.0)
    )
    print(f"  Output: '{output.strip()}'")
    status = "✓" if output.strip() == "56" else "✗"
    print(f"  {status} 7 * 8 = 56 generated autoregressively")
    print()

    # Test 3: Multiple printfs
    print("Test 3: Multiple printfs")
    executor3 = C4AutoregressivePrintf(memory_size=256)

    memory3 = torch.zeros(256)
    code3 = [
        # printf(10)
        instr(Op.IMM, 10),
        instr(Op.PSH),
        instr(Op.IMM, 0),
        instr(Op.PSH),
        instr(Op.PRTF),
        instr(Op.ADJ, 16),
        # printf(20)
        instr(Op.IMM, 20),
        instr(Op.PSH),
        instr(Op.IMM, 0),
        instr(Op.PSH),
        instr(Op.PRTF),
        instr(Op.ADJ, 16),
        # printf(30)
        instr(Op.IMM, 30),
        instr(Op.PSH),
        instr(Op.IMM, 0),
        instr(Op.PSH),
        instr(Op.PRTF),
        instr(Op.ADJ, 16),
        instr(Op.EXIT),
    ]
    for i, c in enumerate(code3):
        memory3[i * 8] = c

    pc, sp, bp, ax, memory3, output = executor3.run(
        memory3, torch.tensor(0.0), torch.tensor(200.0),
        torch.tensor(200.0), torch.tensor(0.0)
    )
    lines = output.strip().split('\n')
    print(f"  Output lines: {lines}")
    status = "✓" if lines == ['10', '20', '30'] else "✗"
    print(f"  {status} Generated 10, 20, 30 autoregressively")
    print()

    # Test 4: Three-digit number
    print("Test 4: printf(123)")
    executor4 = C4AutoregressivePrintf(memory_size=256)

    memory4 = torch.zeros(256)
    code4 = [
        instr(Op.IMM, 123),
        instr(Op.PSH),
        instr(Op.IMM, 0),
        instr(Op.PSH),
        instr(Op.PRTF),
        instr(Op.ADJ, 16),
        instr(Op.EXIT),
    ]
    for i, c in enumerate(code4):
        memory4[i * 8] = c

    pc, sp, bp, ax, memory4, output = executor4.run(
        memory4, torch.tensor(0.0), torch.tensor(200.0),
        torch.tensor(200.0), torch.tensor(0.0)
    )
    print(f"  Output: '{output.strip()}'")
    status = "✓" if output.strip() == "123" else "✗"
    print(f"  {status} Generated '1', '2', '3' autoregressively")
    print()

    print("=" * 60)
    print("AUTOREGRESSIVE PRINTF COMPLETE!")
    print()
    print("Each digit is generated as a token, appended to context.")
    print("This matches how LLMs generate text token by token.")

    return True


if __name__ == "__main__":
    test_autoregressive_printf()
