"""
C4 with Thinking Token Tags.

Internal computations (register predictions) are tagged as 'thinking' tokens.
Output tokens (printf) are tagged as 'output' tokens.

This matches the reasoning model paradigm where:
- <thinking> tokens = internal reasoning/computation
- <output> tokens = visible results

Context layout with types:
  [mem_0, ..., mem_n, PC, SP, BP, AX, OUT_0, OUT_1, ...]

Type tags (5 bits now):
  memory:   [0,0,0,0,0] - addressable memory
  PC:       [1,0,0,0,0] - thinking token
  SP:       [0,1,0,0,0] - thinking token
  BP:       [0,0,1,0,0] - thinking token
  AX:       [0,0,0,1,0] - thinking token (result register)
  OUTPUT:   [0,0,0,0,1] - output token

Each step generates thinking tokens (PC→SP→BP→AX) before producing output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from c4_vm import Op


# Token type constants
TYPE_MEMORY = 0
TYPE_THINKING_PC = 1
TYPE_THINKING_SP = 2
TYPE_THINKING_BP = 3
TYPE_THINKING_AX = 4
TYPE_OUTPUT = 5


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


class ThinkingContext(nn.Module):
    """
    Context with explicit thinking token tags.

    Each token has:
    - value (the data)
    - type_tag (which kind of token)
    - thinking_flag (is this a thinking/internal token?)

    Thinking tokens: PC, SP, BP, AX (invisible to user, internal computation)
    Output tokens: printf output (visible result)
    Memory tokens: regular memory (addressable data)
    """

    def __init__(self, memory_size=256, output_size=32, scale=10.0):
        super().__init__()
        self.memory_size = memory_size
        self.output_size = output_size
        self.scale = scale

        # Position indices
        self.PC_IDX = memory_size
        self.SP_IDX = memory_size + 1
        self.BP_IDX = memory_size + 2
        self.AX_IDX = memory_size + 3
        self.OUT_START = memory_size + 4

        # Total context size
        self.total_size = memory_size + 4 + output_size

        # Binary address encoder
        self.num_addr_bits = int(math.ceil(math.log2(self.total_size)))
        self.register_buffer('powers', 2 ** torch.arange(self.num_addr_bits))

        # Type tag templates (using -N/+N encoding for sharp matching)
        # 6 types: memory, PC, SP, BP, AX, output
        self.num_type_bits = 6

        type_tags = torch.full((self.total_size, self.num_type_bits), -scale)

        # Memory tokens: all -N (type 0)
        # Already initialized above

        # Thinking tokens: PC, SP, BP, AX
        type_tags[self.PC_IDX, 0] = scale   # PC thinking
        type_tags[self.SP_IDX, 1] = scale   # SP thinking
        type_tags[self.BP_IDX, 2] = scale   # BP thinking
        type_tags[self.AX_IDX, 3] = scale   # AX thinking

        # Output tokens
        for i in range(output_size):
            type_tags[self.OUT_START + i, 4] = scale

        # Thinking flag: 1 for thinking tokens, 0 for others
        thinking_flags = torch.zeros(self.total_size)
        thinking_flags[self.PC_IDX] = 1.0
        thinking_flags[self.SP_IDX] = 1.0
        thinking_flags[self.BP_IDX] = 1.0
        thinking_flags[self.AX_IDX] = 1.0

        # Output flag
        output_flags = torch.zeros(self.total_size)
        output_flags[self.OUT_START:self.OUT_START + output_size] = 1.0

        self.register_buffer('type_tags', type_tags)
        self.register_buffer('thinking_flags', thinking_flags)
        self.register_buffer('output_flags', output_flags)

        # Build full key matrix (address + type)
        addresses = torch.arange(self.total_size)
        addr_keys = self._encode_address(addresses)
        full_keys = torch.cat([addr_keys, type_tags], dim=1)
        self.register_buffer('all_keys', full_keys)

    def _encode_address(self, x):
        """Binary encode address to (-N, +N) vectors."""
        x = x.long().unsqueeze(-1)
        bits = (x // self.powers) % 2
        return (2 * bits.float() - 1) * self.scale

    def build_context(self, memory, pc, sp, bp, ax, output=None, out_ptr=0):
        """Build full context sequence with type tags."""
        context = torch.zeros(self.total_size)
        context[:self.memory_size] = memory[:self.memory_size].float()
        context[self.PC_IDX] = pc.float()
        context[self.SP_IDX] = sp.float()
        context[self.BP_IDX] = bp.float()
        context[self.AX_IDX] = ax.float()

        if output is not None:
            context[self.OUT_START:self.OUT_START + len(output)] = output.float()

        return context

    def attend(self, context, query_addr, query_type='memory'):
        """Attend to context position with type filtering."""
        addr_query = self._encode_address(query_addr.unsqueeze(0)).squeeze(0)

        # Get type tag for query
        type_idx = {
            'memory': None,  # Default all -N
            'pc': 0,
            'sp': 1,
            'bp': 2,
            'ax': 3,
            'output': 4,
        }.get(query_type)

        type_query = torch.full((self.num_type_bits,), -self.scale)
        if type_idx is not None:
            type_query[type_idx] = self.scale

        query = torch.cat([addr_query, type_query])

        scores = torch.matmul(self.all_keys[:len(context)], query)
        scores = scores / math.sqrt(len(query))
        weights = F.softmax(scores, dim=0)

        return torch.sum(weights * context)

    def get_thinking_tokens(self, context):
        """Extract only the thinking token values."""
        return {
            'PC': context[self.PC_IDX].item(),
            'SP': context[self.SP_IDX].item(),
            'BP': context[self.BP_IDX].item(),
            'AX': context[self.AX_IDX].item(),
        }

    def get_output_tokens(self, context, out_ptr):
        """Extract only the output token values."""
        tokens = []
        for i in range(int(out_ptr)):
            tokens.append(int(context[self.OUT_START + i].item()))
        return tokens

    def format_trace(self, context, out_ptr, step_num):
        """Format a trace showing thinking vs output tokens."""
        thinking = self.get_thinking_tokens(context)
        output = self.get_output_tokens(context, out_ptr)

        lines = [f"Step {step_num}:"]
        lines.append(f"  <thinking>")
        lines.append(f"    PC={int(thinking['PC']):3d}  SP={int(thinking['SP']):3d}  "
                    f"BP={int(thinking['BP']):3d}  AX={int(thinking['AX']):3d}")
        lines.append(f"  </thinking>")

        if output:
            chars = ''.join(chr(c) if 32 <= c <= 126 else f'\\x{c:02x}' for c in output)
            lines.append(f"  <output>{chars}</output>")

        return '\n'.join(lines)


class C4WithThinking(nn.Module):
    """
    C4 executor that tracks thinking vs output tokens.

    During execution:
    - Register updates are <thinking> tokens (internal computation)
    - Printf output is <output> tokens (visible result)

    This matches reasoning models where:
    - Thinking = chain-of-thought, scratchpad, planning
    - Output = final answer visible to user
    """

    def __init__(self, memory_size=256, output_size=32):
        super().__init__()
        self.memory_size = memory_size
        self.output_size = output_size

        self.ctx = ThinkingContext(memory_size, output_size)

        # Import base executor
        from c4_autoregressive import C4AutoregressiveExecutor
        self.executor = C4AutoregressiveExecutor(memory_size)

    def decode(self, instruction):
        instruction = instruction.float()
        imm = torch.floor(instruction / 256.0)
        opcode = instruction - imm * 256.0
        return opcode, imm

    def step(self, pc, sp, bp, ax, memory, output, out_ptr, trace=False):
        """
        Execute one step, generating thinking tokens then possibly output tokens.

        Returns: (pc, sp, bp, ax, memory, output, out_ptr, trace_str)
        """
        # Build context with thinking tokens
        context = self.ctx.build_context(memory, pc, sp, bp, ax, output, out_ptr)

        # Fetch instruction
        instruction = self.ctx.attend(context, pc, 'memory')
        opcode, imm = self.decode(instruction)

        # Check for printf
        is_printf = eq_gate(opcode, torch.tensor(float(Op.PRTF))) > 0.5

        if is_printf:
            # Get value from stack
            value = self.ctx.attend(context, sp + 8, 'memory')

            # Generate output tokens (ASCII digits)
            val_int = abs(int(value.item()))
            digits = str(val_int)

            new_output = output.clone()
            new_ptr = out_ptr

            # Generate digit tokens as output
            for ch in digits + '\n':
                if new_ptr < self.output_size:
                    new_output[int(new_ptr)] = float(ord(ch))
                    new_ptr = new_ptr + 1

            # Update thinking tokens (PC advances)
            new_pc = pc + 8

            trace_str = ""
            if trace:
                new_ctx = self.ctx.build_context(memory, new_pc, sp, bp, ax, new_output, new_ptr)
                trace_str = self.ctx.format_trace(new_ctx, new_ptr, 0)

            return new_pc, sp, bp, ax, memory, new_output, new_ptr, trace_str

        # Regular execution: generate thinking tokens
        new_pc, new_sp, new_bp, new_ax, new_memory = self.executor.step(
            pc, sp, bp, ax, memory
        )

        trace_str = ""
        if trace:
            new_ctx = self.ctx.build_context(new_memory, new_pc, new_sp, new_bp, new_ax, output, out_ptr)
            trace_str = self.ctx.format_trace(new_ctx, out_ptr, 0)

        return new_pc, new_sp, new_bp, new_ax, new_memory, output, out_ptr, trace_str

    def is_exit(self, memory, pc):
        return self.executor.is_exit(memory, pc)

    def run(self, memory, pc, sp, bp, ax, max_steps=100, trace=False):
        """Run program, showing thinking vs output token separation."""
        output = torch.zeros(self.output_size)
        out_ptr = torch.tensor(0.0)

        all_traces = []

        for step in range(max_steps):
            if self.is_exit(memory, pc):
                break

            pc, sp, bp, ax, memory, output, out_ptr, trace_str = self.step(
                pc, sp, bp, ax, memory, output, out_ptr, trace=trace
            )

            if trace and trace_str:
                all_traces.append(trace_str)

        # Convert output to string
        output_str = ''.join(
            chr(int(output[i].item()))
            for i in range(int(out_ptr.item()))
            if 32 <= int(output[i].item()) <= 126 or int(output[i].item()) == 10
        )

        return pc, sp, bp, ax, memory, output_str, all_traces


def test_thinking_tokens():
    print("C4 WITH THINKING TOKEN TAGS")
    print("=" * 60)
    print()
    print("Token types:")
    print("  <thinking> PC, SP, BP, AX - internal register state")
    print("  <output>   printf chars   - visible output")
    print("  <memory>   addresses      - data storage")
    print()

    def instr(op, imm=0):
        return float(op + (imm << 8))

    # Test: compute and print
    print("Test: 3 + 4 → printf")
    executor = C4WithThinking(memory_size=256)

    memory = torch.zeros(256)
    code = [
        instr(Op.IMM, 3),       # AX = 3
        instr(Op.PSH),          # push 3
        instr(Op.IMM, 4),       # AX = 4
        instr(Op.ADD),          # AX = 3 + 4 = 7
        instr(Op.PSH),          # push 7
        instr(Op.IMM, 0),       # format addr
        instr(Op.PSH),          # push format
        instr(Op.PRTF),         # printf(7) → generates output tokens
        instr(Op.ADJ, 16),      # cleanup
        instr(Op.EXIT),
    ]
    for i, c in enumerate(code):
        memory[i * 8] = c

    pc, sp, bp, ax, memory, output, traces = executor.run(
        memory,
        torch.tensor(0.0),
        torch.tensor(200.0),
        torch.tensor(200.0),
        torch.tensor(0.0),
        max_steps=20,
        trace=True
    )

    print("Execution trace (first 5 steps):")
    for i, t in enumerate(traces[:5]):
        print(t)
        print()

    print(f"Final output: '{output.strip()}'")
    status = "✓" if output.strip() == "7" else "✗"
    print(f"{status} Thinking tokens computed 3+4=7, output token shows '7'")
    print()

    print("=" * 60)
    print("THINKING/OUTPUT TOKEN SEPARATION COMPLETE")
    print()
    print("Key insight:")
    print("  - Register predictions are 'thinking' (internal computation)")
    print("  - Printf output is 'output' (visible result)")
    print("  - This matches reasoning model architectures")


if __name__ == "__main__":
    test_thinking_tokens()
