#!/usr/bin/env python3
"""
Special Tokens for C4 Transformer VM

Implements LLM-style special tokens:
- <think> / </think> - reasoning mode
- <|user|> / <|assistant|> - role switching
- <|code|> / </code> - code blocks
- <|exec|> - execute code
- <|result|> - execution result

Vocab layout (256 bytes + special tokens):
  0-255: ASCII bytes
  256+:  Special tokens
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Optional
from c4_byte_to_nibble import C4ByteNibbleVM
from c4_compiler_full import compile_c

# =============================================================================
# TOKEN DEFINITIONS
# =============================================================================

class SpecialToken(IntEnum):
    """Special tokens beyond ASCII (256+)."""
    # Thinking/reasoning
    THINK_START = 256      # <think>
    THINK_END = 257        # </think>

    # Role markers
    USER = 258             # <|user|>
    ASSISTANT = 259        # <|assistant|>
    SYSTEM = 260           # <|system|>

    # Code handling
    CODE_START = 261       # <|code|>
    CODE_END = 262         # </code>
    EXEC = 263             # <|exec|> - execute the code
    RESULT = 264           # <|result|> - execution result follows

    # Control
    EOS = 265              # End of sequence
    PAD = 266              # Padding
    BOS = 267              # Beginning of sequence

    # Numbers (for efficient number encoding)
    NUM_START = 268        # Start of number literal
    NUM_END = 269          # End of number literal


# Token name mapping
TOKEN_NAMES = {
    SpecialToken.THINK_START: "<think>",
    SpecialToken.THINK_END: "</think>",
    SpecialToken.USER: "<|user|>",
    SpecialToken.ASSISTANT: "<|assistant|>",
    SpecialToken.SYSTEM: "<|system|>",
    SpecialToken.CODE_START: "<|code|>",
    SpecialToken.CODE_END: "</code>",
    SpecialToken.EXEC: "<|exec|>",
    SpecialToken.RESULT: "<|result|>",
    SpecialToken.EOS: "<|eos|>",
    SpecialToken.PAD: "<|pad|>",
    SpecialToken.BOS: "<|bos|>",
    SpecialToken.NUM_START: "<num>",
    SpecialToken.NUM_END: "</num>",
}

# Reverse mapping
NAME_TO_TOKEN = {v: k for k, v in TOKEN_NAMES.items()}


# =============================================================================
# TOKENIZER
# =============================================================================

class C4Tokenizer:
    """
    Tokenizer for C4 VM with special tokens.

    Vocab:
      0-255: ASCII bytes (standard text)
      256+:  Special tokens (think, roles, code, etc.)
    """

    def __init__(self):
        self.vocab_size = 270  # 256 ASCII + 14 special tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text to tokens.

        Handles:
        - Regular ASCII characters
        - Special token markers like <think>, <|user|>, etc.
        """
        tokens = []
        i = 0

        while i < len(text):
            # Check for special tokens
            matched = False
            for name, token in NAME_TO_TOKEN.items():
                if text[i:].startswith(name):
                    tokens.append(int(token))
                    i += len(name)
                    matched = True
                    break

            if not matched:
                # Regular ASCII byte
                tokens.append(ord(text[i]))
                i += 1

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back to text."""
        result = []
        for t in tokens:
            if t < 256:
                result.append(chr(t))
            elif t in TOKEN_NAMES:
                result.append(TOKEN_NAMES[SpecialToken(t)])
            else:
                result.append(f"<unk:{t}>")
        return "".join(result)

    def encode_with_roles(self, messages: List[Tuple[str, str]]) -> List[int]:
        """
        Encode a conversation with role markers.

        Args:
            messages: List of (role, content) tuples
                      role is "user", "assistant", or "system"
        """
        tokens = [int(SpecialToken.BOS)]

        for role, content in messages:
            if role == "user":
                tokens.append(int(SpecialToken.USER))
            elif role == "assistant":
                tokens.append(int(SpecialToken.ASSISTANT))
            elif role == "system":
                tokens.append(int(SpecialToken.SYSTEM))

            tokens.extend(self.encode(content))

        tokens.append(int(SpecialToken.EOS))
        return tokens


# =============================================================================
# EXECUTION ENGINE WITH THINKING
# =============================================================================

class ThinkingVM:
    """
    VM that supports thinking tokens and code execution.

    Can process sequences like:
    <|user|> What is 3+4*2? <|assistant|> <think> Need to handle precedence </think> The answer is 11
    """

    def __init__(self):
        self.vm = C4ByteNibbleVM()
        self.tokenizer = C4Tokenizer()
        self.thinking_log = []

    def process(self, tokens: List[int]) -> Tuple[List[int], Optional[int]]:
        """
        Process a token sequence.

        Returns:
            (output_tokens, execution_result)
        """
        output = []
        code_buffer = []
        in_code = False
        in_think = False
        result = None

        i = 0
        while i < len(tokens):
            t = tokens[i]

            if t == SpecialToken.THINK_START:
                in_think = True
                output.append(t)

            elif t == SpecialToken.THINK_END:
                in_think = False
                output.append(t)

            elif t == SpecialToken.CODE_START:
                in_code = True
                code_buffer = []
                output.append(t)

            elif t == SpecialToken.CODE_END:
                in_code = False
                output.append(t)

            elif t == SpecialToken.EXEC:
                # Execute the code buffer
                output.append(t)
                if code_buffer:
                    code_str = "".join(chr(c) for c in code_buffer if c < 256)
                    try:
                        bytecode, data = compile_c(code_str)
                        self.vm.reset()
                        self.vm.load_bytecode(bytecode, data)
                        result = self.vm.run(max_steps=100000)
                        output.append(int(SpecialToken.RESULT))
                        # Encode result as tokens
                        result_str = str(result)
                        output.extend(ord(c) for c in result_str)
                    except Exception as e:
                        output.extend(ord(c) for c in f"Error: {e}")

            elif in_code:
                code_buffer.append(t)
                output.append(t)

            elif in_think:
                self.thinking_log.append(t)
                output.append(t)

            else:
                output.append(t)

            i += 1

        return output, result


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 70)
    print("  C4 TRANSFORMER VM - SPECIAL TOKENS")
    print("=" * 70)
    print()

    tokenizer = C4Tokenizer()

    # Show vocab
    print("Vocabulary:")
    print("  0-255:   ASCII bytes")
    print("  256-269: Special tokens")
    print()

    print("Special tokens:")
    for token, name in TOKEN_NAMES.items():
        print(f"  {int(token):3d}: {name}")
    print()

    # Test encoding
    print("Encoding examples:")
    print("-" * 50)

    examples = [
        "Hello",
        "<think>reasoning</think>",
        "<|user|>What is 2+2?<|assistant|>4",
        "<|code|>int main() { return 42; }</code><|exec|>",
    ]

    for text in examples:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"  '{text}'")
        print(f"  → {tokens}")
        print(f"  → '{decoded}'")
        print()

    # Test conversation encoding
    print("Conversation encoding:")
    print("-" * 50)

    messages = [
        ("system", "You are a calculator."),
        ("user", "What is 6*7?"),
        ("assistant", "<think>multiply</think>42"),
    ]

    tokens = tokenizer.encode_with_roles(messages)
    print(f"  Messages: {messages}")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: '{tokenizer.decode(tokens)}'")
    print()

    # Test execution with thinking
    print("Execution with thinking:")
    print("-" * 50)

    vm = ThinkingVM()

    # Create a sequence: user asks, assistant thinks and executes code
    sequence = tokenizer.encode(
        "<|user|>Calculate 3+4*2"
        "<|assistant|><think>precedence: multiply first</think>"
        "<|code|>int main() { return 3 + 4 * 2; }</code><|exec|>"
    )

    print(f"  Input tokens: {len(sequence)} tokens")
    output, result = vm.process(sequence)
    print(f"  Output: '{tokenizer.decode(output)}'")
    print(f"  Execution result: {result}")
    print()

    # More complex example
    print("Complex example with Fibonacci:")
    print("-" * 50)

    sequence = tokenizer.encode(
        "<|user|>What is fib(10)?"
        "<|assistant|><think>recursive fibonacci</think>"
        "<|code|>int fib(int n) { if (n < 2) return n; return fib(n-1) + fib(n-2); } "
        "int main() { return fib(10); }</code><|exec|>"
    )

    output, result = vm.process(sequence)
    print(f"  Result: {result}")
    print(f"  Thinking: {vm.tokenizer.decode(vm.thinking_log)}")
    print()

    print("=" * 70)
    print("  TOKEN ARCHITECTURE")
    print("=" * 70)
    print("""
Vocab size: 270 tokens
  Bytes 0-255:  ASCII characters
  Token 256:    <think>    - Start reasoning
  Token 257:    </think>   - End reasoning
  Token 258:    <|user|>   - User turn
  Token 259:    <|assistant|> - Assistant turn
  Token 261:    <|code|>   - Code block start
  Token 263:    <|exec|>   - Execute code
  Token 264:    <|result|> - Result follows

The transformer processes this token stream:
  1. See <|code|> → buffer code tokens
  2. See <|exec|> → compile & run on transformer VM
  3. Emit <|result|> followed by result tokens

This matches how LLMs handle special tokens!
""")


if __name__ == "__main__":
    demo()
