"""
Tokenizer for C4 Transformer VM

Implements byte-level tokenization with special tokens:
- Vocab 0-255: ASCII/Unicode bytes
- Vocab 256+: Special tokens (think, roles, code, etc.)

This matches how modern LLMs handle text at the byte level.
"""

from enum import IntEnum
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


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
    EXEC = 263             # <|exec|>
    RESULT = 264           # <|result|>

    # Control tokens
    EOS = 265              # End of sequence
    PAD = 266              # Padding
    BOS = 267              # Beginning of sequence

    # Number encoding
    NUM_START = 268        # Start of number
    NUM_END = 269          # End of number


# Token display names
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


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    vocab_size: int = 270
    pad_token_id: int = int(SpecialToken.PAD)
    bos_token_id: int = int(SpecialToken.BOS)
    eos_token_id: int = int(SpecialToken.EOS)


class C4Tokenizer:
    """
    Byte-level tokenizer with special tokens.

    Tokenization is simple: ord() for each character, with
    special multi-character sequences mapped to token IDs 256+.

    This is exactly how byte-level language models work!
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.vocab_size = self.config.vocab_size

        # Sort special tokens by length (longest first) for matching
        self._special_tokens = sorted(
            NAME_TO_TOKEN.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.config.bos_token_id)

        i = 0
        while i < len(text):
            # Check for special tokens (longest match first)
            matched = False
            for name, token in self._special_tokens:
                if text[i:].startswith(name):
                    tokens.append(int(token))
                    i += len(name)
                    matched = True
                    break

            if not matched:
                # Regular byte
                tokens.append(ord(text[i]))
                i += 1

        if add_special_tokens:
            tokens.append(self.config.eos_token_id)

        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: List of token IDs
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded text
        """
        result = []

        for t in tokens:
            if t < 256:
                # Regular byte
                result.append(chr(t))
            elif t in [int(s) for s in SpecialToken]:
                if not skip_special_tokens:
                    result.append(TOKEN_NAMES[SpecialToken(t)])
            else:
                result.append(f"<unk:{t}>")

        return "".join(result)

    def encode_conversation(self, messages: List[Dict[str, str]]) -> List[int]:
        """
        Encode a conversation with role markers.

        Args:
            messages: List of {"role": str, "content": str} dicts

        Returns:
            Token IDs with role markers
        """
        tokens = [self.config.bos_token_id]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                tokens.append(int(SpecialToken.USER))
            elif role == "assistant":
                tokens.append(int(SpecialToken.ASSISTANT))
            elif role == "system":
                tokens.append(int(SpecialToken.SYSTEM))

            tokens.extend(self.encode(content))

        tokens.append(self.config.eos_token_id)
        return tokens

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        vocab = {}

        # Byte tokens
        for i in range(256):
            if 32 <= i < 127:
                vocab[chr(i)] = i
            else:
                vocab[f"<byte:{i}>"] = i

        # Special tokens
        for token, name in TOKEN_NAMES.items():
            vocab[name] = int(token)

        return vocab

    def __call__(self, text: str, **kwargs) -> Dict[str, List[int]]:
        """HuggingFace-style call interface."""
        tokens = self.encode(text, add_special_tokens=kwargs.get('add_special_tokens', False))
        return {
            'input_ids': tokens,
            'attention_mask': [1] * len(tokens),
        }


class NumberTokenizer:
    """
    Efficient number tokenization.

    Instead of encoding "123" as [49, 50, 51] (3 tokens),
    can encode as [NUM_START, 123_encoded, NUM_END].
    """

    def __init__(self, base_tokenizer: C4Tokenizer):
        self.base = base_tokenizer

    def encode_number(self, n: int) -> List[int]:
        """Encode number efficiently."""
        tokens = [int(SpecialToken.NUM_START)]

        # Encode as bytes (little-endian)
        if n == 0:
            tokens.append(0)
        else:
            while n > 0:
                tokens.append(n & 0xFF)
                n >>= 8

        tokens.append(int(SpecialToken.NUM_END))
        return tokens

    def decode_number(self, tokens: List[int]) -> Optional[int]:
        """Decode number from tokens."""
        if not tokens or tokens[0] != int(SpecialToken.NUM_START):
            return None

        result = 0
        shift = 0

        for t in tokens[1:]:
            if t == int(SpecialToken.NUM_END):
                break
            result |= (t << shift)
            shift += 8

        return result


__all__ = [
    'C4Tokenizer',
    'NumberTokenizer',
    'SpecialToken',
    'TokenizerConfig',
    'TOKEN_NAMES',
]
