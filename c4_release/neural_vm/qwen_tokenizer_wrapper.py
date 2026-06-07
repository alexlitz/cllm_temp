"""Phase R7 — Tokenizer Option B byte-level wrapper for Qwen export.

The neural VM uses byte-level tokens (see ``Token`` in ``vm_step.py``):
the byte values 0-255 are token IDs directly, and special tokens
(``CODE_START``, ``REG_PC``, ``STEP_END``, ``HALT`` etc.) sit at IDs
256..275.

Qwen's stock tokenizer is BPE over UTF-8 text, so a model exported to
the Qwen layout cannot be driven by Qwen text prompts unmodified. This
module supplies the translation layer:

  * ``encode_program(c_source)`` compiles a C4 program to bytecode and
    returns the exact token ID sequence the VM consumes
    (``[CODE_START, op0, imm0_b0..b3, pad0..2, ..., CODE_END,
    DATA_START, ..., DATA_END]``).
  * ``encode_bpe(text)`` Qwen-tokenizes ``text`` with the wrapped HF
    tokenizer, decodes the resulting BPE IDs back to a UTF-8 string,
    then re-encodes that string byte-by-byte into our VM token IDs.
    This is the path that turns an arbitrary Qwen prompt into a byte
    stream the VM understands.
  * ``decode(ids)`` is the inverse: each VM byte ID 0-255 emits the
    corresponding raw byte, and each special-token ID emits a stable
    tag string (e.g. ``"<|c4:STEP_END|>"``). The result decodes
    cleanly through the HF tokenizer.

By construction, ``decode(encode_program(src))`` round-trips byte-by-
byte, and each special token has a unique stable string form whose own
round-trip recovers the original ID.

The class does NOT require the Qwen HF tokenizer to actually be on
disk. Callers pass in a tokenizer-like object; tests use the
``StubByteTokenizer`` defined in
``tests/test_qwen_tokenizer.py``.

Plan reference: ``docs/QWEN_STRUCTURAL_ADAPTER_PLAN_2026_06_07.md``
Phase R7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from .vm_step import Token


# Map of every Token special token (ID >= 256) to a stable, BPE-safe
# string tag. The ``<|c4:NAME|>`` form mirrors Qwen's own special-token
# convention (``<|im_start|>``, ``<|endoftext|>`` ...), which means we
# can register these as ``additional_special_tokens`` on a real Qwen
# tokenizer without colliding with any text byte.
SPECIAL_TOKEN_TAGS: Dict[int, str] = {
    Token.SEP: "<|c4:SEP|>",
    Token.REG_PC: "<|c4:REG_PC|>",
    Token.REG_AX: "<|c4:REG_AX|>",
    Token.REG_SP: "<|c4:REG_SP|>",
    Token.REG_BP: "<|c4:REG_BP|>",
    Token.MEM: "<|c4:MEM|>",
    Token.STEP_END: "<|c4:STEP_END|>",
    Token.HALT: "<|c4:HALT|>",
    Token.CODE_START: "<|c4:CODE_START|>",
    Token.CODE_END: "<|c4:CODE_END|>",
    Token.DATA_START: "<|c4:DATA_START|>",
    Token.DATA_END: "<|c4:DATA_END|>",
    Token.STACK0: "<|c4:STACK0|>",
    Token.USER_INPUT_START: "<|c4:USER_INPUT_START|>",
    Token.USER_INPUT_END: "<|c4:USER_INPUT_END|>",
    Token.TOOL_CALL: "<|c4:TOOL_CALL|>",
    Token.THINKING_START: "<|c4:THINKING_START|>",
    Token.THINKING_END: "<|c4:THINKING_END|>",
    Token.IO_STATE_EMIT_BYTE: "<|c4:IO_STATE_EMIT_BYTE|>",
    Token.IO_STATE_EMIT_THINKING: "<|c4:IO_STATE_EMIT_THINKING|>",
}


def _validate_special_token_tags() -> None:
    """Sanity-check the SPECIAL_TOKEN_TAGS table.

    Every special token ID (256..VOCAB_SIZE-1) must have a tag, and the
    tags must be unique strings disjoint from the raw byte alphabet
    (0-255 reproduces as a literal byte/char, never as a tag).
    """

    expected_ids = set(range(256, Token.VOCAB_SIZE))
    actual_ids = set(SPECIAL_TOKEN_TAGS.keys())
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    if missing or extra:
        raise RuntimeError(
            "SPECIAL_TOKEN_TAGS out of sync with Token vocab "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )

    tags = list(SPECIAL_TOKEN_TAGS.values())
    if len(set(tags)) != len(tags):
        raise RuntimeError(
            "SPECIAL_TOKEN_TAGS contains duplicate tags; every special "
            "token must round-trip to a unique string."
        )


_validate_special_token_tags()


# Reverse lookup: tag -> ID. Built once at import time after the
# uniqueness check above.
TAG_TO_SPECIAL_ID: Dict[str, int] = {tag: tid for tid, tag in SPECIAL_TOKEN_TAGS.items()}


class HFTokenizerLike(Protocol):
    """Minimal duck-typed interface we need from a Qwen HF tokenizer."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> List[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = ...,
    ) -> str: ...


@dataclass
class C4QwenTokenizerWrapper:
    """Translation layer between a Qwen BPE tokenizer and the VM byte vocab.

    Parameters
    ----------
    hf_tokenizer:
        Any object exposing ``encode(text, add_special_tokens=...)`` and
        ``decode(ids, skip_special_tokens=...)`` with HF semantics. The
        wrapper never inspects the BPE vocabulary directly.
    code_section_terminators:
        Optional override; defaults to the standard
        ``[CODE_END, DATA_START, DATA_END]`` triple emitted by
        ``run_vm.AutoregressiveRunner._build_context`` when there is no
        data segment, stdin, or argv.
    """

    hf_tokenizer: HFTokenizerLike
    code_section_terminators: Tuple[int, ...] = field(
        default_factory=lambda: (
            Token.CODE_END,
            Token.DATA_START,
            Token.DATA_END,
        )
    )

    # -- public API -----------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """VM-side vocabulary size; matches ``Token.VOCAB_SIZE``."""

        return Token.VOCAB_SIZE

    def encode_program(
        self,
        c_source: str,
        *,
        data: Optional[Sequence[int]] = None,
        stdin: str = "",
        argv: Optional[Sequence[str]] = None,
        link_stdlib: bool = True,
        immediate_size: int = 4,
        padding_size: int = 3,
    ) -> List[int]:
        """Compile ``c_source`` and return the VM context token IDs.

        The token layout mirrors
        ``AutoregressiveRunner._build_context``:

            [CODE_START] (op, imm0, imm1, imm2, imm3, pad, pad, pad)*
            [CODE_END] [DATA_START] data_bytes [DATA_END]
            ([USER_INPUT_START] stdin_bytes [USER_INPUT_END])?
            (argv chars + null terminator)?
        """

        # Lazy import keeps this module importable without a full src
        # tree (the tests don't need the C compiler).
        from src.compiler import compile_c

        bytecode, compiled_data = compile_c(c_source, link_stdlib=link_stdlib)
        data_bytes = list(compiled_data) if data is None else list(data)

        tokens: List[int] = [Token.CODE_START]
        for instr in bytecode:
            op = instr & 0xFF
            imm = instr >> 8
            tokens.append(op)
            for i in range(immediate_size):
                tokens.append((imm >> (i * 8)) & 0xFF)
            for _ in range(padding_size):
                tokens.append(0)

        tokens.append(Token.CODE_END)
        tokens.append(Token.DATA_START)
        tokens.extend(int(b) & 0xFF for b in data_bytes)
        tokens.append(Token.DATA_END)

        if stdin:
            tokens.append(Token.USER_INPUT_START)
            tokens.extend(ord(c) & 0xFF for c in stdin)
            tokens.append(Token.USER_INPUT_END)

        if argv:
            for arg in argv:
                tokens.extend(ord(c) & 0xFF for c in arg)
                tokens.append(0)

        return tokens

    def encode_bpe(self, text: str) -> List[int]:
        """Translate a Qwen-text prompt into VM byte token IDs.

        We round-trip ``text`` through the HF tokenizer to materialize
        the exact byte sequence Qwen will see, then re-encode it byte
        by byte. Embedded special tags (``<|c4:STEP_END|>`` etc.) are
        recovered as special token IDs so a prompt can include VM
        boundaries verbatim.
        """

        bpe_ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
        decoded = self.hf_tokenizer.decode(bpe_ids, skip_special_tokens=False)
        return self._encode_bytes_with_tags(decoded)

    def encode_bpe_ids(self, bpe_ids: Sequence[int]) -> List[int]:
        """Translate raw Qwen BPE IDs into VM byte token IDs.

        Convenience wrapper for callers who already have a tokenized
        prompt in hand.
        """

        decoded = self.hf_tokenizer.decode(list(bpe_ids), skip_special_tokens=False)
        return self._encode_bytes_with_tags(decoded)

    def decode(self, ids: Iterable[int]) -> str:
        """Render VM token IDs as a Qwen-decodable string.

        Byte IDs (0-255) emit a single Latin-1 codepoint so the string
        is reversible to bytes via ``s.encode('latin-1')``. Special
        token IDs emit their stable ``<|c4:NAME|>`` tag.
        """

        out: List[str] = []
        for raw in ids:
            tok = int(raw)
            if 0 <= tok < 256:
                out.append(chr(tok))
            elif tok in SPECIAL_TOKEN_TAGS:
                out.append(SPECIAL_TOKEN_TAGS[tok])
            else:
                raise ValueError(
                    f"token id {tok} is outside the VM vocabulary "
                    f"(0..{Token.VOCAB_SIZE - 1})"
                )
        return "".join(out)

    def additional_special_tokens(self) -> List[str]:
        """Tags to register with the HF tokenizer as new specials.

        Callers can pass this list to
        ``tokenizer.add_special_tokens({'additional_special_tokens': ...})``
        on a real Qwen tokenizer so that prompts containing C4
        boundaries survive a BPE round-trip unmangled.
        """

        return list(SPECIAL_TOKEN_TAGS.values())

    # -- internals ------------------------------------------------------

    def _encode_bytes_with_tags(self, text: str) -> List[int]:
        """Walk ``text`` and emit VM IDs, recognizing ``<|c4:NAME|>`` tags.

        Anything that isn't a known tag falls back to its UTF-8 byte
        encoding, with each byte mapped to its own ID.
        """

        out: List[int] = []
        i = 0
        n = len(text)
        prefix = "<|c4:"
        while i < n:
            if text.startswith(prefix, i):
                end = text.find("|>", i + len(prefix))
                if end != -1:
                    tag = text[i : end + 2]
                    if tag in TAG_TO_SPECIAL_ID:
                        out.append(TAG_TO_SPECIAL_ID[tag])
                        i = end + 2
                        continue
            # Not a recognized tag — emit one character's UTF-8 bytes.
            for b in text[i].encode("utf-8"):
                out.append(b)
            i += 1
        return out


def make_wrapper(hf_tokenizer: HFTokenizerLike) -> C4QwenTokenizerWrapper:
    """Convenience constructor; mirrors HF's ``from_pretrained`` ergonomics."""

    return C4QwenTokenizerWrapper(hf_tokenizer=hf_tokenizer)


__all__ = [
    "C4QwenTokenizerWrapper",
    "SPECIAL_TOKEN_TAGS",
    "TAG_TO_SPECIAL_ID",
    "HFTokenizerLike",
    "make_wrapper",
]
