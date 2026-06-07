"""Phase R7 — tests for the byte-level Qwen tokenizer wrapper.

These tests intentionally avoid downloading the Qwen tokenizer assets.
A ``StubByteTokenizer`` mimics the HF behaviour we depend on
(``encode(text, add_special_tokens=...)`` / ``decode(ids,
skip_special_tokens=...)``) over a trivial Latin-1 vocabulary, which
is enough to validate that the wrapper:

  * compiles a C program and produces VM token IDs that round-trip
    byte-identically through ``decode`` + ``encode_bpe``;
  * maps every special token to a unique stable HF-decodable tag and
    recovers the original ID through the round-trip.
"""

import os
import sys
import unittest
from typing import List, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neural_vm.qwen_tokenizer_wrapper import (
    C4QwenTokenizerWrapper,
    SPECIAL_TOKEN_TAGS,
    TAG_TO_SPECIAL_ID,
)
from neural_vm.vm_step import Token


class StubByteTokenizer:
    """Tiny HF-shaped tokenizer for offline testing.

    ``encode`` maps each UTF-8 byte of ``text`` to its own integer
    (with a stable offset so special tokens have distinct IDs), and
    ``decode`` is the inverse. Special tokens registered via
    ``additional_special_tokens`` round-trip through dedicated IDs so
    embedded ``<|c4:...|>`` tags survive a tokenize-then-detokenize.
    """

    BYTE_OFFSET = 1000  # leave room for specials at IDs 0..999

    def __init__(self) -> None:
        self._special_to_id: dict[str, int] = {}
        self._id_to_special: dict[int, str] = {}

    def add_special_tokens(self, mapping: dict) -> None:
        for tag in mapping.get("additional_special_tokens", []):
            if tag in self._special_to_id:
                continue
            new_id = len(self._special_to_id)
            self._special_to_id[tag] = new_id
            self._id_to_special[new_id] = tag

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        out: List[int] = []
        i = 0
        n = len(text)
        # Greedy match registered special tokens, otherwise emit
        # UTF-8 bytes shifted by BYTE_OFFSET so byte IDs never overlap
        # special-token IDs.
        while i < n:
            matched = False
            if text.startswith("<|", i):
                end = text.find("|>", i + 2)
                if end != -1:
                    tag = text[i : end + 2]
                    if tag in self._special_to_id:
                        out.append(self._special_to_id[tag])
                        i = end + 2
                        matched = True
            if not matched:
                for b in text[i].encode("utf-8"):
                    out.append(self.BYTE_OFFSET + b)
                i += 1
        return out

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> str:
        parts: List[str] = []
        byte_buf: List[int] = []

        def flush_bytes() -> None:
            if byte_buf:
                parts.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()

        for tid in token_ids:
            if tid in self._id_to_special:
                flush_bytes()
                if not skip_special_tokens:
                    parts.append(self._id_to_special[tid])
            elif tid >= self.BYTE_OFFSET:
                byte_buf.append(tid - self.BYTE_OFFSET)
            else:
                # Unknown low ID — ignore (no-op for the test stub).
                continue
        flush_bytes()
        return "".join(parts)


def _make_wrapper() -> C4QwenTokenizerWrapper:
    stub = StubByteTokenizer()
    wrapper = C4QwenTokenizerWrapper(hf_tokenizer=stub)
    stub.add_special_tokens(
        {"additional_special_tokens": wrapper.additional_special_tokens()}
    )
    return wrapper


C_PROGRAM = "int main(){return 42;}"


class TestC4QwenTokenizerWrapper(unittest.TestCase):
    """Phase R7 acceptance: byte-identical round-trip + stable specials."""

    def test_special_tokens_have_unique_stable_ids(self) -> None:
        """Every special token ID maps to a unique tag and back."""

        # Vocab coverage: every special slot in 256..VOCAB_SIZE-1
        # appears in the tag table.
        expected_ids = set(range(256, Token.VOCAB_SIZE))
        self.assertEqual(set(SPECIAL_TOKEN_TAGS.keys()), expected_ids)

        # Tags themselves are unique.
        tags = list(SPECIAL_TOKEN_TAGS.values())
        self.assertEqual(len(set(tags)), len(tags))

        # Reverse map is consistent.
        for tid, tag in SPECIAL_TOKEN_TAGS.items():
            self.assertEqual(TAG_TO_SPECIAL_ID[tag], tid)

    def test_special_tokens_round_trip_through_hf(self) -> None:
        """``decode([tid])`` then re-encode recovers the same ID."""

        wrapper = _make_wrapper()
        for tid in range(256, Token.VOCAB_SIZE):
            text = wrapper.decode([tid])
            ids = wrapper.encode_bpe(text)
            self.assertEqual(
                ids,
                [tid],
                f"special token id={tid} did not survive round-trip "
                f"(text={text!r}, ids={ids})",
            )

    def test_decode_then_encode_bpe_round_trips_a_program(self) -> None:
        """A compiled C program survives the HF wrapper byte-identically."""

        wrapper = _make_wrapper()
        ids = wrapper.encode_program(C_PROGRAM)

        # Sanity: encode_program respects the VM boundary markers.
        self.assertEqual(ids[0], Token.CODE_START)
        self.assertIn(Token.CODE_END, ids)
        self.assertIn(Token.DATA_START, ids)
        self.assertIn(Token.DATA_END, ids)

        text = wrapper.decode(ids)
        round_tripped = wrapper.encode_bpe(text)
        self.assertEqual(
            round_tripped,
            ids,
            "C program token ids did not round-trip byte-identically",
        )

    def test_decode_emits_only_bytes_and_known_tags(self) -> None:
        """Decoded text is the concatenation of raw bytes + known tags."""

        wrapper = _make_wrapper()
        ids = wrapper.encode_program(C_PROGRAM)
        text = wrapper.decode(ids)

        # Every special tag present in the decoded text must be a
        # registered one (no orphans / no malformed tags).
        i = 0
        while True:
            i = text.find("<|c4:", i)
            if i == -1:
                break
            end = text.find("|>", i)
            self.assertNotEqual(end, -1, "unterminated <|c4: tag")
            tag = text[i : end + 2]
            self.assertIn(tag, TAG_TO_SPECIAL_ID)
            i = end + 2

    def test_decode_byte_only_ids_are_raw_latin1(self) -> None:
        """IDs 0-255 emit the corresponding Latin-1 character."""

        wrapper = _make_wrapper()
        # A handful of arbitrary bytes including 0x00 and 0xFF.
        sample = [0x00, 0x41, 0x42, 0x7F, 0x80, 0xFF]
        text = wrapper.decode(sample)
        self.assertEqual([ord(c) for c in text], sample)

    def test_encode_program_layout_matches_runner(self) -> None:
        """The token shape matches ``AutoregressiveRunner._build_context``.

        Spec: 1 op byte + 4 immediate bytes + 3 padding bytes = 8
        tokens per instruction.
        """

        wrapper = _make_wrapper()
        ids = wrapper.encode_program(C_PROGRAM)
        code_end = ids.index(Token.CODE_END)
        # ids[0] is CODE_START; ids[1:code_end] is the instruction block.
        instr_region = ids[1:code_end]
        self.assertEqual(
            len(instr_region) % 8,
            0,
            "instruction region must be a multiple of 8 tokens (op + 4 imm + 3 pad)",
        )

    def test_encode_bpe_handles_plain_text(self) -> None:
        """A pure-text Qwen prompt becomes a stream of byte IDs."""

        wrapper = _make_wrapper()
        ids = wrapper.encode_bpe("hi")
        self.assertEqual(ids, [ord("h"), ord("i")])

    def test_encode_bpe_ids_path(self) -> None:
        """A caller can hand BPE IDs directly to ``encode_bpe_ids``."""

        wrapper = _make_wrapper()
        bpe_ids = wrapper.hf_tokenizer.encode("ab", add_special_tokens=False)
        ids = wrapper.encode_bpe_ids(bpe_ids)
        self.assertEqual(ids, [ord("a"), ord("b")])

    def test_decode_rejects_out_of_range_ids(self) -> None:
        wrapper = _make_wrapper()
        with self.assertRaises(ValueError):
            wrapper.decode([Token.VOCAB_SIZE])


if __name__ == "__main__":
    unittest.main()
