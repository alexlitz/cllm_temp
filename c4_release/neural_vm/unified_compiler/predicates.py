"""
Predicate DSL for declarative-IR semantic contracts.

Build-time-only: parse predicate strings, normalize to DNF, decide entailment
via structural matching (no SMT solver).

Public API:
    parse(text: str) -> Predicate
    dnf(p: Predicate) -> list[frozenset[Atom]]
    entails(p: Predicate, q: Predicate) -> bool
    explain_failure(p: Predicate, q: Predicate) -> Optional[str]

Grammar (precedence: NOT > AND > OR, parens explicit):

    predicate := or_expr
    or_expr   := and_expr ('OR' and_expr)*
    and_expr  := not_expr ('AND' not_expr)*
    not_expr  := 'NOT' not_expr | primary
    primary   := '(' predicate ')' | atom

Atoms (closed set):

    # Marker atoms
    mark == SP | AX | PC | BP | MEM | STACK0 | SE | NONE
    mark in {SP, BP, ...}

    # Boolean atoms (no operand)
    is_byte, has_se, step_is_fresh, in_step_fresh,
    addr_b0_valid, addr_b1_valid, addr_b2_valid,
    sp_gathered_this_step

    # Step-index atoms
    step_index == <int>
    step_index in [<int>, <int>)
    step_index in {<int>, ...}

    # Opcode-context atoms
    opcode_at_AX == <NAME>
    opcode_at_AX in {<NAME>, ...}
    opcode_in_step in {<NAME>, ...}

    # Byte-index atoms
    byte_index == <int>
    byte_index in {<int>, ...}

    # Value-witness atoms
    byte_value == 0x<hex>
    byte_value in {0x<hex>, ...}
    byte_value.lo_nibble == 0x<hex>
    byte_value.hi_nibble == 0x<hex>
    sp_byte0 == 0x<hex>

    # Output self-reference atoms
    output_lo_nibble == 0x<hex>
    output_hi_nibble == 0x<hex>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Union


# ---------------------------------------------------------------------------
# Atom families
# ---------------------------------------------------------------------------

# Valid marker role names.
_MARK_ROLES = frozenset({"SP", "AX", "PC", "BP", "MEM", "STACK0", "SE", "NONE"})

# Valid boolean atom names.
_BOOL_ATOMS = frozenset({
    "is_byte",
    "has_se",
    "step_is_fresh",
    "in_step_fresh",
    "addr_b0_valid",
    "addr_b1_valid",
    "addr_b2_valid",
    "sp_gathered_this_step",
})


@dataclass(frozen=True)
class Atom:
    """Base for atoms; subclasses are leaf predicates."""

    def __str__(self) -> str:  # pragma: no cover - subclasses override
        raise NotImplementedError


# --- Marker atoms ----------------------------------------------------------


@dataclass(frozen=True)
class MarkEq(Atom):
    role: str

    def __str__(self) -> str:
        return f"mark == {self.role}"


@dataclass(frozen=True)
class MarkIn(Atom):
    roles: frozenset[str]

    def __str__(self) -> str:
        body = ", ".join(sorted(self.roles))
        return f"mark in {{{body}}}"


@dataclass(frozen=True)
class MarkNotEq(Atom):
    """Negation of MarkEq."""

    role: str

    def __str__(self) -> str:
        return f"NOT mark == {self.role}"


@dataclass(frozen=True)
class MarkNotIn(Atom):
    """Negation of MarkIn."""

    roles: frozenset[str]

    def __str__(self) -> str:
        body = ", ".join(sorted(self.roles))
        return f"NOT mark in {{{body}}}"


# --- Boolean atoms ---------------------------------------------------------


@dataclass(frozen=True)
class BoolAtom(Atom):
    name: str
    negated: bool = False

    def __str__(self) -> str:
        return f"NOT {self.name}" if self.negated else self.name


# --- Step-index atoms ------------------------------------------------------


@dataclass(frozen=True)
class StepIndexEq(Atom):
    n: int

    def __str__(self) -> str:
        return f"step_index == {self.n}"


@dataclass(frozen=True)
class StepIndexInRange(Atom):
    lo: int  # inclusive
    hi: int  # exclusive

    def __str__(self) -> str:
        return f"step_index in [{self.lo}, {self.hi})"


@dataclass(frozen=True)
class StepIndexInSet(Atom):
    values: frozenset[int]

    def __str__(self) -> str:
        body = ", ".join(str(v) for v in sorted(self.values))
        return f"step_index in {{{body}}}"


@dataclass(frozen=True)
class StepIndexNotEq(Atom):
    n: int

    def __str__(self) -> str:
        return f"NOT step_index == {self.n}"


@dataclass(frozen=True)
class StepIndexNotInRange(Atom):
    lo: int
    hi: int

    def __str__(self) -> str:
        return f"NOT step_index in [{self.lo}, {self.hi})"


@dataclass(frozen=True)
class StepIndexNotInSet(Atom):
    values: frozenset[int]

    def __str__(self) -> str:
        body = ", ".join(str(v) for v in sorted(self.values))
        return f"NOT step_index in {{{body}}}"


# --- Opcode atoms ----------------------------------------------------------


@dataclass(frozen=True)
class OpcodeAtAxEq(Atom):
    opcode: str

    def __str__(self) -> str:
        return f"opcode_at_AX == {self.opcode}"


@dataclass(frozen=True)
class OpcodeAtAxIn(Atom):
    opcodes: frozenset[str]

    def __str__(self) -> str:
        body = ", ".join(sorted(self.opcodes))
        return f"opcode_at_AX in {{{body}}}"


@dataclass(frozen=True)
class OpcodeAtAxNotEq(Atom):
    opcode: str

    def __str__(self) -> str:
        return f"NOT opcode_at_AX == {self.opcode}"


@dataclass(frozen=True)
class OpcodeAtAxNotIn(Atom):
    opcodes: frozenset[str]

    def __str__(self) -> str:
        body = ", ".join(sorted(self.opcodes))
        return f"NOT opcode_at_AX in {{{body}}}"


@dataclass(frozen=True)
class OpcodeInStepIn(Atom):
    opcodes: frozenset[str]

    def __str__(self) -> str:
        body = ", ".join(sorted(self.opcodes))
        return f"opcode_in_step in {{{body}}}"


@dataclass(frozen=True)
class OpcodeInStepNotIn(Atom):
    opcodes: frozenset[str]

    def __str__(self) -> str:
        body = ", ".join(sorted(self.opcodes))
        return f"NOT opcode_in_step in {{{body}}}"


# --- Byte index atoms ------------------------------------------------------


@dataclass(frozen=True)
class ByteIndexEq(Atom):
    n: int

    def __str__(self) -> str:
        return f"byte_index == {self.n}"


@dataclass(frozen=True)
class ByteIndexIn(Atom):
    values: frozenset[int]

    def __str__(self) -> str:
        body = ", ".join(str(v) for v in sorted(self.values))
        return f"byte_index in {{{body}}}"


@dataclass(frozen=True)
class ByteIndexNotEq(Atom):
    n: int

    def __str__(self) -> str:
        return f"NOT byte_index == {self.n}"


@dataclass(frozen=True)
class ByteIndexNotIn(Atom):
    values: frozenset[int]

    def __str__(self) -> str:
        body = ", ".join(str(v) for v in sorted(self.values))
        return f"NOT byte_index in {{{body}}}"


# --- Byte value atoms ------------------------------------------------------


@dataclass(frozen=True)
class ByteValueEq(Atom):
    value: int

    def __str__(self) -> str:
        return f"byte_value == 0x{self.value:02X}"


@dataclass(frozen=True)
class ByteValueIn(Atom):
    values: frozenset[int]

    def __str__(self) -> str:
        body = ", ".join(f"0x{v:02X}" for v in sorted(self.values))
        return f"byte_value in {{{body}}}"


@dataclass(frozen=True)
class ByteValueLoNibbleEq(Atom):
    nibble: int

    def __str__(self) -> str:
        return f"byte_value.lo_nibble == 0x{self.nibble:X}"


@dataclass(frozen=True)
class ByteValueHiNibbleEq(Atom):
    nibble: int

    def __str__(self) -> str:
        return f"byte_value.hi_nibble == 0x{self.nibble:X}"


@dataclass(frozen=True)
class ByteValueNotEq(Atom):
    value: int

    def __str__(self) -> str:
        return f"NOT byte_value == 0x{self.value:02X}"


@dataclass(frozen=True)
class ByteValueNotIn(Atom):
    values: frozenset[int]

    def __str__(self) -> str:
        body = ", ".join(f"0x{v:02X}" for v in sorted(self.values))
        return f"NOT byte_value in {{{body}}}"


@dataclass(frozen=True)
class ByteValueLoNibbleNotEq(Atom):
    nibble: int

    def __str__(self) -> str:
        return f"NOT byte_value.lo_nibble == 0x{self.nibble:X}"


@dataclass(frozen=True)
class ByteValueHiNibbleNotEq(Atom):
    nibble: int

    def __str__(self) -> str:
        return f"NOT byte_value.hi_nibble == 0x{self.nibble:X}"


@dataclass(frozen=True)
class SpByte0Eq(Atom):
    value: int

    def __str__(self) -> str:
        return f"sp_byte0 == 0x{self.value:02X}"


@dataclass(frozen=True)
class SpByte0NotEq(Atom):
    value: int

    def __str__(self) -> str:
        return f"NOT sp_byte0 == 0x{self.value:02X}"


# --- Output self-reference atoms ------------------------------------------


@dataclass(frozen=True)
class OutputLoNibbleEq(Atom):
    nibble: int

    def __str__(self) -> str:
        return f"output_lo_nibble == 0x{self.nibble:X}"


@dataclass(frozen=True)
class OutputHiNibbleEq(Atom):
    nibble: int

    def __str__(self) -> str:
        return f"output_hi_nibble == 0x{self.nibble:X}"


@dataclass(frozen=True)
class OutputLoNibbleNotEq(Atom):
    nibble: int

    def __str__(self) -> str:
        return f"NOT output_lo_nibble == 0x{self.nibble:X}"


@dataclass(frozen=True)
class OutputHiNibbleNotEq(Atom):
    nibble: int

    def __str__(self) -> str:
        return f"NOT output_hi_nibble == 0x{self.nibble:X}"


# ---------------------------------------------------------------------------
# Connective AST nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class And:
    children: tuple["Predicate", ...]

    def __str__(self) -> str:
        return "(" + " AND ".join(str(c) for c in self.children) + ")"


@dataclass(frozen=True)
class Or:
    children: tuple["Predicate", ...]

    def __str__(self) -> str:
        return "(" + " OR ".join(str(c) for c in self.children) + ")"


@dataclass(frozen=True)
class Not:
    child: "Predicate"

    def __str__(self) -> str:
        return f"NOT {self.child}"


Predicate = Union[Atom, And, Or, Not]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class _Tokenizer:
    """Tokenizes the predicate DSL. Tokens are tuples (kind, value, pos)."""

    SPECIALS = {"(", ")", "{", "}", "[", "]", ",", "."}

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.tokens: list[tuple[str, str, int]] = []
        self._tokenize()
        self.idx = 0

    def _tokenize(self) -> None:
        text = self.text
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if c.isspace():
                i += 1
                continue
            if c in self.SPECIALS:
                self.tokens.append((c, c, i))
                i += 1
                continue
            if c == "=":
                if i + 1 < n and text[i + 1] == "=":
                    self.tokens.append(("==", "==", i))
                    i += 2
                    continue
                raise ValueError(f"unexpected character {c!r} at position {i}")
            # Identifier / number / hex literal (including signs for ints).
            if c == "-" and i + 1 < n and text[i + 1].isdigit():
                # signed integer
                j = i + 2
                while j < n and text[j].isdigit():
                    j += 1
                self.tokens.append(("INT", text[i:j], i))
                i = j
                continue
            if c.isdigit():
                # hex (0x..) or decimal integer
                if c == "0" and i + 1 < n and text[i + 1] in ("x", "X"):
                    j = i + 2
                    start_hex = j
                    while j < n and (text[j].isdigit() or text[j] in "abcdefABCDEF"):
                        j += 1
                    if j == start_hex:
                        raise ValueError(
                            f"invalid hex literal at position {i}: expected hex digits after 0x"
                        )
                    self.tokens.append(("HEX", text[i:j], i))
                    i = j
                    continue
                j = i
                while j < n and text[j].isdigit():
                    j += 1
                self.tokens.append(("INT", text[i:j], i))
                i = j
                continue
            if c.isalpha() or c == "_":
                j = i
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                word = text[i:j]
                kind = "WORD"
                if word in ("AND", "OR", "NOT", "in"):
                    kind = word
                self.tokens.append((kind, word, i))
                i = j
                continue
            raise ValueError(f"unexpected character {c!r} at position {i}")

    def peek(self, offset: int = 0) -> Optional[tuple[str, str, int]]:
        idx = self.idx + offset
        if idx >= len(self.tokens):
            return None
        return self.tokens[idx]

    def consume(self) -> tuple[str, str, int]:
        if self.idx >= len(self.tokens):
            raise ValueError("unexpected end of input")
        tok = self.tokens[self.idx]
        self.idx += 1
        return tok

    def expect(self, kind: str) -> tuple[str, str, int]:
        tok = self.peek()
        if tok is None:
            raise ValueError(f"unexpected end of input, expected {kind!r}")
        if tok[0] != kind:
            raise ValueError(
                f"expected {kind!r} but found {tok[1]!r} at position {tok[2]}"
            )
        return self.consume()

    def at_end(self) -> bool:
        return self.idx >= len(self.tokens)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse_int(tok: tuple[str, str, int]) -> int:
    try:
        return int(tok[1], 10)
    except ValueError as exc:
        raise ValueError(
            f"invalid integer literal {tok[1]!r} at position {tok[2]}"
        ) from exc


def _parse_hex(tok: tuple[str, str, int]) -> int:
    try:
        return int(tok[1], 16)
    except ValueError as exc:
        raise ValueError(
            f"invalid hex literal {tok[1]!r} at position {tok[2]}"
        ) from exc


def _parse_int_set(tz: _Tokenizer) -> frozenset[int]:
    tz.expect("{")
    values: set[int] = set()
    if tz.peek() is not None and tz.peek()[0] == "}":
        tz.consume()
        return frozenset(values)
    while True:
        tok = tz.consume()
        if tok[0] != "INT":
            raise ValueError(
                f"expected integer in set literal but found {tok[1]!r} at position {tok[2]}"
            )
        values.add(_parse_int(tok))
        nxt = tz.peek()
        if nxt is None:
            raise ValueError("unexpected end of input inside set literal")
        if nxt[0] == ",":
            tz.consume()
            continue
        if nxt[0] == "}":
            tz.consume()
            break
        raise ValueError(
            f"expected ',' or '}}' in set literal but found {nxt[1]!r} at position {nxt[2]}"
        )
    return frozenset(values)


def _parse_hex_set(tz: _Tokenizer) -> frozenset[int]:
    tz.expect("{")
    values: set[int] = set()
    if tz.peek() is not None and tz.peek()[0] == "}":
        tz.consume()
        return frozenset(values)
    while True:
        tok = tz.consume()
        if tok[0] != "HEX":
            raise ValueError(
                f"expected hex literal in set but found {tok[1]!r} at position {tok[2]}"
            )
        values.add(_parse_hex(tok))
        nxt = tz.peek()
        if nxt is None:
            raise ValueError("unexpected end of input inside set literal")
        if nxt[0] == ",":
            tz.consume()
            continue
        if nxt[0] == "}":
            tz.consume()
            break
        raise ValueError(
            f"expected ',' or '}}' in set literal but found {nxt[1]!r} at position {nxt[2]}"
        )
    return frozenset(values)


def _parse_word_set(tz: _Tokenizer) -> frozenset[str]:
    tz.expect("{")
    values: set[str] = set()
    if tz.peek() is not None and tz.peek()[0] == "}":
        tz.consume()
        return frozenset(values)
    while True:
        tok = tz.consume()
        if tok[0] != "WORD":
            raise ValueError(
                f"expected name in set literal but found {tok[1]!r} at position {tok[2]}"
            )
        values.add(tok[1])
        nxt = tz.peek()
        if nxt is None:
            raise ValueError("unexpected end of input inside set literal")
        if nxt[0] == ",":
            tz.consume()
            continue
        if nxt[0] == "}":
            tz.consume()
            break
        raise ValueError(
            f"expected ',' or '}}' in set literal but found {nxt[1]!r} at position {nxt[2]}"
        )
    return frozenset(values)


def _parse_atom(tz: _Tokenizer) -> Atom:
    tok = tz.peek()
    if tok is None:
        raise ValueError("unexpected end of input, expected atom")
    if tok[0] != "WORD":
        raise ValueError(
            f"expected atom name but found {tok[1]!r} at position {tok[2]}"
        )
    name = tok[1]
    pos = tok[2]

    if name == "mark":
        tz.consume()
        op = tz.consume()
        if op[0] == "==":
            roletok = tz.consume()
            if roletok[0] != "WORD" or roletok[1] not in _MARK_ROLES:
                raise ValueError(
                    f"unknown mark role: {roletok[1]!r} at position {roletok[2]}"
                )
            return MarkEq(roletok[1])
        if op[0] == "in":
            roles = _parse_word_set(tz)
            for r in roles:
                if r not in _MARK_ROLES:
                    raise ValueError(f"unknown mark role: {r!r} at position {pos}")
            return MarkIn(roles)
        raise ValueError(
            f"expected '==' or 'in' after 'mark' but found {op[1]!r} at position {op[2]}"
        )

    if name == "step_index":
        tz.consume()
        op = tz.consume()
        if op[0] == "==":
            inttok = tz.consume()
            if inttok[0] != "INT":
                raise ValueError(
                    f"expected integer after 'step_index ==' but found {inttok[1]!r} at position {inttok[2]}"
                )
            return StepIndexEq(_parse_int(inttok))
        if op[0] == "in":
            nxt = tz.peek()
            if nxt is None:
                raise ValueError("unexpected end of input after 'step_index in'")
            if nxt[0] == "[":
                tz.consume()
                lo_tok = tz.consume()
                if lo_tok[0] != "INT":
                    raise ValueError(
                        f"expected integer for range lower bound at position {lo_tok[2]}"
                    )
                tz.expect(",")
                hi_tok = tz.consume()
                if hi_tok[0] != "INT":
                    raise ValueError(
                        f"expected integer for range upper bound at position {hi_tok[2]}"
                    )
                tz.expect(")")
                return StepIndexInRange(_parse_int(lo_tok), _parse_int(hi_tok))
            if nxt[0] == "{":
                values = _parse_int_set(tz)
                return StepIndexInSet(values)
            raise ValueError(
                f"expected '[' or '{{' after 'step_index in' but found {nxt[1]!r} at position {nxt[2]}"
            )
        raise ValueError(
            f"expected '==' or 'in' after 'step_index' but found {op[1]!r} at position {op[2]}"
        )

    if name == "opcode_at_AX":
        tz.consume()
        op = tz.consume()
        if op[0] == "==":
            optok = tz.consume()
            if optok[0] != "WORD":
                raise ValueError(
                    f"expected opcode name after 'opcode_at_AX ==' but found {optok[1]!r} at position {optok[2]}"
                )
            return OpcodeAtAxEq(optok[1])
        if op[0] == "in":
            opcodes = _parse_word_set(tz)
            return OpcodeAtAxIn(opcodes)
        raise ValueError(
            f"expected '==' or 'in' after 'opcode_at_AX' but found {op[1]!r} at position {op[2]}"
        )

    if name == "opcode_in_step":
        tz.consume()
        op = tz.consume()
        if op[0] == "in":
            opcodes = _parse_word_set(tz)
            return OpcodeInStepIn(opcodes)
        raise ValueError(
            f"expected 'in' after 'opcode_in_step' but found {op[1]!r} at position {op[2]}"
        )

    if name == "byte_index":
        tz.consume()
        op = tz.consume()
        if op[0] == "==":
            inttok = tz.consume()
            if inttok[0] != "INT":
                raise ValueError(
                    f"expected integer after 'byte_index ==' but found {inttok[1]!r} at position {inttok[2]}"
                )
            return ByteIndexEq(_parse_int(inttok))
        if op[0] == "in":
            values = _parse_int_set(tz)
            return ByteIndexIn(values)
        raise ValueError(
            f"expected '==' or 'in' after 'byte_index' but found {op[1]!r} at position {op[2]}"
        )

    if name == "byte_value":
        tz.consume()
        nxt = tz.peek()
        if nxt is not None and nxt[0] == ".":
            tz.consume()
            field_tok = tz.consume()
            if field_tok[0] != "WORD" or field_tok[1] not in ("lo_nibble", "hi_nibble"):
                raise ValueError(
                    f"unknown byte_value field: {field_tok[1]!r} at position {field_tok[2]}"
                )
            tz.expect("==")
            hextok = tz.consume()
            if hextok[0] != "HEX":
                raise ValueError(
                    f"expected hex literal after 'byte_value.{field_tok[1]} ==' but found {hextok[1]!r} at position {hextok[2]}"
                )
            value = _parse_hex(hextok)
            if not (0 <= value <= 0xF):
                raise ValueError(
                    f"nibble value out of range at position {hextok[2]}: 0x{value:X}"
                )
            if field_tok[1] == "lo_nibble":
                return ByteValueLoNibbleEq(value)
            return ByteValueHiNibbleEq(value)
        op = tz.consume()
        if op[0] == "==":
            hextok = tz.consume()
            if hextok[0] != "HEX":
                raise ValueError(
                    f"expected hex literal after 'byte_value ==' but found {hextok[1]!r} at position {hextok[2]}"
                )
            return ByteValueEq(_parse_hex(hextok))
        if op[0] == "in":
            values = _parse_hex_set(tz)
            return ByteValueIn(values)
        raise ValueError(
            f"expected '==' or 'in' after 'byte_value' but found {op[1]!r} at position {op[2]}"
        )

    if name == "sp_byte0":
        tz.consume()
        tz.expect("==")
        hextok = tz.consume()
        if hextok[0] != "HEX":
            raise ValueError(
                f"expected hex literal after 'sp_byte0 ==' but found {hextok[1]!r} at position {hextok[2]}"
            )
        return SpByte0Eq(_parse_hex(hextok))

    if name in ("output_lo_nibble", "output_hi_nibble"):
        tz.consume()
        tz.expect("==")
        hextok = tz.consume()
        if hextok[0] != "HEX":
            raise ValueError(
                f"expected hex literal after '{name} ==' but found {hextok[1]!r} at position {hextok[2]}"
            )
        value = _parse_hex(hextok)
        if not (0 <= value <= 0xF):
            raise ValueError(
                f"nibble value out of range at position {hextok[2]}: 0x{value:X}"
            )
        if name == "output_lo_nibble":
            return OutputLoNibbleEq(value)
        return OutputHiNibbleEq(value)

    if name in _BOOL_ATOMS:
        tz.consume()
        return BoolAtom(name)

    raise ValueError(f"unknown atom: {name!r} at position {pos}")


def _parse_primary(tz: _Tokenizer) -> Predicate:
    tok = tz.peek()
    if tok is None:
        raise ValueError("unexpected end of input, expected primary")
    if tok[0] == "(":
        tz.consume()
        inner = _parse_or(tz)
        nxt = tz.peek()
        if nxt is None or nxt[0] != ")":
            raise ValueError(
                f"unmatched '(' starting at position {tok[2]}"
            )
        tz.consume()
        return inner
    return _parse_atom(tz)


def _parse_not(tz: _Tokenizer) -> Predicate:
    tok = tz.peek()
    if tok is not None and tok[0] == "NOT":
        tz.consume()
        return Not(_parse_not(tz))
    return _parse_primary(tz)


def _parse_and(tz: _Tokenizer) -> Predicate:
    first = _parse_not(tz)
    items: list[Predicate] = [first]
    while True:
        tok = tz.peek()
        if tok is None or tok[0] != "AND":
            break
        tz.consume()
        items.append(_parse_not(tz))
    if len(items) == 1:
        return items[0]
    return And(tuple(items))


def _parse_or(tz: _Tokenizer) -> Predicate:
    first = _parse_and(tz)
    items: list[Predicate] = [first]
    while True:
        tok = tz.peek()
        if tok is None or tok[0] != "OR":
            break
        tz.consume()
        items.append(_parse_and(tz))
    if len(items) == 1:
        return items[0]
    return Or(tuple(items))


def parse(text: str) -> Predicate:
    """Parse a predicate string into AST. Raises ValueError on bad input."""
    if not text or not text.strip():
        raise ValueError("empty predicate text")
    tz = _Tokenizer(text)
    result = _parse_or(tz)
    if not tz.at_end():
        leftover = tz.peek()
        raise ValueError(
            f"unexpected trailing token {leftover[1]!r} at position {leftover[2]}"
        )
    return result


# ---------------------------------------------------------------------------
# Negation pushdown (De Morgan)
# ---------------------------------------------------------------------------


def _negate_atom(atom: Atom) -> Atom:
    """Return an atom representing the logical negation of `atom`.

    Each atom family has a natural negation atom; using these lets us push
    NOT all the way to the leaves so that DNF has no Not nodes.
    """
    if isinstance(atom, MarkEq):
        return MarkNotEq(atom.role)
    if isinstance(atom, MarkNotEq):
        return MarkEq(atom.role)
    if isinstance(atom, MarkIn):
        return MarkNotIn(atom.roles)
    if isinstance(atom, MarkNotIn):
        return MarkIn(atom.roles)
    if isinstance(atom, BoolAtom):
        return BoolAtom(atom.name, negated=not atom.negated)
    if isinstance(atom, StepIndexEq):
        return StepIndexNotEq(atom.n)
    if isinstance(atom, StepIndexNotEq):
        return StepIndexEq(atom.n)
    if isinstance(atom, StepIndexInRange):
        return StepIndexNotInRange(atom.lo, atom.hi)
    if isinstance(atom, StepIndexNotInRange):
        return StepIndexInRange(atom.lo, atom.hi)
    if isinstance(atom, StepIndexInSet):
        return StepIndexNotInSet(atom.values)
    if isinstance(atom, StepIndexNotInSet):
        return StepIndexInSet(atom.values)
    if isinstance(atom, OpcodeAtAxEq):
        return OpcodeAtAxNotEq(atom.opcode)
    if isinstance(atom, OpcodeAtAxNotEq):
        return OpcodeAtAxEq(atom.opcode)
    if isinstance(atom, OpcodeAtAxIn):
        return OpcodeAtAxNotIn(atom.opcodes)
    if isinstance(atom, OpcodeAtAxNotIn):
        return OpcodeAtAxIn(atom.opcodes)
    if isinstance(atom, OpcodeInStepIn):
        return OpcodeInStepNotIn(atom.opcodes)
    if isinstance(atom, OpcodeInStepNotIn):
        return OpcodeInStepIn(atom.opcodes)
    if isinstance(atom, ByteIndexEq):
        return ByteIndexNotEq(atom.n)
    if isinstance(atom, ByteIndexNotEq):
        return ByteIndexEq(atom.n)
    if isinstance(atom, ByteIndexIn):
        return ByteIndexNotIn(atom.values)
    if isinstance(atom, ByteIndexNotIn):
        return ByteIndexIn(atom.values)
    if isinstance(atom, ByteValueEq):
        return ByteValueNotEq(atom.value)
    if isinstance(atom, ByteValueNotEq):
        return ByteValueEq(atom.value)
    if isinstance(atom, ByteValueIn):
        return ByteValueNotIn(atom.values)
    if isinstance(atom, ByteValueNotIn):
        return ByteValueIn(atom.values)
    if isinstance(atom, ByteValueLoNibbleEq):
        return ByteValueLoNibbleNotEq(atom.nibble)
    if isinstance(atom, ByteValueLoNibbleNotEq):
        return ByteValueLoNibbleEq(atom.nibble)
    if isinstance(atom, ByteValueHiNibbleEq):
        return ByteValueHiNibbleNotEq(atom.nibble)
    if isinstance(atom, ByteValueHiNibbleNotEq):
        return ByteValueHiNibbleEq(atom.nibble)
    if isinstance(atom, SpByte0Eq):
        return SpByte0NotEq(atom.value)
    if isinstance(atom, SpByte0NotEq):
        return SpByte0Eq(atom.value)
    if isinstance(atom, OutputLoNibbleEq):
        return OutputLoNibbleNotEq(atom.nibble)
    if isinstance(atom, OutputLoNibbleNotEq):
        return OutputLoNibbleEq(atom.nibble)
    if isinstance(atom, OutputHiNibbleEq):
        return OutputHiNibbleNotEq(atom.nibble)
    if isinstance(atom, OutputHiNibbleNotEq):
        return OutputHiNibbleEq(atom.nibble)
    raise TypeError(f"cannot negate atom of type {type(atom).__name__}")


def _push_not(p: Predicate) -> Predicate:
    """Push Not nodes to atoms via De Morgan, returning a Not-free-on-connectives AST."""
    if isinstance(p, Atom):
        return p
    if isinstance(p, And):
        return And(tuple(_push_not(c) for c in p.children))
    if isinstance(p, Or):
        return Or(tuple(_push_not(c) for c in p.children))
    if isinstance(p, Not):
        inner = p.child
        if isinstance(inner, Atom):
            return _negate_atom(inner)
        if isinstance(inner, Not):
            return _push_not(inner.child)
        if isinstance(inner, And):
            # NOT (A AND B) -> NOT A OR NOT B
            return _push_not(Or(tuple(Not(c) for c in inner.children)))
        if isinstance(inner, Or):
            # NOT (A OR B) -> NOT A AND NOT B
            return _push_not(And(tuple(Not(c) for c in inner.children)))
    raise TypeError(f"unknown predicate node: {type(p).__name__}")


# ---------------------------------------------------------------------------
# DNF
# ---------------------------------------------------------------------------


DNF = list[frozenset[Atom]]


def _dnf_of_nnf(p: Predicate) -> DNF:
    """Compute DNF assuming p is in negation-normal form (no Not on connectives)."""
    if isinstance(p, Atom):
        return [frozenset({p})]
    if isinstance(p, Or):
        result: DNF = []
        for child in p.children:
            for disj in _dnf_of_nnf(child):
                result.append(disj)
        return result
    if isinstance(p, And):
        # Cross product of child DNFs.
        result_lists: list[DNF] = [_dnf_of_nnf(c) for c in p.children]
        combined: DNF = [frozenset()]
        for child_dnf in result_lists:
            new_combined: DNF = []
            for existing in combined:
                for disj in child_dnf:
                    new_combined.append(existing | disj)
            combined = new_combined
        return combined
    raise TypeError(f"_dnf_of_nnf: unexpected node {type(p).__name__}")


def dnf(p: Predicate) -> DNF:
    """Normalize predicate to disjunctive normal form.

    Returns a list of disjuncts; each disjunct is a frozenset of atoms whose
    conjunction yields one alternative. Empty list means an unsatisfiable
    predicate (none currently produced); single empty frozenset would mean
    trivially-true (not produced — DSL has no constant atoms).
    """
    nnf = _push_not(p)
    return _dnf_of_nnf(nnf)


# ---------------------------------------------------------------------------
# Atom subsumption
# ---------------------------------------------------------------------------


def atom_subsumes(a: Atom, b: Atom) -> bool:
    """Return True iff atom `a` implies atom `b` (a is more specific or equal).

    Equivalent: every concrete position satisfying `a` also satisfies `b`.
    """
    # Trivial equality.
    if a == b:
        return True

    # --- Marker family -----------------------------------------------------
    if isinstance(a, MarkEq):
        if isinstance(b, MarkEq):
            return a.role == b.role
        if isinstance(b, MarkIn):
            return a.role in b.roles
        if isinstance(b, MarkNotEq):
            return a.role != b.role
        if isinstance(b, MarkNotIn):
            return a.role not in b.roles
        return False
    if isinstance(a, MarkIn):
        if isinstance(b, MarkIn):
            return a.roles.issubset(b.roles)
        if isinstance(b, MarkEq):
            # only subsumes singleton {role}
            return a.roles == {b.role}
        if isinstance(b, MarkNotEq):
            return b.role not in a.roles
        if isinstance(b, MarkNotIn):
            return a.roles.isdisjoint(b.roles)
        return False
    if isinstance(a, MarkNotEq):
        if isinstance(b, MarkNotEq):
            return a.role == b.role
        if isinstance(b, MarkNotIn):
            # NOT mark==X subsumes NOT mark in S iff S subset {X}
            return b.roles.issubset({a.role})
        return False
    if isinstance(a, MarkNotIn):
        if isinstance(b, MarkNotIn):
            return b.roles.issubset(a.roles)
        if isinstance(b, MarkNotEq):
            return b.role in a.roles
        return False

    # --- Boolean atoms -----------------------------------------------------
    if isinstance(a, BoolAtom):
        if isinstance(b, BoolAtom):
            return a.name == b.name and a.negated == b.negated
        return False

    # --- Step index family -------------------------------------------------
    if isinstance(a, StepIndexEq):
        if isinstance(b, StepIndexEq):
            return a.n == b.n
        if isinstance(b, StepIndexInSet):
            return a.n in b.values
        if isinstance(b, StepIndexInRange):
            return b.lo <= a.n < b.hi
        if isinstance(b, StepIndexNotEq):
            return a.n != b.n
        if isinstance(b, StepIndexNotInSet):
            return a.n not in b.values
        if isinstance(b, StepIndexNotInRange):
            return not (b.lo <= a.n < b.hi)
        return False
    if isinstance(a, StepIndexInSet):
        if isinstance(b, StepIndexInSet):
            return a.values.issubset(b.values)
        if isinstance(b, StepIndexEq):
            return a.values == {b.n}
        if isinstance(b, StepIndexInRange):
            return all(b.lo <= v < b.hi for v in a.values)
        if isinstance(b, StepIndexNotEq):
            return b.n not in a.values
        if isinstance(b, StepIndexNotInSet):
            return a.values.isdisjoint(b.values)
        if isinstance(b, StepIndexNotInRange):
            return all(not (b.lo <= v < b.hi) for v in a.values)
        return False
    if isinstance(a, StepIndexInRange):
        if isinstance(b, StepIndexInRange):
            # a ⊆ b
            if a.lo >= a.hi:
                return True  # empty
            return b.lo <= a.lo and a.hi <= b.hi
        if isinstance(b, StepIndexEq):
            # a is singleton {b.n}
            return a.lo == b.n and a.hi == b.n + 1
        if isinstance(b, StepIndexInSet):
            if a.lo >= a.hi:
                return True
            return all(v in b.values for v in range(a.lo, a.hi))
        if isinstance(b, StepIndexNotEq):
            return not (a.lo <= b.n < a.hi)
        if isinstance(b, StepIndexNotInSet):
            if a.lo >= a.hi:
                return True
            return all(v not in b.values for v in range(a.lo, a.hi))
        if isinstance(b, StepIndexNotInRange):
            # a's range disjoint from b's range
            if a.lo >= a.hi or b.lo >= b.hi:
                return True
            return a.hi <= b.lo or b.hi <= a.lo
        return False
    if isinstance(a, StepIndexNotEq):
        if isinstance(b, StepIndexNotEq):
            return a.n == b.n
        if isinstance(b, StepIndexNotInSet):
            return b.values.issubset({a.n})
        return False
    if isinstance(a, StepIndexNotInSet):
        if isinstance(b, StepIndexNotInSet):
            return b.values.issubset(a.values)
        if isinstance(b, StepIndexNotEq):
            return b.n in a.values
        return False
    if isinstance(a, StepIndexNotInRange):
        if isinstance(b, StepIndexNotInRange):
            # a says step ∉ [a.lo, a.hi); subsumes b only if b.range ⊆ a.range
            return a.lo <= b.lo and b.hi <= a.hi
        return False

    # --- Opcode-at-AX family ----------------------------------------------
    if isinstance(a, OpcodeAtAxEq):
        if isinstance(b, OpcodeAtAxEq):
            return a.opcode == b.opcode
        if isinstance(b, OpcodeAtAxIn):
            return a.opcode in b.opcodes
        if isinstance(b, OpcodeAtAxNotEq):
            return a.opcode != b.opcode
        if isinstance(b, OpcodeAtAxNotIn):
            return a.opcode not in b.opcodes
        return False
    if isinstance(a, OpcodeAtAxIn):
        if isinstance(b, OpcodeAtAxIn):
            return a.opcodes.issubset(b.opcodes)
        if isinstance(b, OpcodeAtAxEq):
            return a.opcodes == {b.opcode}
        if isinstance(b, OpcodeAtAxNotEq):
            return b.opcode not in a.opcodes
        if isinstance(b, OpcodeAtAxNotIn):
            return a.opcodes.isdisjoint(b.opcodes)
        return False
    if isinstance(a, OpcodeAtAxNotEq):
        if isinstance(b, OpcodeAtAxNotEq):
            return a.opcode == b.opcode
        if isinstance(b, OpcodeAtAxNotIn):
            return b.opcodes.issubset({a.opcode})
        return False
    if isinstance(a, OpcodeAtAxNotIn):
        if isinstance(b, OpcodeAtAxNotIn):
            return b.opcodes.issubset(a.opcodes)
        if isinstance(b, OpcodeAtAxNotEq):
            return b.opcode in a.opcodes
        return False

    # --- Opcode-in-step family --------------------------------------------
    if isinstance(a, OpcodeInStepIn):
        if isinstance(b, OpcodeInStepIn):
            return a.opcodes.issubset(b.opcodes)
        if isinstance(b, OpcodeInStepNotIn):
            return a.opcodes.isdisjoint(b.opcodes)
        return False
    if isinstance(a, OpcodeInStepNotIn):
        if isinstance(b, OpcodeInStepNotIn):
            return b.opcodes.issubset(a.opcodes)
        return False

    # --- Byte index --------------------------------------------------------
    if isinstance(a, ByteIndexEq):
        if isinstance(b, ByteIndexEq):
            return a.n == b.n
        if isinstance(b, ByteIndexIn):
            return a.n in b.values
        if isinstance(b, ByteIndexNotEq):
            return a.n != b.n
        if isinstance(b, ByteIndexNotIn):
            return a.n not in b.values
        return False
    if isinstance(a, ByteIndexIn):
        if isinstance(b, ByteIndexIn):
            return a.values.issubset(b.values)
        if isinstance(b, ByteIndexEq):
            return a.values == {b.n}
        if isinstance(b, ByteIndexNotEq):
            return b.n not in a.values
        if isinstance(b, ByteIndexNotIn):
            return a.values.isdisjoint(b.values)
        return False
    if isinstance(a, ByteIndexNotEq):
        if isinstance(b, ByteIndexNotEq):
            return a.n == b.n
        if isinstance(b, ByteIndexNotIn):
            return b.values.issubset({a.n})
        return False
    if isinstance(a, ByteIndexNotIn):
        if isinstance(b, ByteIndexNotIn):
            return b.values.issubset(a.values)
        if isinstance(b, ByteIndexNotEq):
            return b.n in a.values
        return False

    # --- Byte value --------------------------------------------------------
    if isinstance(a, ByteValueEq):
        if isinstance(b, ByteValueEq):
            return a.value == b.value
        if isinstance(b, ByteValueIn):
            return a.value in b.values
        if isinstance(b, ByteValueLoNibbleEq):
            return (a.value & 0xF) == b.nibble
        if isinstance(b, ByteValueHiNibbleEq):
            return ((a.value >> 4) & 0xF) == b.nibble
        if isinstance(b, ByteValueNotEq):
            return a.value != b.value
        if isinstance(b, ByteValueNotIn):
            return a.value not in b.values
        if isinstance(b, ByteValueLoNibbleNotEq):
            return (a.value & 0xF) != b.nibble
        if isinstance(b, ByteValueHiNibbleNotEq):
            return ((a.value >> 4) & 0xF) != b.nibble
        return False
    if isinstance(a, ByteValueIn):
        if isinstance(b, ByteValueIn):
            return a.values.issubset(b.values)
        if isinstance(b, ByteValueEq):
            return a.values == {b.value}
        if isinstance(b, ByteValueLoNibbleEq):
            return all((v & 0xF) == b.nibble for v in a.values)
        if isinstance(b, ByteValueHiNibbleEq):
            return all(((v >> 4) & 0xF) == b.nibble for v in a.values)
        if isinstance(b, ByteValueNotEq):
            return b.value not in a.values
        if isinstance(b, ByteValueNotIn):
            return a.values.isdisjoint(b.values)
        if isinstance(b, ByteValueLoNibbleNotEq):
            return all((v & 0xF) != b.nibble for v in a.values)
        if isinstance(b, ByteValueHiNibbleNotEq):
            return all(((v >> 4) & 0xF) != b.nibble for v in a.values)
        return False
    if isinstance(a, ByteValueLoNibbleEq):
        if isinstance(b, ByteValueLoNibbleEq):
            return a.nibble == b.nibble
        if isinstance(b, ByteValueLoNibbleNotEq):
            return a.nibble != b.nibble
        return False
    if isinstance(a, ByteValueHiNibbleEq):
        if isinstance(b, ByteValueHiNibbleEq):
            return a.nibble == b.nibble
        if isinstance(b, ByteValueHiNibbleNotEq):
            return a.nibble != b.nibble
        return False
    if isinstance(a, ByteValueNotEq):
        if isinstance(b, ByteValueNotEq):
            return a.value == b.value
        if isinstance(b, ByteValueNotIn):
            return b.values.issubset({a.value})
        return False
    if isinstance(a, ByteValueNotIn):
        if isinstance(b, ByteValueNotIn):
            return b.values.issubset(a.values)
        if isinstance(b, ByteValueNotEq):
            return b.value in a.values
        return False
    if isinstance(a, ByteValueLoNibbleNotEq):
        if isinstance(b, ByteValueLoNibbleNotEq):
            return a.nibble == b.nibble
        return False
    if isinstance(a, ByteValueHiNibbleNotEq):
        if isinstance(b, ByteValueHiNibbleNotEq):
            return a.nibble == b.nibble
        return False

    # --- sp_byte0 ---------------------------------------------------------
    if isinstance(a, SpByte0Eq):
        if isinstance(b, SpByte0Eq):
            return a.value == b.value
        if isinstance(b, SpByte0NotEq):
            return a.value != b.value
        return False
    if isinstance(a, SpByte0NotEq):
        if isinstance(b, SpByte0NotEq):
            return a.value == b.value
        return False

    # --- Output nibbles ---------------------------------------------------
    if isinstance(a, OutputLoNibbleEq):
        if isinstance(b, OutputLoNibbleEq):
            return a.nibble == b.nibble
        if isinstance(b, OutputLoNibbleNotEq):
            return a.nibble != b.nibble
        return False
    if isinstance(a, OutputHiNibbleEq):
        if isinstance(b, OutputHiNibbleEq):
            return a.nibble == b.nibble
        if isinstance(b, OutputHiNibbleNotEq):
            return a.nibble != b.nibble
        return False
    if isinstance(a, OutputLoNibbleNotEq):
        if isinstance(b, OutputLoNibbleNotEq):
            return a.nibble == b.nibble
        return False
    if isinstance(a, OutputHiNibbleNotEq):
        if isinstance(b, OutputHiNibbleNotEq):
            return a.nibble == b.nibble
        return False

    # Unknown atom pair → unrelated.
    return False


def conj_subsumes(c1: frozenset[Atom], c2: frozenset[Atom]) -> bool:
    """True iff conjunction c1 is at least as strong as c2.

    Equivalently: every concrete position satisfying all atoms in c1 also
    satisfies every atom in c2. We check that for every atom in c2 there
    exists an atom in c1 that subsumes it.
    """
    for q_atom in c2:
        if not any(atom_subsumes(p_atom, q_atom) for p_atom in c1):
            return False
    return True


# ---------------------------------------------------------------------------
# Entailment
# ---------------------------------------------------------------------------


def entails(p: Predicate, q: Predicate) -> bool:
    """Decide whether p ⊨ q (every model of p is a model of q)."""
    p_dnf = dnf(p)
    q_dnf = dnf(q)
    if not p_dnf:
        # p is unsatisfiable -> vacuously entails anything.
        return True
    # Filter out internally-contradictory p disjuncts: they represent
    # the empty set and entail anything vacuously. Our DNF builder
    # never emits an empty list, so contradictions surface as
    # frozensets containing structurally-incompatible atoms (e.g.,
    # {mark == MEM, mark == AX}); _disjunct_satisfiable rejects those.
    sat_p_dnf = [d for d in p_dnf if _disjunct_satisfiable(d)]
    if not sat_p_dnf:
        # Every disjunct of p was contradictory -> p is unsatisfiable.
        return True
    if not q_dnf:
        # q is unsatisfiable; only satisfied if p is too (already handled).
        return False
    for p_disj in sat_p_dnf:
        if not any(conj_subsumes(p_disj, q_disj) for q_disj in q_dnf):
            return False
    return True


def _format_conj(conj: frozenset[Atom]) -> str:
    if not conj:
        return "TRUE"
    return " AND ".join(sorted(str(a) for a in conj))


# ---------------------------------------------------------------------------
# Satisfiability and overlap
# ---------------------------------------------------------------------------


def atom_contradicts(a: Atom, b: Atom) -> bool:
    """Return True iff atoms `a` and `b` cannot both hold at any position.

    Implemented via subsumption against the negation of one operand:
    a ⊥ b iff a ⊨ NOT b (or symmetrically b ⊨ NOT a). For cross-family
    pairs (e.g., mark vs step_index), neither subsumption holds and the
    function correctly returns False (independent dimensions).
    """
    if a == b:
        return False
    # Try a entails NOT b, then symmetric.
    try:
        neg_b = _negate_atom(b)
    except TypeError:
        neg_b = None
    if neg_b is not None and atom_subsumes(a, neg_b):
        return True
    try:
        neg_a = _negate_atom(a)
    except TypeError:
        neg_a = None
    if neg_a is not None and atom_subsumes(b, neg_a):
        return True
    return False


def _disjunct_satisfiable(conj: frozenset[Atom]) -> bool:
    """A conjunction of atoms is satisfiable iff no pair contradicts."""
    atoms = list(conj)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if atom_contradicts(atoms[i], atoms[j]):
                return False
    return True


def satisfiable(p: Predicate) -> bool:
    """Decide if predicate p has any satisfying position.

    Decidable structurally over our atom families: a conjunction is
    unsatisfiable iff some pair of atoms contradicts (e.g., mark==SP
    AND mark==AX); a disjunction is satisfiable iff any disjunct is.
    """
    p_dnf = dnf(p)
    for disj in p_dnf:
        if _disjunct_satisfiable(disj):
            return True
    return False


def overlaps(p: Predicate, q: Predicate) -> bool:
    """Decide if there exists any position satisfying BOTH p and q.

    Equivalent to satisfiable(p AND q). Two scopes "overlap" iff there's
    a position where both rules could fire — they compete for the output
    at that position.
    """
    return satisfiable(And((p, q)))


def explain_failure(p: Predicate, q: Predicate) -> Optional[str]:
    """Return None if p ⊨ q else a one-line explanation."""
    p_dnf = dnf(p)
    q_dnf = dnf(q)
    if not p_dnf:
        return None
    for p_disj in p_dnf:
        matching = [qd for qd in q_dnf if conj_subsumes(p_disj, qd)]
        if matching:
            continue
        # Find the q disjunct closest to matching, and list missing atoms.
        if q_dnf:
            best = None
            best_missing: list[Atom] = []
            for q_disj in q_dnf:
                missing = [
                    qa for qa in q_disj
                    if not any(atom_subsumes(pa, qa) for pa in p_disj)
                ]
                if best is None or len(missing) < len(best_missing):
                    best = q_disj
                    best_missing = missing
            missing_str = ", ".join(sorted(str(a) for a in best_missing))
            return (
                f"disjunct {{{_format_conj(p_disj)}}} of p does not entail any "
                f"disjunct of q; closest q-disjunct {{{_format_conj(best)}}} "
                f"missing: [{missing_str}]"
            )
        return (
            f"disjunct {{{_format_conj(p_disj)}}} of p has no matching disjunct "
            f"in q (q is unsatisfiable)"
        )
    return None
