"""
RASP (Restricted Access Sequence Processing) Language Implementation.

RASP is a programming language designed to express computations that
transformers can implement. This module provides:

1. S-ops (Sequence operations) - Values at each sequence position
2. Selectors - Attention patterns (which positions attend to which)
3. Aggregate - Combine values using a selector

Reference: "Thinking Like Transformers" (Weiss et al., 2021)

Example:
    from rasp import tokens, indices, select, aggregate, Map

    # Reverse a sequence
    flip = select(indices, indices, lambda k, q: k == length - 1 - q)
    reversed_tokens = aggregate(flip, tokens)

    # Count tokens equal to current
    same = select(tokens, tokens, lambda k, q: k == q)
    count = aggregate(same, ones)
"""

from __future__ import annotations
from typing import Callable, Any, Optional, List, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import operator


# =============================================================================
# CORE TYPES
# =============================================================================

class SOp(ABC):
    """
    Sequence Operation - a value at each position in a sequence.

    S-ops are the fundamental data type in RASP. Each S-op represents
    a sequence of values, one per position.
    """

    _counter = 0

    def __init__(self, name: Optional[str] = None):
        SOp._counter += 1
        self.id = SOp._counter
        self.name = name or f"sop_{self.id}"
        self._encoding: Optional[Encoding] = None

    @abstractmethod
    def get_dependencies(self) -> List[SOp]:
        """Return S-ops this one depends on."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    # Arithmetic operations return new S-ops
    def __add__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(operator.add, self, _ensure_sop(other), "+")

    def __radd__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(operator.add, _ensure_sop(other), self, "+")

    def __sub__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(operator.sub, self, _ensure_sop(other), "-")

    def __rsub__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(operator.sub, _ensure_sop(other), self, "-")

    def __mul__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(operator.mul, self, _ensure_sop(other), "*")

    def __rmul__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(operator.mul, _ensure_sop(other), self, "*")

    def __truediv__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(operator.truediv, self, _ensure_sop(other), "/")

    def __rtruediv__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(operator.truediv, _ensure_sop(other), self, "/")

    def __neg__(self) -> SOp:
        return ElementwiseOp(operator.neg, self, None, "neg")

    def __eq__(self, other: Union[SOp, float, int]) -> SOp:  # type: ignore
        return ElementwiseOp(operator.eq, self, _ensure_sop(other), "==")

    def __ne__(self, other: Union[SOp, float, int]) -> SOp:  # type: ignore
        return ElementwiseOp(operator.ne, self, _ensure_sop(other), "!=")

    def __lt__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(operator.lt, self, _ensure_sop(other), "<")

    def __le__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(operator.le, self, _ensure_sop(other), "<=")

    def __gt__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(operator.gt, self, _ensure_sop(other), ">")

    def __ge__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(operator.ge, self, _ensure_sop(other), ">=")

    def __and__(self, other: SOp) -> SOp:
        return ElementwiseOp(lambda a, b: a and b, self, other, "and")

    def __or__(self, other: SOp) -> SOp:
        return ElementwiseOp(lambda a, b: a or b, self, other, "or")

    def __invert__(self) -> SOp:
        return ElementwiseOp(lambda a, _: not a, self, None, "not")

    # =========================================================================
    # C4-COMPATIBLE OPERATIONS
    # =========================================================================

    def __mod__(self, other: Union[SOp, float, int]) -> SOp:
        """Modulo: a % b = a - floor(a/b) * b"""
        return ElementwiseOp(lambda a, b: a - int(a / b) * b if b != 0 else 0,
                            self, _ensure_sop(other), "%")

    def __rmod__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(lambda a, b: a - int(a / b) * b if b != 0 else 0,
                            _ensure_sop(other), self, "%")

    def __floordiv__(self, other: Union[SOp, float, int]) -> SOp:
        """Floor division: a // b"""
        return ElementwiseOp(lambda a, b: int(a / b) if b != 0 else 0,
                            self, _ensure_sop(other), "//")

    def __rfloordiv__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(lambda a, b: int(a / b) if b != 0 else 0,
                            _ensure_sop(other), self, "//")

    def __lshift__(self, other: Union[SOp, float, int]) -> SOp:
        """Left shift: a << b = a * 2^b"""
        return ElementwiseOp(lambda a, b: int(a * (2 ** int(b))),
                            self, _ensure_sop(other), "<<")

    def __rlshift__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(lambda a, b: int(a * (2 ** int(b))),
                            _ensure_sop(other), self, "<<")

    def __rshift__(self, other: Union[SOp, float, int]) -> SOp:
        """Right shift: a >> b = a // 2^b"""
        return ElementwiseOp(lambda a, b: int(a / (2 ** int(b))),
                            self, _ensure_sop(other), ">>")

    def __rrshift__(self, other: Union[SOp, float, int]) -> SOp:
        return ElementwiseOp(lambda a, b: int(a / (2 ** int(b))),
                            _ensure_sop(other), self, ">>")

    def __pos__(self) -> SOp:
        """Unary plus: +x (identity)"""
        return self

    # Helper methods for C-style operations
    def increment(self) -> SOp:
        """C-style ++: returns x + 1"""
        return self + 1

    def decrement(self) -> SOp:
        """C-style --: returns x - 1"""
        return self - 1


def _ensure_sop(x: Union[SOp, float, int]) -> SOp:
    """Convert scalars to constant S-ops."""
    if isinstance(x, SOp):
        return x
    return ConstantSOp(x)


class Selector(ABC):
    """
    Selector - an attention pattern.

    A selector is a boolean matrix where selector[i, j] indicates
    whether position i should attend to position j.
    """

    _counter = 0

    def __init__(self, name: Optional[str] = None):
        Selector._counter += 1
        self.id = Selector._counter
        self.name = name or f"sel_{self.id}"

    @abstractmethod
    def get_dependencies(self) -> Tuple[List[SOp], List[Selector]]:
        """Return (S-ops, Selectors) this selector depends on."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __and__(self, other: Selector) -> Selector:
        return SelectorAnd(self, other)

    def __or__(self, other: Selector) -> Selector:
        return SelectorOr(self, other)

    def __invert__(self) -> Selector:
        return SelectorNot(self)


# =============================================================================
# PRIMITIVE S-OPS
# =============================================================================

class InputSOp(SOp):
    """Input tokens - the raw input to the transformer."""

    def __init__(self):
        super().__init__("tokens")

    def get_dependencies(self) -> List[SOp]:
        return []


class IndicesSOp(SOp):
    """Position indices: [0, 1, 2, ..., seq_len-1]."""

    def __init__(self):
        super().__init__("indices")

    def get_dependencies(self) -> List[SOp]:
        return []


class LengthSOp(SOp):
    """Sequence length (same value at all positions)."""

    def __init__(self):
        super().__init__("length")

    def get_dependencies(self) -> List[SOp]:
        return []


class OnesSOp(SOp):
    """All ones: [1, 1, 1, ...]."""

    def __init__(self):
        super().__init__("ones")

    def get_dependencies(self) -> List[SOp]:
        return []


class ConstantSOp(SOp):
    """Constant value at all positions."""

    def __init__(self, value: Union[float, int]):
        super().__init__(f"const_{value}")
        self.value = value

    def get_dependencies(self) -> List[SOp]:
        return []


# Global primitive S-ops
tokens = InputSOp()
indices = IndicesSOp()
length = LengthSOp()
ones = OnesSOp()


# =============================================================================
# DERIVED S-OPS
# =============================================================================

class ElementwiseOp(SOp):
    """Element-wise operation on one or two S-ops."""

    def __init__(
        self,
        op: Callable,
        left: SOp,
        right: Optional[SOp],
        op_name: str,
        name: Optional[str] = None
    ):
        super().__init__(name or f"({left.name}{op_name}{right.name if right else ''})")
        self.op = op
        self.left = left
        self.right = right
        self.op_name = op_name

    def get_dependencies(self) -> List[SOp]:
        if self.right is None:
            return [self.left]
        return [self.left, self.right]


class MapOp(SOp):
    """Apply a function element-wise to an S-op."""

    def __init__(self, fn: Callable, arg: SOp, name: Optional[str] = None):
        super().__init__(name or f"map({arg.name})")
        self.fn = fn
        self.arg = arg

    def get_dependencies(self) -> List[SOp]:
        return [self.arg]


class SequenceMapOp(SOp):
    """Apply a function element-wise to multiple S-ops."""

    def __init__(self, fn: Callable, args: List[SOp], name: Optional[str] = None):
        arg_names = ", ".join(a.name for a in args)
        super().__init__(name or f"seqmap({arg_names})")
        self.fn = fn
        self.args = args

    def get_dependencies(self) -> List[SOp]:
        return list(self.args)


class AggregateOp(SOp):
    """
    Aggregate values using a selector (attention pattern).

    For each position i:
        output[i] = mean(values[j] for j where selector[i, j] is True)

    If no positions are selected, uses the default value.
    """

    def __init__(
        self,
        selector: Selector,
        values: SOp,
        default: Optional[Union[SOp, float, int]] = None,
        name: Optional[str] = None
    ):
        super().__init__(name or f"agg({selector.name}, {values.name})")
        self.selector = selector
        self.values = values
        self.default = _ensure_sop(default) if default is not None else ConstantSOp(0)

    def get_dependencies(self) -> List[SOp]:
        return [self.values, self.default]


class SelectorWidthOp(SOp):
    """Count how many positions each position attends to."""

    def __init__(self, selector: Selector, name: Optional[str] = None):
        super().__init__(name or f"width({selector.name})")
        self.selector = selector

    def get_dependencies(self) -> List[SOp]:
        return []


# =============================================================================
# SELECTORS
# =============================================================================

class SelectOp(Selector):
    """
    Create a selector from key and query S-ops with a predicate.

    selector[i, j] = predicate(keys[j], queries[i])

    Example:
        # Causal attention: each position attends to earlier positions
        causal = select(indices, indices, lambda k, q: k <= q)

        # Match tokens: attend to positions with same token
        same_token = select(tokens, tokens, lambda k, q: k == q)
    """

    def __init__(
        self,
        keys: SOp,
        queries: SOp,
        predicate: Callable[[Any, Any], bool],
        name: Optional[str] = None
    ):
        super().__init__(name or f"select({keys.name}, {queries.name})")
        self.keys = keys
        self.queries = queries
        self.predicate = predicate

    def get_dependencies(self) -> Tuple[List[SOp], List[Selector]]:
        return ([self.keys, self.queries], [])


class ConstantSelector(Selector):
    """Selector that's True everywhere or nowhere."""

    def __init__(self, value: bool, name: Optional[str] = None):
        super().__init__(name or f"sel_{'all' if value else 'none'}")
        self.value = value

    def get_dependencies(self) -> Tuple[List[SOp], List[Selector]]:
        return ([], [])


class SelectorAnd(Selector):
    """Intersection of two selectors."""

    def __init__(self, left: Selector, right: Selector, name: Optional[str] = None):
        super().__init__(name or f"({left.name} & {right.name})")
        self.left = left
        self.right = right

    def get_dependencies(self) -> Tuple[List[SOp], List[Selector]]:
        left_sops, left_sels = self.left.get_dependencies()
        right_sops, right_sels = self.right.get_dependencies()
        return (left_sops + right_sops, [self.left, self.right] + left_sels + right_sels)


class SelectorOr(Selector):
    """Union of two selectors."""

    def __init__(self, left: Selector, right: Selector, name: Optional[str] = None):
        super().__init__(name or f"({left.name} | {right.name})")
        self.left = left
        self.right = right

    def get_dependencies(self) -> Tuple[List[SOp], List[Selector]]:
        left_sops, left_sels = self.left.get_dependencies()
        right_sops, right_sels = self.right.get_dependencies()
        return (left_sops + right_sops, [self.left, self.right] + left_sels + right_sels)


class SelectorNot(Selector):
    """Complement of a selector."""

    def __init__(self, inner: Selector, name: Optional[str] = None):
        super().__init__(name or f"~{inner.name}")
        self.inner = inner

    def get_dependencies(self) -> Tuple[List[SOp], List[Selector]]:
        sops, sels = self.inner.get_dependencies()
        return (sops, [self.inner] + sels)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def select(
    keys: SOp,
    queries: SOp,
    predicate: Callable[[Any, Any], bool],
    name: Optional[str] = None
) -> Selector:
    """Create a selector from key/query S-ops with a predicate."""
    return SelectOp(keys, queries, predicate, name)


def aggregate(
    selector: Selector,
    values: SOp,
    default: Optional[Union[SOp, float, int]] = None,
    name: Optional[str] = None
) -> SOp:
    """Aggregate values using a selector (attention pattern)."""
    return AggregateOp(selector, values, default, name)


def selector_width(selector: Selector, name: Optional[str] = None) -> SOp:
    """Count how many positions each position attends to."""
    return SelectorWidthOp(selector, name)


def Map(fn: Callable, arg: SOp, name: Optional[str] = None) -> SOp:
    """Apply a function element-wise."""
    return MapOp(fn, arg, name)


def SequenceMap(fn: Callable, *args: SOp, name: Optional[str] = None) -> SOp:
    """Apply a function element-wise to multiple S-ops."""
    return SequenceMapOp(fn, list(args), name)


# Common selectors
def select_all() -> Selector:
    """Selector that's True everywhere."""
    return ConstantSelector(True, "all")


def select_none() -> Selector:
    """Selector that's False everywhere."""
    return ConstantSelector(False, "none")


# =============================================================================
# C4-COMPATIBLE HELPER FUNCTIONS
# =============================================================================

def ternary(condition: SOp, true_val: Union[SOp, float, int],
            false_val: Union[SOp, float, int], name: Optional[str] = None) -> SOp:
    """
    C-style ternary operator: condition ? true_val : false_val

    Implementation: condition * true_val + (1 - condition) * false_val
    Assumes condition is 0 or 1 (boolean).
    """
    cond = _ensure_sop(condition)
    tv = _ensure_sop(true_val)
    fv = _ensure_sop(false_val)
    result = cond * tv + (ConstantSOp(1) - cond) * fv
    if name:
        result.name = name
    return result


def c_sizeof(type_name: str) -> SOp:
    """
    C-style sizeof - returns size in bytes as constant.

    In C4: char=1, int=8 (64-bit), pointer=8
    """
    sizes = {
        'char': 1,
        'int': 8,      # C4 uses 64-bit ints
        'ptr': 8,
        'pointer': 8,
        'void*': 8,
        'char*': 8,
        'int*': 8,
    }
    return ConstantSOp(sizes.get(type_name, 8))


def floor_div(a: SOp, b: Union[SOp, float, int]) -> SOp:
    """Floor division: a // b"""
    return a // b


def modulo(a: SOp, b: Union[SOp, float, int]) -> SOp:
    """Modulo: a % b"""
    return a % b


def left_shift(a: SOp, b: Union[SOp, float, int]) -> SOp:
    """Left shift: a << b = a * 2^b"""
    return a << b


def right_shift(a: SOp, b: Union[SOp, float, int]) -> SOp:
    """Right shift: a >> b = a // 2^b"""
    return a >> b


def negate(a: SOp) -> SOp:
    """Unary negation: -a"""
    return -a


def logical_not(a: SOp) -> SOp:
    """Logical NOT: !a (returns 1 if a==0, else 0)"""
    return ternary(a == 0, 1, 0)


def abs_val(a: SOp) -> SOp:
    """Absolute value: |a|"""
    return ternary(a < 0, -a, a)


def min_val(a: SOp, b: Union[SOp, float, int]) -> SOp:
    """Minimum of two values"""
    b_sop = _ensure_sop(b)
    return ternary(a < b_sop, a, b_sop)


def max_val(a: SOp, b: Union[SOp, float, int]) -> SOp:
    """Maximum of two values"""
    b_sop = _ensure_sop(b)
    return ternary(a > b_sop, a, b_sop)


def clamp(a: SOp, lo: Union[SOp, float, int], hi: Union[SOp, float, int]) -> SOp:
    """Clamp value between lo and hi"""
    return min_val(max_val(a, lo), hi)


# =============================================================================
# BITWISE OPERATIONS (via bit decomposition)
# =============================================================================

def get_bit(x: SOp, bit_pos: int) -> SOp:
    """
    Extract bit at position bit_pos from x.

    bit_i = floor(x / 2^i) % 2

    Returns 0 or 1.
    """
    shifted = x >> bit_pos  # floor(x / 2^bit_pos)
    return shifted % 2      # extract lowest bit


def bits_to_int(bits: List[SOp]) -> SOp:
    """
    Reconstruct integer from list of bits (LSB first).

    x = sum(bits[i] * 2^i)
    """
    result = bits[0]
    for i, bit in enumerate(bits[1:], 1):
        result = result + bit * (2 ** i)
    return result


def decompose_bits(x: SOp, num_bits: int) -> List[SOp]:
    """
    Decompose integer x into a list of bits (LSB first).

    Returns [bit_0, bit_1, ..., bit_{n-1}]
    """
    return [get_bit(x, i) for i in range(num_bits)]


def bitwise_and(a: SOp, b: SOp, num_bits: int = 8) -> SOp:
    """
    Bitwise AND: a & b

    For each bit position: a_i AND b_i = a_i * b_i (since bits are 0 or 1)
    Then reconstruct the result.
    """
    a_bits = decompose_bits(a, num_bits)
    b_bits = decompose_bits(b, num_bits)
    result_bits = [a_bits[i] * b_bits[i] for i in range(num_bits)]
    return bits_to_int(result_bits)


def bitwise_or(a: SOp, b: SOp, num_bits: int = 8) -> SOp:
    """
    Bitwise OR: a | b

    For each bit position: a_i OR b_i = a_i + b_i - a_i * b_i
    (This is: max(a,b) for bits, or: 1 - (1-a)(1-b))
    """
    a_bits = decompose_bits(a, num_bits)
    b_bits = decompose_bits(b, num_bits)
    result_bits = [a_bits[i] + b_bits[i] - a_bits[i] * b_bits[i] for i in range(num_bits)]
    return bits_to_int(result_bits)


def bitwise_xor(a: SOp, b: SOp, num_bits: int = 8) -> SOp:
    """
    Bitwise XOR: a ^ b

    For each bit position: a_i XOR b_i = a_i + b_i - 2 * a_i * b_i
    (This is: (a + b) mod 2, or: a != b)
    """
    a_bits = decompose_bits(a, num_bits)
    b_bits = decompose_bits(b, num_bits)
    result_bits = [a_bits[i] + b_bits[i] - 2 * a_bits[i] * b_bits[i] for i in range(num_bits)]
    return bits_to_int(result_bits)


def bitwise_not(a: SOp, num_bits: int = 8) -> SOp:
    """
    Bitwise NOT: ~a (for num_bits width)

    ~a = (2^num_bits - 1) - a

    Or per-bit: NOT a_i = 1 - a_i
    """
    a_bits = decompose_bits(a, num_bits)
    result_bits = [1 - a_bits[i] for i in range(num_bits)]
    return bits_to_int(result_bits)


def count_ones(x: SOp, num_bits: int = 8) -> SOp:
    """
    Population count (popcount): count number of 1 bits in x.

    popcount(x) = sum of all bits
    """
    bits = decompose_bits(x, num_bits)
    result = bits[0]
    for bit in bits[1:]:
        result = result + bit
    return result


def leading_zeros(x: SOp, num_bits: int = 8) -> SOp:
    """
    Count leading zeros in x.

    Start from MSB, count consecutive zeros until first 1.
    """
    bits = decompose_bits(x, num_bits)
    # Scan from MSB to LSB
    # leading_zeros = num_bits - (highest set bit position + 1)
    # This is complex - use cumulative approach

    # Alternative: check if x < 2^i for each i
    count = ConstantSOp(0)
    for i in range(num_bits - 1, -1, -1):
        # If bit i is the highest set bit, leading zeros = num_bits - 1 - i
        # We accumulate: if bit[i] == 0, add 1 to count (until we hit a 1)
        bit_clear = 1 - bits[i]
        # We need to track "still counting" state - tricky without state
        # Simplified: return num_bits - floor(log2(x)) - 1
        pass

    # Simpler approximation: num_bits - popcount(x | (x-1)) ... complex
    # For now, return placeholder using comparison chain
    result = ConstantSOp(num_bits)
    threshold = 1
    for i in range(num_bits):
        # If x >= 2^i, then leading zeros <= num_bits - i - 1
        is_ge = ternary(x >= threshold, 1, 0)
        result = result - is_ge
        threshold *= 2
    return result


# =============================================================================
# SOFTMAX1-BASED OPERATIONS
# =============================================================================
"""
Softmax1 (quiet attention) provides several primitives "for free":

1. sigmoid(s) = softmax1([s])  -- single element case
2. count via leftover: n = 1/leftover - 1
3. soft thresholding: sigmoid(β*(x-t)) → step(x-t) as β→∞
4. reciprocal: sigmoid(-log(x)) = 1/(1+x) exactly
5. soft max/argmax: softmax1(β*x) → one-hot at argmax

These can simplify many operations compared to explicit arithmetic.
"""


class Softmax1Op(SOp):
    """
    Softmax1 operation: exp(x) / (1 + sum(exp(x)))

    Returns attention weights that sum to < 1.
    The "leftover" (1 - sum) encodes additional information.
    """
    def __init__(self, scores: SOp, beta: float = 1.0, name: Optional[str] = None):
        super().__init__(name or f"softmax1({scores.name})")
        self.scores = scores
        self.beta = beta

    def get_dependencies(self) -> List[SOp]:
        return [self.scores]


class SigmoidOp(SOp):
    """
    Sigmoid: exp(x) / (1 + exp(x))

    Equivalent to softmax1 on a single element.
    With temperature β: sigmoid(β*x) → step function as β→∞
    """
    def __init__(self, x: SOp, beta: float = 1.0, name: Optional[str] = None):
        super().__init__(name or f"sigmoid({x.name})")
        self.x = x
        self.beta = beta

    def get_dependencies(self) -> List[SOp]:
        return [self.x]


def sigmoid(x: SOp, beta: float = 1.0) -> SOp:
    """
    Sigmoid function: 1 / (1 + exp(-x))

    With temperature β: sigmoid(β*x) → step function as β→∞
    - sigmoid(βx) ≈ 1 if x > 0
    - sigmoid(βx) ≈ 0 if x < 0
    """
    return SigmoidOp(x, beta)


def soft_gt(a: SOp, b: Union[SOp, float, int], beta: float = 10.0) -> SOp:
    """
    Soft greater-than: approximates (a > b) ? 1 : 0

    Returns sigmoid(β*(a-b)), which approaches step function as β→∞
    """
    return sigmoid(a - _ensure_sop(b), beta)


def soft_lt(a: SOp, b: Union[SOp, float, int], beta: float = 10.0) -> SOp:
    """
    Soft less-than: approximates (a < b) ? 1 : 0
    """
    return sigmoid(_ensure_sop(b) - a, beta)


def soft_eq(a: SOp, b: Union[SOp, float, int], tolerance: float = 0.5, beta: float = 10.0) -> SOp:
    """
    Soft equality: approximates (|a - b| < tolerance) ? 1 : 0

    Uses: sigmoid(β*(tolerance - |a-b|))
    """
    diff = abs_val(a - _ensure_sop(b))
    return sigmoid(ConstantSOp(tolerance) - diff, beta)


def soft_max_value(values: List[SOp], beta: float = 10.0) -> SOp:
    """
    Soft maximum: returns approximately max(values)

    Uses softmax1(β*values) as weights, then weighted sum.
    As β→∞, this approaches the hard max.
    """
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        a, b = values
        weight_a = sigmoid((a - b) * beta)
        return weight_a * a + (ConstantSOp(1) - weight_a) * b
    # For more values, would need full softmax1 aggregation
    # Approximate with pairwise max
    result = values[0]
    for v in values[1:]:
        result = soft_max_value([result, v], beta)
    return result


def soft_min_value(values: List[SOp], beta: float = 10.0) -> SOp:
    """
    Soft minimum: returns approximately min(values)
    """
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        a, b = values
        weight_a = sigmoid((b - a) * beta)  # reversed
        return weight_a * a + (ConstantSOp(1) - weight_a) * b
    result = values[0]
    for v in values[1:]:
        result = soft_min_value([result, v], beta)
    return result


def soft_ternary(cond: SOp, true_val: Union[SOp, float, int],
                 false_val: Union[SOp, float, int], beta: float = 10.0) -> SOp:
    """
    Soft ternary: cond ? true_val : false_val

    Uses sigmoid to smoothly interpolate based on condition.
    When cond >> 0: returns true_val
    When cond << 0: returns false_val

    For boolean cond (0 or 1), use beta to sharpen.
    """
    tv = _ensure_sop(true_val)
    fv = _ensure_sop(false_val)
    weight = sigmoid(cond, beta)
    return weight * tv + (ConstantSOp(1) - weight) * fv


def reciprocal_via_softmax1(x: SOp) -> SOp:
    """
    Compute 1/(1+x) using sigmoid(-log(x)).

    Note: This requires log(x) which is non-trivial.
    For the special case where we know the count n:
        leftover = 1/(1+n), so we get 1/(1+n) directly!
    """
    # For now, fall back to the algebraic approach
    # 1/(1+x) ≈ 1 - x + x² - x³ + ... (Taylor for |x| < 1)
    # Or use the attention-based reciprocal from hidden_dim_ops
    return ConstantSOp(1) / (ConstantSOp(1) + x)


def count_via_softmax1_leftover(selector: Selector) -> SOp:
    """
    Count positions selected by a selector using softmax1 leftover.

    When all selected positions have score 0:
        leftover = 1/(1+n)
        n = 1/leftover - 1

    This is an alternative to selector_width that comes naturally
    from softmax1 attention.

    Note: In standard RASP, selector_width is the direct approach.
    This function shows the softmax1 interpretation.
    """
    # This is equivalent to selector_width conceptually
    return SelectorWidthOp(selector)


def soft_bit_extract(x: SOp, bit_pos: int, beta: float = 100.0) -> SOp:
    """
    Extract bit at position bit_pos using sigmoid thresholding.

    bit_i ≈ sigmoid(β * (x % 2^(i+1) - 2^i + 0.5))

    As β→∞, this gives 1 if bit is set, 0 otherwise.
    """
    mod_val = x % (2 ** (bit_pos + 1))
    threshold = 2 ** bit_pos - 0.5
    return sigmoid(mod_val - ConstantSOp(threshold), beta)


# =============================================================================
# ENCODING (for compilation)
# =============================================================================

@dataclass
class Encoding:
    """
    Encoding of an S-op's values into residual stream dimensions.

    Each S-op gets assigned a "basis" - a set of dimensions in the
    residual stream where its values are stored.

    For categorical S-ops (like tokens), we use one-hot encoding.
    For numerical S-ops (like indices), we use direct encoding.
    """

    class Type(Enum):
        CATEGORICAL = auto()  # One-hot encoding
        NUMERICAL = auto()     # Direct numerical value

    encoding_type: Type
    basis_start: int  # Starting dimension in residual stream
    basis_size: int   # Number of dimensions used
    categories: Optional[List[Any]] = None  # For categorical encoding


# =============================================================================
# RASP PROGRAM
# =============================================================================

@dataclass
class RASPProgram:
    """
    A complete RASP program ready for compilation.

    Contains:
    - outputs: S-ops whose values we want to compute
    - vocab: Input vocabulary (for token encoding)
    - max_seq_len: Maximum sequence length
    """

    outputs: List[SOp]
    vocab: List[Any]
    max_seq_len: int
    name: str = "program"

    def get_all_ops(self) -> Tuple[List[SOp], List[Selector]]:
        """Get all S-ops and Selectors used in the program."""
        all_sops: List[SOp] = []
        all_selectors: List[Selector] = []
        visited_sops: Set[int] = set()
        visited_sels: Set[int] = set()

        def visit_sop(sop: SOp):
            if sop.id in visited_sops:
                return
            visited_sops.add(sop.id)

            # Visit dependencies first
            for dep in sop.get_dependencies():
                visit_sop(dep)

            # Handle AggregateOp specially - also visit selector
            if isinstance(sop, AggregateOp):
                visit_selector(sop.selector)
            if isinstance(sop, SelectorWidthOp):
                visit_selector(sop.selector)

            all_sops.append(sop)

        def visit_selector(sel: Selector):
            if sel.id in visited_sels:
                return
            visited_sels.add(sel.id)

            sops, sels = sel.get_dependencies()
            for s in sops:
                visit_sop(s)
            for s in sels:
                visit_selector(s)

            all_selectors.append(sel)

        for output in self.outputs:
            visit_sop(output)

        return all_sops, all_selectors


# =============================================================================
# EXAMPLE PROGRAMS
# =============================================================================

def make_length_program() -> RASPProgram:
    """Program that computes sequence length at each position."""
    # Select all positions, count them with selector_width
    all_selector = select(ones, ones, lambda k, q: True, "all")
    seq_length = selector_width(all_selector, name="length")

    return RASPProgram(
        outputs=[seq_length],
        vocab=list(range(10)),  # dummy vocab
        max_seq_len=16,
        name="length"
    )


def make_reverse_program() -> RASPProgram:
    """Program that reverses the input sequence."""
    # flip[i, j] = (i + j == length - 1)
    # This requires knowing length, which we compute first
    all_sel = select(ones, ones, lambda k, q: True)
    seq_len = aggregate(all_sel, ones)  # This gives length at each position

    # Now create flip selector: attend to position (length - 1 - i)
    # flip[i, j] = (j == length - 1 - i) = (i + j == length - 1)
    flip = select(indices, indices + seq_len - 1, lambda k, q: k == q, "flip")

    reversed_tokens = aggregate(flip, tokens, name="reversed")

    return RASPProgram(
        outputs=[reversed_tokens],
        vocab=list("abcdefghij"),
        max_seq_len=16,
        name="reverse"
    )


def make_histogram_program() -> RASPProgram:
    """Program that counts occurrences of each token."""
    # same[i, j] = (tokens[i] == tokens[j])
    same = select(tokens, tokens, lambda k, q: k == q, "same")

    # Count matches using selector_width
    count = selector_width(same, name="count")

    return RASPProgram(
        outputs=[count],
        vocab=list("abcdefghij"),
        max_seq_len=16,
        name="histogram"
    )


def make_sort_program() -> RASPProgram:
    """
    Program that sorts the input sequence.

    Uses the "smaller or equal count" trick:
    - For each position, count how many tokens are smaller
    - That count is the sorted position
    """
    # Count tokens that are smaller (attend to smaller tokens)
    smaller = select(tokens, tokens, lambda k, q: k < q, "smaller")
    smaller_count = aggregate(smaller, ones, name="smaller_count")

    # Count tokens that are equal but at earlier position (tiebreaker)
    same_earlier = select(
        SequenceMap(lambda t, i: (t, i), tokens, indices),
        SequenceMap(lambda t, i: (t, i), tokens, indices),
        lambda k, q: k[0] == q[0] and k[1] < q[1],
        "same_earlier"
    )
    same_earlier_count = aggregate(same_earlier, ones, name="same_earlier_count")

    # Final position in sorted order
    sorted_position = smaller_count + same_earlier_count

    # Now select from original based on sorted position
    target_pos = select(sorted_position, indices, lambda k, q: k == q, "target")
    sorted_tokens = aggregate(target_pos, tokens, name="sorted")

    return RASPProgram(
        outputs=[sorted_tokens],
        vocab=list(range(10)),
        max_seq_len=16,
        name="sort"
    )


def make_dyck_program(n_parens: int = 1) -> RASPProgram:
    """
    Program that checks if parentheses are balanced (Dyck language).

    For single paren type:
    - '(' adds 1 to depth
    - ')' subtracts 1 from depth
    - Valid if depth never goes negative and ends at 0
    """
    # Map tokens to +1 (open) or -1 (close)
    is_open = (tokens == '(')
    is_close = (tokens == ')')
    delta = is_open * 1 + is_close * (-1)

    # Running sum of deltas (depth at each position)
    prefix = select(indices, indices, lambda k, q: k <= q, "prefix")
    depth = aggregate(prefix, delta, name="depth")

    # Check if depth ever goes negative
    was_negative = (depth < 0)
    any_negative = aggregate(select_all(), was_negative, name="any_negative")

    # Check final depth is 0 (at last position)
    # This is trickier - we need to identify the last position

    return RASPProgram(
        outputs=[depth, any_negative],
        vocab=['(', ')'],
        max_seq_len=32,
        name="dyck"
    )


# =============================================================================
# INTERPRETER (for testing)
# =============================================================================

def interpret(program: RASPProgram, input_sequence: List[Any]) -> Dict[str, List[Any]]:
    """
    Interpret a RASP program on an input sequence.

    This is a reference implementation for testing - it directly
    executes the RASP operations without compiling to a transformer.
    """
    seq_len = len(input_sequence)

    # Cache for computed values
    sop_values: Dict[int, List[Any]] = {}
    selector_values: Dict[int, List[List[bool]]] = {}

    def eval_sop(sop: SOp) -> List[Any]:
        if sop.id in sop_values:
            return sop_values[sop.id]

        if isinstance(sop, InputSOp):
            result = list(input_sequence)

        elif isinstance(sop, IndicesSOp):
            result = list(range(seq_len))

        elif isinstance(sop, LengthSOp):
            result = [seq_len] * seq_len

        elif isinstance(sop, OnesSOp):
            result = [1] * seq_len

        elif isinstance(sop, ConstantSOp):
            result = [sop.value] * seq_len

        elif isinstance(sop, ElementwiseOp):
            left_vals = eval_sop(sop.left)
            if sop.right is not None:
                right_vals = eval_sop(sop.right)
                result = [sop.op(l, r) for l, r in zip(left_vals, right_vals)]
            else:
                # Unary operation - check if op takes 1 or 2 args
                import inspect
                try:
                    sig = inspect.signature(sop.op)
                    num_params = len(sig.parameters)
                except (ValueError, TypeError):
                    num_params = 1  # Assume unary for built-ins like operator.neg

                if num_params == 1:
                    result = [sop.op(l) for l in left_vals]
                else:
                    result = [sop.op(l, None) for l in left_vals]

        elif isinstance(sop, MapOp):
            arg_vals = eval_sop(sop.arg)
            result = [sop.fn(v) for v in arg_vals]

        elif isinstance(sop, SequenceMapOp):
            arg_vals = [eval_sop(arg) for arg in sop.args]
            result = [sop.fn(*vals) for vals in zip(*arg_vals)]

        elif isinstance(sop, AggregateOp):
            sel_matrix = eval_selector(sop.selector)
            values = eval_sop(sop.values)
            default_vals = eval_sop(sop.default)

            result = []
            for i in range(seq_len):
                selected = [values[j] for j in range(seq_len) if sel_matrix[i][j]]
                if selected:
                    # Average (what attention does)
                    if all(isinstance(v, (int, float)) for v in selected):
                        result.append(sum(selected) / len(selected))
                    else:
                        # For non-numeric, just take first (or could error)
                        result.append(selected[0])
                else:
                    result.append(default_vals[i])

        elif isinstance(sop, SelectorWidthOp):
            sel_matrix = eval_selector(sop.selector)
            result = [sum(row) for row in sel_matrix]

        elif isinstance(sop, SigmoidOp):
            x_vals = eval_sop(sop.x)
            import math
            result = [1 / (1 + math.exp(-sop.beta * v)) for v in x_vals]

        elif isinstance(sop, Softmax1Op):
            score_vals = eval_sop(sop.scores)
            import math
            # For single values, softmax1 = sigmoid
            scaled = [sop.beta * v for v in score_vals]
            max_val = max(scaled) if scaled else 0
            exp_vals = [math.exp(v - max_val) for v in scaled]
            denominator = math.exp(-max_val) + sum(exp_vals)  # The "+1" in softmax1
            result = [e / denominator for e in exp_vals]

        else:
            raise ValueError(f"Unknown S-op type: {type(sop)}")

        sop_values[sop.id] = result
        return result

    def eval_selector(sel: Selector) -> List[List[bool]]:
        if sel.id in selector_values:
            return selector_values[sel.id]

        if isinstance(sel, SelectOp):
            keys = eval_sop(sel.keys)
            queries = eval_sop(sel.queries)
            result = [
                [sel.predicate(keys[j], queries[i]) for j in range(seq_len)]
                for i in range(seq_len)
            ]

        elif isinstance(sel, ConstantSelector):
            result = [[sel.value] * seq_len for _ in range(seq_len)]

        elif isinstance(sel, SelectorAnd):
            left = eval_selector(sel.left)
            right = eval_selector(sel.right)
            result = [
                [left[i][j] and right[i][j] for j in range(seq_len)]
                for i in range(seq_len)
            ]

        elif isinstance(sel, SelectorOr):
            left = eval_selector(sel.left)
            right = eval_selector(sel.right)
            result = [
                [left[i][j] or right[i][j] for j in range(seq_len)]
                for i in range(seq_len)
            ]

        elif isinstance(sel, SelectorNot):
            inner = eval_selector(sel.inner)
            result = [
                [not inner[i][j] for j in range(seq_len)]
                for i in range(seq_len)
            ]

        else:
            raise ValueError(f"Unknown Selector type: {type(sel)}")

        selector_values[sel.id] = result
        return result

    # Evaluate all outputs
    outputs = {}
    for output_sop in program.outputs:
        outputs[output_sop.name] = eval_sop(output_sop)

    return outputs


# =============================================================================
# TESTING
# =============================================================================

def test_rasp():
    """Test RASP primitives and interpreter."""
    print("Testing RASP Language")
    print("=" * 50)

    # Test 1: Length
    print("\n1. Length program:")
    prog = make_length_program()
    result = interpret(prog, [1, 2, 3, 4, 5])
    print(f"   Input: [1, 2, 3, 4, 5]")
    print(f"   Length at each position: {result['length']}")
    assert result['length'] == [5, 5, 5, 5, 5]

    # Test 2: Histogram
    print("\n2. Histogram program:")
    prog = make_histogram_program()
    result = interpret(prog, list("aabbc"))
    print(f"   Input: 'aabbc'")
    print(f"   Counts: {result['count']}")
    assert result['count'] == [2, 2, 2, 2, 1]

    # Test 3: Simple arithmetic
    print("\n3. Arithmetic (indices * 2 + 1):")
    double_plus_one = indices * 2 + 1
    prog = RASPProgram([double_plus_one], list(range(5)), 8, "arith")
    result = interpret(prog, [0, 1, 2, 3, 4])
    print(f"   Input indices: [0, 1, 2, 3, 4]")
    print(f"   indices * 2 + 1: {result[double_plus_one.name]}")

    # Test 4: Select and aggregate
    print("\n4. Prefix sum:")
    prefix_sel = select(indices, indices, lambda k, q: k <= q)
    prefix_sum = aggregate(prefix_sel, indices + 1)  # sum of 1,2,3,...
    prog = RASPProgram([prefix_sum], list(range(5)), 8, "prefix")
    result = interpret(prog, [0, 1, 2, 3, 4])
    print(f"   Prefix sums (of 1,2,3,4,5): {result[prefix_sum.name]}")
    # Expected: [1, 1.5, 2, 2.5, 3] (averages, not sums, because aggregate averages)

    print("\n" + "=" * 50)
    print("All RASP tests passed!")
    return True


def test_c4_operations():
    """Test C4-compatible operations."""
    print("\nTesting C4-Compatible Operations")
    print("=" * 50)

    # Helper to run a simple program
    def run_op(sop, input_vals):
        prog = RASPProgram([sop], list(range(10)), 16, "test")
        result = interpret(prog, input_vals)
        return result[sop.name]

    # Test 1: Modulo
    print("\n1. Modulo (%):")
    mod_op = indices % 3
    result = run_op(mod_op, [0, 1, 2, 3, 4, 5, 6])
    print(f"   [0,1,2,3,4,5,6] % 3 = {result}")
    assert result == [0, 1, 2, 0, 1, 2, 0], f"Expected [0,1,2,0,1,2,0], got {result}"
    print("   ✓ Modulo passed")

    # Test 2: Floor division
    print("\n2. Floor Division (//):")
    floordiv_op = indices // 2
    result = run_op(floordiv_op, [0, 1, 2, 3, 4, 5, 6])
    print(f"   [0,1,2,3,4,5,6] // 2 = {result}")
    assert result == [0, 0, 1, 1, 2, 2, 3], f"Expected [0,0,1,1,2,2,3], got {result}"
    print("   ✓ Floor division passed")

    # Test 3: Left shift
    print("\n3. Left Shift (<<):")
    lshift_op = (indices + 1) << 2  # (x+1) * 4
    result = run_op(lshift_op, [0, 1, 2, 3])
    print(f"   [1,2,3,4] << 2 = {result}")
    assert result == [4, 8, 12, 16], f"Expected [4,8,12,16], got {result}"
    print("   ✓ Left shift passed")

    # Test 4: Right shift
    print("\n4. Right Shift (>>):")
    # Use tokens directly for specific values
    tok = tokens
    rshift_op = tok >> 1  # x // 2
    prog = RASPProgram([rshift_op], list(range(20)), 16, "test")
    result = interpret(prog, [8, 16, 32, 7, 15])
    result = result[rshift_op.name]
    print(f"   [8,16,32,7,15] >> 1 = {result}")
    assert result == [4, 8, 16, 3, 7], f"Expected [4,8,16,3,7], got {result}"
    print("   ✓ Right shift passed")

    # Test 5: Increment/Decrement
    print("\n5. Increment/Decrement:")
    inc_op = indices.increment()
    dec_op = indices.decrement()
    inc_result = run_op(inc_op, [0, 1, 2, 3, 4])
    dec_result = run_op(dec_op, [0, 1, 2, 3, 4])
    print(f"   [0,1,2,3,4]++ = {inc_result}")
    print(f"   [0,1,2,3,4]-- = {dec_result}")
    assert inc_result == [1, 2, 3, 4, 5], f"Expected [1,2,3,4,5], got {inc_result}"
    assert dec_result == [-1, 0, 1, 2, 3], f"Expected [-1,0,1,2,3], got {dec_result}"
    print("   ✓ Increment/decrement passed")

    # Test 6: Unary negation
    print("\n6. Unary Negation (-):")
    neg_op = -indices
    result = run_op(neg_op, [0, 1, 2, 3, 4])
    print(f"   -[0,1,2,3,4] = {result}")
    assert result == [0, -1, -2, -3, -4], f"Expected [0,-1,-2,-3,-4], got {result}"
    print("   ✓ Unary negation passed")

    # Test 7: Ternary operator
    print("\n7. Ternary Operator (?:):")
    # condition: indices > 2, true: 100, false: 0
    cond = indices > 2
    tern_op = ternary(cond, 100, 0)
    result = run_op(tern_op, [0, 1, 2, 3, 4])
    print(f"   (i > 2) ? 100 : 0 for i in [0,1,2,3,4] = {result}")
    assert result == [0, 0, 0, 100, 100], f"Expected [0,0,0,100,100], got {result}"
    print("   ✓ Ternary operator passed")

    # Test 8: sizeof
    print("\n8. sizeof:")
    char_size = c_sizeof('char')
    int_size = c_sizeof('int')
    ptr_size = c_sizeof('ptr')
    prog = RASPProgram([char_size, int_size, ptr_size], [0], 4, "sizeof")
    result = interpret(prog, [0])
    print(f"   sizeof(char) = {result[char_size.name][0]}")
    print(f"   sizeof(int) = {result[int_size.name][0]}")
    print(f"   sizeof(ptr) = {result[ptr_size.name][0]}")
    assert result[char_size.name][0] == 1
    assert result[int_size.name][0] == 8
    assert result[ptr_size.name][0] == 8
    print("   ✓ sizeof passed")

    # Test 9: Logical NOT
    print("\n9. Logical NOT (!):")
    lnot_op = logical_not(indices)
    result = run_op(lnot_op, [0, 1, 2, 3, 4])
    print(f"   ![0,1,2,3,4] = {result}")
    assert result == [1, 0, 0, 0, 0], f"Expected [1,0,0,0,0], got {result}"
    print("   ✓ Logical NOT passed")

    # Test 10: abs, min, max, clamp
    print("\n10. abs, min, max, clamp:")
    # Test with tokens for negative values
    abs_op = abs_val(tokens)
    prog = RASPProgram([abs_op], list(range(-10, 10)), 16, "abs")
    result = interpret(prog, [-3, -1, 0, 2, 5])
    print(f"   abs([-3,-1,0,2,5]) = {result[abs_op.name]}")
    assert result[abs_op.name] == [3, 1, 0, 2, 5], f"Got {result[abs_op.name]}"

    min_op = min_val(tokens, 2)
    prog = RASPProgram([min_op], list(range(10)), 16, "min")
    result = interpret(prog, [0, 1, 3, 5])
    print(f"   min([0,1,3,5], 2) = {result[min_op.name]}")
    assert result[min_op.name] == [0, 1, 2, 2], f"Got {result[min_op.name]}"

    max_op = max_val(tokens, 2)
    prog = RASPProgram([max_op], list(range(10)), 16, "max")
    result = interpret(prog, [0, 1, 3, 5])
    print(f"   max([0,1,3,5], 2) = {result[max_op.name]}")
    assert result[max_op.name] == [2, 2, 3, 5], f"Got {result[max_op.name]}"

    clamp_op = clamp(tokens, 1, 4)
    prog = RASPProgram([clamp_op], list(range(10)), 16, "clamp")
    result = interpret(prog, [0, 2, 5, 3])
    print(f"   clamp([0,2,5,3], 1, 4) = {result[clamp_op.name]}")
    assert result[clamp_op.name] == [1, 2, 4, 3], f"Got {result[clamp_op.name]}"
    print("   ✓ abs/min/max/clamp passed")

    # Test 11: Combined operations (simulating C expressions)
    print("\n11. Combined C expressions:")
    # (a % 4) << 1 + 1
    combined = ((indices % 4) << 1) + 1
    result = run_op(combined, [0, 1, 2, 3, 4, 5, 6, 7])
    print(f"   ((i % 4) << 1) + 1 for i in [0..7] = {result}")
    expected = [1, 3, 5, 7, 1, 3, 5, 7]  # (i%4)*2 + 1
    assert result == expected, f"Expected {expected}, got {result}"
    print("   ✓ Combined expressions passed")

    # Test 12: Division edge cases
    print("\n12. Division edge cases:")
    # Division by varying amounts
    div_op = tokens // (indices + 1)  # token[i] / (i+1)
    prog = RASPProgram([div_op], list(range(20)), 16, "div")
    result = interpret(prog, [10, 10, 10, 10])  # 10/1, 10/2, 10/3, 10/4
    print(f"   10 // [1,2,3,4] = {result[div_op.name]}")
    assert result[div_op.name] == [10, 5, 3, 2], f"Got {result[div_op.name]}"
    print("   ✓ Division edge cases passed")

    # Test 13: Bitwise AND
    print("\n13. Bitwise AND (via bit decomposition):")
    and_op = bitwise_and(tokens, ConstantSOp(0b1010), num_bits=4)  # AND with 10
    prog = RASPProgram([and_op], list(range(16)), 16, "bitand")
    # 0b1111 & 0b1010 = 0b1010 = 10
    # 0b1100 & 0b1010 = 0b1000 = 8
    # 0b0101 & 0b1010 = 0b0000 = 0
    result = interpret(prog, [15, 12, 5, 10])
    print(f"   [15,12,5,10] & 10 = {result[and_op.name]}")
    expected_and = [15 & 10, 12 & 10, 5 & 10, 10 & 10]
    assert result[and_op.name] == expected_and, f"Expected {expected_and}, got {result[and_op.name]}"
    print("   ✓ Bitwise AND passed")

    # Test 14: Bitwise OR
    print("\n14. Bitwise OR (via bit decomposition):")
    or_op = bitwise_or(tokens, ConstantSOp(0b0101), num_bits=4)  # OR with 5
    prog = RASPProgram([or_op], list(range(16)), 16, "bitor")
    result = interpret(prog, [0, 2, 8, 10])
    print(f"   [0,2,8,10] | 5 = {result[or_op.name]}")
    expected_or = [0 | 5, 2 | 5, 8 | 5, 10 | 5]
    assert result[or_op.name] == expected_or, f"Expected {expected_or}, got {result[or_op.name]}"
    print("   ✓ Bitwise OR passed")

    # Test 15: Bitwise XOR
    print("\n15. Bitwise XOR (via bit decomposition):")
    xor_op = bitwise_xor(tokens, ConstantSOp(0b1111), num_bits=4)  # XOR with 15
    prog = RASPProgram([xor_op], list(range(16)), 16, "bitxor")
    result = interpret(prog, [0, 5, 10, 15])
    print(f"   [0,5,10,15] ^ 15 = {result[xor_op.name]}")
    expected_xor = [0 ^ 15, 5 ^ 15, 10 ^ 15, 15 ^ 15]
    assert result[xor_op.name] == expected_xor, f"Expected {expected_xor}, got {result[xor_op.name]}"
    print("   ✓ Bitwise XOR passed")

    # Test 16: Bitwise NOT
    print("\n16. Bitwise NOT (via bit decomposition):")
    not_op = bitwise_not(tokens, num_bits=4)  # NOT with 4-bit width
    prog = RASPProgram([not_op], list(range(16)), 16, "bitnot")
    result = interpret(prog, [0, 5, 10, 15])
    print(f"   ~[0,5,10,15] (4-bit) = {result[not_op.name]}")
    # 4-bit NOT: ~0 = 15, ~5 = 10, ~10 = 5, ~15 = 0
    expected_not = [15, 10, 5, 0]
    assert result[not_op.name] == expected_not, f"Expected {expected_not}, got {result[not_op.name]}"
    print("   ✓ Bitwise NOT passed")

    # Test 17: Population count (popcount)
    print("\n17. Population count (count 1 bits):")
    pop_op = count_ones(tokens, num_bits=4)
    prog = RASPProgram([pop_op], list(range(16)), 16, "popcount")
    result = interpret(prog, [0, 1, 3, 7, 15])
    print(f"   popcount([0,1,3,7,15]) = {result[pop_op.name]}")
    expected_pop = [0, 1, 2, 3, 4]  # number of 1 bits
    assert result[pop_op.name] == expected_pop, f"Expected {expected_pop}, got {result[pop_op.name]}"
    print("   ✓ Population count passed")

    # Test 18: Bit extraction
    print("\n18. Bit extraction (get individual bits):")
    bit0_op = get_bit(tokens, 0)  # LSB
    bit1_op = get_bit(tokens, 1)
    bit2_op = get_bit(tokens, 2)
    bit3_op = get_bit(tokens, 3)  # MSB for 4-bit
    prog = RASPProgram([bit0_op, bit1_op, bit2_op, bit3_op], list(range(16)), 16, "getbit")
    result = interpret(prog, [0b1010, 0b0101, 0b1111])  # 10, 5, 15
    print(f"   Bits of 10 (0b1010): [{result[bit3_op.name][0]},{result[bit2_op.name][0]},{result[bit1_op.name][0]},{result[bit0_op.name][0]}]")
    print(f"   Bits of 5  (0b0101): [{result[bit3_op.name][1]},{result[bit2_op.name][1]},{result[bit1_op.name][1]},{result[bit0_op.name][1]}]")
    print(f"   Bits of 15 (0b1111): [{result[bit3_op.name][2]},{result[bit2_op.name][2]},{result[bit1_op.name][2]},{result[bit0_op.name][2]}]")
    # 10 = 0b1010: bits = [0,1,0,1]
    assert result[bit0_op.name][0] == 0
    assert result[bit1_op.name][0] == 1
    assert result[bit2_op.name][0] == 0
    assert result[bit3_op.name][0] == 1
    print("   ✓ Bit extraction passed")

    # Test 19: Sigmoid operation
    print("\n19. Sigmoid (softmax1 single element):")
    sig_op = sigmoid(indices - 2, beta=5.0)  # sigmoid(5*(i-2))
    result = run_op(sig_op, [0, 1, 2, 3, 4])
    print(f"   sigmoid(5*(i-2)) for i in [0..4]: {[f'{v:.3f}' for v in result]}")
    # At i=2: sigmoid(0) = 0.5
    # At i=4: sigmoid(10) ≈ 1
    # At i=0: sigmoid(-10) ≈ 0
    assert abs(result[2] - 0.5) < 0.01, f"Expected ~0.5 at i=2, got {result[2]}"
    assert result[4] > 0.99, f"Expected ~1 at i=4, got {result[4]}"
    assert result[0] < 0.01, f"Expected ~0 at i=0, got {result[0]}"
    print("   ✓ Sigmoid passed")

    # Test 20: Soft greater-than
    print("\n20. Soft greater-than (via sigmoid):")
    soft_gt_op = soft_gt(indices, 2, beta=10.0)  # i > 2 ?
    result = run_op(soft_gt_op, [0, 1, 2, 3, 4])
    print(f"   soft_gt(i, 2) for i in [0..4]: {[f'{v:.3f}' for v in result]}")
    # i=0,1,2 should be ~0; i=3,4 should be ~1
    assert result[0] < 0.1 and result[1] < 0.1
    assert result[3] > 0.9 and result[4] > 0.9
    print("   ✓ Soft greater-than passed")

    # Test 21: Soft bit extraction (via sigmoid threshold)
    print("\n21. Soft bit extraction (sigmoid threshold):")
    soft_bit0 = soft_bit_extract(tokens, 0, beta=100.0)
    soft_bit1 = soft_bit_extract(tokens, 1, beta=100.0)
    prog = RASPProgram([soft_bit0, soft_bit1], list(range(16)), 16, "softbit")
    result = interpret(prog, [0, 1, 2, 3, 4, 5, 6, 7])
    bit0_results = [round(v) for v in result[soft_bit0.name]]
    bit1_results = [round(v) for v in result[soft_bit1.name]]
    print(f"   soft_bit0([0..7]): {bit0_results}")
    print(f"   soft_bit1([0..7]): {bit1_results}")
    expected_bit0 = [0, 1, 0, 1, 0, 1, 0, 1]
    expected_bit1 = [0, 0, 1, 1, 0, 0, 1, 1]
    assert bit0_results == expected_bit0, f"Expected {expected_bit0}, got {bit0_results}"
    assert bit1_results == expected_bit1, f"Expected {expected_bit1}, got {bit1_results}"
    print("   ✓ Soft bit extraction passed")

    print("\n" + "=" * 50)
    print("All C4 operation tests passed!")
    return True


if __name__ == "__main__":
    test_rasp()
    test_c4_operations()
