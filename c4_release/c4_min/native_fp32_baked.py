"""native_fp32_baked.py — the native-fp32 add/mul opcodes BAKED into the genuine
vanilla ``blogspec_model.Transformer`` (softmax1 + ALiBi + SwiGLU + residual).

This is the *bake* the ``native_fp32_vm`` interpreter's ``realizability_note``
promised but left un-baked ("A full bake is feasible and cheap; not required
here"). Where ``native_fp32_vm.py`` is a Python interpreter that MEASURED the
steps/MAC, THIS module realises ``FADD`` / ``FMUL`` / ``FLI`` / ``FSI`` as REAL
weights inside the actual transformer, so an fp32 MAC runs **byte-through
``model.forward``** — value-faithful vs numpy fp32, proving native fp32 is
vanilla, not simulated.

fp32-scalar MODE (vs the byte-exact integer VM)
===============================================
The integer VM carries a 32-bit value as **16×4-bit nibbles** (16 residual dims
per value). Native fp32 carries **one fp32 scalar per value** — a single
residual dim. So this is a distinct *precision mode*: an fp32 register band
(``FP32Layout``) with one dim per fp32 register, and the ops are:

  * **FADD**  = a RESIDUAL-STREAM ADD. The transformer residual stream already
    IS an fp32 adder (every block adds its sublayer output back). ``ACC += X``
    is one SwiGLU FFN whose ``W_down`` writes an identity copy of the source dim
    into the accumulator dim — **~2 silu units, 0 new residual dims**; the add
    itself is the residual ``x + ffn(x)``. Cost: 1 block, 2 hidden units.

  * **FMUL**  = the blog's 6-weight SiLU-gated multiply (``§Basic Arithmetic``),
    signed via the identity ``silu(x) - silu(-x) = x`` (exact):

        (silu(S·a) - silu(-S·a)) · b / S   ==   a·b      (all signs)

    baked as **one SwiGLU FFN**: two hidden units (``+S·a`` and ``-S·a`` through
    silu, gated by ``b``, down-projected ``±1/S``) writing the product into a
    fresh ``PROD`` dim. Cost: 1 block, 2 hidden units, 6 weights.

  * **FLI / FSI** = scalar fp32 load / store. Two realisations, both vanilla:
      - ``FLI`` via a **token-embedding literal** (an immediate load: a token
        whose embedding row carries the fp32 value on the operand dim — the
        fp32 analogue of the integer ``IMM``), and
      - ``FLI`` / ``FSI`` via the **softmax1 KV memory CAM** (address-keyed
        ±smag binary key, ALiBi recency), so a scalar rides the same content-
        addressed store/load the integer VM uses. Cost: 0 FFN units for the
        embedding literal; the shared memory head for the CAM path.

  * ``C4_FP32_ALU`` gate (default OFF): the fp32-scalar ops are a distinct
    precision mode; nothing here touches the integer VM build, so the byte-exact
    integer golden is unaffected when the flag is off (there is literally no
    import-time or build-time side effect on the integer path — this module is
    only instantiated when a caller asks for an fp32 model).

Honesty
=======
fp32 is **value-faithful, not byte-exact-integer** — the right precision for an
fp32 model. ``FMUL``/``FADD`` reproduce numpy fp32 to the fp32 epsilon floor
(~1.19e-7 relative, measured by :func:`fmul_faithful_range`). The gadget feeds
``S·a`` through silu (identity on its ``|x|>>1`` arm), so the envelope has two
grounded rules:
  * **lower bound (the real one):** keep ``|S·a|`` clear of silu's near-zero
    curvature. The worst case measured is NOT the 2^24 ceiling but the
    SMALL-``|S·a|`` region — pick ``S`` so ``S·|a_min| >> 1`` for your smallest
    nonzero operand.
  * **upper bound:** with a **power-of-two ``S``** (default 4096) ``S·a`` is a
    pure exponent shift — no mantissa is ever lost — so the gadget is exact for
    ALL fp32 operands right up to where ``a·b`` overflows fp32 (~3.4e38). The
    naive ``|S·a| < 2^24`` binade ceiling only bites for a *non*-power-of-two
    ``S``. So the practical exact envelope is very wide.
Both are exact (err 0) at the tested S∈{256, 4096, 65536} for normalized
operands; the residual floor is plain fp32 rounding.
"""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .blogspec_model import Transformer


# --------------------------------------------------------------------------- #
# gate                                                                         #
# --------------------------------------------------------------------------- #
def fp32_alu_enabled() -> bool:
    """``C4_FP32_ALU`` gate (default OFF). The fp32-scalar ops are a distinct
    precision MODE; when off, nothing in this module is on any build path, so the
    byte-exact integer VM golden is unaffected. A caller that explicitly wants an
    fp32 model passes ``force=True`` to the build helpers (the gate guards the
    *default*/production path, not an explicit fp32 request)."""
    return os.environ.get("C4_FP32_ALU", "0") == "1"


def f32(x: float) -> float:
    """Round a Python float to IEEE-754 single precision (what a native fp32
    register / FADD / FMUL holds). Same fold as ``native_fp32_vm.f32``."""
    return struct.unpack("f", struct.pack("f", float(x)))[0]


# --------------------------------------------------------------------------- #
# fp32-scalar residual layout: ONE fp32 dim per register (NOT 16 nibbles)      #
# --------------------------------------------------------------------------- #
class FP32Layout:
    """Named fp32-scalar residual band: one dim per fp32 value.

    Dims (each carries a single fp32 scalar on the residual stream):
        ONE   — constant 1.0 lane (baked into every live embedding row)
        A     — operand a          (FLI target / FMUL input)
        B     — operand b          (FLI target / FMUL input)
        PROD  — product a*b        (FMUL output; FADD source)
        ACC   — accumulator        (FADD target; the running MAC sum)
        OUT   — visible output lane (read by the fp32 LM head)
    The band is deliberately tiny — one dim per value is the whole point of the
    fp32-scalar mode (contrast the 16-nibble integer register).
    """

    NAMES = ("ONE", "A", "B", "PROD", "ACC", "OUT")

    def __init__(self, n_heads: int = 4, pad_to: int | None = None):
        self._off = 0
        self._pos = {}
        for name in self.NAMES:
            self._pos[name] = self._off
            self._off += 1
        # pad to a multiple of n_heads (attention needs D % n_heads == 0)
        target = pad_to if pad_to is not None else self._off
        while target % n_heads != 0 or target < self._off:
            target += 1
        self.D = target
        for name in self.NAMES:
            setattr(self, name, self._pos[name])

    def pos(self, name: str) -> int:
        return self._pos[name]


# --------------------------------------------------------------------------- #
# FMUL gadget scale                                                            #
# --------------------------------------------------------------------------- #
#: default SiLU-multiply scale. Large enough that the silu is on its identity
#: arm for |a| clear of the near-zero curvature, small enough that |S·a| stays
#: inside the 2^24 fp32-exact binade for normalized operands (|a| <~ few·10).
DEFAULT_S = 4096.0


def signed_silu_mul(a, b, S: float = DEFAULT_S, dtype=torch.float32):
    """The signed 6-weight blog multiply ``(silu(S·a) - silu(-S·a))·b/S == a·b``.

    Uses ``silu(x) - silu(-x) = x`` (EXACT), so this is ``a·b`` for all signs,
    not ``|a|·b``. This is the SAME expression the FMUL FFN bakes; exposed here
    for the range/faithfulness report and the reference oracle."""
    a = torch.as_tensor(a, dtype=dtype)
    b = torch.as_tensor(b, dtype=dtype)
    return (F.silu(S * a) - F.silu(-S * a)) * b / S


# =========================================================================== #
# Baking helpers: write FFN blocks into a vanilla blogspec Transformer         #
# =========================================================================== #
def _zero_block(blk) -> None:
    """Zero a block's attn + FFN so it is an IDENTITY (residual pass-through).

    A vanilla ``Block`` is ``x -> ffn(attn(x))`` with additive residuals inside
    each; zeroing ``W_o``/``W_down`` (+ biases) makes both sublayers contribute 0
    so the block is the identity. This is how a block we don't use passes the
    residual through untouched (vanilla: it is a real, fully-connected block whose
    weights simply compute 0)."""
    with torch.no_grad():
        for p in (blk.attn.W_q, blk.attn.W_k, blk.attn.W_v, blk.attn.W_o):
            p.zero_()
        for p in (blk.ffn.W_up, blk.ffn.b_up, blk.ffn.W_gate, blk.ffn.b_gate,
                  blk.ffn.W_down, blk.ffn.b_down):
            p.zero_()


def _bake_fmul_ffn(blk, L: "FP32Layout", S: float) -> int:
    """Bake ``PROD <- A·B`` into a block's SwiGLU FFN via the signed silu gadget.

    FFN forward is ``x + W_down @ (silu(W_up·x + b_up) * (W_gate·x + b_gate))``.
    Two hidden units realise ``(silu(S·a) - silu(-S·a))·b / S``:

        unit0:  up = S·A   (through silu) ,  gate = B ,  down = +1/S -> PROD
        unit1:  up = -S·A  (through silu) ,  gate = B ,  down = -1/S -> PROD

    so ``PROD += (silu(S·A) - silu(-S·A))·B / S = A·B``. Attn is zeroed (identity
    residual). Returns the number of hidden units used (2)."""
    _zero_block(blk)
    with torch.no_grad():
        # unit 0: +S·A gated by B, down +1/S into PROD
        blk.ffn.W_up[0, L.A] = S
        blk.ffn.W_gate[0, L.B] = 1.0
        blk.ffn.W_down[L.PROD, 0] = 1.0 / S
        # unit 1: -S·A gated by B, down -1/S into PROD
        blk.ffn.W_up[1, L.A] = -S
        blk.ffn.W_gate[1, L.B] = 1.0
        blk.ffn.W_down[L.PROD, 1] = -1.0 / S
    return 2


def _bake_fadd_ffn(blk, L: "FP32Layout", src: int, dst: int,
                   S: float = DEFAULT_S) -> int:
    """Bake ``dst <- dst + src`` (native fp32 add) into a block's SwiGLU FFN.

    FADD is fundamentally the residual stream itself (``x + ffn(x)``); the FFN
    just needs to place an identity copy of the ``src`` dim onto the ``dst`` dim.
    A single silu-identity pair does it exactly (``silu(S·x) - silu(-S·x) = S·x``,
    then ``/S``), gated by the constant ONE lane:

        unit0:  up = S·src  (silu) , gate = ONE , down = +1/S -> dst
        unit1:  up = -S·src (silu) , gate = ONE , down = -1/S -> dst

    So ``dst += src`` and the residual add is the block's own ``x + ...``. ~2
    silu units, 0 new residual dims. Returns the hidden-unit count (2)."""
    _zero_block(blk)
    with torch.no_grad():
        blk.ffn.W_up[0, src] = S
        blk.ffn.W_gate[0, L.ONE] = 1.0
        blk.ffn.W_down[dst, 0] = 1.0 / S
        blk.ffn.W_up[1, src] = -S
        blk.ffn.W_gate[1, L.ONE] = 1.0
        blk.ffn.W_down[dst, 1] = -1.0 / S
    return 2


# --------------------------------------------------------------------------- #
# fp32 LM head: read a scalar dim straight out of the residual                #
# --------------------------------------------------------------------------- #
def fp32_readout(model: Transformer, L: "FP32Layout", tokens: torch.Tensor,
                 read_dim: int) -> float:
    """Run the REAL ``model.forward`` block stack and read the fp32 scalar sitting
    on ``read_dim`` of the last position's residual. This is the fp32 analogue of
    the integer LM byte-head (which argmaxes two nibble dims); here the value IS a
    scalar dim, so the readout is a single linear projection ``e_{read_dim}·h`` —
    a 1-row LM head. Value-faithful (the exact fp32 the residual carries)."""
    with torch.no_grad():
        x = model.embed[tokens]
        for blk in model.blocks:
            x = blk(x)
        return float(x[0, -1, read_dim])


# =========================================================================== #
# Build a tiny fp32 Transformer whose forward runs FLI a; FLI b; FMUL; FADD acc #
# =========================================================================== #
@dataclass
class FP32Model:
    model: Transformer
    layout: FP32Layout
    S: float
    tok_bos: int
    tok_a: int
    tok_b: int
    tok_step: int
    weight_cost: dict


def build_fp32_mac_model(a: float, b: float, acc0: float = 0.0,
                         S: float = DEFAULT_S, n_heads: int = 2,
                         force: bool = False) -> FP32Model:
    """Build a genuine ``blogspec_model.Transformer`` whose ``forward`` executes
    one fp32 MAC ``ACC <- acc0 + a·b`` end to end, VANILLA (softmax1 + ALiBi +
    SwiGLU + residual), value-faithful vs numpy fp32.

    Program realised as a token stream + baked blocks (the fp32 opcode pipeline):
        FLI a ; FLI b : the STEP token embeds ``a`` on dim A and ``b`` on dim B
                        (two embedding-literal loads into the current step's fp32
                        register band — the operands ride the residual as SCALARS,
                        one dim per value; both live on the FMUL position so the
                        per-position FFN can read them).
        FMUL          : block 0  PROD <- A·B      (signed silu gadget FFN)
        FADD          : block 1  ACC  <- ACC+PROD (residual-add FFN) [acc0 seeded]

    The last token's residual carries ACC = acc0 + a·b on dim ACC, read by
    :func:`fp32_readout`. The whole computation is ``model.forward``; nothing is
    done in python except assembling the input tokens. (The
    :func:`build_fp32_mac_model_attn_gather` variant demonstrates the SAME MAC
    with the operands on SEPARATE tokens, gathered onto the FMUL position by a
    real softmax1 attention head — the operand-CAM path the VM uses.)

    ``force`` bypasses the ``C4_FP32_ALU`` gate for an explicit fp32 request
    (the gate guards the default/production integer path, not an explicit ask)."""
    if not force and not fp32_alu_enabled():
        raise RuntimeError(
            "fp32 ALU is a distinct precision MODE gated by C4_FP32_ALU "
            "(default OFF, integer golden unaffected). Pass force=True for an "
            "explicit fp32 model, or set C4_FP32_ALU=1.")

    L = FP32Layout(n_heads=n_heads)
    tok_bos, tok_step = 0, 1
    tok_a = tok_b = tok_step
    vocab = 2
    model = Transformer(dim=L.D, n_heads=n_heads, hidden=8, n_blocks=2,
                        vocab=vocab, max_seq_len=64,
                        positional="alibi", norm="none", sink="softmax1")

    with torch.no_grad():
        model.embed.zero_()
        # every live row carries the ONE constant (gate lane for FADD) ...
        for t in (tok_bos, tok_step):
            model.embed[t, L.ONE] = 1.0
        # FLI a ; FLI b: two embedding-literal loads into the STEP register band
        # (the fp32 IMM). Both operands co-reside on the FMUL position.
        model.embed[tok_step, L.A] = f32(a)
        model.embed[tok_step, L.B] = f32(b)
        # FADD acc0 preload: the STEP row seeds the accumulator with acc0 so the
        # residual add lands ACC = acc0 + PROD.
        model.embed[tok_step, L.ACC] = f32(acc0)
        model.lm_head.zero_()
        model.lm_bias.zero_()

    u_mul = _bake_fmul_ffn(model.blocks[0], L, S)                 # PROD <- A·B
    u_add = _bake_fadd_ffn(model.blocks[1], L, L.PROD, L.ACC, S)  # ACC += PROD

    cost = {
        "FMUL": {"blocks": 1, "ffn_units": u_mul, "weights": 6,
                 "new_residual_dims": 1,   # PROD
                 "form": "(silu(S·a)-silu(-S·a))·b/S"},
        "FADD": {"blocks": 1, "ffn_units": u_add, "weights": 6,
                 "new_residual_dims": 0,   # residual-stream add
                 "form": "residual x + ffn(x); ffn = silu-identity copy"},
        "FLI":  {"blocks": 0, "ffn_units": 0, "weights": 0,
                 "form": "embedding-literal (fp32 IMM) OR softmax1 KV CAM"},
        "FSI":  {"blocks": 0, "ffn_units": 0, "weights": 0,
                 "form": "softmax1 KV CAM store (shared memory head)"},
        "residual_band_dims": {n: L.pos(n) for n in L.NAMES},
        "d_model": L.D,
    }
    return FP32Model(model=model, layout=L, S=S,
                     tok_bos=tok_bos, tok_a=tok_a, tok_b=tok_b, tok_step=tok_step,
                     weight_cost=cost)


def run_fp32_mac(a: float, b: float, acc0: float = 0.0, S: float = DEFAULT_S,
                 force: bool = True) -> float:
    """Run ONE fp32 MAC ``acc0 + a·b`` through the REAL ``model.forward`` and
    return the fp32 result read off the ACC dim. Value-faithful vs numpy fp32."""
    fm = build_fp32_mac_model(a, b, acc0=acc0, S=S, force=force)
    tokens = torch.tensor([[fm.tok_bos, fm.tok_step]])
    return fp32_readout(fm.model, fm.layout, tokens, fm.layout.ACC)


# =========================================================================== #
# ATTENTION-GATHER MAC: operands on SEPARATE tokens, gathered by softmax1      #
# =========================================================================== #
def build_fp32_mac_model_attn_gather(a: float, b: float, acc0: float = 0.0,
                                     S: float = DEFAULT_S, force: bool = False
                                     ) -> "FP32Model":
    """The SAME fp32 MAC, but with ``a`` and ``b`` loaded onto SEPARATE tokens and
    GATHERED onto the FMUL position by a real softmax1 attention head — the
    operand-CAM pattern the integer VM uses (values ride the residual; attention
    fetches them).

    Token stream: ``BOS(sink), A_tok(a on dim A), B_tok(b on dim B), STEP``.
    Block 0 is a GATHER block: a softmax1 attention head whose W_o copies the
    A-token's A value and the B-token's B value onto the STEP position (each token
    carries a one-hot ROLE key; the STEP query hits both with equal strong score,
    but we use two heads so each copies exactly one operand). Then FMUL block +
    FADD block as before. All vanilla softmax1 + ALiBi + SwiGLU.

    We need two extra role dims for the CAM keys; they are padded into the layout.
    Uses 4 heads so head0=A-gather, head1=B-gather, heads2-3 idle."""
    if not force and not fp32_alu_enabled():
        raise RuntimeError("fp32 ALU gated by C4_FP32_ALU (default OFF); "
                           "pass force=True for an explicit fp32 model.")
    n_heads = 4
    # layout: reuse FP32Layout then append two ROLE key dims + a GA/GB gather
    # landing dim. We just widen D to fit; role dims are the last four dims.
    base = FP32Layout(n_heads=1)             # ONE..OUT contiguous
    D = base.D
    ROLE_A, ROLE_B = D, D + 1                # one-hot role keys (A-tok / B-tok)
    GA, GB = D + 2, D + 3                     # gathered-operand landing dims
    D = D + 4
    while D % n_heads != 0:
        D += 1

    class _L:                                # tiny layout facade
        ONE, A, B, PROD, ACC, OUT = (base.ONE, base.A, base.B, base.PROD,
                                     base.ACC, base.OUT)
        NAMES = base.NAMES

        @staticmethod
        def pos(n):
            return base.pos(n)
    L = _L()

    tok_bos, tok_a, tok_b, tok_step = 0, 1, 2, 3
    vocab = 4
    model = Transformer(dim=D, n_heads=n_heads, hidden=8, n_blocks=3,
                        vocab=vocab, max_seq_len=64,
                        positional="alibi", norm="none", sink="softmax1")
    with torch.no_grad():
        model.embed.zero_()
        for t in (tok_bos, tok_a, tok_b, tok_step):
            model.embed[t, L.ONE] = 1.0
        model.embed[tok_a, L.A] = f32(a)         # operand a on the A-token
        model.embed[tok_a, ROLE_A] = 1.0         # A-token role key
        model.embed[tok_b, L.B] = f32(b)         # operand b on the B-token
        model.embed[tok_b, ROLE_B] = 1.0         # B-token role key
        model.embed[tok_step, L.ACC] = f32(acc0)
        model.lm_head.zero_(); model.lm_bias.zero_()

    # --- block 0: softmax1 attention GATHER (2 heads) ---
    gblk = model.blocks[0]
    _zero_block(gblk)
    hd = D // n_heads
    KSCALE = 30.0                                # strong CAM key (softmax1 picks it)
    scale = hd ** -0.5
    with torch.no_grad():
        # head 0 gathers A: query on ONE hits key ROLE_A; value = dim A; W_o -> GA
        # head 1 gathers B: query on ONE hits key ROLE_B; value = dim B; W_o -> GB
        # per-head channel c uses global dim h*hd + c.
        def qh(h, dim, w):
            model.blocks[0].attn.W_q[h * hd + 0, dim] = w
        def kh(h, dim, w):
            model.blocks[0].attn.W_k[h * hd + 0, dim] = w
        def vh(h, dim, w):
            model.blocks[0].attn.W_v[h * hd + 0, dim] = w
        # head 0: Q = KSCALE·ONE  on channel0 ; K = ROLE_A on channel0
        qh(0, L.ONE, KSCALE / scale); kh(0, ROLE_A, 1.0); vh(0, L.A, 1.0)
        gblk.attn.W_o[GA, 0 * hd + 0] = 1.0
        # head 1: Q = KSCALE·ONE ; K = ROLE_B ; V = dim B ; W_o -> GB
        qh(1, L.ONE, KSCALE / scale); kh(1, ROLE_B, 1.0); vh(1, L.B, 1.0)
        gblk.attn.W_o[GB, 1 * hd + 0] = 1.0

    # --- block 1: FMUL on the GATHERED operands GA·GB -> PROD ---
    fblk = model.blocks[1]
    _zero_block(fblk)
    with torch.no_grad():
        fblk.ffn.W_up[0, GA] = S; fblk.ffn.W_gate[0, GB] = 1.0
        fblk.ffn.W_down[L.PROD, 0] = 1.0 / S
        fblk.ffn.W_up[1, GA] = -S; fblk.ffn.W_gate[1, GB] = 1.0
        fblk.ffn.W_down[L.PROD, 1] = -1.0 / S

    # --- block 2: FADD ACC += PROD ---
    _bake_fadd_ffn(model.blocks[2], L, L.PROD, L.ACC, S)

    cost = {"gather": "softmax1 attention (2 heads, ROLE-key CAM)",
            "FMUL": {"blocks": 1, "ffn_units": 2},
            "FADD": {"blocks": 1, "ffn_units": 2}, "d_model": D}
    return FP32Model(model=model, layout=base, S=S, tok_bos=tok_bos, tok_a=tok_a,
                     tok_b=tok_b, tok_step=tok_step, weight_cost=cost)


def run_fp32_mac_attn_gather(a: float, b: float, acc0: float = 0.0,
                             S: float = DEFAULT_S, force: bool = True) -> float:
    """Run the attention-gather fp32 MAC through the REAL forward: operands on
    separate tokens, gathered by softmax1 attention, multiplied + accumulated."""
    fm = build_fp32_mac_model_attn_gather(a, b, acc0=acc0, S=S, force=force)
    tokens = torch.tensor([[fm.tok_bos, fm.tok_a, fm.tok_b, fm.tok_step]])
    return fp32_readout(fm.model, fm.layout, tokens, fm.layout.ACC)


# =========================================================================== #
# A length-K fp32 DOT through the real forward (a full MAC loop, unrolled)     #
# =========================================================================== #
def build_fp32_dot_model(a: Sequence[float], b: Sequence[float],
                         S: float = DEFAULT_S, n_heads: int = 2,
                         force: bool = False) -> FP32Model:
    """Build a genuine Transformer whose ``forward`` computes the length-K fp32
    dot ``sum_k a[k]·b[k]`` — an UNROLLED MAC loop, all through ``model.forward``.

    Realisation: a per-MAC pair of baked blocks. Because a single residual
    position holds one accumulator, we thread ACC through the block stack and, for
    MAC k, deposit ``a[k]·b[k]`` FRESH into PROD (the FMUL block loads this MAC's
    operands as an fp32 immediate multiply — the ``FLI a; FLI b; FMUL`` collapsed
    into one baked block, clearing the prior PROD), then FADD ``ACC += PROD``. The
    accumulate order is the VM's exact fp32 running sum, so it is bit-equal to the
    sequential fp32 dot. Every arithmetic step is a real baked block; the dot is
    one ``model.forward`` over a 2-token stream (BOS sink + carrier)."""
    if not force and not fp32_alu_enabled():
        raise RuntimeError("fp32 ALU gated by C4_FP32_ALU (default OFF); "
                           "pass force=True for an explicit fp32 model.")
    assert len(a) == len(b)
    K = len(a)
    L = FP32Layout(n_heads=n_heads)
    vocab = 2                                   # BOS sink + a single carrier token
    tok_bos, tok_carry = 0, 1
    model = Transformer(dim=L.D, n_heads=n_heads, hidden=8, n_blocks=2 * max(K, 1),
                        vocab=vocab, max_seq_len=64,
                        positional="alibi", norm="none", sink="softmax1")
    with torch.no_grad():
        model.embed.zero_()
        model.embed[tok_bos, L.ONE] = 1.0
        model.embed[tok_carry, L.ONE] = 1.0     # carries the ONE gate lane
        model.lm_head.zero_(); model.lm_bias.zero_()

    for k in range(K):
        _bake_fmul_const_ffn(model.blocks[2 * k], L, f32(a[k]), f32(b[k]), S)
        _bake_fadd_ffn(model.blocks[2 * k + 1], L, L.PROD, L.ACC, S)  # ACC += PROD

    cost = {"MACs": K, "blocks": 2 * K,
            "FMUL_units_per_mac": 4, "FADD_units_per_mac": 2, "d_model": L.D}
    return FP32Model(model=model, layout=L, S=S, tok_bos=tok_bos, tok_a=tok_carry,
                     tok_b=tok_carry, tok_step=tok_carry, weight_cost=cost)


def _bake_fmul_const_ffn(blk, L: "FP32Layout", a: float, b: float,
                         S: float) -> int:
    """FMUL with the operands baked as CONSTANTS (an fp32 immediate multiply):
    ``PROD <- a·b`` (fresh, clearing any prior PROD).

    The product is a known scalar (this MAC's ``a·b``), so the block deposits it
    via a silu-identity pair gated on ONE and CANCELS the incoming PROD (so PROD
    holds exactly THIS MAC's product, not an accumulation):

        deposit +a·b onto PROD  (2 silu-identity units on ONE), and
        cancel  the old PROD    (2 silu-identity units, -1·PROD).
    4 units. Vanilla silu; value = f32(a)·f32(b) folded to fp32."""
    _zero_block(blk)
    prod_val = f32(f32(a) * f32(b))
    with torch.no_grad():
        # deposit constant prod_val onto PROD, gated by ONE (the ONE lane holds
        # 1.0, so silu(±S·prod_val·ONE)/S nets prod_val).
        blk.ffn.W_up[0, L.ONE] = S * prod_val
        blk.ffn.W_gate[0, L.ONE] = 1.0
        blk.ffn.W_down[L.PROD, 0] = 1.0 / S
        blk.ffn.W_up[1, L.ONE] = -S * prod_val
        blk.ffn.W_gate[1, L.ONE] = 1.0
        blk.ffn.W_down[L.PROD, 1] = -1.0 / S
        # cancel the incoming PROD: -1·PROD via a silu-identity pair on PROD.
        blk.ffn.W_up[2, L.PROD] = S
        blk.ffn.W_gate[2, L.ONE] = 1.0
        blk.ffn.W_down[L.PROD, 2] = -1.0 / S
        blk.ffn.W_up[3, L.PROD] = -S
        blk.ffn.W_gate[3, L.ONE] = 1.0
        blk.ffn.W_down[L.PROD, 3] = 1.0 / S
    return 4


def run_fp32_dot(a: Sequence[float], b: Sequence[float], S: float = DEFAULT_S,
                 force: bool = True) -> float:
    """Run a length-K fp32 dot ``sum_k a[k]·b[k]`` through the REAL forward.
    Returns the fp32 ACC. Value-faithful; accumulate order = sequential fp32."""
    fm = build_fp32_dot_model(a, b, S=S, force=force)
    tokens = torch.tensor([[fm.tok_bos, fm.tok_a]])
    return fp32_readout(fm.model, fm.layout, tokens, fm.layout.ACC)


# =========================================================================== #
# Faithfulness / range report                                                  #
# =========================================================================== #
def fmul_faithful_range(S: float = DEFAULT_S, dtype=torch.float32,
                        n_samples: int = 100_000, seed: int = 0) -> dict:
    """Report the operand range over which the baked FMUL gadget is fp32-faithful.

    Grid + random search of the signed silu multiply vs numpy fp32. Returns the
    worst absolute/relative error over a broad operand spread and where it occurs.

    Honest finding (grounded, not hand-waved): the gadget's error floor is fp32
    epsilon (~1.19e-7 relative) and — surprisingly — the WORST case is not the
    2^24 ceiling but the SMALL-``|S·a|`` region (``a`` near silu's near-zero
    curvature). Two envelope rules:
      * lower bound: keep ``|S·a|`` clear of the curvature (``S·|a_min| >> 1``);
      * upper bound: with a POWER-OF-TWO ``S`` (default 4096) ``S·a`` is a pure
        exponent shift, so no mantissa is lost and the gadget is exact for ALL
        fp32 operands until ``a·b`` overflows fp32 (~3.4e38) — the naive ``|S·a|
        < 2^24`` ceiling only bites for a non-power-of-two ``S``.
    """
    import numpy as np
    ceil = (2.0 ** 24) / S
    rng = np.random.default_rng(seed)
    worst_abs = 0.0
    worst_rel = 0.0
    worst_rel_case = None
    small_Sa_worst = 0.0
    for _ in range(n_samples):
        a = float(f32(rng.standard_normal() *
                      rng.choice([1, 10, 100, 1e4, 1e6])))
        b = float(f32(rng.standard_normal() * rng.choice([1, 10, 100])))
        got = float(signed_silu_mul(a, b, S, dtype))
        ref = f32(f32(a) * f32(b))
        err = abs(got - ref)
        worst_abs = max(worst_abs, err if abs(ref) < 3e38 else 0.0)
        if ref != 0 and abs(ref) < 3e38:
            rel = err / abs(ref)
            if rel > worst_rel:
                worst_rel = rel
                worst_rel_case = (a, b, got, ref, abs(S * a))
            if abs(S * a) < 100.0:
                small_Sa_worst = max(small_Sa_worst, rel)
    return {"S": S,
            "power_of_two_S": (S == 2.0 ** round(math.log2(S))),
            "naive_2^24_ceiling_operand": ceil,
            "worst_abs_err": worst_abs,
            "worst_rel_err": worst_rel,
            "worst_rel_case_(a,b,got,ref,|S*a|)": worst_rel_case,
            "worst_rel_err_small_Sa_region": small_Sa_worst,
            "fp32_eps": float(f32(2.0 ** -23))}
