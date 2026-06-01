"""
Chunk-generic MOD pipeline: nibble-level long division (no fp64).

MOD shares the long-division core with DIV: 8 outer iterations produce both
quotient and remainder simultaneously. The MOD entry point emits the
remainder; the DIV entry point emits the quotient. See
``divmod_longdiv.py`` for algorithm details.

Pipeline:
  Layer 1: ClearDivSlotsFFN -- clear scratch slots
  Layer 2: LongDivisionModule -- long division -> SLOT_REMAINDER, SLOT_QUOTIENT
  Layer 3: EmitDivResultModule -- copy SLOT_REMAINDER[*] -> RESULT[*]

FIXME(investigation/expr-mod-divergences, 2026-05-31): All 25 ``expr_mod_N``
tests (IDs 875-899) in the 1096 declarative diagnostic suite return wrong
``neural`` final values under ``C4_DECLARATIONS_ONLY_BAKE=1``. The simpler
``mod_N`` tests (IDs 200-249, ``int main() { return a % b; }``) also fail
value-dependently: e.g. mod_0 (390%19=10) -> neural=1, mod_1 (89%10=9) ->
neural=7, mod_3 (377%12=5) -> neural=9, but mod_2 (154%8=2) -> neural=2
(correct).

Key findings:
  * ``expr_mul_div_N`` (DIV path) passes 2/2 (offset 850, declarations-only).
  * ``test_smoke.TestSmokeBasic::test_mod_basic`` (43%10=3) passes under
    default flags.
  * MOD and DIV share the same ``FlattenedDivMod`` composite; only the
    ``emit_remainder`` flag and opcode-merge step differ.
  * The reported ``first_token_divergence=step0:SP_byte0`` (expected 0x00,
    neural 0xf8) is infrastructure noise: it occurs on every declarations-
    only test because the symbolic-execution SP convention differs from
    the L3-baked initial-SP-marker rule (L3 emits 0xf8; symbolic emits
    0x00). The semantic break is the wrong final value, not that token.
  * ``block27 layer=27`` is the lowering audit's terminal report; the
    actual MOD compute divergence happens earlier in the pipeline but is
    masked by the universal SP_byte0 first-divergence trace.
  * mod_2 (b=8 power of 2, remainder fits in low nibble) producing the
    correct answer suggests the long-division compute itself works for
    trivial cases; failure mode is value-dependent -- possibly in the
    OUTPUT_LO/HI projection of the multi-nibble remainder back through
    L11-L17 when the remainder has a high nibble or competes with stale
    quotient state.

Next steps (not in this commit):
  1. Capture intermediate residual state at the MOD post-op output for
     mod_0 (390%19=10) and mod_2 (154%8=2) and diff the residual to
     pinpoint whether SLOT_REMAINDER is computed correctly inside the
     long-division module.
  2. If SLOT_REMAINDER is correct, walk forward through L11-L15 to find
     the layer that scrambles the multi-nibble remainder into the wrong
     OUTPUT_LO/HI bytes.
  3. Compare with DIV in the same regime (e.g. div_0 from offset 150)
     under declarations-only mode to confirm DIV vs MOD asymmetry vs
     value-dependent vs systemic.
"""

from torch import nn

from ..chunk_config import ChunkConfig


def build_mod_layers(config: ChunkConfig, opcode: int = 29) -> nn.ModuleList:
    """Build MOD pipeline using nibble-level long division (no fp64).

    All arithmetic is fp32. See ``divmod_longdiv.py``.
    """
    from .divmod_longdiv import build_mod_layers_longdiv
    return build_mod_layers_longdiv(config, opcode)
