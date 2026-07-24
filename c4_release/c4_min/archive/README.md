# Archived c4_min modules

These are **superseded bakeoff losers** — experimental ALU-gadget variants that
competed in a bake-off and lost. Their winners are live in `c4_min/`. Kept here for
reference (reversible `git mv`); nothing in `c4_min/` or the test suite imports them.

Archived 2026-07-23. To restore one: `git mv c4_min/archive/<name>.py c4_min/<name>.py`.

| Archived module | Superseded by (live) |
|---|---|
| `div_radix16_attn.py` | `c4_min/div_radix16_hardened.py` |
| `div_radix16_lean.py` | `c4_min/div_radix16_hardened.py` |
| `mul_bakeoff.py` | `c4_min/mul_lookahead.py` |
| `mul_bakeoff_11bit.py` | `c4_min/mul_lookahead.py` |
| `mul_byte_accum.py` | `c4_min/mul_lookahead.py` |
| `mul_byte_carrysave.py` | `c4_min/mul_lookahead.py` |

Verified zero importers (production + tests) and not wired by the #735 integration
branch before archiving. `shift_tight_nibble.py` was **excluded** — it is LIVE
(imported by `nibble_bitwise.py`).
