# Unified per-op ALU unit registry (`alu_units.py`)

ONE interface to select the **implementation unit** for ADD / MUL / DIV in the fused
Qwen VM, exposing the **depth ↔ weight** tradeoff so the arithmetic can be *fit to a
target model's layer / width budget*.

Code: [`alu_units.py`](alu_units.py) · tests: [`test_alu_units.py`](test_alu_units.py)
· wired non-invasively into [`qwen_full_vm.build`](qwen_full_vm.py) (a single
`apply_to_env()` projection).

---

## 1. What existed before (the scattered switches)

Each op family had grown its own selection switch, in different places and forms:

| op | switch | form | where read | selects |
|----|--------|------|-----------|---------|
| DIV/MOD | `C4_DIV_LONGDIV` | env (default `0`) | `qwen_full_vm._div_longdiv` | radix-16 digit-recurrence (off) vs base-16 long division ~262 blocks (on) |
| DIV/MOD | `C4_DIV_LEAN` | env (default `1`) | `qwen_full_vm._div_lean` | radix-16 **lean** 80-block (on) vs **hardened** 88-block (off) |
| DIV/MOD | `recurrent_divmod=` | kwarg (default `False`) | `qwen_full_vm.build` / `_block_specs` | unroll 8 iters (80 stored) vs fold to ONE reused iteration body (~17 stored, applied 8×) |
| DIV/MOD | `C4_KB_BATCHED` | env (default `1`) | `nibble_alu32.kb_batched` | KB-precompute: batched Kogge-Stone prefix (~9 blk) vs 15×6 serial ripple (~92 blk) |
| MUL | `C4_MUL_LOOKAHEAD` | env (default `1`) | `nibble_alu32.mul_lookahead` | Kogge-Stone prefix resolve 8 blk (on) vs 7-ripple 10 blk (off); byte-identical result |
| MUL | `C4_CONST_OPERAND` | env (default `0`) | `qwen_full_vm._const_operand` | const-multiplier fast path (`const_mul`) detection hook — **not** the default path |
| ADD/SUB | *(none)* | — | — | the fused VM folds ADD/SUB into ONE scalar `base_dispatch_rules` FFN rule; **no per-op unit toggle** |
| ADD/SUB | `C4_VM_WIDTH32` / `C4_VM_TWO_LIMB` | env (default off) | `nibble_vm.vm_width32` / `vm_two_limb` | *value substrate* (8-bit fold vs fp32 two-limb), **not** an ADD unit |

**Honest per-op verdict:**

* **ADD/SUB** — essentially ONE production unit (the scalar dispatch rule). A real
  byte-carry-chain gadget (`compile_addsub_blocks`, 8 blocks) exists but is only
  wired on the deep `nibble_pure_forward_complete` path, **not** the fused VM.
* **MUL** — TWO real resolve variants (ripple vs lookahead, byte-identical result,
  differ only in depth) + a genuine `const` multiply (const operand only).
* **DIV** — THREE production-wired variants sharing the `compile_divmod_blocks`
  interface (base long division, radix-16 lean, radix-16 hardened), + the orthogonal
  `recurrent` fold, + a `const`-divisor magic divide. The attention-select /
  automaton / microcode divides (`div_radix16_attn`, `div_automaton_attn`,
  `microcode_div`) are **measure-only bakeoffs** — NOT wired into the builder.

The measure-only bakeoffs and const paths are NOT invented variants — they are real
modules, but they are `wired=False` in the registry and cannot be selected as a
fused-VM unit until the owning agent promotes them. The **sibling "sub-40 divide"**
agent owns adding a new *wired* divide; it registers via `register_div_unit(...)`.

## 2. The unified interface

`C4_ADD_UNIT` / `C4_MUL_UNIT` / `C4_DIV_UNIT` select the unit per op. Each **defaults
to the current production unit**, so an unset environment builds byte-identically to
golden. The registry projects the selection onto the legacy builder flags at the top
of `build()` (`apply_to_env()`), a **no-op when nothing is set** — the proven
flag-read sites are untouched.

```bash
# byte-identical to golden (nothing set):
python -m c4_min.qwen_full_vm ...

# select units explicitly:
C4_DIV_UNIT=radix16_hardened C4_MUL_UNIT=ripple python -m c4_min.qwen_full_vm ...
```

Programmatic:

```python
from c4_min import alu_units as AU
AU.selected_unit("div")              # -> AluUnit(radix16_lean)  (default)
AU.select_units_for_model(80, 8192)  # -> FitPlan for Qwen-72B
```

## 3. Per-op depth / weight / const-vs-variable metadata

`applied` = unrolled forward blocks (total compute); `stored` = distinct layers when
unrolled; `recur` = distinct layers when the divide is folded recurrently (reused
8×) — **this is what a shallow model's layer count must hold**.

```
[ADD]  C4_ADD_UNIT (default -> scalar)
  unit          applied stored recur     nz  wired const  module
  scalar             -      0     -       -  True  False  (rule)      *DEFAULT
  byte_chain         8      8     -       -  False False  nibble_alu32

[MUL]  C4_MUL_UNIT (default -> lookahead)
  unit          applied stored recur     nz  wired const  module
  lookahead          8      8     -   11430  True  False  nibble_alu32 *DEFAULT
  ripple            10     10     -   11430  True  False  nibble_alu32
  const              -      -     -       -  False True   const_mul

[DIV]  C4_DIV_UNIT (default -> radix16_lean)
  unit          applied stored recur     nz  wired const  module
  radix16_lean      80     80    17       -  True  False  div_radix16_lean     *DEFAULT
  radix16_hardened  88     88    18       -  True  False  div_radix16_hardened
  longdiv          262    262    42       -  True  False  nibble_alu32
  const              -      -     -       -  False True   const_divmod_digitrec
  radix16_attn       -      -     -       -  False False  div_radix16_attn      (measure-only)
  automaton_attn     -      -     -       -  False True   div_automaton_attn    (measure-only)
  microcode          -      -     -       -  False False  microcode_div         (measure-only)
```

## 4. Fit-to-model selector

`select_units_for_model(layers, d_model, const_operands=False)` picks a unit per op
to satisfy a layer/width budget. The DIVIDE is the binding constraint (80 applied
blocks). Two mechanisms let it fit a smaller model, and the selector chooses by the
STORED layer cost:

* **UNROLLED** (80 distinct layers) — only genuinely deep models; prefer HARDENED
  when deep AND wide enough for the robustness headroom.
* **RECURRENT fold** (~17 stored layers reused 8×, the production `recurrent_divmod`
  path) — this is how a 24-layer Qwen-0.5B runs an 80-block divide (it *reuses*
  layers).
* **const_operands** — prefer the const magic-multiply / const-divisor magic divide
  where an operand is a compile-time constant (shallowest of all, ~0 added depth).

### Worked examples

| model | layers / d_model | variable-operand plan | const-operand plan |
|-------|------------------|-----------------------|--------------------|
| Qwen-0.5B | 24 / 896 | add=scalar, mul=lookahead, **div=radix16_lean + recurrent** — but even the folded ~17 stored layers + ~16 machinery **exceed 24**, so a variable divide does **not** fit stock-24 (matches the known gap); restrict to const divisors or grow layers | add=scalar, mul=**const**, div=**const** (magic divide) — ~0 arith depth, fits easily |
| Qwen-1.5B | 28 / 1536 | same as 0.5B (still too shallow to unroll; recurrent fold) | mul=const, div=const |
| Qwen-7B | 28 / 3584 | same (layer-bound, not width-bound) | mul=const, div=const |
| Qwen-72B | 80 / 8192 | add=scalar, mul=lookahead, **div=radix16_lean + recurrent** (80 unrolled overflows the 56-layer divide budget after the machinery reserve; folded ~17 stored layers fit with huge headroom) | mul=const, div=const |

Run `python -m c4_min.alu_units` to print the full inventory + all worked examples.

**Honest finding surfaced by the selector:** a *variable* 32-bit divide is ~80 blocks
however you slice it; the only way it "fits" a 24–80-layer Qwen is (a) the recurrent
layer-reuse fold, or (b) a compile-time-constant divisor (magic divide). The selector
says so rather than pretending a variable divide fits stock-24 as distinct layers —
consistent with the recorded `full_native_fast` gap ("recurrent_divmod → 291 layers,
doesn't fit stock-24").

## 5. Byte-identity guarantee

* `apply_to_env()` mutates **nothing** when no `C4_*_UNIT` flag is set → the fused-VM
  build resolves `_div_longdiv()==False`, `_div_lean()==True`, `mul_lookahead()==True`
  (production defaults), so the state dict is unchanged from golden.
* Verified by `test_alu_units.py` (defaults resolve to production, `apply_to_env` is a
  no-op, projection drives the builder, non-wired/unknown tokens raise) and a CPU
  builder-drive check (`C4_TEST_ALU_UNIT_BUILD=1`) that confirms
  `C4_DIV_UNIT=radix16_hardened` switches the divmod builder variant end-to-end.

## 6. Extending (sibling-agent promotion path)

A new *wired* divide (e.g. the "sub-40" build) registers itself without editing
`alu_units.py`:

```python
from c4_min.alu_units import register_div_unit, AluUnit
register_div_unit(AluUnit(
    op="div", name="sub40", depth=38, stored_blocks=38, recurrent_blocks=12,
    wired=True, module="sub40_divide", builder="compile_divmod_blocks",
    note="attention-select radix-16, <40 blocks"))
```

`apply_to_env` / the fit selector pick it up automatically. If the new unit does not
map onto the legacy `C4_DIV_LEAN` / `C4_DIV_LONGDIV` flags (a genuinely new builder
wire), extend the DIV branch of `apply_to_env` to route it — that is the one
integration point.
