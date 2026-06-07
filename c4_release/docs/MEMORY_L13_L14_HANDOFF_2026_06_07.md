# Memory cluster L13->L14 residual handoff investigation — 2026-06-07

Followup to `docs/MEMORY_PHASE4_BLOCKER_2026_06_05.md`. The Phase 4 V2
retry (commit `acdee94d`) verified that even with `layer13_mem_addr_gather`
correctly pinned at L13 and `layer14_mem_generation` at its baseline
placement, the 5 SI/LI/SC/LC smoke tests still fail. This doc reports a
hooked-runtime trace of `test_si_li_roundtrip` and pinpoints where the
residual chain breaks.

## TL;DR

The bug is **NOT** an L13->L14 handoff. The L13 `mem_addr_gather` heads
fire at the LI step's emitted MEM block but write the **wrong nibble
values** because the upstream MEM tokens (addr bytes d=1..4 of the MEM
section) **are already corrupted before L13 runs**. Specifically:

* The model autoregressively emits the MEM block (marker + 4 addr bytes
  + 4 val bytes) on every SI/LI step. Those tokens carry the address
  the operation will store-at / load-from.
* For `IMM 0x200, PSH, IMM 42, SI`, the SI step's MEM block should
  carry `addr_bytes = [0, 2, 0, 0]` (little-endian 0x200) and
  `val_bytes = [42, 0, 0, 0]`.
* Actual emitted tokens: `addr_bytes = [224, 255, 0, 0] = 0xFFE0`,
  `val_bytes = [0, 255, 255, 0]`. The address is the SP underflow
  value, not the pushed 0x200.
* `_inject_mem_section` in `run_vm.py` has been a **no-op** since
  V6 retirement (2026-05-12). The Python overrides do NOT
  populate the MEM tokens; the model has to predict them itself
  via `layer14_mem_generation` heads 0-7 (autoregressively).

`layer14_mem_generation` is the SOLE producer of the MEM block's
addr+val bytes in pure_neural mode. Its heads K-side for SI reads
`STACK0_BYTE0` (head 0) and `BP/L1H4+BP_I` thresholds (heads 1-3) —
the latter assume the value source is laid out like a BP-stack frame.
But STACK0 in this build only stores byte 0 in `STACK0_BYTE0`;
bytes 1-3 are NOT in any STACK0_BYTE* dim. So heads 1-3 attend to
the wrong tokens and emit zero / spurious nibbles for addr bytes
1, 2, 3.

## Per-layer residual trace (test_si_li_roundtrip, last full forward)

Sequence: `[IMM 0x200, PSH, IMM 42, SI, IMM 0x200, LI, EXIT]`,
seq_len=253, result=**0** (expected 42; baseline V2 reported "gets 512"
which is the same class of failure — AX retains the IMM 0x200 value
because LI's lookup returns 0 and the L19 routing layers don't override).

MEM markers land at positions 85 (PSH), 118 (SI), 153 (IMM-with-empty-MEM?), 188 (LI).

### L8 output (after `layer7_memory_heads` heads 2-4 write ADDR_B*_LO/HI at AX positions for LI Q-side)

At LAST `MARK_AX` @224 (LI step):
```
d=0 pos=224 tok=258: B0_LO=argmax=0 B0_HI=0 B1_LO=2 B1_HI=0 B2_LO=0 B2_HI=0
d=1 pos=225 tok=0:   B0_LO=0       B0_HI=0 B1_LO=2 B1_HI=0 B2_LO=0 B2_HI=0
```
Q-side ADDR_B*_LO/HI at LI AX position correctly encodes 0x200 (B1_LO=2). **L7 works.**

### L13 output (after `layer13_mem_addr_gather`)

```
MARK_MEM positions: [85, 118, 153, 188]

At MEM @118 (SI step, MEM_STORE=1.0, MEM_ADDR_SRC=0):
  d=4..7 (MEM_VAL_B0..3 positions): B0_LO=. B0_HI=. B1_LO=. B1_HI=. (ALL ZERO)

At MEM @188 (LI step, MEM_STORE=1.0, MEM_ADDR_SRC=2):
  d=4..7: B0_LO=argmax=0 (0.79)  B0_HI=0 (1.24)
          B1_LO=0 (1.11)  B1_HI=0 (1.50)
          (Expected: B1_LO=2 for address byte 1 = 0x02 of 0x200)
```

L13 head 1 (`addr_b1`) gather K side reads `L1H2+MEM_I` (positive) and
`L1H1+MEM_I` (negative), so the head attends to the token at d=2 from
the MEM marker (the addr byte 1 token). The V copies CLEAN_EMBED from
that token. The CLEAN_EMBED at MEM @188 d=2 is `LO=0, HI=0` (token id 0,
not 2 as expected). So the gather correctly copies what's there — but
what's there is wrong.

### MEM block token IDs (the smoking gun)

```
MEM @85  (PSH step):  d0=261 d1..8=[0, 0, 0, 0, 0, 0, 0, 0]
MEM @118 (SI step):   d0=261 d1..8=[224, 255, 0, 0, 0, 255, 255, 0]
MEM @153 (extra):     d0=261 d1..8=[0, 0, 0, 0, 0, 0, 0, 0]
MEM @188 (LI step):   d0=261 d1..8=[0, 0, 1, 0, 42, 0, 0, 2]
```

The SI step's MEM block addr decodes to `0x0000FFE0` (= SP underflow
`-32`) — the model emitted the SP pointer instead of the popped STACK0
value (which was the actual store address 0x200). The val bytes are
also garbled (`[0, 255, 255, 0]`).

The LI step's MEM block has bytes `[0, 0, 1, 0]` for addr — neither
0x200 nor SP. Some downstream attention is leaking unrelated values.

### Confirmation: `_inject_mem_section` is a no-op

```python
# c4_release/neural_vm/run_vm.py:3105
def _inject_mem_section(self, addr, value):
    """V6 RETIRED (2026-05-12): no-op."""
    if self._debug_memory:
        print(...)
```

So in pure_neural mode, the only source of MEM block contents is the
neural network's autoregressive emission, which is driven by
`layer14_mem_generation`.

## The actual bug: `layer14_mem_generation` heads 1-3 SI source path

`c4_release/neural_vm/unified_compiler/ops/l14_ops.py:407-545`
(`_layer14_mem_generation_head_specs`).

For each addr-byte head h in 0..3, slot 2 (STACK0 source bonus, gated
by `MEM_ADDR_SRC` for SI/SC) writes K reads:

```python
if h == 0:
    k.append(AP(2, BD.STACK0_BYTE0, L))            # OK: STACK0 byte 0 in STACK0_BYTE0 dim
elif h == 1:
    k.append(AP(2, BD.L1H4 + BP_I, L))             # wrong: BP-relative threshold
    k.append(AP(2, BD.H1 + BP_I, -L))
elif h == 2:
    k.append(AP(2, BD.H2 + BP_I, L))               # wrong: BP-relative threshold
    k.append(AP(2, BD.L1H4 + BP_I, -L))
elif h == 3:
    k.append(AP(2, BD.H3 + BP_I, L))               # wrong
    k.append(AP(2, BD.H2 + BP_I, -L))
```

These K writes use **BP-relative position thresholds** (`H1/H2/H3/L1H4
+ BP_I`), which fire at byte indices 1, 2, 3 of the BP-marked register.
This was the design when STACK0 was implemented as a "BP byte stack"
(STACK0_BYTE0..3 aliased into BP value bytes). But in the current
dim registry (`dim_registry_dynamic.py`), `STACK0_BYTE1/2/3` are
distinct slots at compact positions 730/731/732 (not aliased onto BP)
— and the L14 head specs were never migrated.

So at the SI step's MEM block construction, head 1 attempts to attend
to "byte 1 of STACK0" via `L1H4+BP_I` threshold (which fires at BP
byte positions, NOT STACK0 byte 1 positions). The attention lands on
zero or garbage rows, and the V passes CLEAN_EMBED of those rows
through to the emitted addr byte 1 nibble — which is 0 (or the SP
underflow value 0xFF).

Head 0 still works because `STACK0_BYTE0` is a stable dim that L8
sp_gather writes correctly. That's why **byte 0 of the addr is at
least sometimes right (= 0 = low byte of 0x200)**, but bytes 1-3 are
garbage.

## Per-hypothesis verdict from the brief

The brief listed 4 hypotheses; here is what the trace says:

| # | Hypothesis | Verdict |
|---|---|---|
| 1 | `mem_generation` reads stale `ADDR_B` cross-step | **No.** `mem_generation` does not read `ADDR_B*`; it WRITES the addr bytes via L14 attention heads to the emitted MEM block. The cross-step `.*.-1` SSA aliases (e.g. `layer14_addr_key_neural_decode` reading `ADDR_B0_LO.*.-1`) resolve to the same numeric dim slot, so the runtime read is in-step. |
| 2 | `mem_addr_gather` writes to wrong dim | **No.** L13 heads 0-2 correctly write `ADDR_B0/1/2_LO/HI` at MEM val byte positions when the upstream addr-byte tokens are valid. At MEM @188 it fires; the values are wrong because **the upstream addr-byte tokens are themselves wrong**. |
| 3 | L14 head 1 attention failure | **YES — but it is `layer14_mem_generation` head 1 (the MEM block addr generator), not the gather.** Heads 1-3's K-side reads BP-relative thresholds (`H1/H2/H3/L1H4 + BP_I`) for the STACK0 source branch (slot 2). STACK0 in the current dim registry only exposes byte 0 (`STACK0_BYTE0`); bytes 1-3 (`STACK0_BYTE1/2/3` at compact slots 730/731/732) are NOT read by the head specs. Result: SI emits wrong addr bytes 1/2/3, and L13 gathers those wrong values into `ADDR_B*_LO/HI` at MEM val byte positions, and L15 memory_lookup's K-side never matches the Q-side address. |
| 4 | L15 memory_lookup reads wrong | **No.** L15 reads `ADDR_KEY` / `ADDR_B*_LO/HI` correctly. The Q-side at the LI's AX byte position has the right address (0x200) via L7. The K-side has the wrong stored address (garbage) because L14 emitted garbage on the SI step. |

## Recommended fix

`c4_release/neural_vm/unified_compiler/ops/l14_ops.py:478-491`
(slot 2 K writes for heads 1-3):

```python
# OLD (broken — BP-relative thresholds for STACK0 byte 1-3 source):
elif h == 1:
    k.append(AP(2, BD.L1H4 + BP_I,  L))
    k.append(AP(2, BD.H1   + BP_I, -L))
elif h == 2:
    k.append(AP(2, BD.H2   + BP_I,  L))
    k.append(AP(2, BD.L1H4 + BP_I, -L))
elif h == 3:
    k.append(AP(2, BD.H3   + BP_I,  L))
    k.append(AP(2, BD.H2   + BP_I, -L))

# NEW (read the dedicated STACK0_BYTE1/2/3 dims that the post-Phase
# 8 dim registry pins at compact slots 730/731/732):
elif h == 1:
    k.append(AP(2, BD.STACK0_BYTE1, L))
elif h == 2:
    k.append(AP(2, BD.STACK0_BYTE2, L))
elif h == 3:
    k.append(AP(2, BD.STACK0_BYTE3, L))
```

This requires verifying that `STACK0_BYTE1/2/3` are populated by
upstream layers at the same token position the head's Q fires on
(the MEM addr byte d=1/2/3 position from MEM marker). If they are not,
add a pre-L14 gather op (analogous to `layer8_sp_gather_bake` which
populates `STACK0_BYTE0` at the STACK0 marker position) to broadcast
STACK0 bytes 1-3 to MEM addr-byte positions BEFORE L17 fires.

The PSH step's `_inject_mem_section` no-op confirms there is no
Python-side rescue path — every byte of the MEM section must be
emitted neurally. The legacy implementation assumed STACK0 was BP-
backed (`STACK0_BYTE1..3` aliased onto BP byte 1..3), and the head
specs were never updated when the dim registry separated the two
families.

## Files referenced

* `c4_release/neural_vm/unified_compiler/ops/l14_ops.py:407-641`
  — `_layer14_mem_generation_head_specs`, lines 478-491 are the slot-2
  K writes for heads 1-3 (the broken STACK0 byte 1/2/3 reads).
* `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:282-375`
  — `_layer13_mem_addr_gather_head_specs` (works correctly; not the bug
  site, despite Phase 4's premise).
* `c4_release/neural_vm/unified_compiler/ops/l15_ops.py:401-652`
  — `layer15_memory_lookup` (heads 0-3 binary address match at dims
  4-27; reads correctly).
* `c4_release/neural_vm/dim_registry_dynamic.py:539-549`
  — `STACK0_BYTE1/2/3` pins at compact positions 730/731/732 (the
  dim slots L14 should be reading).
* `c4_release/neural_vm/run_vm.py:3105-3121`
  — `_inject_mem_section` no-op confirmation.
* `c4_release/docs/MEMORY_PHASE4_BLOCKER_2026_06_05.md`
  — Phase 4 V2 retry's structural-objective-met-but-still-fails report
  (root cause is upstream of L13, not in the L13->L14 handoff).
