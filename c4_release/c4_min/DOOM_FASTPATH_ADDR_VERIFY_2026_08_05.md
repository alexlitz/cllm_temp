# Direct-CAM O(1) ADDRESS verification (C4_DIRECT_CAM_VERIFY_ADDR)

Closes the "memory-read ADDRESS is DRAFT-TRUSTED" gap the faithfulness audit
(`DOOM_FASTPATH_FAITHFULNESS_AUDIT_2026_08_05.md`, scenario E) proved: on the
direct-CAM fast path a *self-consistent* wrong-address draft was ACCEPTED 7/7 because
the read's address→row→value was resolved entirely by `resolve_load_rows(draft)` and
injected, while **the model's own `W_q` query for the CAM heads was computed then
discarded**. This lever makes the fast path INDEPENDENTLY VERIFY that address at O(1) per
read — without touching the large-KV speedup. Default OFF; golden `069cc32f` / CFM
`7d19cdc3` unchanged.

## The mechanism (files:lines)

The model's query for a memory CAM head **is** the binary address key. In
`_bake_cam_head` (`nibble_pure_forward_complete.py:571-575`) each address bit `b` is
written as
```
Q[base+b] = 2*smag*QRY_BIN[b] - smag*ONE  = smag*(2*bit - 1)   (smag > 0)
```
so `sign(Q[base+b])` decodes the queried address bit `b` exactly (`>0` ⟺ 1, `<0` ⟺ 0).

- `direct_cam_batched.py`
  - `verify_addr_enabled()` — the `C4_DIRECT_CAM_VERIFY_ADDR` flag (default OFF).
  - `ResolvedTable.addr` / `.code_addr` + `build_resolved_table` (`:249-266`) — records
    the **draft-claimed** address (`ResolvedRead.addr`) per read, alongside the value.
  - `_decode_model_query_addr(Qh_rows, n_bits)` (`:131-155`) — the O(n_bits) sign decode
    of the model's OWN query at each read row. `_n_addr_bits` = 32 (mem/pop/lev/uni) or
    12 (code).
  - the check inside `direct_forward` (`:191-240`) — after `Q = W_q.linear(x)`
    (`:474`, the query the gather is about to discard), for each CAM read row it decodes
    the model's address and compares to `tbl.addr`. On mismatch it calls
    `DivergenceSink.report`. **O(1) per read**, no O(n_store) softmax — a per-bit sign
    read of one query vector against the one draft-claimed address.
- `pf_speculative.py` (`verify_blocks`, `:1940-1997`) — after each span forward, polls
  `_dcam_tbl.addr_sink`; a hit becomes a terminal FAIL `VerifyResult(all_matched=False,
  first_mismatch={"kind":"cam_addr", ...})` — the SAME first-divergence-stop path the
  register/token compare uses (`accepted = min(accepted, step_of_divergence)`).

## Before / after (the audit's wrong-address experiment)

`_agent_verify_addr_experiment.py` re-runs the audit's store/load program
(`ENT 1; LEA 0; PSH; IMM 777; SI; LEA 0; LI; HALT`, final AX 777) and corrupts the LI
read to a **self-consistent decoy address** (a decoy store at a different address whose
value is still 777, so the injected value matches the frame — the register-decode still
passes; only the resolved ADDRESS is wrong):

| case | flag | result |
|---|---|---|
| clean draft | ON | ACCEPTED 8/8, AX=777 (no false positive) |
| wrong-address, self-consistent | **OFF** | **ACCEPTED 8/8, AX=777 — rubber-stamped** (the audit gap) |
| wrong-address, self-consistent | **ON** | **CAUGHT** — FAIL at the read step, `kind=cam_addr`, `got addr=248 (model's own query) != want addr=252 (draft resolved)` |

`before: ACCEPTED (rubber-stamp)  →  after: CAUGHT`.

## Byte-exact on correct execution (no false positive)

`_agent_verify_addr_byteexact.py` (flag ON, vs the softmax path):
add / mul / **div** / mem / func(LEV) / bigimm / jmp / bnz — all L-inf=0 accept, same
final AX; and the **6315-step deep loop** (heavy mem/pop/lev traffic) accepts
6315/6315, AX=45150, L-inf=0. `_agent_verify_addr_doom_slice.py` — a **real id-doom
slice** (compiled from the actual `doom.c`, 1500 drafted steps) decodes byte-exact:
softmax == direct == direct+verify (all 1500/1500, identical AX). The check never
false-fails a correct read.

## fps impact ≈ 0

`_agent_verify_addr_fps.py` — direct-CAM `verify_blocks` steps/sec on the 6315-step deep
loop, K=256, best of 3:
- verify-addr **OFF**: 4.33 ms/step, ~231 steps/s
- verify-addr **ON** : 3.99 ms/step, ~250 steps/s

Within run-to-run noise (the O(1) per-read sign-decode is negligible vs the multi-block
forward). The composed CUDA-graph replay path (`install_composed` / precomputed schedule)
does not go through `verify_blocks` and is entirely unaffected.

## Honest scope — what it now VERIFIES vs still TRUSTS

- **VERIFIES (independently): the memory-read ADDRESS.** The `mem` LOAD address (LI/LC
  `mem[addr]`) is the genuinely draft-trusted, non-register-covered quantity, and it is
  now checked against the model's OWN decoded query. `pop`/`lev` (SP / BP+4) are checked
  too (defence in depth; those are also register-derived). A wrong-address draft
  pointing to any NON-ZERO address the model would not have queried is CAUGHT.
- **The CODE-fetch address (PC) is covered elsewhere, so it is EXCLUDED from this check**
  (`direct_cam_batched.py:200-207`): the register-transition compare already verifies
  the PC every step (`got_pc != want_pc`), and the code head's query encoding at the
  value-injection row differs from the memory heads' `smag*(2*bit-1)` pattern (a
  sign-decode there would false-positive).
- **STILL TRUSTS: the VALUE at the correct address.** A self-consistent value swap AT the
  verified address is not caught here — re-deriving the value requires the full O(S)
  softmax gather over the stores (that is the full-attention job, out of scope; the
  softmax path, flag-OFF, does re-derive it and catches a wrong value).
- **Documented conservative limit:** the check fires only when the model asserts a
  NON-ZERO address that disagrees (`ma != 0 and ma != da`). Under the batched/recurrent
  span the block-input residual does not always carry the read's `QRY_BIN` (an all-zero
  query = "address not asserted in THIS residual view"), which is ambiguous with a
  genuine address-0 read; skipping `ma==0` mismatches is the conservative choice (never a
  false positive). The residual honest gap: a draft that resolves a read to a non-zero
  address the model would have read as 0 is indistinguishable from an unasserted query
  and is NOT caught. This is strictly a superset of the old behaviour (which verified the
  address for NONE of the reads).

## Goldens

- default (flags-OFF): `069cc32fa7cecfbceae448a7dbf6e2140b3db6cf6857c8accec5639b9c55c0ca`
- CFM (`C4_PF_CFM=1`): `7d19cdc3f8190954a913f60dedc8599254584e735e70efdce867b12fa53baa05`

Both UNCHANGED (`python -m c4_min._fingerprint_build`). The lever is a runtime forward
wrapper installed only when `C4_DIRECT_CAM_BATCHED=1 C4_DIRECT_CAM_VERIFY_ADDR=1`; no
weights touched.
