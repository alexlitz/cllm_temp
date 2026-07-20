# c4_min Consolidation Report — `c4min-consolidated-2026-07-20`

Consolidates this session's VERIFIED c4_min feature branches into one canonical
branch, byte-identity-gated. `main` is untouched (stays at `e45eb4ff`); nothing
is merged to main.

## Base

- Branch created from `fix-zfod-malloc-neural` (`8426cad5`), the tip of the
  verified trunk chain:
  `49907f07` (unify-full-vm-no-lean-splits) →
  `cc16ceed` (lib-into-unified-model, ported lib + addr32) →
  `8426cad5` (fix-zfod-malloc-neural, malloc-family byte-exact + eviction-default).
  Chain confirmed linear; all three descend from `main` at `e45eb4ff`.

## Byte-identity gate

Fingerprint = deterministic sha256 over the DENSE-reconstructed per-block weights
(fp64, sorted order) of the memory-safe streaming sparse build
(`build_compact_sparse_streaming`, peak ~10-11 GB RSS). Tool:
`c4_min/_fingerprint_build.py`. The full dense build is a 54-117 GB hazard; the
streaming build is byte-identical (L∞=0, `dense_kernel`) and stays under 20 GB.

| Stage | Fingerprint (sha256, first 16) | Note |
|-------|--------------------------------|------|
| base `fix-zfod-malloc-neural` (`8426cad5`) | `3566e0560d3b71f9` | reference |

## Per-branch log

### 1. `fix-frame-local-ptr-store` (#660) — MERGED CLEAN (fingerprint changed, EXPECTED)

- Merge: clean, no conflicts. `--no-ff` merge commit.
- NOTE: #660 is a strict descendant of #648 `libprog-neural-phase2` — merging it
  ALSO brings in the entire libprog corpus (item 10 below becomes a no-op).
- Build-path change: adds a NEW LEA-gated physical block `lea-addr-nib` (after
  `ax-nib-split`, before `ax-byte-nib`) that recomputes the frame-address nibbles
  from `BP_LOW+4*imm` with a round-to-nearest integer decode. Gated on
  `OP_IS[LEA]` → writes nothing on every non-LEA op.
- Fingerprint: `3566e0560d3b71f9` → `f61decf217c718ba` — **CHANGED, and this is
  EXPECTED**: a new physical block is added (n_blocks 305 → 306), so the
  state_dict necessarily differs. This is a real, VERIFIED functional fix
  (9/9 BP-low-byte LEA sweep, no regression per #660), listed explicitly in the
  consolidation brief as an additive coexisting block — NOT a byte-identity-neutral
  change. Functional non-regression is confirmed by the final test suite.
- New expected baseline from here: `f61decf217c718ba`.
