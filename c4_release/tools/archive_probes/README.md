# archive_probes/ — retired one-off probe scripts

These are **per-session throwaway diagnostic probes** that were moved out of the
active `tools/` surface by the Infra-J1 probe-script archive sweep
(task #371, 2026-07-03). They are preserved (not deleted) so their history and
any one-shot logic remain available; they are simply out of the way of the
~500-script `tools/` sprawl.

## What was archived vs kept

A probe was **KEPT** in the active `tools/` surface if it met ANY of:

- its filename appears in a **build-path provenance comment** in `neural_vm/`
  (e.g. `# see tools/probe_X.py`) — this is cited evidence for a live model rule;
- its filename appears in a **live design doc** under `docs/` (not `docs/archive/`);
- its filename is cited in the **user-memory** project notes;
- it is **imported** by another kept tool or a test
  (`probe_lib.py`, `probe_ifvar_gt_true.py`, `probe_ifvar_result_step.py`);
- it was **touched on/after 2026-06-27** (possibly in-flight from a concurrent agent);
- it is a canonical reusable probe (`probe_groundtruth.py`, `probe_lib.py`).

Everything else — the `probe_inc3_*`, `probe_var_*`, `_probe_func_*`,
`_probe_loopsum_*`, `_probe_lev_*`, etc. single-session throwaways with no
surviving citation — was moved here via `git mv`.

## Model is unchanged

This sweep is **tooling-only**. Nothing on the build path moved; the default
build state_dict sha256 remained
`b4d2ab273438b3b2fa1bd024b48d0b06a15e63a81f66dec2812ada580b3ec70e`
(byte-identity gate `tools/_isa_golden_hash.py`) before and after.

## Restoring a probe

If you need one back: `git mv tools/archive_probes/<name>.py tools/<name>.py`.
