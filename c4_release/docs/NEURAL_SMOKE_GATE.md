# Neural Smoke Gate

The authoritative smoke gate is `tests/test_smoke.py` with strict defaults.
It uses `BatchedPureNeuralRunner` for per-class batching, but the default
mode is still raw neural decode:

- `C4_SMOKE_SPEC_K=0` is the default.
- `spec_k=0` runs one model token per forward and does not create speculative
  DraftVMs.
- The smoke harness also disables DraftVM length bucketing when
  `C4_SMOKE_SPEC_K=0`, so the default gate does not use DraftVM for execution
  or scheduling.
- Results are decoded from the model-emitted context. Do not restore any path
  that substitutes `DraftVM.ax`, `DraftVM.output`, or other DraftVM state as
  the smoke result.
- `C4_SMOKE_MAX_STEPS_CAP` can override the per-group `max_steps` passed into
  `run_batch` for stress or timeout triage. Leave it unset for the gate.

Speculative smoke is a performance experiment only:

```bash
C4_SMOKE_SPEC_K=8 python -m pytest c4_release/tests/test_smoke.py -v --tb=short
```

Do not use speculative smoke as the merge gate unless the strict run below has
already passed.

## Strict Smoke

From the repository root:

```bash
C4_SMOKE_SPEC_K=0 C4_SMOKE_TIMING=1 timeout 1800 \
  python -m pytest c4_release/tests/test_smoke.py -v --tb=short --timeout=900
```

If `timeout` is unavailable on the host, run the same `python -m pytest ...`
command without the outer `timeout`; keep `--timeout=900` when the
`pytest-timeout` plugin is installed.

## Phase Order

Run phases in dependency order and stop at the first unexpected failure or
timeout. Many phase files intentionally contain `xfail` tests; an XPASS is
useful signal and should be triaged rather than hidden.

```bash
timeout 300  python -m pytest c4_release/tests/test_pure_neural_fixture.py -v --tb=short
timeout 600  python -m pytest c4_release/tests/test_pure_neural_pc.py -v --tb=short --timeout=200
timeout 900  python -m pytest c4_release/tests/test_pure_neural_psh_add.py -v --tb=short --timeout=200
timeout 900  python -m pytest c4_release/tests/test_pure_neural_multibyte.py -v --tb=short --timeout=200
timeout 900  python -m pytest c4_release/tests/test_pure_neural_jmp_bz.py -v --tb=short --timeout=200
timeout 900  python -m pytest c4_release/tests/test_pure_neural_jsr_ent_lev.py -v --tb=short --timeout=200
timeout 900  python -m pytest c4_release/tests/test_phase6_syscall_status.py c4_release/tests/test_pure_neural_io.py -v --tb=short --timeout=200
timeout 900  python -m pytest c4_release/tests/test_pure_neural_heap_div.py -v --tb=short --timeout=200
C4_SMOKE_SPEC_K=0 C4_SMOKE_TIMING=1 timeout 1800 \
  python -m pytest c4_release/tests/test_smoke.py -v --tb=short --timeout=900
```

For a cheap harness-only check that avoids model construction:

```bash
python -m pytest c4_release/tests/test_smoke.py::TestSmokeHarnessConfig -v --tb=short
```
