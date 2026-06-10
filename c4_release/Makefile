# C4 Neural VM development targets.
#
# Targets here wrap routine workflows so contributors don't have to
# remember pytest marker invocations. The headline gate is `validate`,
# which runs the declarative verifier suite over the production VM build.

.PHONY: validate validate-strict pure-neural-ratchet pure-neural-ratchet-micro

# Mode A static claim verification + Mode B+ multistep dynamic verifier.
# Runs under the `validator` pytest marker (registered in pyproject.toml).
# Wall time is roughly 60-90s on a warm cache; the verifier shares one
# bake across the multistep tests via class-scoped fixtures.
validate:
	pytest -m validator --tb=short

# Strict mode escalates partial-claim residue (cells the bake wrote but
# the op did not declare) to errors. Currently informational; switch on
# nightly once ops adopt full-coverage claim declarations.
validate-strict:
	pytest -m validator_strict --tb=short

# Pure-neural smoke ratchet (no Python ALU cheats). Re-runs the full
# test_smoke.py with C4_DISABLE_BATCHED_ALU_RECOVERY=1 and asserts the
# pass count >= _PURE_NEURAL_BASELINE in
# tests/test_pure_neural_smoke_ratchet.py. Wall ~5-15 min — gated behind
# @pytest.mark.slow so it only fires when --runslow is passed.
# Bump _PURE_NEURAL_BASELINE in the same commit as any improving fix.
pure-neural-ratchet:
	pytest --runslow tests/test_pure_neural_smoke_ratchet.py -v -s

# Fast (<30s) micro version — runs only TestSmokeBasic under the no-cheats
# env var. Use as a pre-push sanity check; if this regresses the full
# ratchet almost certainly does too.
pure-neural-ratchet-micro:
	pytest tests/test_pure_neural_smoke_ratchet_micro.py -v -s
