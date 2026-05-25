# C4 Neural VM development targets.
#
# Targets here wrap routine workflows so contributors don't have to
# remember pytest marker invocations. The headline gate is `validate`,
# which runs the declarative verifier suite over the production VM build.

.PHONY: validate validate-strict

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
