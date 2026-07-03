"""
neural_vm.verification — CI/dev verification, symbolic-forward interpreters,
dim-contract audits, and byte-faithful interpreters for the Neural VM.

This package holds the *verification / interpreter* dev-tooling subgraph that
was relocated out of ``neural_vm.unified_compiler`` (reduction map ②). None of
these modules is on the weight-authoring build path
(``compile_full_vm_dynamic`` → ``full_vm_compiler_dynamic`` → ``layer_compiler``
→ ``ops/`` → ``ir`` / ``primitives`` / ``building_blocks`` / allocators), which
is why they can live outside the compiler core.

Import direction rule:

* ``verification`` MAY import FROM ``unified_compiler`` (``ir``, ``primitives``,
  ``layer_compiler``, ``predicates``, ``full_vm_compiler_dynamic`` …) freely.
* ``unified_compiler`` MUST import ``verification`` LATE / locally only
  (the two ``layer_compiler`` audit hooks — ``run_dim_integrity_check`` and
  ``run_attention_gate_audit`` — and the HF-export ``model_shape_constraint``
  path), to avoid an import cycle at package load.
"""
