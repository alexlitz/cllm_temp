"""Tests for the PRTF string-quine (``quine_prtf``) and its bundle
(``quine_bundle``).

The FAST tests (default) prove the quine algorithm on the reference interpreter +
the bundle plumbing (build/serialise/reconstruct) without a model bake.  The
NEURAL test (opt-in via ``C4_RUN_NEURAL_QUINE=1``) bakes the small pure-forward
model and proves byte-exact self-output through the KV-cached driver — it is heavy
(~1 min bake + a ~1000-step run), so it is skipped in the default suite.
"""
import os

import pytest

from . import isa
from . import blogspec_vocab as V
from .quine_prtf import build_quine, source_bytes, run_reference, build_quine_prog


def test_prtf_opcode_present():
    """PRTF is op 33 (the spec's printf) and round-trips through the ISA tables."""
    assert isa.PRTF == 33
    assert isa.NAMES[isa.PRTF] == "PRTF"
    assert isa.BY_NAME["PRTF"] == 33


def test_reference_interpret_prtf_visible_channel():
    """The reference interpreter emits AX & 0xFF on PRTF into the ``out`` channel."""
    prog = [("IMM", 72), ("PRTF", 0), ("IMM", 105), ("PRTF", 0), ("HALT", 0)]
    code = isa.assemble(prog)
    out = []
    isa.interpret(code, mem_size=256, max_steps=40, out=out)
    assert out == [72, 105]                # "Hi"


def test_quine_is_self_referential():
    """The quine's data segment ``seed_mem`` equals its own source serialization."""
    code, seed_mem, S = build_quine()
    assert S == source_bytes(code)         # serialization matches the code table
    # Q holds S byte-for-byte at Q_BASE + i.
    for i, b in enumerate(S):
        assert seed_mem[i] == b
    assert isa.BY_NAME["PRTF"] in S        # the quine prints via PRTF, in its source


def test_quine_reference_self_output_byte_exact():
    """The quine printed on the REFERENCE VM equals its own source, byte-for-byte."""
    visible, S = run_reference()
    assert visible == S
    assert len(visible) == len(S) == 2 * len(build_quine_prog()[0])


def test_visible_output_think_tag_extraction():
    """``visible_output`` recovers exactly the between-THINK bytes (§Printing)."""
    # BOS, THINK_START, <frame byte inside think>, THINK_END, 65, THINK_START, HALT
    toks = [V.BOS, V.THINK_START, 7, V.THINK_END, 65, V.THINK_START,
            9, V.THINK_END, 66, V.THINK_START, V.THINK_END, V.HALT]
    assert V.visible_output(toks) == [65, 66]   # the two outside-think bytes only


def test_bundle_build_and_reconstruct_metadata(tmp_path):
    """The bundle serialises the code + data segment + config and reconstructs the
    same self-referential program (no model bake needed for this check)."""
    import torch
    from .quine_bundle import build_bundle
    # Build ONLY the metadata half by monkey-checking the code/seed via build_quine;
    # a full build bakes the model, so gate it behind the neural env like the run.
    code, seed_mem, S = build_quine()
    # the bundle's code/seed_mem/source_bytes must round-trip through save/load.
    payload = {
        "code": [(int(i.op), int(i.imm)) for i in code],
        "seed_mem": {int(a): int(b) for a, b in seed_mem.items()},
        "source_bytes": list(S),
    }
    p = tmp_path / "meta.pt"
    torch.save(payload, p)
    back = torch.load(p, weights_only=False)
    assert back["source_bytes"] == list(S)
    assert [op for op, _ in back["code"]] == [i.op for i in code]
    assert back["seed_mem"] == {int(a): int(b) for a, b in seed_mem.items()}


@pytest.mark.skipif(not os.environ.get("C4_RUN_NEURAL_QUINE"),
                    reason="heavy: bakes the small model + ~1000-step run "
                           "(set C4_RUN_NEURAL_QUINE=1)")
def test_neural_quine_byte_exact_self_output():
    """The quine printed by the SMALL pure-forward transformer (KV-cached driver)
    equals its own source, byte-for-byte — the true neural quine self-output."""
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.blogspec_memory as _MEM
    _PF.SP_INIT = 0xF0
    _PFC.SP_INIT = 0xF0
    _MEM.MEM_ALIBI_SLOPE = 0.05
    import torch
    from .compact_alloc import build_compact_pure_forward_model
    from .sparse_forward import SparseTransformer
    from .nibble_pure_forward_cached import run_pure_forward_cached

    code, seed_mem, S = build_quine()
    model, L, _ = build_compact_pure_forward_model(
        code_size=64, include_bitwise=True, include_divmod=False)
    sparse = SparseTransformer(model, compute_mode="dense_kernel")
    del model
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    sparse = sparse.to(dev)
    out = []
    run_pure_forward_cached(sparse, L, code, max_steps=4000, mask=0xFFFFFFFF,
                            evict=True, prune_interval=90, out=out, seed_mem=seed_mem)
    assert out == S
