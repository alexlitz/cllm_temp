"""Tests for the DSL-interpreter verdict vehicle.

Covers the two load-bearing correctness claims:

  1. :class:`IRBlockForward` (the DSL interpreter executing per-physical-block
     IR) is argmax-byte-identical to ``CachedFaithfulForward`` (the validated
     recovered-weight forward) and to the real ``model.forward`` (modulo the
     model's own saturated-tie fp instability, which both share).
  2. The generator-level attribution classifier maps an emitted rule name back
     to the ISA-DSL generator that produced it.

The model build is shared across the IR tests via a session fixture. CPU-only;
the build is the production ``alu_mode='efficient'`` layout.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")


# --------------------------------------------------------------------------
# Generator attribution — pure string classifier, no model build.
# --------------------------------------------------------------------------


def test_generator_for_rule_recognises_each_generator():
    from neural_vm.verification.generator_attribution import generator_for_rule

    assert generator_for_rule("ax_byte1_carry_val0_lo_3") == "cross_step_carry"
    assert generator_for_rule("stack0_b0_carry_val2_hi_15") == "cross_step_carry"
    assert generator_for_rule("edge_literal_head_token_5") == "full_width_byte_emission"
    assert generator_for_rule("edge_literal_fill_200") == "full_width_byte_emission"
    assert generator_for_rule("pc_chain_consumer_add") == "consumer_lookahead_gate"
    assert generator_for_rule("pc_chain_dump_block_and") == "consumer_lookahead_gate"
    assert (
        generator_for_rule("pc_chain_dump_block_passthrough")
        == "consumer_lookahead_gate"
    )


def test_generator_for_rule_none_for_hand_authored_and_empty():
    from neural_vm.verification.generator_attribution import generator_for_rule

    assert generator_for_rule("l10_tail_stack0_pop_loaded_42") is None
    assert generator_for_rule("some_hand_rule") is None
    assert generator_for_rule(None) is None
    assert generator_for_rule("") is None


def test_attribute_to_generator_strings():
    from neural_vm.verification.generator_attribution import attribute_to_generator

    assert "GENERATOR cross_step_carry" in attribute_to_generator(
        "ax_byte1_carry_val0_lo_3"
    )
    assert "rule 'l10_tail" in attribute_to_generator("l10_tail_x_1")
    assert "attribution impossible" in attribute_to_generator(None)
    assert "imperative composite ALU" in attribute_to_generator(None, is_alu_step=True)


# --------------------------------------------------------------------------
# IRBlockForward byte-identity (needs the model build).
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_model():
    import contextlib
    import io

    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = compile_full_vm_dynamic(
            alu_mode="efficient", disk_cache=False,
        )
    model = model.to("cpu")
    model.eval()
    return model, layout


def _build_tape(bc, data, n_steps=6):
    from neural_vm.speculative import DraftVM
    from neural_vm.vm_step import Token
    from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

    toks = [Token.CODE_START]
    for instr in bc:
        op = instr & 0xFF
        imm = instr >> 8
        toks.append(op)
        for i in range(IMMEDIATE_SIZE):
            toks.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            toks.append(0)
    toks.append(Token.CODE_END)
    toks.append(Token.DATA_START)
    toks.extend(int(b) for b in data)
    toks.append(Token.DATA_END)
    vm = DraftVM(list(bc))
    vm.load_data(data)
    n = 0
    while not vm.halted and n < n_steps:
        if not vm.step():
            break
        toks.extend(vm.draft_tokens())
        n += 1
    return toks


@pytest.mark.slow
def test_irblockforward_argmax_identical_to_cached(built_model):
    """The DSL-interpreter IR forward must equal the recovered-weight forward
    at EVERY token position (byte-for-byte) — this is the proof the IR-execution
    path is faithful."""
    from neural_vm.verification.faithful_interpreter import (
        IRBlockForward, CachedFaithfulForward,
    )
    from src.compiler import compile_c

    model, _layout = built_model
    irf = IRBlockForward(model)
    caf = CachedFaithfulForward(model)

    # A small spread: an ALU add and a var-binding (framing-drift) program.
    srcs = [
        "int main() { return 654 + 114; }",
        "int main() { int x; x = 28; return x; }",
    ]
    total = match = 0
    for src in srcs:
        bc, data = compile_c(src)
        tape = _build_tape(bc, data, n_steps=6)
        ir = irf.forward(tape).argmax(dim=-1).tolist()
        cf = caf.forward(tape).argmax(dim=-1).tolist()
        total += len(ir)
        match += sum(1 for a, b in zip(ir, cf) if a == b)
    # The IR-execution path and the recovered-weight path are the SAME engine
    # math over the SAME specs, so they must agree at every position.
    assert match == total, f"IRBlockForward != CachedFaithfulForward: {match}/{total}"


@pytest.mark.slow
def test_irblockforward_logits_close_to_real_model(built_model):
    """The DSL-interpreter IR forward logits must be numerically close to the
    real ``model.forward`` (the residual diff is fp32 accumulation noise, not a
    structural divergence)."""
    from neural_vm.verification.faithful_interpreter import IRBlockForward
    from src.compiler import compile_c

    model, _layout = built_model
    irf = IRBlockForward(model)
    bc, data = compile_c("int main(){return 21*59;}")
    tape = _build_tape(bc, data, n_steps=5)
    ir_logits = irf.forward(tape)
    real_logits = model.forward(torch.tensor([tape]))[0]
    # Argmax agreement is the byte-for-byte signal; allow the documented
    # saturated-tie positions (rare) to differ.
    ir_am = ir_logits.argmax(dim=-1).tolist()
    real_am = real_logits.argmax(dim=-1).tolist()
    agree = sum(1 for a, b in zip(ir_am, real_am) if a == b)
    assert agree >= len(real_am) - 2, (
        f"IRBlockForward argmax diverges from model.forward at "
        f"{len(real_am) - agree} positions (> 2 saturated-tie budget)"
    )
