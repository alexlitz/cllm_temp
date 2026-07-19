"""The fused Qwen C4 VM is a GENUINE HuggingFace causal LM.

Every test drives the VM through the NATIVE HF generation stack —
``model.generate(do_sample=False)`` / ``TextIteratorStreamer`` / the chat template +
reasoning parser — and asserts byte-exactness vs the ``isa.interpret`` reference (the
key deliverable: greedy ``generate()`` IS the VM executing).  See
``c4_min/vm_causal_lm.py``.
"""
from __future__ import annotations

import threading

import pytest
import torch

from c4_min import isa
from c4_min import blogspec_vocab as V
from c4_min import qwen_full_vm as Q
from c4_min import vm_causal_lm as C
from c4_min.vm_causal_lm import (
    C4VMForCausalLM, build_c4_causal_lm, run_via_generate, decode_ax_trace,
    generate_matches_reference, HaltStoppingCriteria,
)
from c4_min.vm_tokenizer import C4VMTokenizer


# -- the wrapper is a real PreTrainedModel + GenerationMixin -----------------
def test_it_is_a_genuine_hf_causal_lm():
    from transformers import PreTrainedModel, GenerationMixin
    from transformers.models.qwen2 import Qwen2Model
    model = build_c4_causal_lm(isa.assemble([("IMM", 1), ("HALT", 0)]),
                               subset=Q.SUBSET_BASE)
    assert isinstance(model, PreTrainedModel)
    assert isinstance(model, GenerationMixin)
    assert model.can_generate()
    # the embedded engine is a genuine Qwen2Model; the LM head is over the frame vocab
    assert isinstance(model.qmodel, Qwen2Model)
    assert model.config.vocab_size == V.VOCAB          # ~267 frame tokens, not 151k


# -- forward() returns a CausalLMOutputWithPast whose argmax is the next frame token
def test_forward_returns_causal_lm_output():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    model = build_c4_causal_lm(
        isa.assemble([("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)]),
        subset=Q.SUBSET_BASE)
    out = model(input_ids=torch.tensor([[V.BOS]]))
    assert isinstance(out, CausalLMOutputWithPast)
    assert out.logits.shape == (1, 1, V.VOCAB)
    # after BOS the VM opens its think block (THINK_START).
    assert int(out.logits[0, -1].argmax()) == V.THINK_START


# -- THE key deliverable: greedy generate() runs the VM byte-exact -----------
@pytest.mark.parametrize("name,subset,prog", C.BATTERY)
def test_generate_runs_the_vm_byte_exact(name, subset, prog):
    """model.generate(do_sample=False) autoregressively RUNS the VM; the emitted
    frame stream decodes byte-exact to the isa.interpret reference."""
    model = build_c4_causal_lm(isa.assemble(prog), subset=subset)
    r = generate_matches_reference(model)
    assert r["exact"], (name, r["ax_trace"], r["ref_trace"])


def test_generate_equals_custom_driver():
    """Greedy generate() and the engine's own run_program driver are the SAME greedy
    argmax over the frame stream -> identical AX trace (no native-vs-wrapped divergence)."""
    prog = [("IMM", 3), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]
    code = isa.assemble(prog)
    vm = Q.build(code_size=len(code) + 2, subset=Q.SUBSET_BASE)
    driver = Q.run_program(vm, code, max_steps=48)["ax_trace"]
    model = C4VMForCausalLM(vm, C.C4Program(code=code, store_log=[])).eval()
    gen = decode_ax_trace(run_via_generate(model))
    # run_program appends the pre-loop initial-frame AX(=0); align on the tail.
    assert gen[-len(driver):] == driver or gen == driver, (gen, driver)
    assert isa.interpret(code) == gen[-len(isa.interpret(code)):]


# -- the reasoning tokenizer: reasoning_content vs content -------------------
def test_reasoning_tokenizer_splits_think_and_output():
    tok = C4VMTokenizer()
    # a frame (reasoning) then a THINK_END/<byte 'A'>/THINK_START (visible OUTPUT).
    frame = V.build_step_frame(pc=1, ax=65, sp=0, bp=0)
    ids = [V.BOS, V.THINK_START] + frame + [V.THINK_END, ord("A"), V.THINK_START, V.HALT]
    parts = tok.split(ids)
    assert parts["content"] == "A"                     # the visible PRTF byte
    assert "[step]" in parts["reasoning_content"]      # the register-frame reasoning
    raw = tok.decode(ids)
    assert "<think>" in raw and "</think>" in raw and "A" in raw
    assert tok.decode(ids, skip_special_tokens=True) == "A"   # clean assistant msg


def test_chat_template_enable_thinking():
    tok = C4VMTokenizer()
    text = tok.apply_chat_template(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
        tokenize=False, enable_thinking=True)
    assert "<|im_start|>user" in text and "hi" in text and "<think>" in text
    noth = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                   tokenize=False, enable_thinking=False)
    assert "<think>" not in noth
    seed = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True)
    assert seed.tolist() == [[V.BOS]]                  # the VM decodes from BOS


# -- the genuine TextIteratorStreamer streams the frame tokens ---------------
def test_text_iterator_streamer_streams_the_vm():
    from transformers import TextIteratorStreamer, StoppingCriteriaList
    model = build_c4_causal_lm(
        isa.assemble([("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)]),
        subset=Q.SUBSET_BASE)
    tok = C4VMTokenizer()
    streamer = TextIteratorStreamer(tok, skip_prompt=True)
    kw = dict(input_ids=torch.tensor([[V.BOS]]), do_sample=False, max_new_tokens=300,
              streamer=streamer, pad_token_id=V.HALT,
              stopping_criteria=StoppingCriteriaList([HaltStoppingCriteria()]))
    th = threading.Thread(target=model.generate, kwargs=kw)
    th.start()
    text = "".join(list(streamer))
    th.join()
    assert "<think>" in text and "[step]" in text
    assert "AX=13" in text                             # the computed 6+7 result frame


# -- the ELIZA tool-use I/O agentic loop -------------------------------------
@pytest.mark.slow
def test_eliza_turn_through_native_generate():
    from c4_min import chat_eliza as E
    from c4_min import run_eliza_qwen_hf as HF
    from c4_min.run_eliza_causal_lm import eliza_turn_causal_lm
    eliza = E.build_chat_min(rules=HF.FAST_RULES)
    reply, thinking, meta = eliza_turn_causal_lm(eliza, "hi there")
    assert reply == E.chat_turn_ref(eliza, "hi there")  # byte-exact vs reference
    assert reply == "HELLO. HOW ARE YOU?\n"
    assert meta["n_tool"] == 2                          # READ + PRTF serviced
    assert "[step]" in thinking                         # reasoning frames exposed
