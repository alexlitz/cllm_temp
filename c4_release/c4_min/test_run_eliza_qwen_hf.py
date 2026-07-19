"""ELIZA runs through a genuine ``transformers.Qwen2Model.forward``, byte-exact.

Each test drives the ELIZA chat program (``chat_eliza.build_chat_min``) through an
actual ``Qwen2Model`` (``qwen_full_vm.build(subset=SUBSET_MEM_CMP)``): the user
message enters via the tool-use I/O (READ fd 0), the prefix-match pattern-match
runs entirely inside ``Qwen2Model.forward`` (LC loads through the RoPE §Memory CAM,
EQ compares, BZ/JMP branch), and the reply comes out via PRTF. The reply is
asserted BYTE-EXACT to the plain-python reference (``run_eliza_reference``) on the
identical bytecode. A tiny 2-rule ELIZA keeps the per-forward window small so the
suite runs on CPU in seconds. See ``c4_min/run_eliza_qwen_hf.py``.
"""
from __future__ import annotations

import pytest

from c4_min import chat_eliza as E
from c4_min import run_eliza_qwen_hf as R


# A minimal 2-rule ELIZA: tiny code + data segment => small forward window => fast.
_TINY_RULES = [("hi", "HELLO THERE.\n"), ("bye", "GOODBYE.\n")]


@pytest.fixture(scope="module")
def chat():
    eliza = E.build_chat_min(rules=_TINY_RULES)
    return R.build_qwen_eliza(eliza=eliza, verbose=False)


@pytest.mark.parametrize("msg", ["hi there", "bye now", "something else"])
def test_reply_is_byte_exact_through_qwen(chat, msg):
    """The Qwen2Model.forward reply is byte-identical to the reference."""
    reply, steps, ok = chat.reply(msg, check_reference=True)
    ref = E.chat_turn_ref(chat.eliza, msg)
    assert reply == ref, (msg, repr(reply), repr(ref))
    assert ok is True
    assert steps > 0


def test_keyword_match_produces_the_right_response(chat):
    reply, _, ok = chat.reply("hi there friend", check_reference=True)
    assert reply == "HELLO THERE.\n" and ok


def test_fallback_on_no_match(chat):
    reply, _, ok = chat.reply("zzz nothing matches", check_reference=True)
    assert reply == E.ELIZA_FALLBACK and ok


def test_the_model_is_a_genuine_qwen2(chat):
    from transformers.models.qwen2 import Qwen2Model
    assert isinstance(chat.vm.qmodel, Qwen2Model)
    cfg = chat.vm.qmodel.config
    # a real Qwen2 config: GQA 14/2, SwiGLU, RMSNorm, RoPE head_dim 64 / theta 1e6.
    assert cfg.num_attention_heads == 14 and cfg.num_key_value_heads == 2
    assert cfg.hidden_act == "silu" and cfg.rms_norm_eps == 1e-6
    assert cfg.head_dim == 64 and cfg.rope_theta == 1_000_000.0


def test_a_multi_turn_exchange_is_byte_exact(chat):
    convo = ["hi", "blah", "bye"]
    for msg in convo:
        reply, steps, ok = chat.reply(msg, check_reference=True)
        assert ok, (msg, reply)


def test_eliza_avoids_the_muldiv_d_budget_wall(chat):
    """ELIZA needs mem+cmp only (no mul/div/mod) => it does not cross the 45 GB
    byte-table wall. The intermediate_size stays tiny (896), nowhere near 160465."""
    assert chat.vm.intermediate_size <= 896
    assert not chat.vm.subset.muldiv and not chat.vm.subset.bitwise
    assert chat.vm.subset.memory and chat.vm.subset.cmp
