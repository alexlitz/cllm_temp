"""Tests for the genuinely-vanilla + byte-exact fixed-point MAC through the
discrete-token round-trip (``native_fixed_mac_discrete``)."""
import random

import pytest

from c4_min import native_fixed_mac_discrete as FM


@pytest.fixture(scope="module")
def fm():
    return FM.build(force=True)


def test_it_is_a_real_qwen2_for_causal_lm(fm):
    from transformers.models.qwen2 import Qwen2ForCausalLM
    assert isinstance(fm.model, Qwen2ForCausalLM)
    # a real, populated embedding + lm_head (no zeroed / dummy tables).
    assert float(fm.model.model.embed_tokens.weight.abs().sum()) > 0.0
    assert float(fm.model.lm_head.weight.abs().sum()) > 0.0


def test_gate_default_off_blocks_build(monkeypatch):
    monkeypatch.delenv("C4_FIXED_MAC_DISCRETE", raising=False)
    assert not FM.fixed_mac_discrete_enabled()
    with pytest.raises(RuntimeError):
        FM.build(force=False)      # gated OFF unless force=True or the flag is set


def test_reference_truncate_toward_zero():
    # the fixed_mul reference is truncate-toward-zero (C4/C convention).
    assert FM.from_fixed(FM.fixed_mul(FM.to_fixed(1.5), FM.to_fixed(-3.0))) == -4.5
    assert FM.from_fixed(FM.fixed_mul(FM.to_fixed(-2.5), FM.to_fixed(-4.0))) == 10.0


@pytest.mark.parametrize("a,b,acc0", [
    (2.5, 4.0, 0.0), (3.0, 4.0, 5.0), (-2.5, 4.0, 1.0), (1.5, -3.0, 0.0),
    (-2.5, -4.0, 0.0), (0.5, 0.5, 0.0), (7.0, 13.0, -10.0), (100.25, 2.0, 0.0),
    (-1.0, 1.0, 0.0), (0.5, -0.5, 0.0),
])
def test_mac_byte_exact_discrete_round_trip(fm, a, b, acc0):
    _res, info = FM.run_mac(fm, a, b, acc0)
    assert info["exact"], (a, b, acc0, info["result_value"], info["ref_value"])
    # the answer round-trips through the discrete tokens, NOT a continuous read.
    assert info["used_inputs_embeds_for_computed_value"] is False
    assert info["reencoded_state"] is False
    assert info["tokens_per_mac"] == FM.W_NIB


def test_kv_cache_equals_full_recompute(fm):
    for a, b in [(2.5, 4.0), (-3.0, 7.0), (0.25, -0.5)]:
        r_kv, _ = FM.run_mac(fm, a, b, 0.0, use_kv_cache=True)
        r_full, _ = FM.run_mac(fm, a, b, 0.0, use_kv_cache=False)
        assert r_kv == r_full


def test_random_signed_macs_byte_exact(fm):
    rng = random.Random(7)
    fails = []
    for _ in range(20):
        a = round(rng.uniform(-80, 80), 3)
        b = round(rng.uniform(-80, 80), 3)
        acc0 = round(rng.uniform(-400, 400), 3)
        _res, info = FM.run_mac(fm, a, b, acc0)
        if not info["exact"]:
            fails.append((a, b, acc0, info["result_value"], info["ref_value"]))
    assert not fails, fails


def test_dot_is_a_mac_chain_byte_exact(fm):
    rng = random.Random(3)
    for K in (1, 4, 8):
        a = [round(rng.uniform(-8, 8), 3) for _ in range(K)]
        b = [round(rng.uniform(-8, 8), 3) for _ in range(K)]
        _res, info = FM.run_dot(fm, a, b)
        assert info["exact"], (K, info["result_value"], info["ref_value"])
        assert info["forwards_per_mac"] == FM.W_NIB   # the discrete round-trip cost/MAC
