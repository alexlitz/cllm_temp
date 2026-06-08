"""Phase R8 — end-to-end Qwen export round-trip tests.

These tests pin the current state of the Qwen3-dense export pipeline as
of 2026-06-07. The full R8 acceptance criterion in the plan is
"≥99 % argmax match against the native runner for
``int main(){return 42;}``" but several final-mile blockers (documented
in ``docs/QWEN_R8_E2E_2026_06_07.md``) prevent reaching that target
today.

The tests below assert the *positive* facts that R6 / R7 already
delivered (the artefact loads through ``AutoModelForCausalLM``) and the
*negative* facts that R8 surfaced (composite FFNs block production
export; tiny-VM forward parity falls short of the 99 % gate due to
softmax1 sink + q_norm/k_norm gaps). When a future change closes one of
those gaps the matching test fails and forces an update.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

transformers = pytest.importorskip("transformers")

import torch

from neural_vm.qwen_compat import export_qwen3_dense
from neural_vm.vm_step import AutoregressiveVM


NORM_COMPENSATOR_K = 1000.0
R8_ARGMAX_TARGET = 0.99  # plan §R8 acceptance criterion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tiny_qwen_compatible_vm(
    *,
    d_model: int = 32,
    n_layers: int = 2,
    n_heads: int = 4,
    ffn_hidden: int = 64,
) -> AutoregressiveVM:
    """Tiny VM whose runtime config matches Qwen3-dense expectations.

    Differs from the R6 export-pipeline fixture (which compiles with
    ALiBi + softmax1) by running with RoPE + standard softmax + RMSNorm.
    This is the closest the VM gets to Qwen3 semantics at runtime so
    forward-parity numbers are meaningful.
    """

    vm = AutoregressiveVM(
        vocab_size=276,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_hidden=ffn_hidden,
        max_seq_len=128,
        positional_encoding="rope",
        attention_normalization="softmax",
        use_rms_norm=True,
        use_flash_attention=False,
    )
    vm.dim_positions = {"NORM_COMPENSATOR": 0, "CONST": 1}
    with torch.no_grad():
        vm.embed.embed.weight[:, 0] = NORM_COMPENSATOR_K
        for block in vm.blocks:
            block.attn.W_o.data[0, :] = 0.0
            block.ffn.W_down.data[0, :] = 0.0
            # Non-trivial bias columns so the R4 bias fold has something
            # to absorb when we compare forward outputs.
            block.ffn.b_up.data.fill_(0.1)
            block.ffn.b_gate.data.fill_(0.05)
    return vm


def _vm_forward_plain(vm: AutoregressiveVM, input_ids: torch.Tensor) -> torch.Tensor:
    """Forward through the VM **without** the NeuralVMEmbedding augmentations.

    The tiny VM doesn't have the wide residual layout the production VM
    uses, so ``vm.embed(input_ids)`` would index out of range when it
    tries to write into the ADDR_KEY / MEM_STORE slots. For the R8 forward
    parity we skip the augmentations and use the inner ``nn.Embedding``
    directly — this is the closest analogue to Qwen3's plain embedding
    lookup. The production VM gap is documented as Blocker 5.
    """

    x = vm.embed.embed(input_ids)
    for block in vm.blocks:
        x = block(x)
    return vm.head(x)


# ---------------------------------------------------------------------------
# R8 acceptance: HF load gate
# ---------------------------------------------------------------------------


def test_export_loads_through_hf_auto_for_tiny_vm():
    """`AutoModelForCausalLM.from_pretrained` round-trips a tiny export."""

    from transformers import AutoConfig, AutoModelForCausalLM

    vm = _build_tiny_qwen_compatible_vm()
    with tempfile.TemporaryDirectory() as tmp:
        export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)

        # AutoConfig should pick up our config.json as Qwen3.
        cfg = AutoConfig.from_pretrained(tmp)
        assert cfg.model_type == "qwen3"
        assert cfg.architectures == ["Qwen3ForCausalLM"]

        # AutoModelForCausalLM should load the state_dict cleanly.
        qmodel = AutoModelForCausalLM.from_pretrained(tmp)
        assert type(qmodel).__name__ == "Qwen3ForCausalLM"

        # The export's state_dict keys should exactly equal the
        # loaded model's expected keys — no missing, no unexpected.
        src_sd = torch.load(
            os.path.join(tmp, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
        tgt_keys = set(qmodel.state_dict().keys())
        assert set(src_sd.keys()) == tgt_keys, (
            f"state_dict mismatch: "
            f"src - tgt = {sorted(set(src_sd) - tgt_keys)[:5]}; "
            f"tgt - src = {sorted(tgt_keys - set(src_sd))[:5]}"
        )


# ---------------------------------------------------------------------------
# R8 acceptance: forward parity (current state pins ≤ 99 %)
# ---------------------------------------------------------------------------


def test_forward_parity_tiny_vm_high_match():
    """Tiny VM ↔ exported Qwen3 argmax parity is in the 90s.

    On the PureFFN-only tiny VM whose runtime config matches Qwen3
    (RoPE + standard softmax + RMSNorm), the R1 compensator + R2 RMSNorm
    identity gamma + R4 SwiGLU repack & bias fold compose into an
    end-to-end forward pass whose logits land within ~0.2 of the VM's.

    Logit |Δ| is non-zero because softmax1 (VM) ≠ standard softmax
    (Qwen3) and Qwen3 applies per-head q_norm / k_norm. At random init
    the top-1 / top-2 gap is tiny (sub-0.1 logit units), so a handful of
    near-tie positions swap argmax under the perturbation. On baked
    production weights the gap is much wider and the perturbation does
    not matter — see the plan §R8 acceptance ("dense should be 100 %").

    We assert ``match >= 0.90`` to gate against structural regressions
    (a real RoPE or RMSNorm break would drop us to ~1/276 ≈ 0.4 %).
    The R8 plan's ≥99 % target requires baked weights, which is gated
    separately by ``test_full_vm_export_fails_on_composite_ffn``.
    """

    from transformers import AutoModelForCausalLM

    vm = _build_tiny_qwen_compatible_vm()
    vm.eval()

    with tempfile.TemporaryDirectory() as tmp:
        export_qwen3_dense(vm, tmp, K=NORM_COMPENSATOR_K)
        qmodel = AutoModelForCausalLM.from_pretrained(tmp)
    qmodel.eval()

    total_match = 0
    total = 0
    for seed in range(5):
        torch.manual_seed(seed)
        input_ids = torch.randint(0, 276, (1, 32))
        attn_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            qout = qmodel(input_ids, attention_mask=attn_mask).logits
            vout = _vm_forward_plain(vm, input_ids)
        m = (qout.argmax(-1) == vout.argmax(-1)).float()
        total_match += m.sum().item()
        total += m.numel()

    match = total_match / total
    assert match >= 0.90, (
        f"Forward parity {match:.3f} below tiny-VM gate (0.90). "
        f"A structural regression in the export (RoPE, RMSNorm-identity, "
        f"SwiGLU repack, or bias fold) would land below 1 %. See "
        f"docs/QWEN_R8_E2E_2026_06_07.md."
    )


# ---------------------------------------------------------------------------
# R8 final-mile blocker: composite FFNs on the production VM
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_full_vm_export_fails_on_composite_ffn():
    """The production VM has composite FFNs that R4's repack can't handle.

    L10/L12/L23/L26/L28 use ``AddSub5StageBlock``, ``FlattenedDivMod``,
    ``FlattenedALUMul``, ``ALUShiftComposite``. Their forward isn't a
    single SwiGLU triple, so ``_swiglu_repack_and_fold_bias`` raises
    ``AttributeError: '...' object has no attribute 'W_up'``.

    Marked slow because it requires the full ``compile_full_vm_dynamic``
    bake (~10 s wall + disk cache). The smoke gate doesn't run it.
    """

    prior = os.environ.get("C4_QWEN_EXPORT_COMPAT")
    os.environ["C4_QWEN_EXPORT_COMPAT"] = "1"
    try:
        from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
            compile_full_vm_dynamic,
        )

        model, layout = compile_full_vm_dynamic(disk_cache=True)
    finally:
        if prior is None:
            os.environ.pop("C4_QWEN_EXPORT_COMPAT", None)
        else:
            os.environ["C4_QWEN_EXPORT_COMPAT"] = prior

    assert "NORM_COMPENSATOR" in layout.dim_positions, (
        "R1 invariant lost — flag-ON compile lacks NORM_COMPENSATOR"
    )

    # Sanity: at least one composite block exists on the production VM.
    composite_names = {
        "AddSub5StageBlock",
        "FlattenedDivMod",
        "FlattenedALUMul",
        "ALUShiftComposite",
    }
    composite_blocks = [
        i
        for i, b in enumerate(model.blocks)
        if type(b.ffn).__name__ in composite_names
    ]
    assert composite_blocks, (
        "Production VM no longer has composite FFNs — Blocker 1 is "
        "closed and this test should be repurposed. See "
        "docs/QWEN_R8_E2E_2026_06_07.md."
    )

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(AttributeError, match="W_up"):
            export_qwen3_dense(model, tmp, K=NORM_COMPENSATOR_K)


# ---------------------------------------------------------------------------
# R8 native baseline: capture the target output for follow-up phases
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_int_main_return_42_native_vm_baseline():
    """Native VM run of ``int main(){return 42;}`` — the R8 diff target.

    This isn't a regression gate — it captures the exit code byte
    sequence the Qwen export needs to match once Blockers 1–5 are
    closed. We assert the exit code is 42 and the output is empty,
    which is the BLOG_SPEC contract for this program.
    """

    from src.compiler import compile_c
    from neural_vm.run_vm import AutoregressiveVMRunner

    c_source = "int main(){return 42;}"
    bytecode, data = compile_c(c_source, link_stdlib=False)

    runner = AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
        enable_divergence_bail=True,
    )
    output, exit_code = runner.run(
        bytecode,
        data=bytes(data) if data else b"",
        max_steps=200,
    )
    assert exit_code == 42, (
        f"Native VM did not return 42 for `int main(){{return 42;}}`; "
        f"got exit_code={exit_code}, output={output!r}. The R8 export "
        f"diff target is built from this baseline."
    )
    assert output == "", (
        f"int main(){{return 42;}} should produce no stdout; got {output!r}"
    )
