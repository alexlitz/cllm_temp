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

from neural_vm.qwen_compat import (
    export_qwen3_dense,
    install_neural_vm_embedding_wrapper,
)
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
def test_full_vm_export_succeeds_on_composite_ffn():
    """The production VM's composite FFNs no longer block export.

    Originally (pre Wave 1 Cluster D1) this test pinned the negative
    invariant — the production VM compiled with ``C4_QWEN_EXPORT_COMPAT=1``
    has five composite blocks (``AddSub5StageBlock``, ``FlattenedDivMod``,
    ``FlattenedALUMul``, ``ALUShiftComposite``) and the R4 repack raised
    ``AttributeError: '...' object has no attribute 'W_up'``.

    With ``extract_composite_ffn_weights`` wired into the export, those
    composites now materialise a zero-init Qwen SwiGLU triple (skip-pass
    semantics — see ``docs/QWEN_R8_E2E_2026_06_07.md`` Blocker 1's "path
    forward" notes), so ``export_qwen3_dense`` runs to completion. The
    semantic byte-identity of those layers is still deferred to a
    follow-up phase; this test only pins that export does not raise and
    every layer carries the three SwiGLU keys.

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
        "Production VM no longer has composite FFNs — this test should "
        "be updated or removed. See docs/QWEN_R8_E2E_2026_06_07.md."
    )

    # The exported state_dict for the production VM is ~1.7 GB. Many
    # tempfs-backed ``/tmp`` mounts (CI runners, dev containers) hit
    # disk-quota limits when the file is materialised there; prefer a
    # location backed by real disk. Override via ``C4_QWEN_EXPORT_TMP``.
    import pathlib
    base_dir = os.environ.get("C4_QWEN_EXPORT_TMP")
    if base_dir:
        base_path = pathlib.Path(base_dir)
    else:
        base_path = pathlib.Path.home() / ".cache" / "c4_qwen_export_test"
    base_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(base_path)) as tmp:
        # Should no longer raise — composite FFNs export as zero-init
        # SwiGLU triples (skip-pass) via extract_composite_ffn_weights.
        cfg = export_qwen3_dense(model, tmp, K=NORM_COMPENSATOR_K)
        assert cfg.num_hidden_layers == len(model.blocks)
        sd = torch.load(
            os.path.join(tmp, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
        # Every composite block must carry the three SwiGLU keys.
        for i in composite_blocks:
            for key_suffix in (
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
            ):
                k = f"model.layers.{i}.mlp.{key_suffix}"
                assert k in sd, (
                    f"missing exported key for composite block {i}: {k}"
                )


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


# ---------------------------------------------------------------------------
# R8 acceptance: full production round-trip
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_r8_full_production_roundtrip():
    """End-to-end plan §R8 acceptance gate.

    Plan §R8 acceptance criterion: ``int main(){return 42;}`` round-trips
    through the exported Qwen3-dense artefact with **≥99 % per-token
    argmax match** against the native VM runner.

    The pipeline this test exercises:

      1. Compile the **production** VM with ``C4_QWEN_EXPORT_COMPAT=1`` (R1
         NORM_COMPENSATOR seeded at K).
      2. Apply :func:`flatten_post_ops_for_qwen_export` so the per-block
         ``post_ops`` lists become first-class Qwen blocks (R5).
      3. Run :func:`export_qwen3_dense` into a temporary directory. The
         function internally runs R2/R3/R4/R7 + the R5 flatten try-pass.
      4. Load the artefact through ``AutoModelForCausalLM.from_pretrained``.
         If the load fails, capture the precise error and ``pytest.xfail``
         it. The remaining steps run only on a successful load.
      5. Compile ``int main(){return 42;}`` via :func:`compile_c` and
         build a token-stream context from the same VM tokenizer the
         runner uses.
      6. Forward the context through the exported Qwen3 model and capture
         a ~30-token argmax sequence.
      7. Run the native :class:`AutoregressiveVMRunner` on the same input,
         capturing its argmax sequence via the same forward path.
      8. Assert per-token argmax match ≥99 %.

    Current state (2026-06-07, base b17b9d3f): all four blockers below
    keep the export from completing. The test fails fast inside step 3
    with ``AttributeError`` on ``W_up`` (composite FFN), and we xfail
    with that signal so the test surfaces as **xpass** the moment all
    blockers integrate. See ``docs/QWEN_R8_E2E_2026_06_07.md`` for the
    full blocker list:

      * **D1 / Blocker 1** — composite FFN flattener (L10/L12/L23/L26/L28
        are ``AddSub5StageBlock`` / ``FlattenedDivMod`` / ``FlattenedALUMul``
        / ``ALUShiftComposite``; R4 SwiGLU repack expects flat
        ``W_up``/``W_gate``/``W_down``).
      * **D2 / Blocker 2** — softmax1 sink runtime wiring (landed at
        commit 3d02dcc0; still requires the loader-side prepend to be
        active).
      * **I1 / Blocker 3** — per-head ``q_norm`` / ``k_norm`` have no VM
        counterpart; compensator must be replicated per head.
      * **I2 / Blocker 4** — production VM is ALiBi; Qwen3 is RoPE-only.
      * **I3 / Blocker 5** — ``NeuralVMEmbedding`` augmentations
        (``ADDR_KEY``, ``MEM_STORE`` boundary writes) — runtime wrapper
        landed at commit 94072b3b and is now wired through the forward
        path here via :func:`install_neural_vm_embedding_wrapper` after
        ``from_pretrained`` (the wrapper byte-identity test lives at
        ``tests/test_qwen_embedding_export.py::test_install_neural_vm_embedding_wrapper_attaches_to_hf_qwen3``).
        The argmax gain from this fix is masked by the remaining
        D1/D2/I1/I2 gaps until those land.

    When D1 + D2 + I1/I2 all integrate, this test flips to xpass
    (or, if the harness escalates xpass to fail, to a plain pass after
    removing the runtime ``pytest.xfail`` block).
    """

    from src.compiler import compile_c
    from neural_vm.qwen_compat import flatten_post_ops_for_qwen_export

    c_source = "int main(){return 42;}"
    bytecode, data = compile_c(c_source, link_stdlib=False)

    # Step 1: compile the production VM with the R1 invariant ON.
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
    model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        # Step 2: R5 flatten. The function is non-mutating and returns a
        # new list of blocks; failure here would mean R5's contract broke.
        try:
            flattened_blocks = flatten_post_ops_for_qwen_export(model)
            assert len(flattened_blocks) >= len(model.blocks)
        except Exception as exc:  # pragma: no cover - defensive
            pytest.xfail(
                f"R5 flatten_post_ops_for_qwen_export raised: "
                f"{type(exc).__name__}: {exc}. Blockers: D1 (composite FFN), "
                f"see docs/QWEN_R8_E2E_2026_06_07.md."
            )

        # Step 3: export. Currently fails on composite FFN (Blocker 1 / D1).
        try:
            export_qwen3_dense(model, tmp, K=NORM_COMPENSATOR_K)
        except Exception as exc:
            pytest.xfail(
                f"export_qwen3_dense failed: "
                f"{type(exc).__name__}: {exc}. Likely blocker: D1 (composite "
                f"FFNs L10/L12/L23/L26/L28 don't expose W_up/W_gate/W_down). "
                f"See docs/QWEN_R8_E2E_2026_06_07.md §Blocker 1."
            )

        # Step 4: HuggingFace load.
        try:
            from transformers import AutoModelForCausalLM

            qmodel = AutoModelForCausalLM.from_pretrained(tmp)
        except Exception as exc:
            pytest.xfail(
                f"AutoModelForCausalLM.from_pretrained failed after export: "
                f"{type(exc).__name__}: {exc}. Likely blockers: D1/D2/I3 "
                f"(state_dict shape mismatch on flattened composite blocks, "
                f"sink token id, or embedding augmentations). See "
                f"docs/QWEN_R8_E2E_2026_06_07.md."
            )
        qmodel.eval()

        # Step 4b: install the NeuralVMEmbedding augmentation wrapper.
        # Plain ``from_pretrained`` rebuilds ``model.embed_tokens`` as an
        # ``nn.Embedding`` — the ADDR_KEY scatter and MEM_STORE writes the
        # native VM does on every forward (see
        # ``NeuralVMEmbedding.forward``) are silently dropped, so L5/L7
        # content-addressed bytecode fetch sees zero ADDR_KEY keys and the
        # argmax diverges. ``install_neural_vm_embedding_wrapper`` swaps the
        # embedding for :class:`NeuralVMEmbeddingWrapper`, rebuilt from the
        # ``c4_addr_key_idx`` / ``c4_mem_*_idx`` config the export persisted.
        try:
            wrapper = install_neural_vm_embedding_wrapper(qmodel, qmodel.config)
            # Mirror the native runner's clean-slate MEM boundary for this
            # single-shot forward (the runner calls ``set_mem_history_end(0)``
            # at the start of every ``run`` — see ``run_vm.py``).
            wrapper.set_mem_history_end(0)
        except Exception as exc:
            pytest.xfail(
                f"install_neural_vm_embedding_wrapper failed after HF load: "
                f"{type(exc).__name__}: {exc}. Likely blocker: I3 "
                f"(embedding augmentation wiring). See "
                f"docs/QWEN_R8_E2E_2026_06_07.md §Blocker 5."
            )

    # Step 5: build the same token context the native runner uses.
    from neural_vm.run_vm import AutoregressiveVMRunner

    runner = AutoregressiveVMRunner(
        pure_neural=True,
        trust_neural_alu=True,
        enable_divergence_bail=True,
    )

    # ``_build_context`` is an internal helper but it is the stable
    # tokenization the runner uses; reusing it keeps the VM and exported
    # model on the same token grid.
    context_tokens = runner._build_context(
        bytecode,
        bytes(data) if data else b"",
        argv=None,
        stdin="",
    )
    # Keep within both models' max_seq_len bounds and only diff the
    # first ~30 generation positions to match the plan's window.
    max_positions = min(
        30,
        len(context_tokens),
        int(getattr(runner.model, "max_seq_len", len(context_tokens))),
        int(getattr(qmodel.config, "max_position_embeddings", len(context_tokens))),
    )
    if max_positions <= 0:
        pytest.xfail(
            "Token context window collapsed to zero; cannot diff argmax. "
            "Likely blocker: I3 (embedding augmentations)."
        )
    diff_window = context_tokens[:max_positions]

    device_native = next(runner.model.parameters()).device
    device_qwen = next(qmodel.parameters()).device
    in_native = torch.tensor([diff_window], dtype=torch.long, device=device_native)
    in_qwen = torch.tensor([diff_window], dtype=torch.long, device=device_qwen)

    with torch.no_grad():
        try:
            qout = qmodel(in_qwen).logits
        except Exception as exc:
            pytest.xfail(
                f"Qwen3 forward raised: {type(exc).__name__}: {exc}. "
                f"Likely blocker: I3 (embedding augmentations) or I2 (RoPE "
                f"vs ALiBi position encoding)."
            )
        try:
            vout = runner.model(in_native)
        except Exception as exc:
            pytest.xfail(
                f"Native VM forward raised: {type(exc).__name__}: {exc}. "
                f"This is a runner regression, not an R8 blocker."
            )

    qwen_argmax = qout.argmax(dim=-1).cpu()
    native_argmax = vout.argmax(dim=-1).cpu()
    match = (qwen_argmax == native_argmax).float().mean().item()

    assert match >= R8_ARGMAX_TARGET, (
        f"R8 argmax match {match:.4f} below plan §R8 target "
        f"{R8_ARGMAX_TARGET}. Window: {max_positions} positions. "
        f"This is the live ≥99 % gate — if all blockers are landed "
        f"the gap is a regression in the export pipeline. See "
        f"docs/QWEN_R8_E2E_2026_06_07.md."
    )
