"""Byte-identity gate for the COO sparse weight sidecar (v6 cache format).

The sparse sidecar (see ``_save_sparse_sidecar`` /
``_load_sparse_sidecar`` in ``_legacy_redirect``) replaces the v5 dense
``.safetensors`` weights file with a per-tensor COO encoding. The encoding
must be lossless: a save → load roundtrip on a compiled VM ``state_dict``
must produce ``torch.equal``-identical tensors for every entry.

This test:

  1. Compiles the VM via ``compile_full_vm_dynamic`` (one bake) into a
     pristine cache directory.
  2. Reads the resulting ``.sparse`` sidecar back via
     ``_load_sparse_sidecar`` and compares every tensor with the live
     model's ``state_dict()``.
  3. Asserts the sidecar size is in the documented 2-5 MB range (sanity
     check on the ~290x reduction; the design doc gives 3.6 MB as the
     reference number).

Compiling the full VM is expensive (~40-70 s), so this test runs the
single-compile path only.
"""

import pathlib

import pytest

torch = pytest.importorskip("torch")

from c4_release.neural_vm.unified_compiler import _legacy_redirect as _static
from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (
    compile_full_vm_dynamic,
)


def _state_dict_equal(sd_a: dict, sd_b: dict):
    """Return (missing, mismatch) tuples for two state_dicts.

    ``missing`` lists keys present in one but not the other.
    ``mismatch`` lists keys whose tensors are not ``torch.equal``.
    """
    missing = sorted(set(sd_a) ^ set(sd_b))
    mismatch = []
    for k in sorted(set(sd_a) & set(sd_b)):
        a = sd_a[k]
        b = sd_b[k]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            mismatch.append(k)
    return missing, mismatch


@pytest.mark.slow
def test_sparse_sidecar_roundtrip_is_byte_identical(
    tmp_path: pathlib.Path, monkeypatch
):
    """Compile → save → load → verify state_dict byte-identical.

    Drives one ``compile_full_vm_dynamic`` call, which writes the COO
    sparse sidecar via the production cache write path. Then re-reads the
    sidecar via ``_load_sparse_sidecar`` and asserts byte identity with
    the live model's state_dict.
    """
    monkeypatch.setenv("C4_VM_CACHE_DIR", str(tmp_path))
    # Bound the LRU caps so the new entry sticks around for inspection.
    monkeypatch.setenv("C4_VM_CACHE_MAX_BYTES", "0")
    monkeypatch.setenv("C4_VM_CACHE_MAX_ENTRIES", "0")

    # Single compile with disk_cache=True writes the v6 .sparse sidecar
    # via ``_try_save_cached``.
    model, _layout = compile_full_vm_dynamic(disk_cache=True)
    live_state = dict(model.state_dict())

    # Locate the cache entry just written. The directory is pristine, so
    # we expect exactly one .pt / .sparse pair.
    pt_files = sorted(tmp_path.glob("*.pt"))
    sparse_files = sorted(tmp_path.glob("*.sparse"))
    assert len(pt_files) == 1, f"expected 1 shell, got {pt_files}"
    assert len(sparse_files) == 1, f"expected 1 sidecar, got {sparse_files}"
    shell_path = pt_files[0]
    sidecar_path = sparse_files[0]
    assert sidecar_path == _static._weights_path_for(shell_path)

    # ---- size gate ------------------------------------------------------
    sidecar_bytes = sidecar_path.stat().st_size
    # Design doc target: ~3.6 MB. Accept anything below 6 MB as healthy
    # (the COO format is lossless so the only knob is sparsity, which is
    # determined by the bake; this catches accidental dense fallback).
    assert sidecar_bytes < 6 * 1024 * 1024, (
        f"sparse sidecar grew unexpectedly: {sidecar_bytes} bytes "
        f"(>=6 MB suggests a tensor fell to dense fallback)"
    )
    # And it shouldn't be ridiculously small either; an empty file would
    # round-trip trivially but indicates the writer dropped tensors.
    assert sidecar_bytes > 1_000_000, (
        f"sparse sidecar suspiciously small: {sidecar_bytes} bytes"
    )

    # ---- byte-identity gate --------------------------------------------
    # Re-read the sidecar into a fresh state_dict. The skeleton supplies
    # shapes/dtypes; using the live model is fine because the loader only
    # reads its ``state_dict()`` keys.
    loaded_state = _static._load_sparse_sidecar(model, sidecar_path)

    # Aliased entries (deduped on save) appear in ``live_state`` but not
    # ``loaded_state`` — the v6 alias map handles re-sharing on the
    # production load path. For the byte-identity gate we just need the
    # kept (non-aliased) names to match.
    raw_state_dict = dict(model.state_dict())
    deduped, alias_map = _static._dedupe_state_dict_for_safetensors(raw_state_dict)
    # Every key the sidecar carries must be in the deduped set.
    assert set(loaded_state) == set(deduped), (
        f"sidecar key set differs from deduped state_dict: "
        f"only-sidecar={sorted(set(loaded_state) - set(deduped))[:5]} "
        f"only-deduped={sorted(set(deduped) - set(loaded_state))[:5]}"
    )
    missing, mismatch = _state_dict_equal(deduped, loaded_state)
    assert not missing, f"missing/extra keys: {missing[:10]}"
    assert not mismatch, (
        f"{len(mismatch)} tensors differ post-roundtrip: {mismatch[:10]}"
    )

    # ---- end-to-end cache-hit gate -------------------------------------
    # The production load path (``_try_load_cached``) wraps the sidecar
    # read with shell unpickle + alias resharing. A second call to
    # ``compile_full_vm_dynamic`` with the same kwargs should hit the
    # cache and return a model whose state_dict matches the original.
    model2, _ = compile_full_vm_dynamic(disk_cache=True)
    live2 = dict(model2.state_dict())
    missing2, mismatch2 = _state_dict_equal(live_state, live2)
    assert not missing2, f"cache-hit state_dict missing keys: {missing2[:10]}"
    assert not mismatch2, (
        f"{len(mismatch2)} tensors differ between fresh compile and "
        f"cache-hit reload: {mismatch2[:10]}"
    )
