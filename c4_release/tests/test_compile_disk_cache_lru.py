"""LRU eviction tests for the ``compile_full_vm_dynamic`` on-disk cache.

The disk cache lives at ``~/.cache/c4_release/compiled_vm/`` (or
``$C4_VM_CACHE_DIR``) and each entry is ~830 MB. Without an evictor a few
compile-variant kwargs accumulate to 30+ GB. The evictor is invoked at the
tail of ``_try_save_cached``; these tests exercise the helper directly with
fake ``*.pt`` files so they run in milliseconds without invoking torch.
"""

import os
import pathlib
import time

import pytest

from c4_release.neural_vm.unified_compiler import _legacy_redirect as _static

try:
    import torch as _torch

    class _TinyPickleModel(_torch.nn.Module):
        """Module-scope so ``torch.save``/``pickle`` can locate the class."""

        def __init__(self):
            super().__init__()
            self.w = _torch.nn.Parameter(_torch.zeros(2))
except Exception:  # torch unavailable -> the dependent test skips below
    _TinyPickleModel = None  # type: ignore[assignment]


def _make_fake_entry(d: pathlib.Path, name: str, size: int, mtime: float) -> pathlib.Path:
    """Write a fake ``*.pt`` file of the requested size and set its mtime."""
    p = d / name
    with open(p, "wb") as f:
        f.write(b"\0" * size)
    os.utime(p, (mtime, mtime))
    return p


def test_evict_cache_lru_by_entry_count(tmp_path: pathlib.Path):
    """With ``max_entries=3`` and 5 entries, the 2 oldest are evicted."""
    now = time.time()
    paths = []
    for i in range(5):
        # Older mtimes for lower indices; index 4 is newest.
        paths.append(_make_fake_entry(tmp_path, f"entry_{i}.pt", 16, now - (100 - i)))

    evicted = _static._evict_cache_lru(
        tmp_path, max_bytes=0, max_entries=3
    )

    evicted_names = {p.name for p in evicted}
    assert evicted_names == {"entry_0.pt", "entry_1.pt"}, evicted_names

    remaining = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert remaining == ["entry_2.pt", "entry_3.pt", "entry_4.pt"], remaining


def test_evict_cache_lru_by_total_bytes(tmp_path: pathlib.Path):
    """With a tight byte budget, the oldest files are dropped first."""
    now = time.time()
    # 4 entries of 100 bytes each = 400 bytes total. Cap at 250 -> evict 2.
    for i in range(4):
        _make_fake_entry(tmp_path, f"entry_{i}.pt", 100, now - (50 - i))

    evicted = _static._evict_cache_lru(
        tmp_path, max_bytes=250, max_entries=0
    )

    assert {p.name for p in evicted} == {"entry_0.pt", "entry_1.pt"}
    remaining_bytes = sum(p.stat().st_size for p in tmp_path.glob("*.pt"))
    assert remaining_bytes <= 250


def test_evict_cache_lru_protects_keep_path(tmp_path: pathlib.Path):
    """The just-written file (``keep_path``) is never evicted."""
    now = time.time()
    # The "kept" entry is the oldest — without keep_path it would go first.
    keep = _make_fake_entry(tmp_path, "kept.pt", 100, now - 1000)
    _make_fake_entry(tmp_path, "other_0.pt", 100, now - 500)
    _make_fake_entry(tmp_path, "other_1.pt", 100, now - 100)

    evicted = _static._evict_cache_lru(
        tmp_path, keep_path=keep, max_bytes=0, max_entries=1
    )

    assert keep.exists(), "keep_path must not be evicted"
    assert {p.name for p in evicted} <= {"other_0.pt", "other_1.pt"}


def test_evict_cache_lru_disabled_when_caps_zero(tmp_path: pathlib.Path):
    """Both caps at 0 means no eviction even with many large entries."""
    now = time.time()
    for i in range(5):
        _make_fake_entry(tmp_path, f"entry_{i}.pt", 1000, now - (10 - i))

    evicted = _static._evict_cache_lru(
        tmp_path, max_bytes=0, max_entries=0
    )

    assert evicted == []
    assert len(list(tmp_path.glob("*.pt"))) == 5


def test_evict_cache_lru_ignores_non_pt_files(tmp_path: pathlib.Path):
    """Stale ``*.tmp`` writers or unrelated files must not be touched."""
    now = time.time()
    other = tmp_path / "stale.tmp"
    other.write_bytes(b"x" * 1000)
    os.utime(other, (now - 9999, now - 9999))

    for i in range(3):
        _make_fake_entry(tmp_path, f"entry_{i}.pt", 100, now - (10 - i))

    _static._evict_cache_lru(
        tmp_path, max_bytes=0, max_entries=1
    )

    assert other.exists(), "non-*.pt files must be left alone"


def test_evict_cache_lru_env_defaults(monkeypatch, tmp_path: pathlib.Path):
    """Env vars override the built-in defaults."""
    monkeypatch.setenv(_static._CACHE_MAX_ENTRIES_ENV, "2")
    monkeypatch.setenv(_static._CACHE_MAX_BYTES_ENV, "0")

    now = time.time()
    for i in range(4):
        _make_fake_entry(tmp_path, f"entry_{i}.pt", 100, now - (10 - i))

    # No explicit args -> pick up env values.
    evicted = _static._evict_cache_lru(tmp_path)

    assert {p.name for p in evicted} == {"entry_0.pt", "entry_1.pt"}
    assert len(list(tmp_path.glob("*.pt"))) == 2


def test_try_save_cached_triggers_eviction(monkeypatch, tmp_path: pathlib.Path):
    """A real ``_try_save_cached`` call evicts older entries.

    We pre-seed the directory with 3 fake older entries and ask
    ``_try_save_cached`` to write a real cache file (uses ``torch.save``
    under the hood) with ``max_entries=2``. Expectation: after the write
    the directory holds at most 2 entries and the new file is one of them.
    """
    pytest.importorskip("torch")
    if _TinyPickleModel is None:
        pytest.skip("torch unavailable")
    from c4_release.neural_vm.unified_compiler.layer_compiler import ModelLayout

    monkeypatch.setenv(_static._CACHE_MAX_ENTRIES_ENV, "2")
    monkeypatch.setenv(_static._CACHE_MAX_BYTES_ENV, "0")

    now = time.time()
    for i in range(3):
        _make_fake_entry(tmp_path, f"old_{i}.pt", 16, now - (1000 - i))

    target = tmp_path / "new_entry.pt"
    layout = ModelLayout(
        d_model=4,
        n_layers=1,
        ops_per_layer=[[]],
        dim_positions={},
        dim_sizes={},
        block_ops=[],
        model_ops=[],
        ffn_widths={},
    )

    _static._try_save_cached(target, _TinyPickleModel(), layout, {"k": 1})

    assert target.exists(), "new cache entry must be written"
    remaining = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert len(remaining) <= 2, remaining
    assert "new_entry.pt" in remaining
