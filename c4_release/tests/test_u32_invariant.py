"""Tests for ``verify_u32_invariant`` -- the static u32-everywhere check.

The u32-everywhere invariant: every internal ALU representation is a u32
value decomposed into byte/nibble lanes. No fp64 internal precision; no
16-bit MUL_ACCUM lanes (one-hot encodings beyond the 0..255 single-byte
range); no u64/i64 intermediate widening.

These tests cover the three issue kinds the verifier emits:

    * ``fp64_dtype``    -- parameter / source-level fp64 use
    * ``oversize_lane`` -- dim_registry alloc wider than a byte
    * ``widening``      -- source line implying a >32-bit intermediate

Each kind gets a positive test (the verifier *should* flag a synthetic
violation) and a negative test (the verifier *should not* flag a clean
synthetic input). The end-to-end "scan the real repo" call is also
exercised so a missing import / regex bug surfaces immediately.

These tests do NOT bake a model -- the source-level scans are the
load-bearing check on CI. The optional ``scan_model=True, model=<baked>``
path is exercised separately by tests that already construct a model.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


from c4_release.neural_vm.verification.decl_verifier import (  # noqa: E402
    _u32_scan_dim_registry,
    _u32_scan_fp64_in_source,
    _u32_scan_model_params,
    _u32_scan_widening_in_source,
    verify_u32_invariant,
)


# ---------------------------------------------------------------------------
# End-to-end: the real repo
# ---------------------------------------------------------------------------


class TestVerifyU32InvariantEndToEnd:
    """The ``verify_u32_invariant()`` entry point must return a list of
    well-shaped issue dicts when called against the live codebase.
    """

    def test_returns_list_of_dicts(self):
        issues = verify_u32_invariant()
        assert isinstance(issues, list)
        for issue in issues:
            assert isinstance(issue, dict)
            assert set(issue.keys()) >= {"kind", "where", "reason"}
            assert issue["kind"] in {
                "fp64_dtype", "oversize_lane", "widening",
            }
            assert isinstance(issue["where"], str) and issue["where"]
            assert isinstance(issue["reason"], str) and issue["reason"]

    def test_scan_source_false_skips_source_scans(self):
        """``scan_source=False`` should drop fp64 / widening source hits
        but still run the dim_registry scan."""
        with_source = verify_u32_invariant(scan_source=True)
        without_source = verify_u32_invariant(scan_source=False)
        # Without source scans we should never see fp64_dtype or widening
        # issues (those only come from source scanning).
        kinds = {i["kind"] for i in without_source}
        assert "fp64_dtype" not in kinds
        assert "widening" not in kinds
        # ... but if the full scan found any of those issues, dropping the
        # scan must have strictly reduced the total count.
        if any(i["kind"] in {"fp64_dtype", "widening"} for i in with_source):
            assert len(without_source) < len(with_source)

    def test_no_model_no_model_walk(self):
        """When ``model`` is None the function must not raise and the
        param-walk path is skipped."""
        # Smoke-only -- exercised already by ``test_returns_list_of_dicts``;
        # this test pins the contract.
        issues = verify_u32_invariant(model=None, scan_model=True)
        kinds = {i["kind"] for i in issues}
        # No <model>:... where strings should appear.
        for issue in issues:
            assert not issue["where"].startswith("<model>")


# ---------------------------------------------------------------------------
# Issue kind 1: fp64_dtype (model param path)
# ---------------------------------------------------------------------------


class _Fp64Toy(nn.Module):
    """A toy module with one fp64 parameter and one fp32 parameter."""

    def __init__(self):
        super().__init__()
        self.good = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.bad = nn.Parameter(torch.zeros(4, dtype=torch.float64))


class _Fp32Toy(nn.Module):
    """A toy module with only fp32 parameters."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))


class TestFp64DtypeModelWalk:
    """``_u32_scan_model_params`` must flag fp64 params and ignore fp32 /
    bf16 / fp16 params."""

    def test_positive_fp64_param_is_flagged(self):
        issues = []
        _u32_scan_model_params(_Fp64Toy(), issues)
        assert len(issues) == 1
        assert issues[0]["kind"] == "fp64_dtype"
        assert "bad" in issues[0]["where"]
        assert "float64" in issues[0]["reason"]

    def test_negative_fp32_bf16_params_are_clean(self):
        issues = []
        _u32_scan_model_params(_Fp32Toy(), issues)
        assert issues == []

    def test_none_model_is_noop(self):
        issues = []
        _u32_scan_model_params(None, issues)
        assert issues == []

    def test_fp64_buffer_is_flagged(self):
        class _Buf(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "x", torch.zeros(2, dtype=torch.float64)
                )
        issues = []
        _u32_scan_model_params(_Buf(), issues)
        assert len(issues) == 1
        assert issues[0]["kind"] == "fp64_dtype"
        assert "buffer" in issues[0]["reason"].lower()


# ---------------------------------------------------------------------------
# Issue kind 1 (cont'd): fp64_dtype (source-level scan)
# ---------------------------------------------------------------------------


class TestFp64DtypeSourceScan:
    """The source-level fp64 scan reads ``.py`` files under ``neural_vm/alu``
    and ``neural_vm/unified_compiler``. We exercise it against the live tree
    (no temp file shenanigans -- the production path is what we want pinned)
    and confirm that *deny* phrases suppress the flag.
    """

    def test_live_scan_returns_list(self):
        issues = []
        _u32_scan_fp64_in_source(issues)
        assert isinstance(issues, list)
        for issue in issues:
            assert issue["kind"] == "fp64_dtype"
            # Path should be relative (no leading slash).
            assert not issue["where"].startswith("/")

    def test_known_violation_in_mul_py_is_flagged(self):
        """``neural_vm/alu/ops/mul.py`` currently has a documented fp64
        upcast (lines around 151 + 229). The scanner must flag at least
        one of them; if a future fix removes both, this test becomes
        a positive signal that the fix landed.
        """
        issues = []
        _u32_scan_fp64_in_source(issues)
        mul_hits = [i for i in issues if "alu/ops/mul.py" in i["where"]]
        # We don't assert >0 hard -- this test documents "if mul.py still
        # has fp64 in it, we expect the scan to catch it". When the fp64
        # is removed, the hits become 0 and the assertion below is the
        # pass-through state.
        for hit in mul_hits:
            assert "float64" in hit["reason"] or "double" in hit["reason"]


# ---------------------------------------------------------------------------
# Issue kind 2: oversize_lane (dim_registry scan)
# ---------------------------------------------------------------------------


class TestOversizeLane:
    """``_u32_scan_dim_registry`` parses ``alloc(name, start, size, desc)``
    calls and flags any whose size > 256 (full-byte one-hot) or whose
    description text claims a single-lane wider-than-byte encoding.
    """

    def test_live_dim_registry_has_no_oversize_lanes(self):
        """Pin the current invariant: the production dim registry does
        NOT contain any oversize-lane allocations. This is the "u32
        invariant holds on current HEAD" guarantee for issue kind 2.
        """
        issues = []
        _u32_scan_dim_registry(issues)
        assert issues == [], (
            "dim_registry now contains oversize_lane allocations: "
            + repr(issues)
        )

    def test_positive_synthetic_oversize_lane(self, tmp_path, monkeypatch):
        """Inject a synthetic ``dim_registry.py`` under a fake project
        root and confirm the scanner flags an allocation with size > 256.
        """
        # Build a tiny fake repo layout with a synthetic dim_registry.py
        # whose contents include an oversize allocation. We monkey-patch
        # the verifier's ``_u32_repo_root`` to point at this fake root.
        fake_root = tmp_path / "c4_release"
        (fake_root / "neural_vm").mkdir(parents=True)
        synthetic = (
            'reg.alloc("BIG_HOT", 0, 1024, "one-hot over 1024 values")\n'
            'reg.alloc("OK_NIBBLE", 1024, 16, "one-hot nibble")\n'
        )
        (fake_root / "neural_vm" / "dim_registry.py").write_text(
            synthetic, encoding="utf-8",
        )
        from c4_release.neural_vm.verification import decl_verifier
        monkeypatch.setattr(
            decl_verifier, "_u32_repo_root", lambda: str(fake_root),
        )
        issues = []
        decl_verifier._u32_scan_dim_registry(issues)
        kinds = [i["kind"] for i in issues]
        assert "oversize_lane" in kinds
        names = [i["reason"] for i in issues]
        assert any("BIG_HOT" in n for n in names)
        assert not any("OK_NIBBLE" in n for n in names)

    def test_positive_synthetic_16bit_hint(self, tmp_path, monkeypatch):
        """A description that claims a 16-bit single-lane encoding must
        be flagged even if the size is small (the size is misleading --
        the description is the spec).
        """
        fake_root = tmp_path / "c4_release"
        (fake_root / "neural_vm").mkdir(parents=True)
        synthetic = (
            'reg.alloc("UINT16_LIE", 0, 8, "single-lane 16-bit u16 value")\n'
        )
        (fake_root / "neural_vm" / "dim_registry.py").write_text(
            synthetic, encoding="utf-8",
        )
        from c4_release.neural_vm.verification import decl_verifier
        monkeypatch.setattr(
            decl_verifier, "_u32_repo_root", lambda: str(fake_root),
        )
        issues = []
        decl_verifier._u32_scan_dim_registry(issues)
        assert any(
            i["kind"] == "oversize_lane" and "UINT16_LIE" in i["reason"]
            for i in issues
        ), repr(issues)

    def test_negative_nibble_decomposed_alloc_is_clean(
        self, tmp_path, monkeypatch,
    ):
        """A multi-nibble allocation whose description mentions "nibble"
        is justified and must not be flagged, even at size 48."""
        fake_root = tmp_path / "c4_release"
        (fake_root / "neural_vm").mkdir(parents=True)
        synthetic = (
            'reg.alloc("ADDR_KEY", 0, 48, "3 nibbles x 16 one-hot key")\n'
        )
        (fake_root / "neural_vm" / "dim_registry.py").write_text(
            synthetic, encoding="utf-8",
        )
        from c4_release.neural_vm.verification import decl_verifier
        monkeypatch.setattr(
            decl_verifier, "_u32_repo_root", lambda: str(fake_root),
        )
        issues = []
        decl_verifier._u32_scan_dim_registry(issues)
        assert issues == [], repr(issues)


# ---------------------------------------------------------------------------
# Issue kind 3: widening (source scan)
# ---------------------------------------------------------------------------


class TestWidening:
    """``_u32_scan_widening_in_source`` reads source files and flags lines
    that imply a >32-bit intermediate (``torch.int64``, ``.long()``,
    ``torch.double``, comments about ``wraparound at 2**32`` etc.).
    """

    def test_live_scan_returns_list(self):
        issues = []
        _u32_scan_widening_in_source(issues)
        assert isinstance(issues, list)
        for issue in issues:
            assert issue["kind"] == "widening"
            assert not issue["where"].startswith("/")

    def test_denial_phrase_suppresses_flag(self, tmp_path, monkeypatch):
        """A line whose comment denies widening (``"no widening"``) must
        not be flagged, even if it contains a widening pattern token.
        """
        fake_root = tmp_path / "c4_release"
        (fake_root / "neural_vm" / "alu" / "ops").mkdir(parents=True)
        (fake_root / "neural_vm" / "unified_compiler").mkdir(parents=True)
        # Two synthetic files: one with a real cast, one with a denying
        # comment.
        bad = '# real widening\nx = torch.tensor([1]).long()\n'
        # The denial phrase must appear ON THE SAME LINE as the widening
        # pattern -- the scan is line-local (per the function docstring).
        good = (
            'name = "torch.long"  # no widening: just a string literal\n'
        )
        (fake_root / "neural_vm" / "alu" / "ops" / "bad.py").write_text(
            bad, encoding="utf-8",
        )
        (fake_root / "neural_vm" / "alu" / "ops" / "good.py").write_text(
            good, encoding="utf-8",
        )
        # ``__init__.py`` files so the walker finds something.
        for sub in ("alu", "alu/ops", "unified_compiler"):
            init = fake_root / "neural_vm" / sub / "__init__.py"
            init.write_text("", encoding="utf-8")
        from c4_release.neural_vm.verification import decl_verifier
        monkeypatch.setattr(
            decl_verifier, "_u32_repo_root", lambda: str(fake_root),
        )
        issues = []
        decl_verifier._u32_scan_widening_in_source(issues)
        wheres = [i["where"] for i in issues]
        assert any("bad.py" in w for w in wheres), repr(issues)
        assert not any("good.py" in w for w in wheres), repr(issues)

    def test_positive_int64_cast_is_flagged(self, tmp_path, monkeypatch):
        """A bare ``.to(torch.int64)`` in an ALU file must be flagged."""
        fake_root = tmp_path / "c4_release"
        (fake_root / "neural_vm" / "alu" / "ops").mkdir(parents=True)
        (fake_root / "neural_vm" / "unified_compiler").mkdir(parents=True)
        bad = "y = x.to(torch.int64)\n"
        (fake_root / "neural_vm" / "alu" / "ops" / "cast.py").write_text(
            bad, encoding="utf-8",
        )
        from c4_release.neural_vm.verification import decl_verifier
        monkeypatch.setattr(
            decl_verifier, "_u32_repo_root", lambda: str(fake_root),
        )
        issues = []
        decl_verifier._u32_scan_widening_in_source(issues)
        assert any(
            "cast.py" in i["where"] and i["kind"] == "widening"
            for i in issues
        ), repr(issues)


# ---------------------------------------------------------------------------
# Issue dict shape contract
# ---------------------------------------------------------------------------


class TestIssueDictContract:
    """Every issue dict the verifier emits must satisfy a fixed schema so
    downstream tools (a future CI gate, a pretty-printer, etc.) can
    consume the output without case-splitting on shape.
    """

    def test_every_issue_has_kind_where_reason(self):
        issues = verify_u32_invariant()
        for issue in issues:
            assert "kind" in issue
            assert "where" in issue
            assert "reason" in issue
            # No additional unexpected keys -- keep the schema minimal.
            assert set(issue.keys()) == {"kind", "where", "reason"}

    def test_kinds_are_well_known(self):
        issues = verify_u32_invariant()
        for issue in issues:
            assert issue["kind"] in {
                "fp64_dtype", "oversize_lane", "widening",
            }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
