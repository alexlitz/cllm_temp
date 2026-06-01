"""S-7: smoke test for the backbone-contribution observer tool.

Runs ``tools/observe_backbone_contributions.py`` with a tiny corpus
(``--limit 2``) and verifies the produced JSON has the expected
top-level shape. The neural VM build is disk-cached
(``C4_VM_CACHE_DIR``), so the second run completes in seconds even
though the cold build takes ~minutes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_observe_smoke():
    """Run with --limit 2 and confirm the JSON has the right shape."""
    c4_release_dir = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as td:
        output = Path(td) / "v1.json"
        env = {
            **os.environ,
            "PYTHONPATH": f".:{c4_release_dir}:{os.environ.get('PYTHONPATH', '')}",
        }
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.observe_backbone_contributions",
                "--output",
                str(output),
                "--limit",
                "2",
            ],
            check=True,
            env=env,
            cwd=str(c4_release_dir),
        )
        with open(output) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert data["corpus_size"] >= 1  # at least 1 of the 2 programs ran
        assert "bounds" in data
        assert isinstance(data["bounds"], dict)
        assert len(data["bounds"]) > 0
        # spot-check shape of one (output_dim, position_class) entry
        first_key = next(iter(data["bounds"].keys()))
        per_class = data["bounds"][first_key]
        assert isinstance(per_class, dict)
        assert len(per_class) > 0
        first_class_entry = next(iter(per_class.values()))
        assert "max_positive_contribution" in first_class_entry
        assert "max_negative_contribution" in first_class_entry
        assert "samples" in first_class_entry
