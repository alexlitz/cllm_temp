#!/usr/bin/env python3
"""Inc-3: WIDE residual diff at the AX byte-1 predictor row (step2 off6) for
var_simple_0, GOLDEN vs CAMPAIGN. Dumps ALL named dims so we can diff which
condition the firing 0xFF emitter reads that the 30-tok frame lacks.

Writes /tmp/inc3_wide_<cfg>.json; run twice (golden, campaign) then diff.
"""
import os, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic)

SRC = "int main() { int x; x = 990; return x; }"


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "campaign" if nostk else "golden"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    _m, _layout = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_layout.dim_positions)
    # Explicit dims the ax_lea_local_addr_byte1 rule reads.
    explicit = {}
    for base, off in [("FETCH_HI", 15), ("TEMP", 10), ("OP_ENT", 0),
                      ("OP_LEA", 0), ("H1", 1), ("CLEAN_EMBED_LO", 8),
                      ("CLEAN_EMBED_HI", 14), ("BYTE_INDEX_0", 0),
                      ("FETCH_LO", 0), ("FETCH_HI", 0)]:
        if base in dp:
            explicit[f"{base}+{off}" if off else base] = dp[base] + off
    # All single-cell dims (skip the .*.-1 cross-step aliases - same base dim).
    dim_names = {k: v for k, v in dp.items() if ".*." not in k}
    dim_names.update(explicit)
    STEP = int(Token.STEP_TOKENS)
    last_block = len(p.runner.model.blocks) - 1
    prompt_len = len(p._build_context(bytecode))
    pos = prompt_len + 2 * STEP + 6  # step2 byte-1 predictor row (off 6)
    vals = p.residual_at(bytecode, last_block, pos, dim_names, max_steps=6)
    out = {k: round(v, 3) for k, v in vals.items() if abs(v) > 0.05}
    json.dump(out, open(f"/tmp/inc3_wide_{cfg}.json", "w"), indent=0)
    keyv = {k: round(vals.get(k, 0.0), 3) for k in explicit}
    print(f"[{cfg}] step2 off6: {keyv}")


if __name__ == "__main__":
    main()
