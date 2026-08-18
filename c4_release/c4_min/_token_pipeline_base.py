"""TOKEN-PIPELINED BASE VM — the un-built half of the literal-0.5B whole-ISA fit
(#923, closing #922/#916).

The sibling ``_token_pipeline_divmod`` (#922) proved the DIVMOD half fits a literal
0.5B (hidden 896, <=24 layers/token, 6 tokens/DIV) byte-exact by SLICING the wide
log-sink scratch across the VM step's emission tokens and carrying the boundary
values across via the real softmax1+ALiBi KV-memory CAM.

THIS module builds the OTHER half: the BASE VM (fetch/decode/PC/AX/SP/BP/STACK0/
OP_IS/mem-CAM/ALU-add-sub/mul/cmp/bitwise/dispatch/branch/fold — everything EXCEPT
the DIV/MOD long-division bank).  The remaining gap the task named was a "liveness
PROJECTION" (349 dims = 163 persistent + 186 max-live scratch), NOT built.  Here we
BUILD it and MEASURE the actual constructed per-token residual:

  * The base VM is exactly the base blocks of ``qwen_full_vm._block_specs`` with the
    whole-ISA layout (L.D=2926) — but those 18 base FFN blocks only TOUCH 327 distinct
    residual dims (measured, not projected).  Their max-simultaneously-live footprint
    is 210 dims (measured over the base block order).  Both are << 896.
  * We REMAP the base footprint into a COMPACT residual (each touched dim -> a dense
    compact index) so every base block runs at a per-token d_model of exactly the
    compact width — a pure column/row selection of the weights, so the SwiGLU FFN math
    is BYTE-IDENTICAL to running on the full 2926-dim residual.
  * We then SLICE the base blocks across tokens (<= cap FFN layers/token) and carry the
    live scratch across each token boundary via the SAME byte-exact ``kv_gather`` CAM
    the divmod pipeline uses (unique address per live value).  The persistent VM state
    (PC/AX/SP/BP/STACK0 + ONE) stays resident and re-seeded each token.

MEASURED, off-build-path.  Golden 174ece66 UNTOUCHED (NEW file, no build-path change;
fingerprint verified before + after).  Sparse-free tiny fp32 residual; poll RSS, abort
> 4 GB (the block specs are small; the whole thing runs in a few hundred MB).
"""
from __future__ import annotations

import os
import resource
import threading
import time
from typing import Dict, List, Set, Tuple

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

# Reuse the byte-exact CAM gather + RSS helpers from the divmod pipeline.
from ._token_pipeline_divmod import kv_gather, rss_mb, start_watchdog  # noqa: F401
from ._measure_critpath_921 import _block_rw


# ===========================================================================
# BASE block extraction from the whole-ISA block list.
# ===========================================================================
def _is_divmod_block(name: str) -> bool:
    """The DIV/MOD long-division bank ONLY (radix-16 lean 'lean-*') — the wide
    digit-recurrence bank whose fp32 form is only ~2^16-exact and which the #922
    divmod token pipeline replaces with the full-32-bit sliced log-sink divide.

    MUL ('alu-mul-*') STAYS in the base VM: it is a base op, byte-exact in the single
    forward, and its 8 schoolbook blocks fold into the base compact residual (411 dims,
    <=896) with base critpath 19 (<=24).  SHL/SHR route through MUL/DIV via
    shift_via_mul, so the base ax-mux/dispatch read the MUL/DIV RES bands unchanged."""
    return name.startswith("lean-")


def build_base_blocks():
    """Return ``(base_specs, L, QL)`` — the base VM FFN blocks (name, spec) extracted
    from the whole-ISA ``qwen_full_vm._block_specs`` (SUBSET_FULL, radix-16-lean divmod,
    shift-via-mul), plus the whole-ISA layout ``L`` they address."""
    from . import qwen_full_vm as Q
    subset = Q.SUBSET_FULL
    QL = Q.QwenFullLayout(24, subset, efficient_alu=True, recurrent_divmod=False,
                          code_from_memory=True, shift_via_mul=True, div_logsink=False)
    L = QL.L
    specs = Q._block_specs(L, 24, subset, efficient_alu=True, recurrent_divmod=False,
                           code_from_memory=True, shift_via_mul=QL.shift_via_mul,
                           div_logsink=QL.div_logsink)
    base = [(n, s) for n, s in specs if not _is_divmod_block(n)]
    return base, L, QL


# ===========================================================================
# COMPACT REMAP: full 2926-dim residual -> a dense compact residual of exactly the
# base footprint width.  A pure column(read)/row(write) selection of the FFN weights,
# so the SwiGLU math is BYTE-IDENTICAL on the compacted residual.
# ===========================================================================
class BaseCompactMap:
    def __init__(self, base_specs, L):
        foot: Set[int] = set()
        for _, s in base_specs:
            r, w = _block_rw(s)
            foot |= r
            foot |= w
        foot.add(L.ONE)  # the bias/const lane must survive
        self.dims = sorted(foot)
        self.D = len(self.dims)                    # CONSTRUCTED compact per-token width
        self.old2new = {d: i for i, d in enumerate(self.dims)}
        self.ONE = self.old2new[L.ONE]
        # persistent VM-state dims (resident every token) + their compact indices
        self.persistent_old: List[int] = []
        for nm in ("PC", "AX", "SP", "BP", "STACK0"):
            base = getattr(L, nm, None)
            if isinstance(base, int):
                for k in range(16):
                    if (base + k) in self.old2new:
                        self.persistent_old.append(base + k)
        # scalar VM-state lanes the dispatch/fold write & the driver reads
        for nm in ("PC_VAL", "AX_VAL", "SP_VAL", "BP_VAL", "STK_VAL", "HALTED",
                   "OP_VAL", "IMM"):
            base = getattr(L, nm, None)
            if isinstance(base, int) and base in self.old2new:
                self.persistent_old.append(base)
        self.persistent_old.append(L.ONE)
        self.persistent_new = set(self.old2new[d] for d in self.persistent_old)

    def remap_spec(self, spec) -> Dict[str, torch.Tensor]:
        """Column/row-select a full-D spec into the compact residual.  BYTE-IDENTICAL
        SwiGLU: W_up/W_gate keep only the footprint columns, W_down keeps only the
        footprint rows; biases unchanged."""
        idx = torch.tensor(self.dims, dtype=torch.long)
        out = {}
        out["W_up"] = spec["W_up"].index_select(1, idx).contiguous()
        out["W_gate"] = spec["W_gate"].index_select(1, idx).contiguous()
        out["W_down"] = spec["W_down"].index_select(0, idx).contiguous()
        out["b_up"] = spec["b_up"].clone()
        out["b_gate"] = spec["b_gate"].clone()
        out["b_down"] = spec["b_down"].index_select(0, idx).contiguous()
        return out


def build_compact_base(base_specs, L):
    """Return ``(cmap, compact_specs)`` — the compact map + the base blocks remapped
    onto the compact residual."""
    cmap = BaseCompactMap(base_specs, L)
    compact = [(n, cmap.remap_spec(s)) for n, s in base_specs]
    return cmap, compact


# ===========================================================================
# FFN forward (SwiGLU, additive residual) — identical math to the divmod pipeline
# and to ``_e2e_inline_bake_916.SparseLeanVM`` (silu(W_up x) * (W_gate x)).
# ===========================================================================
def _ffn_forward(x, spec):
    up = F.linear(x, spec["W_up"]) + spec["b_up"]
    gate = F.linear(x, spec["W_gate"]) + spec["b_gate"]
    hidden = F.silu(up) * gate
    return x + F.linear(hidden, spec["W_down"], spec.get("b_down"))


# ===========================================================================
# SINGLE-TOKEN base forward on the compact residual (the reference the sliced
# pipeline must match byte-for-byte).
# ===========================================================================
def run_base_single_token(x0, compact_specs):
    x = x0.clone()
    for _, spec in compact_specs:
        x = _ffn_forward(x, spec)
    return x


# ===========================================================================
# TOKEN-PIPELINED base forward: slice the base blocks into tokens of <= cap FFN
# layers each; at EVERY token boundary carry the live scratch (dims written so far,
# minus persistent state) across via the byte-exact softmax1+ALiBi CAM (each live
# dim its own address).  This exercises the cross-token gather over the BASE scratch.
# ===========================================================================
def run_base_token_pipeline(x0, cmap: BaseCompactMap, compact_specs, cap: int = 23,
                            verbose: bool = False):
    """Run the base blocks token-sliced.  Returns ``(x_final, n_tokens)``.

    Persistent VM state (PC/AX/SP/BP/STACK0 + scalar lanes + ONE) rides resident in
    every token's seed; transient scratch crosses boundaries via ``kv_gather`` at a
    UNIQUE address per compact dim (exact-match softmax1 weight ~1.0 in fp64 -> abs
    err 0)."""
    D = cmap.D
    # partition into tokens of <= cap FFN layers (+1 gather-attn stage = <= cap+1 layers)
    tokens: List[List] = []
    cur: List = []
    for nb in compact_specs:
        if len(cur) >= cap:
            tokens.append(cur)
            cur = []
        cur.append(nb)
    if cur:
        tokens.append(cur)

    store_rows: List[Tuple[int, int, list]] = []
    pos = [0]

    def carry_out(x, dims):
        for d in sorted(dims):
            pos[0] += 30                      # one 30-token frame apart (spec spacing)
            store_rows.append((pos[0], 0x2000 + d, [float(x[d])] + [0.0] * 15))

    def carry_in(x, dims):
        for d in sorted(dims):
            pos[0] += 1
            x[d] = float(kv_gather(store_rows, 0x2000 + d)[0])

    x = x0.clone()
    written: Set[int] = set()
    persistent = cmap.persistent_new
    for ti, tok in enumerate(tokens):
        if ti > 0:
            live = written - persistent          # transient scratch to relay
            carry_out(x, live)
            xn = x0.clone()                      # re-seed persistent state (resident)
            carry_in(xn, live)                   # gather the transient scratch back
            # persistent dims already in xn (from x0 seed); overwrite w/ current values
            for d in persistent:
                xn[d] = float(x[d])
            x = xn
        for name, spec in tok:
            _, w = _block_rw(spec)
            x = _ffn_forward(x, spec)
            written |= (w - {cmap.ONE})
        if verbose:
            print(f"    token {ti}: ran {len(tok)} FFN blocks, "
                  f"written scratch so far={len(written - persistent)}")
    return x, len(tokens)
