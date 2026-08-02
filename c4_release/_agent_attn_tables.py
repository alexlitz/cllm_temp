"""C4_ATTN_TABLES — attention-baked static lookup tables for c4_min.

THE IDEA (gated, default OFF -> golden 069cc32f unchanged)
=========================================================
The blog's §Memory subsystem (``c4_min.blogspec_memory.KVMemory``) is ALREADY a
softmax1 + ALiBi content-addressed memory (CAM): a store writes an address→value
KV row, a load queries the address and the exact-match row's value is relayed.

For a STATIC, READ-ONLY table (finesine, palette, reciprocal-for-divide), the
current VM pays twice:
  (a) INIT COMPUTE — code computes+writes each entry (``init_sin`` in doom.c is a
      fixed-point Taylor loop: per entry 5 MUL + 3 DIV + adds, then 3 mirror
      loops, then one SI store per entry = thousands of VM steps), and
  (b) STORE-LOG RESIDENCY — every entry lives in the GROWING dynamic store-log
      (each step re-runs the CAM score over ALL committed store rows, and the
      eviction policy has to keep them live).

This module bakes the table ONCE as a set of FIXED KV rows
``{key = binary(index), value = table[index]}`` and adds a table-lookup op that
is a single content-addressed attention query keyed on the index -> the value V,
decoded exactly like a memory read.  The fixed table KV lives in a SEPARATE,
pre-loaded, never-evicted KV region (the block's ``past_kv`` cache prefix), so:
  * NO init compute (0 VM steps to build the table), and
  * the table is NOT in the dynamic per-step store-log (heap/eviction stays
    small — a program's live store-log only holds its real writes).

It reuses the EXACT §Memory CAM machinery (``bake_memory_head`` / the ±smag
binary-address key/query, the ZFOD bias channel, the store-role gate, the value
relay, softmax1) — a table row is just a permanent "store" whose address is the
index and whose value is ``table[index]``.  Because each index appears exactly
ONCE in a static table, ALiBi recency is irrelevant (there are no duplicate
addresses), so the exact-match address CAM alone resolves every lookup.

Nothing here is on any production build path.  ``C4_ATTN_TABLES`` gates the demo;
default OFF, additive, new file only (``_agent_*.py``) -> golden 069cc32f
unchanged.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from c4_min import blogspec_vocab as V
from c4_min.blogspec_layout import NIB_PER_REG
from c4_min.blogspec_memory import (
    MemoryLayout, build_memory_model, bake_memory_head,
    address_bits, ADDR_BITS, _decode_byte,
)


def attn_tables_enabled() -> bool:
    """C4_ATTN_TABLES (DEFAULT OFF): the attention-baked static lookup table demo.

    OFF -> nothing built here touches the golden path (golden 069cc32f unchanged).
    ON  -> the demo / measurement harness runs.  This is a MECHANISM demonstrator;
    it does not alter any production op.
    """
    return os.environ.get("C4_ATTN_TABLES", "0") not in ("0", "", "false", "False")


# ===========================================================================
# The mechanism: a static array -> FIXED KV rows in a separate CAM region.
# ===========================================================================
@dataclass
class AttentionTable:
    """A static read-only table baked as FIXED attention KV rows.

    ``table[i]`` (a 32-bit value, signed values stored two's-complement) is baked
    as one permanent KV row ``{key = binary(i), value = table[i]}`` using the
    SAME §Memory CAM head as ``KVMemory``.  ``lookup(i)`` is a single softmax1
    attention query keyed on ``i`` -> the value V of the exact-match row, decoded
    with the LM byte-head (identical to a memory load).

    The rows are computed at BAKE time (host side, known ahead) and pinned into
    the block's ``past_kv`` cache prefix — a SEPARATE, never-evicted KV region
    that is NOT part of the dynamic per-step store-log.

    ``base_addr`` offsets the index into an address space so the table can share
    a CAM with a live store-log without collision (index i -> address
    ``base_addr + i*stride``).  ``stride`` defaults to 4 (word-addressed, §Memory
    4-byte alignment); use 1 for a byte table (palette).
    """

    values: List[int]
    base_addr: int = 0x10000
    stride: int = 4
    name: str = "table"
    signed: bool = False
    _model: object = field(default=None, repr=False)
    _L: object = field(default=None, repr=False)
    _kv: object = field(default=None, repr=False)          # baked (K, V, pos) prefix
    _bake_stats: dict = field(default_factory=dict, repr=False)

    # -- address <-> index ---------------------------------------------------
    def addr_of(self, i: int) -> int:
        return (self.base_addr + i * self.stride) & 0xFFFFFFFF

    # -- build the model + bake the fixed KV rows ----------------------------
    def bake(self, model=None, L: Optional[MemoryLayout] = None) -> "AttentionTable":
        """Bake every ``table[i]`` as a FIXED KV row into the CAM cache prefix.

        Returns ``self``.  After this the table's KV ``(K, V, pos)`` prefix is
        computed ONCE; ``lookup`` attends the query over it via the block's
        ``past_kv`` incremental path — so the table entries never enter the
        per-step token stream (store-log).
        """
        t0 = time.time()
        if model is None:
            model, L = build_memory_model()
        self._model, self._L = model, L
        n = len(self.values)

        # Build ONE residual row per table entry, each carrying the §Memory store
        # bands: ADDR_BIN = binary(addr), VAL_NIB = value nibbles, IS_STORE = 1.
        # These are the KEY/VALUE source rows of the fixed CAM region.
        rows = torch.zeros(n, L.D)
        rows[:, L.ONE] = 1.0                       # ±smag key bias lane
        for i, val in enumerate(self.values):
            addr = self.addr_of(i)
            v32 = val & 0xFFFFFFFF
            for b, bit in enumerate(address_bits(addr)):
                rows[i, L.ADDR_BIN + b] = bit
            for j, nv in enumerate(V.nibbles_of_value(v32, NIB_PER_REG)):
                rows[i, L.VAL_NIB + j] = float(nv)
            rows[i, L.IS_STORE] = 1.0

        # Run the ATTENTION K/V projection over the table rows ONCE and store the
        # (K, V, pos) prefix.  This is the "bake": the fixed table KV is computed
        # a single time and kept in a separate region (never re-scored per step,
        # never evicted, never in the dynamic store-log).
        attn = model.blocks[0].attn
        with torch.no_grad():
            x = rows.unsqueeze(0)                  # [1, n, D]
            B, S, D = x.shape
            H, HD = attn.n_heads, attn.head_dim
            K = F.linear(x, attn.W_k).view(B, S, H, HD).transpose(1, 2)
            Vv = F.linear(x, attn.W_v).view(B, S, H, HD).transpose(1, 2)
            pos = torch.arange(S)
        self._kv = (K, Vv, pos)
        self._bake_stats = {
            "entries": n,
            "kv_rows": n,
            "bake_wall_s": time.time() - t0,
            # residency: K + V floats for the fixed prefix (per block, one block).
            "kv_floats": int(K.numel() + Vv.numel()),
            "kv_bytes_fp32": int((K.numel() + Vv.numel()) * 4),
        }
        return self

    # -- the table-lookup op: one content-addressed attention query ----------
    def lookup(self, i: int) -> int:
        """LOOKUP(table, i) == table[i], via ONE softmax1 attention query.

        A query row keyed on ``binary(addr_of(i))`` attends over the FIXED table
        KV prefix; the exact-match row's value nibbles are relayed into AX and
        decoded by the LM byte-head — identical to a §Memory load, but the table
        was never computed at runtime nor stored in the dynamic store-log.
        """
        return self._lookup_batch([i])[0]

    def lookup_all(self) -> List[int]:
        """Every ``table[i]`` via a single batched attention query (proof path)."""
        return self._lookup_batch(list(range(len(self.values))))

    def _lookup_batch(self, idxs: Sequence[int]) -> List[int]:
        assert self._kv is not None, "call .bake() first"
        L, model = self._L, self._model
        attn = model.blocks[0].attn
        K, Vv, kv_pos = self._kv
        n_tab = K.shape[2]
        m = len(idxs)

        # Build the QUERY rows: each carries QRY_BIN = binary(addr_of(i)),
        # IS_LOAD = 1 (identical to a §Memory load query).
        qrows = torch.zeros(m, L.D)
        qrows[:, L.ONE] = 1.0
        for r, i in enumerate(idxs):
            addr = self.addr_of(i)
            for b, bit in enumerate(address_bits(addr)):
                qrows[r, L.QRY_BIN + b] = bit
            qrows[r, L.IS_LOAD] = 1.0

        with torch.no_grad():
            x = qrows.unsqueeze(0)                          # [1, m, D]
            B, S, D = x.shape
            H, HD = attn.n_heads, attn.head_dim
            Q = F.linear(x, attn.W_q).view(B, S, H, HD).transpose(1, 2)
            # queries sit AFTER the table prefix (positions n_tab .. n_tab+m-1),
            # so the causal mask lets each query see the whole fixed table.
            q_pos = torch.arange(n_tab, n_tab + m)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale  # [1,H,m,n_tab]
            dist = (q_pos.unsqueeze(1) - kv_pos.unsqueeze(0)).abs().float()
            scores = scores - attn.alibi_slopes.view(1, H, 1, 1) * dist.unsqueeze(0)
            # all table rows precede every query -> fully visible (no causal mask).
            from c4_min.blogspec_model import softmax1
            w = softmax1(scores, dim=-1)                    # softmax1 (ZFOD sink)
            out = torch.matmul(w, Vv).transpose(1, 2).contiguous().view(B, S, D)
            state = x + F.linear(out, attn.W_o)             # [1, m, D]

        vals: List[int] = []
        for r in range(m):
            row = state[0, r]
            v = 0
            for bi in range(4):
                v |= _decode_byte(row, L, L.AX, bi) << (8 * bi)
            if self.signed and (v & 0x80000000):
                v -= 1 << 32
            vals.append(v)
        return vals


# ===========================================================================
# init_sin cost model (doom.c) — the STEPS the baked table replaces.
# ===========================================================================
def init_sin_reference(circ: int = 256, fp: int = 1024) -> List[int]:
    """The exact doom.c ``init_sin`` sine table (first-quadrant Taylor + mirror).

    Mirrors doom.c lines 155-179 verbatim (integer fixed-point), so the baked
    table's values ARE the doom sintab; the cost model below counts the VM ops
    this computation would take.
    """
    t = [0] * circ
    i = 0
    while i <= circ // 4:
        x = i * 3217 // 128
        x2 = x * x // fp
        x3 = x2 * x // fp
        x5 = x3 * x2 // fp
        x7 = x5 * x2 // fp
        t[i] = x - x3 // 6 + x5 // 120 - x7 // 5040
        i += 1
    i = circ // 4 + 1
    while i < circ // 2:
        t[i] = t[circ // 2 - i]; i += 1
    i = circ // 2
    while i < 3 * circ // 4:
        t[i] = -t[i - circ // 2]; i += 1
    i = 3 * circ // 4
    while i < circ:
        t[i] = -t[circ - i]; i += 1
    return t


def init_sin_vm_step_cost(circ: int = 256) -> dict:
    """Analytic VM-step (and DIV-op) cost of computing ``init_sin`` at runtime.

    The blog VM executes one C-level operation per VM step (each op = one 30-token
    frame).  Counting the C4 ops doom.c's ``init_sin`` emits (the compiled body,
    per the ISA the VM runs):

      Quadrant loop (i = 0 .. CIRC/4, inclusive -> CIRC/4 + 1 iterations), body:
        x  = i*3217/128        : 1 MUL + 1 DIV
        x2 = x*x/FP            : 1 MUL + 1 DIV
        x3 = x2*x/FP           : 1 MUL + 1 DIV
        x5 = x3*x2/FP          : 1 MUL + 1 DIV
        x7 = x5*x2/FP          : 1 MUL + 1 DIV
        t[i]= x - x3/6 + x5/120 - x7/5040 : 3 DIV + 3 SUB/ADD
        + loop test/incr + index math + 1 SI store           (~8 book-keeping ops)
      -> ~5 MUL, ~8 DIV, ~14 other = ~27 ops/iter.
      Mirror loops (3 loops covering the other 3 quadrants): each iter is
        1 index-math + 1 load + 1 negate/copy + 1 SI store + loop overhead
        -> ~6 ops/iter, over ~3*CIRC/4 iterations.

    A DIV is the ISA's most expensive op — the recurrent long-division megablock
    is 168 of the VM's 242 blocks (``batched_block_skip.py``), i.e. every DIV
    step pays the full division layer stack.
    """
    q_iters = circ // 4 + 1
    muls = 5 * q_iters
    divs = 8 * q_iters
    other_q = 14 * q_iters
    mirror_iters = 3 * circ // 4          # the three mirror loops together
    mirror_ops = 6 * mirror_iters
    si_stores = circ                      # one SI per table entry (both loops)
    total_ops = muls + divs + other_q + mirror_ops
    return {
        "circ": circ,
        "quadrant_iters": q_iters,
        "mul_ops": muls,
        "div_ops": divs,                  # the expensive long-division steps
        "other_quadrant_ops": other_q,
        "mirror_ops": mirror_ops,
        "si_store_ops": si_stores,        # per-entry stores avoided (store-log)
        "total_vm_steps": total_ops,
        "div_block_fraction": "168/242 blocks per DIV (batched_block_skip.py)",
    }
