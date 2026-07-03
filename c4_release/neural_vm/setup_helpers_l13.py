"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 13 helpers: MEM addr gather + shift FFN.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


def _set_layer13_mem_addr_gather(attn, S, BD, HD):
    """L13 attention heads 0-2: Gather MEM addr bytes → MEM val byte positions.

    For L15 K-side address keys: copies addr byte nibbles from MEM addr
    positions (d=0..3 from MEM marker) to MEM val byte positions (d=4..8).

    Head 0 additionally drives an ``ADDR_B0_VALID`` lifecycle bit (B7-4 / B6-K
    slot 97) on slot 34. See the inline block below the gather loop for the
    detailed Q/K/V/O routing; downstream consumers (L10 tail addr0 family
    per B4-H §3.2) read ADDR_B0_VALID to distinguish freshly-computed
    ADDR_B0 nibbles from stale residue.
    """
    L = 15.0
    MEM_I = 4

    for j in range(3):
        base = j * HD
        addr_lo_out = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][j]
        addr_hi_out = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][j]

        # Q: fires at MEM val byte positions (d=5..8 from MEM)
        # Use MEM_VAL_B0-3 flags computed in L2 FFN
        attn.W_q[base, BD.MEM_VAL_B0] = L
        attn.W_q[base, BD.MEM_VAL_B1] = L
        attn.W_q[base, BD.MEM_VAL_B2] = L
        attn.W_q[base, BD.MEM_VAL_B3] = L

        # K: fires at MEM addr byte J position.
        # Addr byte 0 is at d=1 (after MEM marker), byte 1 at d=2, byte 2 at d=3.
        if j == 0:
            # Addr byte 0 at d=1: L1H1[MEM]=1 (d≤1.5), subtract L1H0[MEM] (d=0 only)
            attn.W_k[base, BD.L1H1 + MEM_I] = L
            attn.W_k[base, BD.L1H0 + MEM_I] = -L  # exclude MEM marker (d=0)
        elif j == 1:
            # Addr byte 1 at d=2: L1H2[MEM]=1 (d≤2.5), subtract L1H1[MEM] (d≤1.5)
            attn.W_k[base, BD.L1H2 + MEM_I] = L
            attn.W_k[base, BD.L1H1 + MEM_I] = -L
        elif j == 2:
            # Addr byte 2 at d=3: H0[MEM]=1 (d≤3.5), subtract L1H2[MEM] (d≤2.5)
            attn.W_k[base, BD.H0 + MEM_I] = L
            attn.W_k[base, BD.L1H2 + MEM_I] = -L

        # Anti-leakage gate
        attn.W_q[base + 33, BD.MEM_VAL_B0] = L
        attn.W_q[base + 33, BD.CONST] = -L / 2
        attn.W_k[base + 33, BD.CONST] = L

        # V: copy CLEAN_EMBED nibbles (addr byte value)
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: write to ADDR_BJ_LO/HI (gathered to val byte positions)
        for k in range(16):
            attn.W_o[addr_lo_out + k, base + 1 + k] = 1.0
            attn.W_o[addr_hi_out + k, base + 17 + k] = 1.0

        # ----------------------------------------------------------------
        # ADDR_B0_VALID lifecycle bit (B7-4): only head 0 produces it,
        # because only head 0 gathers ADDR_B0_LO/HI. The slot mirrors head
        # 0's primary Q/K so attention is locked to the MEM addr byte 0
        # row, then reads ``L1H1+MEM_I`` through V and routes through W_o
        # into ADDR_B0_VALID.
        #
        # Reading ``L1H1+MEM_I`` (not CONST) is what gives the lifecycle
        # bit its discrimination: ``L1H1+MEM_I`` is 1.0 only at MEM-band
        # rows close to a MEM marker (d ≤ 1.5 from MEM), so at attended
        # MEM addr byte 0 rows V=1 → output=1, while at unrelated rows
        # (no MEM context anywhere) V=0 → output=0 even when softmax
        # spreads mass.
        #
        # The result: at every MEM val byte position, ADDR_B0_VALID
        # residual carries ~1.0 *iff* head 0 found a real MEM addr byte 0
        # row to attend to. Downstream consumers (L10 tail addr0 family
        # per B4-H §3.2) can gate their writes on ADDR_B0_VALID +50 as
        # positive evidence that the ADDR_B0 lanes are fresh.
        if j == 0:
            VALID_SLOT = 34  # one past the anti-leakage gate at slot 33
            # Q mirrors slot 0 (fires at MEM val byte positions).
            attn.W_q[base + VALID_SLOT, BD.MEM_VAL_B0] = L
            attn.W_q[base + VALID_SLOT, BD.MEM_VAL_B1] = L
            attn.W_q[base + VALID_SLOT, BD.MEM_VAL_B2] = L
            attn.W_q[base + VALID_SLOT, BD.MEM_VAL_B3] = L
            # K mirrors head 0 slot 0 (selects MEM addr byte 0 row at d=1).
            attn.W_k[base + VALID_SLOT, BD.L1H1 + MEM_I] = L
            attn.W_k[base + VALID_SLOT, BD.L1H0 + MEM_I] = -L
            # V: read L1H1+MEM_I so the attended MEM addr byte 0 row
            # delivers 1.0 and unrelated rows deliver 0.
            attn.W_v[base + VALID_SLOT, BD.L1H1 + MEM_I] = 1.0
            # O: route gathered value into ADDR_B0_VALID.
            attn.W_o[BD.ADDR_B0_VALID, base + VALID_SLOT] = 1.0

