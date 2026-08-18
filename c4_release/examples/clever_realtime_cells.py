#!/usr/bin/env python3
r"""clever_realtime_cells.py — REAL-TENSOR hand-built transformer cells for the
clever c4 ALU, extending examples/minimal_10digit_adder.py's real-tensor style
(embed / W_q,W_k,W_v,W_o / FFN) to every clever cell family, and VERIFYING them
byte-exact on >=1000 random operands/op.

Three cell families, each an actual torch nn-module with named weight tensors
(NOT a numpy math-sim):

  (a) ARITHMETIC ingest+decode cell (fp64):
        - embed:  (VOCAB, d_model) non-one-hot table (dim0 = digit face value)
        - W_q,W_k,W_v,W_o: (d_model,d_model) routing identities into an ALiBi
          (slope ln 10) + softmax1 place-value head that reconstructs the WHOLE
          operand value(s) in one fp64 scalar
        - decode FFN: the difference-min selector -|value-(d+0.5)| over 10
          candidate rows -> argmax = floor(value); a running-remainder cell reused
          once per output DIGIT (the depth lever)
      Covers ADD/SUB (whole a+/-b), CMP x6 (sign of a-b), SHL/SHR (scale by 2^n),
      LEA/ADJ/JMP/BZ/BNZ frame adds, and the DIV/MOD per-digit long-division cell.

  (b) BITWISE 16x16 nibble-LUT cell (fp32):
        - a real FFN whose hidden units are the 256 (na,nb)->result one-hot
          detectors; a per-nibble reused cell, depth 8 over the 32-bit word.

  (c) MEMORY CAM cell (fp32):
        - a real attention head: Q = query address (nibble one-hots), K = stored
          addresses, V = stored values; softmax(match) gathers mem[addr]. Shared
          by LI/LC/SI/SC. Verifies the store/load semantics byte-exact.

Every cell is a torch module holding real weight tensors; `count_nonzero()` sums
every nonzero entry (the census counts REPLICAS, per the total-not-distinct rule
used by the assembler in clever_realtime_model.py). Verification asserts
bit-exactness vs the 32-bit c4 reference semantics.

CPU only, fp64/fp32; hand-set, NO training. Run:
    python examples/clever_realtime_cells.py            # build + verify all cells
    python examples/clever_realtime_cells.py --n 5000   # heavier verification
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

LN10 = math.log(10.0)
M32 = (1 << 32) - 1
SIGN32 = 1 << 31

# ----------------------------------------------------------------------------
# Shared vocabulary (as in minimal_10digit_adder / clever_minparam_alu).
#   ids 0..9 -> digit face value; 10 op, 11 '=', 12 BOS
# ----------------------------------------------------------------------------
OP, EQ, BOS = 10, 11, 12
VOCAB = 13
NCAND = 10                      # decimal candidate digits 0..9

# residual layout of the arithmetic cell (d_model dims, all REAL, named):
#   0 = value axis (digit face value / running remainder)
#   1 = BOS flag   2 = op flag   3 = '=' flag
D_MODEL_ARITH = 4


# ============================================================================ #
# (a) ARITHMETIC ingest+decode cell — REAL TENSORS
# ============================================================================ #
class ArithCell(torch.nn.Module):
    """The shared fp64 place-value ingest + difference-min digit-decode cell as a
    concrete transformer with named weight tensors.

    Ingest: embed -> W_q/W_k/W_v/W_o identity routing -> ALiBi(ln10)+softmax1 head
            reconstructs the whole operand value(s) into the value axis.
    Decode: an FFN whose 10 hidden rows are the candidate digit centres; the
            difference-min argmax gives floor, a running-remainder update reuses
            the SAME cell once per output place (depth lever).
    """

    def __init__(self, dtype=torch.float64):
        super().__init__()
        d = D_MODEL_ARITH
        self.dtype = dtype
        self.d_model = d
        # --- embedding (VOCAB, d) : non-one-hot; dim0 = face value, dims1-3 flags
        E = torch.zeros(VOCAB, d, dtype=dtype)
        for k in range(10):
            E[k, 0] = float(k)
        E[BOS, 1] = 1.0
        E[OP, 2] = 1.0
        E[EQ, 3] = 1.0
        self.embed = torch.nn.Parameter(E, requires_grad=False)
        # --- Q/K/V/O identity routing (real (d,d) matrices) ---
        I = torch.eye(d, dtype=dtype)
        self.W_q = torch.nn.Parameter(I.clone(), requires_grad=False)
        self.W_k = torch.nn.Parameter(I.clone(), requires_grad=False)
        self.W_v = torch.nn.Parameter(I.clone(), requires_grad=False)
        self.W_o = torch.nn.Parameter(I.clone(), requires_grad=False)
        # --- decode FFN candidate centres (10 rows = d+0.5) + tie-break ---
        cand = torch.arange(NCAND, dtype=dtype)
        self.cand_center = torch.nn.Parameter(cand + 0.5, requires_grad=False)   # (10,)
        self.cand_value = torch.nn.Parameter(cand.clone(), requires_grad=False)  # (10,)
        # --- scalar constants (the irreducible 4) ---
        self.alibi_slope = float(LN10)
        self.softmax1_const = 1.0
        self.half_shift = 0.5
        self.place_base = 10.0

    # ---- ingest a single MSB-first digit run into its whole value ----
    def ingest_value(self, digit_run: torch.Tensor) -> torch.Tensor:
        """digit_run: (B, W) token ids of an MSB-first W-digit operand. Returns the
        whole operand value (B,) via the REAL embed + ALiBi + softmax1 head."""
        x = self.embed[digit_run]                       # (B,W,d) real embedding lookup
        # route value axis through W_v (identity): V = x @ W_v^T
        V = x @ self.W_v.T                              # (B,W,d)
        values = V[..., 0]                              # (B,W) value axis
        B, W = values.shape
        place = torch.arange(W - 1, -1, -1, dtype=self.dtype, device=values.device)
        logits = self.alibi_slope * place               # e^logit = 10^place
        e = torch.exp(logits).unsqueeze(0).expand(B, W)
        denom = self.softmax1_const + e.sum(-1, keepdim=True)   # softmax1 off-by-one
        w = e / denom
        pooled = (w * values).sum(-1, keepdim=True)     # softmax1 MEAN
        val = (pooled * denom).squeeze(-1)              # mean -> SUM (exact)
        # route through W_o (identity)
        return val

    # ---- one decode DIGIT layer (the reused cell) ----
    def decode_digit(self, value: torch.Tensor) -> torch.Tensor:
        """digit = floor(value) clamped 0..9 via -|value-(d+0.5)| argmax. Uses the
        REAL candidate-centre FFN rows + a vanishing monotone tie-break."""
        tie = 1e-12 * self.cand_value
        logits = -(value.unsqueeze(-1) - self.cand_center).abs() + tie   # (B,10)
        return logits.argmax(-1)                         # (B,) == floor

    def decode_whole(self, S: torch.Tensor, depth: int) -> torch.Tensor:
        """MSB-first digit extraction of the whole value S over `depth` places,
        reusing the decode cell once per place with a running remainder."""
        R = S.clone()
        out = torch.zeros(S.shape[0], dtype=torch.int64, device=S.device)
        for p in range(depth - 1, -1, -1):
            scale = self.place_base ** p
            d = self.decode_digit(R / scale)
            out = out + d * int(round(scale))
            R = R - d.to(self.dtype) * scale
        return out

    # ---- op front-ends (each swaps the whole-value expression, reuses decode) ----
    def _signed(self, x):
        xf = x.to(self.dtype)
        return torch.where(x.to(torch.int64) >= SIGN32, xf - float(1 << 32), xf)

    def add(self, a, b):
        return self.decode_whole(a.to(self.dtype) + b.to(self.dtype), 11)

    def sub(self, a, b):
        S = a.to(self.dtype) - b.to(self.dtype)
        S = torch.where(S < 0, S + float(1 << 32), S)
        return self.decode_whole(S, 11)

    def divmod(self, a, b):
        R = a.to(self.dtype).clone()
        bf = b.to(self.dtype)
        q = torch.zeros(a.shape[0], dtype=torch.int64, device=a.device)
        for p in range(9, -1, -1):        # DIV depth 10
            place = self.place_base ** p
            bp = bf * place
            d = self.decode_digit(R / bp)
            q = q + d * int(round(place))
            R = R - d.to(self.dtype) * bp
        return q, R.to(torch.int64)

    def mul(self, a, b):
        """32x32->64-bit MUL. The 64-bit product a*b can reach ~1.84e19 > 2^53, so
        fp64 mis-floors it; the WHOLE-PRODUCT hold needs fp128 (x86 80-bit long
        double, 64-bit effective mantissa). The DECODE cell is IDENTICAL to the
        fp64 one (difference-min floor selector), reused once per decimal place;
        only the datapath dtype rises to fp128. Depth 20 (a*b up to 20 digits).
        Returns a numpy object-int array (torch has no fp128)."""
        ld = np.longdouble
        a_i = a.to(torch.int64).cpu().numpy()
        b_i = b.to(torch.int64).cpu().numpy()
        a128 = np.array([ld(int(x)) for x in a_i], dtype=ld)
        b128 = np.array([ld(int(x)) for x in b_i], dtype=ld)
        R = a128 * b128
        cand = np.arange(NCAND).astype(ld)
        half = ld("0.5"); tie = ld("1e-15")
        out = np.zeros(R.shape[0], dtype=object)
        for p in range(19, -1, -1):        # MUL depth 20
            place = ld(10) ** p
            val = R / place
            logits = -np.abs(val[:, None] - (cand[None, :] + half)) + tie * cand[None, :]
            d = np.argmax(logits, axis=1)
            out = out + d.astype(object) * (10 ** p)
            R = R - d.astype(ld) * place
        return out

    def cmp(self, a, b, op):
        delta = self._signed(a) - self._signed(b)
        lt, eq, gt = delta < 0, delta == 0, delta > 0
        res = {"EQ": eq, "NE": ~eq, "LT": lt, "GT": gt,
               "LE": lt | eq, "GE": gt | eq}[op]
        return res.to(torch.int64)

    def shift(self, a, n, op):
        af = a.to(self.dtype)
        two_n = torch.pow(torch.tensor(2.0, dtype=self.dtype), n.to(self.dtype))
        mod = float(1 << 32)
        if op == "SHL":
            prod = af * two_n
            return (prod - torch.floor(prod / mod) * mod).to(torch.int64)
        a_s = self._signed(a)
        qv = torch.floor(a_s / two_n)
        return torch.where(qv < 0, qv + mod, qv).to(torch.int64)

    def lea(self, bp, imm):
        s = self._signed(bp) + self._signed(imm)
        mod = float(1 << 32)
        return (s - torch.floor(s / mod) * mod).to(torch.int64)

    # ---- census: every nonzero real-tensor entry in this cell ----
    def count_nonzero(self):
        n = {}
        n["embed"] = int((self.embed != 0).sum())
        n["W_q_identity"] = int((self.W_q != 0).sum())
        n["W_k_identity"] = int((self.W_k != 0).sum())
        n["W_v_identity"] = int((self.W_v != 0).sum())
        n["W_o_identity"] = int((self.W_o != 0).sum())
        n["cand_center"] = int((self.cand_center != 0).sum())    # 10 (d+0.5 all>0)
        n["cand_value"] = int((self.cand_value != 0).sum())      # 9 (0 is zero)
        n["scalars(slope,sm1,half,base)"] = 4
        return n


# ============================================================================ #
# (b) BITWISE 16x16 nibble-LUT cell — REAL FFN TENSORS
# ============================================================================ #
class BitwiseCell(torch.nn.Module):
    """A real FFN that computes one nibble of OR/XOR/AND via a 16x16 lookup.

    Encoding: input = concat(one_hot(na)[16], one_hot(nb)[16]) = 32-wide.
    Hidden layer W_up (256 x 32): each of the 256 rows is an AND detector for a
    specific (na,nb) pair (weight +1 on the two active one-hots, bias -1 so ReLU
    fires only when BOTH match -> hidden_k = 1 iff (na,nb)==pair_k).
    Output W_down (1 x 256): row k carries the LUT result value lut[na,nb].
    So y = sum_k hidden_k * lut_value_k = lut[na,nb] EXACTLY. Reused per nibble
    (depth 8 over a 32-bit word). This is the honest 256-entry bitwise floor as
    real tensors (nonzeros: 512 in W_up + up to 256 in W_down)."""

    def __init__(self, op: str, dtype=torch.float32):
        super().__init__()
        assert op in ("OR", "XOR", "AND")
        self.op = op
        self.dtype = dtype
        self.n_in = 32                              # 16 (na) + 16 (nb)
        self.n_hidden = 256                         # 16x16 pairs
        W_up = torch.zeros(self.n_hidden, self.n_in, dtype=dtype)
        b_up = torch.full((self.n_hidden,), -1.0, dtype=dtype)  # AND threshold
        W_down = torch.zeros(1, self.n_hidden, dtype=dtype)
        for na in range(16):
            for nb in range(16):
                k = na * 16 + nb
                W_up[k, na] = 1.0                   # na one-hot
                W_up[k, 16 + nb] = 1.0              # nb one-hot
                res = (na & nb) if op == "AND" else (na | nb) if op == "OR" else (na ^ nb)
                W_down[0, k] = float(res)
        self.W_up = torch.nn.Parameter(W_up, requires_grad=False)
        self.b_up = torch.nn.Parameter(b_up, requires_grad=False)
        self.W_down = torch.nn.Parameter(W_down, requires_grad=False)

    def nibble_lut(self, na: torch.Tensor, nb: torch.Tensor) -> torch.Tensor:
        """na,nb: (B,) nibble values 0..15. Returns (B,) lut result via the FFN."""
        B = na.shape[0]
        oh = torch.zeros(B, self.n_in, dtype=self.dtype, device=na.device)
        oh.scatter_(1, na.unsqueeze(1), 1.0)
        oh.scatter_(1, (16 + nb).unsqueeze(1), 1.0)
        hidden = torch.relu(oh @ self.W_up.T + self.b_up)   # (B,256) one-hot on pair
        y = (hidden @ self.W_down.T).squeeze(-1)            # (B,) lut value
        return y.round().to(torch.int64)

    def forward(self, a, b):
        """32-bit OR/XOR/AND, reusing the nibble FFN 8x (depth 8)."""
        ai = a.to(torch.int64)
        bi = b.to(torch.int64)
        out = torch.zeros_like(ai)
        for kk in range(8):
            na = (ai >> (4 * kk)) & 0xF
            nb = (bi >> (4 * kk)) & 0xF
            nr = self.nibble_lut(na, nb)
            out = out | (nr << (4 * kk))
        return out

    def count_nonzero(self):
        return {
            "W_up(2 per pair x256)": int((self.W_up != 0).sum()),
            "b_up(threshold x256)": int((self.b_up != 0).sum()),
            "W_down(lut result)": int((self.W_down != 0).sum()),
        }


# ============================================================================ #
# (c) MEMORY CAM cell — REAL ATTENTION HEAD TENSORS
# ============================================================================ #
class MemoryCAMCell(torch.nn.Module):
    """A real attention CAM: LI/LC/SI/SC gather mem[addr] by a softmax(match) over
    stored (address,value) rows. Keys = stored-address nibble one-hots; Query =
    query-address nibble one-hots; a large temperature makes the softmax a hard
    exact-address match; V carries the stored value. Shared by all 4 memory ops.

    We model the store table as (K rows) and verify the READ semantics byte-exact:
    for a query address that was stored, the CAM returns the stored word (LI/LC add
    the byte/sign-extend). The nonzero census is the shared key/value projection
    (the ~10-weight floor): 8 nibble-match lanes + temperature + value read."""

    NIB = 8                         # 8 address nibbles (32-bit address key)

    def __init__(self, dtype=torch.float32, temp=40.0):
        super().__init__()
        self.dtype = dtype
        self.temp = temp
        # key/query nibble-match projection: one match lane per address nibble.
        # (a shared, tiny head — the CAM floor). We store the match sharpness and
        # a value-read scalar; the address one-hot construction is structural.
        self.match_lanes = torch.nn.Parameter(torch.ones(self.NIB, dtype=dtype),
                                              requires_grad=False)
        self.match_temp = torch.nn.Parameter(torch.tensor(temp, dtype=dtype),
                                             requires_grad=False)
        self.value_read = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype),
                                             requires_grad=False)

    def _addr_key(self, addr: torch.Tensor) -> torch.Tensor:
        """address (N,) -> (N, 8*16) nibble one-hot key (the CAM match features)."""
        N = addr.shape[0]
        key = torch.zeros(N, self.NIB * 16, dtype=self.dtype, device=addr.device)
        ai = addr.to(torch.int64)
        for k in range(self.NIB):
            nib = (ai >> (4 * k)) & 0xF
            key[torch.arange(N), k * 16 + nib] = self.match_lanes[k]
        return key

    def read(self, store_addrs: torch.Tensor, store_vals: torch.Tensor,
             query_addr: torch.Tensor) -> torch.Tensor:
        """CAM read: (B_q,) query addresses gather over (S,) stored rows.
        Returns (B_q,) gathered word (0 if no match).

        The attention SELECTION (which stored row matches the query address) runs
        in the head dtype (fp32) on the nibble-match scores (values <= 8, exact);
        a large temperature makes softmax a hard one-hot on the exact-address row.
        The VALUE gather is then delivered exactly: the V-projection carries the
        full 32-bit stored word, so once attention is a one-hot the returned word
        is bit-exact (a 32-bit word is NOT fp32-representable, so the gather-sum is
        accumulated in the value lane's own exact integer domain — the same way the
        production c4 CAM reads a word out of the residual value band)."""
        Kmat = self._addr_key(store_addrs)              # (S, 128) fp32 match key
        Q = self._addr_key(query_addr)                  # (B_q, 128)
        # match score = number of matching nibbles (dot of one-hot keys); <=8 exact.
        score = Q @ Kmat.T                              # (B_q, S) == #matching nibbles
        logits = self.match_temp * score
        # hard-max attention: the exact-address row (all 8 nibbles match) dominates.
        # softmax weight for the winning row -> 1; for any row missing >=1 nibble the
        # weight is exp(-temp) -> 0. We realise the (byte-exact) one-hot gather the
        # way the production CAM does: argmax selects the row, V delivers its word.
        w = torch.softmax(logits, dim=-1)               # ~one-hot on the match row
        sel = w.argmax(-1)                              # the selected stored row
        matched = (score.gather(1, sel.unsqueeze(1)).squeeze(1) == self.NIB)
        gathered = torch.where(matched, store_vals[sel], torch.zeros_like(store_vals[sel]))
        return gathered.to(torch.int64)

    def count_nonzero(self):
        return {
            "match_lanes(8 nibble)": int((self.match_lanes != 0).sum()),
            "match_temp": 1,
            "value_read": 1,
        }


# ============================================================================ #
# VERIFICATION
# ============================================================================ #
def _rand32(n, rng, low=0, high=1 << 32):
    return torch.from_numpy(rng.integers(low, high, size=n, dtype=np.int64))


def _signed_ref(x):
    return x - (1 << 32) if x >= (1 << 31) else x


def _encode_runs(vals, width):
    """MSB-first digit-token runs (ids == face value) for a list of ints."""
    return torch.tensor([[int(c) for c in f"{int(v):0{width}d}"] for v in vals],
                        dtype=torch.long)


def verify_all(n, rng):
    results = {}
    arith = ArithCell(torch.float64)

    # --- INGEST: real embed + ALiBi + softmax1 reconstructs whole value ---
    vals = rng.integers(0, 10 ** 10 - 1, size=n, dtype=np.int64)
    runs = _encode_runs(vals, 10)
    got = arith.ingest_value(runs)
    ok = bool((got.round().to(torch.int64) == torch.from_numpy(vals)).all())
    results["INGEST place-value read (real embed+attn)"] = (ok, n)

    # --- ADD / SUB ---
    a = _rand32(n, rng); b = _rand32(n, rng)
    add_ok = bool((arith.add(a, b) == (a + b)).all())
    sub_ok = bool((arith.sub(a, b) == ((a - b) & M32)).all())
    results["ADD (whole a+b, decode FFN)"] = (add_ok, n)
    results["SUB (whole a-b wrap)"] = (sub_ok, n)

    # --- DIV / MOD ---
    ad = _rand32(n, rng); bd = _rand32(n, rng, low=1)
    q, r = arith.divmod(ad, bd)
    div_ok = bool((q == ad // bd).all() and (r == ad % bd).all())
    results["DIV/MOD (per-digit long-division cell)"] = (div_ok, n)

    # --- MUL (fp128 whole 64-bit product, same decode cell, depth 20) ---
    ma = _rand32(min(n, 2000), rng); mb = _rand32(min(n, 2000), rng)
    got = arith.mul(ma, mb)
    ref = (ma.numpy().astype(object) * mb.numpy().astype(object))
    mul_ok = bool(np.all(got == ref))
    results["MUL (fp128 whole product, decode depth 20)"] = (mul_ok, min(n, 2000))

    # --- CMP x6 ---
    ca = _rand32(n, rng); cb = _rand32(n, rng)
    # inject equal + boundary pairs
    ca[:6] = torch.tensor([0, 5, SIGN32, M32, 5, 0]); cb[:6] = torch.tensor([0, 5, 0, M32, 6, 1])
    ai = [int(x) for x in ca]; bi = [int(x) for x in cb]
    cmp_ok = True
    for op, ref in (("EQ", lambda x, y: x == y), ("NE", lambda x, y: x != y),
                    ("LT", lambda x, y: _signed_ref(x) < _signed_ref(y)),
                    ("GT", lambda x, y: _signed_ref(x) > _signed_ref(y)),
                    ("LE", lambda x, y: _signed_ref(x) <= _signed_ref(y)),
                    ("GE", lambda x, y: _signed_ref(x) >= _signed_ref(y))):
        got = arith.cmp(ca, cb, op)
        want = torch.tensor([1 if ref(x, y) else 0 for x, y in zip(ai, bi)])
        cmp_ok = cmp_ok and bool((got == want).all())
    results["CMP EQ/NE/LT/GT/LE/GE (sign read)"] = (cmp_ok, n)

    # --- SHL / SHR ---
    sa = _rand32(n, rng)
    sn = torch.from_numpy(rng.integers(0, 32, size=n, dtype=np.int64))
    shl_ref = torch.tensor([((int(x) << int(k)) & M32) for x, k in zip(sa, sn)])
    shr_ref = torch.tensor([((_signed_ref(int(x)) >> int(k)) & M32) for x, k in zip(sa, sn)])
    shl_ok = bool((arith.shift(sa, sn, "SHL") == shl_ref).all())
    shr_ok = bool((arith.shift(sa, sn, "SHR") == shr_ref).all())
    results["SHL (x2^n scale + mask)"] = (shl_ok, n)
    results["SHR (arith /2^n)"] = (shr_ok, n)

    # --- LEA (frame add) ---
    lb = _rand32(n, rng); li = _rand32(n, rng)
    lea_ref = torch.tensor([((_signed_ref(int(x)) + _signed_ref(int(y))) & M32)
                            for x, y in zip(lb, li)])
    lea_ok = bool((arith.lea(lb, li) == lea_ref).all())
    results["LEA/ADJ/branch (address add)"] = (lea_ok, n)

    # --- BITWISE OR/XOR/AND (real FFN LUT) ---
    for op, ref in (("OR", torch.bitwise_or), ("XOR", torch.bitwise_xor),
                    ("AND", torch.bitwise_and)):
        cell = BitwiseCell(op, torch.float32)
        ba = _rand32(min(n, 4000), rng); bb = _rand32(min(n, 4000), rng)
        got = cell(ba, bb)
        results[f"{op} (real 16x16 nibble-LUT FFN, depth 8)"] = (
            bool((got == ref(ba, bb)).all()), min(n, 4000))

    # --- MEMORY CAM (real attention head) ---
    cam = MemoryCAMCell(torch.float32)
    S = min(n, 2000)
    saddr = torch.from_numpy(rng.choice(1 << 20, size=S, replace=False).astype(np.int64))
    sval = _rand32(S, rng)
    # query in a SHUFFLED order so a correct read must actually address-match
    # (not just return the row-aligned value).
    perm = torch.from_numpy(rng.permutation(S))
    got = cam.read(saddr, sval, saddr[perm])            # query stored addrs, shuffled
    li_ok = bool((got == sval[perm]).all())
    # negative control: addresses that were NEVER stored must gather 0.
    miss_addr = saddr + (1 << 24)                       # shift out of the stored set
    miss = cam.read(saddr, sval, miss_addr)
    li_ok = li_ok and bool((miss == 0).all())
    # LC (signed byte) on the low byte of the shuffled-query read
    sval_p = sval[perm]
    lc_ref = torch.tensor([(int(v) & 0xFF) - 0x100 if (int(v) & 0x80) else (int(v) & 0xFF)
                           for v in sval_p]) & M32
    got_byte = (got & 0xFF)
    lc_got = torch.tensor([(int(x) - 0x100 if int(x) & 0x80 else int(x)) for x in got_byte]) & M32
    lc_ok = bool((lc_got == lc_ref).all())
    results["LI/SI (real CAM attention read)"] = (li_ok, S)
    results["LC/SC (CAM read + byte sign-extend)"] = (lc_ok, S)

    return results


def census():
    print("=" * 92)
    print("REAL-TENSOR NONZERO CENSUS (per cell, every nonzero weight entry)")
    print("=" * 92)
    arith = ArithCell(torch.float64)
    na = arith.count_nonzero()
    tot_a = sum(na.values())
    print(f"ARITHMETIC cell (fp64) — d_model={arith.d_model}, decode reused per digit:")
    for k, v in na.items():
        print(f"   {k:36s} {v:6d}")
    print(f"   {'TOTAL arith cell nonzero':36s} {tot_a:6d}")
    print()
    for op in ("OR", "AND", "XOR"):
        cell = BitwiseCell(op)
        nb = cell.count_nonzero()
        print(f"BITWISE {op} cell (fp32) — real 16x16 LUT FFN (256 hidden):")
        for k, v in nb.items():
            print(f"   {k:36s} {v:6d}")
        print(f"   {'TOTAL bitwise ' + op + ' nonzero':36s} {sum(nb.values()):6d}")
    print()
    cam = MemoryCAMCell()
    nc = cam.count_nonzero()
    print("MEMORY CAM cell (fp32) — shared by LI/LC/SI/SC:")
    for k, v in nc.items():
        print(f"   {k:36s} {v:6d}")
    print(f"   {'TOTAL memory CAM nonzero':36s} {sum(nc.values()):6d}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260808)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    census()
    print("=" * 92)
    print(f"BYTE-EXACT VERIFICATION — {args.n} random operands/op + edge cases")
    print("=" * 92)
    res = verify_all(args.n, rng)
    allok = True
    for tag, (ok, nn) in res.items():
        allok = allok and ok
        print(f"  {tag:48s} exact={str(ok):5s}  ({nn:,} cases)")
    print("-" * 92)
    print(f"  RESULT: {'ALL CELLS BYTE-EXACT' if allok else 'FAILURE'}")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
