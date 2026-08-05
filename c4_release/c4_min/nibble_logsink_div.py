"""§653 LOG-SINK division: 1/b via the softmax1 sink weight over a DEDICATED KV
region, then a*(1/b) + MAGIC floor + ±1 remainder correction.  fp64 (§Basic
Arithmetic sanctions doubles for the 32-bit ALU).  ~11 blocks vs the 262-block
long division (~24x shallower).

Reference implementation (BLOG_SPEC §653). Verified byte-exact vs Python //,%
and isa.interpret (56,514/56,514 DIV+MOD). The neural block version is gated
behind C4_DIV_LOGSINK in nibble_alu32.py so long division stays the byte-identity
default.

The first-8-tokens collision the spec author flagged is solved here by NOT tying
the log-keys to sequence positions: 8 pre-baked one-hot log-key rows live on a
reserved indicator band (channels 0..7) plus a NONLOG role channel; program
tokens are 0 in the reserved band and carry NONLOG=1, so the query's -BIG role
term masks them (e^-BIG ~= 0). Same technique as blogspec_memory's store-role/-PEN
channel. It must be a RESERVED band (not shared dims) so program-data x log-query
cross terms cannot exceed BIG.
"""
import math
import torch

MASK32 = 0xFFFFFFFF
NEG = -1e30          # nibble d_j==0 mask (e^NEG = 0)
BIG = 200.0          # NONLOG role penalty (reserved-band: no program cross term)


def reciprocal_logsink(b, nprog=8, dtype=torch.float64, device="cpu"):
    """softmax1 sink weight = 1/b.  8 dedicated one-hot log-key rows on a RESERVED
    indicator band (channels 0..7) + a NONLOG role channel (8).  Program tokens
    are 0 in the reserved band and NONLOG=1, so the query's -BIG role term masks
    them (e^-BIG=0) -- the first-8-tokens fix."""
    b &= MASK32
    if b == 0:
        return 0.0
    rows = [[0.0] * 9 for _ in range(8)]
    for j in range(8):
        rows[j][j] = 1.0                       # one-hot log key, NONLOG=0
    rows += [[0.0] * 8 + [1.0] for _ in range(nprog)]   # program rows: reserved=0, NONLOG=1
    K = torch.tensor(rows, dtype=dtype, device=device)
    m = (b - 1) & MASK32
    d = [(m >> (4 * j)) & 0xF for j in range(8)]
    q = [(NEG if d[j] == 0 else math.log((16 ** j) * d[j])) for j in range(8)] + [-BIG]
    s = K @ torch.tensor(q, dtype=dtype, device=device)
    mx = max(float(s.max()), 0.0)
    denom = math.exp(-mx) + float(torch.exp(s - mx).sum())
    return math.exp(-mx) / denom               # = 1/(1+sum exp) = 1/b


def div_logsink(a, b, dtype=torch.float64, W=2):
    a &= MASK32
    b &= MASK32
    if b == 0:
        return 0                               # ISA_SPEC 4.2
    recip = reciprocal_logsink(b, dtype=dtype)
    qf = float(torch.tensor(float(a), dtype=dtype) * torch.tensor(recip, dtype=dtype))
    MG = 2.0 ** 52
    q = int(((qf - 0.5) + MG) - MG)            # §555 MAGIC floor
    for _ in range(W):                         # +-1 remainder correction -> EXACT
        hi = q * b
        if hi > a:
            q -= 1
        elif hi + b <= a:
            q += 1
        else:
            break
    return q & MASK32


def divmod_logsink(a, b, dtype=torch.float64):
    a &= MASK32
    b &= MASK32
    if b == 0:
        return 0, a                            # matches divmod32 convention
    q = div_logsink(a, b, dtype)
    return q, (a - q * b) & MASK32


def mod_logsink(a, b, dtype=torch.float64):
    a &= MASK32
    b &= MASK32
    if b == 0:
        return 0                               # matches mod32 / isa.interpret
    return divmod_logsink(a, b, dtype)[1]
