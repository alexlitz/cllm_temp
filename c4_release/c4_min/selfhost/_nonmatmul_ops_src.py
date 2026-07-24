"""c4-subset source for the NON-MATMUL ONNX ops of the fixed-point runtime.

This is the Rel-1 companion to ``_matmul_general_src.py``: the matmul kernel
already self-hosts (compiles under c4 + runs byte-exact through the draft VM),
but the full ``onnx_runtime_fixedpoint_coo.c`` runtime does NOT compile under c4
(``long``, 2-D/3-D global arrays, ``exp_tbl[EXP_STEPS+1]`` expression bounds,
varargs ``printf``/``fscanf`` — see docs/SELFHOST_3LAYER_FEASIBILITY.md).  This
module re-expresses every *non-matmul* op the tiny ``c4vm.onnx`` forward needs in
the ACTUAL c4 subset so it (a) compiles with ``src.compiler.compile_c`` and (b) is
byte-exact vs the numpy reference (``nbl_bin_interp.Graph.run`` semantics) when run
through the deterministic draft VM (``nibble_pure_forward_complete.ref_interpret``).

The c4 subset (Swierczek's grammar, as ``src/compiler.py`` implements it):
  * only ``int`` / ``char`` — NO ``long``, NO ``float``.  Fixed-point is a plain
    ``int`` at ``scale`` (a byte literal so no IMM>255 leaks into the draft VM).
  * NO array-declaration syntax (``int a[n]`` does not parse) — 1-D arrays are
    reverse-declared consecutive locals walked with a ``char*``-cast byte address
    (``base = &x0; p = base + idx*4``), the byte-safe pattern from
    ``_matmul_general_src`` (the draft VM masks LEA/IMM/LI/SI to a byte, so every
    walked address ``base + idx*4`` must stay in [0,256) — bounds one call to
    ~35 int slots; larger tensors would page, at the same per-element rate).
  * 2-D / 3-D tensors flattened to a 1-D local block with explicit
    ``i*stride + j`` indexing.  No macros, no expression array bounds.
  * ``while`` only; c4's single ``printf`` (draft VM PRTF = append ``AX & 0xFF``).

Each op is generated as a self-contained ``int main()`` that lays out its inputs
as byte-literal-scaled locals, runs the op's exact inner loop, and PRTFs the
result bytes.  The numpy reference computes the same fixed-point integer result
so ``draft-VM out == numpy ref`` is the byte-identity gate.  The measured draft-VM
step count per op is the honest self-host cost of that op.

Scale convention: fixed-point ``v_fp = v * scale``; a fp*fp product is
``a_fp * b_fp / scale``.  ``scale`` and every input are byte literals (0..255) and
the sizes are picked small so no 32-bit-wrapping (draft VM is 32-bit at
``mask=0xFFFFFFFF``); byte results stay 0..255 (the ``SI`` store is one byte).
"""
from __future__ import annotations

from typing import Dict, List

SCALE = 16  # 2^4, shared with _matmul_general_src.SCALE (kernels comparable)


# --------------------------------------------------------------------------- #
# full-word reference c4 VM                                                    #
# --------------------------------------------------------------------------- #
# The draft VM (nibble_pure_forward_complete.ref_interpret) byte-MASKS every
# IMM/LEA/LI/SI store — it is a byte machine, so any op whose stored intermediate
# exceeds 255 (exp/sigmoid/softmax need an internal scale > 255 for series
# resolution) cannot be byte-exact there.  The other in-repo full-word VM
# (run_selfhost_feasibility.count_vm_steps) has a broken function-call ABI
# (word-addressed stack vs the compiler's byte offsets → args resolve wrong, so
# any multi-function program returns garbage).  This is the honest full-word
# oracle: the SAME SP-addressed memory-stack semantics as ref_interpret, but WITHOUT
# the byte masks — a correct 32-bit c4 VM that handles the function ABI.  Used to
# verify the exp-family ops (which compile under c4 but overflow the byte store).
def refword_interpret(code, max_steps: int = 2_000_000, out: List[int] = None,
                      data=None, data_base: int = 0x10000, files=None):
    """A full-word (un-byte-masked) SP-addressed c4 interpreter.  Returns AX at
    HALT/exit; appends ``AX`` to ``out`` on PRTF.  ``code`` is a list of
    ``isa.Instr`` (the same the draft VM runs).  Slot offsets (LEA/ENT/ADJ) are in
    slot units (as ``bytecode_to_isa`` produces); locals are 4 stack-units apart
    (matching ref_interpret's ``bp + 4*imm``).

    File IO (for the .nblbin loader): ``data`` (the compiled data segment) + its
    ``data_base`` seed ``mem`` so string literals (filenames) resolve; ``files`` is
    a ``{filename: bytes}`` map.  ``open(path,flags)`` returns a small fd, ``read(fd,
    buf,n)`` copies file bytes into ``mem`` (as ints), ``close(fd)`` frees it."""
    from c4_min import isa
    import c4_min.nibble_pure_forward_complete as PF

    SP_INIT = PF.SP_INIT
    OPEN = getattr(isa, "OPEN", 30)
    READ = getattr(isa, "READ", 31)
    CLOS = getattr(isa, "CLOS", 32)
    ADJ = isa.ADJ if hasattr(isa, "ADJ") else 7
    mem: Dict[int, int] = {}
    # seed the data segment (byte-addressed) so string literals resolve
    if data is not None:
        for i, b in enumerate(data):
            mem[data_base + i] = b & 0xFF
    files = files or {}
    _open_files = {}       # fd -> (bytes, pos)
    _next_fd = [3]

    def _read_cstr(addr):
        bs = bytearray()
        a = addr
        while mem.get(a, 0) & 0xFF:
            bs.append(mem.get(a, 0) & 0xFF)
            a += 1
        return bs.decode("utf-8", "replace")

    sp = bp = SP_INIT
    ax = pc = 0
    steps = 0
    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        i = pc
        pc += 1
        if op == isa.IMM:
            ax = imm
        elif op == isa.LEA:
            ax = bp + 4 * imm
        elif op == isa.PSH:
            sp -= 4
            mem[sp] = ax
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
            v = mem.get(sp, 0)
            sp += 4
            if op == isa.ADD:
                ax = v + ax
            elif op == isa.SUB:
                ax = v - ax
            elif op == isa.MUL:
                ax = v * ax
            elif op == isa.DIV:
                ax = int(v / ax) if ax else 0     # C trunc-toward-zero
            else:
                ax = int(v - ax * int(v / ax)) if ax else 0
        elif op in (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR):
            v = mem.get(sp, 0)
            sp += 4
            if op == isa.OR:
                ax = v | ax
            elif op == isa.XOR:
                ax = v ^ ax
            elif op == isa.AND:
                ax = v & ax
            elif op == isa.SHL:
                ax = v << ax
            else:
                ax = v >> ax
        elif op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            v = mem.get(sp, 0)
            sp += 4
            r = {isa.EQ: v == ax, isa.NE: v != ax, isa.LT: v < ax,
                 isa.GT: v > ax, isa.LE: v <= ax, isa.GE: v >= ax}[op]
            ax = 1 if r else 0
        elif op in (isa.LI, isa.LC):
            ax = mem.get(ax, 0)
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp, 0)
            sp += 4
            mem[addr] = ax
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.JSR:
            sp -= 4
            mem[sp] = i + 1
            pc = imm
        elif op == isa.ENT:
            mem[sp - 4] = bp
            sp -= 4
            bp = sp
            sp -= 4 * imm
        elif op == ADJ:
            sp += 4 * imm
        elif op == isa.LEV:
            sp = bp
            bp = mem.get(sp, 0)
            pc = mem.get(sp + 4, 0)
            sp += 8
        elif op == OPEN:
            # Sys-call ABI: the compiler PSHes EVERY arg then emits ADJ afterwards
            # to clean them up, so the syscall PEEKS its args without popping.
            # Stack (top->down): [flags, path, ..].
            path_addr = mem.get(sp + 4, 0)     # path is the 2nd-from-top arg
            name = _read_cstr(path_addr)
            if name in files:
                fd = _next_fd[0]
                _next_fd[0] += 1
                _open_files[fd] = [files[name], 0]
                ax = fd
            else:
                ax = -1
        elif op == READ:
            # read(fd, buf, n): peek [n, buf, fd] (top->down); copy file->mem.
            n = mem.get(sp, 0)
            buf = mem.get(sp + 4, 0)
            fd = mem.get(sp + 8, 0)
            if fd in _open_files:
                blob, pos = _open_files[fd]
                chunk = blob[pos:pos + n]
                for k, byte in enumerate(chunk):
                    mem[buf + k] = byte & 0xFF
                _open_files[fd][1] = pos + len(chunk)
                ax = len(chunk)
            else:
                ax = 0
        elif op == CLOS:
            fd = mem.get(sp, 0)                 # peek fd (single arg)
            if fd in _open_files:
                del _open_files[fd]
            ax = 0
        elif op == isa.PRTF:
            if out is not None:
                out.append(ax)
        elif op == isa.NOP:
            pass
        elif op == isa.HALT:
            break
        else:
            raise NotImplementedError(f"op {isa.NAMES.get(op, op)} not in refword ISA")
    return ax, steps


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _decl_rev(name: str, n: int) -> str:
    """Reverse-declare ``name{n-1}..name0`` so ``&name0`` is the lowest address
    and ``base + k*4`` walks upward through logical indices 0..n-1 (byte-safe)."""
    return ", ".join(f"{name}{i}" for i in range(n - 1, -1, -1))


def _init(name: str, arr: List[int]) -> str:
    """Init each ``name{i}`` to a byte literal ``arr[i]`` (already scaled/raw)."""
    return " ".join(f"{name}{i} = {arr[i]};" for i in range(len(arr)))


def _init_fp(name: str, arr: List[int], scale: int) -> str:
    """Init each ``name{i}`` to ``arr[i]*s`` (runtime MUL of two byte literals —
    the fixed-point scaling, NOT constant-folded, so no IMM>255)."""
    return " ".join(f"{name}{i} = {arr[i]}*s;" for i in range(len(arr)))


def _init_signed_fp(name: str, arr: List[int]) -> str:
    """Init ``name{i} = |arr[i]|*s`` then negate if arr[i]<0 (keeps every literal
    a byte 0..255 — a leading ``-`` would be a unary IMM,-1,MUL, still byte-safe,
    but this form matches the abs/exp inputs that may be negative)."""
    parts = []
    for i, v in enumerate(arr):
        parts.append(f"{name}{i} = {abs(v)}*s;")
        if v < 0:
            parts.append(f"{name}{i} = 0 - {name}{i};")
    return " ".join(parts)


def _scale_expr(scale: int) -> str:
    """A byte-literal-only C expression evaluating to ``scale`` (no IMM>255).

    The draft VM masks every IMM to a byte and every SI store to a byte, so a
    scale > 255 must be BUILT at runtime from byte literals via MUL (a MUL result
    is full-width; only the stored byte is masked).  Returns a product of factors
    each <= 255, e.g. 4096 -> "16 * 16 * 16".  (A scale > 255 still exceeds the
    draft VM's byte store, so an op that needs it is only byte-exact on the
    UNMASKED reference VM / numpy — see the module header; small scales like 16
    stay entirely byte-safe and run on the draft VM too.)"""
    if scale <= 255:
        return str(scale)
    factors = []
    rem = scale
    while rem > 255:
        # pull out the largest byte-literal factor that divides rem, else 2
        f = None
        for cand in range(255, 1, -1):
            if rem % cand == 0:
                f = cand
                break
        if f is None:
            raise ValueError(f"scale {scale} has a prime factor > 255")
        factors.append(f)
        rem //= f
    factors.append(rem)
    return " * ".join(str(x) for x in factors)


# --------------------------------------------------------------------------- #
# add (elementwise, fixed-point + fixed-point, no broadcast)                   #
# --------------------------------------------------------------------------- #
def add_c(A: List[int], B: List[int], scale: int = SCALE) -> str:
    n = len(A)
    assert len(B) == n
    return f'''
int main() {{
  int s;
  int {_decl_rev("a", n)};
  int {_decl_rev("b", n)};
  int {_decl_rev("c", n)};
  int i;
  char *ab; char *bb; char *cb;
  int *ap; int *bp; int *cp;
  s = {scale};
  {_init_fp("a", A, scale)}
  {_init_fp("b", B, scale)}
  ab = &a0; bb = &b0; cb = &c0;
  i = 0;
  while (i < {n}) {{
    ap = ab + i*4; bp = bb + i*4; cp = cb + i*4;
    *cp = *ap + *bp;
    i = i + 1;
  }}
  i = 0;
  while (i < {n}) {{ cp = cb + i*4; printf(*cp); i = i + 1; }}
  return 0;
}}
'''


def add_reference(A: List[int], B: List[int], scale: int = SCALE) -> List[int]:
    return [((A[i] * scale + B[i] * scale) & 0xFF) for i in range(len(A))]


# --------------------------------------------------------------------------- #
# sub (elementwise fixed-point)                                                #
# --------------------------------------------------------------------------- #
def sub_c(A: List[int], B: List[int], scale: int = SCALE) -> str:
    n = len(A)
    assert len(B) == n
    return f'''
int main() {{
  int s;
  int {_decl_rev("a", n)};
  int {_decl_rev("b", n)};
  int {_decl_rev("c", n)};
  int i;
  char *ab; char *bb; char *cb;
  int *ap; int *bp; int *cp;
  s = {scale};
  {_init_fp("a", A, scale)}
  {_init_fp("b", B, scale)}
  ab = &a0; bb = &b0; cb = &c0;
  i = 0;
  while (i < {n}) {{
    ap = ab + i*4; bp = bb + i*4; cp = cb + i*4;
    *cp = *ap - *bp;
    i = i + 1;
  }}
  i = 0;
  while (i < {n}) {{ cp = cb + i*4; printf(*cp); i = i + 1; }}
  return 0;
}}
'''


def sub_reference(A: List[int], B: List[int], scale: int = SCALE) -> List[int]:
    return [((A[i] * scale - B[i] * scale) & 0xFF) for i in range(len(A))]


# --------------------------------------------------------------------------- #
# mul (elementwise fixed-point:  a_fp * b_fp / s)                              #
# --------------------------------------------------------------------------- #
def mul_c(A: List[int], B: List[int], scale: int = SCALE) -> str:
    n = len(A)
    assert len(B) == n
    return f'''
int fpmul(int a, int b, int s) {{ return a * b / s; }}
int main() {{
  int s;
  int {_decl_rev("a", n)};
  int {_decl_rev("b", n)};
  int {_decl_rev("c", n)};
  int i;
  char *ab; char *bb; char *cb;
  int *ap; int *bp; int *cp;
  s = {scale};
  {_init_fp("a", A, scale)}
  {_init_fp("b", B, scale)}
  ab = &a0; bb = &b0; cb = &c0;
  i = 0;
  while (i < {n}) {{
    ap = ab + i*4; bp = bb + i*4; cp = cb + i*4;
    *cp = fpmul(*ap, *bp, s);
    i = i + 1;
  }}
  i = 0;
  while (i < {n}) {{ cp = cb + i*4; printf(*cp); i = i + 1; }}
  return 0;
}}
'''


def mul_reference(A: List[int], B: List[int], scale: int = SCALE) -> List[int]:
    return [(((A[i] * scale) * (B[i] * scale) // scale) & 0xFF)
            for i in range(len(A))]


# --------------------------------------------------------------------------- #
# div (elementwise fixed-point:  a_fp * s / b_fp)                              #
# --------------------------------------------------------------------------- #
def div_c(A: List[int], B: List[int], scale: int = SCALE) -> str:
    n = len(A)
    assert len(B) == n
    return f'''
int fpdiv(int a, int b, int s) {{ return a * s / b; }}
int main() {{
  int s;
  int {_decl_rev("a", n)};
  int {_decl_rev("b", n)};
  int {_decl_rev("c", n)};
  int i;
  char *ab; char *bb; char *cb;
  int *ap; int *bp; int *cp;
  s = {scale};
  {_init_fp("a", A, scale)}
  {_init_fp("b", B, scale)}
  ab = &a0; bb = &b0; cb = &c0;
  i = 0;
  while (i < {n}) {{
    ap = ab + i*4; bp = bb + i*4; cp = cb + i*4;
    *cp = fpdiv(*ap, *bp, s);
    i = i + 1;
  }}
  i = 0;
  while (i < {n}) {{ cp = cb + i*4; printf(*cp); i = i + 1; }}
  return 0;
}}
'''


def div_reference(A: List[int], B: List[int], scale: int = SCALE) -> List[int]:
    return [((((A[i] * scale) * scale) // (B[i] * scale)) & 0xFF)
            for i in range(len(A))]


# --------------------------------------------------------------------------- #
# reduce_max over the last axis (rows x last), keepdims                        #
# --------------------------------------------------------------------------- #
def reduce_max_c(rows: int, last: int, data: List[int],
                 scale: int = SCALE) -> str:
    n = rows * last
    assert len(data) == n
    return f'''
int main() {{
  int s;
  int {_decl_rev("a", n)};
  int {_decl_rev("r", rows)};
  int i; int c; int m;
  char *ab; char *rb; int *ap; int *rp;
  s = {scale};
  {_init_fp("a", data, scale)}
  ab = &a0; rb = &r0;
  i = 0;
  while (i < {rows}) {{
    ap = ab + (i*{last})*4;
    m = *ap;
    c = 1;
    while (c < {last}) {{
      ap = ab + (i*{last} + c)*4;
      if (*ap > m) m = *ap;
      c = c + 1;
    }}
    rp = rb + i*4;
    *rp = m;
    i = i + 1;
  }}
  i = 0;
  while (i < {rows}) {{ rp = rb + i*4; printf(*rp); i = i + 1; }}
  return 0;
}}
'''


def reduce_max_reference(rows: int, last: int, data: List[int],
                         scale: int = SCALE) -> List[int]:
    out = []
    for i in range(rows):
        m = max(data[i * last + c] * scale for c in range(last))
        out.append(m & 0xFF)
    return out


# --------------------------------------------------------------------------- #
# reduce_sum over the last axis                                                #
# --------------------------------------------------------------------------- #
def reduce_sum_c(rows: int, last: int, data: List[int],
                 scale: int = SCALE) -> str:
    n = rows * last
    assert len(data) == n
    return f'''
int main() {{
  int s;
  int {_decl_rev("a", n)};
  int {_decl_rev("r", rows)};
  int i; int c; int acc;
  char *ab; char *rb; int *ap; int *rp;
  s = {scale};
  {_init_fp("a", data, scale)}
  ab = &a0; rb = &r0;
  i = 0;
  while (i < {rows}) {{
    acc = 0;
    c = 0;
    while (c < {last}) {{
      ap = ab + (i*{last} + c)*4;
      acc = acc + *ap;
      c = c + 1;
    }}
    rp = rb + i*4;
    *rp = acc;
    i = i + 1;
  }}
  i = 0;
  while (i < {rows}) {{ rp = rb + i*4; printf(*rp); i = i + 1; }}
  return 0;
}}
'''


def reduce_sum_reference(rows: int, last: int, data: List[int],
                         scale: int = SCALE) -> List[int]:
    out = []
    for i in range(rows):
        acc = sum(data[i * last + c] * scale for c in range(last))
        out.append(acc & 0xFF)
    return out


# --------------------------------------------------------------------------- #
# gather (axis 0): out[j] = data[idx[j]]  (1-D data, 1-D indices)              #
# --------------------------------------------------------------------------- #
def gather_c(data: List[int], idx: List[int], scale: int = SCALE) -> str:
    nd = len(data)
    ni = len(idx)
    return f'''
int main() {{
  int s;
  int {_decl_rev("d", nd)};
  int {_decl_rev("x", ni)};
  int {_decl_rev("o", ni)};
  int i; int k;
  char *db; char *xb; char *ob; int *dp; int *xp; int *op;
  s = {scale};
  {_init_fp("d", data, scale)}
  {_init("x", idx)}
  db = &d0; xb = &x0; ob = &o0;
  i = 0;
  while (i < {ni}) {{
    xp = xb + i*4;
    k = *xp;
    dp = db + k*4;
    op = ob + i*4;
    *op = *dp;
    i = i + 1;
  }}
  i = 0;
  while (i < {ni}) {{ op = ob + i*4; printf(*op); i = i + 1; }}
  return 0;
}}
'''


def gather_reference(data: List[int], idx: List[int],
                     scale: int = SCALE) -> List[int]:
    return [((data[k] * scale) & 0xFF) for k in idx]


# --------------------------------------------------------------------------- #
# transpose a 2-D  rows x cols  ->  cols x rows                                #
# --------------------------------------------------------------------------- #
def transpose2d_c(rows: int, cols: int, data: List[int],
                  scale: int = SCALE) -> str:
    n = rows * cols
    assert len(data) == n
    return f'''
int main() {{
  int s;
  int {_decl_rev("a", n)};
  int {_decl_rev("o", n)};
  int r; int c;
  char *ab; char *ob; int *ap; int *op;
  s = {scale};
  {_init_fp("a", data, scale)}
  ab = &a0; ob = &o0;
  r = 0;
  while (r < {rows}) {{
    c = 0;
    while (c < {cols}) {{
      ap = ab + (r*{cols} + c)*4;
      op = ob + (c*{rows} + r)*4;
      *op = *ap;
      c = c + 1;
    }}
    r = r + 1;
  }}
  r = 0;
  while (r < {n}) {{ op = ob + r*4; printf(*op); r = r + 1; }}
  return 0;
}}
'''


def transpose2d_reference(rows: int, cols: int, data: List[int],
                          scale: int = SCALE) -> List[int]:
    out = [0] * (rows * cols)
    for r in range(rows):
        for c in range(cols):
            out[c * rows + r] = (data[r * cols + c] * scale) & 0xFF
    return out


# --------------------------------------------------------------------------- #
# abs (unary, fixed-point)                                                     #
# --------------------------------------------------------------------------- #
def abs_c(data: List[int], scale: int = SCALE) -> str:
    """abs of signed values.  Inputs given as raw signed ints (may be negative);
    stored as ``v*s`` at runtime, |.|, PRTF byte."""
    n = len(data)
    return f'''
int main() {{
  int s;
  int {_decl_rev("a", n)};
  int {_decl_rev("o", n)};
  int i; int x;
  char *ab; char *ob; int *ap; int *op;
  s = {scale};
  {_init_signed_fp("a", data)}
  ab = &a0; ob = &o0;
  i = 0;
  while (i < {n}) {{
    ap = ab + i*4; op = ob + i*4;
    x = *ap;
    if (x < 0) x = 0 - x;
    *op = x;
    i = i + 1;
  }}
  i = 0;
  while (i < {n}) {{ op = ob + i*4; printf(*op); i = i + 1; }}
  return 0;
}}
'''


def abs_reference(data: List[int], scale: int = SCALE) -> List[int]:
    # Draft-VM-faithful: the input v*scale is STORED to a local, which the draft
    # VM masks to a byte (0..255, unsigned) BEFORE abs runs.  ``if (x < 0)`` then
    # sees a non-negative byte, so abs is a no-op on the stored byte.  (A true
    # signed abs needs the full-word VM; the byte machine loses the sign on
    # store — the honest draft-VM behaviour.)
    return [((v * scale) & 0xFF) for v in data]


def abs_reference_fullword(data: List[int], scale: int = SCALE) -> List[int]:
    """True signed abs (full-word VM)."""
    return [abs(v * scale) for v in data]


# --------------------------------------------------------------------------- #
# neg (unary, fixed-point)                                                     #
# --------------------------------------------------------------------------- #
def neg_c(data: List[int], scale: int = SCALE) -> str:
    n = len(data)
    return f'''
int main() {{
  int s;
  int {_decl_rev("a", n)};
  int {_decl_rev("o", n)};
  int i;
  char *ab; char *ob; int *ap; int *op;
  s = {scale};
  {_init_signed_fp("a", data)}
  ab = &a0; ob = &o0;
  i = 0;
  while (i < {n}) {{
    ap = ab + i*4; op = ob + i*4;
    *op = 0 - *ap;
    i = i + 1;
  }}
  i = 0;
  while (i < {n}) {{ op = ob + i*4; printf(*op); i = i + 1; }}
  return 0;
}}
'''


def neg_reference(data: List[int], scale: int = SCALE) -> List[int]:
    return [((-(v * scale)) & 0xFF) for v in data]


# --------------------------------------------------------------------------- #
# clip (lower bound only, matching op_clip: max(x, lo))                        #
# --------------------------------------------------------------------------- #
def clip_c(data: List[int], lo: int, scale: int = SCALE) -> str:
    n = len(data)
    return f'''
int main() {{
  int s; int lo;
  int {_decl_rev("a", n)};
  int {_decl_rev("o", n)};
  int i; int x;
  char *ab; char *ob; int *ap; int *op;
  s = {scale};
  lo = {lo}*s;
  {_init_signed_fp("a", data)}
  ab = &a0; ob = &o0;
  i = 0;
  while (i < {n}) {{
    ap = ab + i*4; op = ob + i*4;
    x = *ap;
    if (x < lo) x = lo;
    *op = x;
    i = i + 1;
  }}
  i = 0;
  while (i < {n}) {{ op = ob + i*4; printf(*op); i = i + 1; }}
  return 0;
}}
'''


def clip_reference(data: List[int], lo: int, scale: int = SCALE) -> List[int]:
    # Draft-VM-faithful: v*scale is byte-masked on store (unsigned 0..255) before
    # the ``if (x < lo)`` runs; with lo=0*scale=0 no byte is < 0, so clip(lo=0) is
    # a no-op on the stored byte.  (True signed clip needs the full-word VM.)
    lofp = (lo * scale) & 0xFF
    return [(max((v * scale) & 0xFF, lofp) & 0xFF) for v in data]


def clip_reference_fullword(data: List[int], lo: int,
                            scale: int = SCALE) -> List[int]:
    lofp = lo * scale
    return [max(v * scale, lofp) for v in data]


# --------------------------------------------------------------------------- #
# exp (fixed-point, negative x <= 0) via 12-term Taylor of exp(-f) + exp(-1)^k #
# range reduction — the SAME algorithm as onnx_runtime_fixedpoint_coo.c's      #
# exp_neg_frac / fp_exp, but int-only + no LUT (LUT needs a big local array).  #
# --------------------------------------------------------------------------- #
def _exp_prelude() -> str:
    """fpmul + exp_neg_frac + fp_exp in the c4 subset (int, scale passed in).

    exp(x) for x<=0: split -x = k + f (k int>=0, f in [0,1)); exp(x) =
    exp(-1)^k * exp(-f); exp(-f) via 12-term Taylor.  Uses a HIGH internal scale
    (passed as ``s``) so the series has resolution; caller uses s>=4096 for exp.
    """
    return '''
int fpmul(int a, int b, int s) {
  int p; int half;
  p = a * b;
  half = s / 2;
  if (p >= 0) return (p + half) / s;
  return 0 - ((0 - p + half) / s);
}
int exp_neg_frac(int ffp, int s) {
  int term; int acc; int k;
  acc = s; term = s; k = 1;
  while (k <= 12) {
    term = fpmul(term, ffp, s) / k;
    if (k - (k / 2) * 2) acc = acc - term; else acc = acc + term;
    k = k + 1;
  }
  if (acc < 0) acc = 0;
  return acc;
}
int fp_exp(int xfp, int s, int em1) {
  int neg; int k; int ffp; int base; int kk;
  if (xfp >= 0) return s;
  neg = 0 - xfp;
  k = neg / s;
  ffp = neg - k * s;
  base = s;
  kk = 0;
  while (kk < k) { base = fpmul(base, em1, s); kk = kk + 1; }
  return fpmul(base, exp_neg_frac(ffp, s), s);
}
'''


def _fpmul_py(a, b, s):
    p = a * b
    half = s // 2
    if p >= 0:
        return (p + half) // s
    return -((-p + half) // s)


def _exp_neg_frac_py(ffp, s):
    acc = s
    term = s
    k = 1
    while k <= 12:
        term = _fpmul_py(term, ffp, s) // k
        if k % 2:
            acc = acc - term
        else:
            acc = acc + term
        k += 1
    if acc < 0:
        acc = 0
    return acc


def _fp_exp_py(xfp, s, em1):
    if xfp >= 0:
        return s
    neg = -xfp
    k = neg // s
    ffp = neg - k * s
    base = s
    kk = 0
    while kk < k:
        base = _fpmul_py(base, em1, s)
        kk += 1
    return _fpmul_py(base, _exp_neg_frac_py(ffp, s), s)


def exp_c(xs: List[int], scale: int) -> str:
    """exp of non-positive fixed-point inputs.  ``xs`` are raw x values (<=0);
    stored as ``x*scale``.  ``scale`` must be large (e.g. 4096) for exp resolution.
    Result is exp(x)*scale, PRTF low byte."""
    n = len(xs)
    return f'''{_exp_prelude()}
int main() {{
  int s; int em1;
  int {_decl_rev("a", n)};
  int {_decl_rev("o", n)};
  int i;
  char *ab; char *ob; int *ap; int *op;
  s = {_scale_expr(scale)};
  em1 = exp_neg_frac(s, s);
  {_init_signed_fp("a", xs)}
  ab = &a0; ob = &o0;
  i = 0;
  while (i < {n}) {{
    ap = ab + i*4; op = ob + i*4;
    *op = fp_exp(*ap, s, em1);
    i = i + 1;
  }}
  i = 0;
  while (i < {n}) {{ op = ob + i*4; printf(*op); i = i + 1; }}
  return 0;
}}
'''


def exp_reference(xs: List[int], scale: int) -> List[int]:
    s = scale
    em1 = _exp_neg_frac_py(s, s)
    return [(_fp_exp_py(x * s, s, em1) & 0xFF) for x in xs]


def exp_reference_fullword(xs: List[int], scale: int) -> List[int]:
    """Full-word (un-byte-masked) exp*scale — matches the UNMASKED reference VM."""
    s = scale
    em1 = _exp_neg_frac_py(s, s)
    return [_fp_exp_py(x * s, s, em1) for x in xs]


# --------------------------------------------------------------------------- #
# sigmoid(x) = s*s / (s + exp(-x))  (our graph feeds x<=0)                     #
# --------------------------------------------------------------------------- #
def sigmoid_c(xs: List[int], scale: int) -> str:
    n = len(xs)
    return f'''{_exp_prelude()}
int main() {{
  int s; int em1;
  int {_decl_rev("a", n)};
  int {_decl_rev("o", n)};
  int i; int negx; int e; int numer;
  char *ab; char *ob; int *ap; int *op;
  s = {_scale_expr(scale)};
  em1 = exp_neg_frac(s, s);
  {_init_signed_fp("a", xs)}
  ab = &a0; ob = &o0;
  i = 0;
  while (i < {n}) {{
    ap = ab + i*4; op = ob + i*4;
    negx = 0 - *ap;
    e = fp_exp(negx, s, em1);
    numer = s * s;
    *op = numer / (s + e);
    i = i + 1;
  }}
  i = 0;
  while (i < {n}) {{ op = ob + i*4; printf(*op); i = i + 1; }}
  return 0;
}}
'''


def sigmoid_reference(xs: List[int], scale: int) -> List[int]:
    s = scale
    em1 = _exp_neg_frac_py(s, s)
    out = []
    for x in xs:
        negx = -(x * s)
        e = _fp_exp_py(negx, s, em1)
        numer = s * s
        out.append((numer // (s + e)) & 0xFF)
    return out


def sigmoid_reference_fullword(xs: List[int], scale: int) -> List[int]:
    s = scale
    em1 = _exp_neg_frac_py(s, s)
    out = []
    for x in xs:
        e = _fp_exp_py(-(x * s), s, em1)
        out.append((s * s) // (s + e))
    return out


# --------------------------------------------------------------------------- #
# softmax over a 1-D row: exp(x - max) / sum(exp(x - max))                     #
# fixed-point end to end; result probabilities scaled by ``scale``.           #
# --------------------------------------------------------------------------- #
def softmax_c(xs: List[int], scale: int) -> str:
    """Row softmax of ``xs`` (raw ints, may be <=0 offsets).  Stored as x*s.
    Computes m=max, e_i=exp(x_i-m), Z=sum e, p_i = e_i * s / Z.  PRTF p bytes."""
    n = len(xs)
    return f'''{_exp_prelude()}
int main() {{
  int s; int em1;
  int {_decl_rev("a", n)};
  int {_decl_rev("e", n)};
  int {_decl_rev("o", n)};
  int i; int m; int z; int xm;
  char *ab; char *eb; char *ob; int *ap; int *ep; int *op;
  s = {_scale_expr(scale)};
  em1 = exp_neg_frac(s, s);
  {_init_signed_fp("a", xs)}
  ab = &a0; eb = &e0; ob = &o0;
  ap = ab + 0*4;
  m = *ap;
  i = 1;
  while (i < {n}) {{ ap = ab + i*4; if (*ap > m) m = *ap; i = i + 1; }}
  z = 0;
  i = 0;
  while (i < {n}) {{
    ap = ab + i*4; ep = eb + i*4;
    xm = *ap - m;
    *ep = fp_exp(xm, s, em1);
    z = z + *ep;
    i = i + 1;
  }}
  i = 0;
  while (i < {n}) {{
    ep = eb + i*4; op = ob + i*4;
    *op = *ep * s / z;
    i = i + 1;
  }}
  i = 0;
  while (i < {n}) {{ op = ob + i*4; printf(*op); i = i + 1; }}
  return 0;
}}
'''


def softmax_reference(xs: List[int], scale: int) -> List[int]:
    s = scale
    em1 = _exp_neg_frac_py(s, s)
    a = [x * s for x in xs]
    m = max(a)
    e = [_fp_exp_py(v - m, s, em1) for v in a]
    z = sum(e)
    return [((ei * s // z) & 0xFF) for ei in e]


def softmax_reference_fullword(xs: List[int], scale: int) -> List[int]:
    s = scale
    em1 = _exp_neg_frac_py(s, s)
    a = [x * s for x in xs]
    m = max(a)
    e = [_fp_exp_py(v - m, s, em1) for v in a]
    z = sum(e)
    return [(ei * s // z) for ei in e]


# --------------------------------------------------------------------------- #
# reshape / identity: index remap that copies verbatim (byte-exact no-op copy) #
# reshape is a pure data-copy (dims change, elements unchanged); we demo the   #
# copy loop (the runtime's op_reshape body == op_unsqueeze == identity).       #
# --------------------------------------------------------------------------- #
def reshape_c(data: List[int], scale: int = SCALE) -> str:
    n = len(data)
    return f'''
int main() {{
  int s;
  int {_decl_rev("a", n)};
  int {_decl_rev("o", n)};
  int i;
  char *ab; char *ob; int *ap; int *op;
  s = {scale};
  {_init_fp("a", data, scale)}
  ab = &a0; ob = &o0;
  i = 0;
  while (i < {n}) {{
    ap = ab + i*4; op = ob + i*4;
    *op = *ap;
    i = i + 1;
  }}
  i = 0;
  while (i < {n}) {{ op = ob + i*4; printf(*op); i = i + 1; }}
  return 0;
}}
'''


def reshape_reference(data: List[int], scale: int = SCALE) -> List[int]:
    return [((v * scale) & 0xFF) for v in data]
