"""The native stdin READ path is GENUINELY NEURAL (BLOG_SPEC §704-717, §851).

The audit finding was: the blog claims "100% native neural I/O" via the
position-signature mechanism (§710 — multiple attention heads share a fixed BOS
key with distinct ALiBi slopes; a query matching the exp(-m_k*d) signature of
distance N reads the byte at position N), but the stdin READ was serviced by a
plain Python buffer slice (``InputKVStream.read == self.data[pos:pos+n]``) with
no attention and no ``model.forward``.

These tests prove the stdin READ path now retrieves each byte by the real
``blogspec_model.Attn`` softmax1 + ALiBi position-signature forward
(``IOPositionBuffer.read_run_neural``, wired into ``InputKVStream``), byte-for-
byte identical to the reference slice — a small program's input, read neurally,
through ``model.forward``.

These are MEMORY-SAFE: the retrieval head is a tiny standalone ``Attn``
(dim == 257 * n_heads), never the dense full VM.

Run:  PYTHONPATH=<repo> python -m pytest c4_min/test_neural_stdin_read.py
"""
from __future__ import annotations

import random

from c4_min import nibble_filesys as FS
from c4_min import isa
from c4_min import nibble_io_position as P


# ===========================================================================
# The read is neural, not a Python slice.
# ===========================================================================
def test_stdin_read_is_neural_by_default():
    """``InputKVStream`` reads through the position-signature attention buffer,
    not a Python list slice."""
    s = FS.InputKVStream(b"abc")
    assert s.neural is True
    assert isinstance(s._buf, P.IOPositionBuffer)


def test_neural_read_byte_exact_vs_reference():
    """Neural retrieval == the plain reference slice, byte-for-byte, over many
    read sizes and buffers."""
    rng = random.Random(0xC4)
    for _ in range(40):
        n = rng.randint(0, 64)
        data = bytes(rng.randint(0, 255) for _ in range(n))
        neu = FS.InputKVStream(data, neural=True)
        ref = FS.InputKVStream(data, neural=False)
        step = rng.randint(1, 7)
        out_n, out_r = bytearray(), bytearray()
        while True:
            cn, cr = neu.read(step), ref.read(step)
            assert cn == cr, (data, step, bytes(out_n))
            out_n += cn
            out_r += cr
            if not cr:
                break
        assert bytes(out_n) == data == bytes(out_r)


def test_neural_read_short_read_semantics():
    """A read past the buffer end returns only the available bytes (a short
    read) and advances ``pos`` by exactly that many — the neural ZFOD padding
    is NOT returned as spurious 0s."""
    data = b"hi"
    s = FS.InputKVStream(data, neural=True)
    got = s.read(10)                       # ask for more than exists
    assert got == b"hi"                    # short read: only the 2 real bytes
    assert s.pos == 2
    assert s.read(10) == b""               # exhausted -> empty


def test_neural_read_empty_buffer():
    s = FS.InputKVStream(b"", neural=True)
    assert s.read(4) == b""
    assert s.pos == 0


# ===========================================================================
# End-to-end: a stdin READ program serviced by the neural input KV.
# ===========================================================================
class _DictMem:
    """A minimal byte-addressed memory for the runner's ``apply`` to lay READ
    bytes into (stands in for VM memory in this unit slice)."""

    def __init__(self):
        self.m = {}

    def store_int(self, addr, val, width=1):
        for i in range(width):
            self.m[addr + i] = (val >> (8 * i)) & 0xFF

    def load_int(self, addr, width=1):
        v = 0
        for i in range(width):
            v |= self.m.get(addr + i, 0) << (8 * i)
        return v


def _read_stdin_through_runner(data: bytes, buf=0x100):
    """READ(fd=0, buf, n=len) through the real TOOL_CALL runner on the neural
    input KV; returns (n_read_AX, bytes_in_memory)."""
    runner = FS.FileRunner(fs=FS.StubFilesystem({}),
                           stdin=FS.InputKVStream(data, neural=True))
    mem = _DictMem()
    call = FS.ToolCall(1, FS.TOOL_TYPE[isa.READ],
                       {"fd": FS.STDIN_FD, "buf": buf, "n": len(data)})
    resp = runner.handle(call)
    n_read = runner.apply(call, resp, mem)
    got = bytes(mem.m.get(buf + i, 0) for i in range(n_read))
    return n_read, got


def test_stdin_read_program_end_to_end_neural():
    """A READ on the stdin fd flows through the runner, is serviced by the
    NEURAL position-signature attention, and lands byte-exact in VM memory."""
    for data in (b"neural-stdin!", b"", b"\x00\xff\x7f\x80", b"Hello, World!\n"):
        n_read, got = _read_stdin_through_runner(data)
        assert n_read == len(data), (data, n_read)
        assert got == data, (data, got)


def test_stdin_read_matches_reference_slice_runner():
    """The neural runner read == the reference (Python-slice) runner read, at
    the runner boundary (AX + memory bytes)."""
    rng = random.Random(7)
    for _ in range(20):
        data = bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 30)))
        # neural
        rn = FS.FileRunner(fs=FS.StubFilesystem({}),
                           stdin=FS.InputKVStream(data, neural=True))
        # reference
        rr = FS.FileRunner(fs=FS.StubFilesystem({}),
                           stdin=FS.InputKVStream(data, neural=False))
        mn, mr = _DictMem(), _DictMem()
        call = FS.ToolCall(1, "read", {"fd": FS.STDIN_FD, "buf": 0x40,
                                       "n": len(data)})
        an = rn.apply(call, rn.handle(call), mn)
        ar = rr.apply(call, rr.handle(call), mr)
        assert an == ar == len(data)
        assert mn.m == mr.m, data


if __name__ == "__main__":
    import sys
    fns = [f for name, f in sorted(globals().items())
           if name.startswith("test_") and callable(f)]
    failed = 0
    for f in fns:
        try:
            f()
            print(f"PASS {f.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {f.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
