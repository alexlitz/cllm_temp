"""Scratch probe: prove the READ(stdin)->LC->PRTF loop through the pure-forward
transformer VM.  A user message enters via the input-KV (READ fd=0), a byte is
read back with LC, and PRTF emits output — the whole thing through model.forward.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.nibble_pure_forward_complete import (
    build_pure_forward_complete_model, run_pure_forward_complete,
)

BUF = 0x40   # low-256 window so the LC CAM can read the read-back bytes


def _seed_cstring(d, addr, s):
    if isinstance(s, str):
        s = s.encode("latin-1")
    for i, b in enumerate(s):
        d[addr + i] = b
    d[addr + len(s)] = 0


def main():
    # program: READ(fd=0, buf=BUF, n=5) from stdin; then LC buf[0], LC buf[1],
    # then PRTF a literal greeting.  Marshalling matches dispatch_file_op:
    #   READ: n=AX, buf=pop, fd=pop  -> push fd, push buf, IMM n, READ
    # NOTE: pointers ride the 8-bit AX byte (the emitted register is masked 0xFF),
    # so string/buffer addresses must live in the LOW-256 window.
    FMT = 0x80
    code = isa.assemble([
        ("IMM", 0), ("PSH", 0),          # push fd=0 (stdin)
        ("IMM", BUF), ("PSH", 0),        # push buf
        ("IMM", 5),                      # AX = n = 5
        ("READ", 0),                     # AX = read(0, buf, 5) -> n_read
        ("IMM", BUF + 0), ("LC", 0),     # AX = buf[0]
        ("IMM", BUF + 1), ("LC", 0),     # AX = buf[1]
        ("IMM", FMT), ("PSH", 0), ("PRTF", 0),  # printf("hi\n")
        ("HALT", 0),
    ])
    data = {}
    _seed_cstring(data, FMT, "hi\n")

    m, L = build_pure_forward_complete_model(
        code_size=len(code) + 2, include_bitwise=True, include_divmod=False)

    fs = FS.StubFilesystem({})
    fio = FS.FileOpState(runner=FS.FileRunner(fs=fs, stdin=FS.InputKVStream(b"AB")))
    tr = run_pure_forward_complete(
        m, L, code, max_steps=64, mask=0xFF, fio=fio, data_seg=data, verbose=True)

    print("TRACE:", tr)
    print("STDOUT:", bytes(fio.runner.stdout))
    print("CALLS:", [c.type for c in fio.calls])
    # buf[0]='A'=65, buf[1]='B'=66
    print("LC buf[0]:", tr[7], "LC buf[1]:", tr[9])


if __name__ == "__main__":
    main()
