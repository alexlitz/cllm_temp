#!/usr/bin/env python3
"""CAPSTONE PHASE B — snapshot-compile the CURRENT id_port/doom_run.c ONCE.

Guardrail: another agent is concurrently editing id_port/doom_run.c (32-bit
re-target).  We compile it ONCE at t0 and save the instruction + data arrays
into c4_release, then work ONLY from the snapshot (id_port is read-only-at-t0).
A's edits change word-width correctness, NOT instruction count/structure, so
the LOAD/FETCH/ATTENTION infra we build on the snapshot is valid.

Saves:
  _doom_bytecode_snapshot.npz  -- bytecode(int64), data(int64), isa op/imm arrays
  _doom_bytecode_snapshot.json -- meta (counts, opcode mix, hashes)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

DOOM_C = Path("/home/alexlitz/Documents/misc/c4_doom/id_port/doom_run.c")
OUT_NPZ = _HERE / "_doom_bytecode_snapshot.npz"
OUT_JSON = _HERE / "_doom_bytecode_snapshot.json"


def main():
    from src.compiler import compile_c, Op
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    from c4_min import isa

    import re
    src = DOOM_C.read_text()
    src_sha = hashlib.sha256(src.encode()).hexdigest()
    print(f"doom_run.c: {len(src)} bytes  sha256={src_sha[:16]}")

    t0 = time.time()
    # doom_run.c references fstat/lseek — these are VM SYSCALLS the driver
    # registers (not real undefined functions).  Iteratively stub any Undefined
    # function/global so the instruction STREAM materializes; the stub set is
    # recorded in meta.  A's edits change word-width correctness, NOT the
    # instruction count/structure, so this snapshot is valid for the
    # LOAD/FETCH/ATTENTION infra.
    stubs_f, stubs_g, seen = [], [], set()
    for _ in range(400):
        stub_src = ("".join(f"int {g};\n" for g in stubs_g)
                    + "".join(f"int {f}(){{ return 0; }}\n" for f in stubs_f)
                    + src)
        try:
            bytecode, data = compile_c(stub_src)
            break
        except SyntaxError as e:
            s = str(e)
            mf = re.search(r"Undefined function: (\w+)", s)
            mg = re.search(r"Undefined: (\w+)", s)
            if mf and mf.group(1) not in seen:
                seen.add(mf.group(1)); stubs_f.append(mf.group(1)); continue
            if mg and mg.group(1) not in seen:
                seen.add(mg.group(1)); stubs_g.append(mg.group(1)); continue
            raise
    else:
        raise RuntimeError("stub-iter exhausted")
    print(f"compile_c: {len(bytecode)} instrs, {len(data)} data words, "
          f"{time.time()-t0:.1f}s  (stubbed fns={stubs_f}, globals={stubs_g})")

    # ISA decode (op, imm) with LEA/ENT/ADJ slot re-encode.
    code = bytecode_to_isa(bytecode)
    ops = np.array([c.op for c in code], dtype=np.int64)
    imms = np.array([c.imm for c in code], dtype=np.int64)

    bc = np.array([int(w) for w in bytecode], dtype=np.int64)
    dt = np.array([int(w) for w in data], dtype=np.int64)

    # opcode mix
    op_names = {}
    for nm in dir(Op):
        v = getattr(Op, nm)
        if isinstance(v, int) and not nm.startswith("_"):
            op_names[int(v)] = nm
    raw_counts = Counter(int(w) & 0xFF for w in bytecode)
    mix = {op_names.get(op, f"op{op}"): n for op, n in
           sorted(raw_counts.items(), key=lambda x: -x[1])}

    np.savez(OUT_NPZ, bytecode=bc, data=dt, ops=ops, imms=imms)
    bc_sha = hashlib.sha256(bc.tobytes()).hexdigest()

    meta = {
        "doom_c_path": str(DOOM_C),
        "doom_c_sha256": src_sha,
        "doom_c_bytes": len(src),
        "n_instr": len(bytecode),
        "n_data": len(data),
        "bytecode_sha256": bc_sha,
        "opcode_mix": mix,
        "op_names": op_names,
        "pc_max": len(bytecode) - 1,
        "isa_names": {int(k): v for k, v in isa.NAMES.items()},
        "stubbed_functions": stubs_f,
        "stubbed_globals": stubs_g,
        "compiled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2))
    print(f"SNAPSHOT n_instr={len(bytecode)} pc_max={len(bytecode)-1} "
          f"bc_sha={bc_sha[:16]}")
    print("opcode mix (top 15):")
    for i, (nm, n) in enumerate(mix.items()):
        if i >= 15:
            break
        print(f"  {nm:6s} {n:8d}  {100.0*n/len(bytecode):5.1f}%")
    print(f"saved: {OUT_NPZ.name}, {OUT_JSON.name}")


if __name__ == "__main__":
    main()
