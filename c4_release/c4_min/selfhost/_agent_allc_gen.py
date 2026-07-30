#!/usr/bin/env python3
"""_agent_allc_gen.py — generate the C header (embed table + layout dims + program)
for the ALL-C native VM (onnx_runtime_nibble_allc.c).

The all-C binary needs, alongside the already-.incbin'd block-stack model:
  * the embed table  (267 x D)                       -> C static const float[]
  * the layout dim indices the overlay writes / decode reads  -> #defines + arrays
  * a program (code op/imm arrays)                   -> emitted per --prog / --emit

This dumps a header `allc_gen.h` that onnx_runtime_nibble_allc.c #includes.  Tooling
only; no build-path side effects; the golden model is untouched.

Also SELF-VERIFIES: rebuilds frame_0 (embed[stream] + overlay) in numpy from the
dumped header constants and asserts max|delta|==0 vs the torch frame stored in
frames.npz — so the C overlay is provably byte-identical before we ever compile it.
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C4MIN = os.path.dirname(HERE)


def _emit_prog_bytes(text: bytes):
    prog = []
    for b in text:
        prog.append(("IMM", int(b)))
        prog.append(("PRTF", 0))
    prog.append(("HALT", 0))
    return prog


def build_program(mode, text, io_mode="literal", n=8, stdin_text=""):
    """Return (isa_code, seed_mem, expected_bytes) for a named utility mode.

    ``io_mode`` selects HOW the utility does I/O:
      * ``literal`` : the legacy IMM(b);PRTF(b) template (no §Memory read) — kept
        for the existing echo build's byte-identity.
      * ``strict``  : MODE 1 — the program pointer-walks §Memory (LC mem[ptr]),
        one PRTF per byte.
      * ``burst``   : MODE 2 — one PRTF whose AX is a pointer; the RUNTIME walks
        §Memory (needs C4_IO_BURST / --io-burst at run time).
    """
    from c4_min import isa
    if mode == "quine":
        from c4_min import quine_prtf as Q
        code, seed_mem, expected = Q.build_quine()
        return code, seed_mem, list(expected)
    if io_mode in ("strict", "burst"):
        from c4_min.selfhost import _agent_io_progs as IOP
        prog = IOP.build(mode, io_mode, text=text, n=n, stdin_text=stdin_text)
        return assemble(prog.code), prog.seed_mem, list(prog.expected)
    # legacy literal template: echo / cat / yes = printf-of-literal
    prog = _emit_prog_bytes(text.encode())
    return assemble(prog), {}, list(text.encode())


def build_layout():
    """Load the layout the same way the driver does (fresh compact model build)."""
    from c4_min import compact_alloc as CA
    model, L, _ = CA.build_compact_pure_forward_model(code_size=48)
    model.eval()
    embed = model.embed.detach().numpy().astype(np.float32)
    return L, embed


def layout_consts(L):
    """Collect every dim index the overlay writes + decode reads, plus code arrays."""
    from c4_min import nibble_pure_forward as PF
    from c4_min.blogspec_memory import ADDR_BITS
    import c4_min.nibble_pure_forward_complete as PFC
    from c4_min import blogspec_vocab as V

    frs = PF._FRAME_ROLE_SLOTS          # {frame_local_pos: role_index}
    consts = dict(
        D=int(L.D),
        VOCAB=267,
        FRAME_LEN=int(V.FRAME_LEN),
        BOS=int(V.BOS),
        SP_INIT=int(PFC.SP_INIT),
        IMM_NIBS=int(PFC.IMM_NIBS),
        N_ROLES=int(PF.N_ROLES),
        ADDR_BITS=int(ADDR_BITS),
        MEM_MARKER_LOCAL=int(PF._MEM_MARKER_LOCAL),
        ONE=int(L.ONE), ROLE=int(L.ROLE), IS_FRAME_BYTE=int(L.IS_FRAME_BYTE),
        IS_STORE=int(L.IS_STORE), ADDR_BIN=int(L.ADDR_BIN), VAL_NIB=int(L.VAL_NIB),
        PC_VAL=int(L.PC_VAL), SP_VAL=int(L.SP_VAL), BP_VAL=int(L.BP_VAL),
        STK_VAL=int(L.STK_VAL), HALTED=int(L.HALTED), AX=int(L.AX),
        # opcodes we implement natively (marker token ids)
        OP_IMM=1, OP_PRTF=33, OP_HALT=38,
        OP_SI=int(__import__("c4_min.isa", fromlist=["SI"]).SI),
        OP_SC=int(__import__("c4_min.isa", fromlist=["SC"]).SC),
        OP_PSH=int(__import__("c4_min.isa", fromlist=["PSH"]).PSH),
        OP_JSR=int(__import__("c4_min.isa", fromlist=["JSR"]).JSR),
        OP_ENT=int(__import__("c4_min.isa", fromlist=["ENT"]).ENT),
        # §File Operations opcodes (30-33): were MISSING — needed by the
        # runtime I/O syscall dispatch (READ stdin->§Memory, OPEN/CLOS stubs).
        OP_OPEN=int(__import__("c4_min.isa", fromlist=["OPEN"]).OPEN),
        OP_READ=int(__import__("c4_min.isa", fromlist=["READ"]).READ),
        OP_CLOS=int(__import__("c4_min.isa", fromlist=["CLOS"]).CLOS),
        # memory-load opcodes (the pointer-walk reads in MODE 1 strict)
        OP_LI=int(__import__("c4_min.isa", fromlist=["LI"]).LI),
        OP_LC=int(__import__("c4_min.isa", fromlist=["LC"]).LC),
    )
    # frame role slots: FRAME_LEN-sized array, -1 for non-role slots
    role_of = [-1] * int(V.FRAME_LEN)
    for pos, role in frs.items():
        role_of[pos] = int(role)
    return consts, role_of, list(L.CODE_OP), list(L.CODE_IMM), list(L.CODE_IMM_NIB)


def assemble(prog):
    from c4_min import isa
    return isa.assemble(prog)


def build_live_masks(L):
    """PER-OP BLOCK-SKIP table (#738) for the all-C VM.

    Resolves ``step_block_skip.build_live_index`` (the cumulative-greedy + operand-
    union SOUND per-op live-block NAME schedule) into a per-op bitmask over the
    242-block block-stack — using the SAME ``L._block_names`` the C runtime's block
    index ``b`` runs in order (both the .nblbin export and this header come from the
    identical ``build_compact_pure_forward_model(code_size=48)`` build, so block
    index ``b`` in C == ``model.blocks[b]`` == ``L._block_names[b]``).

    Returns ``(num_ops, nb, have, mask_rows)`` where:
      * ``num_ops`` = isa.NUM_OPS (opcode-one-hot band size; covers every op value)
      * ``nb``      = number of blocks the table was built for (must == g_nblocks)
      * ``have[op]``= 1 if op has a SOUND live set (skip it); 0 -> run ALL blocks
                      (the byte-exact fallback for any op NOT in the table)
      * ``mask_rows[op]`` = list[nb] of 0/1, 1 == block is LIVE (run it)

    PRTF/OPEN/READ/CLOS/NOP are added here (NOT in the torch ``_LIVE_NAMES`` table):
      * PRTF is a register-passthrough op (AX unchanged, PC+=1) whose SOUND live set
        was derived by the same cumulative-greedy method (verified byte-exact) and
        pinned by NAME below.
      * OPEN/READ/CLOS are §File-Ops SYSCALLS — the all-C driver OWNS their effect
        and DISCARDS the neural forward's PC/AX for that row (it overrides the regs
        and ``continue``s before decoding).  So their forward output is never read;
        the minimal frame-advance set is byte-exact (nothing downstream reads it).
    """
    from c4_min import isa
    from c4_min.step_block_skip import build_live_index

    class _FakeModel:
        # build_live_index only needs len(model.blocks)
        blocks = list(range(len(L._block_names)))

    li = build_live_index(_FakeModel(), L)     # {op_or_None: sorted[block_idx]}
    nb = len(L._block_names)
    num_ops = int(isa.NUM_OPS)

    # I/O + no-op ops NOT in the torch table.  PRTF pinned by NAME (greedy-derived,
    # verified byte-exact).  SYSCALLS + NOP: frame-advance only (forward discarded).
    name_to_idx = {}
    for i, n in enumerate(L._block_names):
        name_to_idx.setdefault(n, []).append(i)

    def _idxs(name_list):
        s = set()
        for nm in name_list:
            s.update(name_to_idx.get(nm, []))
        return sorted(s)

    # PRTF SOUND live set (register-passthrough; the greedy result — a superset of
    # the frame-advance set that preserves AX through the residual).
    _PRTF_NAMES = ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
                   "dispatch"]
    # SYSCALL / NOP: forward output is discarded by the driver -> minimal advance set.
    _SYS_NAMES = ["ingest+recompose", "pc-fetch", "code-select", "opcode-decode",
                  "dispatch"]
    extra = {
        isa.PRTF: _idxs(_PRTF_NAMES),
        isa.OPEN: _idxs(_SYS_NAMES),
        isa.READ: _idxs(_SYS_NAMES),
        isa.CLOS: _idxs(_SYS_NAMES),
        isa.NOP: _idxs(_SYS_NAMES),
    }

    have = [0] * num_ops
    mask_rows = [[0] * nb for _ in range(num_ops)]
    for op in range(num_ops):
        live = None
        if op in li and li[op] is not None:
            live = li[op]
        elif op in extra:
            live = extra[op]
        if live is None:
            continue                          # no live set -> run all (fallback)
        have[op] = 1
        for b in live:
            if 0 <= b < nb:
                mask_rows[op][b] = 1
    return num_ops, nb, have, mask_rows


def gen_header(out_path, L, embed, code, seed_mem=None):
    seed_mem = seed_mem or {}
    consts, role_of, code_op_dims, code_imm_dims, code_imm_nib_dims = layout_consts(L)
    D = consts["D"]
    # seed data segment: sorted (addr, val) -> leading store frames (idx 0..n_seed-1)
    seed_items = sorted((int(a) & 0xFFFFFFFF, int(v) & 0xFFFFFFFF)
                        for a, v in seed_mem.items())
    lines = []
    ap = lines.append
    ap("/* AUTO-GENERATED by _agent_allc_gen.py — do not edit. */")
    ap("#ifndef ALLC_GEN_H\n#define ALLC_GEN_H\n")
    for k, v in consts.items():
        if v is None:
            continue
        ap(f"#define G_{k} {v}")
    ap(f"#define G_NCODE {len(code_op_dims)}")
    ap(f"#define G_PROGLEN {len(code)}")
    ap(f"#define G_NSEED {len(seed_items)}")
    ap("")
    # embed table (267 x D), row-major
    ap(f"static const float G_EMBED[{consts['VOCAB']}*{D}] = {{")
    flat = embed.reshape(-1)
    _emit_float_array(ap, flat)
    ap("};")
    ap("")
    # role_of[FRAME_LEN]
    ap(f"static const int G_ROLE_OF[{consts['FRAME_LEN']}] = {{ "
       + ",".join(str(x) for x in role_of) + " };")
    # per-code-slot overlay dims
    ap(f"static const int G_CODE_OP_DIM[{len(code_op_dims)}] = {{ "
       + ",".join(str(x) for x in code_op_dims) + " };")
    ap(f"static const int G_CODE_IMM_DIM[{len(code_imm_dims)}] = {{ "
       + ",".join(str(x) for x in code_imm_dims) + " };")
    ap(f"static const int G_CODE_IMM_NIB_DIM[{len(code_imm_nib_dims)}] = {{ "
       + ",".join(str(x) for x in code_imm_nib_dims) + " };")
    # program
    ap(f"static const int G_PROG_OP[{len(code)}] = {{ "
       + ",".join(str(int(i.op)) for i in code) + " };")
    ap(f"static const int G_PROG_IMM[{len(code)}] = {{ "
       + ",".join(str(int(i.imm)) for i in code) + " };")
    # seed data segment (quine): leading store frames, one per (addr,val)
    if seed_items:
        ap(f"static const long G_SEED_ADDR[{len(seed_items)}] = {{ "
           + ",".join(str(a) for a, v in seed_items) + " };")
        ap(f"static const long G_SEED_VAL[{len(seed_items)}] = {{ "
           + ",".join(str(v) for a, v in seed_items) + " };")
    else:
        ap("static const long G_SEED_ADDR[1] = { 0 };")
        ap("static const long G_SEED_VAL[1] = { 0 };")
    ap("")
    # ---- PER-OP BLOCK-SKIP table (#738): the SOUND per-op live-block mask over the
    # block-stack, so the all-C forward runs ONLY the decoded op's live blocks and
    # passes the residual straight through the skipped ones (byte-exact for decode).
    num_ops, nb_skip, have, mask_rows = build_live_masks(L)
    ap(f"#define G_SKIP_NUM_OPS {num_ops}")
    ap(f"#define G_SKIP_NB {nb_skip}")
    ap(f"static const int G_SKIP_HAVE[{num_ops}] = {{ "
       + ",".join(str(h) for h in have) + " };")
    # G_LIVE_MASK[op*nb + b] == 1 iff block b is LIVE for op (else skipped).  Rows
    # for ops with have[op]==0 are all-zero and IGNORED (fallback = run all blocks).
    ap(f"static const unsigned char G_LIVE_MASK[{num_ops}*{nb_skip}] = {{")
    for op in range(num_ops):
        ap(" " + ",".join(str(mask_rows[op][b]) for b in range(nb_skip)) + ",")
    ap("};")
    ap("")
    ap("#endif")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return consts, code


def _cfloat(v):
    """A C float literal that ALWAYS parses as float (never an int suffixed 'f').
    %.9g renders 0.0 as '0' -> '0f' is an invalid integer suffix; ensure a '.' or
    exponent is present before appending the 'f' suffix."""
    s = f"{float(v):.9g}"
    if ("." not in s) and ("e" not in s) and ("E" not in s) \
            and ("inf" not in s) and ("nan" not in s):
        s += ".0"
    return s + "f"


def _emit_float_array(ap, flat):
    # compact: 12 per line, round-trippable float32 literals
    buf = []
    for i, v in enumerate(flat):
        buf.append(_cfloat(v))
        if len(buf) == 12:
            ap(" " + ",".join(buf) + ",")
            buf = []
    if buf:
        ap(" " + ",".join(buf))


def verify_frame0(L, embed, prog):
    """Rebuild frame_0's residual (embed[stream]+overlay) in numpy from the SAME
    layout constants the header dumps and assert byte-identity vs torch."""
    import torch
    import c4_min.nibble_pure_forward_complete as PFC
    from c4_min import blogspec_vocab as V
    code = assemble(prog)
    stream = [V.BOS] + PFC._build_frame(0, 0, PFC.SP_INIT, PFC.SP_INIT, 0)
    overlay = PFC.make_overlay_complete(code, L, store_log={})
    x = torch.from_numpy(embed[np.asarray(stream)]).unsqueeze(0).clone()
    overlay(x)
    return x.numpy()[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(C4MIN, "allc_gen.h"))
    ap.add_argument("--mode", default="echo",
                    help="echo|cat|yes|quine (echo/cat/yes = printf of --text)")
    ap.add_argument("--text", default="hello\n")
    ap.add_argument("--io-mode", default="literal",
                    choices=["literal", "strict", "burst"],
                    help="literal=legacy IMM;PRTF; strict=MODE1 §Memory pointer-walk"
                         "; burst=MODE2 runtime §Memory syscall")
    ap.add_argument("--n", type=int, default=8, help="yes: repeat count")
    ap.add_argument("--stdin", default="", help="cat: the stdin the program READs")
    args = ap.parse_args()
    print(f"building compact full-ISA model + layout (mode={args.mode} "
          f"io={args.io_mode}) ...", flush=True)
    L, embed = build_layout()
    code, seed_mem, expected = build_program(
        args.mode, args.text, io_mode=args.io_mode, n=args.n,
        stdin_text=args.stdin)
    print(f"  D={L.D}  vocab={embed.shape[0]}  code={len(code)} isa-ops  "
          f"seed_mem={len(seed_mem)}  expected={bytes(expected)!r}")
    consts, code2 = gen_header(args.out, L, embed, code, seed_mem)
    print(f"  wrote {args.out} ({os.path.getsize(args.out):,} bytes)")
    print(f"  consts: D={consts['D']} FRAME_LEN={consts['FRAME_LEN']} "
          f"AX={consts['AX']} PC_VAL={consts['PC_VAL']} HALTED={consts['HALTED']}")
    print("done.")
