"""Tests for the native fused WAD lump-NAME compare op NAMEEQ (``doom_nameeq``).

Covers: the gate (default OFF, golden-neutral), byte-exactness of the native
reference vs the ``__name_eq`` C semantics (record uppercased, want as-is,
NUL-terminated 8-byte compare), the fused megablock block schedule, the
intrinsic recognition byte-identity (JSR __name_eq -> NAMEEQ + NOP, length
preserved), and — when the id_port tree is present — byte-exactness vs the REAL
compiled ``__name_eq`` as it runs on ``c4vm32.py`` (the 32-bit substrate the
transformer's Doom image runs).
"""
import os

import pytest

from c4_min import doom_nameeq as NE
from c4_min import isa


# --------------------------------------------------------------------------- #
# gate                                                                        #
# --------------------------------------------------------------------------- #
def test_gate_default_off(monkeypatch):
    monkeypatch.delenv("C4_DOOM_NAMEEQ", raising=False)
    assert not NE.nameeq_enabled()


def test_gate_on(monkeypatch):
    monkeypatch.setenv("C4_DOOM_NAMEEQ", "1")
    assert NE.nameeq_enabled()


def test_register_opcode_idempotent():
    NE.register_opcode()
    NE.register_opcode()
    assert isa.BY_NAME["NAMEEQ"] == NE.NAMEEQ
    # the opcode sits PAST the neural one-hot band -> registering it is byte-neutral
    assert NE.NAMEEQ >= isa.NUM_OPS
    # does not collide with the other Doom native ops (FIXEDMUL/DIV 42/43, BLIT 44,
    # MEMCPY 45)
    assert NE.NAMEEQ == 46


# --------------------------------------------------------------------------- #
# reference: byte-exact vs the __name_eq C semantics                          #
# --------------------------------------------------------------------------- #
def test_toupper_matches_c():
    # c4_toupper: 'a'..'z' -> upper, everything else unchanged (incl. boundaries)
    assert NE.c4_toupper(ord("a")) == ord("A")
    assert NE.c4_toupper(ord("z")) == ord("Z")
    assert NE.c4_toupper(ord("A")) == ord("A")       # already upper
    assert NE.c4_toupper(96) == 96                    # '`' just below 'a'
    assert NE.c4_toupper(123) == 123                  # '{' just above 'z'
    assert NE.c4_toupper(ord("5")) == ord("5")        # digit unchanged
    assert NE.c4_toupper(0) == 0                       # NUL unchanged


def test_equal_names():
    def pad8(b):
        return b.ljust(8, b"\x00")
    assert NE.name_eq_bytes(pad8(b"MAP01"), pad8(b"MAP01")) == 1
    assert NE.name_eq_bytes(pad8(b"PLAYPAL"), pad8(b"PLAYPAL")) == 1
    assert NE.name_eq_bytes(b"ABCDEFGH", b"ABCDEFGH") == 1     # full 8, no NUL


def test_case_insensitive_record_side_only():
    # __name_eq uppercases the RECORD side; the want side is taken as-is.
    def pad8(b):
        return b.ljust(8, b"\x00")
    assert NE.name_eq_bytes(pad8(b"map01"), pad8(b"MAP01")) == 1   # lower record OK
    assert NE.name_eq_bytes(pad8(b"MaP01"), pad8(b"MAP01")) == 1   # mixed record OK
    # but a lower-case WANT byte does NOT match an upper record byte (want not
    # uppercased by __name_eq) — this is the asymmetry of the C rule.
    assert NE.name_eq_bytes(pad8(b"MAP01"), pad8(b"map01")) == 0


def test_differ_at_each_position():
    base = b"TEXTURE1"
    for pos in range(8):
        bad = bytearray(base)
        bad[pos] = ord("X") if bad[pos] != ord("X") else ord("Y")
        assert NE.name_eq_bytes(bytes(bad), base) == 0, pos


def test_pad_and_nul_edges():
    def pad8(b):
        return b.ljust(8, b"\x00")
    assert NE.name_eq_bytes(pad8(b"E1M1"), pad8(b"E1M1")) == 1
    assert NE.name_eq_bytes(pad8(b"E1M1"), pad8(b"E1M11")) == 0     # short vs longer
    assert NE.name_eq_bytes(pad8(b""), pad8(b"")) == 1              # empty == empty
    assert NE.name_eq_bytes(pad8(b""), pad8(b"A")) == 0            # empty vs non-empty
    assert NE.name_eq_bytes(b"ABCDEFGH", b"ABCDEFGX") == 0          # differ last byte


def test_vm_memory_reference_matches_pure():
    """The VM-memory reference (:func:`name_eq`) agrees with the pure-bytes
    reference on the full battery."""
    REC, WANT = 0x400000, 0x410000
    for rec_bytes, want_bytes in NE.battery_cases():
        mem = bytearray(WANT + 64)
        mem[REC:REC + len(rec_bytes)] = rec_bytes
        mem[WANT:WANT + len(want_bytes)] = want_bytes
        assert NE.name_eq(mem, REC, WANT) == NE.name_eq_bytes(rec_bytes, want_bytes)


def test_verify_byte_exact_battery():
    r = NE.verify_byte_exact()
    assert r["ref_fail"] == 0
    assert r["mem_fail"] == 0
    assert r["n"] >= 100          # a real battery, not a stub


# --------------------------------------------------------------------------- #
# fused megablock schedule                                                     #
# --------------------------------------------------------------------------- #
def test_megablock_block_schedule():
    mb = NE.nameeq_megablock()
    assert mb.name == "NAMEEQ"
    assert mb.n_blocks == 6                     # one decoded VM step
    for b in ("alu-expand", "name-gather", "name-upper", "name-cmp",
              "name-reduce", "ax-mux"):
        assert b in mb.blocks
    # reuses the shared expand + ax-mux blocks (like BLIT / FIXEDMUL)
    assert mb.blocks[0] == "alu-expand" and mb.blocks[-1] == "ax-mux"


# --------------------------------------------------------------------------- #
# intrinsic recognition byte-identity                                          #
# --------------------------------------------------------------------------- #
def test_intrinsic_substitution_rewrites_jsr_tuple():
    NAME_PC = 548188
    code = [(isa.PSH, 0), (isa.PSH, 0), (isa.JSR, NAME_PC), (isa.ADJ, 16),
            (isa.LEA, 0)]
    out, n = NE.substitute_intrinsics(code, NE.IntrinsicMap.for_doom(NAME_PC))
    assert n == 1
    assert out[2] == (NE.NAMEEQ, 0)             # JSR -> native op
    assert out[3] == (isa.NOP, 0)               # ADJ -> NOP
    assert out[4] == (isa.LEA, 0)               # everything else unchanged
    assert len(out) == len(code)                # length preserved (targets stable)


def test_intrinsic_substitution_rewrites_jsr_instr():
    NAME_PC = 548188
    code = [isa.Instr(isa.PSH, 0), isa.Instr(isa.JSR, NAME_PC),
            isa.Instr(isa.ADJ, 16)]
    out, n = NE.substitute_intrinsics(code, NE.IntrinsicMap.for_doom(NAME_PC))
    assert n == 1
    assert out[1].op == NE.NAMEEQ and out[1].imm == 0
    assert out[2].op == isa.NOP
    assert len(out) == len(code)


def test_intrinsic_only_touches_target_jsr():
    NAME_PC = 548188
    OTHER_PC = 999
    code = [(isa.JSR, OTHER_PC), (isa.ADJ, 16), (isa.JSR, NAME_PC), (isa.ADJ, 16)]
    out, n = NE.substitute_intrinsics(code, NE.IntrinsicMap.for_doom(NAME_PC))
    assert n == 1
    assert out[0] == (isa.JSR, OTHER_PC)        # unrelated JSR untouched
    assert out[1] == (isa.ADJ, 16)              # its ADJ untouched
    assert out[2] == (NE.NAMEEQ, 0)


# --------------------------------------------------------------------------- #
# step cost model                                                             #
# --------------------------------------------------------------------------- #
def test_step_cost_short_circuits():
    # a name that differs at byte 0 examines 1 iteration; equal 8-char examines 8.
    def pad8(b):
        return b.ljust(8, b"\x00")
    c_diff0 = NE.nameeq_step_cost_c(pad8(b"XMAP01"), pad8(b"YMAP01"))
    c_eq8 = NE.nameeq_step_cost_c(b"ABCDEFGH", b"ABCDEFGH")
    # both are MANY VM steps for the function path; the native op is ONE.
    assert c_diff0 < c_eq8                       # short-circuit is cheaper
    assert c_eq8 > 50                            # 8 iters incl. nested toupper


# --------------------------------------------------------------------------- #
# on-VM byte-exactness vs the REAL compiled __name_eq (c4vm32)                  #
# --------------------------------------------------------------------------- #
_ID_PORT = "/home/alexlitz/Documents/misc/c4_doom/id_port"


@pytest.mark.slow
def test_byte_exact_vs_real_compiled_name_eq_on_c4vm32():
    """The native NAMEEQ reference is byte-exact vs the ACTUAL compiled
    ``__name_eq`` executed on ``c4vm32`` (the substrate the transformer's Doom
    image runs).  Skipped if the id_port tree / measurement harness is absent."""
    import importlib.util
    import sys
    from pathlib import Path

    work = Path(__file__).resolve().parent / "doom_nameeq_work"
    if not (Path(_ID_PORT) / "c4vm32.py").exists() or not work.exists():
        pytest.skip("id_port/c4vm32.py or doom_nameeq_work harness not present")
    sys.path.insert(0, _ID_PORT)
    sys.path.insert(0, str(work))
    spec = importlib.util.spec_from_file_location("_measure_nameeq",
                                                  work / "measure_nameeq.py")
    M = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(M)
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"measure_nameeq harness not importable: {e}")
    code, data, _argc = M.P.compile_doom()
    name_pc = M.entry_pcs()["__name_eq"]
    res = M.verify_byte_exact_onvm(code, data, name_pc, NE.battery_cases())
    assert res["fails"] == 0, res["fail_detail"][:10]
    assert res["checked"] >= 100
