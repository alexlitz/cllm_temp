"""
Dynamic mirror of :func:`neural_vm.dim_registry.build_default_registry`.

This module re-encodes every static ``reg.alloc(NAME, START, SIZE, ...)``
call from the production registry using
:class:`neural_vm.dim_allocator.Allocator` with ``pin=START`` so the
resulting :class:`DimRegistry` is **byte-identical** to the static one:
every slot has the same ``(start, size, semantics)`` tuple.

The only purpose of building it this way today is to prove the allocator
can reconstruct the existing layout. Once the static registry stabilises
(after the registry-completeness pass lands), a follow-up commit can
drop ``pin=`` from individual dims (and entire op families) so the
allocator picks the lowest free gap automatically — that's where the
Phase B plan's slot-conflict question (Q2: ``OUTPUT_HI_PREV_STEP``
vs B8-B address-byte hi nibbles) gets answered.

This file is intentionally a near-mechanical translation of
``build_default_registry``. The aliasing structure (e.g. ``AX_FULL_LO``
overlaying ``FORMAT_PTR_LO`` at 471) is preserved via
``allow_overlap=True`` on the alias call, NOT by reshuffling slots.
"""

from __future__ import annotations

from neural_vm.dim_allocator import Allocator
from neural_vm.dim_registry import DimRegistry


def build_default_registry_dynamic() -> DimRegistry:
    """Mirror :func:`build_default_registry` via the dynamic allocator.

    The returned :class:`DimRegistry` should compare equal — slot-by-slot
    on ``(name, start, size, semantics)`` — to the registry returned by
    :func:`neural_vm.dim_registry.build_default_registry`. This is
    enforced by ``test_allocator_byte_identical_to_static`` in
    ``tests/test_dim_allocator.py``.

    Every existing dim is allocated with ``pin=<its_existing_start>``.
    Intentional aliases (slots that share bytes with a parent) pass
    ``allow_overlap=True`` so the allocator's collision check doesn't
    fire on them.
    """
    # d_model expanded from 512 -> 736 to mirror the static registry's
    # compact pin_io_only layout block at positions 510..732. See the
    # ``# --- Compact pin_io_only layout mirrors ---`` block at the end
    # of this function for the ``_PIN``-suffix family.
    a = Allocator(d_model=736)

    def pin(name, start, size, desc, semantics=None, alias=False):
        # Tiny wrapper that mirrors the (name, start, size, desc, semantics)
        # signature of ``DimRegistry.alloc`` so the body below stays
        # visually identical to ``build_default_registry``. ``alias``
        # toggles ``allow_overlap`` for intentional alias allocations.
        a.alloc(
            name, size,
            pin=start,
            description=desc,
            semantics=semantics,
            allow_overlap=alias,
        )

    # ------------------------------------------------------------------
    # Marker identity flags (set by embedding)
    # ------------------------------------------------------------------
    pin("MARK_PC",      0, 1, "PC register marker flag", "mark == PC")
    pin("MARK_AX",      1, 1, "AX register marker flag", "mark == AX")
    pin("MARK_SP",      2, 1, "SP register marker flag", "mark == SP")
    pin("MARK_BP",      3, 1, "BP register marker flag", "mark == BP")
    pin("MARK_MEM",     4, 1, "MEM marker flag", "mark == MEM")
    pin("MARK_SE",      5, 1, "STEP_END/DATA_END marker flag", "mark == SE")
    pin("IS_BYTE",      6, 1, "Token is a byte value (0-255)", "is_byte")
    pin("IS_MARK",      7, 1, "Token is a marker", "NOT is_byte")
    pin("CONST",        8, 1, "Constant 1.0 on all tokens", "is_byte OR NOT is_byte")
    pin("MARK_CS",      9, 1, "CODE_START only marker", "NOT is_byte")
    pin("MARK_SE_ONLY", 10, 1, "STEP_END only (not DATA_END)", "mark == SE")
    pin("MARK_STACK0",  11, 1, "STACK0 marker flag", "mark == STACK0")

    # ------------------------------------------------------------------
    # Address byte nibbles
    # ------------------------------------------------------------------
    pin("ADDR_B0_LO",  12, 16, "One-hot addr byte 0 low nibble", "mark == MEM")
    pin("ADDR_B1_LO",  28, 16, "One-hot addr byte 1 low nibble", "mark == MEM")
    pin("ADDR_B2_LO",  44, 16, "One-hot addr byte 2 low nibble", "mark == MEM")

    # ------------------------------------------------------------------
    # Layer 0 attention output (H0..H7)
    # ------------------------------------------------------------------
    pin("H0",  60, 7, "L0 head 0: marker within dist 3.5",  "is_byte OR NOT is_byte")
    pin("H1",  67, 7, "L0 head 1: marker within dist 4.5",  "is_byte OR NOT is_byte")
    pin("H2",  74, 7, "L0 head 2: marker within dist 5.5",  "is_byte OR NOT is_byte")
    pin("H3",  81, 7, "L0 head 3: marker within dist 9.5",  "is_byte OR NOT is_byte")
    pin("H4",  88, 7, "L0 head 4: marker within dist 10.5", "is_byte OR NOT is_byte")
    pin("H5",  95, 7, "L0 head 5: marker within dist 14.5", "is_byte OR NOT is_byte")
    pin("H6", 102, 7, "L0 head 6: marker within dist 15.5", "is_byte OR NOT is_byte")
    pin("H7", 109, 7, "L0 head 7: marker within dist 19.5", "is_byte OR NOT is_byte")

    # ------------------------------------------------------------------
    # Layer 1 attention output (L1H0..L1H2) + HAS_SE
    # ------------------------------------------------------------------
    pin("L1H0", 116, 7, "L1 head 0: marker within dist 0.5", "is_byte OR NOT is_byte")
    pin("L1H1", 123, 7, "L1 head 1: marker within dist 1.5", "is_byte OR NOT is_byte")
    pin("L1H2", 130, 7, "L1 head 2: marker within dist 2.5", "is_byte OR NOT is_byte")
    pin("HAS_SE", 137, 1, "STEP_END existence flag", "has_se")

    # ------------------------------------------------------------------
    # Byte index within register
    # ------------------------------------------------------------------
    pin("BYTE_INDEX_0", 138, 1, "Byte index 0 flag", "is_byte AND byte_index == 0")
    pin("BYTE_INDEX_1", 139, 1, "Byte index 1 flag", "is_byte AND byte_index == 1")
    pin("BYTE_INDEX_2", 140, 1, "Byte index 2 flag", "is_byte AND byte_index == 2")
    pin("BYTE_INDEX_3", 141, 1, "Byte index 3 flag", "is_byte AND byte_index == 3")

    # ------------------------------------------------------------------
    # Nibble encoding: EMBED / OUTPUT
    # ------------------------------------------------------------------
    pin("EMBED_LO",  142, 16, "Embedding input low nibble (one-hot)",  "is_byte")
    pin("EMBED_HI",  158, 16, "Embedding input high nibble (one-hot)", "is_byte")
    pin("OUTPUT_LO", 174, 16, "Output decoding low nibble (one-hot)",
        "is_byte OR NOT is_byte")
    pin("OUTPUT_HI", 190, 16, "Output decoding high nibble (one-hot)",
        "is_byte OR NOT is_byte")

    # ------------------------------------------------------------------
    # Memory address key
    # ------------------------------------------------------------------
    pin("ADDR_KEY", 206, 48,
        "One-hot address key for memory matching (3 nibbles x 16)",
        "mark == MEM")

    # ------------------------------------------------------------------
    # NEXT_* transition flags
    # ------------------------------------------------------------------
    pin("NEXT_PC",     254, 1, "Next token is PC register",     "NOT is_byte")
    pin("NEXT_AX",     255, 1, "Next token is AX register",     "NOT is_byte")
    pin("NEXT_SP",     256, 1, "Next token is SP register",     "NOT is_byte")
    pin("NEXT_BP",     257, 1, "Next token is BP register",     "NOT is_byte")
    pin("NEXT_STACK0", 258, 1, "Next token is STACK0 marker",   "NOT is_byte")
    pin("NEXT_MEM",    259, 1, "Next token is MEM marker",      "NOT is_byte")
    pin("NEXT_SE",     260, 1, "Next token is STEP_END",        "NOT is_byte")
    pin("NEXT_HALT",   261, 1, "Emit HALT instead of STEP_END", "NOT is_byte")

    # ------------------------------------------------------------------
    # OPCODE_FLAGS (262..295) parent slot
    # ------------------------------------------------------------------
    pin("OPCODE_FLAGS", 262, 34, "One-hot opcode flags (LEA..GETCHAR)",
        "mark == AX OR (is_byte AND byte_index == 0)")

    # ------------------------------------------------------------------
    # IO PUTCHAR flag
    # ------------------------------------------------------------------
    pin("IO_IS_PUTCHAR", 296, 1, "OP_PUTCHAR detected this step (L6 FFN)",
        "mark == AX AND opcode_at_AX == PUTCHAR")

    # ------------------------------------------------------------------
    # ADJ implementation dims
    # ------------------------------------------------------------------
    pin("SP_OLD_LO", 297, 8, "ADJ: old SP value low nibbles (4 bytes)",
        "(mark == SP OR mark == AX) AND opcode_in_step in {ADJ}")
    pin("SP_OLD_HI", 305, 8, "ADJ: old SP value high nibbles (4 bytes)",
        "(mark == SP OR mark == AX) AND opcode_in_step in {ADJ}")
    pin("ADJ_CARRY", 313, 2, "ADJ: multi-byte carry propagation",
        "(mark == SP OR mark == AX) AND opcode_in_step in {ADJ}")

    # ------------------------------------------------------------------
    # Reserved
    # ------------------------------------------------------------------
    pin("RESERVED_315_327", 315, 13, "Reserved (ENT/LEV staging)",
        "mark == PC AND mark == AX")

    # ------------------------------------------------------------------
    # AX carry-forward staging
    # ------------------------------------------------------------------
    pin("AX_CARRY_LO", 328, 16, "Carried-forward AX lo nibble",
        "mark == AX OR (is_byte AND byte_index == 0)")
    pin("AX_CARRY_HI", 344, 16, "Carried-forward AX hi nibble",
        "mark == AX OR (is_byte AND byte_index == 0)")

    # ------------------------------------------------------------------
    # ALU result staging
    # ------------------------------------------------------------------
    pin("ALU_LO", 360, 16, "ALU result lo nibble",
        "mark == AX OR (is_byte AND byte_index == 0)")
    pin("ALU_HI", 376, 16, "ALU result hi nibble",
        "mark == AX OR (is_byte AND byte_index == 0)")

    # ------------------------------------------------------------------
    # Carry / comparison cascade
    # ------------------------------------------------------------------
    pin("CARRY", 392, 4, "Inter-byte carry for ADD/SUB/MUL",
        "is_byte AND byte_index in {0, 1, 2, 3}")
    pin("CMP",   396, 4, "Comparison cascade: LT, EQ, GT, ZERO",
        "mark == AX OR (is_byte AND byte_index == 0)")

    # ------------------------------------------------------------------
    # Reserved 400..415
    # ------------------------------------------------------------------
    pin("RESERVED_400_415", 400, 16, "Reserved (future PC binary encoding/IO)",
        "mark == PC AND mark == AX")

    # ------------------------------------------------------------------
    # MUL/DIV staging
    # ------------------------------------------------------------------
    pin("MUL_ACCUM",   416, 16, "Multiplication accumulator",
        "mark == AX OR (is_byte AND byte_index == 0)")
    pin("DIV_STAGING", 432, 16, "Division quotient/remainder",
        "mark == AX OR (is_byte AND byte_index == 0)")

    # ------------------------------------------------------------------
    # Immediate staging
    # ------------------------------------------------------------------
    pin("IMM_STAGING", 448, 16, "Fetched immediate bytes",
        "mark == PC OR (is_byte AND byte_index in {0, 1, 2, 3})")

    # ------------------------------------------------------------------
    # CS distance thermometer
    # ------------------------------------------------------------------
    pin("CS_DIST_THERMO", 464, 16, "Thermometer-coded distance from CODE_START",
        "is_byte OR NOT is_byte")

    # ------------------------------------------------------------------
    # General temporaries / reserved scratch (parent of OUTPUT_BYTE_*,
    # STACK0_BYTE1..3, LAST_WAS_*, ACTIVE_OPCODE_*, MARK_THINKING_*)
    # ------------------------------------------------------------------
    pin("TEMP", 480, 32, "General temporaries / reserved", "is_byte OR NOT is_byte")

    # ==================================================================
    # F-4-extension: aliases that OVERLAP parent slots (OPCODE_FLAGS,
    # ADDR_KEY, MUL_ACCUM, DIV_STAGING, MEM_VAL/RELAY, TEMP, ...). The
    # parent above already claimed the bytes; these all need
    # alias=True so the allocator's collision check accepts them.
    # ==================================================================

    # --- Per-opcode flags (sub-offsets within OPCODE_FLAGS 262..295) ---
    _OPCODES = [
        ("OP_LEA", 262), ("OP_IMM", 263), ("OP_JMP", 264), ("OP_JSR", 265),
        ("OP_BZ",  266), ("OP_BNZ", 267), ("OP_ENT", 268), ("OP_ADJ", 269),
        ("OP_LEV", 270), ("OP_LI",  271), ("OP_LC",  272), ("OP_SI",  273),
        ("OP_SC",  274), ("OP_PSH", 275), ("OP_OR",  276), ("OP_XOR", 277),
        ("OP_AND", 278), ("OP_EQ",  279), ("OP_NE",  280), ("OP_LT",  281),
        ("OP_GT",  282), ("OP_LE",  283), ("OP_GE",  284), ("OP_SHL", 285),
        ("OP_SHR", 286), ("OP_ADD", 287), ("OP_SUB", 288), ("OP_MUL", 289),
        ("OP_DIV", 290), ("OP_MOD", 291), ("OP_EXIT", 292), ("OP_NOP", 293),
        ("OP_PUTCHAR", 294), ("OP_GETCHAR", 295),
    ]
    _DSL_KEYWORDS = {"OR", "AND", "NOT"}
    for _name, _pos in _OPCODES:
        _opname = _name[3:]
        if _opname in _DSL_KEYWORDS:
            _sem = "mark == AX OR (is_byte AND byte_index == 0)"
        else:
            _sem = f"mark == AX AND opcode_at_AX == {_opname}"
        pin(_name, _pos, 1,
            f"OPCODE_FLAGS[{_pos - 262}] = {_name} active flag",
            _sem, alias=True)

    # --- STACK0 byte position flags ---
    # STACK0_BYTE0 lives at 304, between SP_OLD_HI (305..312) and the
    # IO_IS_PUTCHAR slot at 296 — actually IO_IS_PUTCHAR is 296 (1 wide)
    # and SP_OLD_LO/HI cover 297..312. So 304 falls inside SP_OLD_HI
    # (305..312)? No: 304 is exactly between IO_IS_PUTCHAR (296) and
    # SP_OLD_LO (297..304). 297..304 = SP_OLD_LO (8 wide). So 304 IS
    # the last byte of SP_OLD_LO — alias.
    pin("STACK0_BYTE0", 304, 1, "STACK0 byte 0 position flag",
        "mark == STACK0 OR (is_byte AND byte_index == 0)", alias=True)
    pin("STACK0_BYTE1", 508, 1, "STACK0 byte 1 position flag",
        "mark == STACK0 OR (is_byte AND byte_index == 1)", alias=True)
    pin("STACK0_BYTE2", 509, 1, "STACK0 byte 2 position flag",
        "mark == STACK0 OR (is_byte AND byte_index == 2)", alias=True)
    pin("STACK0_BYTE3", 510, 1, "STACK0 byte 3 position flag",
        "mark == STACK0 OR (is_byte AND byte_index == 3)", alias=True)

    # --- L1H4 / L2H0 fine threshold heads ---
    # L1H4 at 297..303 overlaps SP_OLD_LO (297..304). L2H0 at 452..458
    # overlaps DIV_STAGING (432..447)? Wait: DIV_STAGING is 432..447, so
    # 452..458 is between DIV_STAGING and IMM_STAGING (448..463). Hmm
    # 448..463 = IMM_STAGING (16 wide). So 452..458 IS inside IMM_STAGING.
    pin("L1H4", 297, 7, "L1 head 4: threshold 6.5 from nearest IS_MARK",
        "is_byte OR NOT is_byte", alias=True)
    pin("L2H0", 452, 7, "L2 head 0: threshold 5.5 from nearest IS_MARK",
        "is_byte OR NOT is_byte", alias=True)

    # --- L0 head 5 aliases (95..100 share with H5 95..101) ---
    pin("SP_BYTE0_IS_F8", 95, 1, "SP byte 0 equals 0xF8 (aliases H5+0)",
        "(mark == SP OR mark == AX) AND sp_byte0 == 0xF8", alias=True)
    pin("IN_STEP_FRESH", 96, 1, "In-step freshness flag (aliases H5+1)",
        "in_step_fresh", alias=True)
    pin("ADDR_B0_VALID", 97, 1, "Address byte 0 gathered (aliases H5+2)",
        "addr_b0_valid", alias=True)
    pin("SP_GATHERED_THIS_STEP", 98, 1,
        "SP gather fired this step (aliases H5+3)",
        "sp_gathered_this_step", alias=True)
    pin("ADDR_B1_VALID", 99, 1, "Address byte 1 gathered (aliases H5+4)",
        "addr_b1_valid", alias=True)
    pin("ADDR_B2_VALID", 100, 1, "Address byte 2 gathered (aliases H5+5)",
        "addr_b2_valid", alias=True)

    # --- CMP_GROUP (305) inside SP_OLD_HI ---
    pin("CMP_GROUP", 305, 1, "Any EQ/NE/LT/GT/LE/GE active at AX",
        "mark == AX AND opcode_at_AX in {EQ, NE, LT, GT, LE, GE}", alias=True)

    # --- CLEAN_EMBED nibbles ---
    # 306..321 overlaps SP_OLD_HI (305..312) and RESERVED_315_327 (315..327).
    # 404..419 overlaps CMP (396..399? no, CMP is 4 wide → 396..399) +
    # RESERVED_400_415 (400..415) + MUL_ACCUM (416..431).
    pin("CLEAN_EMBED_LO", 306, 16, "Clean embedding lo nibble (one-hot)",
        "is_byte", alias=True)
    pin("CLEAN_EMBED_HI", 404, 16, "Clean embedding hi nibble (one-hot)",
        "is_byte", alias=True)

    # --- IO/conversational tool-call transition + state flags
    # 322..327 sits inside RESERVED_315_327 (315..327) ---
    pin("IO_IS_TOOL_CALL", 322, 1, "Any of OPEN/READ/CLOS/PRTF active",
        "mark == AX", alias=True)
    pin("NEXT_TOOL_CALL",        323, 1, "Next token is TOOL_CALL",
        "NOT is_byte", alias=True)
    pin("NEXT_THINKING_START",   324, 1, "Next token is THINKING_START",
        "NOT is_byte", alias=True)
    pin("NEXT_THINKING_END",     325, 1, "Next token is THINKING_END",
        "NOT is_byte", alias=True)
    pin("NEXT_IO_STATE_EMIT_BYTE",      326, 1,
        "Next token is IO_STATE_EMIT_BYTE", "NOT is_byte", alias=True)
    pin("NEXT_IO_STATE_EMIT_THINKING",  327, 1,
        "Next token is IO_STATE_EMIT_THINKING", "NOT is_byte", alias=True)

    # --- FETCH aliases — alias MUL_ACCUM (416..431) and DIV_STAGING (432..447) ---
    pin("FETCH_LO", 420, 16, "Fetched immediate lo nibble (aliases MUL_ACCUM)",
        "mark == PC OR (is_byte AND byte_index in {0, 1, 2, 3})",
        alias=True)
    pin("FETCH_HI", 436, 16, "Fetched immediate hi nibble (aliases DIV_STAGING)",
        "mark == PC OR (is_byte AND byte_index in {0, 1, 2, 3})",
        alias=True)

    # --- ADDR_B0/1/2_HI alias ADDR_KEY (206..253) ---
    pin("ADDR_B0_HI", 206, 16,
        "Gathered addr byte 0 hi nibble (aliases ADDR_KEY[0:16])",
        "mark == MEM", alias=True)
    pin("ADDR_B1_HI", 222, 16,
        "Gathered addr byte 1 hi nibble (aliases ADDR_KEY[16:32])",
        "mark == MEM", alias=True)
    pin("ADDR_B2_HI", 238, 16,
        "Gathered addr byte 2 hi nibble (aliases ADDR_KEY[32:48])",
        "mark == MEM", alias=True)

    # --- MEM control + value bus (459..464 inside IMM_STAGING 448..463;
    # 464 is start of CS_DIST_THERMO 464..479) ---
    pin("MEM_STORE", 459, 1,
        "Store op (SI/SC/PSH) active, relayed to MEM positions",
        "mark == MEM AND opcode_in_step in {SI, SC, PSH}", alias=True)
    pin("MEM_ADDR_SRC", 460, 1,
        "Address source: 1=STACK0 (SI/SC), 0=SP (PSH)",
        "mark == MEM AND opcode_in_step in {SI, SC, PSH}", alias=True)
    pin("MEM_VAL_B0", 461, 1, "Predicts MEM val byte 0 (d=4 from MEM)",
        "mark == MEM OR (is_byte AND byte_index == 0)", alias=True)
    pin("MEM_VAL_B1", 462, 1, "Predicts MEM val byte 1 (d=5 from MEM)",
        "mark == MEM OR (is_byte AND byte_index == 1)", alias=True)
    pin("MEM_VAL_B2", 463, 1, "Predicts MEM val byte 2 (d=6 from MEM)",
        "mark == MEM OR (is_byte AND byte_index == 2)", alias=True)
    pin("MEM_VAL_B3", 464, 1, "Predicts MEM val byte 3 (d=7 from MEM)",
        "mark == MEM OR (is_byte AND byte_index == 3)", alias=True)

    # --- LI/LC opcode relays + PSH-at-SP flag (465..467 inside CS_DIST_THERMO) ---
    pin("OP_LI_RELAY", 465, 1, "LI active, relayed to AX byte positions",
        "mark == AX AND opcode_in_step in {LI}", alias=True)
    pin("OP_LC_RELAY", 466, 1, "LC active, relayed to AX byte positions",
        "mark == AX AND opcode_in_step in {LC}", alias=True)
    pin("PSH_AT_SP", 467, 1, "PSH opcode flag relayed to SP/STACK0",
        "(mark == SP OR mark == STACK0) AND opcode_in_step in {PSH}",
        alias=True)

    # --- PRTF/READ IO state machine scratch (alias MEM_VAL/RELAY slots) ---
    pin("IO_IS_PRTF", 464, 1, "PRTF opcode detected (aliases MEM_VAL_B3)",
        "mark == AX", alias=True)
    pin("IO_IS_READ", 465, 1, "READ opcode detected (aliases OP_LI_RELAY)",
        "mark == AX", alias=True)
    pin("IO_STATE",   466, 1, "IO state machine (aliases OP_LC_RELAY)",
        "mark == AX OR NOT is_byte", alias=True)
    pin("IO_OUTPUT_COUNT", 467, 1,
        "Output bytes remaining (aliases PSH_AT_SP)",
        "mark == AX OR NOT is_byte", alias=True)
    pin("IO_FORMAT_POS",   468, 1, "Position in format string (aliases MEM_EXEC)",
        "mark == AX OR NOT is_byte", alias=True)
    pin("MEM_EXEC", 468, 1, "Deprecated; retained as IO_FORMAT_POS alias",
        "mark == AX OR NOT is_byte", alias=True)
    pin("IO_IN_OUTPUT_MODE",  469, 1, "Currently emitting output bytes",
        "is_byte OR NOT is_byte", alias=True)
    pin("IO_OUTPUT_COMPLETE", 470, 1, "Format string complete",
        "is_byte OR NOT is_byte", alias=True)

    # --- FORMAT_PTR / AX_FULL nibble pointers (471..502) ---
    # 471..486 sits inside CS_DIST_THERMO (464..479) + TEMP (480..511).
    # 487..502 sits inside TEMP.
    pin("FORMAT_PTR_LO", 471, 16,
        "Format string ptr lo nibble (aliases AX_FULL_LO)",
        "mark == AX OR (is_byte AND byte_index == 0)", alias=True)
    pin("FORMAT_PTR_HI", 487, 16,
        "Format string ptr hi nibble (aliases AX_FULL_HI)",
        "mark == AX OR (is_byte AND byte_index == 0)", alias=True)
    pin("AX_FULL_LO",    471, 16, "Full AX lo nibble (aliases FORMAT_PTR_LO)",
        "mark == AX OR (is_byte AND byte_index == 0)", alias=True)
    pin("AX_FULL_HI",    487, 16, "Full AX hi nibble (aliases FORMAT_PTR_HI)",
        "mark == AX OR (is_byte AND byte_index == 0)", alias=True)

    # --- OUTPUT_BYTE nibbles — alias TEMP+0..15 / TEMP+16..31 ---
    pin("OUTPUT_BYTE_LO", 480, 16, "Output byte lo nibble (aliases TEMP[0:16])",
        "is_byte OR NOT is_byte", alias=True)
    pin("OUTPUT_BYTE_HI", 496, 16, "Output byte hi nibble (aliases TEMP[16:32])",
        "is_byte OR NOT is_byte", alias=True)

    # --- "LAST_WAS_*" / "ACTIVE_OPCODE_*" / "MARK_THINKING_*" flags ---
    pin("LAST_WAS_THINKING_END",   501, 1, "Prev token was THINKING_END",
        "NOT is_byte", alias=True)
    pin("LAST_WAS_THINKING_START", 502, 1, "Prev token was THINKING_START",
        "NOT is_byte", alias=True)
    pin("LAST_WAS_BYTE", 503, 1, "Prev token was byte (0-255)",
        "is_byte OR NOT is_byte", alias=True)
    pin("LAST_WAS_IO_STATE_EMIT_BYTE", 462, 1,
        "Prev token was IO_STATE_EMIT_BYTE (aliases MEM_VAL_B1)",
        "is_byte OR NOT is_byte", alias=True)
    pin("LAST_WAS_IO_STATE_EMIT_THINKING", 463, 1,
        "Prev token was IO_STATE_EMIT_THINKING (aliases MEM_VAL_B2)",
        "is_byte OR NOT is_byte", alias=True)
    pin("ACTIVE_OPCODE_PRTF", 504, 1, "Current opcode is PRTF",
        "mark == AX AND opcode_at_AX == PRTF", alias=True)
    pin("ACTIVE_OPCODE_READ", 505, 1, "Current opcode is READ",
        "mark == AX AND opcode_at_AX == READ", alias=True)
    pin("MARK_THINKING_START", 506, 1, "THINKING_START token marker",
        "NOT is_byte", alias=True)
    pin("MARK_THINKING_END",   507, 1, "THINKING_END token marker",
        "NOT is_byte", alias=True)

    # --- POST_PRTF aliases (471..502 / 328..359) ---
    pin("POST_PRTF_PC_LO", 471, 16, "Post-PRTF PC lo (aliases AX_FULL_LO)",
        "mark == AX OR (is_byte AND byte_index == 0)", alias=True)
    pin("POST_PRTF_PC_HI", 487, 16, "Post-PRTF PC hi (aliases AX_FULL_HI)",
        "mark == AX OR (is_byte AND byte_index == 0)", alias=True)
    pin("POST_PRTF_SP_LO", 328, 16, "Post-PRTF SP lo (aliases AX_CARRY_LO)",
        "mark == AX OR (is_byte AND byte_index == 0)", alias=True)
    pin("POST_PRTF_SP_HI", 344, 16, "Post-PRTF SP hi (aliases AX_CARRY_HI)",
        "mark == AX OR (is_byte AND byte_index == 0)", alias=True)

    # --- Opcode-byte aliases (12, 28) — alias ADDR_B0_LO / ADDR_B1_LO ---
    pin("OPCODE_BYTE_LO", 12, 16, "Opcode byte lo nibble (aliases ADDR_B0_LO)",
        "mark == MEM OR (is_byte AND byte_index == 0)", alias=True)
    pin("OPCODE_BYTE_HI", 28, 16, "Opcode byte hi nibble (aliases ADDR_B1_LO)",
        "mark == MEM OR (is_byte AND byte_index == 0)", alias=True)

    # --- OPCODE_BASE alias (262) — alias of OPCODE_FLAGS / OP_LEA ---
    pin("OPCODE_BASE", 262, 1, "Base of opcode one-hot (aliases OP_LEA)",
        "mark == AX OR (is_byte AND byte_index == 0)", alias=True)

    # =========================================================================
    # Compact pin_io_only layout mirrors (positions 510..732)
    # -----------------------------------------------------------------------
    # Mirror of the registry-completeness ``_PIN`` block in
    # ``build_default_registry`` (commit 57c6a04). The compact
    # ``pin_io_only=True`` layout lays the address / nibble / scratch
    # families out at positions 510..732; the attention IR emitted by
    # ops references those positions directly, so the registry needs
    # named slots at them for ``verify_attention_head`` to resolve dim
    # ints back to names. Each ``_PIN`` alloc mirrors the semantics of
    # its legacy counterpart.
    # =========================================================================

    # OPCODE_BYTE_HI mirror (compact pos 510..525). Legacy OPCODE_BYTE_HI
    # lives at 28 (aliased onto ADDR_B1_LO). Compact layout places it at
    # 510. Note: 510..526 straddles TEMP's tail (480..512) by 2 bytes, so
    # this mirror is registered as an alias of TEMP at the boundary.
    pin("OPCODE_BYTE_HI_PIN", 510, 16,
        "Compact-layout OPCODE_BYTE_HI (mirrors legacy OPCODE_BYTE_HI at 28)",
        "mark == MEM OR (is_byte AND byte_index == 0)", alias=True)

    # ADDR_B*_LO mirrors (compact pos 526..573). Legacy ADDR_B0_LO=12,
    # ADDR_B1_LO=28, ADDR_B2_LO=44 (one-hot low nibbles of gathered addr
    # bytes). Compact layout places the family at 526..573.
    pin("ADDR_B0_LO_PIN", 526, 16,
        "Compact-layout ADDR_B0_LO (mirrors legacy at 12)",
        "mark == MEM")
    pin("ADDR_B1_LO_PIN", 542, 16,
        "Compact-layout ADDR_B1_LO (mirrors legacy at 28)",
        "mark == MEM")
    pin("ADDR_B2_LO_PIN", 558, 16,
        "Compact-layout ADDR_B2_LO (mirrors legacy at 44)",
        "mark == MEM")

    # ADDR_B*_HI mirrors (compact pos 574..621). Legacy ADDR_B0_HI=206,
    # ADDR_B1_HI=222, ADDR_B2_HI=238 (hi nibbles, aliased onto ADDR_KEY).
    # Compact layout places the family at 574..621.
    pin("ADDR_B0_HI_PIN", 574, 16,
        "Compact-layout ADDR_B0_HI (mirrors legacy at 206)",
        "mark == MEM")
    pin("ADDR_B1_HI_PIN", 590, 16,
        "Compact-layout ADDR_B1_HI (mirrors legacy at 222)",
        "mark == MEM")
    pin("ADDR_B2_HI_PIN", 606, 16,
        "Compact-layout ADDR_B2_HI (mirrors legacy at 238)",
        "mark == MEM")

    # FORMAT_PTR_*/AX_FULL_* mirrors (compact pos 622..653). Legacy
    # FORMAT_PTR_LO=471, FORMAT_PTR_HI=487 (aliased with AX_FULL_LO/HI).
    # Compact layout places the family at 622..653.
    pin("FORMAT_PTR_LO_PIN", 622, 16,
        "Compact-layout FORMAT_PTR_LO/AX_FULL_LO (mirrors legacy at 471)",
        "mark == AX OR (is_byte AND byte_index == 0)")
    pin("FORMAT_PTR_HI_PIN", 638, 16,
        "Compact-layout FORMAT_PTR_HI/AX_FULL_HI (mirrors legacy at 487)",
        "mark == AX OR (is_byte AND byte_index == 0)")

    # OUTPUT_BYTE_* mirrors (compact pos 654..685). Legacy OUTPUT_BYTE_LO=480,
    # OUTPUT_BYTE_HI=496 (aliased onto TEMP). Compact layout places them at
    # 654..685.
    pin("OUTPUT_BYTE_LO_PIN", 654, 16,
        "Compact-layout OUTPUT_BYTE_LO (mirrors legacy at 480)",
        "is_byte OR NOT is_byte")
    pin("OUTPUT_BYTE_HI_PIN", 670, 16,
        "Compact-layout OUTPUT_BYTE_HI (mirrors legacy at 496)",
        "is_byte OR NOT is_byte")

    # CARRY mirror (compact pos 686..689). Legacy CARRY=392 (4-wide
    # inter-byte ADD/SUB/MUL carry cascade).
    pin("CARRY_PIN", 686, 4,
        "Compact-layout CARRY (mirrors legacy at 392)",
        "is_byte AND byte_index in {0, 1, 2, 3}")

    # CMP mirror (compact pos 690..697). Legacy CMP=396 (size 4 in legacy
    # registry; compact layout widens to 8 to match the declared size in
    # ``declare_setdim_compat_dims`` -- the ``eight_dim`` family).
    pin("CMP_PIN", 690, 8,
        "Compact-layout CMP (mirrors legacy at 396; widened to 8)",
        "mark == AX OR (is_byte AND byte_index == 0)")

    # TEMP mirror (compact pos 698..729). Legacy TEMP=480 size 32.
    pin("TEMP_PIN", 698, 32,
        "Compact-layout TEMP (mirrors legacy at 480)",
        "is_byte OR NOT is_byte")

    # STACK0_BYTE1/2/3 mirrors (compact pos 730..732). Legacy
    # STACK0_BYTE1/2/3 = 508/509/510 (aliased onto TEMP+28..30).
    pin("STACK0_BYTE1_PIN", 730, 1,
        "Compact-layout STACK0_BYTE1 (mirrors legacy at 508)",
        "mark == STACK0 OR (is_byte AND byte_index == 1)")
    pin("STACK0_BYTE2_PIN", 731, 1,
        "Compact-layout STACK0_BYTE2 (mirrors legacy at 509)",
        "mark == STACK0 OR (is_byte AND byte_index == 2)")
    pin("STACK0_BYTE3_PIN", 732, 1,
        "Compact-layout STACK0_BYTE3 (mirrors legacy at 510)",
        "mark == STACK0 OR (is_byte AND byte_index == 3)")

    return a.to_registry()


__all__ = ["build_default_registry_dynamic"]
