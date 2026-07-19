"""Per-op-class representative test programs (isa format).

Extracted from the removed _probe_pf_corpus_sample runner (kept for the
top-1 MoE + byte-identity test batteries). Side-effect-free: NO SP_INIT
mutation, NO model import — just the program table.
"""
from __future__ import annotations

DATA = 0x40          # small data address in place of the oracle's 0x10000

def progs_by_class():
    """One representative isa-format program per op-class (mirrors oracle.py)."""
    P = {}
    # --- binops: IMM a; PSH; IMM b; <op>; HALT ---
    binop_cases = {
        "ADD": (3, 4), "SUB": (9, 4), "MUL": (6, 7), "DIV": (84, 7), "MOD": (84, 5),
        "AND": (0x6C, 0x3A), "OR": (0x6C, 0x3A), "XOR": (0x6C, 0x3A),
        "SHL": (5, 3), "SHR": (200, 2),
        "EQ": (5, 5), "NE": (7, 9), "LT": (7, 9), "GT": (9, 7), "LE": (7, 9), "GE": (9, 7),
    }
    for op, (a, b) in binop_cases.items():
        P[op] = [(("IMM", a), ("PSH", 0), ("IMM", b), (op, 0), ("HALT", 0))]
    # --- IMM ---
    P["IMM"] = [(("IMM", 42), ("HALT", 0))]
    # --- PSH (round-trip through ADD 0) ---
    P["PSH"] = [(("IMM", 200), ("PSH", 0), ("IMM", 0), ("ADD", 0), ("HALT", 0))]
    # --- JMP ---
    P["JMP"] = [(("JMP", 2), ("IMM", 99), ("IMM", 5), ("HALT", 0))]
    # --- BZ / BNZ ---
    P["BZ"] = [(("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0))]
    P["BNZ"] = [(("IMM", 1), ("BNZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0))]
    # --- LI / LC / SI / SC : store-then-load (the pure-forward KV memory) ---
    P["SI"] = [(("IMM", DATA), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
               ("IMM", DATA), ("LI", 0), ("HALT", 0))]
    P["SC"] = [(("IMM", DATA), ("PSH", 0), ("IMM", 0x41), ("SC", 0),
               ("IMM", DATA), ("LC", 0), ("HALT", 0))]
    P["LI"] = [(("IMM", DATA), ("PSH", 0), ("IMM", 0x5A), ("SI", 0),
               ("IMM", DATA), ("LI", 0), ("HALT", 0))]
    P["LC"] = [(("IMM", DATA), ("PSH", 0), ("IMM", 0x5A), ("SC", 0),
               ("IMM", DATA), ("LC", 0), ("HALT", 0))]
    # --- LEA / ENT : frame local store+load (ENT n; LEA slot; SI; LEA slot; LI) ---
    #   ENT 1 (imm=SLOTS): reserve 1 local at BP-4 -> LEA slot -1 (BP-4).
    P["ENT"] = [(("ENT", 1), ("LEA", -1), ("PSH", 0), ("IMM", 0x2A), ("SI", 0),
                ("LEA", -1), ("LI", 0), ("HALT", 0))]
    P["LEA"] = [(("ENT", 1), ("LEA", -1), ("PSH", 0), ("IMM", 0x37), ("SI", 0),
                ("LEA", -1), ("LI", 0), ("HALT", 0))]
    # --- ADJ : push then discard ---
    P["ADJ"] = [(("IMM", 0x99), ("PSH", 0), ("ADJ", 1), ("IMM", 0x11), ("HALT", 0))]
    # --- JSR / LEV : a tiny call to a leaf returning 0x2A ---
    #   [0] IMM 0 ; [1] JSR 3 ; [2] HALT ; [3] IMM 0x2A ; [4] LEV
    call = (("IMM", 0), ("JSR", 3), ("HALT", 0), ("IMM", 0x2A), ("LEV", 0))
    P["JSR"] = [call]
    P["LEV"] = [call]
    return P



ALL = ("ADD", "SUB", "MUL", "DIV", "MOD",
       "EQ", "NE", "LT", "GT", "LE", "GE",
       "AND", "OR", "XOR", "SHL", "SHR",
       "LI", "LC", "SI", "SC", "PSH",
       "LEA", "IMM", "JMP", "JSR", "ENT", "ADJ", "LEV", "BZ", "BNZ")
