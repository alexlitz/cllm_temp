#!/usr/bin/env python3
"""MODEL-LEVEL byte-identity gate for the RECURRENT DIV/MOD refactor.

Builds the pure-forward VM two ways via the PRODUCTION streaming-sparse path —
the UNROLLED 300-block divmod and the RECURRENT single-reused-iteration-body
divmod (153 stored blocks, 300 applications) — and runs a battery of DIV/MOD
programs (incl. edge cases) through the REAL driver (``run_pure_forward_complete``,
every step one ``model.forward``).  Asserts the recurrent model produces the
byte-EXACT same AX trace as the unrolled one, and matches the C-source expected
value.  This is the deliverable's byte-identity criterion (greedy/argmax) at the
whole-model level, exercising the apply-order + dim-remap + save/reload path.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python c4_min/_test_recurrent_divmod_model.py
"""
from __future__ import annotations
import os, sys, tempfile
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min.nibble_pure_forward_complete import run_pure_forward_complete
from c4_min.compact_alloc import (build_compact_sparse_streaming,
                                  save_sparse_transformer, load_sparse_transformer)
from c4_min.run_1096_pure_forward import bytecode_to_isa


# DIV/MOD battery: name, C source, expected (values in the corpus <=9999 range the
# operand-load path supports; the divmod ALU itself is 32-bit-exact — see the
# gadget test for the full 32-bit / edge-case coverage incl. 1e9/7).
CASES = [
    ("div_exact",     "int main(){ return 720 / 6; }",       120),
    ("div_142",       "int main(){ return 999 / 7; }",       142),
    ("mod_84_5",      "int main(){ return 84 % 5; }",          4),
    ("mod_1000_13",   "int main(){ return 1000 % 13; }",      12),
    ("div_by_zero",   "int main(){ return 5 / 0; }",           0),
    ("mod_by_zero",   "int main(){ return 5 % 0; }",           0),
    ("div_100_7",     "int main(){ return 100 / 7; }",        14),
    ("mod_100_7",     "int main(){ return 100 % 7; }",         2),
    ("div_a_lt_b",    "int main(){ return 41 / 42; }",         0),
    ("mod_a_lt_b",    "int main(){ return 41 % 42; }",        41),
    ("div_eq",        "int main(){ return 42 / 42; }",         1),
    ("div_9999_3",    "int main(){ return 9999 / 3; }",     3333),
    ("mod_9999_7",    "int main(){ return 9999 % 7; }",        3),
]


def _build(recurrent):
    sparse, L, _ = build_compact_sparse_streaming(
        code_size=44,
        compute_mode="dense_kernel", recurrent_divmod=recurrent)
    return sparse, L


def _trace(model, L, source, cap=512):
    from src.compiler import compile_c
    code = bytecode_to_isa(compile_c(source)[0])
    return run_pure_forward_complete(model, L, code, max_steps=cap, mask=0xFFFFFFFF)


def main():
    print("building UNROLLED divmod (streaming) ...", flush=True)
    m_un, L_un = _build(False)
    print(f"  unrolled stored={len(getattr(m_un,'_phys_blocks',m_un.blocks))} "
          f"applied={len(m_un.blocks)}", flush=True)
    print("building RECURRENT divmod (streaming) ...", flush=True)
    m_rec, L_rec = _build(True)
    print(f"  recurrent stored={len(getattr(m_rec,'_phys_blocks',m_rec.blocks))} "
          f"applied={len(m_rec.blocks)}", flush=True)

    # also exercise SAVE/RELOAD of the recurrent artifact (apply-order round-trip).
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    save_sparse_transformer(m_rec, L_rec, None, path)
    m_reload, L_reload = load_sparse_transformer(path)
    os.unlink(path)
    print(f"  reloaded stored={len(getattr(m_reload,'_phys_blocks',m_reload.blocks))} "
          f"applied={len(m_reload.blocks)}", flush=True)

    ok_ident = ok_val = ok_reload = 0
    for name, source, expected in CASES:
        tr_un = _trace(m_un, L_un, source)
        tr_rec = _trace(m_rec, L_rec, source)
        tr_rel = _trace(m_reload, L_reload, source)
        got_rec = (tr_rec[-1] & 0xFFFFFFFF) if tr_rec else None
        got_un = (tr_un[-1] & 0xFFFFFFFF) if tr_un else None
        ident = (tr_un == tr_rec)
        valok = (got_rec == expected)
        relok = (tr_rel == tr_rec)
        ok_ident += ident; ok_val += valok; ok_reload += relok
        flag = "OK " if (ident and valok and relok) else "MISMATCH"
        print(f"[{flag}] {name:12s} rec={got_rec} unrolled={got_un} want={expected} "
              f"ident={ident} reload_ident={relok}", flush=True)

    n = len(CASES)
    print(f"\n{ok_ident}/{n} recurrent trace == unrolled trace (byte-identity gate)")
    print(f"{ok_val}/{n} recurrent == C-source expected value")
    print(f"{ok_reload}/{n} reloaded artifact == recurrent (save/load round-trip)")
    sys.exit(0 if (ok_ident == n and ok_val == n and ok_reload == n) else 1)


if __name__ == "__main__":
    main()
