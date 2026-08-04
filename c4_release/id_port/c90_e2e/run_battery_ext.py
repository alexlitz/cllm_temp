"""Extended C90 end-to-end battery — three-stage, durable, memory-guarded.

For every case in ``cases_ext.CASES`` runs THREE stages and records a byte-exact
comparison in ``CONFORMANCE_MATRIX_EXT.json``:

  stage 1  gcc          : gcc-15 -std=c90 exit code (independent ground truth)
  stage 2  native ./c4  : native_c4.run (faithful full-word c4 VM, compiler ABI)
  stage 3  transformer  : run_pure_forward_complete on the LEAN STREAMING CFM build
                          with the proven flag set.

The transformer is compared to BOTH native-c4 and gcc.  Because the transformer
uses a 4-byte-cell / *4-LEA frame ABI (vs the compiler's 8-byte-cell / byte-offset
ABI), function-call-with-args and other frame-crossing cases are EXPECTED to hit an
ABI wall on the transformer — the matrix classifies every transformer miss.

MEMORY DISCIPLINE (MEMORY.md): C4_PF_CFM=1 lean streaming ONLY, never
load_sparse_transformer; the build is the ~1.5 GB CFM perf build.  Checks
/proc/meminfo and ABORTS if MemAvailable < 25 GB.  CUDA_VISIBLE_DEVICES=0,1.

Run (proven flag set):
  CUDA_VISIBLE_DEVICES=0,1 C4_PF_CFM=1 C4_DRAFT_CMP32=1 C4_CMP32=1 \
    C4_CMP32_ORDER=1 C4_MEM_ADDR_BITS=18 C4_EXACT_EVICT=1 C4_MEM_EFF=500000 \
    python -m id_port.c90_e2e.run_battery_ext
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

RESULTS = os.environ.get("RESULTS_PATH", os.path.join(HERE, "CONFORMANCE_MATRIX_EXT.json"))
MEM_FLOOR_GB = float(os.environ.get("C90_MEM_FLOOR_GB", "25"))


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 1e9


def _flush(*a, **k):
    k.setdefault("flush", True)
    print(*a, **k)


def _guard(where):
    g = mem_available_gb()
    if g < MEM_FLOOR_GB:
        _flush(f"*** MEM GUARD: MemAvailable {g:.1f} GB < {MEM_FLOOR_GB} GB at {where} — ABORT ***")
        sys.exit(3)
    return g


# small stack window so the transformer's frame-relative LEA reaches locals (same
# as the 1096 tests + the #817 battery); the compiler's absolute data/global
# addresses (>= 0x10000) are unaffected.
import c4_min.nibble_pure_forward as _PF  # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min import isa  # noqa: E402
from c4_min.nibble_pure_forward_complete import (  # noqa: E402
    build_pure_forward_complete_model, run_pure_forward_complete, ref_interpret,
)
from src.compiler import compile_c  # noqa: E402
import native_c4  # noqa: E402
from cases_ext import CASES  # noqa: E402


def _compile(src):
    words, data = compile_c(src)
    code = []
    for w in words:
        op = w & 0xFF
        imm = w >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        code.append(isa.Instr(op, imm))
    return code, data


# ---- transformer-failure divergence classification -------------------------
def classify(name, cat, src, code, native_ax, tf_ax, ref_ax):
    """Assign a divergence CLASS to a transformer miss (the fix backlog bucket).

    native_ax = faithful c4 (compiler ABI, == gcc).  ref_ax = the shipped 8-bit
    ref_interpret (the transformer's own oracle, 4-byte/ *4 ABI).  tf_ax = model.
    """
    ops = {isa.NAMES.get(i.op, i.op) for i in code}
    has_call = "JSR" in ops and any(
        i.op == isa.JSR and 0 <= i.imm < len(code) and code[i.imm].op == isa.ENT
        for i in code)
    # count real (non-startup) function bodies: an ENT that is a JSR target other
    # than the very first startup JSR.
    ent_targets = {i.imm for i in code if i.op == isa.JSR}
    n_funcs = sum(1 for t in ent_targets if 0 <= t < len(code) and code[t].op == isa.ENT)
    passes_args = any(i.op == isa.ADJ for i in code)   # ADJ n present -> args were pushed
    uses_data = any(i.op == isa.IMM and i.imm >= 65536 for i in code)  # string/global literal ptr
    if tf_ax == native_ax:
        return "ok"
    # transformer matches its OWN (4-byte-ABI) oracle but not the faithful c4/gcc:
    if ref_ax is not None and tf_ax == ref_ax and ref_ax != native_ax:
        if passes_args and n_funcs >= 1:
            return "abi-wall-4byte-cell (frame *4 vs compiler byte-offset args)"
        return "ref-abi-divergence (ref_interpret 4-byte cell)"
    # transformer diverges from BOTH oracles -> a genuine model wall:
    if passes_args and n_funcs >= 1:
        return "abi-wall + call-frame (JSR/ENT arg passing)"
    if uses_data:
        return "data-segment-recall (string/global literal load)"
    if cat == "cmp" or ("EQ" in ops or "NE" in ops or "LT" in ops or "GT" in ops):
        return "cmp/EQ-decode"
    if "MUL" in ops or "DIV" in ops or "MOD" in ops:
        return "width/ALU-32bit (mul/div/mod)"
    if "SHL" in ops or "SHR" in ops:
        return "shift-width"
    if "LI" in ops or "SI" in ops:
        return "CAM-recall (memory load/store)"
    return "decode-fidelity/other"


def main():
    flags = {k: os.environ.get(k) for k in
             ("C4_PF_CFM", "C4_DRAFT_CMP32", "C4_CMP32", "C4_CMP32_ORDER",
              "C4_MEM_ADDR_BITS", "C4_EXACT_EVICT", "C4_MEM_EFF", "C4_DIVMOD_SIGNED")}
    _flush("FLAGS: " + " ".join(f"{k}={v}" for k, v in flags.items()))
    _flush(f"cases={len(CASES)}  MemAvailable={mem_available_gb():.1f} GB  floor={MEM_FLOOR_GB} GB")
    _guard("startup")
    if os.environ.get("C4_PF_CFM", "0") not in ("1", "true", "True"):
        _flush("*** REFUSING TO RUN: C4_PF_CFM must be 1 (lean streaming). ***")
        return 2

    # compile everything first (also gives the native + gcc baselines)
    compiled = []
    maxlen = 0
    for name, cat, src, expect in CASES:
        code, data = _compile(src)
        compiled.append((name, cat, src, expect, code, data))
        maxlen = max(maxlen, len(code))

    # FAST ORDER (C90_ORDER=fast): run untested cases fastest-first (by native step
    # count) so a long CPU run maximises category coverage before any slow loop/array
    # case blocks — the divergence CLASS of a category is captured by its fastest rep.
    if os.environ.get("C90_ORDER") == "fast":
        order = {}
        for name, cat, src, expect, code, data in compiled:
            order[name] = native_c4.run(code, data=data)[1]
        compiled.sort(key=lambda r: order[r[0]])

    _flush(f"building LEAN CFM model (code_size={maxlen + 2}) ...")
    _guard("pre-build")
    t0 = time.time()
    model, L = build_pure_forward_complete_model(code_size=maxlen + 2)
    model.eval()
    _flush(f"built in {time.time()-t0:.1f}s.  MemAvailable={_guard('post-build'):.1f} GB")

    # RESUME: reuse completed cases from a prior (interrupted) matrix so a restart
    # after a per-case step-cap tweak does not re-run the already-clean fast cases.
    # Reads C90_RESUME_FROM (a saved snapshot) so it survives RESULTS being overwritten.
    prior = {}
    resume_src = os.environ.get("C90_RESUME_FROM", RESULTS)
    if os.environ.get("C90_RESUME", "1") not in ("0", "", "false") and os.path.exists(resume_src):
        try:
            pj = json.load(open(resume_src))
            for r in pj.get("results", []):
                if r.get("transformer") is not None:
                    prior[r["name"]] = r
            _flush(f"RESUME: {len(prior)} prior case results loaded from {resume_src}")
        except Exception:
            pass

    results = []
    n_native_ok = n_tf_ran = n_tf_exact = 0
    classes = {}
    for i, (name, cat, src, expect, code, data) in enumerate(compiled):
        native_ax, native_steps = native_c4.run(code, data=data)
        native = native_ax & 0xFF
        n_native_ok += (native == expect)
        # per-case transformer step cap: enough slack for a CORRECT run to HALT,
        # but a diverging run is cut off quickly (avoids a 200k growing-stream stall).
        tf_cap = min(int(os.environ.get("C90_CAP_MAX", "2500")), native_steps * 3 + 80)

        # the transformer's own shipped oracle (4-byte cell ABI, 8-bit AX trace)
        try:
            rtr = ref_interpret(code, max_steps=max(20000, tf_cap), mask=0xFFFFFFFF)
            ref_ax = (rtr[-1] & 0xFF) if rtr else None
        except Exception:
            ref_ax = None

        # seed the transformer's data segment (string/global literals) so LC/LI
        # reads see the same bytes gcc/native do.
        seed_mem = {65536 + k: b for k, b in enumerate(data) if b} if data else None

        if name in prior:                      # reuse a prior clean result
            rec = prior[name]
            tf = rec.get("transformer")
            cls = rec.get("class", "ok")
            n_tf_ran += (tf is not None)
            exact = bool(rec.get("tf_exact"))
            n_tf_exact += exact
            if cls != "ok":
                classes[cls] = classes.get(cls, 0) + 1
            results.append(rec)
            with open(RESULTS, "w") as f:
                json.dump({"flags": flags, "n_cases": len(CASES),
                           "n_native_ok": n_native_ok, "n_tf_ran": n_tf_ran,
                           "n_tf_exact": n_tf_exact, "classes": classes,
                           "done": i + 1, "results": results}, f, indent=2)
            _flush(f"  [{i+1:3d}/{len(CASES)}] {name:22s} [{cat:9s}] (resumed) "
                   f"exp={expect:3} nat={native:3} tf={tf if tf is not None else '--':>3} "
                   f"{'OK' if exact else 'MISS['+cls+']'}")
            continue

        tf = None
        tf_err = None
        got = None
        t_case = time.time()
        try:
            got = run_pure_forward_complete(
                model, L, code, max_steps=tf_cap, mask=0xFFFFFFFF,
                seed_mem=seed_mem)
            tf = (got[-1] & 0xFF) if got else None
            n_tf_ran += 1
        except Exception as e:
            tf_err = f"{type(e).__name__}: {str(e)[:80]}"
        case_dt = time.time() - t_case
        # a run that used the whole cap (and missed) never HALTed -> timeout/non-halt.
        hit_cap = (got is not None and len(got) >= tf_cap - 1) if not tf_err else False

        exact = (tf is not None and tf == native)
        n_tf_exact += exact
        cls = "ok" if exact else (
            "tf-exception" if tf_err else
            ("tf-timeout/non-halt (cap %d)" % tf_cap if hit_cap else
             classify(name, cat, src, code, native, tf, ref_ax)))
        if cls != "ok":
            classes[cls] = classes.get(cls, 0) + 1

        rec = {"name": name, "category": cat, "expect": expect,
               "native": native, "native_steps": native_steps, "ref8": ref_ax,
               "transformer": tf, "tf_cap": tf_cap, "case_secs": round(case_dt, 1),
               "native_ok": bool(native == expect),
               "tf_exact": bool(exact), "class": cls, "steps_code": len(code)}
        if tf_err:
            rec["tf_error"] = tf_err
        results.append(rec)

        # durable persist after every case
        with open(RESULTS, "w") as f:
            json.dump({"flags": flags, "n_cases": len(CASES),
                       "n_native_ok": n_native_ok, "n_tf_ran": n_tf_ran,
                       "n_tf_exact": n_tf_exact, "classes": classes,
                       "done": i + 1, "results": results}, f, indent=2)

        mark = "OK" if exact else f"MISS[{cls}]"
        _flush(f"  [{i+1:3d}/{len(CASES)}] {name:22s} [{cat:9s}] "
               f"exp={expect:3} nat={native:3} tf={tf if tf is not None else '--':>3} {mark}")
        if (i + 1) % 10 == 0:
            _guard(f"after case {i+1}")

    _flush(f"\n=== EXTENDED C90 BATTERY ===")
    _flush(f"native ./c4 == gcc : {n_native_ok}/{len(CASES)}")
    _flush(f"transformer ran    : {n_tf_ran}/{len(CASES)}")
    _flush(f"transformer exact  : {n_tf_exact}/{len(CASES)}")
    _flush(f"\ndivergence classes (fix backlog):")
    for cls, n in sorted(classes.items(), key=lambda x: -x[1]):
        _flush(f"  {n:3d}  {cls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
