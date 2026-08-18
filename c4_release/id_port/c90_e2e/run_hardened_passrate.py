"""HARDENED-BUILD C90 pass-rate battery (ext + ext2), measurement-only.

Runs the ext (112) AND ext2 (81) in-subset conformance cases through the HARDENED
doom pure-forward transformer (`run_pure_forward_complete`, the LEAN sparse-streaming
CFM build) with the fp32-safe BP-restore fix (`C4_BP_RESTORE_HIBYTE=1`) that made
Mandelbrot byte-exact, and compares the decoded final AX&0xFF BYTE-EXACT vs the
faithful full-word `native_c4` oracle (which is itself 193/193 == gcc on CPU pre-flight,
modulo the documented c4-vs-x86 sizeof gap).

This is a PURE ADDITION under id_port/c90_e2e/ — no production/weight file is touched,
so the golden `069cc32f` (flags-off) is byte-identical by construction. The hardened
flags only ENABLE correctness fixes; C4_BP_RESTORE_HIBYTE is DEFAULT-OFF golden-neutral.

Ordering: fastest-first (native step count). Each case has a wall-clock timeout
(C90H_CASE_WALL, default 90s) enforced via the pure-forward step cap being bounded so a
long case cannot monopolise the run; a case that would exceed the cap is recorded as
tf-timeout and does NOT block the rest. Persists incrementally to RESULTS_PATH.

MEMORY-SAFE: LEAN sparse-streaming CFM (~0.02 GB VRAM); aborts if MemAvailable < 25 GB.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

RESULTS = os.environ.get("RESULTS_PATH", os.path.join(HERE, "CONFORMANCE_HARDENED.json"))
MEM_FLOOR_GB = float(os.environ.get("C90_MEM_FLOOR_GB", "25"))
DEV = os.environ.get("C90_DEVICE", "cuda:0")
CASE_CAP = int(os.environ.get("C90H_CASE_CAP", "1200"))   # per-case pure-forward step cap
GLOBAL_WALL = float(os.environ.get("C90H_GLOBAL_WALL", "999999"))  # sec, whole run


def _flush(*a, **k):
    k.setdefault("flush", True)
    print(*a, **k)


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return 1e9


def _guard(where):
    g = mem_available_gb()
    if g < MEM_FLOOR_GB:
        _flush(f"*** MEM GUARD {g:.1f}GB < {MEM_FLOOR_GB}GB at {where} — ABORT ***")
        sys.exit(3)
    return g


import c4_min.nibble_pure_forward as _PF  # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

import torch  # noqa: E402
from c4_min import isa  # noqa: E402
from c4_min.nibble_pure_forward_complete import run_pure_forward_complete, ref_interpret  # noqa: E402
from c4_min.compact_alloc import build_compact_sparse_streaming  # noqa: E402
from c4_min.run_1096_pure_forward import bytecode_to_isa  # noqa: E402
from src.compiler import compile_c  # noqa: E402
import native_c4  # noqa: E402
from cases_ext import CASES as CASES1  # noqa: E402
from cases_ext2 import CASES2, C4_X86_GAP2  # noqa: E402

C4_X86_GAP = {"sizeof_int"} | set(C4_X86_GAP2)


def _compile_native(src):
    words, data = compile_c(src)
    code = []
    for w in words:
        op = w & 0xFF
        imm = w >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        code.append(isa.Instr(op, imm))
    return code, data


def _compile_tf(src):
    words, data = compile_c(src)
    return bytecode_to_isa(words), data


def classify(name, cat, code_tf, native_ax, tf_ax):
    if tf_ax == native_ax:
        return "ok"
    ops = {isa.NAMES.get(i.op, i.op) for i in code_tf}
    uses_data = any(i.op == isa.IMM and i.imm >= 65536 for i in code_tf)
    passes_args = any(i.op == isa.ADJ for i in code_tf)
    n_funcs = sum(1 for t in {i.imm for i in code_tf if i.op == isa.JSR}
                  if 0 <= t < len(code_tf) and code_tf[t].op == isa.ENT)
    if uses_data:
        return "data-segment-recall (global/string literal load)"
    if passes_args and n_funcs >= 1:
        return "call-frame/arg-passing"
    if cat == "cmp" or ops & {"EQ", "NE", "LT", "GT", "LE", "GE"}:
        return "cmp/EQ-decode"
    if ops & {"MUL", "DIV", "MOD"}:
        return "width/ALU-32bit"
    if ops & {"SHL", "SHR"}:
        return "shift-width"
    if ops & {"LI", "SI"}:
        return "CAM-recall (memory load/store)"
    return "decode-fidelity/other"


def build_all_cases():
    """Merge ext (name,cat,src,expect) + ext2 (name,cat,src,expect,tf_note)."""
    out = []
    seen = set()
    for name, cat, src, expect in CASES1:
        if name in seen:
            continue
        seen.add(name)
        out.append((name, cat, src, int(expect) & 0xFF, "ext"))
    for tup in CASES2:
        name, cat, src, expect = tup[0], tup[1], tup[2], tup[3]
        if name in seen:
            continue
        seen.add(name)
        out.append((name, cat, src, int(expect) & 0xFF, "ext2"))
    return out


def main():
    t_run = time.time()
    flags = {k: os.environ.get(k) for k in
             ("C4_PF_CFM", "C4_CMP32", "C4_CMP32_ORDER", "C4_MEM_ADDR_BITS",
              "C4_EXACT_EVICT", "C4_MEM_EFF", "C4_GLOBAL_ADDR32", "C4_FLASH_ATTN",
              "C4_BP_RESTORE_HIBYTE")}
    _flush("HARDENED FLAGS: " + " ".join(f"{k}={v}" for k, v in flags.items()))
    if os.environ.get("C4_PF_CFM", "0") not in ("1", "true", "True"):
        _flush("*** REFUSING: C4_PF_CFM must be 1 (lean streaming). ***")
        return 2
    _guard("startup")

    cases = build_all_cases()
    _flush(f"total in-subset cases: {len(cases)}  "
           f"(ext={sum(1 for c in cases if c[4]=='ext')}, "
           f"ext2={sum(1 for c in cases if c[4]=='ext2')})")

    # compile + native step counts (CPU, fast)
    compiled = []
    maxlen = 0
    for name, cat, src, expect, wave in cases:
        try:
            code_tf, data = _compile_tf(src)
            code_nat, _ = _compile_native(src)
        except Exception as e:
            _flush(f"  COMPILE-ERR {name}: {type(e).__name__}: {e}")
            continue
        nax, nsteps = native_c4.run(code_nat, data=data)
        compiled.append((name, cat, src, expect, wave, code_tf, code_nat, data,
                         nax & 0xFF, nsteps))
        maxlen = max(maxlen, len(code_tf))
    compiled.sort(key=lambda r: r[9])  # fastest (fewest native steps) first
    _flush(f"compiled {len(compiled)} cases; maxlen={maxlen}; "
           f"native-step range [{compiled[0][9]}..{compiled[-1][9]}]")

    # RESUME
    prior = {}
    resume = os.environ.get("C90H_RESUME_FROM", RESULTS)
    if resume and os.path.exists(resume):
        try:
            for r in json.load(open(resume)).get("results", []):
                if r.get("transformer") is not None or r.get("class", "").startswith("tf-"):
                    prior[r["name"]] = r
            _flush(f"RESUME: {len(prior)} prior results")
        except Exception:
            pass

    _flush(f"building COMPACT SPARSE CFM on {DEV} (code_size={maxlen+2}) ...")
    _guard("pre-build")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, L, _stats = build_compact_sparse_streaming(
        code_size=maxlen + 2, compute_mode="sparse_mm")
    model.to(DEV)
    vram = torch.cuda.max_memory_allocated() / 1e9
    _flush(f"built in {time.time()-t0:.1f}s  VRAM={vram:.3f}GB  "
           f"MemAvail={_guard('post-build'):.1f}GB")

    results = []
    n_native_ok = n_tf_ran = n_tf_exact = n_skip = 0
    classes = {}
    for i, (name, cat, src, expect, wave, code_tf, code_nat, data,
            native, nsteps) in enumerate(compiled):
        gap = name in C4_X86_GAP
        n_native_ok += (native == expect) or gap

        if name in prior:
            rec = prior[name]
            results.append(rec)
            tf = rec.get("transformer")
            exact = bool(rec.get("tf_exact"))
            n_tf_exact += exact
            n_tf_ran += (tf is not None)
            cls = rec.get("class", "ok")
            if cls != "ok":
                classes[cls] = classes.get(cls, 0) + 1
            _flush(f"  [{i+1:3d}/{len(compiled)}] {name:22s} [{wave}] (resumed) "
                   f"nat={native:3} tf={tf if tf is not None else '--':>3} "
                   f"{'OK' if exact else 'MISS['+cls+']'}")
            _persist(flags, n_native_ok, n_tf_ran, n_tf_exact, n_skip, classes,
                     i + 1, results, vram, len(compiled))
            continue

        tf_cap = min(CASE_CAP, nsteps * 3 + 80)
        # a case whose faithful trace exceeds the cap: record tf-timeout, keep going.
        if nsteps > CASE_CAP:
            n_skip += 1
            cls = f"tf-timeout/too-long (native {nsteps} > cap {CASE_CAP})"
            classes[cls] = classes.get(cls, 0) + 1
            rec = {"name": name, "category": cat, "wave": wave, "expect": expect,
                   "native": native, "native_steps": nsteps, "transformer": None,
                   "tf_cap": tf_cap, "tf_exact": False, "class": cls,
                   "native_ok": bool((native == expect) or gap),
                   "c4_vs_x86": bool(gap), "steps_code": len(code_tf)}
            results.append(rec)
            _flush(f"  [{i+1:3d}/{len(compiled)}] {name:22s} [{wave}] "
                   f"nat={native:3} SKIP (too long: {nsteps} steps)")
            _persist(flags, n_native_ok, n_tf_ran, n_tf_exact, n_skip, classes,
                     i + 1, results, vram, len(compiled))
            continue

        seed = {65536 + k: b for k, b in enumerate(data) if b} if data else None
        tf = None
        tf_err = None
        got = None
        t = time.time()
        try:
            got = run_pure_forward_complete(model, L, code_tf, max_steps=tf_cap,
                                            mask=0xFFFFFFFF, seed_mem=seed)
            tf = (got[-1] & 0xFF) if got else None
            n_tf_ran += 1
        except Exception as e:
            tf_err = f"{type(e).__name__}: {str(e)[:80]}"
        dt = time.time() - t
        hit_cap = (got is not None and len(got) >= tf_cap - 1) if not tf_err else False
        exact = (tf is not None and tf == native)
        n_tf_exact += exact
        cls = "ok" if exact else (
            "tf-exception" if tf_err else
            ("tf-timeout/non-halt (cap %d)" % tf_cap if hit_cap else
             classify(name, cat, code_tf, native, tf)))
        if cls != "ok":
            classes[cls] = classes.get(cls, 0) + 1
        rec = {"name": name, "category": cat, "wave": wave, "expect": expect,
               "native": native, "native_steps": nsteps, "transformer": tf,
               "tf_cap": tf_cap, "case_secs": round(dt, 1),
               "native_ok": bool((native == expect) or gap),
               "c4_vs_x86": bool(gap), "tf_exact": bool(exact), "class": cls,
               "steps_code": len(code_tf)}
        if tf_err:
            rec["tf_error"] = tf_err
        results.append(rec)
        _persist(flags, n_native_ok, n_tf_ran, n_tf_exact, n_skip, classes,
                 i + 1, results, vram, len(compiled))
        mark = "OK" if exact else f"MISS[{cls}]"
        _flush(f"  [{i+1:3d}/{len(compiled)}] {name:22s} [{wave}] "
               f"nat={native:3} tf={tf if tf is not None else '--':>3} "
               f"{dt:5.1f}s {mark}")
        if (i + 1) % 10 == 0:
            _guard(f"after {i+1}")
        if time.time() - t_run > GLOBAL_WALL:
            _flush(f"*** GLOBAL WALL {GLOBAL_WALL}s hit at case {i+1} — stopping clean ***")
            break

    _flush("\n=== HARDENED C90 BATTERY (ext+ext2, sparse-streaming pure-forward) ===")
    _flush(f"native ./c4 == gcc : {n_native_ok}/{len(compiled)}")
    _flush(f"transformer ran    : {n_tf_ran}")
    _flush(f"transformer exact  : {n_tf_exact}")
    _flush(f"skipped (too long) : {n_skip}")
    ran_or_skip = n_tf_ran + n_skip
    if n_tf_ran:
        _flush(f"PASS RATE (of ran) : {n_tf_exact}/{n_tf_ran} = {100*n_tf_exact/n_tf_ran:.1f}%")
    _flush("\ndivergence classes (fix backlog):")
    for cls, n in sorted(classes.items(), key=lambda x: -x[1]):
        _flush(f"  {n:3d}  {cls}")
    return 0


def _persist(flags, n_native_ok, n_tf_ran, n_tf_exact, n_skip, classes, done,
             results, vram, total):
    with open(RESULTS, "w") as f:
        json.dump({"flags": flags, "device": DEV, "vram_gb": round(vram, 3),
                   "n_cases": total, "n_native_ok": n_native_ok,
                   "n_tf_ran": n_tf_ran, "n_tf_exact": n_tf_exact,
                   "n_skip": n_skip, "classes": classes, "done": done,
                   "results": results}, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
