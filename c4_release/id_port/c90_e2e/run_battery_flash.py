"""Extended C90 battery — GPU SPARSE-STREAMING + FLASH ATTENTION (task follow-on to #839).

Same three stages as run_battery_ext.py, but the transformer stage runs the COMPACT
SPARSE-STREAMING CFM build on the GPU with BYTE-EXACT FLASH ATTENTION, which makes the
long control / loop / function / recursion cases TRACTABLE (O(S) memory + fast) — the
#839 run stopped at 61/112 because the CPU O(S^2) masked-full attention OOMed / was
minutes-per-step on those.

The three fixes over #839's transformer stage:
  1. FLASH ATTENTION (C4_FLASH_ATTN=1): O(S) memory softmax1+ALiBi (SDPA mem-efficient
     un-cached full case; Triton online-softmax1 cached/windowed case).  Byte-exact
     (softmax1 == plain-softmax over a BOS-sink column) — verified vs the masked-full
     path in verify_flash_byte_exact.py (worst 3.2e-5, below the nibble margin).
  2. bytecode_to_isa (/8 SLOT re-encode) for the TRANSFORMER code: the compiler emits
     LEA/ENT/ADJ immediates as 8-byte-cell BYTE offsets; the transformer's ABI is
     4-byte cells with a *4 LEA/ENT/ADJ scale, so the immediates must be re-encoded
     byte-offset -> slot count (imm//8), exactly as the doom port + the 1096 corpus do
     (run_1096_pure_forward.bytecode_to_isa).  This is the documented /8->*4 slot
     isomorphism the doom aligned-oracle proves byte-exact — it dissolves the #839
     "fn_call ABI wall" (a REFERENCE/feeding bug, not a transformer gap).
  3. C4_GLOBAL_ADDR32=1: widen the LI/LC load-address CAM query to the full 32 bits so
     a global / string-literal pointer at data-segment address 0x10000+ recalls its own
     store instead of aliasing addr&0xFF -> the ZFOD sink (fixes global_rw/global_two +
     str_first_char/str_literal_index, the #838 fix).

native_c4 stays on its own consistent 8-byte-cell / byte-offset ABI (SCALE=1, CELL=8):
it is already 112/112 == gcc, and it feeds RAW byte-offset immediates (its native ABI).
The transformer feeds the /8 slot-re-encoded immediates.  Both model the SAME c4
semantics; they agree byte-exact.

MEMORY DISCIPLINE (MEMORY.md): LEAN STREAMING only — the compact sparse build is
CSR-resident (~0.02-0.1 GB VRAM); flash keeps per-step attention O(S).  NEVER
load_sparse_transformer / the 108 GB dense load.  Checks /proc/meminfo, aborts if
MemAvailable < 25 GB.  CUDA_VISIBLE_DEVICES=0,1.

Run:
  CUDA_VISIBLE_DEVICES=0,1 C4_PF_CFM=1 C4_CMP32=1 C4_CMP32_ORDER=1 C4_MEM_ADDR_BITS=18 \
    C4_EXACT_EVICT=1 C4_MEM_EFF=500000 C4_GLOBAL_ADDR32=1 C4_FLASH_ATTN=1 \
    PYTHONPATH=<c4_release> python id_port/c90_e2e/run_battery_flash.py
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
DEV = os.environ.get("C90_DEVICE", "cuda:0")


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
from cases_ext import CASES  # noqa: E402


def _compile_native(src):
    """Native-c4 code: RAW compiler byte-offset immediates (8-byte-cell ABI)."""
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
    """Transformer code: /8 SLOT-re-encoded immediates (4-byte-cell *4 ABI)."""
    words, data = compile_c(src)
    return bytecode_to_isa(words), data


# c4-vs-x86 EXPECTED semantic gap (c4 sizeof(int)==8, x86 4): NOT a fail.
C4_X86_GAP = {"sizeof_int"}


def classify(name, cat, code_tf, native_ax, tf_ax, ref_ax):
    """Divergence CLASS for a transformer miss (native_ax == faithful c4 == gcc)."""
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


def main():
    flags = {k: os.environ.get(k) for k in
             ("C4_PF_CFM", "C4_CMP32", "C4_CMP32_ORDER", "C4_MEM_ADDR_BITS",
              "C4_EXACT_EVICT", "C4_MEM_EFF", "C4_GLOBAL_ADDR32", "C4_FLASH_ATTN")}
    _flush("FLAGS: " + " ".join(f"{k}={v}" for k, v in flags.items()))
    _flush(f"cases={len(CASES)}  MemAvailable={mem_available_gb():.1f} GB  device={DEV}")
    _guard("startup")
    if os.environ.get("C4_PF_CFM", "0") not in ("1", "true", "True"):
        _flush("*** REFUSING: C4_PF_CFM must be 1 (lean streaming). ***")
        return 2
    if os.environ.get("C4_FLASH_ATTN", "0") != "1":
        _flush("*** WARNING: C4_FLASH_ATTN != 1 — the long cases will NOT be tractable. ***")

    compiled = []
    maxlen = 0
    for name, cat, src, expect in CASES:
        code_tf, data = _compile_tf(src)
        code_nat, _ = _compile_native(src)
        compiled.append((name, cat, src, expect, code_tf, code_nat, data))
        maxlen = max(maxlen, len(code_tf))

    # FAST ORDER: run untested cases fastest-first (by native step count) so a long run
    # maximises coverage before the slow loop/recursion cases.
    if os.environ.get("C90_ORDER", "fast") == "fast":
        order = {}
        for name, cat, src, expect, code_tf, code_nat, data in compiled:
            order[name] = native_c4.run(code_nat, data=data)[1]
        compiled.sort(key=lambda r: order[r[0]])

    _flush(f"building COMPACT SPARSE CFM on {DEV} (code_size={maxlen + 2}) ...")
    _guard("pre-build")
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    model, L, _stats = build_compact_sparse_streaming(
        code_size=maxlen + 2, compute_mode="sparse_mm")
    model.to(DEV)
    vram = torch.cuda.max_memory_allocated() / 1e9
    _flush(f"built in {time.time()-t0:.1f}s.  VRAM={vram:.3f} GB  "
           f"MemAvailable={_guard('post-build'):.1f} GB")

    # RESUME from a prior snapshot (survives RESULTS being overwritten).
    prior = {}
    resume_src = os.environ.get("C90_RESUME_FROM", "")
    if resume_src and os.path.exists(resume_src):
        try:
            for r in json.load(open(resume_src)).get("results", []):
                if r.get("transformer") is not None:
                    prior[r["name"]] = r
            _flush(f"RESUME: {len(prior)} prior results from {resume_src}")
        except Exception:
            pass

    results = []
    n_native_ok = n_tf_ran = n_tf_exact = 0
    classes = {}
    for i, (name, cat, src, expect, code_tf, code_nat, data) in enumerate(compiled):
        native_ax, native_steps = native_c4.run(code_nat, data=data)
        native = native_ax & 0xFF
        gap = name in C4_X86_GAP
        n_native_ok += (native == expect) or gap
        tf_cap = min(int(os.environ.get("C90_CAP_MAX", "3000")), native_steps * 3 + 80)

        if name in prior:
            rec = prior[name]
            tf = rec.get("transformer")
            cls = rec.get("class", "ok")
            n_tf_ran += (tf is not None)
            exact = bool(rec.get("tf_exact"))
            n_tf_exact += exact
            if cls != "ok":
                classes[cls] = classes.get(cls, 0) + 1
            results.append(rec)
            _flush(f"  [{i+1:3d}/{len(CASES)}] {name:22s} [{cat:9s}] (resumed) "
                   f"exp={expect:3} nat={native:3} tf={tf if tf is not None else '--':>3} "
                   f"{'OK' if exact else 'MISS['+cls+']'}")
            _persist(flags, n_native_ok, n_tf_ran, n_tf_exact, classes, i + 1, results, vram)
            continue

        # the model's OWN 8-bit oracle (ref_interpret) on the /8 slot-re-encoded code.
        try:
            rtr = ref_interpret(code_tf, max_steps=max(20000, tf_cap), mask=0xFFFFFFFF)
            ref_ax = (rtr[-1] & 0xFF) if rtr else None
        except Exception:
            ref_ax = None

        seed_mem = {65536 + k: b for k, b in enumerate(data) if b} if data else None

        tf = None
        tf_err = None
        got = None
        t_case = time.time()
        try:
            got = run_pure_forward_complete(
                model, L, code_tf, max_steps=tf_cap, mask=0xFFFFFFFF, seed_mem=seed_mem)
            tf = (got[-1] & 0xFF) if got else None
            n_tf_ran += 1
        except Exception as e:
            tf_err = f"{type(e).__name__}: {str(e)[:80]}"
        case_dt = time.time() - t_case
        hit_cap = (got is not None and len(got) >= tf_cap - 1) if not tf_err else False

        # a transformer miss on the sizeof_int c4-vs-x86 case: only an EXPECTED gap if
        # the transformer matches native c4 (== 8); it should, since it runs c4 semantics.
        exact = (tf is not None and tf == native)
        n_tf_exact += exact
        cls = "ok" if exact else (
            "tf-exception" if tf_err else
            ("tf-timeout/non-halt (cap %d)" % tf_cap if hit_cap else
             classify(name, cat, code_tf, native, tf, ref_ax)))
        if cls != "ok":
            classes[cls] = classes.get(cls, 0) + 1

        rec = {"name": name, "category": cat, "expect": expect,
               "native": native, "native_steps": native_steps, "ref8": ref_ax,
               "transformer": tf, "tf_cap": tf_cap, "case_secs": round(case_dt, 1),
               "native_ok": bool((native == expect) or gap),
               "c4_vs_x86": bool(gap),
               "tf_exact": bool(exact), "class": cls, "steps_code": len(code_tf)}
        if tf_err:
            rec["tf_error"] = tf_err
        results.append(rec)
        _persist(flags, n_native_ok, n_tf_ran, n_tf_exact, classes, i + 1, results, vram)

        mark = "OK" if exact else f"MISS[{cls}]"
        _flush(f"  [{i+1:3d}/{len(CASES)}] {name:22s} [{cat:9s}] "
               f"exp={expect:3} nat={native:3} tf={tf if tf is not None else '--':>3} "
               f"{case_dt:5.1f}s {mark}")
        if (i + 1) % 10 == 0:
            _guard(f"after case {i+1}")

    _flush(f"\n=== EXTENDED C90 BATTERY (GPU sparse-streaming + flash) ===")
    _flush(f"native ./c4 == gcc : {n_native_ok}/{len(CASES)}")
    _flush(f"transformer ran    : {n_tf_ran}/{len(CASES)}")
    _flush(f"transformer exact  : {n_tf_exact}/{len(CASES)}")
    _flush(f"\ndivergence classes (fix backlog):")
    for cls, n in sorted(classes.items(), key=lambda x: -x[1]):
        _flush(f"  {n:3d}  {cls}")
    return 0


def _persist(flags, n_native_ok, n_tf_ran, n_tf_exact, classes, done, results, vram):
    with open(RESULTS, "w") as f:
        json.dump({"flags": flags, "device": DEV, "vram_gb": round(vram, 3),
                   "n_cases": len(CASES), "n_native_ok": n_native_ok,
                   "n_tf_ran": n_tf_ran, "n_tf_exact": n_tf_exact,
                   "classes": classes, "done": done, "results": results}, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
