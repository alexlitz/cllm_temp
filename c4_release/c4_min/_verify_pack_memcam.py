"""Byte-exact ON-vs-OFF verification for C4_PACK_MEMCAM.

Runs the memory battery (LI/LC/SI/SC + var round-trip + ZFOD + latest-write-
wins) plus a spread of arith / branch / func / cmp programs through the REAL
fused Qwen forward, in two SEPARATE processes (flag OFF, flag ON), and asserts
the emitted per-step AX trace is byte-identical.  Also reports the D_used /
hidden floor delta.
"""
import os, sys, json, subprocess


PROGS = {
    # memory battery
    "mem_store_load": [("IMM", 5), ("PSH", 0), ("IMM", 0x23), ("SI", 0),
                       ("IMM", 5), ("LI", 0), ("HALT", 0)],
    "mem_zfod":       [("IMM", 50), ("LI", 0), ("HALT", 0)],
    "mem_latest_win": [("IMM", 30), ("PSH", 0), ("IMM", 1), ("SI", 0),
                       ("IMM", 30), ("PSH", 0), ("IMM", 9), ("SI", 0),
                       ("IMM", 30), ("LI", 0), ("HALT", 0)],
    "var_via_mem":    [("IMM", 10), ("PSH", 0), ("IMM", 7), ("SI", 0),
                       ("IMM", 10), ("LI", 0), ("PSH", 0), ("IMM", 3),
                       ("ADD", 0), ("HALT", 0)],
    "mem_multilocal": [("IMM", 20), ("PSH", 0), ("IMM", 11), ("SI", 0),
                       ("IMM", 24), ("PSH", 0), ("IMM", 22), ("SI", 0),
                       ("IMM", 20), ("LI", 0), ("PSH", 0), ("IMM", 24),
                       ("LI", 0), ("ADD", 0), ("HALT", 0)],
    "mem_lc_signed":  [("IMM", 7), ("PSH", 0), ("IMM", 0x80), ("SC", 0),
                       ("IMM", 7), ("LC", 0), ("HALT", 0)],
    # arith / branch / control / cmp (regression across the packed layout)
    "add":            [("IMM", 200), ("PSH", 0), ("IMM", 55), ("ADD", 0), ("HALT", 0)],
    "sub":            [("IMM", 7), ("PSH", 0), ("IMM", 9), ("SUB", 0), ("HALT", 0)],
    "bz":             [("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)],
    "cmp_lt":         [("IMM", 3), ("PSH", 0), ("IMM", 5), ("LT", 0), ("HALT", 0)],
    "cmp_gt_neg":     [("IMM", 0xFF), ("PSH", 0), ("IMM", 1), ("GT", 0), ("HALT", 0)],
}


def _run_all():
    import c4_min.qwen_full_vm as Q
    from c4_min import isa
    vm = Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)
    out = {"__meta__": {"D_used": vm.QL.D_used, "hidden": vm.hidden_size,
                        "fits_stock": vm.fits_stock,
                        "pack": bool(int(os.environ.get("C4_PACK_MEMCAM", "0")))}}
    for name, prog in PROGS.items():
        r = Q.run_program(vm, isa.assemble(prog), max_steps=48)
        out[name] = {"ax": r["ax_trace"], "ref": r["ref_trace"],
                     "exact": r["exact"]}
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    if os.environ.get("_PMC_CHILD") == "1":
        _run_all()
        sys.exit(0)
    # parent: spawn OFF then ON child, compare
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = {}
    for tag, flag in (("OFF", "0"), ("ON", "1")):
        env = dict(os.environ, _PMC_CHILD="1", C4_PACK_MEMCAM=flag,
                   PYTHONPATH=here, OMP_NUM_THREADS="4")
        p = subprocess.run([sys.executable, "-m", "c4_min._verify_pack_memcam"],
                           capture_output=True, text=True, env=env, cwd=here)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(f"[{tag}] CHILD FAILED\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr[-3000:]}")
            sys.exit(2)
        results[tag] = json.loads(line[0][len("RESULT "):])

    off, on = results["OFF"], results["ON"]
    print("META OFF:", off["__meta__"])
    print("META ON :", on["__meta__"])
    all_ok = True
    for name in PROGS:
        o, n = off[name], on[name]
        ax_same = o["ax"] == n["ax"]
        ref_same = o["ref"] == n["ref"]
        # both must be byte-identical ON vs OFF; and each must be exact vs reference
        ok = ax_same and ref_same and o["exact"] and n["exact"]
        all_ok &= ok
        status = "OK " if ok else "FAIL"
        extra = "" if ok else (
            f"  ax_same={ax_same} ref_same={ref_same} "
            f"off.exact={o['exact']} on.exact={n['exact']}\n"
            f"    OFF.ax={o['ax']}\n    ON .ax={n['ax']}")
        print(f"  [{status}] {name}{extra}")
    print("\nALL BYTE-IDENTICAL ON==OFF AND EXACT:" , all_ok)
    sys.exit(0 if all_ok else 1)
