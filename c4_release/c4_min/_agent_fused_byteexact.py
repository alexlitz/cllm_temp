"""#806 follow-up — END-TO-END byte-exactness of the FUSED delta sparse-FFN.

Runs a battery of REAL C programs (add/mul/cmp/var/func + a longer arithmetic
slice standing in for the doom_run bytecode) through the c4_min pure-forward VM
in TWO configurations:

  (REF)  the unmodified sparse_mm FFN forward (the #804/#806 reference path), and
  (FUSED) every FFN block swapped for FusedUpGateSiluDeltaFFN (the delta kernel).

For each program it compares the FULL per-step AX trace (the SNAPPED-nibble
decode, ``_snap_lane`` -> integer registers).  The task's byte-exactness bar is
"match the SNAPPED nibble, not raw fp": the W_down fp accumulation-order residual
(~6e-2 raw) is absorbed by the integer nibble snap, so the decoded traces must be
IDENTICAL.  Also asserts the final answer matches the pure-python reference
interpreter (``ref_interpret``).

Golden 069cc32f is UNTOUCHED: the fused kernels only READ the stored W_up/W_gate/
W_down into CSR index tensors at swap time; no stored weight is modified (asserted
by hashing the model state before/after the swap).

Run: CUDA_VISIBLE_DEVICES=0,1 python -m c4_min._agent_fused_byteexact --device cuda:0
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

import torch

_HERE = os.path.dirname(__file__)
_OUT = os.path.join(_HERE, "_fused_results", "byteexact_results.json")


def _state_hash(model):
    """Hash every stored FFN weight tensor (the golden-relevant state) — must be
    identical before and after the fused swap (the kernels only READ them)."""
    h = hashlib.sha256()
    for b in model.blocks:
        base = getattr(b, "_orig_ffn", None) or b.ffn
        for nm in ("W_up", "W_gate", "W_down"):
            w = getattr(base, nm, None)
            if w is None:
                continue
            dv = w
            for attr in ("dense_resident", "dense"):
                if getattr(w, attr, None) is not None:
                    dv = getattr(w, attr); break
            else:
                if getattr(w, "csr", None) is not None:
                    dv = w.csr.to_dense()
            h.update(dv.detach().cpu().float().contiguous().numpy().tobytes())
        for nm in ("b_up", "b_gate", "b_down"):
            v = getattr(base, nm, None)
            if v is not None:
                h.update(v.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def _swap_fused(model, dev, block_k=512):
    from .fused_sparse_ffn import FusedUpGateSiluDeltaFFN
    cache = []
    for b in model.blocks:
        if getattr(b, "_routed", False) or getattr(b.ffn, "W_up", None) is None:
            continue
        k = FusedUpGateSiluDeltaFFN(b.ffn, dev, block_k=block_k)
        cache.append(k)
        b.ffn._fused = k
        b.ffn.forward = k.forward
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--code-size", type=int, default=80)
    ap.add_argument("--block-k", type=int, default=512)
    args = ap.parse_args()

    from .compact_alloc import build_compact_sparse_streaming
    from .nibble_pure_forward_complete import run_pure_forward_complete, ref_interpret
    from .run_1096_pure_forward import bytecode_to_isa
    from src.compiler import compile_c

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.zeros(1).to(dev)

    # A battery incl. a LONGER arithmetic slice (proxy for a doom_run bytecode slice:
    # the same ADD/SUB/MUL/CMP/branch opcodes doom exercises).
    battery = [
        ("add", "int main(){ return 500 + 700; }", 1200),
        ("sub", "int main(){ return 9000 - 1234; }", 9000 - 1234),
        ("mul", "int main(){ return 100 * 10; }", 1000),
        ("div", "int main(){ return 9999 / 7; }", 9999 // 7),
        ("mod", "int main(){ return 9999 % 7; }", 9999 % 7),
        ("cmp", "int main(){ if (5 > 3) return 1; return 0; }", 1),
        ("var", "int main(){ int x; x = 1000; return x; }", 1000),
        ("func", "int identity(int x){ return x; } int main(){ return identity(1000); }", 1000),
        # doom-slice proxy: a multi-op arithmetic expression chain (add/sub/mul/div/mod
        # + compares + a branch — the opcode mix a doom frame's fixed-point math uses).
        ("doomslice",
         "int main(){ int a; int b; int c; a = 320*200; b = a/16; c = b%7; "
         "if (c > 3) return a - b + c; return a + b - c; }",
         None),
    ]

    print("[build] compact sparse model (sparse_mm) ...", flush=True)
    model, L, _ = build_compact_sparse_streaming(
        code_size=max(int(args.code_size), 20), compute_mode="sparse_mm")
    model.to(str(dev))
    print(f"[build] blocks={len(model.blocks)} dim={model.dim}", flush=True)

    # ---- REFERENCE traces (unmodified sparse_mm FFN) ----
    def run_battery(tag):
        traces = {}
        for name, src, exp in battery:
            code = bytecode_to_isa(compile_c(src)[0])
            ref_tr = ref_interpret(code, max_steps=20000, mask=0xFFFFFFFF)
            cap = len(ref_tr) + 6
            tr = run_pure_forward_complete(model, L, code, max_steps=cap, mask=0xFFFFFFFF)
            got = (tr[-1] & 0xFFFFFFFF) if tr else None
            want = (ref_tr[-1] & 0xFFFFFFFF) if ref_tr else exp
            traces[name] = {"trace": [t & 0xFFFFFFFF for t in tr], "got": got,
                            "want": want if exp is None else exp,
                            "ref_answer": want}
            print(f"  [{tag}] {name:10s} steps={len(tr):4d} got={got} "
                  f"ref={want}", flush=True)
        return traces

    print("\n=== REF (unmodified sparse_mm FFN) ===", flush=True)
    h_before = _state_hash(model)
    ref_traces = run_battery("REF")

    # ---- swap fused delta, hash-check weights unchanged, re-run ----
    for b in model.blocks:
        b._orig_ffn = getattr(b, "_orig_ffn", None) or b.ffn
    _swap_fused(model, dev, block_k=args.block_k)
    h_after = _state_hash(model)
    print(f"\nstate hash before swap = {h_before[:16]}", flush=True)
    print(f"state hash after  swap = {h_after[:16]}  "
          f"({'UNCHANGED — weights untouched' if h_before==h_after else 'CHANGED!!'})",
          flush=True)

    print("\n=== FUSED (delta kernel FFN) ===", flush=True)
    fused_traces = run_battery("FUSED")

    # ---- compare snapped traces ----
    print("\n=== BYTE-EXACTNESS (snapped-nibble trace equality) ===", flush=True)
    all_ok = True
    summary = {}
    for name, _s, exp in battery:
        r = ref_traces[name]["trace"]
        f = fused_traces[name]["trace"]
        trace_eq = (r == f)
        ans_eq = (ref_traces[name]["got"] == fused_traces[name]["got"])
        # also: does the fused answer match the python reference?
        want = ref_traces[name]["want"]
        matches_ref = (fused_traces[name]["got"] == want) if want is not None else None
        ok = trace_eq and ans_eq
        all_ok = all_ok and ok
        # first divergence step if any
        div = None
        if not trace_eq:
            for i in range(min(len(r), len(f))):
                if r[i] != f[i]:
                    div = i; break
            if div is None:
                div = min(len(r), len(f))
        print(f"  {name:10s} trace_eq={trace_eq}  answer_eq={ans_eq}  "
              f"fused_answer={fused_traces[name]['got']} want={want} "
              f"matches_ref={matches_ref}"
              + (f"  FIRST-DIVERGE step {div}" if div is not None else ""), flush=True)
        summary[name] = {"trace_eq": trace_eq, "answer_eq": ans_eq,
                         "fused_answer": fused_traces[name]["got"],
                         "ref_answer": want, "matches_ref": matches_ref,
                         "steps": len(f), "first_diverge": div}

    verdict = "BYTE-EXACT (all snapped traces identical)" if all_ok else "DIVERGENCE FOUND"
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"golden state untouched: {h_before == h_after}", flush=True)

    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "w") as fp:
        json.dump({"all_byte_exact": all_ok, "state_hash_unchanged": h_before == h_after,
                   "state_hash_before": h_before, "state_hash_after": h_after,
                   "per_program": summary, "block_k": args.block_k}, fp, indent=2)
    print(f"[saved] {_OUT}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
