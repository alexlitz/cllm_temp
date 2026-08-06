"""BYTE-EXACT + GENUINENESS gate for C4_GENUINE_STRUCTURED_ATTN through the REAL model path.

Mirrors _agent_faithful_sd_byteexact.py but with C4_GENUINE_STRUCTURED_ATTN=1 ON TOP of
C4_FAITHFUL_SINGLE_DISPATCH: the faithful single dispatch's VALUE re-resolution now RUNS the
model's softmax1+ALiBi attention over the ONE hash-resolved winner store row (physical K/V
reconstructed from the store residual) instead of the store_log[frame][1] value atom.

Asserts on the DIV-free battery + a 6315-step deep loop:
  (a) the genuine-structured faithful SD ACCEPTS every step (verdict ok, no false positive);
  (b) same accepted-prefix + same final AX as the plain fast single-dispatch (draft-trusted)
      AND the wanted answer -> L-inf=0 (byte-exact on correct execution).
"""
import warnings; warnings.filterwarnings('ignore')
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
import c4_min.nibble_pure_forward_cached as _PFCa
_PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC
from c4_min import isa
from c4_min.lib_neural import build_lib_model_streaming
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from c4_min.tight_attn_compose import install_composed, uninstall_composed
from c4_min._agent_faithful_sd_byteexact import _programs, COMPOSED


def _flags(mode):
    """mode: 'fast' (draft-trusted) | 'gsa' (faithful + genuine-structured value read)."""
    for f in COMPOSED:
        os.environ[f] = "1"
    for f in ("C4_ONCHIP_RESIDUAL", "C4_RESIDENT_BATCH", "C4_PRECOMPUTED_SCHEDULE",
              "C4_SCHED_FAST_BUILD", "C4_SCHED_GPU_BUILD", "C4_FFN_FUSED_HIDDEN"):
        os.environ[f] = "1"
    os.environ.pop("C4_FAITHFUL_ATTN_EVICT", None)
    if mode == "gsa":
        os.environ["C4_FAITHFUL_SINGLE_DISPATCH"] = "1"
        os.environ["C4_GENUINE_STRUCTURED_ATTN"] = "1"
    else:
        os.environ.pop("C4_FAITHFUL_SINGLE_DISPATCH", None)
        os.environ.pop("C4_GENUINE_STRUCTURED_ATTN", None)


def _run(model, L, code_isa, draft, device, mode):
    _flags(mode); install_composed(model, verbose=False); st = {}
    try:
        vr = verify_blocks(model, L, code_isa, draft, block_steps=512, device=device,
                           evict=True, mask=0xFFFFFFFF, stats=st, fast=True,
                           evict_interval_steps=64, exact_evict=None)
    finally:
        uninstall_composed(model)
        os.environ.pop("C4_FAITHFUL_SINGLE_DISPATCH", None)
        os.environ.pop("C4_GENUINE_STRUCTURED_ATTN", None)
    return vr, st


def main():
    device = "cuda:0"
    model, L, _ = build_lib_model_streaming(code_size=64); model = model.to(device)
    print("[build] done\n", flush=True)
    P = _programs()
    allok = True
    for name, (prog, want_ax) in P.items():
        code_isa = prog if isinstance(prog[0], isa.Instr) else isa.assemble(prog)
        draft = draft_pf_program(code_isa, max_steps=20000, mask=0xFFFFFFFF)
        vg, sg = _run(model, L, code_isa, draft, device, "gsa")     # genuine structured
        vp, sp = _run(model, L, code_isa, draft, device, "fast")    # fast (trusted)
        fa = (vg.decoded_final_ax if vg.all_matched else None)
        pa = (vp.decoded_final_ax if vp.all_matched else None)
        same = (vg.accepted_steps == vp.accepted_steps and vg.all_matched == vp.all_matched
                and fa == pa)
        ax_ok = (vg.all_matched and (fa & 0xFFFF) == (want_ax & 0xFFFF))
        ok = same and ax_ok and sg.get("faithful_ok", None) is True
        allok = allok and ok
        print(f"  [{name:>12}] gsa acc={vg.accepted_steps}/{vg.total_steps} ax={fa} "
              f"| fast acc={vp.accepted_steps}/{vp.total_steps} ax={pa} | want={want_ax} "
              f"| verify_ok={sg.get('faithful_ok')} "
              f"chk[a={sg.get('faithful_addr_checked')} v={sg.get('faithful_value_checked')} "
              f"rt={sg.get('faithful_routing_checked')}] -> {'OK' if ok else 'FAIL'}", flush=True)
    print(f"\n=== GSA BYTE-EXACT battery: "
          f"{'ALL OK (L-inf=0, no false positive; genuine value == fast SD == want)' if allok else 'FAIL'} ===",
          flush=True)


if __name__ == "__main__":
    main()
