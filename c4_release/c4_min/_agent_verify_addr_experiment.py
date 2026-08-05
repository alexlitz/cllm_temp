"""RE-RUN of the audit's divergence experiment for C4_DIRECT_CAM_VERIFY_ADDR.

Reconstructs the DOOM_FASTPATH_FAITHFULNESS_AUDIT_2026_08_05 "deciding experiment"
(scenario table) on the tiny store/load program, and adds the WRONG-ADDRESS
self-consistent case the audit flagged as DRAFT-TRUSTED (rubber-stamped).

Program (the audit's `mem` prog):  ENT 1; LEA 0; PSH; IMM 777; SI; LEA 0; LI; HALT
  -> stores 777 at a local addr, loads it back, final AX = 777.  At the LI step the
  `mem` CAM head reads that address; `resolve_load_rows` picks the store row.

Scenarios (each drafted CLEAN, then CORRUPTED, run through verify_blocks on the
direct-CAM path):

  A) CLEAN draft, direct-CAM         -> should ACCEPT (byte-exact), AX=777.
  B) WRONG-ADDRESS, self-consistent  -> the draft resolves the LI to a DECOY store at
     a DIFFERENT address (value still 777, so the injected value matches the frame the
     register-decode compares).  The model's OWN query is still the CORRECT address
     (the program/residual is unchanged).  With addr-verify OFF this is ACCEPTED
     (rubber-stamped, the audit's scenario E for the address).  With addr-verify ON the
     model's decoded query != the draft's resolved address -> CAUGHT (terminal FAIL).

Run:  python -m c4_min._agent_verify_addr_experiment
"""
import warnings; warnings.filterwarnings('ignore')
import os, copy
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['C4_PF_CFM'] = '1'
os.environ.setdefault('C4_DRAFT_CMP32', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
import torch
from c4_min import isa
from c4_min.pf_speculative import draft_pf_program, verify_blocks
from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.local_attention import install_local_attention, uninstall_local_attention
from c4_min.live_head_attention import install_dead_block_fusion, uninstall_dead_block_fusion
from c4_min import direct_cam_batched as DCB


MEM_PROG = [isa.Instr(isa.ENT, 1), isa.Instr(isa.LEA, 0), isa.Instr(isa.PSH),
            isa.Instr(isa.IMM, 777), isa.Instr(isa.SI),
            isa.Instr(isa.LEA, 0), isa.Instr(isa.LI), isa.Instr(isa.HALT)]
EXPECT_AX = 777


def _reset_forwards(sparse):
    uninstall_dead_block_fusion(sparse)
    uninstall_local_attention(sparse)
    install_local_attention(sparse, window=96, drop_local_kv=True, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)


def _run(sparse, L, code, draft, dev, verify_addr):
    """Run verify_blocks on the direct-CAM path with addr-verify on/off."""
    _reset_forwards(sparse)
    os.environ['C4_DIRECT_CAM_BATCHED'] = '1'
    os.environ['C4_DIRECT_CAM_VERIFY_ADDR'] = '1' if verify_addr else '0'
    tbl = DCB.install_direct_cam_batched(sparse, L, draft, code, verbose=False)
    stats = {}
    vr = verify_blocks(sparse, L, code, draft, block_steps=64, device=dev,
                       evict=False, mask=0xFFFFFFFF, stats=stats, fast=True)
    os.environ['C4_DIRECT_CAM_BATCHED'] = '0'
    os.environ['C4_DIRECT_CAM_VERIFY_ADDR'] = '0'
    return vr, stats


def _corrupt_wrong_address(draft):
    """Make a SELF-CONSISTENT wrong-ADDRESS draft: re-resolve the LI read to a DECOY
    store at a DIFFERENT address whose value is still 777.  The injected value stays
    777 so the register-decode (frames) still passes; only the ADDRESS the draft claims
    the read resolves to is wrong -- exactly the address the model would NOT select."""
    d = copy.deepcopy(draft)
    # the LI read frame + its correct address (mem head).
    li_rf = None
    correct_addr = None
    for rf, rl in d.read_log.items():
        for (head, addr) in rl:
            if head == "mem":
                li_rf, correct_addr = rf, addr
    assert li_rf is not None, "no mem read found to corrupt"
    decoy_addr = (correct_addr + 4) & 0xFFFFFFFF     # a DIFFERENT, 4-aligned address
    # 1) add a DECOY store to the decoy address (value 777) at a frame BEFORE the read,
    #    so latest-write-wins resolves the read to it self-consistently.
    decoy_frame = li_rf - 1
    while decoy_frame in d.store_log:
        decoy_frame -= 1                              # find a free frame slot < li_rf
    d.store_log[decoy_frame] = (decoy_addr, EXPECT_AX)
    # 2) re-point the read's ADDRESS to the decoy (self-consistent: read looks for the
    #    decoy, store wrote the decoy, value matches).  The MODEL's query is unchanged
    #    (built from the real program residual -> still the CORRECT address).
    d.read_log[li_rf] = [("mem", decoy_addr)]
    return d, correct_addr, decoy_addr


def main():
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sparse, L, _ = build_compact_sparse_streaming(
        code_size=64, recurrent_divmod=True, compute_mode='dense_kernel')
    if dev != 'cpu':
        sparse = sparse.to(dev)
    print(f"built dim={sparse.embed.shape[1]} blocks={len(sparse.blocks)} dev={dev}",
          flush=True)

    clean = draft_pf_program(MEM_PROG, max_steps=60, mask=0xFFFFFFFF)
    print(f"\nCLEAN draft: steps={clean.step_count} final_ax={clean.final_ax_masked}",
          flush=True)

    # --- A) CLEAN draft, direct-CAM, addr-verify ON (must NOT false-positive) ---------
    vr, _ = _run(sparse, L, MEM_PROG, clean, dev, verify_addr=True)
    print(f"[A clean, verify ON ] matched={vr.all_matched} "
          f"acc={vr.accepted_steps}/{vr.total_steps} ax={vr.decoded_final_ax} "
          f"{'-> ACCEPTED (no false positive)' if vr.all_matched else '-> FALSE POSITIVE!'}",
          flush=True)
    a_ok = vr.all_matched and vr.decoded_final_ax == EXPECT_AX

    # --- B) WRONG-ADDRESS self-consistent draft: BEFORE (verify OFF) vs AFTER (ON) ----
    bad, correct_addr, decoy_addr = _corrupt_wrong_address(clean)
    print(f"\nWRONG-ADDRESS draft: read re-pointed {correct_addr} -> {decoy_addr} "
          f"(decoy store value {EXPECT_AX}, self-consistent)", flush=True)

    vr_off, _ = _run(sparse, L, MEM_PROG, bad, dev, verify_addr=False)
    print(f"[B wrong-addr, verify OFF] matched={vr_off.all_matched} "
          f"acc={vr_off.accepted_steps}/{vr_off.total_steps} ax={vr_off.decoded_final_ax} "
          f"{'-> ACCEPTED (rubber-stamped, the audit gap)' if vr_off.all_matched else '-> caught'}",
          flush=True)

    vr_on, stats_on = _run(sparse, L, MEM_PROG, bad, dev, verify_addr=True)
    caught = (not vr_on.all_matched
              and vr_on.first_mismatch is not None
              and vr_on.first_mismatch.get("kind") == "cam_addr")
    print(f"[B wrong-addr, verify ON ] matched={vr_on.all_matched} "
          f"acc={vr_on.accepted_steps}/{vr_on.total_steps} "
          f"{'-> CAUGHT (divergence)' if caught else '-> NOT CAUGHT'}", flush=True)
    if vr_on.first_mismatch is not None:
        print(f"    first_mismatch: {vr_on.first_mismatch}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  A clean+verifyON accepted (no false positive): {a_ok}", flush=True)
    print(f"  B wrong-addr verifyOFF accepted (before)      : {vr_off.all_matched}", flush=True)
    print(f"  B wrong-addr verifyON  CAUGHT   (after)       : {caught}", flush=True)
    overall = a_ok and vr_off.all_matched and caught
    print(f"  => before: ACCEPTED (rubber-stamp)  ->  after: CAUGHT  : "
          f"{'PASS' if overall else 'FAIL'}", flush=True)
    return 0 if overall else 1


if __name__ == '__main__':
    raise SystemExit(main())
