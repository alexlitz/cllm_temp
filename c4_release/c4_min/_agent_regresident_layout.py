"""READ-ONLY register-resident VM-state / bandwidth probe (#register-resident lever).

Builds the DOOM-config LAYOUT ONLY (no weights, memory-safe: layout is a pure
python dim allocator) and categorises every residual band into:

  (A) CROSS-STEP LIVE  — the persistent VM state the driver EMITS at end-of-step
      and the next step re-ingests (registers + the value lanes the frame carries)
  (B) TRANSIENT         — scratch recomputed every step from the ingested registers
      (fetch/decode/ALU/CMP/address-bin/CAM scratch + the CODE program table)

Also enumerates what the driver actually READS from the last-token residual
(``run_pure_forward_complete``: PC_VAL/SP_VAL/BP_VAL/STK_VAL + AX nibbles + HALTED)
and what ``_build_frame`` re-emits (PC/AX/SP/BP/STACK0 registers => the ingest set).

Prints the band table, the cross-step-live dim count vs transient, and the
register-resident vs full-residual per-step BYTES-CARRIED estimate.

Run:  C4_PF_CFM=1 C4_CODE_ADDR_BITS=20 C4_MEM_ADDR_BITS=18 \
      OMP_NUM_THREADS=4 python -m c4_min._agent_regresident_layout
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

from . import isa
from .nibble_pure_forward_complete import PureForwardCompleteLayout


# The 5 registers the frame carries (INGEST_REGS) — the CROSS-STEP LIVE set the
# next step re-ingests via the frame-ingest CAM.
INGEST_REGS = ["PC", "AX", "SP", "BP", "STACK0"]

# The scalar value lanes the driver READS at end-of-step (the transformer's
# actual cross-step OUTPUT) — see run_pure_forward_complete.
DRIVER_READ_SCALARS = ["PC_VAL", "SP_VAL", "BP_VAL", "STK_VAL", "HALTED"]

# category classifier by band-name prefix/name.
def classify(name: str) -> str:
    # the 5 emitted registers (nibble bands) — CROSS-STEP LIVE (re-ingested)
    if name in INGEST_REGS:
        return "LIVE_REG"
    # the value-lane scalar images of those registers (driver reads these to emit)
    if name in ("PC_VAL", "SP_VAL", "BP_VAL", "STK_VAL", "AX_VAL"):
        return "LIVE_REG_SCALAR"
    if name == "HALTED":
        return "LIVE_CTRL"          # 1 dim: halt flag (emitted as the loop guard)
    # the frame-ingest role tags (structural, set by the overlay each step; a
    # positional tag NOT computed state) — these are INPUT tags, not carried.
    if name in ("ROLE", "IS_FRAME_BYTE", "CUR_NIB", "CTX", "BYTE_OFS"):
        return "INGEST_TAG"
    # the program CODE table (static input, not carried in the residual — it is a
    # baked table dim OR a CFM code-frame band; recomputed/re-read each step)
    if name.startswith("CODE_") or name.startswith("PC_IS"):
        return "CODE_TABLE"
    # the KV-memory CAM bands (stack/heap live in the KV cache, addressed by these
    # per-step-recomputed address bins — NOT cross-step residual state)
    if name in ("ADDR_BIN", "QRY_BIN", "VAL_NIB", "IS_STORE", "IS_LOAD",
                "SP_QRY_BIN", "LEV_QRY_BIN", "UNI_QRY_BIN", "UNI_VAL",
                "IS_POP", "IS_LEV", "IS_MEMREAD",
                "POP_ADDR", "LEV_ADDR", "LEV_RET", "LEV_RET_VAL",
                "CODE_KEY_BIN", "CODE_QRY_BIN", "CODE_OPV", "CODE_IMM_NIB_MEM",
                "IS_CODE", "IS_FETCH"):
        return "MEM_CAM_SCRATCH"
    if name == "ONE" or name == "POS":
        return "CONST"
    if name.startswith("_pad") or name.startswith("_pf") or name.startswith("_bw") \
       or name.startswith("_hd"):
        return "PAD"
    # everything else = per-step TRANSIENT scratch (fetch/decode/ALU/CMP/addr)
    return "TRANSIENT"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-size", type=int, default=int(os.environ.get("C4_PROBE_CODE_SIZE", "1")))
    ap.add_argument("--n-heads", type=int, default=int(os.environ.get("C4_PROBE_NHEADS", "24")))
    args = ap.parse_args()

    L = PureForwardCompleteLayout(code_size=args.code_size, n_heads=args.n_heads)
    D = L.D
    print(f"[layout] PureForwardCompleteLayout D={D}  code_size={args.code_size}  "
          f"n_heads={args.n_heads}  cfm={getattr(L,'cfm',False)}")
    print(f"[layout] flags: PF_CFM={os.environ.get('C4_PF_CFM')} "
          f"CODE_ADDR_BITS={os.environ.get('C4_CODE_ADDR_BITS')} "
          f"MEM_ADDR_BITS={os.environ.get('C4_MEM_ADDR_BITS')} "
          f"CMP32={os.environ.get('C4_CMP32')} SHIFT32={os.environ.get('C4_SHIFT32')} "
          f"LEA_WIDE={os.environ.get('C4_LEA_WIDE')}")

    # aggregate by category
    from collections import defaultdict
    cat_dims = defaultdict(int)
    cat_bands = defaultdict(list)
    per_band = []
    for name, (base, size) in L._names.items():
        c = classify(name)
        cat_dims[c] += size
        cat_bands[c].append((name, base, size))
        per_band.append((base, name, size, c))

    per_band.sort()
    print("\n[bands] base  size  category           name")
    for base, name, size, c in per_band:
        # skip the huge CODE_* per-slot repetition noise in the print
        if name.startswith("CODE_") and args.code_size > 4 and \
           name.split("_")[-1].isdigit() and int(name.split("_")[-1]) > 2:
            continue
        print(f"  {base:5d} {size:4d}  {c:18s} {name}")

    print("\n[category totals]  dims  (of D=%d)" % D)
    order = ["LIVE_REG", "LIVE_REG_SCALAR", "LIVE_CTRL", "INGEST_TAG",
             "CODE_TABLE", "MEM_CAM_SCRATCH", "TRANSIENT", "CONST", "PAD"]
    for c in order:
        if cat_dims[c]:
            print(f"  {c:18s} {cat_dims[c]:5d}  ({100.0*cat_dims[c]/D:5.1f}%)  "
                  f"[{len(cat_bands[c])} bands]")

    # ----- the TRUE cross-step live set -----
    live_reg_dims = cat_dims["LIVE_REG"]                 # 5 registers x 16 nibble dims
    live_scalar_dims = cat_dims["LIVE_REG_SCALAR"]
    live_ctrl = cat_dims["LIVE_CTRL"]
    live_total_residual = live_reg_dims + live_scalar_dims + live_ctrl
    print(f"\n[cross-step LIVE residual dims] "
          f"registers(nibble)={live_reg_dims} + reg-scalars={live_scalar_dims} + "
          f"ctrl={live_ctrl} = {live_total_residual}  ({100.0*live_total_residual/D:.1f}% of D)")

    # what the driver ACTUALLY reads at end-of-step (the transformer's real output)
    read_dims = 0
    for nm in DRIVER_READ_SCALARS:
        if nm in L._names:
            read_dims += L._names[nm][1]
    ax_nib = L._names["AX"][1]                          # AX read as 16 nibbles
    driver_read = read_dims + ax_nib
    print(f"[driver end-of-step READ] {DRIVER_READ_SCALARS} scalars ({read_dims}) + "
          f"AX nibbles ({ax_nib}) = {driver_read} residual dims are the ONLY dims "
          f"of the {D}-dim last-token residual the driver consumes.")

    # the emitted FRAME payload = 5 registers x 4 bytes = 20 bytes (+ opt mem 8 bytes)
    frame_reg_bytes = len(INGEST_REGS) * 4
    print(f"[emitted frame payload] {len(INGEST_REGS)} registers x 4 bytes = "
          f"{frame_reg_bytes} bytes of TRUE VM state (+ up to 8 bytes mem addr/val "
          f"on a store op).  This is what carries to step N+1 via the frame-ingest CAM.")

    # ----- BANDWIDTH: full-residual carry vs register-resident carry -----
    # The per-block residual moved through HBM (the ~371 KB/step figure) is
    # D dims x 4 bytes (fp32) per token per block, over the active span.  If ONLY
    # the cross-step live set had to survive between blocks, the per-block carry
    # would be live_total_residual dims instead of D dims.  (This is the UPPER bound
    # of the residual-carry reduction; the compute still touches transient scratch
    # WITHIN a block.)
    print(f"\n[residual carry ratio] full D={D} dims vs true-live "
          f"{live_total_residual} dims => {D/live_total_residual:.1f}x smaller "
          f"cross-step residual state.")
    print(f"[residual carry ratio] full D={D} dims vs driver-read {driver_read} dims "
          f"=> {D/driver_read:.1f}x smaller if only the emitted state survives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
