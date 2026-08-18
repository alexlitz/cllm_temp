# CLEVER VM RUNTIME — Phase 1 (the fetch-decode-execute FOUNDATION)

Status: **BUILT + BYTE-EXACT** (see `examples/clever_vm_runtime.py --verify`).
Golden `174ece66` not in this repo (stale brief ref); current build-path golden
untouched — this is a NEW file under `examples/`, off every model build path.

## Goal (the go/no-go for the whole build)

Prove the **clever fp32 datapath** supports a **proper sequencer** — a real
fetch → decode/dispatch → execute → update-(PC/SP/BP/AX/stack/memory) loop —
by running a MINIMAL but REAL c4 program BYTE-EXACT end-to-end against the c4
reference (`c4_min.nibble_pure_forward_complete.ref_interpret`, the
SP-addressed memory-stack oracle). Not isolated op-cells: a machine.

## What already existed (reused, not rebuilt)

The byte-exact clever fp32 **op-cells** — all ISOLATED, no runtime:

- `examples/clever_compact_scoring_realtime.py`
  - `_diffmin_decode(value_f, r)` / `torch.floor` — fp32-EXACT digit / floor
    extraction (the difference-min decode the production cell uses). This is
    the substrate every "read a byte / a nibble / a bit out of an fp32 scalar"
    op stands on.
  - `CompactStepModel` — the SwiGLU + (T=1 direct) attention transformer step:
    the real fp32 *compute* datapath, radix-4096, compact candidate band.
- `examples/clever_fp32_fullops.py`
  - `limb_mul_from_limbs` — fp32-exact 8-bit-limb MUL (col-acc 260,864 < 2^24).
- `examples/clever_honest_attn_realtime.py`
  - `DirectCAMReadHead.gather_value(store_val, resolve)` — the **O(1)
    direct-CAM** latest-write-wins memory read (the KV heap/stack read that a
    softmax1+ALiBi CAM collapses to). This is the memory-read mechanism.

Nothing in the pre-existing set has a **fetch-decode-execute loop**, control
flow that actually moves PC, a real stack machine (SP/BP/PSH + the c4 calling
convention), or a real memory model. Phase 1 builds exactly that around the
op-cells.

## The clever fetch-decode-execute architecture

The machine carries EVERY architectural register as an **fp32 scalar** and
EVERY step's transition is computed by fp32 tensor ops (the same primitives the
op-cells use). The state per lane:

    PC, SP, BP, AX  : fp32 scalars   (exact integers, all < 2^24 in these progs)
    stack + heap    : one direct-CAM write-log (addr,val) per lane, latest-wins
    code            : an address-keyed CODE segment (op, imm per PC), CAM-read

Batched: all of the above are `(B,)` fp32 tensors, so B lanes each run their own
independent VM in lock-step — the batched-verify execution model the clever VM
is built for.

### (a) FETCH — direct-CAM read on the CODE segment

The program is laid into a **code CAM**: for each instruction address `i`, a row
keyed on `bits(i)` carrying `(op_i, imm_i)`. Fetch = resolve the query address
`PC` to its row and gather `(op, imm)` — the SAME O(1) `gather_value` resolve
the data-CAM uses (`DirectCAMReadHead`), on the code store. Because the code
address is unique per row, this is a pure content-addressed gather (no recency
decay needed). In Phase 1 the code CAM is a dense `(n_code,)` op/imm table and
the fetch is `code_op[PC]` / `code_imm[PC]` — the byte-exact collapse of the
address-CAM at unique keys (identical values, S-independent bandwidth). This
matches the reference VM's `_bake_code_cam_head` fetch@PC path.

### (b) DECODE / DISPATCH — opcode one-hot select, shared datapath

Decode = build the **opcode one-hot** `onehot[op] ∈ {0,1}^NUM_OPS` from the
fetched `op` scalar (an fp32-exact equality comparison against each opcode id —
`(op == k)`, the difference-min / one-hot indicator the neural model bakes).

Every op is computed on the SAME datapath (one candidate next-state per op), and
the one-hot **routes** which candidate becomes the committed next state:

    next_PC = Σ_k onehot[k] · PC_candidate_k
    next_AX = Σ_k onehot[k] · AX_candidate_k
    next_SP = Σ_k onehot[k] · SP_candidate_k    (etc. for BP, mem writes)

This is exactly a neural dispatch: the one-hot is the routing gate, the
candidate states are the shared datapath's per-op outputs, and the commit is a
one-hot-weighted **gather** (mul-add reduction). No `if`-ladder — all ops'
candidates are computed, then masked. Collisions are structurally impossible
because the one-hot is exactly-one-hot (an fp32 argmax over exact-integer
equality scores is a single 1.0), so exactly one candidate survives per lane.

### (c) EXECUTE — the op-cells on the shared datapath

Each op's candidate next-state is computed by the reused fp32 op-cells:

- **IMM** `ax = imm & 0xFF` — a floor/mask on the fetched immediate.
- **LEA** `ax = (bp + 4*imm) & 0xFF` — fp32 add + mask.
- **PSH** `sp -= 4; mem[sp] = ax` — a stack write to the direct-CAM store.
- **LI**  `ax = mem[ax] & 0xFF` — a direct-CAM read (query = AX).
- **SI**  `mem[pop()] = ax` — pop (a CAM read at SP) then a CAM write.
- **ADD** `v = pop(); ax = (v + ax) & mask` — CAM read at SP + fp32 add + mask.
- **CMP** (EQ/NE/LT/GT/LE/GE) `ax = (v ? ax)` — fp32 compare -> {0,1}.
- **BZ/BNZ** `pc = imm if ax(≠0/==0) else pc` — a gated PC mux.
- **JMP** `pc = imm`.
- **JSR** `mem[sp-4] = i+1; sp -= 4; pc = imm` — push return addr + jump.
- **ENT** `mem[sp-4] = bp; sp -= 4; bp = sp; sp -= 4*imm` — frame prologue.
- **ADJ** `sp += 4*imm`.
- **LEV** `sp = bp; bp = mem[sp]; pc = mem[sp+4]; sp += 8` — frame epilogue
  (TWO CAM reads: saved BP at `sp`, return PC at `sp+4`).

The masks/floors are `torch.floor`-based (byte-exact in fp32 for these
integers); the pops/loads/reads are `gather_value` on the write-log; the writes
append `(addr, val)` to the write-log.

### (d) UPDATE — one-hot commit of PC/SP/BP/AX + memory write

After the candidates are muxed by the one-hot, the committed `(PC, SP, BP, AX)`
become the next step's registers, and any op that wrote memory has appended its
`(addr, val)` to the per-lane write-log (latest-write-wins ordering by append
index — exactly the direct-CAM recency the reference uses). The default PC
candidate is `PC+1` (the sequential fetch bump); JMP/BZ/BNZ/JSR/LEV override it
through their candidates.

## Memory model — one write-log, three roles

`stack`, `heap`, and `code` are all **address-keyed KV**. The heap+stack share
ONE per-lane write-log (a growing `(addr, val)` list); a read resolves the
query address to the **latest** matching write (`resolve_latest`), returning 0
(ZFOD) if unwritten. This is the byte-exact host-side collapse of the
softmax1+ALiBi latest-write-wins CAM (`softmax_cam_read`), i.e. the O(1)
`DirectCAMReadHead.gather_value` mechanism. The stack (`PSH`/pop/`JSR`/`ENT`/
`LEV`) and the program stores (`SI`/`SC`) are the SAME kind of address-keyed
write — exactly the reference's unification (`nibble_pure_forward_complete`
docstring: "a push lays a store MEM token … EXACTLY as SI does").

## The HARD parts (hit in Phase 1, and how handled)

1. **Opcode dispatch without collision.** Handled by the exactly-one-hot commit:
   an fp32 equality one-hot over exact-integer opcode ids is a single 1.0 (no two
   ops fire), so the mul-add commit selects exactly one candidate. There is no
   datapath entanglement at Phase-1 width because each register's next value is a
   one-hot-weighted sum of per-op candidates — the ops never write a shared band
   simultaneously. This is the clean version of what the WIDE model fights
   (its shared OUTPUT/ALU bands, per memory notes) — a min-param runtime can keep
   per-op candidates in private lanes and only entangle at the final commit
   reduction, which is collision-free by construction.

2. **The applied-DEPTH problem (variable per-op depth).** MUL runs an 8-limb
   schoolbook (depth ~8 carry rounds); DIV runs a radix long-division (depth ~n
   quotient places); ADD-class is depth ~1. A single fixed-depth forward can't
   serve all. **Phase 1's answer: the SEQUENCER already IS the depth loop.** Each
   VM step is one iteration of the outer fetch-decode-execute loop; a
   variable-depth op is just an op-cell whose *candidate* computation internally
   runs its own fixed number of sub-passes (limb/quotient rounds) — bounded and
   known per op, so it is a static unroll, NOT a data-dependent loop. The outer
   loop's depth is constant (one step); the inner op-cell depth varies but is
   statically-known per opcode. Phase 1 exercises the shallow ops (IMM/LEA/LI/SI/
   ADD/CMP/branch/call); MUL/DIV depth is proven byte-exact already in
   `clever_fp32_fullops.py` and slots in as a deeper candidate branch (deferred to
   Phase 2 dispatch-integration, see plan).

3. **Datapath entanglement.** Deferred/avoided at Phase-1 width by private per-op
   candidate lanes + a one-hot commit. The real entanglement risk is at SCALE
   (all 40 ops sharing a min-param band) — flagged as the biggest risk below.

## Byte-exact verification

`run_state_trace(code)` runs an INSTRUMENTED copy of the reference
`ref_interpret` that records the FULL per-step state `(pc, sp, bp, ax, mem)`.
The clever machine records the SAME per-step state. Verification asserts
`L-inf = 0` between the two traces at EVERY step, for EVERY field
(PC, SP, BP, AX, and every touched memory cell), on a REAL program that
computes → stores → loads → branches → calls.

## Phased plan to the whole ISA + a real c4 program

- **Phase 1 (this).** fetch-decode-execute FOUNDATION: IMM/LEA/LI/SI/ADD/CMP/
  BZ/BNZ/JMP + ENT/ADJ/LEV (function call+return), byte-exact, L-inf=0.
- **Phase 2 — full ISA dispatch.** Add every remaining op's candidate to the
  one-hot commit (OR/XOR/AND, SHL/SHR, SUB/MUL/DIV/MOD, LC/SC, PSH-depth,
  PRTF/READ I/O via the tool-call protocol). MUL/DIV bring their statically-
  unrolled inner depth. Re-verify byte-exact on the 8-bit + 32-bit reference
  batteries. RISK: the min-param datapath must hold all 40 candidates without a
  shared-band collision at the commit — this is where entanglement bites.
- **Phase 3 — into the real neural datapath.** Replace the host fp32 scalar ops
  with the `CompactStepModel` transformer forward (the one-hot dispatch baked as
  FFN routing, the CAM reads as the attention heads) — a min-param model that IS
  the sequencer. Verify byte-exact vs Phase-2 host machine.
- **Phase 4 — a real c4 program.** Run a real module (a doom sub-routine or the
  c4 self-host slice) byte-exact vs the c4 bundler/interpreter oracle
  (`bundler/c4_compile.c`). RISK: PC/address width (12-16 bit code CAM ceiling —
  the reference lifts it with the position-invariant address-CAM); the deep-loop
  step budget; and the variable-depth MUL/DIV integration at production step
  counts.

### Biggest risk (blunt)

**Datapath entanglement at full-ISA min-param width.** Phase 1 keeps per-op
candidates in private lanes so the one-hot commit is collision-free — cheap when
there are ~12 ops and the model is a host scalar machine. At 40 ops in a genuine
min-param NEURAL datapath (Phase 3), the ops must SHARE a narrow residual/ALU
band, and that shared-band write-read is precisely the entanglement the WIDE
model spent the whole campaign fighting (the OUTPUT/ALU-band self-reinforcement
megaroots in the memory notes). The one-hot commit is clean *as arithmetic*; the
open question is whether a MIN-PARAM neural realization can keep the 40 per-op
candidates separable enough that the commit stays exactly-one-hot under fp32
without a shared-band collision. Phase 1 proves the *sequencer* works; it does
NOT prove the min-param NEURAL datapath scales to 40 ops without re-hitting
entanglement — that is the Phase-3 wall to watch.
