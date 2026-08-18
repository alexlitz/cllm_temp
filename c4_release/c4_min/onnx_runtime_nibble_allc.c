/*
 * onnx_runtime_nibble_allc.c  —  the WHOLE VM in C.  ZERO Python.
 *
 * This is onnx_runtime_nibble_incremental.c (windowed-KV incremental block stack,
 * .incbin-embedded COO model) PLUS the ENTIRE driver ported into C:
 *   - the embed table (267 x D), .incbin'd via allc_gen.h   -> token->row lookup
 *   - the program bytecode (op/imm) .incbin'd via allc_gen.h -> overlay in C
 *   - the overlay (port of make_overlay_complete): ONE + CODE_OP/CODE_IMM/
 *     CODE_IMM_NIB per code slot + ROLE/IS_FRAME_BYTE frame tags + the MEM-store
 *     KV row (IS_STORE/ADDR_BIN/VAL_NIB) + the last-row all-ROLE ingest one-hot
 *   - the VM step loop: build residual [1,S,D], run_incremental(S), read x[0,-1]
 *   - byte-decode: _snap_lane (PC/SP/BP/STK/HALTED) + per-byte nibble argmax (AX)
 *   - I/O through §Memory (NOT hardcoded literals), TWO selectable modes:
 *       MODE 1  STRICT  (default)         the PROGRAM pointer-walks §Memory: LC
 *          mem[ptr] each step, PRTF the ONE decoded byte, ptr++ (faithful c4 VM
 *          loop; one model forward per output byte).
 *       MODE 2  BURST   (C4_IO_BURST=1 / --io-burst)   the RUNTIME does the whole
 *          buffer in ONE op: PRTF-burst walks §Memory mem[ptr..NUL] -> stdout in a
 *          C loop (one VM step / no per-byte think-frame); READ injects the whole
 *          stdin buffer into §Memory as MEM-store KV in one op.  Byte-exact vs
 *          MODE 1, far fewer forwards.
 *     §File Ops: OP_READ/OP_OPEN/OP_CLOS are RUNTIME-owned syscalls (the model has
 *     no rules for them) — the runtime marshals the args off the store-log stack,
 *     performs the I/O, and overrides the decoded PC/SP/AX.  READ lays each byte
 *     into §Memory as its own KV frame so a later LC reads it (both modes).
 * ONE static binary does echo/cat/yes/eliza/quine end-to-end in `--allc` mode: no
 * serve process, no per-step stdin/stdout tensor marshalling, no Python anywhere.
 *
 * BUILD (via selfhost/_agent_allc_gen.py -> allc_gen.h, then _agent_allc_build.sh):
 *   python -m c4_min.selfhost._agent_allc_gen --mode echo --out allc_gen.h
 *   gcc -O3 -march=native -DUSE_PTHREADS -pthread -static -I. \
 *       -DMODEL_BLOB_PATH=/path/blockstack.nblbin -o prog onnx_runtime_nibble_allc.c -lm
 *   ./prog --allc              # runs the WHOLE embedded VM; PRTF -> stdout
 *
 * MEASURED (this machine, 242 blocks, 23 heads, D=1725, pthreads):
 *   - echo prints 'hello\n' BYTE-EXACT; yes prints 'y\n'x8 BYTE-EXACT; both all-C.
 *   - Removing Python collapses the per-step PLUMBING: the Python hybrid spent
 *     ~286 ms/step OFF the forward (overlay ~227ms + torch embed ~13ms + decode
 *     ~46ms + [1,S,D] pipe marshal at S=391).  The all-C binary does that same
 *     work in ~1.7 ms/step (overlay ~1.7ms + decode ~0.01ms) — a ~170x cut.
 *   - Per-step is now DOMINATED by the block-stack forward (attention): a clean
 *     step0 (S=31) forward ~218 ms; attention ops grow O(tail*S) (414M@S31 ->
 *     9.7B@S391), while Q/K/V/O proj ~21K and FFN ~5M ops per step (attention is
 *     ~2000x the FFN+proj work).  Plumbing is no longer the bottleneck.
 *   - BOUNDED-KV (ALLC_BOUNDED_W=W): windows the LOCAL (ingest-recency slope 6.0/
 *     0.5) heads to the last W keys — BYTE-EXACT vs the exact path (echo/yes
 *     identical) since a local head's softmax1 weight is exactly 0 beyond W.  Cuts
 *     attention ops 9.7B -> 2.7B (W=64) / 2.0B (W=32) at S=391 (3.6x/4.9x).  The
 *     GLOBAL (memory) heads stay full-history (exact); bounding them to store rows
 *     is the remaining lever (unimplemented — needs the store-row set, byte-exact
 *     risk).
 *   - PER-OP BLOCK-SKIP (ALLC_BLOCK_SKIP=1 / --block-skip, #738): each step runs
 *     ONLY the decoded opcode's LIVE blocks (run_full_skip) and passes the residual
 *     straight through the skipped ones — BYTE-EXACT for the decode (verified: echo/
 *     yes/cat identical base==skip; the torch harness _step_block_skip_verify proves
 *     the same live-set byte-exact over the whole opcode corpus).  echo's ops (IMM=7/
 *     PRTF=5/HALT=5 live of 242) drop the block-applies ~41x (3146 -> 77); MEASURED
 *     wall echo 5.5s -> 0.65s (~8.5x), yes 1.9s -> 0.20s (~9.7x) — both SUB-SECOND.
 *     The live table (G_LIVE_MASK/G_SKIP_HAVE/G_SKIP_NB in allc_gen.h) is a per-op
 *     bitmask emitted by _agent_allc_gen.build_live_masks (from step_block_skip.
 *     build_live_index).  DIV/MOD are operand-dependent -> the whole 188-block divmod
 *     span (~1.3x).  Any op WITHOUT a live set (MALC/FREE/MSET/MCMP) runs all blocks
 *     (byte-safe fallback).  COST model: run_full_skip is a fresh P=0 recompute over
 *     the live blocks (attention O(live*S^2), no cross-step cache — the cache is NOT
 *     skip-safe since the live set varies per step, changing every prefix row's
 *     per-block input).  It is a large net win for SHORT/interactive streams (echo/
 *     yes) but its O(S^2) attention loses to the incremental O(S) cache at large S;
 *     so the driver AUTO-CROSSOVERS to run_incremental once live_count*S >= NB*31
 *     and LATCHES there (S grows monotonically -> one cache rebuild, no thrash), so
 *     block-skip never regresses a long stream (cat 89 steps S->2851: 205s base vs
 *     209s skip, break-even, byte-exact).  Composes with BOUNDED-KV.  Default OFF
 *     (full forward, run_incremental — byte-identical to before this feature).
 *   - FREE-DRIVEN EVICTION (ALLC_EVICT=1 / --evict / --evict-w=W): each step runs the
 *     block stack over a COMPACTED KEEP-SET — {BOS} + {every LIVE §Memory store row:
 *     the LATEST store to each address, superseded/freed rows dropped} + {the last W
 *     recency rows the LOCAL/ingest heads key on} — instead of the growing full stream,
 *     with ABSOLUTE-position ALiBi (run_full_evict + ac_build_residual_evict).  This
 *     keeps the whole LIVE HEAP (LI/LC still content-address ANY written address) while
 *     the KV row-count + RSS stay FLAT at ~heap+W over ARBITRARY-LENGTH I/O (no OOM).
 *     BYTE-EXACT (proven max|delta|=0 over all 242 blocks & 23 heads): the 20 LOCAL
 *     heads (slope 6.0) have softmax1 weight EXACTLY 0 beyond ~one frame; the 3 GLOBAL
 *     memory-CAM heads (slope ~0.004) gate their VALUE to EXACTLY 0 on every non-store
 *     row AND the softmax1 denom is unchanged by dropping them (their score is at/below
 *     the softmax1 sink).  MEASURED: cat streams 4080 bytes (full stream S=133,231
 *     tokens) at a FLAT peak of 256 compacted rows / ~1010 MB RSS — same as at 1020
 *     bytes (S=33,751); base==evict byte-exact on strict cat (Hi) + seeded-string
 *     read-back (KeepHeap!123, 177 steps, byte C read ~3000 tokens past the window,
 *     peak 77 rows).  Composes with --block-skip + --io-burst.  Default OFF (full
 *     stream, byte-identical to before this feature).  ARBITRARY-LENGTH cat via the
 *     `catchunk` program (loop READ(chunk<=192)->burst-print->supersede->repeat);
 *     HONEST 8-bit limit: the burst PRTF is a NUL-terminated C-string walk so a literal
 *     NUL byte can't pass THIS path, and a single §Memory string is capped at the 8-bit
 *     window (~200 B) — but arbitrary TOTAL via chunking IS in scope (16/32-bit = future).
 *   - The QUINE runs fully in C (SI/PSH/LI/BZ + seed_mem KV all ported; the step
 *     trace's memory-load path is correct through the loop setup) but its end-to-end
 *     byte-exactness vs the reference is NOT yet confirmed (the ground-truth hybrid
 *     comparison was blocked on machine contention).
 *   - DEAD-BLOCK-FUSION (default ON; ALLC_NO_ATTN_FUSION=1 to disable): only 3 of the
 *     242 blocks carry a LIVE attention head (audited from the baked weights, ALLC_
 *     FUSION_AUDIT=1: b0 ingest Wv/Wo nnz=40, b7 nnz=16, b11 memory-CAM/LEV nnz=32);
 *     the other 239 blocks have an ALL-ZERO W_v OR W_o slice, so their attention output
 *     is mathematically-exact ZERO and the attn residual add is pure passthrough
 *     (xout=xin).  We SKIP the entire attention sublayer (Q/K/V proj + softmax + O proj)
 *     for those 239 blocks and run only the FFN — BYTE-EXACT (verified: fusion ON==OFF
 *     stdout+exit byte-identical on echo/yes/cat/quine, and the kill-switch OFF==the
 *     pre-fusion binary).  This collapses the per-step forward from ~473 ms/step to
 *     ~24 ms/step (~19x): MEASURED echo 6.5s->0.41s (~16x full), 0.28s block-skip
 *     (~23x); yes(y\\n x8) 26s->1.1s (~24x); cat 48s->1.8s (~27x).  Composes with
 *     --block-skip / --evict / --io-burst.  _weight_all_zero() + g_attn_dead[] in
 *     incr_discover(); applied in run_incremental / run_full_skip / run_full_evict.
 *   - DIRECT-CAM (opt-in --direct-cam / ALLC_DIRECT_CAM=1; default OFF): resolve a
 *     memory-LOAD op's AX (LI full-word / LC byte) O(1) from the store-log (load addr
 *     = pre-step AX, latest-write-wins ac_mem_top/ac_mem_read) instead of the neural
 *     memory-CAM — the SAME latest-write CAM rule the global heads implement, so it is
 *     BYTE-EXACT (verified: 0 LI/LC steps where store-log != neural decode on cat +
 *     quine; output byte-identical ON vs OFF).  HONEST: in the autoregressive all-C
 *     path this is byte-exactness-proving but NOT a speedup lever — the neural forward
 *     still runs in full (the b0 ingest block + b7/b11 are needed to decode PC/SP), so
 *     the AX override happens AFTER the forward; the timing win is entirely dead-block-
 *     fusion.  Kept as a verified O(1) resolution + a neural/store-log CAM-agreement
 *     canary (ac_dcam_diff == 0 confirms the two CAMs never disagree on these progs).
 *   - OPTIMIZED LIVE-BLOCK ATTENTION (the headline speedup on TOP of dead-block-fusion;
 *     both default ON, each with a kill-switch).  Post-fusion, only 3 blocks run
 *     attention: b0 (20 LOCAL ingest heads, ALiBi slope 6.0) and b7/b11 (3 GLOBAL
 *     memory-CAM heads, slope ~0.004).  #879's fusion skipped the 239 DEAD blocks but
 *     the 3 LIVE blocks still scanned ALL S past keys each step -> O(S) per query,
 *     O(S^2) over a run.  Two byte-exact levers cut that to O(W)/O(#stores+W):
 *       (a) LIVE-BLOCK LOCAL WINDOW (ALLC_NO_LIVE_WIN=1 to disable): b0's LOCAL heads
 *           attend only the last AC_LIVE_LOCAL_WIN=32 keys.  BYTE-EXACT: an ingest head
 *           (slope 6.0) has softmax1 weight EXACTLY 0.0f beyond its recency horizon —
 *           MEASURED max nonzero-weight distance 28 over echo/cat/quine (ALLC_WINPROBE),
 *           so W=32 is a proven-exact margin.  Turns b0's per-query O(S) scan into O(W).
 *       (b) GLOBAL-HEAD KEY-SUBSET / direct-CAM-attn (ALLC_NO_GKEY=1 to disable): the 3
 *           GLOBAL memory-CAM heads scan ONLY the keep-set keys (BOS + every §Memory
 *           store row + last AC_GLOBAL_WIN=60 recency rows) instead of all S — O(#stores
 *           +W) not O(S).  BYTE-EXACT for the SAME reason the eviction path is (V==0 on
 *           every non-store row -> zero numerator; the softmax1 denom delta over the
 *           dropped rows rounds away), scanned in the same ascending-k order + real
 *           distances.  NOTE these GLOBAL heads are a broad soft blend (measured gnz=all
 *           causal keys, top weight ~1/S), NOT a hard 1-hot gather — so a true O(1)
 *           hard-gather of the attention output is NOT byte-exact; the byte-exact O(1)
 *           resolution is the store-log AX override (DIRECT-CAM above).  This key-subset
 *           is the byte-exact attention-level analogue.  g_gkey/ac_build_gkey.
 *     MEASURED (quine, 363 steps, S 1651->12541, all-C incremental path): attention
 *     ops stay FLAT ~13-16M/step ON vs growing 534M->3980M/step OFF (the O(S)->O(W)+
 *     O(#stores) flattening); per-step 284ms->100ms; QUINE end-to-end 182s->36s (~5x),
 *     byte-identical stdout+exit.  Both levers compose with dead-block-fusion / block-
 *     skip / evict / io-burst; each is byte-exact ON vs OFF vs the #879 base on echo/
 *     yes/cat/quine.  The FFN was ALREADY O(nnz)-sparse (coo_project, no dense W_up/
 *     W_gate/W_down); per-op BLOCK-SKIP (below) already skips a skipped block's FFN too.
 *   - OVERHEAD-KILL PASS (this change; all byte-exact, each with a kill-switch; profiled
 *     with ALLC_TIMING=1 [per-phase overlay/fwd/decode] + ALLC_PHASE=1 [attn/ffn split]).
 *     PROFILE found the runtime was OVERHEAD-bound, NOT flop-bound (~162K MACs/step but
 *     24-73ms/step), with the dominant cost being TWO per-step O(S) leaks the forward
 *     didn't even need + serial FFN elementwise:
 *       (1) TAIL-ONLY OVERLAY (the biggest long-run lever): the incremental forward reads
 *           ONLY the tail input rows [P..S-1] (prefix served from the K/V cache), yet the
 *           driver rebuilt+copied the WHOLE [1,S,D] residual every step — pure O(S) waste
 *           that GREW with the stream (MEASURED overlay 30ms/step at S=7651, ~40% of the
 *           step).  ac_build_residual_tail(S,P) builds only [P..S-1] with the IDENTICAL
 *           per-row overlay math (byte-exact for every row the forward reads; the prefix
 *           rows in ac_resid/the input tensor are left unread), and the driver copies only
 *           the tail.  overlay 30ms -> 0.2ms (150x).  incr_prefix_len(S) mirrors the P the
 *           forward uses; block-skip/fresh-cache paths still full-build (byte-safe).
 *       (2) GLOBAL-KEEP-SET from the STORE-LOG (ac_build_gkey): was a strided IS_STORE read
 *           over the full [1,S,D] residual (a ~60MB cache-miss scan/step once the residual
 *           is only tail-built); now derives the store rows directly from the store-log
 *           (row 1+FRAME_LEN*fi+MEM_MARKER_LOCAL) — identical mask, no O(S) strided read.
 *       (3) SPARSE FFN DOWN-RESIDUAL + persistent scratch (ALLC_NO_FFN_SPARSE=1 off): the
 *           down-bias is all-zero every block, so `xout += hh@Wdown` is nonzero only on the
 *           down weight's distinct output columns; add ONLY those (O(#cols) not O(D)) and
 *           zero only them in dn — byte-exact +0.0 no-op elimination.  FFN scratch is a
 *           persistent small buffer (steady-state T~31), one-shot local for the rare large-T
 *           cold recompute (so the giant step-0 buffer never stays resident to THRASH the
 *           steady-state working set out of cache — that thrash cost more than the mallocs
 *           it saved; the T-cap fixed it).  ffn elementwise 82->64ms on echo.
 *       (4) LARGE-T FFN THREADING (pthreads, T>=96 only): the rare full-recompute (quine
 *           step 0, evict/skip fresh passes) forks the row-independent FFN across cores —
 *           byte-exact (disjoint rows, unchanged per-(row,col) order).  step-0 fwd 1191->698ms.
 *           The steady-state tail (T~31) stays serial (fork-join would cost more than it saves).
 *     MEASURED end-to-end (multi-thread): quine 363 steps S->12541  23.3s -> 14.6s (1.6x,
 *     the tail-overlay dominates the long-run win); echo 0.33->0.31s; yes8 0.74->0.67s; a
 *     late step (S=7651) 73ms -> 40ms.  Byte-identical stdout+exit vs the #879+af32b58 base
 *     on echo/yes/cat/quine for EVERY lever ON and OFF (fusion / live-win / gkey / block-skip
 *     / evict / io-burst / direct-cam / ffn-sparse).  The C stays portable (no non-portable
 *     intrinsics; -march=native is only host tuning — ALLC_MARCH=-mtune=generic builds a
 *     byte-identical portable static binary).
 *   - The binary is ~6.4 MB, fully static (`ldd: not a dynamic executable`), zero
 *     python/torch/.so deps — model + embed + program all embedded.
 *
 * ---- (original incremental.c header follows) ----
 *
 * onnx_runtime_nibble_incremental.c  —  SELF-CONTAINED sparse (COO) runtime with
 * a WINDOWED KV-CACHED INCREMENTAL block-stack forward (kills the O(S^2)).
 *
 * This is the self-contained (`.incbin`-embedded model) sparse runtime PLUS an
 * incremental decode path (`--incremental`, serve mode) that makes the per-step
 * wall FLAT in the stream length S instead of growing O(S^2).
 *
 * THE O(S^2) PROBLEM the base sparse runtime has: the corpus driver runs one VM
 * step as `x = embed[stream]; overlay(x); hidden = blocks(x)` and reads only the
 * LAST row `hidden[0,-1]`.  Each step APPENDS a 30-token register frame to the
 * stream, so S grows by 30/step; the base runtime re-runs the WHOLE block stack
 * over ALL S rows every step (attention is O(S^2), FFN O(S)).  ~1.7s@S31 growing
 * to ~45s@S211 — quine (S=1651) / mandelbrot intractable.
 *
 * THE FIX — windowed KV-cached incremental forward (BYTE-EXACT).  The stream is
 * append-only: verified (torch, max|delta|=0 over ALL 242 blocks) that between
 * step N (S_prev rows) and step N+1, rows [0 .. S_prev-2] have BYTE-IDENTICAL
 * block outputs at EVERY block — only the previous query row S_prev-1 changes
 * (the overlay clears its all-ROLE one-hot when it stops being last) and the 30
 * new rows are added.  Blocks are causal, so a stable prefix row's K/V is fixed
 * forever.  Each step therefore recomputes only the TAIL rows [S_prev-1 .. S-1]
 * (31 rows: 1 changed + 30 new), attending over a CACHED prefix K/V (rows
 * 0..S_prev-2) concatenated with the tail's own K/V.  Per-step: attention is
 * O(31 * S) (LINEAR, not S^2); Q/K/V/O projections + FFN are O(31) (CONSTANT in
 * S).  After the step, tail rows [S_prev-1 .. S-2] become the new stable prefix
 * (K/V appended to the cache); the new last row S-1 is recomputed next step.
 * Byte-exact vs the full-recompute at row -1 (max|delta| ~1e-19, decode argmax
 * identical) because the same COO ascending-r matmul + same softmax1 fp order.
 *
 * The incremental forward (incr_forward) is a HAND-WRITTEN port of
 * blogspec_model.Attn.forward + FFN.forward (softmax1 + ALiBi + SwiGLU) that
 * reads the per-block Q/K/V/O + FFN up/gate/down COO weights and the alibi-slope
 * / scale / bias constants straight out of the loaded graph (the block stack is a
 * perfectly regular 242 x 69-node pattern, so the weight tensor ids are DERIVED
 * from the node structure at load, not hardcoded).
 *
 * SIMD: the hot dense loops (COO scatter accumulation, softmax1 exp/reduce,
 * attn@V, FFN gate/up/down) are written so `gcc -O3 -march=native` auto-
 * vectorises them; the fp ACCUMULATION ORDER is preserved (no -ffast-math
 * reassociation) so byte-exactness holds.
 *
 * MULTICORE: the 242-block loop is sequential (data dependency), but WITHIN a
 * block the 23 attention heads and the projection/FFN row loops are parallelised
 * over cores.  Two mechanisms, either static-linkable to keep ONE self-contained
 * binary:
 *   - OpenMP (`-fopenmp`): `#pragma omp parallel for` on the head/row loops.  Needs
 *     static libgomp (or a libgomp .so dep); falls back to serial without -fopenmp.
 *   - pthreads (`-DUSE_PTHREADS -pthread`): a tiny fork-join parallel-for over the
 *     23 heads (glibc static pthreads works where static libgomp is absent).  This
 *     is the SELF-CONTAINED multicore path on toolchains without static libgomp.
 * Byte-exactness holds either way: each head writes a disjoint output slice, so the
 * per-head fp reduction order is unchanged by the parallel split.
 *
 * ---- (inherited from the sparse runtime) ----
 * THE DIFFERENCE from the dense runtime (onnx_runtime_nibble.c lines ~202-214):
 * the dense runtime RECONSTRUCTS every COO-sparse weight initializer into a dense
 * buffer and does an O(M*K*N) dense MatMul — the WRONG compute for a ~99.99%
 * sparse model.  Here a COO-sparse RHS weight is KEPT sparse (parallel val/row/col
 * triples, in stored flat-index order) and op_matmul_coo iterates ONLY the nnz
 * nonzeros: for each activation row it scatters nnz terms, so the inner-loop count
 * is O(M*nnz), NOT O(M*K*N).  Non-matmul consumers of a sparse tensor (none in the
 * block-stack graph, but kept general) still read a zero-filled DENSE payload, so
 * the whole 23-op forward is intact and every other op is bit-identical to dense.
 *
 * BYTE-EXACT vs the dense runtime.  The load-bearing subtlety at VM residual
 * magnitudes ~1e29 is fp ACCUMULATION ORDER: adding the same terms in a different
 * order flips low bits and can flip a decode argmax.  The dense MatMul accumulates
 * out[p][q] = sum_r A[p][r]*W[r][q] over r = 0..K-1 IN ORDER.  The COO nonzeros of
 * a row-major weight W[K,N] are stored sorted by flat index f = r*N + q (ascending)
 * — i.e. r-major, q-minor.  A single scatter pass over the COO array in stored
 * order therefore delivers, for each fixed output column q, its contributions in
 * ASCENDING-r order — EXACTLY the dense reduction order.  Skipped zero terms add
 * +0.0 in dense (no-op in fp), so iterating only the nonzeros in stored order is
 * bit-identical to the dense O(K) reduction.  (op_matmul_coo asserts this ordering
 * by scattering in stored COO order, never re-sorting per column.)
 *
 * float parity: native `float`, argmax(byte-head)-identical to torch/ORT — the VM
 * decodes its state by argmax over the byte head.  Same convention as the dense
 * runtime.
 *
 * Build:   gcc -O2 -static -lm -o rt_sparse onnx_runtime_nibble_sparse.c
 * Run:     ./rt_sparse model.nblbin input [--dump-argmax|--dump-logits]
 *                                          [--residual-in [--serve]]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>          /* write(1,...) / read(0,...) for I/O */
#include <time.h>            /* per-step timing (ALLC_TIMING) */
#include "allc_gen.h"        /* embed table + layout dims + program (generated) */

static double _now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_PTHREADS
#include <pthread.h>
#endif

/* ---- op enum (must match onnx_to_c4bin.OP) ---- */
#define OP_MATMUL 0
#define OP_ADD 1
#define OP_SUB 2
#define OP_MUL 3
#define OP_DIV 4
#define OP_GATHER 5
#define OP_RESHAPE 6
#define OP_TRANSPOSE 7
#define OP_CONCAT 8
#define OP_UNSQUEEZE 9
#define OP_SHAPE 10
#define OP_CAST 11
#define OP_EXP 12
#define OP_NEG 13
#define OP_ABS 14
#define OP_RANGE 15
#define OP_REDUCEMAX 16
#define OP_REDUCESUM 17
#define OP_CLIP 18
#define OP_TRILU 19
#define OP_CONSTANTOFSHAPE 20
#define OP_SIGMOID 21
#define OP_IDENTITY 22

/* ---- attr-key enum (must match onnx_to_c4bin.ATTR) ---- */
#define A_PERM 0
#define A_AXIS 1
#define A_TO 2
#define A_UPPER 3
#define A_KEEPDIMS 4
#define A_AXES 5
#define A_ALLOWZERO 6
#define A_COS_FILL_BITS 7

#define DT_FLOAT 0
#define DT_INT64 1

/* Sized for the ONE UNIFIED full-op VM graph (dim~1633, 305 blocks): the full op
 * set (incl. the ~300 DIV/MOD blocks) yields ~33k nodes / ~34k distinct tensors.
 * The old lean/compact graph (dim=1564, 42 blocks: 4461 tensors / 3186 nodes) and
 * the tiny proof model (dim=104, 2 blocks) both fit far inside — one runtime for
 * all.  These tables are BSS globals (~20 MB), never on the stack. */
#define MAX_TENSORS 65536
#define MAX_NODES 65536
#define MAX_IO 8
#define MAX_RANK 8
#define MAX_ATTRS 4

/* A COO-sparse weight's zero-filled DENSE payload is reconstructed ONLY when its
 * numel is <= this (biases a non-matmul op reads).  Larger weights are MatMul-RHS
 * only (directly or via Identity), so their dense buffer is never read — skip it. */
#define COO_DENSE_MAX 1048576

/* ---- tensor table (Structure-of-Arrays, C4-subset friendly) ---- */
int   t_dtype[MAX_TENSORS];       /* 0 float, 1 int64 */
int   t_rank[MAX_TENSORS];
int   t_dims[MAX_TENSORS][MAX_RANK];
int   t_size[MAX_TENSORS];        /* element count */
float *tf[MAX_TENSORS];           /* float payload (dtype 0) */
long  *ti[MAX_TENSORS];           /* int64 payload (dtype 1) */
int   t_is_init[MAX_TENSORS];
char  t_name[MAX_TENSORS][80];
int   n_tensors;
int   input_tid;
int   output_tid;

/* ---- SPARSE (COO) payload — the load-bearing difference from the dense runtime.
 * A 2-D weight that arrives COO-sparse (is_init==2) is kept SPARSE: its nonzeros
 * are parallel (val, row, col) triples in STORED flat-index order (row-major:
 * ascending r, then ascending q within a row) and t_is_coo[tid]=1.  op_matmul
 * routes such a RHS to the O(M*nnz) sparse path; every other consumer reads the
 * zero-filled dense payload tf[tid] (also reconstructed at load).  Stored order
 * is preserved so the per-column reduction order matches dense (byte-exact). */
int   t_is_coo[MAX_TENSORS];      /* 1 => weight stored as COO (sparse matmul RHS) */
int   t_nnz[MAX_TENSORS];         /* number of nonzeros */
int  *t_coo_row[MAX_TENSORS];     /* nnz row indices (dim -2), stored order */
int  *t_coo_col[MAX_TENSORS];     /* nnz col indices (dim -1), stored order */
float *t_coo_val[MAX_TENSORS];    /* nnz float values, stored order */

/* diagnostic counters: the ACTUAL matmul inner-iteration counts executed this run,
 * so a measurement can print the realised sparse work (SPARSE_ITERS) vs the dense
 * work it replaced (DENSE_EQUIV_ITERS = M*K*N over the sparse-RHS matmuls). */
long  SPARSE_ITERS;
long  DENSE_EQUIV_ITERS;

/* ---- node table ---- */
int n_op[MAX_NODES];
int n_nin[MAX_NODES];
int n_in[MAX_NODES][MAX_IO];
int n_nout[MAX_NODES];
int n_out[MAX_NODES][MAX_IO];
int n_nattr[MAX_NODES];
int n_akey[MAX_NODES][MAX_ATTRS];
int n_anv[MAX_NODES][MAX_ATTRS];
int n_aval[MAX_NODES][MAX_ATTRS][MAX_RANK];
int n_nodes;

/* ============ binary reader — MEMORY cursor over the EMBEDDED model blob ========
 * This self-contained build has the .nblbin bytes linked in via `.incbin`
 * (see the _model_blob asm at the bottom); the loader reads from that byte cursor,
 * so the binary needs NO external model file. */
extern const unsigned char _model_blob[];
extern const unsigned char _model_blob_end[];
const unsigned char *g_blob;
long g_pos;

int rd_i32(void) {
    const unsigned char *b;
    b = g_blob + g_pos;
    g_pos += 4;
    return (int)(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
}
/* read a stored int64 (two int32, little-endian lo/hi) as a C long */
long rd_i64(void) {
    unsigned int lo;
    int hi;
    lo = (unsigned int)rd_i32();
    hi = rd_i32();
    return ((long)hi << 32) | (long)lo;
}
float rd_f32(void) {
    int bits;
    float x;
    bits = rd_i32();
    memcpy(&x, &bits, 4);
    return x;
}
/* copy `n` raw bytes from the cursor into dst, advancing the cursor. */
void rd_bytes(void *dst, int n) {
    memcpy(dst, g_blob + g_pos, n);
    g_pos += n;
}

/* ============ tensor helpers ============ */
int prod(int rank, int *dims) {
    int p; int i;
    p = 1; i = 0;
    while (i < rank) { p = p * dims[i]; i = i + 1; }
    return p;
}

/* allocate a runtime tensor's storage for the given shape + dtype.  A dense
 * (re)allocation clears any COO-alias status the tensor id previously held. */
void alloc_tensor(int tid, int dtype, int rank, int *dims) {
    int sz; int i;
    t_is_coo[tid] = 0;                 /* dense alloc -> not a COO alias */
    t_dtype[tid] = dtype;
    t_rank[tid] = rank;
    i = 0;
    while (i < rank) { t_dims[tid][i] = dims[i]; i = i + 1; }
    sz = prod(rank, dims);
    t_size[tid] = sz;
    if (tf[tid]) { free(tf[tid]); tf[tid] = 0; }
    if (ti[tid]) { free(ti[tid]); ti[tid] = 0; }
    if (dtype == DT_FLOAT) tf[tid] = malloc(sz * sizeof(float));
    else ti[tid] = malloc(sz * sizeof(long));
}

/* strides for row-major layout */
void strides_of(int rank, int *dims, int *st) {
    int i; int s;
    s = 1; i = rank - 1;
    while (i >= 0) { st[i] = s; s = s * dims[i]; i = i - 1; }
}

/* ============ load the EMBEDDED .nblbin (memory cursor, no file) ============ */
void load(char *path) {
    int magic; int i; int j; int k;
    int nl; int isi; int dt; int rank; int dims[MAX_RANK]; int ne;
    int op; int nin; int nout; int nattr; int key; int nv;

    (void)path;                       /* embedded blob — path ignored */
    g_blob = _model_blob;
    g_pos = 0;
    magic = rd_i32();
    if (magic != 0x314C424E) { printf("bad magic %x\n", magic); exit(1); }
    n_tensors = rd_i32();
    n_nodes = rd_i32();
    input_tid = rd_i32();
    output_tid = rd_i32();

    i = 0;
    while (i < n_tensors) {
        nl = rd_i32();
        rd_bytes(t_name[i], nl);
        t_name[i][nl] = 0;
        isi = rd_i32();
        t_is_init[i] = isi;
        tf[i] = 0; ti[i] = 0;
        t_is_coo[i] = 0; t_nnz[i] = 0;
        t_coo_row[i] = 0; t_coo_col[i] = 0; t_coo_val[i] = 0;
        dt = rd_i32();
        rank = rd_i32();
        j = 0;
        while (j < rank) { dims[j] = rd_i32(); j = j + 1; }
        ne = rd_i32();
        if (isi == 1) {
            /* dense initializer: ne stored values */
            alloc_tensor(i, dt, rank, dims);
            if (dt == DT_FLOAT) {
                j = 0; while (j < ne) { tf[i][j] = rd_f32(); j = j + 1; }
            } else {
                j = 0; while (j < ne) { ti[i][j] = rd_i64(); j = j + 1; }
            }
        } else if (isi == 2) {
            /* COO-sparse float initializer: ne = DENSE numel, then nnz, then
             * nnz int64 flat-indices (ASCENDING = row-major stored order) + nnz
             * float values.  KEEP IT SPARSE: store parallel (row,col,val) triples
             * in the SAME stored order (so op_matmul_coo's per-column reduction
             * order == dense; op_identity propagates the COO alias to MatMul RHS).
             *
             * The dense payload tf[i] is reconstructed ONLY for SMALL tensors
             * (numel <= COO_DENSE_MAX): those are biases (e.g. blocks.34.ffn.b_gate,
             * [4320]) that a non-matmul op (Add) consumes and therefore need a dense
             * read.  Large weight matrices are consumed ONLY as MatMul RHS (directly
             * or through Identity), so their dense payload is never read — skipping
             * it saves the ~12 GB / ~3 B-float dense reconstruction and its zero-fill
             * time.  If a large sparse tensor WERE read densely, tf[i]==0 would be a
             * loud NULL-deref, not a silent wrong answer. */
            int nnz; long *s_idx; int lastdim; long fidx; int want_dense;
            t_dtype[i] = DT_FLOAT; t_rank[i] = rank;
            { int dd; dd = 0; while (dd < rank) { t_dims[i][dd] = dims[dd]; dd = dd + 1; } }
            t_size[i] = ne;
            want_dense = (ne <= COO_DENSE_MAX) ? 1 : 0;
            if (want_dense) {
                tf[i] = malloc(ne * sizeof(float));
                j = 0; while (j < ne) { tf[i][j] = 0.0f; j = j + 1; }
            }
            lastdim = dims[rank - 1];             /* N (col stride) */
            nnz = rd_i32();
            s_idx = malloc(nnz * sizeof(long));
            j = 0;
            while (j < nnz) { s_idx[j] = rd_i64(); j = j + 1; }
            t_is_coo[i] = 1;
            t_nnz[i] = nnz;
            t_coo_row[i] = malloc(nnz * sizeof(int));
            t_coo_col[i] = malloc(nnz * sizeof(int));
            t_coo_val[i] = malloc(nnz * sizeof(float));
            j = 0;
            while (j < nnz) {
                fidx = s_idx[j];
                t_coo_row[i][j] = (int)(fidx / lastdim);
                t_coo_col[i][j] = (int)(fidx - (fidx / lastdim) * lastdim);
                j = j + 1;
            }
            j = 0;
            while (j < nnz) {
                float v; v = rd_f32();
                t_coo_val[i][j] = v;
                if (want_dense) tf[i][s_idx[j]] = v;   /* scatter into dense payload */
                j = j + 1;
            }
            free(s_idx);
        } else {
            t_dtype[i] = -1; t_rank[i] = 0; t_size[i] = 0;
        }
        i = i + 1;
    }

    i = 0;
    while (i < n_nodes) {
        op = rd_i32();
        n_op[i] = op;
        nin = rd_i32();
        n_nin[i] = nin;
        j = 0; while (j < nin) { n_in[i][j] = rd_i32(); j = j + 1; }
        nout = rd_i32();
        n_nout[i] = nout;
        j = 0; while (j < nout) { n_out[i][j] = rd_i32(); j = j + 1; }
        nattr = rd_i32();
        n_nattr[i] = nattr;
        j = 0;
        while (j < nattr) {
            key = rd_i32();
            nv = rd_i32();
            n_akey[i][j] = key;
            n_anv[i][j] = nv;
            k = 0; while (k < nv) { n_aval[i][j][k] = rd_i32(); k = k + 1; }
            j = j + 1;
        }
        i = i + 1;
    }
    /* diagnostics on stderr so --residual-in / --dump-logits keep a clean stdout */
    fprintf(stderr, "loaded EMBEDDED model: %ld bytes, %d tensors, %d nodes "
            "(in=%d out=%d)\n", (long)(_model_blob_end - _model_blob),
            n_tensors, n_nodes, input_tid, output_tid);
}

/* find an attribute value on node `nd`, key `key`; return value index 0 or dflt */
int attr1(int nd, int key, int dflt) {
    int j;
    j = 0;
    while (j < n_nattr[nd]) {
        if (n_akey[nd][j] == key) return n_aval[nd][j][0];
        j = j + 1;
    }
    return dflt;
}
/* count of values for an attr key (0 if absent) */
int attr_nv(int nd, int key) {
    int j;
    j = 0;
    while (j < n_nattr[nd]) {
        if (n_akey[nd][j] == key) return n_anv[nd][j];
        j = j + 1;
    }
    return 0;
}
int attr_val(int nd, int key, int idx) {
    int j;
    j = 0;
    while (j < n_nattr[nd]) {
        if (n_akey[nd][j] == key) return n_aval[nd][j][idx];
        j = j + 1;
    }
    return 0;
}

/* read tensor element t[flat] as float regardless of dtype */
float getf(int tid, int flat) {
    if (t_dtype[tid] == DT_FLOAT) return tf[tid][flat];
    return (float)ti[tid][flat];
}
long geti(int tid, int flat) {
    if (t_dtype[tid] == DT_INT64) return ti[tid][flat];
    return (long)tf[tid][flat];
}

/* broadcasting elementwise: op 0=add 1=sub 2=mul 3=div. writes out (float). */
void ew_broadcast(int out, int a, int b, int kind) {
    int rank; int rdims[MAX_RANK];
    int ar; int br; int i;
    int ast[MAX_RANK]; int bst[MAX_RANK];
    int adims[MAX_RANK]; int bdims[MAX_RANK];
    int idx[MAX_RANK];
    int flat; int sz;
    int ai; int bi; int d;
    float va; float vb; float vr;

    ar = t_rank[a]; br = t_rank[b];
    rank = ar > br ? ar : br;
    /* right-align dims (pad with 1 on the left) */
    i = 0;
    while (i < rank) {
        d = rank - 1 - i;
        adims[d] = (i < ar) ? t_dims[a][ar - 1 - i] : 1;
        bdims[d] = (i < br) ? t_dims[b][br - 1 - i] : 1;
        rdims[d] = adims[d] > bdims[d] ? adims[d] : bdims[d];
        i = i + 1;
    }
    strides_of(rank, adims, ast);
    strides_of(rank, bdims, bst);
    alloc_tensor(out, DT_FLOAT, rank, rdims);
    sz = t_size[out];

    i = 0; while (i < rank) { idx[i] = 0; i = i + 1; }
    flat = 0;
    while (flat < sz) {
        ai = 0; bi = 0; i = 0;
        while (i < rank) {
            ai = ai + (adims[i] == 1 ? 0 : idx[i]) * ast[i];
            bi = bi + (bdims[i] == 1 ? 0 : idx[i]) * bst[i];
            i = i + 1;
        }
        va = getf(a, ai); vb = getf(b, bi);
        if (kind == 0) vr = va + vb;
        else if (kind == 1) vr = va - vb;
        else if (kind == 2) vr = va * vb;
        else vr = va / vb;
        tf[out][flat] = vr;
        /* increment odometer */
        i = rank - 1;
        while (i >= 0) {
            idx[i] = idx[i] + 1;
            if (idx[i] < rdims[i]) { i = -1; }
            else { idx[i] = 0; i = i - 1; }
        }
        flat = flat + 1;
    }
}

/* ===== SPARSE (COO) MatMul: out[...,M,N] = A[...,M,K] @ Wcoo[K,N] ==============
 * The RHS weight W[K,N] is stored COO (val,row,col triples in row-major stored
 * order: ascending flat = r*N+q).  A is a DENSE activation of rank ar, with M =
 * A.dims[-2], K = A.dims[-1]; its leading dims are the batch.  W is 2-D and
 * broadcast across every batch slice.  Output rank = ar, dims = A.dims[:-1] + [N].
 *
 * For each (batch, output-row p) we zero the N-wide output row, then scatter EVERY
 * nonzero in STORED order:  out[p][col] += A[p][row]*val.  Because the COO array
 * is row-major (all q of r=0, then all q of r=1, ...), each out[p][q] receives its
 * contributions in ascending-r order — identical to the dense sum_r loop, so the
 * fp accumulation is BIT-EXACT to op_matmul on the reconstructed-dense weight.
 * O(batch*M*nnz) inner iterations instead of O(batch*M*K*N). */
void op_matmul_coo(int out, int a, int b) {
    int ar; int rank; int M; int K; int N; int nnz;
    int i; int batch; int rdims[MAX_RANK];
    int *row; int *col; float *val;
    int bt; int p; int j; int aoff; int ooff; int base;
    float av;

    ar = t_rank[a];
    M = t_dims[a][ar - 2]; K = t_dims[a][ar - 1];
    N = t_dims[b][t_rank[b] - 1];
    rank = ar;                                   /* W is 2-D, broadcast; out rank=ar */
    i = 0; while (i < rank - 2) { rdims[i] = t_dims[a][i]; i = i + 1; }
    rdims[rank - 2] = M; rdims[rank - 1] = N;
    alloc_tensor(out, DT_FLOAT, rank, rdims);

    batch = 1;
    i = 0; while (i < rank - 2) { batch = batch * rdims[i]; i = i + 1; }

    nnz = t_nnz[b];
    row = t_coo_row[b]; col = t_coo_col[b]; val = t_coo_val[b];

    /* zero the whole output first (dense sets each out[p][q] once; here we scatter
     * additively, so pre-zeroing gives the identical starting value 0.0f). */
    i = 0; while (i < batch * M * N) { tf[out][i] = 0.0f; i = i + 1; }

    bt = 0;
    while (bt < batch) {
        aoff = bt * M * K;
        ooff = bt * M * N;
        p = 0;
        while (p < M) {
            base = ooff + p * N;
            j = 0;
            while (j < nnz) {
                av = tf[a][aoff + p * K + row[j]];
                tf[out][base + col[j]] = tf[out][base + col[j]] + av * val[j];
                j = j + 1;
            }
            p = p + 1;
        }
        bt = bt + 1;
    }
    SPARSE_ITERS = SPARSE_ITERS + (long)batch * (long)M * (long)nnz;
    DENSE_EQUIV_ITERS = DENSE_EQUIV_ITERS + (long)batch * (long)M * (long)K * (long)N;
}

/* generic N-D MatMul: batched over leading dims, contract last of A with 2nd-last of B.
 * If the RHS is a COO-sparse weight, dispatch to the O(nnz) sparse path (byte-exact). */
void op_matmul(int out, int a, int b) {
    if (t_is_coo[b]) { op_matmul_coo(out, a, b); return; }
    int ar; int br; int rank;
    int M; int K; int N;
    int i; int batch;
    int rdims[MAX_RANK];
    int abatch; int bbatch;
    int adims2; int bdims2;
    int p; int q; int r;
    int aoff; int boff; int ooff;
    float acc;

    ar = t_rank[a]; br = t_rank[b];
    M = t_dims[a][ar - 2]; K = t_dims[a][ar - 1];
    N = t_dims[b][br - 1];
    /* output rank = max(ar,br); leading batch dims come from the higher-rank
       operand (nibble graph only ever broadcasts a 2-D weight against 4-D acts
       or matches batch dims exactly). */
    rank = ar > br ? ar : br;
    i = 0;
    while (i < rank - 2) {
        int ad; int bd;
        int da; int db;
        da = rank - ar; db = rank - br;
        ad = (i >= da) ? t_dims[a][i - da] : 1;
        bd = (i >= db) ? t_dims[b][i - db] : 1;
        rdims[i] = ad > bd ? ad : bd;
        i = i + 1;
    }
    rdims[rank - 2] = M; rdims[rank - 1] = N;
    alloc_tensor(out, DT_FLOAT, rank, rdims);

    batch = 1;
    i = 0; while (i < rank - 2) { batch = batch * rdims[i]; i = i + 1; }
    adims2 = M * K; bdims2 = K * N;
    /* does A / B carry the batch dim? (broadcast a 2-D operand across batch) */
    abatch = (ar == rank) ? 1 : 0;
    bbatch = (br == rank) ? 1 : 0;

    i = 0;
    while (i < batch) {
        aoff = abatch ? i * adims2 : 0;
        boff = bbatch ? i * bdims2 : 0;
        ooff = i * M * N;
        p = 0;
        while (p < M) {
            q = 0;
            while (q < N) {
                acc = 0.0f;
                r = 0;
                while (r < K) {
                    acc = acc + tf[a][aoff + p * K + r] * tf[b][boff + r * N + q];
                    r = r + 1;
                }
                tf[out][ooff + p * N + q] = acc;
                q = q + 1;
            }
            p = p + 1;
        }
        i = i + 1;
    }
}

/* Gather: out = take(data, indices, axis) */
void op_gather(int out, int data, int ind, int axis) {
    int dr; int ir; int i; int j;
    int outer; int inner; int axlen; int nidx;
    int rdims[MAX_RANK]; int rank;
    int o; int a; int in_i; int base; int src;

    dr = t_rank[data]; ir = t_rank[ind];
    axlen = t_dims[data][axis];
    outer = 1; i = 0; while (i < axis) { outer = outer * t_dims[data][i]; i = i + 1; }
    inner = 1; i = axis + 1; while (i < dr) { inner = inner * t_dims[data][i]; i = i + 1; }
    nidx = t_size[ind];

    /* output shape = data[:axis] + ind.shape + data[axis+1:] */
    rank = 0;
    i = 0; while (i < axis) { rdims[rank] = t_dims[data][i]; rank = rank + 1; i = i + 1; }
    i = 0; while (i < ir) { rdims[rank] = t_dims[ind][i]; rank = rank + 1; i = i + 1; }
    i = axis + 1; while (i < dr) { rdims[rank] = t_dims[data][i]; rank = rank + 1; i = i + 1; }
    if (rank == 0) { rdims[0] = 1; rank = 1; }
    alloc_tensor(out, t_dtype[data], rank, rdims);

    o = 0;
    while (o < outer) {
        j = 0;
        while (j < nidx) {
            in_i = (int)geti(ind, j);
            if (in_i < 0) in_i = in_i + axlen;
            base = (o * nidx + j) * inner;
            src = (o * axlen + in_i) * inner;
            a = 0;
            while (a < inner) {
                if (t_dtype[data] == DT_FLOAT) tf[out][base + a] = tf[data][src + a];
                else ti[out][base + a] = ti[data][src + a];
                a = a + 1;
            }
            j = j + 1;
        }
        o = o + 1;
    }
}

/* Reshape: copy data, set shape from int64 shape-tensor (supports one -1) */
void op_reshape(int out, int in, int shp) {
    int rank; int i; int sz; int known; int neg;
    int rdims[MAX_RANK]; long v;
    rank = t_size[shp];
    known = 1; neg = -1;
    i = 0;
    while (i < rank) {
        v = geti(shp, i);
        if (v == -1) { neg = i; rdims[i] = 1; }
        else { rdims[i] = (int)v; known = known * (int)v; }
        i = i + 1;
    }
    if (neg >= 0) rdims[neg] = t_size[in] / known;
    sz = t_size[in];
    alloc_tensor(out, t_dtype[in], rank, rdims);
    i = 0;
    while (i < sz) {
        if (t_dtype[in] == DT_FLOAT) tf[out][i] = tf[in][i];
        else ti[out][i] = ti[in][i];
        i = i + 1;
    }
}

/* Transpose with an explicit perm */
void op_transpose(int out, int in, int nd, int *perm) {
    int rank; int i; int sz;
    int idims[MAX_RANK]; int odims[MAX_RANK];
    int ist[MAX_RANK]; int ost[MAX_RANK];
    int idx[MAX_RANK]; int flat; int src;
    rank = t_rank[in];
    i = 0; while (i < rank) { idims[i] = t_dims[in][i]; i = i + 1; }
    i = 0; while (i < rank) { odims[i] = idims[perm[i]]; i = i + 1; }
    strides_of(rank, idims, ist);
    alloc_tensor(out, t_dtype[in], rank, odims);
    strides_of(rank, odims, ost);
    sz = t_size[in];
    i = 0; while (i < rank) { idx[i] = 0; i = i + 1; }
    flat = 0;
    while (flat < sz) {
        src = 0; i = 0;
        while (i < rank) { src = src + idx[i] * ist[perm[i]]; i = i + 1; }
        if (t_dtype[in] == DT_FLOAT) tf[out][flat] = tf[in][src];
        else ti[out][flat] = ti[in][src];
        i = rank - 1;
        while (i >= 0) {
            idx[i] = idx[i] + 1;
            if (idx[i] < odims[i]) { i = -1; }
            else { idx[i] = 0; i = i - 1; }
        }
        flat = flat + 1;
    }
}

/* Concat along an axis (all int64 here — shape plumbing) */
void op_concat(int nd, int out, int axis) {
    int nin; int i; int j; int rank;
    int rdims[MAX_RANK]; int total; int dt;
    int outer; int inner; int k; int o; int a; int off; int cin; int seg;
    nin = n_nin[nd];
    dt = t_dtype[n_in[nd][0]];
    rank = t_rank[n_in[nd][0]];
    if (rank == 0) { rank = 1; }
    /* sum the axis dim */
    total = 0;
    i = 0;
    while (i < nin) {
        int r; r = t_rank[n_in[nd][i]];
        total = total + (r == 0 ? 1 : t_dims[n_in[nd][i]][axis]);
        i = i + 1;
    }
    i = 0; while (i < rank) { rdims[i] = t_dims[n_in[nd][0]][i]; i = i + 1; }
    if (t_rank[n_in[nd][0]] == 0) rdims[0] = 1;
    rdims[axis] = total;
    alloc_tensor(out, dt, rank, rdims);
    outer = 1; i = 0; while (i < axis) { outer = outer * rdims[i]; i = i + 1; }
    inner = 1; i = axis + 1; while (i < rank) { inner = inner * rdims[i]; i = i + 1; }
    o = 0;
    while (o < outer) {
        off = 0;
        i = 0;
        while (i < nin) {
            cin = n_in[nd][i];
            seg = (t_rank[cin] == 0) ? 1 : t_dims[cin][axis];
            k = 0;
            while (k < seg) {
                a = 0;
                while (a < inner) {
                    int dstf; int srcf;
                    dstf = (o * total + off + k) * inner + a;
                    srcf = (o * seg + k) * inner + a;
                    if (dt == DT_FLOAT) tf[out][dstf] = tf[cin][srcf];
                    else ti[out][dstf] = ti[cin][srcf];
                    a = a + 1;
                }
                k = k + 1;
            }
            off = off + seg;
            i = i + 1;
        }
        o = o + 1;
    }
}

/* Unsqueeze: insert size-1 dims at the given axes (data copied verbatim) */
void op_unsqueeze(int out, int in, int axtid) {
    int naxes; int i; int rank; int newrank;
    int axes[MAX_RANK]; int rdims[MAX_RANK]; int sz; int mark[MAX_RANK];
    int p; int a;
    naxes = t_size[axtid];
    rank = t_rank[in];
    newrank = rank + naxes;
    i = 0; while (i < naxes) {
        a = (int)geti(axtid, i);
        if (a < 0) a = a + newrank;
        axes[i] = a; i = i + 1;
    }
    i = 0; while (i < newrank) { mark[i] = 0; i = i + 1; }
    i = 0; while (i < naxes) { mark[axes[i]] = 1; i = i + 1; }
    p = 0;
    i = 0;
    while (i < newrank) {
        if (mark[i]) rdims[i] = 1;
        else { rdims[i] = (rank == 0) ? 1 : t_dims[in][p]; p = p + 1; }
        i = i + 1;
    }
    sz = (rank == 0) ? 1 : t_size[in];
    alloc_tensor(out, t_dtype[in], newrank, rdims);
    i = 0;
    while (i < sz) {
        if (t_dtype[in] == DT_FLOAT) tf[out][i] = (rank == 0) ? getf(in, 0) : tf[in][i];
        else ti[out][i] = (rank == 0) ? geti(in, 0) : ti[in][i];
        i = i + 1;
    }
}

void op_unary(int out, int in, int kind);   /* fwd decl (defined below) */

/* Identity that PRESERVES COO sparsity.  ONNX often routes a weight through an
 * Identity node before the MatMul (onnx::MatMul_* -> Identity -> MatMul).  If the
 * input is a COO-sparse weight, make the output a COO ALIAS (share the same
 * row/col/val arrays + dims + nnz, is_coo=1, no dense buffer) so the downstream
 * MatMul reads it via the O(nnz) sparse path instead of a reconstructed-dense
 * O(K*N) copy.  Aliasing is safe: COO arrays are never freed mid-run, and these
 * Identity outputs are consumed only as MatMul RHS.  For a non-COO input this
 * falls back to a plain dense copy (op_unary identity). */
void op_identity(int out, int in) {
    if (t_is_coo[in]) {
        int i;
        t_dtype[out] = DT_FLOAT;
        t_rank[out] = t_rank[in];
        i = 0; while (i < t_rank[in]) { t_dims[out][i] = t_dims[in][i]; i = i + 1; }
        t_size[out] = t_size[in];
        if (tf[out]) { free(tf[out]); tf[out] = 0; }
        if (ti[out]) { free(ti[out]); ti[out] = 0; }
        t_is_coo[out] = 1;
        t_nnz[out] = t_nnz[in];
        t_coo_row[out] = t_coo_row[in];      /* share (never freed) */
        t_coo_col[out] = t_coo_col[in];
        t_coo_val[out] = t_coo_val[in];
        return;
    }
    op_unary(out, in, 4);
}

/* Shape: out (int64) = the input's dims */
void op_shape(int out, int in) {
    int rank; int i; int d[MAX_RANK];
    rank = t_rank[in];
    d[0] = rank;
    alloc_tensor(out, DT_INT64, 1, d);
    i = 0; while (i < rank) { ti[out][i] = (long)t_dims[in][i]; i = i + 1; }
}

/* Cast: to==1 -> float32, to==7 -> int64 (onnx TensorProto codes) */
void op_cast(int out, int in, int to) {
    int i; int sz; int dt;
    sz = t_size[in];
    dt = (to == 1) ? DT_FLOAT : DT_INT64;
    alloc_tensor(out, dt, t_rank[in], t_dims[in]);
    i = 0;
    while (i < sz) {
        if (dt == DT_FLOAT) tf[out][i] = getf(in, i);
        else ti[out][i] = geti(in, i);
        i = i + 1;
    }
}

/* Range: out (int64) = [start, limit) step delta.  scalars are int64 tensors. */
void op_range(int out, int st, int lim, int dl) {
    long s; long l; long d; int cnt; int i; int dim[1];
    s = geti(st, 0); l = geti(lim, 0); d = geti(dl, 0);
    cnt = 0;
    { long v; v = s; while ((d > 0 && v < l) || (d < 0 && v > l)) { cnt = cnt + 1; v = v + d; } }
    dim[0] = cnt;
    alloc_tensor(out, DT_INT64, 1, dim);
    i = 0; { long v; v = s; while (i < cnt) { ti[out][i] = v; v = v + d; i = i + 1; } }
}

/* unary float map: 0 exp, 1 neg, 2 abs, 3 sigmoid, 4 identity */
void op_unary(int out, int in, int kind) {
    int i; int sz; float x;
    sz = t_size[in];
    alloc_tensor(out, DT_FLOAT, t_rank[in], t_dims[in]);
    i = 0;
    while (i < sz) {
        x = getf(in, i);
        if (kind == 0) tf[out][i] = expf(x);
        else if (kind == 1) tf[out][i] = -x;
        else if (kind == 2) tf[out][i] = fabsf(x);
        else if (kind == 3) tf[out][i] = 1.0f / (1.0f + expf(-x));
        else tf[out][i] = x;
        i = i + 1;
    }
}

/* Reduce over the last axis (nibble graph reduces last-axis only), keepdims=1.
   kind 0=max, 1=sum. */
void op_reduce_last(int out, int in, int kind) {
    int rank; int i; int last; int rows; int r; int c;
    int rdims[MAX_RANK]; float acc;
    rank = t_rank[in];
    last = t_dims[in][rank - 1];
    rows = t_size[in] / last;
    i = 0; while (i < rank) { rdims[i] = t_dims[in][i]; i = i + 1; }
    rdims[rank - 1] = 1;
    alloc_tensor(out, DT_FLOAT, rank, rdims);
    r = 0;
    while (r < rows) {
        if (kind == 0) {
            acc = tf[in][r * last];
            c = 1; while (c < last) { if (tf[in][r * last + c] > acc) acc = tf[in][r * last + c]; c = c + 1; }
        } else {
            acc = 0.0f;
            c = 0; while (c < last) { acc = acc + tf[in][r * last + c]; c = c + 1; }
        }
        tf[out][r] = acc;
        r = r + 1;
    }
}

/* Clip: lower bound only (nibble uses min=0.0, no upper).  in2 optional. */
void op_clip(int out, int in, int lotid) {
    int i; int sz; float lo; float x;
    sz = t_size[in];
    lo = (lotid >= 0) ? getf(lotid, 0) : -1e30f;
    alloc_tensor(out, DT_FLOAT, t_rank[in], t_dims[in]);
    i = 0;
    while (i < sz) { x = getf(in, i); tf[out][i] = (x < lo) ? lo : x; i = i + 1; }
}

/* Trilu upper: keep elements where col-row >= k, else 0.  Operates on the last
   two dims of an N-D tensor (here rank 2, the [S,S] mask). */
void op_trilu(int out, int in, int upper, int k) {
    int rank; int rows; int cols; int i; int mat; int nmat; int rr; int cc;
    rank = t_rank[in];
    rows = t_dims[in][rank - 2];
    cols = t_dims[in][rank - 1];
    nmat = t_size[in] / (rows * cols);
    alloc_tensor(out, DT_FLOAT, rank, t_dims[in]);
    mat = 0;
    while (mat < nmat) {
        rr = 0;
        while (rr < rows) {
            cc = 0;
            while (cc < cols) {
                int f; int keep;
                f = mat * rows * cols + rr * cols + cc;
                if (upper) keep = (cc - rr >= k) ? 1 : 0;
                else keep = (cc - rr <= k) ? 1 : 0;
                tf[out][f] = keep ? tf[in][f] : 0.0f;
                cc = cc + 1;
            }
            rr = rr + 1;
        }
        mat = mat + 1;
    }
}

/* ConstantOfShape: fill a tensor of the given int64 shape with a float bit-pattern */
void op_constofshape(int out, int shp, int bits) {
    int rank; int i; int rdims[MAX_RANK]; int sz; float fill;
    rank = t_size[shp];
    i = 0; while (i < rank) { rdims[i] = (int)geti(shp, i); i = i + 1; }
    memcpy(&fill, &bits, 4);
    alloc_tensor(out, DT_FLOAT, rank, rdims);
    sz = t_size[out];
    i = 0; while (i < sz) { tf[out][i] = fill; i = i + 1; }
}

/* ============ execute ============ */
void run() {
    int nd; int op; int o; int i0; int i1; int i2;
    int perm[MAX_RANK]; int i; int np;
    nd = 0;
    while (nd < n_nodes) {
        op = n_op[nd];
        o = n_out[nd][0];
        i0 = (n_nin[nd] > 0) ? n_in[nd][0] : -1;
        i1 = (n_nin[nd] > 1) ? n_in[nd][1] : -1;
        i2 = (n_nin[nd] > 2) ? n_in[nd][2] : -1;

        if (op == OP_MATMUL) op_matmul(o, i0, i1);
        else if (op == OP_ADD) ew_broadcast(o, i0, i1, 0);
        else if (op == OP_SUB) ew_broadcast(o, i0, i1, 1);
        else if (op == OP_MUL) ew_broadcast(o, i0, i1, 2);
        else if (op == OP_DIV) ew_broadcast(o, i0, i1, 3);
        else if (op == OP_GATHER) op_gather(o, i0, i1, attr1(nd, A_AXIS, 0));
        else if (op == OP_RESHAPE) op_reshape(o, i0, i1);
        else if (op == OP_TRANSPOSE) {
            np = attr_nv(nd, A_PERM);
            i = 0; while (i < np) { perm[i] = attr_val(nd, A_PERM, i); i = i + 1; }
            op_transpose(o, i0, np, perm);
        }
        else if (op == OP_CONCAT) op_concat(nd, o, attr1(nd, A_AXIS, 0));
        else if (op == OP_UNSQUEEZE) op_unsqueeze(o, i0, i1);
        else if (op == OP_SHAPE) op_shape(o, i0);
        else if (op == OP_CAST) op_cast(o, i0, attr1(nd, A_TO, 1));
        else if (op == OP_EXP) op_unary(o, i0, 0);
        else if (op == OP_NEG) op_unary(o, i0, 1);
        else if (op == OP_ABS) op_unary(o, i0, 2);
        else if (op == OP_RANGE) op_range(o, i0, i1, i2);
        else if (op == OP_REDUCEMAX) op_reduce_last(o, i0, 0);
        else if (op == OP_REDUCESUM) op_reduce_last(o, i0, 1);
        else if (op == OP_CLIP) op_clip(o, i0, i1);
        else if (op == OP_TRILU) op_trilu(o, i0, attr1(nd, A_UPPER, 1), (int)geti(i1, 0));
        else if (op == OP_CONSTANTOFSHAPE) op_constofshape(o, i0, attr1(nd, A_COS_FILL_BITS, 0));
        else if (op == OP_SIGMOID) op_unary(o, i0, 3);
        else if (op == OP_IDENTITY) op_identity(o, i0);
        else { printf("unknown op %d at node %d\n", op, nd); exit(1); }
        nd = nd + 1;
    }
}

/* =====================================================================
 * WINDOWED KV-CACHED INCREMENTAL BLOCK-STACK FORWARD  (the O(S^2) fix)
 * =====================================================================
 * The block stack is a perfectly regular pattern: 2248 leading IDENTITY
 * nodes (they alias the COO weights onto the MatMul-RHS tensor ids, no
 * compute), then 242 identical 69-node blocks (attn softmax1+ALiBi + SwiGLU
 * FFN).  We discover the per-block weight tensor ids from the FIRST block's
 * node structure, then run a hand-written incremental forward that is
 * byte-exact to the full graph at the last row.  See file header. */

#define NB_MAX 512          /* max blocks (242 today) */
#define BLK_NODES 69         /* nodes per block (regular pattern) */

int   g_incr_ready;          /* 1 once the block layout is discovered */
int   g_nblocks;             /* number of transformer blocks */
int   g_H;                   /* heads */
int   g_HD;                  /* head dim */
int   g_D;                   /* d_model = H*HD */
float g_scale;               /* attention score scale */
int   g_slope_tid;           /* alibi-slope tensor id ([1,H,1,1]) */
/* per-block COO weight tensor ids + FFN bias ids */
int   g_wq[NB_MAX], g_wk[NB_MAX], g_wv[NB_MAX], g_wo[NB_MAX];
int   g_wup[NB_MAX], g_wgate[NB_MAX], g_wdown[NB_MAX];
int   g_bup[NB_MAX], g_bgate[NB_MAX], g_bdown[NB_MAX];
int   g_ffn_hidden[NB_MAX];  /* FFN hidden width PER BLOCK (varies! DIV/MOD megablocks) */
int   g_ffn_hidden_max;      /* max over blocks (for scratch sizing) */

/* ---- SPARSE FFN DOWN-RESIDUAL (byte-exact, default ON; ALLC_NO_FFN_SPARSE=1 off):
 * the FFN residual add is  xout[j] += dn[j] + bdown[j]  over ALL D=g_D columns and
 * ALL 242 blocks (~13M float ops/step) — but dn = hh@Wdown is SPARSE (only the
 * distinct OUTPUT columns of the COO Wdown are ever nonzero; every other column of
 * dn stays exactly 0.0 after the zero-fill) AND bdown is all-zero/absent for every
 * block (audited: down-bias nnz total == 0).  So for such a block the add over the
 * OTHER columns is `xout[j] += 0.0f + 0.0f`, an fp NO-OP.  We add ONLY the distinct
 * down-output columns -> O(#down-cols) not O(D) per block, BYTE-EXACT.  We keep the
 * dn accumulation from 0.0 unchanged (same ascending-r reduction order), then add
 * each column to xout exactly once — bit-identical to the full-D loop.  Only enabled
 * per-block when bdown is provably all-zero (else fall back to the full-D add). */
int   g_ffn_sparse;              /* 1 => apply the sparse down-residual add */
int   g_bdown_zero[NB_MAX];      /* 1 => this block's down bias is all-zero/absent */
int  *g_down_cols[NB_MAX];       /* sorted distinct output columns of Wdown (or NULL) */
int   g_down_ncols[NB_MAX];      /* count of distinct down-output columns */

/* ---- DEAD-BLOCK-FUSION (attention->0): a block's attention SUBLAYER produces NO
 * live output when its VALUE projection W_v is all-zero (=> V==0 => attn out==0)
 * OR its OUTPUT projection W_o is all-zero (=> attn@W_o==0).  Either way the attn
 * residual add is xout=xin (pure passthrough), so we can skip the ENTIRE attention
 * sublayer (Q/K/V proj + softmax + O proj) and run only the FFN — BYTE-EXACT (the
 * skipped ops contribute mathematically-exact zero to the residual).  Only ~4 of
 * 242 blocks carry a LIVE attention head (block-0 ingest + the memory-CAM/stack-pop/
 * LEV heads); the other ~238 have a fully-pruned attention slice.  Computed once at
 * discover time; default ON (byte-neutral), disable with ALLC_NO_ATTN_FUSION=1. */
int   g_attn_dead[NB_MAX];   /* 1 => attention sublayer is a no-op (skip it) */
int   g_attn_dead_n;         /* count of dead-attention blocks (audit) */
int   g_attn_fuse;           /* 1 => apply dead-block-fusion (default ON) */

/* return 1 iff the weight tensor `tid` is all-zero (nnz==0, or every stored value
 * is exactly 0.0f for COO; every dense entry exactly 0.0f otherwise).  A missing
 * tensor (tid<0) counts as all-zero.  This is the mathematically-exact test for
 * "this projection outputs 0 for every input" (the residual passthrough condition). */
static int _weight_all_zero(int tid) {
    if (tid < 0) return 1;
    if (t_is_coo[tid]) {
        int nnz = t_nnz[tid]; int j;
        float *val = t_coo_val[tid];
        if (nnz == 0 || !val) return 1;
        for (j = 0; j < nnz; j = j + 1) if (val[j] != 0.0f) return 0;
        return 1;
    } else {
        const float *W = tf[tid];
        long n = 1; int r;
        if (!W) return 1;
        for (r = 0; r < t_rank[tid]; r = r + 1) n *= t_dims[tid][r];
        { long i; for (i = 0; i < n; i = i + 1) if (W[i] != 0.0f) return 0; }
        return 1;
    }
}

/* ---- BOUNDED-KV (opt-in via ALLC_BOUNDED_W): a head is LOCAL if its ALiBi slope
 * matches an ingest recency slope (6.0 or 0.5) -> its softmax1 weight is EXACTLY
 * 0 beyond a small recency window, so scanning only the last-W keys is byte-exact.
 * GLOBAL heads (any other slope) stay full-history (exact).  We compute a per-head
 * LOCAL flag once at discover time (slopes are shared across blocks). */
#define AC_ING_SLOPE0 6.0f
#define AC_ING_SLOPE1 0.5f
/* byte-exact local window for the LIVE ingest heads (b0, slope 6.0): MEASURED max
 * nonzero softmax1-weight distance is 28 over echo/cat/quine (ALLC_WINPROBE), so 32
 * keys is a proven-exact margin.  Any key beyond it has exp(score-m)==0.0f exactly. */
#define AC_LIVE_LOCAL_WIN 32
int   g_head_local[512];     /* 1 if head h is a LOCAL (windowed) head */
int   g_bounded_w;           /* window for local heads (0 = bounded-KV OFF) */

/* WINDOW PROBE (ALLC_WINPROBE=1): record, per head, the MAX absolute-distance at
 * which any key received a NONZERO softmax1 weight over the whole run — the true
 * byte-exact local window for the LOCAL heads (diagnostic only, no output effect). */
int   g_winprobe;
int   g_winprobe_max[512];
int   g_winprobe_gnz[512];    /* GLOBAL head: max #keys with nonzero softmax1 weight */
float g_winprobe_gtop[512];   /* GLOBAL head: min top softmax1 weight (1.0 == hard gather) */

/* per-phase work counters (contention-independent profile: attention QK/AV dot
 * ops vs projection+FFN COO nnz scatter ops).  Reset per step by the driver. */
long  g_iter_attn;           /* attention inner (dot + attn@V) fp ops */
long  g_iter_proj;           /* Q/K/V/O projection COO scatter ops */
long  g_iter_ffn;            /* FFN up/gate/down COO scatter ops */
long  g_iter_attn_h[512];    /* per-head attention ops (race-free: each thread own h) */

/* fine-grained per-phase WALL timers (ALLC_PHASE=1): attention sublayer vs FFN body
 * vs per-block FFN scratch alloc.  Diagnostic only; accumulated per step. */
int    g_phase_on;
double g_ph_attn, g_ph_ffn, g_ph_alloc;

/* ---- per-block KV cache (the stable prefix, rows 0..prefix_len-1) ---- */
/* K/V stored [H][cap*HD] (row-major per head: pos-major, hd-minor). */
float *g_Kc[NB_MAX];         /* cached K, per block */
float *g_Vc[NB_MAX];         /* cached V, per block */
int    g_cap[NB_MAX];        /* allocated capacity (in positions) */
int    g_prefix_len;         /* number of cached (stable) prefix positions */
int    g_prev_S;             /* S of the previous step (0 = fresh) */

/* find the 2nd input of a node relative to the first block base */
static int _blk_node_in(int blk_base, int off, int which) {
    int nd = blk_base + off;
    return (n_nin[nd] > which) ? n_in[nd][which] : -1;
}

/* the bias operand of an Add(matmul_out, bias) node: the input that is NOT the
 * output of the preceding MatMul node `mm`.  Returns -1 if no such operand. */
static int _bias_of_add(int addn, int mm) {
    int mmout = (mm >= 0 && n_nout[mm] > 0) ? n_out[mm][0] : -2;
    if (n_nin[addn] < 2) return -1;
    if (n_in[addn][0] != mmout) return n_in[addn][0];
    return n_in[addn][1];
}

/* discover the regular block layout + weight ids.  Returns 1 on success.
 * FIRST runs the prologue IDENTITY nodes: they alias the COO weights onto the
 * MatMul-RHS tensor ids (onnx::MatMul_* -> weight), so after this the per-block
 * weight ids carry their COO row/col/val + dims.  Cheap (pure pointer aliasing,
 * no compute). */
int incr_discover(void) {
    int base; int b;
    /* the first transformer block starts at the first non-IDENTITY node */
    base = 0;
    while (base < n_nodes && n_op[base] == OP_IDENTITY) base = base + 1;
    if (base >= n_nodes) return 0;
    g_nblocks = (n_nodes - base) / BLK_NODES;
    if (g_nblocks <= 0 || g_nblocks > NB_MAX) return 0;
    if ((n_nodes - base) % BLK_NODES != 0) return 0;

    /* run the prologue IDENTITY nodes so the COO aliases are populated */
    { int nd; for (nd = 0; nd < base; nd = nd + 1)
        if (n_op[nd] == OP_IDENTITY) op_identity(n_out[nd][0], n_in[nd][0]); }

    /* per-block weight ids, from the regular node offsets (validated against
     * the graph dump):  Q=off6.in1 K=off18.in1 V=off20.in1 O=off57.in1
     * FFN up=off59.in1 gate=off61.in1 down=off66.in1; biases from the
     * following Add nodes (the non-MatMul-output operand). */
    g_ffn_hidden_max = 0;
    b = 0;
    while (b < g_nblocks) {
        int bb;
        bb = base + b * BLK_NODES;
        g_wq[b] = _blk_node_in(bb, 6, 1);
        g_wk[b] = _blk_node_in(bb, 18, 1);
        g_wv[b] = _blk_node_in(bb, 20, 1);
        g_wo[b] = _blk_node_in(bb, 57, 1);
        g_wup[b] = _blk_node_in(bb, 59, 1);
        g_wgate[b] = _blk_node_in(bb, 61, 1);
        g_wdown[b] = _blk_node_in(bb, 66, 1);
        /* biases: the non-MatMul-output operand of the Add.  May be absent
         * (folded-away zero: is_init==0, size 0) or COO — coo_bias() reads them
         * generically.  We keep the id; a NULL/absent bias reads as +0. */
        g_bup[b] = _bias_of_add(bb + 60, bb + 59);
        g_bgate[b] = _bias_of_add(bb + 62, bb + 61);
        g_bdown[b] = _bias_of_add(bb + 67, bb + 66);
        /* per-block FFN hidden = last dim of the (COO) up weight [D, hidden] */
        if (g_wup[b] < 0) return 0;
        g_ffn_hidden[b] = t_dims[g_wup[b]][t_rank[g_wup[b]] - 1];
        if (g_ffn_hidden[b] > g_ffn_hidden_max) g_ffn_hidden_max = g_ffn_hidden[b];
        b = b + 1;
    }
    /* dims: Q weight is [D,D] (COO after aliasing). */
    if (g_wq[0] < 0) return 0;
    g_D = t_dims[g_wq[0]][0];

    /* SPARSE FFN DOWN-RESIDUAL: precompute, per block, whether the down-bias is
     * all-zero (=> the residual add over non-down-columns is a no-op) and the SORTED
     * DISTINCT output columns of the COO Wdown (the only columns dn is ever nonzero).
     * Default ON; ALLC_NO_FFN_SPARSE=1 disables (full-D residual add every block). */
    {
        char *e = getenv("ALLC_NO_FFN_SPARSE");
        g_ffn_sparse = (e && atoi(e)) ? 0 : 1;
        for (b = 0; b < g_nblocks; b = b + 1) {
            g_bdown_zero[b] = _weight_all_zero(g_bdown[b]);
            g_down_cols[b] = 0; g_down_ncols[b] = 0;
            if (g_bdown_zero[b] && t_is_coo[g_wdown[b]] && t_nnz[g_wdown[b]] > 0) {
                int nnz = t_nnz[g_wdown[b]]; int *col = t_coo_col[g_wdown[b]];
                float *val = t_coo_val[g_wdown[b]];
                /* mark distinct columns that carry a NONZERO value (a stored 0.0f
                 * scatters +0.0 -> still a no-op, so it need not be visited) */
                unsigned char *seen = calloc((size_t)g_D, 1);
                int j, nc = 0;
                for (j = 0; j < nnz; j = j + 1)
                    if (val[j] != 0.0f && !seen[col[j]]) { seen[col[j]] = 1; nc++; }
                g_down_cols[b] = malloc((size_t)(nc > 0 ? nc : 1) * sizeof(int));
                { int c, w = 0; for (c = 0; c < g_D; c = c + 1) if (seen[c]) g_down_cols[b][w++] = c; }
                g_down_ncols[b] = nc;
                free(seen);
            }
        }
    }

    /* alibi slope tensor: node off33 (Mul_1) input 0 = onnx::Mul_2978x [1,H,1,1] */
    g_slope_tid = _blk_node_in(base, 33, 0);
    if (g_slope_tid < 0) return 0;
    g_H = t_dims[g_slope_tid][1];
    g_HD = g_D / g_H;
    /* scale constant: node off25 (Mul) input 1 = Constant_10 (scalar dense) */
    {
        int sc = _blk_node_in(base, 25, 1);
        g_scale = (sc >= 0 && tf[sc]) ? tf[sc][0] : 1.0f;
    }
    /* BOUNDED-KV / LIVE-BLOCK LOCAL WINDOW: classify each head LOCAL/GLOBAL by its
     * ALiBi slope (LOCAL == an ingest recency slope 6.0/0.5 -> softmax1 weight is
     * EXACTLY 0 beyond a small recency window, so windowing is byte-exact).
     *
     * TWO ways to set the local window (both byte-exact for the LOCAL heads):
     *   - ALLC_BOUNDED_W=W : explicit legacy override (any W).
     *   - LIVE-BLOCK LOCAL WINDOW (default ON): the 3 LIVE blocks (b0 ingest + b7/b11
     *     memory-CAM) are the only ones that run attention post-dead-block-fusion; b0's
     *     20 LOCAL heads (slope 6.0) have a MEASURED max nonzero-weight distance of 28
     *     (ALLC_WINPROBE over echo/cat/quine), so a window of AC_LIVE_LOCAL_WIN=32 keys
     *     is byte-exact and turns b0's per-query O(S) scan into O(W).  Kill-switch:
     *     ALLC_NO_LIVE_WIN=1 (=> g_bounded_w stays 0 => full-history local scan).
     * The GLOBAL memory-CAM heads (b7/b11, slope ~0.004) stay full-history here (their
     * O(1) resolution is the separate DIRECT-CAM-ATTN lever below). */
    {
        char *e = getenv("ALLC_BOUNDED_W");
        char *nl = getenv("ALLC_NO_LIVE_WIN");
        int h;
        if (e) g_bounded_w = atoi(e);                       /* explicit override */
        else if (nl && atoi(nl)) g_bounded_w = 0;           /* kill-switch: full */
        else g_bounded_w = AC_LIVE_LOCAL_WIN;               /* default byte-exact win */
        for (h = 0; h < g_H; h = h + 1) {
            float sl = tf[g_slope_tid][h];
            float d0 = sl - AC_ING_SLOPE0; if (d0 < 0) d0 = -d0;
            float d1 = sl - AC_ING_SLOPE1; if (d1 < 0) d1 = -d1;
            g_head_local[h] = (d0 <= 1e-3f || d1 <= 1e-3f) ? 1 : 0;
        }
    }
    /* DEAD-BLOCK-FUSION: mark every block whose attention sublayer is a no-op
     * (all-zero W_v OR all-zero W_o -> attn residual == passthrough).  Default ON
     * (byte-neutral); ALLC_NO_ATTN_FUSION=1 forces the full attention every block. */
    {
        char *e = getenv("ALLC_NO_ATTN_FUSION");
        g_attn_fuse = (e && atoi(e)) ? 0 : 1;
        g_attn_dead_n = 0;
        for (b = 0; b < g_nblocks; b = b + 1) {
            g_attn_dead[b] = (_weight_all_zero(g_wv[b]) || _weight_all_zero(g_wo[b]))
                             ? 1 : 0;
            if (g_attn_dead[b]) g_attn_dead_n = g_attn_dead_n + 1;
        }
        if (getenv("ALLC_FUSION_AUDIT")) {
            fprintf(stderr, "allc: LIVE-attention blocks (nnz Wv/Wo):");
            for (b = 0; b < g_nblocks; b = b + 1)
                if (!g_attn_dead[b])
                    fprintf(stderr, " b%d(Wv=%d,Wo=%d)", b,
                            t_is_coo[g_wv[b]] ? t_nnz[g_wv[b]] : -1,
                            t_is_coo[g_wo[b]] ? t_nnz[g_wo[b]] : -1);
            fprintf(stderr, "\n");
            /* ALLC_FUSION_AUDIT=2: deep per-head structure of the LIVE blocks —
             * which heads carry nnz V/O, their ALiBi slope, and the Q/K nnz so we
             * can classify each live head local (windowed) vs global (memory-CAM). */
            if (atoi(getenv("ALLC_FUSION_AUDIT")) >= 2) {
                /* FFN sparsity audit: total nnz + coverage over the 242 blocks. */
                long tup = 0, tgate = 0, tdn = 0; int nzero_ffn = 0, ncoo = 0;
                for (b = 0; b < g_nblocks; b = b + 1) {
                    int nu = t_is_coo[g_wup[b]] ? t_nnz[g_wup[b]] : -1;
                    int ng = t_is_coo[g_wgate[b]] ? t_nnz[g_wgate[b]] : -1;
                    int nd = t_is_coo[g_wdown[b]] ? t_nnz[g_wdown[b]] : -1;
                    if (t_is_coo[g_wup[b]]) ncoo++;
                    if (nu > 0) tup += nu;
                    if (ng > 0) tgate += ng;
                    if (nd > 0) tdn += nd;
                    if ((nu <= 0) && (ng <= 0) && (nd <= 0)) nzero_ffn++;
                }
                fprintf(stderr, "allc: FFN audit: %d/%d blocks COO; total nnz up=%ld "
                        "gate=%ld down=%ld; %d all-zero-FFN blocks; ffn is ALREADY "
                        "O(nnz) sparse (coo_project)\n", ncoo, g_nblocks, tup, tgate,
                        tdn, nzero_ffn);
                { long sh = 0; int nbup = 0, nbgate = 0, nbdown = 0; int bd_nz = 0;
                  for (b = 0; b < g_nblocks; b = b + 1) sh += g_ffn_hidden[b];
                  for (b = 0; b < g_nblocks; b = b + 1) {
                      if (g_bup[b] >= 0 && !_weight_all_zero(g_bup[b])) nbup++;
                      if (g_bgate[b] >= 0 && !_weight_all_zero(g_bgate[b])) nbgate++;
                      if (g_bdown[b] >= 0 && !_weight_all_zero(g_bdown[b])) { nbdown++;
                          { int cc; const float *bb2 = tf[g_bdown[b]];
                            if (bb2) for (cc=0; cc<g_D; cc++) if (bb2[cc]!=0.0f) bd_nz++; } }
                  }
                  fprintf(stderr, "allc: FFN elemwise audit: sum(hid)=%ld over %d blocks "
                          "(silu loop size/step at T=1); nonzero biases up=%d gate=%d "
                          "down=%d (down bias nnz total=%d over D=%d)\n",
                          sh, g_nblocks, nbup, nbgate, nbdown, bd_nz, g_D);
                  /* silu-skip potential: sum(hid) over blocks where BOTH up-bias and
                   * gate-bias are zero (there up==0 && gate==0 -> silu(0)*0==0, so
                   * only the up-nnz rows need computing).  Report the skippable hid. */
                  { long sh_skip = 0, sh_bias = 0; int nskip = 0;
                    for (b = 0; b < g_nblocks; b = b + 1) {
                        int upz = _weight_all_zero(g_bup[b]);
                        int gz  = _weight_all_zero(g_bgate[b]);
                        if (upz && gz) { sh_skip += g_ffn_hidden[b]; nskip++; }
                        else sh_bias += g_ffn_hidden[b];
                    }
                    fprintf(stderr, "allc: silu-skip audit: %d blocks (hid sum=%ld) have "
                            "zero up+gate bias -> silu computable ONLY on up-nnz rows; "
                            "%ld hid in bias-carrying blocks (must compute full)\n",
                            nskip, sh_skip, sh_bias); }
                  /* FULL-NOOP audit: a block whose attn is dead AND whose FFN delta
                   * (hh@Wdown + bdown) is identically 0 (Wdown all-zero AND bdown
                   * all-zero) is a pure passthrough xout==xin -> skip it entirely. */
                  { int nnoop = 0;
                    for (b = 0; b < g_nblocks; b = b + 1)
                        if (g_attn_dead[b] && _weight_all_zero(g_wdown[b])
                            && _weight_all_zero(g_bdown[b])) nnoop++;
                    fprintf(stderr, "allc: FULL-NOOP audit: %d/%d blocks are attn-dead + "
                            "zero-down-proj + zero-down-bias -> pure passthrough (skippable)\n",
                            nnoop, g_nblocks); } }
                fprintf(stderr, "allc: per-head slopes:");
                { int h; for (h = 0; h < g_H; h = h + 1)
                    fprintf(stderr, " h%d=%.5g%s", h, tf[g_slope_tid][h],
                            g_head_local[h] ? "(L)" : "(G)"); }
                fprintf(stderr, "\n");
                for (b = 0; b < g_nblocks; b = b + 1) {
                    if (g_attn_dead[b]) continue;
                    fprintf(stderr, "allc: block %d live heads:\n", b);
                    { int h; for (h = 0; h < g_H; h = h + 1) {
                        int col0 = h * g_HD, col1 = col0 + g_HD;
                        int j, vnnz = 0, onnz = 0, qnnz = 0, knnz = 0;
                        /* V writes head-h output cols [col0,col1); count nnz there */
                        if (t_is_coo[g_wv[b]]) for (j = 0; j < t_nnz[g_wv[b]]; j++)
                            { int c = t_coo_col[g_wv[b]][j];
                              if (c >= col0 && c < col1 && t_coo_val[g_wv[b]][j] != 0.0f) vnnz++; }
                        /* O reads head-h input rows [col0,col1); count nnz there */
                        if (t_is_coo[g_wo[b]]) for (j = 0; j < t_nnz[g_wo[b]]; j++)
                            { int r = t_coo_row[g_wo[b]][j];
                              if (r >= col0 && r < col1 && t_coo_val[g_wo[b]][j] != 0.0f) onnz++; }
                        if (t_is_coo[g_wq[b]]) for (j = 0; j < t_nnz[g_wq[b]]; j++)
                            { int c = t_coo_col[g_wq[b]][j];
                              if (c >= col0 && c < col1 && t_coo_val[g_wq[b]][j] != 0.0f) qnnz++; }
                        if (t_is_coo[g_wk[b]]) for (j = 0; j < t_nnz[g_wk[b]]; j++)
                            { int c = t_coo_col[g_wk[b]][j];
                              if (c >= col0 && c < col1 && t_coo_val[g_wk[b]][j] != 0.0f) knnz++; }
                        if (vnnz || onnz || qnnz || knnz)
                            fprintf(stderr, "   h%2d slope=%.5g %s  Vnnz=%d Onnz=%d "
                                    "Qnnz=%d Knnz=%d\n", h, tf[g_slope_tid][h],
                                    g_head_local[h] ? "LOCAL " : "GLOBAL", vnnz, onnz,
                                    qnnz, knnz);
                    } }
                }
            }
        }
    }
    g_incr_ready = 1;
    g_prefix_len = 0;
    g_prev_S = 0;
    for (b = 0; b < g_nblocks; b = b + 1) {
        g_Kc[b] = 0; g_Vc[b] = 0; g_cap[b] = 0;
    }
    return 1;
}

/* row projection over a ROW RANGE [r0,r1):  out[t][*] = sum_r act[t][r] * W[r][*].
 * act is [.,K] (row-major); out is [.,N].  W may be:
 *   (a) COO (t_is_coo): scatter the nnz in STORED order (ascending flat r*N+q),
 *       so each out[t][q] accumulates ascending-r == the dense reduction order.
 *   (b) DENSE (is_init==1, a small weight kept dense): a plain row-major dense
 *       matmul, ALSO ascending-r per column (same order).
 * Both are byte-exact to the graph's op_matmul.  Each row writes a disjoint out
 * slice -> byte-exact under any row split (no counter side-effects here). */
static void proj_rows(const float *act, int r0, int r1, int K, int Wt,
                      float *out, int N) {
    int t;
    if (t_is_coo[Wt]) {
        int nnz = t_nnz[Wt];
        int *row = t_coo_row[Wt]; int *col = t_coo_col[Wt]; float *val = t_coo_val[Wt];
        for (t = r0; t < r1; t = t + 1) {
            const float *arow = act + (long)t * K;
            float *orow = out + (long)t * N;
            int j;
            memset(orow, 0, (size_t)N * sizeof(float));   /* 0.0f == all-zero bits */
            for (j = 0; j < nnz; j = j + 1)
                orow[col[j]] += arow[row[j]] * val[j];
        }
    } else {
        /* dense fallback: out[t][q] = sum_r act[t][r] * W[r*N+q], ascending r */
        const float *W = tf[Wt];
        for (t = r0; t < r1; t = t + 1) {
            const float *arow = act + (long)t * K;
            float *orow = out + (long)t * N;
            int q; int r;
            for (q = 0; q < N; q = q + 1) orow[q] = 0.0f;
            for (r = 0; r < K; r = r + 1) {
                float a = arow[r];
                const float *wr = W + (long)r * N;
                for (q = 0; q < N; q = q + 1) orow[q] += a * wr[q];
            }
        }
    }
}

/* whole-matrix projection (all T rows), used by the attention Q/K/V/O paths.  Keeps
 * the SPARSE_ITERS / DENSE_EQUIV_ITERS diagnostic tallies. */
static void coo_project(const float *act, int T, int K, int Wt,
                        float *out, int N) {
    proj_rows(act, 0, T, K, Wt, out, N);
    if (t_is_coo[Wt]) {
        SPARSE_ITERS += (long)T * (long)t_nnz[Wt];
        DENSE_EQUIV_ITERS += (long)T * (long)K * (long)N;
    } else {
        DENSE_EQUIV_ITERS += (long)T * (long)K * (long)N;
        SPARSE_ITERS += (long)T * (long)K * (long)N;
    }
}

/* grow a per-block K/V cache to hold at least `need` positions */
static void cache_reserve(int b, int need) {
    if (g_cap[b] >= need) return;
    {
        int newcap = g_cap[b] ? g_cap[b] : 64;
        while (newcap < need) newcap = newcap * 2;
        g_Kc[b] = realloc(g_Kc[b], (long)newcap * g_D * sizeof(float));
        g_Vc[b] = realloc(g_Vc[b], (long)newcap * g_D * sizeof(float));
        g_cap[b] = newcap;
    }
}

/* ---- PERSISTENT FFN SCRATCH (kills the per-block malloc/free of up/gate/hh/dn):
 * the FFN runs 242x/step and previously malloc'd 4 buffers each time.  These grow-
 * only scratch buffers are reused across ALL blocks + steps (single-threaded FFN
 * body).  Byte-neutral (same values, same order).
 *
 * CACHE-FOOTPRINT DISCIPLINE: the persistent buffers are sized to the STEADY-STATE
 * tail (T ~= FRAME_LEN+1 = 31 rows), NOT to the one-off cold-start full recompute
 * (T = S, e.g. 1651 at quine step 0, which would leave ~85MB permanently resident
 * and THRASH the tiny steady-state working set out of L2/L3 for every later step).
 * A step whose T exceeds the persistent cap (only the rare fresh/full recompute)
 * malloc/free's a one-shot local buffer instead — so the resident scratch stays
 * small (a few MB) after the cold step.  ffn_alloc() returns the buffer to use and
 * sets *owned=1 when the caller must free it. */
#define FFN_SCRATCH_MAX_T 64      /* persistent only up to this tail size */
static float *g_ffn_up, *g_ffn_gate, *g_ffn_hh, *g_ffn_dn;
static long   g_ffn_scratch_hid, g_ffn_scratch_d;
static void ffn_scratch_reserve(long need_hid, long need_d) {
    if (need_hid > g_ffn_scratch_hid) {
        g_ffn_up   = realloc(g_ffn_up,   need_hid * sizeof(float));
        g_ffn_gate = realloc(g_ffn_gate, need_hid * sizeof(float));
        g_ffn_hh   = realloc(g_ffn_hh,   need_hid * sizeof(float));
        g_ffn_scratch_hid = need_hid;
    }
    if (need_d > g_ffn_scratch_d) {
        g_ffn_dn = realloc(g_ffn_dn, need_d * sizeof(float));
        g_ffn_scratch_d = need_d;
    }
}

/* ---- FFN over a ROW-RANGE [r0,r1): all work is per-row-independent, so this is the
 * unit a pthread worker owns.  up/gate/hh/dn are the (shared) scratch; each worker
 * writes disjoint row slices -> byte-exact under any row split (same per-(row,col)
 * ascending-r reduction order regardless of which thread runs the row). */
static void ffn_rows(int b, float *xout, int r0, int r1, int hid,
                     const float *bup, const float *bgate, const float *bdown,
                     float *up, float *gate, float *hh, float *dn) {
    int t;
    /* up/gate projection for these rows (COO scatter, ascending-r, byte-exact) */
    proj_rows(xout, r0, r1, g_D, g_wup[b], up, hid);
    proj_rows(xout, r0, r1, g_D, g_wgate[b], gate, hid);
    /* silu(up)*gate */
    for (t = r0; t < r1; t++) {
        const float *upr = up + (long)t * hid, *gar = gate + (long)t * hid;
        float *hhr = hh + (long)t * hid; int j;
        for (j = 0; j < hid; j++) {
            float u = upr[j] + (bup ? bup[j] : 0.0f);
            float ga = gar[j] + (bgate ? bgate[j] : 0.0f);
            float su = u / (1.0f + expf(-u));
            hhr[j] = su * ga;
        }
    }
    /* down projection + residual add (sparse when down-bias is zero) */
    if (g_ffn_sparse && g_bdown_zero[b] && g_down_cols[b]) {
        int *dc = g_down_cols[b]; int nc = g_down_ncols[b];
        int nnz = t_nnz[g_wdown[b]];
        int *drow = t_coo_row[g_wdown[b]], *dcol = t_coo_col[g_wdown[b]]; float *dval = t_coo_val[g_wdown[b]];
        for (t = r0; t < r1; t++) {
            const float *hr = hh + (long)t * hid; float *dr = dn + (long)t * g_D;
            long ro = (long)t * g_D; int c, j;
            for (c = 0; c < nc; c++) dr[dc[c]] = 0.0f;
            for (j = 0; j < nnz; j++) dr[dcol[j]] += hr[drow[j]] * dval[j];
            for (c = 0; c < nc; c++) { int jj = dc[c]; xout[ro + jj] += dr[jj]; }
        }
    } else {
        proj_rows(hh, r0, r1, hid, g_wdown[b], dn, g_D);
        for (t = r0; t < r1; t++) {
            float *xr = xout + (long)t * g_D; const float *dr = dn + (long)t * g_D; int j;
            for (j = 0; j < g_D; j++) xr[j] += dr[j] + (bdown ? bdown[j] : 0.0f);
        }
    }
}

#ifdef USE_PTHREADS
/* fork-join over FFN row ranges — only used for LARGE T (cold/full-recompute steps),
 * where the per-row work dwarfs the thread spawn cost.  Byte-exact (disjoint rows). */
typedef struct { int b; float *xout; int r0, r1, hid;
                 const float *bup, *bgate, *bdown;
                 float *up, *gate, *hh, *dn; } _ffn_arg;
static void *_ffn_worker(void *a) {
    _ffn_arg *w = (_ffn_arg *)a;
    ffn_rows(w->b, w->xout, w->r0, w->r1, w->hid, w->bup, w->bgate, w->bdown,
             w->up, w->gate, w->hh, w->dn);
    return 0;
}
static int _nthreads(void);   /* fwd decl (defined with the attn pool below) */
#endif

/* FFN THREADING THRESHOLD: below this T the fork-join overhead exceeds the per-row
 * work (steady-state incremental tail T~31 stays serial), so parallelise only the
 * rare large-T full-recompute (quine step 0, evict/skip fresh passes). */
#define FFN_PAR_MIN_T 96

/* ---- the SwiGLU FFN sublayer for block b over T tail rows in `xout` (in place).
 * Shared by run_incremental / run_full_skip / run_full_evict (was inlined 3x,
 * byte-identical).  Persistent scratch for small T (steady state); one-shot local
 * buffers for the rare large-T cold recompute (keeps the resident set small so the
 * steady-state working set stays in cache).  For large T the row loops fork-join
 * across cores (byte-exact). */
static void ffn_block(int b, float *xout, int T) {
    int hid = g_ffn_hidden[b];
    const float *bup = (g_bup[b] >= 0) ? tf[g_bup[b]] : 0;
    const float *bgate = (g_bgate[b] >= 0) ? tf[g_bgate[b]] : 0;
    const float *bdown = (g_bdown[b] >= 0) ? tf[g_bdown[b]] : 0;
    float *up, *gate, *hh, *dn;
    int owned = 0;
    double _a0 = g_phase_on ? _now_ms() : 0;
    if (T <= FFN_SCRATCH_MAX_T) {
        ffn_scratch_reserve((long)T * hid, (long)T * g_D);
        up = g_ffn_up; gate = g_ffn_gate; hh = g_ffn_hh; dn = g_ffn_dn;
    } else {
        up = malloc((long)T * hid * sizeof(float));
        gate = malloc((long)T * hid * sizeof(float));
        hh = malloc((long)T * hid * sizeof(float));
        dn = malloc((long)T * g_D * sizeof(float));
        owned = 1;
    }
    if (g_phase_on) { g_ph_alloc += _now_ms() - _a0; }
    /* diagnostic iter tallies (proj_rows itself is counter-free): up/gate read D->hid,
     * down reads hid->D — all three COO-sparse in this model. */
    g_iter_ffn += (long)T * (t_nnz[g_wup[b]] + t_nnz[g_wgate[b]] + t_nnz[g_wdown[b]]);
    SPARSE_ITERS += (long)T * (t_nnz[g_wup[b]] + t_nnz[g_wgate[b]] + t_nnz[g_wdown[b]]);
    DENSE_EQUIV_ITERS += (long)T * ((long)g_D * hid * 2 + (long)hid * g_D);

#ifdef USE_PTHREADS
    { int nth = (T >= FFN_PAR_MIN_T) ? _nthreads() : 1;
      if (nth > T) nth = T;
      if (nth > 1) {
        pthread_t th[256]; _ffn_arg args[256]; int i;
        int per = (T + nth - 1) / nth;
        for (i = 0; i < nth; i++) {
            args[i].b = b; args[i].xout = xout; args[i].hid = hid;
            args[i].bup = bup; args[i].bgate = bgate; args[i].bdown = bdown;
            args[i].up = up; args[i].gate = gate; args[i].hh = hh; args[i].dn = dn;
            args[i].r0 = i * per; args[i].r1 = (i + 1) * per;
            if (args[i].r1 > T) args[i].r1 = T;
            pthread_create(&th[i], 0, _ffn_worker, &args[i]);
        }
        for (i = 0; i < nth; i++) pthread_join(th[i], 0);
      } else ffn_rows(b, xout, 0, T, hid, bup, bgate, bdown, up, gate, hh, dn);
    }
#else
    ffn_rows(b, xout, 0, T, hid, bup, bgate, bdown, up, gate, hh, dn);
#endif

    if (g_phase_on) g_ph_ffn += _now_ms() - _a0;   /* whole FFN body (proj+silu+resid) */
    if (owned) { free(up); free(gate); free(hh); free(dn); }
}

/* ABSOLUTE-position map for the EVICT (compacted-stream) path.  When set (non-NULL)
 * the K/V cache rows are a COMPACTED subset of the logical stream (evicted rows are
 * physically absent), so a cache row index `k` no longer equals its absolute stream
 * position.  ALiBi's distance MUST be computed from ABSOLUTE positions (the model's
 * positional recency is a function of the real token distance, not the compacted
 * index), and the causal mask likewise (k is a valid key for query q iff
 * abs_pos[k] <= abs_pos[q]).  g_abs_pos[i] = absolute stream position of cache row i.
 * NULL (all other paths: incremental / full-skip) => identity (abs_pos[i]==i), the
 * exact prior behaviour (attn_head then reads k / qpos directly). */
const int *g_abs_pos;        /* NULL => identity; else abs position per cache row */

/* ---- GLOBAL-HEAD KEY-SUBSET (DIRECT-CAM-ATTN, default ON; ALLC_NO_GKEY=1 to disable):
 * the 3 GLOBAL memory-CAM heads (b7 h20, b11 h21/h22, ALiBi slope ~0.004) gate their
 * VALUE projection to EXACTLY 0 on every NON-store row, so only §Memory store rows
 * carry a nonzero value into their attention output; and the softmax1 denominator is
 * unchanged when the non-store rows OUTSIDE the recency window are dropped (the SAME
 * byte-exact keep-set the FREE-DRIVEN EVICTION path is proven on: BOS + live store
 * rows + last-W recency rows).  So for a GLOBAL head we can scan ONLY the keep-set
 * keys instead of all SK keys — O(#stores + W) not O(S) per query — BYTE-EXACT.
 * g_gkey[k] (per cache row, built per forward) = 1 iff key k is in the keep-set;
 * NULL => scan every key (the exact prior behaviour).  The LOCAL heads always scan
 * their full windowed range (g_gkey is ignored for them). */
const unsigned char *g_gkey;     /* NULL => scan all keys; else GLOBAL keep-set mask */
int   g_gkey_on;                 /* 1 => build+use the global keep-set (default ON) */
#define AC_GLOBAL_WIN (2 * G_FRAME_LEN)   /* recency window kept for GLOBAL heads */

/* one head's softmax1+ALiBi attention over the tail rows: Qt[t] (head h) attends
 * over cached K/V[0..SK-1], writes attnout[t][h*HD..].  Each head is independent
 * (disjoint output slice + private scratch) so this is parallel-safe. */
static void attn_head(int h, int b, int P, int T, int SK,
                      const float *Qt, float *attnout) {
    float slope = tf[g_slope_tid][h];
    int hoff = h * g_HD;
    float *sch = malloc((long)SK * sizeof(float));
    /* BOUNDED-KV: for a LOCAL head (ingest-recency slope), keys older than the
     * window have softmax1 weight EXACTLY 0 (ALiBi -slope*dist underflows exp),
     * so starting the scan at qpos-W+1 is BYTE-EXACT.  GLOBAL heads: full history
     * (klo=0).  klo is computed per query row below (window relative to qpos).
     * NOTE: under the EVICT path (g_abs_pos set) the window is applied on the
     * COMPACTED index — safe because the compacted stream ALREADY keeps every row
     * within the recency window (the evict keep-set is store-rows + last-W rows),
     * so no windowed local key is ever missing; the local_win here is a harmless
     * upper bound (all kept local-window rows are contiguous at the tail). */
    int local_win = (g_bounded_w > 0 && g_head_local[h]) ? g_bounded_w : 0;
    /* GLOBAL-HEAD KEY-SUBSET: for a GLOBAL memory-CAM head, restrict the scan to the
     * keep-set keys (g_gkey[k]==1: store rows + BOS + recency window).  Byte-exact:
     * the dropped keys carry V==0 (zero numerator) and their softmax1 exp-weight sums
     * to a denom delta that rounds away — the SAME keep-set proven byte-exact by the
     * eviction path (scanned here in the same ascending-k order + same real distances).
     * A LOCAL head ignores g_gkey (it uses local_win).  NULL g_gkey => scan all. */
    const unsigned char *gk = (!g_head_local[h] && g_gkey) ? g_gkey : 0;
    int t;
    long ops = 0;
    for (t = 0; t < T; t = t + 1) {
        int qpos = P + t;
        int qabs = g_abs_pos ? g_abs_pos[qpos] : qpos;   /* absolute pos of query */
        const float *qv = Qt + (long)t * g_D + hoff;
        int k; float m; float denom; int hd;
        int klo = 0;
        if (local_win > 0) { klo = qpos - local_win + 1; if (klo < 0) klo = 0; }
        m = 0.0f;                   /* softmax1 sink: clamp(max,0) starts at 0 */
        for (k = klo; k < SK; k = k + 1) {
            int kabs = g_abs_pos ? g_abs_pos[k] : k;
            if (kabs > qabs) { sch[k] = -1e30f; continue; }   /* causal (abs) */
            if (gk && !gk[k]) { sch[k] = -1e30f; continue; }  /* global keep-set skip */
            {
                const float *kv = g_Kc[b] + (long)k * g_D + hoff;
                float dot = 0.0f;
                for (hd = 0; hd < g_HD; hd = hd + 1) dot += qv[hd] * kv[hd];
                {
                    float dist = (float)(qabs - kabs);   /* |q-k| in ABSOLUTE pos */
                    sch[k] = dot * g_scale - slope * dist;
                }
                if (sch[k] > m) m = sch[k];
                ops += 2 * g_HD;                          /* realised dot + attn@V */
            }
        }
        denom = expf(-m);           /* softmax1: exp(x-m)/(exp(-m)+sum exp(x-m)) */
        for (k = klo; k < SK; k = k + 1) {
            int kabs = g_abs_pos ? g_abs_pos[k] : k;
            if (kabs > qabs) continue;                       /* skip non-causal */
            if (gk && !gk[k]) continue;                      /* global keep-set skip */
            sch[k] = expf(sch[k] - m);
            denom += sch[k];
        }
        if (g_winprobe && !g_head_local[h]) {   /* GLOBAL head: how many keys blend? */
            int nz = 0; float wtop = 0.0f;
            for (k = klo; k < SK; k = k + 1) {
                float w;
                if (gk && !gk[k]) continue;
                w = sch[k] / denom;
                if (w != 0.0f) { nz++; if (w > wtop) wtop = w; }
            }
            if (nz > g_winprobe_gnz[h]) g_winprobe_gnz[h] = nz;
            /* record min top-weight (1.0 => hard 1-hot gather; <1 => soft blend + sink) */
            if (wtop < g_winprobe_gtop[h] || g_winprobe_gtop[h] == 0.0f)
                g_winprobe_gtop[h] = wtop;
        }
        {
            float *outv = attnout + (long)t * g_D + hoff;
            for (hd = 0; hd < g_HD; hd = hd + 1) outv[hd] = 0.0f;
            for (k = klo; k < SK; k = k + 1) {   /* attn@V, ascending-k order */
                int kabs = g_abs_pos ? g_abs_pos[k] : k;
                float w;
                const float *vv;
                if (kabs > qabs) continue;
                if (gk && !gk[k]) continue;                  /* global keep-set skip */
                w = sch[k] / denom;
                if (g_winprobe && sch[k] != 0.0f) {   /* nonzero softmax1 weight */
                    int dist = qabs - kabs;
                    if (dist > g_winprobe_max[h]) g_winprobe_max[h] = dist;
                }
                vv = g_Vc[b] + (long)k * g_D + hoff;
                for (hd = 0; hd < g_HD; hd = hd + 1) outv[hd] += w * vv[hd];
            }
        }
    }
    g_iter_attn_h[h] = ops;   /* race-free: this thread owns head h */
    free(sch);
}

#ifdef USE_PTHREADS
/* fork-join parallel-for over the g_H heads (self-contained, static-linkable) */
typedef struct { int b, P, T, SK; const float *Qt; float *attnout;
                 int h0, h1; } _attn_arg;
static void *_attn_worker(void *a) {
    _attn_arg *w = (_attn_arg *)a; int h;
    for (h = w->h0; h < w->h1; h = h + 1)
        attn_head(h, w->b, w->P, w->T, w->SK, w->Qt, w->attnout);
    return 0;
}
static int _nthreads(void) {
    static int n = 0;
    if (n == 0) {
        char *e = getenv("INCR_THREADS");
        n = e ? atoi(e) : (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (n < 1) n = 1;
    }
    return n;
}
#endif

static void attn_all_heads(int b, int P, int T, int SK,
                           const float *Qt, float *attnout) {
#ifdef USE_PTHREADS
    int nth = _nthreads();
    if (nth > g_H) nth = g_H;
    if (nth > 1) {
        pthread_t th[256]; _attn_arg args[256]; int i;
        int per = (g_H + nth - 1) / nth;
        for (i = 0; i < nth; i = i + 1) {
            args[i].b = b; args[i].P = P; args[i].T = T; args[i].SK = SK;
            args[i].Qt = Qt; args[i].attnout = attnout;
            args[i].h0 = i * per; args[i].h1 = (i + 1) * per;
            if (args[i].h1 > g_H) args[i].h1 = g_H;
            pthread_create(&th[i], 0, _attn_worker, &args[i]);
        }
        for (i = 0; i < nth; i = i + 1) pthread_join(th[i], 0);
        return;
    }
#endif
    {
        int h;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (h = 0; h < g_H; h = h + 1)
            attn_head(h, b, P, T, SK, Qt, attnout);
    }
}

/* the incremental block-stack forward over a residual [1,S,D] input already in
 * tf[input_tid].  Recomputes only the TAIL rows [P .. S-1] where P is the stable
 * prefix length (= prev_S-1, 0 on a fresh stream), reading each block's cached
 * prefix K/V.  Writes the exact hidden row -1 into tf[output_tid] row S-1 (only
 * that row is read by the driver).  Byte-exact vs run() at row -1.  Returns 0. */
/* the EXACT prefix length run_incremental(S) will use for the CURRENT cache state —
 * lets the driver build ONLY the tail input rows [P..S-1] (the only rows the forward
 * reads).  Mirrors the P logic below; MUST stay in sync with it. */
int incr_prefix_len(int S) {
    int P;
    if (!g_incr_ready) return 0;
    if (g_prev_S == 0 || S <= g_prev_S) P = 0;
    else P = g_prev_S - 1;
    if (P != g_prefix_len) P = 0;
    return P;
}

void run_incremental(int S) {
    int P; int T; int b;
    float *xin;                 /* current block-input residual for tail rows [T,D] */
    float *xout;                /* current block-output residual for tail rows [T,D] */
    float *Kt; float *Vt; float *Qt; float *attnout;

    if (!g_incr_ready) { run(); return; }

    /* prefix length: rows [0..P-1] are stable (cached).  On a fresh stream (or a
     * reset when S went backwards) P=0 and caches are empty.  incr_prefix_len(S)
     * MUST return this same P (the driver relies on it to tail-build the input). */
    if (g_prev_S == 0 || S <= g_prev_S) {
        P = 0; g_prefix_len = 0;
        for (b = 0; b < g_nblocks; b = b + 1) g_cap[b] = g_cap[b]; /* keep alloc */
    } else {
        P = g_prev_S - 1;       /* verified: rows 0..prev_S-2 immutable */
    }
    if (P != g_prefix_len) {
        /* the cache holds exactly g_prefix_len positions; if the requested prefix
         * differs (shouldn't in steady state) fall back to a full rebuild. */
        P = 0; g_prefix_len = 0;
    }
    T = S - P;

    xin = malloc((long)T * g_D * sizeof(float));
    xout = malloc((long)T * g_D * sizeof(float));
    Qt = malloc((long)T * g_D * sizeof(float));
    Kt = malloc((long)T * g_D * sizeof(float));
    Vt = malloc((long)T * g_D * sizeof(float));
    attnout = malloc((long)T * g_D * sizeof(float));

    /* copy the tail rows of the input residual (positions P..S-1) */
    {
        int t;
        for (t = 0; t < T; t = t + 1)
            memcpy(xin + (long)t * g_D, tf[input_tid] + (long)(P + t) * g_D,
                   g_D * sizeof(float));
    }

    for (b = 0; b < g_nblocks; b = b + 1) {
        int SK = P + T;         /* total keys = prefix + tail */
        int h; int t;
        double _pt0 = g_phase_on ? _now_ms() : 0;
        if (g_attn_fuse && g_attn_dead[b]) {
            /* DEAD-BLOCK-FUSION: attention sublayer is a no-op (W_v or W_o all-zero
             * -> attn out == 0 -> residual passthrough).  Skip Q/K/V proj + softmax
             * + O proj entirely; the block-input residual passes straight to the FFN.
             * BYTE-EXACT (the skipped ops add mathematically-exact zero). */
            memcpy(xout, xin, (long)T * g_D * sizeof(float));
        } else {
        /* Q/K/V projections for the tail rows (COO, ascending-r, byte-exact) */
        coo_project(xin, T, g_D, g_wq[b], Qt, g_D);
        coo_project(xin, T, g_D, g_wk[b], Kt, g_D);
        coo_project(xin, T, g_D, g_wv[b], Vt, g_D);
        g_iter_proj += (long)T * (t_nnz[g_wq[b]] + t_nnz[g_wk[b]] + t_nnz[g_wv[b]]);

        /* append the tail K/V to the cache buffer so K/V is contiguous per head.
         * cache buffers hold rows [0..P-1]; we write tail rows into [P..SK-1]. */
        cache_reserve(b, SK);
        for (t = 0; t < T; t = t + 1) {
            memcpy(g_Kc[b] + (long)(P + t) * g_D, Kt + (long)t * g_D, g_D * sizeof(float));
            memcpy(g_Vc[b] + (long)(P + t) * g_D, Vt + (long)t * g_D, g_D * sizeof(float));
        }

        /* per-head attention: Qt[t] attends over K/V[0..SK-1], softmax1+ALiBi,
         * causal (mask k>q).  Parallelised over heads (disjoint output slices)
         * via OpenMP or self-contained pthreads. */
        attn_all_heads(b, P, T, SK, Qt, attnout);
        for (h = 0; h < g_H; h = h + 1) g_iter_attn += g_iter_attn_h[h];
        (void)t;

        /* output projection W_o (COO) + residual: xout = xin + attnout @ W_o */
        coo_project(attnout, T, g_D, g_wo[b], xout, g_D);
        g_iter_proj += (long)T * t_nnz[g_wo[b]];
        for (t = 0; t < T * g_D; t = t + 1) xout[t] = xout[t] + xin[t];
        }   /* end attention sublayer */
        if (g_phase_on) g_ph_attn += _now_ms() - _pt0;

        /* SwiGLU FFN (shared helper; sparse down-residual + persistent scratch). */
        ffn_block(b, xout, T);

        /* this block's output tail becomes the next block's input tail */
        { float *tmp = xin; xin = xout; xout = tmp; }
    }

    /* Materialise the hidden output tensor [1,S,D].  The single-step driver reads
     * only row -1; the SPECULATIVE (batched-K) driver reads the K frame-end rows,
     * all of which live in the freshly-computed tail [P..S-1].  So we write the
     * EXACT block-stack output for EVERY tail row (positions P..S-1) into the
     * output tensor, and zero the stable-prefix rows [0..P-1] (never read).  This
     * is BYTE-EXACT: each tail row's value in xin is the same value the row would
     * have as row -1 of a stream truncated there (causal attention → a row's
     * output depends only on rows <= it), so row (P+i) here == the single-step
     * decode of a stream that ended at P+i.  The row-(S-1) copy is bit-identical
     * to the prior single-row behaviour. */
    {
        int rdim[3]; rdim[0] = 1; rdim[1] = S; rdim[2] = g_D;
        int t;
        alloc_tensor(output_tid, DT_FLOAT, 3, rdim);
        if (P > 0) memset(tf[output_tid], 0, (long)P * g_D * sizeof(float));
        for (t = 0; t < T; t = t + 1)
            memcpy(tf[output_tid] + (long)(P + t) * g_D, xin + (long)t * g_D,
                   g_D * sizeof(float));
    }

    /* update the stable prefix for the next step: rows [0..S-2] are now stable.
     * The cache buffers already hold K/V for rows [0..SK-1] (=[0..S-1]); the new
     * stable prefix is S-1 positions (drop the current last row, which will be
     * recomputed next step).  We simply record the new prefix length; the last
     * row's cached K/V slot is overwritten next step. */
    g_prefix_len = S - 1;
    g_prev_S = S;

    free(xin); free(xout); free(Qt); free(Kt); free(Vt); free(attnout);
}

/* =====================================================================
 * PER-OP BLOCK-SKIP FORWARD  (#738 port — byte-exact for decode)
 * =====================================================================
 * Runs the block stack over ALL S rows (fresh, P=0, no cross-step KV cache) but
 * applies ONLY the decoded opcode's LIVE blocks; a SKIPPED block passes the
 * residual straight through (xout=xin, no attn / no FFN).  This is the EXACT
 * semantics of step_block_skip.StepBlockSkipRunner driven from the torch
 * pure-forward driver (which also re-embeds fresh + skips per step, no cache), so
 * it is byte-exact vs run_incremental at row -1 for every op whose live-set is a
 * SUPERSET of its true decode-live set (verified end-to-end by the harness).
 *
 * WHY P=0 (no cache): the cross-step KV cache in run_incremental freezes a prefix
 * row's per-block K/V at the step that row was the tail.  Under block-skip the
 * live set VARIES per step, so a prefix row's per-block input differs from the
 * current op's recompute — the cache is NOT block-skip-safe.  The block-COUNT
 * reduction (242 -> ~8 for echo) is the contention-independent win here; it is
 * orthogonal to the O(S) cache win and MUCH larger for the tiny-live-set I/O ops.
 *
 * live_mask: G_SKIP_NB bytes, live_mask[b]==1 -> run block b.  Caller guarantees
 * G_SKIP_NB == g_nblocks. */
void run_full_skip(int S, const unsigned char *live_mask) {
    int b, t;
    int T = S;                  /* recompute ALL rows */
    float *xin, *xout, *Qt, *Kt, *Vt, *attnout;

    if (!g_incr_ready) { run(); return; }

    xin = malloc((long)T * g_D * sizeof(float));
    xout = malloc((long)T * g_D * sizeof(float));
    Qt = malloc((long)T * g_D * sizeof(float));
    Kt = malloc((long)T * g_D * sizeof(float));
    Vt = malloc((long)T * g_D * sizeof(float));
    attnout = malloc((long)T * g_D * sizeof(float));

    /* full input residual (all rows) */
    memcpy(xin, tf[input_tid], (long)S * g_D * sizeof(float));

    for (b = 0; b < g_nblocks; b = b + 1) {
        int SK = T;
        if (!live_mask[b]) continue;    /* SKIP: xout==xin, residual passes through
                                         * (xin is already the running residual) */
        if (g_attn_fuse && g_attn_dead[b]) {
            /* DEAD-BLOCK-FUSION: attn no-op (W_v/W_o all-zero); residual passthrough
             * into the FFN.  BYTE-EXACT (skipped ops add exact zero). */
            memcpy(xout, xin, (long)T * g_D * sizeof(float));
        } else {
        /* Q/K/V projections for ALL rows (COO, ascending-r, byte-exact) */
        coo_project(xin, T, g_D, g_wq[b], Qt, g_D);
        coo_project(xin, T, g_D, g_wk[b], Kt, g_D);
        coo_project(xin, T, g_D, g_wv[b], Vt, g_D);
        g_iter_proj += (long)T * (t_nnz[g_wq[b]] + t_nnz[g_wk[b]] + t_nnz[g_wv[b]]);

        /* K/V live in Qt-parallel buffers Kt/Vt (rows 0..SK-1).  Point the cache
         * pointers at them for attn_head (which reads g_Kc[b]/g_Vc[b]); this call
         * is a FULL recompute so the "cache" is exactly the freshly-projected K/V. */
        cache_reserve(b, SK);
        memcpy(g_Kc[b], Kt, (long)SK * g_D * sizeof(float));
        memcpy(g_Vc[b], Vt, (long)SK * g_D * sizeof(float));

        attn_all_heads(b, 0, T, SK, Qt, attnout);
        { int h; for (h = 0; h < g_H; h = h + 1) g_iter_attn += g_iter_attn_h[h]; }

        /* output projection + residual: xout = xin + attnout @ W_o */
        coo_project(attnout, T, g_D, g_wo[b], xout, g_D);
        g_iter_proj += (long)T * t_nnz[g_wo[b]];
        for (t = 0; t < T * g_D; t = t + 1) xout[t] = xout[t] + xin[t];
        }   /* end attention sublayer */

        /* SwiGLU FFN (shared helper; identical to run_incremental) */
        ffn_block(b, xout, T);
        { float *tmp = xin; xin = xout; xout = tmp; }   /* out becomes next in */
    }

    /* materialise hidden [1,S,D]; the driver reads only row -1 (row S-1). */
    {
        int rdim[3]; rdim[0] = 1; rdim[1] = S; rdim[2] = g_D;
        alloc_tensor(output_tid, DT_FLOAT, 3, rdim);
        memcpy(tf[output_tid], xin, (long)S * g_D * sizeof(float));
    }
    /* block-skip runs a FRESH full recompute each step; it does NOT maintain the
     * incremental prefix cache.  Invalidate it so a later non-skip step rebuilds. */
    g_prefix_len = 0; g_prev_S = 0;

    free(xin); free(xout); free(Qt); free(Kt); free(Vt); free(attnout);
}

/* =====================================================================
 * FREE-DRIVEN-EVICTION COMPACTED FORWARD  (the arbitrary-length / flat-memory fix)
 * =====================================================================
 * Runs a FRESH full block-stack forward (P=0, no cross-step cache) over a COMPACTED
 * stream of `S_eff` rows whose ABSOLUTE stream positions are `abs_pos[0..S_eff-1]`
 * (strictly ascending).  The compacted stream is the byte-exact KEEP-SET:
 *
 *   KEEP  = { every LIVE §Memory store-KV row (the heap; a store is live until a
 *             newer store SUPERSEDES its address or a zero-write FREES it) }
 *         ∪ { BOS (position 0) }
 *         ∪ { the last W rows (the recency window the LOCAL/ingest heads key on) }
 *   DROP  = every other row (old register/ingest frames + un-stored MEM-slot rows).
 *
 * WHY BYTE-EXACT (proven, max|delta|=0 over all 242 blocks & heads):
 *   - The 20 LOCAL heads (ALiBi slope 6.0) have softmax1 weight EXACTLY 0 beyond
 *     ~one frame, so dropping rows older than the recency window W (>= one frame)
 *     changes nothing on them.  (Any kept far-behind store row also sits outside
 *     the local window -> weight 0, so keeping it doesn't perturb the local heads.)
 *   - The 3 GLOBAL heads (memory-CAM, slope ~0.004) gate their VALUE projection to
 *     EXACTLY 0 on every NON-store row (measured: |V|=0 for BOS/register/query/
 *     un-stored MEM-slot rows; |V|=4 only on store rows).  A dropped non-store row
 *     therefore contributes 0 to the numerator AND — crucially — the softmax1 denom
 *     is unchanged too (verified: dropping the non-store rows leaves the global-head
 *     output bit-identical, so their score is at/below the softmax1 sink floor).
 *   ALiBi's recency is a function of the ABSOLUTE token distance, so we feed the
 *   real absolute positions (g_abs_pos) into attn_head; the compacted INDEX is only
 *   the storage layout.  Result: the memory-CAM reads the LIVE HEAP over O(heap)
 *   rows and the local heads read O(W) rows — total O(heap+W), FLAT in stream length.
 *
 * COST: a fresh P=0 recompute over S_eff rows => attention O(242 * S_eff^2), FFN/proj
 * O(242 * S_eff).  Because S_eff is BOUNDED (~heap+W, not the growing S), this is
 * flat per step regardless of how long the I/O stream runs (no OOM, no O(S^2) blowup).
 * live_mask (per-op block skip) composes: pass it non-NULL to skip dead blocks. */
void run_full_evict(int S_eff, const int *abs_pos, const unsigned char *live_mask) {
    int b, t;
    int T = S_eff;
    float *xin, *xout, *Qt, *Kt, *Vt, *attnout;

    if (!g_incr_ready) { run(); return; }

    xin = malloc((long)T * g_D * sizeof(float));
    xout = malloc((long)T * g_D * sizeof(float));
    Qt = malloc((long)T * g_D * sizeof(float));
    Kt = malloc((long)T * g_D * sizeof(float));
    Vt = malloc((long)T * g_D * sizeof(float));
    attnout = malloc((long)T * g_D * sizeof(float));

    /* the compacted input residual already lives in tf[input_tid] (S_eff rows) */
    memcpy(xin, tf[input_tid], (long)S_eff * g_D * sizeof(float));

    g_abs_pos = abs_pos;             /* attn_head now reads ABSOLUTE positions */

    for (b = 0; b < g_nblocks; b = b + 1) {
        int SK = T;
        if (live_mask && !live_mask[b]) continue;   /* per-op skip (byte-exact) */
        if (g_attn_fuse && g_attn_dead[b]) {
            /* DEAD-BLOCK-FUSION: attn no-op (W_v/W_o all-zero); residual passthrough
             * into the FFN.  BYTE-EXACT (skipped ops add exact zero). */
            memcpy(xout, xin, (long)T * g_D * sizeof(float));
        } else {
        coo_project(xin, T, g_D, g_wq[b], Qt, g_D);
        coo_project(xin, T, g_D, g_wk[b], Kt, g_D);
        coo_project(xin, T, g_D, g_wv[b], Vt, g_D);
        g_iter_proj += (long)T * (t_nnz[g_wq[b]] + t_nnz[g_wk[b]] + t_nnz[g_wv[b]]);

        cache_reserve(b, SK);
        memcpy(g_Kc[b], Kt, (long)SK * g_D * sizeof(float));
        memcpy(g_Vc[b], Vt, (long)SK * g_D * sizeof(float));

        attn_all_heads(b, 0, T, SK, Qt, attnout);
        { int h; for (h = 0; h < g_H; h = h + 1) g_iter_attn += g_iter_attn_h[h]; }

        coo_project(attnout, T, g_D, g_wo[b], xout, g_D);
        g_iter_proj += (long)T * t_nnz[g_wo[b]];
        for (t = 0; t < T * g_D; t = t + 1) xout[t] = xout[t] + xin[t];
        }   /* end attention sublayer */

        /* SwiGLU FFN (shared helper; sparse down-residual + persistent scratch). */
        ffn_block(b, xout, T);
        { float *tmp = xin; xin = xout; xout = tmp; }
    }

    g_abs_pos = 0;                   /* restore identity for all other paths */

    /* materialise hidden [1,S_eff,D]; the driver reads only the LAST compacted row
     * (S_eff-1), which is the current query row (always kept in the recency window). */
    {
        int rdim[3]; rdim[0] = 1; rdim[1] = S_eff; rdim[2] = g_D;
        alloc_tensor(output_tid, DT_FLOAT, 3, rdim);
        memcpy(tf[output_tid], xin, (long)S_eff * g_D * sizeof(float));
    }
    /* fresh full recompute; invalidate the incremental prefix cache. */
    g_prefix_len = 0; g_prev_S = 0;

    free(xin); free(xout); free(Qt); free(Kt); free(Vt); free(attnout);
}

/* =====================================================================
 * THE WHOLE-VM-IN-C DRIVER  (port of nibble_pure_forward_complete.run_program)
 * =====================================================================
 * embed + overlay + VM step loop + byte-decode + I/O, all in C.  Reads the
 * embedded embed table / layout dims / program from allc_gen.h.  No Python. */

/* ---- the growing token stream + the store log (KV memory bookkeeping) ---- */
#define AC_MAX_STREAM 200000
#define AC_MAX_STORES 100000
static int   ac_stream[AC_MAX_STREAM];   /* token ids */
static int   ac_S;                       /* current stream length */
static long  ac_store_addr[AC_MAX_STORES];
static long  ac_store_val[AC_MAX_STORES];
static int   ac_store_fidx[AC_MAX_STORES]; /* frame index that owns this store */
static int   ac_nstore;

/* append the 30-token register frame (little-endian bytes) for one VM step.
 * mirror of blogspec_vocab.build_step_frame: on a store step the MEM slot carries
 * (addr,val); else it carries stack0 in the val slot.  Marker token ids from the
 * generated header would be nicer but they are fixed by the spec: */
#define TK_REG_PC 256
#define TK_REG_AX 257
#define TK_REG_SP 258
#define TK_REG_BP 259
#define TK_MEM    260
#define TK_STEP_END 261

static void ac_push_bytes_le(long v, int n) {
    int i;
    for (i = 0; i < n; i = i + 1) { ac_stream[ac_S++] = (int)((v >> (8 * i)) & 0xFF); }
}
static void ac_build_frame(long pc, long ax, long sp, long bp, long stk,
                           long mem_addr, long mem_val, int is_store) {
    ac_stream[ac_S++] = TK_REG_PC; ac_push_bytes_le(pc, 4);
    ac_stream[ac_S++] = TK_REG_AX; ac_push_bytes_le(ax, 4);
    ac_stream[ac_S++] = TK_REG_SP; ac_push_bytes_le(sp, 4);
    ac_stream[ac_S++] = TK_REG_BP; ac_push_bytes_le(bp, 4);
    ac_stream[ac_S++] = TK_MEM;
    if (is_store || mem_addr || mem_val) {
        ac_push_bytes_le(mem_addr, 4); ac_push_bytes_le(mem_val, 4);
    } else {
        ac_push_bytes_le(0, 4); ac_push_bytes_le(stk & 0xFFFFFFFFL, 4);
    }
    ac_stream[ac_S++] = TK_STEP_END;
}

/* the residual buffer [1 x S x D] the overlay writes and run_incremental reads */
static float *ac_resid;      /* size ac_resid_cap * G_D */
static int    ac_resid_cap;

static void ac_resid_reserve(int S) {
    if (ac_resid_cap >= S) return;
    { int nc = ac_resid_cap ? ac_resid_cap : 256;
      while (nc < S) nc = nc * 2;
      ac_resid = realloc(ac_resid, (long)nc * G_D * sizeof(float));
      ac_resid_cap = nc; }
}

/* nibbles-of-value helper (little-endian 4-bit) */
static void ac_write_nibbles(float *row, int base, long v, int n) {
    int j;
    for (j = 0; j < n; j = j + 1) row[base + j] = (float)((v >> (4 * j)) & 0xF);
}

/* build the FULL residual [1,S,D]: embed[stream] then overlay.  Byte-identical
 * port of embed[toks] + make_overlay_complete(code, L, store_log).  Every row is
 * SET (not accumulated): the embed row is copied, then the overlay OVERWRITES its
 * layout dims. */
static void ac_build_residual(int S) {
    int i, k, j, b;
    ac_resid_reserve(S);
    /* 1) embed lookup: row i = G_EMBED[token] */
    for (i = 0; i < S; i = i + 1) {
        int tok = ac_stream[i];
        memcpy(ac_resid + (long)i * G_D, G_EMBED + (long)tok * G_D,
               G_D * sizeof(float));
    }
    /* 2) overlay: the program DATA bands at EVERY position + ONE */
    for (i = 0; i < S; i = i + 1) {
        float *row = ac_resid + (long)i * G_D;
        row[G_ONE] = 1.0f;
        for (k = 0; k < G_PROGLEN; k = k + 1) {
            row[G_CODE_OP_DIM[k]]  = (float)G_PROG_OP[k];
            row[G_CODE_IMM_DIM[k]] = (float)G_PROG_IMM[k];
            ac_write_nibbles(row, G_CODE_IMM_NIB_DIM[k],
                             (long)G_PROG_IMM[k] & 0xFFFFFFFFL, G_IMM_NIBS);
        }
    }
    /* 3) frame ROLE / IS_FRAME_BYTE tags + MEM-store KV rows.  Frames start at
     * pos=1 (after BOS), FRAME_LEN each; frame_idx counts frames. */
    {
        int pos = 1, frame_idx = 0;
        while (pos + G_FRAME_LEN <= S) {
            for (j = 0; j < G_FRAME_LEN; j = j + 1) {
                int role = G_ROLE_OF[j];
                if (role >= 0) {
                    float *row = ac_resid + (long)(pos + j) * G_D;
                    row[G_ROLE + role] = 1.0f;
                    row[G_IS_FRAME_BYTE] = 1.0f;
                }
            }
            /* store row for this frame? (latest store with this fidx wins) */
            {
                int si; long addr = 0, val = 0; int have = 0;
                for (si = 0; si < ac_nstore; si = si + 1)
                    if (ac_store_fidx[si] == frame_idx) {
                        addr = ac_store_addr[si]; val = ac_store_val[si]; have = 1;
                    }
                if (have) {
                    int mem_pos = pos + G_MEM_MARKER_LOCAL;
                    float *row = ac_resid + (long)mem_pos * G_D;
                    row[G_IS_STORE] = 1.0f;
                    row[G_IS_FRAME_BYTE] = 0.0f;
                    for (b = 0; b < G_ADDR_BITS; b = b + 1)
                        row[G_ADDR_BIN + b] = (float)((addr >> b) & 1L);
                    ac_write_nibbles(row, G_VAL_NIB, val & 0xFFFFFFFFL, 16);
                }
            }
            pos += G_FRAME_LEN;
            frame_idx += 1;
        }
        /* the query row (last position) carries ALL ROLE one-hots for ingest */
        {
            float *row = ac_resid + (long)(S - 1) * G_D;
            for (j = 0; j < G_N_ROLES; j = j + 1) row[G_ROLE + j] = 1.0f;
        }
    }
}

/* ---- TAIL-ONLY residual build (the O(S)->O(T) overlay fix for the incremental path):
 * the incremental forward reads ONLY the tail input rows [P..S-1] (rows [0..P-1] are
 * the stable prefix, served from the cross-step K/V cache — their input residual is
 * never read).  So building + copying the FULL [1,S,D] residual every step is pure
 * O(S) waste that GROWS with the stream (measured: 30ms/step overlay at S=7651, ~40%
 * of the whole step).  This builds ONLY rows [P..S-1] into ac_resid (at their absolute
 * offsets, so run_incremental's tf[input_tid]+(P+t)*D indexing is unchanged), with the
 * IDENTICAL per-row overlay math as ac_build_residual -> BYTE-EXACT for every tail row
 * the forward reads.  Prefix rows in ac_resid/the input tensor are left stale (unread).
 * P must equal the prefix run_incremental will use (= g_prev_S-1, or 0 on fresh). */
static void ac_build_residual_tail(int S, int P) {
    int i, k, j, b;
    if (P <= 0) { ac_build_residual(S); return; }   /* fresh -> full build */
    ac_resid_reserve(S);
    /* 1+2) embed + program-data overlay for tail rows [P..S-1] */
    for (i = P; i < S; i = i + 1) {
        int tok = ac_stream[i];
        float *row = ac_resid + (long)i * G_D;
        memcpy(row, G_EMBED + (long)tok * G_D, G_D * sizeof(float));
        row[G_ONE] = 1.0f;
        for (k = 0; k < G_PROGLEN; k = k + 1) {
            row[G_CODE_OP_DIM[k]]  = (float)G_PROG_OP[k];
            row[G_CODE_IMM_DIM[k]] = (float)G_PROG_IMM[k];
            ac_write_nibbles(row, G_CODE_IMM_NIB_DIM[k],
                             (long)G_PROG_IMM[k] & 0xFFFFFFFFL, G_IMM_NIBS);
        }
    }
    /* 3) frame ROLE / IS_FRAME_BYTE tags + store-KV rows, ONLY for frames overlapping
     * the tail.  A frame at absolute base `pos` (=1+FRAME_LEN*frame_idx) touches rows
     * [pos..pos+FRAME_LEN); we visit only frames whose span intersects [P..S-1]. */
    {
        int first_fi = (P - 1) / G_FRAME_LEN; if (first_fi < 0) first_fi = 0;
        int pos = 1 + G_FRAME_LEN * first_fi, frame_idx = first_fi;
        while (pos + G_FRAME_LEN <= S) {
            for (j = 0; j < G_FRAME_LEN; j = j + 1) {
                int abspos = pos + j;
                if (abspos < P || abspos >= S) continue;    /* tail rows only */
                { int role = G_ROLE_OF[j];
                  if (role >= 0) {
                      float *row = ac_resid + (long)abspos * G_D;
                      row[G_ROLE + role] = 1.0f;
                      row[G_IS_FRAME_BYTE] = 1.0f;
                  } }
            }
            {
                int mem_pos = pos + G_MEM_MARKER_LOCAL;
                if (mem_pos >= P && mem_pos < S) {
                    int si; long addr = 0, val = 0; int have = 0;
                    for (si = 0; si < ac_nstore; si = si + 1)
                        if (ac_store_fidx[si] == frame_idx) {
                            addr = ac_store_addr[si]; val = ac_store_val[si]; have = 1;
                        }
                    if (have) {
                        float *row = ac_resid + (long)mem_pos * G_D;
                        row[G_IS_STORE] = 1.0f;
                        row[G_IS_FRAME_BYTE] = 0.0f;
                        for (b = 0; b < G_ADDR_BITS; b = b + 1)
                            row[G_ADDR_BIN + b] = (float)((addr >> b) & 1L);
                        ac_write_nibbles(row, G_VAL_NIB, val & 0xFFFFFFFFL, 16);
                    }
                }
            }
            pos += G_FRAME_LEN;
            frame_idx += 1;
        }
        /* the query row (last position, always in the tail) carries ALL ROLE one-hots */
        {
            float *row = ac_resid + (long)(S - 1) * G_D;
            for (j = 0; j < G_N_ROLES; j = j + 1) row[G_ROLE + j] = 1.0f;
        }
    }
}

/* =====================================================================
 * FREE-DRIVEN EVICTION: build the COMPACTED residual over the KEEP-SET only
 * =====================================================================
 * Instead of materialising the full [1,S,D] (O(S) rows), build ONLY the kept rows
 * into ac_resid (compacted, contiguous) and record each kept row's ABSOLUTE stream
 * position in abs_pos_out.  KEEP-SET (proven byte-exact, see run_full_evict):
 *   - BOS (position 0)
 *   - every LIVE store-KV row: the LATEST store to each address (a superseded
 *     older-same-address store is dropped — latest-write-wins CAM makes it weight-0)
 *   - every row in the last W absolute positions (the recency window; W>=FRAME_LEN
 *     is byte-exact for the LOCAL/ingest heads whose weight is 0 beyond ~one frame).
 * Returns the compacted length S_eff (rows written to ac_resid, ascending abs pos).
 *
 * Memory: O(#distinct-live-addresses + W) rows — FLAT in the stream length S.  Each
 * kept store row / window row is built with the SAME overlay math as ac_build_residual
 * (byte-identical per-row values); only the ROWS PRESENT differ (evicted rows gone). */
static int ac_build_residual_evict(int S, int win, int *abs_pos_out) {
    int i, k, b, seff = 0;
    int lo_win = S - win; if (lo_win < 1) lo_win = 1;   /* window start (>=1; 0=BOS) */

    /* precompute, per store-log entry, whether it is the LATEST store to its address
     * (only the latest survives; older same-address stores are superseded & dropped).
     * ac_store_* is in commit order, so the LAST entry for an address is the latest. */
    /* (linear scan per store row below; ac_nstore is O(distinct addrs + churn)). */

    ac_resid_reserve(win + G_ADDR_BITS /*slack*/ + ac_nstore + 2);

    /* helper: build one row (embed + overlay) at absolute position `pos`, frame_idx,
     * is_query, into the compacted slot `seff`. */
    #define AC_EMIT_ROW(POS, FIDX, ISQ, STORE_SI) do {                              \
        int _pos = (POS); float *row = ac_resid + (long)seff * G_D;                 \
        int _tok = ac_stream[_pos];                                                 \
        memcpy(row, G_EMBED + (long)_tok * G_D, G_D * sizeof(float));               \
        row[G_ONE] = 1.0f;                                                          \
        for (k = 0; k < G_PROGLEN; k = k + 1) {                                     \
            row[G_CODE_OP_DIM[k]]  = (float)G_PROG_OP[k];                           \
            row[G_CODE_IMM_DIM[k]] = (float)G_PROG_IMM[k];                          \
            ac_write_nibbles(row, G_CODE_IMM_NIB_DIM[k],                            \
                             (long)G_PROG_IMM[k] & 0xFFFFFFFFL, G_IMM_NIBS);        \
        }                                                                           \
        /* frame ROLE / IS_FRAME_BYTE tag for this row's local slot */              \
        if ((FIDX) >= 0) {                                                          \
            int _rel = (_pos - 1) - G_FRAME_LEN * (FIDX);                           \
            if (_rel >= 0 && _rel < G_FRAME_LEN) {                                  \
                int _role = G_ROLE_OF[_rel];                                        \
                if (_role >= 0) { row[G_ROLE + _role] = 1.0f; row[G_IS_FRAME_BYTE] = 1.0f; } \
            }                                                                       \
        }                                                                           \
        /* store-KV row (overrides the frame-byte tag) */                          \
        if ((STORE_SI) >= 0) {                                                      \
            long _a = ac_store_addr[(STORE_SI)], _v = ac_store_val[(STORE_SI)];     \
            row[G_IS_STORE] = 1.0f; row[G_IS_FRAME_BYTE] = 0.0f;                    \
            for (b = 0; b < G_ADDR_BITS; b = b + 1)                                 \
                row[G_ADDR_BIN + b] = (float)((_a >> b) & 1L);                      \
            ac_write_nibbles(row, G_VAL_NIB, _v & 0xFFFFFFFFL, 16);                 \
        }                                                                           \
        /* the query row (== last abs position, S-1) carries ALL ROLE one-hots */  \
        if (ISQ) { for (k = 0; k < G_N_ROLES; k = k + 1) row[G_ROLE + k] = 1.0f; }  \
        abs_pos_out[seff] = _pos; seff = seff + 1;                                  \
    } while (0)

    /* 1) BOS */
    AC_EMIT_ROW(0, -1, 0, -1);

    /* 2) LIVE store rows OLDER than the window (window store rows are emitted in
     *    step 3 in position order).  A store at frame fi has its KV row at absolute
     *    position store_pos = 1 + FRAME_LEN*fi + MEM_MARKER_LOCAL.  Emit it iff it is
     *    the LATEST store to its address (supersede-drop) AND its store_pos < lo_win. */
    for (i = 0; i < ac_nstore; i = i + 1) {
        int fi = ac_store_fidx[i];
        int store_pos = 1 + G_FRAME_LEN * fi + G_MEM_MARKER_LOCAL;
        int latest = 1, j;
        if (store_pos >= lo_win) continue;                  /* in window -> step 3 */
        if (store_pos >= S) continue;                       /* not yet materialised */
        for (j = i + 1; j < ac_nstore; j = j + 1)
            if (ac_store_addr[j] == ac_store_addr[i]) { latest = 0; break; }
        if (!latest) continue;                              /* superseded -> drop */
        AC_EMIT_ROW(store_pos, fi, 0, i);
    }

    /* 3) the RECENCY WINDOW: every row in [lo_win, S).  These are contiguous and
     *    ascending; each may itself be a store row (find its store-log entry). */
    for (i = lo_win; i < S; i = i + 1) {
        int fidx = (i >= 1) ? (i - 1) / G_FRAME_LEN : -1;
        int store_si = -1, si;
        int is_query = (i == S - 1);
        /* is this the MEM-marker slot of its frame with a store committed? */
        if (fidx >= 0) {
            int mem_pos = 1 + G_FRAME_LEN * fidx + G_MEM_MARKER_LOCAL;
            if (i == mem_pos)
                for (si = 0; si < ac_nstore; si = si + 1)
                    if (ac_store_fidx[si] == fidx) store_si = si;   /* latest wins */
        }
        AC_EMIT_ROW(i, fidx, is_query, store_si);
    }
    #undef AC_EMIT_ROW
    return seff;
}

/* argmax_n (2*n*x - n*n) over n in 0..15 — the nibble re-quantiser (no round). */
static int ac_snap_nib(double x) {
    int n, bn = 0; double best = -1e30;
    for (n = 0; n < 16; n = n + 1) {
        double s = 2.0 * n * x - (double)n * n;
        if (s > best) { best = s; bn = n; }
    }
    return bn;
}

/* _snap_lane: flat argmax_v (2*v*x - v*v) over the value vocab (VALVOCAB ~ 0x10100).
 * We reproduce it with the closed-form integer snap floor(x+0.5) clamped to >=0,
 * which is argmax-identical for the small single-scalar lanes (PC<4096, SP/BP≈
 * 0x10000, STK) — the exact same result the flat argmax returns for x<VALVOCAB. */
static long ac_snap_lane(double x) {
    long v;
    if (x < 0.0) x = 0.0;
    v = (long)floor(x + 0.5);
    return v;
}

/* _decode_reg_from_nibbles: 4 bytes, each byte = lo + 16*hi via per-nibble argmax */
static long ac_decode_reg(const float *state, int reg_base) {
    long val = 0; int bi;
    for (bi = 0; bi < 4; bi = bi + 1) {
        int lo = ac_snap_nib((double)state[reg_base + 2 * bi + 0]);
        int hi = ac_snap_nib((double)state[reg_base + 2 * bi + 1]);
        val |= (long)(lo + 16 * hi) << (8 * bi);
    }
    return val;
}

/* MEM[sp]: the most-recent store to address sp (latest-write-wins) */
static long ac_mem_top(long sp) {
    long val = 0; int si;
    for (si = 0; si < ac_nstore; si = si + 1)
        if (ac_store_addr[si] == sp) val = ac_store_val[si];
    return val;
}

/* ---- §Memory byte access for the I/O syscalls (PRTF/READ burst + strict) ----
 * §Memory in the all-C VM is the store-log KV (ac_store_*): seed frames + every
 * SI/SC/PSH store the VM has made.  ac_mem_read(addr) returns the latest-write byte
 * at ``addr`` (latest-write-wins, same rule the model's memory CAM uses), i.e. the
 * SAME byte the program's LC/LI would decode.  This is what makes the burst syscall
 * (runtime C-loop over §Memory) byte-exact vs the strict per-byte pointer-walk. */
static long ac_mem_read(long addr) {
    long val = 0; int si;
    addr &= 0xFFFFFFFFL;
    for (si = 0; si < ac_nstore; si = si + 1)
        if (ac_store_addr[si] == addr) val = ac_store_val[si];
    return val & 0xFF;
}

/* I/O-burst mode (MODE 2): PRTF/READ do the WHOLE buffer in ONE op inside the
 * runtime handler (a C loop over §Memory) with NO per-byte VM step / think-frame.
 * Gated by env C4_IO_BURST=1 or --io-burst.  0 = strict per-byte (MODE 1). */
static int ac_io_burst = 0;

/* DIRECT-CAM (opt-in ALLC_DIRECT_CAM=1): resolve LI/LC (memory-load) AX directly
 * from the store-log (address = pre-step AX, latest-write-wins) instead of the
 * neural memory-CAM.  Same CAM rule -> intended byte-exact; DEFAULT OFF.  A per-run
 * mismatch count (ac_dcam_diff) surfaces any neural-vs-store-log CAM disagreement. */
static int  ac_direct_cam = 0;
static long ac_dcam_diff = 0;

/* PER-OP BLOCK-SKIP (#738): each step, run ONLY the decoded op's LIVE blocks
 * (run_full_skip) instead of the full 242-block forward — the residual passes
 * straight through the skipped blocks (byte-exact for decode).  echo's ops
 * (IMM/PRTF/HALT) drop 242 -> ~7 live blocks (~34x fewer block applies).  Gated
 * by env ALLC_BLOCK_SKIP=1 or --block-skip; 0 = full forward (run_incremental).
 * Enabled only when the header's G_SKIP_NB matches the built g_nblocks (else the
 * live table is stale for this model -> fall back to the full forward, byte-safe). */
static int ac_block_skip = 0;
static int ac_skip_latched_off = 0;          /* set once crossover picks incremental */
static long ac_live_count[G_SKIP_NUM_OPS];   /* per-op #live blocks (crossover est) */

/* FREE-DRIVEN EVICTION (--evict / ALLC_EVICT): each step, run the block stack over a
 * COMPACTED KEEP-SET (BOS + live store rows + last-W rows) with ABSOLUTE-position
 * ALiBi, instead of the growing full stream.  Keeps the whole LIVE HEAP (every live
 * store) — evicting only superseded stores + old register/ingest frames — so LI/LC
 * still content-address ANY written address, byte-exact, while the KV row-count (and
 * RSS) stays FLAT at ~heap+window over arbitrary-length I/O (no OOM).  ALLC_EVICT_W
 * sets the recency window (default 2*FRAME_LEN=60, a byte-exact margin over the proven
 * one-frame local horizon).  Gated OFF by default (full stream, byte-identical). */
static int ac_evict = 0;
static int ac_evict_w = 0;                    /* recency window (0 => default below) */
static int *ac_abs_pos;                       /* abs stream pos per compacted row */
static int  ac_abs_pos_cap;
static int  ac_evict_peak_rows;               /* max compacted row-count seen (report) */

/* GLOBAL-HEAD KEY-SUBSET (DIRECT-CAM-ATTN): the per-forward keep-set mask fed to the
 * GLOBAL memory-CAM heads (see g_gkey in attn_head).  ac_gkey[k]=1 for BOS, every
 * §Memory store row (IS_STORE==1 in the residual), and the last AC_GLOBAL_WIN rows;
 * 0 for the dropped non-store rows (which carry V==0 and round away in the denom).
 * Built directly off the residual's IS_STORE dim so it matches what the model sees.
 * Default ON; ALLC_NO_GKEY=1 disables (g_gkey stays NULL -> full global scan).
 * NOT used on the evict path (there the rows are ALREADY compacted to the keep-set). */
static unsigned char *ac_gkey;
static int  ac_gkey_cap;
static long ac_gkey_kept;                     /* total kept keys across the run (report) */
static long ac_gkey_total;                    /* total keys scanned-if-full (report) */

/* precompute the per-op live-block count from the header mask (once, at startup). */
static void ac_init_live_counts(void) {
    int op, b;
    for (op = 0; op < G_SKIP_NUM_OPS; op = op + 1) {
        long c = 0;
        for (b = 0; b < G_SKIP_NB; b = b + 1) c += G_LIVE_MASK[(long)op * G_SKIP_NB + b];
        ac_live_count[op] = c;
    }
}

/* Build the GLOBAL-HEAD keep-set mask: keep BOS(0), every §Memory STORE row, and the
 * last AC_GLOBAL_WIN rows.  This is the SAME keep-set the eviction path is proven
 * byte-exact on, so restricting the GLOBAL memory-CAM heads to these keys is byte-exact.
 *
 * The store-row set is derived DIRECTLY from the store-log (a store at frame fi lives at
 * absolute row 1+FRAME_LEN*fi+MEM_MARKER_LOCAL) instead of a strided read of the full
 * [1,S,D] residual — critical now that the residual is only tail-built (its prefix rows
 * would otherwise be a ~60MB strided cache-miss scan every step, an O(S) leak).  Result
 * is IDENTICAL: a row carries IS_STORE==1 iff it is a committed store's marker slot.
 * Returns the kept count. */
static int ac_build_gkey(int S) {
    int i, kept = 0, win = AC_GLOBAL_WIN;
    int lo_win = S - win; if (lo_win < 0) lo_win = 0;
    if (ac_gkey_cap < S) {
        ac_gkey_cap = S < 256 ? 256 : S;
        ac_gkey = realloc(ac_gkey, (long)ac_gkey_cap * sizeof(unsigned char));
    }
    /* window + BOS in one pass (contiguous, cache-friendly), then scatter store rows */
    if (lo_win > 0) memset(ac_gkey, 0, (size_t)lo_win);
    for (i = (lo_win > 0 ? lo_win : 0); i < S; i = i + 1) ac_gkey[i] = 1;   /* window */
    if (lo_win > 0) ac_gkey[0] = 1;                                        /* BOS */
    { int si;
      for (si = 0; si < ac_nstore; si = si + 1) {
          int fi = ac_store_fidx[si];
          long sp = 1 + (long)G_FRAME_LEN * fi + G_MEM_MARKER_LOCAL;
          if (sp >= 0 && sp < lo_win) ac_gkey[sp] = 1;   /* store rows below window */
      } }
    for (i = 0; i < S; i = i + 1) kept += ac_gkey[i];
    return kept;
}

/* PRTF-burst: AX = pointer.  Walk §Memory mem[ptr], mem[ptr+1], ... until a NUL
 * byte and write ALL of them to stdout in one go (a C loop in the opcode handler).
 * Returns the number of bytes written.  8-bit pointer window: wrap at 256. */
static int ac_prtf_burst(long ptr) {
    unsigned char buf[512]; int n = 0; long a = ptr & 0xFF;
    while (n < (int)sizeof(buf)) {
        long b = ac_mem_read(a);
        if (b == 0) break;              /* NUL terminator ends the C string */
        buf[n++] = (unsigned char)b;
        a = (a + 1) & 0xFF;             /* 8-bit pointer window */
    }
    if (n > 0) { ssize_t w = write(1, buf, n); (void)w; }
    return n;
}

/* READ syscall: (fd, buf, n).  Read up to n bytes from stdin (fd 0 = neural
 * input-KV) into the caller buffer ``dst`` in ONE op.  Returns the number of bytes
 * actually read (short-read at EOF).  The caller lays each byte into §Memory as its
 * OWN store frame (one MEM-store KV per byte, mirroring the pure-forward driver) so
 * a later LC(buf+i) attends to exactly the byte the file delivered. */
static int ac_read_syscall(long fd, long n, unsigned char *dst, int cap) {
    int got = 0;
    if (fd != 0) return 0;
    if (n > (long)cap) n = (long)cap;
    while (got < (int)n) {
        ssize_t r = read(0, dst + got, (size_t)((int)n - got));
        if (r <= 0) break;
        got += (int)r;
    }
    return got;
}

/* run the WHOLE VM in C.  Prints PRTF bytes to stdout.  Returns the number of
 * steps executed.  Byte-exact to run_program in measure_incremental_windowed. */
static int ac_run_vm(int max_steps) {
    long cur_pc = 0, cur_sp = G_SP_INIT, cur_bp = G_SP_INIT, cur_ax = 0;
    long mask = 0xFF;
    int frame_idx;
    int step, k;

    /* seed stream: BOS + seed data-segment store frames + init frame.
     * The n_seed seed frames occupy frame indices 0..n_seed-1 (store_log keys);
     * the init frame is index n_seed; the first emitted step frame is n_seed+1. */
    ac_S = 0; ac_nstore = 0;
    ac_stream[ac_S++] = G_BOS;
    for (k = 0; k < G_NSEED; k = k + 1) {
        ac_build_frame(0, 0, G_SP_INIT, G_SP_INIT, 0,
                       G_SEED_ADDR[k], G_SEED_VAL[k], 1);
        ac_store_addr[ac_nstore] = G_SEED_ADDR[k] & 0xFFFFFFFFL;
        ac_store_val[ac_nstore]  = G_SEED_VAL[k] & 0xFFFFFFFFL;
        ac_store_fidx[ac_nstore] = k;               /* seed frame idx = k */
        ac_nstore += 1;
    }
    frame_idx = G_NSEED;                            /* init frame index */
    ac_build_frame(0, 0, G_SP_INIT, G_SP_INIT, 0, 0, 0, 0);

    int timing = getenv("ALLC_TIMING") ? atoi(getenv("ALLC_TIMING")) : 0;
    g_phase_on = getenv("ALLC_PHASE") ? atoi(getenv("ALLC_PHASE")) : 0;
    double t_overlay = 0, t_fwd = 0, t_decode = 0;
    for (step = 0; step < max_steps; step = step + 1) {
        int S = ac_S;
        long pc, sp, bp, stk, ax; int halted;
        int op, imm;
        const float *state;
        long s_addr = 0, s_val = 0; int is_store = 0;
        double ta, tb, tc, td;

        int S_fwd = S;               /* rows fed to the forward (S, or S_eff if evict) */
        int build_P = 0;             /* tail-build start (0 = full build) */
        ta = _now_ms();

        /* decode the CURRENT opcode (from the pre-step PC) FIRST — it needs only
         * cur_pc — so we know the forward path (evict / block-skip / incremental)
         * and can build ONLY the input rows that path will actually read. */
        op  = (cur_pc >= 0 && cur_pc < G_PROGLEN) ? G_PROG_OP[cur_pc] : -1;
        imm = (cur_pc >= 0 && cur_pc < G_PROGLEN) ? G_PROG_IMM[cur_pc] : 0;
        (void)imm;

        if (ac_evict) {
            /* build the COMPACTED keep-set residual (flat memory); S_eff <= S rows */
            int win = ac_evict_w > 0 ? ac_evict_w : (2 * G_FRAME_LEN);
            if (ac_abs_pos_cap < S + 4) {
                ac_abs_pos_cap = S + 4;
                ac_abs_pos = realloc(ac_abs_pos, (long)ac_abs_pos_cap * sizeof(int));
            }
            S_fwd = ac_build_residual_evict(S, win, ac_abs_pos);
            if (S_fwd > ac_evict_peak_rows) ac_evict_peak_rows = S_fwd;
        } else {
            /* TAIL-ONLY BUILD: the incremental forward reads ONLY rows [P..S-1], so
             * build just the tail (O(T) not O(S)) when the plain-incremental path
             * will run.  block-skip / a fresh cache (P=0) need the full build. */
            int will_skip = (ac_block_skip && g_incr_ready && !ac_skip_latched_off
                             && G_SKIP_NB == g_nblocks && op >= 0
                             && op < G_SKIP_NUM_OPS && G_SKIP_HAVE[op]
                             && ((long)ac_live_count[op] * S < (long)g_nblocks * 31));
            if (!will_skip && g_incr_ready) build_P = incr_prefix_len(S);
            if (build_P > 0) ac_build_residual_tail(S, build_P);
            else             ac_build_residual(S);
        }

        /* feed the residual into the incremental forward's input tensor + run.  For a
         * tail build we copy ONLY the tail rows [P..S-1] — run_incremental reads no
         * prefix input row, so the prefix stays stale (unread), saving the O(S) copy. */
        {
            int rdim[3]; rdim[0] = 1; rdim[1] = S_fwd; rdim[2] = G_D;
            alloc_tensor(input_tid, DT_FLOAT, 3, rdim);
            t_is_init[input_tid] = 1;
            if (build_P > 0)
                memcpy(tf[input_tid] + (long)build_P * G_D,
                       ac_resid + (long)build_P * G_D,
                       (long)(S_fwd - build_P) * G_D * sizeof(float));
            else
                memcpy(tf[input_tid], ac_resid, (long)S_fwd * G_D * sizeof(float));
        }
        tb = _now_ms();
        g_iter_attn = 0; g_iter_proj = 0; g_iter_ffn = 0;
        {
            /* BLOCK-SKIP: run only op's live blocks (byte-exact) if enabled, the
             * header's live table matches this model (G_SKIP_NB==g_nblocks), and
             * this op HAS a SOUND live set (G_SKIP_HAVE[op]).  Both paths are
             * byte-exact for the decode; they differ only in COST:
             *   - run_full_skip: P=0 recompute over ~live_count blocks -> attention
             *     is O(live_count * S^2), but FFN/proj is O(live_count * S) (vs 242).
             *   - run_incremental: O(S) attention over the CACHED prefix, but FFN/
             *     proj over ALL 242 blocks (incl the wide DIV/MOD megablocks).
             * CROSSOVER: skip wins while its attention estimate (live_count*S) is
             * below the incremental tail estimate (NB * ~31).  S grows monotonically
             * so the switch (if any) happens ONCE -> at most one cache rebuild, no
             * thrash.  A skip step invalidates the cache (run_full_skip sets
             * prev_S=0); the first incremental step after it rebuilds from P=0. */
            int has_live = (ac_block_skip && g_incr_ready && !ac_skip_latched_off
                            && G_SKIP_NB == g_nblocks
                            && op >= 0 && op < G_SKIP_NUM_OPS && G_SKIP_HAVE[op]);
            int use_skip = 0;
            if (has_live) {
                long lc = ac_live_count[op];
                /* skip attention est ~ lc*S*S ; incremental est ~ NB*31*S.
                 * -> skip cheaper while  lc*S < NB*31. */
                use_skip = ((long)lc * S < (long)g_nblocks * 31);
            }
            /* LATCH: once ANY step (with a live set) prefers incremental at this S,
             * S only grows -> skip never becomes optimal again.  Latch skip OFF so
             * the incremental cache warms ONCE and stays warm (no per-op crossover
             * thrash that would re-invalidate the cache every other step). */
            if (ac_block_skip && !ac_skip_latched_off && has_live && !use_skip)
                ac_skip_latched_off = 1;
            /* GLOBAL-HEAD KEY-SUBSET: build the keep-set mask (store rows + BOS +
             * recency window) and feed it to the GLOBAL memory-CAM heads so they scan
             * O(#stores+W) keys not O(S) — byte-exact (same keep-set as eviction).  The
             * evict path already compacts to this set physically, so g_gkey is left
             * NULL there (all rows ARE keep-set rows). */
            if (ac_evict) {
                /* EVICT: compacted full recompute over S_fwd keep-set rows with
                 * ABSOLUTE-position ALiBi.  Composes with block-skip (pass the op's
                 * live mask when it has a sound live set) — byte-exact either way. */
                const unsigned char *lm =
                    (ac_block_skip && G_SKIP_NB == g_nblocks && op >= 0
                     && op < G_SKIP_NUM_OPS && G_SKIP_HAVE[op])
                        ? &G_LIVE_MASK[(long)op * G_SKIP_NB] : 0;
                g_gkey = 0;
                run_full_evict(S_fwd, ac_abs_pos, lm);
            }
            else {
                if (g_gkey_on) {
                    int kept = ac_build_gkey(S);
                    g_gkey = ac_gkey;
                    ac_gkey_kept += kept; ac_gkey_total += S;
                } else g_gkey = 0;
                if (use_skip) run_full_skip(S, &G_LIVE_MASK[(long)op * G_SKIP_NB]);
                else if (g_incr_ready) run_incremental(S);
                else               run();
                g_gkey = 0;
            }
        }
        tc = _now_ms();

        state = tf[output_tid] + (long)(S_fwd - 1) * G_D;   /* row -1 (query) */
        pc  = ac_snap_lane((double)state[G_PC_VAL]);
        sp  = ac_snap_lane((double)state[G_SP_VAL]);
        bp  = ac_snap_lane((double)state[G_BP_VAL]);
        stk = ac_snap_lane((double)state[G_STK_VAL]);
        halted = ((double)state[G_HALTED] > 0.5) ? 1 : 0;
        ax = ac_decode_reg(state, G_AX);

        /* ---- DIRECT-CAM (opt-in ALLC_DIRECT_CAM=1): resolve a MEMORY-LOAD op's AX
         * from the store-log O(1) (address = pre-step AX, latest-write-wins) instead
         * of the neural memory-CAM.  This is the SAME latest-write CAM rule the neural
         * global heads implement (and that the burst I/O already relies on), so it is
         * intended byte-exact.  LI loads a full 32-bit word (ac_mem_top); LC loads one
         * byte (ac_mem_read).  DEFAULT OFF (the neural forward's AX is authoritative);
         * a mismatch counter (ac_dcam_diff) is bumped so any neural/store-log CAM
         * disagreement is visible rather than silently overriding. */
        if (ac_direct_cam && (op == G_OP_LI || op == G_OP_LC)) {
            long dcaddr = cur_ax & 0xFFFFFFFFL;   /* load address = pre-step AX */
            long dv = (op == G_OP_LI) ? (ac_mem_top(dcaddr) & 0xFFFFFFFFL)
                                      : (ac_mem_read(dcaddr) & 0xFF);
            if (dv != (ax & (op == G_OP_LI ? 0xFFFFFFFFL : 0xFFL)))
                ac_dcam_diff += 1;
            ax = dv;
        }

        /* ---- SYSCALL opcodes (READ/OPEN/CLOS): the ONE class NOT computed
         * neurally (§Tool Use Mode — the model has no rules for them, so its
         * forward's PC/AX for this row are meaningless).  The RUNTIME owns the
         * effect: it marshals the args off the store-log stack, performs the I/O,
         * OVERRIDES the registers (pc=cur_pc+1, sp+=popped, ax=result), and — for
         * READ — lays each delivered byte into §Memory as its OWN MEM-store KV frame
         * (one frame per byte, byte-identical to the pure-forward driver) so a later
         * LC(buf+i) attends to exactly that byte.  This whole op is done here (before
         * the normal single-frame build) and then we ``continue``. */
        if (op == G_OP_READ || op == G_OP_OPEN || op == G_OP_CLOS) {
            long syspc = cur_pc + 1, syssp = cur_sp, sysax = 0;
            unsigned char rbuf[512]; int got = 0, bi; long buf = 0;
            if (op == G_OP_READ) {
                /* READ(fd=pop, buf=pop, n=AX): isa.interpret order = buf=pop() (TOP),
                 * fd=pop() (next), n = pre-step AX. */
                buf = ac_mem_top(cur_sp);               /* top of stack */
                long fd = ac_mem_top(cur_sp + 4);       /* next slot down */
                long n  = cur_ax & mask;                /* count from PRE-step AX */
                got = ac_read_syscall(fd, n, rbuf, (int)sizeof(rbuf));
                syssp = cur_sp + 8;                     /* two pops (fd, buf) */
                sysax = got & mask;
                if (getenv("ALLC_TRACE") && step < 64)
                    fprintf(stderr, "  [READ] fd=%ld buf=%ld n=%ld -> %d bytes into "
                            "§Memory (input-KV, one KV frame/byte)\n", fd, buf, n, got);
            } else if (op == G_OP_OPEN) {
                syssp = cur_sp + 4; sysax = 3;          /* stub fd */
            } else {                                     /* CLOS */
                syssp = cur_sp + 4; sysax = 0;
            }
            /* 1) the register frame for the syscall itself (no store) */
            ac_build_frame(syspc, sysax, syssp, cur_bp, stk, 0, 0, 0);
            frame_idx += 1;
            /* 2) READ: one MEM-store KV frame per delivered byte */
            for (bi = 0; bi < got; bi = bi + 1) {
                long baddr = (buf + bi) & 0xFF, bval = rbuf[bi];
                ac_build_frame(syspc, sysax, syssp, cur_bp, stk, baddr, bval, 1);
                frame_idx += 1;
                ac_store_addr[ac_nstore] = baddr;
                ac_store_val[ac_nstore]  = bval;
                ac_store_fidx[ac_nstore] = frame_idx;
                ac_nstore += 1;
            }
            cur_pc = syspc; cur_sp = syssp; cur_ax = sysax;
            if (getenv("ALLC_TRACE") && step < 16)
                fprintf(stderr, "  step%d pc=%ld op=%d [SYSCALL] -> pc=%ld ax=%ld "
                        "sp=%ld (+%d byte-frames)\n", step, cur_pc - 1, op, syspc,
                        sysax, syssp, got);
            if (syspc < 0 || syspc >= G_PROGLEN) { step = step + 1; break; }
            continue;                                    /* next VM step */
        }

        /* store bookkeeping (port of the driver's SI/PSH/JSR/ENT contract) */
        if (op == G_OP_SI || op == G_OP_SC) {
            is_store = 1; s_addr = ac_mem_top(cur_sp); s_val = ax & mask;
        } else if (op == G_OP_PSH) {
            is_store = 1; s_addr = cur_sp - 4; s_val = ax & mask;
        } else if (op == G_OP_JSR) {
            is_store = 1; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFFL;
        } else if (op == G_OP_ENT) {
            is_store = 1; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFFL;
        }

        ac_build_frame(pc, ax, sp, bp, stk,
                       is_store ? s_addr : 0, is_store ? s_val : 0, is_store);
        frame_idx += 1;
        if (is_store) {
            ac_store_addr[ac_nstore] = s_addr & 0xFFFFFFFFL;
            ac_store_val[ac_nstore]  = s_val & 0xFFFFFFFFL;
            ac_store_fidx[ac_nstore] = frame_idx;
            ac_nstore += 1;
        }

        if (getenv("ALLC_TRACE") && step < 16)
            fprintf(stderr, "  step%d pc=%ld op=%d -> pc=%ld ax=%ld sp=%ld "
                    "store=%d@%ld=%ld\n", step, cur_pc, op, pc, ax & 0xFF, sp,
                    is_store, s_addr, s_val);

        /* ---- I/O syscalls: PRTF (write) and READ (stdin -> §Memory) ---- */
        if (op == G_OP_PRTF) {
            if (ac_io_burst) {
                /* MODE 2 (BURST): AX = pointer.  The RUNTIME walks §Memory
                 * mem[ptr..NUL] and writes the WHOLE string to stdout in ONE op —
                 * no per-byte VM step / think-frame. */
                int nw = ac_prtf_burst(ax & 0xFF);
                if (getenv("ALLC_TRACE") && step < 64)
                    fprintf(stderr, "  [BURST PRTF] ptr=%ld -> %d bytes from "
                            "§Memory (runtime C-loop)\n", ax & 0xFF, nw);
            } else {
                /* MODE 1 (STRICT): the PROGRAM already walked §Memory (LC mem[ptr])
                 * so AX holds ONE decoded byte; emit it (one PRTF per byte). */
                unsigned char c = (unsigned char)(ax & 0xFF);
                ssize_t w = write(1, &c, 1); (void)w;
            }
        }
        /* READ/OPEN/CLOS syscalls are handled ABOVE (before the frame build) so
         * their runtime-owned state override is reflected in the emitted frame. */

        cur_pc = pc; cur_sp = sp; cur_bp = bp; cur_ax = ax; (void)cur_ax;
        td = _now_ms();
        t_overlay += tb - ta; t_fwd += tc - tb; t_decode += td - tc;
        if (timing)
            fprintf(stderr, "  step %d S=%d: overlay %.2f  fwd %.2f  decode %.2f "
                    "(total %.2f ms)  [attn %ld  proj %ld  ffn %ld ops]\n",
                    step, S, tb - ta, tc - tb, td - tc, td - ta,
                    g_iter_attn, g_iter_proj, g_iter_ffn);
        if (halted || pc < 0 || pc >= G_PROGLEN) { step = step + 1; break; }
    }
    if (timing)
        fprintf(stderr, "TIMING: overlay %.1f  fwd %.1f  decode %.1f ms total "
                "(%d steps, %.1f ms/step)\n", t_overlay, t_fwd, t_decode,
                step, (t_overlay + t_fwd + t_decode) / (step > 0 ? step : 1));
    if (g_phase_on)
        fprintf(stderr, "PHASE (within fwd, ms total): attn %.1f  ffn %.1f  "
                "ffn-alloc %.1f  (%d steps)\n",
                g_ph_attn, g_ph_ffn, g_ph_alloc, step);
    return step;
}

/* ============ main ============ */
int main(int argc, char **argv) {
    FILE *tf_in;
    int B; int S; int i; int dump; int a;
    int residual_in; int stats; int incremental;
    int dim[2];
    char *model; char *tokfile;

    /* ---- ALL-C mode: `./prog --allc [max_steps]` runs the WHOLE embedded VM in
     * C (embed + overlay + step loop + decode + I/O), no model-file arg needed. */
    {
        int is_allc = 0, max_steps = 20000, k;
        for (k = 1; k < argc; k = k + 1) {
            if (strcmp(argv[k], "--allc") == 0) is_allc = 1;
            else if (strcmp(argv[k], "--io-burst") == 0) ac_io_burst = 1;
            else if (strcmp(argv[k], "--block-skip") == 0) ac_block_skip = 1;
            else if (strcmp(argv[k], "--evict") == 0) ac_evict = 1;
            else if (strcmp(argv[k], "--direct-cam") == 0) ac_direct_cam = 1;
            else if (strncmp(argv[k], "--evict-w=", 10) == 0)
                { ac_evict = 1; ac_evict_w = atoi(argv[k] + 10); }
            else if (is_allc && argv[k][0] != '-') max_steps = atoi(argv[k]);
        }
        /* env override (C4_IO_BURST=1) — same gate, for the measure harness */
        if (getenv("C4_IO_BURST") && atoi(getenv("C4_IO_BURST"))) ac_io_burst = 1;
        if (getenv("ALLC_BLOCK_SKIP") && atoi(getenv("ALLC_BLOCK_SKIP")))
            ac_block_skip = 1;
        if (getenv("ALLC_EVICT") && atoi(getenv("ALLC_EVICT"))) ac_evict = 1;
        if (getenv("ALLC_EVICT_W")) ac_evict_w = atoi(getenv("ALLC_EVICT_W"));
        if (getenv("ALLC_DIRECT_CAM") && atoi(getenv("ALLC_DIRECT_CAM")))
            ac_direct_cam = 1;
        if (getenv("ALLC_WINPROBE") && atoi(getenv("ALLC_WINPROBE"))) {
            int _h; g_winprobe = 1;
            for (_h = 0; _h < 512; _h = _h + 1) g_winprobe_max[_h] = -1;
        }
        /* GLOBAL-HEAD KEY-SUBSET default ON; ALLC_NO_GKEY=1 disables it. */
        g_gkey_on = 1;
        if (getenv("ALLC_NO_GKEY") && atoi(getenv("ALLC_NO_GKEY"))) g_gkey_on = 0;
        if (is_allc) {
            int nsteps;
            a = 0; while (a < MAX_TENSORS) { tf[a] = 0; ti[a] = 0; a = a + 1; }
            SPARSE_ITERS = 0; DENSE_EQUIV_ITERS = 0;
            fprintf(stderr, "allc: I/O mode = %s\n",
                    ac_io_burst ? "BURST (MODE 2, runtime §Memory syscall)"
                                : "STRICT (MODE 1, per-byte pointer-walk)");
            load(0);                       /* embedded blob */
            if (incr_discover()) {
                fprintf(stderr, "allc: %d blocks H=%d HD=%d D=%d ffn<=%d "
                        "(windowed KV, O(S)/step)\n", g_nblocks, g_H, g_HD, g_D,
                        g_ffn_hidden_max);
                if (g_attn_fuse)
                    fprintf(stderr, "allc: DEAD-BLOCK-FUSION ON: %d/%d blocks have "
                            "a LIVE attention head; %d dead-attention blocks skip "
                            "Q/K/V/O+softmax (byte-exact)\n",
                            g_nblocks - g_attn_dead_n, g_nblocks, g_attn_dead_n);
            } else
                fprintf(stderr, "allc: block layout not regular; full-recompute\n");
            if (ac_block_skip) {
                if (G_SKIP_NB == g_nblocks) {
                    ac_init_live_counts();
                    fprintf(stderr, "allc: BLOCK-SKIP ON (per-op live-set, "
                            "G_SKIP_NB=%d matches %d blocks; auto-crossover to "
                            "incremental at large S)\n", G_SKIP_NB, g_nblocks);
                } else {
                    fprintf(stderr, "allc: BLOCK-SKIP requested but G_SKIP_NB=%d "
                            "!= %d blocks -> DISABLED (full forward, byte-safe)\n",
                            G_SKIP_NB, g_nblocks);
                    ac_block_skip = 0;
                }
            }
            if (ac_evict)
                fprintf(stderr, "allc: FREE-DRIVEN EVICTION ON (keep-set = BOS + live "
                        "store rows + last-%d recency rows; ABSOLUTE-pos ALiBi; KV "
                        "flat at ~heap+window over arbitrary length)\n",
                        ac_evict_w > 0 ? ac_evict_w : (2 * G_FRAME_LEN));
            if (ac_direct_cam)
                fprintf(stderr, "allc: DIRECT-CAM ON (LI/LC AX resolved O(1) from the "
                        "store-log, latest-write-wins; overrides the neural memory-CAM "
                        "decode)\n");
            nsteps = ac_run_vm(max_steps);
            fprintf(stderr, "allc: %d VM steps, final S=%d\n", nsteps, ac_S);
            if (g_gkey_on && ac_gkey_total > 0)
                fprintf(stderr, "allc: GLOBAL-HEAD KEY-SUBSET ON: scanned %ld/%ld keys "
                        "for the 3 global memory-CAM heads (%.1fx fewer)\n",
                        ac_gkey_kept, ac_gkey_total,
                        (double)ac_gkey_total / (double)(ac_gkey_kept > 0 ? ac_gkey_kept : 1));
            if (g_winprobe) {
                int _h, gmaxL = -1, gmaxG = -1;
                fprintf(stderr, "allc: WINPROBE max nonzero-weight distance per head:\n");
                for (_h = 0; _h < g_H; _h = _h + 1) {
                    if (g_winprobe_max[_h] < 0) continue;
                    fprintf(stderr, "   h%2d %s max_dist=%d  gnz=%d gtop=%.6f\n", _h,
                            g_head_local[_h] ? "LOCAL " : "GLOBAL", g_winprobe_max[_h],
                            g_winprobe_gnz[_h], g_winprobe_gtop[_h]);
                    if (g_head_local[_h]) { if (g_winprobe_max[_h] > gmaxL) gmaxL = g_winprobe_max[_h]; }
                    else { if (g_winprobe_max[_h] > gmaxG) gmaxG = g_winprobe_max[_h]; }
                }
                fprintf(stderr, "allc: WINPROBE LOCAL max_dist=%d  GLOBAL max_dist=%d\n",
                        gmaxL, gmaxG);
            }
            if (ac_direct_cam)
                fprintf(stderr, "allc: DIRECT-CAM: %ld LI/LC steps where store-log != "
                        "neural decode (0 == byte-exact CAM agreement)\n", ac_dcam_diff);
            if (ac_evict)
                fprintf(stderr, "allc: EVICT peak compacted rows = %d (vs full stream "
                        "S=%d) -> %d live stores kept\n", ac_evict_peak_rows, ac_S,
                        ac_nstore);
            return 0;
        }
    }

    if (argc < 3) {
        printf("usage: %s model.nblbin input.txt "
               "[--dump-argmax|--dump-logits] [--residual-in] [--incremental] "
               "[--stats]\n   or: %s --allc [max_steps]\n", argv[0], argv[0]);
        return 1;
    }
    model = argv[1];
    tokfile = argv[2];
    dump = 0;
    residual_in = 0;
    stats = 0;
    incremental = 0;
    /* scan all trailing flags (order-independent) */
    a = 3;
    while (a < argc) {
        if (strcmp(argv[a], "--dump-argmax") == 0) dump = 1;
        else if (strcmp(argv[a], "--dump-logits") == 0) dump = 2;
        else if (strcmp(argv[a], "--residual-in") == 0) residual_in = 1;
        else if (strcmp(argv[a], "--incremental") == 0) incremental = 1;
        else if (strcmp(argv[a], "--stats") == 0) stats = 1;
        a = a + 1;
    }
    SPARSE_ITERS = 0; DENSE_EQUIV_ITERS = 0;

    i = 0; a = 0;
    while (a < MAX_TENSORS) { tf[a] = 0; ti[a] = 0; a = a + 1; }

    load(model);

    /* discover the regular block layout for the incremental path (no-op if the
     * graph is not the regular block stack — then --incremental falls back to the
     * full run() per frame). */
    if (incremental) {
        if (!incr_discover()) {
            fprintf(stderr, "incremental: block layout NOT regular; "
                    "falling back to full-recompute run()\n");
            incremental = 0;
        } else {
            fprintf(stderr, "incremental: %d blocks, H=%d HD=%d D=%d "
                    "ffn_hidden<=%d scale=%.9g  (windowed KV-cache, O(S) per step)\n",
                    g_nblocks, g_H, g_HD, g_D, g_ffn_hidden_max, g_scale);
        }
    }

    if (residual_in) {
        /* residual-in mode: input is a BINARY file — int32 B,S,D header then
         * B*S*D float32 values.  The input tensor is the pre-embedded /
         * program-overlaid residual [B,S,D] the corpus driver feeds; the C
         * runtime runs the block stack and dumps the hidden residual.  Output is
         * written as raw little-endian float32 bytes (no text) for speed. */
        int D; int n; int serve;
        /* --serve + tokfile "-": load the graph ONCE, then loop reading residual
         * frames from stdin (int32 B,S,D header + B*S*D float32) and writing the
         * hidden residual to stdout, until EOF.  This is the persistent-process
         * mode the corpus driver uses so the ~2 MB graph load is amortized over the
         * whole autoregressive run (no per-step re-load / re-spawn). */
        serve = (strcmp(tokfile, "-") == 0) ? 1 : 0;
        if (serve) {
            while (fread(&B, sizeof(int), 1, stdin) == 1) {
                fread(&S, sizeof(int), 1, stdin);
                fread(&D, sizeof(int), 1, stdin);
                {
                    int rdim[3]; rdim[0] = B; rdim[1] = S; rdim[2] = D;
                    alloc_tensor(input_tid, DT_FLOAT, 3, rdim);
                }
                t_is_init[input_tid] = 1;
                n = B * S * D;
                fread(tf[input_tid], sizeof(float), n, stdin);
                if (incremental) run_incremental(S); else run();
                { int sz; sz = t_size[output_tid];
                  fwrite(tf[output_tid], sizeof(float), sz, stdout); }
                fflush(stdout);
            }
            return 0;
        }
        tf_in = fopen(tokfile, "rb");
        if (!tf_in) { printf("cannot open %s\n", tokfile); return 1; }
        fread(&B, sizeof(int), 1, tf_in);
        fread(&S, sizeof(int), 1, tf_in);
        fread(&D, sizeof(int), 1, tf_in);
        {
            int rdim[3]; rdim[0] = B; rdim[1] = S; rdim[2] = D;
            alloc_tensor(input_tid, DT_FLOAT, 3, rdim);
        }
        t_is_init[input_tid] = 1;
        n = B * S * D;
        fread(tf[input_tid], sizeof(float), n, tf_in);
        fclose(tf_in);
        if (incremental) run_incremental(S); else run();
        /* dump the block-stack output residual as raw float32 (stdout binary) */
        {
            int sz; sz = t_size[output_tid];
            fwrite(tf[output_tid], sizeof(float), sz, stdout);
        }
        if (stats) {
            fprintf(stderr, "SPARSE_MATMUL_ITERS %ld  DENSE_EQUIV_ITERS %ld  "
                    "ratio %.2f\n", SPARSE_ITERS, DENSE_EQUIV_ITERS,
                    (double)DENSE_EQUIV_ITERS / (double)(SPARSE_ITERS > 0 ? SPARSE_ITERS : 1));
        }
        return 0;
    }

    /* read tokens: "B S" then B*S ints */
    tf_in = fopen(tokfile, "r");
    if (!tf_in) { printf("cannot open %s\n", tokfile); return 1; }
    fscanf(tf_in, "%d %d", &B, &S);
    dim[0] = B; dim[1] = S;
    alloc_tensor(input_tid, DT_INT64, 2, dim);
    t_is_init[input_tid] = 1;
    i = 0;
    while (i < B * S) { int tk; fscanf(tf_in, "%d", &tk); ti[input_tid][i] = (long)tk; i = i + 1; }
    fclose(tf_in);
    fprintf(stderr, "tokens: B=%d S=%d\n", B, S);

    run();

    /* logits: [B, S, vocab] — argmax over vocab per (b,s) */
    {
        int vocab; int rank; int rows; int r; int c; int best; float bv;
        rank = t_rank[output_tid];
        vocab = t_dims[output_tid][rank - 1];
        rows = t_size[output_tid] / vocab;
        fprintf(stderr, "logits shape:");
        r = 0; while (r < rank) { fprintf(stderr, " %d", t_dims[output_tid][r]); r = r + 1; }
        fprintf(stderr, "\n");
        if (dump == 2) {
            /* raw logits, one float per line (%.9g for full precision) */
            r = 0;
            while (r < rows * vocab) { printf("%.9g\n", tf[output_tid][r]); r = r + 1; }
        } else if (dump == 1) {
            r = 0;
            while (r < rows) {
                best = 0; bv = tf[output_tid][r * vocab];
                c = 1;
                while (c < vocab) {
                    if (tf[output_tid][r * vocab + c] > bv) { bv = tf[output_tid][r * vocab + c]; best = c; }
                    c = c + 1;
                }
                printf("%d\n", best);
                r = r + 1;
            }
        } else {
            /* print last-position argmax + a checksum of all argmaxes */
            int sum; sum = 0;
            r = 0;
            while (r < rows) {
                best = 0; bv = tf[output_tid][r * vocab];
                c = 1;
                while (c < vocab) {
                    if (tf[output_tid][r * vocab + c] > bv) { bv = tf[output_tid][r * vocab + c]; best = c; }
                    c = c + 1;
                }
                sum = sum + best * (r + 1);
                r = r + 1;
            }
            printf("argmax_checksum=%d rows=%d\n", sum, rows);
        }
    }
    return 0;
}

/* ============ embedded model blob ============================================
 * The full-ISA COO .nblbin is linked in as a static byte array via GNU-as
 * `.incbin`.  The (bare) path is supplied at compile time with
 *   -DMODEL_BLOB_PATH=/abs/path/blockstack.nblbin
 * so `prog` needs NO external model file — the weights + graph are IN the binary.
 * `_model_blob` / `_model_blob_end` bracket the bytes (see the memory-cursor
 * reader above).  STRINGIZE the bare path so the .incbin gets a quoted string. */
#ifndef MODEL_BLOB_PATH
#error "compile with -DMODEL_BLOB_PATH=/path/to/blockstack.nblbin"
#endif
#define _STR2(x) #x
#define _STR(x) _STR2(x)
__asm__(
    ".section .rodata\n"
    ".global _model_blob\n"
    ".align 16\n"
    "_model_blob:\n"
    ".incbin \"" _STR(MODEL_BLOB_PATH) "\"\n"
    ".global _model_blob_end\n"
    "_model_blob_end:\n"
    ".previous\n"
);
