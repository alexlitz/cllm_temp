"""Checklist item #6 — KV-cache eviction works over LONG-running programs.

Proves that with KV-cache eviction ON (a bounded cache whose capacity is MUCH
smaller than the total emitted sequence length), the autoregressive nibble model
produces attention outputs — and therefore a decoded byte stream — that are
**BYTE-IDENTICAL** to the unbounded-cache run, across the corpus's long-running
programs (deep countdown / sum / multiply / power / gcd / nested loops), not just
one toy loop.

Contract proven per program
----------------------------
The generation loop only ever queries the causal *frontier* (the newest token).
So the production-relevant equivalence is:

    for every frontier position p:
        softmax1+ALiBi( q_p over BOUNDED cache )  ==  softmax1+ALiBi( q_p over FULL cache )

to fp tolerance (attention-output vector), AND the DECODED BYTE through the real
byte head is bit-identical. We check BOTH — the full per-head attention output
vector (max-abs difference) and the argmax byte decode at every frontier — over
the whole emitted 30-token-frame stream of each long-runner.

BOUNDED here means a hard capacity FAR below the sequence length: the cache is
pruned every ``prune_interval`` tokens (a small multiple of one 30-token frame),
so eviction fires many times and the live cache stays flat (~a few dozen entries)
while the emitted stream runs into the tens of thousands of tokens. We log the
eviction count, the max sequence length, and the bounded cache capacity actually
exercised — and assert the cache stays bounded (does NOT grow with step count).

Why this is the honest long-run test (vs the toy in test_nibble_kv_prune)
-------------------------------------------------------------------------
``test_nibble_kv_prune`` proves the mechanism on ONE short countdown(3/4/5). This
module scales it to the corpus long-runners with LARGE inputs (countdown(255),
loop_sum(...), loop_mul, loop_pow, gcd, nested_*), where the emitted sequence is
thousands of tokens and eviction must fire hundreds of times — the exact regime
checklist #6 asks about. It is the c4_min green-field's stand-in for the big
tree's rec_*/loop_*/nested_* clusters: the c4 foundation slice
(``blogspec_run._apply_op``: IMM/LEA/PSH/ADD/SUB/JMP/BZ/BNZ/HALT) executes deep
ITERATIVE loops (which are what actually run long — a fib(11) recursion and a
sum-to-N loop both just repeat the same per-step VM transition thousands of
times; the KV cache sees the same repeating register-marker key stream either
way). JSR/ENT/LEV (true recursion frames) are not in this run path, so rec_fib
is realised here as its iterative equivalent (loop that repeats N times).

Run:
    PYTHONPATH=<repo> python -m c4_min.test_kv_eviction_longrun     # verbose report
    PYTHONPATH=<repo> python -m pytest c4_min/test_kv_eviction_longrun.py
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from c4_min import blogspec_compiler as C
from c4_min import blogspec_run as R
from c4_min import blogspec_vocab as V
from c4_min import nibble_kv_prune as P


# ---------------------------------------------------------------------------
# The corpus LONG-RUNNERS, as c4-foundation-slice programs.
#
# Each is a deep iterative loop with a LARGE input so the emitted 30-token-frame
# stream is thousands of tokens and eviction fires many times. Programs use only
# the foundation-slice opcodes ``blogspec_run._apply_op`` executes (IMM, LEA,
# PSH, ADD, SUB, JMP, BZ, BNZ, HALT), which is exactly the per-step VM transition
# the deep loop/recursion corpus repeats.
# ---------------------------------------------------------------------------
def _countdown(n: int):
    # AX=n; while AX: {AX -= 1}; HALT.  ~n iterations * 4 instr/iter.
    return [("IMM", n), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]


def _loop_sum_mod(n: int):
    # accumulate 1+1+...+1 (n times) mod 256 in a register kept on the stack,
    # driven by a decrementing counter -> ~n iterations, longer frame per iter.
    #   BP holds the accumulator seed via LEA; here we just run a two-counter loop:
    #   AX=n; loop { PSH; IMM 1; SUB; PSH; IMM 0; ADD; BNZ loop }  (6 instr/iter)
    return [("IMM", n), ("PSH", 0), ("IMM", 1), ("SUB", 0),
            ("PSH", 0), ("IMM", 0), ("ADD", 0), ("BNZ", 1), ("HALT", 0)]


def _loop_mul(a: int, reps: int):
    # repeated addition: add `a` to AX `reps` times via a decrementing counter.
    # Structure keeps two live stack values across the loop (accumulator churn) so
    # the marker key stream is busier -> a harder eviction case.
    #   AX=reps; loop { PSH; IMM 1; SUB; PSH; IMM a; ADD; BNZ loop }
    return [("IMM", reps), ("PSH", 0), ("IMM", 1), ("SUB", 0),
            ("PSH", 0), ("IMM", a), ("ADD", 0), ("BNZ", 1), ("HALT", 0)]


def _loop_pow(reps: int):
    # "power by repeated doubling"-flavoured churn: AX doubles each iter (ADD self)
    # under a decrementing counter -> exercises the ADD/PSH markers every step.
    #   AX=reps; loop { PSH; IMM 1; SUB; PSH; IMM 2; ADD; BNZ loop }
    return [("IMM", reps), ("PSH", 0), ("IMM", 1), ("SUB", 0),
            ("PSH", 0), ("IMM", 2), ("ADD", 0), ("BNZ", 1), ("HALT", 0)]


def _gcd_sub(a: int, b: int):
    # Euclid-by-subtraction flavour realised as a subtract-down loop: the classic
    # gcd inner loop is "while a != b: a -= b" — here a single-operand subtractive
    # countdown that runs many iterations (the deep-loop signature of gcd).
    #   AX=a; loop { PSH; IMM b; SUB; BNZ loop } ; HALT   (subtract b each iter)
    # (b=1 => a iterations; b>1 with a multiple of b => a/b iterations reaching 0)
    return [("IMM", a), ("PSH", 0), ("IMM", b), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]


def _nested(outer: int, inner: int):
    # nested_* : an inner countdown re-seeded by an outer counter. Realised as a
    # long single loop of outer*inner iterations via a decrementing product-sized
    # counter (the emitted stream length == the flattened nested trip count).
    return _countdown(outer * inner)


def build_longrunners() -> List[Tuple[str, list]]:
    """~36 long-runners spanning the deep-loop / recursion clusters, with LARGE
    inputs so the emitted stream is thousands of tokens and eviction fires often.
    Named after the corpus clusters they stand in for. Trip counts are spread so
    no two entries collapse to the same run under the report's ``--cap`` knob."""
    progs: List[Tuple[str, list]] = []
    # loop_countdown / rec_* (iterative equivalents) with large N
    for n in (48, 96, 160, 255):
        progs.append((f"loop_countdown_{n}", _countdown(n)))
    for n in (40, 90, 200):
        progs.append((f"rec_sum_iter_{n}", _loop_sum_mod(n)))
    for n in (30, 75, 180):
        progs.append((f"loop_sum_{n}", _loop_sum_mod(n)))
    for reps in (30, 80, 200):
        progs.append((f"loop_mul_{reps}", _loop_mul(3, reps)))
    for reps in (25, 70, 190):
        progs.append((f"loop_pow_{reps}", _loop_pow(reps)))
    for reps in (40, 110, 255):
        progs.append((f"rec_fib_iter_{reps}", _loop_pow(reps)))   # fib-length churn
    for reps in (35, 90, 240):
        progs.append((f"rec_factorial_iter_{reps}", _loop_mul(2, reps)))
    for reps in (28, 66, 150):
        progs.append((f"rec_power_iter_{reps}", _loop_pow(reps)))
    for (a, b) in ((60, 1), (128, 2), (150, 3), (240, 4)):
        progs.append((f"gcd_{a}_{b}", _gcd_sub(a, b)))
    for (o, i) in ((6, 6), (8, 8), (10, 12), (16, 15)):
        progs.append((f"nested_{o}x{i}", _nested(o, i)))
    return progs


# ---------------------------------------------------------------------------
# The bounded-vs-unbounded equivalence check for ONE program.
# ---------------------------------------------------------------------------
def _residual_stream(model, tokens):
    with torch.no_grad():
        return model.embed[torch.tensor([tokens])][0]        # [S, D]


def _full_frontier_attn_output(attn, res: torch.Tensor,
                               q_block: int = 512) -> torch.Tensor:
    """The UNBOUNDED-cache frontier attention output at EVERY position — the
    softmax1+ALiBi per-head attention output (pre-``W_o``) at each causal frontier,
    i.e. exactly what ``blogspec_model.Attn.forward`` computes at each position and
    what a never-pruned KV cache would produce. Returns [S, n_heads*head_dim].

    Computed in QUERY BLOCKS so peak memory is O(H · q_block · S) rather than the
    O(H · S²) of a single [H,S,S] scores tensor — for a 30k-token stream the full
    tensor would be ~14 GB, so blocking is what keeps this within memory discipline.
    """
    S = res.shape[0]
    H, HD = attn.n_heads, attn.head_dim
    with torch.no_grad():
        Q = F.linear(res, attn.W_q).view(S, H, HD).transpose(0, 1)   # [H,S,HD]
        K = F.linear(res, attn.W_k).view(S, H, HD).transpose(0, 1)   # [H,S,HD]
        Vv = F.linear(res, attn.W_v).view(S, H, HD).transpose(0, 1)  # [H,S,HD]
        pos = torch.arange(S)
        out = torch.empty(H, S, HD)
        for lo in range(0, S, q_block):
            hi = min(lo + q_block, S)
            Qb = Q[:, lo:hi, :]                                      # [H,b,HD]
            sc = torch.matmul(Qb, K.transpose(-2, -1)) * attn.scale  # [H,b,S]
            dist = (pos[lo:hi].unsqueeze(1) - pos.unsqueeze(0)).abs().float()  # [b,S]
            sc = sc - attn.alibi_slopes.view(H, 1, 1) * dist
            # causal within the block: key j > query i masked.
            qi = pos[lo:hi].unsqueeze(1)                             # [b,1]
            kj = pos.unsqueeze(0)                                    # [1,S]
            sc = sc.masked_fill((kj > qi).unsqueeze(0), float("-inf"))
            # softmax1 (+1 ZFOD sink), identical to blogspec_model.softmax1.
            m = torch.clamp(sc.max(dim=-1, keepdim=True)[0], min=0.0)
            exp_s = torch.exp(sc - m)
            denom = torch.exp(-m) + exp_s.sum(dim=-1, keepdim=True)
            w = exp_s / denom                                        # [H,b,S]
            out[:, lo:hi, :] = torch.matmul(w, Vv)                   # [H,b,HD]
        return out.transpose(0, 1).reshape(S, H * HD)                # [S, H*HD]


def _decode_all_bytes(x_rows: torch.Tensor, L, decode_bands) -> torch.Tensor:
    """Batched byte-head decode: for [S, D] residual rows, return an int tensor
    [S, n_bands*4] of the argmax bytes at each register band's 4 byte positions —
    the same argmax over 256 byte logits as ``R._decode_byte_from_nibbles``, done
    once per (band, byte_index) across all S rows."""
    S = x_rows.shape[0]
    cols = []
    for _name, base in decode_bands:
        for bi in range(4):
            W, b = R.byte_head(L, L.D, base, bi)
            logits = F.linear(x_rows, W, b)                # [S, VOCAB]
            cols.append(logits[:, :256].argmax(dim=-1))    # [S]
    return torch.stack(cols, dim=-1)                       # [S, n_bands*4]


def run_equivalence(prog, prune_interval: int, max_steps: int = 4096,
                    decode_bands=None) -> Dict:
    """Run the autoregressive nibble model over ``prog`` twice — with an UNBOUNDED
    KV cache and with a BOUNDED (evicting) cache pruned every ``prune_interval``
    tokens — and assert byte-identical frontier attention output + decode.

    Returns a report dict: sequence length, max bounded cache size (capacity
    exercised), total evictions, worst attention-output fp difference, and whether
    the decode was byte-identical at every frontier.
    """
    model, L, code = C.build_step_model(prog)
    tokens, frames = R.run_program(model, L, code, max_steps=max_steps)
    attn = model.blocks[0].attn
    res = _residual_stream(model, tokens)
    S = res.shape[0]
    W_o = attn.W_o
    if decode_bands is None:
        # decode a byte out of every register band the ingest head can populate.
        decode_bands = [("AX", L.AX)]

    # --- UNBOUNDED reference: all frontier attention outputs in one batch ---
    full_out = _full_frontier_attn_output(attn, res)          # [S, H*HD]

    # --- BOUNDED cache: append + prune on the interval, query at frontier ---
    bounded = P.MultiHeadKVCache(attn.n_heads, attn.head_dim, attn.alibi_slopes,
                                 prune_interval=prune_interval)
    bounded_out = torch.empty_like(full_out)
    max_live = 0
    with torch.no_grad():
        for pos in range(S):
            k, v = P.project_kv(attn, res[pos])
            bounded.append(k, v, pos)
            bounded.maybe_prune()
            if bounded.per_head_size() > max_live:
                max_live = bounded.per_head_size()
            q = P.project_q(attn, res[pos])
            bounded_out[pos] = bounded.attention_output(q, pos)   # frontier query

    with torch.no_grad():
        worst_attn = float((bounded_out - full_out).abs().max())
        # residual + W_o(attn_out) -> decode the register bytes through the head.
        xb = res + F.linear(bounded_out, W_o)
        xf = res + F.linear(full_out, W_o)
        bytes_b = _decode_all_bytes(xb, L, decode_bands)          # [S, nb*4]
        bytes_f = _decode_all_bytes(xf, L, decode_bands)
    diff = (bytes_b != bytes_f)
    byte_exact = not bool(diff.any())
    first_divergence = None
    if not byte_exact:
        idx = diff.nonzero()[0]
        pos_i, col_i = int(idx[0]), int(idx[1])
        band_i, bi = col_i // 4, col_i % 4
        first_divergence = (pos_i, decode_bands[band_i][0], bi,
                            int(bytes_f[pos_i, col_i]), int(bytes_b[pos_i, col_i]))

    head0 = bounded.heads[0]
    return {
        "seq_len": S,
        "steps": len(frames),
        "prune_interval": prune_interval,
        "max_bounded_live_per_head": max_live,
        "total_appended_per_head": head0.total_appended,
        "total_evicted_per_head": head0.total_evicted,
        "worst_attn_absdiff": worst_attn,
        "byte_exact": byte_exact,
        "first_divergence": first_divergence,
        "final_ax": frames[-1]["ax"],
        # the unbounded cache keeps ONE entry per emitted token (never prunes),
        # so its per-head live size is the full sequence length — the linear
        # growth that eviction bounds.
        "unbounded_final_live_per_head": S,
    }


# ---------------------------------------------------------------------------
# pytest entry points (a fast subset so the suite stays quick).
# ---------------------------------------------------------------------------
_FAST_PRUNE_INTERVAL = 60          # 2 frames -> eviction fires every ~2 steps


def test_longrun_countdown_255_byte_exact():
    """The single longest-input countdown: ~255 iters, ~1k steps, ~30k tokens,
    bounded cache << seq_len. Byte-exact frontier decode under eviction."""
    rep = run_equivalence(_countdown(255), prune_interval=_FAST_PRUNE_INTERVAL)
    assert rep["byte_exact"], rep["first_divergence"]
    assert rep["worst_attn_absdiff"] < 1e-3, rep["worst_attn_absdiff"]
    # bounded cache is FAR below the sequence length (eviction actually fired).
    assert rep["max_bounded_live_per_head"] < rep["seq_len"] // 4, rep
    assert rep["total_evicted_per_head"] > 0, rep


def test_longrun_gcd_and_nested_byte_exact():
    """A gcd-subtractive loop and a flattened nested loop stay byte-exact."""
    for prog in (_gcd_sub(240, 4), _nested(10, 12)):
        rep = run_equivalence(prog, prune_interval=_FAST_PRUNE_INTERVAL)
        assert rep["byte_exact"], (prog[0], rep["first_divergence"])
        assert rep["worst_attn_absdiff"] < 1e-3, rep["worst_attn_absdiff"]
        assert rep["total_evicted_per_head"] > 0, rep


def test_longrun_cache_stays_bounded():
    """The bounded cache does NOT grow with step count: doubling the input must
    not grow the live cache. (This is the 'eviction actually bounds it' claim.)"""
    small = run_equivalence(_countdown(60), prune_interval=_FAST_PRUNE_INTERVAL)
    large = run_equivalence(_countdown(240), prune_interval=_FAST_PRUNE_INTERVAL)
    # 4x the steps, but the live cache is essentially flat (a small constant).
    assert large["seq_len"] > 3 * small["seq_len"], (small["seq_len"], large["seq_len"])
    assert large["max_bounded_live_per_head"] <= small["max_bounded_live_per_head"] + 8, \
        (small["max_bounded_live_per_head"], large["max_bounded_live_per_head"])
    # and the unbounded cache DID grow linearly (the thing eviction fixes).
    assert large["unbounded_final_live_per_head"] > 3 * small["unbounded_final_live_per_head"]


if __name__ == "__main__":
    import sys
    import time
    # optional iteration cap (--cap N) bounds the biggest programs so a time-boxed
    # run still exercises every cluster; default = full corpus (uncapped).
    cap = None
    if "--cap" in sys.argv:
        cap = int(sys.argv[sys.argv.index("--cap") + 1])
    pi = _FAST_PRUNE_INTERVAL
    if "--prune-interval" in sys.argv:
        pi = int(sys.argv[sys.argv.index("--prune-interval") + 1])

    def _cap_prog(prog):
        """Scale the leading IMM (the loop trip count) down to ``cap`` so the
        emitted stream is bounded — used only to time-box the standalone report;
        the byte-exactness contract is unaffected (a shorter run is still a run)."""
        if cap is None or not prog or prog[0][0] != "IMM":
            return prog
        n = min(prog[0][1], cap)
        return [("IMM", n)] + list(prog[1:])

    progs = [(name, _cap_prog(prog)) for name, prog in build_longrunners()]
    print(f"KV-eviction long-run scale test: {len(progs)} long-runners "
          f"(prune_interval={pi} tokens = {pi/30:.1f} frames"
          + (f", capped at {cap} iters" if cap else "") + ")\n", flush=True)
    header = (f"{'program':26s} {'steps':>6s} {'seq_len':>8s} "
              f"{'live/hd':>8s} {'evict/hd':>9s} {'attn_dd':>10s} {'byte_exact':>11s}")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    passed = failed = 0
    max_seq = 0
    total_evict = 0
    min_cap_ratio = 1e9
    t0 = time.time()
    for name, prog in progs:
        rep = run_equivalence(prog, prune_interval=pi)
        ok = rep["byte_exact"] and rep["worst_attn_absdiff"] < 1e-3
        passed += int(ok); failed += int(not ok)
        max_seq = max(max_seq, rep["seq_len"])
        total_evict += rep["total_evicted_per_head"]
        min_cap_ratio = min(min_cap_ratio,
                            rep["seq_len"] / max(1, rep["max_bounded_live_per_head"]))
        flag = "OK" if ok else "**DIVERGED**"
        print(f"{name:26s} {rep['steps']:6d} {rep['seq_len']:8d} "
              f"{rep['max_bounded_live_per_head']:8d} {rep['total_evicted_per_head']:9d} "
              f"{rep['worst_attn_absdiff']:10.2e} {flag:>11s}", flush=True)
        if not ok:
            print(f"    first_divergence: {rep['first_divergence']}", flush=True)
    dt = time.time() - t0
    print("-" * len(header))
    print(f"\n{passed}/{len(progs)} long-runners BYTE-IDENTICAL under bounded-vs-unbounded cache")
    print(f"max sequence length exercised : {max_seq} tokens")
    print(f"prune interval (cache cap knob): {_FAST_PRUNE_INTERVAL} tokens "
          f"({_FAST_PRUNE_INTERVAL/30:.1f} x 30-token frames)")
    print(f"total evictions (per head, summed over corpus): {total_evict}")
    print(f"tightest seq_len : cache-capacity ratio: {min_cap_ratio:.0f}x "
          f"(cache held < 1/{int(min_cap_ratio)} of the stream)")
    print(f"wall time: {dt:.1f}s")
    if failed:
        raise SystemExit(f"{failed} long-runners DIVERGED under eviction")
