"""COMPOSED CONTINUOUS DOOM RUN to the first byte-exact output.

Composes the full fast substrate on the 20-bit-IMM pure-forward model:
  * build_lib_model_streaming(addr32=True)            -- wall #1 (32-bit addressing),
  * efficient_alu recurrent divmod (default)          -- wall #2 (32-bit ALU),
  * depth-N lean stack (default)                      -- wall #3,
  * C4_DRAFT_CMP32=1                                  -- doom loop counters > 255,
  * C4_MEM_ADDR_BITS=18                               -- wall #1 CAM lanes,
  * local-attn + dead-block-fusion                    -- bounded KV / GPU-frugal,
  * verify_blocks big-K speculation                   -- the FAST throughput lever,
  * compiler-ABI FileRunner (run_c4_min) + data_seg   -- the multi-arg printf I/O.

Runs doom with stdin 'q' from step 0 until the FIRST printf ("%c[2J%c[H", ESC, ESC).
VERIFY: neural stdout's first bytes == `printf 'q' | ./c4 doom.c` (first 7 = 1b5b324a1b5b48).

Env knobs: K (block_steps, default 500), C4_MAX_STEPS (default 30200), C4_LOCAL_WINDOW
(default 96).  Requires C4_DRAFT_CMP32=1.
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_DRAFT_CMP32", "1")
os.environ.setdefault("C4_MEM_ADDR_BITS", "18")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/home/alexlitz/Documents/misc/c4_doom")

import torch
from pathlib import Path
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.pf_speculative import draft_pf_program, verify_blocks, _draft_cmp32
from src.compiler import compile_c
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import (tag_compiler_syscalls, install_compiler_abi_file_dispatcher,
                        data_segment)

DOOM = "/home/alexlitz/Documents/misc/c4_doom/doom.c"
REF_BIN = "/tmp/doom_ref.bin"          # printf 'q' | ./c4 doom.c  (1986 bytes)


def main():
    assert _draft_cmp32(), "C4_DRAFT_CMP32 must be on for doom's >255 loop counters"
    max_steps = int(os.environ.get("C4_MAX_STEPS", "30200"))
    K = int(os.environ.get("K", "500"))
    window = int(os.environ.get("C4_LOCAL_WINDOW", "96"))

    src = Path(DOOM).read_text()
    bc, data = compile_c(src)
    code = tag_compiler_syscalls(bytecode_to_isa(bc), isa)
    data_seg = data_segment(data)
    print(f"[doom] instrs={len(code)} data={len(data or [])} K={K} "
          f"max_steps={max_steps} window={window} addr_bits={os.environ['C4_MEM_ADDR_BITS']}",
          flush=True)

    ref = Path(REF_BIN).read_bytes()
    print(f"[doom] reference ./c4 first 7 = {ref[:7].hex()} (want 1b5b324a1b5b48)", flush=True)

    # ---- DRAFT (free) : run the whole init phase + first printf with I/O -----------
    install_compiler_abi_file_dispatcher()
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(b"q", neural=True)))
    t0 = time.time()
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF,
                             data_seg=data_seg, fio=fio)
    draft_out = bytes(fio.runner.stdout)
    print(f"[doom] draft: steps={draft.step_count} halted={draft.halted} "
          f"n_prtf={len(draft.prtf_steps)} first_prtf_step={draft.prtf_steps[:1]} "
          f"stdout={draft_out[:16].hex()} draft_wall={time.time()-t0:.2f}s", flush=True)
    dm = sum(1 for f in draft.frames if f["op"] in ("DIV", "MOD"))
    print(f"[doom] draft divmod steps={dm} ({100*dm/max(draft.step_count,1):.1f}%)", flush=True)
    if not draft.prtf_steps:
        print("[doom] DRAFT did not reach the first printf within max_steps", flush=True)
        return 2

    # ---- BUILD the model (streaming sparse) ----------------------------------------
    from c4_min.lib_neural import build_lib_model_streaming
    cs = max(len(code) + 2, 64)
    print(f"[doom] building model code_size={cs} (this is the scale wall) ...", flush=True)
    t0 = time.time()
    sparse, L, _ = build_lib_model_streaming(code_size=cs, recurrent_divmod=True,
                                             addr32=True, compute_mode="dense_kernel")
    print(f"[doom] built blocks={len(sparse.blocks)} dim={sparse.embed.shape[1]} "
          f"build_wall={time.time()-t0:.0f}s", flush=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dev != "cpu":
        sparse = sparse.to(dev)
    from c4_min.local_attention import install_local_attention
    from c4_min.live_head_attention import install_dead_block_fusion
    install_local_attention(sparse, window=window, drop_local_kv=True,
                            content_bound_global=True, verbose=False)
    install_dead_block_fusion(sparse, verbose=False)
    print(f"[doom] installed local-attn(window={window}) + dead-block-fusion", flush=True)

    # ---- VERIFY (fast big-K) : model must accept every draft step ------------------
    stats = {}
    t0 = time.time()
    vr = verify_blocks(sparse, L, code, draft, block_steps=K, device=dev,
                       evict=True, mask=0xFFFFFFFF, stats=stats, fast=True,
                       oom_backoff=True, min_block_steps=4)
    run_wall = time.time() - t0
    ms = run_wall / max(draft.step_count, 1) * 1e3
    print(f"[doom] FAST verify_blocks: matched={vr.all_matched} forwards={vr.forwards} "
          f"wall={run_wall:.1f}s ms/step={ms:.1f} eff_K={stats.get('effective_block_steps')} "
          f"peak_vram={stats.get('peak_vram_gb', 0):.1f}GB", flush=True)

    # ---- VERDICT -------------------------------------------------------------------
    n = min(len(draft_out), len(ref))
    match = 0
    for i in range(n):
        if draft_out[i] == ref[i]:
            match += 1
        else:
            break
    print(f"\n[doom] neural stdout first bytes = {draft_out[:7].hex()}", flush=True)
    print(f"[doom] byte-exact prefix vs ./c4 = {match} bytes "
          f"({'ESC[2J ESC[H OK' if draft_out[:7] == ref[:7] else 'MISMATCH'})", flush=True)
    print(f"[doom] model accepts every step (speculation valid): {vr.all_matched}", flush=True)
    ok = (draft_out[:7] == ref[:7]) and vr.all_matched
    print(f"\n[doom] CONTINUOUS BYTE-EXACT DOOM RUN to first output: {ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
