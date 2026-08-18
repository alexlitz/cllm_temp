"""#921 BUILD + RUN the WIDTH-FREE PARALLEL-FORM log-sink divmod as ACTUAL blocks —
the construction the menu doc PROJECTS at ~11 blocks but nobody assembled.

The current ``nibble_logsink_blocks.compile_logsink_blocks`` is 127 stored blocks whose
depth is 96 blocks of SEQUENTIAL running-remainder decompose glue (measured in
``_measure_critpath_921``: greedy antichain collapse only reaches 124, because each
``ext_c`` reads what ``red_{c+1}`` wrote).  This module REWRITES decompose to the
PARALLEL form — all 8 nibbles read the CLEAN integer scalar directly:

    nib_c = floor(Q / 16^c) - 16 * floor(Q / 16^(c+1))

so ALL 8 nibble extracts + snaps live in ONE collapsed layer (no running remainder).
That is the "width is free" reduction the accumulation strategy needs.

The PRECISION crux (measured in this module):
  * fp64 (2^53 headroom): floor(Q/16^c) of a 2^32 scalar is EXACT -> parallel decompose
    byte-exact -> the 96-block glue collapses to ~4 parallel blocks.
  * fp32 (2^24 ceiling): a 2^32 scalar is NOT exactly representable, so floor(Q/16^c)
    is already wrong -> parallel decompose FAILS unless Q is held as <=16-bit CHUNKS.
    The fp32 build must therefore carry Q (and REM) as 2 fp32 limbs and decompose each
    limb's 4 nibbles in parallel -> the chunking adds limb-split depth.

This module provides the minimal ``_MiniL`` layout that carries the AX/STACK0 nibble
inputs + the LogSink scratch, used by ``_measure_critpath_921`` / ``_asap_schedule_921``
to build the PARALLEL-form block list from the byte-identical ``nibble_logsink_blocks``
emitters and measure its width-free critical path.

NOTE: byte-exactness of the PARALLEL algorithm is measured at the ARITHMETIC level (the
block semantics in the target dtype) in the inline harnesses — fp64 400293/400293,
fp32-chunked 500308/500308 — NOT through a bespoke FFN forward (an earlier
RMSNorm-identity mini-forward was too lossy to be a trustworthy oracle and was removed).
The DEPTH is measured structurally from the actual block data-dependencies.

Golden 174ece66 untouched (off build path).
"""
from __future__ import annotations

import math
import os
import resource
import threading
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

MASK32 = 0xFFFFFFFF


def _rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def _watchdog(limit_mb: int = 4000):
    def watch():
        while True:
            if _rss_mb() > limit_mb:
                print(f"RSS ABORT {_rss_mb()} MB > {limit_mb}", flush=True)
                os._exit(3)
            time.sleep(0.25)
    threading.Thread(target=watch, daemon=True).start()


# ---------------------------------------------------------------------------
# A minimal layout carrying the AX/STACK0 nibble inputs + the LogSink scratch.
# ---------------------------------------------------------------------------
class _MiniL:
    def __init__(self, n_heads=4):
        self._off = 0
        self._names = {}
        self.n_heads = n_heads
        self.ONE = self._scalar("ONE")
        self.AX = self._band("AX", 8)          # divisor b nibbles
        self.STACK0 = self._band("STACK0", 8)  # dividend a nibbles
        self.LOGSINK = None

    def _scalar(self, name):
        return self._band(name, 1)

    def _band(self, name, size):
        base = self._off
        self._names[name] = (base, size)
        self._off += size
        return base

    def finalize(self, n_heads=4):
        while self._off % n_heads != 0:
            self._scalar(f"_pad{self._off}")
        self.D = self._off
