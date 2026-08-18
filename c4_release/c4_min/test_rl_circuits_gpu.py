"""GPU-GATED RL-circuit self-checks (part of the #923 Wave-2 RL-circuits
consolidation).

These wrap the FULL RL self-checks of the two grafted-ALU-under-RL studies:

  * C1 ``_agent_graft_rl_variance_reduction`` — grafts the #910 bounded byte-exact
    ALU into a real Qwen2.5-0.5B and sweeps the variance-reduction toolkit
    (big group / advantage-norm / low temperature / KL trust region) under
    UNPROTECTED reward-aligned RL; the finding is that the routing/compute SPLIT
    is the load-bearing survival lever, not variance reduction.
  * C4 ``_agent_graft_rl_unprotected`` — the companion: with NO training-time
    protection, a REINFORCE "use-the-ALU-correctly" reward is a DESTABILIZER of
    the byte-exact attractor.

BOTH load a real Qwen2.5-0.5B host (``AutoModelForCausalLM.from_pretrained`` at
fp32, several GB) and run policy-gradient sweeps, so they are GPU-ONLY: loading
the stock 0.5B on CPU costs ~7-9 GB RSS (over the 4 GB session cap → OOM risk)
and the RL sweeps are slow.  They are marked ``@pytest.mark.gpu`` and are NOT run
in the CPU / default lane — AUTHORED + MARKED here, executed only where a GPU is
available.  Each uses the module's own QUICK flag to bound the sweep.

Off-build-path: imports no neural-model build-path module — golden 174ece66 is
untouched (see ``test_golden_174ece66_untouched``).
"""
from __future__ import annotations

import os
import sys

import pytest

# The RL modules use bare top-level sibling imports (``from _agent_graft_sgd_vm_rl
# import ...``); put the c4_min package dir on sys.path so they resolve.
_C4MIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _C4MIN_DIR not in sys.path:
    sys.path.insert(0, _C4MIN_DIR)


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("GPU-only RL self-check (loads a real Qwen2.5-0.5B host); "
                    "no CUDA device available")


@pytest.mark.gpu
def test_c1_variance_reduction_rl_selfcheck():
    """C1: the byte-exact-ALU-under-variance-reduction RL sweep runs end-to-end on a
    real Qwen2.5-0.5B host (GPU-only).  Uses ``GRAFT_VR_QUICK=1`` to bound the sweep;
    also asserts the grafted ALU is byte-exact AT CONSTRUCTION (the #910 attractor
    the RL study probes for survival)."""
    _require_cuda()
    os.environ["GRAFT_VR_QUICK"] = "1"

    import _agent_graft_rl_variance_reduction as VR

    # byte-exact at construction (the reward-optimum the RL sweep starts from)
    alu = VR.GraftedByteExactALU(temp=30.0)
    VR.construct_byte_exact(alu)

    # full RL self-check sweep (loads the 0.5B host; QUICK-bounded)
    VR.main()


@pytest.mark.gpu
def test_c4_unprotected_reward_rl_selfcheck():
    """C4: the UNPROTECTED reward-aligned RL sweep runs end-to-end on a real
    Qwen2.5-0.5B host (GPU-only).  Uses ``GRAFT_RL_QUICK=1`` to bound the sweep;
    also asserts the grafted ALU is byte-exact AT CONSTRUCTION."""
    _require_cuda()
    os.environ["GRAFT_RL_QUICK"] = "1"

    import _agent_graft_rl_unprotected as UP

    alu = UP.GraftedByteExactALU(temp=30.0)
    UP.construct_byte_exact(alu)

    UP.main()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-m", "gpu"]))
