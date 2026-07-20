"""EXEC-PATH VANILLA GUARD — a RUNTIME proof that the c4_min per-step decode is
purely the transformer forward + the LM-head argmax re-quantiser, with **zero**
``torch.round`` (or any ad-hoc float->int quantiser) executing on the compute
path.

Motivation
----------
The vanilla requantisation the spec mandates is the model's OWN emit-token snap:
each register value re-enters the residual as an exact-integer byte token, decoded
by the LM byte-head argmax (``argmax_v (2*v*x - v^2)`` == nearest integer, but
residue-IMMUNE).  A ``torch.round`` on the exec path is a NON-VANILLA short-cut:
it snaps the fp residual with a hand-rolled quantiser instead of the transformer's
own head.  This test asserts that no such short-cut runs.

Two complementary guards
-------------------------
1. STATIC (AST): the decode-path modules contain no ``round`` / ``floor`` /
   ``trunc`` CALL node.  (``test_blogspec_foundation`` already does this for the
   blogspec modules; here we cover the nibble-VM / pure-forward / recurrent /
   memory decode modules too — the exec-path requant.)
2. RUNTIME (monkeypatch + settrace): we actually EXECUTE a representative program
   on each exec path (recurrent, pure-forward-complete, blogspec_run) with
   ``torch.round`` / ``torch.floor`` / ``torch.trunc`` / ``Tensor.round`` /
   ``Tensor.floor`` / ``Tensor.trunc`` swapped for tripwires that RAISE on any
   call — and assert none fired.  For the PURE-FORWARD paths (whole VM transition
   in ``model.forward``) a settrace census over the SAME run additionally confirms
   no python VM-compute helper (``_apply_op`` / a per-call ALU gadget) leaked in —
   the ONLY things allowed to run per step are the transformer forward, the argmax
   decode (``_snap_lane`` / ``_snap_nib`` / ``_decode_byte``), and the tool-call
   I/O boundary (the one allowed external step, absent in these progs).  The
   ``blogspec_run`` REFERENCE driver computes the VM transition in python by design
   (its job is to prove the nibble + 30-token + argmax requant, not the neural
   transition), so it is checked for the quantiser tripwire only.  A negative
   control proves the tripwire is not vacuous (it DOES catch a real torch.round).

Run: OMP_NUM_THREADS=4 PYTHONPATH=<repo> python -m pytest c4_min/test_exec_path_vanilla.py
(or directly: OMP_NUM_THREADS=4 PYTHONPATH=<repo> python c4_min/test_exec_path_vanilla.py).
"""
from __future__ import annotations

import ast
import inspect
import sys
from typing import List

import torch

from c4_min import isa


# --- programs that exercise arith / cmp / branch / deep-loop ------------------
_ADD = [("IMM", 6), ("PSH", 0), ("IMM", 7), ("ADD", 0), ("HALT", 0)]
_SUB_UF = [("IMM", 7), ("PSH", 0), ("IMM", 9), ("SUB", 0), ("HALT", 0)]
_BRANCH = [("IMM", 0), ("BZ", 3), ("IMM", 99), ("IMM", 7), ("HALT", 0)]
_LOOP = [("IMM", 8), ("PSH", 0), ("IMM", 1), ("SUB", 0), ("BNZ", 1), ("HALT", 0)]


# ===========================================================================
# RUNTIME tripwire: swap every float->int quantiser torch op for a raiser.
# A settrace census runs alongside to catch python VM-compute helpers.
# ===========================================================================
_FORBIDDEN_PY = {
    "_apply_op",                                    # blogspec_run if/elif dispatch
    "DictMemStack.store_int", "DictMemStack.load_int",
    "nibble_add_gadget", "nibble_sub_gadget",       # per-call ALU gadgets
    "mul32", "div32", "mod32", "compare", "to_bit",
    "or_gadget", "xor_gadget", "and_gadget", "shl_gadget", "shr_gadget",
}


class _VanillaExecGuard:
    """Records (and raises on) every exec-time ``torch.round`` / ``.round()`` /
    floor / trunc call (the non-vanilla quantisers) AND records every forbidden
    python VM-compute call while active.  Empty ``violations`` == the step ran
    purely as transformer-forward + argmax-decode."""

    def __init__(self):
        self.round_calls: List[str] = []
        self.py_calls: List[str] = []
        self._saved = {}
        self._prev_trace = None

    def _make_trip(self, label):
        rec = self.round_calls

        def _trip(*a, **k):
            rec.append(label)
            raise AssertionError(
                f"NON-VANILLA quantiser {label} executed on the exec path")
        return _trip

    def _tracer(self, frame, event, arg):
        if event == "call":
            name = frame.f_code.co_name
            qual = getattr(frame.f_code, "co_qualname", name)
            if name in _FORBIDDEN_PY or qual in _FORBIDDEN_PY or \
               any(qual.endswith("." + f) for f in _FORBIDDEN_PY):
                self.py_calls.append(qual)
        return None

    def __enter__(self):
        for nm in ("round", "floor", "trunc"):
            self._saved[("torch", nm)] = getattr(torch, nm)
            setattr(torch, nm, self._make_trip(f"torch.{nm}"))
        for nm in ("round", "round_", "floor", "floor_", "trunc", "trunc_"):
            self._saved[("Tensor", nm)] = getattr(torch.Tensor, nm)
            setattr(torch.Tensor, nm, self._make_trip(f"Tensor.{nm}"))
        self._prev_trace = sys.gettrace()
        sys.settrace(self._tracer)
        return self

    def __exit__(self, *exc):
        sys.settrace(self._prev_trace)
        for (owner, nm), fn in self._saved.items():
            setattr(torch if owner == "torch" else torch.Tensor, nm, fn)
        return False


def _assert_vanilla_exec(run_fn, *args, enforce_no_py_compute=True, **kwargs):
    """Run ``run_fn`` under the guard.  ALWAYS asserts NO round/floor/trunc
    quantiser ran (the load-bearing check: the requant is the LM-head argmax, not a
    hand-rolled snap).  ``enforce_no_py_compute`` additionally asserts no python
    VM-compute helper (``_apply_op`` / a per-call ALU gadget) ran — true for the
    PURE-FORWARD paths (the whole VM transition is ``model.forward``); the
    ``blogspec_run`` reference driver computes the transition in python by design
    (its contribution is proving the NIBBLE + 30-token + argmax requant), so it is
    checked for the quantiser only.  Returns ``run_fn``'s result."""
    guard = _VanillaExecGuard()
    with guard:
        result = run_fn(*args, **kwargs)
    assert not guard.round_calls, \
        f"exec-time float->int quantiser leaked: {sorted(set(guard.round_calls))}"
    if enforce_no_py_compute:
        assert not guard.py_calls, \
            f"python VM-compute leaked into the step: {sorted(set(guard.py_calls))}"
    return result


# ===========================================================================
# 1. RECURRENT path — the ONE step-block applied autoregressively (deep loops).
#    This is the path whose per-step requant WAS ``torch.round`` and is now the
#    vanilla LM-head argmax.  We prove BOTH the tripwire fires zero AND the trace
#    is byte-exact vs the reference (so the swap is byte-identical).
# ===========================================================================
def test_recurrent_exec_is_vanilla_argmax():
    from c4_min.recurrent import build_step_model, run_recurrent
    for prog in (_ADD, _SUB_UF, _BRANCH, _LOOP):
        code = isa.assemble(prog)
        model, L, _ = build_step_model(code)
        trace = _assert_vanilla_exec(
            run_recurrent, model, L, code, max_steps=100000, requantize=True)
        assert trace == isa.interpret(code, max_steps=100000), (prog, trace)


# ===========================================================================
# 2. PURE-FORWARD-COMPLETE path — one model.forward per step; decode via the
#    LM-head argmax (``_snap_lane`` / ``_snap_nib``).  Built via the memory-SAFE
#    streaming path (``guarded_complete_build`` -> ``build_compact_sparse_streaming``,
#    peak ~5 GB) instead of the DENSE ``build_pure_forward_complete_model`` (which
#    pads every block to the ~160k-row MUL/DIV/MOD FFN and peaks at 54-108 GB RSS
#    — this test at code_size=16 ballooned the box to 105 GB).  The streaming model
#    is the SAME full-op-set interpreter, byte-identical (L-inf=0, dense_kernel) and
#    driven by the SAME ``run_pure_forward_complete`` runner.
# ===========================================================================
def test_pure_forward_complete_exec_is_vanilla_argmax():
    from c4_min.nibble_pure_forward_complete import run_pure_forward_complete
    from c4_min._build_guard import guarded_complete_build
    model, L = guarded_complete_build(code_size=16)
    for prog in (_ADD, _SUB_UF, _BRANCH, _LOOP):
        code = isa.assemble(prog)
        trace = _assert_vanilla_exec(
            run_pure_forward_complete, model, L, code, max_steps=256)
        ref = [v & 0xFF for v in isa.interpret(code, max_steps=256)]
        assert trace == ref, (prog, trace, ref)


# ===========================================================================
# 3. BLOGSPEC_RUN path — the spec-faithful autoregressive driver; register bytes
#    decoded from the nibble state by the LM byte-head argmax.
# ===========================================================================
def test_blogspec_run_exec_is_vanilla_argmax():
    from c4_min import blogspec_compiler as C
    from c4_min import blogspec_run as R
    model, L, code = C.build_step_model(_ADD)
    # blogspec_run computes the VM transition in python (the reference driver); its
    # requant is still the vanilla LM-head argmax, so we assert ONLY the quantiser
    # tripwire (no torch.round), not the python-compute census.
    tokens, frames = _assert_vanilla_exec(
        R.run_program, model, L, code, enforce_no_py_compute=False)
    trace = R.decode_trace(frames)
    assert trace == isa.interpret(code), (trace, isa.interpret(code))


# ===========================================================================
# 4. STATIC (AST) — no round/floor/trunc CALL node in the decode modules (the
#    exec-path requant).  The blogspec modules are also covered by
#    test_blogspec_foundation; here we additionally cover the nibble-VM /
#    pure-forward / recurrent / memory decode modules.
# ===========================================================================
def test_guard_is_not_vacuous_negative_control():
    """The tripwire MUST fire if a ``torch.round`` executes on the traced path —
    otherwise the passing tests above prove nothing.  Run a function that DOES call
    ``torch.round`` under the guard and assert it is caught."""
    def _rounds_a_tensor():
        return torch.round(torch.tensor([0.4, 1.6]))
    caught = False
    try:
        _assert_vanilla_exec(_rounds_a_tensor, enforce_no_py_compute=False)
    except AssertionError as e:
        caught = "torch.round" in str(e)
    assert caught, "guard did NOT catch a real torch.round — it is vacuous"


def test_no_quantizer_call_node_in_decode_modules():
    import c4_min.recurrent as REC
    import c4_min.nibble_vm as NVM
    import c4_min.nibble_pure_forward as NPF
    import c4_min.nibble_pure_forward_complete as NPFC
    import c4_min.blogspec_memory as BM
    import c4_min.blogspec_run as BR
    forbidden = {"round", "floor", "trunc"}
    for mod in (REC, NVM, NPF, NPFC, BM, BR):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute) else None)
                assert name not in forbidden, (mod.__name__, name, ast.dump(node))


if __name__ == "__main__":
    import traceback
    tests = [
        test_recurrent_exec_is_vanilla_argmax,
        test_pure_forward_complete_exec_is_vanilla_argmax,
        test_blogspec_run_exec_is_vanilla_argmax,
        test_guard_is_not_vacuous_negative_control,
        test_no_quantizer_call_node_in_decode_modules,
    ]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} exec-path-vanilla tests passed")
