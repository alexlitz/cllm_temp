#!/usr/bin/env python3
"""Static scan for the "opcode-broadcast defeats intent-blocker" bug class.

NO model build, NO weights, NO probing. Imports the per-layer ``make_*``
op factories, collects their *resolved* ``FFNRule`` objects (all variable /
tuple-concat / helper resolution already done), and for every rule whose
``conditions`` (or ``gate_terms``) contain a POSITIVE ``("OP_<X>", W)`` term for
a *broadcasting* frame/control opcode, checks whether ``OP_<X>_broadcast * W``
can defeat the rule's negative intent-blockers (markers / IS_BYTE / byte_index
/ marker-distance hints).

Measured broadcast magnitudes (tools/audit_opcode_broadcast.py, spec_k=0):
  OP_JSR : ~15.5 marker, ~17.2 byte, ~11.5 STEP_END
  OP_ENT : up to ~11.4 at the ENT-step AX/SP marker rows (commit f04e9f02)
  OP_LEA : ~5.2 marker
  OP_LEV : (probed separately on a LEV program)
  OP_ADJ : (probed separately on an ADJ program)
  ALL ALU opcodes (ADD/SUB/OR/AND/MUL/EQ/LT/...) : 0.0  -> do NOT broadcast.

Classification per rule:
  (a) genuinely-vulnerable : opcode term ALONE clears (threshold - other
      positive baseline) AND the strongest blocker is too weak to veto, with
      the broadcast at the CURRENT measured magnitude.
  (b) latent              : safe at the current broadcast but the margin is
      thin (< 3x); would mis-fire if the broadcast grows or a second opcode
      co-asserts.
  (c) safe                : blocker is a hard NOT-blocker (>=1e5) OR the
      opcode weight is small enough that broadcast*W can never reach
      threshold past the blocker.

Usage:
    python tools/scan_opcode_broadcast_rules.py            # full report
    python tools/scan_opcode_broadcast_rules.py --csv      # machine-readable
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys

os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_TEST_SPEC_K", "0")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

# Empirically measured in-step broadcast magnitudes (peak |residual| over the
# marker + byte rows). ALU opcodes broadcast 0 -> not in this table, so any
# OP_<ALU> condition is inert for the bug class.
BROADCAST = {
    "OP_JSR": 17.2,   # measured: 17.2 byte rows / 15.5 markers / 11.5 STEP_END
    "OP_ENT": 11.4,   # measured: 7.3 (LI/LEV-program rows) .. 11.4 (ENT-step
                      #           AX/SP marker, commit f04e9f02). Use the peak.
    "OP_LEA": 5.2,    # measured 5.2 at the ENT-step LEA marker (id 262)
    # OP_LEV / OP_ADJ measured ~0.0 at the sampled L20-input rows (the flag is
    # only briefly asserted at its own single late step). 5.0 is a CONSERVATIVE
    # upper bound so their rules are flagged cautiously; in practice they are
    # even safer than reported. JSR + ENT are the only practical drivers.
    "OP_LEV": 5.0,
    "OP_ADJ": 5.0,
}
HARD_BLOCKER = 1e5   # >= this is treated as a hard NOT-blocker (immune)

# Marker / byte / hint dims that act as intent-blockers when negative.
BLOCKER_PREFIXES = (
    "MARK_", "IS_BYTE", "BYTE_INDEX_", "H1+", "H2+", "H3+", "HAS_SE",
    "MEM_STORE", "PSH_AT_SP",
)


def _is_blocker_dim(name: str) -> bool:
    return any(name == p or name.startswith(p) for p in BLOCKER_PREFIXES)


def _term_pairs(terms):
    """Normalize a sequence of ConditionTerm (or (name, weight) tuples) into
    ``[(key_string, weight), ...]`` where key is ``name`` for offset 0 else
    ``name+offset`` — matching how the source spells the dims."""
    out = []
    for t in terms or ():
        if isinstance(t, tuple):
            name, w = t[0], float(t[1])
        else:
            dr = t.dim
            name = dr.name if dr.offset == 0 else f"{dr.name}+{dr.offset}"
            w = float(t.weight)
        out.append((name, w))
    return out


def collect_all_rules():
    """Import every ops module, call each ``make_*`` factory, collect rules.

    Returns list of (module, factory_name, FFNRule)."""
    from neural_vm.unified_compiler.decl_verifier import _collect_ffn_rules_from_op
    from neural_vm.unified_compiler.ir import FFNRule, FFNOp, CompilerIR

    ops_dir = os.path.join(_PKG, "neural_vm", "unified_compiler", "ops")
    modnames = sorted(
        f[:-3] for f in os.listdir(ops_dir)
        if f.endswith("_ops.py") and not f.startswith("__")
    )
    # Also include non-lN modules that carry opcode rules.
    for extra in ("control_flow_heads", "all_core_ops", "shared",
                  "flag_gated_ops", "model_ops"):
        if extra not in modnames and os.path.exists(
                os.path.join(ops_dir, extra + ".py")):
            modnames.append(extra)

    out = []
    seen = set()
    for modname in modnames:
        try:
            mod = importlib.import_module(
                f"neural_vm.unified_compiler.ops.{modname}")
        except Exception as e:
            print(f"# skip module {modname}: {e}", file=sys.stderr)
            continue
        for fname, fn in inspect.getmembers(mod, inspect.isfunction):
            if not (fname.startswith("make_") or fname.startswith("_layer")
                    or fname.endswith("_rules") or fname.endswith("_ir")):
                continue
            sig = inspect.signature(fn)
            # Only call zero-arg-or-all-default factories.
            if any(p.default is inspect.Parameter.empty
                   and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                   for p in sig.parameters.values()):
                continue
            try:
                res = fn()
            except Exception:
                continue
            # Extract rules from whatever shape came back.
            rules = []
            if isinstance(res, FFNRule):
                rules = [res]
            elif isinstance(res, (list, tuple)):
                rules = [r for r in res if isinstance(r, FFNRule)]
                for r in res:
                    if isinstance(r, FFNOp):
                        rules.extend(r.rules)
            elif isinstance(res, FFNOp):
                rules = list(res.rules)
            elif isinstance(res, CompilerIR) or hasattr(res, "compiler_ir") \
                    or hasattr(res, "layers"):
                try:
                    rules = _collect_ffn_rules_from_op(
                        res if hasattr(res, "compiler_ir")
                        else type("X", (), {"compiler_ir": res})())
                except Exception:
                    rules = []
            for r in rules:
                key = (r.name or id(r), tuple(r.conditions),
                       tuple(getattr(r, "gate_terms", ()) or ()))
                if key in seen:
                    continue
                seen.add(key)
                out.append((modname, fname, r))
    return out


def analyze_rule(rule):
    """Return None if no broadcasting OP_<X> positive term; else a dict."""
    conds = _term_pairs(rule.conditions) + _term_pairs(
        getattr(rule, "gate_terms", ()) or ())
    op_terms = [(n, w) for (n, w) in conds
                if n in BROADCAST and w > 0]
    if not op_terms:
        return None

    pos_terms = [(n, w) for (n, w) in conds if w > 0 and n not in BROADCAST]
    neg_terms = [(n, w) for (n, w) in conds if w < 0]
    blockers = [(n, w) for (n, w) in neg_terms if _is_blocker_dim(n)]

    thr = float(rule.threshold)
    # Strongest broadcast contribution available from a single opcode term.
    op_contribs = [(n, w, w * BROADCAST[n]) for (n, w) in op_terms]
    best_op = max(op_contribs, key=lambda t: t[2])
    op_name, op_w, op_contrib = best_op

    # Other co-firing opcode broadcasts (a JSR+ENT step asserts both).
    total_op_contrib = sum(c for _, _, c in op_contribs)

    # Strongest single negative blocker magnitude.
    strongest_blocker = max((abs(w) for _, w in blockers), default=0.0)
    has_hard_blocker = any(abs(w) >= HARD_BLOCKER for _, w in blockers)

    # CO-FIRING blocker model at a realistic mis-fire row. The opcode broadcast
    # hits EVERY row, but the rule's marker/IS_BYTE blockers also co-fire at the
    # wrong row. Worst plausible mis-fire = a register BYTE row of a register
    # that is NOT the rule's intended one:
    #   * IS_BYTE = 1 (byte row)            -> the IS_BYTE blocker fires
    #   * exactly ONE wrong-register marker may be ~1 -> the largest single
    #     MARK_* / H* blocker fires
    #   * BYTE_INDEX_{1,2,3} blockers may fire on byte rows
    #   * the rule's legit non-opcode positives (its own MARK / BYTE_INDEX_0)
    #     are typically ABSENT -> they don't help the opcode.
    is_byte_blk = sum(abs(w) for (n, w) in blockers if n.startswith("IS_BYTE"))
    mark_blks = [abs(w) for (n, w) in blockers
                 if n.startswith("MARK_") or n.startswith("H1+")
                 or n.startswith("H2+") or n.startswith("H3+")]
    # one wrong-register marker active -> the single largest MARK blocker fires
    one_mark_blk = max(mark_blks) if mark_blks else 0.0
    byteidx_blks = sum(abs(w) for (n, w) in blockers
                       if n.startswith("BYTE_INDEX_") and "BYTE_INDEX_0" not in n)
    cofire_blocker = is_byte_blk + one_mark_blk + byteidx_blks
    if cofire_blocker == 0.0:
        cofire_blocker = strongest_blocker

    # The "baseline" positive contribution from the rule's INTENDED on-state
    # non-opcode positives (markers it legitimately keys on). At a MIS-FIRE
    # row, these may be 0 (wrong marker) — worst case the opcode term must
    # carry the whole threshold alone. But discriminator terms (e.g. an
    # OUTPUT_HI gate that is negative at the wrong row) help the blocker.
    # We classify on the conservative "opcode alone vs threshold-minus-other"
    # but report the strongest blocker for the operator.

    # Can the opcode broadcast alone (no legit positives) clear threshold?
    op_alone_clears = total_op_contrib >= thr
    # Realistic mis-fire score = opcode broadcast minus the CO-FIRING blockers
    # active at the wrong row.
    misfire_score = total_op_contrib - cofire_blocker
    misfire_clears = misfire_score >= thr

    if has_hard_blocker:
        cls = "c"
        reason = "hard NOT-blocker (>=1e5) present"
    elif cofire_blocker == 0.0 and op_alone_clears:
        # No marker/byte blocker at all and opcode clears -> vulnerable unless
        # a non-opcode positive co-requirement genuinely scopes it (the rule
        # SHOULD fire on every step of this opcode at that marker).
        cls = "a" if not pos_terms else "b"
        reason = ("no marker/IS_BYTE blocker; opcode broadcast clears threshold"
                  if cls == "a" else
                  "no hard blocker but rule co-requires non-opcode positives")
    elif misfire_clears:
        cls = "a"
        reason = (f"OP broadcast {total_op_contrib:.0f} - co-fire blockers "
                  f"{cofire_blocker:.0f} = {misfire_score:.0f} >= thr {thr:.0f} "
                  f"(mis-fires at wrong byte/marker row)")
    else:
        # blocker wins. Latent if the margin is thin (< 1.5x the opcode contrib)
        # so it would flip if the broadcast grew or a co-opcode co-asserted.
        margin = cofire_blocker - misfire_score
        if margin < 0.5 * op_contrib and op_alone_clears:
            cls = "b"
            reason = (f"co-fire blockers {cofire_blocker:.0f} beat OP "
                      f"{total_op_contrib:.0f} now (score {misfire_score:.0f} "
                      f"vs thr {thr:.0f}), thin margin (latent)")
        else:
            cls = "c"
            reason = (f"co-fire blockers {cofire_blocker:.0f} decisively veto "
                      f"OP {total_op_contrib:.0f} (score {misfire_score:.0f} "
                      f"< thr {thr:.0f})")

    return {
        "name": rule.name or "<anon>",
        "threshold": thr,
        "op_terms": op_terms,
        "best_op": (op_name, op_w, op_contrib),
        "total_op_contrib": total_op_contrib,
        "strongest_blocker": strongest_blocker,
        "cofire_blocker": cofire_blocker,
        "n_blockers": len(blockers),
        "has_hard_blocker": has_hard_blocker,
        "class": cls,
        "reason": reason,
        "blockers": blockers,
    }


import re

_FAMILY_SUFFIX = re.compile(
    r"(?:[._](?:lo|hi))?(?:[._](?:lane|at|byte|nib|nibble|k|b|idx))?[._]?\d+$"
)


def family_of(name: str) -> str:
    """Collapse band/lane-expanded rule names to their family stem.

    e.g. l16_lev_stack0_byte0_preserve_lo_7 -> l16_lev_stack0_byte0_preserve
         tail_mem_store_addr0_f8_initial_jsr_exact.lo.lane_9
              -> tail_mem_store_addr0_f8_initial_jsr_exact
    """
    prev = None
    n = name
    # strip repeatedly (e.g. ``_preserve_lo_7`` -> ``_preserve_lo`` -> ``_preserve``)
    while prev != n:
        prev = n
        n = _FAMILY_SUFFIX.sub("", n)
        n = re.sub(r"[._](?:lo|hi)$", "", n)
    return n


def main():
    csv = "--csv" in sys.argv
    fam = "--families" in sys.argv
    rules = collect_all_rules()
    print(f"# collected {len(rules)} unique FFNRules from ops factories",
          file=sys.stderr)

    findings = []
    for modname, fname, rule in rules:
        a = analyze_rule(rule)
        if a is None:
            continue
        a["module"] = modname
        a["factory"] = fname
        findings.append(a)

    by_class = {"a": [], "b": [], "c": []}
    for f in findings:
        by_class[f["class"]].append(f)

    if fam:
        # Collapse to families: a family's class = worst (a>b>c) over members.
        rank = {"a": 0, "b": 1, "c": 2}
        fams = {}
        for f in findings:
            key = (f["module"], family_of(f["name"]))
            cur = fams.get(key)
            if cur is None or rank[f["class"]] < rank[cur["class"]]:
                fams[key] = {**f, "n_members": 1, "family": key[1]}
            else:
                cur["n_members"] += 1
            if key in fams and fams[key] is not f:
                fams[key]["n_members"] = fams[key].get("n_members", 1)
        # recount members
        counts = {}
        for f in findings:
            key = (f["module"], family_of(f["name"]))
            counts[key] = counts.get(key, 0) + 1
        fam_class = {"a": [], "b": [], "c": []}
        for key, rep in fams.items():
            rep["n_members"] = counts[key]
            fam_class[rep["class"]].append(rep)
        print(f"\n# FAMILY-LEVEL (band/lane-collapsed): {len(fams)} distinct "
              f"rule families touch a broadcasting opcode")
        print(f"#   (a) genuinely-vulnerable families : {len(fam_class['a'])}")
        print(f"#   (b) latent families               : {len(fam_class['b'])}")
        print(f"#   (c) safe families                 : {len(fam_class['c'])}")
        for cls, label in (("a", "GENUINELY-VULNERABLE"), ("b", "LATENT"),
                           ("c", "SAFE")):
            items = sorted(fam_class[cls], key=lambda x: -x["total_op_contrib"])
            print(f"\n{'='*78}\n# FAMILY CLASS ({cls}) {label}  "
                  f"({len(items)} families)\n{'='*78}")
            for f in items:
                on, ow, oc = f["best_op"]
                ops_str = " ".join(f"{n}*{w:.0f}" for n, w in f["op_terms"])
                print(f"  [{f['module']:<12}] {f['family']}  (x{f['n_members']})")
                print(f"      thr={f['threshold']:.0f} ops=[{ops_str}] "
                      f"op_total={f['total_op_contrib']:.0f} "
                      f"blocker={f['strongest_blocker']:.0f} -> {f['reason']}")
        return

    if csv:
        print("class,module,name,threshold,best_op,best_op_w,op_contrib,"
              "total_op_contrib,strongest_blocker,has_hard_blocker,reason")
        for f in sorted(findings, key=lambda x: (x["class"], x["module"])):
            on, ow, oc = f["best_op"]
            print(f'{f["class"]},{f["module"]},{f["name"]},{f["threshold"]:.0f},'
                  f'{on},{ow:.0f},{oc:.0f},{f["total_op_contrib"]:.0f},'
                  f'{f["strongest_blocker"]:.0f},{f["has_hard_blocker"]},'
                  f'"{f["reason"]}"')
        return

    print(f"\n# {len(findings)} rules reference a broadcasting frame/control "
          f"opcode as a positive condition.")
    print(f"#   (a) genuinely-vulnerable : {len(by_class['a'])}")
    print(f"#   (b) latent               : {len(by_class['b'])}")
    print(f"#   (c) safe                 : {len(by_class['c'])}")

    for cls, label in (("a", "GENUINELY-VULNERABLE"), ("b", "LATENT"),
                       ("c", "SAFE")):
        items = by_class[cls]
        print(f"\n{'='*78}\n# CLASS ({cls}) {label}  ({len(items)} rules)\n{'='*78}")
        for f in sorted(items, key=lambda x: -x["total_op_contrib"]):
            on, ow, oc = f["best_op"]
            ops_str = " ".join(f"{n}*{w:.0f}" for n, w in f["op_terms"])
            print(f"  [{f['module']:<10}] {f['name']}")
            print(f"      thr={f['threshold']:.0f}  op_terms=[{ops_str}]  "
                  f"best={on} contrib={oc:.0f} total_op={f['total_op_contrib']:.0f}")
            print(f"      strongest_blocker={f['strongest_blocker']:.0f}  "
                  f"hard={f['has_hard_blocker']}  -> {f['reason']}")


if __name__ == "__main__":
    main()
