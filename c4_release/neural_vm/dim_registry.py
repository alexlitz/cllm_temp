"""
Dimension Registry & Layer Contract Tracking for the Autoregressive Neural VM.

Tracks all d_model dimension allocations, per-layer read/write contracts,
and validates data-flow invariants (no unintended overlaps, write-before-read,
double-write detection).

Also provides static weight inspection: auto-derive read/write contracts from
actual weight matrices after set_vm_weights() runs, and compare against
manually-declared contracts.

Usage:
    python3 -m neural_vm.dim_registry    # print dim map + validate + weight inspection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class DimSlot:
    """A named allocation within the d_model embedding dimensions."""
    name: str
    start: int
    size: int
    desc: str
    semantics: Optional[str] = None  # predicate-DSL string describing
                                     # when this dim fires; checked by
                                     # decl_verifier.verify_rule_scopes
                                     # (F-7). None means "not yet specified
                                     # — tolerated this merge cycle but will
                                     # become required."
    # Phase 7.E.1 — semantic category & role for the (category, role) ->
    # offset lookup path. ``category`` names a family of conceptually
    # equivalent dims (``"register_lo"``, ``"memory_hi"``, ``"opcode_flag"``,
    # …) and ``role`` distinguishes members within that family
    # (``"AX"``, ``"SP"``, ``"OP_JMP"``, …). Together they uniquely
    # identify a slot via ``DimRegistry.resolve_dim(category, role)``.
    # Both are optional so legacy ``+N`` offset references and slot-name
    # lookups continue to work unchanged — the category index is purely
    # additive. Phase 7.E.2 will start migrating rules to the new path.
    category: Optional[str] = None
    role: Optional[str] = None

    @property
    def end(self) -> int:
        return self.start + self.size

    @property
    def range(self) -> range:
        return range(self.start, self.end)

    def overlaps(self, other: 'DimSlot') -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class LayerIO:
    """Declares which dim slots a layer reads from and writes to."""
    layer: str          # e.g. "embed", "L0_attn", "L0_ffn", "L3_attn"
    reads: List[str]    # slot names read
    writes: List[str]   # slot names written
    notes: str = ""
    additive_writes: List[str] = field(default_factory=list)  # intentional multi-writer slots


class DimRegistry:
    """Registry of all dimension allocations within d_model."""

    def __init__(self, d_model: int = 256):
        self.d_model = d_model
        self.slots: Dict[str, DimSlot] = {}
        # Phase 7.E.1 — (category, role) -> slot_name index. Built up
        # incrementally as slots are allocated with category/role
        # arguments OR registered post-hoc via ``register_category``.
        # ``resolve_dim`` reads from this map. A given (category, role)
        # pair MUST map to exactly one slot — duplicate registration
        # raises ``ValueError`` so accidental collisions are loud.
        self._category_index: Dict[Tuple[str, str], str] = {}

    def alloc(
        self,
        name: str,
        start: int,
        size: int,
        desc: str,
        semantics: Optional[str] = None,
        category: Optional[str] = None,
        role: Optional[str] = None,
    ) -> DimSlot:
        """Register a dimension allocation. Returns the DimSlot.

        `semantics` is a predicate-DSL string (parse with
        neural_vm.unified_compiler.predicates.parse) describing when this
        dim fires. Optional during F-3 tolerant rollout; will become
        required in a follow-on commit once all ~80 dims are backfilled.

        Phase 7.E.1: `category` + `role` populate the
        ``(category, role) -> offset`` index consumed by
        :meth:`resolve_dim`. Both must be provided together or both
        omitted; passing one without the other raises ``ValueError``.
        """
        if name in self.slots:
            raise ValueError(f"Duplicate slot name: {name}")
        if start < 0 or start + size > self.d_model:
            raise ValueError(f"Slot {name} [{start}, {start+size}) out of bounds [0, {self.d_model})")
        if semantics is None:
            import warnings
            warnings.warn(
                f"DimSlot {name!r} declared without semantics (DEPRECATED; "
                f"will be required after F-4 backfill lands). Add a "
                f"`semantics='<predicate>'` keyword arg.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        if (category is None) != (role is None):
            raise ValueError(
                f"Slot {name!r}: category and role must both be provided "
                f"or both omitted (got category={category!r}, role={role!r})"
            )
        slot = DimSlot(name, start, size, desc, semantics,
                       category=category, role=role)
        self.slots[name] = slot
        if category is not None and role is not None:
            self._index_category(name, category, role)
        return slot

    def semantics(self, name: str) -> Optional[str]:
        """Return the semantics predicate string for `name`, or None if
        not declared."""
        if name not in self.slots:
            raise KeyError(f"Unknown dim: {name!r}")
        return self.slots[name].semantics

    # ------------------------------------------------------------------
    # Phase 7.E.1 — semantic category / role index
    # ------------------------------------------------------------------
    def _index_category(self, name: str, category: str, role: str) -> None:
        """Insert ``(category, role) -> name`` into the lookup map.

        Raises ``ValueError`` if the pair is already claimed by a
        different slot. Used internally by :meth:`alloc` and
        :meth:`register_category`.
        """
        key = (category, role)
        existing = self._category_index.get(key)
        if existing is not None and existing != name:
            raise ValueError(
                f"Category ({category!r}, {role!r}) already maps to "
                f"{existing!r}; cannot re-register for {name!r}"
            )
        self._category_index[key] = name

    def register_category(
        self,
        name: str,
        category: str,
        role: str,
    ) -> None:
        """Attach a ``(category, role)`` label to an already-allocated slot.

        Used for slots whose ``alloc`` call doesn't yet carry category
        kwargs (the bulk of ``build_default_registry``'s body). Idempotent
        when the slot is re-registered with the same ``(category, role)``;
        raises ``ValueError`` on any conflicting registration.

        The slot itself is updated in place so
        :attr:`DimSlot.category`/``.role`` reflect the registration.
        """
        if name not in self.slots:
            raise KeyError(f"Unknown dim: {name!r}")
        slot = self.slots[name]
        if slot.category is not None or slot.role is not None:
            # Re-registration with identical pair is a no-op; conflicting
            # pair is an error.
            if slot.category == category and slot.role == role:
                return
            raise ValueError(
                f"Slot {name!r} already has category=({slot.category!r}, "
                f"{slot.role!r}); cannot re-tag as ({category!r}, {role!r})"
            )
        slot.category = category
        slot.role = role
        self._index_category(name, category, role)

    def resolve_dim(self, category: str, role: str) -> int:
        """Return the absolute dim offset for ``(category, role)``.

        Phase 7.E.1 lookup helper: rules that today reference dims by
        ``"OUTPUT_LO+15"``-style strings will start consuming
        ``resolve_dim("output_lo", "nibble_15")`` (or analogous role
        names) once Phase 7.E.2 migrates the call sites. The returned
        offset is the slot's ``start`` — for multi-cell slots like
        ``OUTPUT_LO``, the role names a specific cell and the slot is
        sized 1; for whole-family slots like ``ALU_LO`` the role is the
        family name and the returned offset is the family base.

        Raises ``KeyError`` if no slot has been registered with the
        requested ``(category, role)`` pair.
        """
        key = (category, role)
        if key not in self._category_index:
            raise KeyError(
                f"No dim registered for (category={category!r}, "
                f"role={role!r})"
            )
        slot_name = self._category_index[key]
        return self.slots[slot_name].start

    def categories(self) -> Dict[str, List[str]]:
        """Return ``{category: [role, ...]}`` for every registered pair.

        Roles within each category are sorted by their resolved offset
        so the output groups conceptually-equivalent dims together.
        """
        out: Dict[str, List[Tuple[str, int]]] = {}
        for (cat, role), name in self._category_index.items():
            out.setdefault(cat, []).append((role, self.slots[name].start))
        return {
            cat: [r for r, _ in sorted(pairs, key=lambda x: x[1])]
            for cat, pairs in out.items()
        }

    def check_overlaps(self) -> List[str]:
        """Return error messages for any overlapping slots."""
        errors = []
        names = sorted(self.slots.keys(), key=lambda n: self.slots[n].start)
        for i, a_name in enumerate(names):
            for b_name in names[i+1:]:
                a, b = self.slots[a_name], self.slots[b_name]
                if a.overlaps(b):
                    errors.append(
                        f"OVERLAP: {a.name}[{a.start}:{a.end}) and "
                        f"{b.name}[{b.start}:{b.end})"
                    )
        return errors

    def free_ranges(self) -> List[Tuple[int, int]]:
        """Return list of (start, end) for unallocated dim ranges."""
        used = set()
        for slot in self.slots.values():
            used.update(slot.range)
        free = []
        start = None
        for d in range(self.d_model):
            if d not in used:
                if start is None:
                    start = d
            else:
                if start is not None:
                    free.append((start, d))
                    start = None
        if start is not None:
            free.append((start, self.d_model))
        return free

    def report(self) -> str:
        """Human-readable dim map."""
        lines = [f"Dimension Map (d_model={self.d_model})", "=" * 60]

        sorted_slots = sorted(self.slots.values(), key=lambda s: s.start)
        for slot in sorted_slots:
            if slot.size == 1:
                lines.append(f"  Dim {slot.start:3d}       : {slot.name:20s} — {slot.desc}")
            else:
                lines.append(
                    f"  Dims {slot.start:3d}-{slot.end-1:<3d}  : "
                    f"{slot.name:20s} ({slot.size:2d}) — {slot.desc}"
                )

        free = self.free_ranges()
        if free:
            lines.append("")
            lines.append("Free ranges:")
            for s, e in free:
                lines.append(f"  [{s}, {e})  ({e - s} dims)")

        used = sum(s.size for s in self.slots.values())
        lines.append("")
        lines.append(f"Used: {used}/{self.d_model} dims, Free: {self.d_model - used}")
        return "\n".join(lines)

    def resolve_names(self, patterns: List[str]) -> List[str]:
        """Resolve slot name patterns (supports trailing * wildcard)."""
        result = []
        for pat in patterns:
            if pat.endswith("*"):
                prefix = pat[:-1]
                matched = [n for n in self.slots if n.startswith(prefix)]
                if not matched:
                    result.append(pat)  # keep unresolved for error reporting
                else:
                    result.extend(matched)
            else:
                result.append(pat)
        return result


class ContractValidator:
    """Validates layer contracts against a DimRegistry."""

    def __init__(self, registry: DimRegistry, layers: List[LayerIO]):
        self.registry = registry
        self.layers = layers

    def validate(self) -> List[str]:
        """Run all validation checks. Returns list of error/warning strings."""
        errors = []
        errors.extend(self._check_overlaps())
        errors.extend(self._check_slot_refs())
        errors.extend(self._check_write_before_read())
        errors.extend(self._check_double_writes())
        return errors

    def _check_overlaps(self) -> List[str]:
        return self.registry.check_overlaps()

    def _check_slot_refs(self) -> List[str]:
        """Check that all referenced slot names exist in registry."""
        errors = []
        for lio in self.layers:
            resolved_reads = self.registry.resolve_names(lio.reads)
            resolved_writes = self.registry.resolve_names(lio.writes)
            resolved_additive = self.registry.resolve_names(lio.additive_writes)
            for name in resolved_reads + resolved_writes + resolved_additive:
                if name not in self.registry.slots:
                    errors.append(f"UNKNOWN SLOT: {lio.layer} references '{name}'")
        return errors

    def _check_write_before_read(self) -> List[str]:
        """Check that every read has a prior write."""
        errors = []
        written: Set[str] = set()
        for lio in self.layers:
            resolved_reads = self.registry.resolve_names(lio.reads)
            for name in resolved_reads:
                if name not in self.registry.slots:
                    continue  # already caught by _check_slot_refs
                if name not in written:
                    errors.append(
                        f"READ-BEFORE-WRITE: {lio.layer} reads '{name}' "
                        f"but no prior layer writes it"
                    )
            resolved_writes = self.registry.resolve_names(lio.writes)
            resolved_additive = self.registry.resolve_names(lio.additive_writes)
            written.update(resolved_writes)
            written.update(resolved_additive)
        return errors

    def _check_double_writes(self) -> List[str]:
        """Warn when multiple layers write the same slot (unless marked additive)."""
        warnings = []
        writers: Dict[str, List[str]] = {}  # slot_name -> [layer_names]
        additive_slots: Set[str] = set()

        for lio in self.layers:
            resolved_writes = self.registry.resolve_names(lio.writes)
            resolved_additive = self.registry.resolve_names(lio.additive_writes)
            for name in resolved_writes:
                if name not in self.registry.slots:
                    continue
                writers.setdefault(name, []).append(lio.layer)
            for name in resolved_additive:
                if name not in self.registry.slots:
                    continue
                writers.setdefault(name, []).append(lio.layer)
                additive_slots.add(name)

        for name, layer_list in writers.items():
            if len(layer_list) > 1 and name not in additive_slots:
                warnings.append(
                    f"DOUBLE-WRITE: slot '{name}' written by "
                    f"{', '.join(layer_list)}"
                )
        return warnings

    def dep_graph(self) -> str:
        """Text dependency visualization: which layers produce/consume each slot."""
        lines = ["Layer Dependency Graph", "=" * 60]

        # Collect all slot names that are read or written
        all_slots: Set[str] = set()
        for lio in self.layers:
            resolved_reads = self.registry.resolve_names(lio.reads)
            resolved_writes = self.registry.resolve_names(lio.writes)
            resolved_additive = self.registry.resolve_names(lio.additive_writes)
            all_slots.update(resolved_reads)
            all_slots.update(resolved_writes)
            all_slots.update(resolved_additive)

        # For each slot, show writers -> readers
        slot_writers: Dict[str, List[str]] = {}
        slot_readers: Dict[str, List[str]] = {}
        for lio in self.layers:
            resolved_reads = self.registry.resolve_names(lio.reads)
            resolved_writes = self.registry.resolve_names(lio.writes)
            resolved_additive = self.registry.resolve_names(lio.additive_writes)
            for name in resolved_reads:
                slot_readers.setdefault(name, []).append(lio.layer)
            for name in resolved_writes + resolved_additive:
                slot_writers.setdefault(name, []).append(lio.layer)

        for name in sorted(all_slots):
            if name not in self.registry.slots:
                continue
            w = slot_writers.get(name, ["(none)"])
            r = slot_readers.get(name, ["(none)"])
            lines.append(f"  {name}:")
            lines.append(f"    writers: {', '.join(w)}")
            lines.append(f"    readers: {', '.join(r)}")

        # Layer execution order summary
        lines.append("")
        lines.append("Execution Order:")
        for lio in self.layers:
            resolved_reads = self.registry.resolve_names(lio.reads)
            resolved_writes = self.registry.resolve_names(lio.writes)
            resolved_additive = self.registry.resolve_names(lio.additive_writes)
            r_str = ", ".join(resolved_reads) if resolved_reads else "(none)"
            w_str = ", ".join(resolved_writes + resolved_additive) if (resolved_writes or resolved_additive) else "(none)"
            lines.append(f"  {lio.layer}:")
            if lio.notes:
                lines.append(f"    {lio.notes}")
            lines.append(f"    reads:  {r_str}")
            lines.append(f"    writes: {w_str}")

        return "\n".join(lines)


# ============================================================================
# Default registry matching current _BakeDim allocations
# ============================================================================

def build_default_registry() -> DimRegistry:
    """Build a DimRegistry matching the current _BakeDim allocations (d_model=512).

    Every alloc carries a ``semantics=`` predicate string (parseable by
    ``neural_vm.unified_compiler.predicates.parse``) describing where the
    slot's value fires across the token sequence. Predicates are
    best-effort and conservative: a permissive-but-parseable predicate is
    preferred to no predicate so that the F-3 tolerant shim can flip to
    hard-error once F-4 lands. See ``# FIXME(F-4): ...`` comments where
    the exact semantics is uncertain.
    """
    # d_model expanded from 512 → 736 to fit the ``pin_io_only=True``
    # compact layout's high-position dims (510..732). The historical 512-
    # dim positions are preserved unchanged below; the new compact-layout
    # aliases live at positions 510..732 with ``_PIN`` suffixes so that
    # ``verify_attention_head`` can resolve dim ints from the compiled IR
    # back to named slots. See the ``# --- Compact pin_io_only layout
    # mirrors ---`` block below ``OPCODE_BASE`` for the suffix family.
    #
    # B16: the registry now builds itself via
    # :class:`neural_vm.dim_allocator.Allocator`. Every dim is ``pin=``-ed
    # to its historical start so the resulting layout is byte-identical
    # to the pre-allocator hand-rolled form (trained weights stay valid
    # because nothing moves). Intentional aliases pass ``alias=True`` to
    # bypass the allocator's collision check; once new dims start to
    # land via the allocator's auto-placement path, ``pin=`` arguments
    # can be progressively dropped family-by-family.
    from neural_vm.dim_allocator import Allocator

    a = Allocator(d_model=736)

    def _pin(name, start, size, desc, semantics=None, alias=False):
        """Thin wrapper that mirrors ``DimRegistry.alloc``'s signature so
        the body below stays visually identical to its pre-allocator
        form. ``alias=True`` toggles ``allow_overlap`` for the many
        intentional alias allocations (FETCH_LO==MUL_ACCUM,
        FORMAT_PTR_LO==AX_FULL_LO, OPCODE_BYTE_LO==ADDR_B0_LO, etc.).
        """
        a.alloc(
            name, size,
            pin=start,
            description=desc,
            semantics=semantics,
            allow_overlap=alias,
        )

    # Marker identity flags (set by embedding)
    _pin("MARK_PC",      0, 1, "PC register marker flag",
              semantics="mark == PC")
    _pin("MARK_AX",      1, 1, "AX register marker flag",
              semantics="mark == AX")
    _pin("MARK_SP",      2, 1, "SP register marker flag",
              semantics="mark == SP")
    _pin("MARK_BP",      3, 1, "BP register marker flag",
              semantics="mark == BP")
    _pin("MARK_MEM",     4, 1, "MEM marker flag",
              semantics="mark == MEM")
    _pin("MARK_SE",      5, 1, "STEP_END/DATA_END marker flag",
              semantics="mark == SE")
    _pin("IS_BYTE",      6, 1, "Token is a byte value (0-255)",
              semantics="is_byte")
    _pin("IS_MARK",      7, 1, "Token is a marker",
              semantics="NOT is_byte")
    # CONST is always-on (set to 1.0 at every token by the embedding); model
    # as a tautology so the predicate parses but imposes no firing constraint.
    _pin("CONST",        8, 1, "Constant 1.0 on all tokens",
              semantics="is_byte OR NOT is_byte")
    # FIXME(F-4): refine — CODE_START is a distinct marker token but the DSL
    # lacks a CS role; conservatively model as "any marker" (NOT is_byte).
    _pin("MARK_CS",      9, 1, "CODE_START only marker",
              semantics="NOT is_byte")
    _pin("MARK_SE_ONLY", 10, 1, "STEP_END only (not DATA_END)",
              semantics="mark == SE")
    _pin("MARK_STACK0",  11, 1, "STACK0 marker flag",
              semantics="mark == STACK0")

    # Address byte nibbles (gathered by memory address layers). Each 16-wide
    # slot is a one-hot encoding written at MEM-adjacent positions during
    # address gather; describe the slot family by its firing region.
    # FIXME(F-4): refine per-cell — the slot AS A WHOLE fires at MEM-region
    # positions; individual cells fire on specific nibble values. The DSL
    # has no per-cell hook, so the slot-level predicate is conservative.
    _pin("ADDR_B0_LO",  12, 16, "One-hot addr byte 0 low nibble",
              semantics="mark == MEM")
    _pin("ADDR_B1_LO",  28, 16, "One-hot addr byte 1 low nibble",
              semantics="mark == MEM")
    _pin("ADDR_B2_LO",  44, 16, "One-hot addr byte 2 low nibble",
              semantics="mark == MEM")

    # Layer 0 attention output: 8 threshold heads. Each 7-wide slot is one
    # cell per marker type, written everywhere by the threshold attention
    # so the firing condition is essentially "every token where a marker
    # is within threshold distance" — conservatively model as a tautology.
    # FIXME(F-4): refine — head outputs are best characterized by
    # marker-distance proximity which the DSL does not yet expose.
    _pin("H0",  60, 7, "L0 head 0: marker within dist 3.5",
              semantics="is_byte OR NOT is_byte")
    _pin("H1",  67, 7, "L0 head 1: marker within dist 4.5",
              semantics="is_byte OR NOT is_byte")
    _pin("H2",  74, 7, "L0 head 2: marker within dist 5.5",
              semantics="is_byte OR NOT is_byte")
    _pin("H3",  81, 7, "L0 head 3: marker within dist 9.5",
              semantics="is_byte OR NOT is_byte")
    _pin("H4",  88, 7, "L0 head 4: marker within dist 10.5",
              semantics="is_byte OR NOT is_byte")
    _pin("H5",  95, 7, "L0 head 5: marker within dist 14.5",
              semantics="is_byte OR NOT is_byte")
    _pin("H6", 102, 7, "L0 head 6: marker within dist 15.5",
              semantics="is_byte OR NOT is_byte")
    _pin("H7", 109, 7, "L0 head 7: marker within dist 19.5",
              semantics="is_byte OR NOT is_byte")

    # Layer 1 attention output: fine thresholds + SE detect
    # FIXME(F-4): refine — same fine-threshold characterization gap as L0.
    _pin("L1H0", 116, 7, "L1 head 0: marker within dist 0.5",
              semantics="is_byte OR NOT is_byte")
    _pin("L1H1", 123, 7, "L1 head 1: marker within dist 1.5",
              semantics="is_byte OR NOT is_byte")
    _pin("L1H2", 130, 7, "L1 head 2: marker within dist 2.5",
              semantics="is_byte OR NOT is_byte")
    _pin("HAS_SE", 137, 1, "STEP_END existence flag",
              semantics="has_se")

    # Byte index within register (4-byte register layout). The byte_index
    # atom fires on byte tokens at the given offset within their register.
    _pin("BYTE_INDEX_0", 138, 1, "Byte index 0 flag",
              semantics="is_byte AND byte_index == 0")
    _pin("BYTE_INDEX_1", 139, 1, "Byte index 1 flag",
              semantics="is_byte AND byte_index == 1")
    _pin("BYTE_INDEX_2", 140, 1, "Byte index 2 flag",
              semantics="is_byte AND byte_index == 2")
    _pin("BYTE_INDEX_3", 141, 1, "Byte index 3 flag",
              semantics="is_byte AND byte_index == 3")

    # Nibble encoding. EMBED_* is set by the embedding at byte tokens
    # (one-hot per nibble); OUTPUT_* is written by the decoder ops.
    # FIXME(F-4): refine — the 16-wide slot fires at byte positions; each
    # cell encodes a specific nibble value (byte_value.lo_nibble == c).
    _pin("EMBED_LO",  142, 16, "Embedding input low nibble (one-hot)",
              semantics="is_byte")
    _pin("EMBED_HI",  158, 16, "Embedding input high nibble (one-hot)",
              semantics="is_byte")
    # FIXME(F-4): refine — OUTPUT_* is autoregressively populated at marker
    # positions where the next emitted token is a byte; conservatively
    # model as "any token" since the DSL has no "next-token-is-byte" atom.
    _pin("OUTPUT_LO", 174, 16, "Output decoding low nibble (one-hot)",
              semantics="is_byte OR NOT is_byte")
    _pin("OUTPUT_HI", 190, 16, "Output decoding high nibble (one-hot)",
              semantics="is_byte OR NOT is_byte")
    # Phase 7.A.3 OUTPUT_LO split: OUTPUT_LO_PREV_STEP aliases the same
    # numeric slot as OUTPUT_LO so byte-identity is preserved. Cross-step
    # readers (L3 head 5 AX_FULL relay, L8 head 6 AX_CARRY refresh) attend
    # back to the prior step's AX marker row, where this slot holds the
    # prev-step value. Mirrors the B9 OUTPUT_HI_THIS_STEP pattern. See
    # docs/B9_OUTPUT_HI_SPLIT_SPEC.md.
    _pin("OUTPUT_LO_PREV_STEP", 174, 16,
              "OUTPUT_LO from previous step (aliases OUTPUT_LO)",
              semantics="is_byte OR NOT is_byte", alias=True)

    # Memory address key (3 nibbles × 16 one-hot = 48 dims). Written at
    # positions feeding the L15 memory lookup attention.
    # FIXME(F-4): refine — fires at MEM-region query positions.
    _pin("ADDR_KEY", 206, 48, "One-hot address key for memory matching (3 nibbles x 16)",
              semantics="mark == MEM")

    # NEXT_* transition flags. Written at the marker position preceding
    # the transition; conservatively scoped to "any marker" since the
    # exact preceding-marker family varies per NEXT_*.
    # FIXME(F-4): refine — NEXT_PC fires at MARK_SE positions (transition
    # from STEP_END to next step's PC); same pattern for the other NEXT_*.
    _pin("NEXT_PC",     254, 1, "Next token is PC register",
              semantics="NOT is_byte")
    _pin("NEXT_AX",     255, 1, "Next token is AX register",
              semantics="NOT is_byte")
    _pin("NEXT_SP",     256, 1, "Next token is SP register",
              semantics="NOT is_byte")
    _pin("NEXT_BP",     257, 1, "Next token is BP register",
              semantics="NOT is_byte")
    _pin("NEXT_STACK0", 258, 1, "Next token is STACK0 marker",
              semantics="NOT is_byte")
    _pin("NEXT_MEM",    259, 1, "Next token is MEM marker",
              semantics="NOT is_byte")
    _pin("NEXT_SE",     260, 1, "Next token is STEP_END",
              semantics="NOT is_byte")
    _pin("NEXT_HALT",   261, 1, "Emit HALT instead of STEP_END",
              semantics="NOT is_byte")

    # Opcode one-hot flags (34 opcodes). The slot is written at AX byte
    # positions during step decode, indicating which opcode is active.
    # FIXME(F-4): refine per-cell — each cell fires when the active
    # opcode matches that index (opcode_at_AX == OP_NAME). Slot-level
    # predicate is "any AX byte position" conservatively widened.
    _pin("OPCODE_FLAGS", 262, 34, "One-hot opcode flags (LEA..GETCHAR)",
              semantics="mark == AX OR (is_byte AND byte_index == 0)")

    # IO PUTCHAR flag — fires at the AX marker when current opcode is PUTCHAR.
    _pin("IO_IS_PUTCHAR", 296, 1, "OP_PUTCHAR detected this step (L6 FFN)",
              semantics="mark == AX AND opcode_at_AX == PUTCHAR")

    # ADJ implementation dimensions (SP + signed immediate). Written during
    # ADJ opcode execution at SP byte positions.
    # FIXME(F-4): refine — these stagings fire at SP byte positions when
    # opcode_in_step contains ADJ.
    _pin("SP_OLD_LO", 297, 8, "ADJ: old SP value low nibbles (4 bytes)",
              semantics="(mark == SP OR mark == AX) AND opcode_in_step in {ADJ}")
    _pin("SP_OLD_HI", 305, 8, "ADJ: old SP value high nibbles (4 bytes)",
              semantics="(mark == SP OR mark == AX) AND opcode_in_step in {ADJ}")
    _pin("ADJ_CARRY", 313, 2, "ADJ: multi-byte carry propagation",
              semantics="(mark == SP OR mark == AX) AND opcode_in_step in {ADJ}")

    # Reserved (remaining space for ENT/LEV)
    # FIXME(F-4): refine — placeholder reserved slot; model as never-fires
    # via an unsatisfiable conjunction (mark must be both PC and AX).
    _pin("RESERVED_315_327", 315, 13, "Reserved (ENT/LEV staging)",
              semantics="mark == PC AND mark == AX")

    # AX carry-forward staging. Populated at AX byte positions by the
    # carry-forward attention so downstream layers can read AX as a value.
    _pin("AX_CARRY_LO", 328, 16, "Carried-forward AX lo nibble",
              semantics="mark == AX OR (is_byte AND byte_index == 0)")
    _pin("AX_CARRY_HI", 344, 16, "Carried-forward AX hi nibble",
              semantics="mark == AX OR (is_byte AND byte_index == 0)")

    # ALU result staging. Written at AX byte positions when an ALU opcode
    # is active in the current step.
    # FIXME(F-4): refine — exact ALU opcode set is ADD/SUB/MUL/DIV/MOD/
    # OR/XOR/AND/SHL/SHR; conservatively gate on AX position only.
    _pin("ALU_LO", 360, 16, "ALU result lo nibble",
              semantics="mark == AX OR (is_byte AND byte_index == 0)")
    _pin("ALU_HI", 376, 16, "ALU result hi nibble",
              semantics="mark == AX OR (is_byte AND byte_index == 0)")

    # Carry / comparison cascade. Per-byte carry propagation slot for
    # add/sub/mul cascade and comparison-flag bus.
    # FIXME(F-4): refine — fires at AX byte positions during ALU ops.
    _pin("CARRY", 392, 4, "Inter-byte carry for ADD/SUB/MUL",
              semantics="is_byte AND byte_index in {0, 1, 2, 3}")
    _pin("CMP",   396, 4, "Comparison cascade: LT, EQ, GT, ZERO",
              semantics="mark == AX OR (is_byte AND byte_index == 0)")

    # Reserved (formerly PC_BIT, available for future use)
    # FIXME(F-4): refine — unused; model as unsatisfiable.
    _pin("RESERVED_400_415", 400, 16, "Reserved (future PC binary encoding/IO)",
              semantics="mark == PC AND mark == AX")

    # MUL/DIV staging — written at AX byte positions when MUL/DIV active.
    # FIXME(F-4): refine — exact gating opcode set unknown at slot level.
    _pin("MUL_ACCUM",   416, 16, "Multiplication accumulator",
              semantics="mark == AX OR (is_byte AND byte_index == 0)")
    _pin("DIV_STAGING", 432, 16, "Division quotient/remainder",
              semantics="mark == AX OR (is_byte AND byte_index == 0)")

    # Immediate staging — fetched immediate bytes following the opcode.
    # FIXME(F-4): refine — fires at PC byte positions during fetch.
    _pin("IMM_STAGING", 448, 16, "Fetched immediate bytes",
              semantics="mark == PC OR (is_byte AND byte_index in {0, 1, 2, 3})")

    # CS distance thermometer — thermometer-coded distance from CODE_START.
    # FIXME(F-4): refine — fires at every position; model as tautology
    # since the DSL has no "distance from CS" atom.
    _pin("CS_DIST_THERMO", 464, 16, "Thermometer-coded distance from CODE_START",
              semantics="is_byte OR NOT is_byte")

    # General temporaries / reserved scratch. Many sub-uses (PRTF/READ
    # capture state, OUTPUT_BYTE) so we use a permissive tautology.
    # FIXME(F-4): refine — split TEMP into sub-slots with per-feature
    # predicates once consumers stabilize.
    _pin("TEMP", 480, 32, "General temporaries / reserved",
              semantics="is_byte OR NOT is_byte")
    # Phase 7.A.3 TEMP split: TEMP_PREV_STEP aliases the same numeric slot
    # as TEMP so byte-identity is preserved. Mirrors the B9 OUTPUT_HI /
    # Phase 7.A.3.b OUTPUT_LO PREV_STEP pattern. TEMP is heavily
    # cell-multiplexed; the alias documents the prev-step semantic and
    # leaves room for future cross-step migrations.
    _pin("TEMP_PREV_STEP", 480, 32,
              "TEMP from previous step (aliases TEMP)",
              semantics="is_byte OR NOT is_byte", alias=True)

    # =========================================================================
    # F-4-extension: per-opcode sub-offsets, MEM/STACK0 control flags,
    # FETCH/AX_FULL aliases, CLEAN_EMBED, and IO/PRTF/READ scratch.
    # These OVERLAP with the parent slot they live within (OPCODE_FLAGS,
    # MUL_ACCUM, DIV_STAGING, TEMP, etc.) — overlaps are NOT checked at
    # alloc() time (only the validator's check_overlaps() reports them),
    # and the verifier needs each name individually present so it can
    # look up a semantics predicate per dim.
    # =========================================================================

    # --- Per-opcode flags (sub-offsets within OPCODE_FLAGS 262..295) ---
    # FIXME(F-4-ext): refine — each fires at AX byte positions when the
    # active opcode matches the named opcode; conservative gate models
    # this as "mark == AX AND opcode_at_AX == NAME" so the verifier can
    # see both the position and opcode constraints.
    _OPCODES = [
        ("OP_LEA", 262), ("OP_IMM", 263), ("OP_JMP", 264), ("OP_JSR", 265),
        ("OP_BZ",  266), ("OP_BNZ", 267), ("OP_ENT", 268), ("OP_ADJ", 269),
        ("OP_LEV", 270), ("OP_LI",  271), ("OP_LC",  272), ("OP_SI",  273),
        ("OP_SC",  274), ("OP_PSH", 275), ("OP_OR",  276), ("OP_XOR", 277),
        ("OP_AND", 278), ("OP_EQ",  279), ("OP_NE",  280), ("OP_LT",  281),
        ("OP_GT",  282), ("OP_LE",  283), ("OP_GE",  284), ("OP_SHL", 285),
        ("OP_SHR", 286), ("OP_ADD", 287), ("OP_SUB", 288), ("OP_MUL", 289),
        ("OP_DIV", 290), ("OP_MOD", 291), ("OP_EXIT", 292), ("OP_NOP", 293),
        ("OP_PUTCHAR", 294), ("OP_GETCHAR", 295),
    ]
    # Reserved DSL keywords (AND/OR/NOT) cannot appear as opcode names in
    # the predicate grammar, so OP_OR/OP_AND fall back to a position-only
    # gate. FIXME(F-4-ext): extend the DSL grammar to allow quoted/escaped
    # opcode literals so OP_OR/OP_AND can be tightened.
    _DSL_KEYWORDS = {"OR", "AND", "NOT"}
    for _name, _pos in _OPCODES:
        # Strip the OP_ prefix to get the opcode atom name used in the DSL.
        _opname = _name[3:]
        if _opname in _DSL_KEYWORDS:
            _sem = "mark == AX OR (is_byte AND byte_index == 0)"
        else:
            _sem = f"mark == AX AND opcode_at_AX == {_opname}"
        _pin(
            _name, _pos, 1,
            f"OPCODE_FLAGS[{_pos - 262}] = {_name} active flag",
            semantics=_sem,
            alias=True,
        )

    # --- STACK0 byte position flags (304, 508, 509, 510) ---
    # Set at STACK0 byte positions; conservatively gate on MARK_STACK0
    # since byte_index alone doesn't tell us which marker family.
    # FIXME(F-4-ext): STACK0_BYTE0 fires at byte_index==0 within a STACK0
    # cluster; this widening covers any STACK0-marked position.
    _pin("STACK0_BYTE0", 304, 1, "STACK0 byte 0 position flag",
              semantics="mark == STACK0 OR (is_byte AND byte_index == 0)",
              alias=True)
    # FIXME(F-4-ext): STACK0_BYTE1/2/3 alias TEMP+28/29/30 (positions
    # 508-510 = TEMP base 480 + 28..30). Overlap with TEMP is intentional.
    _pin("STACK0_BYTE1", 508, 1, "STACK0 byte 1 position flag",
              semantics="mark == STACK0 OR (is_byte AND byte_index == 1)",
              alias=True)
    _pin("STACK0_BYTE2", 509, 1, "STACK0 byte 2 position flag",
              semantics="mark == STACK0 OR (is_byte AND byte_index == 2)",
              alias=True)
    _pin("STACK0_BYTE3", 510, 1, "STACK0 byte 3 position flag",
              semantics="mark == STACK0 OR (is_byte AND byte_index == 3)",
              alias=True)

    # --- L1H4 / L2H0 fine threshold heads ---
    # FIXME(F-4-ext): refine — head outputs are best characterized by
    # marker-distance proximity; permissive tautology mirrors the H*
    # head treatments above.
    _pin("L1H4", 297, 7, "L1 head 4: threshold 6.5 from nearest IS_MARK",
              semantics="is_byte OR NOT is_byte", alias=True)
    _pin("L2H0", 452, 7, "L2 head 0: threshold 5.5 from nearest IS_MARK",
              semantics="is_byte OR NOT is_byte", alias=True)

    # --- L0 head 5 aliases (SP_BYTE0_IS_F8, IN_STEP_FRESH, ADDR_B0_VALID,
    # SP_GATHERED_THIS_STEP). These alias H5+0..H5+3 and carry pseudo-boolean
    # state set by later layers; use boolean atoms when supported by the DSL.
    _pin("SP_BYTE0_IS_F8", 95, 1, "SP byte 0 equals 0xF8 (aliases H5+0)",
              semantics="(mark == SP OR mark == AX) AND sp_byte0 == 0xF8",
              alias=True)
    _pin("IN_STEP_FRESH", 96, 1, "In-step freshness flag (aliases H5+1)",
              semantics="in_step_fresh", alias=True)
    _pin("ADDR_B0_VALID", 97, 1, "Address byte 0 gathered (aliases H5+2)",
              semantics="addr_b0_valid", alias=True)
    _pin("SP_GATHERED_THIS_STEP", 98, 1,
              "SP gather fired this step (aliases H5+3)",
              semantics="sp_gathered_this_step", alias=True)
    # B8-A: ADDR_B1_VALID / ADDR_B2_VALID lifecycle bits — mirror ADDR_B0_VALID
    # on L13 heads 1 and 2 (slot 34 of each head). Aliases H5+4 / H5+5, the
    # next two dormant L0 head-5 lanes after the B7 quartet at 95-98.
    _pin("ADDR_B1_VALID", 99, 1, "Address byte 1 gathered (aliases H5+4)",
              semantics="addr_b1_valid", alias=True)
    _pin("ADDR_B2_VALID", 100, 1, "Address byte 2 gathered (aliases H5+5)",
              semantics="addr_b2_valid", alias=True)

    # --- CMP_GROUP (305): set at AX when any cmp opcode active ---
    # FIXME(F-4-ext): refine — fires only at AX positions during cmp ops.
    _pin("CMP_GROUP", 305, 1, "Any EQ/NE/LT/GT/LE/GE active at AX",
              semantics="mark == AX AND opcode_at_AX in {EQ, NE, LT, GT, LE, GE}",
              alias=True)

    # --- CLEAN_EMBED nibbles (clones of EMBED, written after a clean pass) ---
    # Fire at byte positions (mirror EMBED_LO/HI semantics).
    _pin("CLEAN_EMBED_LO", 306, 16, "Clean embedding lo nibble (one-hot)",
              semantics="is_byte", alias=True)
    _pin("CLEAN_EMBED_HI", 404, 16, "Clean embedding hi nibble (one-hot)",
              semantics="is_byte", alias=True)

    # --- IO/conversational tool-call transition + state flags ---
    # FIXME(F-4-ext): refine — these are autoregressive transition flags
    # set at marker positions to drive next-token emission; conservatively
    # gate on "any marker" since the DSL has no PRTF/READ atom.
    _pin("IO_IS_TOOL_CALL", 322, 1, "Any of OPEN/READ/CLOS/PRTF active",
              semantics="mark == AX", alias=True)
    _pin("NEXT_TOOL_CALL",        323, 1, "Next token is TOOL_CALL",
              semantics="NOT is_byte", alias=True)
    _pin("NEXT_THINKING_START",   324, 1, "Next token is THINKING_START",
              semantics="NOT is_byte", alias=True)
    _pin("NEXT_THINKING_END",     325, 1, "Next token is THINKING_END",
              semantics="NOT is_byte", alias=True)
    _pin("NEXT_IO_STATE_EMIT_BYTE",      326, 1,
              "Next token is IO_STATE_EMIT_BYTE",
              semantics="NOT is_byte", alias=True)
    _pin("NEXT_IO_STATE_EMIT_THINKING",  327, 1,
              "Next token is IO_STATE_EMIT_THINKING",
              semantics="NOT is_byte", alias=True)

    # --- FETCH aliases — fetched immediate nibble staging.
    # FETCH_LO/HI alias MUL_ACCUM (420..435) and DIV_STAGING (436..451).
    # FIXME(F-4-ext): refine — fired at PC byte positions during opcode
    # fetch; current MUL_ACCUM/DIV_STAGING base predicates use AX gating.
    _pin("FETCH_LO", 420, 16, "Fetched immediate lo nibble (aliases MUL_ACCUM)",
              semantics="mark == PC OR (is_byte AND byte_index in {0, 1, 2, 3})",
              alias=True)
    _pin("FETCH_HI", 436, 16, "Fetched immediate hi nibble (aliases DIV_STAGING)",
              semantics="mark == PC OR (is_byte AND byte_index in {0, 1, 2, 3})",
              alias=True)

    # --- ADDR_B0_HI / ADDR_B1_HI / ADDR_B2_HI — hi nibble of gathered addr
    # bytes. These OVERLAP with ADDR_KEY (206..253), which itself is the
    # 3 nibbles × 16 one-hot. The hi-nibble overlay is read at MEM-region
    # positions just like ADDR_B*_LO.
    _pin("ADDR_B0_HI", 206, 16, "Gathered addr byte 0 hi nibble (aliases ADDR_KEY[0:16])",
              semantics="mark == MEM", alias=True)
    _pin("ADDR_B1_HI", 222, 16, "Gathered addr byte 1 hi nibble (aliases ADDR_KEY[16:32])",
              semantics="mark == MEM", alias=True)
    _pin("ADDR_B2_HI", 238, 16, "Gathered addr byte 2 hi nibble (aliases ADDR_KEY[32:48])",
              semantics="mark == MEM", alias=True)

    # --- MEM control + value bus (459..464 plus relays 465..467) ---
    # FIXME(F-4-ext): refine — MEM_STORE is set at MEM positions when a
    # store op (SI/SC/PSH) is active; the {SI, SC, PSH} gate captures
    # the intent at the slot level.
    _pin("MEM_STORE", 459, 1,
              "Store op (SI/SC/PSH) active, relayed to MEM positions",
              semantics="mark == MEM AND opcode_in_step in {SI, SC, PSH}",
              alias=True)
    _pin("MEM_ADDR_SRC", 460, 1,
              "Address source: 1=STACK0 (SI/SC), 0=SP (PSH)",
              semantics="mark == MEM AND opcode_in_step in {SI, SC, PSH}",
              alias=True)
    _pin("MEM_VAL_B0", 461, 1, "Predicts MEM val byte 0 (d=4 from MEM)",
              semantics="mark == MEM OR (is_byte AND byte_index == 0)",
              alias=True)
    _pin("MEM_VAL_B1", 462, 1, "Predicts MEM val byte 1 (d=5 from MEM)",
              semantics="mark == MEM OR (is_byte AND byte_index == 1)",
              alias=True)
    _pin("MEM_VAL_B2", 463, 1, "Predicts MEM val byte 2 (d=6 from MEM)",
              semantics="mark == MEM OR (is_byte AND byte_index == 2)",
              alias=True)
    _pin("MEM_VAL_B3", 464, 1, "Predicts MEM val byte 3 (d=7 from MEM)",
              semantics="mark == MEM OR (is_byte AND byte_index == 3)",
              alias=True)

    # --- LI/LC opcode relays + PSH-at-SP flag ---
    # These alias IO_IS_PRTF/IO_IS_READ/IO_STATE/IO_OUTPUT_COUNT positions.
    _pin("OP_LI_RELAY", 465, 1, "LI active, relayed to AX byte positions",
              semantics="mark == AX AND opcode_in_step in {LI}",
              alias=True)
    _pin("OP_LC_RELAY", 466, 1, "LC active, relayed to AX byte positions",
              semantics="mark == AX AND opcode_in_step in {LC}",
              alias=True)
    _pin("PSH_AT_SP", 467, 1, "PSH opcode flag relayed to SP/STACK0",
              semantics="(mark == SP OR mark == STACK0) AND opcode_in_step in {PSH}",
              alias=True)

    # --- PRTF/READ IO state machine scratch (alias MEM_VAL/RELAY slots) ---
    # FIXME(F-4-ext): refine — these are autoregressive IO state flags;
    # firing positions depend on the PRTF/READ tool-call state machine.
    # Conservative gating: AX positions when PRTF/READ-class opcode active.
    _pin("IO_IS_PRTF", 464, 1, "PRTF opcode detected (aliases MEM_VAL_B3)",
              semantics="mark == AX", alias=True)
    _pin("IO_IS_READ", 465, 1, "READ opcode detected (aliases OP_LI_RELAY)",
              semantics="mark == AX", alias=True)
    _pin("IO_STATE",   466, 1, "IO state machine (aliases OP_LC_RELAY)",
              semantics="mark == AX OR NOT is_byte", alias=True)
    _pin("IO_OUTPUT_COUNT", 467, 1,
              "Output bytes remaining (aliases PSH_AT_SP)",
              semantics="mark == AX OR NOT is_byte", alias=True)
    _pin("IO_FORMAT_POS",   468, 1, "Position in format string (aliases MEM_EXEC)",
              semantics="mark == AX OR NOT is_byte", alias=True)
    _pin("MEM_EXEC", 468, 1, "Deprecated; retained as IO_FORMAT_POS alias",
              semantics="mark == AX OR NOT is_byte", alias=True)
    _pin("IO_IN_OUTPUT_MODE",  469, 1, "Currently emitting output bytes",
              semantics="is_byte OR NOT is_byte", alias=True)
    _pin("IO_OUTPUT_COMPLETE", 470, 1, "Format string complete",
              semantics="is_byte OR NOT is_byte", alias=True)

    # --- FORMAT_PTR / AX_FULL nibble pointers (471..502 — two views) ---
    # FORMAT_PTR_* and AX_FULL_* share the same byte range; both are
    # populated at AX positions.
    _pin("FORMAT_PTR_LO", 471, 16, "Format string ptr lo nibble (aliases AX_FULL_LO)",
              semantics="mark == AX OR (is_byte AND byte_index == 0)",
              alias=True)
    _pin("FORMAT_PTR_HI", 487, 16, "Format string ptr hi nibble (aliases AX_FULL_HI)",
              semantics="mark == AX OR (is_byte AND byte_index == 0)",
              alias=True)
    _pin("AX_FULL_LO",    471, 16, "Full AX lo nibble (aliases FORMAT_PTR_LO)",
              semantics="mark == AX OR (is_byte AND byte_index == 0)",
              alias=True)
    _pin("AX_FULL_HI",    487, 16, "Full AX hi nibble (aliases FORMAT_PTR_HI)",
              semantics="mark == AX OR (is_byte AND byte_index == 0)",
              alias=True)

    # --- OUTPUT_BYTE nibbles — alias TEMP+0..15 / TEMP+16..31 ---
    # Set at marker positions where next-emitted token will be a byte.
    _pin("OUTPUT_BYTE_LO", 480, 16, "Output byte lo nibble (aliases TEMP[0:16])",
              semantics="is_byte OR NOT is_byte", alias=True)
    _pin("OUTPUT_BYTE_HI", 496, 16, "Output byte hi nibble (aliases TEMP[16:32])",
              semantics="is_byte OR NOT is_byte", alias=True)

    # --- "LAST_WAS_*" / "ACTIVE_OPCODE_*" / "MARK_THINKING_*" flags ---
    # All single-dim history/marker flags within the TEMP region.
    _pin("LAST_WAS_THINKING_END",   501, 1, "Prev token was THINKING_END",
              semantics="NOT is_byte", alias=True)
    _pin("LAST_WAS_THINKING_START", 502, 1, "Prev token was THINKING_START",
              semantics="NOT is_byte", alias=True)
    _pin("LAST_WAS_BYTE", 503, 1, "Prev token was byte (0-255)",
              semantics="is_byte OR NOT is_byte", alias=True)
    _pin("LAST_WAS_IO_STATE_EMIT_BYTE", 462, 1,
              "Prev token was IO_STATE_EMIT_BYTE (aliases MEM_VAL_B1)",
              semantics="is_byte OR NOT is_byte", alias=True)
    _pin("LAST_WAS_IO_STATE_EMIT_THINKING", 463, 1,
              "Prev token was IO_STATE_EMIT_THINKING (aliases MEM_VAL_B2)",
              semantics="is_byte OR NOT is_byte", alias=True)
    _pin("ACTIVE_OPCODE_PRTF", 504, 1, "Current opcode is PRTF",
              semantics="mark == AX AND opcode_at_AX == PRTF", alias=True)
    _pin("ACTIVE_OPCODE_READ", 505, 1, "Current opcode is READ",
              semantics="mark == AX AND opcode_at_AX == READ", alias=True)
    _pin("MARK_THINKING_START", 506, 1, "THINKING_START token marker",
              semantics="NOT is_byte", alias=True)
    _pin("MARK_THINKING_END",   507, 1, "THINKING_END token marker",
              semantics="NOT is_byte", alias=True)

    # --- POST_PRTF aliases (471..502 / 328..359) ---
    # FIXME(F-4-ext): refine — these alias AX_FULL/AX_CARRY ranges and are
    # populated when a PRTF tool-call returns; conservatively gate on AX.
    _pin("POST_PRTF_PC_LO", 471, 16, "Post-PRTF PC lo (aliases AX_FULL_LO)",
              semantics="mark == AX OR (is_byte AND byte_index == 0)",
              alias=True)
    _pin("POST_PRTF_PC_HI", 487, 16, "Post-PRTF PC hi (aliases AX_FULL_HI)",
              semantics="mark == AX OR (is_byte AND byte_index == 0)",
              alias=True)
    _pin("POST_PRTF_SP_LO", 328, 16, "Post-PRTF SP lo (aliases AX_CARRY_LO)",
              semantics="mark == AX OR (is_byte AND byte_index == 0)",
              alias=True)
    _pin("POST_PRTF_SP_HI", 344, 16, "Post-PRTF SP hi (aliases AX_CARRY_HI)",
              semantics="mark == AX OR (is_byte AND byte_index == 0)",
              alias=True)

    # --- Opcode-byte aliases (12, 28) — unused in autoregressive but
    # referenced by some legacy ops. Alias ADDR_B0_LO / ADDR_B1_LO.
    _pin("OPCODE_BYTE_LO", 12, 16, "Opcode byte lo nibble (aliases ADDR_B0_LO)",
              semantics="mark == MEM OR (is_byte AND byte_index == 0)",
              alias=True)
    _pin("OPCODE_BYTE_HI", 28, 16, "Opcode byte hi nibble (aliases ADDR_B1_LO)",
              semantics="mark == MEM OR (is_byte AND byte_index == 0)",
              alias=True)

    # --- OPCODE_BASE alias (262) — alias of OPCODE_FLAGS / OP_LEA. ---
    _pin("OPCODE_BASE", 262, 1, "Base of opcode one-hot (aliases OP_LEA)",
              semantics="mark == AX OR (is_byte AND byte_index == 0)",
              alias=True)

    # =========================================================================
    # Compact pin_io_only layout mirrors (positions 510..732)
    # -----------------------------------------------------------------------
    # The historical 512-dim positions above match the legacy ``_BakeDim`` /
    # ``pin_to_setdim`` layout. The current compiler runs with
    # ``pin_io_only=True`` (see ``declare_setdim_compat_dims``) which lays
    # out IO-required dims in a compact block at 0..~242 and bump-pointer-
    # allocates the remaining dims above that, producing a different set of
    # positions for the address / nibble / scratch families. The bake IR
    # emitted by attention ops references those compact positions directly,
    # so the registry needs slots at those positions for
    # ``verify_attention_head`` to resolve them by name.
    #
    # These ``_PIN`` aliases mirror the semantics of the legacy slots above
    # at the compact-layout positions. They are intentional overlays — the
    # legacy slot and the ``_PIN`` mirror describe the same dim family,
    # they just live at different positions across the two layouts. The
    # ``check_overlaps`` validator already tolerates the many intentional
    # legacy aliases (FETCH_LO==MUL_ACCUM, FORMAT_PTR_LO==AX_FULL_LO,
    # ADDR_B0_HI==ADDR_KEY[0:16], etc.) so the new ``_PIN`` overlays add no
    # new pattern.
    # =========================================================================

    # OPCODE_BYTE_HI mirror (compact pos 510..525). Legacy OPCODE_BYTE_HI
    # lives at 28 (aliased onto ADDR_B1_LO). Compact layout places it at 510.
    # Note: 510..526 straddles TEMP's tail (480..512) by 2 bytes, so this
    # mirror is registered as an alias of TEMP at the boundary.
    _pin("OPCODE_BYTE_HI_PIN", 510, 16,
              "Compact-layout OPCODE_BYTE_HI (mirrors legacy OPCODE_BYTE_HI at 28)",
              semantics="mark == MEM OR (is_byte AND byte_index == 0)",
              alias=True)

    # ADDR_B*_LO mirrors (compact pos 526..573). Legacy ADDR_B0_LO=12,
    # ADDR_B1_LO=28, ADDR_B2_LO=44 (one-hot low nibbles of gathered addr
    # bytes). Compact layout places the family at 526..573.
    _pin("ADDR_B0_LO_PIN", 526, 16,
              "Compact-layout ADDR_B0_LO (mirrors legacy at 12)",
              semantics="mark == MEM")
    _pin("ADDR_B1_LO_PIN", 542, 16,
              "Compact-layout ADDR_B1_LO (mirrors legacy at 28)",
              semantics="mark == MEM")
    _pin("ADDR_B2_LO_PIN", 558, 16,
              "Compact-layout ADDR_B2_LO (mirrors legacy at 44)",
              semantics="mark == MEM")

    # ADDR_B*_HI mirrors (compact pos 574..621). Legacy ADDR_B0_HI=206,
    # ADDR_B1_HI=222, ADDR_B2_HI=238 (hi nibbles, aliased onto ADDR_KEY).
    # Compact layout places the family at 574..621.
    _pin("ADDR_B0_HI_PIN", 574, 16,
              "Compact-layout ADDR_B0_HI (mirrors legacy at 206)",
              semantics="mark == MEM")
    _pin("ADDR_B1_HI_PIN", 590, 16,
              "Compact-layout ADDR_B1_HI (mirrors legacy at 222)",
              semantics="mark == MEM")
    _pin("ADDR_B2_HI_PIN", 606, 16,
              "Compact-layout ADDR_B2_HI (mirrors legacy at 238)",
              semantics="mark == MEM")

    # FORMAT_PTR_*/AX_FULL_* mirrors (compact pos 622..653). Legacy
    # FORMAT_PTR_LO=471, FORMAT_PTR_HI=487 (aliased with AX_FULL_LO/HI).
    # Compact layout places the family at 622..653.
    _pin("FORMAT_PTR_LO_PIN", 622, 16,
              "Compact-layout FORMAT_PTR_LO/AX_FULL_LO (mirrors legacy at 471)",
              semantics="mark == AX OR (is_byte AND byte_index == 0)")
    _pin("FORMAT_PTR_HI_PIN", 638, 16,
              "Compact-layout FORMAT_PTR_HI/AX_FULL_HI (mirrors legacy at 487)",
              semantics="mark == AX OR (is_byte AND byte_index == 0)")

    # OUTPUT_BYTE_* mirrors (compact pos 654..685). Legacy OUTPUT_BYTE_LO=480,
    # OUTPUT_BYTE_HI=496 (aliased onto TEMP). Compact layout places them at
    # 654..685.
    _pin("OUTPUT_BYTE_LO_PIN", 654, 16,
              "Compact-layout OUTPUT_BYTE_LO (mirrors legacy at 480)",
              semantics="is_byte OR NOT is_byte")
    _pin("OUTPUT_BYTE_HI_PIN", 670, 16,
              "Compact-layout OUTPUT_BYTE_HI (mirrors legacy at 496)",
              semantics="is_byte OR NOT is_byte")

    # CARRY mirror (compact pos 686..689). Legacy CARRY=392 (4-wide
    # inter-byte ADD/SUB/MUL carry cascade).
    _pin("CARRY_PIN", 686, 4,
              "Compact-layout CARRY (mirrors legacy at 392)",
              semantics="is_byte AND byte_index in {0, 1, 2, 3}")

    # CMP mirror (compact pos 690..697). Legacy CMP=396 (size 4 in legacy
    # registry; compact layout widens to 8 to match the declared size in
    # ``declare_setdim_compat_dims`` -- the ``eight_dim`` family).
    _pin("CMP_PIN", 690, 8,
              "Compact-layout CMP (mirrors legacy at 396; widened to 8)",
              semantics="mark == AX OR (is_byte AND byte_index == 0)")

    # TEMP mirror (compact pos 698..729). Legacy TEMP=480 size 32.
    _pin("TEMP_PIN", 698, 32,
              "Compact-layout TEMP (mirrors legacy at 480)",
              semantics="is_byte OR NOT is_byte")

    # STACK0_BYTE1/2/3 mirrors (compact pos 730..732). Legacy
    # STACK0_BYTE1/2/3 = 508/509/510 (aliased onto TEMP+28..30).
    _pin("STACK0_BYTE1_PIN", 730, 1,
              "Compact-layout STACK0_BYTE1 (mirrors legacy at 508)",
              semantics="mark == STACK0 OR (is_byte AND byte_index == 1)")
    _pin("STACK0_BYTE2_PIN", 731, 1,
              "Compact-layout STACK0_BYTE2 (mirrors legacy at 509)",
              semantics="mark == STACK0 OR (is_byte AND byte_index == 2)")
    _pin("STACK0_BYTE3_PIN", 732, 1,
              "Compact-layout STACK0_BYTE3 (mirrors legacy at 510)",
              semantics="mark == STACK0 OR (is_byte AND byte_index == 3)")

    reg = a.to_registry()
    _register_default_categories(reg)
    return reg


# ============================================================================
# Phase 7.E.1 — semantic category / role bindings for the default registry.
# ----------------------------------------------------------------------------
# Categories let Phase 7.E.2 rules reference dims by ``(category, role)``
# instead of ``"OUTPUT_LO+15"``-style strings. The mapping below covers
# the suggested initial set:
#
#   register_lo / register_hi     PC/AX/SP/BP/STACK0 lo/hi nibble families
#   memory_lo / memory_hi         MEM value bus + addr nibble families
#   output_lo / output_hi         Decoder output nibble families
#   temp_scratch                  General scratch
#   addr_key_nibble               Memory address one-hot key
#   opcode_flag                   One-hot opcode flags (LEA..GETCHAR)
#   marker                        Marker identity flags (PC/AX/SP/BP/MEM/SE/…)
#   byte_index                    Byte index within register
#   carry                         Inter-byte carry cascades
#   cmp_flag                      Comparison cascade
#   alu_lo / alu_hi               ALU result nibbles
#   ax_carry_lo / ax_carry_hi     AX carry-forward staging
#
# Roles inside a category are slot-level (the family base) for now;
# Phase 7.E.2 can attach per-cell roles if a rule needs ``+N`` granularity.
# Registration uses :meth:`DimRegistry.register_category` so this block is
# additive — existing slots aren't re-allocated and trained weights stay
# in place.
# ============================================================================
def _register_default_categories(reg: 'DimRegistry') -> None:
    """Tag :func:`build_default_registry` slots with semantic categories.

    Called by :func:`build_default_registry`; safe to skip in callers
    that don't need the ``(category, role)`` resolution path. Every
    pair must be unique — duplicate registration raises ``ValueError``
    via :meth:`DimRegistry._index_category`.
    """
    bindings = (
        # ---- marker (register identity flags + token-class flags) ----
        ("MARK_PC",      "marker", "PC"),
        ("MARK_AX",      "marker", "AX"),
        ("MARK_SP",      "marker", "SP"),
        ("MARK_BP",      "marker", "BP"),
        ("MARK_MEM",     "marker", "MEM"),
        ("MARK_SE",      "marker", "SE"),
        ("MARK_CS",      "marker", "CS"),
        ("MARK_SE_ONLY", "marker", "SE_ONLY"),
        ("MARK_STACK0",  "marker", "STACK0"),

        # ---- byte_index ----
        ("BYTE_INDEX_0", "byte_index", "0"),
        ("BYTE_INDEX_1", "byte_index", "1"),
        ("BYTE_INDEX_2", "byte_index", "2"),
        ("BYTE_INDEX_3", "byte_index", "3"),

        # ---- register_lo / register_hi (full + carry views are split
        # into their own ax_carry_* category; register_* refers to the
        # value-bus view at AX positions). The compiler-IR consumers
        # use ``AX_FULL_LO`` / ``AX_FULL_HI`` (471/487) for the AX
        # value bus; SP/BP/PC analogues are POST_PRTF_PC_LO/HI and
        # POST_PRTF_SP_LO/HI which alias those ranges. ----
        ("AX_FULL_LO",     "register_lo", "AX"),
        ("AX_FULL_HI",     "register_hi", "AX"),
        ("POST_PRTF_PC_LO", "register_lo", "PC"),
        ("POST_PRTF_PC_HI", "register_hi", "PC"),
        ("POST_PRTF_SP_LO", "register_lo", "SP"),
        ("POST_PRTF_SP_HI", "register_hi", "SP"),
        ("SP_OLD_LO",      "register_lo", "SP_OLD"),
        ("SP_OLD_HI",      "register_hi", "SP_OLD"),

        # ---- ax_carry_lo / ax_carry_hi ----
        ("AX_CARRY_LO", "ax_carry_lo", "AX"),
        ("AX_CARRY_HI", "ax_carry_hi", "AX"),

        # ---- alu_lo / alu_hi ----
        ("ALU_LO", "alu_lo", "result"),
        ("ALU_HI", "alu_hi", "result"),

        # ---- carry (inter-byte cascades) ----
        ("CARRY",     "carry", "alu"),
        ("ADJ_CARRY", "carry", "adj"),

        # ---- cmp_flag ----
        ("CMP",       "cmp_flag", "cascade"),
        ("CMP_GROUP", "cmp_flag", "group"),

        # ---- memory_lo / memory_hi (address byte nibbles + MEM value bus
        # single-bit cells). ADDR_B*_LO are the low-nibble one-hot
        # encodings; ADDR_B*_HI alias ADDR_KEY's first 3 nibbles. ----
        ("ADDR_B0_LO", "memory_lo", "addr_b0"),
        ("ADDR_B1_LO", "memory_lo", "addr_b1"),
        ("ADDR_B2_LO", "memory_lo", "addr_b2"),
        ("ADDR_B0_HI", "memory_hi", "addr_b0"),
        ("ADDR_B1_HI", "memory_hi", "addr_b1"),
        ("ADDR_B2_HI", "memory_hi", "addr_b2"),
        ("MEM_VAL_B0", "memory_lo", "val_b0"),
        ("MEM_VAL_B1", "memory_lo", "val_b1"),
        ("MEM_VAL_B2", "memory_lo", "val_b2"),
        ("MEM_VAL_B3", "memory_lo", "val_b3"),

        # ---- addr_key_nibble ----
        ("ADDR_KEY", "addr_key_nibble", "key"),

        # ---- output_lo / output_hi ----
        ("OUTPUT_LO",      "output_lo", "nibble"),
        ("OUTPUT_HI",      "output_hi", "nibble"),
        ("OUTPUT_BYTE_LO", "output_lo", "byte"),
        ("OUTPUT_BYTE_HI", "output_hi", "byte"),

        # ---- temp_scratch ----
        ("TEMP", "temp_scratch", "general"),
    )
    for slot_name, cat, role in bindings:
        reg.register_category(slot_name, cat, role)

    # ---- opcode_flag (one role per opcode, role is the opcode mnemonic) ----
    _OPCODE_ROLES = [
        "LEA", "IMM", "JMP", "JSR", "BZ", "BNZ", "ENT", "ADJ", "LEV",
        "LI", "LC", "SI", "SC", "PSH", "OR", "XOR", "AND", "EQ", "NE",
        "LT", "GT", "LE", "GE", "SHL", "SHR", "ADD", "SUB", "MUL",
        "DIV", "MOD", "EXIT", "NOP", "PUTCHAR", "GETCHAR",
    ]
    for op in _OPCODE_ROLES:
        reg.register_category(f"OP_{op}", "opcode_flag", op)


def build_default_contracts(registry: DimRegistry) -> List[LayerIO]:
    """Build layer contracts matching current set_vm_weights data flow (16 layers)."""
    return [
        LayerIO(
            layer="embed",
            reads=[],
            writes=[
                "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
                "MARK_SE", "IS_BYTE", "IS_MARK", "CONST", "MARK_CS",
                "MARK_SE_ONLY", "MARK_STACK0", "EMBED_LO", "EMBED_HI",
            ],
            notes="Embedding layer sets marker flags + nibble encodings",
        ),
        LayerIO(
            layer="L0_attn",
            reads=["CONST", "IS_MARK", "MARK_*"],
            writes=["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"],
            notes="Threshold attention: 8 heads for 39-token step structure",
        ),
        LayerIO(
            layer="L0_ffn",
            reads=["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"],
            writes=["NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
                    "NEXT_STACK0", "NEXT_MEM", "NEXT_SE"],
            notes="Detect register transitions for 39-token step",
        ),
        LayerIO(
            layer="L1_attn",
            reads=["CONST", "IS_MARK", "MARK_*"],
            writes=["L1H0", "L1H1", "L1H2", "HAS_SE"],
            notes="Fine thresholds 0.5/1.5/2.5 + global STEP_END detection",
        ),
        LayerIO(
            layer="L1_ffn",
            reads=["HAS_SE", "NEXT_SE"],
            writes=["NEXT_HALT"],
            notes="HALT override (instruction decode moved to L4-L5)",
        ),
        # Layers 2-6: instruction fetch + opcode decode + routing
        LayerIO(
            layer="L6_ffn_io",
            reads=["OPCODE_FLAGS", "MARK_AX", "AX_CARRY_LO", "AX_CARRY_HI"],
            writes=["IO_IS_PUTCHAR"],
            additive_writes=["OUTPUT_LO", "OUTPUT_HI"],
            notes="PUTCHAR: detect opcode, route AX_CARRY → OUTPUT (autoregressive)",
        ),
        # Layers 7-14: reserved for STACK0, ALU, MEM ops
        # GETCHAR: runner-side IO (not in weights, see run_vm.py)
        LayerIO(
            layer="L15_attn",
            reads=["ADDR_KEY", "EMBED_LO", "EMBED_HI"],
            writes=[],
            additive_writes=["OUTPUT_LO", "OUTPUT_HI"],
            notes="Memory lookup via softmax1 + ALiBi (ZFOD semantics)",
        ),
        LayerIO(
            layer="output_head",
            reads=["OUTPUT_LO", "OUTPUT_HI", "NEXT_PC", "NEXT_AX",
                   "NEXT_SP", "NEXT_BP", "NEXT_STACK0", "NEXT_MEM",
                   "NEXT_SE", "NEXT_HALT"],
            writes=[],
            notes="Nibble decoding + transition token selection",
        ),
    ]


def validate_default() -> Tuple[DimRegistry, List[str]]:
    """Build and validate the default registry + contracts. Returns (registry, errors)."""
    reg = build_default_registry()
    contracts = build_default_contracts(reg)
    validator = ContractValidator(reg, contracts)
    errors = validator.validate()
    return reg, errors


# ============================================================================
# Static Weight Inspection — auto-derive contracts from actual weight matrices
# ============================================================================

def _nonzero_dims(weight, axis: int) -> Set[int]:
    """Find dims with any non-zero entry along the given axis.

    Args:
        weight: 2D tensor (dense or sparse COO)
        axis: 0 = find non-zero columns (input dims), 1 = find non-zero rows (output dims)

    For axis=0: returns set of column indices where any row has non-zero.
        i.e. dims d where weight[:, d] has any non-zero → "reads from d"
    For axis=1: returns set of row indices where any column has non-zero.
        i.e. dims d where weight[d, :] has any non-zero → "writes to d"
    """
    import torch
    if weight.is_sparse:
        indices = weight.coalesce().indices()
        if indices.numel() == 0:
            return set()
        # indices[0] = row indices, indices[1] = column indices
        # axis=0 → want columns → indices[1]; axis=1 → want rows → indices[0]
        if axis == 0:
            return set(indices[1].tolist())
        else:
            return set(indices[0].tolist())
    else:
        # W shape [M, N]. sum(dim=0) → [N] (column sums), sum(dim=1) → [M] (row sums)
        # axis=0 (columns/inputs): sum over rows → dim=0 → shape [N] → column indices
        # axis=1 (rows/outputs): sum over cols → dim=1 → shape [M] → row indices
        summed = weight.abs().sum(dim=axis)
        return set(torch.nonzero(summed).squeeze(-1).tolist())


def _nonzero_bias_dims(bias) -> Set[int]:
    """Find dims where bias vector has non-zero entries."""
    import torch
    if bias is None:
        return set()
    if bias.is_sparse:
        indices = bias.coalesce().indices()
        if indices.numel() == 0:
            return set()
        return set(indices[0].tolist())
    else:
        return set(torch.nonzero(bias.abs()).squeeze(-1).tolist())


def _build_dim_to_slot(registry: DimRegistry) -> Dict[int, str]:
    """Map each dim index to its registry slot name.

    Dims not covered by any slot are omitted (reported as unregistered).
    """
    dim_to_slot = {}
    for name, slot in registry.slots.items():
        for d in slot.range:
            dim_to_slot[d] = name
    return dim_to_slot


def _build_setdim_lookup() -> Dict[int, str]:
    """Build a fallback lookup from _SetDim constants for unregistered dims.

    Returns a mapping from dim index → _SetDim attribute name (e.g. 455 → "MEM_STORE").
    Multi-dim ranges (16-wide nibble arrays etc.) get names like "CLEAN_EMBED_LO+3".
    """
    from .vm_step import _SetDim

    # Collect all integer class attributes
    attrs = {}
    for name in dir(_SetDim):
        if name.startswith('_') or name.startswith('NUM_') or name == 'MARKS':
            continue
        val = getattr(_SetDim, name)
        if isinstance(val, int) and 0 <= val < 512:
            attrs[name] = val

    # Sort by value to detect ranges — attrs with same value are aliases
    by_val = sorted(attrs.items(), key=lambda x: x[1])

    # Known multi-dim ranges (name, start, size)
    RANGES = [
        ("ADDR_B0_LO", 12, 16), ("ADDR_B1_LO", 28, 16), ("ADDR_B2_LO", 44, 16),
        ("EMBED_LO", 142, 16), ("EMBED_HI", 158, 16),
        ("OUTPUT_LO", 174, 16), ("OUTPUT_HI", 190, 16),
        ("ADDR_KEY", 206, 48),
        ("OPCODE_FLAGS", 262, 34),
        ("CLEAN_EMBED_LO", 306, 16), ("CLEAN_EMBED_HI", 400, 16),
        ("AX_CARRY_LO", 328, 16), ("AX_CARRY_HI", 344, 16),
        ("ALU_LO", 360, 16), ("ALU_HI", 376, 16),
        ("CARRY", 392, 4), ("CMP", 396, 4),
        ("MUL_ACCUM", 416, 16), ("DIV_STAGING", 432, 16),
        ("TEMP", 480, 32),
        ("L1H4", 297, 7), ("L2H0", 448, 7),
    ]
    # Also the L0/L1 head ranges
    for h in range(8):
        base = 60 + h * 7
        RANGES.append((f"H{h}", base, 7))
    for h in range(3):
        base = 116 + h * 7
        RANGES.append((f"L1H{h}", base, 7))

    lookup = {}
    for rname, rstart, rsize in RANGES:
        for offset in range(rsize):
            d = rstart + offset
            if offset == 0:
                lookup[d] = rname
            else:
                lookup[d] = f"{rname}+{offset}"

    # Add single-dim constants
    for name, val in by_val:
        if val not in lookup:
            lookup[val] = name

    return lookup


def _dim_name(d: int, dim_to_slot: Dict[int, str], setdim_lookup: Dict[int, str]) -> str:
    """Get a human-readable name for dimension d.

    Prefers registry slot name, falls back to _SetDim name, then "unregistered[d]".
    """
    if d in dim_to_slot:
        return dim_to_slot[d]
    if d in setdim_lookup:
        return f"_{setdim_lookup[d]}"  # prefix with _ to indicate non-registry
    return f"unregistered[{d}]"


def _dims_to_slot_names(dims: Set[int], dim_to_slot: Dict[int, str]) -> Tuple[Set[str], Set[int]]:
    """Map dim indices to registry slot names.

    Returns (slot_names, unregistered_dims).
    """
    slot_names = set()
    unregistered = set()
    for d in dims:
        if d in dim_to_slot:
            slot_names.add(dim_to_slot[d])
        else:
            unregistered.add(d)
    return slot_names, unregistered


@dataclass
class LayerWeightInfo:
    """Auto-derived weight info for one layer component (FFN or attn head)."""
    layer: str              # e.g. "L0_ffn", "L0_attn", "L0_attn_h3"
    read_dims: Set[int]     # raw dim indices read
    write_dims: Set[int]    # raw dim indices written
    bias_dims: Set[int] = field(default_factory=set)  # dims activated by bias alone
    active_hidden: int = 0  # number of active hidden units (FFN only)
    total_hidden: int = 0   # total hidden units (FFN only)


def extract_weight_info(model) -> List[LayerWeightInfo]:
    """Extract per-layer read/write dim sets from actual weight matrices.

    Inspects W_up, W_gate, W_down for FFNs and W_q, W_k, W_v, W_o for
    attention layers. Works with both dense and sparse weights.

    Returns a list of LayerWeightInfo, two per block (attn + ffn), plus
    per-head detail for attention.
    """
    import torch
    infos = []

    for i, block in enumerate(model.blocks):
        # --- Attention ---
        attn = block.attn
        attn_read = set()
        attn_write = set()
        for w_name in ('W_q', 'W_k', 'W_v'):
            W = getattr(attn, w_name)
            attn_read |= _nonzero_dims(W, axis=0)
        attn_write = _nonzero_dims(attn.W_o, axis=1)

        infos.append(LayerWeightInfo(
            layer=f"L{i}_attn",
            read_dims=attn_read,
            write_dims=attn_write,
        ))

        # Per-head detail
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        for h in range(num_heads):
            h_start = h * head_dim
            h_end = (h + 1) * head_dim
            h_read = set()
            h_write = set()
            for w_name in ('W_q', 'W_k', 'W_v'):
                W = getattr(attn, w_name)
                if W.is_sparse:
                    idx = W.coalesce().indices()
                    if idx.numel() > 0:
                        mask = (idx[0] >= h_start) & (idx[0] < h_end)
                        h_read |= set(idx[1][mask].tolist())
                else:
                    h_read |= _nonzero_dims(W[h_start:h_end], axis=0)
            # W_o: columns h_start:h_end → which output dims
            W_o = attn.W_o
            if W_o.is_sparse:
                idx = W_o.coalesce().indices()
                if idx.numel() > 0:
                    mask = (idx[1] >= h_start) & (idx[1] < h_end)
                    h_write |= set(idx[0][mask].tolist())
            else:
                h_write = _nonzero_dims(W_o[:, h_start:h_end], axis=1)

            if h_read or h_write:
                infos.append(LayerWeightInfo(
                    layer=f"L{i}_attn_h{h}",
                    read_dims=h_read,
                    write_dims=h_write,
                ))

        # --- FFN ---
        ffn = block.ffn
        ffn_read = _nonzero_dims(ffn.W_up, axis=0) | _nonzero_dims(ffn.W_gate, axis=0)
        ffn_write = _nonzero_dims(ffn.W_down, axis=1)
        # Bias: find hidden units with non-zero b_up or b_gate (constant activations)
        bias_hidden = _nonzero_bias_dims(ffn.b_up) | _nonzero_bias_dims(ffn.b_gate)
        # Map bias-activated hidden units to which output dims they can affect
        # via W_down[d, h] — if W_down[d, h] != 0 and h has non-zero bias
        bias_output_dims = set()
        if bias_hidden:
            W_down = ffn.W_down
            if W_down.is_sparse:
                idx = W_down.coalesce().indices()
                vals = W_down.coalesce().values()
                if idx.numel() > 0:
                    for h in bias_hidden:
                        mask = (idx[1] == h) & (vals != 0)
                        bias_output_dims |= set(idx[0][mask].tolist())
            else:
                for h in bias_hidden:
                    if h < W_down.shape[1]:
                        col = W_down[:, h]
                        bias_output_dims |= set(torch.nonzero(col.abs()).squeeze(-1).tolist())

        # Active hidden units: rows of W_up or W_gate with any non-zero
        active_up = _nonzero_dims(ffn.W_up, axis=1)
        active_gate = _nonzero_dims(ffn.W_gate, axis=1)
        active_hidden = len(active_up | active_gate)
        total_hidden = ffn.hidden_dim

        infos.append(LayerWeightInfo(
            layer=f"L{i}_ffn",
            read_dims=ffn_read,
            write_dims=ffn_write,
            bias_dims=bias_output_dims,  # d_model dims affected by bias
            active_hidden=active_hidden,
            total_hidden=total_hidden,
        ))

    return infos


def extract_model_contracts(model, registry: DimRegistry) -> List[LayerIO]:
    """Auto-derive per-layer read/write contracts from actual weight matrices.

    Maps non-zero dimension indices back to DimRegistry slot names.
    Returns one LayerIO per block component (attn + ffn per layer).
    """
    dim_to_slot = _build_dim_to_slot(registry)
    infos = extract_weight_info(model)

    contracts = []
    for info in infos:
        # Skip per-head detail for contract comparison
        if '_h' in info.layer:
            continue
        read_slots, unregistered_reads = _dims_to_slot_names(info.read_dims, dim_to_slot)
        write_slots, unregistered_writes = _dims_to_slot_names(info.write_dims, dim_to_slot)

        notes_parts = []
        if unregistered_reads:
            notes_parts.append(f"unregistered reads: {sorted(unregistered_reads)}")
        if unregistered_writes:
            notes_parts.append(f"unregistered writes: {sorted(unregistered_writes)}")
        if info.active_hidden:
            notes_parts.append(f"{info.active_hidden}/{info.total_hidden} active hidden units")

        contracts.append(LayerIO(
            layer=info.layer,
            reads=sorted(read_slots),
            writes=sorted(write_slots),
            notes="; ".join(notes_parts),
        ))

    return contracts


def compare_contracts(
    declared: List[LayerIO],
    derived: List[LayerIO],
    registry: DimRegistry,
) -> List[str]:
    """Compare manually-declared contracts against auto-derived ones.

    Reports:
    - Undeclared reads: slots read by weights but not in declared contract
    - Undeclared writes: slots written by weights but not in declared contract
    - Phantom reads: declared but not actually read (stale contract)
    - Phantom writes: declared but not actually written
    """
    declared_map = {lio.layer: lio for lio in declared}
    derived_map = {lio.layer: lio for lio in derived}

    issues = []

    # Check derived layers not covered by any declaration
    for layer_name, derived_lio in sorted(derived_map.items()):
        if layer_name not in declared_map:
            d_reads = set(derived_lio.reads)
            d_writes = set(derived_lio.writes)
            if d_reads or d_writes:
                issues.append(
                    f"UNDECLARED LAYER: {layer_name} "
                    f"reads={sorted(d_reads)}, writes={sorted(d_writes)}"
                )
            continue

        decl = declared_map[layer_name]
        decl_reads = set(registry.resolve_names(decl.reads))
        decl_writes = set(registry.resolve_names(decl.writes))
        decl_additive = set(registry.resolve_names(decl.additive_writes))
        decl_all_writes = decl_writes | decl_additive

        derived_reads = set(derived_lio.reads)
        derived_writes = set(derived_lio.writes)

        undeclared_reads = derived_reads - decl_reads
        undeclared_writes = derived_writes - decl_all_writes
        phantom_reads = decl_reads - derived_reads
        phantom_writes = decl_all_writes - derived_writes

        for slot in sorted(undeclared_reads):
            issues.append(f"UNDECLARED READ: {layer_name} reads '{slot}' (not in contract)")
        for slot in sorted(undeclared_writes):
            issues.append(f"UNDECLARED WRITE: {layer_name} writes '{slot}' (not in contract)")
        for slot in sorted(phantom_reads):
            issues.append(f"PHANTOM READ: {layer_name} declares read '{slot}' but weights don't touch it")
        for slot in sorted(phantom_writes):
            issues.append(f"PHANTOM WRITE: {layer_name} declares write '{slot}' but weights don't touch it")

    return issues


def ffn_summary(model, registry: DimRegistry, per_head: bool = True) -> str:
    """Human-readable per-layer FFN/attn input/output report.

    For each layer, prints:
      L{i} attn: (reads/writes with slot names)
        Per head detail if per_head=True
      L{i} FFN: {active}/{total} active hidden units
        reads:  SLOT_A, SLOT_B, ...
        writes: SLOT_C, SLOT_D, ...
    """
    dim_to_slot = _build_dim_to_slot(registry)
    setdim_lookup = _build_setdim_lookup()
    infos = extract_weight_info(model)

    lines = ["Weight Inspection Summary", "=" * 70]

    def _format_dims(dims: Set[int]) -> str:
        """Format dim set as slot-grouped names.

        For fully-used registry slots, shows "SLOT_NAME".
        For partially-used slots, shows individual _SetDim names when available,
        e.g. "MEM_STORE, MEM_ADDR_SRC" instead of "IMM_STAGING[2/16]".
        """
        if not dims:
            return "(none)"
        # Group by registry slot
        by_slot: Dict[str, List[int]] = {}
        unregistered: List[int] = []
        for d in sorted(dims):
            if d in dim_to_slot:
                by_slot.setdefault(dim_to_slot[d], []).append(d)
            else:
                unregistered.append(d)

        parts = []
        for base in sorted(by_slot.keys(), key=lambda b: min(by_slot[b])):
            dim_list = by_slot[base]
            slot = registry.slots.get(base)
            if slot and len(dim_list) == slot.size:
                # Full slot used
                parts.append(base)
            elif slot and len(dim_list) < slot.size:
                # Partial slot — show _SetDim names if they exist and are
                # more specific than the slot name
                setdim_names = []
                for d in dim_list:
                    sd = setdim_lookup.get(d)
                    if sd and sd != base:
                        setdim_names.append(sd)
                    else:
                        setdim_names.append(f"{base}[{d - slot.start}]")
                # If all have _SetDim names, show them individually (more useful)
                if len(setdim_names) <= 6:
                    parts.extend(setdim_names)
                else:
                    parts.append(f"{base}[{len(dim_list)}/{slot.size}]")
            else:
                parts.append(base)

        # Unregistered dims — show _SetDim names
        for d in unregistered:
            sd = setdim_lookup.get(d)
            if sd:
                parts.append(f"_{sd}")
            else:
                parts.append(f"dim[{d}]")

        return ", ".join(parts)

    for info in infos:
        is_head = '_h' in info.layer
        indent = "    " if is_head else "  "

        if is_head:
            if not info.read_dims and not info.write_dims:
                continue
            lines.append(f"{indent}{info.layer}:")
            lines.append(f"{indent}  reads:  {_format_dims(info.read_dims)}")
            lines.append(f"{indent}  writes: {_format_dims(info.write_dims)}")
        elif info.layer.endswith("_attn"):
            lines.append("")
            lines.append(f"{indent}{info.layer}:")
            lines.append(f"{indent}  reads:  {_format_dims(info.read_dims)}")
            lines.append(f"{indent}  writes: {_format_dims(info.write_dims)}")
        else:
            # FFN
            hidden_str = ""
            if info.total_hidden:
                hidden_str = f" ({info.active_hidden}/{info.total_hidden} active hidden units)"
            lines.append(f"{indent}{info.layer}:{hidden_str}")
            lines.append(f"{indent}  reads:  {_format_dims(info.read_dims)}")
            lines.append(f"{indent}  writes: {_format_dims(info.write_dims)}")
            if info.bias_dims:
                lines.append(f"{indent}  bias→out:  {_format_dims(info.bias_dims)}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    reg = build_default_registry()
    contracts = build_default_contracts(reg)
    validator = ContractValidator(reg, contracts)

    print(reg.report())
    print()

    errors = validator.validate()
    if errors:
        print("Validation Issues:")
        for e in errors:
            print(f"  {e}")
    else:
        print("All contracts valid.")
    print()

    print(validator.dep_graph())
    print()

    # --- Weight inspection ---
    print("=" * 70)
    print("WEIGHT INSPECTION (auto-derived from actual weight matrices)")
    print("=" * 70)
    print()

    import torch
    from .unified_compiler.full_vm_compiler import compile_full_vm

    print("Building model and setting weights...")
    model, _ = compile_full_vm()
    print()

    # Per-layer summary
    print(ffn_summary(model, reg, per_head=True))
    print()

    # Compare against declared contracts
    derived = extract_model_contracts(model, reg)
    issues = compare_contracts(contracts, derived, reg)
    if issues:
        print("Contract Comparison (declared vs auto-derived):")
        print("-" * 50)
        for issue in issues:
            print(f"  {issue}")
    else:
        print("All declared contracts match auto-derived contracts.")
    print()
