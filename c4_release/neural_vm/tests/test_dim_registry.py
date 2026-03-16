"""Tests for DimRegistry & ContractValidator."""

import unittest
from neural_vm.dim_registry import (
    DimRegistry, DimSlot, LayerIO, ContractValidator,
    build_default_registry, build_default_contracts, validate_default,
)


class TestDimSlot(unittest.TestCase):
    def test_overlap_detection(self):
        a = DimSlot("A", 0, 10, "")
        b = DimSlot("B", 5, 10, "")
        c = DimSlot("C", 10, 5, "")
        self.assertTrue(a.overlaps(b))
        self.assertTrue(b.overlaps(a))
        self.assertFalse(a.overlaps(c))  # adjacent, not overlapping
        self.assertFalse(c.overlaps(a))

    def test_range(self):
        s = DimSlot("X", 10, 5, "")
        self.assertEqual(list(s.range), [10, 11, 12, 13, 14])
        self.assertEqual(s.end, 15)


class TestDimRegistry(unittest.TestCase):
    def test_alloc_basic(self):
        reg = DimRegistry(d_model=64)
        slot = reg.alloc("FOO", 0, 10, "test slot")
        self.assertEqual(slot.name, "FOO")
        self.assertEqual(slot.start, 0)
        self.assertEqual(slot.size, 10)
        self.assertIn("FOO", reg.slots)

    def test_alloc_duplicate_name(self):
        reg = DimRegistry(d_model=64)
        reg.alloc("FOO", 0, 10, "first")
        with self.assertRaises(ValueError):
            reg.alloc("FOO", 20, 5, "duplicate")

    def test_alloc_out_of_bounds(self):
        reg = DimRegistry(d_model=64)
        with self.assertRaises(ValueError):
            reg.alloc("BAD", 60, 10, "exceeds d_model")

    def test_check_overlaps_none(self):
        reg = DimRegistry(d_model=64)
        reg.alloc("A", 0, 10, "")
        reg.alloc("B", 10, 10, "")
        reg.alloc("C", 30, 5, "")
        self.assertEqual(reg.check_overlaps(), [])

    def test_check_overlaps_detected(self):
        reg = DimRegistry(d_model=64)
        reg.alloc("A", 0, 10, "")
        reg.alloc("B", 5, 10, "")
        errors = reg.check_overlaps()
        self.assertEqual(len(errors), 1)
        self.assertIn("OVERLAP", errors[0])
        self.assertIn("A", errors[0])
        self.assertIn("B", errors[0])

    def test_free_ranges(self):
        reg = DimRegistry(d_model=32)
        reg.alloc("A", 0, 5, "")
        reg.alloc("B", 10, 5, "")
        free = reg.free_ranges()
        self.assertEqual(free, [(5, 10), (15, 32)])

    def test_free_ranges_full(self):
        reg = DimRegistry(d_model=8)
        reg.alloc("ALL", 0, 8, "")
        self.assertEqual(reg.free_ranges(), [])

    def test_free_ranges_empty(self):
        reg = DimRegistry(d_model=8)
        self.assertEqual(reg.free_ranges(), [(0, 8)])

    def test_report(self):
        reg = DimRegistry(d_model=32)
        reg.alloc("FOO", 0, 1, "single dim")
        reg.alloc("BAR", 10, 5, "multi dim")
        report = reg.report()
        self.assertIn("FOO", report)
        self.assertIn("BAR", report)
        self.assertIn("d_model=32", report)
        self.assertIn("Free", report)

    def test_resolve_names_wildcard(self):
        reg = DimRegistry(d_model=64)
        reg.alloc("MARK_PC", 0, 1, "")
        reg.alloc("MARK_AX", 1, 1, "")
        reg.alloc("OTHER", 10, 1, "")
        resolved = reg.resolve_names(["MARK_*", "OTHER"])
        self.assertIn("MARK_PC", resolved)
        self.assertIn("MARK_AX", resolved)
        self.assertIn("OTHER", resolved)
        self.assertEqual(len(resolved), 3)

    def test_resolve_names_no_match(self):
        reg = DimRegistry(d_model=64)
        reg.alloc("FOO", 0, 1, "")
        resolved = reg.resolve_names(["NONEXIST_*"])
        # unresolved wildcard kept as-is
        self.assertEqual(resolved, ["NONEXIST_*"])


class TestContractValidator(unittest.TestCase):
    def _make_simple(self):
        reg = DimRegistry(d_model=32)
        reg.alloc("A", 0, 5, "")
        reg.alloc("B", 10, 5, "")
        reg.alloc("C", 20, 5, "")
        return reg

    def test_valid_contracts(self):
        reg = self._make_simple()
        contracts = [
            LayerIO("embed", reads=[], writes=["A", "B"]),
            LayerIO("L0", reads=["A"], writes=["C"]),
            LayerIO("head", reads=["B", "C"], writes=[]),
        ]
        errors = ContractValidator(reg, contracts).validate()
        self.assertEqual(errors, [])

    def test_read_before_write(self):
        reg = self._make_simple()
        contracts = [
            LayerIO("embed", reads=[], writes=["A"]),
            LayerIO("L0", reads=["B"], writes=[]),  # B never written
        ]
        errors = ContractValidator(reg, contracts).validate()
        rbw = [e for e in errors if "READ-BEFORE-WRITE" in e]
        self.assertEqual(len(rbw), 1)
        self.assertIn("B", rbw[0])

    def test_double_write_warning(self):
        reg = self._make_simple()
        contracts = [
            LayerIO("embed", reads=[], writes=["A"]),
            LayerIO("L0", reads=["A"], writes=["B"]),
            LayerIO("L1", reads=["A"], writes=["B"]),  # double-write
        ]
        errors = ContractValidator(reg, contracts).validate()
        dw = [e for e in errors if "DOUBLE-WRITE" in e]
        self.assertEqual(len(dw), 1)
        self.assertIn("B", dw[0])

    def test_additive_write_no_warning(self):
        reg = self._make_simple()
        contracts = [
            LayerIO("embed", reads=[], writes=["A"]),
            LayerIO("L0", reads=["A"], writes=[], additive_writes=["B"]),
            LayerIO("L1", reads=["A"], writes=[], additive_writes=["B"]),
        ]
        errors = ContractValidator(reg, contracts).validate()
        dw = [e for e in errors if "DOUBLE-WRITE" in e]
        self.assertEqual(len(dw), 0)

    def test_unknown_slot(self):
        reg = self._make_simple()
        contracts = [
            LayerIO("L0", reads=["NONEXIST"], writes=[]),
        ]
        errors = ContractValidator(reg, contracts).validate()
        unknown = [e for e in errors if "UNKNOWN SLOT" in e]
        self.assertEqual(len(unknown), 1)

    def test_overlap_propagated(self):
        reg = DimRegistry(d_model=32)
        reg.alloc("X", 0, 10, "")
        reg.alloc("Y", 5, 10, "")
        contracts = [LayerIO("embed", reads=[], writes=["X", "Y"])]
        errors = ContractValidator(reg, contracts).validate()
        overlap = [e for e in errors if "OVERLAP" in e]
        self.assertEqual(len(overlap), 1)

    def test_dep_graph(self):
        reg = self._make_simple()
        contracts = [
            LayerIO("embed", reads=[], writes=["A"]),
            LayerIO("L0", reads=["A"], writes=["B"]),
        ]
        graph = ContractValidator(reg, contracts).dep_graph()
        self.assertIn("Dependency Graph", graph)
        self.assertIn("embed", graph)
        self.assertIn("L0", graph)


class TestDefaultRegistry(unittest.TestCase):
    def test_no_overlaps(self):
        reg = build_default_registry()
        errors = reg.check_overlaps()
        self.assertEqual(errors, [], f"Unexpected overlaps: {errors}")

    def test_all_setdim_slots_present(self):
        """Verify registry covers key _SetDim allocations."""
        reg = build_default_registry()
        expected_names = [
            "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
            "MARK_SE", "IS_BYTE", "IS_MARK", "CONST", "MARK_CS",
            "MARK_SE_ONLY", "MARK_STACK0",
            "ADDR_B0_LO", "ADDR_B1_LO", "ADDR_B2_LO",
            "H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7",
            "L1H0", "L1H1", "L1H2", "HAS_SE",
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
            "EMBED_LO", "EMBED_HI", "OUTPUT_LO", "OUTPUT_HI",
            "ADDR_KEY",
            "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
            "NEXT_STACK0", "NEXT_MEM", "NEXT_SE", "NEXT_HALT",
            "OPCODE_FLAGS",
            "STACK0_LO", "STACK0_HI", "AX_CARRY_LO", "AX_CARRY_HI",
            "ALU_LO", "ALU_HI", "CARRY", "CMP",
            "PC_BIT", "MUL_ACCUM", "DIV_STAGING", "IMM_STAGING",
            "CS_DIST_THERMO", "TEMP",
        ]
        for name in expected_names:
            self.assertIn(name, reg.slots, f"Missing slot: {name}")

    def test_setdim_start_values_match(self):
        """Verify registry start values match _SetDim constants."""
        from neural_vm.vm_step import _SetDim as BD
        reg = build_default_registry()

        checks = {
            "MARK_PC": BD.MARK_PC, "MARK_AX": BD.MARK_AX,
            "MARK_SP": BD.MARK_SP, "MARK_BP": BD.MARK_BP,
            "MARK_MEM": BD.MARK_MEM, "MARK_SE": BD.MARK_SE,
            "IS_BYTE": BD.IS_BYTE, "IS_MARK": BD.IS_MARK,
            "CONST": BD.CONST, "MARK_CS": BD.MARK_CS,
            "MARK_SE_ONLY": BD.MARK_SE_ONLY, "MARK_STACK0": BD.MARK_STACK0,
            "H0": BD.H0, "H1": BD.H1, "H2": BD.H2, "H3": BD.H3,
            "H4": BD.H4, "H5": BD.H5, "H6": BD.H6, "H7": BD.H7,
            "L1H0": BD.L1H0, "L1H1": BD.L1H1, "L1H2": BD.L1H2,
            "HAS_SE": BD.HAS_SE,
            "NEXT_PC": BD.NEXT_PC, "NEXT_AX": BD.NEXT_AX,
            "NEXT_SP": BD.NEXT_SP, "NEXT_BP": BD.NEXT_BP,
            "NEXT_STACK0": BD.NEXT_STACK0,
            "NEXT_MEM": BD.NEXT_MEM, "NEXT_SE": BD.NEXT_SE,
            "NEXT_HALT": BD.NEXT_HALT,
            "EMBED_LO": BD.EMBED_LO, "EMBED_HI": BD.EMBED_HI,
            "OUTPUT_LO": BD.OUTPUT_LO, "OUTPUT_HI": BD.OUTPUT_HI,
            "ADDR_B0_LO": BD.ADDR_B0_LO, "ADDR_B1_LO": BD.ADDR_B1_LO,
            "ADDR_B2_LO": BD.ADDR_B2_LO,
            "ADDR_KEY": BD.ADDR_KEY,
        }
        for name, expected_start in checks.items():
            self.assertEqual(
                reg.slots[name].start, expected_start,
                f"{name}: registry start={reg.slots[name].start} != _SetDim={expected_start}"
            )

    def test_d_model_512(self):
        """Registry should use d_model=512."""
        reg = build_default_registry()
        self.assertEqual(reg.d_model, 512)

    def test_report_runs(self):
        reg = build_default_registry()
        report = reg.report()
        self.assertIn("d_model=512", report)
        self.assertIn("ADDR_KEY", report)
        self.assertIn("OPCODE_FLAGS", report)


class TestDefaultContracts(unittest.TestCase):
    def test_validate_default(self):
        """Default contracts should produce only expected warnings, no errors."""
        reg, errors = validate_default()
        # Separate real errors from expected double-write warnings
        real_errors = [e for e in errors
                       if "DOUBLE-WRITE" not in e
                       and "READ-BEFORE-WRITE" not in e]
        self.assertEqual(real_errors, [], f"Unexpected errors: {real_errors}")

    def test_output_double_write_is_additive(self):
        """OUTPUT_LO/HI should NOT produce double-write warnings (marked additive)."""
        reg = build_default_registry()
        contracts = build_default_contracts(reg)
        errors = ContractValidator(reg, contracts).validate()
        output_dw = [e for e in errors if "DOUBLE-WRITE" in e and "OUTPUT" in e]
        self.assertEqual(output_dw, [],
                         f"OUTPUT_LO/HI should be additive, not flagged: {output_dw}")

    def test_dep_graph_runs(self):
        reg = build_default_registry()
        contracts = build_default_contracts(reg)
        graph = ContractValidator(reg, contracts).dep_graph()
        self.assertIn("Execution Order", graph)
        self.assertIn("L15_attn", graph)


class TestIntegration(unittest.TestCase):
    def test_set_vm_weights_runs_validation(self):
        """set_vm_weights should complete without raising."""
        import torch
        from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
        model = AutoregressiveVM()
        set_vm_weights(model)  # should not raise


if __name__ == "__main__":
    unittest.main()
