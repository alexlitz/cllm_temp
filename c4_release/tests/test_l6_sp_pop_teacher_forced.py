"""Focused teacher-forced probes for L6 stack-pop SP high-byte handling."""

from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
from neural_vm.verification.decl_verifier import (
    build_teacher_forced_symbolic_trace,
    verify_teacher_forced_token_support,
)
from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs


_SP_POP_CLUSTER_IDS = (0, 9, 350, 825)


def test_l6_binary_pop_sp_high_bytes_survive_final_tail():
    runner = BatchedPureNeuralRunner(max_seq_len=4096)
    programs = generate_test_programs()

    for test_idx in _SP_POP_CLUSTER_IDS:
        source, _, _ = programs[test_idx]
        bytecode, data = compile_c(source)
        trace = build_teacher_forced_symbolic_trace(bytecode, data)

        assert trace.token_for(3, "SP_byte1") == 0x00
        assert trace.token_for(3, "SP_byte2") == 0x01

        for slot in ("SP_byte1", "SP_byte2"):
            token_index = trace.token_index(3, slot)
            report = verify_teacher_forced_token_support(
                runner.model,
                trace.context,
                token_index=token_index,
                prefix_len=trace.prefix_len,
                mem_store_positions=trace.mem_store_positions,
                probe_name=f"id={test_idx:04d}:step3:{slot}",
            )
            after_l6 = next(
                snapshot
                for snapshot in report.snapshots
                if snapshot.original_layer_index == 6
            )

            assert after_l6.supports_expected_argmax, report.format()
            assert after_l6.supports_expected_byte_channel, report.format()
            assert after_l6.expected_margin >= 0.0, report.format()
            assert report.supported, report.format()
