"""Phase 6 PRTF/READ syscall status diagnostics."""

from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner


def _runner_without_model():
    runner = AutoregressiveVMRunner.__new__(AutoregressiveVMRunner)
    runner._phase6_syscall_events = []
    return runner


def test_phase6_syscall_status_documents_prtf_and_read_gaps():
    runner = _runner_without_model()

    status = runner.get_phase6_syscall_status()

    assert set(status) == {"PRTF", "READ"}
    for name in ("PRTF", "READ"):
        assert status[name]["status"] == "external-shim"
        assert status[name]["implemented"] is True
        assert status[name]["neural_complete"] is False
        assert status[name]["diagnostic"]


def test_phase6_syscall_diagnostics_include_prtf_read_events():
    runner = _runner_without_model()

    runner._record_phase6_syscall_event(
        Opcode.PRTF, "pure_neural_skipped_op", exec_idx=4, argc=1, bytes=2
    )
    runner._record_phase6_syscall_event(
        Opcode.READ, "pure_neural_direct", fd=0, buf_ptr=0x10000,
        count=1, bytes=1,
    )

    diagnostics = runner.get_phase6_syscall_diagnostics()

    assert [event["name"] for event in diagnostics["events"]] == ["PRTF", "READ"]
    assert diagnostics["events"][0]["path"] == "pure_neural_skipped_op"
    assert diagnostics["events"][1]["buf_ptr"] == 0x10000
    assert {item["name"] for item in diagnostics["unresolved"]} == {"PRTF", "READ"}
