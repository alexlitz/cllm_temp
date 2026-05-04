"""
ONNX export for the 8-bit Neural VM.

Exports the AutoregressiveVM's single-step forward() to ONNX format,
and provides an ONNX-based runner for verification.

Usage:
    from vm_8bit.onnx_export import export_onnx, ONNXRunner

    # Export
    path = export_onnx("vm8bit.onnx")

    # Run
    runner = ONNXRunner("vm8bit.onnx")
    runner.load(bytecode)
    output, exit_code = runner.run()
"""

import tempfile
import numpy as np
import torch

from .autoregressive_vm import AutoregressiveVM, FE
from .config import STACK_INIT


def export_onnx(path=None, opset_version=18):
    model = AutoregressiveVM()
    model.eval()
    dummy = torch.zeros(1, FE.DIM)
    dummy[0, FE.CONST] = 1.0
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".onnx")
    torch.onnx.export(
        model, dummy, path,
        input_names=["state"],
        output_names=["state_out"],
        opset_version=opset_version,
        do_constant_folding=True,
    )
    return path


class ONNXRunner:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path)
        self._input_buf = []

    def load(self, bytecode):
        self.bytecode = bytecode

    def set_input(self, chars):
        self._input_buf = list(chars)

    def run(self, max_steps=1000):
        state = np.zeros((1, FE.DIM), dtype=np.float32)
        for i, b in enumerate(self.bytecode):
            state[0, FE.MEM + i] = float(b & 0xFF)
        state[0, FE.SP] = float(STACK_INIT)
        state[0, FE.BP] = float(STACK_INIT)
        state[0, FE.CONST] = 1.0
        output = []

        snap_integer = list(range(FE.PC)) + [FE.PC, FE.AX, FE.SP, FE.BP,
            FE.OPCODE, FE.IMM, FE.STACK_TOP, FE.ALU_RESULT,
            FE.NEXT_PC, FE.SP_DEC, FE.SP_INC, FE.OUTPUT_CHAR,
            FE.MEM_WR_ADDR, FE.MEM_WR_VAL, FE.LI_VAL,
            FE.LEV_NEW_BP, FE.LEV_NEW_PC, FE.INPUT_VAL]
        snap_binary = [
            FE.OP_IMM, FE.OP_PSH, FE.OP_ADD, FE.OP_SUB, FE.OP_MUL,
            FE.OP_AND, FE.OP_OR, FE.OP_XOR, FE.OP_DIV, FE.OP_MOD,
            FE.OP_SHL, FE.OP_SHR, FE.OP_PUTCHAR, FE.OP_EXIT,
            FE.OP_JMP, FE.OP_BZ, FE.OP_BNZ, FE.OP_JSR, FE.OP_ENT,
            FE.OP_LEV, FE.OP_ADJ, FE.OP_LI, FE.OP_SI, FE.OP_GETCHAR,
            FE.OP_EQ, FE.OP_NE, FE.OP_LT, FE.OP_GT, FE.OP_LE, FE.OP_GE,
            FE.AX_IS_ZERO, FE.AX_IS_ZERO_NEG, FE.IS_ALU_OP, FE.IS_CMP_OP,
            FE.BZ_TAKEN, FE.BNZ_TAKEN, FE.OP_DEFAULT_AX, FE.OP_DEFAULT_SP,
            FE.OP_DEFAULT_PC, FE.PC_GO_IMM, FE.PC_GO_NEXT,
            FE.HALT, FE.ALU_IS_ZERO, FE.CMP_LT]

        for _ in range(max_steps):
            state[0, FE.INPUT_VAL] = float(ord(self._input_buf[0]) & 0xFF) if self._input_buf else 255.0

            state = self.session.run(None, {"state": state})[0]

            for slot in snap_integer:
                v = state[0, slot]
                state[0, slot] = np.round(v)
            for slot in snap_binary:
                state[0, slot] = np.round(state[0, slot]).clip(0, 1)

            if state[0, FE.OP_PUTCHAR].item() > 0.5:
                output.append(chr(int(round(state[0, FE.OUTPUT_CHAR].item())) & 0xFF))

            if state[0, FE.OP_GETCHAR].item() > 0.5 and self._input_buf:
                self._input_buf.pop(0)

            if state[0, FE.HALT].item() > 0.5:
                break

        return "".join(output), 0
