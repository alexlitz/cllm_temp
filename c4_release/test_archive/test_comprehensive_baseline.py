"""Comprehensive baseline test - check multiple opcodes."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    """Build minimal context."""
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

# Test programs
tests = [
    ("IMM 5", [Opcode.IMM | (5 << 8), Opcode.EXIT]),
    ("JMP 12", [Opcode.JMP | (12 << 8), Opcode.EXIT]),
    ("PSH", [Opcode.IMM | (42 << 8), Opcode.PSH, Opcode.EXIT]),
    ("ADD", [Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.ADD, Opcode.EXIT]),
]

# Load model once
print("Loading model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()
print("Model loaded\n")

print("Comprehensive Baseline Test")
print("="*70)

for name, bytecode in tests:
    print(f"\nTest: {name}")
    print("-"*70)

    # Get DraftVM ground truth
    draft_vm = DraftVM(bytecode)
    context = build_context(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    # Generate with neural model
    context_with_draft = context + draft_tokens
    ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    # Compare predictions
    ctx_len = len(context)
    mismatches = []

    for i in range(min(35, len(draft_tokens))):
        pos = ctx_len + i
        if pos >= logits.shape[1]:
            break

        expected = draft_tokens[i]
        predicted = logits[0, pos - 1, :].argmax(-1).item()

        if expected != predicted:
            mismatches.append((i, expected, predicted))

    # Report
    match_rate = (35 - len(mismatches)) / 35 * 100
    print(f"  Match rate: {match_rate:.1f}% ({35 - len(mismatches)}/35 tokens)")

    if mismatches:
        print(f"  Mismatches:")
        for pos, exp, pred in mismatches[:5]:  # Show first 5
            marker_names = {257: "REG_PC", 258: "REG_AX", 259: "REG_SP",
                          260: "REG_BP", 261: "STACK0", 262: "MEM", 263: "STEP_END"}
            pos_name = f"pos {pos}"
            if pos == 0:
                pos_name = "marker"
            elif pos <= 4:
                pos_name = f"byte {pos-1}"

            exp_str = marker_names.get(exp, str(exp))
            pred_str = marker_names.get(pred, str(pred))
            print(f"    {pos_name:12s}: expected {exp_str:10s}, got {pred_str:10s}")

        if len(mismatches) > 5:
            print(f"    ... and {len(mismatches) - 5} more")

print("\n" + "="*70)
print("Summary: Baseline has prediction issues that need fixing")
