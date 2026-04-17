"""Fast test of all opcodes - single step only."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
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

# Initialize model once
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

def test_opcode_fast(name, bytecode):
    """Quick single-step test."""
    draft_vm = DraftVM(bytecode)
    context = build_context(bytecode)

    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)

    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    ctx_len = len(context)
    matches = sum(
        1 for i in range(35)
        if logits[0, ctx_len - 1 + i, :].argmax().item() == draft_tokens[i]
    )

    match_rate = matches / 35 * 100
    status = "✓" if match_rate == 100.0 else ("⚠" if match_rate >= 97 else "✗")
    print(f"{status} {name:25} {match_rate:5.1f}% ({matches}/35)")
    return match_rate

print("=" * 70)
print("Fast Opcode Test - All Opcodes")
print("=" * 70)

results = {}

print("\n--- Load/Store/Stack ---")
results['IMM'] = test_opcode_fast("IMM", [Opcode.IMM | (42 << 8)])
results['LEA'] = test_opcode_fast("LEA", [Opcode.LEA | (8 << 8)])
results['PSH'] = test_opcode_fast("PSH", [Opcode.IMM | (10 << 8), Opcode.PSH])
results['NOP'] = test_opcode_fast("NOP", [Opcode.NOP])
results['EXIT'] = test_opcode_fast("EXIT", [Opcode.EXIT])

print("\n--- Jumps/Branches ---")
results['JMP'] = test_opcode_fast("JMP", [Opcode.JMP | (12 << 8)])
results['BZ (taken)'] = test_opcode_fast("BZ (taken)", [Opcode.IMM | (0 << 8), Opcode.BZ | (12 << 8)])
results['BZ (not taken)'] = test_opcode_fast("BZ (not taken)", [Opcode.IMM | (5 << 8), Opcode.BZ | (12 << 8)])
results['BNZ (taken)'] = test_opcode_fast("BNZ (taken)", [Opcode.IMM | (5 << 8), Opcode.BNZ | (12 << 8)])
results['BNZ (not taken)'] = test_opcode_fast("BNZ (not taken)", [Opcode.IMM | (0 << 8), Opcode.BNZ | (12 << 8)])

print("\n--- Arithmetic ---")
results['ADD'] = test_opcode_fast("ADD", [Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.ADD])
results['SUB'] = test_opcode_fast("SUB", [Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.SUB])
results['MUL'] = test_opcode_fast("MUL", [Opcode.IMM | (6 << 8), Opcode.PSH, Opcode.IMM | (7 << 8), Opcode.MUL])
results['DIV'] = test_opcode_fast("DIV", [Opcode.IMM | (20 << 8), Opcode.PSH, Opcode.IMM | (4 << 8), Opcode.DIV])
results['MOD'] = test_opcode_fast("MOD", [Opcode.IMM | (17 << 8), Opcode.PSH, Opcode.IMM | (5 << 8), Opcode.MOD])

print("\n--- Bitwise ---")
results['OR'] = test_opcode_fast("OR", [Opcode.IMM | (0x0F << 8), Opcode.PSH, Opcode.IMM | (0xF0 << 8), Opcode.OR])
results['XOR'] = test_opcode_fast("XOR", [Opcode.IMM | (0xAA << 8), Opcode.PSH, Opcode.IMM | (0x55 << 8), Opcode.XOR])
results['AND'] = test_opcode_fast("AND", [Opcode.IMM | (0xFF << 8), Opcode.PSH, Opcode.IMM | (0x0F << 8), Opcode.AND])
results['SHL'] = test_opcode_fast("SHL", [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.SHL])
results['SHR'] = test_opcode_fast("SHR", [Opcode.IMM | (16 << 8), Opcode.PSH, Opcode.IMM | (2 << 8), Opcode.SHR])

print("\n--- Comparison ---")
results['EQ'] = test_opcode_fast("EQ", [Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (5 << 8), Opcode.EQ])
results['NE'] = test_opcode_fast("NE", [Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.NE])
results['LT'] = test_opcode_fast("LT", [Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.LT])
results['GT'] = test_opcode_fast("GT", [Opcode.IMM | (3 << 8), Opcode.PSH, Opcode.IMM | (10 << 8), Opcode.GT])
results['LE'] = test_opcode_fast("LE", [Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (5 << 8), Opcode.LE])
results['GE'] = test_opcode_fast("GE", [Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (5 << 8), Opcode.GE])

print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)

perfect = [k for k, v in results.items() if v == 100.0]
good = [k for k, v in results.items() if 97.0 <= v < 100.0]
bad = [k for k, v in results.items() if v < 97.0]

print(f"✓ Perfect (100%): {len(perfect)} opcodes")
if perfect[:10]:
    print(f"  {', '.join(perfect[:10])}")
if len(perfect) > 10:
    print(f"  ... and {len(perfect) - 10} more")

print(f"⚠ Good (97%+): {len(good)} opcodes")
if good:
    print(f"  {', '.join(good)} - Expected (one-step delay)")

if bad:
    print(f"✗ Issues (<97%): {len(bad)} opcodes")
    print(f"  {', '.join(bad)}")
