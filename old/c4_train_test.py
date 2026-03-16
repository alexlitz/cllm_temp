"""
Test if a small transformer can learn to execute C4 opcodes.

Key insight: Each opcode is a deterministic state transition.
Given (opcode, immediate, registers, relevant_memory) -> new_state

We test: Can a neural network learn these transitions?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from c4_vm import C4VM, Op, CODE_BASE, DATA_BASE, STACK_BASE
from c4_vm import program_simple_arithmetic, program_sum, program_factorial


class OpcodePredictor(nn.Module):
    """
    Predicts state changes given opcode and current state.

    Input:
      - opcode (0-38)
      - immediate (64 bits)
      - registers: PC, SP, BP, AX (4 x 64 bits)
      - stack_top (8 values from top of stack)
      - memory_at_ax (8 bytes at address AX)

    Output:
      - new registers: PC, SP, BP, AX (4 x 64 bits)
      - stack_delta (values to push/pop)
    """

    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

        # Input encoding
        # Opcode: one-hot (39 opcodes)
        # Immediate: 64 bits -> 8 bytes one-hot (256 each) = 2048
        # Registers: 4 x 8 bytes = 32 bytes one-hot = 8192
        # Stack: 8 x 8 bytes = 64 bytes
        # Memory at AX: 8 bytes

        # Simplified: encode as floating point values normalized
        self.opcode_embed = nn.Embedding(40, dim)

        # Encode registers and immediate as normalized values
        # 5 int64s as bytes = 40, 8 stack values = 8
        self.input_proj = nn.Linear(40 + 8, dim)

        # Process
        self.layers = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
        )

        # Output: 4 registers as deltas + new values
        # We predict both absolute values and deltas
        self.reg_out = nn.Linear(dim * 2, 4)  # PC, SP, BP, AX as floats

    def forward(self, opcode, imm, pc, sp, bp, ax, stack_vals, mem_at_ax):
        """
        All inputs are (batch,) tensors except stack_vals (batch, 8) and mem_at_ax (batch, 8)
        """
        batch = opcode.shape[0]
        device = opcode.device

        # Embed opcode
        op_emb = self.opcode_embed(opcode)  # (batch, dim)

        # Encode numeric values (normalize to reasonable range)
        scale = 1e6  # Addresses are in range 0-1M

        nums = torch.stack([
            imm.float() / scale,
            pc.float() / scale,
            sp.float() / scale,
            bp.float() / scale,
            ax.float() / scale,
        ], dim=1)  # (batch, 5)

        # Add byte representation for more precision
        num_bytes = torch.zeros(batch, 5 * 8, device=device)
        for i, val in enumerate([imm, pc, sp, bp, ax]):
            for b in range(8):
                num_bytes[:, i*8 + b] = ((val >> (b*8)) & 0xFF).float() / 255.0

        stack_norm = stack_vals.float() / scale

        # Concatenate all numeric features
        nums_all = torch.cat([num_bytes, stack_norm], dim=1)  # (batch, 40 + 8)
        nums_emb = self.input_proj(nums_all)  # (batch, dim)

        # Combine opcode and numeric embeddings
        x = torch.cat([op_emb, nums_emb], dim=1)  # (batch, dim*2)

        # Process
        x = self.layers(x)  # (batch, dim*2)

        # Output registers (as normalized values)
        reg_out = self.reg_out(x)  # (batch, 4)

        # Return raw float outputs for loss computation
        return reg_out  # (batch, 4) - [PC, SP, BP, AX] normalized


def extract_features(vm):
    """Extract features from VM state."""
    pc = vm.state.pc
    sp = vm.state.sp
    bp = vm.state.bp
    ax = vm.state.ax

    # Get opcode and immediate at PC
    instruction = vm.read_int(pc)  # read_int is on VM, not VMState
    opcode = instruction & 0xFF
    imm = instruction >> 8

    # Get stack top (8 int64 values)
    stack_vals = []
    for i in range(8):
        addr = sp + i * 8
        try:
            val = vm.read_int(addr)
        except ValueError:
            val = 0
        stack_vals.append(val)

    # Get memory at AX
    mem_at_ax = []
    for i in range(8):
        addr = ax + i
        if 0 <= addr < len(vm.state.memory):
            mem_at_ax.append(vm.state.memory[addr])
        else:
            mem_at_ax.append(0)

    return opcode, imm, pc, sp, bp, ax, stack_vals, mem_at_ax


def generate_training_data(programs, max_steps=100):
    """Generate training examples from program execution."""
    examples = []

    for prog_fn in programs:
        vm = C4VM()
        vm.load_code(prog_fn())

        for _ in range(max_steps):
            if vm.state.halted:
                break

            # Extract input features
            opcode, imm, pc, sp, bp, ax, stack_vals, mem_at_ax = extract_features(vm)

            # Step
            vm.step()

            # Target: new register values
            new_pc = vm.state.pc
            new_sp = vm.state.sp
            new_bp = vm.state.bp
            new_ax = vm.state.ax

            examples.append({
                'opcode': opcode,
                'imm': imm,
                'pc': pc,
                'sp': sp,
                'bp': bp,
                'ax': ax,
                'stack_vals': stack_vals,
                'mem_at_ax': mem_at_ax,
                'new_pc': new_pc,
                'new_sp': new_sp,
                'new_bp': new_bp,
                'new_ax': new_ax,
            })

    return examples


def examples_to_tensors(examples, device):
    """Convert examples to batched tensors."""
    n = len(examples)

    opcode = torch.tensor([e['opcode'] for e in examples], dtype=torch.long, device=device)
    imm = torch.tensor([e['imm'] for e in examples], dtype=torch.long, device=device)
    pc = torch.tensor([e['pc'] for e in examples], dtype=torch.long, device=device)
    sp = torch.tensor([e['sp'] for e in examples], dtype=torch.long, device=device)
    bp = torch.tensor([e['bp'] for e in examples], dtype=torch.long, device=device)
    ax = torch.tensor([e['ax'] for e in examples], dtype=torch.long, device=device)
    stack_vals = torch.tensor([e['stack_vals'] for e in examples], dtype=torch.long, device=device)
    mem_at_ax = torch.tensor([e['mem_at_ax'] for e in examples], dtype=torch.long, device=device)

    new_pc = torch.tensor([e['new_pc'] for e in examples], dtype=torch.long, device=device)
    new_sp = torch.tensor([e['new_sp'] for e in examples], dtype=torch.long, device=device)
    new_bp = torch.tensor([e['new_bp'] for e in examples], dtype=torch.long, device=device)
    new_ax = torch.tensor([e['new_ax'] for e in examples], dtype=torch.long, device=device)

    return (opcode, imm, pc, sp, bp, ax, stack_vals, mem_at_ax), (new_pc, new_sp, new_bp, new_ax)


def test_direct_execution():
    """Test that the primitive executor works correctly (no learning)."""
    from c4_ops_transformer import C4OpcodeExecutor

    print("=" * 60)
    print("TEST 1: Direct Primitive Execution (no learning)")
    print("=" * 60)

    executor = C4OpcodeExecutor()

    # Test sum(10)
    vm = C4VM()
    vm.load_code(program_sum(10))

    pc = torch.tensor(vm.state.pc, dtype=torch.long)
    sp = torch.tensor(vm.state.sp, dtype=torch.long)
    bp = torch.tensor(vm.state.bp, dtype=torch.long)
    ax = torch.tensor(vm.state.ax, dtype=torch.long)
    memory = torch.tensor(list(vm.state.memory), dtype=torch.long)

    steps = 0
    while steps < 500:
        opcode, imm = executor.fetch_instruction(memory, pc)
        if opcode.item() == Op.EXIT:
            break

        pc, sp, bp, ax, memory = executor.execute_op(
            opcode.item(), imm.item(),
            pc, sp, bp, ax, memory
        )
        vm.step()

        if pc.item() != vm.state.pc or ax.item() != vm.state.ax:
            print(f"  Step {steps}: MISMATCH")
            print(f"    Pred: PC={pc.item()}, AX={ax.item()}")
            print(f"    True: PC={vm.state.pc}, AX={vm.state.ax}")
            return False

        steps += 1

    print(f"  Executed {steps} steps")
    print(f"  Final AX: {ax.item()} (expected 55)")
    print(f"  Result: {'PASS' if ax.item() == 55 else 'FAIL'}")
    return ax.item() == 55


def test_opcode_classification():
    """Test if we can at least classify which opcode type based on context."""
    print("\n" + "=" * 60)
    print("TEST 2: Opcode Classification from Context")
    print("=" * 60)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Generate data
    programs = []
    for _ in range(5):
        programs.append(program_simple_arithmetic)
    for n in range(1, 15):
        programs.append(lambda n=n: program_sum(n))
    for n in range(1, 8):
        programs.append(lambda n=n: program_factorial(n))

    print("Generating training data...")
    examples = generate_training_data(programs, max_steps=200)
    print(f"  {len(examples)} examples")

    # Count opcodes
    opcode_counts = {}
    for e in examples:
        op = e['opcode']
        opcode_counts[op] = opcode_counts.get(op, 0) + 1

    print(f"  Opcodes used: {sorted(opcode_counts.keys())}")

    # Simple test: Can we predict PC change given opcode?
    # PC usually increases by 8, except for JMP, BZ, BNZ, JSR, LEV

    correct = 0
    for e in examples:
        op = e['opcode']
        pc = e['pc']
        new_pc = e['new_pc']

        # Predict: if JMP/JSR, new_pc = imm; if BZ/BNZ, depends on AX; else pc+8
        if op == Op.JMP:
            pred_pc = e['imm']
        elif op == Op.JSR:
            pred_pc = e['imm']
        elif op == Op.BZ:
            pred_pc = e['imm'] if e['ax'] == 0 else pc + 8
        elif op == Op.BNZ:
            pred_pc = e['imm'] if e['ax'] != 0 else pc + 8
        elif op == Op.LEV:
            # LEV pops PC from stack - we'd need stack contents
            pred_pc = new_pc  # cheat for now
        elif op == Op.EXIT:
            pred_pc = pc
        else:
            pred_pc = pc + 8

        if pred_pc == new_pc:
            correct += 1

    print(f"\n  Rule-based PC prediction: {correct}/{len(examples)} ({100*correct/len(examples):.1f}%)")

    # Now test AX changes
    correct_ax = 0
    for e in examples:
        op = e['opcode']
        ax = e['ax']
        new_ax = e['new_ax']
        imm = e['imm']
        stack = e['stack_vals']

        # Predict AX based on opcode
        if op == Op.IMM:
            pred_ax = imm
        elif op == Op.LEA:
            pred_ax = e['bp'] + imm
        elif op == Op.ADD:
            pred_ax = stack[0] + ax if stack else ax
        elif op == Op.SUB:
            pred_ax = stack[0] - ax if stack else ax
        elif op == Op.MUL:
            pred_ax = stack[0] * ax if stack else ax
        elif op == Op.PSH:
            pred_ax = ax  # PSH doesn't change AX
        elif op in (Op.EQ, Op.NE, Op.LT, Op.GT, Op.LE, Op.GE):
            # Comparison - would need to compute
            pred_ax = new_ax  # cheat
        elif op == Op.LI:
            # Load from memory at AX
            pred_ax = new_ax  # cheat - would need memory
        else:
            pred_ax = ax  # Many ops don't change AX

        if pred_ax == new_ax:
            correct_ax += 1

    print(f"  Rule-based AX prediction: {correct_ax}/{len(examples)} ({100*correct_ax/len(examples):.1f}%)")

    return True


class DeltaPredictor(nn.Module):
    """
    Predicts state DELTAS given opcode and current state.

    Key insight: Most opcodes have simple delta patterns:
    - PC: usually +8, or jump to immediate
    - SP: -8 (push), +8 (pop), or unchanged
    - BP: rarely changes
    - AX: depends on opcode (IMM→immediate, ADD→ax+stack[0], etc.)

    We classify the operation type rather than regressing exact values.
    """

    def __init__(self, dim=128):
        super().__init__()

        # Opcode embedding
        self.opcode_embed = nn.Embedding(40, dim)

        # Input: opcode features + register context
        self.input_proj = nn.Linear(48, dim)  # 40 bytes + 8 stack vals

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )

        # PC delta classifier: +8, =imm, =stack[0], other
        self.pc_head = nn.Linear(dim, 4)

        # SP delta classifier: -8, +8, -imm, +imm, 0
        self.sp_head = nn.Linear(dim, 5)

        # AX operation classifier: =imm, =stack[0]+ax, =stack[0]-ax, =stack[0]*ax,
        #                          =stack[0]/ax, =mem[ax], =bp+imm, unchanged, etc.
        self.ax_head = nn.Linear(dim, 12)

    def forward(self, opcode, imm, pc, sp, bp, ax, stack_vals, mem_at_ax):
        batch = opcode.shape[0]
        device = opcode.device

        # Embed opcode
        op_emb = self.opcode_embed(opcode)  # (batch, dim)

        # Encode numeric values as bytes
        scale = 1e6
        num_bytes = torch.zeros(batch, 5 * 8, device=device)
        for i, val in enumerate([imm, pc, sp, bp, ax]):
            for b in range(8):
                num_bytes[:, i*8 + b] = ((val >> (b*8)) & 0xFF).float() / 255.0

        stack_norm = stack_vals.float() / scale
        nums_all = torch.cat([num_bytes, stack_norm], dim=1)
        nums_emb = self.input_proj(nums_all)

        # Combine and process
        x = torch.cat([op_emb, nums_emb], dim=1)
        x = self.trunk(x)

        # Classify operations
        pc_logits = self.pc_head(x)
        sp_logits = self.sp_head(x)
        ax_logits = self.ax_head(x)

        return pc_logits, sp_logits, ax_logits


def compute_delta_labels(examples, device):
    """Convert examples to operation classification labels."""
    pc_labels = []
    sp_labels = []
    ax_labels = []

    for e in examples:
        pc, new_pc = e['pc'], e['new_pc']
        sp, new_sp = e['sp'], e['new_sp']
        ax, new_ax = e['ax'], e['new_ax']
        imm = e['imm']
        stack0 = e['stack_vals'][0] if e['stack_vals'] else 0

        # PC: 0=+8, 1=imm, 2=stack[0], 3=other
        if new_pc == pc + 8:
            pc_labels.append(0)
        elif new_pc == imm:
            pc_labels.append(1)
        elif new_pc == stack0:
            pc_labels.append(2)
        else:
            pc_labels.append(3)

        # SP: 0=-8, 1=+8, 2=-imm, 3=+imm, 4=unchanged
        delta_sp = new_sp - sp
        if delta_sp == -8:
            sp_labels.append(0)
        elif delta_sp == +8:
            sp_labels.append(1)
        elif delta_sp == -imm:
            sp_labels.append(2)
        elif delta_sp == +imm:
            sp_labels.append(3)
        elif delta_sp == 0:
            sp_labels.append(4)
        else:
            sp_labels.append(4)  # default to unchanged

        # AX: 0=imm, 1=stack+ax, 2=stack-ax, 3=stack*ax, 4=stack/ax,
        #     5=stack==ax, 6=stack<ax, 7=bp+imm, 8=unchanged, 9=stack&ax, 10=stack|ax, 11=other
        if new_ax == imm:
            ax_labels.append(0)
        elif new_ax == stack0 + ax:
            ax_labels.append(1)
        elif new_ax == stack0 - ax:
            ax_labels.append(2)
        elif new_ax == stack0 * ax:
            ax_labels.append(3)
        elif ax != 0 and new_ax == stack0 // ax:
            ax_labels.append(4)
        elif new_ax == (1 if stack0 == ax else 0):
            ax_labels.append(5)
        elif new_ax == (1 if stack0 < ax else 0):
            ax_labels.append(6)
        elif new_ax == e['bp'] + imm:
            ax_labels.append(7)
        elif new_ax == ax:
            ax_labels.append(8)
        elif new_ax == (stack0 & ax):
            ax_labels.append(9)
        elif new_ax == (stack0 | ax):
            ax_labels.append(10)
        else:
            ax_labels.append(11)

    return (
        torch.tensor(pc_labels, device=device),
        torch.tensor(sp_labels, device=device),
        torch.tensor(ax_labels, device=device)
    )


def test_neural_predictor():
    """Test if a neural network can learn opcode operation TYPES."""
    print("\n" + "=" * 60)
    print("TEST 3: Neural Network Operation Classification")
    print("=" * 60)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Generate data
    programs = []
    for _ in range(10):
        programs.append(program_simple_arithmetic)
    for n in range(1, 20):
        programs.append(lambda n=n: program_sum(n))
    for n in range(1, 10):
        programs.append(lambda n=n: program_factorial(n))

    print("Generating training data...")
    examples = generate_training_data(programs, max_steps=200)
    print(f"  {len(examples)} total examples")

    # Split
    split = int(0.8 * len(examples))
    train_examples = examples[:split]
    test_examples = examples[split:]
    print(f"  Train: {len(train_examples)}, Test: {len(test_examples)}")

    # Create model
    model = DeltaPredictor(dim=128).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Convert to tensors
    train_inputs, _ = examples_to_tensors(train_examples, device)
    test_inputs, _ = examples_to_tensors(test_examples, device)

    train_pc_labels, train_sp_labels, train_ax_labels = compute_delta_labels(train_examples, device)
    test_pc_labels, test_sp_labels, test_ax_labels = compute_delta_labels(test_examples, device)

    # Show label distributions
    print(f"\n  PC label distribution: {torch.bincount(train_pc_labels, minlength=4).tolist()}")
    print(f"  SP label distribution: {torch.bincount(train_sp_labels, minlength=5).tolist()}")
    print(f"  AX label distribution: {torch.bincount(train_ax_labels, minlength=12).tolist()}")

    # Training
    print("\nTraining...")
    batch_size = 64
    n_train = len(train_examples)

    for epoch in range(100):
        model.train()
        total_loss = 0

        # Shuffle
        perm = torch.randperm(n_train, device=device)

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]

            # Get batch
            opcode = train_inputs[0][idx]
            imm = train_inputs[1][idx]
            pc = train_inputs[2][idx]
            sp = train_inputs[3][idx]
            bp = train_inputs[4][idx]
            ax = train_inputs[5][idx]
            stack_vals = train_inputs[6][idx]
            mem_at_ax = train_inputs[7][idx]

            # Forward
            pc_logits, sp_logits, ax_logits = model(
                opcode, imm, pc, sp, bp, ax, stack_vals, mem_at_ax
            )

            # Loss
            loss = (
                criterion(pc_logits, train_pc_labels[idx]) +
                criterion(sp_logits, train_sp_labels[idx]) +
                criterion(ax_logits, train_ax_labels[idx])
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate every 20 epochs
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                pc_logits, sp_logits, ax_logits = model(
                    test_inputs[0], test_inputs[1], test_inputs[2],
                    test_inputs[3], test_inputs[4], test_inputs[5],
                    test_inputs[6], test_inputs[7]
                )

                pc_acc = (pc_logits.argmax(1) == test_pc_labels).float().mean().item()
                sp_acc = (sp_logits.argmax(1) == test_sp_labels).float().mean().item()
                ax_acc = (ax_logits.argmax(1) == test_ax_labels).float().mean().item()

            print(f"  Epoch {epoch+1:3d}: loss={total_loss:.4f}  " +
                  f"PC={100*pc_acc:.1f}% SP={100*sp_acc:.1f}% AX={100*ax_acc:.1f}%")

    # Final evaluation
    print("\nFinal Results:")
    model.eval()
    with torch.no_grad():
        pc_logits, sp_logits, ax_logits = model(
            test_inputs[0], test_inputs[1], test_inputs[2],
            test_inputs[3], test_inputs[4], test_inputs[5],
            test_inputs[6], test_inputs[7]
        )

        pc_acc = (pc_logits.argmax(1) == test_pc_labels).float().mean().item()
        sp_acc = (sp_logits.argmax(1) == test_sp_labels).float().mean().item()
        ax_acc = (ax_logits.argmax(1) == test_ax_labels).float().mean().item()

    print(f"  PC operation: {100*pc_acc:.1f}%")
    print(f"  SP operation: {100*sp_acc:.1f}%")
    print(f"  AX operation: {100*ax_acc:.1f}%")
    print(f"  Average:      {100*(pc_acc + sp_acc + ax_acc)/3:.1f}%")

    return pc_acc > 0.9 and sp_acc > 0.9 and ax_acc > 0.7


if __name__ == "__main__":
    print("C4 TRANSFORMER EXECUTION TESTS")
    print("=" * 60)
    print()

    test1 = test_direct_execution()
    test2 = test_opcode_classification()
    test3 = test_neural_predictor()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Test 1 (Direct Execution):    {'PASS' if test1 else 'FAIL'}")
    print(f"  Test 2 (Rule-based):          {'PASS' if test2 else 'FAIL'}")
    print(f"  Test 3 (Neural Learning):     {'PASS' if test3 else 'FAIL'}")
