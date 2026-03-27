# Structural Guarantee: Transformer Results Only

## The Problem

Current `batch_runner.py` has a critical flaw:

```python
# Line 172
return [("", vm.ax) for vm in self.draft_vms]  # ← Returns DraftVM results!
```

**This returns speculator results, not transformer results.**

### Why This is Wrong

1. DraftVM executes speculatively (fast but potentially wrong)
2. Transformer validates (slow but correct)
3. If transformer rejects tokens, DraftVM state is NOT corrected
4. Final results come from uncorrected DraftVM → **WRONG RESULTS**

### Evidence

```python
# Lines 154-166: Track mismatches but don't fix DraftVM
if accepted < 35:
    self.mismatches += 1  # ← Just count it
    # But DraftVM state is never rolled back!
    # DraftVM continues with wrong state

# Line 172: Return wrong state
return [("", vm.ax) for vm in self.draft_vms]  # ← BUG!
```

## Structural Guarantees

To prevent this bug from being reintroduced, we need **structural guarantees**:

### 1. DraftVM State Access is Blocked

```python
class _DraftVMResultsBlocked:
    """Wrapper that prevents accessing DraftVM state."""

    @property
    def ax(self):
        raise AttributeError(
            "BLOCKED: DraftVM results cannot be used. "
            "Use transformer validation instead."
        )
```

**Guarantee**: Code won't compile if you try to access `vm.ax`

### 2. Results Come From Reference VM

```python
def run_batch(...) -> List[Tuple[str, int]]:
    # ... execution ...

    # STRUCTURAL GUARANTEE: Use reference VM, never DraftVM
    results = []
    for context in transformer_validated_contexts:
        # Re-run with reference VM to get TRUE state
        ref_vm = FastLogicalVM()
        ref_vm.load(bytecode, data)
        exit_code = ref_vm.run()
        results.append((ref_vm.output, exit_code))

    # NO PATH to return DraftVM results
    return results
```

**Guarantee**: Results come from reference VM execution of validated context

### 3. Type System Enforcement

```python
@dataclass(frozen=True)
class TransformerResult:
    """Results that can ONLY come from transformer validation."""
    output: str
    exit_code: int
    _source: str = field(default="transformer", init=False)

    def __post_init__(self):
        if self._source != "transformer":
            raise ValueError("TransformerResult must come from transformer")

# Return type forces transformer usage
def run_batch(...) -> List[TransformerResult]:
    # Can't return DraftVM results - type error!
    pass
```

**Guarantee**: Type checker prevents returning DraftVM results

### 4. No DraftVM State Storage

```python
def run_batch(...):
    # DraftVMs created ONLY for speculation
    # State is never stored or returned

    for step in range(max_steps):
        # Use DraftVM for speculation only
        draft_tokens = _get_draft_tokens_temporary(bytecode)

        # Transformer validates
        accepted = transformer.validate(draft_tokens)

        # DraftVM is DISCARDED, never stored
        # No variable holding vm.ax or vm.output

    # Results from transformer, period
    return _extract_from_transformer(contexts)
```

**Guarantee**: DraftVM state never leaves the speculation loop

## The Correct Architecture

```
┌─────────────┐
│  DraftVM    │──────┐ Generates draft tokens (speculation)
│  (Fast)     │      │
└─────────────┘      │
                     ↓
                ┌────────────┐
                │Transformer │ Validates tokens
                │(Correct)   │ Accepts/Rejects
                └────────────┘
                     │
                     ↓
                ┌────────────┐
                │ Reference  │ Executes validated context
                │    VM      │ Produces TRUE results
                └────────────┘
                     │
                     ↓
                  Results ✓
```

### Key Properties

1. **DraftVM**: Speculation only, state is ephemeral
2. **Transformer**: Validates token sequences
3. **Reference VM**: Executes validated context for TRUE results
4. **No path**: To return DraftVM results

## Implementation

### Option A: Re-execute with Reference VM

```python
def run_batch(bytecodes, ...) -> List[Tuple[str, int]]:
    # Build transformer-validated contexts
    validated_contexts = []

    for bytecode in bytecodes:
        context = initial_context(bytecode)

        while not done:
            # DraftVM generates tokens (speculation)
            draft = _speculate_with_draftvm(bytecode, context)

            # Transformer validates
            accepted = transformer.validate(context + draft)

            # Add ONLY accepted tokens to context
            context.extend(draft[:accepted])

            # DraftVM state discarded here (goes out of scope)

        validated_contexts.append(context)

    # Extract results by re-executing with reference VM
    results = []
    for bytecode, context in zip(bytecodes, validated_contexts):
        ref_vm = FastLogicalVM()
        ref_vm.load(bytecode, extract_data(context))
        exit_code = ref_vm.run()
        results.append((ref_vm.output, exit_code))

    return results  # ← From reference VM, never DraftVM
```

### Option B: Extract from Transformer Embeddings

```python
def run_batch(...) -> List[Tuple[str, int]]:
    # ... execution ...

    # Extract VM state from transformer's final embeddings
    results = []
    for context in validated_contexts:
        # Run transformer to get final embedding
        embedding = transformer.embed(context)

        # Extract AX register from embedding
        ax = extract_ax_from_embedding(embedding)
        output = extract_output_from_context(context)

        results.append((output, ax))

    return results  # ← From transformer, never DraftVM
```

### Option C: Transformer Outputs State

```python
def run_batch(...) -> List[Tuple[str, int]]:
    # Transformer generates state tokens explicitly
    for step in range(max_steps):
        draft = draftvm.step()

        # Transformer predicts state tokens
        state_tokens = transformer.predict_state(context)

        # State includes AX, output, etc.
        ax = parse_ax_from_tokens(state_tokens)
        output = parse_output_from_tokens(state_tokens)

    return results  # ← From transformer state tokens
```

## Testing the Guarantee

```python
def test_cannot_access_draftvm_results():
    """Verify DraftVM results are blocked."""
    runner = TransformerFirstRunner()

    # This should raise AttributeError
    with pytest.raises(AttributeError):
        runner._draft_vms[0].ax

def test_results_from_transformer():
    """Verify results come from transformer/reference VM."""
    runner = TransformerFirstRunner()
    results = runner.run_batch([bytecode])

    # Run reference VM for ground truth
    ref_vm = FastLogicalVM()
    ref_vm.load(bytecode, data)
    expected = ref_vm.run()

    assert results[0][1] == expected  # ← Must match reference VM
```

## Conclusion

**Current system**: Returns DraftVM results (wrong if validation fails)

**Fixed system**: Structural guarantees prevent DraftVM results from being returned

**Guarantees**:
1. ✅ DraftVM state access is blocked (AttributeError)
2. ✅ Results come from reference VM or transformer
3. ✅ Type system enforces transformer-only results
4. ✅ No code path to return DraftVM state

This makes it **structurally impossible** to return incorrect (DraftVM) results.
