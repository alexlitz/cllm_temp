# Purity Check System Protection

## Purpose

This document explains how to protect the purity checking system from unintended modifications and ensure test integrity over time.

## Protected Files

### Core Protected Files

1. **`neural_vm/purity_check.py`**
   - Contains opcode classifications
   - Implements purity violation detection
   - **Protection Level**: HIGH - Changes require justification

2. **`neural_vm/PURITY_CHECK_CHANGELOG.md`**
   - Documents all changes to purity system
   - **Protection Level**: HIGH - All modifications must be logged

3. **`docs/PURITY_PROTECTION.md`** (this file)
   - Protection guidelines
   - **Protection Level**: MEDIUM - Update when procedures change

### Related Protected Files

4. **`tests/test_purity.py`**
   - Validates purity checking
   - **Protection Level**: MEDIUM - Tests must remain comprehensive

5. **`docs/HANDLER_DEPENDENCY_DISCOVERY.md`**
   - Historical context
   - **Protection Level**: LOW - Read-only historical record

## Protection Mechanisms

### 1. File Integrity Checking

**Hash Validation**:
```python
# In purity_check.py
def verify_purity_check_integrity() -> bool:
    """Verify this file hasn't been modified unexpectedly."""
    # Compares current file hash with expected hash
    # Warns if modifications detected
```

**Usage**:
```python
from neural_vm.purity_check import verify_purity_check_integrity

if not verify_purity_check_integrity():
    print("WARNING: Purity check system may have been modified!")
```

### 2. Mandatory Changelog

**Rule**: Every change to `purity_check.py` MUST be documented in `PURITY_CHECK_CHANGELOG.md`

**Enforcement**:
- Pre-commit hook (TODO: implement)
- Code review requirement
- CI/CD check (TODO: implement)

**Template**:
```markdown
### Version X.Y.Z - YYYY-MM-DD
**Title**: Brief description
**Author**: Your name
**Justification**: Why was this necessary?
**Changes**: What was modified?
**Testing**: How was it validated?
```

### 3. Classification Lock

**Opcode classifications should not change without investigation**:

```python
# Before moving opcode between classifications:
# 1. Document why in PURITY_CHECK_CHANGELOG.md
# 2. Run comprehensive tests
# 3. Update related documentation

# Example: Moving EQ from BROKEN to PURE_NEURAL
# Requires:
# - Test showing EQ works without handler
# - Update in PURITY_CHECK_CHANGELOG.md
# - Update in OPCODE_TEST_STATUS.md
# - Update in HANDLER_DEPENDENCY_DISCOVERY.md
```

### 4. Test Protection

**Purity tests must not be disabled or weakened**:

```python
# tests/test_purity.py

def test_pure_neural_opcodes_work_without_handlers():
    """CRITICAL: This test must always pass.

    If this test fails, it means:
    1. A pure neural opcode has regressed, OR
    2. The opcode classification is wrong

    DO NOT simply disable this test. Investigate and fix the root cause.
    """
    for opcode in PURE_NEURAL_OPCODES:
        # Test each opcode without handlers
        assert opcode_works_neurally(opcode)
```

## Modification Procedures

### Adding a New Opcode Classification

**Scenario**: You discovered ADD now works neurally.

**Steps**:
1. **Verify** with comprehensive tests (multiple test cases)
2. **Document** in PURITY_CHECK_CHANGELOG.md:
   ```markdown
   ### Version 1.1.0 - 2026-04-10
   **Title**: Move ADD to PURE_NEURAL_OPCODES
   **Author**: Developer Name
   **Justification**: Fixed neural weights for ADD operation.
   **Changes**:
   - Moved opcode 25 (ADD) from HANDLER_REQUIRED_OPCODES to PURE_NEURAL_OPCODES
   **Testing**:
   - Tested ADD with various operands without handlers
   - All tests pass with pure neural execution
   ```
3. **Update** `purity_check.py`:
   ```python
   # Move from HANDLER_REQUIRED_OPCODES
   HANDLER_REQUIRED_OPCODES = {
       # 25,  # ADD - REMOVED, now pure neural
       26,  # SUB
       # ...
   }

   PURE_NEURAL_OPCODES = {
       # ...existing...
       25,  # ADD - Added 2026-04-10
   }
   ```
4. **Update** related documentation:
   - `docs/OPCODE_TEST_STATUS.md`
   - `docs/HANDLER_DEPENDENCY_DISCOVERY.md`
5. **Run** validation:
   ```bash
   python3 -m pytest tests/test_purity.py -v
   python3 neural_vm/purity_check.py  # Self-test
   ```
6. **Commit** with clear message:
   ```
   fix: Move ADD to pure neural execution

   ADD operation now works without handler after fixing
   neural weights. See PURITY_CHECK_CHANGELOG.md v1.1.0.

   Tests: test_purity.py::test_add_pure_neural
   ```

### Changing Purity Checker Logic

**Scenario**: You need to modify `PurityChecker.check_handler_use()`.

**Steps**:
1. **Justify** the change in PURITY_CHECK_CHANGELOG.md first
2. **Write tests** for the new behavior
3. **Implement** the change
4. **Validate** existing tests still pass
5. **Update** documentation
6. **Review** impact on all usages

**Example Bad Change** (DO NOT DO):
```python
# BAD: Silently allows violations
def check_handler_use(self, opcode, opcode_name):
    # Disabled purity checking for debugging
    return  # ← WRONG! Documents the violation
```

**Example Good Change**:
```python
# GOOD: Properly justified exception
def check_handler_use(self, opcode, opcode_name):
    # Allow handlers for debugging if DEBUG_MODE env var set
    # Documented in PURITY_CHECK_CHANGELOG.md v1.2.0
    if os.environ.get('DEBUG_MODE'):
        print(f"DEBUG: Handler used for {opcode_name}", file=sys.stderr)
        return

    # Normal purity checking continues
    # ...existing logic...
```

### Disabling Purity Checks (Discouraged)

**When is it acceptable to disable purity checks?**

**Acceptable**:
- Temporary debugging (must re-enable before commit)
- Performance profiling (comparing handler vs neural)
- Testing handler implementations

**NOT Acceptable**:
- To make failing tests pass
- Because purity violations are inconvenient
- Without documentation

**How to temporarily disable**:
```python
# Option 1: Use UNRESTRICTED mode (for testing only)
checker = PurityChecker(ExecutionMode.UNRESTRICTED)

# Option 2: Set environment variable
# export DISABLE_PURITY_CHECKS=1
if not os.environ.get('DISABLE_PURITY_CHECKS'):
    checker.check_handler_use(opcode, name)
```

**Requirements if disabling**:
1. Document reason in code comments
2. Set expiration date (`# TODO: Re-enable by 2026-04-15`)
3. Add tracking issue
4. Never commit with checks disabled

## Integrity Verification

### Manual Verification

```bash
# Run purity check self-test
python3 neural_vm/purity_check.py

# Run purity tests
python3 -m pytest tests/test_purity.py -v

# Check for unauthorized modifications
git diff neural_vm/purity_check.py
git log --oneline neural_vm/purity_check.py
```

### Automated Verification (TODO)

**Pre-commit Hook**:
```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check if purity_check.py modified
if git diff --cached --name-only | grep -q "neural_vm/purity_check.py"; then
    echo "WARNING: purity_check.py modified"
    echo "Checking for changelog entry..."

    if ! git diff --cached neural_vm/PURITY_CHECK_CHANGELOG.md | grep -q "Version"; then
        echo "ERROR: Must update PURITY_CHECK_CHANGELOG.md when modifying purity_check.py"
        exit 1
    fi
fi
```

**CI/CD Check**:
```yaml
# .github/workflows/purity_check.yml
name: Purity Check Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run purity tests
        run: python3 -m pytest tests/test_purity.py -v
      - name: Verify changelog
        run: |
          if git diff --name-only ${{ github.event.before }}..${{ github.sha }} | grep -q purity_check.py; then
            grep -q "Version.*$(date +%Y-%m-%d)" neural_vm/PURITY_CHECK_CHANGELOG.md || exit 1
          fi
```

## Common Pitfalls

### Pitfall 1: Weakening Classifications

**Symptom**: Moving opcodes from PURE_NEURAL to HANDLER_REQUIRED to "fix" tests.

**Why it's bad**: Masks regressions in neural weights.

**Correct approach**: Investigate why opcode stopped working neurally, fix root cause.

### Pitfall 2: Adding Handlers to Pure Opcodes

**Symptom**: Adding handler for IMM to "improve performance" without documentation.

**Why it's bad**: Creates handler dependency, reduces pure neural coverage.

**Correct approach**: Document justification, update classifications, note in changelog.

### Pitfall 3: Disabling Violation Warnings

**Symptom**: Commenting out warnings to reduce console noise.

**Why it's bad**: Hides real issues, allows violations to accumulate.

**Correct approach**: Fix violations, or properly classify opcodes as HANDLER_REQUIRED.

### Pitfall 4: Undocumented Changes

**Symptom**: Modifying `purity_check.py` without updating CHANGELOG.

**Why it's bad**: Loses history, makes debugging harder, no review trail.

**Correct approach**: Always update CHANGELOG with every change.

## Review Checklist

Before modifying purity system, verify:

- [ ] Change is necessary and justified
- [ ] PURITY_CHECK_CHANGELOG.md updated
- [ ] Tests updated/added for new behavior
- [ ] All existing tests pass
- [ ] Related documentation updated
- [ ] No classifications weakened without investigation
- [ ] Commit message references changelog entry
- [ ] Code review completed

## Emergency Procedures

### If Purity System is Broken

**Scenario**: Purity check system has a critical bug blocking development.

**Steps**:
1. **DO NOT** simply disable the system
2. **File** an issue documenting the problem
3. **Create** a hotfix branch
4. **Fix** the issue with minimal changes
5. **Test** thoroughly
6. **Document** in CHANGELOG
7. **Review** and merge quickly
8. **Post-mortem** to prevent recurrence

### If Changelog is Out of Sync

**Scenario**: Discovered purity_check.py was modified without changelog update.

**Steps**:
1. **Identify** what changed (`git diff` or `git log`)
2. **Investigate** why (who made change, when, why)
3. **Retroactively document** in CHANGELOG with note: "Retroactive entry for undocumented change"
4. **Add** protections to prevent recurrence (pre-commit hook)

## Questions and Answers

**Q: Why is this protection necessary?**
A: Without protection, tests can be weakened over time, making it hard to know if opcodes truly work neurally or just via handlers.

**Q: Can I add a handler for a pure neural opcode?**
A: Only with strong justification (e.g., 32-bit correctness, performance). Must be documented in CHANGELOG.

**Q: What if I disagree with an opcode classification?**
A: Test it! If opcode works without handler, move to PURE_NEURAL. If not, move to HANDLER_REQUIRED. Document evidence.

**Q: How do I test if an opcode is truly pure neural?**
A: Remove its handler (if any), run comprehensive tests, verify correct behavior.

**Q: This seems overly bureaucratic...**
A: The alternative is slowly regressing to all-handler execution without noticing. Protection maintains neural execution integrity.

## Related Documents

- `neural_vm/purity_check.py` - Core implementation
- `neural_vm/PURITY_CHECK_CHANGELOG.md` - Change history
- `docs/HANDLER_DEPENDENCY_DISCOVERY.md` - Why this exists
- `docs/OPCODE_TEST_STATUS.md` - Current opcode status
- `tests/test_purity.py` - Validation tests

---

**Date**: 2026-04-07
**Version**: 1.0.0
**Status**: Active protection
