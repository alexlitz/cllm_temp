#!/usr/bin/env python3
"""Test runner initialization step by step."""

import sys
print("1. Starting...", flush=True)

from neural_vm.run_vm import AutoregressiveVMRunner
print("2. Imported AutoregressiveVMRunner", flush=True)

print("3. Creating runner...", flush=True)
sys.stdout.flush()

runner = AutoregressiveVMRunner()
print("4. Runner created successfully!", flush=True)

print("✅ SUCCESS - Runner initialization works")
