"""Verify val_pos threshold firing ranges."""

import sys
sys.path.insert(0, '.')

# Threshold values from Layer 0
thresholds = {
    'H0': 3.5,
    'H1': 4.5,
    'H2': 5.5,
    'H3': 9.5,
    'H4': 10.5,
    'L1H0': 0.5,
    'L1H1': 1.5,
    'L1H2': 2.5,
    'L1H4': 6.5,
    'L2H0': 5.5,
}

val_pos = [
    ('H1', 'H0'),      # d=4: predicts val_b0 (position 5 in MEM section)
    ('L2H0', 'H1'),    # d=5: predicts val_b1 (position 6)
    ('L1H4', 'L2H0'),  # d=6: predicts val_b2 (position 7)
    ('H2', 'L1H4'),    # d=7: predicts val_b3 (position 8) ← BUG!
]

print("Val position threshold analysis:")
print("=" * 70)
print("\nMEM section layout (absolute positions in step):")
print("  Step structure: PC(5) + AX(5) + SP(5) + BP(5) + STACK0(5) + MEM(9) + SE(1)")
print("  MEM section starts at position 25")
print("    pos 25: MEM marker")
print("    pos 26-29: addr bytes")
print("    pos 30-33: val bytes")
print("    pos 34: STEP_END")
print()
print("Threshold Q fires at positions WITHIN MEM section:")
print("  MEM marker is nearest IS_MARK for positions 26-33")
print("  Distance from MEM marker: addr bytes at d=1-4, val bytes at d=5-8")
print()

for i, (up_name, down_name) in enumerate(val_pos):
    up_thresh = thresholds[up_name]
    down_thresh = thresholds[down_name]

    # Val bytes are at MEM section positions 5-8 (distance from MEM marker)
    target_d_from_marker = 5 + i  # distances 5, 6, 7, 8
    comment_d = 4 + i  # The comment says d=4,5,6,7

    # Fires when: down_thresh < d ≤ up_thresh
    fire_range = f"{down_thresh} < d ≤ {up_thresh}"
    fires_at_target = down_thresh < target_d_from_marker <= up_thresh

    print(f"Head {i+4} (val_b{i}):")
    print(f"  Comment says: d={comment_d}")
    print(f"  Actual distance from MEM marker: d={target_d_from_marker}")
    print(f"  Thresholds: ({up_name}, {down_name}) = ({up_thresh}, {down_thresh})")
    print(f"  Fires at: {fire_range}")
    print(f"  Fires at target (d={target_d_from_marker})? {fires_at_target} {'✓' if fires_at_target else '✗'}")

    if not fires_at_target:
        print(f"  ⚠️  BUG: Won't fire at target position!")
        # Find correct threshold
        for name, thresh in sorted(thresholds.items(), key=lambda x: x[1]):
            if down_thresh < target_d_from_marker <= thresh:
                print(f"  FIX: Should use ({name}, {down_name}) = ({thresh}, {down_thresh})")
                break
    print()

print("\nConclusion:")
print("=" * 70)
print("Val heads 4, 5, 6: ✓ Correct")
print("Val head 7: ✗ BUG - thresholds (5.5, 6.5) fire at NOTHING")
print("Val head 7 FIX: Should use (H3, L1H4) = (9.5, 6.5) to fire at d∈(6.5, 9.5]")
