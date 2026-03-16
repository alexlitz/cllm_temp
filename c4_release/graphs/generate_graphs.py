"""Generate all graphs for the Neural VM documentation."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = os.path.dirname(os.path.abspath(__file__))


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 50, 1.0, np.where(x < -50, 0.0, 1 / (1 + np.exp(-x))))


def silu(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 50, x, np.where(x < -50, 0.0, x / (1 + np.exp(-x))))


def neural_step(x, S, eps):
    r"""Neural step: [silu(S(x+eps)) - silu(Sx)] / (S*eps)"""
    return (silu(S * (x + eps)) - silu(S * x)) / (S * eps)


def neural_point(x, k, S, eps):
    r"""Point indicator via second-difference of silu:

    f_eps(x-k) = [silu(S(x-k+eps)) - 2*silu(S(x-k)) + silu(S(x-k-eps))]
                 / [S * eps * (2*sigma(S*eps) - 1)]
    """
    z = x - k
    num = silu(S * (z + eps)) - 2 * silu(S * z) + silu(S * (z - eps))
    denom = S * eps * (2 * sigmoid(S * eps) - 1)
    return num / denom


# ── 1. SiLU Function (zoomed out) ──

fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(-20, 20, 2000)
ax.plot(x, silu(x), 'b-', linewidth=2.5, label=r'SiLU(x) = x $\cdot$ $\sigma$(x)')
ax.plot(x, x, '--', color='gray', linewidth=1.5, label='y = x')
ax.plot(x, np.maximum(x, 0), ':', color='orange', linewidth=1.5, alpha=0.7, label='ReLU(x)')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('SiLU(x)', fontsize=13)
ax.set_title('SiLU Activation Function', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'silu_function.png'), dpi=150)
plt.close(fig)
print("  silu_function.png")


# ── 2. Step Function ──

fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(-3, 3, 8000)

ax.plot([-3, 0, 0, 3], [0, 0, 1, 1], '--', color='gray', linewidth=2.5, label='Ideal step(x)')

configs = [
    (20,   1.0, 'C0', r'$\epsilon$=1, S=20'),
    (100,  0.3, 'C2', r'$\epsilon$=0.3, S=100'),
    (1000, 0.1, 'C3', r'$\epsilon$=0.1, S=1000'),
]
for S, eps, color, label in configs:
    y = neural_step(x, S, eps)
    ax.plot(x, y, color=color, linewidth=2, label=label)

ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('step(x)', fontsize=13)
ax.set_title('Neural Step Function', fontsize=15)
ax.legend(fontsize=12)
ax.set_ylim(-0.15, 1.25)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'step_function.png'), dpi=150)
plt.close(fig)
print("  step_function.png")


# ── 3. Point Indicator ──

fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(-3, 3, 16000)

configs = [
    (20,   1.0, 'C0', r'$\epsilon$=1, S=20'),
    (100,  0.3, 'C2', r'$\epsilon$=0.3, S=100'),
    (1000, 0.1, 'C3', r'$\epsilon$=0.1, S=1000'),
]
for S, eps, color, label in configs:
    y = neural_point(x, 0, S, eps)
    ax.plot(x, y, color=color, linewidth=2, label=label)

ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('point(x, 0)', fontsize=13)
ax.set_title('Point Indicator Function', fontsize=15)
ax.legend(fontsize=12)
ax.set_ylim(-0.15, 1.25)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'point_indicator.png'), dpi=150)
plt.close(fig)
print("  point_indicator.png")


# ── 4. Point Indicators for Nibble Values ──

fig, ax = plt.subplots(figsize=(12, 5))
x = np.linspace(-1, 16, 16000)
S = 1000
eps = 0.1

for k, color in [(0, 'C0'), (5, 'C1'), (10, 'C2'), (15, 'C3')]:
    y = neural_point(x, k, S, eps)
    ax.plot(x, y, color=color, linewidth=2, label=f'k={k}')

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('point(x, k)', fontsize=13)
ax.set_title(f'Point Indicators for Nibble Values  (S={S}, $\\epsilon$={eps})', fontsize=15)
ax.legend(fontsize=12)
ax.set_ylim(-0.1, 1.25)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'point_indicators_nibble.png'), dpi=150)
plt.close(fig)
print("  point_indicators_nibble.png")


# ── 5. SwiGLU Multiplication ──

fig, ax = plt.subplots(figsize=(10, 6))
S = 200
b = 7
x = np.linspace(-2, 15, 2000)

ideal = x * b
swiglu = silu(S * x) * b / S

ax.plot(x, ideal, '--', color='gray', linewidth=2, label=f'Ideal: a * {b}')
ax.plot(x, swiglu, 'b-', linewidth=2.5, label='SwiGLU approx')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('a', fontsize=13)
ax.set_ylabel(f'a * {b}', fontsize=13)
ax.set_title('SwiGLU Multiplication', fontsize=15)
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'swiglu_multiply.png'), dpi=150)
plt.close(fig)
print("  swiglu_multiply.png")


# ── 6. SwiGLU Multiplication (Zoomed Out) ──

fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(-50, 50, 2000)

ideal = x * b
swiglu = silu(S * x) * b / S

ax.plot(x, ideal, '--', color='gray', linewidth=2, label=f'Ideal: a * {b}')
ax.plot(x, swiglu, 'b-', linewidth=2.5, label='SwiGLU approx')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('a', fontsize=13)
ax.set_ylabel(f'a * {b}', fontsize=13)
ax.set_title('SwiGLU Multiplication (Zoomed Out)', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'swiglu_zoomed_out.png'), dpi=150)
plt.close(fig)
print("  swiglu_zoomed_out.png")


# ── 7. Neural Floor Function ──
# floor(x) ≈ silu(S*(x - 1 + eps)) / S + 1 - eps
# Single SiLU term: captures one step at x=1.
# For a full staircase, sum K such terms (one per integer threshold).

def neural_floor_single(x, S, eps):
    r"""Single-threshold floor: silu(S*(x-1+eps))/S + 1 - eps"""
    return silu(S * (x - 1 + eps)) / S + 1 - eps

def neural_floor(x, S, eps, K=16):
    r"""Full staircase: sum of K step terms.

    floor(x) ≈ sum_{k=1}^{K} step(x - k)
    where step(z) ≈ [silu(S*(z+eps)) - silu(S*z)] / (S*eps)
    """
    result = np.zeros_like(x, dtype=np.float64)
    for k in range(1, K + 1):
        result += neural_step(x - k, S, eps)
    return result

fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(-0.5, 5.5, 16000)

# Ideal floor
for k in range(6):
    ax.plot([k, k + 1], [k, k], '--', color='gray', linewidth=2,
            label='Ideal floor(x)' if k == 0 else None)
    if k < 5:
        ax.plot([k + 1, k + 1], [k, k + 1], ':', color='gray', linewidth=1)

configs = [
    (20,  0.5, 'C0', r'$\epsilon$=0.5, S=20'),
    (100, 0.1, 'C2', r'$\epsilon$=0.1, S=100'),
    (1000, 0.01, 'C3', r'$\epsilon$=0.01, S=1000'),
]
for S, eps, color, label in configs:
    y = neural_floor(x, S, eps, K=6)
    ax.plot(x, y, color=color, linewidth=2, label=label)

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('floor(x)', fontsize=13)
ax.set_title('Neural Floor Function', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'floor_function.png'), dpi=150)
plt.close(fig)
print("  floor_function.png")


# ── 8. Neural Modulus (fixed base) ──
# x mod N = x - floor(x/N) * N

def neural_mod(x, N, S, eps):
    r"""x mod N ≈ x - floor(x/N) * N"""
    fl = neural_floor(x / N, S, eps, K=16)
    return x - fl * N

fig, ax = plt.subplots(figsize=(10, 6))
N = 16
x = np.linspace(0, 4 * N, 16000)

# Ideal mod
ax.plot(x, x % N, '--', color='gray', linewidth=2, label=f'Ideal x mod {N}')

configs = [
    (20,  0.5, 'C0', r'$\epsilon$=0.5, S=20'),
    (100, 0.1, 'C2', r'$\epsilon$=0.1, S=100'),
    (1000, 0.01, 'C3', r'$\epsilon$=0.01, S=1000'),
]
for S, eps, color, label in configs:
    y = neural_mod(x, N, S, eps)
    ax.plot(x, y, color=color, linewidth=2, label=label)

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel(f'x mod {N}', fontsize=13)
ax.set_title(f'Neural Modulus (N={N})', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'mod_function.png'), dpi=150)
plt.close(fig)
print("  mod_function.png")


print("\nAll graphs regenerated.")
