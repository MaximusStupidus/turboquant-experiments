"""
Pedagogical demo of the Johnson-Lindenstrauss lemma.

Claim being demonstrated:
    If you take vectors in a high-dimensional space R^d and multiply each by
    the *same* random Gaussian projection matrix R (shape d x k) with k << d,
    the pairwise dot products of the projected vectors approximate the pairwise
    dot products of the original vectors. Error shrinks as k grows, roughly
    like 1/sqrt(k). The original dimension d barely matters.

This is the property TurboQuant exploits for KV cache compression: it lets you
shrink K and V vectors (which only need to preserve dot products with Q) from
many dimensions down to far fewer, with provable bounded error.

Run:  python3 jl_demo.py
Output: jl_demo.png
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# --- setup ---
d = 1024            # original dimension (typical of K/V vectors per head, ish)
N = 200             # number of vectors we'll generate
ks = [4, 8, 16, 32, 64, 128, 256, 512]   # projection dimensions to test

# Generate N random vectors in R^d.
# Each entry is N(0, 1/d), so each vector has expected squared norm = 1.
X = rng.standard_normal((N, d)) / np.sqrt(d)

def pairwise_sq_dists(M):
    """Pairwise squared L2 distances between rows of M, returned as a flat array."""
    sq_norms = (M * M).sum(axis=1)
    G = M @ M.T
    D = sq_norms[:, None] + sq_norms[None, :] - 2 * G
    iu = np.triu_indices(M.shape[0], k=1)
    return D[iu]

# Ground truth: true pairwise squared distances in d dimensions.
true_d2 = pairwise_sq_dists(X)

# --- sweep over projection dimensions k, measure relative error on distances ---
median_errors = []
for k in ks:
    # JL projection matrix: Gaussian entries with the standard JL scaling.
    R = rng.standard_normal((d, k)) / np.sqrt(k)
    Xp = X @ R                                    # projected vectors, shape (N, k)
    proj_d2 = pairwise_sq_dists(Xp)
    rel_err = np.abs(proj_d2 - true_d2) / true_d2
    median_errors.append(np.median(rel_err))

# --- detailed scatter at one chosen k, for the "see it on the y=x line" plot ---
k_demo = 128
R_demo = rng.standard_normal((d, k_demo)) / np.sqrt(k_demo)
Xp_demo = X @ R_demo
proj_demo = pairwise_sq_dists(Xp_demo)

# --- plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(true_d2, proj_demo, s=8, alpha=0.4, label='vector pairs')
lo = float(min(true_d2.min(), proj_demo.min())) * 0.9
hi = float(max(true_d2.max(), proj_demo.max())) * 1.05
ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='y = x (perfect)')
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel(f'True squared distance (d = {d})')
ax.set_ylabel(f'Projected squared distance (k = {k_demo})')
ax.set_title(f'JL: true vs projected squared distances\n{N} vectors, {len(true_d2)} pairs')
ax.legend(loc='upper left')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.loglog(ks, median_errors, 'o-', linewidth=2, markersize=8, label='measured')
# Theoretical 1/sqrt(k) reference, anchored at first data point
ref = median_errors[0] * np.sqrt(ks[0]) / np.sqrt(np.array(ks))
ax.loglog(ks, ref, 'r--', linewidth=1.5, label=r'$\propto 1/\sqrt{k}$')
ax.set_xlabel('Projection dimension k')
ax.set_ylabel('Median relative error of squared distances')
ax.set_title(f'Error decay as k grows (d = {d} fixed)')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('jl_demo.png', dpi=120)

# --- print numerical summary ---
print(f'd = {d}, N = {N}, num pairs = {len(true_d2)}')
print()
print(f'{"k":>6}  {"median rel err":>16}  {"compression vs d":>20}')
print('-' * 48)
for k, e in zip(ks, median_errors):
    print(f'{k:>6}  {e:>16.4f}  {d/k:>19.1f}x')
print()
print('Saved plot: jl_demo.png')
