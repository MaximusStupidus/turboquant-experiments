"""
Pedagogical demo: random projection flattens outlier-heavy distributions.

This is the *second* property of random projection that TurboQuant exploits
(the first being dot-product preservation, demonstrated in jl_demo.py).

Setup:
    Generate vectors that mimic the empirical structure of LLM K/V vectors:
    most channels are small Gaussian noise, but a few "outlier channels"
    have values 30-50x bigger than typical. Naive bit-quantization on this
    data fails because the outliers eat the bucket budget.

Then:
    Apply a random Gaussian projection. Show the histogram of values
    BEFORE projection (peaky with huge outliers in a few dims) vs AFTER
    projection (smooth, near-Gaussian, no concentrated outliers).

The visual takeaway: the outlier *energy* is still in the data
(dot products / norms are preserved by JL), but it's been spread evenly
across all the projected dimensions instead of being concentrated in a few.
A bit-quantizer applied to the post-projection data sees a friendly
distribution and uses its bucket budget efficiently.

Run:  python3 outlier_demo.py
Output: outlier_demo.png
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# --- setup ---
d = 1024                       # original dimension
N = 500                        # number of vectors
k = 256                        # projection dimension
outlier_channels = [47, 312, 800]   # which dims are "outliers"
outlier_scale = 40.0           # how big the outlier values are

# Generate baseline Gaussian noise
X = rng.standard_normal((N, d))

# Inject outliers: in the chosen channels, multiply values by a huge factor
for c in outlier_channels:
    X[:, c] *= outlier_scale

# --- random projection ---
R = rng.standard_normal((d, k)) / np.sqrt(k)
Xp = X @ R                     # shape (N, k)

# --- compare distributions ---
orig_values = X.flatten()
proj_values = Xp.flatten()

# --- plot ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Determine a shared x-range that covers both distributions
xrange = max(np.abs(orig_values).max(), np.abs(proj_values).max()) * 1.05

ax = axes[0]
ax.hist(orig_values, bins=200, range=(-xrange, xrange),
        color='steelblue', edgecolor='none')
ax.set_yscale('log')
ax.set_xlabel('Value')
ax.set_ylabel('Count (log scale)')
ax.set_title(f'BEFORE projection: original {d}-dim vectors\n'
             f'{len(outlier_channels)} outlier channels at ~±{outlier_scale}')
ax.axvline(0, color='k', linewidth=0.5, alpha=0.4)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(proj_values, bins=200, range=(-xrange, xrange),
        color='seagreen', edgecolor='none')
ax.set_yscale('log')
ax.set_xlabel('Value')
ax.set_ylabel('Count (log scale)')
ax.set_title(f'AFTER projection: same vectors in {k} dims\n'
             f'(outliers smeared across all dimensions)')
ax.axvline(0, color='k', linewidth=0.5, alpha=0.4)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outlier_demo.png', dpi=120)

# --- print numerical summary ---
def summary(name, v):
    print(f'{name:>22}  '
          f'mean={v.mean():+.3f}  '
          f'std={v.std():.3f}  '
          f'min={v.min():+.2f}  '
          f'max={v.max():+.2f}  '
          f'p99.9={np.quantile(np.abs(v), 0.999):.2f}')

print(f'd={d}  k={k}  N={N}  outlier_channels={outlier_channels}  outlier_scale={outlier_scale}')
print()
summary('original', orig_values)
summary('projected', proj_values)
print()
print('Saved plot: outlier_demo.png')
