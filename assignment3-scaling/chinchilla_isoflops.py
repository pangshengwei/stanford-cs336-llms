import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
with open("data/isoflops_curves.json") as f:
    runs = json.load(f)

# Group by compute budget
from collections import defaultdict
budgets = defaultdict(list)
for run in runs:
    budgets[run["compute_budget"]].append(run)

# Find optimal N and D for each compute budget
C_vals = []
N_opt_vals = []
D_opt_vals = []

for C in sorted(budgets.keys()):
    best = min(budgets[C], key=lambda r: r["final_loss"])
    N_opt = best["parameters"]
    D_opt = C / (6 * N_opt)
    C_vals.append(C)
    N_opt_vals.append(N_opt)
    D_opt_vals.append(D_opt)
    print(f"C={C:.0e}, N_opt={N_opt:.2e}, D_opt={D_opt:.2e}, loss={best['final_loss']:.4f}")

C_vals = np.array(C_vals)
N_opt_vals = np.array(N_opt_vals)
D_opt_vals = np.array(D_opt_vals)

# Fit power law: N_opt = k * C^a  (in log space: log(N) = log(k) + a*log(C))
def power_law(x, k, a):
    return k * x**a

popt_N, _ = curve_fit(power_law, C_vals, N_opt_vals, p0=[1, 0.5])
popt_D, _ = curve_fit(power_law, C_vals, D_opt_vals, p0=[1, 0.5])

k_N, a_N = popt_N
k_D, a_D = popt_D
print(f"\nN_opt = {k_N:.4e} * C^{a_N:.4f}")
print(f"D_opt = {k_D:.4e} * C^{a_D:.4f}")

# Extrapolate
for C_target in [1e23, 1e24]:
    N_pred = power_law(C_target, *popt_N)
    D_pred = power_law(C_target, *popt_D)
    print(f"\nC = {C_target:.0e}:")
    print(f"  Predicted N_opt = {N_pred:.2e}")
    print(f"  Predicted D_opt = {D_pred:.2e}")

# Plot 1: N_opt vs C
C_plot = np.logspace(np.log10(C_vals.min()), 24, 200)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(C_vals, N_opt_vals, color='blue', zorder=5, label='Data points')
ax.plot(C_plot, power_law(C_plot, *popt_N), 'r--',
        label=f'$N_{{opt}} = {k_N:.2e} \\cdot C^{{{a_N:.3f}}}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Compute Budget C (FLOPs)')
ax.set_ylabel('Optimal Model Size $N_{opt}$ (parameters)')
ax.set_title('Compute-Optimal Model Size (IsoFLOPs)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: D_opt vs C
ax = axes[1]
ax.scatter(C_vals, D_opt_vals, color='green', zorder=5, label='Data points')
ax.plot(C_plot, power_law(C_plot, *popt_D), 'r--',
        label=f'$D_{{opt}} = {k_D:.2e} \\cdot C^{{{a_D:.3f}}}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Compute Budget C (FLOPs)')
ax.set_ylabel('Optimal Dataset Size $D_{opt}$ (tokens)')
ax.set_title('Compute-Optimal Dataset Size (IsoFLOPs)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("isoflops_scaling_laws.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to isoflops_scaling_laws.png")