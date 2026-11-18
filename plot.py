"""Draw Figure 4A (accuracy and RMSE over training) from results.npz."""

import numpy as np
import matplotlib.pyplot as plt
from train import SOLVED_ACC, BATCH_SIZE

# key: (label, colour, line style) - colours follow the paper (ESN black, SpaRCe red, HL blue)
STYLE = {
    "esn":      ("ESN", "black", "-"),
    "l1":       ("ESN + L1", "tab:green", "-"),
    "hl30":     ("ESN + HL, N_h = 30", "lightskyblue", "-"),
    "hl":       ("ESN + HL, N_h = 100", "tab:blue", "-"),
    "sparce":   ("ESN + SpaRCe, P_n = 50", "tab:red", "-"),
    "sparce90": ("ESN + SpaRCe, P_n = 90", "darkred", "-"),
}

r = np.load("results.npz")
batch = r["batch"][0]
n_seeds = r["acc_esn"].shape[0]

fig, (ax_acc, ax_rmse) = plt.subplots(1, 2, figsize=(13, 4.8))

print(f"{'model':26s} {'final acc':>14s} {'best acc':>9s} {'solved':>7s} {'solve batch':>12s}")
for key, (label, colour, style) in STYLE.items():
    acc, rmse = r[f"acc_{key}"], r[f"rmse_{key}"]          # shape (n_seeds, n_evaluations)

    for ax, data in [(ax_acc, acc), (ax_rmse, rmse)]:
        mean, sd = data.mean(axis=0), data.std(axis=0)
        ax.plot(batch, mean, color=colour, linestyle=style, label=label)
        ax.fill_between(batch, mean - sd, mean + sd, color=colour, alpha=0.15, linewidth=0)

    # dots: the first evaluation at which each seed solves the task
    solve_batches = []
    for s in range(n_seeds):
        hits = np.flatnonzero(acc[s] >= SOLVED_ACC)
        if len(hits):
            i = hits[0]
            solve_batches.append(batch[i])
            ax_acc.plot(batch[i], acc[s, i], "o", color=colour, markersize=6)
            ax_rmse.plot(batch[i], rmse[s, i], "o", color=colour, markersize=6)

    final = acc[:, -10:].mean(axis=1)                       # last 10 evaluations, per seed
    solve_text = f"{np.mean(solve_batches):.0f}" if solve_batches else "-"
    print(f"{label:26s} {final.mean():6.3f} ± {final.std():.3f} {acc.max():9.3f} "
          f"{len(solve_batches):3d}/{n_seeds} {solve_text:>12s}")

ax_acc.set_ylabel("Accuracy")
ax_acc.set_ylim(0.45, 1.02)
ax_rmse.set_ylabel("Root mean square error")
ax_rmse.set_ylim(0, 1)
for ax in (ax_acc, ax_rmse):
    ax.set_xlabel(f"Minibatch ({BATCH_SIZE} samples each)")
    ax.grid(alpha=0.3)
ax_acc.legend(loc="center right")
fig.suptitle(f"192 sequences, mean ± SD over {n_seeds} seeds; dots = task solved "
             f"(accuracy ≥ {SOLVED_ACC:.0%})")
fig.tight_layout()
fig.savefig("figure4a.png", dpi=200)
print("\nSaved figure4a.png")
plt.show()