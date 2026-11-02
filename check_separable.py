"""Sanity check: is the task learnable with this reservoir?

Fits the best possible linear read-out directly (ridge regression) instead of
by gradient descent, on raw V and on SpaRCe's thresholded x (fixed P50).
"""

import numpy as np
from data import load_odours
from inputs import make_noisy_input
from reservoir import Reservoir
from models import one_hot, SpaRCeReadout


def ridge_accuracy(X_train, y_train, X_test, y_test, lam):
    """Best linear read-out (with bias) by ridge regression; returns test accuracy."""
    add_bias = lambda X: np.hstack([X, np.ones((len(X), 1))])
    A, B = add_bias(X_train), add_bias(X_test)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ one_hot(y_train))
    return np.mean(np.argmax(B @ W, axis=1) == y_test)


odours, _ = load_odours()
saved = np.load("sequences.npz")
sequences, labels = saved["sequences"], saved["labels"]
res = Reservoir(n_inputs=odours.shape[1], input_scale=1.0 / odours.max(), seed=0)
rng = np.random.default_rng(4)

n_draws = 20
print(f"Running the reservoir on {n_draws} noisy presentations of each sequence...")
V_train = np.concatenate([res.run(make_noisy_input(odours, sequences, rng)) for _ in range(n_draws)])
y_train = np.tile(labels, n_draws)
V_test = res.run(make_noisy_input(odours, sequences, rng))     # fresh noise
y_test = labels

sparce = SpaRCeReadout(V_train, percentile=50)
X_train, X_test = sparce.thresholded(V_train), sparce.thresholded(V_test)

print(f"\n{'lambda':>8s} {'raw V':>8s} {'SpaRCe x':>9s}")
for lam in [1e-6, 1e-4, 1e-2, 1e0]:
    acc_V = ridge_accuracy(V_train, y_train, V_test, y_test, lam)
    acc_x = ridge_accuracy(X_train, y_train, X_test, y_test, lam)
    print(f"{lam:8.0e} {acc_V:8.3f} {acc_x:9.3f}")