"""Choose the L1 and L2 penalty strengths, which the paper does not give.

Trains ESN+L1 and ESN+L2 read-outs for several lambdas on a separate reservoir
(seed 100, not used in the main runs) and keeps the lambda with the best accuracy
on held-out noisy presentations.
"""

import numpy as np
from data import load_odours
from reservoir import Reservoir
from models import ESNReadout
from train import train, reservoir_states, make_reservoir, EVAL_DRAWS, ETA_ESN

LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
TUNE_SEED = 100
N_BATCHES = 10000

odours, _ = load_odours()
saved = np.load("sequences.npz")
sequences, labels = saved["sequences"], saved["labels"]

res = make_reservoir(odours, TUNE_SEED)
rng = np.random.default_rng(TUNE_SEED)
V_val = reservoir_states(res, odours, sequences, rng, EVAL_DRAWS)
labels_val = np.tile(labels, EVAL_DRAWS)
n_nodes = V_val.shape[1]

models = {}
for lam in LAMBDAS:
    models[f"l1_{lam:.0e}"] = ESNReadout(n_nodes, reg="l1", lam=lam, eta=ETA_ESN)
    models[f"l2_{lam:.0e}"] = ESNReadout(n_nodes, reg="l2", lam=lam, eta=ETA_ESN)

print(f"Training {len(models)} models for {N_BATCHES} minibatches (takes a minute or two)...")
hist = train(models, res, odours, sequences, labels, rng, V_val, labels_val,
             n_batches=N_BATCHES, verbose=False)

best = {}
for reg in ["l1", "l2"]:
    print(f"\n{reg.upper()}:   lambda   validation accuracy")
    scores = {}
    for lam in LAMBDAS:
        scores[lam] = hist[f"acc_{reg}_{lam:.0e}"][-10:].mean()   # average of the last 10 evaluations
        print(f"        {lam:8.0e}   {scores[lam]:.3f}")
    best[reg] = max(scores, key=scores.get)
    print(f"  best: {best[reg]:.0e}")

np.savez("lambdas.npz", **best)
print("\nSaved lambdas.npz")