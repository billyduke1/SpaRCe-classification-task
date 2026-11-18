"""Step 6: train all read-outs and record learning curves for Figure 4A.

Usage:
    python train.py                         # full run: 5 seeds x 20000 minibatches
    python train.py --seeds 1 --batches 2000   # quick test

Design choices (not specified in the paper):
  - all models train on the same minibatches (20 sequences, fresh noise each time)
  - test set: EVAL_DRAWS noisy presentations of every sequence, noise fixed per seed
  - evaluated every EVAL_EVERY minibatches
  - each seed: new reservoir, new hidden-layer initialisation, new noise; sequences fixed
  - L1/L2 strengths from lambdas.npz (made by tune_lambda.py)
"""

import argparse
import time
import numpy as np

from data import load_odours
from inputs import make_noisy_input
from reservoir import Reservoir
from models import ESNReadout, SpaRCeReadout, HiddenLayerReadout, one_hot

BATCH_SIZE = 20
N_BATCHES = 300000       # minibatches per run (matches the paper's x-axis, read as minibatches)
EVAL_EVERY = 1000        # evaluate on the test set every this many minibatches
EVAL_DRAWS = 5           # noisy presentations of each sequence in the test set
PERCENTILE_DRAWS = 5     # noisy presentations used to compute SpaRCe's percentiles
SOLVED_ACC = 1.0         # accuracy that counts as "solving" the task
DEFAULT_LAM = {"l1": 1e-4, "l2": 1e-4}
INPUT_GAIN = 3.0    # inputs are scaled to lie in 0..INPUT_GAIN (calibrated against Fig. 4A)
ETA_ESN = 2e-5      # learning rate for ESN, ESN+L1, ESN+L2 (grid search; Table 1 values diverge)
ETA_HL = 2e-4       # learning rate for the hidden layers (second eta_W value in Table 1)


def make_reservoir(odours, seed):
    return Reservoir(n_inputs=odours.shape[1], input_scale=INPUT_GAIN / odours.max(), seed=seed)


def reservoir_states(res, odours, sequences, rng, n_draws):
    """Final reservoir states for n_draws noisy presentations of every sequence."""
    return np.concatenate([res.run(make_noisy_input(odours, sequences, rng))
                           for _ in range(n_draws)])


def make_models(V_train, lam_l1, lam_l2, seed):
    n_nodes = V_train.shape[1]
    return {
        "esn": ESNReadout(n_nodes, eta=ETA_ESN),
        "l1": ESNReadout(n_nodes, reg="l1", lam=lam_l1, eta=ETA_ESN),
        "l2": ESNReadout(n_nodes, reg="l2", lam=lam_l2, eta=ETA_ESN),
        "sparce": SpaRCeReadout(V_train, percentile=50),
        "hl": HiddenLayerReadout(n_nodes, n_hidden=100, eta=ETA_HL, seed=seed),
        "sparce90": SpaRCeReadout(V_train, percentile=90),
        "hl30": HiddenLayerReadout(n_nodes, n_hidden=30, eta=ETA_HL, seed=seed),
    }


def train(models, res, odours, sequences, labels, rng, V_test, labels_test,
          n_batches=N_BATCHES, eval_every=EVAL_EVERY, verbose=True):
    """Train every model on the same minibatches, evaluating on a fixed test set.

    Returns a dict of arrays: "batch", "acc_<model>", "rmse_<model>", and
    "active" (SpaRCe's fraction of active nodes, if SpaRCe is one of the models).
    """
    hist = {"batch": [], "active": []}
    for k in models:
        hist[f"acc_{k}"] = []
        hist[f"rmse_{k}"] = []

    def record(b):
        hist["batch"].append(b)
        for k, m in models.items():
            acc, rmse = m.evaluate(V_test, labels_test)
            hist[f"acc_{k}"].append(acc)
            hist[f"rmse_{k}"].append(rmse)
        if "sparce" in models:
            hist["active"].append(models["sparce"].active_fraction(V_test))

    record(0)
    start = time.time()
    for b in range(1, n_batches + 1):
        idx = rng.integers(0, len(sequences), size=BATCH_SIZE)
        V = res.run(make_noisy_input(odours, sequences[idx], rng))
        Y = one_hot(labels[idx])
        for m in models.values():
            m.step(V, Y)

        if b % eval_every == 0:
            record(b)
            if verbose and b % (eval_every * 20) == 0:
                accs = "  ".join(f"{k} {hist[f'acc_{k}'][-1]:.3f}" for k in models)
                print(f"  batch {b:6d} ({time.time() - start:4.0f}s)  {accs}")

    return {key: np.array(values) for key, values in hist.items()}


def run_seed(seed, odours, sequences, labels, lam, n_batches):
    """One complete run: build a reservoir and models for this seed, then train."""
    res = make_reservoir(odours, seed)
    rng = np.random.default_rng(1000 + seed)

    V_train = reservoir_states(res, odours, sequences, rng, PERCENTILE_DRAWS)
    V_test = reservoir_states(res, odours, sequences, rng, EVAL_DRAWS)
    labels_test = np.tile(labels, EVAL_DRAWS)

    models = make_models(V_train, lam["l1"], lam["l2"], seed)
    return train(models, res, odours, sequences, labels, rng, V_test, labels_test, n_batches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all read-outs for Figure 4A.")
    parser.add_argument("--seeds", type=int, default=5, help="number of independent runs")
    parser.add_argument("--batches", type=int, default=N_BATCHES, help="minibatches per run")
    args = parser.parse_args()

    odours, _ = load_odours()
    saved = np.load("sequences.npz")
    sequences, labels = saved["sequences"], saved["labels"]

    try:
        tuned = np.load("lambdas.npz")
        lam = {"l1": float(tuned["l1"]), "l2": float(tuned["l2"])}
        print("Using tuned lambdas:", lam)
    except FileNotFoundError:
        lam = DEFAULT_LAM
        print("lambdas.npz not found - using defaults", lam, "(run tune_lambda.py first)")

    runs = []
    for seed in range(args.seeds):
        print(f"\nSeed {seed}  ({args.batches} minibatches)")
        runs.append(run_seed(seed, odours, sequences, labels, lam, args.batches))

    # stack runs: each array becomes shape (n_seeds, n_evaluations)
    results = {key: np.stack([run[key] for run in runs]) for key in runs[0]}
    np.savez("results.npz", lam_l1=lam["l1"], lam_l2=lam["l2"], **results)
    print("\nSaved results.npz")