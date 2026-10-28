"""build the odour sequences"""

import numpy as np

ODOURS_PER_SEQUENCE = 3
NUM_CLASSES = 2

def _generate_batch(rng,n_odours):
    # initial sequence - repeats allowed
    initial = rng.integers(0, n_odours, size=(NUM_CLASSES, ODOURS_PER_SEQUENCE))

    # have to be distinct from odours in initial sequence
    dissimilar_odours = np.setdiff1d(np.arange(n_odours), initial)
    new = rng.choice(dissimilar_odours, size=(ODOURS_PER_SEQUENCE,NUM_CLASSES), replace=False)

    sequences = [i.copy() for i in initial]
    labels = list(range(NUM_CLASSES))

    for pos in reversed(range(ODOURS_PER_SEQUENCE)): # last position first as in paper
        for i in range(NUM_CLASSES):
            variant = initial[i].copy()
            variant[pos] = new[pos,i]
            sequences.append(variant)
            labels.append((i+1)%NUM_CLASSES) # opposite label to initial

    return sequences, labels


def make_sequences(n_odours, n_sequences=192, seed=0):
    # build the full set of labelled sequences

    per_round = NUM_CLASSES * (ODOURS_PER_SEQUENCE + 1) # initial + variants
    if n_sequences % per_round != 0:
        raise ValueError(f"n_sequences must be a multiple of {per_round}")

    rng = np.random.default_rng(seed)
    seen = set() # initialise a set of seen sequences
    sequences, labels = [], []

    while len(sequences) < n_sequences:
        batch, batch_labels = _generate_batch(rng,n_odours)
        keys = [tuple(seq) for seq in batch]
        # reject the batch if any sequence has been seen before
        if len(set(keys)) < len(keys) or len(set(keys).intersection(seen)) > 0:
            continue
        seen.update(keys)
        sequences.extend(batch)
        labels.extend(batch_labels)

    return np.array(sequences), np.array(labels)


if __name__ == "__main__":
    from data import load_odours

    odours, odour_names = load_odours()
    sequences, labels = make_sequences(len(odours), n_sequences=192, seed=0)

    print("Sequences shape:", sequences.shape)
    print("Labels shape:", labels.shape)
    print("Class counts:", np.bincount(labels))
    print("Unique sequences:", len(np.unique(sequences, axis=0)))
    print("distinct odours used:", len(np.unique(sequences)), "of", len(odours))

    print("\nFirst batch of sequences and labels:")
    for s,l in zip(sequences[:NUM_CLASSES*(ODOURS_PER_SEQUENCE+1)], labels[:NUM_CLASSES*(ODOURS_PER_SEQUENCE+1)]):
        print("  " + " -> ".join(odour_names[i] for i in s) + f"  (label {l})")

    np.savez("sequences.npz", sequences=sequences, labels=labels)
    print("\nSaved sequences and labels to sequences.npz")