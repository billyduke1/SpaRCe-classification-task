"""turn sequences into noisy input signals"""

import numpy as np

# constants taken from reference paper

DELTA = 0.01 # simulation time step (seconds)
ODOUR_SIGNAL_DURATION = 0.1 # duration of each odour signal (seconds)
SIGMA_S = 0.3 # multiplicative noise strength as seen in reference paper
STEPS_PER_ODOUR = round(ODOUR_SIGNAL_DURATION / DELTA) # number of time steps per odour signal


def make_clean_inputs(odours,sequences):
    per_odour = odours[sequences] # replaces odour indices with their corresponding glomerular response patterns
    return np.repeat(per_odour, STEPS_PER_ODOUR, axis=1) # repeat each odour's response pattern for the number of time steps per odour


def make_noisy_input(odours, sequences, rng, sigma_s=SIGMA_S):
    signals = make_clean_inputs(odours, sequences)
    xi = rng.standard_normal(size=signals.shape) # shape (n_sequences, n_odours_per_sequence * n_steps_per_odour, n_glomeruli)
    return signals * sigma_s * xi + signals # paper's formula for multiplicative noise, with the clean signal added back in to ensure the mean is preserved


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data import load_odours

    odours, odour_names = load_odours()
    saved = np.load("sequences.npz")
    sequences, labels = saved["sequences"], saved["labels"]

    rng = np.random.default_rng(1)

    # one sequence, clean then with noise
    first = sequences[:1]
    clean = make_clean_inputs(odours, first)
    noisy_first = make_noisy_input(odours, first, rng)
    noisy_second = make_noisy_input(odours, first, rng)

    print("Steps per odour:", STEPS_PER_ODOUR)
    print("Clean input shape:", clean.shape)
    print("Noisy input shape:", noisy_first.shape)
    print("Odour held constant within a block?",
          np.allclose(clean[0, 0], clean[0, STEPS_PER_ODOUR - 1]))
    print("Two noisy presentations identical?", np.allclose(noisy_first, noisy_second))

    # check noise level across all sequences

    clean_signals = make_clean_inputs(odours, sequences)
    noisy_signals = make_noisy_input(odours, sequences, rng)
    nonzero = clean_signals != 0
    relative = (noisy_signals[nonzero] - clean_signals[nonzero]) / clean_signals[nonzero]
    print(f"Relative noise: mean {relative.mean():.3f}, std {relative.std():.3f}  (want ~0 and ~{SIGMA_S})")
    print(f"Fraction of inputs pushed below zero: {(noisy_signals < 0).mean():.5f}")

    # plot of what the network sees
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, grid, title in [(axes[0], clean[0], "Clean input"),
                            (axes[1], noisy_first[0], "Noisy input 1")]:
        im = ax.imshow(grid.T, aspect="auto", cmap="viridis", origin="lower")
        for boundary in (STEPS_PER_ODOUR, 2 * STEPS_PER_ODOUR):
            ax.axvline(boundary - 0.5, color="white", lw=1)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Glomerulus")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Firing rate (Hz)")

    channel = np.argmax(clean[0].max(axis=0)) # glomerulus with largest response
    axes[2].plot(clean[0, :, channel], "k-", lw=2, label="Clean")
    axes[2].plot(noisy_first[0, :, channel], ".-", label="Noisy 1")
    axes[2].plot(noisy_second[0, :, channel], ".-", label="Noisy 2")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Firing rate (Hz)")
    axes[2].set_title(f"Glomerulus (#{channel}) response")
    axes[2].legend()

    names = " -> ".join([odour_names[i] for i in sequences[0]])
    fig.suptitle(f"Sequence: {names}  (label {labels[0]})", fontsize=16)
    fig.tight_layout()
    fig.savefig("example_inputs.png", dpi=150)
    print("Saved example_inputs.png")
    plt.show()