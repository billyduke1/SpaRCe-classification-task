"""the echo state network reservoir model"""

import numpy as np

N = 1000 # number of reservoir neurons
ALPHA = 0.1 # leak rate
RHO = 0.95 # spectral radius
GAMMA = 1.0 # input scaling
P_ER = 0.001 # Erdos-Renyi connection probability for two neurons to be connected

def relu(x):
    return np.maximum(0.0, x)


class Reservoir:
    def __init__(self, n_inputs, input_scale, seed=0,
                 n=N, alpha=ALPHA, rho=RHO, gamma=GAMMA, p_er=P_ER):
        rng = np.random.default_rng(seed)
        self.n = n
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.input_scale = input_scale

        # create the recurrent reservoir weight matrix - sparse and random.
        # at p_er = 0.001 (~1 connection per node) a random draw can have no loops,
        # giving spectral radius 0, so redraw until it has some.

        for attempt in range(100):
            mask = rng.random((n, n)) < p_er
            w = rng.standard_normal((n, n)) * mask
            radius = np.max(np.abs(np.linalg.eigvals(w)))
            if radius > 1e-6:
                break
        else:
            raise ValueError("Could not build a reservoir with nonzero spectral radius")
        self.w = w / radius
        self.redraws = attempt # how many networks were rejected

        # input weight matrix - lognormal, one row per reservoir node
        W_in = rng.lognormal(mean=0.0, sigma=1.0, size=(n, n_inputs))
        self.W_in = W_in / W_in.sum(axis=1).mean()


    def run(self, S, return_trajectory=False):
        """run the reservoir on a sequence of inputs S
        S is shape (n_sequences, n_steps, n_inputs)
        returns the final reservoir state for each sequence, shape (n_sequences, n)
        if return_trajectory is True, returns every step instead, shape (n_sequences, n_steps, n)
        """

        batch, n_steps, n_inputs = S.shape
        drive_in = self.gamma * (S * self.input_scale) @ self.W_in.T # shape (batch, n_steps, n)

        V = np.zeros((batch, self.n)) # initial reservoir state
        states = [] # stores V at every step, only used if return_trajectory is True

        for t in range(n_steps):
            V = (1 - self.alpha) * V + self.alpha * relu(drive_in[:, t] + self.rho * V @ self.w.T)
            if return_trajectory:
                states.append(V)

        if return_trajectory:
            return np.stack(states, axis=1) # list of (batch, n) -> (batch, n_steps, n)
        return V 



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data import load_odours
    from inputs import make_noisy_input, make_clean_inputs, STEPS_PER_ODOUR

    odours, odour_names = load_odours()
    saved = np.load("sequences.npz")
    sequences, labels = saved["sequences"], saved["labels"]

    res = Reservoir(n_inputs=odours.shape[1], input_scale=1.0 / odours.max(), seed=0) # scale inputs to [0,1] range

    # test - are weights what we want?
    eigenvalues = np.linalg.eigvals(res.rho * res.w) # rho scales the spectral radius from 1 to 0.95
    print("Recurrent connections:", np.count_nonzero(res.w), f"(expected ~{N * N * P_ER:.0f})")
    print(f"Spectral radius of rho*W: {np.abs(eigenvalues).max():.3f}  (want {RHO})")
    print("Input weight shape:", res.W_in.shape)

    # run the reservoir on all 192 sequences
    rng = np.random.default_rng(2)
    V_clean = res.run(make_clean_inputs(odours, sequences))
    V_noisy = res.run(make_noisy_input(odours, sequences, rng))
    V_noisy2 = res.run(make_noisy_input(odours, sequences, rng))

    print("\nV(T) shape:", V_noisy.shape)
    print(f"V(T) values: mean {V_noisy.mean():.4f}, max {V_noisy.max():.4f}")
    print(f"Fraction of V(T) entries that are exactly 0: {(V_noisy == 0).mean():.3f}")
    print(f"Nodes silent for every sequence: {(V_noisy.max(axis=0) == 0).sum()} of {N}")

    # can reservoir distinguish sequences?

    def correlation(a, b):
        return np.corrcoef(a, b)[0, 1] # correlation between two 1D arrays

    n_rounds = len(sequences) // 8
    same_seq = np.mean([correlation(V_noisy[i], V_noisy2[i]) for i in range(len(sequences))])
    twin_last = np.mean([correlation(V_clean[8 * k], V_clean[8 * k + 2]) for k in range(n_rounds)])
    twin_first = np.mean([correlation(V_clean[8 * k], V_clean[8 * k + 6]) for k in range(n_rounds)])
    pairs = rng.integers(0, len(sequences), size=(300, 2))
    unrelated = np.mean([correlation(V_clean[i], V_clean[j]) for i, j in pairs if i != j])

    print("\nAverage correlation between final states V(T):")
    print(f"  same sequence, two noise draws : {same_seq:.3f}")
    print(f"  twins differing in LAST odour  : {twin_last:.3f}")
    print(f"  twins differing in FIRST odour : {twin_first:.3f}")
    print(f"  unrelated sequences            : {unrelated:.3f}")

        # distance check: are twins further apart than noise moves a sequence?
    noise_dist = np.mean(np.linalg.norm(V_noisy - V_noisy2, axis=1))
    last_dist = np.mean([np.linalg.norm(V_clean[8 * k] - V_clean[8 * k + 2]) for k in range(n_rounds)])
    first_dist = np.mean([np.linalg.norm(V_clean[8 * k] - V_clean[8 * k + 6]) for k in range(n_rounds)])

    print("\nAverage distance between final states V(T):")
    print(f"  same sequence, two noise draws : {noise_dist:.3f}")
    print(f"  twins differing in LAST odour  : {last_dist:.3f}  ({last_dist / noise_dist:.1f}x noise)")
    print(f"  twins differing in FIRST odour : {first_dist:.3f}  ({first_dist / noise_dist:.1f}x noise)")

    # graphical summary of reservoir behaviour
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # eigenvalues inside the unit circle
    ax = axes[0, 0]
    angle = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(angle), np.sin(angle), "k--", linewidth=1, label="unit circle")
    ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=4)
    ax.set_aspect("equal")
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.set_title("Eigenvalues of rho*W")
    ax.legend()

    # a few nodes over time for one sequence
    ax = axes[0, 1]
    traj = res.run(make_clean_inputs(odours, sequences[:1]), return_trajectory=True)[0]
    active = np.flatnonzero(traj[-1] > 0)
    show = rng.choice(active, size=min(10, len(active)), replace=False)
    ax.plot(traj[:, show])
    for boundary in (STEPS_PER_ODOUR, 2 * STEPS_PER_ODOUR):
        ax.axvline(boundary - 0.5, color="grey", linestyle=":")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Activity V")
    ax.set_title("10 nodes during sequence 0 (dotted = odour switch)")

    # distribution of final activities
    ax = axes[1, 0]
    ax.hist(V_noisy[V_noisy > 0].ravel(), bins=100)
    ax.set_yscale("log")
    ax.set_xlabel("V(T) value (nonzero entries)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Distribution of final activities")

    # fading memory: sequence, then silence
    ax = axes[1, 1]
    silence = np.zeros((1, 200, odours.shape[1]))
    S_ext = np.concatenate([make_clean_inputs(odours, sequences[:1]), silence], axis=1)
    traj_ext = res.run(S_ext, return_trajectory=True)[0]
    ax.plot(traj_ext.mean(axis=1))
    ax.axvline(3 * STEPS_PER_ODOUR - 0.5, color="grey", linestyle=":", label="input stops")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean activity")
    ax.set_title("Activity fades after the input stops")
    ax.legend()

    fig.tight_layout()
    fig.savefig("step4_reservoir.png", dpi=150)
    print("\nSaved figure to step4_reservoir.png")
    plt.show()