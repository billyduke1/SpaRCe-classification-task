"""The read-outs compared in Figure 4A.

Every model maps the reservoir's final state V(T), shape (batch, 1000), to 2 outputs y
and is trained by gradient descent on the mean squared error (Eq. 3 of the paper):
    E = 1/2 * sum_j (target_j - y_j)^2, averaged over the minibatch.

Models:
  ESNReadout          y = W V + b                 (optionally + L1 or L2 penalty on W)
  SpaRCeReadout       y = W x + b, x = sign(V) relu(|V| - theta), theta = P_n + theta_tilde
  HiddenLayerReadout  y = W2 relu(W1 V + b1) + b2  (backpropagation)


"""

import numpy as np
from reservoir import relu

N_CLASSES = 2
ETA_W = 2e-3       # learning rate for weights (Table 1)
ETA_THETA = 2e-4   # learning rate for SpaRCe thresholds (Table 1)


def one_hot(labels, n_classes=N_CLASSES):
    """Turn labels like [0, 1, 1] into targets [[1, 0], [0, 1], [0, 1]]."""
    Y = np.zeros((len(labels), n_classes))
    Y[np.arange(len(labels)), labels] = 1.0
    return Y


class Readout:
    """Machinery shared by all read-outs.

    Each subclass must define:
      self.params : dict of name -> array   (the trainable parameters)
      self.lrs    : dict of name -> learning rate
      forward(V)  : outputs y, shape (batch, N_CLASSES)
      grads(V, Y) : dict of name -> dE/dparam (same shapes as self.params)
    """

    def penalty(self):
        return 0.0   # no regularisation unless a subclass says otherwise

    def loss(self, V, Y):
        err = Y - self.forward(V)
        return 0.5 * np.mean(np.sum(err ** 2, axis=1)) + self.penalty()

    def step(self, V, Y):
        grads = self.grads(V, Y)            # every gradient uses the current parameters...
        for name, g in grads.items():       # ...then all parameters are updated together
            self.params[name] -= self.lrs[name] * g

    def evaluate(self, V, labels):
        y = self.forward(V)
        accuracy = np.mean(np.argmax(y, axis=1) == labels)
        rmse = np.sqrt(np.mean((one_hot(labels) - y) ** 2))
        return accuracy, rmse

    @property
    def n_params(self):
        return sum(p.size for p in self.params.values())


class ESNReadout(Readout):
    """Standard ESN read-out: y = W V + b, with an optional L1 or L2 penalty on W."""

    def __init__(self, n_nodes, reg=None, lam=0.0, eta=ETA_W):
        self.params = {"W": np.zeros((N_CLASSES, n_nodes)),
                       "b": np.zeros(N_CLASSES)}
        self.lrs = {"W": eta, "b": eta}
        self.reg = reg      # None, "l1" or "l2"
        self.lam = lam

    def forward(self, V):
        return V @ self.params["W"].T + self.params["b"]

    def penalty(self):
        W = self.params["W"]
        if self.reg == "l2":
            return 0.5 * self.lam * np.sum(W ** 2)
        if self.reg == "l1":
            return self.lam * np.sum(np.abs(W))
        return 0.0

    def grads(self, V, Y):
        W = self.params["W"]
        err = Y - self.forward(V)                  # (batch, classes)
        gW = -(err.T @ V) / len(V)                 # (classes, nodes)
        gb = -err.mean(axis=0)                     # (classes,)
        if self.reg == "l2":
            gW = gW + self.lam * W
        if self.reg == "l1":
            gW = gW + self.lam * np.sign(W)
        return {"W": gW, "b": gb}


class SpaRCeReadout(Readout):
    """SpaRCe: y = W x + b, x = sign(V) relu(|V| - theta), theta = P_n(|V|) + theta_tilde."""

    def __init__(self, V_train, percentile=50, eta_w=ETA_W, eta_theta=ETA_THETA):
        n_nodes = V_train.shape[1]
        # fixed part of the threshold: each node's own percentile over the training data
        self.P_n = np.percentile(np.abs(V_train), percentile, axis=0)   # (nodes,)
        self.params = {"W": np.zeros((N_CLASSES, n_nodes)),
                       "b": np.zeros(N_CLASSES),
                       "theta_tilde": np.zeros(n_nodes)}             # learned part
        self.lrs = {"W": eta_w, "b": eta_w, "theta_tilde": eta_theta}

    def thresholded(self, V):
        theta = self.P_n + self.params["theta_tilde"]
        return np.sign(V) * relu(np.abs(V) - theta)                  # Eq. 4

    def active_fraction(self, V):
        return np.mean(self.thresholded(V) != 0)

    def forward(self, V):
        return self.thresholded(V) @ self.params["W"].T + self.params["b"]

    def grads(self, V, Y):
        W = self.params["W"]
        x = self.thresholded(V)
        err = Y - (x @ W.T + self.params["b"])
        gW = -(err.T @ x) / len(V)
        gb = -err.mean(axis=0)
        # dE/dtheta_k = sum_j err_j * W_jk * sign(x_k), averaged over the batch (Eq. B3)
        gtheta = np.mean((err @ W) * np.sign(x), axis=0)
        return {"W": gW, "b": gb, "theta_tilde": gtheta}


class HiddenLayerReadout(Readout):
    """Hidden layer: h = relu(W1 V + b1), y = W2 h + b2, trained by backpropagation."""

    def __init__(self, n_nodes, n_hidden=100, eta=ETA_W, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.standard_normal((n_hidden, n_nodes)) * np.sqrt(2.0 / n_nodes),
            "b1": np.zeros(n_hidden),
            "W2": rng.standard_normal((N_CLASSES, n_hidden)) * np.sqrt(1.0 / n_hidden),
            "b2": np.zeros(N_CLASSES),
        }
        self.lrs = {name: eta for name in self.params}

    def _hidden(self, V):
        pre = V @ self.params["W1"].T + self.params["b1"]
        return pre, relu(pre)

    def forward(self, V):
        _, h = self._hidden(V)
        return h @ self.params["W2"].T + self.params["b2"]

    def grads(self, V, Y):
        pre, h = self._hidden(V)
        err = Y - (h @ self.params["W2"].T + self.params["b2"])
        B = len(V)
        gW2 = -(err.T @ h) / B
        gb2 = -err.mean(axis=0)
        d_pre = -(err @ self.params["W2"]) * (pre > 0)    # error passed back through ReLU
        gW1 = (d_pre.T @ V) / B
        gb1 = d_pre.mean(axis=0)
        return {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}


def gradient_check(model, V, Y, rng, n_checks=5, eps=1e-5):
    """Compare analytic gradients with numerical slopes; return the worst relative error."""
    analytic = model.grads(V, Y)
    worst = 0.0
    for name, p in model.params.items():
        for _ in range(n_checks):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            original = p[idx]
            p[idx] = original + eps
            up = model.loss(V, Y)
            p[idx] = original - eps
            down = model.loss(V, Y)
            p[idx] = original
            numeric = (up - down) / (2 * eps)
            a = analytic[name][idx]
            rel = abs(numeric - a) / max(abs(numeric) + abs(a), 1e-7)
            worst = max(worst, rel)
    return worst


if __name__ == "__main__":
    import copy
    from data import load_odours
    from inputs import make_noisy_input
    from reservoir import Reservoir

    odours, _ = load_odours()
    saved = np.load("sequences.npz")
    sequences, labels = saved["sequences"], saved["labels"]
    res = Reservoir(n_inputs=odours.shape[1], input_scale=1.0 / odours.max(), seed=0)
    rng = np.random.default_rng(3)

    # reservoir states for 5 noisy presentations of every sequence
    n_draws = 5
    V_train = np.concatenate([res.run(make_noisy_input(odours, sequences, rng))
                              for _ in range(n_draws)])
    labels_train = np.tile(labels, n_draws)
    n_nodes = V_train.shape[1]

    models = {
        "ESN": ESNReadout(n_nodes),
        "ESN + L1": ESNReadout(n_nodes, reg="l1", lam=1e-4),
        "ESN + L2": ESNReadout(n_nodes, reg="l2", lam=1e-4),
        "SpaRCe (P50)": SpaRCeReadout(V_train, percentile=50),
        "Hidden layer (100)": HiddenLayerReadout(n_nodes, n_hidden=100),
    }

    # --- 1. parameter counts vs the paper ---
    paper = {"ESN": 2002, "SpaRCe (P50)": 3002, "Hidden layer (100)": 100302}
    print("Trainable parameters:")
    for name, m in models.items():
        note = f"   (paper: {paper[name]})" if name in paper else ""
        print(f"  {name:20s} {m.n_params:7d}{note}")

    # --- 2. SpaRCe's starting sparsity ---
    sparce = models["SpaRCe (P50)"]
    print(f"\nSpaRCe active fraction at start: {sparce.active_fraction(V_train):.3f}  (want ~0.5)")

    # --- 3. gradient check, with random parameters so every gradient is non-trivial ---
    V_small, Y_small = V_train[:20], one_hot(labels_train[:20])
    print("\nGradient check, worst relative error (want < 1e-4):")
    for name, m in models.items():
        test = copy.deepcopy(m)
        for p in test.params.values():
            p[...] = rng.normal(0.0, 0.05, p.shape)
        print(f"  {name:20s} {gradient_check(test, V_small, Y_small, rng):.1e}")

    # --- 4. short training run: does the error go down? ---
    V_test = res.run(make_noisy_input(odours, sequences, rng))
    Y_test = one_hot(labels)
    start = {name: m.loss(V_test, Y_test) for name, m in models.items()}

    n_batches = 300
    print(f"\nTraining every model for {n_batches} minibatches of 20 (takes ~30 s)...")
    for _ in range(n_batches):
        idx = rng.integers(0, len(sequences), size=20)
        V = res.run(make_noisy_input(odours, sequences[idx], rng))
        Y = one_hot(labels[idx])
        for m in models.values():
            m.step(V, Y)

    print(f"\n{'model':20s} {'loss before':>12s} {'loss after':>11s} {'accuracy':>9s} {'RMSE':>6s}")
    for name, m in models.items():
        acc, rmse = m.evaluate(V_test, labels)
        print(f"{name:20s} {start[name]:12.4f} {m.loss(V_test, Y_test):11.4f} {acc:9.3f} {rmse:6.3f}")
    print(f"\nSpaRCe active fraction after training: {sparce.active_fraction(V_test):.3f}")