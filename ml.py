import math
import random
from utils import normalize_signal
from oracle import bfs_hops, extract_local_features, propagation_quality, simulate_propagation

MAX_EPOCHS = 10000
EPOCH_INTERVAL = 10
MAX_HOP_NORM = 20.0
FEATURE_DIM = 6

def features_to_vector(features, n_nodes):
    return [
        min(features["expected_rx_count"], 10.0) / 10.0,
        normalize_signal(features["best_signal"]),
        normalize_signal(features["mean_signal"]),
        features["neighbor_count"] / max(n_nodes - 1, 1),
        features["hop_count"] / MAX_HOP_NORM if features["hop_count"] >= 0 else 1.0,
        features["node_p_reach"],
    ]

def build_training_data(nodes, adj, radj):
    n = len(nodes)
    samples = []
    full_set = set(range(n))
    for source in range(n):
        relay_set = full_set - {source}
        p_full = simulate_propagation(source, relay_set, adj, radj, n)
        hops = bfs_hops(source, adj, n)
        q_full = propagation_quality(source, relay_set, adj, radj, n, p_full)
        for node in relay_set:
            features = extract_local_features(
                node, source, relay_set, adj, radj, n, p_full, hops
            )
            p_reduced = simulate_propagation(source, relay_set - {node}, adj, radj, n)
            q_reduced = propagation_quality(
                source, relay_set - {node}, adj, radj, n, p_reduced
            )
            utility = q_full - q_reduced
            samples.append((features_to_vector(features, n), utility))
    return samples

class RelayUtilityModel:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(feature_dim)]
        self.bias = 0.0

    def predict(self, features):
        s = self.bias
        for w, x in zip(self.weights, features):
            s += w * x
        return s

class Adam:
    def __init__(self, lr=0.003, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1

    def update(self, key, param, grad):
        if key not in self.m:
            self.m[key] = 0.0
            self.v[key] = 0.0
        self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * grad * grad
        m_hat = self.m[key] / (1.0 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1.0 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        return param

def train(model, samples):
    opt = Adam()
    best_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    patience = 120
    min_delta = 1e-4
    mean_target = sum(t for _f, t in samples) / len(samples)
    ss_tot = sum((t - mean_target) ** 2 for _f, t in samples)
    for epoch in range(MAX_EPOCHS):
        random.shuffle(samples)
        total_sq_err = 0.0
        total_abs_err = 0.0
        for features, target in samples:
            opt.step()
            pred = model.predict(features)
            err = pred - target
            total_sq_err += err * err
            total_abs_err += abs(err)
            for i in range(model.feature_dim):
                model.weights[i] = opt.update(("W", i), model.weights[i], 2.0 * err * features[i])
            model.bias = opt.update(("B",), model.bias, 2.0 * err)
        mae = total_abs_err / len(samples)
        mse = total_sq_err / len(samples)
        r2 = 1.0 - total_sq_err / max(ss_tot, 1e-12)
        improvement = best_mae - mae
        if improvement > min_delta:
            best_mae = mae
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch % EPOCH_INTERVAL == 0:
            print(
                f"epoch={epoch:5d}  "
                f"mse={mse:.6f}  "
                f"mae={mae:.6f}  "
                f"r2={r2:.4f}"
            )
        if mae < 1e-5:
            print(f"Converged at epoch {epoch}. mae={mae:.6f}  r2={r2:.4f}")
            break
        if epochs_without_improvement >= patience:
            print(
                f"Stabilized at epoch {epoch}. "
                f"best_mae={best_mae:.6f} "
                f"(best epoch={best_epoch})"
            )
            break
